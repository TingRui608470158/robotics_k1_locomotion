# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.Robotics_K1 import K1_HUMANOID_CFG


@configclass
class K1SingleLegWalkEnvCfg(DirectRLEnvCfg):
    debug_vis: bool = True

    # env
    decimation = 2
    episode_length_s = 20.0
    # - spaces definition
    action_space = 23
    observation_space = 85
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot
    robot_cfg = K1_HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # action
    # 每個關節群組的 (action_scale, max_delta), 單位: rad
    # 設計原則: action_scale >= max_delta, 這樣 clamp 才會實際生效
    # 上限範圍由 max_delta 決定, action_scale 只是敏感度係數
    #
    # 腿部原本是 (1.0, 1.0), 但 zero_agent 站得穩、一有(甚至是訓練初期近乎隨機的)action 就會
    # 馬上倒的現象顯示: 單腳站立的支撐面很小, PPO 剛開始 log_std=0 + clip_actions=False,
    # 隨機動作一步就可能讓髖/膝瞬間偏移到 ~1 rad(57 度), 支撐腳根本撐不住, policy 還沒機會學
    # 就已經摔了。把腿部 max_delta 縮小, 讓隨機探索不會一步就把單腳平衡摔垮。
    joint_action_scale_map: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            # 腿部: 主動驅動步行, 但單腳站立支撐面小, 不能一步跳太多
            "hip_pitch_joint": (1, 0.5),
            "hip_roll_joint": (0.3, 0.3),  # 側向, 對單腳平衡最敏感, 最保守
            "hip_yaw_joint": (0.3, 0.3),
            "knee_joint": (1.0, 0.5),
            "ankle_pitch_joint": (0.3, 0.3),
            "ankle_roll_joint": (0.3, 0.3),
            # 腰部: 保守, 過大會破壞平衡
            "waist_yaw_joint": (0.3, 0.3),
            # 上半身: 允許自然擺動, 但幅度小
            "shoulder_pitch_joint": (0.20, 0.17),
            "shoulder_roll_joint": (0.15, 0.12),
            "shoulder_yaw_joint": (0.15, 0.12),
            "elbow_joint": (0.15, 0.14),
            # 手腕: 幾乎鎖住
            "wrist_roll_joint": (0.05, 0.05),
        }
    )

    # reset 判斷條件
    min_torso_height: float = 0.6
    max_torso_tilt: float = 0.45

    # --- 接觸感測: stance_contact_reward / slip_penalty / swing_clearance_penalty 都要靠這個 ---
    # prim_path 依實際 link 名稱調整(跟 self._feet_ids 找的是同一組 body: .*ankle_roll_link)
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*ankle_roll_link",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )
    contact_force_threshold: float = 100.0  # N, 超過這個力才算「有觸地」

    # --- 步態相位: 標準雙足步態, 60% 站立相 / 40% 擺動相, 兩腳交替支撐 ---
    stance_fraction: float = 0.6
    swing_height: float = 0.09  # 擺盪最高點高度 (m)
    gait_cycle_time: float = 0.8  # 一個完整步態週期的時間 (s)
    origin_height: float = 0.065

    # --- reward scales(全新設計, 6 大類/9 項, 不是照抄舊版數值, 都是待調的起始猜測) ---
    # 1a/1b 站立相(is_stance 時, 由 gate=command!=0 蓋掉)
    stance_contact_reward_scale: float = 1.0
    slip_penalty_scale: float = -1.0
    # 2a/2b 擺動相(is_swing 時, 由 gate=command!=0 蓋掉)
    swing_height_penalty_scale: float = -10.0
    swing_clearance_penalty_scale: float = -2.0
    # 3/4 線速度/角速度追蹤(全程都在, 不受 gate 影響)
    lin_vel_tracking_reward_scale: float = 2.0
    ang_vel_tracking_reward_scale: float = 1.5
    lin_vel_std: float = 0.25
    ang_vel_std: float = 0.25
    # 11 直線方向鎖定(只在指令 wz=0 時計分, 涵蓋站立/直走/直退, 不含 stage 2 的轉彎模式): 見
    # env.py 該項註解——ang_vel_tracking 只管瞬時角速度, 容忍區間內的殘留誤差累積一整個 episode
    # 下來會偏移很多, 這項直接懲罰「目前朝向」偏離「reset 當下朝向」多少, 才能真正拉直路線
    heading_drift_penalty_scale: float = -2.0
    # 5 存活獎勵 + action rate(全程都在)
    alive_reward_scale: float = 0.5
    action_rate_penalty_scale: float = -0.3
    # 6 站立指令(command≈0)專用姿態懲罰, 由 command_is_zero 蓋(跟站立/擺動的 gate 相反)
    stand_still_penalty_scale: float = -1.0
    # 7 軀幹 roll/pitch 懲罰(全程都在, 不受 gate 影響): 直接對「傾斜」給連續梯度, 不像
    # termination 只在超過 max_torso_tilt 才有訊號, 讓 policy 在真的倒下之前就有機會被導正
    torso_orientation_penalty_scale: float = -1.0
    # 8 hip_roll 內收懲罰(全程都在, 不受 gate 影響): 腳掌朝向就算是對的, 也可能是「腿伸直、只靠
    # hip_roll 把腿往中線夾」造成兩腳互撞, 腳掌朝向量不到這個(曾試過, 已改用這項取代), 直接管
    # hip_roll 本身比較準。只罰內收方向, 外展(把腳張開)不罰
    hip_roll_penalty_scale: float = -1.0
    # 9 擺動腳方向追蹤(只在 is_swing 時, 且由 gate=command!=0 蓋掉): 只管方向(cosine 相似度,
    # -1~1), 不管速度大小——原本用 exp-kernel 同時要求方向+大小貼近指令速度, 會跟 stride_length
    # (要求跨步夠遠, 常常需要比指令速度更快)互相打架, 拿掉方向約束又會讓腳往內偏。方向跟大小
    # 解耦後, 大小交給 stride_length 決定, 這裡只負責不讓擺動腳偏離該走的方向
    swing_vel_tracking_reward_scale: float = 1.0
    # 10 跨步長度獎勵(只在單支撐、gate=command!=0 時計分): 目前在擺動的那隻腳, 沿著指令方向
    # 投影, 領先支撐腳的距離跟「依指令速度算出來的目標跨步」的比例, 連續、按比例給分, 範圍
    # -1(落後支撐腳一個跨步, 擺動剛開始的起始狀態)到 +1(領先支撐腳一個跨步, 真正交叉過去)。
    # 下限故意不卡在 0(=兩腳平行)——若卡在 0, 「還沒追上」到「追平」這一整段會是平坦 0 分、
    # 梯度消失, 而這正好是「平行步態」卡住的操作點。目標跨步: 一個完整步態週期機身要移動
    # |vx|*gait_cycle_time, 標準雙足交替步態一個週期邁兩步, 單步理論上該負責一半
    stride_length_cycle_fraction: float = 0.5
    stride_length_reward_scale: float = 2.0

    # --- command: 離散分類 + 分階段 curriculum ---
    # 原本用連續 uniform 分布同時取樣 vx/vy/wz, 容易產生「三個方向都有一點點」的複合指令,
    # 對機器人來說難學、reward 訊號雜。改成離散分類: 每次 reset 從目前 stage 開放的模式裡
    # 均勻隨機選一種, 讓每個 env 在一段時間內目標單純明確(只有一個方向有速度, 其餘為 0)。
    max_lin_speed_x: float = 1.5  # m/s, 前進模式用
    max_lin_speed_x_backward: float = 0.75  # m/s, 後退模式用, 後退步態較不自然, 上限抓比前進低
    max_lin_speed_y: float = 0.5  # m/s, 目前沒有任何模式使用, 保留給之後的側移模式
    max_ang_speed: float = 1.0  # rad/s, 左轉/右轉模式用

    # 訓練啟動時手動指定要用哪個 stage; 觀察訓練狀況、確認完成度夠高後, 從該 stage 的
    # checkpoint 接續訓練並手動切到下一個 stage, 不寫自動判斷/切換邏輯。
    # key 用字串: isaaclab 的 class_to_dict() 會假設所有 dict key 都是字串(key.startswith("__")),
    # int key 在 hydra 轉換設定時會直接噴 AttributeError。
    command_stage: str = "0"
    command_stage_modes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "0": ["stand", "forward"],
            "1": ["stand", "forward", "backward"],
            "2": ["stand", "forward", "backward"],
        }
    )
    # 每個模式的取樣權重(reset 時用, 不是均等機率): stand 幾乎不會倒、一抽到就撐到滿集, 存活
    # 時間遠長於 forward/backward——即使 reset 時三者機率一樣, 時間拉長後平行環境裡「當下正在
    # 跑哪個模式」的佔比也會被存活時間拉偏, stand 佔比會遠超過取樣機率, 稀釋掉 forward/backward
    # 真正需要的訓練資料量。調低 stand 權重去補償這個偏差
    command_mode_weights: dict[str, float] = field(
        default_factory=lambda: {"stand": 1.0, "forward": 6.0, "backward": 3.0}
    )
    # 是否在 forward/backward 模式上疊加 wz(弧線走法); stand 不管哪個 stage 一律 wz=0(真的站定)。
    # 前面 stage 先練純直走, 到 stage 2 才加入邊走邊轉, 不是新增獨立的模式
    command_stage_wz_enabled: dict[str, bool] = field(default_factory=lambda: {"0": False, "1": False, "2": True})
