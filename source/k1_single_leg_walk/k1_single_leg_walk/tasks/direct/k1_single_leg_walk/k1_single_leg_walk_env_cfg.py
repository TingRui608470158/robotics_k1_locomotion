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
            "hip_pitch_joint": (1.0, 0.4),
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
    contact_force_threshold: float = 5.0  # N, 超過這個力才算「有觸地」

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
    # 同一個 stage 內的模式一律等機率, 不額外做機率加權(要調的話等有需要再加)。
    # key 用字串: isaaclab 的 class_to_dict() 會假設所有 dict key 都是字串(key.startswith("__")),
    # int key 在 hydra 轉換設定時會直接噴 AttributeError。
    command_stage: str = "1"
    command_stage_modes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "0": ["stand", "forward"],
            "1": ["stand", "forward", "backward"],
            "2": ["stand", "forward", "backward", "turn_left", "turn_right"],
        }
    )
