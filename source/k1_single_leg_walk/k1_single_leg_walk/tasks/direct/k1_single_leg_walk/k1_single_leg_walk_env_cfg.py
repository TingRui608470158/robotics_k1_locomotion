# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import field

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.Robotics_K1 import K1_HUMANOID_CFG

@configclass
class EventCfg:
    """Domain randomization 事件設定。"""

    # 地面摩擦力隨機化：只作用在腳掌（左右腳分別獨立取樣）
    randomize_foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "static_friction_range": (0.4, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )

    # 外力推撞：每個 env 各自獨立隨機倒數(interval_range_s), 時間到就給機身一個隨機水平速度,
    # 模擬被撞/被推——訓練 policy 學會抵抗擾動、保持平衡, 不是只會走預先排好的步態
    push_robot: EventTerm | None = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 骨盆質量隨機化(模擬製造誤差/負重變化): 官方文件建議只在初始化做一次(mode="startup"),
    # 不要每次 reset 都跑(這個操作用 CPU tensor, 對所有 env 一次做比較划算)
    randomize_base_mass: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class K1SingleLegWalkEnvCfg(DirectRLEnvCfg):
    debug_vis: bool = True

    # env
    decimation = 2
    episode_length_s = 20.0
    # - spaces definition
    action_space = 23
    # actor(policy): root_ang_vel_b(3) + projected_gravity_b(3) + commands(3) +
    # joint_pos-default(23) + joint_pos-previous_joint_pos(23) + joint_vel(23) + joint_torque(23) +
    # actions(23) + phase_sin(2) + heading_error(1) = 127
    observation_space = 127
    # critic(value, asymmetric actor-critic): actor 的全部 127 項 + root_lin_vel_b(3) +
    # torso_height(1) + joint_vel(23, 重複列一次) + feet_height(2) + contact_forces(6) = 162,
    # 只有 critic 看得到、訓練完就丟掉, 不用管實機拿不拿得到(見 _get_observations 說明)
    state_space = 162

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot
    robot_cfg = K1_HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    # domain randomization
    events: EventCfg = EventCfg()
    # 每項獨立開關, 預設全部關閉, 由使用者自己決定何時開(見 __post_init__)——避免新增的 DR
    # 默默改變目前訓練難度
    dr_enable_push: bool = False
    dr_enable_mass: bool = False
    dr_enable_obs_noise: bool = False
    # 觀測雜訊: 只在「真的像感測器讀值」的欄位上加(角速度/重力投影/關節位置差分/關節角速度/
    # 關節力矩/朝向誤差), std 各自可調; commands/actions/phase_sin 不是感測器讀值(分別是目標值、
    # policy 自己剛輸出的值、內部步態時鐘), 不該加雜訊, 由 __post_init__ 組 std tensor 時補 0
    obs_noise_ang_vel_std: float = 0.05
    obs_noise_gravity_std: float = 0.02
    obs_noise_joint_pos_std: float = 0.01
    obs_noise_joint_vel_std: float = 0.5
    obs_noise_joint_torque_std: float = 1.0
    obs_noise_heading_std: float = 0.01
    observation_noise_model: NoiseModelCfg | None = None  # __post_init__ 依 dr_enable_obs_noise 決定要不要建

    def __post_init__(self):
        # 依開關決定要不要保留對應的 EventTerm——EventManager 看到某個 term 是 None 就會直接
        # 跳過, 是 IsaacLab 關閉單一 DR 項目的標準做法
        if not self.dr_enable_push:
            self.events.push_robot = None
        if not self.dr_enable_mass:
            self.events.randomize_base_mass = None

        if self.dr_enable_obs_noise:
            # 跟 observation_space 上面的分段註解對齊: 3+3+3+23+23+23+23+23+2+1 = 127
            std = torch.cat(
                [
                    torch.full((3,), self.obs_noise_ang_vel_std),  # root_ang_vel_b
                    torch.full((3,), self.obs_noise_gravity_std),  # projected_gravity_b
                    torch.zeros(3),  # commands, 不加雜訊
                    torch.full((23,), self.obs_noise_joint_pos_std),  # joint_pos_delta_default
                    torch.full((23,), self.obs_noise_joint_pos_std),  # joint_pos_delta_step
                    torch.full((23,), self.obs_noise_joint_vel_std),  # joint_vel
                    torch.full((23,), self.obs_noise_joint_torque_std),  # joint_torque
                    torch.zeros(23),  # actions, 不加雜訊
                    torch.zeros(2),  # phase_sin, 不加雜訊
                    torch.full((1,), self.obs_noise_heading_std),  # heading_error
                ]
            )
            self.observation_noise_model = NoiseModelCfg(noise_cfg=GaussianNoiseCfg(mean=0.0, std=std, operation="add"))

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

    # --- reward scales(14 項, 分 6 大類 A~F, 呼應 env.py _get_rewards() 的分類, 都是待調的
    # 起始猜測) ---
    # ========== A. 支撐相 (STANCE) ==========
    stance_contact_reward_scale: float = 1.0
    slip_penalty_scale: float = -1.0

    # ========== B. 擺動相 (SWING) ==========
    swing_height_penalty_scale: float = -10.0
    swing_clearance_penalty_scale: float = 0.0 #-2.0
    swing_vel_tracking_reward_scale: float = 1.0  # 只管方向不管大小, 大小交給 stride_length 決定
    stride_length_cycle_fraction: float = 0.5
    stride_length_reward_scale: float = 2.0  # 範圍 -1~+1, 見 env.py 該項計算

    # ========== C. 指令追蹤 (VELOCITY TRACKING) ==========
    lin_vel_tracking_reward_scale: float = 2.0
    lin_vel_std: float = 0.25
    ang_vel_tracking_reward_scale: float = 1.5
    ang_vel_std: float = 0.25
    heading_drift_penalty_scale: float = -2.0  # 累積朝向偏移, 補 ang_vel_tracking 抓不到的長期漂移

    # ========== D. 姿態穩定 (POSTURE) ==========
    torso_orientation_penalty_scale: float = -5.0
    hip_roll_penalty_scale: float = -10.0  # 只罰內收方向, 見 env.py 該項計算

    # ========== E. 站立指令專用 (STAND-STILL) ==========
    stand_still_penalty_scale: float = -1.0

    # ========== F. 基礎/正則化 (BASE) ==========
    alive_reward_scale: float = 1.0
    action_rate_penalty_scale: float = -0.3
    joint_torque_penalty_scale: float = -1.0e-4  # 抓小: 力矩平方和原始量級遠大於其他項

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
        default_factory=lambda: {"stand": 2.0, "forward": 5.0, "backward": 3.0}
    )
    # 是否在 forward/backward 模式上疊加 wz(弧線走法); stand 不管哪個 stage 一律 wz=0(真的站定)。
    # 前面 stage 先練純直走, 到 stage 2 才加入邊走邊轉, 不是新增獨立的模式
    command_stage_wz_enabled: dict[str, bool] = field(default_factory=lambda: {"0": False, "1": False, "2": True})
    # 是否在目前 stage 開啟「episode 中途重新抽指令」(不 reset 整個 episode, 只換目標), 讓
    # policy 在訓練時就會遇到指令切換, 而不是只有 keyboard_play.py 互動操作才第一次碰到。預設
    # 全部關閉, 由使用者自己決定何時開, 避免默默改變目前訓練難度
    command_stage_resample_enabled: dict[str, bool] = field(
        default_factory=lambda: {"0": False, "1": False, "2": False}
    )
    # 每個 env 各自倒數多久(秒)重新抽一次指令, 各自獨立隨機取樣, 避免全部 env 同步換指令
    command_resample_time_range_s: tuple[float, float] = (2.0, 4.0)
