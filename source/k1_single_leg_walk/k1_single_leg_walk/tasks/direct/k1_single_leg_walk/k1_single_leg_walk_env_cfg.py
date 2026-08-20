# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
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
    joint_action_scale_map: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            # 腿部: 主動驅動步行, 需要較大範圍
            "hip_pitch_joint": (1.0, 1.0),
            "hip_roll_joint": (1.0, 1.0),
            "hip_yaw_joint": (1.0, 1.0),
            "knee_joint": (1.0, 1.0),
            "ankle_pitch_joint": (0.5, 1.0),
            "ankle_roll_joint": (0.5, 1.0),
            # 腰部: 保守, 過大會破壞平衡
            "waist_yaw_joint": (0.5, 1.0),
            # 上半身: 允許自然擺動, 但幅度小
            "shoulder_pitch_joint": (0.20, 0.17),
            "shoulder_roll_joint": (0.15, 0.12),
            "shoulder_yaw_joint": (0.15, 0.12),
            "elbow_joint": (0.15, 0.14),
            # 手腕: 幾乎鎖住
            "wrist_roll_joint": (0.05, 0.05),
        }
    )

    # 每個關節群組相對預設姿態偏移的懲罰權重
    pose_weights_map: dict[str, float] = field(
        default_factory=lambda: {
            # 腿部: 走路必要動作，權重低，允許較大偏移不重罰
            "hip_pitch_joint": 0.2,
            "hip_roll_joint": 3.0,
            "hip_yaw_joint": 1.0,
            "knee_joint": 0.2,
            "ankle_pitch_joint": 1.0,
            "ankle_roll_joint": 2.0,
            # 腰部: 應保持穩定，權重中等偏高
            "waist_yaw_joint": 5.0,
            # 上半身: 允許小幅自然擺動，但不希望偏離太多
            "shoulder_pitch_joint": 5.0,
            "shoulder_roll_joint": 5.0,
            "shoulder_yaw_joint": 5.0,
            "elbow_joint": 5.0,
            # 手腕: 幾乎鎖住，一旦偏離要重罰
            "wrist_roll_joint": 5.0,
        }
    )

    # reset 判斷條件
    min_torso_height: float = 0.5
    max_torso_tilt: float = 0.45

    # 單腳站立的目標軀幹高度(m), 給 torso_height_penalty_scale 用
    # 實測 default_root_state[:, 2] = 0.78
    target_torso_height: float = 0.78

    # --- reward scales (7 大類) ---
    lin_vel_tracking_reward_scale: float = 2.0  # 1. 線速度跟隨
    ang_vel_tracking_reward_scale: float = 1.0  # 1. 角速度跟隨
    foot_height_reward_scale: float = 3.0  # 2. 腳掌高度追蹤
    joint_deviation_penalty_scale: float = -3.0  # 3. 預設姿態懲罰
    feet_ori_penalty_scale: float = -10.0  # 4. 腳掌朝向懲罰(左右腳 yaw 差)
    close_feet_xy_penalty_scale: float = -5.0  # 4. 腳掌間距過近懲罰
    feet_pitch_penalty_scale: float = -0.0  # 4. 腳掌平行(pitch)懲罰
    alive_reward_scale: float = 1.0  # 5. 存活獎勵
    torso_orientation_penalty_scale: float = -3.0  # 6. 軀幹直立姿態懲罰
    ang_vel_xy_penalty_scale: float = -1.0  # 6. 軀幹晃動角速度懲罰
    torso_height_penalty_scale: float = -50.0  # 6c. 軀幹高度懲罰(避免蹲低鑽 termination 門檻)
    action_rate_penalty_scale: float = -0.01  # 7. 動作變化率懲罰
    joint_vel_penalty_scale: float = -0.01  # 7b. 關節速度懲罰
    joint_acc_penalty_scale: float = -0.0  # 7b. 關節加速度懲罰
    termination_penalty_scale: float = -100.0

    # --- kernel std / 目標值 ---
    lin_vel_std: float = 0.25
    ang_vel_std: float = 0.25
    origin_height: float = 0.065
    swing_height: float = 0.05  # 擺盪最高點高度 (m)
    gait_tracking_sigma: float = 0.002  # 追蹤誤差的容忍度（越小越嚴格）
    gait_cycle_time: float = 0.8  # 一個完整步態週期的時間 (s)

    close_feet_threshold: float = 0.15  # 4. 腳掌側向間距小於此值開始懲罰
