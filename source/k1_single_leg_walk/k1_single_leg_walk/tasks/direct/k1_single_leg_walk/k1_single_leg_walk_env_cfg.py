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
    #
    # 腿部原本是 (1.0, 1.0), 但 zero_agent 站得穩、一有(甚至是訓練初期近乎隨機的)action 就會
    # 馬上倒的現象顯示: 單腳站立的支撐面很小, PPO 剛開始 log_std=0 + clip_actions=False,
    # 隨機動作一步就可能讓髖/膝瞬間偏移到 ~1 rad(57 度), 支撐腳根本撐不住, policy 還沒機會學
    # 就已經摔了。把腿部 max_delta 縮小, 讓隨機探索不會一步就把單腳平衡摔垮。
    joint_action_scale_map: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            # 腿部: 主動驅動步行, 但單腳站立支撐面小, 不能一步跳太多
            "hip_pitch_joint": (0.4, 0.4),
            "hip_roll_joint": (0.3, 0.3),  # 側向, 對單腳平衡最敏感, 最保守
            "hip_yaw_joint": (0.3, 0.3),
            "knee_joint": (0.5, 0.5),
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

    # 每個關節群組相對預設姿態偏移的懲罰權重
    # 大致對齊 holosoma T1/G1 loco preset 的 pose_weights 規律(hip_yaw/ankle 不該亂動給高權重,
    # 腰部以上幾乎鎖住), 但 hip_pitch/knee 沒有照抄 holosoma 的 0.01(幾乎免費)——holosoma 是雙腳
    # 交替的一般步態, 有第二隻腳兜底; K1 是單腳站立, 支撐面小很多, 髖/膝幾乎零成本地亂動很容易
    # 弓身/前傾到失衡, 所以給中等權重, 讓邁步仍然划算但不能無限彎曲
    pose_weights_map: dict[str, float] = field(
        default_factory=lambda: {
            # 腿部
            "hip_pitch_joint": 0.5,
            "hip_roll_joint": 1.0,
            "hip_yaw_joint": 5.0,
            "knee_joint": 0.2,
            "ankle_pitch_joint": 5.0,
            "ankle_roll_joint": 5.0,
            # 腰部以上: 幾乎鎖住
            "waist_yaw_joint": 50.0,
            "shoulder_pitch_joint": 50.0,
            "shoulder_roll_joint": 50.0,
            "shoulder_yaw_joint": 50.0,
            "elbow_joint": 50.0,
            "wrist_roll_joint": 50.0,
        }
    )

    # reset 判斷條件
    min_torso_height: float = 0.5
    max_torso_tilt: float = 0.45

    # 單腳站立的目標軀幹高度(m), 給 torso_height_penalty_scale 用
    # 實測 default_root_state[:, 2] = 0.78
    target_torso_height: float = 0.78

    # --- reward scales ---
    # lin_vel_tracking / ang_vel_tracking / foot_height / joint_deviation / feet_ori /
    # close_feet_xy / torso_orientation / ang_vel_xy / action_rate / alive 這 10 項的數值
    # 對齊 holosoma T1/G1 雙足 locomotion reward preset(見 reward_terms.py 開頭的說明)。
    #
    # feet_ori / close_feet_xy 這兩項「規定腳掌姿態細節」的懲罰改成預設關閉: 參考
    # arXiv:2404.19173(Revisiting Reward Design and Evaluation for Robust Humanoid
    # Standing and Walking)的論點, 這類過度規定性(overly prescriptive)的懲罰會一條一條
    # 砍掉可行解空間, policy 可能被逼到只剩奇怪姿勢能同時滿足所有規定。
    lin_vel_tracking_reward_scale: float = 2.0  # 1. 線速度跟隨
    ang_vel_tracking_reward_scale: float = 1.5  # 1. 角速度跟隨
    foot_height_reward_scale: float = 5.0  # 2. 腳掌高度追蹤
    joint_deviation_penalty_scale: float = -0.5  # 3. 預設姿態懲罰
    feet_ori_penalty_scale: float = 0.0  # 4. 腳掌平整度懲罰(pitch+roll, 非左右腳 yaw 差)
    close_feet_xy_penalty_scale: float = 0.0  # 4. 腳掌間距過近懲罰(二元)
    alive_reward_scale: float = 1.0  # 5. 存活獎勵
    torso_orientation_penalty_scale: float = -10.0  # 6. 軀幹直立姿態懲罰
    ang_vel_xy_penalty_scale: float = -1.0  # 6. 軀幹晃動角速度懲罰
    action_rate_penalty_scale: float = -2.0  # 7. 動作變化率懲罰

    # 以下幾項不在 holosoma T1/G1 preset 裡(它們靠上面較強的 orientation/action_rate
    # 懲罰撐住, 不靠這些), 先預設關閉, 函式留著, 要重新開啟就把 scale 改回非 0
    torso_height_penalty_scale: float = 0.0  # 6c. 軀幹高度懲罰
    joint_vel_penalty_scale: float = 0.0  # 7b. 關節速度懲罰
    joint_acc_penalty_scale: float = 0.0  # 7b. 關節加速度懲罰
    termination_penalty_scale: float = 0.0  # 8. 提早終止懲罰

    # --- kernel std / 目標值 ---
    lin_vel_std: float = 0.25
    ang_vel_std: float = 0.25
    origin_height: float = 0.065
    swing_height: float = 0.09  # 擺盪最高點高度 (m)
    gait_tracking_sigma: float = 0.008  # 追蹤誤差的容忍度（越小越嚴格）
    gait_cycle_time: float = 0.8  # 一個完整步態週期的時間 (s)

    close_feet_threshold: float = 0.15  # 4. 腳掌側向間距小於此值開始懲罰
