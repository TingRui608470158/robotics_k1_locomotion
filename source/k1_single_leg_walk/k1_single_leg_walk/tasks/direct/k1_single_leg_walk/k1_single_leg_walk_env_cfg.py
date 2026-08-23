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
    # decimation=4, sim.dt=1/200 -> step_dt = 4*(1/200) = 20ms(跟論文 50Hz policy 頻率一樣),
    # 但物理只跑 200Hz 不是論文的 2kHz —— 2kHz 對這種有接觸碰撞的人形機器人來說, 不管訓練
    # (headless, 只在乎總吞吐量, 其實沒差)還是即時互動(keyboard_play.py/play.py 這種要
    # 邊看畫面邊操作的, 對任何硬體都是很吃緊的即時預算)都太重, 訓練變慢 ~10 倍、互動也會卡,
    # 所以退回 200Hz, 用 decimation 補回一樣的 step_dt, 犧牲一點物理保真度換流暢度。
    decimation = 4
    episode_length_s = 16.0
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

    # 手臂(含手腕)關節 —— arm_deviation_reward 只鬆散限制這幾個關節, 不再管腿/腰。
    # 舊版 pose_weights_map 對腿部也給懲罰(即使權重低), arXiv:2404.19173 的論點是
    # 這種「規定關節該待在哪」的懲罰對走路必要的腿部動作太過限制性, 所以新版直接
    # 不管腿/腰, 只用鬆散的 arm reward 防手臂自碰撞。
    arm_joint_keys: tuple[str, ...] = (
        "shoulder_pitch_joint",
        "shoulder_roll_joint",
        "shoulder_yaw_joint",
        "elbow_joint",
        "wrist_roll_joint",
    )

    # reset 判斷條件
    min_torso_height: float = 0.5
    max_torso_tilt: float = 0.45

    # 單腳站立的目標軀幹高度(m), 給 base_height_reward 用
    # 實測 default_root_state[:, 2] = 0.78
    target_torso_height: float = 0.78

    # --- 指令取樣範圍(對齊論文 cu=[cx,cy,cyaw]的範圍) ---
    # cx 前後速度(m/s): 前進(2.0)跟後退(-0.5)不對稱, 論文原文就是這樣設計
    min_vx: float = -0.5
    max_vx: float = 2.0
    # cy 側向速度(m/s)
    min_vy: float = -0.5
    max_vy: float = 0.5
    # cyaw 是論文原文定義的「轉向角速度指令」(rad/s), 不是絕對朝向。實際做法: 每次抽指令時
    # 抽一個固定的 yaw rate, 每個 step 用它去積分更新內部的目標朝向(self._commands[:,2]),
    # yaw_tracking_reward 追蹤的還是這個積分出來的絕對朝向(quat_dist 對到 c_yaw), 只是
    # c_yaw 現在會隨時間持續轉動, 不是固定不動的一次性取樣值。
    min_yaw_rate: float = -0.5
    max_yaw_rate: float = 0.5

    # 對齊論文的指令抽樣協定: 五類指令均勻抽一類([站立, 矢狀面行走(只有 cx), 側向行走
    # (只有 cy), 原地旋轉(只有 cyaw), 全向(三軸都可能非零)]), 不相關的軸鎖 0; 而且不是
    # 整個 episode 只抽一次, 是每 [min/max_resample_interval_s] 秒就在 episode 中途重新
    # 抽一次(見 env.py 的 _sample_commands())。
    min_resample_interval_s: float = 2.0
    max_resample_interval_s: float = 6.0

    # --- 訓練中隨機推力(對齊論文, 鼓勵抗擾動/推撞回穩能力) ---
    # 每個 step、每個 env 各自獨立以 random_push_prob 的機率被推一下(論文原文是每個 frame
    # 1% 機率), 力道在 [min_push_force, max_push_force] N 均勻抽, 方向在水平面隨機, 只
    # 施加在骨盆(pelvis, articulation 的 root body)上, 只持續這一個 step(=論文的 20ms,
    # 剛好等於我們現在 decimation=40、sim.dt=1/2000 算出來的 step_dt)。
    #
    # enable_random_push 預設 True(訓練用); play.py/keyboard_play.py 會在建立環境前把它
    # 設成 False, 不然 play/測試時機器人會被無預警亂推, 看起來像「沒下指令自己亂動」。
    enable_random_push: bool = True
    random_push_prob: float = 0.01
    min_push_force: float = 200.0  # N
    max_push_force: float = 800.0  # N

    # ============================================================
    # reward 架構對齊 Revisiting Reward Design and Evaluation for Robust Humanoid
    # Standing and Walking(arXiv:2404.19173)的 Reward Term Definition/Weighting 表格,
    # 每一項的公式、係數、weight 都照論文原文抄(唯一例外: feet_orientation 的核函數
    # 係數在 OCR 掃描裡缺字看不清楚, 沿用之前抓的 5.0 佔位)。細節/理由見
    # reward_terms.py 檔頭說明。
    # ============================================================

    # 1. 核心指令追蹤
    lin_vel_tracking_reward_scale: float = 0.15
    lin_vel_tracking_coeff: float = 5.0
    yaw_tracking_reward_scale: float = 0.1
    yaw_tracking_coeff: float = 300.0
    roll_pitch_tracking_reward_scale: float = 0.2
    roll_pitch_tracking_coeff: float = 30.0

    # 2. 單腳接觸(取代相位時鐘 + 腳掌高度追蹤; 論文比較五種抑制雙腳跳躍步態的方法後,
    # 認為這項最可靠、最不需要調參, 也最不限制行為)
    #
    # 「著地」用腳踝高度判定, 不用接觸力: origin_height 是腳踝(ankle_roll_link)原點到
    # 腳底的偏移量, 「腳踝 z - origin_height」就是淨離地高度, 低於 contact_height_threshold
    # 就算著地。這個 threshold 同時也是事實上的最低抬腳高度要求 —— 擺盪腳沒有真的舉過這個
    # 高度就不會被判定成「離地」, 直接解決「貼地拖著走也能拿到 single_foot_contact/
    # feet_air_time 滿分」的問題, 不用另外加一個腳掌離地高度的 reward。平地地形下腳踝高度
    # 是可靠的接觸替代量測。
    single_foot_contact_reward_scale: float = 0.1
    origin_height: float = 0.065  # m, 腳踝(ankle_roll_link)原點到腳底的偏移量
    single_contact_grace_period: float = 0.2  # s, 過去這段時間內只要出現過單腳著地就算數

    # contact_height_threshold 課程式訓練(簡化版, 純看訓練進度、不看表現): 把
    # curriculum_total_timesteps 這整段訓練切成 3 等分, 每過 1/3 就晉級一次門檻, 依序套用
    # contact_height_threshold_stages 這三個值(第一等分用 [0], 第二等分用 [1], 最後一等分
    # 用 [2])。不是一開始就固定用最終目標值硬練 —— 實測直接固定在 0.05~0.08 會讓
    # single_foot_contact 學不起來, 太嚴的門檻在 policy 還不會踏步的階段等於完全沒有梯度
    # 可以爬。curriculum_total_timesteps 要自己對齊實際打算訓練的總步數(例如 skrl
    # --max_iterations * rollouts, 或直接抓 train.py 印出來的總 timesteps), 這裡沒辦法
    # 自動讀到 agent 端的設定。目前的課程階段存在 K1SingleLegWalkEnv._contact_height_threshold
    # (不是 cfg 常數, 見 env.py 的 _update_contact_height_curriculum())。
    contact_height_threshold_stages: tuple[float, float, float] = (0.02, 0.04, 0.06)  # m
    curriculum_total_timesteps: int = 300_000

    # 2b. 兩腳角色對稱(不在論文原文裡, 是補的): single_foot_contact 只看「恰好一隻腳著地」,
    # 不管是哪隻腳, 訓練中發現 policy 會鑽漏洞 —— 固定一腳整場貼地拖著走、另一腳負責所有
    # 抬腳動作, 完全不需要真正輪流交替步態就能拿到接近滿分。這項用兩腳「最近一段時間著地
    # 時間比例」的差距來懲罰這種不對稱解法, 非站立指令時才生效。
    leg_symmetry_reward_scale: float = 0.05
    leg_symmetry_coeff: float = 20.0
    leg_symmetry_tau: float = 1.0  # s, 著地比例 EMA 的時間常數

    # 2c. 站立時的靜止獎勵(不在論文原文裡, 是補的): single_foot_contact / feet_position
    # 都只管「腳的位置/是否著地」, 不管腳有沒有在原地抖動亂動, 訓練中發現站立指令下機器人
    # 還是會不停小幅度移動腳。這項直接鼓勵站立指令時受控關節速度趨近 0, 非站立指令時關閉。
    stand_still_reward_scale: float = 0.05
    stand_still_coeff: float = 0.1

    # 3. 風格 / sim-to-real 輔助項
    base_height_reward_scale: float = 0.05
    base_height_coeff: float = 20.0

    feet_air_time_reward_scale: float = 1.0  # 全套裡唯一的稀疏 reward, 故權重特別高
    feet_air_time_threshold: float = 0.4  # s

    # gate 改用「有沒有下 cyaw 轉向指令(|yaw_rate_cmd|>0)」判斷, 不再用量測到的
    # heading error 門檻(turning_yaw_error_threshold 這個舊欄位已經拿掉)。
    feet_orientation_reward_scale: float = 0.05
    feet_orientation_coeff: float = 5.0  # 論文原文係數 OCR 缺字看不清楚, 沿用佔位值

    feet_position_reward_scale: float = 0.05  # 只在站立指令時生效, 非站立固定給滿分
    # 論文原文是 3.0, 但實測太鬆會讓站立時腳掌分岔張開(之前已經驗證過調到 20 左右能改善),
    # 這裡刻意偏離論文原文數字, 選用實測有效的值。
    feet_position_coeff: float = 20.0

    arm_deviation_reward_scale: float = 0.03
    arm_deviation_coeff: float = 3.0

    base_acceleration_reward_scale: float = 0.1
    base_acceleration_coeff: float = 0.01

    action_diff_reward_scale: float = 0.02
    action_diff_coeff: float = 0.02

    torque_reward_scale: float = 0.02
    torque_coeff: float = 0.02
