# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, wrap_to_pi

from .k1_single_leg_walk_env_cfg import K1SingleLegWalkEnvCfg


class K1SingleLegWalkEnv(DirectRLEnv):
    cfg: K1SingleLegWalkEnvCfg

    def __init__(self, cfg: K1SingleLegWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.debug_vis:
            # 目標(指令)速度箭頭 - 用綠色
            goal_marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            goal_marker_cfg.prim_path = "/Visuals/Command/velocity_goal"
            goal_marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            self.goal_vel_visualizer = VisualizationMarkers(goal_marker_cfg)

            # 機器人實際速度箭頭 - 用藍色
            # current_marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
            # current_marker_cfg.prim_path = "/Visuals/Command/velocity_current"
            # current_marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            # self.current_vel_visualizer = VisualizationMarkers(current_marker_cfg)

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # reset 當下的朝向基準, 給 heading_drift 懲罰用(見 _get_rewards 該項說明)
        self._initial_heading = torch.zeros(self.num_envs, device=self.device)

        joint_names = self.robot.data.joint_names
        self._action_scale = torch.zeros(len(joint_names), device=self.device)
        self._max_delta = torch.zeros(len(joint_names), device=self.device)
        for i, name in enumerate(joint_names):
            for key, (scale, delta) in self.cfg.joint_action_scale_map.items():
                if key in name:
                    self._action_scale[i] = scale
                    self._max_delta[i] = delta
                    break
        # 找出「受控關節」的 idx
        self._controlled_idx = torch.nonzero(self._action_scale > 0).squeeze(-1)

        # 找出左右腳掌對應的 body index(腳掌高度、水平速度、接觸偵測都用同一組 body)
        self._feet_ids, self._feet_names = self.robot.find_bodies(".*ankle_roll_link")

        # ContactSensor 掃到的 body 順序不保證跟 find_bodies 一樣, 用名字對齊, 避免左右腳接觸力兜錯
        self._contact_feet_idx = [self.contact_sensor.body_names.index(name) for name in self._feet_names]

        # 找出左右 hip_roll 關節 idx, 給 hip_roll 內夾懲罰用。兩邊用同一個旋轉軸慣例(axis="1 0 0",
        # 沒有鏡像), 已用 URDF joint limit 驗證方向: left 是 (-0.61, +2.53)、right 是 (-2.53, +0.61)
        # 剛好鏡像對稱, 代表 left_hip_roll<0 / right_hip_roll>0 是「內收(往中線夾)」的方向
        hip_roll_ids, hip_roll_names = self.robot.find_joints(".*hip_roll_joint")

        # 腿部關節 index, 給 Domain Randomization 的初始關節偏移用
        leg_joint_patterns = [
            ".*hip_pitch_joint", ".*hip_roll_joint", ".*hip_yaw_joint", 
            ".*knee_joint", ".*ankle_pitch_joint", ".*ankle_roll_joint",
        ]
        self._leg_joint_ids, _ = self.robot.find_joints(leg_joint_patterns)
        
        self._left_hip_roll_idx = hip_roll_ids[[i for i, n in enumerate(hip_roll_names) if "left" in n][0]]
        self._right_hip_roll_idx = hip_roll_ids[[i for i, n in enumerate(hip_roll_names) if "right" in n][0]]

        # hip_roll 是目前唯一左右鏡像的關節(其他關節如 hip_pitch/knee/ankle_pitch/ankle_roll
        # 都已驗證兩邊 URDF limit 完全對稱, 沒有這個問題), 動作套用/觀測端在這之前都直接共用同一個
        # scale/delta, 沒有處理「正值方向左右相反」——policy 對左右腳輸出類似數值時, 兩腳會被推向
        # 不對稱的物理方向(例如都輸出正值, 左腳外展、右腳卻內收)。這裡把右腳 hip_roll 的動作/觀測
        # 都乘 -1 校正, 讓兩邊「正值 action」都對應同一個物理方向(外展), 跟左腳一致
        self._action_sign = torch.ones(len(joint_names), device=self.device)
        self._action_sign[self._right_hip_roll_idx] = -1.0

        # 左右腳的步態相位，各自 shape (num_envs,)，範圍 [-pi, pi]
        # 右腳相位比左腳落後半個週期(pi)，讓兩腳自然交替
        self._gait_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self._gait_phase[:, 1] = torch.pi  # 右腳相位初始差 pi
        # 每個 physics step，相位要推進多少（一個週期 = 2*pi）
        self._phase_dt = (2 * torch.pi / self.cfg.gait_cycle_time) * self.step_dt

        # log
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "stance_contact",
                "slip",
                "swing_height",
                "swing_clearance",
                "lin_vel_tracking",
                "ang_vel_tracking",
                "swing_vel_tracking",
                "alive",
                "action_rate",
                "stand_still",
                "torso_orientation",
                "hip_roll_adduction",
                "stride_length",
                "heading_drift",
            ]
        }

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # stance_contact_reward / slip_penalty / swing_clearance_penalty 都要靠實際接觸力
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _update_velocity_markers(self):
        if not self.cfg.debug_vis:
            return

        # 箭頭位置：放在機器人軀幹上方一點，方便觀察
        base_pos = self.robot.data.root_pos_w.clone()
        base_pos[:, 2] += 0.5  # 抬高 0.5m，避免跟機器人本體重疊

        # ---------- 目標速度箭頭 ----------
        # 指令是 (vx, vy, yaw_rate), vx/vy 是機體座標(相對目前朝向), 這裡只取 x-y 平面方向,
        # 要先轉到世界座標箭頭才會跟著機器人朝向轉
        cmd_lin_w = torch.cat([self._commands[:, :2], torch.zeros_like(self._commands[:, :1])], dim=-1)
        goal_vel_xy = quat_apply(self.robot.data.root_quat_w, cmd_lin_w)[:, :2]
        goal_speed = torch.norm(goal_vel_xy, dim=-1)
        goal_heading = torch.atan2(goal_vel_xy[:, 1], goal_vel_xy[:, 0])

        goal_scale = torch.stack([goal_speed, torch.ones_like(goal_speed), torch.ones_like(goal_speed)], dim=-1)
        goal_quat = quat_from_euler_xyz(torch.zeros_like(goal_heading), torch.zeros_like(goal_heading), goal_heading)
        self.goal_vel_visualizer.visualize(translations=base_pos, orientations=goal_quat, scales=goal_scale)

        # ---------- 實際速度箭頭(尚未接上, 保留 visualizer 供之後接線) ----------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # clamp 原始 policy 輸出，避免異常大的取樣值平方後在 action_rate_penalty 或
        # observation 回饋中溢位成 inf/nan（實際套用到物理的量已經有 max_delta clamp，
        # 這裡是保護原始值本身，跟 skrl_ppo_cfg.yaml 的 clip_actions 是兩道防線）
        self._actions = actions.clone().clamp(-10.0, 10.0)
        default_q = self.robot.data.default_joint_pos

        # _action_sign 校正 hip_roll 左右鏡像的問題(見 __init__ 說明), 其餘關節都是 1.0 不影響
        raw_delta = self._action_scale[self._controlled_idx] * self._action_sign[self._controlled_idx] * self._actions
        clipped_delta = torch.clamp(
            raw_delta,
            -self._max_delta[self._controlled_idx],
            self._max_delta[self._controlled_idx],
        )

        self._processed_actions = default_q.clone()
        self._processed_actions[:, self._controlled_idx] = default_q[:, self._controlled_idx] + clipped_delta

        # 指令是 stand(全 0)時相位不推進, 鎖在 reset 時設定的雙腳支撐相正中央(見 _reset_idx)。
        # 步態相位不管指令是什麼原本都會持續推進, 但 stand 時所有跟步態有關的 reward 都被 gate
        # 蓋成 0, 沒有任何訊號告訴 policy 該怎麼處理相位推進到擺動區間這件事——會讓 observation
        # 出現「該啟步了」的訊號, 但指令卻要求站定, 兩個矛盾訊號疊加, 容易讓 policy 在站立時也
        # 想抬腳
        moving = (~torch.all(self._commands == 0.0, dim=1)).float().unsqueeze(-1)
        self._gait_phase = (
            torch.remainder(self._gait_phase + self._phase_dt * moving + torch.pi, 2 * torch.pi) - torch.pi
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        phase_sin = torch.sin(self._gait_phase)
        phase_cos = torch.cos(self._gait_phase)
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,  # 3
                self.robot.data.root_ang_vel_b,  # 3
                self.robot.data.projected_gravity_b,  # 3
                self._commands,  # 3, vx 是機體座標(相對目前朝向), wz 是世界座標角速度, 見 _get_rewards
                # _action_sign 校正 hip_roll 左右鏡像(見 __init__ 說明), 讓左右腳的關節偏移/角速度
                # 觀測值方向意義一致, 其餘關節都是 1.0 不影響
                (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self._controlled_idx]
                * self._action_sign[self._controlled_idx],  # 23
                self.robot.data.joint_vel[:, self._controlled_idx] * self._action_sign[self._controlled_idx],  # 23
                self._actions,  # 23
                phase_sin,  # 2
                phase_cos,  # 2
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self._update_velocity_markers()

        # ---------- 從機器人狀態擷取各項 reward 需要的原始量 ----------
        # 步態相位 -> 正規化 [0,1) -> 站立/擺動判斷(stance_fraction 門檻, 兩腳交替支撐)
        gait_phase_norm = (self._gait_phase + torch.pi) / (2 * torch.pi)
        is_stance = gait_phase_norm < self.cfg.stance_fraction  # (num_envs, 2)
        is_swing = ~is_stance

        # 接觸偵測: 用 ContactSensor 的實際接觸力, 已在 __init__ 對齊成跟 self._feet_ids 一樣的左右順序
        contact_forces = self.contact_sensor.data.net_forces_w[:, self._contact_feet_idx]  # (num_envs, 2, 3)
        contact_detected = torch.norm(contact_forces, dim=-1) > self.cfg.contact_force_threshold  # (num_envs, 2)

        # 腳掌高度(扣掉 origin_height)與擺動相的目標高度曲線(三次貝茲曲線分兩段完成「上升 -> 下降」)
        feet_pos_z = self.robot.data.body_pos_w[:, self._feet_ids, 2] - self.cfg.origin_height  # (num_envs, 2)

        def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            y_diff = y_end - y_start
            bezier = x**3 + 3 * (x**2 * (1 - x))
            return y_start + y_diff * bezier

        swing_span = 1.0 - self.cfg.stance_fraction
        t = torch.clamp((gait_phase_norm - self.cfg.stance_fraction) / swing_span, 0.0, 1.0)
        rising = cubic_bezier_interpolation(torch.zeros_like(t), torch.full_like(t, self.cfg.swing_height), 2 * t)
        falling = cubic_bezier_interpolation(torch.full_like(t, self.cfg.swing_height), torch.zeros_like(t), 2 * t - 1)
        swing_target = torch.where(t <= 0.5, rising, falling)
        foot_height_target = torch.where(is_stance, torch.zeros_like(swing_target), swing_target)

        # 腳掌水平速度(滑動懲罰用)
        foot_vel_horizontal = self.robot.data.body_lin_vel_w[:, self._feet_ids, :2]  # (num_envs, 2, 2)

        # command≈0(站立指令): 目前的 command 是整個 episode 固定不變, "stand" 模式就是精確的 0,
        # 直接判斷相等即可, 不需要額外容忍誤差
        command_is_zero = torch.all(self._commands == 0.0, dim=1)  # (num_envs,)
        gate = (~command_is_zero).float()  # 站立/擺動 reward 只在有移動指令時才算, 避免跟指令脫鉤

        # ---------- 各項 reward/penalty 公式 ----------
        # 1a/1b 站立相(is_stance 判斷本身走路/站立都適用, 不受 gate 影響——腳有沒有踩穩地面
        # 跟該不該交替擺動是兩件事, 站立時一樣需要腳確實接觸地面, 不該因為沒有移動指令就不管)
        stance_correct = (contact_detected == is_stance).float() * 2.0 - 1.0
        stance_contact = torch.sum(stance_correct, dim=1) * self.cfg.stance_contact_reward_scale

        slip = torch.sum(torch.square(foot_vel_horizontal), dim=-1)
        slip = torch.sum(slip * is_stance.float(), dim=1) * self.cfg.slip_penalty_scale

        # 2a/2b 擺動相(is_swing 時, 由 gate=command!=0 蓋掉)
        swing_height_error = torch.square(feet_pos_z - foot_height_target)
        swing_height = torch.sum(swing_height_error * is_swing.float(), dim=1) * self.cfg.swing_height_penalty_scale

        swing_clearance_violation = (is_swing & contact_detected).float()
        swing_clearance = torch.sum(swing_clearance_violation, dim=1) * self.cfg.swing_clearance_penalty_scale

        # 3 線速度追蹤(全程都在, 不受 gate 影響): vx 是機體座標(相對目前朝向前進/後退), 直接跟
        # root_lin_vel_b 比——這樣目標方向會跟著機器人目前朝向走, 不會跟 wz(見下面)打架
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_tracking = (
            torch.exp(-lin_vel_error / (self.cfg.lin_vel_std**2)) * self.cfg.lin_vel_tracking_reward_scale
        )

        # 4 角速度追蹤(全程都在, 不受 gate 影響): wz 用世界座標(root_ang_vel_w)——「轉彎速度」該
        # 是相對地面的朝向變化速度, 不該受機器人瞬間 roll/pitch 傾斜影響機體座標的量測結果
        ang_vel_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_w[:, 2])
        ang_vel_tracking = (
            torch.exp(-ang_vel_error / (self.cfg.ang_vel_std**2)) * self.cfg.ang_vel_tracking_reward_scale
        )

        # 11 直線方向鎖定(只在指令 wz=0 時計分, 涵蓋站立/直走/直退, 不含 stage 2 的轉彎模式):
        # ang_vel_tracking 只管瞬時角速度趨近 0, exp-kernel 容忍區間內殘留一點點角速度誤差幾乎不
        # 扣分, 但這種殘留誤差如果持續存在, 累積一整個 episode(20s)下來偏移量會很可觀(例如殘留
        # 0.1 rad/s 沒被抓到, 20 秒就轉了 2 rad, 約 115 度)——用 reset 當下的朝向當基準, 直接懲罰
        # 目前朝向偏離基準朝向多少(累積偏移, 不是瞬時角速度), 才能真正把路線拉直
        current_heading = euler_xyz_from_quat(self.robot.data.root_quat_w)[2]
        heading_error = wrap_to_pi(current_heading - self._initial_heading)
        heading_drift = torch.square(heading_error) * self.cfg.heading_drift_penalty_scale
        wz_command_is_zero = self._commands[:, 2] == 0.0
        heading_drift = heading_drift * wz_command_is_zero.float()

        # 9 擺動腳方向追蹤(只在 is_swing 時, 由 gate=command!=0 蓋掉): 只管方向(cosine 相似度,
        # -1~1), 不管速度大小, 見 cfg 該項註解。cmd 是機體座標, foot_vel_horizontal 是世界座標
        # (body_lin_vel_w), 要先把 cmd 轉到世界座標才能比較
        cmd_lin_w = torch.cat([self._commands[:, :2], torch.zeros_like(self._commands[:, :1])], dim=-1)
        cmd_dir_w = torch.nn.functional.normalize(quat_apply(self.robot.data.root_quat_w, cmd_lin_w)[:, :2], dim=-1)
        foot_vel_dir = torch.nn.functional.normalize(foot_vel_horizontal, dim=-1)  # (num_envs, 2, 2)
        swing_dir_alignment = torch.sum(foot_vel_dir * cmd_dir_w.unsqueeze(1), dim=-1)  # (num_envs, 2)
        swing_vel_tracking = torch.sum(swing_dir_alignment * is_swing.float(), dim=1)
        swing_vel_tracking = swing_vel_tracking * self.cfg.swing_vel_tracking_reward_scale

        # 5 存活獎勵 + action rate(全程都在)
        alive = torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward_scale

        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_rate = action_rate * self.cfg.action_rate_penalty_scale

        # 6 站立指令(command≈0)專用姿態懲罰, 由 command_is_zero 蓋(跟站立/擺動的 gate 相反)
        stand_still_error = torch.sum(
            torch.square((self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self._controlled_idx]),
            dim=1,
        )
        stand_still = stand_still_error * self.cfg.stand_still_penalty_scale

        # 7 軀幹 roll/pitch 懲罰(全程都在, 不受 gate 影響): projected_gravity 在機體座標下的
        # xy 分量應接近 0(代表軀幹接近水平)
        torso_orientation_error = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        torso_orientation = torso_orientation_error * self.cfg.torso_orientation_penalty_scale

        # 8 hip_roll 內收懲罰(全程都在, 不受 gate 影響): 腳掌朝向就算是對的, 也可能是「腿伸直、
        # 只靠 hip_roll 把腿往中線夾」造成兩腳互撞, 腳掌朝向量不到這個, 直接管 hip_roll 本身比較
        # 準。left_hip_roll<0 / right_hip_roll>0 是往中線夾的方向(見 __init__ 的 URDF limit 驗證),
        # 只罰內收方向, 外展(把腳張開)不罰
        left_hip_roll = self.robot.data.joint_pos[:, self._left_hip_roll_idx]
        right_hip_roll = self.robot.data.joint_pos[:, self._right_hip_roll_idx]
        hip_roll_adduction_error = torch.clamp(-left_hip_roll, min=0.0) + torch.clamp(right_hip_roll, min=0.0)
        hip_roll_adduction = torch.square(hip_roll_adduction_error) * self.cfg.hip_roll_penalty_scale

        # 10 跨步長度獎勵(只在單支撐、gate=command!=0 時計分): 見 cfg 該項註解。把兩腳位置沿著
        # 機體目前朝向(root_forward_xy)投影, 擺動腳領先支撐腳的距離(乘上 sign(vx) 處理前進/後退)
        # 跟目標跨步的比例, 連續、按比例給分。下限故意不卡在 0(=兩腳平行), 而是延伸到 -1(=跟
        # 支撐腳一樣落後): 擺動腳從落後支撐腳一個跨步開始擺動, 理論上該一路線性爬升到領先一個跨步,
        # 若下限卡在 0, 「還沒追上」到「剛好追平」這一整段會是平坦的 0 分, 梯度消失——而這正好是
        # 「平行步態」卡住的操作點, 沒有方向性訊號可以把 policy 推過去。下限延伸到 -1 後, 整條路徑
        # (落後 -> 追平 -> 領先)都有連續的線性梯度, 追平本身是 0 分(中性), 不是梯度死區
        feet_pos_xy = self.robot.data.body_pos_w[:, self._feet_ids, :2]  # (num_envs, 2, 2)
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        root_forward_xy_raw = quat_apply(self.robot.data.root_quat_w, forward_local)[:, :2]
        root_forward_xy = torch.nn.functional.normalize(root_forward_xy_raw, dim=-1)
        feet_proj = torch.sum(feet_pos_xy * root_forward_xy.unsqueeze(1), dim=-1)  # (num_envs, 2)
        exactly_one_swinging = is_swing.sum(dim=1) == 1
        swing_proj = torch.sum(feet_proj * is_swing.float(), dim=1)
        stance_proj = torch.sum(feet_proj * is_stance.float(), dim=1)
        stride_length_signed = (swing_proj - stance_proj) * torch.sign(self._commands[:, 0])
        target_stride_length = (
            torch.abs(self._commands[:, 0]) * self.cfg.gait_cycle_time * self.cfg.stride_length_cycle_fraction
        )
        stride_ratio = torch.where(
            target_stride_length > 1e-6,
            torch.clamp(stride_length_signed / (target_stride_length + 1e-6), -1.0, 1.0),
            torch.zeros_like(stride_length_signed),
        )
        stride_length = stride_ratio * self.cfg.stride_length_reward_scale * exactly_one_swinging.float()

        rewards = {
            "stance_contact": stance_contact,
            "slip": slip,
            "swing_height": swing_height * gate,
            "swing_clearance": swing_clearance * gate,
            "lin_vel_tracking": lin_vel_tracking,
            "ang_vel_tracking": ang_vel_tracking,
            "heading_drift": heading_drift,
            "swing_vel_tracking": swing_vel_tracking * gate,
            "alive": alive,
            "action_rate": action_rate,
            "stand_still": stand_still * command_is_zero.float(),
            "torso_orientation": torso_orientation,
            "hip_roll_adduction": hip_roll_adduction,
            "stride_length": stride_length * gate,
        }
        rewards = {key: value * self.step_dt for key, value in rewards.items()}

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        torso_height = self.robot.data.root_state_w[:, 2]
        fell_down = torso_height < self.cfg.min_torso_height

        tilt = torch.acos(torch.clamp(-self.robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        tilted_too_much = tilt > self.cfg.max_torso_tilt

        terminated = fell_down | tilted_too_much

        return terminated, time_out

    def _sample_commands(self, env_ids: torch.Tensor) -> torch.Tensor:
        """每次 reset:
        1) 從目前 cfg.command_stage 開放的模式裡依 cfg.command_mode_weights 加權隨機選
        stand / forward / backward, vx 固定為 0 / +1.0 / -1.0。不是均等機率——stand 幾乎不會
        倒、一抽到就撐到滿集, 存活時間遠長於 forward/backward, 即使 reset 時機率一樣, 平行環境
        裡「當下正在跑哪個模式」的佔比也會被存活時間拉偏, 稀釋掉 forward/backward 真正需要的
        訓練資料量, 所以要調低 stand 的取樣權重去補償。
        2) 若目前 stage 開啟 wz(cfg.command_stage_wz_enabled), 只在 forward/backward 這兩個
        真的在移動的模式上疊加一個獨立取樣的 wz, 做出邊走邊轉的弧線步態; stand 不管哪個 stage
        一律 wz=0(真的站定), 不會變成原地旋轉。
        """
        active_modes = self.cfg.command_stage_modes[self.cfg.command_stage]
        n = env_ids.shape[0]
        mode_weights = torch.tensor([self.cfg.command_mode_weights[m] for m in active_modes], device=self.device)
        mode_idx = torch.multinomial(mode_weights, n, replacement=True)

        commands = torch.zeros(n, 3, device=self.device)
        moving_mask = torch.zeros(n, dtype=torch.bool, device=self.device)

        # --- vx: forward / backward 定速 ---
        for i, mode in enumerate(active_modes):
            mask = mode_idx == i
            if mask.sum() == 0:
                continue
            if mode == "forward":
                commands[mask, 0] = 1.0
                moving_mask |= mask
            elif mode == "backward":
                commands[mask, 0] = -1.0
                moving_mask |= mask
            # stand: vx 維持 0, 不算 moving

        # --- wz: 只在目前 stage 開啟時, 疊加在真的在移動(forward/backward)的 env 上 ---
        if self.cfg.command_stage_wz_enabled.get(self.cfg.command_stage, False) and moving_mask.any():
            count = int(moving_mask.sum().item())
            commands[moving_mask, 2] = torch.empty(count, device=self.device).uniform_(
                -self.cfg.max_ang_speed, self.cfg.max_ang_speed
            )

        return commands

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        # ------------ 姿態重置 ------------
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # --- Domain Randomization: 腿部關節初始角度加隨機偏移 ---
        n_reset = env_ids.shape[0]
        joint_offset = torch.empty(n_reset, len(self._leg_joint_ids), device=self.device).uniform_(-0.05, 0.05)
        joint_pos[:, self._leg_joint_ids] += joint_offset
        if 0 in env_ids:
            idx = (env_ids == 0).nonzero(as_tuple=True)[0].item()
            # print(f"[DR check] env0 leg joint offset: {joint_offset[idx].cpu().numpy()}")
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 記錄這次 reset 當下的朝向, 當作 heading_drift 懲罰的基準(見 _get_rewards 該項說明)
        self._initial_heading[env_ids] = euler_xyz_from_quat(default_root_state[:, 3:7])[2]

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # ------------ 指令重置(離散分類 + 分階段 curriculum, 見 env_cfg.py 說明) ------------
        self._commands[env_ids] = self._sample_commands(env_ids)

        # 隨機化 reset 時的初始相位(維持左右腳 pi 的交替偏移), 避免每個 env 都固定在同一個
        # 時間點(第 8 步左右)同時觸發「該切換到擺動相」——固定起始相位會讓所有 env 在還沒
        # 建立任何單腳承重能力前, 就被同時要求抬腳, 是跟 reward/action 幅度無關的時序問題。
        # stand 指令(全 0)例外: 直接鎖進雙腳支撐相正中央(phase=0), 不給隨機相位, 也不讓它在
        # _pre_physics_step 裡繼續推進(見該處說明)——沒理由讓一個「站定」的 episode 一開始
        # 就被 observation 告知「現在該擺動」, 這種矛盾訊號是站立時還想抬腳的根源之一
        n = env_ids.shape[0]
        phi0 = torch.empty(n, device=self.device).uniform_(-torch.pi, torch.pi)
        phase_right = torch.remainder(phi0 + 2 * torch.pi, 2 * torch.pi) - torch.pi
        command_is_zero_reset = torch.all(self._commands[env_ids] == 0.0, dim=1)
        phi0 = torch.where(command_is_zero_reset, torch.zeros_like(phi0), phi0)
        phase_right = torch.where(command_is_zero_reset, torch.zeros_like(phase_right), phase_right)
        self._gait_phase[env_ids, 0] = phi0
        self._gait_phase[env_ids, 1] = phase_right

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
