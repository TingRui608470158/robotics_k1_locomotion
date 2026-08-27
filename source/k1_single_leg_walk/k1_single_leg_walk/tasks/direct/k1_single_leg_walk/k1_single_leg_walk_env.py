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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz

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
        self._left_hip_roll_idx = hip_roll_ids[[i for i, n in enumerate(hip_roll_names) if "left" in n][0]]
        self._right_hip_roll_idx = hip_roll_ids[[i for i, n in enumerate(hip_roll_names) if "right" in n][0]]

        # 軀幹 local +X = 面朝方向, 給 heading 對齊懲罰轉到世界座標用
        self._root_forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)

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
                "heading",
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
        # 指令是 (vx, vy, yaw_rate)，這裡只取 x-y 平面方向
        goal_vel_xy = self._commands[:, :2]
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

        raw_delta = self._action_scale[self._controlled_idx] * self._actions
        clipped_delta = torch.clamp(
            raw_delta,
            -self._max_delta[self._controlled_idx],
            self._max_delta[self._controlled_idx],
        )

        self._processed_actions = default_q.clone()
        self._processed_actions[:, self._controlled_idx] = default_q[:, self._controlled_idx] + clipped_delta

        self._gait_phase = torch.remainder(self._gait_phase + self._phase_dt + torch.pi, 2 * torch.pi) - torch.pi

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        phase_sin = torch.sin(self._gait_phase)
        phase_cos = torch.cos(self._gait_phase)

        # self._commands 現在是世界座標的目標(見 _get_rewards 說明), policy 自己感知不到世界朝向
        # (projected_gravity_b 對 yaw 不敏感), 額外把指令投影到機體座標當觀測, 讓 policy 有「以
        # 目前朝向來看, 目標在哪個方向」這個可以直接拿來行動的資訊
        cmd_lin_w = torch.cat([self._commands[:, :2], torch.zeros_like(self._commands[:, :1])], dim=-1)
        cmd_lin_b = quat_apply_inverse(self.robot.data.root_quat_w, cmd_lin_w)[:, :2]

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,  # 3
                self.robot.data.root_ang_vel_b,  # 3
                self.robot.data.projected_gravity_b,  # 3
                self._commands,  # 3
                (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self._controlled_idx],  # 23
                self.robot.data.joint_vel[:, self._controlled_idx],  # 23
                self._actions,  # 23
                phase_sin,  # 2
                phase_cos,  # 2
                cmd_lin_b,  # 2
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
        # 1a/1b 站立相(is_stance 時, 由 gate=command!=0 蓋掉)
        stance_correct = (contact_detected == is_stance).float() * 2.0 - 1.0
        stance_contact = torch.sum(stance_correct, dim=1) * self.cfg.stance_contact_reward_scale

        slip = torch.sum(torch.square(foot_vel_horizontal), dim=-1)
        slip = torch.sum(slip * is_stance.float(), dim=1) * self.cfg.slip_penalty_scale

        # 2a/2b 擺動相(is_swing 時, 由 gate=command!=0 蓋掉)
        swing_height_error = torch.square(feet_pos_z - foot_height_target)
        swing_height = torch.sum(swing_height_error * is_swing.float(), dim=1) * self.cfg.swing_height_penalty_scale

        swing_clearance_violation = (is_swing & contact_detected).float()
        swing_clearance = torch.sum(swing_clearance_violation, dim=1) * self.cfg.swing_clearance_penalty_scale

        # 3/4 線速度/角速度追蹤(全程都在, 不受 gate 影響): 用世界座標(root_lin_vel_w/root_ang_vel_w),
        # 不是機體座標——body frame 只看得出「相對自己目前朝向有沒有在前進」, 看不出整體路徑有沒有
        # 慢慢偏航/繞圈走, 世界座標才抓得到這種漂移。self._commands 現在定義成世界座標的目標
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_w[:, :2]), dim=1)
        lin_vel_tracking = (
            torch.exp(-lin_vel_error / (self.cfg.lin_vel_std**2)) * self.cfg.lin_vel_tracking_reward_scale
        )

        ang_vel_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_w[:, 2])
        ang_vel_tracking = (
            torch.exp(-ang_vel_error / (self.cfg.ang_vel_std**2)) * self.cfg.ang_vel_tracking_reward_scale
        )

        # 9 擺動腳速度追蹤(只在 is_swing 時, 由 gate=command!=0 蓋掉): 目標向量方向抓指令方向、
        # 大小抓指令速度, 跟 lin_vel_tracking 同樣用 exp-kernel, 有界、有明確最佳解(不是越快越好)。
        # self._commands 跟 foot_vel_horizontal 現在都是世界座標, 直接比較, 不用再轉機體座標
        cmd_lin = self._commands[:, :2]
        target_foot_vel = cmd_lin.unsqueeze(1).expand(-1, 2, -1)  # (num_envs, 2, 2), 兩腳目標一樣
        swing_vel_error = torch.sum(torch.square(foot_vel_horizontal - target_foot_vel), dim=-1)  # (num_envs, 2)
        swing_vel_tracking = torch.exp(-swing_vel_error / (self.cfg.swing_vel_std**2)) * is_swing.float()
        swing_vel_tracking = torch.sum(swing_vel_tracking, dim=1) * self.cfg.swing_vel_tracking_reward_scale

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

        # 10 朝向對齊懲罰(由 gate=command!=0 蓋掉, stand 時沒有方向可對齊): 機器人朝向(root 局部
        # +X 投影到世界 XY)應該對齊指令方向, 不要用側身/背對指令方向的方式走
        root_forward_w = quat_apply(self.robot.data.root_quat_w, self._root_forward_local.expand(self.num_envs, 3))
        root_forward_xy = torch.nn.functional.normalize(root_forward_w[:, :2], dim=-1)
        cmd_dir_xy = torch.nn.functional.normalize(self._commands[:, :2], dim=-1, eps=1e-6)
        heading_alignment = torch.sum(root_forward_xy * cmd_dir_xy, dim=-1)  # (num_envs,), 1=完全對齊
        heading = (1.0 - heading_alignment) * self.cfg.heading_penalty_scale

        rewards = {
            "stance_contact": stance_contact * gate,
            "slip": slip * gate,
            "swing_height": swing_height * gate,
            "swing_clearance": swing_clearance * gate,
            "lin_vel_tracking": lin_vel_tracking,
            "ang_vel_tracking": ang_vel_tracking,
            "swing_vel_tracking": swing_vel_tracking * gate,
            "alive": alive,
            "action_rate": action_rate,
            "stand_still": stand_still * command_is_zero.float(),
            "torso_orientation": torso_orientation,
            "hip_roll_adduction": hip_roll_adduction,
            "heading": heading * gate,
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

    # def _sample_commands(self, env_ids: torch.Tensor) -> torch.Tensor:
    #     """每次 reset 從目前 cfg.command_stage 開放的模式裡均勻隨機選一種, 只讓其中一個
    #     方向(vx 或 wz)非零, 其餘維持 0——避免連續 uniform 取樣三軸同時產生難學的複合指令。
    #     """
    #     active_modes = self.cfg.command_stage_modes[self.cfg.command_stage]
    #     n = env_ids.shape[0]
    #     mode_idx = torch.randint(0, len(active_modes), (n,), device=self.device)

    #     commands = torch.zeros(n, 3, device=self.device)
    #     for i, mode in enumerate(active_modes):
    #         mask = mode_idx == i
    #         count = int(mask.sum().item())
    #         if count == 0:
    #             continue
    #         if mode == "stand":
    #             continue  # 保持 0
    #         if mode == "forward":
    #             commands[mask, 0] = torch.empty(count, device=self.device).uniform_(0.0, self.cfg.max_lin_speed_x)
    #         elif mode == "backward":
    #             commands[mask, 0] = torch.empty(count, device=self.device).uniform_(
    #                 -self.cfg.max_lin_speed_x_backward, 0.0
    #             )
    #         elif mode == "turn_left":
    #             commands[mask, 2] = torch.empty(count, device=self.device).uniform_(0.0, self.cfg.max_ang_speed)
    #         elif mode == "turn_right":
    #             commands[mask, 2] = torch.empty(count, device=self.device).uniform_(-self.cfg.max_ang_speed, 0.0)

    #     return commands
    def _sample_commands(self, env_ids: torch.Tensor) -> torch.Tensor:
        """每次 reset:
        1) 從目前 cfg.command_stage 開放的模式裡均勻隨機選 forward 或 backward,
        線速度固定為 ±1.0(不再 uniform 取樣)。
        2) 額外獨立取樣一個 wz(旋轉角速度), 與 vx 同時存在、互不互斥。
        """
        active_modes = self.cfg.command_stage_modes[self.cfg.command_stage]
        n = env_ids.shape[0]
        mode_idx = torch.randint(0, len(active_modes), (n,), device=self.device)

        commands = torch.zeros(n, 3, device=self.device)

        # --- vx: forward / backward 定速 ---
        for i, mode in enumerate(active_modes):
            mask = mode_idx == i
            if mask.sum() == 0:
                continue
            if mode == "forward":
                commands[mask, 0] = 1.0
            elif mode == "backward":
                commands[mask, 0] = -1.0
            # 若還保留 "stand" 之類模式, vx 維持 0

        # --- wz: 獨立取樣, 與 vx 疊加而非互斥 ---
        commands[:, 2] = torch.empty(n, device=self.device).uniform_(-self.cfg.max_ang_speed, self.cfg.max_ang_speed)

        return commands

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        # ------------ 指令重置(離散分類 + 分階段 curriculum, 見 env_cfg.py 說明) ------------
        # 要先取樣 command, 才知道 reset 姿態該面向哪個世界方向(見下面姿態重置)
        self._commands[env_ids] = self._sample_commands(env_ids)

        # ------------ 姿態重置 ------------
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self._commands 是世界座標指令(forward=+X, backward=-X), 但機器人永遠用同一個預設姿態
        # reset。如果 reset 時面向不跟 command 方向一致(例如抽到 backward 卻面向 +X), 機器人得
        # 先學會轉身 180 度才可能同時滿足 lin_vel_tracking(世界座標)+ heading(朝向對齊), 學習
        # 起點不一致會拖慢訓練——所以 reset 時直接讓它面向 command 的世界方向, backward 面向 -X
        yaw = torch.where(
            self._commands[env_ids, 0] < 0,
            torch.full_like(self._commands[env_ids, 0], torch.pi),
            torch.zeros_like(self._commands[env_ids, 0]),
        )
        default_root_state[:, 3:7] = torch.stack(
            [torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1
        )

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # 隨機化 reset 時的初始相位(維持左右腳 pi 的交替偏移), 避免每個 env 都固定在同一個
        # 時間點(第 8 步左右)同時觸發「該切換到擺動相」——固定起始相位會讓所有 env 在還沒
        # 建立任何單腳承重能力前, 就被同時要求抬腳, 是跟 reward/action 幅度無關的時序問題。
        n = env_ids.shape[0]
        phi0 = torch.empty(n, device=self.device).uniform_(-torch.pi, torch.pi)
        self._gait_phase[env_ids, 0] = phi0
        self._gait_phase[env_ids, 1] = torch.remainder(phi0 + 2 * torch.pi, 2 * torch.pi) - torch.pi

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
