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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz

from .k1_single_leg_walk_env_cfg import K1SingleLegWalkEnvCfg
from .reward_terms import (
    action_rate_penalty,
    alive_reward,
    ang_vel_tracking_reward,
    ang_vel_xy_penalty,
    close_feet_xy_penalty,
    feet_orientation_penalty,
    feet_pitch_penalty,
    foot_height_tracking_reward,
    joint_acc_penalty,
    joint_deviation_penalty,
    joint_vel_penalty,
    lin_vel_tracking_reward,
    termination_penalty,
    torso_orientation_penalty,
)


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
        self._pose_weights = torch.zeros(len(joint_names), device=self.device)
        self._max_delta = torch.zeros(len(joint_names), device=self.device)
        for i, name in enumerate(joint_names):
            for key, (scale, delta) in self.cfg.joint_action_scale_map.items():
                if key in name:
                    self._action_scale[i] = scale
                    self._max_delta[i] = delta
                    break
            for key, weight in self.cfg.pose_weights_map.items():
                if key in name:
                    self._pose_weights[i] = weight
                    break
        # 找出「受控關節」的 idx
        self._controlled_idx = torch.nonzero(self._action_scale > 0).squeeze(-1)

        # 找出左右腳掌對應的 body index(用於腳掌高度、朝向、間距獎勵)
        self._feet_ids, _ = self.robot.find_bodies(".*ankle_roll_link")
        self._ankle_ids, _ = self.robot.find_bodies(".*ankle_pitch_link")

        # 找出軀幹(base/torso)body index，用於姿態懲罰
        self._torso_id, _ = self.robot.find_bodies("pelvis")
        self._gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device)

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
                "lin_vel_tracking",
                "ang_vel_tracking",
                "foot_height",
                "joint_deviation",
                "feet_ori",
                "close_feet_xy",
                "feet_pitch",
                "alive",
                "torso_orientation",
                "ang_vel_xy",
                "action_rate",
                "joint_vel",
                "joint_acc",
                "termination",
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
        self.phase_sin = torch.sin(self._gait_phase)
        self.phase_cos = torch.cos(self._gait_phase)
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,  # 3
                self.robot.data.root_ang_vel_b,  # 3
                self.robot.data.projected_gravity_b,  # 3
                self._commands,  # 3
                (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self._controlled_idx],  # 23
                self.robot.data.joint_vel[:, self._controlled_idx],  # 23
                self._actions,  # 23
                self.phase_sin,  # 2
                self.phase_cos,  # 2
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self._update_velocity_markers()

        # ---------- 從機器人狀態擷取各項 reward 需要的原始量(四元數 -> 純量的轉換留在這裡) ----------
        feet_pos_z = self.robot.data.body_pos_w[:, self._ankle_ids, 2]  # (num_envs, 2)

        feet_quat = self.robot.data.body_quat_w[:, self._ankle_ids]  # (N, 2, 4)
        feet_quat_flat = feet_quat.reshape(-1, 4)  # (N*2, 4)

        # 左右腳 yaw 差, wrap 到 [-pi, pi]
        _, _, feet_yaw = euler_xyz_from_quat(feet_quat_flat)
        feet_yaw = feet_yaw.reshape(self.num_envs, 2)
        raw_yaw_diff = feet_yaw[:, 0] - feet_yaw[:, 1]
        feet_yaw_diff = torch.atan2(torch.sin(raw_yaw_diff), torch.cos(raw_yaw_diff))

        # 左右腳側向間距(投影到軀幹 y 軸)
        _, _, base_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        cos_y, sin_y = torch.cos(base_yaw), torch.sin(base_yaw)
        feet_xy = self.robot.data.body_pos_w[:, self._feet_ids, :2]
        delta_xy = feet_xy[:, 0] - feet_xy[:, 1]
        feet_lateral = torch.abs(cos_y * delta_xy[:, 1] - sin_y * delta_xy[:, 0])

        # 左右腳重力在本體座標的投影(改用重力投影,避開歐拉角 wrap 問題)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(feet_quat_flat.shape[0], 3)
        feet_gravity_b = quat_apply_inverse(feet_quat_flat, gravity_vec).reshape(self.num_envs, 2, 3)

        # 軀幹姿態 / 角速度在本體座標的投影
        torso_quat = self.robot.data.body_quat_w[:, self._torso_id[0]]  # (N, 4)
        torso_gravity_b = quat_apply_inverse(torso_quat, self._gravity_vec.expand(self.num_envs, 3))
        torso_ang_vel_w = self.robot.data.body_ang_vel_w[:, self._torso_id[0]]  # (N, 3) 世界座標
        torso_ang_vel_b = quat_apply_inverse(torso_quat, torso_ang_vel_w)  # 轉到 torso 本體座標

        # ---------- 呼叫各項 reward/penalty 公式(都在 reward_terms.py) ----------
        rewards = {
            "lin_vel_tracking": lin_vel_tracking_reward(
                self._commands,
                self.robot.data.root_lin_vel_b,
                self.cfg.lin_vel_std,
                self.cfg.lin_vel_tracking_reward_scale,
            ),
            "ang_vel_tracking": ang_vel_tracking_reward(
                self._commands,
                self.robot.data.root_ang_vel_b,
                self.cfg.ang_vel_std,
                self.cfg.ang_vel_tracking_reward_scale,
            ),
            "foot_height": foot_height_tracking_reward(
                feet_pos_z,
                self.cfg.origin_height,
                self._gait_phase,
                self.cfg.swing_height,
                self.cfg.gait_tracking_sigma,
                self.cfg.foot_height_reward_scale,
            ),
            "joint_deviation": joint_deviation_penalty(
                self.robot.data.joint_pos,
                self.robot.data.default_joint_pos,
                self._controlled_idx,
                self._pose_weights,
                self.cfg.joint_deviation_penalty_scale,
            ),
            "feet_ori": feet_orientation_penalty(feet_yaw_diff, self.cfg.feet_ori_penalty_scale),
            "close_feet_xy": close_feet_xy_penalty(
                feet_lateral, self.cfg.close_feet_threshold, self.cfg.close_feet_xy_penalty_scale
            ),
            "feet_pitch": feet_pitch_penalty(feet_gravity_b[:, :, 0], self.cfg.feet_pitch_penalty_scale),
            "alive": alive_reward(self.num_envs, self.device, self.cfg.alive_reward_scale),
            "torso_orientation": torso_orientation_penalty(
                torso_gravity_b[:, :2], self.cfg.torso_orientation_penalty_scale
            ),
            "ang_vel_xy": ang_vel_xy_penalty(torso_ang_vel_b[:, :2], self.cfg.ang_vel_xy_penalty_scale),
            "action_rate": action_rate_penalty(
                self._actions, self._previous_actions, self.cfg.action_rate_penalty_scale
            ),
            "joint_vel": joint_vel_penalty(
                self.robot.data.joint_vel, self._controlled_idx, self.cfg.joint_vel_penalty_scale
            ),
            "joint_acc": joint_acc_penalty(
                self.robot.data.joint_acc, self._controlled_idx, self.cfg.joint_acc_penalty_scale
            ),
            "termination": termination_penalty(self.reset_terminated, self.cfg.termination_penalty_scale),
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

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        # ------------ 姿態重置 ------------
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # ------------ 指令重置 ------------
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        self._gait_phase[env_ids, 0] = 0.0
        self._gait_phase[env_ids, 1] = torch.pi

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
