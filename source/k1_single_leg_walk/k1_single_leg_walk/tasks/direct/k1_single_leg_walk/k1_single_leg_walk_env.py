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
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz

from .k1_single_leg_walk_env_cfg import K1SingleLegWalkEnvCfg
from .reward_terms import (
    action_diff_reward,
    arm_deviation_reward,
    base_acceleration_reward,
    base_height_reward,
    feet_air_time_reward,
    feet_orientation_reward,
    feet_position_reward,
    leg_symmetry_reward,
    lin_vel_xy_tracking_reward,
    roll_pitch_tracking_reward,
    single_foot_contact_reward,
    stand_still_reward,
    torque_reward,
    yaw_from_quat,
    yaw_tracking_reward,
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
        # (vx_cmd, vy_cmd, yaw_cmd) —— 第三欄是「目標朝向」(絕對 heading), 不是 yaw rate 本身。
        # 對齊論文 cu=[cx,cy,cyaw]的定義, cyaw 是轉向角速度指令(rad/s), 每次抽指令時固定
        # 抽一個值存在 self._yaw_rate_cmd, 每個 step 在 _get_rewards() 開頭把它積分進這裡的
        # yaw_cmd, yaw_tracking_reward 追蹤的還是這個(持續轉動的)絕對朝向。
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_rate_cmd = torch.zeros(self.num_envs, device=self.device)
        self._is_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 對齊論文的五類指令: [站立, 矢狀面(只有 cx), 側向(只有 cy), 原地旋轉(只有 cyaw), 全向
        # (三軸都可能非零)], 每一列是 (vx_active, vy_active, yaw_active) 的 0/1 mask, 用抽到的
        # category(0~4)去 index 這張表, 決定哪幾軸要保留、哪幾軸要鎖 0(見 _sample_commands()）。
        self._command_axes_active = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 0: 站立
                [1.0, 0.0, 0.0],  # 1: 矢狀面行走
                [0.0, 1.0, 0.0],  # 2: 側向行走
                [0.0, 0.0, 1.0],  # 3: 原地旋轉
                [1.0, 1.0, 1.0],  # 4: 全向
            ],
            device=self.device,
        )
        # 每個 env 距離下一次重新抽指令還剩幾個 step(2~6 秒隨機, 對齊論文); 不是整個 episode
        # 只在 reset 時抽一次, 是 episode 中途也會照這個倒數重新抽(見 _get_rewards() 開頭)。
        self._steps_until_resample = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        joint_names = self.robot.data.joint_names
        self._action_scale = torch.zeros(len(joint_names), device=self.device)
        self._max_delta = torch.zeros(len(joint_names), device=self.device)
        arm_mask = torch.zeros(len(joint_names), dtype=torch.bool, device=self.device)
        for i, name in enumerate(joint_names):
            for key, (scale, delta) in self.cfg.joint_action_scale_map.items():
                if key in name:
                    self._action_scale[i] = scale
                    self._max_delta[i] = delta
                    break
            for key in self.cfg.arm_joint_keys:
                if key in name:
                    arm_mask[i] = True
                    break
        # 找出「受控關節」/「手臂關節」的 idx
        self._controlled_idx = torch.nonzero(self._action_scale > 0).squeeze(-1)
        self._arm_idx = torch.nonzero(arm_mask).squeeze(-1)

        # 找出左右腳掌(ankle_roll_link, 實際碰地的那節)對應的 body index
        # 順序固定是 [left, right](find_bodies 對 "left_..."/"right_..." 字母序排序)
        self._feet_ids, _ = self.robot.find_bodies(".*_ankle_roll_link")

        # 站立時腳掌相對骨盆(pelvis-yaw 座標)的 nominal xy 位置, 取自 URDF 預設姿態
        # (joint_pos=0)下沿運動鏈把各關節 origin 加總算出來的靜態值, 見 reward_terms.py
        # feet_position_reward 的說明 —— 只是鬆散的參考值, 不是精確量測。
        self._nominal_feet_local_xy = torch.tensor([[0.0, 0.125], [0.0, -0.125]], device=self.device)

        # pelvis 就是 articulation 的 root body, 姿態直接用 root_quat_w, 不用另外查 body index;
        # 但訓練中隨機推力要用 set_forces_and_torques 指定 body_ids, 這裡才需要查一次 index
        self._torso_body_id, _ = self.robot.find_bodies("pelvis")
        self._gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        # 這個 step 要不要推、往哪個方向推多大力(見 _pre_physics_step() 抽樣, _apply_action()
        # 每個 physics substep 都重新施加, 讓力道整個 step(=20ms)內保持不變)
        self._push_force_w = torch.zeros(self.num_envs, 3, device=self.device)

        # 單腳接觸的「寬限期」歷史緩衝區: 過去 single_contact_grace_period 秒內只要出現過
        # 恰好單腳著地就算數, 避免站立/擺動相位切太死(見 reward_terms.py 說明)
        self._contact_hist_len = max(1, round(self.cfg.single_contact_grace_period / self.step_dt))
        self._single_contact_hist = torch.zeros(
            self.num_envs, self._contact_hist_len, dtype=torch.bool, device=self.device
        )
        # 快取這個 step 算出的腳掌接觸狀態, 給 _get_observations 用(_get_rewards 先跑)
        self._last_foot_in_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        # 「著地」改用腳踝高度判定(不用 ContactSensor 的接觸力), 平地地形下腳踝高度是很可靠
        # 的替代量測, 而且 contact_height_threshold 順便就是事實上的最低抬腳高度要求 ——
        # 擺盪腳一定要把高度舉過這個門檻才會被判定成「離地」, single_foot_contact /
        # feet_air_time 才會給credit, 不會再出現貼地拖著走也能拿滿分的情況。因為不再靠
        # ContactSensor 內建的 air-time 追蹤(那是依接觸力算的), 這裡自己維護每隻腳的
        # 騰空計時器: 著地時歸零, 騰空時每個 step 累加, 從「著地」變「騰空」的那一刻記錄
        # last_air_time。
        self._prev_foot_in_contact = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._foot_air_time = torch.zeros(self.num_envs, 2, device=self.device)

        # contact_height_threshold 課程式訓練狀態(全體共用一個純量, 不是每個 env 各自的):
        # 把 curriculum_total_timesteps 切成 3 等分, 每過 1/3 晉級一次, 依序套用
        # contact_height_threshold_stages 三個值。見 env_cfg.py 的說明跟
        # _update_contact_height_curriculum()。
        self._contact_height_threshold = self.cfg.contact_height_threshold_stages[0]

        # 軀幹線速度上一步的值, 用有限差分算 base_acceleration_reward(論文公式只看線加速度)
        self._prev_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # 左右腳「最近一段時間著地時間比例」的 EMA, 給 leg_symmetry_reward 用; 初始值 0.5
        # 代表「還不知道, 先當作對稱」, 避免剛 reset 就被判定不對稱
        self._foot_contact_frac = torch.full((self.num_envs, 2), 0.5, device=self.device)

        # log
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel_tracking",
                "yaw_tracking",
                "roll_pitch_tracking",
                "single_foot_contact",
                "leg_symmetry",
                "stand_still",
                "base_height",
                "feet_air_time",
                "feet_orientation",
                "feet_position",
                "arm_deviation",
                "base_acceleration",
                "action_diff",
                "torque",
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
        # 指令是 (vx, vy, yaw_cmd)，這裡只取 x-y 平面方向。
        # 站立指令時 vx=vy=0, 箭頭長度(scale x)會是 0 而直接從畫面消失(看起來像 bug) ——
        # 改成箭頭改指向目標朝向(yaw_cmd)、長度固定為 min_arrow_len, 這樣才看得出
        # 「這個環境現在的指令是站在原地、面向這個方向」, 而不是單純沒有箭頭。
        min_arrow_len = 0.3
        goal_vel_xy = self._commands[:, :2]
        goal_speed = torch.norm(goal_vel_xy, dim=-1)
        vel_heading = torch.atan2(goal_vel_xy[:, 1], goal_vel_xy[:, 0])

        goal_heading = torch.where(self._is_standing, self._commands[:, 2], vel_heading)
        goal_len = torch.where(
            self._is_standing, torch.full_like(goal_speed, min_arrow_len), torch.clamp(goal_speed, min=min_arrow_len)
        )

        goal_scale = torch.stack([goal_len, torch.ones_like(goal_len), torch.ones_like(goal_len)], dim=-1)
        goal_quat = quat_from_euler_xyz(torch.zeros_like(goal_heading), torch.zeros_like(goal_heading), goal_heading)
        self.goal_vel_visualizer.visualize(translations=base_pos, orientations=goal_quat, scales=goal_scale)

        # ---------- 實際速度箭頭(尚未接上, 保留 visualizer 供之後接線) ----------

    def _sample_push(self) -> None:
        """每個 step 獨立抽這個 step 要不要推、往哪個方向、多大力(對齊論文的訓練中隨機推力,
        見 env_cfg.py 的 random_push_prob/min_push_force/max_push_force 說明)。水平面隨機
        方向, 只推骨盆。enable_random_push=False 時(play.py/keyboard_play.py 會這樣設)完全
        不推, 不然評估/互動測試時機器人會被無預警亂推, 看起來像「沒下指令自己亂動」。
        """
        if not self.cfg.enable_random_push:
            self._push_force_w.zero_()
            return
        push_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.random_push_prob
        push_mag = torch.zeros(self.num_envs, device=self.device).uniform_(
            self.cfg.min_push_force, self.cfg.max_push_force
        )
        push_angle = torch.zeros(self.num_envs, device=self.device).uniform_(0.0, 2 * torch.pi)
        push_dir = torch.stack([torch.cos(push_angle), torch.sin(push_angle), torch.zeros_like(push_angle)], dim=-1)
        self._push_force_w = push_dir * push_mag.unsqueeze(-1) * push_mask.float().unsqueeze(-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # clamp 原始 policy 輸出，避免異常大的取樣值平方後在 action_diff_reward 或
        # observation 回饋中溢位成 inf/nan（實際套用到物理的量已經有 max_delta clamp，
        # 這裡是保護原始值本身，跟 skrl_ppo_cfg.yaml 的 clip_actions 是兩道防線）
        self._actions = actions.clone().clamp(-10.0, 10.0)
        default_q = self.robot.data.default_joint_pos

        self._sample_push()

        raw_delta = self._action_scale[self._controlled_idx] * self._actions
        clipped_delta = torch.clamp(
            raw_delta,
            -self._max_delta[self._controlled_idx],
            self._max_delta[self._controlled_idx],
        )

        self._processed_actions = default_q.clone()
        self._processed_actions[:, self._controlled_idx] = default_q[:, self._controlled_idx] + clipped_delta

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._processed_actions)

        # 每個 physics substep 都重新施加(instantaneous_wrench_composer 只在單一 sim step
        # 有效, 每次呼叫都要重設), 讓推力整個 env.step()(=論文的 20ms)內持續不變, 不需要
        # 之後手動清除, 下個 env.step() 的 _sample_push() 決定新的力(絕大多數情況下是 0)
        # 就自然蓋掉了
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._push_force_w.unsqueeze(1),
            torques=torch.zeros_like(self._push_force_w).unsqueeze(1),
            body_ids=self._torso_body_id,
            is_global=True,
        )

    def _sample_commands(self, env_ids: torch.Tensor, base_yaw: torch.Tensor) -> None:
        """依論文的五類指令協定重新抽指令(reset 時、以及 episode 中途 _steps_until_resample
        倒數到 0 時都會呼叫這裡, 不是只有 reset 才抽)。base_yaw 是這些 env 目前實際的 yaw,
        目標朝向從這裡開始重新起算(中途重抽指令時也用當下朝向, 不是原本 reset 時的朝向)。
        """
        n = env_ids.numel()
        category = torch.randint(0, 5, (n,), device=self.device)
        axes_active = self._command_axes_active[category]  # (n, 3)

        vx_raw = torch.zeros(n, device=self.device).uniform_(self.cfg.min_vx, self.cfg.max_vx)
        vy_raw = torch.zeros(n, device=self.device).uniform_(self.cfg.min_vy, self.cfg.max_vy)
        yaw_rate_raw = torch.zeros(n, device=self.device).uniform_(self.cfg.min_yaw_rate, self.cfg.max_yaw_rate)

        self._commands[env_ids, 0] = vx_raw * axes_active[:, 0]
        self._commands[env_ids, 1] = vy_raw * axes_active[:, 1]
        self._commands[env_ids, 2] = base_yaw
        self._yaw_rate_cmd[env_ids] = yaw_rate_raw * axes_active[:, 2]
        self._is_standing[env_ids] = category == 0

        resample_s = torch.zeros(n, device=self.device).uniform_(
            self.cfg.min_resample_interval_s, self.cfg.max_resample_interval_s
        )
        self._steps_until_resample[env_ids] = torch.round(resample_s / self.step_dt).long()

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # heading error(目標 yaw - 目前 yaw, wrap 到 [-pi,pi])用 sin/cos 表示, 避免不連續
        base_yaw = yaw_from_quat(self.robot.data.root_quat_w)
        heading_err = torch.atan2(
            torch.sin(self._commands[:, 2] - base_yaw), torch.cos(self._commands[:, 2] - base_yaw)
        )

        # 腳掌接觸狀態(_get_rewards 每個 step 都會先跑, 已經是這個 step 算好的快取值)
        contact_state = self._last_foot_in_contact.float()

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,  # 3
                self.robot.data.root_ang_vel_b,  # 3
                self.robot.data.projected_gravity_b,  # 3
                self._commands,  # 3
                (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self._controlled_idx],  # 23
                self.robot.data.joint_vel[:, self._controlled_idx],  # 23
                self._actions,  # 23
                torch.stack([torch.sin(heading_err), torch.cos(heading_err)], dim=-1),  # 2
                contact_state,  # 2
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _update_contact_height_curriculum(self) -> None:
        """把 curriculum_total_timesteps 切成 3 等分, 每過 1/3 就晉級一次
        contact_height_threshold(依序套用 contact_height_threshold_stages 三個值), 不看
        表現、只看訓練進度 —— 簡單版課程, 見 env_cfg.py 的說明。
        """
        stages = self.cfg.contact_height_threshold_stages
        progress = self.common_step_counter / max(1, self.cfg.curriculum_total_timesteps)
        stage_idx = min(int(progress * 3), len(stages) - 1)
        self._contact_height_threshold = stages[stage_idx]

    def _get_rewards(self) -> torch.Tensor:
        self._update_contact_height_curriculum()

        base_quat = self.robot.data.root_quat_w
        base_yaw = yaw_from_quat(base_quat)
        lin_vel_b = self.robot.data.root_lin_vel_b

        # episode 中途指令重新抽樣(對齊論文每 2~6 秒重抽一次的協定, 不是整個 episode 只抽一次)
        self._steps_until_resample -= 1
        need_resample = torch.nonzero(self._steps_until_resample <= 0).squeeze(-1)
        if need_resample.numel() > 0:
            self._sample_commands(need_resample, base_yaw[need_resample])

        # 用 yaw rate 指令積分更新目標朝向(站立/剛重抽的 env 這步的 yaw_rate_cmd 就是新值)
        self._commands[:, 2] = torch.remainder(
            self._commands[:, 2] + self._yaw_rate_cmd * self.step_dt + torch.pi, 2 * torch.pi
        ) - torch.pi

        self._update_velocity_markers()

        # ---------- 腳掌接觸狀態: single foot contact(含 grace period) + feet air time ----------
        # 「著地」用腳踝高度判定(不用接觸力): 淨離地高度 = 腳踝 z - origin_height(腳踝到
        # 腳底的偏移量), 低於 contact_height_threshold 就算著地。這個 threshold 同時也是
        # 事實上的最低抬腳高度要求, 見 __init__ 裡的說明。
        feet_z = self.robot.data.body_pos_w[:, self._feet_ids, 2]  # (N, 2)
        ground_clearance = feet_z - self.cfg.origin_height
        foot_in_contact = ground_clearance < self._contact_height_threshold  # (N, 2)
        self._last_foot_in_contact = foot_in_contact

        single_contact_now = torch.sum(foot_in_contact, dim=1) == 1
        self._single_contact_hist = torch.roll(self._single_contact_hist, shifts=-1, dims=1)
        self._single_contact_hist[:, -1] = single_contact_now
        grace_credit = torch.any(self._single_contact_hist, dim=1)
        contact_credit = torch.where(self._is_standing, torch.ones_like(grace_credit), grace_credit)

        # 自己維護每隻腳的騰空計時器(見 __init__ 說明, 不能再靠 ContactSensor 內建的
        # 依接觸力算的 air-time 追蹤): first_contact = 這個 step 剛從騰空變成著地;
        # last_air_time = 這次著地前總共騰空了幾秒(擷取歸零前的累加值)。
        first_contact = (~self._prev_foot_in_contact) & foot_in_contact
        last_air_time = self._foot_air_time.clone()
        self._foot_air_time = torch.where(
            foot_in_contact, torch.zeros_like(self._foot_air_time), self._foot_air_time + self.step_dt
        )
        self._prev_foot_in_contact = foot_in_contact

        # 左右腳著地比例的 EMA(給 leg_symmetry_reward 用), 時間常數 leg_symmetry_tau
        ema_alpha = min(1.0, self.step_dt / self.cfg.leg_symmetry_tau)
        self._foot_contact_frac = (
            1.0 - ema_alpha
        ) * self._foot_contact_frac + ema_alpha * foot_in_contact.float()

        # ---------- 軀幹線加速度(有限差分, 給 base_acceleration_reward) ----------
        lin_acc = (lin_vel_b - self._prev_lin_vel_b) / self.step_dt
        self._prev_lin_vel_b = lin_vel_b.clone()

        # ---------- 腳掌相對骨盆(pelvis-yaw 座標)的 xy 位置, 給 feet_position_reward ----------
        cos_y, sin_y = torch.cos(base_yaw), torch.sin(base_yaw)
        feet_xy_w = self.robot.data.body_pos_w[:, self._feet_ids, :2]  # (N, 2, 2)
        delta_xy = feet_xy_w - self.robot.data.root_pos_w[:, :2].unsqueeze(1)  # (N, 2, 2)
        feet_local_x = cos_y.unsqueeze(-1) * delta_xy[..., 0] + sin_y.unsqueeze(-1) * delta_xy[..., 1]
        feet_local_y = -sin_y.unsqueeze(-1) * delta_xy[..., 0] + cos_y.unsqueeze(-1) * delta_xy[..., 1]
        feet_local_xy = torch.stack([feet_local_x, feet_local_y], dim=-1)  # (N, 2, 2)

        # ---------- 左右腳重力在各自本體座標的投影(~roll+pitch), 給 feet_orientation_reward ----------
        feet_quat = self.robot.data.body_quat_w[:, self._feet_ids]  # (N, 2, 4)
        feet_quat_flat = feet_quat.reshape(-1, 4)
        gravity_vec = self._gravity_vec.expand(feet_quat_flat.shape[0], 3)
        feet_gravity_b = quat_apply_inverse(feet_quat_flat, gravity_vec).reshape(self.num_envs, 2, 3)

        # ---------- 左右腳 yaw 相對軀幹 yaw 的誤差(不轉彎時 feet_orientation_reward 才會用) ----------
        feet_yaw = yaw_from_quat(feet_quat_flat).reshape(self.num_envs, 2)
        feet_yaw_err = torch.atan2(
            torch.sin(feet_yaw - base_yaw.unsqueeze(-1)), torch.cos(feet_yaw - base_yaw.unsqueeze(-1))
        )

        # ---------- 「轉彎中」mask: 對齊論文用「有沒有下 cyaw 轉向指令」判斷, 不是量測到的
        # heading error(turning_yaw_error_threshold 這個舊機制已經拿掉) ----------
        is_turning = torch.abs(self._yaw_rate_cmd) > 1e-6

        # ---------- 呼叫各項 reward 公式(都在 reward_terms.py) ----------
        rewards = {
            "lin_vel_tracking": lin_vel_xy_tracking_reward(
                self._commands[:, :2],
                lin_vel_b[:, :2],
                self._is_standing,
                self.cfg.lin_vel_tracking_coeff,
                self.cfg.lin_vel_tracking_reward_scale,
            ),
            "yaw_tracking": yaw_tracking_reward(
                base_quat, self._commands[:, 2], self.cfg.yaw_tracking_coeff, self.cfg.yaw_tracking_reward_scale
            ),
            "roll_pitch_tracking": roll_pitch_tracking_reward(
                base_quat, self.cfg.roll_pitch_tracking_coeff, self.cfg.roll_pitch_tracking_reward_scale
            ),
            "single_foot_contact": single_foot_contact_reward(
                contact_credit, self.cfg.single_foot_contact_reward_scale
            ),
            "leg_symmetry": leg_symmetry_reward(
                self._foot_contact_frac,
                ~self._is_standing,
                self.cfg.leg_symmetry_coeff,
                self.cfg.leg_symmetry_reward_scale,
            ),
            "stand_still": stand_still_reward(
                self.robot.data.joint_vel,
                self._controlled_idx,
                self._is_standing,
                self.cfg.stand_still_coeff,
                self.cfg.stand_still_reward_scale,
            ),
            "base_height": base_height_reward(
                self.robot.data.root_state_w[:, 2],
                self.cfg.target_torso_height,
                self.cfg.base_height_coeff,
                self.cfg.base_height_reward_scale,
            ),
            "feet_air_time": feet_air_time_reward(
                last_air_time,
                first_contact,
                self.cfg.feet_air_time_threshold,
                self._is_standing,
                self.cfg.feet_air_time_reward_scale,
            ),
            "feet_orientation": feet_orientation_reward(
                feet_gravity_b[:, :, :2],
                feet_yaw_err,
                self.cfg.feet_orientation_coeff,
                is_turning,
                self.cfg.feet_orientation_reward_scale,
            ),
            "feet_position": feet_position_reward(
                feet_local_xy,
                self._nominal_feet_local_xy,
                self.cfg.feet_position_coeff,
                self._is_standing,
                self.cfg.feet_position_reward_scale,
            ),
            "arm_deviation": arm_deviation_reward(
                self.robot.data.joint_pos,
                self.robot.data.default_joint_pos,
                self._arm_idx,
                self.cfg.arm_deviation_coeff,
                self.cfg.arm_deviation_reward_scale,
            ),
            "base_acceleration": base_acceleration_reward(
                lin_acc, self.cfg.base_acceleration_coeff, self.cfg.base_acceleration_reward_scale
            ),
            "action_diff": action_diff_reward(
                self._actions, self._previous_actions, self.cfg.action_diff_coeff, self.cfg.action_diff_reward_scale
            ),
            "torque": torque_reward(
                self.robot.data.applied_torque,
                self.robot.data.joint_effort_limits,
                self._controlled_idx,
                self.cfg.torque_coeff,
                self.cfg.torque_reward_scale,
            ),
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
        # 用跟 episode 中途重抽指令一樣的邏輯(五類指令均勻抽一類), 目標朝向從重置當下的
        # 朝向開始起算, 詳見 _sample_commands()。
        self._sample_commands(env_ids, yaw_from_quat(default_root_state[:, 3:7]))

        # ------------ 其餘 per-env 狀態重置 ------------
        self._single_contact_hist[env_ids] = False
        self._last_foot_in_contact[env_ids] = False
        self._prev_foot_in_contact[env_ids] = True  # 重置姿態雙腳都貼地
        self._foot_air_time[env_ids] = 0.0
        self._prev_lin_vel_b[env_ids] = 0.0
        self._foot_contact_frac[env_ids] = 0.5

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        # skrl's logger only forwards tensor-valued entries from extras["log"] to TensorBoard;
        # plain python int/float values here get silently dropped (no error, just never shows up)
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).float()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).float()
        extras["Curriculum/contact_height_threshold"] = torch.tensor(self._contact_height_threshold, device=self.device)
        self.extras["log"].update(extras)
