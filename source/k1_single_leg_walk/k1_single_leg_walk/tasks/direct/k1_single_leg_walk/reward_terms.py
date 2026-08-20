# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 single-leg-walk 的每一項 reward/penalty 公式(含 scale)。

只吃/回傳 torch.Tensor 或純量, 只依賴 torch, 不 import 這個套件裡任何其他模組
(尤其不 import isaaclab, 這樣才能被 scripts/reward_curves.py 用 importlib 直接以
檔案路徑載入, 不需要 Isaac Sim)。四元數 -> 純量(yaw 差、重力投影分量等)的轉換屬於
「讀取機器人狀態」, 留在 k1_single_leg_walk_env.py; 這裡只放「純量狀態 -> reward」
的公式本身, 這樣 _get_rewards() 就只需要組出輸入、呼叫這裡的函式。

每個 *_reward / *_penalty 函式都回傳「已乘上 scale」的 (num_envs,) reward tensor,
呼叫端(env.py 或 reward_curves.py)只需要再乘上 step_dt 做累加/畫圖。

lin_vel_tracking_reward / ang_vel_tracking_reward / foot_height_tracking_reward /
joint_deviation_penalty / feet_orientation_penalty / close_feet_xy_penalty /
torso_orientation_penalty / ang_vel_xy_penalty / action_rate_penalty / alive_reward
這 10 項的公式跟預設 scale 對齊 holosoma(../../holosoma)的 T1/G1 雙足 locomotion reward
preset(見 holosoma/src/holosoma/holosoma/config_values/loco/{t1,g1}/reward.py 跟
holosoma/src/holosoma/holosoma/managers/reward/terms/locomotion.py)。holosoma 的兩個
preset 都沒有用到 termination_penalty / torso_height_penalty / joint_vel_penalty /
joint_acc_penalty 這幾項(靠夠強的 orientation + action_rate 懲罰撐住,不是靠這些),
所以這幾項在 env_cfg.py 裡預設 scale=0.0,函式留著但不是 holosoma baseline 的一部分。

feet_orientation_penalty / close_feet_xy_penalty 這兩項「規定腳掌姿態細節」的懲罰在
env_cfg.py 裡改成預設關閉, 參考 Revisiting Reward Design and Evaluation for Robust
Humanoid Standing and Walking(arXiv:2404.19173)的論點: 過度規定性(overly prescriptive)
的懲罰會一條一條砍掉可行解空間, policy 可能被逼到只剩奇怪姿勢能同時滿足所有規定。
"""

from __future__ import annotations

import torch


# --- 共用數學核心 ---
def exp_kernel(error_sq: torch.Tensor, std: float) -> torch.Tensor:
    """速度追蹤 / 腳掌高度追蹤共用的指數核: exp(-error_sq / std)。"""
    return torch.exp(-error_sq / std)


def expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier swing/stance profile."""

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)  # [0, 1)

    # 前半週期: 真正的支撐期, 高度固定為 0
    stance = torch.zeros_like(x)

    # 後半週期: 單獨在這半段裡完成「上升 -> 下降」的完整弧線
    t = torch.clamp((x - 0.5) * 2, 0.0, 1.0)  # 把 [0.5, 1) 重新縮放成 [0, 1]
    rising = cubic_bezier_interpolation(torch.zeros_like(t), torch.full_like(t, swing_height), 2 * t)
    falling = cubic_bezier_interpolation(torch.full_like(t, swing_height), torch.zeros_like(t), 2 * t - 1)
    swing = torch.where(t <= 0.5, rising, falling)

    return torch.where(x <= 0.5, stance, swing)


# --- 1. 線速度 / 角速度追蹤 ---
def lin_vel_tracking_reward(
    commands: torch.Tensor, root_lin_vel_b: torch.Tensor, std: float, scale: float
) -> torch.Tensor:
    error = torch.sum(torch.square(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    return exp_kernel(error, std) * scale


def ang_vel_tracking_reward(
    commands: torch.Tensor, root_ang_vel_b: torch.Tensor, std: float, scale: float
) -> torch.Tensor:
    error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    return exp_kernel(error, std) * scale


# --- 2. 腳掌高度追蹤 ---
def foot_height_tracking_reward(
    feet_pos_z: torch.Tensor,
    origin_height: float,
    gait_phase: torch.Tensor,
    swing_height: float,
    sigma: float,
    scale: float,
) -> torch.Tensor:
    """feet_pos_z / gait_phase: shape (num_envs, 2), 欄位順序為 [left, right]。"""
    foot_z_left = feet_pos_z[:, 0] - origin_height
    foot_z_right = feet_pos_z[:, 1] - origin_height
    rz_left = expected_foot_height(gait_phase[:, 0], swing_height)
    rz_right = expected_foot_height(gait_phase[:, 1], swing_height)
    track_error = torch.square(foot_z_left - rz_left) + torch.square(foot_z_right - rz_right)
    return exp_kernel(track_error, sigma) * scale


# --- 3. 預設姿態懲罰 ---
def joint_deviation_penalty(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    controlled_idx: torch.Tensor,
    pose_weights: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    pose_error = torch.square((joint_pos - default_joint_pos)[:, controlled_idx])
    weighted_pose_error = pose_error * pose_weights[controlled_idx]
    return torch.sum(weighted_pose_error, dim=1) * scale


# --- 4. 腳掌相關懲罰(參考 holosoma penalty_feet_ori / penalty_close_feet_xy) ---
def feet_orientation_penalty(feet_gravity_xy: torch.Tensor, scale: float) -> torch.Tensor:
    """量測腳掌相對地面的平整度(pitch+roll), 不是左右腳 yaw 差。

    feet_gravity_xy: 左右腳重力在各自本體座標的 xy 分量, shape (num_envs, 2 feet, 2)。
    每隻腳先取 xy 分量的 norm(~sin(傾斜角)), 兩腳相加。
    """
    return torch.sum(torch.norm(feet_gravity_xy, dim=-1), dim=-1) * scale


def close_feet_xy_penalty(feet_lateral: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
    """腳掌側向間距小於 threshold 時固定懲罰(二元, 非連續斜坡)。"""
    return (feet_lateral < threshold).float() * scale


# --- 5. 存活獎勵 ---
def alive_reward(num_envs: int, device: torch.device, scale: float) -> torch.Tensor:
    return torch.ones(num_envs, device=device) * scale


# --- 6. 軀幹姿態懲罰(參考 holosoma penalty_orientation) ---
def torso_orientation_penalty(torso_gravity_xy: torch.Tensor, scale: float) -> torch.Tensor:
    """torso_gravity_xy: 軀幹重力在本體 xy 分量, shape (num_envs, 2)。"""
    return torch.sum(torch.square(torso_gravity_xy), dim=1) * scale


def ang_vel_xy_penalty(torso_ang_vel_xy: torch.Tensor, scale: float) -> torch.Tensor:
    """torso_ang_vel_xy: 軀幹角速度在本體 xy 分量, shape (num_envs, 2)。"""
    return torch.sum(torch.square(torso_ang_vel_xy), dim=1) * scale


def torso_height_penalty(torso_height: torch.Tensor, target_height: float, scale: float) -> torch.Tensor:
    """torso_height: 軀幹(root)世界座標 z, shape (num_envs,)。

    torso_orientation_penalty 只管傾斜角度, 不管高度 —— 蹲低但軀幹依然直立時該項是 0。
    這裡補上高度本身的追蹤, 讓「蹲低」在觸發 termination 之前就先有平滑的梯度懲罰,
    而不是只靠 termination 這個硬門檻。target_height 建議設成單腳站立的自然高度
    (可以印 default_root_state[:, 2] 來抓這個值), 不是 min_torso_height 那個摔倒門檻。
    """
    return torch.square(torso_height - target_height) * scale


# --- 7. 動作變化率懲罰 ---
def action_rate_penalty(actions: torch.Tensor, previous_actions: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sum(torch.square(actions - previous_actions), dim=1) * scale


# --- 7b. 關節速度 / 加速度懲罰(抑制關節抖動, 只看受控關節) ---
def joint_vel_penalty(joint_vel: torch.Tensor, controlled_idx: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sum(torch.square(joint_vel[:, controlled_idx]), dim=1) * scale


def joint_acc_penalty(joint_acc: torch.Tensor, controlled_idx: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sum(torch.square(joint_acc[:, controlled_idx]), dim=1) * scale


# --- 8. 提早終止懲罰 ---
def termination_penalty(reset_terminated: torch.Tensor, scale: float) -> torch.Tensor:
    return reset_terminated.float() * scale
