# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 single-leg-walk 的每一項 reward 公式(含 scale)。

只吃/回傳 torch.Tensor 或純量, 只依賴 torch, 不 import 這個套件裡任何其他模組
(尤其不 import isaaclab, 這樣才能被 scripts/reward_curves.py 用 importlib 直接以
檔案路徑載入, 不需要 Isaac Sim)。四元數 -> 純量的轉換(取 body 姿態、算重力投影等)
屬於「讀取機器人狀態」, 留在 k1_single_leg_walk_env.py; 這裡只放「純量/tensor 狀態
-> reward」的公式本身。

整套 reward 架構對齊 Revisiting Reward Design and Evaluation for Robust Humanoid
Standing and Walking(arXiv:2404.19173)的 Reward Term Definition/Weighting 表格,
逐項數字/公式都照論文原文抄, 不是本專案自己抓的起點:

1. 三項核心指令追蹤(xy 速度、yaw 朝向、roll/pitch 朝向) —— 論文指出只用這三項會學出
   雙腳同時跳躍前進的怪異步態(滿足指令但不是想要的走路方式)。
2. 單腳接觸(single foot contact) —— 論文比較五種抑制跳躍步態的方法後, 認為這項最可靠
   也最不限制行為, 取代舊版以相位時鐘追蹤腳掌高度曲線的做法。
3. 風格 / sim-to-real 輔助項(base height、feet air time、feet orientation、
   feet position、arm、base acceleration、action difference、torque)。

論文表格裡的規律: 大多數輔助項用的是「絕對值/範數誤差」(線性), 不是「誤差平方」;
而且好幾項(feet air time、feet position、feet orientation)在站立/不轉彎以外的情況
是給「固定 1 分」而不是「關掉(=0)」——這兩點都跟舊版(本檔案作者自己抓起點時)的做法
不同, 這次改版已經全部照論文改掉。

唯一例外: feet_orientation 的核函數係數在原始 OCR 掃描裡缺字看不清楚, 沿用之前抓的
5.0 當佔位數字, 不是論文原文確切數字, 訓練時可能需要再調。
"""

from __future__ import annotations

import torch


# --- 四元數工具(w, x, y, z 慣例, 對齊 isaaclab 的 quat 慣例) ---
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_dist(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """兩個四元數的「距離」: 1 - <q1,q2>^2, 值域 [0,1]。

    對小夾角來說約等於 sin^2(angle/2) ≈ (angle/2)^2, 是 wrap-safe 的角度誤差度量,
    這也是論文公式裡係數(5 / 300 / 30)偏大的原因 —— 係數乘的是這個「小量」, 不是
    直接乘弧度角。
    """
    dot = torch.sum(q1 * q2, dim=-1)
    return 1.0 - dot * dot


def yaw_twist_quat(q: torch.Tensor) -> torch.Tensor:
    """從姿態四元數 q 取出繞世界 +z 軸的 yaw(twist)分量(swing-twist 分解)。"""
    w, x, y, z = q.unbind(-1)
    twist = torch.stack([w, torch.zeros_like(x), torch.zeros_like(y), z], dim=-1)
    norm = torch.linalg.norm(twist, dim=-1, keepdim=True).clamp_min(1e-6)
    return twist / norm


def swing_quat(q: torch.Tensor) -> torch.Tensor:
    """從姿態四元數 q 取出「去掉 yaw 之後」的傾斜(swing)分量。"""
    twist = yaw_twist_quat(q)
    return quat_mul(q, quat_conjugate(twist))


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """姿態四元數 q 的純量 yaw 角(rad), 用於算 heading error(給 obs 用, 不進 reward 公式本身)。"""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --- 1. 核心指令追蹤(三項, weight 0.15 / 0.1 / 0.2) ---
def lin_vel_xy_tracking_reward(
    commands_xy: torch.Tensor,
    actual_xy: torch.Tensor,
    is_standing: torch.Tensor,
    coeff: float,
    scale: float,
) -> torch.Tensor:
    """站立指令: exp(-coeff * ||v_xy - c_xy||)(誤差不平方); 非站立: exp(-coeff * ||v_xy - c_xy||^2)。

    論文特別指出: 站立指令下如果誤差平方, exp 核在誤差很小時梯度太平, 不足以壓住
    殘餘漂移速度, 所以站立時改用誤差不平方的版本(下降得比較快), 非站立時才平方。
    """
    error = torch.norm(commands_xy - actual_xy, dim=-1)
    linear_track = torch.exp(-coeff * error)
    squared_track = torch.exp(-coeff * error * error)
    return torch.where(is_standing, linear_track, squared_track) * scale


def yaw_tracking_reward(base_quat: torch.Tensor, yaw_cmd: torch.Tensor, coeff: float, scale: float) -> torch.Tensor:
    """exp(-coeff * quat_dist(yaw, c_yaw)); yaw 用 swing-twist 分解取出的純 yaw 四元數比較。"""
    twist = yaw_twist_quat(base_quat)
    half = 0.5 * yaw_cmd
    cmd_twist = torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1
    )
    return torch.exp(-coeff * quat_dist(twist, cmd_twist)) * scale


def roll_pitch_tracking_reward(base_quat: torch.Tensor, coeff: float, scale: float) -> torch.Tensor:
    """exp(-coeff * quat_dist(rp, c_rp)); c_rp 固定為「直立」(identity), 不額外開放傾斜指令。"""
    swing = swing_quat(base_quat)
    identity = torch.zeros_like(swing)
    identity[..., 0] = 1.0
    return torch.exp(-coeff * quat_dist(swing, identity)) * scale


# --- 2. 單腳接觸(取代相位時鐘, weight 0.1) ---
def single_foot_contact_reward(has_credit: torch.Tensor, scale: float) -> torch.Tensor:
    """has_credit 由呼叫端(env.py)算好: 非站立指令時「過去 grace period 內出現過恰好單腳
    著地」就是 True; 站立指令時固定 True(不要求雙腳都著地 —— 論文指出這會懲罰機器人
    被推撞時抬腳回穩的必要動作)。這裡只是套上 scale。
    """
    return has_credit.float() * scale


# --- 2b. 兩腳角色對稱(補的, 不在論文原文裡) ---
def leg_symmetry_reward(contact_frac: torch.Tensor, gate: torch.Tensor, coeff: float, scale: float) -> torch.Tensor:
    """contact_frac: 左右腳「最近一段時間著地時間比例」的 EMA, shape (num_envs, 2 feet)。

    single_foot_contact_reward 只看「當下恰好一隻腳著地」, 不管是哪一隻——固定一腳整場
    貼地拖著走、另一腳負責所有抬腳動作, 一樣能拿到接近滿分, 完全不需要真正輪流交替步態。
    這項懲罰兩腳著地比例差太多, 逼兩腳角色對稱; gate 通常是「非站立指令」(站立時兩腳
    角色本來就不需要對稱)。
    """
    diff_sq = torch.square(contact_frac[:, 0] - contact_frac[:, 1])
    return torch.exp(-coeff * diff_sq) * gate.float() * scale


# --- 2c. 站立時的靜止獎勵(補的, 不在論文原文裡) ---
def stand_still_reward(
    joint_vel: torch.Tensor, controlled_idx: torch.Tensor, gate: torch.Tensor, coeff: float, scale: float
) -> torch.Tensor:
    """站立指令時, 鼓勵受控關節速度趨近 0(真正站定不動); gate 通常是 is_standing,
    非站立指令時關閉, 不限制走路動作。
    """
    err_sq = torch.sum(torch.square(joint_vel[:, controlled_idx]), dim=1)
    return torch.exp(-coeff * err_sq) * gate.float() * scale


# --- 3a. Base height(weight 0.05) ---
def base_height_reward(root_height: torch.Tensor, target_height: float, coeff: float, scale: float) -> torch.Tensor:
    """exp(-coeff * |pz - c_h|), 線性誤差(不平方)。"""
    return torch.exp(-coeff * torch.abs(root_height - target_height)) * scale


# --- 3b. Feet air time(weight 1.0, 全套裡唯一的稀疏 reward, 故權重特別高) ---
def feet_air_time_reward(
    last_air_time: torch.Tensor,
    first_contact: torch.Tensor,
    threshold: float,
    is_standing: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """站立指令時固定給滿分(=1); 非站立時, 觸地(first_contact)瞬間才有值, 等於
    (該次騰空秒數 - threshold), 其餘時間是 0。

    threshold 以下(步頻太快、騰空太短)會被罰負值, threshold 以上才是正獎勵 —— 用騰空
    秒數本身「抵銷」固定罰分, 防止步頻過高。站立指令時沒有腳步事件的概念, 論文直接給
    滿分(不是關掉/0), 跟其他「非站立才生效」的項不同, 要注意別搞混。
    """
    air_time = torch.sum((last_air_time - threshold) * first_contact.float(), dim=1)
    return torch.where(is_standing, torch.ones_like(air_time), air_time) * scale


# --- 3c. Feet orientation(weight 0.05) ---
def feet_orientation_reward(
    feet_gravity_xy: torch.Tensor,
    feet_yaw_err: torch.Tensor,
    coeff: float,
    is_turning: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """feet_gravity_xy: 左右腳重力在各自本體座標的 xy 分量(~roll+pitch 傾角), shape
    (num_envs, 2 feet, 2)。feet_yaw_err: 左右腳 yaw 相對軀幹 yaw 的誤差(rad,已 wrap),
    shape (num_envs, 2 feet)。

    is_turning 是「有沒有下轉向指令(|yaw_rate_cmd| > 0)」, 不是量測到的 heading error:
    轉彎中(is_turning=True)腳掌本來就需要傾斜/外八才能轉向, 只檢查 roll+pitch(rp) 平不
    平; 沒有轉向指令時(直走或站立), 額外要求腳掌 yaw 也要對齊軀幹朝向(rpy), 不能內外八。

    註: coeff 是佔位數字(5.0), 論文原文這項的核函數係數在 OCR 掃描裡缺字看不清楚。
    """
    rp_err = torch.sum(torch.abs(feet_gravity_xy), dim=(-1, -2))
    rpy_err = rp_err + torch.sum(torch.abs(feet_yaw_err), dim=-1)
    err = torch.where(is_turning, rp_err, rpy_err)
    return torch.exp(-coeff * err) * scale


# --- 3d. Feet position(weight 0.05) ---
def feet_position_reward(
    feet_local_xy: torch.Tensor, nominal_xy: torch.Tensor, coeff: float, is_standing: torch.Tensor, scale: float
) -> torch.Tensor:
    """feet_local_xy / nominal_xy: 左右腳相對骨盆(pelvis-yaw 座標)的 xy 位置, shape (num_envs, 2, 2)。

    只在站立指令時追蹤 exp(-coeff * |Δp|)(線性誤差), 鬆散地把腳掌拉回 nominal 站姿,
    避免站著的時候出現腳掌打結、外八到誇張角度等怪站姿; 非站立指令固定給滿分(不是
    關掉/0), 走路時完全不管, 讓步態自由發展。
    """
    err = torch.sum(torch.abs(feet_local_xy - nominal_xy), dim=(-1, -2))
    standing_track = torch.exp(-coeff * err)
    return torch.where(is_standing, standing_track, torch.ones_like(err)) * scale


# --- 3e. Arm(weight 0.03, 鬆散限制, 主要防自碰撞) ---
def arm_deviation_reward(
    joint_pos: torch.Tensor, default_joint_pos: torch.Tensor, arm_idx: torch.Tensor, coeff: float, scale: float
) -> torch.Tensor:
    """exp(-coeff * ||Δθ_arm||), 範數(不平方)。"""
    err = torch.norm((joint_pos - default_joint_pos)[:, arm_idx], dim=-1)
    return torch.exp(-coeff * err) * scale


# --- 3f. Base acceleration(weight 0.1, 抑制軀幹晃動) ---
def base_acceleration_reward(lin_acc: torch.Tensor, coeff: float, scale: float) -> torch.Tensor:
    """exp(-coeff * |b_xyz|); 只看軀幹線加速度(不含角加速度), 範數不平方。"""
    return torch.exp(-coeff * torch.norm(lin_acc, dim=-1)) * scale


# --- 3g. Action difference(weight 0.02) ---
def action_diff_reward(
    actions: torch.Tensor, previous_actions: torch.Tensor, coeff: float, scale: float
) -> torch.Tensor:
    """exp(-coeff * ||Δa||), 範數(不平方)。"""
    err = torch.norm(actions - previous_actions, dim=-1)
    return torch.exp(-coeff * err) * scale


# --- 3h. Torque(weight 0.02, 抑制扭矩使用) ---
def torque_reward(
    applied_torque: torch.Tensor,
    effort_limits: torch.Tensor,
    controlled_idx: torch.Tensor,
    coeff: float,
    scale: float,
) -> torch.Tensor:
    """exp(-coeff * mean(|τ_motor| / τ_max)); 每顆受控馬達的扭矩先除以自己的額定上限
    (effort_limits)做正規化, 再取平均 —— 不是直接對原始 Nm 數值取平方和, 這樣不同
    關節群組(腿部 96.9Nm vs 手臂 47.3Nm)才不會因為量級不同而被不成比例地懲罰。
    """
    ratio = torch.abs(applied_torque[:, controlled_idx]) / effort_limits[:, controlled_idx]
    return torch.exp(-coeff * torch.mean(ratio, dim=-1)) * scale
