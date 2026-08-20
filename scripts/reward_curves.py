"""視覺化 k1_single_leg_walk_env.py 中每一項 reward/penalty 公式的形狀與量級。

純 torch/matplotlib 腳本, 不需要 Isaac Sim, 直接執行:
    python scripts/reward_curves.py

每一項 reward/penalty 直接呼叫 reward_terms.py 裡跟訓練共用的同一份函式(用 importlib
以檔案路徑載入; 不能寫成 `import k1_single_leg_walk...`, 因為套件的 tasks/__init__.py
會透過 isaaclab_tasks.utils.import_packages 自動 import 整個套件, 進而拉進 isaaclab_tasks),
這裡只負責合成輸入 tensor 餵給那些函式, 不重新實作任何一項公式。

下方 DEFAULT_CFG 的 scale/std/sigma 數值仍需與 k1_single_leg_walk_env_cfg.py 手動保持同步
(這些是純數值, env_cfg.py 本身混了 isaaclab 型別的欄位, 沒辦法整份被獨立腳本 import)。
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider

_REWARD_TERMS_PATH = (
    Path(__file__).resolve().parent.parent
    / "source"
    / "k1_single_leg_walk"
    / "k1_single_leg_walk"
    / "tasks"
    / "direct"
    / "k1_single_leg_walk"
    / "reward_terms.py"
)


def _load_reward_terms():
    spec = importlib.util.spec_from_file_location("k1_reward_terms", _REWARD_TERMS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load_reward_terms()

# --- 對應 k1_single_leg_walk_env_cfg.py 的預設值 ---
DEFAULT_CFG = {
    # reward scales(10 項對齊 holosoma T1/G1 preset, 其餘 4 項是額外選配, 預設關閉)
    "lin_vel_tracking_reward_scale": 2.0,
    "ang_vel_tracking_reward_scale": 1.5,
    "foot_height_reward_scale": 5.0,
    "joint_deviation_penalty_scale": -0.5,
    "feet_ori_penalty_scale": 0.0,
    "close_feet_xy_penalty_scale": 0.0,
    "alive_reward_scale": 1.0,
    "torso_orientation_penalty_scale": -10.0,
    "ang_vel_xy_penalty_scale": -1.0,
    "action_rate_penalty_scale": -2.0,
    "torso_height_penalty_scale": 0.0,
    "joint_vel_penalty_scale": 0.0,
    "joint_acc_penalty_scale": 0.0,
    "termination_penalty_scale": 0.0,
    # kernel std / 目標值
    "lin_vel_std": 0.25,
    "ang_vel_std": 0.25,
    "gait_tracking_sigma": 0.008,
    "close_feet_threshold": 0.15,
    "max_torso_tilt": 0.45,
    "swing_height": 0.09,
    "target_torso_height": 0.78,
    # pose_weights_map 的權重範圍(knee 最低 0.2, 腰部以上最高 50.0)
    "pose_weight_min": 0.2,
    "pose_weight_max": 50.0,
    # step_dt = decimation * sim_dt = 2 * (1/200)
    "step_dt": 2 * (1 / 200),
}

# 每項在小圖裡的 x 軸物理範圍上限(x_norm=1 對應到這個值), combined 圖沿用同一套範圍
X_RANGE = {
    "lin_vel": 2.0,  # m/s
    "ang_vel": 3.0,  # rad/s
    "foot_height": 0.05,  # m
    "joint_dev(max)": 1.0,  # rad
    "feet_ori": 1.0,  # sin(單腳傾角), 對稱, 取 [0,1] 代表
    "close_feet": 0.3,  # m
    "alive": 1.0,  # 常數項, x 軸無意義
    "torso_ori": DEFAULT_CFG["max_torso_tilt"] * 1.5,  # rad
    "ang_vel_xy": 5.0,  # rad/s
    "action_rate": 3.0,  # 單一動作維度變化量
    "joint_vel": 10.0,  # rad/s, 單一關節
    "joint_acc": 200.0,  # rad/s^2, 單一關節
    "torso_height": 0.3,  # m, 偏離 target_torso_height 的量
}

REWARD_COLOR = "#2a9d4a"
PENALTY_COLOR = "#d64a2a"

X_NORM = torch.linspace(0.0, 1.0, 300)
N = X_NORM.shape[0]
_ZEROS_N3 = torch.zeros(N, 3)


def term_curves(cfg: dict) -> dict[str, torch.Tensor]:
    """把每一項的合成輸入餵進 reward_terms.py 的對應函式, 回傳 x_norm=[0,1] 上的值。

    輸入都刻意合成成「只有這一項的誤差來源不為零」, 這樣算出來的曲線才會跟小圖裡
    單一變數的 x 軸一一對應。
    """
    lin_vel = _ZEROS_N3.clone()
    lin_vel[:, 0] = X_NORM * X_RANGE["lin_vel"]

    ang_vel = _ZEROS_N3.clone()
    ang_vel[:, 2] = X_NORM * X_RANGE["ang_vel"]

    feet_pos_z = torch.zeros(N, 2)
    feet_pos_z[:, 0] = X_NORM * X_RANGE["foot_height"]
    gait_phase0 = torch.zeros(N, 2)  # phase=0 時兩腳期望高度都是 0(支撐期邊界)

    joint_pos = torch.zeros(N, 1)
    joint_pos[:, 0] = X_NORM * X_RANGE["joint_dev(max)"]
    default_joint_pos = torch.zeros(N, 1)
    controlled_idx = torch.tensor([0])
    pose_weight_max = torch.tensor([cfg["pose_weight_max"]])

    feet_lateral = X_NORM * X_RANGE["close_feet"]

    # 只讓其中一隻腳的 x 分量偏離 0(單腳傾斜), 另一隻腳跟 y 分量維持 0
    feet_gravity_xy = torch.zeros(N, 2, 2)
    feet_gravity_xy[:, 0, 0] = X_NORM * X_RANGE["feet_ori"]

    torso_gravity_xy = torch.zeros(N, 2)
    torso_gravity_xy[:, 0] = torch.sin(X_NORM * X_RANGE["torso_ori"])

    torso_ang_vel_xy = torch.zeros(N, 2)
    torso_ang_vel_xy[:, 0] = X_NORM * X_RANGE["ang_vel_xy"]

    actions = torch.zeros(N, 1)
    actions[:, 0] = X_NORM * X_RANGE["action_rate"]
    previous_actions = torch.zeros(N, 1)

    joint_vel = torch.zeros(N, 1)
    joint_vel[:, 0] = X_NORM * X_RANGE["joint_vel"]

    joint_acc = torch.zeros(N, 1)
    joint_acc[:, 0] = X_NORM * X_RANGE["joint_acc"]

    torso_height = cfg["target_torso_height"] + X_NORM * X_RANGE["torso_height"]

    return {
        "lin_vel": rt.lin_vel_tracking_reward(
            _ZEROS_N3, lin_vel, cfg["lin_vel_std"], cfg["lin_vel_tracking_reward_scale"]
        ),
        "ang_vel": rt.ang_vel_tracking_reward(
            _ZEROS_N3, ang_vel, cfg["ang_vel_std"], cfg["ang_vel_tracking_reward_scale"]
        ),
        "foot_height": rt.foot_height_tracking_reward(
            feet_pos_z,
            0.0,
            gait_phase0,
            cfg["swing_height"],
            cfg["gait_tracking_sigma"],
            cfg["foot_height_reward_scale"],
        ),
        "joint_dev(max)": rt.joint_deviation_penalty(
            joint_pos, default_joint_pos, controlled_idx, pose_weight_max, cfg["joint_deviation_penalty_scale"]
        ),
        "feet_ori": rt.feet_orientation_penalty(feet_gravity_xy, cfg["feet_ori_penalty_scale"]),
        "close_feet": rt.close_feet_xy_penalty(
            feet_lateral, cfg["close_feet_threshold"], cfg["close_feet_xy_penalty_scale"]
        ),
        "alive": rt.alive_reward(N, torch.device("cpu"), cfg["alive_reward_scale"]),
        "torso_ori": rt.torso_orientation_penalty(torso_gravity_xy, cfg["torso_orientation_penalty_scale"]),
        "ang_vel_xy": rt.ang_vel_xy_penalty(torso_ang_vel_xy, cfg["ang_vel_xy_penalty_scale"]),
        "action_rate": rt.action_rate_penalty(actions, previous_actions, cfg["action_rate_penalty_scale"]),
        "joint_vel": rt.joint_vel_penalty(joint_vel, controlled_idx, cfg["joint_vel_penalty_scale"]),
        "joint_acc": rt.joint_acc_penalty(joint_acc, controlled_idx, cfg["joint_acc_penalty_scale"]),
        "torso_height": rt.torso_height_penalty(
            torso_height, cfg["target_torso_height"], cfg["torso_height_penalty_scale"]
        ),
    }


TERM_SCALE_KEY = {
    "lin_vel": "lin_vel_tracking_reward_scale",
    "ang_vel": "ang_vel_tracking_reward_scale",
    "foot_height": "foot_height_reward_scale",
    "joint_dev(max)": "joint_deviation_penalty_scale",
    "feet_ori": "feet_ori_penalty_scale",
    "close_feet": "close_feet_xy_penalty_scale",
    "alive": "alive_reward_scale",
    "torso_ori": "torso_orientation_penalty_scale",
    "ang_vel_xy": "ang_vel_xy_penalty_scale",
    "torso_height": "torso_height_penalty_scale",
    "action_rate": "action_rate_penalty_scale",
    "joint_vel": "joint_vel_penalty_scale",
    "joint_acc": "joint_acc_penalty_scale",
}

# --- 建立圖表: 上方 4x4 為各項獨立公式形狀(14 格用到, 2 格留空), 最下面一整排是量級比較圖 ---
curves0 = term_curves(DEFAULT_CFG)

fig = plt.figure(figsize=(18, 24))
gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 3.0], hspace=0.7, wspace=0.35, bottom=0.18, top=0.95)
axes = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(4)]
combined_ax = fig.add_subplot(gs[4, :])
fig.suptitle("K1 Single Leg Walk - Reward / Penalty Terms", fontsize=14)

lines = {}


def style_ax(ax, title, xlabel, ylabel="reward"):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


# ---------- 1a. 線速度追蹤 ----------
ax = axes[0][0]
lin_vel_err = X_NORM * X_RANGE["lin_vel"]
(lines["lin_vel"],) = ax.plot(lin_vel_err.numpy(), curves0["lin_vel"].numpy(), color=REWARD_COLOR)
style_ax(ax, f"1. Lin vel tracking (scale={DEFAULT_CFG['lin_vel_tracking_reward_scale']})", "|v_cmd - v| (m/s)")

# ---------- 1b. 角速度追蹤 ----------
ax = axes[0][1]
ang_vel_err = X_NORM * X_RANGE["ang_vel"]
(lines["ang_vel"],) = ax.plot(ang_vel_err.numpy(), curves0["ang_vel"].numpy(), color=REWARD_COLOR)
style_ax(ax, f"1. Ang vel tracking (scale={DEFAULT_CFG['ang_vel_tracking_reward_scale']})", "|yaw_cmd - yaw| (rad/s)")

# ---------- 2. 腳掌高度追蹤 ----------
ax = axes[0][2]
foot_err = X_NORM * X_RANGE["foot_height"]
(lines["foot_height"],) = ax.plot(foot_err.numpy(), curves0["foot_height"].numpy(), color=REWARD_COLOR)
style_ax(ax, f"2. Foot height tracking (scale={DEFAULT_CFG['foot_height_reward_scale']})", "|z_target - z| (m)")

# ---------- 3. 預設姿態懲罰 ----------
ax = axes[0][3]
joint_dev = X_NORM * X_RANGE["joint_dev(max)"]
joint_dev_min_curve = rt.joint_deviation_penalty(
    joint_dev.unsqueeze(1),
    torch.zeros(N, 1),
    torch.tensor([0]),
    torch.tensor([DEFAULT_CFG["pose_weight_min"]]),
    DEFAULT_CFG["joint_deviation_penalty_scale"],
)
ax.plot(
    joint_dev.numpy(),
    joint_dev_min_curve.numpy(),
    color=PENALTY_COLOR,
    linestyle="--",
    label=f"weight={DEFAULT_CFG['pose_weight_min']} (legs)",
)
ax.plot(
    joint_dev.numpy(),
    curves0["joint_dev(max)"].numpy(),
    color=PENALTY_COLOR,
    linestyle="-",
    label=f"weight={DEFAULT_CFG['pose_weight_max']} (wrist/shoulder/elbow)",
)
style_ax(
    ax,
    f"3. Joint deviation penalty (scale={DEFAULT_CFG['joint_deviation_penalty_scale']})",
    "per-joint deviation (rad)",
)
ax.legend(fontsize=6, loc="lower left")

# ---------- 4a. 腳掌平整度懲罰(pitch+roll, 非左右腳 yaw 差) ----------
ax = axes[1][0]
feet_tilt = X_NORM * X_RANGE["feet_ori"]
ax.plot(feet_tilt.numpy(), curves0["feet_ori"].numpy(), color=PENALTY_COLOR)
style_ax(
    ax, f"4a. Feet orientation penalty (scale={DEFAULT_CFG['feet_ori_penalty_scale']})", "single foot tilt, sin(angle)"
)

# ---------- 4b. 腳掌間距過近懲罰 ----------
ax = axes[1][1]
feet_lateral = X_NORM * X_RANGE["close_feet"]
(lines["close_feet"],) = ax.plot(feet_lateral.numpy(), curves0["close_feet"].numpy(), color=PENALTY_COLOR)
threshold_line = ax.axvline(DEFAULT_CFG["close_feet_threshold"], color="gray", linestyle=":", linewidth=1)
style_ax(
    ax, f"4b. Close feet penalty (scale={DEFAULT_CFG['close_feet_xy_penalty_scale']})", "feet lateral distance (m)"
)

# 4c 已併入 4a(holosoma penalty_feet_ori 一次涵蓋 pitch+roll), 這格留空
axes[1][2].axis("off")

# ---------- 5. 存活獎勵 ----------
ax = axes[1][3]
ax.plot(X_NORM.numpy(), curves0["alive"].numpy(), color=REWARD_COLOR, linewidth=3)
ax.set_xticks([])
style_ax(ax, f"5. Alive reward (scale={DEFAULT_CFG['alive_reward_scale']})", "constant every step")

# ---------- 6a. 軀幹直立姿態懲罰 ----------
ax = axes[2][0]
tilt = X_NORM * X_RANGE["torso_ori"]
ax.plot(tilt.numpy(), curves0["torso_ori"].numpy(), color=PENALTY_COLOR)
tilt_line = ax.axvline(DEFAULT_CFG["max_torso_tilt"], color="gray", linestyle=":", linewidth=1)
style_ax(
    ax,
    f"6a. Torso orientation penalty (scale={DEFAULT_CFG['torso_orientation_penalty_scale']})",
    "torso tilt angle (rad)",
)

# ---------- 6b. 軀幹晃動角速度懲罰 ----------
ax = axes[2][1]
torso_ang_vel = X_NORM * X_RANGE["ang_vel_xy"]
ax.plot(torso_ang_vel.numpy(), curves0["ang_vel_xy"].numpy(), color=PENALTY_COLOR)
style_ax(ax, f"6b. Torso ang vel penalty (scale={DEFAULT_CFG['ang_vel_xy_penalty_scale']})", "torso ang vel (rad/s)")

# ---------- 7. 動作變化率懲罰 ----------
ax = axes[2][2]
action_delta = X_NORM * X_RANGE["action_rate"]
ax.plot(action_delta.numpy(), curves0["action_rate"].numpy(), color=PENALTY_COLOR)
style_ax(
    ax, f"7. Action rate penalty (scale={DEFAULT_CFG['action_rate_penalty_scale']})", "|delta action| (single dim)"
)

# ---------- 7b1. 關節速度懲罰 ----------
ax = axes[2][3]
joint_vel_x = X_NORM * X_RANGE["joint_vel"]
ax.plot(joint_vel_x.numpy(), curves0["joint_vel"].numpy(), color=PENALTY_COLOR)
style_ax(
    ax, f"7b. Joint vel penalty (scale={DEFAULT_CFG['joint_vel_penalty_scale']})", "|joint_vel| (rad/s, single joint)"
)

# ---------- 7b2. 關節加速度懲罰 ----------
ax = axes[3][0]
joint_acc_x = X_NORM * X_RANGE["joint_acc"]
ax.plot(joint_acc_x.numpy(), curves0["joint_acc"].numpy(), color=PENALTY_COLOR)
style_ax(
    ax,
    f"7b. Joint acc penalty (scale={DEFAULT_CFG['joint_acc_penalty_scale']})",
    "|joint_acc| (rad/s^2, single joint)",
)

# ---------- 6c. 軀幹高度懲罰 ----------
ax = axes[3][1]
torso_height_x = X_NORM * X_RANGE["torso_height"]
ax.plot(torso_height_x.numpy(), curves0["torso_height"].numpy(), color=PENALTY_COLOR)
style_ax(
    ax,
    f"6c. Torso height penalty (scale={DEFAULT_CFG['torso_height_penalty_scale']})",
    f"|height - target| (m, target={DEFAULT_CFG['target_torso_height']})",
)

# 剩下 1 格留空(4x4 格子共 16, 目前 13 條曲線 + 1 個峰值長條圖 = 14 格)
axes[3][2].axis("off")

# ---------- 各項峰值貢獻總覽(乘上 step_dt, 即單一 physics step 實際加減多少 reward) ----------
ax = axes[3][3]
# 用絕對值最大的點當「峰值」: reward 類的峰值在 x=0(零誤差), 大多數 penalty 的峰值在 x=1(最大誤差),
# 但 close_feet 是 hinge 形狀, 峰值反而在 x=0 -- 用 argmax(abs) 才能對所有形狀都正確。
term_peaks = [
    curves0[name][torch.argmax(torch.abs(curves0[name]))].item() * DEFAULT_CFG["step_dt"] for name in TERM_SCALE_KEY
]
bar_colors = [REWARD_COLOR if v >= 0 else PENALTY_COLOR for v in term_peaks]
ax.barh(list(TERM_SCALE_KEY), term_peaks, color=bar_colors)
ax.axvline(0.0, color="black", linewidth=0.8)
style_ax(ax, "Peak contribution per step (scale x x-range max x step_dt)", "reward / step", ylabel="")
ax.tick_params(labelsize=6)

# ---------- 綜合比較圖: 所有項目疊在同一張圖上, 直接比較量級(y 已乘上 step_dt) ----------
palette = plt.get_cmap("tab20").colors
combined_lines = {}
for i, name in enumerate(TERM_SCALE_KEY):
    scale = DEFAULT_CFG[TERM_SCALE_KEY[name]]
    (combined_lines[name],) = combined_ax.plot(
        X_NORM.numpy(),
        (curves0[name] * DEFAULT_CFG["step_dt"]).numpy(),
        color=palette[i % len(palette)],
        linestyle="-" if scale >= 0 else "--",
        linewidth=1.8,
        label=f"{name} (scale={scale})",
    )
combined_ax.axhline(0.0, color="black", linewidth=0.8)
combined_ax.grid(True, alpha=0.3)
combined_ax.set_title(
    "All terms overlaid - direct per-step magnitude comparison (solid = reward, dashed = penalty)", fontsize=10
)
combined_ax.set_xlabel(
    "normalized x per term, 0-1 mapped to the same physical range as its panel above (see panels for units)"
)
combined_ax.set_ylabel("reward per step (x step_dt)")
combined_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.08), bbox_transform=fig.transFigure, ncol=7, fontsize=7)

# --- 滑桿: 調整幾個決定曲線形狀的關鍵超參數, 同步更新小圖與綜合比較圖 ---
slider_axes = {
    "lin_vel_std": plt.axes([0.10, 0.045, 0.32, 0.012]),
    "ang_vel_std": plt.axes([0.10, 0.025, 0.32, 0.012]),
    "gait_tracking_sigma": plt.axes([0.10, 0.005, 0.32, 0.012]),
    "close_feet_threshold": plt.axes([0.58, 0.045, 0.32, 0.012]),
}
sliders = {
    "lin_vel_std": Slider(slider_axes["lin_vel_std"], "lin_vel_std", 0.05, 1.0, valinit=DEFAULT_CFG["lin_vel_std"]),
    "ang_vel_std": Slider(slider_axes["ang_vel_std"], "ang_vel_std", 0.05, 1.0, valinit=DEFAULT_CFG["ang_vel_std"]),
    "gait_tracking_sigma": Slider(
        slider_axes["gait_tracking_sigma"],
        "gait_tracking_sigma",
        0.0005,
        0.02,
        valinit=DEFAULT_CFG["gait_tracking_sigma"],
    ),
    "close_feet_threshold": Slider(
        slider_axes["close_feet_threshold"],
        "close_feet_threshold",
        0.05,
        0.3,
        valinit=DEFAULT_CFG["close_feet_threshold"],
    ),
}


def update(_val):
    cfg = dict(DEFAULT_CFG)
    cfg["lin_vel_std"] = sliders["lin_vel_std"].val
    cfg["ang_vel_std"] = sliders["ang_vel_std"].val
    cfg["gait_tracking_sigma"] = sliders["gait_tracking_sigma"].val
    close_feet_th = sliders["close_feet_threshold"].val
    cfg["close_feet_threshold"] = close_feet_th

    curves = term_curves(cfg)
    for name in ("lin_vel", "ang_vel", "foot_height", "close_feet"):
        lines[name].set_ydata(curves[name].numpy())
    threshold_line.set_xdata([close_feet_th, close_feet_th])

    for name, line in combined_lines.items():
        line.set_ydata((curves[name] * cfg["step_dt"]).numpy())

    fig.canvas.draw_idle()


for slider in sliders.values():
    slider.on_changed(update)

plt.show()
