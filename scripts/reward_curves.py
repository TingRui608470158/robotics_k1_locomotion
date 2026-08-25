"""視覺化 k1_single_leg_walk_env.py 中每一項 reward/penalty 公式的形狀與量級。

純 torch/matplotlib 腳本, 不需要 Isaac Sim, 直接執行:
    python scripts/reward_curves.py

每一項 reward/penalty 直接呼叫 reward_terms.py 裡跟訓練共用的同一份函式(用 importlib
以檔案路徑載入; 不能寫成 `import k1_single_leg_walk...`, 因為套件的 tasks/__init__.py
會透過 isaaclab_tasks.utils.import_packages 自動 import 整個套件, 進而拉進 isaaclab_tasks),
這裡只負責合成輸入 tensor 餵給那些函式, 不重新實作任何一項公式。

站立/擺動相關的 4 項(stance_contact/slip/swing_height/swing_clearance)在實際訓練時會
被 gate(command==0 時乘 0)蓋掉, 這裡只畫「gate=1(有移動指令)時」的曲線形狀。

下方 DEFAULT_CFG 的 scale/std 數值仍需與 k1_single_leg_walk_env_cfg.py 手動保持同步
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
    "stance_contact_reward_scale": 1.0,
    "slip_penalty_scale": -1.0,
    "swing_height_penalty_scale": -200.0,
    "swing_clearance_penalty_scale": -2.0,
    "lin_vel_tracking_reward_scale": 2.0,
    "ang_vel_tracking_reward_scale": 1.5,
    "lin_vel_std": 0.25,
    "ang_vel_std": 0.25,
    "alive_reward_scale": 0.5,
    "action_rate_penalty_scale": -2.0,
    "stand_still_penalty_scale": -1.0,
    "stance_fraction": 0.6,
    "swing_height": 0.09,
    # step_dt = decimation * sim_dt = 2 * (1/200)
    "step_dt": 2 * (1 / 200),
}

# 每項在小圖裡的 x 軸物理範圍上限(x_norm=1 對應到這個值), combined 圖沿用同一套範圍
X_RANGE = {
    "lin_vel": 2.0,  # m/s
    "ang_vel": 3.0,  # rad/s
    "slip": 2.0,  # m/s, 單腳水平滑動速度
    "swing_height": 0.09,  # m, 實際高度偏離目標的量(用 swing_height 當上限, 對稱誤差)
    "action_rate": 3.0,  # 單一動作維度變化量
    "stand_still": 1.0,  # rad, 單一關節偏離預設姿態
}

REWARD_COLOR = "#2a9d4a"
PENALTY_COLOR = "#d64a2a"

X_NORM = torch.linspace(0.0, 1.0, 300)
N = X_NORM.shape[0]
_ZEROS_N3 = torch.zeros(N, 3)


def term_curves(cfg: dict) -> dict[str, torch.Tensor]:
    """把每一項的合成輸入餵進 reward_terms.py 的對應函式, 回傳 x_norm=[0,1] 上的值(gate=1)。

    輸入都刻意合成成「只有這一項的誤差來源不為零」, 這樣算出來的曲線才會跟小圖裡
    單一變數的 x 軸一一對應。
    """
    lin_vel = _ZEROS_N3.clone()
    lin_vel[:, 0] = X_NORM * X_RANGE["lin_vel"]

    ang_vel = _ZEROS_N3.clone()
    ang_vel[:, 2] = X_NORM * X_RANGE["ang_vel"]

    # 只讓一隻腳有水平速度(滑動), is_stance 兩隻腳都設 True(展示站立相滑動被罰的形狀)
    foot_vel_horizontal = torch.zeros(N, 2, 2)
    foot_vel_horizontal[:, 0, 0] = X_NORM * X_RANGE["slip"]
    is_stance_both = torch.ones(N, 2, dtype=torch.bool)

    # 只讓一隻腳的高度偏離目標, is_swing 兩隻腳都設 True
    foot_height_actual = torch.zeros(N, 2)
    foot_height_actual[:, 0] = X_NORM * X_RANGE["swing_height"]
    foot_height_target = torch.zeros(N, 2)
    is_swing_both = torch.ones(N, 2, dtype=torch.bool)

    actions = torch.zeros(N, 1)
    actions[:, 0] = X_NORM * X_RANGE["action_rate"]
    previous_actions = torch.zeros(N, 1)

    joint_pos = torch.zeros(N, 1)
    joint_pos[:, 0] = X_NORM * X_RANGE["stand_still"]
    default_joint_pos = torch.zeros(N, 1)
    controlled_idx = torch.tensor([0])

    return {
        "lin_vel": rt.lin_vel_tracking_reward(
            _ZEROS_N3, lin_vel, cfg["lin_vel_std"], cfg["lin_vel_tracking_reward_scale"]
        ),
        "ang_vel": rt.ang_vel_tracking_reward(
            _ZEROS_N3, ang_vel, cfg["ang_vel_std"], cfg["ang_vel_tracking_reward_scale"]
        ),
        "alive": rt.alive_reward(N, torch.device("cpu"), cfg["alive_reward_scale"]),
        "action_rate": rt.action_rate_penalty(actions, previous_actions, cfg["action_rate_penalty_scale"]),
        "slip": rt.slip_penalty(foot_vel_horizontal, is_stance_both, cfg["slip_penalty_scale"]),
        "swing_height": rt.swing_height_penalty(
            foot_height_actual, foot_height_target, is_swing_both, cfg["swing_height_penalty_scale"]
        ),
        "stand_still": rt.stand_still_penalty(
            joint_pos, default_joint_pos, controlled_idx, cfg["stand_still_penalty_scale"]
        ),
    }


TERM_SCALE_KEY = {
    "lin_vel": "lin_vel_tracking_reward_scale",
    "ang_vel": "ang_vel_tracking_reward_scale",
    "alive": "alive_reward_scale",
    "action_rate": "action_rate_penalty_scale",
    "slip": "slip_penalty_scale",
    "swing_height": "swing_height_penalty_scale",
    "stand_still": "stand_still_penalty_scale",
}

curves0 = term_curves(DEFAULT_CFG)

# --- 建立圖表: 上方 3x4 為各項獨立公式形狀(7 條曲線 + 2 個特殊面板 + 1 個峰值長條圖 = 10 格用到) ---
fig = plt.figure(figsize=(18, 17))
gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 3.0], hspace=0.7, wspace=0.35, bottom=0.16, top=0.94)
axes = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(3)]
combined_ax = fig.add_subplot(gs[3, :])
fig.suptitle("K1 Single Leg Walk - Reward / Penalty Terms (new design, shape at gate=1)", fontsize=14)

lines = {}


def style_ax(ax, title, xlabel, ylabel="reward"):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


# ---------- 3a. 線速度追蹤 ----------
ax = axes[0][0]
lin_vel_err = X_NORM * X_RANGE["lin_vel"]
(lines["lin_vel"],) = ax.plot(lin_vel_err.numpy(), curves0["lin_vel"].numpy(), color=REWARD_COLOR)
style_ax(ax, f"3. Lin vel tracking (scale={DEFAULT_CFG['lin_vel_tracking_reward_scale']})", "|v_cmd - v| (m/s)")

# ---------- 3b. 角速度追蹤 ----------
ax = axes[0][1]
ang_vel_err = X_NORM * X_RANGE["ang_vel"]
(lines["ang_vel"],) = ax.plot(ang_vel_err.numpy(), curves0["ang_vel"].numpy(), color=REWARD_COLOR)
style_ax(ax, f"4. Ang vel tracking (scale={DEFAULT_CFG['ang_vel_tracking_reward_scale']})", "|yaw_cmd - yaw| (rad/s)")

# ---------- 1b. 滑動懲罰(站立相) ----------
ax = axes[0][2]
slip_x = X_NORM * X_RANGE["slip"]
(lines["slip"],) = ax.plot(slip_x.numpy(), curves0["slip"].numpy(), color=PENALTY_COLOR)
style_ax(ax, f"1b. Slip penalty (scale={DEFAULT_CFG['slip_penalty_scale']})", "single foot horizontal speed (m/s)")

# ---------- 2a. 高度追蹤(擺動相) ----------
ax = axes[0][3]
swing_h_x = X_NORM * X_RANGE["swing_height"]
(lines["swing_height"],) = ax.plot(swing_h_x.numpy(), curves0["swing_height"].numpy(), color=PENALTY_COLOR)
style_ax(ax, f"2a. Swing height penalty (scale={DEFAULT_CFG['swing_height_penalty_scale']})", "|actual - target| (m)")

# ---------- 5. 存活獎勵 ----------
ax = axes[1][0]
ax.plot(X_NORM.numpy(), curves0["alive"].numpy(), color=REWARD_COLOR, linewidth=3)
ax.set_xticks([])
style_ax(ax, f"5. Alive reward (scale={DEFAULT_CFG['alive_reward_scale']})", "constant every step")

# ---------- 5. 動作變化率懲罰 ----------
ax = axes[1][1]
action_delta = X_NORM * X_RANGE["action_rate"]
(lines["action_rate"],) = ax.plot(action_delta.numpy(), curves0["action_rate"].numpy(), color=PENALTY_COLOR)
style_ax(
    ax, f"5. Action rate penalty (scale={DEFAULT_CFG['action_rate_penalty_scale']})", "|delta action| (single dim)"
)

# ---------- 6. 站立姿態懲罰(command=0 時) ----------
ax = axes[1][2]
stand_still_x = X_NORM * X_RANGE["stand_still"]
(lines["stand_still"],) = ax.plot(stand_still_x.numpy(), curves0["stand_still"].numpy(), color=PENALTY_COLOR)
style_ax(
    ax,
    f"6. Stand still penalty (scale={DEFAULT_CFG['stand_still_penalty_scale']})",
    "per-joint deviation (rad, command=0 only)",
)

# ---------- 1a / 2b. 接觸相關(二元, 不是 x_norm 曲線, 用文字說明) ----------
ax = axes[1][3]
ax.axis("off")
ax.text(
    0.5,
    0.5,
    "1a. stance_contact_reward / 2b. swing_clearance_penalty\n"
    "are binary +-1/0 signals (comparing contact_detected against\n"
    "is_stance/is_swing) that need real ContactSensor data, so\n"
    "they can't be drawn as a synthetic continuous-x curve;\n"
    "see the peak bar chart below instead\n"
    f"(stance_contact scale={DEFAULT_CFG['stance_contact_reward_scale']}, "
    f"swing_clearance scale={DEFAULT_CFG['swing_clearance_penalty_scale']})",
    ha="center",
    va="center",
    fontsize=7,
    wrap=True,
)

# 剩下的格子留空
for ax in (axes[2][0], axes[2][1], axes[2][2]):
    ax.axis("off")

# ---------- 各項峰值貢獻總覽(乘上 step_dt, 即單一 physics step 實際加減多少 reward) ----------
ax = axes[2][3]
# 用絕對值最大的點當「峰值」: reward 類的峰值在 x=0(零誤差), penalty 類的峰值在 x=1(最大誤差)
term_peaks = {
    name: curves0[name][torch.argmax(torch.abs(curves0[name]))].item() * DEFAULT_CFG["step_dt"]
    for name in TERM_SCALE_KEY
}
# stance_contact / swing_clearance 是二元訊號, 峰值直接用 scale 本身(兩腳都對/都違規時的極端值)
term_peaks["stance_contact"] = 2.0 * DEFAULT_CFG["stance_contact_reward_scale"] * DEFAULT_CFG["step_dt"]
term_peaks["swing_clearance"] = 2.0 * DEFAULT_CFG["swing_clearance_penalty_scale"] * DEFAULT_CFG["step_dt"]

labels = list(term_peaks.keys())
values = list(term_peaks.values())
bar_colors = [REWARD_COLOR if v >= 0 else PENALTY_COLOR for v in values]
ax.barh(labels, values, color=bar_colors)
ax.axvline(0.0, color="black", linewidth=0.8)
style_ax(ax, "Peak contribution per step (x step_dt)", "reward / step", ylabel="")
ax.tick_params(labelsize=6)

# ---------- 綜合比較圖: 有合成曲線的項目疊在同一張圖上, 直接比較量級(y 已乘上 step_dt) ----------
palette = plt.get_cmap("tab10").colors
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
    "Terms with a synthetic curve, overlaid (solid = reward, dashed = penalty); "
    "stance_contact/swing_clearance excluded (binary, see peak bar chart)",
    fontsize=10,
)
combined_ax.set_xlabel(
    "normalized x per term, 0-1 mapped to the same physical range as its panel above (see panels for units)"
)
combined_ax.set_ylabel("reward per step (x step_dt)")
combined_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.08), bbox_transform=fig.transFigure, ncol=7, fontsize=8)

# --- 滑桿: 調整兩個 exp kernel 的 std, 同步更新小圖與綜合比較圖 ---
slider_axes = {
    "lin_vel_std": plt.axes([0.10, 0.03, 0.32, 0.015]),
    "ang_vel_std": plt.axes([0.10, 0.005, 0.32, 0.015]),
}
sliders = {
    "lin_vel_std": Slider(slider_axes["lin_vel_std"], "lin_vel_std", 0.05, 1.0, valinit=DEFAULT_CFG["lin_vel_std"]),
    "ang_vel_std": Slider(slider_axes["ang_vel_std"], "ang_vel_std", 0.05, 1.0, valinit=DEFAULT_CFG["ang_vel_std"]),
}


def update(_val):
    cfg = dict(DEFAULT_CFG)
    cfg["lin_vel_std"] = sliders["lin_vel_std"].val
    cfg["ang_vel_std"] = sliders["ang_vel_std"].val

    curves = term_curves(cfg)
    for name in ("lin_vel", "ang_vel"):
        lines[name].set_ydata(curves[name].numpy())

    for name, line in combined_lines.items():
        line.set_ydata((curves[name] * cfg["step_dt"]).numpy())

    fig.canvas.draw_idle()


for slider in sliders.values():
    slider.on_changed(update)

plt.show()
