import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def expected_foot_height(phi: np.ndarray, swing_height: float) -> np.ndarray:
    """Expected foot height from gait phase using a cubic Bézier profile.

    x in [0, 0.5): 支撐期,腳真正貼地不動 (height = 0)
    x in [0.5, 1): 擺動期,只在這半段內完成一次完整的抬起→放下弧線
    """

    def cubic_bezier_interpolation(y_start, y_end, x):
        x = np.clip(x, 0, 1)
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + np.pi) / (2 * np.pi)  # [0, 1)

    # 前半週期:貼地不動
    stance = np.zeros_like(x)

    # 後半週期:把 [0.5, 1) 重新縮放到 [0, 1),在這半段內獨立完成「抬起→放下」
    t = np.clip((x - 0.5) * 2, 0.0, 1.0)
    rising = cubic_bezier_interpolation(np.zeros_like(t), np.full_like(t, swing_height), 2 * t)
    falling = cubic_bezier_interpolation(np.full_like(t, swing_height), np.zeros_like(t), 2 * t - 1)
    swing = np.where(t <= 0.5, rising, falling)

    return np.where(x <= 0.5, stance, swing)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """把角度包回 [-pi, pi) 範圍。"""
    return (x + np.pi) % (2 * np.pi) - np.pi


# --- 繪圖設定 ---
Y_MAX = 0.3  # 固定 y 軸上限

phi = np.linspace(-np.pi, np.pi, 400)          # 左腳相位(當作時間軸)
phi_right = wrap_to_pi(phi + np.pi)             # 右腳相位:錯開半個週期
init_swing_height = 0.1

fig, ax = plt.subplots(figsize=(7, 4))
plt.subplots_adjust(bottom=0.25)

line_left, = ax.plot(
    phi, expected_foot_height(phi, init_swing_height),
    color="#2a78d6", linewidth=2, label="Left foot"
)
line_right, = ax.plot(
    phi, expected_foot_height(phi_right, init_swing_height),
    color="#d64a2a", linewidth=2, linestyle="--", label="Right foot"
)

ax.set_xlabel("phi (rad)")
ax.set_ylabel("foot height")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(0, Y_MAX)  # y 軸範圍固定,不隨資料變動
ax.grid(True, alpha=0.3)
ax.set_title("Expected foot height vs gait phase (Left vs Right)")
ax.legend(loc="upper right")

# 滑桿:調整 swing_height
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "swing_height", 0.02, 0.3, valinit=init_swing_height, valstep=0.01)


def update(val):
    line_left.set_ydata(expected_foot_height(phi, slider.val))
    line_right.set_ydata(expected_foot_height(phi_right, slider.val))
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()