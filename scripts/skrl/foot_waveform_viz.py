# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play 時的即時關節波形視覺化(尤其是腳踝/腳掌), 用網頁而不是 matplotlib。

跟 play.py 分開成獨立檔案, 對外只暴露 FootWaveformServer 這一個 class(建構、start、
sample、stop 四個方法), 讓 play.py 的迴圈只需要呼叫一兩行, 不用知道 HTTP/SSE/畫面的細節。

用網頁(HTML + <canvas> + Server-Sent Events)而不是 matplotlib, 是因為畫圖完全丟給
瀏覽器這個獨立 process 處理, Python 這邊(跟 Isaac Sim 同一個 process/執行緒)每一步只
需要把數字塞進一個 deque, 幾乎零成本, 不會跟 Omniverse Kit 自己的 UI 事件迴圈搶執行緒,
也不會拖慢 env.step() 的節奏。只用標準庫(http.server/threading/json), 不需要額外裝套件。
"""

from __future__ import annotations

import http.server
import json
import math
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any

# (key, find_joints regex, paired_left_right) —— 只涵蓋腿部+腰部這 13 個關節(手臂/手腕
# 不在這個視覺化工具的範圍內), key 會直接當成前端 JOINT_DEFS 的 key 跟 sample() 塞進
# point dict 的欄位前綴, 兩邊要對得上
_JOINT_SPECS: list[tuple[str, str, bool]] = [
    ("hip_pitch", ".*hip_pitch_joint", True),
    ("hip_roll", ".*hip_roll_joint", True),
    ("hip_yaw", ".*hip_yaw_joint", True),
    ("knee", ".*knee_joint", True),
    ("ankle_pitch", ".*ankle_pitch_joint", True),
    ("ankle_roll", ".*ankle_roll_joint", True),
    ("waist_yaw", ".*waist_yaw_joint", False),
]

# K1 正面站姿的實際渲染圖, 給網頁上的關節選取示意圖當背景用(見 _INDEX_HTML 的 <svg>)。
# 路徑相對這個檔案算, 不用假設呼叫者的工作目錄
_ROBOT_IMAGE_PATH = (
    Path(__file__).resolve().parents[2] / "source" / "k1_single_leg_walk" / "config" / "ai_sapiens_k1_render.png"
)


class FootWaveformServer:
    """在背景執行緒跑一個小型 HTTP 伺服器, 即時把腳踝/腳掌波形推給瀏覽器畫。"""

    def __init__(self, env: Any, env_idx: int = 0, port: int = 8765, window_s: float = 4.0):
        """env 傳 env.unwrapped(要能直接存取 .robot/.contact_sensor/.cfg/._feet_ids 等)。"""
        self._env = env
        self._env_idx = env_idx
        self._port = port

        step_dt = getattr(env, "step_dt", None) or 0.01
        maxlen = max(1, int(window_s / step_dt))
        self._buffer: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        # 關節選取示意圖的背景圖, 開機時讀一次, 每個 /robot.png request 直接回傳同一份 bytes
        try:
            self._robot_png_bytes: bytes | None = _ROBOT_IMAGE_PATH.read_bytes()
        except OSError as exc:
            print(f"[FootWaveformViz] 讀不到機器人示意圖 {_ROBOT_IMAGE_PATH}: {exc}")
            self._robot_png_bytes = None

        self._resolve_indices()

    # ---------- 對外介面 ----------

    def start(self) -> None:
        """啟動背景 HTTP 伺服器執行緒, 並自動開瀏覽器分頁。"""
        handler_cls = self._make_handler()
        try:
            self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self._port), handler_cls)
        except OSError as exc:
            print(f"[FootWaveformViz] 無法啟動伺服器(port {self._port} 可能已被佔用): {exc}")
            print("[FootWaveformViz] 波形視覺化停用, Isaac Sim 繼續正常執行。")
            self._httpd = None
            return
        # 瀏覽器分頁關掉時, SSE 連線常常是被 client 端直接中斷(TCP RST)而不是乾淨關閉,
        # socketserver 預設會把這種連線層級的錯誤印成一整條 traceback——這是無害的(server
        # 本身不會壞, 只是那個連線的執行緒結束), 蓋掉預設的 handle_error 讓它安靜跳過,
        # 不要每次關分頁都在 terminal 洗一條嚇人的錯誤訊息
        self._httpd.handle_error = lambda request, client_address: None

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://127.0.0.1:{self._port}/"
        print(f"[FootWaveformViz] 波形視覺化: {url}")
        webbrowser.open(url)

    def sample(self, step_count: int, step_dt: float) -> None:
        """讀這一步的關節角度/腳掌高度/接觸力, append 進內部的 deque buffer。

        沒有任何 I/O、沒有畫圖, 呼叫成本極低, 不影響模擬節奏。
        """
        if self._httpd is None:
            return

        env = self._env
        idx = self._env_idx
        joint_pos = env.robot.data.joint_pos[idx]
        joint_vel = env.robot.data.joint_vel[idx]

        feet_pos_z = env.robot.data.body_pos_w[idx, env._feet_ids, 2] - env.cfg.origin_height
        contact_forces = env.contact_sensor.data.net_forces_w[idx, env._contact_feet_idx]
        contact_mag = contact_forces.norm(dim=-1)

        point: dict[str, float] = {"t": step_count * step_dt}
        # 角度/角速度都轉成度/度每秒——跟原本 ankle 的慣例一致, 前端只要一套格式化邏輯就好
        for key, _, paired in _JOINT_SPECS:
            ji = self._joint_idx[key]
            if paired:
                point[f"{key}_pos_l"] = math.degrees(joint_pos[ji["l"]].item())
                point[f"{key}_pos_r"] = math.degrees(joint_pos[ji["r"]].item())
                point[f"{key}_vel_l"] = math.degrees(joint_vel[ji["l"]].item())
                point[f"{key}_vel_r"] = math.degrees(joint_vel[ji["r"]].item())
            else:
                point[f"{key}_pos"] = math.degrees(joint_pos[ji["idx"]].item())
                point[f"{key}_vel"] = math.degrees(joint_vel[ji["idx"]].item())

        point["foot_h_l"] = feet_pos_z[self._foot_pos_l].item()
        point["foot_h_r"] = feet_pos_z[self._foot_pos_r].item()
        point["contact_l"] = float(contact_mag[self._foot_pos_l].item() > env.cfg.contact_force_threshold)
        point["contact_r"] = float(contact_mag[self._foot_pos_r].item() > env.cfg.contact_force_threshold)

        with self._lock:
            self._buffer.append(point)

    def stop(self) -> None:
        """關掉背景伺服器執行緒, 釋放 port。"""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # ---------- 內部細節 ----------

    def _resolve_indices(self) -> None:
        """查好 _JOINT_SPECS 這 13 個關節(左右腳各自)的 joint index、feet_ids 裡左右腳的
        位置, 建構時只做一次, sample() 就不用每步重找。

        self._joint_idx 是 {key: {"l": idx, "r": idx}} (成對關節) 或 {key: {"idx": idx}}
        (waist_yaw 這種單一、沒有左右之分的關節)。
        """
        robot = self._env.robot

        def left_right(ids: list[int], names: list[str]) -> tuple[int, int]:
            left = ids[[i for i, n in enumerate(names) if "left" in n][0]]
            right = ids[[i for i, n in enumerate(names) if "right" in n][0]]
            return left, right

        self._joint_idx: dict[str, dict[str, int]] = {}
        for key, pattern, paired in _JOINT_SPECS:
            ids, names = robot.find_joints(pattern)
            if paired:
                left, right = left_right(ids, names)
                self._joint_idx[key] = {"l": left, "r": right}
            else:
                assert len(ids) == 1, f"[FootWaveformViz] {key!r} 預期只有 1 個關節, 找到: {names}"
                self._joint_idx[key] = {"idx": ids[0]}

        feet_names = self._env._feet_names
        self._foot_pos_l = next(i for i, n in enumerate(feet_names) if "left" in n)
        self._foot_pos_r = next(i for i, n in enumerate(feet_names) if "right" in n)

    def _make_handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
                pass  # 不要洗版 terminal

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self._serve_index()
                elif self.path == "/stream":
                    self._serve_stream()
                elif self.path == "/robot.png":
                    self._serve_robot_image()
                else:
                    self.send_error(404)

            def _serve_index(self) -> None:
                body = _INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_robot_image(self) -> None:
                body = outer._robot_png_bytes
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        with outer._lock:
                            snapshot = list(outer._buffer)
                        payload = json.dumps(snapshot)
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(0.05)  # 20Hz 推送, 跟模擬速度脫鉤
                except OSError:
                    # 瀏覽器分頁關掉/連線中斷, 結束這個連線的執行緒即可(BrokenPipeError/
                    # ConnectionResetError 是 POSIX 上的錯誤, Windows 上實際會是
                    # ConnectionAbortedError——三者都是 OSError 的子類, 直接抓 OSError
                    # 比較保險, 不用逐一列舉平台特定的例外)
                    pass

        return Handler


_INDEX_HTML = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>K1 Foot Waveform</title>
<style>
  :root {
    --bg: #0b0c0e;
    --panel: #131519;
    --panel-border: rgba(255,255,255,0.08);
    --text: #e7e8ea;
    --text-dim: #8b9096;
    --text-faint: #565b62;
    --grid-line: rgba(255,255,255,0.06);
    --zero-line: rgba(255,255,255,0.16);
    --accent-l: #5b9dff;
    --accent-r: #ffab4d;
    --ok: #3ecf8e;
    --warn: #d65f5f;
    --radius: 10px;
    --sans: -apple-system, "Segoe UI", system-ui, "PingFang TC", "Microsoft JhengHei", sans-serif;
    --mono: ui-monospace, "SFMono-Regular", "Cascadia Mono", Consolas, monospace;
  }

  * { box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    margin: 0;
    padding: 20px 24px 32px;
  }

  .topbar {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px 20px;
    margin: 0 0 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--panel-border);
  }

  .brand { display: flex; align-items: center; gap: 9px; }

  .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--text-faint);
    box-shadow: 0 0 0 3px transparent;
    transition: background 0.3s ease;
  }
  .status-dot.ok { background: var(--ok); box-shadow: 0 0 0 3px rgba(62,207,142,0.15); }
  .status-dot.warn { background: var(--warn); box-shadow: 0 0 0 3px rgba(214,95,95,0.15); }

  h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: 0.01em; }

  .meta { display: flex; gap: 18px; font-family: var(--mono); font-size: 12px; color: var(--text-dim); }
  .meta b { color: var(--text); font-weight: 500; }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 14px;
  }

  .card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: var(--radius);
    padding: 14px 16px 12px;
  }

  .card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
  }

  .card-head h2 { font-size: 12.5px; font-weight: 500; margin: 0; color: var(--text-dim); }

  .readouts { display: flex; gap: 14px; font-family: var(--mono); font-size: 12.5px; }
  .readouts .readout { display: flex; align-items: baseline; gap: 5px; }
  .readouts i {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    margin-bottom: 1px;
  }
  .readout.l i { background: var(--accent-l); }
  .readout.r i { background: var(--accent-r); }
  .readouts b { font-weight: 500; min-width: 3.5em; text-align: right; }
  .readout.l b { color: var(--accent-l); }
  .readout.r b { color: var(--accent-r); }
  .readouts small { color: var(--text-faint); font-size: 10.5px; }

  .canvas-wrap { position: relative; }
  canvas { display: block; width: 100%; height: 132px; }

  .layout { display: flex; gap: 14px; align-items: flex-start; }

  .picker-panel { flex: 0 0 320px; padding: 14px; }
  .picker-panel h2 { font-size: 12.5px; font-weight: 500; margin: 0 0 10px; color: var(--text-dim); }

  #jointSvg { width: 100%; height: auto; display: block; }
  /* 關節選取改用方形按鈕, 固定排在圖片左右兩側外緣、垂直排整齊, 不再疊在機器人身上的
     實際關節位置——疊在圖上那版每次都要精準對到關節像素位置, 對不準就會像"腳踝跑到膝蓋"
     這樣認錯關節, 而且點很小、擠在一起也不好點; 排成側邊清單後位置固定、好認、好點 */
  .joint-marker { cursor: pointer; }
  .joint-marker .btn {
    fill: var(--panel);
    stroke-width: 3;
    transform-box: fill-box;
    transform-origin: center;
    transition: fill 0.15s ease, stroke 0.15s ease, filter 0.15s ease, transform 0.15s ease;
  }
  .joint-marker[data-side="l"] .btn { stroke: var(--accent-l); }
  .joint-marker[data-side="r"] .btn { stroke: var(--accent-r); }
  .joint-marker:hover .btn { fill: var(--panel-border); }
  /* active 狀態要夠明顯: 換成飽和填色 + 放大 + 同色發光, 不能只靠邊框顏色, 在機器人圖片的
     雜訊背景上太容易被忽略 */
  .joint-marker.active .btn { transform: scale(1.12); }
  .joint-marker.active[data-side="l"] .btn { fill: var(--accent-l); filter: drop-shadow(0 0 10px var(--accent-l)); }
  .joint-marker.active[data-side="r"] .btn { fill: var(--accent-r); filter: drop-shadow(0 0 10px var(--accent-r)); }

  .joint-marker .btn-label {
    fill: var(--text);
    font-family: var(--sans);
    text-anchor: middle;
    dominant-baseline: central;
    pointer-events: none;
    transition: fill 0.15s ease;
  }
  .joint-marker.active .btn-label { fill: #0b0c0e; }

  .charts-col { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 14px; }

  .joint-charts-row { display: flex; gap: 14px; }
  .joint-charts-row .chart-col {
    flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 14px;
  }
</style>
</head>
<body>
<div class="topbar">
  <div class="brand">
    <span class="status-dot" id="statusDot"></span>
    <h1>K1 Foot Waveform</h1>
  </div>
  <div class="meta">
    <span>WINDOW <b id="windowLabel">-</b></span>
    <span>T <b id="clock">0.00s</b></span>
  </div>
</div>
<div class="layout">
  <section class="card picker-panel">
    <h2>關節選擇(點擊切換波形)</h2>
    <svg id="jointSvg" viewBox="0 0 1354 3840" xmlns="http://www.w3.org/2000/svg">
      <image href="/robot.png" width="1354" height="3840" />
      <g id="markersLayer"></g>
    </svg>
  </section>
  <div class="charts-col">
    <div class="grid" id="fixedCharts"></div>
    <div class="joint-charts-row">
      <div class="chart-col" id="jointChartsPos"></div>
      <div class="chart-col" id="jointChartsVel"></div>
    </div>
  </div>
</div>
<script>
const COLOR_L = getComputedStyle(document.documentElement).getPropertyValue("--accent-l").trim();
const COLOR_R = getComputedStyle(document.documentElement).getPropertyValue("--accent-r").trim();
const GRID_LINE = getComputedStyle(document.documentElement).getPropertyValue("--grid-line").trim();
const ZERO_LINE = getComputedStyle(document.documentElement).getPropertyValue("--zero-line").trim();

// 7 種可點選的關節(腿部+腰部), key 對應到 Python sample() 塞進 point dict 的欄位前綴
// (成對關節是 `${key}_pos_l`/`${key}_pos_r`/`${key}_vel_l`/`${key}_vel_r`, waist_yaw 沒有
// 左右之分, 是 `${key}_pos`/`${key}_vel`)。markers 是這個按鈕在機器人示意圖(1354x3840,
// 對應 /robot.png 的原始像素座標)上的位置——key 是機器人「自己的」左右, 不是螢幕畫面上的
// 左右: 正面圖是鏡像的, 機器人的右腳畫在畫面左邊, 所以「r」這一欄放在畫面左側的按鈕欄,
// 「l」放右側。按鈕不疊在機器人身上實際的關節位置上(疊上去那版每次都要精準對到像素,
// 對不準就會認錯關節、而且擠在一起不好點), 而是固定排在圖片左右外緣、由上到下照關節順序
// 垂直排列成兩欄清單, label 是按鈕上顯示的短字, 完整名稱看 hover 的 tooltip(見下面 jointLabel)
const BTN = 150; // 方形按鈕邊長(圖片像素)
const COL_R_X = 90; // 右腳(畫面左側)按鈕欄的中心 x
const COL_L_X = 1354 - 90; // 左腳(畫面右側)按鈕欄的中心 x
// 6 個關節由上到下的列高: 同一個關節的不同自由度(髖的 pitch/roll/yaw、踝的 pitch/roll)
// 排得緊(組內間距 180), 不同關節之間(髖組 -> 膝 -> 踝組)拉開(組間間距 420), 讓分組一眼
//看得出來, 不是均勻等距排一排
const ROW_Y = [1850, 2030, 2210, 2830, 3450, 3630]; // 整體比上一版往下移 300(圖片像素)
const JOINT_DEFS = [
  { key: "hip_pitch",   title: "髖關節 Pitch", label: "HipP", paired: true,  markers: { r: [COL_R_X, ROW_Y[0]], l: [COL_L_X, ROW_Y[0]] } },
  { key: "hip_roll",    title: "髖關節 Roll",  label: "HipR", paired: true,  markers: { r: [COL_R_X, ROW_Y[1]], l: [COL_L_X, ROW_Y[1]] } },
  { key: "hip_yaw",     title: "髖關節 Yaw",   label: "HipY", paired: true,  markers: { r: [COL_R_X, ROW_Y[2]], l: [COL_L_X, ROW_Y[2]] } },
  { key: "knee",        title: "膝關節",       label: "Knee", paired: true,  markers: { r: [COL_R_X, ROW_Y[3]], l: [COL_L_X, ROW_Y[3]] } },
  { key: "ankle_pitch", title: "踝關節 Pitch", label: "AnkP", paired: true,  markers: { r: [COL_R_X, ROW_Y[4]], l: [COL_L_X, ROW_Y[4]] } },
  { key: "ankle_roll",  title: "踝關節 Roll",  label: "AnkR", paired: true,  markers: { r: [COL_R_X, ROW_Y[5]], l: [COL_L_X, ROW_Y[5]] } },
  { key: "waist_yaw",   title: "腰部 Yaw",     label: "Wst",  paired: false, markers: { l: [1354 / 2, 1550] } },
];
const JOINT_DEFS_BY_KEY = Object.fromEntries(JOINT_DEFS.map((d) => [d.key, d]));

// foot_h/contact 不是特定關節的自由度, 一直固定顯示, 不受點選影響
const FIXED_CHARTS = [
  { key: "foot_h", title: "腳掌高度", unit: "m", fmt: (v) => v.toFixed(3), zeroLine: true },
  {
    key: "contact", title: "接觸狀態", unit: "", step: true, fill: true, fixedRange: [-0.1, 1.1],
    fmt: (v) => (v > 0.5 ? "STANCE" : "SWING"),
  },
];

const fixedChartsEl = document.getElementById("fixedCharts");
const jointChartsPosEl = document.getElementById("jointChartsPos");
const jointChartsVelEl = document.getElementById("jointChartsVel");
const statusDot = document.getElementById("statusDot");
const windowLabel = document.getElementById("windowLabel");
const clockEl = document.getElementById("clock");

// 把原本 CHARTS.map(...) 裡建卡片的部分抽成獨立函式, 讓固定圖表(一開始建一次)跟動態的
// 關節圖表(點選時才建、取消點選時要能整張移除)可以共用同一套邏輯
function buildChartCard(cfg, container) {
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML = `
    <div class="card-head">
      <h2>${cfg.title}${cfg.unit ? ` (${cfg.unit})` : ""}</h2>
      <div class="readouts">
        <span class="readout l"><i></i><b data-role="val-l">-</b><small>L</small></span>
        ${cfg.unpaired ? "" : `<span class="readout r"><i></i><b data-role="val-r">-</b><small>R</small></span>`}
      </div>
    </div>
    <div class="canvas-wrap"><canvas></canvas></div>`;
  container.appendChild(card);

  const canvas = card.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const valL = card.querySelector('[data-role="val-l"]');
  const valR = card.querySelector('[data-role="val-r"]');

  const chart = { cfg, canvas, ctx, valL, valR, card, cssW: 0, cssH: 0 };
  resizeChart(chart);
  return chart;
}

function resizeChart(chart) {
  const dpr = window.devicePixelRatio || 1;
  const rect = chart.canvas.getBoundingClientRect();
  chart.cssW = rect.width;
  chart.cssH = rect.height;
  chart.canvas.width = Math.round(rect.width * dpr);
  chart.canvas.height = Math.round(rect.height * dpr);
  chart.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => activeCharts.forEach(resizeChart), 100);
});

function drawChart(chart, samples) {
  const { ctx, cfg } = chart;
  const w = chart.cssW, h = chart.cssH;
  ctx.clearRect(0, 0, w, h);
  if (samples.length < 2) return;

  const padL = 34, padR = 6, padT = 6, padB = 6;
  // unpaired(waist_yaw)只有一條線, 沒有 "_r" 那個欄位——keyR 是 null 時下面每個用到
  // keyR/valR 的地方都要跳過, 不是重新設計整個函式
  const keyL = cfg.unpaired ? cfg.key : cfg.key + "_l";
  const keyR = cfg.unpaired ? null : cfg.key + "_r";
  const tMin = samples[0].t, tMax = samples[samples.length - 1].t;
  const tSpan = Math.max(tMax - tMin, 1e-6);

  let yMin, yMax;
  if (cfg.fixedRange) {
    [yMin, yMax] = cfg.fixedRange;
  } else {
    const vals = [];
    for (const s of samples) { vals.push(s[keyL]); if (keyR) vals.push(s[keyR]); }
    yMin = Math.min(...vals);
    yMax = Math.max(...vals);
    if (cfg.zeroLine) { yMin = Math.min(yMin, 0); yMax = Math.max(yMax, 0); }
    const pad = Math.max((yMax - yMin) * 0.1, 1e-3);
    yMin -= pad; yMax += pad;
  }

  const xOf = (t) => padL + ((t - tMin) / tSpan) * (w - padL - padR);
  const yOf = (v) => h - padB - ((v - yMin) / (yMax - yMin)) * (h - padT - padB);

  // 三條水平格線(上/中/下), 讓數值範圍一眼可讀, 不是純裝飾
  ctx.strokeStyle = GRID_LINE;
  ctx.lineWidth = 1;
  ctx.font = "10px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
  ctx.fillStyle = "#565b62";
  ctx.textBaseline = "middle";
  [yMax, (yMax + yMin) / 2, yMin].forEach((v) => {
    const y = yOf(v);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    if (cfg.key !== "contact") {
      ctx.fillText(v.toFixed(cfg.key === "foot_h" ? 2 : 0), 2, y);
    }
  });

  if (cfg.zeroLine) {
    ctx.strokeStyle = ZERO_LINE;
    ctx.beginPath();
    ctx.moveTo(padL, yOf(0));
    ctx.lineTo(w - padR, yOf(0));
    ctx.stroke();
  }

  function seriesPath(key) {
    ctx.beginPath();
    samples.forEach((s, i) => {
      const x = xOf(s.t);
      const y = yOf(s[key]);
      if (cfg.step && i > 0) ctx.lineTo(x, yOf(samples[i - 1][key]));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
  }

  function drawSeries(key, color) {
    if (cfg.fill) {
      seriesPath(key);
      ctx.lineTo(xOf(samples[samples.length - 1].t), yOf(yMin));
      ctx.lineTo(xOf(samples[0].t), yOf(yMin));
      ctx.closePath();
      ctx.fillStyle = color + "22";
      ctx.fill();
    }
    seriesPath(key);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  drawSeries(keyL, COLOR_L);
  if (keyR) drawSeries(keyR, COLOR_R);

  const last = samples[samples.length - 1];
  chart.valL.textContent = cfg.fmt(last[keyL]);
  if (chart.valR) chart.valR.textContent = cfg.fmt(last[keyR]);
}

// ---------- 固定圖表(foot_h/contact), 一開始就建好, 永遠顯示 ----------
const fixedCharts = FIXED_CHARTS.map((cfg) => buildChartCard(cfg, fixedChartsEl));

// ---------- 動態關節圖表: 點選機器人示意圖上的關節, 加入/移除一對(角度+角速度)卡片 ----------
const dynamicCharts = new Map(); // key -> [posChart, velChart]
let activeCharts = [...fixedCharts]; // SSE 進來時實際要畫的圖表清單, 每次加/減動態圖表都重建

function chartCfgFor(def, kind) {
  // kind: "pos"(角度, deg) | "vel"(角速度, deg/s)
  const isPos = kind === "pos";
  return {
    key: `${def.key}_${kind}`,
    title: `${def.title} ${isPos ? "角度" : "角速度"}`,
    unit: isPos ? "deg" : "deg/s",
    unpaired: !def.paired,
    fmt: (v) => v.toFixed(1),
  };
}

function refreshActiveCharts() {
  activeCharts = [...fixedCharts, ...[...dynamicCharts.values()].flat()];
}

function addJointCharts(key) {
  if (dynamicCharts.has(key)) return;
  const def = JOINT_DEFS_BY_KEY[key];
  const posChart = buildChartCard(chartCfgFor(def, "pos"), jointChartsPosEl);
  const velChart = buildChartCard(chartCfgFor(def, "vel"), jointChartsVelEl);
  dynamicCharts.set(key, [posChart, velChart]);
  refreshActiveCharts();
}

function removeJointCharts(key) {
  const pair = dynamicCharts.get(key);
  if (!pair) return;
  pair.forEach((chart) => chart.card.remove());
  dynamicCharts.delete(key);
  refreshActiveCharts();
}

function updateMarkerActiveState(key) {
  const active = dynamicCharts.has(key);
  document.querySelectorAll(`.joint-marker[data-key="${key}"]`).forEach((m) => {
    m.classList.toggle("active", active);
  });
}

function toggleJoint(key) {
  if (dynamicCharts.has(key)) {
    removeJointCharts(key);
  } else {
    addJointCharts(key);
  }
  updateMarkerActiveState(key);
}

// ---------- 機器人示意圖: 在 <svg> 裡動態生成 13 個可點選的關節標記 ----------
const SVG_NS = "http://www.w3.org/2000/svg";
const markersLayer = document.getElementById("markersLayer");

function jointLabel(def, side) {
  if (!def.paired) return def.title;
  return (side === "r" ? "右" : "左") + def.title;
}

JOINT_DEFS.forEach((def) => {
  Object.entries(def.markers).forEach(([side, [cx, cy]]) => {
    const g = document.createElementNS(SVG_NS, "g");
    g.setAttribute("class", "joint-marker");
    g.dataset.key = def.key;
    g.dataset.side = side;

    // 方形按鈕本身就是點擊區域, 不用再疊一個看不見的熱區——固定大小、固定排列, 比疊在
    // 機器人身上的小圓點好點很多
    const btn = document.createElementNS(SVG_NS, "rect");
    btn.setAttribute("x", cx - BTN / 2);
    btn.setAttribute("y", cy - BTN / 2);
    btn.setAttribute("width", BTN);
    btn.setAttribute("height", BTN);
    btn.setAttribute("rx", 14);
    btn.setAttribute("class", "btn");
    g.appendChild(btn);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", cx);
    label.setAttribute("y", cy);
    label.setAttribute("font-size", 36);
    label.setAttribute("class", "btn-label");
    label.textContent = def.label;
    g.appendChild(label);

    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = jointLabel(def, side);
    g.appendChild(title);

    markersLayer.appendChild(g);
  });
});

document.getElementById("jointSvg").addEventListener("click", (e) => {
  const marker = e.target.closest(".joint-marker");
  if (marker) toggleJoint(marker.dataset.key);
});

// 預設顯示跟以前一樣的 ankle_pitch/ankle_roll, 保留原本行為, 只是現在可以再加其他關節
["ankle_pitch", "ankle_roll"].forEach((key) => {
  addJointCharts(key);
  updateMarkerActiveState(key);
});

const evtSource = new EventSource("/stream");
evtSource.onopen = () => { statusDot.className = "status-dot ok"; };
evtSource.onerror = () => { statusDot.className = "status-dot warn"; };
evtSource.onmessage = (event) => {
  const samples = JSON.parse(event.data);
  if (samples.length < 2) return;
  const span = samples[samples.length - 1].t - samples[0].t;
  windowLabel.textContent = span.toFixed(1) + "s";
  clockEl.textContent = samples[samples.length - 1].t.toFixed(2) + "s";
  activeCharts.forEach((chart) => drawChart(chart, samples));
};
</script>
</body>
</html>
"""
