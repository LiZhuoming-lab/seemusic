from __future__ import annotations

import base64
import colorsys
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

_CACHE_ROOT = Path(tempfile.gettempdir()) / "spectral_tool_cache"
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from .analysis import format_seconds


def plot_waveform(samples: np.ndarray, sr: int, event_times: np.ndarray) -> plt.Figure:
    duration = samples.shape[-1] / sr
    time_axis = np.linspace(0.0, duration, samples.shape[-1], endpoint=False)

    figure, axis = plt.subplots(figsize=(14, 3.5))
    axis.plot(time_axis, samples, linewidth=0.7, color="#1f5c99")
    for event_time in event_times:
        axis.axvline(float(event_time), color="#d1495b", alpha=0.7, linewidth=1.0)

    axis.set_title("波形与自动事件标记")
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("振幅")
    axis.set_xlim(0, duration)
    axis.grid(alpha=0.2, linestyle="--")
    figure.tight_layout()
    return figure


def plot_novelty(
    times: np.ndarray,
    novelty: np.ndarray,
    threshold: np.ndarray,
    peak_indices: np.ndarray,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(14, 3.5))
    axis.plot(times, novelty, color="#2a9d8f", linewidth=1.2, label="新颖度曲线")
    axis.plot(times, threshold, color="#e76f51", linewidth=1.0, linestyle="--", label="检测阈值")
    if peak_indices.size:
        axis.scatter(
            times[peak_indices],
            novelty[peak_indices],
            color="#9c3d54",
            s=35,
            label="自动标记点",
            zorder=3,
        )

    axis.set_title("频谱新颖度检测")
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("新颖度")
    axis.grid(alpha=0.2, linestyle="--")
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def plot_spectrogram(
    spectrogram_db: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    event_times: np.ndarray,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(14.8, 6.2))
    mesh = axis.pcolormesh(
        times,
        freqs / 1000.0,
        spectrogram_db,
        shading="auto",
        cmap="magma",
    )

    upper_y = (freqs[-1] / 1000.0) * 0.98 if freqs.size else 1.0
    for index, event_time in enumerate(event_times, start=1):
        axis.axvline(float(event_time), color="white", alpha=0.8, linewidth=0.9)
        axis.text(
            float(event_time),
            upper_y,
            str(index),
            color="white",
            fontsize=8,
            ha="center",
            va="top",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
        )

    axis.set_title("频谱图与自动事件编号")
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("频率（kHz）", labelpad=12)
    colorbar = figure.colorbar(mesh, ax=axis, format="%+2.0f dB", pad=0.02, fraction=0.035)
    colorbar.set_label("相对幅度（dB）")
    figure.subplots_adjust(left=0.13, right=0.92, bottom=0.11, top=0.92)
    return figure


def plot_local_waveform(
    samples: np.ndarray,
    sr: int,
    center_time: float,
    window_radius_sec: float,
    highlight_time: float | None = None,
) -> plt.Figure:
    if samples.size == 0:
        figure, axis = plt.subplots(figsize=(8.2, 2.6))
        axis.text(0.5, 0.5, "没有可显示的局部波形", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.tight_layout()
        return figure

    start_sec = max(0.0, float(center_time - window_radius_sec))
    end_sec = min(float(samples.shape[-1] / sr), float(center_time + window_radius_sec))
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(samples.shape[-1], max(start_sample + 1, int(end_sec * sr)))

    local_samples = samples[start_sample:end_sample]
    local_time = np.linspace(start_sec, end_sec, local_samples.shape[0], endpoint=False)

    figure, axis = plt.subplots(figsize=(8.2, 2.6))
    axis.plot(local_time, local_samples, linewidth=0.8, color="#1f5c99")
    axis.axhline(0.0, color="#7a7a7a", linewidth=0.6, alpha=0.6)
    if highlight_time is None:
        highlight_time = center_time
    axis.axvline(float(highlight_time), color="#d1495b", linewidth=1.2, alpha=0.95)
    axis.set_title(f"事件附近局部波形：{center_time:.2f}s")
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("振幅")
    axis.set_xlim(start_sec, end_sec)
    axis.grid(alpha=0.18, linestyle="--")
    figure.tight_layout()
    return figure


def build_local_waveform_chart(
    samples: np.ndarray,
    sr: int,
    center_time: float,
    window_radius_sec: float,
    highlight_time: float | None = None,
) -> go.Figure:
    if samples.size == 0:
        figure = go.Figure()
        figure.update_layout(
            title="没有可显示的局部波形",
            template="plotly_white",
            height=240,
        )
        return figure

    start_sec = max(0.0, float(center_time - window_radius_sec))
    end_sec = min(float(samples.shape[-1] / sr), float(center_time + window_radius_sec))
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(samples.shape[-1], max(start_sample + 1, int(end_sec * sr)))

    local_samples = samples[start_sample:end_sample]
    local_time = np.linspace(start_sec, end_sec, local_samples.shape[0], endpoint=False)
    if highlight_time is None:
        highlight_time = center_time

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=local_time,
            y=local_samples,
            mode="lines",
            line={"color": "#1f5c99", "width": 1.0},
            name="波形",
            hovertemplate="时间 %{x:.3f}s<br>振幅 %{y:.4f}<extra></extra>",
        )
    )
    figure.add_vline(
        x=float(highlight_time),
        line_color="#d1495b",
        line_width=2.0,
        opacity=0.95,
    )
    figure.update_layout(
        title=f"事件附近局部波形：{center_time:.2f}s",
        template="plotly_white",
        height=260,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        xaxis={"title": "时间（秒）", "showgrid": True, "gridcolor": "rgba(0,0,0,0.08)"},
        yaxis={"title": "振幅", "showgrid": True, "gridcolor": "rgba(0,0,0,0.08)"},
        hovermode="x unified",
        dragmode="pan",
        showlegend=False,
    )
    return figure


def build_synced_waveform_player_html(
    audio_bytes: bytes,
    samples: np.ndarray,
    sr: int,
    clip_start_sec: float,
    clip_end_sec: float,
    event_time_sec: float | None = None,
    amplitude_scale: float = 1.0,
) -> str:
    if samples.size == 0:
        return """
<div style="font-family: sans-serif; color: #666; padding: 1rem;">
  没有可显示的同步试听波形
</div>
"""

    start_sample = max(0, int(clip_start_sec * sr))
    end_sample = min(samples.shape[-1], max(start_sample + 1, int(clip_end_sec * sr)))
    clip_samples = np.asarray(samples[start_sample:end_sample], dtype=np.float32)
    clip_duration = max(clip_end_sec - clip_start_sec, 1e-6)

    bucket_count = int(min(1200, max(240, clip_samples.shape[0] // 128)))
    buckets = np.array_split(np.abs(clip_samples), bucket_count)
    envelope = [round(float(np.max(bucket)) if bucket.size else 0.0, 6) for bucket in buckets]

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    envelope_json = json.dumps(envelope, ensure_ascii=False)
    event_offset = None if event_time_sec is None else max(0.0, min(clip_duration, event_time_sec - clip_start_sec))
    event_offset_json = "null" if event_offset is None else f"{event_offset:.6f}"

    return f"""
<div style="font-family: 'Helvetica Neue', Arial, sans-serif; color: #1f1f1f; overflow: visible; padding-left: 10px; padding-right: 6px; box-sizing: border-box;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; font-size:0.92rem;">
    <div>同步试听波形</div>
    <div id="timeLabel" style="color:#666;">{format_seconds(clip_start_sec)} / {format_seconds(clip_end_sec)}</div>
  </div>
  <audio id="audio" controls preload="auto" style="width:100%; margin-bottom:0.45rem;">
    <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav" />
  </audio>
  <canvas id="waveCanvas" width="1200" height="220" style="width:100%; height:220px; border:1px solid #e3e3e3; border-radius:8px; background:#fbfbfc; cursor:pointer;"></canvas>
  <div style="display:flex; justify-content:space-between; margin-top:0.3rem; font-size:0.82rem; color:#666;">
    <span>{format_seconds(clip_start_sec)}</span>
    <span>点击波形可跳转；绿线是播放头，红线是事件时刻</span>
    <span>{format_seconds(clip_end_sec)}</span>
  </div>
</div>
<script>
  const audio = document.getElementById("audio");
  const canvas = document.getElementById("waveCanvas");
  const ctx = canvas.getContext("2d");
  const timeLabel = document.getElementById("timeLabel");
  const envelope = {envelope_json};
  const clipDuration = {clip_duration:.6f};
  const clipStart = {clip_start_sec:.6f};
  const eventOffset = {event_offset_json};
  const amplitudeScale = {float(amplitude_scale):.3f};

  function formatTime(totalSeconds) {{
    const safe = Math.max(0, totalSeconds);
    const minutes = Math.floor(safe / 60);
    const seconds = safe - minutes * 60;
    return `${{String(minutes).padStart(2, "0")}}:${{seconds.toFixed(2).padStart(5, "0")}}`;
  }}

  function draw() {{
    const width = canvas.width;
    const height = canvas.height;
    const mid = height / 2;
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#fbfbfc";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#d8dde6";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    const barWidth = width / envelope.length;
    ctx.fillStyle = "#1f5c99";
    for (let i = 0; i < envelope.length; i += 1) {{
      const amp = Math.max(0.01, envelope[i]);
      const barHeight = Math.min(height * 0.48, amp * (height * 0.56) * amplitudeScale);
      const x = i * barWidth;
      ctx.fillRect(x, mid - barHeight, Math.max(1, barWidth * 0.88), barHeight * 2);
    }}

    if (eventOffset !== null && clipDuration > 0) {{
      const eventX = (eventOffset / clipDuration) * width;
      ctx.strokeStyle = "#d1495b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(eventX, 0);
      ctx.lineTo(eventX, height);
      ctx.stroke();
    }}

    const currentX = clipDuration > 0 ? (audio.currentTime / clipDuration) * width : 0;
    ctx.strokeStyle = "#17c964";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(currentX, 0);
    ctx.lineTo(currentX, height);
    ctx.stroke();

    const absoluteCurrent = clipStart + (audio.currentTime || 0);
    timeLabel.textContent = `${{formatTime(absoluteCurrent)}} / {format_seconds(clip_end_sec)}`;
  }}

  function tick() {{
    draw();
    if (!audio.paused && !audio.ended) {{
      window.requestAnimationFrame(tick);
    }}
  }}

  ["play", "pause", "seeked", "loadedmetadata", "timeupdate", "ended"].forEach((eventName) => {{
    audio.addEventListener(eventName, () => {{
      draw();
      if (eventName === "play") {{
        window.requestAnimationFrame(tick);
      }}
    }});
  }});

  canvas.addEventListener("click", (event) => {{
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    audio.currentTime = ratio * clipDuration;
    draw();
  }});

  draw();
</script>
"""


def _spectrogram_png_base64(spectrogram_db: np.ndarray) -> str:
    if spectrogram_db.size == 0:
        empty = Image.new("RGBA", (2, 2), color=(255, 255, 255, 255))
        buffer = io.BytesIO()
        empty.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    clipped = np.clip(np.asarray(spectrogram_db, dtype=np.float32), -90.0, 0.0)
    normalized = (clipped + 90.0) / 90.0
    rgba = (matplotlib.colormaps["magma"](normalized) * 255).astype(np.uint8)
    rgba = np.flipud(rgba)
    image = Image.fromarray(rgba)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_synced_overview_player_html(
    audio_bytes: bytes,
    audio_mime: str,
    samples: np.ndarray,
    sr: int,
    spectrogram_db: np.ndarray,
    spectrogram_times: np.ndarray,
    spectrogram_freqs: np.ndarray,
    duration_sec: float,
    initial_wave_amplitude_max: float | None = None,
    initial_freq_max_hz: float | None = None,
) -> str:
    if samples.size == 0:
        return """
<div style="font-family: sans-serif; color: #666; padding: 1rem;">
  没有可显示的同步总览
</div>
"""

    sample_count = samples.shape[-1]
    duration_sec = max(float(duration_sec), 1e-6)
    bucket_count = int(min(2200, max(480, sample_count // 256)))
    buckets = np.array_split(np.abs(np.asarray(samples, dtype=np.float32)), bucket_count)
    envelope = [round(float(np.max(bucket)) if bucket.size else 0.0, 6) for bucket in buckets]
    envelope_json = json.dumps(envelope, ensure_ascii=False)
    if initial_wave_amplitude_max is None:
        envelope_array = np.asarray(envelope, dtype=np.float32)
        if envelope_array.size:
            auto_wave_amplitude_max = float(np.quantile(envelope_array, 0.98) * 1.08)
        else:
            auto_wave_amplitude_max = 0.3
        visible_wave_amplitude_max = max(0.03, min(1.0, auto_wave_amplitude_max))
    else:
        visible_wave_amplitude_max = max(0.03, min(1.0, float(initial_wave_amplitude_max)))

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    spectrogram_b64 = _spectrogram_png_base64(spectrogram_db)
    full_freq_max = float(spectrogram_freqs[-1]) if spectrogram_freqs.size else 22050.0
    visible_freq_max = full_freq_max
    if initial_freq_max_hz is not None:
        visible_freq_max = max(1000.0, min(float(initial_freq_max_hz), full_freq_max))
    freq_slider_max = max(1000.0, float(int(np.ceil(full_freq_max / 500.0) * 500.0)))

    spec_time_start = float(spectrogram_times[0]) if spectrogram_times.size else 0.0
    spec_time_end = float(spectrogram_times[-1]) if spectrogram_times.size else duration_sec
    spec_time_span = max(spec_time_end - spec_time_start, 1e-6)
    window_slider_max = max(5.0, float(int(np.ceil(duration_sec))))
    initial_window_sec = window_slider_max
    default_focus_window_sec = min(
        duration_sec,
        max(15.0, min(30.0, duration_sec / 6.0 if duration_sec > 60 else duration_sec / 2.0)),
    )

    return f"""
<div style="font-family: 'Helvetica Neue', Arial, sans-serif; color: #1f1f1f;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; font-size:0.95rem;">
    <div>同步总览：纯频谱 + 波形</div>
    <div id="overviewTimeLabel" style="color:#666;">00:00.00 / {format_seconds(duration_sec)}</div>
  </div>
  <audio id="overviewAudio" controls preload="auto" style="width:100%; margin-bottom:0.6rem;">
    <source src="data:{audio_mime};base64,{audio_b64}" type="{audio_mime}" />
  </audio>
  <div style="display:grid; grid-template-columns: 1fr; gap: 0.8rem; margin-bottom:0.6rem; font-size:0.88rem;">
    <div style="display:flex; justify-content:flex-start; margin-bottom:0.1rem;">
      <button
        id="overviewViewModeToggle"
        type="button"
        style="border:1px solid #b8c4d3; background:#f6f9fc; color:#243447; border-radius:999px; padding:0.35rem 0.8rem; font-size:0.82rem; cursor:pointer;"
      >
        切换到局部聚焦
      </button>
    </div>
    <label style="display:flex; flex-direction:column; gap:0.2rem;">
      <span>聚焦窗口：<span id="overviewWindowValue">{initial_window_sec:.0f}</span> 秒</span>
      <input id="overviewWindowSec" type="range" min="5" max="{int(window_slider_max)}" step="1" value="{initial_window_sec:.0f}" />
    </label>
  </div>
  <div id="overviewWindowLabel" style="margin-bottom:0.45rem; font-size:0.84rem; color:#5c6773;">
    当前聚焦区间：00:00.00 - {format_seconds(min(duration_sec, initial_window_sec))}
  </div>
  <div style="display:flex; align-items:stretch; gap:0.8rem; margin-bottom:0.55rem; overflow:visible;">
    <div style="flex:1 1 auto; min-width:0; display:flex; flex-direction:column; gap:0.35rem; overflow:visible;">
      <canvas id="overviewWaveCanvas" width="1400" height="230" style="flex:1; width:100%; height:230px; border:1px solid #e3e3e3; border-radius:8px; background:#fbfbfc; cursor:crosshair; display:block;"></canvas>
      <div id="overviewWaveHover" style="font-size:0.82rem; color:#5c6773;">悬停波形：时间 -- | 包络振幅 --</div>
    </div>
    <div id="overviewWaveControl" style="flex:0 0 72px; width:72px; border:1px solid #d9e0e8; border-radius:10px; background:#f7f9fc; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:0.5rem 0.35rem; gap:0.45rem;">
      <div style="font-size:0.78rem; color:#4f5d75; text-align:center;">波形纵轴上限</div>
      <div id="overviewWaveAmplitudeValue" style="font-size:0.92rem; font-weight:600; color:#1f2d3d;">±{float(visible_wave_amplitude_max):.2f}</div>
      <input
        id="overviewWaveAmplitudeMax"
        type="range"
        min="0.10"
        max="1.00"
        step="0.05"
        value="{float(visible_wave_amplitude_max):.2f}"
        orient="vertical"
        style="-webkit-appearance: slider-vertical; writing-mode: vertical-lr; direction: rtl; width: 26px; height: 150px;"
      />
    </div>
  </div>
  <div style="display:flex; align-items:stretch; gap:0.8rem; overflow:visible;">
    <div style="flex:1 1 auto; min-width:0; display:flex; flex-direction:column; gap:0.35rem; overflow:visible;">
      <canvas id="overviewSpecCanvas" width="1400" height="360" style="flex:1; width:100%; height:360px; border:1px solid #e3e3e3; border-radius:8px; background:#111; cursor:crosshair; display:block;"></canvas>
      <div id="overviewSpecHover" style="font-size:0.82rem; color:#5c6773;">悬停频谱：时间 -- | 频率 --</div>
    </div>
    <div id="overviewFreqControl" style="flex:0 0 84px; width:84px; border:1px solid #2a2f39; border-radius:10px; background:#171b22; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:0.5rem 0.35rem; gap:0.45rem;">
      <div style="font-size:0.78rem; color:#d7dde8; text-align:center;">频率上限</div>
      <div id="overviewFreqValue" style="font-size:0.92rem; font-weight:600; color:#ffffff;">{int(visible_freq_max)} Hz</div>
      <input
        id="overviewFreqMax"
        type="range"
        min="1000"
        max="{int(freq_slider_max)}"
        step="250"
        value="{int(visible_freq_max)}"
        orient="vertical"
        style="-webkit-appearance: slider-vertical; writing-mode: vertical-lr; direction: rtl; width: 26px; height: 230px;"
      />
    </div>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:0.35rem; font-size:0.82rem; color:#666;">
    <span id="overviewWindowStart">00:00.00</span>
    <span>滚轮缩放时间范围，拖拽平移窗口，双击回到播放头；上面的进度条拖到哪里，下面就聚焦哪里</span>
    <span id="overviewWindowEnd">{format_seconds(min(duration_sec, initial_window_sec))}</span>
  </div>
</div>
<script>
  const audio = document.getElementById("overviewAudio");
  const waveCanvas = document.getElementById("overviewWaveCanvas");
  const specCanvas = document.getElementById("overviewSpecCanvas");
  const waveCtx = waveCanvas.getContext("2d");
  const specCtx = specCanvas.getContext("2d");
  const timeLabel = document.getElementById("overviewTimeLabel");
  const viewModeToggle = document.getElementById("overviewViewModeToggle");
  const windowInput = document.getElementById("overviewWindowSec");
  const waveAmplitudeInput = document.getElementById("overviewWaveAmplitudeMax");
  const freqInput = document.getElementById("overviewFreqMax");
  const windowValue = document.getElementById("overviewWindowValue");
  const waveAmplitudeValue = document.getElementById("overviewWaveAmplitudeValue");
  const freqValue = document.getElementById("overviewFreqValue");
  const waveHoverLabel = document.getElementById("overviewWaveHover");
  const specHoverLabel = document.getElementById("overviewSpecHover");
  const windowLabel = document.getElementById("overviewWindowLabel");
  const windowStartLabel = document.getElementById("overviewWindowStart");
  const windowEndLabel = document.getElementById("overviewWindowEnd");
  const envelope = {envelope_json};
  const duration = {duration_sec:.6f};
  const specTimeStart = {spec_time_start:.6f};
  const specTimeSpan = {spec_time_span:.6f};
  const fullFreqMax = {full_freq_max:.3f};
  const spectrogramImage = new Image();
  spectrogramImage.src = "data:image/png;base64,{spectrogram_b64}";
  const waveAxisPad = 96;
  const specAxisPad = 120;
  const rightPad = 12;
  const waveTopPad = 16;
  const waveBottomPad = 16;
  const specTopPad = 18;
  const specBottomPad = 22;
  const minViewSpan = Math.min(duration, 1.0);
  let viewSpan = Math.min(duration, {initial_window_sec:.6f});
  let viewStart = 0;
  let followPlayhead = true;
  let dragState = null;
  let lastFocusSpan = Math.max(minViewSpan, Math.min(duration, {default_focus_window_sec:.6f}));

  function formatTime(totalSeconds) {{
    const safe = Math.max(0, totalSeconds);
    const minutes = Math.floor(safe / 60);
    const seconds = safe - minutes * 60;
    return `${{String(minutes).padStart(2, "0")}}:${{seconds.toFixed(2).padStart(5, "0")}}`;
  }}

  function clampView(start, span) {{
    const safeSpan = Math.max(minViewSpan, Math.min(duration, span));
    const safeStart = Math.max(0, Math.min(duration - safeSpan, start));
    return {{
      start: safeStart,
      end: safeStart + safeSpan,
      span: safeSpan,
    }};
  }}

  function centeredView(center, span) {{
    return clampView(center - span / 2, span);
  }}

  function currentWindowBounds() {{
    return clampView(viewStart, viewSpan);
  }}

  function isFullView(windowBounds) {{
    return windowBounds.start <= 1e-3 && Math.abs(windowBounds.span - duration) <= 0.5;
  }}

  function syncViewToPlayhead() {{
    const centered = centeredView(audio.currentTime || 0, viewSpan);
    viewStart = centered.start;
    viewSpan = centered.span;
  }}

  function playheadX(currentTime, left, plotWidth, windowStart, windowSpan) {{
    const relative = windowSpan > 0 ? (currentTime - windowStart) / windowSpan : 0;
    return left + Math.max(0, Math.min(1, relative)) * plotWidth;
  }}

  function canvasMetrics(canvasEl) {{
    const rect = canvasEl.getBoundingClientRect();
    const internalWidth = canvasEl.width;
    const leftPadInternal = canvasEl === specCanvas ? specAxisPad : waveAxisPad;
    const rightPadInternal = rightPad;
    const leftPadDisplay = (leftPadInternal / internalWidth) * rect.width;
    const rightPadDisplay = (rightPadInternal / internalWidth) * rect.width;
    return {{
      rect,
      leftPadDisplay,
      rightPadDisplay,
      usableDisplayWidth: Math.max(1, rect.width - leftPadDisplay - rightPadDisplay),
    }};
  }}

  function eventRatioWithinPlot(event, canvasEl) {{
    const metrics = canvasMetrics(canvasEl);
    const localX = event.clientX - metrics.rect.left;
    const clampedX = Math.max(
      metrics.leftPadDisplay,
      Math.min(metrics.rect.width - metrics.rightPadDisplay, localX),
    );
    return Math.max(0, Math.min(1, (clampedX - metrics.leftPadDisplay) / metrics.usableDisplayWidth));
  }}

  function eventYRatioWithinCanvas(event, canvasEl) {{
    const rect = canvasEl.getBoundingClientRect();
    const topPad = canvasEl === specCanvas ? specTopPad : waveTopPad;
    const bottomPad = canvasEl === specCanvas ? specBottomPad : waveBottomPad;
    const internalHeight = canvasEl.height;
    const topPadDisplay = (topPad / internalHeight) * rect.height;
    const bottomPadDisplay = (bottomPad / internalHeight) * rect.height;
    const usableDisplayHeight = Math.max(1, rect.height - topPadDisplay - bottomPadDisplay);
    const localY = event.clientY - rect.top;
    const clampedY = Math.max(topPadDisplay, Math.min(rect.height - bottomPadDisplay, localY));
    return Math.max(0, Math.min(1, (clampedY - topPadDisplay) / usableDisplayHeight));
  }}

  function getVisibleEnvelope(windowBounds) {{
    const startIndex = Math.max(0, Math.floor((windowBounds.start / duration) * envelope.length));
    const endIndex = Math.min(envelope.length, Math.ceil((windowBounds.end / duration) * envelope.length));
    return envelope.slice(startIndex, Math.max(startIndex + 1, endIndex));
  }}

  function applyWindowFromSlider() {{
    const requested = parseFloat(windowInput.value || "{initial_window_sec:.0f}");
    const safeSpan = Math.max(minViewSpan, Math.min(duration, requested));
    const center = followPlayhead ? (audio.currentTime || 0) : (viewStart + viewSpan / 2);
    const next = centeredView(center, safeSpan);
    viewStart = next.start;
    viewSpan = next.span;
  }}

  function zoomAroundPointer(event, canvasEl) {{
    event.preventDefault();
    const ratio = eventRatioWithinPlot(event, canvasEl);
    const zoomFactor = event.deltaY < 0 ? 0.84 : 1.18;
    const anchorTime = viewStart + ratio * viewSpan;
    const nextSpan = Math.max(minViewSpan, Math.min(duration, viewSpan * zoomFactor));
    const nextStart = anchorTime - ratio * nextSpan;
    const next = clampView(nextStart, nextSpan);
    viewStart = next.start;
    viewSpan = next.span;
    followPlayhead = false;
    drawAll();
  }}

  function pointerDown(event, canvasEl) {{
    dragState = {{
      canvasEl,
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startViewStart: viewStart,
      moved: false,
    }};
    canvasEl.setPointerCapture(event.pointerId);
  }}

  function pointerMove(event) {{
    if (!dragState || dragState.pointerId !== event.pointerId) {{
      return;
    }}
    const metrics = canvasMetrics(dragState.canvasEl);
    const deltaX = event.clientX - dragState.startClientX;
    if (Math.abs(deltaX) > 3) {{
      dragState.moved = true;
    }}
    if (!dragState.moved) {{
      return;
    }}
    const deltaSeconds = (deltaX / metrics.usableDisplayWidth) * viewSpan;
    const next = clampView(dragState.startViewStart - deltaSeconds, viewSpan);
    viewStart = next.start;
    viewSpan = next.span;
    followPlayhead = false;
    drawAll();
  }}

  function pointerUp(event) {{
    if (!dragState || dragState.pointerId !== event.pointerId) {{
      return;
    }}
    const canvasEl = dragState.canvasEl;
    const wasMoved = dragState.moved;
    dragState = null;
    canvasEl.releasePointerCapture(event.pointerId);
    if (!wasMoved) {{
      const ratio = eventRatioWithinPlot(event, canvasEl);
      const windowBounds = currentWindowBounds();
      audio.currentTime = windowBounds.start + ratio * windowBounds.span;
      if (followPlayhead) {{
        syncViewToPlayhead();
      }}
      drawAll();
    }}
  }}

  function resetToPlayhead() {{
    followPlayhead = true;
    applyWindowFromSlider();
    syncViewToPlayhead();
    drawAll();
  }}

  function toggleOverviewMode() {{
    const windowBounds = currentWindowBounds();
    if (isFullView(windowBounds)) {{
      followPlayhead = true;
      viewSpan = lastFocusSpan;
      if (windowInput) {{
        windowInput.value = `${{Math.round(viewSpan)}}`;
      }}
      syncViewToPlayhead();
    }} else {{
      lastFocusSpan = windowBounds.span;
      followPlayhead = false;
      viewStart = 0;
      viewSpan = duration;
      if (windowInput) {{
        windowInput.value = `${{Math.round(duration)}}`;
      }}
    }}
    drawAll();
  }}

  function drawWaveform() {{
    const width = waveCanvas.width;
    const height = waveCanvas.height;
    const plotTop = waveTopPad;
    const plotBottom = height - waveBottomPad;
    const plotHeight = Math.max(1, plotBottom - plotTop);
    const mid = plotTop + plotHeight / 2;
    const amplitudeMax = Math.max(0.05, parseFloat(waveAmplitudeInput.value));
    const plotLeft = waveAxisPad;
    const plotRight = width - rightPad;
    const plotWidth = Math.max(1, plotRight - plotLeft);
    const windowBounds = currentWindowBounds();
    const visibleEnvelope = getVisibleEnvelope(windowBounds);
    waveCtx.clearRect(0, 0, width, height);
    waveCtx.fillStyle = "#fbfbfc";
    waveCtx.fillRect(0, 0, width, height);

    waveCtx.fillStyle = "#4f5d75";
    waveCtx.font = "13px sans-serif";
    waveCtx.textAlign = "right";
    waveCtx.textBaseline = "middle";
    const waveTicks = [amplitudeMax, amplitudeMax / 2, 0.0, -amplitudeMax / 2, -amplitudeMax];
    for (const tick of waveTicks) {{
      const normalized = amplitudeMax > 0 ? tick / amplitudeMax : 0;
      const y = mid - normalized * (plotHeight * 0.42);
      waveCtx.fillText(tick > 0 ? `+${{tick.toFixed(2)}}` : tick.toFixed(2), plotLeft - 12, y);
      waveCtx.strokeStyle = "rgba(79, 93, 117, 0.18)";
      waveCtx.lineWidth = 1;
      waveCtx.beginPath();
      waveCtx.moveTo(plotLeft - 4, y);
      waveCtx.lineTo(plotRight, y);
      waveCtx.stroke();
    }}
    waveCtx.save();
    waveCtx.translate(24, plotTop + plotHeight / 2);
    waveCtx.rotate(-Math.PI / 2);
    waveCtx.fillStyle = "#4f5d75";
    waveCtx.textAlign = "center";
    waveCtx.fillText("振幅", 0, 0);
    waveCtx.restore();

    waveCtx.strokeStyle = "#9aa5b1";
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    waveCtx.moveTo(plotLeft, plotTop);
    waveCtx.lineTo(plotLeft, plotBottom);
    waveCtx.moveTo(plotLeft, mid);
    waveCtx.lineTo(plotRight, mid);
    waveCtx.stroke();

    const barWidth = plotWidth / visibleEnvelope.length;
    waveCtx.fillStyle = "#1f5c99";
    for (let i = 0; i < visibleEnvelope.length; i += 1) {{
      const amp = Math.max(0.01, visibleEnvelope[i]);
      const normalizedAmp = Math.min(1.0, amp / amplitudeMax);
      const barHeight = normalizedAmp * (plotHeight * 0.42);
      const x = plotLeft + i * barWidth;
      waveCtx.fillRect(x, mid - barHeight, Math.max(1, barWidth * 0.86), barHeight * 2);
    }}

    const x = playheadX(audio.currentTime, plotLeft, plotWidth, windowBounds.start, windowBounds.span);
    waveCtx.strokeStyle = "#17c964";
    waveCtx.lineWidth = 2.5;
    waveCtx.beginPath();
    waveCtx.moveTo(x, plotTop);
    waveCtx.lineTo(x, plotBottom);
    waveCtx.stroke();
  }}

  function drawSpectrogram() {{
    const width = specCanvas.width;
    const height = specCanvas.height;
    const plotTop = specTopPad;
    const plotBottom = height - specBottomPad;
    const plotHeight = Math.max(1, plotBottom - plotTop);
    const freqMax = parseFloat(freqInput.value);
    const plotLeft = specAxisPad;
    const plotRight = width - rightPad;
    const plotWidth = Math.max(1, plotRight - plotLeft);
    const windowBounds = currentWindowBounds();
    specCtx.clearRect(0, 0, width, height);
    specCtx.fillStyle = "#111";
    specCtx.fillRect(0, 0, width, height);

    if (spectrogramImage.complete) {{
      const sourceHeight = spectrogramImage.height * (freqMax / fullFreqMax);
      const sourceY = spectrogramImage.height - sourceHeight;
      const timeStartRatio = specTimeSpan > 0 ? (windowBounds.start - specTimeStart) / specTimeSpan : 0;
      const timeEndRatio = specTimeSpan > 0 ? (windowBounds.end - specTimeStart) / specTimeSpan : 1;
      const sourceX = Math.max(0, Math.min(spectrogramImage.width - 1, timeStartRatio * spectrogramImage.width));
      const sourceX2 = Math.max(sourceX + 1, Math.min(spectrogramImage.width, timeEndRatio * spectrogramImage.width));
      specCtx.drawImage(
        spectrogramImage,
        sourceX,
        sourceY,
        sourceX2 - sourceX,
        sourceHeight,
        plotLeft,
        plotTop,
        plotWidth,
        plotHeight,
      );
    }}

    specCtx.strokeStyle = "rgba(255,255,255,0.22)";
    specCtx.lineWidth = 1;
    specCtx.beginPath();
    specCtx.moveTo(plotLeft, plotTop);
    specCtx.lineTo(plotLeft, plotBottom);
    specCtx.stroke();

    specCtx.fillStyle = "#ffffff";
    specCtx.font = "13px sans-serif";
    specCtx.textAlign = "right";
    specCtx.textBaseline = "middle";
    const freqTicks = 5;
    for (let i = 0; i < freqTicks; i += 1) {{
      const ratio = i / (freqTicks - 1);
      const y = plotBottom - ratio * plotHeight;
      const tickFreq = ratio * freqMax;
      specCtx.fillText(`${{Math.round(tickFreq)}}`, plotLeft - 12, y);
      specCtx.strokeStyle = "rgba(255,255,255,0.14)";
      specCtx.lineWidth = 1;
      specCtx.beginPath();
      specCtx.moveTo(plotLeft - 4, y);
      specCtx.lineTo(plotRight, y);
      specCtx.stroke();
    }}
    specCtx.save();
    specCtx.translate(28, plotTop + plotHeight / 2);
    specCtx.rotate(-Math.PI / 2);
    specCtx.fillStyle = "#ffffff";
    specCtx.textAlign = "center";
    specCtx.fillText("频率 (Hz)", 0, 0);
    specCtx.restore();

    const x = playheadX(audio.currentTime, plotLeft, plotWidth, windowBounds.start, windowBounds.span);
    specCtx.strokeStyle = "#17c964";
    specCtx.lineWidth = 2.5;
    specCtx.beginPath();
    specCtx.moveTo(x, plotTop);
    specCtx.lineTo(x, plotBottom);
    specCtx.stroke();
  }}

  function updateWaveHover(event) {{
    const ratio = eventRatioWithinPlot(event, waveCanvas);
    const windowBounds = currentWindowBounds();
    const hoverTime = windowBounds.start + ratio * windowBounds.span;
    const visibleEnvelope = getVisibleEnvelope(windowBounds);
    const index = Math.min(visibleEnvelope.length - 1, Math.max(0, Math.floor(ratio * visibleEnvelope.length)));
    const amplitude = visibleEnvelope[index] || 0;
    waveHoverLabel.textContent = `悬停波形：时间 ${{formatTime(hoverTime)}} | 包络振幅 ${{amplitude.toFixed(4)}}`;
  }}

  function updateSpecHover(event) {{
    const ratio = eventRatioWithinPlot(event, specCanvas);
    const yRatio = eventYRatioWithinCanvas(event, specCanvas);
    const windowBounds = currentWindowBounds();
    const hoverTime = windowBounds.start + ratio * windowBounds.span;
    const freqMax = parseFloat(freqInput.value);
    const hoverFreq = (1 - yRatio) * freqMax;
    specHoverLabel.textContent = `悬停频谱：时间 ${{formatTime(hoverTime)}} | 频率 ${{hoverFreq.toFixed(1)}} Hz`;
  }}

  function resetHoverLabels() {{
    waveHoverLabel.textContent = "悬停波形：时间 -- | 包络振幅 --";
    specHoverLabel.textContent = "悬停频谱：时间 -- | 频率 --";
  }}

  function drawAll() {{
    if (followPlayhead) {{
      syncViewToPlayhead();
    }}
    const windowBounds = currentWindowBounds();
    if (!isFullView(windowBounds)) {{
      lastFocusSpan = windowBounds.span;
    }}
    windowInput.value = `${{Math.round(windowBounds.span)}}`;
    windowValue.textContent = `${{Math.round(windowBounds.span)}}`;
    waveAmplitudeValue.textContent = `±${{parseFloat(waveAmplitudeInput.value).toFixed(2)}}`;
    freqValue.textContent = `${{Math.round(parseFloat(freqInput.value))}}`;
    viewModeToggle.textContent = isFullView(windowBounds) ? "切换到局部聚焦" : "切换到全曲总览";
    windowLabel.textContent = `当前聚焦区间：${{formatTime(windowBounds.start)}} - ${{formatTime(windowBounds.end)}}`;
    windowStartLabel.textContent = formatTime(windowBounds.start);
    windowEndLabel.textContent = formatTime(windowBounds.end);
    drawWaveform();
    drawSpectrogram();
    timeLabel.textContent = `${{formatTime(audio.currentTime || 0)}} / {format_seconds(duration_sec)}`;
  }}

  function tick() {{
    drawAll();
    if (!audio.paused && !audio.ended) {{
      window.requestAnimationFrame(tick);
    }}
  }}

  ["play", "pause", "seeked", "loadedmetadata", "timeupdate", "ended"].forEach((eventName) => {{
    audio.addEventListener(eventName, () => {{
      if (eventName === "seeked" || eventName === "loadedmetadata") {{
        if (followPlayhead) {{
          syncViewToPlayhead();
        }} else {{
          const next = centeredView(audio.currentTime || 0, viewSpan);
          viewStart = next.start;
          viewSpan = next.span;
        }}
      }}
      drawAll();
      if (eventName === "play") {{
        window.requestAnimationFrame(tick);
      }}
    }});
  }});

  windowInput.addEventListener("input", () => {{
    followPlayhead = false;
    applyWindowFromSlider();
    drawAll();
  }});
  viewModeToggle.addEventListener("click", toggleOverviewMode);
  waveAmplitudeInput.addEventListener("input", drawAll);
  freqInput.addEventListener("input", drawAll);
  [waveCanvas, specCanvas].forEach((canvasEl) => {{
    canvasEl.addEventListener("wheel", (event) => zoomAroundPointer(event, canvasEl), {{ passive: false }});
    canvasEl.addEventListener("pointerdown", (event) => pointerDown(event, canvasEl));
    canvasEl.addEventListener("pointermove", pointerMove);
    canvasEl.addEventListener("pointerup", pointerUp);
    canvasEl.addEventListener("pointercancel", pointerUp);
    canvasEl.addEventListener("dblclick", resetToPlayhead);
  }});
  waveCanvas.addEventListener("mousemove", updateWaveHover);
  specCanvas.addEventListener("mousemove", updateSpecHover);
  waveCanvas.addEventListener("mouseleave", resetHoverLabels);
  specCanvas.addEventListener("mouseleave", resetHoverLabels);
  syncViewToPlayhead();
  spectrogramImage.onload = drawAll;
  resetHoverLabels();
  drawAll();
</script>
"""


def plot_local_spectrogram(
    spectrogram_db: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    center_time: float,
    window_radius_sec: float,
    highlight_time: float | None = None,
    max_freq_hz: float | None = 10000.0,
) -> plt.Figure:
    if times.size == 0 or freqs.size == 0 or spectrogram_db.size == 0:
        figure, axis = plt.subplots(figsize=(8.2, 3.8))
        axis.text(0.5, 0.5, "没有可显示的局部频谱", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.tight_layout()
        return figure

    start_time = max(float(times[0]), float(center_time - window_radius_sec))
    end_time = min(float(times[-1]), float(center_time + window_radius_sec))
    time_mask = (times >= start_time) & (times <= end_time)
    if not np.any(time_mask):
        nearest_time = int(np.argmin(np.abs(times - center_time)))
        time_mask[max(0, nearest_time - 2) : min(times.size, nearest_time + 3)] = True

    freq_mask = np.ones_like(freqs, dtype=bool)
    if max_freq_hz is not None:
        freq_mask = freqs <= float(max_freq_hz)
        if not np.any(freq_mask):
            freq_mask[:] = True

    local_times = times[time_mask]
    local_freqs = freqs[freq_mask]
    local_db = spectrogram_db[np.ix_(freq_mask, time_mask)]

    figure, axis = plt.subplots(figsize=(8.8, 4.2))
    mesh = axis.pcolormesh(
        local_times,
        local_freqs / 1000.0,
        local_db,
        shading="auto",
        cmap="magma",
    )

    if highlight_time is None:
        highlight_time = center_time
    axis.axvline(float(highlight_time), color="#ffffff", alpha=0.95, linewidth=1.2)
    axis.set_title(f"事件附近局部频谱：{center_time:.2f}s")
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("频率（kHz）", labelpad=10)
    axis.set_xlim(float(local_times[0]), float(local_times[-1]))
    colorbar = figure.colorbar(mesh, ax=axis, format="%+2.0f dB", pad=0.02, fraction=0.04)
    colorbar.set_label("相对幅度（dB）")
    figure.subplots_adjust(left=0.15, right=0.90, bottom=0.12, top=0.90)
    return figure


def plot_event_density(event_times: np.ndarray, duration_sec: float) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(14, 2.8))
    if event_times.size == 0:
        axis.text(0.5, 0.5, "未检测到事件", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return figure

    bins = min(24, max(6, int(np.ceil(duration_sec / 20.0))))
    axis.hist(event_times, bins=bins, color="#457b9d", alpha=0.85)
    axis.set_title("事件密度分布")
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("事件数量")
    axis.grid(alpha=0.2, linestyle="--")
    figure.tight_layout()
    return figure


def build_summary_text(summary_lines: list[str]) -> str:
    return "\n".join(summary_lines)


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _event_group_palette(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["#4c78a8"]
    colors: list[str] = []
    for index in range(count):
        hue = float(index) / max(count, 1)
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.90)
        colors.append(f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}")
    return colors


def build_interactive_novelty_chart(feature_table: pd.DataFrame, event_table: pd.DataFrame) -> alt.Chart:
    selector = alt.selection_point(
        name="event_pick",
        fields=["event_id"],
        on="click",
        empty=False,
    )

    event_source = event_table.copy()
    if event_source.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point().encode(x="x:Q", y="y:Q")

    event_source["boundary_type"] = np.where(
        event_source["is_major_boundary"],
        "候选边界",
        "普通事件",
    )
    if "interaction_priority_label" not in event_source.columns:
        event_source["interaction_priority_label"] = np.where(
            event_source["is_major_boundary"],
            "重点变化",
            "一般变化",
        )
    if "interaction_priority" not in event_source.columns:
        event_source["interaction_priority"] = np.where(
            event_source["is_major_boundary"],
            "strong",
            "medium",
        )
    if "similarity_group_label" not in event_source.columns:
        event_source["similarity_group_label"] = [f"相似组 {index}" for index in range(1, len(event_source) + 1)]
    if "similarity_group_size" not in event_source.columns:
        event_source["similarity_group_size"] = 1
    if "max_similarity_in_group" not in event_source.columns:
        event_source["max_similarity_in_group"] = 1.0

    size_map = {"strong": 210, "medium": 135, "weak": 78}
    opacity_map = {"strong": 0.98, "medium": 0.82, "weak": 0.42}
    event_source["point_size"] = event_source["interaction_priority"].map(size_map).fillna(135)
    event_source["point_opacity"] = event_source["interaction_priority"].map(opacity_map).fillna(0.82)
    similarity_domain = list(dict.fromkeys(event_source["similarity_group_label"].astype(str).tolist()))
    similarity_range = _event_group_palette(len(similarity_domain))

    chart = (
        alt.Chart(event_source)
        .mark_circle(size=110)
        .encode(
            x=alt.X("time_sec:Q", title="时间（秒）"),
            y=alt.Y("strength:Q", title="事件强度"),
            size=alt.Size("point_size:Q", legend=None),
            opacity=alt.Opacity("point_opacity:Q", legend=None),
            color=alt.condition(
                selector,
                alt.value("#d62828"),
                alt.Color(
                    "similarity_group_label:N",
                    title="相似事件组",
                    scale=alt.Scale(
                        domain=similarity_domain,
                        range=similarity_range,
                    ),
                ),
            ),
            tooltip=[
                alt.Tooltip("event_id:Q", title="事件"),
                alt.Tooltip("time_label:N", title="时间"),
                alt.Tooltip("strength:Q", title="强度", format=".3f"),
                alt.Tooltip("prominence:Q", title="显著度", format=".3f"),
                alt.Tooltip("similarity_group_label:N", title="相似组"),
                alt.Tooltip("similarity_group_size:Q", title="组内事件数"),
                alt.Tooltip("max_similarity_in_group:Q", title="组内最高相似度", format=".0%"),
                alt.Tooltip("interaction_priority_label:N", title="交互层级"),
                alt.Tooltip("boundary_type:N", title="类型"),
                alt.Tooltip("auto_labels:N", title="候选标签"),
            ],
        )
        .add_params(selector)
        .properties(
            height=320,
            title="可点击的自动事件点图",
        )
    )
    return chart


def build_feature_curve_chart(
    feature_table: pd.DataFrame,
    selected_features: list[str],
    feature_labels: dict[str, str],
    normalize: bool = True,
) -> alt.Chart:
    if not selected_features:
        selected_features = ["rms", "spectral_flux", "novelty"]

    working = feature_table[["time_sec", "time_label", *selected_features]].copy()
    if normalize:
        for feature_name in selected_features:
            values = working[feature_name].astype(float).to_numpy()
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std < 1e-8:
                working[feature_name] = 0.0
            else:
                working[feature_name] = (values - mean) / std

    long_df = working.melt(
        id_vars=["time_sec", "time_label"],
        value_vars=selected_features,
        var_name="feature_name",
        value_name="value",
    )
    long_df["feature_label"] = long_df["feature_name"].map(feature_labels).fillna(long_df["feature_name"])

    chart = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time_sec:Q", title="时间（秒）"),
            y=alt.Y("value:Q", title="归一化特征值" if normalize else "特征值"),
            color=alt.Color("feature_label:N", title="特征"),
            tooltip=[
                alt.Tooltip("time_label:N", title="时间"),
                alt.Tooltip("feature_label:N", title="特征"),
                alt.Tooltip("value:Q", title="值", format=".4f"),
            ],
        )
        .properties(height=320, title="特征曲线")
    )
    return chart


def extract_selected_point_value(
    event_state: Any,
    selection_name: str,
    field_name: str,
) -> int | float | str | None:
    if not event_state:
        return None

    selection = None
    if isinstance(event_state, dict):
        selection = event_state.get("selection")
    else:
        selection = getattr(event_state, "selection", None)
    if selection is None:
        return None

    payload = selection.get(selection_name) if hasattr(selection, "get") else None
    if payload is None:
        payload = getattr(selection, selection_name, None)
    if payload is None:
        return None

    if isinstance(payload, dict):
        value = payload.get(field_name)
        if isinstance(value, list) and value:
            return value[0]
        if value is not None:
            return value
        if len(payload) == 1:
            only_value = next(iter(payload.values()))
            if isinstance(only_value, list) and only_value:
                return only_value[0]
            return only_value

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and field_name in first:
            return first[field_name]
        return first

    return None
