from __future__ import annotations

import io
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import audioread
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample_poly, spectrogram

ChannelMode = Literal["mix", "left", "right"]

EVENT_LABEL_VOCAB = [
    "高频扩展",
    "噪声侵入",
    "稳态持续",
    "材料聚集",
    "消散",
    "新事件出现",
    "段落转换",
    "音色突变",
    "能量增强",
    "能量减弱",
]

SECTION_LABEL_VOCAB = [
    "稳态持续",
    "材料聚集",
    "消散",
    "高频扩展",
    "噪声侵入",
    "能量增强",
    "能量减弱",
]

FEATURE_COLUMN_LABELS = {
    "rms": "RMS",
    "spectral_centroid_hz": "Spectral Centroid",
    "rolloff_hz": "Spectral Rolloff",
    "flatness": "Spectral Flatness",
    "spectral_flux": "Spectral Flux",
    "onset_strength": "Onset Strength",
    "novelty": "Novelty",
}


@dataclass(slots=True)
class AnalysisConfig:
    target_sr: int | None = None
    n_fft: int = 4096
    hop_length: int = 1024
    n_bands: int = 128
    smooth_sigma: float = 1.6
    threshold_sigma: float = 1.0
    prominence_sigma: float = 0.8
    min_event_distance_sec: float = 5.0
    context_window_sec: float = 3.0
    novelty_weight_cosine: float = 0.40
    novelty_weight_flux: float = 0.25
    novelty_weight_onset: float = 0.20
    novelty_weight_rms: float = 0.15
    event_model_preset: str = "balanced_default"

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def format_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    minutes = int(total_seconds // 60)
    secs = total_seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def join_labels(labels: list[str]) -> str:
    return " | ".join(_unique_labels(labels))


def split_label_text(label_text: str) -> list[str]:
    if not label_text:
        return []
    parts = [part.strip() for part in label_text.replace("；", "|").replace("、", "|").split("|")]
    return [part for part in parts if part]


def _unique_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


def _safe_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def _resolve_novelty_weights(config: AnalysisConfig) -> tuple[float, float, float, float]:
    weights = np.array(
        [
            float(config.novelty_weight_cosine),
            float(config.novelty_weight_flux),
            float(config.novelty_weight_onset),
            float(config.novelty_weight_rms),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)):
        return (0.40, 0.25, 0.20, 0.15)

    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(np.sum(weights))
    if total <= 1e-8:
        return (0.40, 0.25, 0.20, 0.15)

    normalized = weights / total
    return tuple(float(value) for value in normalized)


def _resolve_source_name(source: str | Path | BinaryIO) -> str:
    if isinstance(source, Path):
        return source.name
    if isinstance(source, str):
        return Path(source).name
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return Path(name).name
    return "audio_input"


def _write_temp_audio_file(source: BinaryIO) -> tuple[str, str | None]:
    source.seek(0)
    data = source.read()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("音频输入必须是字节流或文件路径。")

    suffix = Path(_resolve_source_name(source)).suffix or ".bin"
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(data)
        temporary.flush()
    finally:
        temporary.close()
    return temporary.name, suffix


def _read_with_audioread(source: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    temp_path: str | None = None
    audio_path: str
    if isinstance(source, (str, Path)):
        audio_path = str(source)
    else:
        audio_path, _ = _write_temp_audio_file(source)
        temp_path = audio_path

    try:
        with audioread.audio_open(audio_path) as handle:
            chunks: list[np.ndarray] = []
            channel_count = int(handle.channels)
            sample_rate = int(handle.samplerate)

            for chunk in handle:
                pcm = np.frombuffer(chunk, dtype="<i2")
                if pcm.size == 0:
                    continue
                trimmed = pcm[: pcm.size - (pcm.size % channel_count)]
                if trimmed.size == 0:
                    continue
                frames = trimmed.reshape(-1, channel_count).astype(np.float32) / 32768.0
                chunks.append(frames)

        if not chunks:
            raise RuntimeError("无法从音频中读取有效的 PCM 数据。")

        audio = np.concatenate(chunks, axis=0)
        return audio.T.astype(np.float32), sample_rate
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _read_audio(source: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        audio, sr = sf.read(source, always_2d=True, dtype="float32")
        return audio.T.astype(np.float32), int(sr)
    except Exception:
        try:
            return _read_with_audioread(source)
        except Exception as audioread_error:
            raise RuntimeError(
                "无法读取该音频文件。请尝试 WAV、FLAC、AIFF、MP3 或 M4A。"
            ) from audioread_error


def _load_audio(source: str | Path | BinaryIO, target_sr: int | None) -> tuple[np.ndarray, int]:
    channels, sr = _read_audio(source)

    if target_sr and target_sr != sr:
        gcd = math.gcd(int(sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(sr) // gcd
        resampled_channels: list[np.ndarray] = []
        for channel in channels:
            resampled = resample_poly(channel, up=up, down=down).astype(np.float32)
            resampled_channels.append(resampled)
        min_length = min(channel.shape[0] for channel in resampled_channels)
        channels = np.vstack([channel[:min_length] for channel in resampled_channels])
        sr = int(target_sr)

    return channels.astype(np.float32), int(sr)


def _select_channel(audio: np.ndarray, mode: ChannelMode) -> np.ndarray:
    if audio.shape[0] == 1:
        return audio[0]
    if mode == "left":
        return audio[0]
    if mode == "right":
        return audio[1]
    return np.mean(audio, axis=0)


def build_audio_excerpt_wav(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    channel_mode: ChannelMode = "mix",
) -> bytes:
    start_sample = max(0, int(start_sec * sr))
    end_sample = max(start_sample + 1, int(end_sec * sr))

    if audio.ndim == 1:
        clip = audio[start_sample:end_sample]
        clip_to_write = clip.reshape(-1, 1)
    else:
        if channel_mode == "left" and audio.shape[0] > 1:
            clip = audio[0, start_sample:end_sample]
            clip_to_write = clip.reshape(-1, 1)
        elif channel_mode == "right" and audio.shape[0] > 1:
            clip = audio[1, start_sample:end_sample]
            clip_to_write = clip.reshape(-1, 1)
        elif channel_mode == "mix":
            clip = np.mean(audio[:, start_sample:end_sample], axis=0)
            clip_to_write = clip.reshape(-1, 1)
        else:
            clip_to_write = audio[:, start_sample:end_sample].T

    buffer = io.BytesIO()
    sf.write(buffer, clip_to_write, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _spectrogram_params(sample_count: int, config: AnalysisConfig) -> tuple[int, int]:
    n_fft = min(config.n_fft, max(256, sample_count))
    hop = min(config.hop_length, max(64, n_fft // 4))
    if hop >= n_fft:
        hop = max(1, n_fft // 2)
    return int(n_fft), int(hop)


def _compute_spectrogram(
    samples: np.ndarray,
    sr: int,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    n_fft, hop = _spectrogram_params(samples.size, config)
    noverlap = max(0, n_fft - hop)
    freqs, times, spec = spectrogram(
        samples,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    return (
        freqs.astype(np.float32),
        times.astype(np.float32),
        spec.astype(np.float32),
        n_fft,
        hop,
    )


def _aggregate_frequency_bins(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if spectrum.shape[0] <= group_count:
        return freqs.astype(np.float32), spectrum.astype(np.float32)

    groups = np.array_split(np.arange(spectrum.shape[0]), group_count)
    aggregated_freqs = np.array([np.mean(freqs[group]) for group in groups], dtype=np.float32)
    aggregated_spectrum = np.stack(
        [np.mean(spectrum[group], axis=0) for group in groups],
        axis=0,
    ).astype(np.float32)
    return aggregated_freqs, aggregated_spectrum


def _aggregate_time_bins(
    times: np.ndarray,
    spectrum: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if spectrum.shape[1] <= group_count:
        return times.astype(np.float32), spectrum.astype(np.float32)

    groups = np.array_split(np.arange(spectrum.shape[1]), group_count)
    aggregated_times = np.array([np.mean(times[group]) for group in groups], dtype=np.float32)
    aggregated_spectrum = np.stack(
        [np.mean(spectrum[:, group], axis=1) for group in groups],
        axis=1,
    ).astype(np.float32)
    return aggregated_times, aggregated_spectrum


def _framewise_cosine_distance(log_spectrum: np.ndarray) -> np.ndarray:
    frame_count = log_spectrum.shape[1]
    distances = np.zeros(frame_count, dtype=np.float32)
    if frame_count < 2:
        return distances

    previous = log_spectrum[:, :-1]
    current = log_spectrum[:, 1:]
    previous_norm = np.linalg.norm(previous, axis=0)
    current_norm = np.linalg.norm(current, axis=0)
    denominator = np.maximum(previous_norm * current_norm, 1e-8)
    similarity = np.sum(previous * current, axis=0) / denominator
    distances[1:] = 1.0 - np.clip(similarity, -1.0, 1.0)
    return distances


def _framewise_rolloff(freqs: np.ndarray, magnitude: np.ndarray, fraction: float = 0.85) -> np.ndarray:
    magnitude = np.maximum(magnitude, 1e-10)
    cumulative = np.cumsum(magnitude, axis=0)
    thresholds = cumulative[-1, :] * fraction
    mask = cumulative >= thresholds[None, :]
    indices = np.argmax(mask, axis=0)
    return freqs[indices].astype(np.float32)


def _framewise_flatness(magnitude: np.ndarray) -> np.ndarray:
    magnitude = np.maximum(magnitude, 1e-10)
    geometric_mean = np.exp(np.mean(np.log(magnitude), axis=0))
    arithmetic_mean = np.mean(magnitude, axis=0)
    return (geometric_mean / np.maximum(arithmetic_mean, 1e-10)).astype(np.float32)


def _framewise_centroid_and_bandwidth(freqs: np.ndarray, magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe_magnitude = np.maximum(magnitude, 1e-10)
    total_energy = np.maximum(np.sum(safe_magnitude, axis=0), 1e-10)
    centroid = (np.sum(freqs[:, None] * safe_magnitude, axis=0) / total_energy).astype(np.float32)
    bandwidth = np.sqrt(
        np.sum(((freqs[:, None] - centroid[None, :]) ** 2) * safe_magnitude, axis=0) / total_energy
    ).astype(np.float32)
    return centroid, bandwidth


def _build_feature_table(
    times: np.ndarray,
    rms: np.ndarray,
    centroid: np.ndarray,
    bandwidth: np.ndarray,
    rolloff: np.ndarray,
    flatness: np.ndarray,
    spectral_flux: np.ndarray,
    onset_strength: np.ndarray,
    novelty: np.ndarray,
    threshold: np.ndarray,
) -> pd.DataFrame:
    frame_table = pd.DataFrame(
        {
            "time_sec": np.round(times.astype(np.float64), 6),
            "time_label": [format_seconds(value) for value in times],
            "rms": rms.astype(np.float64),
            "spectral_centroid_hz": centroid.astype(np.float64),
            "bandwidth_hz": bandwidth.astype(np.float64),
            "rolloff_hz": rolloff.astype(np.float64),
            "flatness": flatness.astype(np.float64),
            "spectral_flux": spectral_flux.astype(np.float64),
            "onset_strength": onset_strength.astype(np.float64),
            "novelty": novelty.astype(np.float64),
            "threshold": threshold.astype(np.float64),
        }
    )
    return frame_table


def _build_novelty_curve(samples: np.ndarray, sr: int, config: AnalysisConfig) -> dict[str, np.ndarray | int]:
    freqs, times, magnitude, n_fft, hop = _compute_spectrogram(samples, sr, config)
    log_magnitude = np.log1p(np.maximum(magnitude, 1e-10))

    _, band_spectrum = _aggregate_frequency_bins(freqs, magnitude, group_count=config.n_bands)
    log_bands = np.log1p(np.maximum(band_spectrum, 1e-10))

    positive_band_diff = np.maximum(np.diff(log_bands, axis=1), 0.0) if log_bands.shape[1] > 1 else np.empty((0, 0))
    spectral_flux = np.zeros(log_bands.shape[1], dtype=np.float32)
    if positive_band_diff.size:
        spectral_flux[1:] = np.mean(positive_band_diff, axis=0)

    positive_full_diff = np.maximum(np.diff(log_magnitude, axis=1), 0.0) if log_magnitude.shape[1] > 1 else np.empty((0, 0))
    onset_strength = np.zeros(log_magnitude.shape[1], dtype=np.float32)
    if positive_full_diff.size:
        freq_weights = np.linspace(0.7, 1.6, positive_full_diff.shape[0], dtype=np.float32)
        onset_strength[1:] = (
            np.sum(positive_full_diff * freq_weights[:, None], axis=0) / np.sum(freq_weights)
        ).astype(np.float32)
        onset_strength = gaussian_filter1d(onset_strength, sigma=1.0)

    cosine_distance = _framewise_cosine_distance(log_bands)
    rms = np.sqrt(np.mean(np.square(magnitude), axis=0, dtype=np.float64)).astype(np.float32)
    rms_delta = np.zeros_like(rms)
    if rms.size > 1:
        rms_delta[1:] = np.abs(np.diff(rms))

    centroid, bandwidth = _framewise_centroid_and_bandwidth(freqs, magnitude)
    rolloff = _framewise_rolloff(freqs, magnitude)
    flatness = _framewise_flatness(magnitude)

    cosine_weight, flux_weight, onset_weight, rms_weight = _resolve_novelty_weights(config)
    novelty = (
        cosine_weight * _zscore(cosine_distance)
        + flux_weight * _zscore(spectral_flux)
        + onset_weight * _zscore(onset_strength)
        + rms_weight * _zscore(rms_delta)
    )
    novelty = gaussian_filter1d(novelty.astype(np.float32), sigma=max(config.smooth_sigma, 0.1))

    display_freqs, display_spectrum = _aggregate_frequency_bins(
        freqs,
        magnitude,
        group_count=min(256, magnitude.shape[0]),
    )
    display_times, display_spectrum = _aggregate_time_bins(
        times,
        display_spectrum,
        group_count=min(2400, magnitude.shape[1]),
    )
    display_db = 20.0 * np.log10(np.maximum(display_spectrum, 1e-8))
    display_db -= float(np.max(display_db))

    return {
        "times": times.astype(np.float32),
        "novelty": novelty.astype(np.float32),
        "rms": rms.astype(np.float32),
        "spectral_centroid_hz": centroid.astype(np.float32),
        "bandwidth_hz": bandwidth.astype(np.float32),
        "rolloff_hz": rolloff.astype(np.float32),
        "flatness": flatness.astype(np.float32),
        "spectral_flux": spectral_flux.astype(np.float32),
        "onset_strength": onset_strength.astype(np.float32),
        "display_spectrogram_db": display_db.astype(np.float32),
        "display_times": display_times.astype(np.float32),
        "display_freqs": display_freqs.astype(np.float32),
        "n_fft": int(n_fft),
        "hop_length": int(hop),
    }


def _detect_peaks(
    novelty: np.ndarray,
    times: np.ndarray,
    config: AnalysisConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    if novelty.size == 0:
        empty = np.array([], dtype=np.int32)
        return empty, {"prominences": np.array([], dtype=np.float32)}, np.array([], dtype=np.float32)

    if times.size > 1:
        frame_duration = float(np.median(np.diff(times)))
    else:
        frame_duration = 0.05

    baseline = gaussian_filter1d(novelty, sigma=max(config.smooth_sigma * 4.0, 1.0))
    residual = novelty - baseline
    residual_std = float(np.std(residual))
    robust_scale = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    scale = max(residual_std, robust_scale, 0.05)

    threshold = baseline + config.threshold_sigma * scale
    min_distance_frames = max(1, int(config.min_event_distance_sec / max(frame_duration, 1e-4)))
    prominence = max(config.prominence_sigma * scale, 0.05)

    peaks, properties = find_peaks(
        novelty,
        height=threshold,
        distance=min_distance_frames,
        prominence=prominence,
    )

    if peaks.size == 0:
        fallback_height = float(np.quantile(novelty, 0.92))
        peaks, properties = find_peaks(
            novelty,
            height=fallback_height,
            distance=min_distance_frames,
        )
        if "prominences" not in properties:
            properties["prominences"] = np.full(peaks.shape, np.nan, dtype=np.float32)

    return peaks.astype(np.int32), properties, threshold.astype(np.float32)


def _segment_features(samples: np.ndarray, sr: int, config: AnalysisConfig) -> dict[str, float | str]:
    if samples.size == 0:
        samples = np.zeros(max(512, config.n_fft), dtype=np.float32)

    freqs, _, magnitude, _, _ = _compute_spectrogram(samples, sr, config)
    mean_spectrum = np.mean(np.maximum(magnitude, 1e-10), axis=1)
    total_energy = float(np.sum(mean_spectrum))

    low_mask = freqs < 250.0
    mid_mask = (freqs >= 250.0) & (freqs < 2000.0)
    high_mask = freqs >= 2000.0

    centroid = float(np.sum(freqs * mean_spectrum) / total_energy)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_spectrum) / total_energy))
    cumulative_energy = np.cumsum(mean_spectrum) / total_energy
    rolloff_index = int(np.searchsorted(cumulative_energy, 0.85, side="left"))
    rolloff_index = min(rolloff_index, freqs.shape[0] - 1)
    rolloff = float(freqs[rolloff_index])
    flatness = float(np.exp(np.mean(np.log(mean_spectrum))) / np.mean(mean_spectrum))

    dominant_indices = np.argsort(mean_spectrum)[-3:][::-1]
    dominant_freqs = ", ".join(f"{freqs[index]:.0f}Hz" for index in dominant_indices)

    return {
        "rms": _safe_rms(samples),
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "rolloff_hz": rolloff,
        "flatness": flatness,
        "low_ratio": float(np.sum(mean_spectrum[low_mask]) / total_energy),
        "mid_ratio": float(np.sum(mean_spectrum[mid_mask]) / total_energy),
        "high_ratio": float(np.sum(mean_spectrum[high_mask]) / total_energy),
        "dominant_freqs": dominant_freqs,
    }


def _channel_bias(audio: np.ndarray, sr: int, center_sec: float, window_sec: float) -> str:
    if audio.shape[0] < 2:
        return "单声道"

    half_window = max(0.5, window_sec) / 2.0
    start = max(0, int((center_sec - half_window) * sr))
    end = min(audio.shape[1], int((center_sec + half_window) * sr))
    if end <= start:
        return "平衡"

    left_rms = _safe_rms(audio[0, start:end])
    right_rms = _safe_rms(audio[1, start:end])

    if left_rms > right_rms * 1.15:
        return "左侧偏强"
    if right_rms > left_rms * 1.15:
        return "右侧偏强"
    return "平衡"


def _event_candidate_labels(
    before: dict[str, float | str],
    after: dict[str, float | str],
    is_major_boundary: bool,
) -> list[str]:
    labels = ["新事件出现"]

    before_centroid = max(float(before["centroid_hz"]), 1.0)
    after_centroid = float(after["centroid_hz"])
    before_bandwidth = max(float(before["bandwidth_hz"]), 1.0)
    after_bandwidth = float(after["bandwidth_hz"])

    centroid_ratio = after_centroid / before_centroid
    bandwidth_ratio = after_bandwidth / before_bandwidth
    low_delta = float(after["low_ratio"]) - float(before["low_ratio"])
    high_delta = float(after["high_ratio"]) - float(before["high_ratio"])
    flatness_delta = float(after["flatness"]) - float(before["flatness"])
    rms_ratio = float(after["rms"]) / max(float(before["rms"]), 1e-8)
    rolloff_ratio = float(after["rolloff_hz"]) / max(float(before["rolloff_hz"]), 1.0)

    if centroid_ratio > 1.18 or high_delta > 0.08 or rolloff_ratio > 1.12:
        labels.append("高频扩展")

    if flatness_delta > 0.05 and high_delta > 0.04:
        labels.append("噪声侵入")

    if rms_ratio > 1.28:
        labels.append("能量增强")
    elif rms_ratio < 0.78:
        labels.append("能量减弱")

    if (
        abs(centroid_ratio - 1.0) > 0.15
        or abs(bandwidth_ratio - 1.0) > 0.16
        or abs(flatness_delta) > 0.05
    ):
        labels.append("音色突变")

    if (bandwidth_ratio > 1.15 and rms_ratio > 1.08) or (high_delta > 0.05 and low_delta > -0.02):
        labels.append("材料聚集")

    if rms_ratio < 0.86 and bandwidth_ratio < 0.92 and high_delta < -0.04:
        labels.append("消散")

    if is_major_boundary:
        labels.append("段落转换")

    return _unique_labels(labels)


def _section_candidate_labels(
    start_features: dict[str, float | str],
    end_features: dict[str, float | str],
    mean_features: dict[str, float | str],
    event_count: int,
) -> list[str]:
    labels: list[str] = []
    start_rms = max(float(start_features["rms"]), 1e-8)
    end_rms = float(end_features["rms"])
    rms_ratio = end_rms / start_rms
    flatness_delta = float(end_features["flatness"]) - float(start_features["flatness"])
    high_delta = float(end_features["high_ratio"]) - float(start_features["high_ratio"])
    centroid_ratio = float(end_features["centroid_hz"]) / max(float(start_features["centroid_hz"]), 1.0)

    if event_count <= 1 and 0.88 <= rms_ratio <= 1.14 and abs(centroid_ratio - 1.0) <= 0.12:
        labels.append("稳态持续")

    if event_count >= 3 or (rms_ratio > 1.12 and float(mean_features["bandwidth_hz"]) > float(start_features["bandwidth_hz"]) * 1.08):
        labels.append("材料聚集")

    if rms_ratio < 0.84:
        labels.append("消散")
        labels.append("能量减弱")
    elif rms_ratio > 1.18:
        labels.append("能量增强")

    if centroid_ratio > 1.15 or high_delta > 0.07:
        labels.append("高频扩展")

    if flatness_delta > 0.05 and high_delta > 0.04:
        labels.append("噪声侵入")

    if not labels:
        labels.append("稳态持续" if event_count <= 1 else "材料聚集")

    return _unique_labels(labels)


def _event_summary(labels: list[str], channel_bias: str) -> str:
    summary_bits = labels[:4]
    if channel_bias not in {"平衡", "单声道"}:
        summary_bits.append(f"{channel_bias}声部更突出")
    return "；".join(summary_bits)


def _build_event_table(
    audio: np.ndarray,
    analysis_samples: np.ndarray,
    sr: int,
    peak_indices: np.ndarray,
    peak_properties: dict[str, np.ndarray],
    novelty: np.ndarray,
    times: np.ndarray,
    duration_sec: float,
    config: AnalysisConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    peak_times = times[peak_indices] if peak_indices.size else np.array([], dtype=np.float32)
    prominences = peak_properties.get("prominences", np.full(peak_indices.shape, np.nan))

    for index, peak_time in enumerate(peak_times):
        context = max(1.0, config.context_window_sec)
        before_start = max(0.0, float(peak_time - context))
        before_end = max(before_start + 0.25, float(peak_time))
        after_start = min(duration_sec, float(peak_time))
        after_end = min(duration_sec, float(peak_time + context))

        before_slice = analysis_samples[int(before_start * sr) : int(before_end * sr)]
        after_slice = analysis_samples[int(after_start * sr) : int(after_end * sr)]
        before_features = _segment_features(before_slice, sr, config)
        after_features = _segment_features(after_slice, sr, config)
        bias = _channel_bias(audio, sr, float(peak_time), context)

        rows.append(
            {
                "event_id": index + 1,
                "time_sec": round(float(peak_time), 3),
                "time_label": format_seconds(float(peak_time)),
                "strength": round(float(novelty[peak_indices[index]]), 4),
                "prominence": round(float(prominences[index]), 4) if index < len(prominences) else math.nan,
                "channel_bias": bias,
                "pre_rms": round(float(before_features["rms"]), 6),
                "post_rms": round(float(after_features["rms"]), 6),
                "post_centroid_hz": round(float(after_features["centroid_hz"]), 1),
                "post_bandwidth_hz": round(float(after_features["bandwidth_hz"]), 1),
                "post_rolloff_hz": round(float(after_features["rolloff_hz"]), 1),
                "post_low_ratio": round(float(after_features["low_ratio"]), 4),
                "post_mid_ratio": round(float(after_features["mid_ratio"]), 4),
                "post_high_ratio": round(float(after_features["high_ratio"]), 4),
                "post_flatness": round(float(after_features["flatness"]), 4),
                "dominant_freqs": str(after_features["dominant_freqs"]),
                "_before_features": before_features,
                "_after_features": after_features,
            }
        )

    event_table = pd.DataFrame(rows)
    if event_table.empty:
        return event_table

    major_count = min(max(1, int(round(math.sqrt(len(event_table))))), min(6, len(event_table)))
    major_indices = (
        event_table.nlargest(major_count, columns=["prominence", "strength"])
        .sort_values("time_sec")
        .index
    )
    event_table["is_major_boundary"] = False
    event_table.loc[major_indices, "is_major_boundary"] = True

    auto_label_texts: list[str] = []
    summaries: list[str] = []
    for _, row in event_table.iterrows():
        labels = _event_candidate_labels(
            before=row["_before_features"],
            after=row["_after_features"],
            is_major_boundary=bool(row["is_major_boundary"]),
        )
        auto_label_texts.append(join_labels(labels))
        summaries.append(_event_summary(labels, str(row["channel_bias"])))

    event_table["auto_labels"] = auto_label_texts
    event_table["descriptor"] = summaries
    return event_table.drop(columns=["_before_features", "_after_features"])


def _build_section_table(
    analysis_samples: np.ndarray,
    sr: int,
    duration_sec: float,
    event_table: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if event_table.empty:
        boundaries = [0.0, duration_sec]
    else:
        major_boundaries = event_table.loc[event_table["is_major_boundary"], "time_sec"].tolist()
        boundaries = [0.0, *major_boundaries, duration_sec]

    cleaned_boundaries = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary - cleaned_boundaries[-1] >= 0.5:
            cleaned_boundaries.append(boundary)
    if cleaned_boundaries[-1] != duration_sec:
        cleaned_boundaries.append(duration_sec)

    rows: list[dict[str, float | int | str]] = []
    event_times = event_table["time_sec"].to_numpy(dtype=float) if not event_table.empty else np.array([], dtype=float)
    for index in range(len(cleaned_boundaries) - 1):
        start = float(cleaned_boundaries[index])
        end = float(cleaned_boundaries[index + 1])
        segment = analysis_samples[int(start * sr) : int(end * sr)]
        duration = end - start
        probe_duration = max(0.6, min(duration * 0.3, 2.5))
        start_segment = analysis_samples[int(start * sr) : int(min(end, start + probe_duration) * sr)]
        end_segment = analysis_samples[int(max(start, end - probe_duration) * sr) : int(end * sr)]

        mean_features = _segment_features(segment, sr, config)
        start_features = _segment_features(start_segment, sr, config)
        end_features = _segment_features(end_segment, sr, config)
        event_count = int(np.sum((event_times >= start) & (event_times < end)))
        labels = _section_candidate_labels(start_features, end_features, mean_features, event_count)

        rows.append(
            {
                "section_id": index + 1,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "start_label": format_seconds(start),
                "end_label": format_seconds(end),
                "duration_sec": round(end - start, 3),
                "event_count": event_count,
                "auto_labels": join_labels(labels),
                "descriptor": "；".join(labels[:4]),
                "centroid_hz": round(float(mean_features["centroid_hz"]), 1),
                "low_ratio": round(float(mean_features["low_ratio"]), 4),
                "mid_ratio": round(float(mean_features["mid_ratio"]), 4),
                "high_ratio": round(float(mean_features["high_ratio"]), 4),
                "flatness": round(float(mean_features["flatness"]), 4),
                "dominant_freqs": str(mean_features["dominant_freqs"]),
            }
        )

    return pd.DataFrame(rows)


def analyze_audio(
    source: str | Path | BinaryIO,
    config: AnalysisConfig | None = None,
    channel_mode: ChannelMode = "mix",
) -> dict[str, object]:
    analysis_config = config or AnalysisConfig()
    audio, sr = _load_audio(source, analysis_config.target_sr)
    selected = _select_channel(audio, channel_mode)
    duration_sec = float(selected.shape[0] / sr)

    novelty_bundle = _build_novelty_curve(selected, sr, analysis_config)
    peak_indices, peak_properties, threshold = _detect_peaks(
        novelty_bundle["novelty"],
        novelty_bundle["times"],
        analysis_config,
    )

    feature_table = _build_feature_table(
        times=novelty_bundle["times"],
        rms=novelty_bundle["rms"],
        centroid=novelty_bundle["spectral_centroid_hz"],
        bandwidth=novelty_bundle["bandwidth_hz"],
        rolloff=novelty_bundle["rolloff_hz"],
        flatness=novelty_bundle["flatness"],
        spectral_flux=novelty_bundle["spectral_flux"],
        onset_strength=novelty_bundle["onset_strength"],
        novelty=novelty_bundle["novelty"],
        threshold=threshold,
    )

    event_table = _build_event_table(
        audio=audio,
        analysis_samples=selected,
        sr=sr,
        peak_indices=peak_indices,
        peak_properties=peak_properties,
        novelty=novelty_bundle["novelty"],
        times=novelty_bundle["times"],
        duration_sec=duration_sec,
        config=analysis_config,
    )
    section_table = _build_section_table(
        analysis_samples=selected,
        sr=sr,
        duration_sec=duration_sec,
        event_table=event_table,
        config=analysis_config,
    )

    summary_lines: list[str] = []
    if not event_table.empty:
        summary_lines.append(
            f"共检测到 {len(event_table)} 个疑似频谱新事件，其中 {int(event_table['is_major_boundary'].sum())} 个被标记为强边界。"
        )
        for row in event_table.loc[event_table["is_major_boundary"]].itertuples():
            summary_lines.append(f"{row.time_label}: {row.descriptor}")
    else:
        summary_lines.append("未检测到显著的新频谱事件，建议降低阈值或缩短最小事件间隔后再试。")

    return {
        "config": analysis_config.to_dict(),
        "audio": audio,
        "selected_audio": selected,
        "sr": sr,
        "duration_sec": duration_sec,
        "channel_mode": channel_mode,
        "times": novelty_bundle["times"],
        "spectrogram_db": novelty_bundle["display_spectrogram_db"],
        "spectrogram_times": novelty_bundle["display_times"],
        "spectrogram_freqs": novelty_bundle["display_freqs"],
        "feature_table": feature_table,
        "novelty": novelty_bundle["novelty"],
        "threshold": threshold,
        "peak_indices": peak_indices,
        "peak_times": novelty_bundle["times"][peak_indices] if peak_indices.size else np.array([], dtype=np.float32),
        "hop_length": novelty_bundle["hop_length"],
        "event_table": event_table,
        "section_table": section_table,
        "summary_lines": summary_lines,
        "event_label_vocab": EVENT_LABEL_VOCAB,
        "section_label_vocab": SECTION_LABEL_VOCAB,
        "feature_column_labels": FEATURE_COLUMN_LABELS,
    }
