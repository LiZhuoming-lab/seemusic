from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import soundfile as sf

from spectral_tool.assistant import (
    annotate_event_interaction_levels,
    annotate_event_similarity_groups,
    build_assistant_overlay_component,
    build_event_assistant_payload,
)
from spectral_tool.analysis import (
    CANDIDATE_BOUNDARY_LABEL,
    AnalysisConfig,
    _boundary_state_change_score,
    _load_audio,
    _resolve_novelty_weights,
    analyze_audio,
    build_audio_excerpt_wav,
)
from spectral_tool.visualization import (
    build_local_waveform_chart,
    build_synced_overview_player_html,
    build_synced_waveform_player_html,
    plot_local_spectrogram,
    plot_local_waveform,
)


class AnalysisTestCase(unittest.TestCase):
    def test_detects_major_spectral_boundaries(self) -> None:
        rng = np.random.default_rng(7)
        sr = 22050
        segment_duration = 2.0
        sample_count = int(sr * segment_duration)
        time_axis = np.linspace(0.0, segment_duration, sample_count, endpoint=False)

        segment_1 = 0.03 * rng.normal(size=sample_count)
        segment_2 = 0.18 * np.sin(2 * np.pi * 220 * time_axis)
        segment_3 = 0.05 * rng.normal(size=sample_count) + 0.08 * np.sin(2 * np.pi * 1600 * time_axis)
        segment_4 = 0.14 * np.sin(2 * np.pi * 60 * time_axis) + 0.04 * rng.normal(size=sample_count)

        audio = np.concatenate([segment_1, segment_2, segment_3, segment_4]).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "synthetic.wav"
            sf.write(path, audio, sr)

            result = analyze_audio(
                path,
                config=AnalysisConfig(
                    target_sr=22050,
                    n_fft=2048,
                    hop_length=512,
                    smooth_sigma=1.0,
                    threshold_sigma=0.5,
                    prominence_sigma=0.4,
                    min_event_distance_sec=1.0,
                    context_window_sec=1.0,
                ),
            )

        event_times = result["event_table"]["time_sec"].to_numpy()
        self.assertGreaterEqual(len(event_times), 3)
        self.assertTrue(np.any(np.abs(event_times - 2.0) < 0.6))
        self.assertTrue(np.any(np.abs(event_times - 4.0) < 0.6))
        self.assertTrue(np.any(np.abs(event_times - 6.0) < 0.6))
        self.assertIn("feature_table", result)
        self.assertTrue(
            {
                "rms",
                "spectral_centroid_hz",
                "band_energy_ratio",
                "high_band_ratio",
                "rolloff_hz",
                "flatness",
                "spectral_flux",
                "onset_strength",
                "novelty",
            }.issubset(set(result["feature_table"].columns))
        )
        self.assertIn("auto_labels", result["event_table"].columns)

    def test_mp3_falls_back_to_audioread(self) -> None:
        class NamedBytesIO(io.BytesIO):
            name = "demo.mp3"

        class FakeAudioHandle:
            channels = 2
            samplerate = 44100

            def __enter__(self) -> "FakeAudioHandle":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def __iter__(self):
                pcm = np.array([0, 1000, -1000, 0, 500, -500], dtype="<i2")
                yield pcm.tobytes()

        source = NamedBytesIO(b"fake-mp3-data")

        with mock.patch("spectral_tool.analysis.sf.read", side_effect=RuntimeError("unsupported")):
            with mock.patch("spectral_tool.analysis.audioread.audio_open", return_value=FakeAudioHandle()):
                audio, sr = _load_audio(source, target_sr=None)

        self.assertEqual(sr, 44100)
        self.assertEqual(audio.shape[0], 2)
        self.assertEqual(audio.shape[1], 3)
        self.assertTrue(np.all(np.abs(audio) <= 1.0))

    def test_m4a_falls_back_to_audioread(self) -> None:
        class NamedBytesIO(io.BytesIO):
            name = "demo.m4a"

        class FakeAudioHandle:
            channels = 2
            samplerate = 48000

            def __enter__(self) -> "FakeAudioHandle":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def __iter__(self):
                pcm = np.array([0, 1200, -1200, 0, 600, -600], dtype="<i2")
                yield pcm.tobytes()

        source = NamedBytesIO(b"fake-m4a-data")

        with mock.patch("spectral_tool.analysis.sf.read", side_effect=RuntimeError("unsupported")):
            with mock.patch("spectral_tool.analysis.audioread.audio_open", return_value=FakeAudioHandle()):
                audio, sr = _load_audio(source, target_sr=None)

        self.assertEqual(sr, 48000)
        self.assertEqual(audio.shape[0], 2)
        self.assertEqual(audio.shape[1], 3)
        self.assertTrue(np.all(np.abs(audio) <= 1.0))

    def test_resolve_novelty_weights_uses_custom_preset_mix(self) -> None:
        config = AnalysisConfig(
            novelty_weight_cosine=0.45,
            novelty_weight_flux=0.30,
            novelty_weight_onset=0.15,
            novelty_weight_rms=0.10,
            event_model_preset="timbre_soundscape",
        )

        weights = _resolve_novelty_weights(config)

        self.assertEqual(weights, (0.45, 0.30, 0.15, 0.10))

    def test_timbre_soundscape_mode_uses_extra_timbre_features(self) -> None:
        sr = 22050
        segment_duration = 2.0
        sample_count = int(sr * segment_duration)
        time_axis = np.linspace(0.0, segment_duration, sample_count, endpoint=False)
        rng = np.random.default_rng(19)

        low_tone = 0.14 * np.sin(2 * np.pi * 180 * time_axis)
        bright_noise = 0.05 * rng.normal(size=sample_count) + 0.08 * np.sin(2 * np.pi * 3400 * time_axis)
        audio = np.concatenate([low_tone, bright_noise]).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "timbre-mode.wav"
            sf.write(path, audio, sr)

            common_kwargs = dict(
                target_sr=22050,
                n_fft=2048,
                hop_length=512,
                smooth_sigma=1.0,
                threshold_sigma=0.8,
                prominence_sigma=0.5,
                min_event_distance_sec=0.8,
                context_window_sec=1.0,
                novelty_weight_cosine=0.45,
                novelty_weight_flux=0.30,
                novelty_weight_onset=0.15,
                novelty_weight_rms=0.10,
            )

            balanced = analyze_audio(
                path,
                config=AnalysisConfig(
                    event_model_preset="balanced_default",
                    **common_kwargs,
                ),
            )
            timbre = analyze_audio(
                path,
                config=AnalysisConfig(
                    event_model_preset="timbre_soundscape",
                    **common_kwargs,
                ),
            )

        self.assertIn("band_energy_ratio", timbre["feature_table"].columns)
        self.assertIn("high_band_ratio", timbre["feature_table"].columns)
        self.assertFalse(np.allclose(balanced["novelty"], timbre["novelty"]))

    def test_build_audio_excerpt_wav_returns_playable_bytes(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
        left = 0.2 * np.sin(2 * np.pi * 220 * time_axis)
        right = 0.2 * np.sin(2 * np.pi * 440 * time_axis)
        stereo = np.vstack([left, right]).astype(np.float32)

        clip = build_audio_excerpt_wav(stereo, sr, 0.5, 1.0, channel_mode="mix")
        self.assertGreater(len(clip), 100)

        decoded, decoded_sr = sf.read(io.BytesIO(clip), always_2d=True, dtype="float32")
        self.assertEqual(decoded_sr, sr)
        self.assertGreater(decoded.shape[0], 0)

    def test_plot_local_spectrogram_returns_figure(self) -> None:
        spectrogram_db = np.random.default_rng(1).normal(size=(32, 60)).astype(np.float32)
        times = np.linspace(0.0, 12.0, 60, dtype=np.float32)
        freqs = np.linspace(0.0, 12000.0, 32, dtype=np.float32)

        figure = plot_local_spectrogram(
            spectrogram_db=spectrogram_db,
            times=times,
            freqs=freqs,
            center_time=6.0,
            window_radius_sec=2.0,
            highlight_time=6.0,
            max_freq_hz=8000.0,
        )

        self.assertGreaterEqual(len(figure.axes), 1)

    def test_plot_local_waveform_returns_figure(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 6.0, sr * 6, endpoint=False)
        samples = (0.15 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)

        figure = plot_local_waveform(
            samples=samples,
            sr=sr,
            center_time=3.0,
            window_radius_sec=1.5,
            highlight_time=3.0,
        )

        self.assertGreaterEqual(len(figure.axes), 1)

    def test_build_local_waveform_chart_returns_plotly_figure(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 4.0, sr * 4, endpoint=False)
        samples = (0.12 * np.sin(2 * np.pi * 330 * time_axis)).astype(np.float32)

        figure = build_local_waveform_chart(
            samples=samples,
            sr=sr,
            center_time=2.0,
            window_radius_sec=1.0,
            highlight_time=2.0,
        )

        self.assertGreaterEqual(len(figure.data), 1)

    def test_build_synced_waveform_player_html_contains_audio_and_playhead_logic(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 3.0, sr * 3, endpoint=False)
        samples = (0.08 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)
        audio_bytes = build_audio_excerpt_wav(samples, sr, 0.0, 2.0, channel_mode="mix")

        html = build_synced_waveform_player_html(
            audio_bytes=audio_bytes,
            samples=samples,
            sr=sr,
            clip_start_sec=0.5,
            clip_end_sec=2.5,
            event_time_sec=1.5,
        )

        self.assertIn("data:audio/wav;base64,", html)
        self.assertIn("audio.currentTime", html)
        self.assertIn("点击波形可跳转", html)

    def test_build_synced_overview_player_html_contains_dual_canvas_and_controls(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 3.0, sr * 3, endpoint=False)
        samples = (0.08 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)
        audio_bytes = build_audio_excerpt_wav(samples, sr, 0.0, 2.0, channel_mode="mix")
        spectrogram_db = np.random.default_rng(3).normal(size=(48, 120)).astype(np.float32)
        spec_times = np.linspace(0.0, 2.0, 120, dtype=np.float32)
        spec_freqs = np.linspace(0.0, 26000.0, 48, dtype=np.float32)

        html = build_synced_overview_player_html(
            audio_bytes=audio_bytes,
            audio_mime="audio/wav",
            samples=samples,
            sr=sr,
            spectrogram_db=spectrogram_db,
            spectrogram_times=spec_times,
            spectrogram_freqs=spec_freqs,
            duration_sec=3.0,
        )

        self.assertIn("overviewWaveCanvas", html)
        self.assertIn("overviewSpecCanvas", html)
        self.assertIn("频率上限", html)
        self.assertIn("聚焦窗口", html)
        self.assertIn("overviewViewModeToggle", html)
        self.assertIn("切换到局部聚焦", html)
        self.assertIn("当前聚焦区间", html)
        self.assertIn("overviewWaveControl", html)
        self.assertIn("overviewFreqControl", html)
        self.assertIn("slider-vertical", html)
        self.assertIn("overviewWaveAmplitudeMax", html)
        self.assertIn("波形纵轴上限", html)
        self.assertIn('max="26000"', html)
        self.assertIn("频率 (Hz)", html)
        self.assertIn("振幅", html)
        self.assertIn("悬停波形", html)
        self.assertIn("包络振幅", html)
        self.assertIn("悬停频谱", html)
        self.assertIn("updateWaveHover", html)
        self.assertIn("updateSpecHover", html)
        self.assertIn("zoomAroundPointer", html)
        self.assertIn("toggleOverviewMode", html)
        self.assertIn("pointerdown", html)
        self.assertIn("dblclick", html)

    def test_build_event_assistant_payload_for_major_boundary(self) -> None:
        payload = build_event_assistant_payload(
            {
                "time_label": "03:24.10",
                "strength": 1.26,
                "prominence": 0.74,
                "pre_rms": 0.12,
                "post_rms": 0.23,
                "post_centroid_hz": 3620.0,
                "post_rolloff_hz": 8610.0,
                "post_high_ratio": 0.34,
                "post_flatness": 0.27,
                "channel_bias": "左侧偏强",
                "dominant_freqs": "312Hz, 1288Hz, 3810Hz",
                "is_major_boundary": True,
                "interaction_priority": "strong",
                "auto_labels": f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
            },
            effective_label_text=f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
        )

        self.assertIn("像不像边界", payload["question"])
        self.assertEqual(payload["role_text"], CANDIDATE_BOUNDARY_LABEL)
        self.assertIn("候选", payload["draft"])
        self.assertEqual(len(payload["primary_actions"]), 3)
        self.assertIn("看它是不是边界", [item["label"] for item in payload["primary_actions"]])
        self.assertIn("看这个变化有没有持续下去", payload["suggested_checks"])
        self.assertEqual(payload["interaction_priority"], "strong")
        self.assertEqual(payload["interaction_depth"], "full")

    def test_build_event_assistant_payload_includes_time_and_labels(self) -> None:
        payload = build_event_assistant_payload(
            {
                "time_label": "01:07.30",
                "strength": 0.85,
                "prominence": 0.44,
                "pre_rms": 0.08,
                "post_rms": 0.11,
                "post_centroid_hz": 2140.0,
                "post_rolloff_hz": 5520.0,
                "post_high_ratio": 0.18,
                "post_flatness": 0.11,
                "channel_bias": "平衡",
                "dominant_freqs": "440Hz, 1760Hz",
                "is_major_boundary": False,
                "interaction_priority": "weak",
                "auto_labels": "新事件出现 | 音色突变",
            },
            effective_label_text="新事件出现 | 音色突变",
        )

        self.assertIn("01:07.30", payload["draft"])
        self.assertIn("新事件出现", payload["label_text"])
        self.assertGreaterEqual(len(payload["evidence_points"]), 4)
        self.assertEqual(payload["assistant_invite"], "AI 小助手")
        self.assertEqual(payload["assistant_invite_hint"], "先看一眼就好")
        self.assertIn("先给我一个轻量观察", [item["label"] for item in payload["primary_actions"]])
        self.assertIn("AI 设置", payload["settings_title"])
        self.assertEqual(payload["interaction_depth"], "light")

    def test_annotate_event_interaction_levels_adds_three_priority_levels(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "strength": 0.22,
                    "prominence": 0.08,
                    "is_major_boundary": False,
                    "auto_labels": "新事件出现",
                    "effective_labels": "新事件出现",
                },
                {
                    "event_id": 2,
                    "strength": 0.61,
                    "prominence": 0.34,
                    "is_major_boundary": False,
                    "auto_labels": "新事件出现 | 音色突变",
                    "effective_labels": "新事件出现 | 音色突变",
                },
                {
                    "event_id": 3,
                    "strength": 1.31,
                    "prominence": 0.86,
                    "is_major_boundary": True,
                    "auto_labels": f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
                    "effective_labels": f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
                },
            ]
        )

        annotated = annotate_event_interaction_levels(frame)

        self.assertIn("interaction_priority", annotated.columns)
        self.assertIn("interaction_priority_label", annotated.columns)
        self.assertIn("interaction_depth", annotated.columns)
        self.assertIn("weak", set(annotated["interaction_priority"]))
        self.assertIn("medium", set(annotated["interaction_priority"]))
        self.assertIn("strong", set(annotated["interaction_priority"]))
        self.assertEqual(
            annotated.loc[annotated["event_id"] == 3, "interaction_priority"].iloc[0],
            "strong",
        )

    def test_annotate_event_similarity_groups_clusters_highly_similar_events(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "post_rms": 0.121,
                    "post_centroid_hz": 2210.0,
                    "post_bandwidth_hz": 1460.0,
                    "post_rolloff_hz": 5280.0,
                    "post_low_ratio": 0.20,
                    "post_mid_ratio": 0.49,
                    "post_high_ratio": 0.31,
                    "post_flatness": 0.14,
                },
                {
                    "event_id": 2,
                    "post_rms": 0.122,
                    "post_centroid_hz": 2225.0,
                    "post_bandwidth_hz": 1455.0,
                    "post_rolloff_hz": 5300.0,
                    "post_low_ratio": 0.201,
                    "post_mid_ratio": 0.488,
                    "post_high_ratio": 0.311,
                    "post_flatness": 0.141,
                },
                {
                    "event_id": 3,
                    "post_rms": 0.042,
                    "post_centroid_hz": 620.0,
                    "post_bandwidth_hz": 380.0,
                    "post_rolloff_hz": 1300.0,
                    "post_low_ratio": 0.72,
                    "post_mid_ratio": 0.23,
                    "post_high_ratio": 0.05,
                    "post_flatness": 0.03,
                },
            ]
        )

        annotated = annotate_event_similarity_groups(frame, threshold=0.90)

        group_1 = int(annotated.loc[annotated["event_id"] == 1, "similarity_group_id"].iloc[0])
        group_2 = int(annotated.loc[annotated["event_id"] == 2, "similarity_group_id"].iloc[0])
        group_3 = int(annotated.loc[annotated["event_id"] == 3, "similarity_group_id"].iloc[0])

        self.assertEqual(group_1, group_2)
        self.assertNotEqual(group_1, group_3)
        self.assertEqual(int(annotated.loc[annotated["event_id"] == 1, "similarity_group_size"].iloc[0]), 2)
        self.assertGreaterEqual(float(annotated.loc[annotated["event_id"] == 1, "max_similarity_in_group"].iloc[0]), 0.90)

    def test_build_assistant_overlay_component_targets_parent_document(self) -> None:
        payload = build_event_assistant_payload(
            {
                "time_label": "03:24.10",
                "strength": 1.26,
                "prominence": 0.74,
                "pre_rms": 0.12,
                "post_rms": 0.23,
                "post_centroid_hz": 3620.0,
                "post_rolloff_hz": 8610.0,
                "post_high_ratio": 0.34,
                "post_flatness": 0.27,
                "channel_bias": "左侧偏强",
                "dominant_freqs": "312Hz, 1288Hz, 3810Hz",
                "is_major_boundary": True,
                "auto_labels": f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
                "descriptor": f"高频扩展；{CANDIDATE_BOUNDARY_LABEL}",
            },
            effective_label_text=f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
        )

        html = build_assistant_overlay_component(3, payload)

        self.assertIn("window.parent.document", html)
        self.assertIn("event-ai-overlay-root", html)
        self.assertIn("AI 小助手", html)
        self.assertIn("建议重点讨论", html)
        self.assertIn("先给我一个初步观察", html)
        self.assertIn("看前后有没有变", html)
        self.assertIn("看它是不是边界", html)
        self.assertIn("AI 设置", html)
        self.assertIn("展开更详细的分析", html)
        self.assertIn("Collaborative Analysis", html)
        self.assertIn("重置位置", html)
        self.assertIn("拖动可移动", html)
        self.assertIn(CANDIDATE_BOUNDARY_LABEL, html)

    def test_boundary_state_change_score_rejects_pure_local_spike(self) -> None:
        before = {
            "rms": 0.12,
            "centroid_hz": 1800.0,
            "bandwidth_hz": 1200.0,
            "rolloff_hz": 5200.0,
            "flatness": 0.10,
            "low_ratio": 0.22,
            "mid_ratio": 0.53,
            "high_ratio": 0.25,
        }
        after = {
            "rms": 0.17,
            "centroid_hz": 1830.0,
            "bandwidth_hz": 1215.0,
            "rolloff_hz": 5260.0,
            "flatness": 0.11,
            "low_ratio": 0.21,
            "mid_ratio": 0.52,
            "high_ratio": 0.27,
        }

        score, is_clear_change = _boundary_state_change_score(before, after)

        self.assertLess(score, 1.05)
        self.assertFalse(is_clear_change)

    def test_boundary_state_change_score_accepts_clear_state_shift(self) -> None:
        before = {
            "rms": 0.09,
            "centroid_hz": 920.0,
            "bandwidth_hz": 640.0,
            "rolloff_hz": 2100.0,
            "flatness": 0.05,
            "low_ratio": 0.62,
            "mid_ratio": 0.30,
            "high_ratio": 0.08,
        }
        after = {
            "rms": 0.18,
            "centroid_hz": 2700.0,
            "bandwidth_hz": 1560.0,
            "rolloff_hz": 6900.0,
            "flatness": 0.18,
            "low_ratio": 0.21,
            "mid_ratio": 0.43,
            "high_ratio": 0.36,
        }

        score, is_clear_change = _boundary_state_change_score(before, after)

        self.assertGreaterEqual(score, 1.05)
        self.assertTrue(is_clear_change)


if __name__ == "__main__":
    unittest.main()
