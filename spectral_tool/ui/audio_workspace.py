from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from spectral_tool.analysis import AnalysisConfig, analyze_audio
from spectral_tool.models.presets import EVENT_MODEL_PRESETS, novelty_explanation
from spectral_tool.state.audio_state import (
    build_analysis_signature,
    filter_event_annotations,
    init_event_annotations,
    init_section_annotations,
)
from spectral_tool.ui.event_editor import render_event_editor, render_feature_chart
from spectral_tool.ui.section_editor import render_section_editor
from spectral_tool.visualization import (
    build_summary_text,
    build_synced_overview_player_html,
    plot_event_density,
    plot_novelty,
    plot_spectrogram,
    plot_waveform,
)


def render_audio_workspace(sidebar_values: dict[str, float | int | str | None]) -> None:
    st.markdown("### 音频 / 频谱事件分析工作台")

    uploaded_file = st.file_uploader(
        "上传音频文件（支持 WAV、FLAC、AIFF、MP3、M4A）",
        type=["wav", "flac", "aif", "aiff", "ogg", "mp3", "m4a"],
    )

    st.markdown(
        """
这个版本已经贴近本地 MVP 的完整工作流：

- 自动提取 `RMS / centroid / band energy ratio / high-frequency ratio / rolloff / flatness / flux / onset strength / novelty`
- 自动检测变化点、候选边界和新事件位置
- 点击自动标记点可显示精确时间
- 支持局部试听与人工改标签
- 支持导出编辑后的 `CSV / JSON`
"""
    )

    if uploaded_file is None:
        st.info("上传一段音频后，系统会自动生成频谱图、特征曲线、事件列表和可编辑标注。")
        st.stop()

    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes)

    config = AnalysisConfig(
        target_sr=sidebar_values["target_sr"],
        n_fft=sidebar_values["n_fft"],
        hop_length=sidebar_values["hop_length"],
        smooth_sigma=sidebar_values["smooth_sigma"],
        threshold_sigma=sidebar_values["threshold_sigma"],
        prominence_sigma=sidebar_values["prominence_sigma"],
        min_event_distance_sec=sidebar_values["min_event_distance"],
        context_window_sec=sidebar_values["context_window"],
        novelty_weight_cosine=sidebar_values["cosine_weight"],
        novelty_weight_flux=sidebar_values["flux_weight"],
        novelty_weight_onset=sidebar_values["onset_weight"],
        novelty_weight_rms=sidebar_values["rms_weight"],
        event_model_preset=str(sidebar_values["preset_key"]),
    )

    channel_mode = str(sidebar_values["channel_mode"])
    preset_key = str(sidebar_values["preset_key"])
    analysis_key = build_analysis_signature(audio_bytes, config, channel_mode)

    with st.spinner("正在分析频谱变化、提取特征并构建候选标注..."):
        result = analyze_audio(uploaded_file, config=config, channel_mode=channel_mode)

    event_state_key = init_event_annotations(result, analysis_key)
    section_state_key = init_section_annotations(result, analysis_key)
    event_annotations = st.session_state[event_state_key].copy()
    section_annotations = st.session_state[section_state_key].copy()
    active_filter_labels = st.session_state.get(f"event_label_filter_{analysis_key}", [])
    overview_annotations = filter_event_annotations(event_annotations, active_filter_labels)
    if active_filter_labels:
        overview_event_times = (
            overview_annotations["time_sec"].to_numpy(dtype=float)
            if not overview_annotations.empty
            else result["peak_times"][:0]
        )
    else:
        overview_event_times = result["peak_times"]

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("时长", f"{result['duration_sec']:.2f} 秒")
    metric_2.metric("采样率", f"{result['sr']} Hz")
    metric_3.metric("检测事件数", int(len(result["event_table"])))
    metric_4.metric(
        "候选边界数",
        int(result["event_table"]["is_major_boundary"].sum()) if not result["event_table"].empty else 0,
    )

    st.subheader("机器摘要")
    st.text(build_summary_text(result["summary_lines"]))

    tab_1, tab_2, tab_3, tab_4 = st.tabs(["总览", "交互分析", "标注编辑", "导出"])

    with tab_1:
        st.markdown("**同步总览播放器**")
        max_analyzable_hz = float(result["sr"]) / 2.0
        st.caption(
            "这一块是不带自动事件编号线的纯频谱总览。播放时，波形和频谱会一起跟着播放头移动，并自动聚焦到播放头附近的局部时间窗口。"
            f"当前分析采样率为 {int(result['sr'])} Hz，所以机器最多能客观分析到约 {int(max_analyzable_hz)} Hz。"
            "如果原文件本身只有 44.1kHz，那么 25kHz 以上的信息并不在文件里。"
        )
        overview_audio_mime = getattr(uploaded_file, "type", None) or "audio/wav"
        components.html(
            build_synced_overview_player_html(
                audio_bytes=audio_bytes,
                audio_mime=overview_audio_mime,
                samples=result["selected_audio"],
                sr=int(result["sr"]),
                spectrogram_db=result["spectrogram_db"],
                spectrogram_times=result["spectrogram_times"],
                spectrogram_freqs=result["spectrogram_freqs"],
                duration_sec=float(result["duration_sec"]),
                initial_wave_amplitude_max=None,
                initial_freq_max_hz=float(result["spectrogram_freqs"][-1]) if len(result["spectrogram_freqs"]) else max_analyzable_hz,
            ),
            height=940,
            scrolling=False,
        )

        st.markdown("**原始总览图**")
        if active_filter_labels:
            st.caption(
                "总览当前只高亮这些标签对应的事件："
                + "、".join(active_filter_labels)
                + f"（{len(overview_event_times)} 个事件）"
            )
        st.pyplot(
            plot_spectrogram(
                result["spectrogram_db"],
                result["spectrogram_times"],
                result["spectrogram_freqs"],
                overview_event_times,
            ),
            clear_figure=True,
        )
        st.pyplot(
            plot_waveform(result["selected_audio"], result["sr"], overview_event_times),
            clear_figure=True,
        )
        st.pyplot(
            plot_event_density(overview_event_times, result["duration_sec"]),
            clear_figure=True,
        )

    with tab_2:
        st.markdown("**自动事件点的判定依据**")
        st.caption(f"当前模式：{EVENT_MODEL_PRESETS[preset_key]['label']}")
        st.markdown(novelty_explanation(config))
        st.pyplot(
            plot_novelty(
                result["times"],
                result["novelty"],
                result["threshold"],
                result["peak_indices"],
            ),
            clear_figure=True,
        )

        render_feature_chart(result, analysis_key)
        event_annotations, _ = render_event_editor(event_annotations, result, analysis_key, preset_key)
        st.session_state[event_state_key] = event_annotations

    with tab_3:
        if event_annotations.empty:
            st.warning("当前参数下没有检测到可编辑的自动事件。")
        else:
            st.markdown("**当前事件修订结果**")
            st.dataframe(
                event_annotations[
                    [
                        "export",
                        "event_id",
                        "time_label",
                        "auto_labels",
                        "manual_labels",
                        "review_notes",
                        "is_major_boundary",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        section_annotations = render_section_editor(section_annotations, analysis_key)
        st.session_state[section_state_key] = section_annotations

    with tab_4:
        exported_events = event_annotations.copy()
        exported_sections = section_annotations.copy()
        if not exported_events.empty:
            exported_events = exported_events.loc[exported_events["export"]].copy()
        if not exported_sections.empty:
            exported_sections = exported_sections.loc[exported_sections["export"]].copy()

        feature_csv = result["feature_table"].to_csv(index=False).encode("utf-8-sig")
        event_csv = exported_events.to_csv(index=False).encode("utf-8-sig")
        section_csv = exported_sections.to_csv(index=False).encode("utf-8-sig")
        summary_text = build_summary_text(result["summary_lines"]).encode("utf-8")

        analysis_json = json.dumps(
            {
                "config": result["config"],
                "duration_sec": result["duration_sec"],
                "sr": result["sr"],
                "channel_mode": result["channel_mode"],
                "summary_lines": result["summary_lines"],
                "events": exported_events.to_dict(orient="records"),
                "sections": exported_sections.to_dict(orient="records"),
                "feature_table": result["feature_table"].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        st.download_button("下载编辑后事件表 CSV", data=event_csv, file_name="annotated_events.csv", mime="text/csv")
        st.download_button("下载编辑后段落表 CSV", data=section_csv, file_name="annotated_sections.csv", mime="text/csv")
        st.download_button("下载特征曲线 CSV", data=feature_csv, file_name="feature_table.csv", mime="text/csv")
        st.download_button("下载摘要 TXT", data=summary_text, file_name="summary.txt", mime="text/plain")
        st.download_button("下载完整分析 JSON", data=analysis_json, file_name="analysis.json", mime="application/json")

    st.markdown(
        """
提示：

- 事件太少：降低“检测阈值强度”或缩短“最小事件间隔”
- 事件太密：提高阈值或增大最小事件间隔
- 点击自动标记点后，右侧会直接显示精确时间与局部试听
- 人工修订后的标签会参与最终导出
"""
    )

