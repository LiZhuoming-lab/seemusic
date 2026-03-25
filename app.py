from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from spectral_tool.assistant import (
    annotate_event_interaction_levels,
    build_assistant_overlay_component,
    build_event_assistant_payload,
)
from spectral_tool.analysis import (
    AnalysisConfig,
    analyze_audio,
    build_audio_excerpt_wav,
    format_seconds,
    join_labels,
    split_label_text,
)
from spectral_tool.visualization import (
    build_feature_curve_chart,
    build_interactive_novelty_chart,
    build_synced_overview_player_html,
    build_synced_waveform_player_html,
    build_summary_text,
    extract_selected_point_value,
    plot_event_density,
    plot_local_spectrogram,
    plot_novelty,
    plot_spectrogram,
    plot_waveform,
)

EVENT_MODEL_PRESETS = {
    "balanced_default": {
        "label": "平衡默认",
        "weights": (0.40, 0.25, 0.20, 0.15),
        "threshold_sigma": 1.0,
        "min_event_distance_sec": 5.0,
        "note": "当前的通用默认模式，适合先做一轮平衡检测。",
    },
    "traditional_structure": {
        "label": "传统结构模式",
        "weights": (0.30, 0.20, 0.30, 0.20),
        "threshold_sigma": 1.2,
        "min_event_distance_sec": 6.5,
        "note": "更强调起音、力度、织体和结构边界，适合古典、浪漫主义钢琴和交响作品。",
    },
    "timbre_soundscape": {
        "label": "音色 / 声景模式",
        "weights": (0.45, 0.30, 0.15, 0.10),
        "threshold_sigma": None,
        "min_event_distance_sec": None,
        "note": "更强调新材料、新音色和频谱状态变化，适合频谱音乐、当代音乐、声景音乐。",
    },
}


def _format_weight_percent(weight: float) -> int:
    return int(round(float(weight) * 100))


def _novelty_explanation(config: AnalysisConfig) -> str:
    return f"""
自动事件点来自“频谱新颖度检测”，它看的是相邻时刻之间的频谱是否发生了明显变化，而不是只看某一个单独音高。

- `{_format_weight_percent(config.novelty_weight_cosine)}%` 频谱形态差异：比较前后频谱整体轮廓是否变了
- `{_format_weight_percent(config.novelty_weight_flux)}%` 谱流量 `spectral flux`：看新的频率能量是否突然涌入
- `{_format_weight_percent(config.novelty_weight_onset)}%` 起音强度 `onset strength`：看瞬时攻击和突发性是否增强
- `{_format_weight_percent(config.novelty_weight_rms)}%` 能量突变 `RMS delta`：看整体响度是否突然变化

当这一条“新颖度曲线”超过阈值并形成峰值时，系统就会在那个时间点标成一个候选事件。
"""


def _analysis_signature(audio_bytes: bytes, config: AnalysisConfig, channel_mode: str) -> str:
    payload = {
        "channel_mode": channel_mode,
        "config": config.to_dict(),
        "audio_sha1": hashlib.sha1(audio_bytes).hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _init_event_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"event_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["event_table"].copy()
        if base.empty:
            st.session_state[state_key] = base
        else:
            base["manual_labels"] = base["auto_labels"]
            base["review_notes"] = ""
            base["export"] = True
            st.session_state[state_key] = base
    return state_key


def _init_section_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"section_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["section_table"].copy()
        if base.empty:
            st.session_state[state_key] = base
        else:
            base["manual_labels"] = base["auto_labels"]
            base["review_notes"] = ""
            base["export"] = True
            st.session_state[state_key] = base
    return state_key


def _event_label(options: list[int], annotations: pd.DataFrame, event_id: int) -> str:
    row = annotations.loc[annotations["event_id"] == event_id].iloc[0]
    priority_label = str(row.get("interaction_priority_label", "")).strip()
    prefix = f"[{priority_label}] " if priority_label else ""
    return f"{prefix}事件 {event_id} | {row['time_label']} | {row['manual_labels'] or row['auto_labels']}"


def _section_label(annotations: pd.DataFrame, section_id: int) -> str:
    row = annotations.loc[annotations["section_id"] == section_id].iloc[0]
    return f"段落 {section_id} | {row['start_label']} - {row['end_label']}"


def _effective_event_labels(row: pd.Series) -> str:
    manual = str(row.get("manual_labels", "")).strip()
    auto = str(row.get("auto_labels", "")).strip()
    return manual or auto


def _available_event_filter_labels(
    annotations: pd.DataFrame,
    preferred_vocab: list[str],
) -> list[str]:
    observed: list[str] = []
    for _, row in annotations.iterrows():
        observed.extend(split_label_text(_effective_event_labels(row)))

    ordered: list[str] = []
    seen: set[str] = set()
    for label in preferred_vocab + observed:
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _filter_event_annotations(
    annotations: pd.DataFrame,
    selected_labels: list[str],
) -> pd.DataFrame:
    if annotations.empty or not selected_labels:
        return annotations.copy()

    selected_set = set(selected_labels)
    mask = annotations.apply(
        lambda row: bool(selected_set.intersection(split_label_text(_effective_event_labels(row)))),
        axis=1,
    )
    return annotations.loc[mask].copy()


def _prepare_event_editor(
    annotations: pd.DataFrame,
    result: dict[str, object],
    analysis_key: str,
) -> tuple[pd.DataFrame, int | None]:
    if annotations.empty:
        return annotations, None

    filter_key = f"event_label_filter_{analysis_key}"
    filter_options = _available_event_filter_labels(annotations, list(result["event_label_vocab"]))
    selected_filter_labels = st.multiselect(
        "按标签筛选事件图",
        options=filter_options,
        key=filter_key,
        help="优先使用人工标签筛选；如果某个事件还没有人工标签，则回退使用自动标签。",
    )

    interaction_annotations = annotations.copy()
    interaction_annotations["effective_labels"] = interaction_annotations.apply(_effective_event_labels, axis=1)
    interaction_annotations = annotate_event_interaction_levels(interaction_annotations)
    filtered_annotations = _filter_event_annotations(interaction_annotations, selected_filter_labels)
    if selected_filter_labels:
        st.caption(f"当前筛选后显示 {len(filtered_annotations)} / {len(annotations)} 个事件。")
    else:
        st.caption(f"当前显示全部 {len(annotations)} 个事件。")

    if filtered_annotations.empty:
        st.warning("当前标签筛选下没有匹配事件。可以清空筛选，或者换一个标签试试。")
        return annotations, None

    show_minor_key = f"show_minor_changes_{analysis_key}"
    show_minor_changes = st.toggle(
        "显示更多较小变化",
        value=False,
        key=show_minor_key,
        help="默认优先突出更值得互动讨论的事件；打开后可把较小变化也放进当前交互层里。",
    )
    weak_count = int((filtered_annotations["interaction_priority"] == "weak").sum())
    strong_count = int((filtered_annotations["interaction_priority"] == "strong").sum())
    medium_count = int((filtered_annotations["interaction_priority"] == "medium").sum())
    st.caption(f"当前交互层级分布：重点变化 {strong_count} 个，一般变化 {medium_count} 个，较小变化 {weak_count} 个。")

    visible_annotations = filtered_annotations.copy()
    if not show_minor_changes:
        visible_annotations = filtered_annotations.loc[filtered_annotations["interaction_priority"] != "weak"].copy()
        if visible_annotations.empty:
            visible_annotations = filtered_annotations.copy()
        elif weak_count:
            st.caption("较小变化默认收起；如果你也想把它们放进同一层互动，可以打开“显示更多较小变化”。")

    selector_key = f"active_event_{analysis_key}"
    event_ids = visible_annotations["event_id"].astype(int).tolist()
    if selector_key not in st.session_state or int(st.session_state[selector_key]) not in event_ids:
        st.session_state[selector_key] = event_ids[0]

    chart_source = visible_annotations.copy()
    chart_source["auto_labels"] = chart_source.apply(_effective_event_labels, axis=1)
    selected_from_chart = st.altair_chart(
        build_interactive_novelty_chart(result["feature_table"], chart_source),
        width="stretch",
        on_select="rerun",
        selection_mode="event_pick",
        key=f"novelty_pick_{analysis_key}",
    )
    clicked_event = extract_selected_point_value(selected_from_chart, "event_pick", "event_id")
    if clicked_event is not None:
        try:
            clicked_event_id = int(clicked_event)
            if clicked_event_id in event_ids:
                st.session_state[selector_key] = clicked_event_id
        except (TypeError, ValueError):
            pass

    st.caption(
        "这些自动事件点不是只看某一个音高或某一根频率线，而是综合看相邻频谱帧之间的结构变化。"
        "当前算法主要依据：频谱形态差异（cosine distance）、谱流量（spectral flux）、起音强度（onset strength）和能量突变（RMS delta）。"
    )
    active_event_id = st.selectbox(
        "当前事件",
        options=event_ids,
        format_func=lambda event_id: _event_label(event_ids, visible_annotations, event_id),
        key=selector_key,
    )

    active_row = visible_annotations.loc[visible_annotations["event_id"] == active_event_id].iloc[0]
    row_index = annotations.index[annotations["event_id"] == active_event_id][0]

    detail_col, right_col = st.columns([1.08, 0.92], gap="large")
    with detail_col:
        st.markdown("**事件详情**")
        st.metric("精确时间", active_row["time_label"])
        st.caption(
            f"交互层级 {active_row['interaction_priority_label']} | 强度 {active_row['strength']:.3f} | 显著度 {active_row['prominence']:.3f} | 声道 {active_row['channel_bias']}"
        )
        st.write(f"自动候选标签：`{active_row['auto_labels']}`")
        st.write(f"规则摘要：{active_row['descriptor']}")
        st.write(f"主导频率：{active_row['dominant_freqs']}")

        default_manual = split_label_text(str(active_row["manual_labels"])) or split_label_text(str(active_row["auto_labels"]))
        default_custom = [
            label for label in default_manual if label not in result["event_label_vocab"]
        ]
        default_manual_vocab = [label for label in default_manual if label in result["event_label_vocab"]]

        manual_vocab_key = f"manual_vocab_{analysis_key}_{active_event_id}"
        manual_custom_key = f"manual_custom_{analysis_key}_{active_event_id}"
        review_note_key = f"review_note_{analysis_key}_{active_event_id}"
        export_key = f"export_event_{analysis_key}_{active_event_id}"
        active_editor_key = f"active_event_editor_{analysis_key}"

        if st.session_state.get(active_editor_key) != active_event_id:
            st.session_state[manual_vocab_key] = default_manual_vocab
            st.session_state[manual_custom_key] = "、".join(default_custom)
            st.session_state[review_note_key] = str(active_row["review_notes"])
            st.session_state[export_key] = bool(active_row["export"])
            st.session_state[active_editor_key] = active_event_id

        chosen_vocab = st.multiselect(
            "人工标签",
            options=result["event_label_vocab"],
            key=manual_vocab_key,
        )
        custom_labels_text = st.text_input(
            "补充标签（可自定义，使用顿号或竖线分隔）",
            key=manual_custom_key,
        )
        review_notes = st.text_area(
            "人工备注",
            key=review_note_key,
            height=100,
        )
        export_flag = st.checkbox(
            "导出该事件",
            key=export_key,
        )

        merged_labels = chosen_vocab + split_label_text(custom_labels_text)
        annotations.at[row_index, "manual_labels"] = join_labels(merged_labels)
        annotations.at[row_index, "review_notes"] = review_notes
        annotations.at[row_index, "export"] = export_flag

    with right_col:
        st.markdown("**局部试听**")
        preview_radius = st.slider(
            "试听窗口（事件前后秒数）",
            min_value=1.0,
            max_value=15.0,
            value=4.0,
            step=0.5,
            key=f"preview_radius_{analysis_key}",
        )
        event_time = float(active_row["time_sec"])
        clip_start = max(0.0, event_time - preview_radius)
        clip_end = min(float(result["duration_sec"]), event_time + preview_radius)
        st.caption(f"试听区间：{format_seconds(clip_start)} - {format_seconds(clip_end)}")
        clip_bytes = build_audio_excerpt_wav(
            result["audio"],
            int(result["sr"]),
            clip_start,
            clip_end,
            channel_mode=result["channel_mode"],
        )

        st.markdown("**同步局部波形**")
        waveform_scale = st.slider(
            "波形放大倍数",
            min_value=1.0,
            max_value=8.0,
            value=3.0,
            step=0.5,
            key=f"waveform_scale_{analysis_key}",
            help="只影响波形显示，不影响音频播放和分析结果。",
        )
        components.html(
            build_synced_waveform_player_html(
                audio_bytes=clip_bytes,
                samples=result["selected_audio"],
                sr=int(result["sr"]),
                clip_start_sec=clip_start,
                clip_end_sec=clip_end,
                event_time_sec=event_time,
                amplitude_scale=float(waveform_scale),
            ),
            height=320,
            scrolling=False,
        )

        st.markdown("**局部频谱区域图**")
        local_spec_max_freq = st.select_slider(
            "局部频谱最高显示频率",
            options=[2000, 5000, 8000, 10000, 16000, 22050],
            value=10000,
            key=f"local_spec_max_freq_{analysis_key}",
        )
        st.pyplot(
            plot_local_spectrogram(
                result["spectrogram_db"],
                result["spectrogram_times"],
                result["spectrogram_freqs"],
                center_time=event_time,
                window_radius_sec=preview_radius,
                highlight_time=event_time,
                max_freq_hz=float(local_spec_max_freq),
            ),
            clear_figure=True,
        )

    st.markdown("**事件标注列表**")
    editor_view = visible_annotations[
        [
            "export",
            "event_id",
            "time_label",
            "interaction_priority_label",
            "is_major_boundary",
            "auto_labels",
            "manual_labels",
            "review_notes",
            "descriptor",
            "channel_bias",
        ]
    ].copy()
    edited = st.data_editor(
        editor_view,
        width="stretch",
        hide_index=True,
        column_config={
            "export": st.column_config.CheckboxColumn("导出"),
            "event_id": st.column_config.NumberColumn("事件", disabled=True),
            "time_label": st.column_config.TextColumn("时间", disabled=True),
            "interaction_priority_label": st.column_config.TextColumn("交互层级", disabled=True),
            "is_major_boundary": st.column_config.CheckboxColumn("强边界", disabled=True),
            "auto_labels": st.column_config.TextColumn("自动标签", disabled=True, width="medium"),
            "manual_labels": st.column_config.TextColumn("人工标签", width="medium"),
            "review_notes": st.column_config.TextColumn("备注", width="large"),
            "descriptor": st.column_config.TextColumn("规则摘要", disabled=True, width="large"),
            "channel_bias": st.column_config.TextColumn("声道", disabled=True),
        },
        key=f"event_editor_{analysis_key}",
    )

    for edited_row in edited.itertuples(index=False):
        target_index = annotations.index[annotations["event_id"] == int(edited_row.event_id)][0]
        annotations.at[target_index, "export"] = bool(edited_row.export)
        annotations.at[target_index, "manual_labels"] = str(edited_row.manual_labels)
        annotations.at[target_index, "review_notes"] = str(edited_row.review_notes)

    effective_label_text = _effective_event_labels(active_row)
    assistant_payload = build_event_assistant_payload(
        active_row.to_dict(),
        effective_label_text=effective_label_text,
    )
    components.html(
        build_assistant_overlay_component(active_event_id, assistant_payload),
        height=1,
        scrolling=False,
    )
    return annotations, active_event_id


def _prepare_section_editor(annotations: pd.DataFrame, analysis_key: str) -> pd.DataFrame:
    if annotations.empty:
        return annotations

    st.markdown("**段落边界与标签列表**")
    editor_view = annotations[
        [
            "export",
            "section_id",
            "start_label",
            "end_label",
            "event_count",
            "auto_labels",
            "manual_labels",
            "review_notes",
            "descriptor",
        ]
    ].copy()

    edited = st.data_editor(
        editor_view,
        width="stretch",
        hide_index=True,
        column_config={
            "export": st.column_config.CheckboxColumn("导出"),
            "section_id": st.column_config.NumberColumn("段落", disabled=True),
            "start_label": st.column_config.TextColumn("起点", disabled=True),
            "end_label": st.column_config.TextColumn("终点", disabled=True),
            "event_count": st.column_config.NumberColumn("事件数", disabled=True),
            "auto_labels": st.column_config.TextColumn("自动标签", disabled=True, width="medium"),
            "manual_labels": st.column_config.TextColumn("人工标签", width="medium"),
            "review_notes": st.column_config.TextColumn("备注", width="large"),
            "descriptor": st.column_config.TextColumn("规则摘要", disabled=True, width="large"),
        },
        key=f"section_editor_{analysis_key}",
    )

    annotations.loc[:, "export"] = edited["export"].astype(bool).to_numpy()
    annotations.loc[:, "manual_labels"] = edited["manual_labels"].astype(str).to_numpy()
    annotations.loc[:, "review_notes"] = edited["review_notes"].astype(str).to_numpy()
    return annotations


st.set_page_config(
    page_title="声景频谱事件分析器",
    page_icon="🎧",
    layout="wide",
)

st.title("声景频谱事件分析器")
st.caption("自动检测音频中的疑似“新频谱/新材料”进入点，并支持试听、点选与人工修订。")

with st.sidebar:
    st.subheader("分析参数")
    preset_key = st.selectbox(
        "自动事件点识别模式",
        options=list(EVENT_MODEL_PRESETS.keys()),
        format_func=lambda value: EVENT_MODEL_PRESETS[value]["label"],
    )
    preset = EVENT_MODEL_PRESETS[preset_key]
    cosine_weight, flux_weight, onset_weight, rms_weight = preset["weights"]
    st.caption(preset["note"])
    st.caption(
        "当前预设权重："
        f"频谱形态差异 {_format_weight_percent(cosine_weight)}% / "
        f"spectral flux {_format_weight_percent(flux_weight)}% / "
        f"onset strength {_format_weight_percent(onset_weight)}% / "
        f"RMS delta {_format_weight_percent(rms_weight)}%"
    )
    channel_mode = st.selectbox(
        "分析通道",
        options=["mix", "left", "right"],
        format_func=lambda value: {"mix": "混合", "left": "左声道", "right": "右声道"}[value],
    )
    target_sr_option = st.selectbox("目标采样率", options=["原始采样率", "44100", "22050"], index=0)
    target_sr = None if target_sr_option == "原始采样率" else int(target_sr_option)
    n_fft = st.select_slider("频谱窗长 n_fft", options=[1024, 2048, 4096, 8192], value=4096)
    hop_length = st.select_slider("帧移 hop_length", options=[256, 512, 1024, 2048], value=1024)
    if preset.get("min_event_distance_sec") is None:
        min_event_distance = st.slider("最小事件间隔（秒）", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    elif preset_key == "balanced_default":
        min_event_distance = st.slider("最小事件间隔（秒）", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    else:
        min_event_distance = float(preset["min_event_distance_sec"])
        st.caption(f"当前预设自动使用：最小事件间隔 {min_event_distance:.1f} 秒")
    context_window = st.slider("事件前后比较窗口（秒）", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    smooth_sigma = st.slider("新颖度平滑强度", min_value=0.5, max_value=4.0, value=1.6, step=0.1)
    if preset.get("threshold_sigma") is None:
        threshold_sigma = st.slider("检测阈值强度", min_value=0.2, max_value=2.5, value=1.0, step=0.1)
    elif preset_key == "balanced_default":
        threshold_sigma = st.slider("检测阈值强度", min_value=0.2, max_value=2.5, value=1.0, step=0.1)
    else:
        threshold_sigma = float(preset["threshold_sigma"])
        st.caption(f"当前预设自动使用：检测阈值强度 {threshold_sigma:.1f}")
    prominence_sigma = st.slider("峰值显著度强度", min_value=0.2, max_value=2.0, value=0.8, step=0.1)

uploaded_file = st.file_uploader(
    "上传音频文件（支持 WAV、FLAC、AIFF、MP3、M4A）",
    type=["wav", "flac", "aif", "aiff", "ogg", "mp3", "m4a"],
)

st.markdown(
    """
这个版本已经贴近本地 MVP 的完整工作流：

- 自动提取 `RMS / centroid / rolloff / flatness / flux / onset strength / novelty`
- 自动检测变化点、强边界和新事件位置
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
    target_sr=target_sr,
    n_fft=n_fft,
    hop_length=hop_length,
    smooth_sigma=smooth_sigma,
    threshold_sigma=threshold_sigma,
    prominence_sigma=prominence_sigma,
    min_event_distance_sec=min_event_distance,
    context_window_sec=context_window,
    novelty_weight_cosine=cosine_weight,
    novelty_weight_flux=flux_weight,
    novelty_weight_onset=onset_weight,
    novelty_weight_rms=rms_weight,
    event_model_preset=preset_key,
)

analysis_key = _analysis_signature(audio_bytes, config, channel_mode)

with st.spinner("正在分析频谱变化、提取特征并构建候选标注..."):
    result = analyze_audio(uploaded_file, config=config, channel_mode=channel_mode)

event_state_key = _init_event_annotations(result, analysis_key)
section_state_key = _init_section_annotations(result, analysis_key)
event_annotations = st.session_state[event_state_key].copy()
section_annotations = st.session_state[section_state_key].copy()
active_filter_labels = st.session_state.get(f"event_label_filter_{analysis_key}", [])
overview_annotations = _filter_event_annotations(event_annotations, active_filter_labels)
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
    "强边界数",
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
    st.markdown(_novelty_explanation(config))
    st.pyplot(
        plot_novelty(
            result["times"],
            result["novelty"],
            result["threshold"],
            result["peak_indices"],
        ),
        clear_figure=True,
    )

    feature_options = list(result["feature_column_labels"].keys())
    default_features = [
        feature_name
        for feature_name in [
            "rms",
            "spectral_centroid_hz",
            "rolloff_hz",
            "flatness",
            "spectral_flux",
            "onset_strength",
            "novelty",
        ]
        if feature_name in feature_options
    ]

    selected_features = st.multiselect(
        "特征曲线",
        options=feature_options,
        default=default_features,
        format_func=lambda value: result["feature_column_labels"].get(value, value),
        key=f"feature_select_{analysis_key}",
    )
    normalize_features = st.toggle("归一化显示特征曲线", value=True, key=f"normalize_{analysis_key}")

    st.altair_chart(
        build_feature_curve_chart(
            result["feature_table"],
            selected_features,
            result["feature_column_labels"],
            normalize=normalize_features,
        ),
        width="stretch",
    )

    event_annotations, active_event_id = _prepare_event_editor(event_annotations, result, analysis_key)
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

    section_annotations = _prepare_section_editor(section_annotations, analysis_key)
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
