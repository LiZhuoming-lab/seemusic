from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from spectral_tool.assistant import (
    annotate_event_interaction_levels,
    annotate_event_similarity_groups,
    build_assistant_overlay_component,
    build_event_assistant_payload,
)
from spectral_tool.analysis import (
    CANDIDATE_BOUNDARY_LABEL,
    AnalysisConfig,
    analyze_audio,
    build_audio_excerpt_wav,
    format_seconds,
    join_labels,
    split_label_text,
)
from spectral_tool.beethoven_sonatas import (
    BEETHOVEN_SONATAS_WEB_URL,
    download_beethoven_sonata_score,
    list_beethoven_sonata_scores,
)
from spectral_tool.symbolic_analysis import (
    SymbolicAnalysisConfig,
    analyze_symbolic_score,
    build_symbolic_export_payload,
    export_symbolic_score_file,
)
from spectral_tool.when_in_rome import (
    WHEN_IN_ROME_WEB_URL,
    download_when_in_rome_score,
    list_when_in_rome_scores,
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

OPERATION_RESULT_OPTIONS = [
    "保留",
    "删除",
    f"改成{CANDIDATE_BOUNDARY_LABEL}",
    "改成局部事件",
    "暂不处理",
]


def _format_weight_percent(weight: float) -> int:
    return int(round(float(weight) * 100))


def _novelty_explanation(config: AnalysisConfig) -> str:
    extra_note = ""
    if config.event_model_preset == "timbre_soundscape":
        extra_note = """
在“音色 / 声景模式”里，上面的“频谱形态差异”这一项还会额外参考：

- `spectral centroid`：看频谱重心有没有整体移动
- `band energy ratio`：看不同频带之间的能量关系有没有改写
- `高频占比`：看高频材料是否明显抬升或退去
- `flatness`：看声音是否更趋向噪声质地

这样可以更敏感地抓到“新音色 / 新材质 / 新频谱状态”的进入。
"""

    return f"""
自动事件点来自“频谱新颖度检测”，它看的是相邻时刻之间的频谱是否发生了明显变化，而不是只看某一个单独音高。

- `{_format_weight_percent(config.novelty_weight_cosine)}%` 频谱形态差异：比较前后频谱整体轮廓是否变了
- `{_format_weight_percent(config.novelty_weight_flux)}%` 谱流量 `spectral flux`：看新的频率能量是否突然涌入
- `{_format_weight_percent(config.novelty_weight_onset)}%` 起音强度 `onset strength`：看瞬时攻击和突发性是否增强
- `{_format_weight_percent(config.novelty_weight_rms)}%` 能量突变 `RMS delta`：看整体响度是否突然变化

当这一条“新颖度曲线”超过阈值并形成峰值时，系统就会在那个时间点标成一个候选事件。
{extra_note}
"""


def _analysis_signature(audio_bytes: bytes, config: AnalysisConfig, channel_mode: str) -> str:
    payload = {
        "channel_mode": channel_mode,
        "config": config.to_dict(),
        "audio_sha1": hashlib.sha1(audio_bytes).hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _symbolic_analysis_signature(score_bytes: bytes, config: SymbolicAnalysisConfig) -> str:
    payload = {
        "config": config.to_dict(),
        "score_sha1": hashlib.sha1(score_bytes).hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def _load_when_in_rome_catalog() -> pd.DataFrame:
    return list_when_in_rome_scores()


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def _load_when_in_rome_score_bytes(path: str) -> tuple[bytes, str]:
    return download_when_in_rome_score(path)


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def _build_musicxml_export_bytes(score_bytes: bytes, source_name: str) -> tuple[bytes, str, str]:
    score_stream = io.BytesIO(score_bytes)
    score_stream.name = source_name
    return export_symbolic_score_file(score_stream, fmt="musicxml")


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def _load_beethoven_sonata_catalog() -> pd.DataFrame:
    return list_beethoven_sonata_scores()


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def _load_beethoven_sonata_score_bytes(path: str) -> tuple[bytes, str]:
    return download_beethoven_sonata_score(path)


def _ensure_event_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ensured = frame.copy()
    if "manual_labels" not in ensured.columns:
        if "auto_labels" in ensured.columns:
            ensured["manual_labels"] = ensured["auto_labels"].astype(str)
        else:
            ensured["manual_labels"] = pd.Series(dtype="object")
    if "review_notes" not in ensured.columns:
        ensured["review_notes"] = ""
    if "export" not in ensured.columns:
        ensured["export"] = True
    if "manual_time_sec" not in ensured.columns:
        ensured["manual_time_sec"] = ensured["time_sec"] if "time_sec" in ensured.columns else pd.Series(dtype="float64")
    if "manual_time_label" not in ensured.columns:
        if "time_label" in ensured.columns:
            ensured["manual_time_label"] = ensured["time_label"].astype(str)
        else:
            ensured["manual_time_label"] = ""
    if "operation_result" not in ensured.columns:
        ensured["operation_result"] = ""
    if "operation_log" not in ensured.columns:
        ensured["operation_log"] = ""
    return ensured


def _init_event_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"event_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["event_table"].copy()
        if base.empty:
            st.session_state[state_key] = base
        else:
            st.session_state[state_key] = _ensure_event_annotation_columns(base)
    else:
        st.session_state[state_key] = _ensure_event_annotation_columns(st.session_state[state_key])
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


def _ensure_harmony_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ensured = frame.copy()
    if "manual_label" not in ensured.columns:
        default_labels = ensured.get("roman_numeral", pd.Series(dtype="object")).astype(str)
        fallback_labels = ensured.get("root", pd.Series(dtype="object")).astype(str)
        ensured["manual_label"] = default_labels.where(default_labels.str.strip() != "", fallback_labels)
    if "review_notes" not in ensured.columns:
        ensured["review_notes"] = ""
    if "export" not in ensured.columns:
        ensured["export"] = True
    return ensured


def _ensure_theme_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ensured = frame.copy()
    if "manual_relation" not in ensured.columns:
        ensured["manual_relation"] = ensured.get("relation_type", pd.Series(dtype="object")).astype(str)
    if "review_notes" not in ensured.columns:
        ensured["review_notes"] = ""
    if "export" not in ensured.columns:
        ensured["export"] = True
    return ensured


def _ensure_cadence_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ensured = frame.copy()
    if "melody_skeleton_class" not in ensured.columns:
        if "melody_scale_degree" in ensured.columns:
            ensured["melody_skeleton_class"] = ensured["melody_scale_degree"].apply(
                lambda value: "主音骨架（1）"
                if pd.notna(value) and int(value) == 1
                else f"非主音骨架（{int(value)}）"
                if pd.notna(value) and int(value) in {3, 5}
                else "骨架未定"
            )
        else:
            ensured["melody_skeleton_class"] = "骨架未定"
    if "manual_label" not in ensured.columns:
        ensured["manual_label"] = ensured.get("cadence_type", pd.Series(dtype="object")).astype(str)
    if "review_notes" not in ensured.columns:
        ensured["review_notes"] = ""
    if "export" not in ensured.columns:
        ensured["export"] = True
    return ensured


def _sync_cadence_annotations(existing: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    latest_ensured = _ensure_cadence_annotation_columns(latest)
    if existing.empty:
        return latest_ensured

    existing_ensured = _ensure_cadence_annotation_columns(existing)
    identity_columns = ["cadence_type", "cadence_window", "cadence_key", "measure_number", "beat"]

    for column_name in identity_columns:
        if column_name not in existing_ensured.columns:
            existing_ensured[column_name] = None
        if column_name not in latest_ensured.columns:
            latest_ensured[column_name] = None

    existing_keyed = existing_ensured.set_index(identity_columns, drop=False)
    latest_keyed = latest_ensured.set_index(identity_columns, drop=False)

    editable_columns = ["manual_label", "review_notes", "export"]
    shared_keys = latest_keyed.index.intersection(existing_keyed.index)
    for column_name in editable_columns:
        if column_name not in existing_keyed.columns or column_name not in latest_keyed.columns:
            continue
        latest_keyed.loc[shared_keys, column_name] = existing_keyed.loc[shared_keys, column_name]

    return latest_keyed.reset_index(drop=True)


def _cadence_result_signature(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    normalized = frame.copy()
    ordered_columns = [
        "cadence_type",
        "cadence_window",
        "cadence_key",
        "measure_number",
        "beat",
        "candidate_score",
        "previous_roman_numeral",
        "current_roman_numeral",
    ]
    available_columns = [column_name for column_name in ordered_columns if column_name in normalized.columns]
    normalized = normalized[available_columns].sort_values(
        [column_name for column_name in ["measure_number", "beat", "cadence_type", "cadence_key"] if column_name in available_columns]
    ).reset_index(drop=True)
    return hashlib.sha1(normalized.to_json(orient="records", force_ascii=False).encode("utf-8")).hexdigest()


def _init_harmony_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"harmony_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["harmony_table"].copy()
        st.session_state[state_key] = _ensure_harmony_annotation_columns(base)
    else:
        st.session_state[state_key] = _ensure_harmony_annotation_columns(st.session_state[state_key])
    return state_key


def _init_theme_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"theme_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["theme_matches"].copy()
        st.session_state[state_key] = _ensure_theme_annotation_columns(base)
    else:
        st.session_state[state_key] = _ensure_theme_annotation_columns(st.session_state[state_key])
    return state_key


def _init_cadence_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"cadence_annotations_{analysis_key}"
    signature_key = f"{state_key}_result_signature"
    editor_key = f"cadence_editor_{analysis_key}"
    base = result["cadence_candidates"].copy()
    latest_signature = _cadence_result_signature(base)
    if state_key not in st.session_state:
        st.session_state[state_key] = _ensure_cadence_annotation_columns(base)
        st.session_state[signature_key] = latest_signature
    else:
        st.session_state[state_key] = _sync_cadence_annotations(st.session_state[state_key], base)
        if st.session_state.get(signature_key) != latest_signature:
            st.session_state[signature_key] = latest_signature
            if editor_key in st.session_state:
                del st.session_state[editor_key]
    return state_key


def _event_label(options: list[int], annotations: pd.DataFrame, event_id: int) -> str:
    row = annotations.loc[annotations["event_id"] == event_id].iloc[0]
    priority_label = str(row.get("interaction_priority_label", "")).strip()
    prefix = f"[{priority_label}] " if priority_label else ""
    display_time = str(row.get("manual_time_label", "")).strip() or str(row["time_label"])
    return f"{prefix}事件 {event_id} | {display_time} | {row['manual_labels'] or row['auto_labels']}"


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


def _format_operation_log(
    mode_label: str,
    row: pd.Series,
    operation_result: str,
    manual_time_sec: float,
    review_notes: str,
) -> str:
    original_time_label = str(row.get("time_label", "--"))
    adjusted_time_label = format_seconds(float(manual_time_sec))
    pre_rms = max(float(row.get("pre_rms", 0.0)), 1e-8)
    post_rms = float(row.get("post_rms", 0.0))
    pre_centroid = max(float(row.get("pre_centroid_hz", 1.0)), 1.0)
    post_centroid = float(row.get("post_centroid_hz", 0.0))
    pre_flatness = float(row.get("pre_flatness", 0.0))
    post_flatness = float(row.get("post_flatness", 0.0))
    pre_high_ratio = float(row.get("pre_high_ratio", 0.0))
    post_high_ratio = float(row.get("post_high_ratio", 0.0))
    rms_ratio = post_rms / pre_rms
    centroid_ratio = post_centroid / pre_centroid
    flatness_delta = post_flatness - pre_flatness
    high_delta = post_high_ratio - pre_high_ratio
    lines = [
        f"事件 {int(row.get('event_id', 0))}",
        f"当前模式：{mode_label}",
        f"原始时间：{original_time_label}",
        f"人工时间：{adjusted_time_label}",
        f"当前自动标签：{str(row.get('auto_labels', ''))}",
        f"当前人工标签：{str(row.get('manual_labels', ''))}",
        (
            "当前事件特征："
            f"rms_ratio={rms_ratio:.3f}, "
            f"centroid_ratio={centroid_ratio:.3f}, "
            f"flatness_delta={flatness_delta:.3f}, "
            f"high_delta={high_delta:.3f}"
        ),
        f"你的操作结果：{operation_result or '未记录'}",
    ]
    note_text = review_notes.strip()
    if note_text:
        lines.append(f"说明：{note_text}")
    return "\n".join(lines)


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
    interaction_annotations = annotate_event_similarity_groups(interaction_annotations, threshold=0.90)
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
    st.caption("事件点颜色现在表示“相似事件组”：如果两个事件的后态声音特征相似度达到约 90% 及以上，它们会用同一种颜色。")
    if str(result["config"].get("event_model_preset", "")) == "timbre_soundscape":
        st.caption("在“音色 / 声景模式”下，频谱形态差异这一项还会补充参考 spectral centroid、band energy ratio、高频占比和 flatness。")
    active_event_id = st.selectbox(
        "当前事件",
        options=event_ids,
        format_func=lambda event_id: _event_label(event_ids, visible_annotations, event_id),
        key=selector_key,
    )

    active_row = visible_annotations.loc[visible_annotations["event_id"] == active_event_id].iloc[0]
    row_index = annotations.index[annotations["event_id"] == active_event_id][0]
    active_mode_label = EVENT_MODEL_PRESETS[preset_key]["label"]

    detail_col, right_col = st.columns([1.08, 0.92], gap="large")
    with detail_col:
        st.markdown("**事件详情**")
        current_time_label = str(active_row.get("manual_time_label", "")).strip() or str(active_row["time_label"])
        st.metric("当前时间", current_time_label)
        if current_time_label != str(active_row["time_label"]):
            st.caption(f"原始检测时间：{active_row['time_label']}")
        st.caption(
            f"交互层级 {active_row['interaction_priority_label']} | {active_row['similarity_group_label']} | 强度 {active_row['strength']:.3f} | 显著度 {active_row['prominence']:.3f} | 声道 {active_row['channel_bias']}"
        )
        st.write(f"自动候选标签：`{active_row['auto_labels']}`")
        st.write(f"规则摘要：{active_row['descriptor']}")
        st.write(f"主导频率：{active_row['dominant_freqs']}")
        st.write(
            f"相似组信息：{active_row['similarity_group_label']}，组内共 {int(active_row['similarity_group_size'])} 个事件，"
            f"当前点与组内其他事件的最高相似度约为 {float(active_row['max_similarity_in_group']):.0%}"
        )

        default_manual = split_label_text(str(active_row["manual_labels"])) or split_label_text(str(active_row["auto_labels"]))
        default_custom = [
            label for label in default_manual if label not in result["event_label_vocab"]
        ]
        default_manual_vocab = [label for label in default_manual if label in result["event_label_vocab"]]

        manual_vocab_key = f"manual_vocab_{analysis_key}_{active_event_id}"
        manual_custom_key = f"manual_custom_{analysis_key}_{active_event_id}"
        review_note_key = f"review_note_{analysis_key}_{active_event_id}"
        export_key = f"export_event_{analysis_key}_{active_event_id}"
        manual_time_key = f"manual_time_{analysis_key}_{active_event_id}"
        operation_result_key = f"operation_result_{analysis_key}_{active_event_id}"
        active_editor_key = f"active_event_editor_{analysis_key}"

        if st.session_state.get(active_editor_key) != active_event_id:
            st.session_state[manual_vocab_key] = default_manual_vocab
            st.session_state[manual_custom_key] = "、".join(default_custom)
            st.session_state[review_note_key] = str(active_row["review_notes"])
            st.session_state[export_key] = bool(active_row["export"])
            st.session_state[manual_time_key] = float(active_row.get("manual_time_sec", active_row["time_sec"]))
            st.session_state[operation_result_key] = str(active_row.get("operation_result", "")).strip() or "暂不处理"
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
        st.markdown("**人工微调事件时间**")
        manual_time_columns = st.columns(3, gap="small")
        if manual_time_columns[0].button("前移 0.2 秒", key=f"shift_left_small_{analysis_key}_{active_event_id}", use_container_width=True):
            st.session_state[manual_time_key] = max(0.0, float(st.session_state.get(manual_time_key, active_row["time_sec"])) - 0.2)
        if manual_time_columns[1].button("回到原始时间", key=f"reset_time_{analysis_key}_{active_event_id}", use_container_width=True):
            st.session_state[manual_time_key] = float(active_row["time_sec"])
        if manual_time_columns[2].button("后移 0.2 秒", key=f"shift_right_small_{analysis_key}_{active_event_id}", use_container_width=True):
            st.session_state[manual_time_key] = min(float(result["duration_sec"]), float(st.session_state.get(manual_time_key, active_row["time_sec"])) + 0.2)

        manual_time_sec = st.number_input(
            "人工时间（秒）",
            min_value=0.0,
            max_value=float(result["duration_sec"]),
            step=0.05,
            key=manual_time_key,
            format="%.2f",
        )
        operation_result = st.selectbox(
            "你的操作结果",
            options=OPERATION_RESULT_OPTIONS,
            key=operation_result_key,
        )

        merged_labels = chosen_vocab + split_label_text(custom_labels_text)
        row_for_log = active_row.copy()
        row_for_log["manual_labels"] = join_labels(merged_labels)
        annotations.at[row_index, "manual_labels"] = join_labels(merged_labels)
        annotations.at[row_index, "review_notes"] = review_notes
        annotations.at[row_index, "export"] = export_flag
        annotations.at[row_index, "manual_time_sec"] = float(manual_time_sec)
        annotations.at[row_index, "manual_time_label"] = format_seconds(float(manual_time_sec))
        annotations.at[row_index, "operation_result"] = str(operation_result)
        annotations.at[row_index, "operation_log"] = _format_operation_log(
            mode_label=active_mode_label,
            row=row_for_log,
            operation_result=str(operation_result),
            manual_time_sec=float(manual_time_sec),
            review_notes=str(review_notes),
        )

        st.markdown("**当前操作日志（可复制）**")
        st.text_area(
            "记录区",
            value=str(annotations.at[row_index, "operation_log"]),
            height=200,
            key=f"operation_log_view_{analysis_key}_{active_event_id}",
        )

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
        event_time = float(annotations.at[row_index, "manual_time_sec"])
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
            "similarity_group_label",
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
            "similarity_group_label": st.column_config.TextColumn("相似组", disabled=True),
            "is_major_boundary": st.column_config.CheckboxColumn("候选边界", disabled=True),
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

    log_rows = annotations.loc[
        annotations.apply(
            lambda row: (
                str(row.get("operation_result", "")).strip() not in {"", "暂不处理"}
                or bool(str(row.get("review_notes", "")).strip())
                or abs(float(row.get("manual_time_sec", row.get("time_sec", 0.0))) - float(row.get("time_sec", 0.0))) > 1e-6
            ),
            axis=1,
        )
    ].sort_values("time_sec")
    if not log_rows.empty:
        combined_log = "\n\n---\n\n".join(
            str(text).strip()
            for text in log_rows["operation_log"].tolist()
            if str(text).strip()
        )
        if combined_log:
            st.markdown("**全部操作记录（可复制）**")
            st.text_area(
                "汇总记录区",
                value=combined_log,
                height=260,
                key=f"combined_operation_log_{analysis_key}",
            )

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
    page_title="音乐分析平台",
    page_icon="🎼",
    layout="wide",
)

st.title("音乐分析平台")
st.caption("保留原有声景 / 频谱事件分析，同时新增第一阶段乐谱符号分析工作台。")
if "workspace_mode" not in st.session_state:
    st.session_state["workspace_mode"] = "audio"

workspace_left, workspace_right = st.columns(2, gap="large")
with workspace_left:
    with st.container(border=True):
        st.markdown("### 左侧工作台")
        st.markdown("**声音频谱事件分析**")
        st.caption("适合做频谱、声景状态、自动事件点、局部试听与人工修订。")
        if st.session_state["workspace_mode"] == "audio":
            st.button("当前正在使用", key="workspace_audio_current", use_container_width=True, disabled=True)
        else:
            if st.button("进入左侧工作台", key="workspace_audio_switch", use_container_width=True):
                st.session_state["workspace_mode"] = "audio"
                st.rerun()

with workspace_right:
    with st.container(border=True):
        st.markdown("### 右侧工作台")
        st.markdown("**music21 符号分析**")
        st.caption("适合做音高、音级、音程、和声切片与主题 / 动机再现初筛。")
        if st.session_state["workspace_mode"] == "score":
            st.button("当前正在使用", key="workspace_score_current", use_container_width=True, disabled=True)
        else:
            if st.button("进入右侧工作台", key="workspace_score_switch", use_container_width=True):
                st.session_state["workspace_mode"] = "score"
                st.rerun()

workspace = str(st.session_state.get("workspace_mode", "audio"))

with st.sidebar:
    st.subheader("当前工作台")
    st.write("声音频谱事件分析" if workspace == "audio" else "music21 符号分析")
    st.caption("页面顶部的左右入口可以直接切换两个并列工作台。")

    if workspace == "audio":
        st.subheader("音频分析参数")
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
        if preset_key == "timbre_soundscape":
            st.caption("当前模式会额外参考：spectral centroid / band energy ratio / 高频占比 / flatness。")
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
    else:
        st.subheader("乐谱分析参数")
        theme_window_notes = st.slider("主题检索窗口（连续音数）", min_value=3, max_value=12, value=6, step=1)
        max_recurrence_results = st.slider("最多展示再现候选数", min_value=5, max_value=40, value=16, step=1)
        measure_summary_top_n = st.slider("每小节显示前几类音级 / 音高类", min_value=2, max_value=5, value=3, step=1)
        symbolic_config = SymbolicAnalysisConfig(
            theme_window_notes=theme_window_notes,
            max_recurrence_results=max_recurrence_results,
            measure_summary_top_n=measure_summary_top_n,
        )

if workspace == "score":
    st.markdown("### 乐谱 / 符号分析工作台")
    source_mode = st.radio(
        "乐谱来源",
        options=["local_upload", "when_in_rome", "beethoven_sonatas"],
        horizontal=True,
        format_func=lambda value: {
            "local_upload": "本地上传",
            "when_in_rome": "When-in-Rome 语料库",
            "beethoven_sonatas": "贝多芬钢琴奏鸣曲 32 首",
        }[value],
        key="symbolic_source_mode",
    )

    st.markdown(
        """
当前是“第一阶段可用版本”，它已经把项目从纯音频频谱工具往音乐分析平台推进了一步：

- 支持 `MusicXML / MXL / MIDI`
- 提取音高统计、音高类分布、音级分布
- 提取旋律音程与音程序类分布
- 生成和声切片与 Roman numeral 候选
- 做连续音窗口的主题 / 动机再现检索
- 允许人工修订和声标签与再现类型
"""
    )

    score_bytes: bytes | None = None
    score_source = None
    score_source_label = ""
    score_source_caption = ""

    if source_mode == "local_upload":
        score_file = st.file_uploader(
            "上传乐谱文件（支持 MusicXML、MXL、MIDI、KRN）",
            type=["musicxml", "xml", "mxl", "mid", "midi", "krn", "kern"],
            key="score_file_uploader",
        )
        if score_file is None:
            st.info("上传一份乐谱后，系统会生成和声、音高、音程和主题再现的第一轮分析结果。")
            st.stop()
        score_bytes = score_file.getvalue()
        score_source = score_file
        score_source_label = str(getattr(score_file, "name", "uploaded_score"))
        score_source_caption = "当前来源：本地上传"
    elif source_mode == "when_in_rome":
        st.markdown(f"**When-in-Rome 语料库**：[打开仓库]({WHEN_IN_ROME_WEB_URL})")
        st.caption("这里接入的是右侧符号分析工作台的外部语料库入口。你可以不上传本地乐谱，直接调用仓库里现成的 MusicXML / MXL / MIDI 文件。")

        try:
            catalog = _load_when_in_rome_catalog()
        except RuntimeError as exc:
            st.warning(str(exc))
            st.stop()
        if catalog.empty:
            st.warning("暂时没有从 When-in-Rome 语料库读取到可用乐谱。")
            st.stop()

        category_options = ["全部"] + sorted(catalog["category_label"].dropna().unique().tolist())
        selected_category = st.selectbox("类别", options=category_options, key="wir_category")
        filtered_catalog = catalog.copy()
        if selected_category != "全部":
            filtered_catalog = filtered_catalog.loc[filtered_catalog["category_label"] == selected_category].copy()

        composer_options = ["全部"] + sorted(filtered_catalog["composer_label"].dropna().unique().tolist())
        selected_composer = st.selectbox("作曲家", options=composer_options, key="wir_composer")
        if selected_composer != "全部":
            filtered_catalog = filtered_catalog.loc[filtered_catalog["composer_label"] == selected_composer].copy()

        keyword = st.text_input("关键词筛选（作品 / 曲目）", key="wir_keyword").strip()
        if keyword:
            keyword_lower = keyword.lower()
            filtered_catalog = filtered_catalog.loc[
                filtered_catalog["display_name"].str.lower().str.contains(keyword_lower, na=False)
                | filtered_catalog["path"].str.lower().str.contains(keyword_lower, na=False)
            ].copy()

        if filtered_catalog.empty:
            st.warning("当前筛选条件下没有匹配到可调用的语料文件。")
            st.stop()

        st.caption(f"当前筛选到 {len(filtered_catalog)} 份可分析乐谱。")
        selected_path = st.selectbox(
            "选择语料文件",
            options=filtered_catalog["path"].tolist(),
            format_func=lambda value: filtered_catalog.loc[filtered_catalog["path"] == value, "display_name"].iloc[0],
            key="wir_path",
        )
        selected_row = filtered_catalog.loc[filtered_catalog["path"] == selected_path].iloc[0]
        st.caption(f"当前文件：{selected_row['path']}")

        try:
            score_bytes, score_name = _load_when_in_rome_score_bytes(str(selected_path))
        except RuntimeError as exc:
            st.warning(str(exc))
            st.stop()
        score_stream = io.BytesIO(score_bytes)
        score_stream.name = str(score_name)
        score_source = score_stream
        score_source_label = str(selected_row["display_name"])
        score_source_caption = f"当前来源：When-in-Rome / {selected_row['path']}"
    else:
        st.markdown(f"**贝多芬钢琴奏鸣曲 32 首语料库**：[打开仓库]({BEETHOVEN_SONATAS_WEB_URL})")
        st.caption("这里接入的是 craigsapp 提供的 Beethoven piano sonatas GitHub 语料库，当前以 Humdrum / kern 文件为主。")

        try:
            catalog = _load_beethoven_sonata_catalog()
        except RuntimeError as exc:
            st.warning(str(exc))
            st.stop()
        if catalog.empty:
            st.warning("暂时没有从贝多芬钢琴奏鸣曲语料库读取到可用乐谱。")
            st.stop()

        sonata_numbers = sorted(int(value) for value in catalog["sonata_number"].dropna().unique().tolist() if int(value) > 0)
        selected_sonata = st.selectbox(
            "选择奏鸣曲",
            options=sonata_numbers,
            format_func=lambda value: f"Sonata No.{int(value)}",
            key="beethoven_sonata_number",
        )
        filtered_catalog = catalog.loc[catalog["sonata_number"] == int(selected_sonata)].copy()
        movement_numbers = sorted(int(value) for value in filtered_catalog["movement_number"].dropna().unique().tolist() if int(value) > 0)
        selected_movement = st.selectbox(
            "选择乐章",
            options=movement_numbers,
            format_func=lambda value: f"Movement {int(value)}",
            key="beethoven_sonata_movement",
        )
        selected_row = filtered_catalog.loc[filtered_catalog["movement_number"] == int(selected_movement)].iloc[0]
        st.caption(f"当前文件：{selected_row['path']}")

        try:
            score_bytes, score_name = _load_beethoven_sonata_score_bytes(str(selected_row["path"]))
        except RuntimeError as exc:
            st.warning(str(exc))
            st.stop()
        score_stream = io.BytesIO(score_bytes)
        score_stream.name = str(score_name)
        score_source = score_stream
        score_source_label = str(selected_row["display_name"])
        score_source_caption = f"当前来源：Beethoven Piano Sonatas / {selected_row['path']}"

    if score_bytes is None or score_source is None:
        st.stop()

    st.caption(score_source_caption)
    symbolic_analysis_key = _symbolic_analysis_signature(score_bytes, symbolic_config)
    with st.spinner("正在解析乐谱、提取音高关系并构建主题再现候选..."):
        symbolic_result = analyze_symbolic_score(score_source, config=symbolic_config)

    harmony_state_key = _init_harmony_annotations(symbolic_result, symbolic_analysis_key)
    cadence_state_key = _init_cadence_annotations(symbolic_result, symbolic_analysis_key)
    theme_state_key = _init_theme_annotations(symbolic_result, symbolic_analysis_key)
    harmony_annotations = st.session_state[harmony_state_key].copy()
    cadence_annotations = st.session_state[cadence_state_key].copy()
    theme_annotations = st.session_state[theme_state_key].copy()

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("乐谱标题", str(symbolic_result["score_title"]))
    metric_2.metric("总音高事件", int(symbolic_result["total_notes"]))
    metric_3.metric("不同音级类", int(symbolic_result["unique_pitch_classes"]))
    metric_4.metric("终止候选", int(len(symbolic_result["cadence_candidates"])))
    metric_5.metric("主题再现候选", int(len(symbolic_result["theme_matches"])))
    st.caption(f"当前分析对象：{score_source_label}")

    st.subheader("机器摘要")
    st.text("\n".join(symbolic_result["summary_lines"]))

    score_tab_1, score_tab_2, score_tab_3, score_tab_4, score_tab_5, score_tab_6, score_tab_7 = st.tabs(
        ["总览", "和声", "终止", "音高", "音程 / 关系", "主题 / 动机", "导出"]
    )

    with score_tab_1:
        st.markdown(f"**全局调性 / 调式估计：** {symbolic_result['global_key']}")
        st.markdown("**声部摘要**")
        st.dataframe(symbolic_result["part_summary"], width="stretch", hide_index=True)

        if not symbolic_result["measure_pitch_summary"].empty:
            st.markdown("**按小节的音高浓度变化**")
            measure_curve = symbolic_result["measure_pitch_summary"].set_index("measure_number")[
                ["note_count", "unique_pitch_classes"]
            ]
            st.line_chart(measure_curve, height=280)
            st.dataframe(symbolic_result["measure_pitch_summary"], width="stretch", hide_index=True)
        else:
            st.info("当前乐谱没有生成可展示的小节级音高摘要。")

    with score_tab_2:
        if harmony_annotations.empty:
            st.warning("当前乐谱没有生成有效的和声切片。")
        else:
            harmony_editor = st.data_editor(
                harmony_annotations[
                    [
                        "export",
                        "slice_id",
                        "measure_number",
                        "beat",
                        "pitch_names",
                        "roman_numeral",
                        "manual_label",
                        "review_notes",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "export": st.column_config.CheckboxColumn("导出"),
                    "slice_id": st.column_config.NumberColumn("切片", disabled=True),
                    "measure_number": st.column_config.NumberColumn("小节", disabled=True),
                    "beat": st.column_config.NumberColumn("拍点", disabled=True),
                    "pitch_names": st.column_config.TextColumn("垂直音响", disabled=True, width="large"),
                    "roman_numeral": st.column_config.TextColumn("候选级数", disabled=True),
                    "manual_label": st.column_config.TextColumn("人工标签", width="medium"),
                    "review_notes": st.column_config.TextColumn("备注", width="large"),
                },
                key=f"harmony_editor_{symbolic_analysis_key}",
            )
            harmony_annotations.loc[:, "export"] = harmony_editor["export"].astype(bool).to_numpy()
            harmony_annotations.loc[:, "manual_label"] = harmony_editor["manual_label"].astype(str).to_numpy()
            harmony_annotations.loc[:, "review_notes"] = harmony_editor["review_notes"].astype(str).to_numpy()
            st.session_state[harmony_state_key] = harmony_annotations

            if not symbolic_result["harmony_table"].empty:
                harmony_counts = (
                    symbolic_result["harmony_table"]["roman_numeral"]
                    .astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .value_counts()
                    .head(12)
                )
                if not harmony_counts.empty:
                    st.markdown("**最常见和声候选**")
                    st.bar_chart(harmony_counts)

    with score_tab_3:
        if cadence_annotations.empty:
            st.warning("当前规则下没有检测到终止候选。")
        else:
            st.caption(
                "当前终止检测分成两类：完满终止候选与半终止候选。"
                "完满终止更看重属准备、主到达、旋律骨架落在 1 与终止窗口的 5-1 支撑；"
                "半终止更看重属到达、属音骨架与句末收束。"
            )
            pac_candidates = cadence_annotations.loc[
                cadence_annotations["cadence_type"].astype(str) == "完满终止候选"
            ].copy()
            hc_candidates = cadence_annotations.loc[
                cadence_annotations["cadence_type"].astype(str) == "半终止候选"
            ].copy()
            tonic_skeleton_candidates = pac_candidates.loc[
                pac_candidates["melody_skeleton_class"].astype(str) == "主音骨架（1）"
            ].copy()
            summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
            summary_col_1.metric("完满终止", int(len(pac_candidates)))
            summary_col_2.metric("半终止", int(len(hc_candidates)))
            summary_col_3.metric("主音骨架 PAC", int(len(tonic_skeleton_candidates)))

            if not tonic_skeleton_candidates.empty:
                st.markdown("**旋律骨架为 1 的候选**")
                st.dataframe(
                    tonic_skeleton_candidates[
                        [
                            "measure_number",
                            "cadence_window",
                            "cadence_key",
                            "candidate_score",
                            "melody_pitch_name",
                            "melody_scale_degree",
                            "melody_skeleton_class",
                            "previous_roman_numeral",
                            "current_roman_numeral",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

            cadence_editor = st.data_editor(
                cadence_annotations[
                    [
                        "export",
                        "cadence_type",
                        "cadence_window",
                        "cadence_key",
                        "candidate_score",
                        "dominant_score",
                        "tonic_score",
                        "melody_score",
                        "bass_score",
                        "closure_score",
                        "measure_number",
                        "beat",
                        "melody_skeleton_class",
                        "previous_roman_numeral",
                        "current_roman_numeral",
                        "previous_bass_scale_degree",
                        "current_bass_scale_degree",
                        "melody_pitch_name",
                        "melody_scale_degree",
                        "manual_label",
                        "review_notes",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "export": st.column_config.CheckboxColumn("导出"),
                    "cadence_type": st.column_config.TextColumn("系统判断", disabled=True),
                    "cadence_window": st.column_config.TextColumn("终止窗口", disabled=True),
                    "cadence_key": st.column_config.TextColumn("判定调性", disabled=True),
                    "candidate_score": st.column_config.NumberColumn("总分", disabled=True, format="%.3f"),
                    "dominant_score": st.column_config.NumberColumn("属准备", disabled=True, format="%.3f"),
                    "tonic_score": st.column_config.NumberColumn("主到达", disabled=True, format="%.3f"),
                    "melody_score": st.column_config.NumberColumn("旋律骨架", disabled=True, format="%.3f"),
                    "bass_score": st.column_config.NumberColumn("低音支撑", disabled=True, format="%.3f"),
                    "closure_score": st.column_config.NumberColumn("句末收束", disabled=True, format="%.3f"),
                    "measure_number": st.column_config.NumberColumn("小节", disabled=True),
                    "beat": st.column_config.NumberColumn("拍点", disabled=True),
                    "melody_skeleton_class": st.column_config.TextColumn("旋律骨架分类", disabled=True),
                    "previous_roman_numeral": st.column_config.TextColumn("前和声", disabled=True),
                    "current_roman_numeral": st.column_config.TextColumn("当前和声", disabled=True),
                    "previous_bass_scale_degree": st.column_config.NumberColumn("前窗口低音级数", disabled=True),
                    "current_bass_scale_degree": st.column_config.NumberColumn("当前窗口低音级数", disabled=True),
                    "melody_pitch_name": st.column_config.TextColumn("旋律音", disabled=True),
                    "melody_scale_degree": st.column_config.NumberColumn("旋律音级", disabled=True),
                    "manual_label": st.column_config.TextColumn("人工标签", width="medium"),
                    "review_notes": st.column_config.TextColumn("备注", width="large"),
                },
                key=f"cadence_editor_{symbolic_analysis_key}",
            )
            cadence_annotations.loc[:, "export"] = cadence_editor["export"].astype(bool).to_numpy()
            cadence_annotations.loc[:, "manual_label"] = cadence_editor["manual_label"].astype(str).to_numpy()
            cadence_annotations.loc[:, "review_notes"] = cadence_editor["review_notes"].astype(str).to_numpy()
            st.session_state[cadence_state_key] = cadence_annotations

    with score_tab_4:
        if not symbolic_result["pitch_class_histogram"].empty:
            st.markdown("**音级类分布**")
            st.bar_chart(
                symbolic_result["pitch_class_histogram"].set_index("pitch_class_label")["count"],
                height=280,
            )
        if not symbolic_result["scale_degree_histogram"].empty:
            st.markdown("**音级分布**")
            degree_chart = symbolic_result["scale_degree_histogram"].copy()
            degree_chart["scale_degree"] = degree_chart["scale_degree"].astype(str)
            st.bar_chart(degree_chart.set_index("scale_degree")["count"], height=240)

        st.markdown("**音高总表（前 300 行）**")
        st.dataframe(symbolic_result["note_table"].head(300), width="stretch", hide_index=True)

    with score_tab_5:
        if symbolic_result["interval_table"].empty:
            st.warning("当前乐谱没有足够的连续旋律音高用于音程分析。")
        else:
            st.markdown("**音程序类分布**")
            st.bar_chart(
                symbolic_result["interval_class_histogram"].set_index("interval_class")["count"],
                height=260,
            )
            if not symbolic_result["directed_interval_histogram"].empty:
                st.markdown("**最常见定向音程**")
                directed_histogram = symbolic_result["directed_interval_histogram"].head(12).set_index("directed_name")["count"]
                st.bar_chart(directed_histogram, height=260)
            st.dataframe(symbolic_result["interval_table"].head(300), width="stretch", hide_index=True)

    with score_tab_6:
        if theme_annotations.empty:
            st.warning("当前窗口长度下没有检测到明显的主题 / 动机再现候选。")
        else:
            theme_editor = st.data_editor(
                theme_annotations[
                    [
                        "export",
                        "relation_type",
                        "manual_relation",
                        "similarity_score",
                        "source_part",
                        "source_measure",
                        "source_excerpt",
                        "match_part",
                        "match_measure",
                        "match_excerpt",
                        "review_notes",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "export": st.column_config.CheckboxColumn("导出"),
                    "relation_type": st.column_config.TextColumn("系统判断", disabled=True),
                    "manual_relation": st.column_config.TextColumn("人工标签"),
                    "similarity_score": st.column_config.NumberColumn("相似度", disabled=True),
                    "source_part": st.column_config.TextColumn("源声部", disabled=True),
                    "source_measure": st.column_config.NumberColumn("源小节", disabled=True),
                    "source_excerpt": st.column_config.TextColumn("源片段", disabled=True, width="large"),
                    "match_part": st.column_config.TextColumn("再现声部", disabled=True),
                    "match_measure": st.column_config.NumberColumn("再现小节", disabled=True),
                    "match_excerpt": st.column_config.TextColumn("再现片段", disabled=True, width="large"),
                    "review_notes": st.column_config.TextColumn("备注", width="large"),
                },
                key=f"theme_editor_{symbolic_analysis_key}",
            )
            theme_annotations.loc[:, "export"] = theme_editor["export"].astype(bool).to_numpy()
            theme_annotations.loc[:, "manual_relation"] = theme_editor["manual_relation"].astype(str).to_numpy()
            theme_annotations.loc[:, "review_notes"] = theme_editor["review_notes"].astype(str).to_numpy()
            st.session_state[theme_state_key] = theme_annotations

    with score_tab_7:
        exported_harmony = harmony_annotations.copy()
        exported_cadence = cadence_annotations.copy()
        exported_theme = theme_annotations.copy()
        if not exported_harmony.empty:
            exported_harmony = exported_harmony.loc[exported_harmony["export"]].copy()
        if not exported_cadence.empty:
            exported_cadence = exported_cadence.loc[exported_cadence["export"]].copy()
        if not exported_theme.empty:
            exported_theme = exported_theme.loc[exported_theme["export"]].copy()

        note_csv = symbolic_result["note_table"].to_csv(index=False).encode("utf-8-sig")
        harmony_csv = exported_harmony.to_csv(index=False).encode("utf-8-sig")
        cadence_csv = exported_cadence.to_csv(index=False).encode("utf-8-sig")
        interval_csv = symbolic_result["interval_table"].to_csv(index=False).encode("utf-8-sig")
        theme_csv = exported_theme.to_csv(index=False).encode("utf-8-sig")
        summary_text = "\n".join(symbolic_result["summary_lines"]).encode("utf-8")
        analysis_json = build_symbolic_export_payload(
            symbolic_result,
            harmony_annotations=exported_harmony,
            cadence_annotations=exported_cadence,
            theme_annotations=exported_theme,
        )
        source_name = str(getattr(score_source, "name", "") or score_source_label or "score_input.musicxml")
        source_suffix = Path(source_name).suffix.lower()
        original_score_mime = {
            ".mxl": "application/vnd.recordare.musicxml",
            ".musicxml": "application/vnd.recordare.musicxml+xml",
            ".xml": "application/xml",
            ".mid": "audio/midi",
            ".midi": "audio/midi",
            ".krn": "text/plain",
            ".kern": "text/plain",
        }.get(source_suffix, "application/octet-stream")
        musicxml_bytes, musicxml_filename, musicxml_mime = _build_musicxml_export_bytes(score_bytes, source_name)

        st.markdown("**分析结果导出**")
        result_export_col_1, result_export_col_2 = st.columns(2, gap="medium")
        with result_export_col_1:
            st.download_button("下载音高总表 CSV", data=note_csv, file_name="score_note_table.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载和声分析 CSV", data=harmony_csv, file_name="harmony_annotations.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载终止候选 CSV", data=cadence_csv, file_name="cadence_candidates.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载音程分析 CSV", data=interval_csv, file_name="interval_table.csv", mime="text/csv", use_container_width=True)
        with result_export_col_2:
            st.download_button("下载主题再现 CSV", data=theme_csv, file_name="theme_matches.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载摘要 TXT", data=summary_text, file_name="symbolic_summary.txt", mime="text/plain", use_container_width=True)
            st.download_button("下载完整分析 JSON", data=analysis_json, file_name="symbolic_analysis.json", mime="application/json", use_container_width=True)

        st.markdown("**乐谱文件导出**")
        st.caption("这一组适合把乐谱文件直接带到其他音频软件、乐谱软件或外部分析环境里继续查看。")
        score_export_col_1, score_export_col_2 = st.columns(2, gap="medium")
        with score_export_col_1:
            if source_suffix in {".mxl", ".musicxml", ".xml", ".krn", ".kern"}:
                st.download_button(
                    "下载原始乐谱文件",
                    data=score_bytes,
                    file_name=Path(source_name).name,
                    mime=original_score_mime,
                    use_container_width=True,
                )
        with score_export_col_2:
            st.download_button(
                "下载 MusicXML 乐谱",
                data=musicxml_bytes,
                file_name=musicxml_filename,
                mime=musicxml_mime,
                use_container_width=True,
            )

    st.markdown(
        """
提示：

- 如果主题再现候选过少，可以缩短“主题检索窗口”
- 如果候选过多，可以增大窗口长度，或者先把焦点放到一个主旋律声部
- Roman numeral 目前是第一阶段候选，并不等于最终和声结论
- 你在“和声”和“主题 / 动机”页中的人工修订会参与导出
"""
    )
    st.stop()

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
            "band_energy_ratio",
            "high_band_ratio",
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
