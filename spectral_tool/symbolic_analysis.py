from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from music21 import chord as m21chord
from music21 import converter
from music21 import interval as m21interval
from music21 import key as m21key
from music21 import note as m21note
from music21 import pitch as m21pitch
from music21 import roman as m21roman
from music21 import stream as m21stream


@dataclass(slots=True)
class SymbolicAnalysisConfig:
    theme_window_notes: int = 6
    max_recurrence_results: int = 16
    measure_summary_top_n: int = 3

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def format_key_label(key_object: m21key.Key | None) -> str:
    if key_object is None:
        return "未能稳定判断"

    mode_map = {
        "major": "大调",
        "minor": "小调",
        "dorian": "多利亚调式",
        "phrygian": "弗里几亚调式",
        "lydian": "利底亚调式",
        "mixolydian": "混合利底亚调式",
        "aeolian": "爱奥利亚调式",
        "locrian": "洛克里亚调式",
    }
    mode_label = mode_map.get(str(key_object.mode), str(key_object.mode))
    return f"{key_object.tonic.name} {mode_label}"


def analyze_symbolic_score(
    source: str | Path | BinaryIO | m21stream.Score,
    config: SymbolicAnalysisConfig | None = None,
) -> dict[str, Any]:
    active_config = config or SymbolicAnalysisConfig()
    score, source_name = _load_symbolic_score(source)
    parts = list(score.parts) or [score]
    global_key = _detect_global_key(score)

    note_table, melodic_sequences, part_summary = _build_note_table(parts, global_key)
    pitch_class_histogram = _build_pitch_class_histogram(note_table)
    pitch_height_histogram = _build_pitch_height_histogram(note_table)
    scale_degree_histogram = _build_scale_degree_histogram(note_table)
    measure_pitch_summary = _build_measure_pitch_summary(note_table, active_config.measure_summary_top_n)
    interval_table, interval_class_histogram, directed_interval_histogram = _build_interval_tables(melodic_sequences)
    harmony_table = _build_harmony_table(score, global_key)
    theme_matches = _build_theme_matches(
        melodic_sequences=melodic_sequences,
        window_size=active_config.theme_window_notes,
        max_results=active_config.max_recurrence_results,
    )

    total_notes = int(len(note_table))
    unique_pitches = int(note_table["midi"].nunique()) if not note_table.empty else 0
    unique_pitch_classes = int(note_table["pitch_class"].nunique()) if not note_table.empty else 0

    summary_lines = _build_summary_lines(
        global_key=global_key,
        total_notes=total_notes,
        unique_pitches=unique_pitches,
        unique_pitch_classes=unique_pitch_classes,
        pitch_class_histogram=pitch_class_histogram,
        interval_class_histogram=interval_class_histogram,
        harmony_table=harmony_table,
        theme_matches=theme_matches,
        theme_window_notes=active_config.theme_window_notes,
    )

    return {
        "source_name": source_name,
        "score_title": _resolve_score_title(score, source_name),
        "config": active_config.to_dict(),
        "global_key": format_key_label(global_key),
        "global_key_object": global_key,
        "part_summary": part_summary,
        "note_table": note_table,
        "pitch_class_histogram": pitch_class_histogram,
        "pitch_height_histogram": pitch_height_histogram,
        "scale_degree_histogram": scale_degree_histogram,
        "measure_pitch_summary": measure_pitch_summary,
        "interval_table": interval_table,
        "interval_class_histogram": interval_class_histogram,
        "directed_interval_histogram": directed_interval_histogram,
        "harmony_table": harmony_table,
        "theme_matches": theme_matches,
        "summary_lines": summary_lines,
        "total_notes": total_notes,
        "unique_pitches": unique_pitches,
        "unique_pitch_classes": unique_pitch_classes,
    }


def _resolve_source_name(source: str | Path | BinaryIO | m21stream.Score) -> str:
    if isinstance(source, Path):
        return source.name
    if isinstance(source, str):
        return Path(source).name
    if isinstance(source, m21stream.Score):
        metadata_title = getattr(getattr(source, "metadata", None), "title", None)
        return str(metadata_title or "score_input")
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return Path(name).name
    return "score_input"


def _write_temp_symbolic_file(source: BinaryIO) -> str:
    source.seek(0)
    data = source.read()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("乐谱输入必须是路径、music21 Score 或字节流。")

    suffix = Path(_resolve_source_name(source)).suffix or ".xml"
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(data)
        temporary.flush()
    finally:
        temporary.close()
    return temporary.name


def _load_symbolic_score(
    source: str | Path | BinaryIO | m21stream.Score,
) -> tuple[m21stream.Score, str]:
    source_name = _resolve_source_name(source)
    temp_path: str | None = None

    try:
        if isinstance(source, m21stream.Score):
            parsed = source
        elif isinstance(source, (str, Path)):
            parsed = converter.parse(str(source))
        else:
            temp_path = _write_temp_symbolic_file(source)
            parsed = converter.parse(temp_path)
    except Exception as exc:
        raise RuntimeError(
            "无法读取该乐谱文件。请尝试 MusicXML、MXL、MIDI 或标准 XML 乐谱文件。"
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if isinstance(parsed, m21stream.Score):
        score = parsed
    else:
        score = m21stream.Score(id="ImportedScore")
        score.insert(0, parsed)

    return score, source_name


def _detect_global_key(score: m21stream.Score) -> m21key.Key | None:
    explicit_keys = list(score.recurse().getElementsByClass(m21key.Key))
    if explicit_keys:
        return explicit_keys[0]

    try:
        analyzed = score.analyze("key")
        if isinstance(analyzed, m21key.Key):
            return analyzed
    except Exception:
        return None
    return None


def _resolve_score_title(score: m21stream.Score, source_name: str) -> str:
    metadata = getattr(score, "metadata", None)
    if metadata is not None:
        title = getattr(metadata, "title", None)
        movement_name = getattr(metadata, "movementName", None)
        if title:
            return str(title)
        if movement_name:
            return str(movement_name)
    return source_name


def _pitch_from_midi(midi_value: int) -> m21pitch.Pitch:
    pitch_object = m21pitch.Pitch()
    pitch_object.midi = int(midi_value)
    return pitch_object


def _pitch_class_name(pitch_class: int) -> str:
    pitch_object = _pitch_from_midi(60 + int(pitch_class))
    return pitch_object.name


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_scale_degree(key_object: m21key.Key | None, pitch_object: m21pitch.Pitch) -> int | None:
    if key_object is None:
        return None
    try:
        return key_object.getScaleDegreeFromPitch(pitch_object)
    except Exception:
        return None


def _build_note_table(
    parts: list[m21stream.Stream],
    global_key: m21key.Key | None,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], pd.DataFrame]:
    note_rows: list[dict[str, Any]] = []
    melodic_sequences: dict[str, list[dict[str, Any]]] = {}
    part_rows: list[dict[str, Any]] = []

    for part_index, part in enumerate(parts, start=1):
        part_name = str(part.partName or part.id or f"Part {part_index}")
        flattened = part.flatten().notes
        onset_index = 0
        melodic_events: list[dict[str, Any]] = []

        for element in flattened:
            onset_index += 1
            measure_number = _safe_int(getattr(element, "measureNumber", 0), 0)
            beat = round(_safe_float(getattr(element, "beat", 0.0), 0.0), 3)
            offset_ql = round(_safe_float(getattr(element, "offset", 0.0), 0.0), 3)
            quarter_length = round(_safe_float(getattr(element, "quarterLength", 0.0), 0.0), 3)

            if isinstance(element, m21chord.Chord):
                event_pitches = [pitch_value for pitch_value in element.pitches if getattr(pitch_value, "midi", None) is not None]
                if not event_pitches:
                    continue
                melodic_pitch = max(event_pitches, key=lambda pitch_value: pitch_value.midi)
                is_chord_event = True
            elif isinstance(element, m21note.Note):
                if getattr(element.pitch, "midi", None) is None:
                    continue
                event_pitches = [element.pitch]
                melodic_pitch = element.pitch
                is_chord_event = False
            else:
                continue

            melodic_events.append(
                {
                    "part_name": part_name,
                    "onset_index": onset_index,
                    "measure_number": measure_number,
                    "beat": beat,
                    "offset_ql": offset_ql,
                    "quarter_length": quarter_length,
                    "midi": int(melodic_pitch.midi),
                    "pitch_name": melodic_pitch.nameWithOctave,
                    "pitch_class": int(melodic_pitch.pitchClass),
                    "scale_degree": _safe_scale_degree(global_key, melodic_pitch),
                }
            )

            chord_cardinality = len(event_pitches)
            for chord_tone_index, pitch_object in enumerate(event_pitches, start=1):
                note_rows.append(
                    {
                        "part_name": part_name,
                        "onset_index": onset_index,
                        "measure_number": measure_number,
                        "beat": beat,
                        "offset_ql": offset_ql,
                        "quarter_length": quarter_length,
                        "pitch_name": pitch_object.nameWithOctave,
                        "pitch_class_label": _pitch_class_name(int(pitch_object.pitchClass)),
                        "midi": int(pitch_object.midi),
                        "pitch_class": int(pitch_object.pitchClass),
                        "octave": _safe_int(pitch_object.octave, 0),
                        "scale_degree": _safe_scale_degree(global_key, pitch_object),
                        "scale_degree_label": (
                            str(_safe_scale_degree(global_key, pitch_object))
                            if _safe_scale_degree(global_key, pitch_object) is not None
                            else ""
                        ),
                        "is_chord_event": is_chord_event,
                        "chord_cardinality": chord_cardinality,
                        "chord_tone_index": chord_tone_index,
                    }
                )

        melodic_sequences[part_name] = melodic_events
        part_rows.append(
            {
                "part_name": part_name,
                "event_count": len(melodic_events),
                "note_count": len([row for row in note_rows if row["part_name"] == part_name]),
            }
        )

    note_table = pd.DataFrame(note_rows)
    part_summary = pd.DataFrame(part_rows)
    if not note_table.empty:
        note_table = note_table.sort_values(["measure_number", "offset_ql", "part_name", "midi"]).reset_index(drop=True)
    return note_table, melodic_sequences, part_summary


def _build_pitch_class_histogram(note_table: pd.DataFrame) -> pd.DataFrame:
    if note_table.empty:
        return pd.DataFrame(columns=["pitch_class", "pitch_class_label", "count", "ratio"])

    histogram = (
        note_table.groupby(["pitch_class", "pitch_class_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["pitch_class"])
        .reset_index(drop=True)
    )
    histogram["ratio"] = histogram["count"] / float(histogram["count"].sum())
    return histogram


def _build_pitch_height_histogram(note_table: pd.DataFrame) -> pd.DataFrame:
    if note_table.empty:
        return pd.DataFrame(columns=["midi", "pitch_name", "count"])

    histogram = (
        note_table.groupby(["midi", "pitch_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["midi", "pitch_name"])
        .reset_index(drop=True)
    )
    return histogram


def _build_scale_degree_histogram(note_table: pd.DataFrame) -> pd.DataFrame:
    if note_table.empty or "scale_degree" not in note_table.columns:
        return pd.DataFrame(columns=["scale_degree", "count", "ratio"])

    filtered = note_table.dropna(subset=["scale_degree"]).copy()
    if filtered.empty:
        return pd.DataFrame(columns=["scale_degree", "count", "ratio"])

    filtered["scale_degree"] = filtered["scale_degree"].astype(int)
    histogram = (
        filtered.groupby("scale_degree", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("scale_degree")
        .reset_index(drop=True)
    )
    histogram["ratio"] = histogram["count"] / float(histogram["count"].sum())
    return histogram


def _top_counts(series: pd.Series, top_n: int) -> str:
    valid = series.astype(str)
    valid = valid[valid != ""]
    if valid.empty:
        return ""
    counts = valid.value_counts().head(top_n)
    return " / ".join(f"{label}×{count}" for label, count in counts.items())


def _build_measure_pitch_summary(note_table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if note_table.empty:
        return pd.DataFrame(
            columns=[
                "measure_number",
                "note_count",
                "unique_pitch_count",
                "unique_pitch_classes",
                "lowest_pitch",
                "highest_pitch",
                "top_pitch_classes",
                "top_scale_degrees",
            ]
        )

    rows: list[dict[str, Any]] = []
    for measure_number, frame in note_table.groupby("measure_number", sort=True):
        low_midi = int(frame["midi"].min())
        high_midi = int(frame["midi"].max())
        rows.append(
            {
                "measure_number": int(measure_number),
                "note_count": int(len(frame)),
                "unique_pitch_count": int(frame["midi"].nunique()),
                "unique_pitch_classes": int(frame["pitch_class"].nunique()),
                "lowest_pitch": _pitch_from_midi(low_midi).nameWithOctave,
                "highest_pitch": _pitch_from_midi(high_midi).nameWithOctave,
                "top_pitch_classes": _top_counts(frame["pitch_class_label"], top_n),
                "top_scale_degrees": _top_counts(frame["scale_degree_label"], top_n),
            }
        )
    return pd.DataFrame(rows)


def _interval_class(semitones: int) -> int:
    absolute = abs(int(semitones)) % 12
    return min(absolute, 12 - absolute)


def _build_interval_tables(
    melodic_sequences: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for part_name, events in melodic_sequences.items():
        if len(events) < 2:
            continue
        for previous, current in zip(events[:-1], events[1:]):
            previous_pitch = _pitch_from_midi(int(previous["midi"]))
            current_pitch = _pitch_from_midi(int(current["midi"]))
            interval_object = m21interval.Interval(previous_pitch, current_pitch)
            semitones = int(current["midi"]) - int(previous["midi"])
            contour = "上行" if semitones > 0 else "下行" if semitones < 0 else "同音重复"
            rows.append(
                {
                    "part_name": part_name,
                    "from_measure": int(previous["measure_number"]),
                    "to_measure": int(current["measure_number"]),
                    "from_pitch": str(previous["pitch_name"]),
                    "to_pitch": str(current["pitch_name"]),
                    "semitones": semitones,
                    "interval_class": _interval_class(semitones),
                    "simple_name": interval_object.simpleName,
                    "directed_name": interval_object.directedName,
                    "contour": contour,
                }
            )

    interval_table = pd.DataFrame(rows)
    if interval_table.empty:
        empty_ic = pd.DataFrame(columns=["interval_class", "count", "ratio"])
        empty_dir = pd.DataFrame(columns=["directed_name", "count", "ratio"])
        return interval_table, empty_ic, empty_dir

    interval_class_histogram = (
        interval_table.groupby("interval_class", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("interval_class")
        .reset_index(drop=True)
    )
    interval_class_histogram["ratio"] = interval_class_histogram["count"] / float(interval_class_histogram["count"].sum())

    directed_interval_histogram = (
        interval_table.groupby("directed_name", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "directed_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    directed_interval_histogram["ratio"] = directed_interval_histogram["count"] / float(
        directed_interval_histogram["count"].sum()
    )
    return interval_table, interval_class_histogram, directed_interval_histogram


def _safe_root_pitch(chord_object: m21chord.Chord) -> m21pitch.Pitch | None:
    try:
        return chord_object.root()
    except Exception:
        return None


def _build_harmony_table(
    score: m21stream.Score,
    global_key: m21key.Key | None,
) -> pd.DataFrame:
    try:
        harmonic_stream = score.chordify()
    except Exception:
        return pd.DataFrame(
            columns=[
                "slice_id",
                "measure_number",
                "beat",
                "quarter_length",
                "pitch_names",
                "pitch_class_set",
                "root",
                "root_scale_degree",
                "quality",
                "roman_numeral",
            ]
        )

    rows: list[dict[str, Any]] = []
    slice_id = 0
    for chord_object in harmonic_stream.flatten().getElementsByClass(m21chord.Chord):
        if len(chord_object.pitches) == 0:
            continue

        slice_id += 1
        unique_pitch_classes = sorted({int(pitch_value.pitchClass) for pitch_value in chord_object.pitches})
        root_pitch = _safe_root_pitch(chord_object)
        roman_numeral = ""
        if global_key is not None and len(chord_object.pitches) >= 2:
            try:
                roman_numeral = str(m21roman.romanNumeralFromChord(chord_object, global_key).figure)
            except Exception:
                roman_numeral = ""

        rows.append(
            {
                "slice_id": slice_id,
                "measure_number": _safe_int(getattr(chord_object, "measureNumber", 0), 0),
                "beat": round(_safe_float(getattr(chord_object, "beat", 0.0), 0.0), 3),
                "quarter_length": round(_safe_float(getattr(chord_object, "quarterLength", 0.0), 0.0), 3),
                "pitch_names": " ".join(pitch_value.nameWithOctave for pitch_value in chord_object.pitches),
                "pitch_class_set": "{" + ", ".join(_pitch_class_name(pitch_class) for pitch_class in unique_pitch_classes) + "}",
                "root": root_pitch.name if root_pitch is not None else "",
                "root_scale_degree": (
                    _safe_scale_degree(global_key, root_pitch) if root_pitch is not None else None
                ),
                "quality": str(getattr(chord_object, "quality", "") or ""),
                "roman_numeral": roman_numeral,
            }
        )

    harmony_table = pd.DataFrame(rows)
    if harmony_table.empty:
        return harmony_table
    return harmony_table.sort_values(["measure_number", "beat", "slice_id"]).reset_index(drop=True)


def _rhythm_signature(durations: list[float]) -> tuple[float, ...]:
    total = float(sum(durations))
    if total <= 1e-8:
        return tuple(0.0 for _ in durations)
    return tuple(round(float(duration) / total, 3) for duration in durations)


def _window_excerpt(events: list[dict[str, Any]]) -> str:
    return " ".join(str(event["pitch_name"]) for event in events)


def _window_occurrences(
    melodic_sequences: dict[str, list[dict[str, Any]]],
    window_size: int,
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for part_name, events in melodic_sequences.items():
        if len(events) < window_size:
            continue
        for start_index in range(0, len(events) - window_size + 1):
            window = events[start_index : start_index + window_size]
            pitches = [int(event["midi"]) for event in window]
            durations = [float(event["quarter_length"]) for event in window]
            interval_signature = tuple(pitches[index + 1] - pitches[index] for index in range(len(pitches) - 1))
            contour_signature = tuple(int(np.sign(value)) for value in interval_signature)
            occurrences.append(
                {
                    "part_name": part_name,
                    "start_index": start_index,
                    "measure_number": int(window[0]["measure_number"]),
                    "beat": float(window[0]["beat"]),
                    "pitch_signature": tuple(pitches),
                    "interval_signature": interval_signature,
                    "contour_signature": contour_signature,
                    "duration_signature": tuple(round(duration, 3) for duration in durations),
                    "rhythm_signature": _rhythm_signature(durations),
                    "first_pitch": pitches[0],
                    "excerpt": _window_excerpt(window),
                }
            )
    return occurrences


def _non_overlapping(
    left: dict[str, Any],
    right: dict[str, Any],
    window_size: int,
) -> bool:
    if left["part_name"] != right["part_name"]:
        return True
    return abs(int(left["start_index"]) - int(right["start_index"])) >= window_size


def _dedupe_occurrences(occurrences: list[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for occurrence in sorted(occurrences, key=lambda item: (str(item["part_name"]), int(item["start_index"]))):
        if all(_non_overlapping(occurrence, existing, window_size) for existing in kept):
            kept.append(occurrence)
    return kept


def _build_theme_matches(
    melodic_sequences: dict[str, list[dict[str, Any]]],
    window_size: int,
    max_results: int,
) -> pd.DataFrame:
    if window_size < 3:
        window_size = 3

    occurrences = _window_occurrences(melodic_sequences, window_size)
    if not occurrences:
        return pd.DataFrame(
            columns=[
                "relation_type",
                "similarity_score",
                "source_part",
                "source_measure",
                "source_beat",
                "source_excerpt",
                "match_part",
                "match_measure",
                "match_beat",
                "match_excerpt",
                "transposition_semitones",
                "window_size",
                "match_detail",
            ]
        )

    exact_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    transposed_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    contour_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for occurrence in occurrences:
        exact_key = (occurrence["pitch_signature"], occurrence["duration_signature"])
        transposed_key = (occurrence["interval_signature"], occurrence["rhythm_signature"])
        contour_key = (occurrence["contour_signature"], occurrence["rhythm_signature"])
        exact_groups.setdefault(exact_key, []).append(occurrence)
        transposed_groups.setdefault(transposed_key, []).append(occurrence)
        contour_groups.setdefault(contour_key, []).append(occurrence)

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()

    def add_group(
        relation_type: str,
        groups: dict[tuple[Any, ...], list[dict[str, Any]]],
        similarity_score: float,
        match_detail: str,
    ) -> None:
        for group_occurrences in groups.values():
            deduped = _dedupe_occurrences(group_occurrences, window_size)
            if len(deduped) < 2:
                continue
            anchor = deduped[0]
            for match in deduped[1:]:
                pair_key = tuple(
                    sorted(
                        [
                            (str(anchor["part_name"]), int(anchor["start_index"])),
                            (str(match["part_name"]), int(match["start_index"])),
                        ]
                    )
                )
                if pair_key in seen_pairs:
                    continue
                if relation_type == "移调再现" and anchor["pitch_signature"] == match["pitch_signature"]:
                    continue
                if relation_type == "轮廓相似" and anchor["interval_signature"] == match["interval_signature"]:
                    continue

                seen_pairs.add(pair_key)
                rows.append(
                    {
                        "relation_type": relation_type,
                        "similarity_score": similarity_score,
                        "source_part": str(anchor["part_name"]),
                        "source_measure": int(anchor["measure_number"]),
                        "source_beat": round(float(anchor["beat"]), 3),
                        "source_excerpt": str(anchor["excerpt"]),
                        "match_part": str(match["part_name"]),
                        "match_measure": int(match["measure_number"]),
                        "match_beat": round(float(match["beat"]), 3),
                        "match_excerpt": str(match["excerpt"]),
                        "transposition_semitones": int(match["first_pitch"]) - int(anchor["first_pitch"]),
                        "window_size": window_size,
                        "match_detail": match_detail,
                    }
                )
                if len(rows) >= max_results:
                    return

    add_group("精确再现", exact_groups, 1.00, "音高与节奏窗口完全一致")
    if len(rows) < max_results:
        add_group("移调再现", transposed_groups, 0.90, "音程骨架与节奏比例一致，但起始音高不同")
    if len(rows) < max_results:
        add_group("轮廓相似", contour_groups, 0.72, "上下行轮廓与节奏比例一致，但具体音程发生变化")

    if not rows:
        return pd.DataFrame(
            columns=[
                "relation_type",
                "similarity_score",
                "source_part",
                "source_measure",
                "source_beat",
                "source_excerpt",
                "match_part",
                "match_measure",
                "match_beat",
                "match_excerpt",
                "transposition_semitones",
                "window_size",
                "match_detail",
            ]
        )

    theme_matches = pd.DataFrame(rows)
    theme_matches = theme_matches.sort_values(
        ["similarity_score", "source_measure", "match_measure"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return theme_matches.head(max_results)


def _top_histogram_statement(
    histogram: pd.DataFrame,
    label_column: str,
    count_column: str = "count",
    limit: int = 3,
) -> str:
    if histogram.empty:
        return "暂无"
    top = histogram.sort_values(count_column, ascending=False).head(limit)
    return " / ".join(f"{row[label_column]}×{int(row[count_column])}" for _, row in top.iterrows())


def _build_summary_lines(
    global_key: m21key.Key | None,
    total_notes: int,
    unique_pitches: int,
    unique_pitch_classes: int,
    pitch_class_histogram: pd.DataFrame,
    interval_class_histogram: pd.DataFrame,
    harmony_table: pd.DataFrame,
    theme_matches: pd.DataFrame,
    theme_window_notes: int,
) -> list[str]:
    roman_count = 0
    if not harmony_table.empty and "roman_numeral" in harmony_table.columns:
        roman_count = int(harmony_table["roman_numeral"].astype(str).str.strip().ne("").sum())

    return [
        f"全局调性 / 调式估计：{format_key_label(global_key)}",
        f"共提取 {total_notes} 个音高事件，包含 {unique_pitches} 个不同音高、{unique_pitch_classes} 个不同音级类。",
        f"音级类重心：{_top_histogram_statement(pitch_class_histogram, 'pitch_class_label')}",
        f"主导音程序类：{_top_histogram_statement(interval_class_histogram, 'interval_class')}",
        f"共生成 {len(harmony_table)} 个和声切片，其中 {roman_count} 个切片得到了 Roman numeral 候选。",
        (
            f"基于连续 {theme_window_notes} 音窗口，共找到 {len(theme_matches)} 条主题 / 动机再现候选。"
            "当前结果属于第一阶段辅助匹配，需要研究者继续复核。"
        ),
    ]


def build_symbolic_export_payload(
    result: dict[str, Any],
    harmony_annotations: pd.DataFrame | None = None,
    theme_annotations: pd.DataFrame | None = None,
) -> bytes:
    payload = {
        "source_name": result["source_name"],
        "score_title": result["score_title"],
        "config": result["config"],
        "global_key": result["global_key"],
        "summary_lines": result["summary_lines"],
        "part_summary": result["part_summary"].to_dict(orient="records"),
        "note_table": result["note_table"].to_dict(orient="records"),
        "pitch_class_histogram": result["pitch_class_histogram"].to_dict(orient="records"),
        "pitch_height_histogram": result["pitch_height_histogram"].to_dict(orient="records"),
        "scale_degree_histogram": result["scale_degree_histogram"].to_dict(orient="records"),
        "measure_pitch_summary": result["measure_pitch_summary"].to_dict(orient="records"),
        "interval_table": result["interval_table"].to_dict(orient="records"),
        "interval_class_histogram": result["interval_class_histogram"].to_dict(orient="records"),
        "directed_interval_histogram": result["directed_interval_histogram"].to_dict(orient="records"),
        "harmony_table": (harmony_annotations if harmony_annotations is not None else result["harmony_table"]).to_dict(
            orient="records"
        ),
        "theme_matches": (theme_annotations if theme_annotations is not None else result["theme_matches"]).to_dict(
            orient="records"
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_symbolic_score_file(
    source: str | Path | BinaryIO | m21stream.Score,
    fmt: str = "musicxml",
) -> tuple[bytes, str, str]:
    score, source_name = _load_symbolic_score(source)
    stem = Path(source_name).stem or "score_input"

    suffix_map = {
        "musicxml": ".musicxml",
        "xml": ".xml",
        "mxl": ".mxl",
    }
    mime_map = {
        "musicxml": "application/vnd.recordare.musicxml+xml",
        "xml": "application/xml",
        "mxl": "application/vnd.recordare.musicxml",
    }
    suffix = suffix_map.get(fmt, ".musicxml")
    mime_type = mime_map.get(fmt, "application/octet-stream")

    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temporary_path = temporary.name
    temporary.close()
    try:
        score.write(fmt, fp=temporary_path)
        data = Path(temporary_path).read_bytes()
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return data, f"{stem}{suffix}", mime_type
