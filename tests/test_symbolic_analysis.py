from __future__ import annotations

import unittest

from music21 import chord, key, meter, note, stream

from spectral_tool.symbolic_analysis import SymbolicAnalysisConfig, analyze_symbolic_score, export_symbolic_score_file


def _build_exact_recurrence_score() -> stream.Score:
    score = stream.Score(id="ExactRecurrence")
    melody = stream.Part(id="Melody")
    melody.partName = "Melody"
    melody.insert(0, key.Key("C"))
    melody.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    for pitch_name in ["C4", "D4", "E4", "G4"]:
        measure_1.append(note.Note(pitch_name, quarterLength=1.0))

    measure_2 = stream.Measure(number=2)
    for pitch_name in ["C4", "D4", "E4", "G4"]:
        measure_2.append(note.Note(pitch_name, quarterLength=1.0))

    melody.append(measure_1)
    melody.append(measure_2)
    score.append(melody)
    return score


def _build_transposed_recurrence_score() -> stream.Score:
    score = stream.Score(id="TransposedRecurrence")
    melody = stream.Part(id="Melody")
    melody.partName = "Melody"
    melody.insert(0, key.Key("G"))
    melody.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    for pitch_name in ["G4", "A4", "B4", "D5"]:
        measure_1.append(note.Note(pitch_name, quarterLength=1.0))

    measure_2 = stream.Measure(number=2)
    for pitch_name in ["A4", "B4", "C#5", "E5"]:
        measure_2.append(note.Note(pitch_name, quarterLength=1.0))

    melody.append(measure_1)
    melody.append(measure_2)
    score.append(melody)
    return score


def _build_harmony_score() -> stream.Score:
    score = stream.Score(id="HarmonyScore")
    part = stream.Part(id="Harmony")
    part.partName = "Harmony"
    part.insert(0, key.Key("C"))
    part.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    measure_1.append(chord.Chord(["C4", "E4", "G4"], quarterLength=2.0))
    measure_1.append(chord.Chord(["G3", "B3", "D4"], quarterLength=2.0))

    measure_2 = stream.Measure(number=2)
    measure_2.append(chord.Chord(["F3", "A3", "C4"], quarterLength=2.0))
    measure_2.append(chord.Chord(["G3", "B3", "D4"], quarterLength=2.0))

    part.append(measure_1)
    part.append(measure_2)
    score.append(part)
    return score


def _build_empty_chord_score() -> stream.Score:
    score = stream.Score(id="EmptyChordScore")
    part = stream.Part(id="SparseHarmony")
    part.partName = "SparseHarmony"
    part.insert(0, key.Key("C"))
    part.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    measure_1.append(chord.Chord([], quarterLength=1.0))
    measure_1.append(note.Note("C4", quarterLength=1.0))
    part.append(measure_1)
    score.append(part)
    return score


class SymbolicAnalysisTestCase(unittest.TestCase):
    def test_exact_recurrence_score_extracts_pitch_statistics(self) -> None:
        result = analyze_symbolic_score(
            _build_exact_recurrence_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=4, max_recurrence_results=8),
        )

        self.assertEqual(result["global_key"], "C 大调")
        self.assertEqual(result["total_notes"], 8)
        self.assertEqual(result["unique_pitch_classes"], 4)
        self.assertEqual(len(result["pitch_class_histogram"]), 4)
        self.assertIn("measure_pitch_summary", result)
        self.assertFalse(result["interval_class_histogram"].empty)
        self.assertTrue((result["pitch_class_histogram"]["count"] == 2).all())

    def test_theme_recurrence_detects_exact_and_transposed_matches(self) -> None:
        exact_result = analyze_symbolic_score(
            _build_exact_recurrence_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=4, max_recurrence_results=8),
        )
        transposed_result = analyze_symbolic_score(
            _build_transposed_recurrence_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=4, max_recurrence_results=8),
        )

        self.assertIn("精确再现", set(exact_result["theme_matches"]["relation_type"]))
        self.assertIn("移调再现", set(transposed_result["theme_matches"]["relation_type"]))

    def test_harmony_table_includes_roman_numeral_candidates(self) -> None:
        result = analyze_symbolic_score(_build_harmony_score())
        roman_candidates = set(result["harmony_table"]["roman_numeral"].astype(str))

        self.assertIn("I", roman_candidates)
        self.assertIn("V", roman_candidates)
        self.assertIn("IV", roman_candidates)

    def test_empty_chord_events_are_skipped_safely(self) -> None:
        result = analyze_symbolic_score(_build_empty_chord_score())

        self.assertEqual(result["total_notes"], 1)
        self.assertEqual(result["unique_pitch_classes"], 1)
        self.assertFalse(result["note_table"].empty)

    def test_export_symbolic_score_file_builds_musicxml_bytes(self) -> None:
        data, filename, mime = export_symbolic_score_file(_build_harmony_score(), fmt="musicxml")

        self.assertTrue(data.startswith(b"<?xml") or b"<score-partwise" in data[:200])
        self.assertTrue(filename.endswith(".musicxml"))
        self.assertEqual(mime, "application/vnd.recordare.musicxml+xml")


if __name__ == "__main__":
    unittest.main()
