import csv
import json
import tempfile
import unittest
from pathlib import Path

from apply_observable_pair_gate import compile_gate
from sweep_observable_multi_source_gate import SourceSpec, sweep_multi_source_rules


def _row(
    pair_index,
    matches,
    margin,
    pfm_correct,
    pfm_wrong,
    lightglue_correct,
    lightglue_wrong,
):
    return {
        "source_name": "unit",
        "split": "eval",
        "pair_index": str(pair_index),
        "pair_type": "same_position_view",
        "base_id": f"b{pair_index:03d}",
        "reference_variant": "nadir",
        "target_variant": "mid_01",
        "pfm_matches": str(pfm_correct + pfm_wrong),
        "pfm_correct": str(pfm_correct),
        "pfm_wrong": str(pfm_wrong),
        "pfm_precision": "1.0",
        "lightglue_matches": str(lightglue_correct + lightglue_wrong),
        "lightglue_correct": str(lightglue_correct),
        "lightglue_wrong": str(lightglue_wrong),
        "lightglue_precision": "1.0",
        "reject_label": "1",
        "teacher_correct_delta": str(lightglue_correct - pfm_correct),
        "teacher_wrong_delta": str(lightglue_wrong - pfm_wrong),
        "feature_matches": str(matches),
        "feature_detail_raw_margin_median": str(margin),
    }


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class ObservableMultiSourceGateSweepTest(unittest.TestCase):
    def test_or_of_and_gate_beats_each_source_without_leaking_truth_fields(self):
        source_a = [
            _row(0, 610, 0.20, 40, 0, 10, 0),
            _row(1, 300, 0.90, 41, 0, 11, 0),
            _row(2, 610, 0.90, 50, 6, 12, 0),
            _row(3, 300, 0.20, 51, 6, 13, 0),
        ]
        source_b = [
            _row(0, 620, 0.20, 35, 0, 9, 0),
            _row(1, 310, 0.90, 36, 0, 10, 0),
            _row(2, 620, 0.90, 45, 7, 11, 0),
            _row(3, 310, 0.20, 46, 7, 12, 0),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_a = root / "a.csv"
            csv_b = root / "b.csv"
            output_dir = root / "out"
            _write_csv(csv_a, source_a)
            _write_csv(csv_b, source_b)

            summary = sweep_multi_source_rules(
                sources=[
                    SourceSpec(name="source_a", dataset_csv=csv_a),
                    SourceSpec(name="source_b", dataset_csv=csv_b),
                ],
                output_dir=output_dir,
                max_thresholds=4,
                beam_width=16,
                max_clauses=2,
                allow_wrong_delta=0,
                min_correct_delta=1,
            )

            best = summary["best_valid"]
            self.assertIsNotNone(best)
            gate = best["gate"]
            self.assertIn(" OR ", gate)
            self.assertIn(" AND ", gate)
            self.assertNotIn("correct", gate)
            self.assertNotIn("wrong", gate)
            self.assertNotIn("label", gate)
            gate_fn = compile_gate(gate)
            self.assertTrue(gate_fn(source_a[0]))
            self.assertTrue(gate_fn(source_a[1]))
            self.assertFalse(gate_fn(source_a[2]))
            self.assertFalse(gate_fn(source_a[3]))
            self.assertGreaterEqual(best["source_a_correct_delta_vs_lightglue"], 1)
            self.assertLessEqual(best["source_a_wrong_delta_vs_lightglue"], 0)
            self.assertGreaterEqual(best["source_b_correct_delta_vs_lightglue"], 1)
            self.assertLessEqual(best["source_b_wrong_delta_vs_lightglue"], 0)
            self.assertTrue((output_dir / "best_gates.csv").exists())
            saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["best_valid"]["gate"], gate)


if __name__ == "__main__":
    unittest.main()
