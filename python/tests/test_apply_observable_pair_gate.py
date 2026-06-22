import csv
import json
import tempfile
import unittest
from pathlib import Path

from apply_observable_pair_gate import apply_observable_gate, build_hybrid_rows, compile_gate, summarize_rows


def _row(pair_index, feature_value, variant, pfm_correct, pfm_wrong, lightglue_correct, lightglue_wrong):
    return {
        "source_name": "unit",
        "split": "lockbox",
        "pair_index": str(pair_index),
        "pair_type": "same_position_view",
        "base_id": f"b{pair_index:03d}",
        "reference_variant": "nadir",
        "target_variant": variant,
        "pfm_matches": str(pfm_correct + pfm_wrong),
        "pfm_correct": str(pfm_correct),
        "pfm_wrong": str(pfm_wrong),
        "pfm_precision": "1.0",
        "lightglue_matches": str(lightglue_correct + lightglue_wrong),
        "lightglue_correct": str(lightglue_correct),
        "lightglue_wrong": str(lightglue_wrong),
        "lightglue_precision": "1.0",
        "feature_x": str(feature_value),
    }


class ApplyObservablePairGateTest(unittest.TestCase):
    def test_compile_gate_supports_numeric_and_variant_conditions(self):
        gate = compile_gate("feature_x <= 3 AND target_variant == mid_01")

        self.assertTrue(gate(_row(0, 2, "mid_01", 20, 0, 10, 0)))
        self.assertFalse(gate(_row(1, 4, "mid_01", 20, 0, 10, 0)))
        self.assertFalse(gate(_row(2, 2, "extreme_03", 20, 0, 10, 0)))

    def test_compile_gate_supports_or_of_and_clauses(self):
        gate = compile_gate(
            "feature_x <= 3 AND target_variant == mid_01 OR "
            "feature_x >= 8 AND target_variant == extreme_03"
        )

        self.assertTrue(gate(_row(0, 2, "mid_01", 20, 0, 10, 0)))
        self.assertTrue(gate(_row(1, 9, "extreme_03", 20, 0, 10, 0)))
        self.assertFalse(gate(_row(2, 9, "mid_01", 20, 0, 10, 0)))
        self.assertFalse(gate(_row(3, 2, "extreme_03", 20, 0, 10, 0)))

    def test_build_hybrid_rows_uses_pfm_for_selected_rows_and_lightglue_for_rejected_rows(self):
        rows = [
            _row(0, 5, "mid_01", 20, 0, 10, 0),
            _row(1, 2, "mid_01", 30, 8, 12, 0),
        ]

        hybrid_rows = build_hybrid_rows(rows, compile_gate("feature_x >= 3"))
        summary = summarize_rows(hybrid_rows)

        self.assertEqual(hybrid_rows[0]["chosen_source"], "pfm")
        self.assertEqual(hybrid_rows[1]["chosen_source"], "lightglue")
        self.assertEqual(summary["kept_pfm_rows"], 1)
        self.assertEqual(summary["fallback_lightglue_rows"], 1)
        self.assertEqual(summary["hybrid_correct"], 32)
        self.assertEqual(summary["hybrid_wrong"], 0)
        self.assertEqual(summary["correct_delta_vs_lightglue"], 10)
        self.assertEqual(summary["wrong_delta_vs_lightglue"], 0)

    def test_apply_observable_gate_writes_csv_json_and_html(self):
        rows = [
            _row(0, 5, "mid_01", 20, 0, 10, 0),
            _row(1, 2, "mid_01", 30, 8, 12, 0),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_csv = root / "dataset.csv"
            output_dir = root / "out"
            with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            summary = apply_observable_gate(
                dataset_csv=dataset_csv,
                gate="feature_x >= 3",
                output_dir=output_dir,
            )

            self.assertEqual(summary["hybrid_correct"], 32)
            self.assertEqual(summary["hybrid_wrong"], 0)
            self.assertTrue((output_dir / "hybrid_rows.csv").exists())
            self.assertTrue((output_dir / "index.html").exists())
            saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["gate"], "feature_x >= 3")


if __name__ == "__main__":
    unittest.main()
