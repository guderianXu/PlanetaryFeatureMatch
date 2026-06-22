import csv
import json
import tempfile
import unittest
from pathlib import Path

from sweep_observable_pair_gate import (
    build_base_rules,
    group_rows_by_split,
    hybrid_summary_for_selected,
    sweep_observable_rules,
)


def _row(
    split,
    pair_index,
    h90,
    variant,
    pfm_correct,
    pfm_wrong,
    lightglue_correct,
    lightglue_wrong,
):
    return {
        "source_name": "unit",
        "split": split,
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
        "reject_label": "1",
        "teacher_correct_delta": str(lightglue_correct - pfm_correct),
        "teacher_wrong_delta": str(lightglue_wrong - pfm_wrong),
        "feature_matches": str(pfm_correct + pfm_wrong),
        "feature_homography_residual_p90_px": str(h90),
        "feature_score_mean": "20.0",
    }


class ObservablePairGateSweepTest(unittest.TestCase):
    def test_base_rules_do_not_leak_truth_or_label_columns(self):
        rows = [
            _row("dev_train", 0, 1.0, "mid_01", 20, 0, 10, 0),
            _row("dev_train", 1, 6.0, "extreme_03", 30, 8, 12, 0),
            _row("dev_val", 2, 1.2, "mid_01", 18, 0, 9, 0),
            _row("dev_val", 3, 5.5, "extreme_03", 25, 5, 11, 0),
        ]

        rules = build_base_rules(group_rows_by_split(rows), max_thresholds=8)
        rule_names = [rule.name for rule in rules]

        self.assertTrue(any(name.startswith("feature_homography_residual_p90_px <=") for name in rule_names))
        self.assertTrue(any(name == "target_variant == mid_01" for name in rule_names))
        forbidden_fragments = ["pfm_correct", "pfm_wrong", "lightglue_correct", "lightglue_wrong", "label", "delta"]
        for name in rule_names:
            self.assertFalse(any(fragment in name for fragment in forbidden_fragments), name)

    def test_hybrid_summary_uses_pfm_only_for_selected_rows(self):
        rows = [
            _row("dev_val", 0, 1.0, "mid_01", 20, 0, 10, 0),
            _row("dev_val", 1, 6.0, "extreme_03", 30, 8, 12, 0),
        ]

        summary = hybrid_summary_for_selected(rows, {0})

        self.assertEqual(summary["use_pfm_pairs"], 1)
        self.assertEqual(summary["lightglue_correct"], 22)
        self.assertEqual(summary["lightglue_wrong"], 0)
        self.assertEqual(summary["hybrid_correct"], 32)
        self.assertEqual(summary["hybrid_wrong"], 0)
        self.assertEqual(summary["correct_delta_vs_lightglue"], 10)
        self.assertEqual(summary["wrong_delta_vs_lightglue"], 0)

    def test_sweep_writes_valid_rule_that_passes_train_and_eval(self):
        rows = [
            _row("dev_train", 0, 1.0, "mid_01", 20, 0, 10, 0),
            _row("dev_train", 1, 6.0, "extreme_03", 30, 8, 12, 0),
            _row("dev_val", 2, 1.2, "mid_01", 18, 0, 9, 0),
            _row("dev_val", 3, 5.5, "extreme_03", 25, 5, 11, 0),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_csv = root / "dataset.csv"
            output_dir = root / "out"
            with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            summary = sweep_observable_rules(
                dataset_csv=dataset_csv,
                output_dir=output_dir,
                train_split="dev_train",
                eval_split="dev_val",
                allow_wrong_delta=0,
                min_correct_delta=1,
                max_thresholds=8,
            )

            self.assertGreater(summary["valid_rule_count"], 0)
            self.assertIn("best_valid", summary)
            self.assertLessEqual(summary["best_valid"]["dev_val_wrong_delta_vs_lightglue"], 0)
            self.assertGreater(summary["best_valid"]["dev_val_correct_delta_vs_lightglue"], 0)
            self.assertTrue((output_dir / "valid_rules.csv").exists())
            saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["valid_rule_count"], summary["valid_rule_count"])


if __name__ == "__main__":
    unittest.main()
