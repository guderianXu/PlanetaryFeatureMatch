import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class ApplyMatchSetRejectionCalibratorTest(unittest.TestCase):
    def write_dataset(self, path: Path) -> None:
        rows = [
            {
                "source_name": "unit",
                "split": "all",
                "pair_index": "0",
                "pair_type": "same_position_view",
                "base_id": "keep_pair",
                "reference_variant": "nadir",
                "target_variant": "mid_01",
                "reject_label": "0",
                "pfm_matches": "100",
                "pfm_correct": "100",
                "pfm_wrong": "0",
                "pfm_precision": "1.0",
                "lightglue_matches": "40",
                "lightglue_correct": "40",
                "lightglue_wrong": "0",
                "lightglue_precision": "1.0",
                "feature_badness": "-5.0",
            },
            {
                "source_name": "unit",
                "split": "all",
                "pair_index": "1",
                "pair_type": "same_position_view",
                "base_id": "fallback_pair",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "reject_label": "1",
                "pfm_matches": "90",
                "pfm_correct": "80",
                "pfm_wrong": "10",
                "pfm_precision": "0.888889",
                "lightglue_matches": "50",
                "lightglue_correct": "50",
                "lightglue_wrong": "0",
                "lightglue_precision": "1.0",
                "feature_badness": "5.0",
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_model(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "standardized_logistic_regression",
                    "feature_columns": ["feature_badness"],
                    "means": [0.0],
                    "scales": [1.0],
                    "weights": [1.0],
                    "bias": 0.0,
                    "label_column": "reject_label",
                    "threshold": 0.5,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_cli_writes_hybrid_predictions_summary_and_html(self) -> None:
        import apply_match_set_rejection_calibrator as apply_calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_csv = root / "rejection_dataset.csv"
            model_json = root / "model.json"
            output_csv = root / "hybrid_summary.csv"
            summary_json = root / "summary.json"
            output_html = root / "index.html"
            self.write_dataset(dataset_csv)
            self.write_model(model_json)

            exit_code = apply_calibrator.main(
                [
                    "--dataset-csv",
                    str(dataset_csv),
                    "--model-json",
                    str(model_json),
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_csv.open(newline="", encoding="utf-8") as handle:
                output_rows = list(csv.DictReader(handle))
            self.assertEqual([row["chosen_source"] for row in output_rows], ["pfm", "lightglue"])
            self.assertEqual(output_rows[0]["matches"], "100")
            self.assertEqual(output_rows[1]["matches"], "50")
            self.assertEqual(output_rows[1]["correct"], "50")
            self.assertEqual(output_rows[1]["wrong"], "0")

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["kept_pfm_rows"], 1)
            self.assertEqual(summary["fallback_lightglue_rows"], 1)
            self.assertEqual(summary["hybrid_correct"], 150)
            self.assertEqual(summary["hybrid_wrong"], 0)
            self.assertEqual(summary["hybrid_correct_delta_vs_lightglue"], 60)
            self.assertEqual(summary["hybrid_wrong_delta_vs_pfm"], -10)
            self.assertIn("Match-set rejection calibrator application", output_html.read_text(encoding="utf-8"))

    def test_cli_can_zero_rejected_pfm_pairs_without_lightglue_fallback(self) -> None:
        import apply_match_set_rejection_calibrator as apply_calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_csv = root / "rejection_dataset.csv"
            model_json = root / "model.json"
            output_csv = root / "pfm_only_summary.csv"
            summary_json = root / "summary.json"
            output_html = root / "index.html"
            self.write_dataset(dataset_csv)
            self.write_model(model_json)

            exit_code = apply_calibrator.main(
                [
                    "--dataset-csv",
                    str(dataset_csv),
                    "--model-json",
                    str(model_json),
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(output_html),
                    "--reject-action",
                    "zero",
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_csv.open(newline="", encoding="utf-8") as handle:
                output_rows = list(csv.DictReader(handle))
            self.assertEqual([row["chosen_source"] for row in output_rows], ["pfm", "rejected"])
            self.assertEqual(output_rows[1]["matches"], "0")
            self.assertEqual(output_rows[1]["correct"], "0")
            self.assertEqual(output_rows[1]["wrong"], "0")

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["kept_pfm_rows"], 1)
            self.assertEqual(summary["fallback_lightglue_rows"], 0)
            self.assertEqual(summary["rejected_rows"], 1)
            self.assertEqual(summary["hybrid_correct"], 100)
            self.assertEqual(summary["hybrid_wrong"], 0)
            self.assertEqual(summary["hybrid_correct_delta_vs_pfm"], -80)
            self.assertEqual(summary["hybrid_wrong_delta_vs_pfm"], -10)

    def test_cli_can_override_model_threshold_for_diagnostic_sweeps(self) -> None:
        import apply_match_set_rejection_calibrator as apply_calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_csv = root / "rejection_dataset.csv"
            model_json = root / "model.json"
            output_csv = root / "threshold_override_summary.csv"
            summary_json = root / "summary.json"
            output_html = root / "index.html"
            self.write_dataset(dataset_csv)
            self.write_model(model_json)

            exit_code = apply_calibrator.main(
                [
                    "--dataset-csv",
                    str(dataset_csv),
                    "--model-json",
                    str(model_json),
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(output_html),
                    "--reject-action",
                    "zero",
                    "--threshold-override",
                    "0.999",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["kept_pfm_rows"], 2)
            self.assertEqual(summary["rejected_rows"], 0)
            self.assertEqual(summary["hybrid_correct"], 180)
            self.assertEqual(summary["hybrid_wrong"], 10)
            self.assertEqual(summary["threshold"], 0.999)


if __name__ == "__main__":
    unittest.main()
