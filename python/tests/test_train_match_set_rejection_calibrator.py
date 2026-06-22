import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class MatchSetRejectionCalibratorTest(unittest.TestCase):
    def sample_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        specs = [
            ("formal", "train_keep_a", "0.10", "100", "100", "0", "0"),
            ("formal", "train_keep_b", "0.20", "80", "79", "1", "0"),
            ("formal", "train_reject_a", "4.80", "120", "112", "8", "1"),
            ("formal", "train_reject_b", "5.20", "90", "82", "8", "1"),
            ("validation", "eval_keep", "0.15", "70", "70", "0", "0"),
            ("validation", "eval_reject", "5.00", "110", "100", "10", "1"),
        ]
        for split, base_id, badness, matches, correct, wrong, label in specs:
            rows.append(
                {
                    "source_name": "unit",
                    "split": split,
                    "pair_index": str(len(rows)),
                    "pair_type": "same_position_view",
                    "base_id": base_id,
                    "reference_variant": "nadir",
                    "target_variant": "extreme_03" if label == "1" else "mid_01",
                    "pfm_matches": matches,
                    "pfm_correct": correct,
                    "pfm_wrong": wrong,
                    "pfm_precision": "1.0" if wrong == "0" else "0.9",
                    "lightglue_matches": "50",
                    "lightglue_correct": "50",
                    "lightglue_wrong": "0",
                    "lightglue_precision": "1.0",
                    "reject_label": label,
                    "keep_label": "1" if label == "0" else "0",
                    "feature_badness": badness,
                    "feature_matches": matches,
                }
            )
        return rows

    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_train_model_uses_feature_columns_only_and_scores_reject_rows_higher(self) -> None:
        import train_match_set_rejection_calibrator as calibrator

        rows = self.sample_rows()
        train_rows = [row for row in rows if row["split"] == "formal"]
        eval_rows = [row for row in rows if row["split"] == "validation"]

        feature_columns = calibrator.select_feature_columns(rows)
        self.assertEqual(feature_columns, ["feature_badness", "feature_matches"])

        model = calibrator.train_model(
            train_rows,
            feature_columns=feature_columns,
            label_column="reject_label",
            epochs=300,
            learning_rate=0.2,
            l2=0.0,
        )
        eval_scores = calibrator.score_rows(model, eval_rows)

        self.assertGreater(eval_scores[1], eval_scores[0])
        self.assertNotIn("pfm_wrong", model.feature_columns)
        self.assertNotIn("reject_label", model.feature_columns)

    def test_cli_writes_model_predictions_sweep_summary_and_html(self) -> None:
        import train_match_set_rejection_calibrator as calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_csv = root / "rejection_dataset.csv"
            output_dir = root / "calibrator"
            self.write_csv(dataset_csv, self.sample_rows())

            exit_code = calibrator.main(
                [
                    "--dataset-csv",
                    str(dataset_csv),
                    "--output-dir",
                    str(output_dir),
                    "--train-split",
                    "formal",
                    "--eval-split",
                    "validation",
                    "--epochs",
                    "300",
                    "--learning-rate",
                    "0.2",
                    "--min-kept-correct-ratio",
                    "0.35",
                ]
            )

            self.assertEqual(exit_code, 0)
            model_json = output_dir / "model.json"
            summary_json = output_dir / "summary.json"
            predictions_csv = output_dir / "validation_predictions.csv"
            sweep_csv = output_dir / "threshold_sweep.csv"
            report_html = output_dir / "index.html"
            self.assertTrue(model_json.exists())
            self.assertTrue(summary_json.exists())
            self.assertTrue(predictions_csv.exists())
            self.assertTrue(sweep_csv.exists())
            self.assertTrue(report_html.exists())

            model_payload = json.loads(model_json.read_text(encoding="utf-8"))
            self.assertEqual(model_payload["feature_columns"], ["feature_badness", "feature_matches"])
            self.assertNotIn("pfm_wrong", model_payload["feature_columns"])

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["train"]["rows"], 4)
            self.assertEqual(summary["eval"]["rows"], 2)
            self.assertEqual(summary["eval"]["predicted_reject_rows"], 1)
            self.assertLess(summary["eval"]["kept_pfm_wrong"], summary["eval"]["pfm_wrong"])
            self.assertEqual(summary["eval"]["hybrid_pfm_lightglue_correct"], 120)
            self.assertEqual(summary["eval"]["hybrid_pfm_lightglue_wrong"], 0)
            self.assertEqual(summary["eval"]["hybrid_correct_delta_vs_lightglue"], 20)

            html_text = report_html.read_text(encoding="utf-8")
            self.assertIn("Match-set rejection calibrator", html_text)
            self.assertIn("validation_predictions.csv", html_text)

    def test_cli_can_evaluate_a_separate_dataset_csv(self) -> None:
        import train_match_set_rejection_calibrator as calibrator

        rows = self.sample_rows()
        train_rows = [row for row in rows if row["split"] == "formal"]
        eval_rows = [row for row in rows if row["split"] == "validation"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "train_rejection_dataset.csv"
            eval_csv = root / "heldout_rejection_dataset.csv"
            output_dir = root / "heldout_eval"
            self.write_csv(train_csv, train_rows)
            self.write_csv(eval_csv, eval_rows)

            exit_code = calibrator.main(
                [
                    "--dataset-csv",
                    str(train_csv),
                    "--eval-dataset-csv",
                    str(eval_csv),
                    "--output-dir",
                    str(output_dir),
                    "--train-split",
                    "formal",
                    "--eval-split",
                    "validation",
                    "--epochs",
                    "300",
                    "--learning-rate",
                    "0.2",
                    "--min-kept-correct-ratio",
                    "0.35",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["dataset_csv"], str(train_csv))
            self.assertEqual(summary["eval_dataset_csv"], str(eval_csv))
            self.assertEqual(summary["train"]["rows"], 4)
            self.assertEqual(summary["eval"]["rows"], 2)
            self.assertEqual(summary["eval"]["predicted_reject_rows"], 1)

    def test_cli_can_choose_threshold_by_hybrid_lightglue_wrong_cap(self) -> None:
        import train_match_set_rejection_calibrator as calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_csv = root / "rejection_dataset.csv"
            output_dir = root / "calibrator_hybrid_cap"
            rows = self.sample_rows()
            for row in rows:
                if row["base_id"] == "train_keep_b":
                    row["pfm_correct"] = "80"
                    row["pfm_wrong"] = "0"
                    row["pfm_precision"] = "1.0"
            self.write_csv(dataset_csv, rows)

            exit_code = calibrator.main(
                [
                    "--dataset-csv",
                    str(dataset_csv),
                    "--output-dir",
                    str(output_dir),
                    "--train-split",
                    "formal",
                    "--eval-split",
                    "validation",
                    "--epochs",
                    "300",
                    "--learning-rate",
                    "0.2",
                    "--threshold-objective",
                    "hybrid_lightglue_wrong_cap",
                    "--max-hybrid-wrong-delta-vs-lightglue",
                    "0",
                    "--min-kept-correct-ratio",
                    "1.0",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["threshold_objective"], "hybrid_lightglue_wrong_cap")
            self.assertEqual(summary["max_hybrid_wrong_delta_vs_lightglue"], 0)
            self.assertLessEqual(summary["train"]["hybrid_wrong_delta_vs_lightglue"], 0)
            self.assertLessEqual(summary["eval"]["hybrid_wrong_delta_vs_lightglue"], 0)
            self.assertGreater(summary["eval"]["hybrid_correct_delta_vs_lightglue"], 0)

            with (output_dir / "threshold_sweep.csv").open(encoding="utf-8", newline="") as handle:
                sweep_fields = csv.DictReader(handle).fieldnames
            self.assertIn("train_hybrid_pfm_lightglue_correct", sweep_fields)
            self.assertIn("train_hybrid_wrong_delta_vs_lightglue", sweep_fields)

    def test_cli_can_choose_threshold_on_eval_split_by_pfm_wrong_cap(self) -> None:
        import train_match_set_rejection_calibrator as calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_csv = root / "rejection_dataset.csv"
            output_dir = root / "calibrator_eval_wrong_cap"
            self.write_csv(dataset_csv, self.sample_rows())

            exit_code = calibrator.main(
                [
                    "--dataset-csv",
                    str(dataset_csv),
                    "--output-dir",
                    str(output_dir),
                    "--train-split",
                    "formal",
                    "--eval-split",
                    "validation",
                    "--epochs",
                    "300",
                    "--learning-rate",
                    "0.2",
                    "--threshold-selection-source",
                    "eval",
                    "--threshold-objective",
                    "pfm_wrong_cap",
                    "--max-kept-pfm-wrong",
                    "0",
                    "--min-kept-correct-ratio",
                    "0.35",
                ]
            )

            self.assertEqual(exit_code, 0)
            model_payload = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["threshold_objective"], "pfm_wrong_cap")
            self.assertEqual(summary["threshold_selection_source"], "eval")
            self.assertEqual(summary["max_kept_pfm_wrong"], 0)
            self.assertEqual(summary["threshold_selection"]["rows"], 2)
            self.assertEqual(summary["eval"]["kept_pfm_wrong"], 0)
            self.assertEqual(summary["eval"]["kept_pfm_correct"], 70)
            self.assertEqual(model_payload["threshold"], summary["threshold"])

    def test_pfm_wrong_cap_threshold_respects_min_kept_correct_ratio(self) -> None:
        import train_match_set_rejection_calibrator as calibrator

        rows = [
            {
                "reject_label": "1",
                "pfm_matches": "110",
                "pfm_correct": "100",
                "pfm_wrong": "10",
                "lightglue_matches": "100",
                "lightglue_correct": "100",
                "lightglue_wrong": "0",
            },
            {
                "reject_label": "1",
                "pfm_matches": "110",
                "pfm_correct": "100",
                "pfm_wrong": "10",
                "lightglue_matches": "100",
                "lightglue_correct": "100",
                "lightglue_wrong": "0",
            },
        ]
        scores = [0.90, 0.80]

        threshold = calibrator.choose_pfm_wrong_cap_threshold(
            rows,
            scores,
            label_column="reject_label",
            max_kept_pfm_wrong=0,
            min_kept_correct_ratio=0.50,
        )

        self.assertEqual(threshold, 1.000001)

    def test_hybrid_wrong_cap_threshold_returns_zero_when_only_all_reject_is_safe(self) -> None:
        import train_match_set_rejection_calibrator as calibrator

        rows = [
            {
                "reject_label": "1",
                "pfm_matches": "10",
                "pfm_correct": "9",
                "pfm_wrong": "1",
                "lightglue_matches": "5",
                "lightglue_correct": "5",
                "lightglue_wrong": "0",
            },
            {
                "reject_label": "1",
                "pfm_matches": "8",
                "pfm_correct": "7",
                "pfm_wrong": "1",
                "lightglue_matches": "4",
                "lightglue_correct": "4",
                "lightglue_wrong": "0",
            },
        ]
        scores = [0.25, 0.75]

        threshold = calibrator.choose_hybrid_lightglue_wrong_cap_threshold(
            rows,
            scores,
            label_column="reject_label",
            max_wrong_delta_vs_lightglue=0,
        )

        self.assertEqual(threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
