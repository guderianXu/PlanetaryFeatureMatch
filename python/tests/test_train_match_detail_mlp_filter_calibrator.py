import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class MatchDetailMlpFilterCalibratorTest(unittest.TestCase):
    def sample_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        specs = [
            ("dev", "pair_a", "24.0", "0.90", "1"),
            ("dev", "pair_a", "22.0", "0.82", "1"),
            ("dev", "pair_a", "2.0", "0.02", "0"),
            ("dev", "pair_b", "25.0", "0.91", "1"),
            ("dev", "pair_b", "3.0", "0.04", "0"),
            ("lockbox", "pair_c", "23.0", "0.88", "1"),
            ("lockbox", "pair_c", "2.5", "0.03", "0"),
        ]
        for split, base_id, score, raw_margin, correct in specs:
            rows.append(
                {
                    "label": "PFM / all",
                    "pair_index": "0" if base_id != "pair_b" else "1",
                    "base_id": base_id,
                    "reference_variant": "extreme_01",
                    "target_variant": "extreme_02",
                    "split": split,
                    "match_index": str(len(rows)),
                    "point_a_x_px": str(10.0 + len(rows)),
                    "point_a_y_px": str(20.0 + len(rows)),
                    "point_b_x_px": str(11.0 + len(rows)),
                    "point_b_y_px": str(21.0 + len(rows)),
                    "score": score,
                    "pair_logit": score,
                    "row_dustbin_logit": "0.4",
                    "col_dustbin_logit": "0.4",
                    "positive_vs_dustbin_margin": score,
                    "raw_similarity": "0.9",
                    "raw_margin": raw_margin,
                    "accept_logit": "2.0",
                    "accept_probability": "0.88",
                    "error_px": "1.0" if correct == "1" else "12.0",
                    "correct": correct,
                    "valid_fraction": "0.8",
                }
            )
        return rows

    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_mlp_model_scores_low_quality_match_as_reject(self) -> None:
        import train_match_detail_filter_calibrator as base_calibrator
        import train_match_detail_mlp_filter_calibrator as mlp_calibrator

        rows = base_calibrator.build_training_rows(self.sample_rows())
        feature_columns = ["feature_score", "feature_raw_margin"]

        model = mlp_calibrator.train_mlp_model(
            rows,
            feature_columns=feature_columns,
            label_column="reject_label",
            hidden_dim=6,
            epochs=180,
            learning_rate=0.03,
            l2=0.0,
            seed=7,
        )
        scores = mlp_calibrator.score_rows(model, rows)

        good_scores = [score for row, score in zip(rows, scores) if row["reject_label"] == "0"]
        bad_scores = [score for row, score in zip(rows, scores) if row["reject_label"] == "1"]
        self.assertEqual(model.model_type, "standardized_mlp_match_filter")
        self.assertGreater(min(bad_scores), max(good_scores))
        self.assertNotIn("feature_true_geometry_error_px", model.feature_columns)

    def test_cli_writes_model_summary_predictions_and_sweep(self) -> None:
        import train_match_detail_mlp_filter_calibrator as mlp_calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "train_match_details.csv"
            eval_csv = root / "eval_match_details.csv"
            output_dir = root / "mlp"
            rows = self.sample_rows()
            self.write_csv(train_csv, [row for row in rows if row["split"] == "dev"])
            self.write_csv(eval_csv, [row for row in rows if row["split"] == "lockbox"])

            exit_code = mlp_calibrator.main(
                [
                    "--train-match-details",
                    str(train_csv),
                    "--eval-match-details",
                    str(eval_csv),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "140",
                    "--hidden-dim",
                    "6",
                    "--learning-rate",
                    "0.03",
                    "--threshold-objective",
                    "pfm_wrong_cap",
                    "--max-kept-wrong",
                    "0",
                    "--max-thresholds",
                    "100",
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(exit_code, 0)
            for relative_path in [
                "model.json",
                "summary.json",
                "threshold_sweep.csv",
                "all_match_predictions.csv",
                "index.html",
            ]:
                self.assertTrue((output_dir / relative_path).exists(), relative_path)

            model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(model["type"], "standardized_mlp_match_filter")
            self.assertFalse(model["include_true_geometry_features"])
            self.assertNotIn("feature_true_geometry_error_px", model["feature_columns"])
            self.assertEqual(summary["eval"]["kept_wrong"], 0)
            self.assertIn("Match-detail MLP filter calibrator", (output_dir / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
