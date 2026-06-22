import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class ApplyMatchDetailFilterCalibratorTest(unittest.TestCase):
    def sample_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        specs = [
            ("pair_a", "22.0", "0.95", "1"),
            ("pair_a", "21.0", "0.90", "1"),
            ("pair_a", "9.0", "0.05", "0"),
            ("pair_b", "23.0", "0.96", "1"),
        ]
        for base_id, score, raw_margin, correct in specs:
            rows.append(
                {
                    "label": "PFM / all-filtered",
                    "pair_index": "0" if base_id == "pair_a" else "1",
                    "base_id": base_id,
                    "reference_variant": "nadir",
                    "target_variant": "mid_01",
                    "split": "fresh",
                    "match_index": str(len(rows)),
                    "point_a_x_px": "10.0",
                    "point_a_y_px": "20.0",
                    "point_b_x_px": "11.0",
                    "point_b_y_px": "21.0",
                    "score": score,
                    "pair_logit": score,
                    "row_dustbin_logit": "0.4",
                    "col_dustbin_logit": "0.4",
                    "positive_vs_dustbin_margin": score,
                    "raw_similarity": "0.9",
                    "raw_margin": raw_margin,
                    "accept_logit": "2.0",
                    "accept_probability": "0.88",
                    "error_px": "1.0" if correct == "1" else "9.0",
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

    def write_model(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "standardized_logistic_regression_match_filter",
                    "feature_columns": ["feature_score"],
                    "means": [15.0],
                    "scales": [1.0],
                    "weights": [-1.0],
                    "bias": 0.0,
                    "label_column": "reject_label",
                    "threshold": 0.5,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_true_geometry_model(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "standardized_logistic_regression_match_filter",
                    "feature_columns": ["feature_true_geometry_error_px"],
                    "means": [0.0],
                    "scales": [1.0],
                    "weights": [1.0],
                    "bias": -5.0,
                    "label_column": "reject_label",
                    "threshold": 0.5,
                    "include_true_geometry_features": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_true_geometry_mlp_model(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "standardized_mlp_match_filter",
                    "feature_columns": ["feature_true_geometry_error_px"],
                    "means": [0.0],
                    "scales": [1.0],
                    "hidden_dim": 1,
                    "layer1_weight": [[1.0]],
                    "layer1_bias": [0.0],
                    "layer2_weight": [1.0],
                    "layer2_bias": -5.0,
                    "label_column": "reject_label",
                    "threshold": 0.5,
                    "include_true_geometry_features": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_cli_filters_match_details_and_writes_pair_summary(self) -> None:
        import apply_match_detail_filter_calibrator as apply_filter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            match_details = root / "all_filtered_match_details.csv"
            model_json = root / "model.json"
            output_dir = root / "applied"
            self.write_csv(match_details, self.sample_rows())
            self.write_model(model_json)

            exit_code = apply_filter.main(
                [
                    "--match-details",
                    str(match_details),
                    "--model-json",
                    str(model_json),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            for relative_path in [
                "match_predictions.csv",
                "kept_match_details.csv",
                "pair_summary.csv",
                "summary.json",
                "index.html",
            ]:
                self.assertTrue((output_dir / relative_path).exists(), relative_path)

            with (output_dir / "kept_match_details.csv").open(newline="", encoding="utf-8") as handle:
                kept_rows = list(csv.DictReader(handle))
            self.assertEqual(len(kept_rows), 3)
            self.assertTrue(all(row["correct"] == "1" for row in kept_rows))

            with (output_dir / "pair_summary.csv").open(newline="", encoding="utf-8") as handle:
                pair_rows = {row["base_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(pair_rows["pair_a"]["matches"], "3")
            self.assertEqual(pair_rows["pair_a"]["wrong"], "1")
            self.assertEqual(pair_rows["pair_a"]["kept_matches"], "2")
            self.assertEqual(pair_rows["pair_a"]["kept_wrong"], "0")

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["matches"], 4)
            self.assertEqual(summary["correct"], 3)
            self.assertEqual(summary["wrong"], 1)
            self.assertEqual(summary["kept_matches"], 3)
            self.assertEqual(summary["kept_correct"], 3)
            self.assertEqual(summary["kept_wrong"], 0)
            self.assertEqual(summary["wrong_reduction"], 1.0)
            self.assertEqual(summary["correct_retention"], 1.0)
            self.assertIn("Match-detail filter application", (output_dir / "index.html").read_text(encoding="utf-8"))

    def test_cli_applies_true_geometry_features_when_model_requests_them(self) -> None:
        import apply_match_detail_filter_calibrator as apply_filter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            match_details = root / "all_match_details.csv"
            model_json = root / "model.json"
            output_dir = root / "applied_true_geometry"
            self.write_csv(match_details, self.sample_rows())
            self.write_true_geometry_model(model_json)

            exit_code = apply_filter.main(
                [
                    "--match-details",
                    str(match_details),
                    "--model-json",
                    str(model_json),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            with (output_dir / "kept_match_details.csv").open(newline="", encoding="utf-8") as handle:
                kept_rows = list(csv.DictReader(handle))
            self.assertEqual(len(kept_rows), 3)
            self.assertTrue(all(row["correct"] == "1" for row in kept_rows))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["include_true_geometry_features"])

    def test_cli_applies_true_geometry_mlp_model(self) -> None:
        import apply_match_detail_filter_calibrator as apply_filter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            match_details = root / "all_match_details.csv"
            model_json = root / "mlp_model.json"
            output_dir = root / "applied_true_geometry_mlp"
            self.write_csv(match_details, self.sample_rows())
            self.write_true_geometry_mlp_model(model_json)

            exit_code = apply_filter.main(
                [
                    "--match-details",
                    str(match_details),
                    "--model-json",
                    str(model_json),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            with (output_dir / "kept_match_details.csv").open(newline="", encoding="utf-8") as handle:
                kept_rows = list(csv.DictReader(handle))
            self.assertEqual(len(kept_rows), 3)
            self.assertTrue(all(row["correct"] == "1" for row in kept_rows))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["model_type"], "standardized_mlp_match_filter")
            self.assertTrue(summary["include_true_geometry_features"])
            self.assertEqual(summary["kept_wrong"], 0)

    def test_cli_can_override_threshold_for_target_variant(self) -> None:
        import apply_match_detail_filter_calibrator as apply_filter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            match_details = root / "all_filtered_match_details.csv"
            model_json = root / "model.json"
            output_dir = root / "applied_variant_threshold"
            rows = self.sample_rows()
            rows[2]["target_variant"] = "extreme_01"
            rows[2]["score"] = "14.6"
            rows[2]["pair_logit"] = "14.6"
            rows[2]["positive_vs_dustbin_margin"] = "14.6"
            self.write_csv(match_details, rows)
            self.write_model(model_json)

            exit_code = apply_filter.main(
                [
                    "--match-details",
                    str(match_details),
                    "--model-json",
                    str(model_json),
                    "--output-dir",
                    str(output_dir),
                    "--variant-threshold",
                    "extreme_01=0.7",
                ]
            )

            self.assertEqual(exit_code, 0)
            with (output_dir / "kept_match_details.csv").open(newline="", encoding="utf-8") as handle:
                kept_rows = list(csv.DictReader(handle))
            self.assertEqual(len(kept_rows), 4)
            self.assertEqual(kept_rows[2]["target_variant"], "extreme_01")

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["kept_wrong"], 1)
            self.assertEqual(summary["variant_thresholds"], {"extreme_01": 0.7})

    def test_prediction_probability_precision_round_trips_threshold_decision(self) -> None:
        import apply_match_detail_filter_calibrator as apply_filter

        rows = self.sample_rows()[:1]
        feature_rows = [{"reject_label": "0"}]
        threshold = 0.1578150000001
        scores = [0.1578150000004]

        prediction_rows = apply_filter.build_prediction_rows(
            rows,
            feature_rows,
            scores,
            threshold=threshold,
        )

        self.assertEqual(prediction_rows[0]["predicted_reject"], "1")
        self.assertGreaterEqual(float(prediction_rows[0]["reject_probability"]), threshold)


if __name__ == "__main__":
    unittest.main()
