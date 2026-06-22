import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class MatchDetailFilterCalibratorTest(unittest.TestCase):
    def sample_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        specs = [
            ("formal", "pair_a", "22.0", "0.95", "1"),
            ("formal", "pair_a", "21.0", "0.90", "1"),
            ("formal", "pair_a", "9.0", "0.05", "0"),
            ("formal", "pair_b", "23.0", "0.96", "1"),
            ("formal", "pair_b", "8.5", "0.04", "0"),
            ("validation", "pair_c", "22.5", "0.95", "1"),
            ("validation", "pair_c", "8.0", "0.03", "0"),
        ]
        for split, base_id, score, raw_margin, correct in specs:
            rows.append(
                {
                    "label": "PFM / all-filtered",
                    "pair_index": str(len(rows) // 3),
                    "base_id": base_id,
                    "reference_variant": "nadir",
                    "target_variant": "mid_01",
                    "split": split,
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

    def test_build_training_rows_uses_inference_features_only(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = calibrator.build_training_rows(self.sample_rows())
        feature_columns = calibrator.select_match_feature_columns(rows)

        self.assertIn("feature_score", feature_columns)
        self.assertIn("feature_raw_margin", feature_columns)
        self.assertNotIn("feature_correct", feature_columns)
        self.assertNotIn("feature_error_px", feature_columns)
        self.assertEqual(rows[2]["reject_label"], "1")
        self.assertEqual(rows[0]["reject_label"], "0")

    def test_build_training_rows_adds_match_coordinate_displacement_features_without_truth_leakage(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        source_rows = self.sample_rows()
        source_rows[0].update(
            {
                "point_a_x_px": "512.0",
                "point_a_y_px": "1024.0",
                "point_b_x_px": "640.0",
                "point_b_y_px": "768.0",
                "correct": "1",
                "error_px": "1.0",
            }
        )

        rows = calibrator.build_training_rows(source_rows)
        feature_columns = calibrator.select_match_feature_columns(rows)
        row = rows[0]

        expected_columns = {
            "feature_point_a_x_norm",
            "feature_point_a_y_norm",
            "feature_point_b_x_norm",
            "feature_point_b_y_norm",
            "feature_displacement_dx_px",
            "feature_displacement_dy_px",
            "feature_displacement_magnitude_px",
            "feature_displacement_dx_norm",
            "feature_displacement_dy_norm",
            "feature_displacement_magnitude_norm",
            "feature_displacement_angle_cos",
            "feature_displacement_angle_sin",
        }
        self.assertTrue(expected_columns.issubset(set(feature_columns)))
        self.assertAlmostEqual(float(row["feature_point_a_x_norm"]), 0.25, places=6)
        self.assertAlmostEqual(float(row["feature_point_a_y_norm"]), 0.5, places=6)
        self.assertAlmostEqual(float(row["feature_point_b_x_norm"]), 0.3125, places=6)
        self.assertAlmostEqual(float(row["feature_point_b_y_norm"]), 0.375, places=6)
        self.assertEqual(row["feature_displacement_dx_px"], "128.000000")
        self.assertEqual(row["feature_displacement_dy_px"], "-256.000000")
        self.assertAlmostEqual(float(row["feature_displacement_magnitude_px"]), 286.216701, places=6)
        self.assertNotIn("feature_correct", feature_columns)
        self.assertNotIn("feature_error_px", feature_columns)
        self.assertNotIn("feature_true_geometry_error_px", feature_columns)

    def test_build_training_rows_can_include_true_geometry_features_when_enabled(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = calibrator.build_training_rows(self.sample_rows(), include_true_geometry_features=True)
        feature_columns = calibrator.select_match_feature_columns(rows)

        self.assertIn("feature_true_geometry_error_px", feature_columns)
        self.assertIn("feature_true_geometry_error_sq_px", feature_columns)
        self.assertIn("feature_true_geometry_valid_fraction", feature_columns)
        self.assertIn("feature_true_geometry_error_le_5px", feature_columns)
        self.assertIn("feature_true_geometry_valid_ge_0_10", feature_columns)
        self.assertIn("feature_true_geometry_rule_error5_valid010", feature_columns)
        self.assertEqual(rows[0]["feature_true_geometry_error_px"], "1.000000")
        self.assertEqual(rows[2]["feature_true_geometry_error_px"], "9.000000")
        self.assertEqual(rows[0]["feature_true_geometry_error_le_5px"], "1")
        self.assertEqual(rows[2]["feature_true_geometry_error_le_5px"], "0")
        self.assertEqual(rows[0]["feature_true_geometry_valid_ge_0_10"], "1")
        self.assertEqual(rows[0]["feature_true_geometry_rule_error5_valid010"], "1")
        self.assertEqual(rows[2]["feature_true_geometry_rule_error5_valid010"], "0")
        self.assertEqual(rows[2]["reject_label"], "1")
        self.assertNotIn("feature_error_px", feature_columns)

    def test_build_training_rows_adds_pair_context_features(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = calibrator.build_training_rows(self.sample_rows())
        feature_columns = calibrator.select_match_feature_columns(rows)

        self.assertIn("feature_pair_match_count", feature_columns)
        self.assertIn("feature_pair_score_rank_fraction", feature_columns)
        self.assertIn("feature_pair_score_zscore", feature_columns)
        self.assertIn("feature_pair_a_center_distance_norm", feature_columns)
        self.assertIn("feature_pair_b_center_distance_norm", feature_columns)
        self.assertIn("feature_pair_displacement_median_distance_norm", feature_columns)
        self.assertNotIn("feature_pair_correct", feature_columns)
        self.assertNotIn("feature_pair_error_px", feature_columns)

        pair_a_rows = [row for row in rows if row["base_id"] == "pair_a"]
        self.assertEqual(pair_a_rows[0]["feature_pair_match_count"], "3")
        self.assertEqual(pair_a_rows[0]["feature_pair_score_rank_fraction"], "1.000000")
        self.assertEqual(pair_a_rows[1]["feature_pair_score_rank_fraction"], "0.500000")
        self.assertEqual(pair_a_rows[2]["feature_pair_score_rank_fraction"], "0.000000")

    def test_build_training_rows_adds_displacement_consensus_features(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = []
        for index, (ax, ay, bx, by) in enumerate(
            [
                (0.0, 0.0, 10.0, 0.0),
                (10.0, 0.0, 20.0, 0.0),
                (20.0, 0.0, 30.0, 0.0),
                (30.0, 0.0, 90.0, 0.0),
            ]
        ):
            rows.append(
                {
                    **self.sample_rows()[0],
                    "pair_index": "42",
                    "base_id": "consensus_pair",
                    "match_index": str(index),
                    "point_a_x_px": str(ax),
                    "point_a_y_px": str(ay),
                    "point_b_x_px": str(bx),
                    "point_b_y_px": str(by),
                    "correct": "0" if index == 3 else "1",
                    "error_px": "40.0" if index == 3 else "1.0",
                }
            )

        training_rows = calibrator.build_training_rows(rows)
        feature_columns = calibrator.select_match_feature_columns(training_rows)
        outlier = training_rows[3]

        self.assertIn("feature_pair_displacement_median_distance_px", feature_columns)
        self.assertIn("feature_pair_displacement_mad_px", feature_columns)
        self.assertIn("feature_pair_local_displacement_median_distance_px", feature_columns)
        self.assertIn("feature_pair_local_displacement_median_distance_norm", feature_columns)
        self.assertIn("feature_pair_displacement_consensus_fraction_10px", feature_columns)
        self.assertIn("feature_pair_is_displacement_consensus_10px", feature_columns)
        self.assertEqual(training_rows[0]["feature_pair_displacement_median_distance_px"], "0.000000")
        self.assertEqual(outlier["feature_pair_displacement_median_distance_px"], "50.000000")
        self.assertEqual(outlier["feature_pair_is_displacement_consensus_10px"], "0")
        self.assertAlmostEqual(
            float(outlier["feature_pair_displacement_consensus_fraction_10px"]),
            0.75,
            places=6,
        )

    def test_build_training_rows_adds_homography_residual_features(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = []
        for index, (ax, ay, bx, by, correct) in enumerate(
            [
                (0.0, 0.0, 10.0, 0.0, "1"),
                (10.0, 0.0, 20.0, 0.0, "1"),
                (0.0, 10.0, 10.0, 10.0, "1"),
                (10.0, 10.0, 20.0, 10.0, "1"),
                (5.0, 5.0, 90.0, 40.0, "0"),
            ]
        ):
            rows.append(
                {
                    **self.sample_rows()[0],
                    "pair_index": "44",
                    "base_id": "homography_pair",
                    "match_index": str(index),
                    "point_a_x_px": str(ax),
                    "point_a_y_px": str(ay),
                    "point_b_x_px": str(bx),
                    "point_b_y_px": str(by),
                    "correct": correct,
                    "error_px": "1.0" if correct == "1" else "40.0",
                }
            )

        training_rows = calibrator.build_training_rows(rows)
        feature_columns = calibrator.select_match_feature_columns(training_rows)
        outlier = training_rows[4]

        self.assertIn("feature_pair_homography_residual_px", feature_columns)
        self.assertIn("feature_pair_homography_residual_norm", feature_columns)
        self.assertIn("feature_pair_homography_residual_p90_px", feature_columns)
        self.assertIn("feature_pair_is_homography_consensus_4px", feature_columns)
        self.assertNotIn("feature_pair_homography_correct", feature_columns)
        self.assertLess(float(training_rows[0]["feature_pair_homography_residual_px"]), 0.5)
        self.assertGreater(float(outlier["feature_pair_homography_residual_px"]), 10.0)
        self.assertEqual(outlier["feature_pair_is_homography_consensus_4px"], "0")

    def test_build_training_rows_adds_variant_transition_features(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = [
            {
                **self.sample_rows()[0],
                "reference_variant": "extreme_01",
                "target_variant": "extreme_02",
                "pair_type": "cross_camera",
            },
            {
                **self.sample_rows()[1],
                "reference_variant": "mid_01",
                "target_variant": "mid_01",
                "pair_type": "same_position_view",
            },
        ]

        training_rows = calibrator.build_training_rows(rows)
        feature_columns = calibrator.select_match_feature_columns(training_rows)

        self.assertIn("feature_reference_variant_extreme_01", feature_columns)
        self.assertIn("feature_reference_variant_mid_01", feature_columns)
        self.assertIn("feature_target_variant_extreme_02", feature_columns)
        self.assertIn("feature_target_variant_mid_01", feature_columns)
        self.assertIn("feature_variant_transition_extreme_01_to_extreme_02", feature_columns)
        self.assertIn("feature_variant_transition_mid_01_to_mid_01", feature_columns)
        self.assertIn("feature_pair_type_cross_camera", feature_columns)
        self.assertIn("feature_pair_type_same_position_view", feature_columns)
        self.assertIn("feature_score_x_target_variant_extreme_02", feature_columns)
        self.assertIn("feature_raw_margin_x_target_variant_extreme_02", feature_columns)
        self.assertIn("feature_accept_probability_x_target_variant_extreme_02", feature_columns)
        self.assertEqual(training_rows[0]["feature_reference_is_extreme"], "1")
        self.assertEqual(training_rows[0]["feature_target_is_extreme"], "1")
        self.assertEqual(training_rows[0]["feature_variant_changed"], "1")
        self.assertEqual(training_rows[0]["feature_pair_type_cross_camera"], "1")
        self.assertEqual(training_rows[0]["feature_score_x_target_variant_extreme_02"], "22.0")
        self.assertEqual(training_rows[1]["feature_reference_is_extreme"], "0")
        self.assertEqual(training_rows[1]["feature_target_is_extreme"], "0")
        self.assertEqual(training_rows[1]["feature_variant_changed"], "0")
        self.assertEqual(training_rows[1]["feature_pair_type_same_position_view"], "1")
        self.assertEqual(training_rows[1].get("feature_score_x_target_variant_extreme_02", "0"), "0")

    def test_displacement_norm_features_use_pixel_scale_floor(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = []
        for index, dx in enumerate([10.00, 10.02, 10.04, 10.06]):
            rows.append(
                {
                    **self.sample_rows()[0],
                    "pair_index": "43",
                    "base_id": "subpixel_jitter_pair",
                    "match_index": str(index),
                    "point_a_x_px": str(index * 10.0),
                    "point_a_y_px": "0.0",
                    "point_b_x_px": str(index * 10.0 + dx),
                    "point_b_y_px": "0.0",
                    "correct": "1",
                    "error_px": "1.0",
                }
            )

        training_rows = calibrator.build_training_rows(rows)

        for row in training_rows:
            self.assertLessEqual(float(row["feature_pair_displacement_median_distance_norm"]), 0.1)
            self.assertLessEqual(float(row["feature_pair_local_displacement_median_distance_norm"]), 0.1)

    def test_limit_training_rows_can_balance_by_target_variant_and_label(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = []
        for index in range(20):
            rows.append(
                {
                    "target_variant": "extreme_02",
                    "reject_label": "1" if index % 2 else "0",
                    "match_index": str(index),
                }
            )
        rows.extend(
            [
                {"target_variant": "extreme_01", "reject_label": "0", "match_index": "e01_keep"},
                {"target_variant": "extreme_01", "reject_label": "1", "match_index": "e01_reject"},
            ]
        )

        sampled = calibrator.limit_training_rows(rows, 4, balance_key="target_variant")

        self.assertEqual(len(sampled), 4)
        self.assertEqual(
            {(row["target_variant"], row["reject_label"]) for row in sampled},
            {
                ("extreme_01", "0"),
                ("extreme_01", "1"),
                ("extreme_02", "0"),
                ("extreme_02", "1"),
            },
        )

    def test_cli_writes_match_filter_reports(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = self.sample_rows()
        train_rows = [row for row in rows if row["split"] == "formal"]
        eval_rows = [row for row in rows if row["split"] == "validation"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "formal_match_details.csv"
            eval_csv = root / "validation_match_details.csv"
            output_dir = root / "match_filter"
            self.write_csv(train_csv, train_rows)
            self.write_csv(eval_csv, eval_rows)

            exit_code = calibrator.main(
                [
                    "--train-match-details",
                    str(train_csv),
                    "--eval-match-details",
                    str(eval_csv),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "300",
                    "--learning-rate",
                    "0.2",
                    "--min-kept-correct-ratio",
                    "0.50",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train"]["matches"], 5)
            self.assertEqual(summary["eval"]["matches"], 2)
            self.assertEqual(summary["eval"]["kept_wrong"], 0)
            self.assertEqual(summary["eval"]["wrong_reduction"], 1.0)
            model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            self.assertNotIn("feature_correct", model["feature_columns"])
            self.assertNotIn("feature_error_px", model["feature_columns"])
            self.assertTrue((output_dir / "validation_match_predictions.csv").exists())
            self.assertTrue((output_dir / "threshold_sweep.csv").exists())
            self.assertIn(
                "Match-detail filter calibrator",
                (output_dir / "index.html").read_text(encoding="utf-8"),
            )

    def test_cli_can_limit_training_rows_with_balanced_sampling(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = self.sample_rows()
        train_rows = [row for row in rows if row["split"] == "formal"]
        eval_rows = [row for row in rows if row["split"] == "validation"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "formal_match_details.csv"
            eval_csv = root / "validation_match_details.csv"
            output_dir = root / "sampled_match_filter"
            self.write_csv(train_csv, train_rows)
            self.write_csv(eval_csv, eval_rows)

            exit_code = calibrator.main(
                [
                    "--train-match-details",
                    str(train_csv),
                    "--eval-match-details",
                    str(eval_csv),
                    "--output-dir",
                    str(output_dir),
                    "--max-train-rows",
                    "4",
                    "--max-thresholds",
                    "4",
                    "--epochs",
                    "50",
                    "--learning-rate",
                    "0.2",
                    "--min-kept-correct-ratio",
                    "0.50",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train_source_matches"], 5)
            self.assertEqual(summary["train"]["matches"], 4)
            self.assertGreater(summary["train"]["wrong"], 0)
            self.assertGreater(summary["train"]["correct"], 0)
            with (output_dir / "threshold_sweep.csv").open("r", encoding="utf-8") as handle:
                sweep_rows = list(csv.DictReader(handle))
            self.assertLessEqual(len(sweep_rows), 4)

    def test_cli_accepts_multiple_train_and_eval_match_detail_files(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = self.sample_rows()
        train_a = rows[:3]
        train_b = rows[3:5]
        eval_a = [rows[5]]
        eval_b = [rows[6]]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_a_csv = root / "formal_a_match_details.csv"
            train_b_csv = root / "formal_b_match_details.csv"
            eval_a_csv = root / "validation_a_match_details.csv"
            eval_b_csv = root / "validation_b_match_details.csv"
            output_dir = root / "multi_match_filter"
            self.write_csv(train_a_csv, train_a)
            self.write_csv(train_b_csv, train_b)
            self.write_csv(eval_a_csv, eval_a)
            self.write_csv(eval_b_csv, eval_b)

            exit_code = calibrator.main(
                [
                    "--train-match-details",
                    str(train_a_csv),
                    "--train-match-details",
                    str(train_b_csv),
                    "--eval-match-details",
                    str(eval_a_csv),
                    "--eval-match-details",
                    str(eval_b_csv),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "80",
                    "--learning-rate",
                    "0.2",
                    "--threshold-objective",
                    "pfm_wrong_cap",
                    "--threshold-selection-source",
                    "eval",
                    "--max-kept-wrong",
                    "0",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["train_match_details"],
                [str(train_a_csv), str(train_b_csv)],
            )
            self.assertEqual(
                summary["eval_match_details"],
                [str(eval_a_csv), str(eval_b_csv)],
            )
            self.assertEqual(summary["train_source_matches"], 5)
            self.assertEqual(summary["train"]["matches"], 5)
            self.assertEqual(summary["eval"]["matches"], 2)
            self.assertTrue((output_dir / "all_match_predictions.csv").exists())

    def test_cli_can_restrict_feature_columns_by_regex(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = self.sample_rows()
        train_rows = [row for row in rows if row["split"] == "formal"]
        eval_rows = [row for row in rows if row["split"] == "validation"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "formal_match_details.csv"
            eval_csv = root / "validation_match_details.csv"
            output_dir = root / "true_geometry_only_match_filter"
            self.write_csv(train_csv, train_rows)
            self.write_csv(eval_csv, eval_rows)

            exit_code = calibrator.main(
                [
                    "--train-match-details",
                    str(train_csv),
                    "--eval-match-details",
                    str(eval_csv),
                    "--output-dir",
                    str(output_dir),
                    "--include-true-geometry-features",
                    "--feature-name-regex",
                    "^feature_true_geometry_",
                    "--epochs",
                    "120",
                    "--learning-rate",
                    "0.2",
                    "--threshold-objective",
                    "pfm_wrong_cap",
                    "--threshold-selection-source",
                    "eval",
                    "--max-kept-wrong",
                    "0",
                ]
            )

            self.assertEqual(exit_code, 0)
            model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(
                model["feature_columns"],
                [
                    "feature_true_geometry_error_px",
                    "feature_true_geometry_error_sq_px",
                    "feature_true_geometry_error_le_5px",
                    "feature_true_geometry_valid_ge_0_10",
                    "feature_true_geometry_rule_error5_valid010",
                    "feature_true_geometry_valid_fraction",
                ],
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["feature_name_regex"], "^feature_true_geometry_")
            self.assertEqual(summary["eval"]["kept_wrong"], 0)

    def test_cli_selects_threshold_from_eval_wrong_cap(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = self.sample_rows()
        train_rows = [row for row in rows if row["split"] == "formal"]
        eval_rows = [row for row in rows if row["split"] == "validation"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "formal_match_details.csv"
            eval_csv = root / "validation_match_details.csv"
            output_dir = root / "wrong_cap_match_filter"
            self.write_csv(train_csv, train_rows)
            self.write_csv(eval_csv, eval_rows)

            exit_code = calibrator.main(
                [
                    "--train-match-details",
                    str(train_csv),
                    "--eval-match-details",
                    str(eval_csv),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "300",
                    "--learning-rate",
                    "0.2",
                    "--threshold-objective",
                    "pfm_wrong_cap",
                    "--threshold-selection-source",
                    "eval",
                    "--max-kept-wrong",
                    "0",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["threshold_objective"], "pfm_wrong_cap")
            self.assertEqual(summary["threshold_selection_source"], "eval")
            self.assertEqual(summary["max_kept_wrong"], 0)
            self.assertEqual(summary["threshold_selection"]["kept_wrong"], 0)
            self.assertEqual(summary["eval"]["kept_wrong"], 0)

    def test_choose_match_wrong_cap_threshold_keeps_most_correct_under_wrong_cap(self) -> None:
        import train_match_detail_filter_calibrator as calibrator

        rows = [
            {"pfm_matches": "1", "pfm_correct": "1", "pfm_wrong": "0", "reject_label": "0"},
            {"pfm_matches": "1", "pfm_correct": "1", "pfm_wrong": "0", "reject_label": "0"},
            {"pfm_matches": "1", "pfm_correct": "1", "pfm_wrong": "0", "reject_label": "0"},
            {"pfm_matches": "1", "pfm_correct": "0", "pfm_wrong": "1", "reject_label": "1"},
            {"pfm_matches": "1", "pfm_correct": "0", "pfm_wrong": "1", "reject_label": "1"},
        ]
        scores = [0.05, 0.10, 0.65, 0.50, 0.80]

        threshold = calibrator.choose_match_wrong_cap_threshold(
            rows,
            scores,
            max_kept_wrong=1,
            max_thresholds=0,
        )
        summary = calibrator.summarize_threshold(rows, scores, threshold=threshold)

        self.assertEqual(summary["kept_wrong"], 1)
        self.assertEqual(summary["kept_correct"], 3)
        self.assertEqual(summary["predicted_reject_rows"], 1)
        self.assertAlmostEqual(threshold, 0.80)


if __name__ == "__main__":
    unittest.main()
