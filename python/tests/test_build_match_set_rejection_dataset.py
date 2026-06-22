import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class MatchSetRejectionDatasetTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def pair_fields(self) -> list[str]:
        return [
            "pair_index",
            "split",
            "pair_type",
            "reference_base_id",
            "reference_variant",
            "target_base_id",
            "target_variant",
            "valid_fraction",
        ]

    def summary_fields(self) -> list[str]:
        return [
            "label",
            "base_id",
            "target_variant",
            "split",
            "valid_fraction",
            "matches",
            "correct",
            "wrong",
            "precision",
            "score_min",
            "score_mean",
            "score_median",
            "score_max",
            "bbox_area_a_px2",
            "bbox_area_b_px2",
            "displacement_median_px",
            "displacement_mad_px",
            "homography_residual_valid",
            "homography_residual_median_px",
            "homography_residual_p90_px",
        ]

    def lightglue_fields(self) -> list[str]:
        return [
            "label",
            "base_id",
            "target_variant",
            "split",
            "pair_index",
            "manifest_pair_index",
            "matches",
            "correct",
            "wrong",
            "precision",
        ]

    def match_detail_fields(self) -> list[str]:
        return [
            "label",
            "pair_index",
            "base_id",
            "reference_variant",
            "target_variant",
            "split",
            "match_index",
            "point_a_x_px",
            "point_a_y_px",
            "point_b_x_px",
            "point_b_y_px",
            "score",
            "pair_logit",
            "row_dustbin_logit",
            "col_dustbin_logit",
            "positive_vs_dustbin_margin",
            "raw_similarity",
            "raw_margin",
            "accept_logit",
            "accept_probability",
            "error_px",
            "correct",
            "valid_fraction",
        ]

    def test_build_rejection_rows_aligns_pfm_summary_with_lightglue_teacher(self) -> None:
        import build_match_set_rejection_dataset as rejection_mod

        pair_rows = [
            {
                "pair_index": "10",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_base_id": "base_a",
                "reference_variant": "nadir",
                "target_base_id": "base_a",
                "target_variant": "extreme_03",
                "valid_fraction": "0.75",
            },
            {
                "pair_index": "11",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_base_id": "base_b",
                "reference_variant": "nadir",
                "target_base_id": "base_b",
                "target_variant": "mid_01",
                "valid_fraction": "0.82",
            },
        ]
        pfm_rows = [
            {
                "label": "PFM / all-filtered",
                "base_id": "base_a",
                "target_variant": "extreme_03",
                "split": "val",
                "valid_fraction": "0.75",
                "matches": "120",
                "correct": "116",
                "wrong": "4",
                "precision": "0.966667",
                "score_min": "8.5",
                "score_mean": "17.5",
                "score_median": "18.0",
                "score_max": "23.0",
                "bbox_area_a_px2": "400",
                "bbox_area_b_px2": "200",
                "displacement_median_px": "90",
                "displacement_mad_px": "25",
                "homography_residual_valid": "1",
                "homography_residual_median_px": "2.2",
                "homography_residual_p90_px": "4.5",
            },
            {
                "label": "PFM / all-filtered",
                "base_id": "base_b",
                "target_variant": "mid_01",
                "split": "val",
                "valid_fraction": "0.82",
                "matches": "80",
                "correct": "80",
                "wrong": "0",
                "precision": "1.0",
                "score_min": "12.0",
                "score_mean": "20.0",
                "score_median": "20.5",
                "score_max": "24.0",
                "bbox_area_a_px2": "300",
                "bbox_area_b_px2": "300",
                "displacement_median_px": "60",
                "displacement_mad_px": "10",
                "homography_residual_valid": "1",
                "homography_residual_median_px": "1.0",
                "homography_residual_p90_px": "2.0",
            },
        ]
        lightglue_rows = [
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "base_a",
                "target_variant": "extreme_03",
                "split": "val",
                "pair_index": "0",
                "manifest_pair_index": "10",
                "matches": "90",
                "correct": "90",
                "wrong": "0",
                "precision": "1.0",
            },
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "base_b",
                "target_variant": "mid_01",
                "split": "val",
                "pair_index": "1",
                "manifest_pair_index": "11",
                "matches": "75",
                "correct": "74",
                "wrong": "1",
                "precision": "0.986667",
            },
        ]
        config = rejection_mod.RejectionLabelConfig(
            reject_wrong_threshold=3,
            reject_precision_threshold=0.99,
            teacher_wrong_excess_threshold=2,
            keep_max_wrong=1,
            keep_min_precision=0.995,
        )

        rows = rejection_mod.build_rejection_rows(
            pair_rows,
            pfm_rows,
            lightglue_rows,
            split="val",
            source_name="phase7h_formal_val",
            teacher_label="LightGlue-SIFT-MAGSAC-min16",
            config=config,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["pair_index"], "10")
        self.assertEqual(rows[0]["base_id"], "base_a")
        self.assertEqual(rows[0]["target_variant"], "extreme_03")
        self.assertEqual(rows[0]["reject_label"], "1")
        self.assertIn("pfm_wrong", rows[0]["reject_reasons"])
        self.assertIn("teacher_wrong_excess", rows[0]["reject_reasons"])
        self.assertEqual(rows[0]["teacher_wrong_delta"], "4")
        self.assertEqual(rows[0]["lightglue_wrong"], "0")
        self.assertEqual(rows[0]["feature_matches"], "120")
        self.assertEqual(rows[0]["feature_target_is_extreme"], "1")
        self.assertEqual(rows[0]["feature_target_is_extreme_01"], "0")
        self.assertEqual(rows[0]["feature_target_is_extreme_02"], "0")
        self.assertEqual(rows[0]["feature_target_is_extreme_03"], "1")
        self.assertEqual(rows[0]["feature_bbox_area_ratio"], "0.500000")
        self.assertEqual(rows[1]["pair_index"], "11")
        self.assertEqual(rows[1]["reject_label"], "0")
        self.assertEqual(rows[1]["keep_label"], "1")
        self.assertEqual(rows[1]["teacher_wrong_delta"], "-1")
        self.assertEqual(rows[1]["feature_target_is_extreme"], "0")
        self.assertEqual(rows[1]["feature_target_is_extreme_01"], "0")
        self.assertEqual(rows[1]["feature_target_is_extreme_02"], "0")
        self.assertEqual(rows[1]["feature_target_is_extreme_03"], "0")

    def test_cli_writes_dataset_summary_and_html_report(self) -> None:
        import build_match_set_rejection_dataset as rejection_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_manifest = root / "pairs.csv"
            pfm_summary = root / "pfm.csv"
            lightglue_metrics = root / "lightglue.csv"
            output_csv = root / "dataset.csv"
            summary_json = root / "summary.json"
            output_html = root / "index.html"
            self.write_csv(
                pair_manifest,
                self.pair_fields(),
                [
                    {
                        "pair_index": "0",
                        "split": "test",
                        "pair_type": "same_position_view",
                        "reference_base_id": "base_c",
                        "reference_variant": "nadir",
                        "target_base_id": "base_c",
                        "target_variant": "extreme_02",
                        "valid_fraction": "0.70",
                    }
                ],
            )
            self.write_csv(
                pfm_summary,
                self.summary_fields(),
                [
                    {
                        "label": "PFM / all-filtered",
                        "base_id": "base_c",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "valid_fraction": "0.70",
                        "matches": "40",
                        "correct": "35",
                        "wrong": "5",
                        "precision": "0.875",
                        "score_min": "7.0",
                        "score_mean": "16.0",
                        "score_median": "16.5",
                        "score_max": "22.0",
                        "bbox_area_a_px2": "100",
                        "bbox_area_b_px2": "50",
                        "displacement_median_px": "75",
                        "displacement_mad_px": "30",
                        "homography_residual_valid": "1",
                        "homography_residual_median_px": "2.5",
                        "homography_residual_p90_px": "4.8",
                    }
                ],
            )
            self.write_csv(
                lightglue_metrics,
                self.lightglue_fields(),
                [
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "base_id": "base_c",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "pair_index": "0",
                        "manifest_pair_index": "0",
                        "matches": "30",
                        "correct": "30",
                        "wrong": "0",
                        "precision": "1.0",
                    }
                ],
            )

            exit_code = rejection_mod.main(
                [
                    "--source",
                    f"test,{pair_manifest},{pfm_summary},{lightglue_metrics}",
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(output_html),
                    "--reject-wrong-threshold",
                    "3",
                    "--reject-precision-threshold",
                    "0.99",
                    "--teacher-wrong-excess-threshold",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["reject_label"], "1")
            self.assertIn("feature_homography_residual_p90_px", rows[0])
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["reject_rows"], 1)
            self.assertEqual(summary["by_variant"]["extreme_02"]["reject_rows"], 1)
            html_text = output_html.read_text(encoding="utf-8")
            self.assertIn("Match-set rejection dataset", html_text)
            self.assertIn("extreme_02", html_text)

    def test_build_rejection_rows_adds_match_detail_features_without_truth_leakage(self) -> None:
        import build_match_set_rejection_dataset as rejection_mod

        pair_rows = [
            {
                "pair_index": "0",
                "split": "test",
                "pair_type": "same_position_view",
                "reference_base_id": "base_detail",
                "reference_variant": "nadir",
                "target_base_id": "base_detail",
                "target_variant": "extreme_03",
                "valid_fraction": "0.70",
            }
        ]
        pfm_rows = [
            {
                "label": "PFM / all-filtered",
                "base_id": "base_detail",
                "target_variant": "extreme_03",
                "split": "test",
                "valid_fraction": "0.70",
                "matches": "3",
                "correct": "2",
                "wrong": "1",
                "precision": "0.666667",
                "score_min": "9.0",
                "score_mean": "12.0",
                "score_median": "12.0",
                "score_max": "15.0",
                "bbox_area_a_px2": "100",
                "bbox_area_b_px2": "80",
                "displacement_median_px": "50",
                "displacement_mad_px": "10",
                "homography_residual_valid": "1",
                "homography_residual_median_px": "1.0",
                "homography_residual_p90_px": "2.0",
            }
        ]
        detail_rows = [
            {
                "label": "PFM / all-filtered",
                "pair_index": "0",
                "base_id": "base_detail",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "split": "test",
                "match_index": "0",
                "point_a_x_px": "0",
                "point_a_y_px": "0",
                "point_b_x_px": "10",
                "point_b_y_px": "5",
                "score": "15",
                "pair_logit": "16",
                "row_dustbin_logit": "0.5",
                "col_dustbin_logit": "0.5",
                "positive_vs_dustbin_margin": "15",
                "raw_similarity": "0.90",
                "raw_margin": "0.10",
                "accept_logit": "1.4",
                "accept_probability": "0.80",
                "error_px": "0.5",
                "correct": "1",
                "valid_fraction": "0.70",
            },
            {
                "label": "PFM / all-filtered",
                "pair_index": "0",
                "base_id": "base_detail",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "split": "test",
                "match_index": "1",
                "point_a_x_px": "2",
                "point_a_y_px": "1",
                "point_b_x_px": "13",
                "point_b_y_px": "7",
                "score": "12",
                "pair_logit": "13",
                "row_dustbin_logit": "0.5",
                "col_dustbin_logit": "0.5",
                "positive_vs_dustbin_margin": "12",
                "raw_similarity": "0.85",
                "raw_margin": "0.03",
                "accept_logit": "0.4",
                "accept_probability": "0.60",
                "error_px": "8.0",
                "correct": "0",
                "valid_fraction": "0.70",
            },
            {
                "label": "PFM / all-filtered",
                "pair_index": "0",
                "base_id": "base_detail",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "split": "test",
                "match_index": "2",
                "point_a_x_px": "4",
                "point_a_y_px": "2",
                "point_b_x_px": "30",
                "point_b_y_px": "20",
                "score": "9",
                "pair_logit": "10",
                "row_dustbin_logit": "0.5",
                "col_dustbin_logit": "0.5",
                "positive_vs_dustbin_margin": "9",
                "raw_similarity": "0.75",
                "raw_margin": "0.01",
                "accept_logit": "-0.4",
                "accept_probability": "0.40",
                "error_px": "20.0",
                "correct": "0",
                "valid_fraction": "0.70",
            },
        ]

        rows = rejection_mod.build_rejection_rows(
            pair_rows,
            pfm_rows,
            [],
            match_detail_rows=detail_rows,
            split="test",
            config=rejection_mod.RejectionLabelConfig(reject_wrong_threshold=1),
        )

        self.assertEqual(rows[0]["feature_detail_count"], "3")
        self.assertEqual(rows[0]["feature_detail_raw_margin_min"], "0.010000")
        self.assertEqual(rows[0]["feature_detail_accept_probability_mean"], "0.600000")
        self.assertEqual(rows[0]["feature_detail_low_accept_fraction"], "0.333333")
        self.assertEqual(rows[0]["feature_detail_low_raw_margin_fraction"], "0.666667")
        self.assertEqual(rows[0]["feature_detail_displacement_dx_mad_px"], "1.000000")
        self.assertEqual(rows[0]["feature_detail_displacement_dy_mad_px"], "1.000000")
        feature_keys = [key for key in rows[0] if key.startswith("feature_")]
        self.assertNotIn("feature_detail_error_px_mean", feature_keys)
        self.assertNotIn("feature_detail_correct", feature_keys)
        self.assertNotIn("feature_detail_wrong", feature_keys)

    def test_parse_source_accepts_optional_match_details_path(self) -> None:
        import build_match_set_rejection_dataset as rejection_mod

        source = rejection_mod.parse_source("validation,pairs.csv,pfm.csv,lightglue.csv,details.csv")

        self.assertEqual(source.split, "validation")
        self.assertEqual(str(source.pair_manifest), "pairs.csv")
        self.assertEqual(str(source.pfm_summary), "pfm.csv")
        self.assertEqual(str(source.lightglue_metrics), "lightglue.csv")
        self.assertEqual(str(source.match_details), "details.csv")

    def test_build_rejection_rows_falls_back_to_ordinal_when_pair_index_collides(self) -> None:
        import build_match_set_rejection_dataset as rejection_mod

        pair_rows = [
            {
                "pair_index": "49",
                "split": "test",
                "pair_type": "same_position_view",
                "reference_base_id": "base_collision",
                "reference_variant": "nadir",
                "target_base_id": "base_collision",
                "target_variant": "extreme_03",
                "valid_fraction": "0.50",
            }
        ]
        pfm_rows = [
            {
                "base_id": "base_collision",
                "target_variant": "extreme_03",
                "split": "test",
                "matches": "10",
                "correct": "9",
                "wrong": "1",
                "precision": "0.9",
            }
        ]
        lightglue_rows = [
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "wrong_pair_for_key_49",
                "target_variant": "mid_01",
                "split": "test",
                "pair_index": "49",
                "manifest_pair_index": "49",
                "matches": "99",
                "correct": "99",
                "wrong": "0",
                "precision": "1.0",
            },
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "base_collision",
                "target_variant": "extreme_03",
                "split": "test",
                "pair_index": "0",
                "manifest_pair_index": "0",
                "matches": "7",
                "correct": "7",
                "wrong": "0",
                "precision": "1.0",
            },
        ]

        rows = rejection_mod.build_rejection_rows(
            pair_rows,
            pfm_rows,
            lightglue_rows,
            split="test",
            teacher_label="LightGlue-SIFT-MAGSAC-min16",
            config=rejection_mod.RejectionLabelConfig(reject_wrong_threshold=3),
        )

        self.assertEqual(rows[0]["lightglue_matches"], "7")
        self.assertEqual(rows[0]["teacher_match_delta"], "3")

    def test_build_rejection_rows_matches_cross_camera_teacher_target_base_id(self) -> None:
        import build_match_set_rejection_dataset as rejection_mod

        pair_rows = [
            {
                "pair_index": "0",
                "split": "train",
                "pair_type": "cross_camera",
                "reference_base_id": "reference_base",
                "reference_variant": "extreme_01",
                "target_base_id": "target_base",
                "target_variant": "extreme_02",
                "valid_fraction": "0.25",
            }
        ]
        pfm_rows = [
            {
                "base_id": "reference_base",
                "target_variant": "extreme_02",
                "split": "train",
                "matches": "100",
                "correct": "80",
                "wrong": "20",
                "precision": "0.8",
            }
        ]
        lightglue_rows = [
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "target_base",
                "target_variant": "extreme_02",
                "split": "train",
                "pair_index": "0",
                "manifest_pair_index": "0",
                "matches": "70",
                "correct": "68",
                "wrong": "2",
                "precision": "0.971429",
            }
        ]

        rows = rejection_mod.build_rejection_rows(
            pair_rows,
            pfm_rows,
            lightglue_rows,
            split="dev",
            teacher_label="LightGlue-SIFT-MAGSAC-min16",
            config=rejection_mod.RejectionLabelConfig(reject_wrong_threshold=3),
        )

        self.assertEqual(rows[0]["base_id"], "reference_base")
        self.assertEqual(rows[0]["lightglue_matches"], "70")
        self.assertEqual(rows[0]["lightglue_correct"], "68")
        self.assertEqual(rows[0]["lightglue_wrong"], "2")
        self.assertEqual(rows[0]["teacher_wrong_delta"], "18")

    def test_build_rejection_rows_prefers_kept_metrics_from_match_detail_filter_summary(self) -> None:
        import build_match_set_rejection_dataset as rejection_mod

        pair_rows = [
            {
                "pair_index": "0",
                "split": "train",
                "pair_type": "cross_camera",
                "reference_base_id": "reference_base",
                "reference_variant": "extreme_01",
                "target_base_id": "target_base",
                "target_variant": "extreme_02",
                "valid_fraction": "0.25",
            }
        ]
        pfm_rows = [
            {
                "base_id": "reference_base",
                "target_variant": "extreme_02",
                "split": "train",
                "matches": "100",
                "correct": "20",
                "wrong": "80",
                "precision": "0.2",
                "kept_matches": "21",
                "kept_correct": "19",
                "kept_wrong": "2",
                "kept_precision": "0.904762",
            }
        ]

        rows = rejection_mod.build_rejection_rows(
            pair_rows,
            pfm_rows,
            [],
            split="dev",
            config=rejection_mod.RejectionLabelConfig(
                reject_wrong_threshold=3,
                reject_precision_threshold=0.80,
            ),
        )

        self.assertEqual(rows[0]["pfm_matches"], "21")
        self.assertEqual(rows[0]["pfm_correct"], "19")
        self.assertEqual(rows[0]["pfm_wrong"], "2")
        self.assertEqual(rows[0]["pfm_precision"], "0.904762")
        self.assertEqual(rows[0]["reject_label"], "0")
        self.assertEqual(rows[0]["feature_matches"], "21")


if __name__ == "__main__":
    unittest.main()
