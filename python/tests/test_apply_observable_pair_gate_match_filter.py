import csv
import json
import tempfile
import unittest
from pathlib import Path

from apply_observable_pair_gate_match_filter import apply_gate_match_filter, parse_args


def _dataset_row(index, feature_x, pfm_correct, pfm_wrong, lightglue_correct, lightglue_wrong, *, target_variant="mid_01"):
    return {
        "source_name": "unit",
        "split": "fresh",
        "pair_index": str(1000 + index),
        "pair_type": "same_position_view",
        "base_id": f"b{index}",
        "reference_variant": "nadir",
        "target_variant": target_variant,
        "pfm_matches": str(pfm_correct + pfm_wrong),
        "pfm_correct": str(pfm_correct),
        "pfm_wrong": str(pfm_wrong),
        "pfm_precision": "1.0",
        "lightglue_matches": str(lightglue_correct + lightglue_wrong),
        "lightglue_correct": str(lightglue_correct),
        "lightglue_wrong": str(lightglue_wrong),
        "lightglue_precision": "1.0",
        "feature_x": str(feature_x),
    }


def _detail_row(pair_ordinal, base_id, x, y, dx=10.0, dy=0.0, correct="1", *, target_variant="mid_01"):
    return {
        "label": "unit",
        "pair_index": str(pair_ordinal),
        "base_id": base_id,
        "reference_variant": "nadir",
        "target_variant": target_variant,
        "split": "fresh",
        "match_index": "0",
        "point_a_x_px": f"{x:.3f}",
        "point_a_y_px": f"{y:.3f}",
        "point_b_x_px": f"{x + dx:.3f}",
        "point_b_y_px": f"{y + dy:.3f}",
        "score": "10.0",
        "pair_logit": "10.0",
        "row_dustbin_logit": "0.0",
        "col_dustbin_logit": "0.0",
        "positive_vs_dustbin_margin": "10.0",
        "raw_similarity": "0.9",
        "raw_margin": "0.1",
        "accept_logit": "1.0",
        "accept_probability": "0.7",
        "error_px": "1.0" if correct == "1" else "5.2",
        "correct": correct,
        "valid_fraction": "1.0",
    }


class ApplyObservablePairGateMatchFilterTest(unittest.TestCase):
    def test_parse_args_merges_repeated_variant_homography_thresholds(self):
        args = parse_args(
            [
                "--dataset-csv",
                "dataset.csv",
                "--match-details",
                "details.csv",
                "--gate",
                "feature_valid_fraction >= 0",
                "--output-dir",
                "out",
                "--variant-homography-residual-px",
                "extreme_01=3.0,extreme_02=4.0",
                "--variant-homography-residual-px",
                "extreme_02=5.0,extreme_03=2.5",
            ]
        )

        self.assertEqual(
            args.variant_homography_residual_px,
            {"extreme_01": 3.0, "extreme_02": 5.0, "extreme_03": 2.5},
        )

    def test_parse_args_defaults_variant_homography_thresholds_to_empty_dict(self):
        args = parse_args(
            [
                "--dataset-csv",
                "dataset.csv",
                "--match-details",
                "details.csv",
                "--gate",
                "feature_valid_fraction >= 0",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.variant_homography_residual_px, {})

    def test_residual_filter_drops_selected_pfm_outlier_and_keeps_lightglue_fallback(self):
        dataset_rows = [
            _dataset_row(0, feature_x=1.0, pfm_correct=4, pfm_wrong=1, lightglue_correct=2, lightglue_wrong=0),
            _dataset_row(1, feature_x=0.0, pfm_correct=4, pfm_wrong=0, lightglue_correct=3, lightglue_wrong=0),
        ]
        detail_rows = [
            _detail_row(0, "b0", 0.0, 0.0),
            _detail_row(0, "b0", 10.0, 0.0),
            _detail_row(0, "b0", 0.0, 10.0),
            _detail_row(0, "b0", 10.0, 10.0),
            _detail_row(0, "b0", 5.0, 5.0, dx=10.0, dy=5.0, correct="0"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_csv = root / "dataset.csv"
            details_csv = root / "details.csv"
            output_dir = root / "out"
            with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0].keys()))
                writer.writeheader()
                writer.writerows(dataset_rows)
            with details_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
                writer.writeheader()
                writer.writerows(detail_rows)

            summary = apply_gate_match_filter(
                dataset_csv=dataset_csv,
                match_details=[details_csv],
                gate="feature_x >= 1",
                output_dir=output_dir,
                max_homography_residual_px=1.0,
            )

            self.assertEqual(summary["kept_pfm_rows"], 1)
            self.assertEqual(summary["fallback_lightglue_rows"], 1)
            self.assertEqual(summary["hybrid_correct"], 7)
            self.assertEqual(summary["hybrid_wrong"], 0)
            self.assertEqual(summary["correct_delta_vs_lightglue"], 2)
            self.assertEqual(summary["wrong_delta_vs_lightglue"], 0)
            self.assertTrue((output_dir / "hybrid_rows.csv").exists())
            saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["max_homography_residual_px"], 1.0)

    def test_selected_zero_match_pfm_row_does_not_require_match_details(self):
        dataset_rows = [
            _dataset_row(0, feature_x=1.0, pfm_correct=0, pfm_wrong=0, lightglue_correct=19, lightglue_wrong=0),
        ]
        detail_header = list(_detail_row(0, "unused", 0.0, 0.0).keys())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_csv = root / "dataset.csv"
            details_csv = root / "details.csv"
            output_dir = root / "out"
            with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0].keys()))
                writer.writeheader()
                writer.writerows(dataset_rows)
            with details_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=detail_header)
                writer.writeheader()

            summary = apply_gate_match_filter(
                dataset_csv=dataset_csv,
                match_details=[details_csv],
                gate="feature_x >= 1",
                output_dir=output_dir,
                max_homography_residual_px=4.9,
            )

            self.assertEqual(summary["kept_pfm_rows"], 1)
            self.assertEqual(summary["fallback_lightglue_rows"], 0)
            self.assertEqual(summary["hybrid_correct"], 0)
            self.assertEqual(summary["hybrid_wrong"], 0)
            self.assertEqual(summary["correct_delta_vs_lightglue"], -19)
            self.assertEqual(summary["wrong_delta_vs_lightglue"], 0)

    def test_multiple_detail_files_tolerate_missing_trailing_zero_match_pairs(self):
        dataset_rows = [
            _dataset_row(0, feature_x=1.0, pfm_correct=4, pfm_wrong=0, lightglue_correct=2, lightglue_wrong=0),
            _dataset_row(1, feature_x=1.0, pfm_correct=0, pfm_wrong=0, lightglue_correct=3, lightglue_wrong=0),
            _dataset_row(2, feature_x=1.0, pfm_correct=4, pfm_wrong=0, lightglue_correct=2, lightglue_wrong=0),
        ]
        first_detail_rows = [
            _detail_row(0, "b0", 0.0, 0.0),
            _detail_row(0, "b0", 10.0, 0.0),
            _detail_row(0, "b0", 0.0, 10.0),
            _detail_row(0, "b0", 10.0, 10.0),
        ]
        second_detail_rows = [
            _detail_row(0, "b2", 20.0, 20.0),
            _detail_row(0, "b2", 30.0, 20.0),
            _detail_row(0, "b2", 20.0, 30.0),
            _detail_row(0, "b2", 30.0, 30.0),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_csv = root / "dataset.csv"
            first_details = root / "first_details.csv"
            second_details = root / "second_details.csv"
            output_dir = root / "out"
            with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0].keys()))
                writer.writeheader()
                writer.writerows(dataset_rows)
            for path, rows in ((first_details, first_detail_rows), (second_details, second_detail_rows)):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

            summary = apply_gate_match_filter(
                dataset_csv=dataset_csv,
                match_details=[first_details, second_details],
                gate="feature_x >= 1",
                output_dir=output_dir,
                max_homography_residual_px=1.0,
            )

            self.assertEqual(summary["kept_pfm_rows"], 3)
            self.assertEqual(summary["hybrid_correct"], 8)
            self.assertEqual(summary["hybrid_wrong"], 0)

    def test_variant_thresholds_override_default_homography_residual(self):
        dataset_rows = [
            _dataset_row(
                0,
                feature_x=1.0,
                pfm_correct=4,
                pfm_wrong=1,
                lightglue_correct=0,
                lightglue_wrong=0,
                target_variant="extreme_01",
            ),
            _dataset_row(
                1,
                feature_x=1.0,
                pfm_correct=4,
                pfm_wrong=1,
                lightglue_correct=0,
                lightglue_wrong=0,
                target_variant="extreme_02",
            ),
        ]
        detail_rows = []
        for pair_ordinal, base_id, target_variant in ((0, "b0", "extreme_01"), (1, "b1", "extreme_02")):
            detail_rows.extend(
                [
                    _detail_row(pair_ordinal, base_id, 0.0, 0.0, target_variant=target_variant),
                    _detail_row(pair_ordinal, base_id, 10.0, 0.0, target_variant=target_variant),
                    _detail_row(pair_ordinal, base_id, 0.0, 10.0, target_variant=target_variant),
                    _detail_row(pair_ordinal, base_id, 10.0, 10.0, target_variant=target_variant),
                    _detail_row(
                        pair_ordinal,
                        base_id,
                        5.0,
                        5.0,
                        dx=10.0,
                        dy=5.0,
                        correct="0",
                        target_variant=target_variant,
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_csv = root / "dataset.csv"
            details_csv = root / "details.csv"
            output_dir = root / "out"
            with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0].keys()))
                writer.writeheader()
                writer.writerows(dataset_rows)
            with details_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
                writer.writeheader()
                writer.writerows(detail_rows)

            summary = apply_gate_match_filter(
                dataset_csv=dataset_csv,
                match_details=[details_csv],
                gate="feature_x >= 1",
                output_dir=output_dir,
                max_homography_residual_px=1.0,
                variant_homography_residual_px={"extreme_02": 10.0},
            )

            self.assertEqual(summary["hybrid_correct"], 8)
            self.assertEqual(summary["hybrid_wrong"], 1)
            self.assertEqual(summary["variant_homography_residual_px"], {"extreme_02": 10.0})


if __name__ == "__main__":
    unittest.main()
