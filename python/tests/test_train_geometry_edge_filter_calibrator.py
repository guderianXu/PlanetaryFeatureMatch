import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


EDGE_FIELDS = [
    "source_name",
    "label",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
    "split",
    "match_index",
    "score",
    "raw_margin",
    "accept_probability",
    "error_px",
    "correct",
    "valid_fraction",
    "geometry_valid_label",
    "geometry_invalid_label",
    "geometry_hard_negative_label",
    "geometry_visibility_label",
    "geometry_supervision_weight",
    "geometry_reprojection_error_px",
    "geometry_valid_fraction",
    "geometry_reason",
    "feature_score",
    "feature_raw_margin",
    "feature_accept_probability",
    "feature_pair_match_count",
    "feature_pair_homography_residual_px",
    "feature_target_variant_extreme_01",
    "feature_valid_fraction",
    "feature_true_geometry_error_px",
]


def _edge_row(index: int, *, split: str, valid: bool, hard_negative: bool = False) -> dict[str, str]:
    target_variant = "extreme_01"
    reject = not valid
    score = 0.90 - index * 0.02 if valid else 0.30 + index * 0.01
    residual = 1.0 + index * 0.1 if valid else 20.0 + index
    error = 2.0 if valid else 14.0 if hard_negative else 7.0
    return {
        "source_name": "phase91_train" if split == "train" else "phase91_eval",
        "label": "PFM / all-filtered",
        "pair_index": str(index // 2),
        "base_id": f"pose_{index // 2}",
        "reference_variant": "nadir",
        "target_variant": target_variant,
        "split": split,
        "match_index": str(index),
        "score": f"{score:.6f}",
        "raw_margin": f"{score - 0.20:.6f}",
        "accept_probability": f"{score:.6f}",
        "error_px": f"{error:.6f}",
        "correct": "1" if valid else "0",
        "valid_fraction": "0.200000",
        "geometry_valid_label": "1" if valid else "0",
        "geometry_invalid_label": "0" if valid else "1",
        "geometry_hard_negative_label": "1" if hard_negative else "0",
        "geometry_visibility_label": "1",
        "geometry_supervision_weight": "3.000000" if hard_negative else "1.000000",
        "geometry_reprojection_error_px": f"{error:.6f}",
        "geometry_valid_fraction": "0.200000",
        "geometry_reason": "valid_error_le_threshold" if valid else "valid_error_gt_hard_negative",
        "feature_score": f"{score:.6f}",
        "feature_raw_margin": f"{score - 0.20:.6f}",
        "feature_accept_probability": f"{score:.6f}",
        "feature_pair_match_count": "4",
        "feature_pair_homography_residual_px": f"{residual:.6f}",
        "feature_target_variant_extreme_01": "1",
        "feature_valid_fraction": "0.200000",
        "feature_true_geometry_error_px": f"{error:.6f}",
    }


def _write_edges(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TrainGeometryEdgeFilterCalibratorTest(unittest.TestCase):
    def test_build_filter_rows_maps_geometry_labels_without_feature_leakage(self) -> None:
        import train_geometry_edge_filter_calibrator as mod

        rows = [
            _edge_row(0, split="train", valid=True),
            _edge_row(1, split="train", valid=False, hard_negative=True),
        ]

        training_rows = mod.build_filter_rows(rows)
        feature_columns = mod.select_geometry_feature_columns(training_rows)

        self.assertEqual([row["reject_label"] for row in training_rows], ["0", "1"])
        self.assertEqual([row["pfm_correct"] for row in training_rows], ["1", "0"])
        self.assertEqual([row["pfm_wrong"] for row in training_rows], ["0", "1"])
        self.assertEqual(training_rows[1]["hard_negative_label"], "1")
        self.assertIn("feature_score", feature_columns)
        self.assertIn("feature_pair_homography_residual_px", feature_columns)
        self.assertNotIn("feature_valid_fraction", feature_columns)
        self.assertNotIn("feature_true_geometry_error_px", feature_columns)
        self.assertNotIn("feature_error_px", feature_columns)
        self.assertNotIn("feature_correct", feature_columns)

    def test_cli_trains_apply_compatible_model_from_geometry_edges(self) -> None:
        import apply_match_detail_filter_calibrator as apply_mod
        import train_geometry_edge_filter_calibrator as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv = root / "train_geometry_edges.csv"
            eval_csv = root / "eval_geometry_edges.csv"
            output_dir = root / "geometry_edge_filter"
            _write_edges(
                train_csv,
                [
                    _edge_row(0, split="train", valid=True),
                    _edge_row(1, split="train", valid=True),
                    _edge_row(2, split="train", valid=False, hard_negative=True),
                    _edge_row(3, split="train", valid=False, hard_negative=True),
                ],
            )
            _write_edges(
                eval_csv,
                [
                    _edge_row(4, split="val", valid=True),
                    _edge_row(5, split="val", valid=False, hard_negative=True),
                ],
            )

            exit_code = mod.main(
                [
                    "--train-geometry-edges",
                    str(train_csv),
                    "--eval-geometry-edges",
                    str(eval_csv),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "200",
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

            model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            loaded_model, include_true_geometry = apply_mod._load_model(output_dir / "model.json")
            with (output_dir / "geometry_edge_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
                prediction_rows = list(csv.DictReader(handle))
            sweep_exists = (output_dir / "threshold_sweep.csv").exists()
            report_text = (output_dir / "index.html").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(model["type"], "standardized_logistic_regression_match_filter")
        self.assertEqual(model["label_column"], "reject_label")
        self.assertFalse(model["include_true_geometry_features"])
        self.assertFalse(include_true_geometry)
        self.assertEqual(loaded_model.feature_columns, model["feature_columns"])
        self.assertNotIn("feature_valid_fraction", model["feature_columns"])
        self.assertNotIn("feature_true_geometry_error_px", model["feature_columns"])
        self.assertEqual(summary["train"]["matches"], 4)
        self.assertEqual(summary["eval"]["matches"], 2)
        self.assertEqual(summary["eval"]["kept_wrong"], 0)
        self.assertEqual(len(prediction_rows), 2)
        self.assertTrue(sweep_exists)
        self.assertIn("Geometry edge filter calibrator", report_text)

    def test_hard_negative_repeat_augments_only_hard_negative_rows(self) -> None:
        import train_geometry_edge_filter_calibrator as mod

        rows = mod.build_filter_rows(
            [
                _edge_row(0, split="train", valid=True),
                _edge_row(1, split="train", valid=False, hard_negative=True),
                _edge_row(2, split="train", valid=False, hard_negative=False),
            ]
        )

        augmented = mod.repeat_hard_negative_rows(rows, repeat=3)

        self.assertEqual(len(augmented), 5)
        self.assertEqual(
            [row["match_index"] for row in augmented],
            ["0", "1", "1", "1", "2"],
        )
        self.assertEqual(sum(1 for row in augmented if row["hard_negative_label"] == "1"), 3)


if __name__ == "__main__":
    unittest.main()
