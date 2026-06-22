import csv
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class BuildLazyFalseMatchCsvTest(unittest.TestCase):
    def test_exports_wrong_match_details_with_lazy_pair_metadata_and_pair_cap(self) -> None:
        import build_lazy_false_match_csv as builder

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            manifest = temp_dir / "train_pairs.csv"
            match_details = temp_dir / "all_match_details.csv"
            output = temp_dir / "false_matches.csv"
            summary_json = temp_dir / "summary.json"
            report_html = temp_dir / "summary.html"

            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "pair_index",
                        "split",
                        "pair_type",
                        "reference_dataset_id",
                        "reference_pose_id",
                        "reference_base_id",
                        "reference_variant",
                        "target_dataset_id",
                        "target_pose_id",
                        "target_base_id",
                        "target_variant",
                        "crop_a_x0",
                        "crop_a_y0",
                        "crop_a_x1",
                        "crop_a_y1",
                        "crop_b_x0",
                        "crop_b_y0",
                        "crop_b_x1",
                        "crop_b_y1",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_index": "7",
                        "split": "train",
                        "pair_type": "cross_camera",
                        "reference_dataset_id": "h100km_fov076",
                        "reference_pose_id": "h100_f076_b000010_extreme_02",
                        "reference_base_id": "h100_f076_b000010",
                        "reference_variant": "extreme_02",
                        "target_dataset_id": "h100km_fov076",
                        "target_pose_id": "h100_f076_b000011_extreme_03",
                        "target_base_id": "h100_f076_b000011",
                        "target_variant": "extreme_03",
                        "crop_a_x0": "0",
                        "crop_a_y0": "0",
                        "crop_a_x1": "2048",
                        "crop_a_y1": "2048",
                        "crop_b_x0": "16",
                        "crop_b_y0": "32",
                        "crop_b_x1": "2064",
                        "crop_b_y1": "2080",
                    }
                )

            with match_details.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
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
                        "raw_similarity",
                        "raw_margin",
                        "accept_probability",
                        "error_px",
                        "correct",
                        "valid_fraction",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "label": "候选样本",
                        "pair_index": "7",
                        "split": "train",
                        "match_index": "0",
                        "point_a_x_px": "10.0",
                        "point_a_y_px": "20.0",
                        "point_b_x_px": "30.0",
                        "point_b_y_px": "40.0",
                        "score": "12.0",
                        "raw_similarity": "0.90",
                        "raw_margin": "0.20",
                        "accept_probability": "0.75",
                        "error_px": "80.0",
                        "correct": "0",
                        "valid_fraction": "0.12",
                    }
                )
                writer.writerow(
                    {
                        "label": "候选样本",
                        "pair_index": "7",
                        "split": "train",
                        "match_index": "1",
                        "point_a_x_px": "50.0",
                        "point_a_y_px": "60.0",
                        "point_b_x_px": "70.0",
                        "point_b_y_px": "80.0",
                        "score": "15.0",
                        "raw_similarity": "0.88",
                        "raw_margin": "0.30",
                        "accept_probability": "0.72",
                        "error_px": "120.0",
                        "correct": "0",
                        "valid_fraction": "0.12",
                    }
                )
                writer.writerow(
                    {
                        "label": "候选样本",
                        "pair_index": "7",
                        "split": "train",
                        "match_index": "2",
                        "point_a_x_px": "90.0",
                        "point_a_y_px": "91.0",
                        "point_b_x_px": "92.0",
                        "point_b_y_px": "93.0",
                        "score": "99.0",
                        "raw_similarity": "0.99",
                        "raw_margin": "0.99",
                        "accept_probability": "0.99",
                        "error_px": "1.0",
                        "correct": "1",
                        "valid_fraction": "0.12",
                    }
                )

            summary = builder.build_lazy_false_match_csv(
                pair_manifest_paths=[manifest],
                match_detail_paths=[match_details],
                output_csv=output,
                min_error_px=5.0,
                min_score=0.0,
                min_raw_similarity=None,
                min_accept_probability=None,
                max_per_pair=1,
                matcher="graph_matcher",
                mine_source="phase73_raw_true_geometry_wrong",
                summary_json=summary_json,
                report_html=report_html,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(1, summary["exported_rows"])
            self.assertEqual(2, summary["candidate_wrong_rows"])
            self.assertEqual(1, len(rows))
            self.assertEqual("50.000", rows[0]["ax"])
            self.assertEqual("60.000", rows[0]["ay"])
            self.assertEqual("70.000", rows[0]["bx"])
            self.assertEqual("80.000", rows[0]["by"])
            self.assertEqual("15.000000", rows[0]["score"])
            self.assertEqual("cross_camera", rows[0]["pair_type"])
            self.assertIn("lazy_pair_false_v1|cross_camera|h100km_fov076", rows[0]["lazy_pair_key"])
            self.assertIn("|0,0,2048,2048|16,32,2064,2080", rows[0]["lazy_pair_key"])
            self.assertEqual("phase73_raw_true_geometry_wrong", rows[0]["mine_source"])
            self.assertEqual("extreme_03", rows[0]["target_variant"])
            self.assertTrue(summary_json.is_file())
            self.assertTrue(report_html.is_file())

    def test_can_filter_by_target_variant_and_error_range(self) -> None:
        import build_lazy_false_match_csv as builder

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            manifest = temp_dir / "train_pairs.csv"
            match_details = temp_dir / "all_match_details.csv"
            output = temp_dir / "false_matches.csv"

            fieldnames = [
                "pair_index",
                "split",
                "pair_type",
                "reference_dataset_id",
                "reference_pose_id",
                "reference_base_id",
                "reference_variant",
                "target_dataset_id",
                "target_pose_id",
                "target_base_id",
                "target_variant",
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for pair_index, target_variant in (("0", "extreme_01"), ("1", "extreme_02")):
                    writer.writerow(
                        {
                            "pair_index": pair_index,
                            "split": "train",
                            "pair_type": "cross_camera",
                            "reference_dataset_id": "h100km_fov076",
                            "reference_pose_id": f"ref_{pair_index}",
                            "reference_base_id": f"ref_{pair_index}",
                            "reference_variant": "nadir",
                            "target_dataset_id": "h100km_fov076",
                            "target_pose_id": f"tgt_{pair_index}",
                            "target_base_id": f"tgt_{pair_index}",
                            "target_variant": target_variant,
                        }
                    )

            detail_fields = [
                "label",
                "pair_index",
                "split",
                "match_index",
                "point_a_x_px",
                "point_a_y_px",
                "point_b_x_px",
                "point_b_y_px",
                "score",
                "error_px",
                "correct",
                "valid_fraction",
            ]
            with match_details.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=detail_fields)
                writer.writeheader()
                for pair_index, match_index, error_px, score in (
                    ("0", "keep_near", "7.0", "30.0"),
                    ("0", "drop_far", "12.0", "40.0"),
                    ("1", "drop_variant", "7.0", "50.0"),
                ):
                    writer.writerow(
                        {
                            "label": "PFM",
                            "pair_index": pair_index,
                            "split": "train",
                            "match_index": match_index,
                            "point_a_x_px": "10.0",
                            "point_a_y_px": "20.0",
                            "point_b_x_px": "30.0",
                            "point_b_y_px": "40.0",
                            "score": score,
                            "error_px": error_px,
                            "correct": "0",
                            "valid_fraction": "0.20",
                        }
                    )

            summary = builder.build_lazy_false_match_csv(
                pair_manifest_paths=[manifest],
                match_detail_paths=[match_details],
                output_csv=output,
                min_error_px=5.0,
                max_error_px=10.0,
                target_variants={"extreme_01"},
                min_score=0.0,
                min_raw_similarity=None,
                min_accept_probability=None,
                max_per_pair=0,
                matcher="graph_matcher",
                mine_source="extreme01_near_boundary_false",
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(1, summary["exported_rows"])
            self.assertEqual(1, summary["candidate_wrong_rows"])
            self.assertEqual(3, summary["wrong_rows"])
            self.assertEqual(2, summary["skipped_invalid_or_threshold"])
            self.assertEqual("keep_near", rows[0]["source_match_index"])
            self.assertEqual("7.000", rows[0]["error_px"])
            self.assertEqual("extreme_01", rows[0]["target_variant"])


if __name__ == "__main__":
    unittest.main()
