import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class ClusterGateDatasetTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def pair_summary_fields(self) -> list[str]:
        return [
            "label",
            "split",
            "pair_index",
            "base_id",
            "reference_variant",
            "target_variant",
            "matches",
            "correct",
            "wrong",
            "precision",
            "wrong_displacement_mad_px",
            "false_cluster",
            "high_confidence_wrong",
            "near_miss_wrong",
            "far_wrong",
        ]

    def pfm_summary_fields(self) -> list[str]:
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

    def test_build_cluster_gate_rows_uses_cluster_labels_and_feature_columns(self) -> None:
        import build_cluster_gate_dataset as cluster_gate

        pair_rows = [
            {
                "label": "PFM / filtered",
                "split": "test",
                "pair_index": "10",
                "base_id": "base_a",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "matches": "120",
                "correct": "112",
                "wrong": "8",
                "precision": "0.933333",
                "wrong_displacement_mad_px": "1.5",
                "false_cluster": "1",
                "high_confidence_wrong": "3",
                "near_miss_wrong": "5",
                "far_wrong": "0",
            },
            {
                "label": "PFM / filtered",
                "split": "test",
                "pair_index": "11",
                "base_id": "base_b",
                "reference_variant": "nadir",
                "target_variant": "mid_01",
                "matches": "90",
                "correct": "90",
                "wrong": "0",
                "precision": "1.0",
                "wrong_displacement_mad_px": "0",
                "false_cluster": "0",
                "high_confidence_wrong": "0",
                "near_miss_wrong": "0",
                "far_wrong": "0",
            },
        ]
        pfm_summary_rows = [
            {
                "label": "PFM / filtered",
                "base_id": "base_a",
                "target_variant": "extreme_03",
                "split": "test",
                "valid_fraction": "0.7",
                "matches": "120",
                "correct": "112",
                "wrong": "8",
                "precision": "0.933333",
                "score_min": "6.0",
                "score_mean": "14.0",
                "score_median": "14.5",
                "score_max": "20.0",
                "displacement_mad_px": "40",
                "homography_residual_valid": "1",
                "homography_residual_median_px": "2.0",
                "homography_residual_p90_px": "5.0",
            },
            {
                "label": "PFM / filtered",
                "base_id": "base_b",
                "target_variant": "mid_01",
                "split": "test",
                "valid_fraction": "0.8",
                "matches": "90",
                "correct": "90",
                "wrong": "0",
                "precision": "1.0",
                "score_min": "10.0",
                "score_mean": "18.0",
                "score_median": "18.5",
                "score_max": "22.0",
                "displacement_mad_px": "12",
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
                "split": "test",
                "pair_index": "0",
                "manifest_pair_index": "10",
                "matches": "80",
                "correct": "80",
                "wrong": "0",
                "precision": "1.0",
            },
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "base_b",
                "target_variant": "mid_01",
                "split": "test",
                "pair_index": "1",
                "manifest_pair_index": "11",
                "matches": "75",
                "correct": "74",
                "wrong": "1",
                "precision": "0.986667",
            },
        ]

        rows = cluster_gate.build_cluster_gate_rows(
            pair_rows,
            pfm_summary_rows,
            lightglue_rows,
            source_name="unit",
            split="formal",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reject_label"], "1")
        self.assertEqual(rows[0]["reject_reasons"], "false_cluster;high_confidence_wrong;pfm_wrong")
        self.assertEqual(rows[0]["keep_label"], "0")
        self.assertEqual(rows[0]["diagnostic_false_cluster"], "1")
        self.assertEqual(rows[0]["diagnostic_high_confidence_wrong"], "3")
        self.assertEqual(rows[0]["feature_target_is_extreme"], "1")
        self.assertEqual(rows[0]["lightglue_wrong"], "0")
        self.assertEqual(rows[1]["reject_label"], "0")
        self.assertEqual(rows[1]["keep_label"], "1")
        self.assertEqual(rows[1]["feature_target_is_extreme"], "0")
        feature_keys = [key for key in rows[0] if key.startswith("feature_")]
        self.assertNotIn("feature_lightglue_wrong", feature_keys)
        self.assertNotIn("feature_reject_label", feature_keys)
        self.assertNotIn("feature_wrong", feature_keys)
        self.assertNotIn("feature_wrong_ratio", feature_keys)
        self.assertNotIn("feature_precision_gap", feature_keys)
        self.assertNotIn("feature_false_cluster", feature_keys)
        self.assertNotIn("feature_high_confidence_wrong", feature_keys)
        self.assertNotIn("feature_near_miss_wrong", feature_keys)
        self.assertNotIn("feature_far_wrong", feature_keys)
        self.assertNotIn("feature_wrong_displacement_mad_px", feature_keys)
        self.assertEqual(rows[0]["diagnostic_wrong"], "8")
        self.assertEqual(rows[0]["diagnostic_wrong_ratio"], "0.066667")
        self.assertEqual(rows[0]["diagnostic_precision_gap"], "0.066667")

    def test_cli_writes_dataset_summary_and_html(self) -> None:
        import build_cluster_gate_dataset as cluster_gate

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_summary = root / "pair_failure_summary.csv"
            pfm_summary = root / "all_filtered_summary.csv"
            lightglue = root / "lightglue.csv"
            output_csv = root / "cluster_gate.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"
            self.write_csv(
                pair_summary,
                self.pair_summary_fields(),
                [
                    {
                        "label": "PFM",
                        "split": "test",
                        "pair_index": "0",
                        "base_id": "base_c",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_02",
                        "matches": "50",
                        "correct": "45",
                        "wrong": "5",
                        "precision": "0.9",
                        "wrong_displacement_mad_px": "2.0",
                        "false_cluster": "1",
                        "high_confidence_wrong": "0",
                        "near_miss_wrong": "5",
                        "far_wrong": "0",
                    }
                ],
            )
            self.write_csv(
                pfm_summary,
                self.pfm_summary_fields(),
                [
                    {
                        "label": "PFM",
                        "base_id": "base_c",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "valid_fraction": "0.7",
                        "matches": "50",
                        "correct": "45",
                        "wrong": "5",
                        "precision": "0.9",
                        "score_min": "8",
                        "score_mean": "15",
                        "score_median": "15",
                        "score_max": "20",
                        "displacement_mad_px": "30",
                        "homography_residual_valid": "1",
                        "homography_residual_median_px": "1.2",
                        "homography_residual_p90_px": "2.5",
                    }
                ],
            )
            self.write_csv(
                lightglue,
                self.lightglue_fields(),
                [
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "base_id": "base_c",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "pair_index": "0",
                        "manifest_pair_index": "0",
                        "matches": "40",
                        "correct": "40",
                        "wrong": "0",
                        "precision": "1.0",
                    }
                ],
            )

            exit_code = cluster_gate.main(
                [
                    "--pair-failure-summary",
                    str(pair_summary),
                    "--pfm-summary",
                    str(pfm_summary),
                    "--lightglue-metrics",
                    str(lightglue),
                    "--source-name",
                    "unit",
                    "--split",
                    "formal",
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(report_html),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_csv.exists())
            self.assertTrue(summary_json.exists())
            self.assertTrue(report_html.exists())
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["reject_rows"], 1)
            self.assertEqual(summary["false_cluster_rows"], 1)
            self.assertIn("cluster_gate.csv", report_html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
