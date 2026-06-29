import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class BuildSpatialCoverageHardMiningManifestTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def match_row(
        self,
        base_id: str,
        match_index: int,
        ax: float,
        ay: float,
        *,
        correct: int,
        reject_probability: float,
        target_variant: str = "extreme_02",
    ) -> dict[str, str]:
        return {
            "label": "PFM",
            "pair_index": "0",
            "base_id": base_id,
            "reference_variant": "nadir",
            "target_variant": target_variant,
            "split": "val",
            "match_index": str(match_index),
            "point_a_x_px": f"{ax:.3f}",
            "point_a_y_px": f"{ay:.3f}",
            "point_b_x_px": f"{ax + 5.0:.3f}",
            "point_b_y_px": f"{ay + 5.0:.3f}",
            "score": "12.0",
            "error_px": "1.0" if correct else "30.0",
            "correct": str(correct),
            "valid_fraction": "0.72",
            "reject_probability": f"{reject_probability:.6f}",
        }

    def train_row(self, base_id: str, target_variant: str = "extreme_02") -> dict[str, str]:
        return {
            "pair_index": "7",
            "split": "train",
            "pair_type": "same-position",
            "reference_dataset_id": "h100km_fov076",
            "reference_pose_id": f"{base_id}_nadir",
            "reference_base_id": base_id,
            "reference_variant": "nadir",
            "target_dataset_id": "h100km_fov076",
            "target_pose_id": f"{base_id}_{target_variant}",
            "target_base_id": base_id,
            "target_variant": target_variant,
            "valid_fraction": "0.70",
            "valid_pixels": "100",
            "attempts": "1",
            "crop_a_x0": "0",
            "crop_a_y0": "0",
            "crop_a_x1": "768",
            "crop_a_y1": "768",
            "crop_b_x0": "0",
            "crop_b_y0": "0",
            "crop_b_x1": "768",
            "crop_b_y1": "768",
        }

    def test_builds_train_only_replay_manifest_and_three_hard_labels(self) -> None:
        import build_spatial_coverage_hard_mining_manifest as mining

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pfm_csv = root / "pfm.csv"
            lightglue_csv = root / "lightglue.csv"
            train_manifest = root / "train_manifest.csv"
            gate_json = root / "gate.json"
            output_manifest = root / "spatial_replay.csv"
            mixed_manifest = root / "mixed.csv"
            mining_csv = root / "spatial_hard_matches.csv"
            false_csv = root / "spatial_false_matches.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"

            self.write_csv(
                pfm_csv,
                [
                    self.match_row("eval_pair", 0, 10, 10, correct=1, reject_probability=0.10),
                    self.match_row("eval_pair", 1, 180, 10, correct=1, reject_probability=0.55),
                    self.match_row("eval_pair", 2, 340, 10, correct=0, reject_probability=0.55),
                ],
            )
            self.write_csv(
                lightglue_csv,
                [
                    self.match_row("eval_pair", 0, 10, 10, correct=1, reject_probability=0.0),
                    self.match_row("eval_pair", 1, 660, 10, correct=1, reject_probability=0.0),
                ],
            )
            self.write_csv(
                train_manifest,
                [
                    self.train_row("train_keep_a"),
                    self.train_row("train_keep_b"),
                    self.train_row("eval_pair"),
                ],
            )
            gate_json.write_text(
                json.dumps({"thresholds_by_target_variant": {"extreme_02": 0.5}}),
                encoding="utf-8",
            )

            exit_code = mining.main(
                [
                    "--pfm-source",
                    f"val,{pfm_csv}",
                    "--lightglue-source",
                    f"val,{lightglue_csv}",
                    "--variant-gate-json",
                    str(gate_json),
                    "--train-manifest",
                    str(train_manifest),
                    "--output-manifest",
                    str(output_manifest),
                    "--mixed-base-manifest",
                    str(train_manifest),
                    "--mixed-output-manifest",
                    str(mixed_manifest),
                    "--mining-csv",
                    str(mining_csv),
                    "--false-match-csv",
                    str(false_csv),
                    "--summary-json",
                    str(summary_json),
                    "--report-html",
                    str(report_html),
                    "--repeat",
                    "2",
                    "--true-geometry-target-scale",
                    "1.0",
                    "--true-geometry-target-min",
                    "2",
                    "--true-geometry-target-max",
                    "10",
                    "--true-geometry-supervision-weight",
                    "0.75",
                    "--pair-accept-weight",
                    "0.25",
                    "--seed",
                    "11",
                ]
            )

            self.assertEqual(exit_code, 0)
            with mining_csv.open(newline="", encoding="utf-8") as handle:
                mined_rows = list(csv.DictReader(handle))
            labels = {row["hard_label"] for row in mined_rows}
            self.assertEqual(
                labels,
                {
                    "base_true_positive",
                    "candidate_rescue_true_positive",
                    "gate_dropped_true_positive",
                    "lightglue_only_cell_gap",
                    "coverage_rescue_false_positive",
                },
            )
            rescue_true_rows = [
                row for row in mined_rows if row["hard_label"] == "candidate_rescue_true_positive"
            ]
            self.assertEqual(len(rescue_true_rows), 1)
            self.assertEqual(rescue_true_rows[0]["cell_pair_key"], "a:1:0|b:1:0")
            self.assertEqual(rescue_true_rows[0]["expands_gate_cells"], "1")
            with false_csv.open(newline="", encoding="utf-8") as handle:
                false_rows = list(csv.DictReader(handle))
            self.assertEqual(len(false_rows), 1)
            self.assertEqual(false_rows[0]["hard_label"], "coverage_rescue_false_positive")

            with output_manifest.open(newline="", encoding="utf-8") as handle:
                replay_rows = list(csv.DictReader(handle))
            self.assertEqual(len(replay_rows), 4)
            self.assertTrue(all(row["split"] == "train" for row in replay_rows))
            self.assertFalse(any(row["reference_base_id"] == "eval_pair" for row in replay_rows))
            self.assertTrue(all(row["spatial_coverage_hard_reason"] for row in replay_rows))
            self.assertTrue(all(row["pair_accept_label"] == "1" for row in replay_rows))
            self.assertTrue(all(row["pair_accept_weight"] == "0.250000" for row in replay_rows))
            self.assertTrue(all(row["true_geometry_positive_matches"] == "3" for row in replay_rows))
            self.assertTrue(
                all(row["true_geometry_supervision_weight"] == "0.750000" for row in replay_rows)
            )
            self.assertIn("candidate_rescue_true_positive", replay_rows[0]["true_geometry_source_labels_json"])

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["hard_label_counts"]["base_true_positive"], 1)
            self.assertEqual(summary["hard_label_counts"]["candidate_rescue_true_positive"], 1)
            self.assertEqual(summary["hard_label_counts"]["gate_dropped_true_positive"], 1)
            self.assertEqual(summary["hard_label_counts"]["lightglue_only_cell_gap"], 1)
            self.assertEqual(summary["hard_label_counts"]["coverage_rescue_false_positive"], 1)
            self.assertEqual(summary["true_geometry_replay"]["extreme_02"]["target_count"], 3)
            self.assertEqual(summary["true_geometry_replay"]["extreme_02"]["source_pair_count"], 1)
            self.assertTrue(report_html.exists())
            self.assertTrue(mixed_manifest.exists())


if __name__ == "__main__":
    unittest.main()
