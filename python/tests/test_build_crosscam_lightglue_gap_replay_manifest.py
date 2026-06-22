import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class CrosscamLightGlueGapReplayManifestTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def pair_rows(self, split: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        specs = [
            ("0", "eval_high_gap", "extreme_02", "0.72"),
            ("1", "eval_low_overlap", "extreme_02", "0.12"),
            ("2", "eval_small_gap", "extreme_03", "0.81"),
        ]
        for pair_index, base_id, target_variant, valid_fraction in specs:
            rows.append(
                {
                    "pair_index": pair_index,
                    "split": split,
                    "pair_type": "cross_camera",
                    "reference_dataset_id": "h100km_fov076",
                    "reference_pose_id": f"{base_id}_nadir",
                    "reference_base_id": base_id,
                    "reference_variant": "nadir",
                    "target_dataset_id": "h100km_fov076",
                    "target_pose_id": f"{base_id}_{target_variant}",
                    "target_base_id": base_id,
                    "target_variant": target_variant,
                    "valid_fraction": valid_fraction,
                    "valid_pixels": "100",
                    "attempts": "1",
                    "crop_a_x0": "0",
                    "crop_a_y0": "0",
                    "crop_a_x1": "2048",
                    "crop_a_y1": "2048",
                    "crop_b_x0": "0",
                    "crop_b_y0": "0",
                    "crop_b_x1": "2048",
                    "crop_b_y1": "2048",
                }
            )
        return rows

    def pfm_rows(self) -> list[dict[str, str]]:
        return [
            {
                "label": "候选样本 / all-filtered",
                "base_id": "pfm_shifted_a",
                "target_variant": "extreme_02",
                "split": "dev",
                "valid_fraction": "0.72",
                "matches": "40",
                "correct": "20",
                "wrong": "20",
                "precision": "0.5",
            },
            {
                "label": "候选样本 / all-filtered",
                "base_id": "pfm_shifted_b",
                "target_variant": "extreme_02",
                "split": "dev",
                "valid_fraction": "0.12",
                "matches": "5",
                "correct": "1",
                "wrong": "4",
                "precision": "0.2",
            },
            {
                "label": "候选样本 / all-filtered",
                "base_id": "pfm_shifted_c",
                "target_variant": "extreme_03",
                "split": "dev",
                "valid_fraction": "0.81",
                "matches": "80",
                "correct": "70",
                "wrong": "10",
                "precision": "0.875",
            },
        ]

    def lightglue_rows(self) -> list[dict[str, str]]:
        return [
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "lg_a",
                "target_variant": "extreme_02",
                "split": "dev",
                "pair_index": "0",
                "valid_fraction": "0.72",
                "matches": "100",
                "correct": "95",
                "wrong": "5",
                "precision": "0.95",
            },
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "lg_b",
                "target_variant": "extreme_02",
                "split": "dev",
                "pair_index": "1",
                "valid_fraction": "0.12",
                "matches": "90",
                "correct": "88",
                "wrong": "2",
                "precision": "0.98",
            },
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "base_id": "lg_c",
                "target_variant": "extreme_03",
                "split": "dev",
                "pair_index": "2",
                "valid_fraction": "0.81",
                "matches": "95",
                "correct": "90",
                "wrong": "5",
                "precision": "0.95",
            },
            {
                "label": "LightGlue-SIFT-raw",
                "base_id": "ignored_raw",
                "target_variant": "extreme_02",
                "split": "dev",
                "pair_index": "0",
                "valid_fraction": "0.72",
                "matches": "130",
                "correct": "100",
                "wrong": "30",
                "precision": "0.77",
            },
        ]

    def train_rows(self) -> list[dict[str, str]]:
        rows = self.pair_rows("train")
        rows[0]["reference_base_id"] = "train_high_gap_a"
        rows[0]["target_base_id"] = "train_high_gap_a"
        rows[0]["target_variant"] = "extreme_02"
        rows[0]["valid_fraction"] = "0.70"
        rows[1]["reference_base_id"] = "train_high_gap_b"
        rows[1]["target_base_id"] = "train_high_gap_b"
        rows[1]["target_variant"] = "extreme_02"
        rows[1]["valid_fraction"] = "0.73"
        rows[2]["reference_base_id"] = "train_wrong_bucket"
        rows[2]["target_base_id"] = "train_wrong_bucket"
        rows[2]["target_variant"] = "extreme_03"
        rows[2]["valid_fraction"] = "0.82"
        return rows

    def test_samples_train_rows_by_row_aligned_gap_patterns(self) -> None:
        import build_crosscam_lightglue_gap_replay_manifest as replay

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_manifest = root / "dev_pairs.csv"
            pfm_summary = root / "pfm_summary.csv"
            lightglue_metrics = root / "lightglue_metrics.csv"
            train_manifest = root / "train_pairs.csv"
            output_manifest = root / "gap_replay_train.csv"
            mixed_manifest = root / "gap_replay_mixed_train.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"
            self.write_csv(pair_manifest, self.pair_rows("dev"))
            self.write_csv(pfm_summary, self.pfm_rows())
            self.write_csv(lightglue_metrics, self.lightglue_rows())
            self.write_csv(train_manifest, self.train_rows())

            exit_code = replay.main(
                [
                    "--source",
                    f"dev,{pair_manifest},{pfm_summary},{lightglue_metrics}",
                    "--train-manifest",
                    str(train_manifest),
                    "--output-manifest",
                    str(output_manifest),
                    "--mixed-base-manifest",
                    str(train_manifest),
                    "--mixed-output-manifest",
                    str(mixed_manifest),
                    "--summary-json",
                    str(summary_json),
                    "--report-html",
                    str(report_html),
                    "--repeat",
                    "2",
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["target_variant"] for row in rows}, {"extreme_02"})
            self.assertTrue(all(row["split"] == "train" for row in rows))
            self.assertFalse(any(row["reference_base_id"].startswith("eval_") for row in rows))
            self.assertTrue(all(row["phase42_gap_replay_reason"] == "pfm_recall_gap_vs_lightglue_pattern" for row in rows))
            self.assertTrue(all(row["phase42_gap_replay_valid_bucket"] == "high" for row in rows))
            self.assertTrue(all(row["phase42_gap_replay_lightglue_correct_sum"] == "95" for row in rows))
            self.assertTrue(report_html.exists())
            with mixed_manifest.open("r", encoding="utf-8", newline="") as handle:
                mixed_rows = list(csv.DictReader(handle))
            self.assertEqual(len(mixed_rows), 7)
            self.assertEqual(
                sum(1 for row in mixed_rows if row["phase42_gap_replay_reason"] == ""),
                3,
            )
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_source_rows"], 1)
            self.assertEqual(summary["output_rows"], 4)
            self.assertEqual(summary["mixed_output_rows"], 7)

    def test_rejects_lockbox_sources_by_default(self) -> None:
        import build_crosscam_lightglue_gap_replay_manifest as replay

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_manifest = root / "lockbox_pairs.csv"
            pfm_summary = root / "pfm_summary.csv"
            lightglue_metrics = root / "lightglue_metrics.csv"
            train_manifest = root / "train_pairs.csv"
            self.write_csv(pair_manifest, self.pair_rows("lockbox"))
            self.write_csv(pfm_summary, self.pfm_rows())
            self.write_csv(lightglue_metrics, self.lightglue_rows())
            self.write_csv(train_manifest, self.train_rows())

            with self.assertRaisesRegex(ValueError, "lockbox"):
                replay.main(
                    [
                        "--source",
                        f"lockbox,{pair_manifest},{pfm_summary},{lightglue_metrics}",
                        "--train-manifest",
                        str(train_manifest),
                        "--output-manifest",
                        str(root / "out.csv"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
