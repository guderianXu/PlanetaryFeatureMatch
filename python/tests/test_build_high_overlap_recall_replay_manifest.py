import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class BuildHighOverlapRecallReplayManifestTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def pair_row(
        self,
        pair_index: int,
        *,
        split: str,
        target_variant: str,
        valid_fraction: float,
    ) -> dict[str, str]:
        return {
            "pair_index": str(pair_index),
            "split": split,
            "pair_type": "cross_camera",
            "reference_dataset_id": "h100km_fov076",
            "reference_pose_id": f"h100_f076_b{pair_index:06d}_nadir",
            "reference_base_id": f"h100_f076_b{pair_index:06d}",
            "reference_variant": "nadir",
            "target_dataset_id": "h100km_fov076",
            "target_pose_id": f"h100_f076_b{pair_index + 1:06d}_{target_variant}",
            "target_base_id": f"h100_f076_b{pair_index + 1:06d}",
            "target_variant": target_variant,
            "valid_fraction": f"{valid_fraction:.6f}",
            "valid_pixels": str(int(valid_fraction * 2048 * 2048)),
            "attempts": "1",
            "crop_a_x0": "0",
            "crop_a_y0": "0",
            "crop_a_x1": "2048",
            "crop_a_y1": "2048",
            "crop_b_x0": "0",
            "crop_b_y0": "0",
            "crop_b_x1": "2048",
            "crop_b_y1": "2048",
            "pair_accept_label": "1",
            "pair_accept_weight": "1.000000",
            "geometry_accept_source_valid_fraction": f"{valid_fraction:.6f}",
            "geometry_accept_reason": "observable_valid_fraction",
        }

    def test_builds_train_only_high_overlap_replay_without_lightglue_inputs(self) -> None:
        import build_high_overlap_recall_replay_manifest as replay

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_manifest = root / "base.csv"
            output_manifest = root / "replay.csv"
            mixed_manifest = root / "mixed.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"
            self.write_csv(
                input_manifest,
                [
                    self.pair_row(0, split="train", target_variant="extreme_02", valid_fraction=0.29),
                    self.pair_row(1, split="train", target_variant="extreme_02", valid_fraction=0.55),
                    self.pair_row(2, split="train", target_variant="extreme_02", valid_fraction=0.82),
                    self.pair_row(3, split="train", target_variant="extreme_03", valid_fraction=0.35),
                    self.pair_row(4, split="val", target_variant="extreme_02", valid_fraction=0.90),
                    self.pair_row(5, split="train", target_variant="small_01", valid_fraction=0.91),
                ],
            )

            exit_code = replay.main(
                [
                    "--input-manifest",
                    str(input_manifest),
                    "--output-manifest",
                    str(output_manifest),
                    "--mixed-output-manifest",
                    str(mixed_manifest),
                    "--summary-json",
                    str(summary_json),
                    "--report-html",
                    str(report_html),
                    "--min-valid-fraction",
                    "0.30",
                    "--target-variant",
                    "extreme_02",
                    "--target-variant",
                    "extreme_03",
                    "--repeat",
                    "2",
                    "--seed",
                    "11",
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                replay_rows = list(csv.DictReader(handle))
            self.assertEqual(len(replay_rows), 6)
            self.assertEqual({row["split"] for row in replay_rows}, {"train"})
            self.assertEqual({row["target_variant"] for row in replay_rows}, {"extreme_02", "extreme_03"})
            self.assertTrue(all(float(row["valid_fraction"]) >= 0.30 for row in replay_rows))
            self.assertTrue(
                all(
                    row["phase43_recall_replay_reason"] == "true_geometry_high_overlap_recall_replay"
                    for row in replay_rows
                )
            )
            self.assertEqual({row["phase43_recall_replay_copy_index"] for row in replay_rows}, {"0", "1"})
            with mixed_manifest.open("r", encoding="utf-8", newline="") as handle:
                mixed_rows = list(csv.DictReader(handle))
            self.assertEqual(len(mixed_rows), 12)
            self.assertEqual(sum(1 for row in mixed_rows if row["phase43_recall_replay_reason"] == ""), 6)
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_input_rows"], 3)
            self.assertEqual(summary["output_rows"], 6)
            self.assertEqual(summary["mixed_output_rows"], 12)
            self.assertTrue(summary["uses_lightglue_labels"] is False)
            self.assertTrue(report_html.exists())


if __name__ == "__main__":
    unittest.main()
