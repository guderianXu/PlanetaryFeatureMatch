import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class FalseClusterReplayManifestTest(unittest.TestCase):
    def rejection_rows(self) -> list[dict[str, str]]:
        return [
            {
                "source_name": "phase7h_kp1536_formal_validation",
                "split": "formal",
                "pair_index": "0",
                "pair_type": "same_position_view",
                "base_id": "formal_a",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "pfm_matches": "120",
                "pfm_correct": "112",
                "pfm_wrong": "8",
                "pfm_precision": "0.933333",
                "teacher_wrong_delta": "8",
                "teacher_precision_delta": "-0.066667",
                "reject_label": "1",
                "reject_reasons": "pfm_wrong|teacher_wrong_excess|teacher_precision_advantage",
                "feature_matches": "120",
            },
            {
                "source_name": "phase7h_kp1536_formal_validation",
                "split": "validation",
                "pair_index": "1",
                "pair_type": "same_position_view",
                "base_id": "validation_b",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "pfm_matches": "90",
                "pfm_correct": "86",
                "pfm_wrong": "4",
                "pfm_precision": "0.955556",
                "teacher_wrong_delta": "3",
                "teacher_precision_delta": "-0.044444",
                "reject_label": "1",
                "reject_reasons": "pfm_wrong|teacher_wrong_excess",
                "feature_matches": "90",
            },
            {
                "source_name": "phase7h_kp1536_formal_validation",
                "split": "validation",
                "pair_index": "2",
                "pair_type": "same_position_view",
                "base_id": "validation_keep",
                "reference_variant": "nadir",
                "target_variant": "mid_01",
                "pfm_matches": "200",
                "pfm_correct": "200",
                "pfm_wrong": "0",
                "pfm_precision": "1.0",
                "teacher_wrong_delta": "0",
                "teacher_precision_delta": "0.0",
                "reject_label": "0",
                "reject_reasons": "",
                "feature_matches": "200",
            },
        ]

    def train_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        specs = [
            ("0", "train", "train_a", "nadir", "extreme_03"),
            ("1", "train", "train_b", "nadir", "extreme_03"),
            ("2", "train", "train_c", "nadir", "mid_01"),
            ("3", "validation", "not_train", "nadir", "extreme_03"),
        ]
        for pair_index, split, base_id, ref_variant, target_variant in specs:
            rows.append(
                {
                    "pair_index": pair_index,
                    "split": split,
                    "pair_type": "same_position_view",
                    "reference_dataset_id": "h100km_fov076",
                    "reference_pose_id": f"{base_id}_{ref_variant}",
                    "reference_base_id": base_id,
                    "reference_variant": ref_variant,
                    "target_dataset_id": "h100km_fov076",
                    "target_pose_id": f"{base_id}_{target_variant}",
                    "target_base_id": base_id,
                    "target_variant": target_variant,
                    "valid_fraction": "0.9",
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

    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_collect_false_cluster_patterns_and_sample_train_rows_only(self) -> None:
        import build_false_cluster_replay_manifest as replay

        patterns = replay.collect_false_cluster_patterns(
            self.rejection_rows(),
            config=replay.FalseClusterReplayConfig(min_source_wrong=2, min_source_rows=1),
        )
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].reference_variant, "nadir")
        self.assertEqual(patterns[0].target_variant, "extreme_03")
        self.assertEqual(patterns[0].source_rows, 2)
        self.assertEqual(patterns[0].wrong_sum, 12)
        self.assertIn("teacher_wrong_excess", patterns[0].reasons)

        sampled = replay.sample_train_rows_by_false_cluster_patterns(
            self.train_rows(),
            patterns,
            max_per_pattern=4,
            seed=7,
        )
        self.assertEqual(len(sampled), 2)
        self.assertTrue(all(row["split"] == "train" for row in sampled))
        self.assertTrue(all(row["target_variant"] == "extreme_03" for row in sampled))
        self.assertEqual(sampled[0]["pair_index"], "0")
        self.assertIn("false_cluster_reasons", sampled[0])
        self.assertEqual(sampled[0]["false_cluster_wrong_sum"], "12")

    def test_rejects_fresh_heldout_sources_by_default(self) -> None:
        import build_false_cluster_replay_manifest as replay

        rows = self.rejection_rows()
        rows[0]["source_name"] = "phase7h_fresh_heldout_diagnostic"
        with self.assertRaisesRegex(ValueError, "heldout"):
            replay.collect_false_cluster_patterns(rows, config=replay.FalseClusterReplayConfig())

    def test_mixed_manifest_interleaves_replay_rows_for_short_training_runs(self) -> None:
        import build_false_cluster_replay_manifest as replay

        base_rows = []
        for index in range(20):
            row = dict(self.train_rows()[0])
            row["pair_index"] = str(index)
            row["reference_pose_id"] = f"base_{index}_nadir"
            row["target_pose_id"] = f"base_{index}_extreme_03"
            row["reference_base_id"] = f"base_{index}"
            row["target_base_id"] = f"base_{index}"
            base_rows.append(row)
        replay_rows = []
        for index in range(2):
            row = dict(base_rows[index])
            row["false_cluster_reasons"] = "pfm_wrong|teacher_wrong_excess"
            row["false_cluster_score"] = "10.0"
            replay_rows.append(row)

        mixed = replay.build_mixed_manifest_rows(
            base_rows,
            replay_rows,
            target_replay_fraction=0.25,
        )

        replay_count = sum(1 for row in mixed if row.get("false_cluster_reasons"))
        self.assertGreaterEqual(replay_count / len(mixed), 0.25)
        self.assertTrue(any(row.get("false_cluster_reasons") for row in mixed[:8]))

    def test_cli_writes_replay_manifest_summary_and_report(self) -> None:
        import build_false_cluster_replay_manifest as replay

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rejection_csv = root / "rejection_dataset.csv"
            train_manifest = root / "overlap_edges_train.csv"
            output_manifest = root / "false_cluster_replay.csv"
            mixed_manifest = root / "false_cluster_mix.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"
            self.write_csv(rejection_csv, self.rejection_rows())
            self.write_csv(train_manifest, self.train_rows())

            exit_code = replay.main(
                [
                    "--rejection-dataset-csv",
                    str(rejection_csv),
                    "--train-manifest",
                    str(train_manifest),
                    "--output-manifest",
                    str(output_manifest),
                    "--mixed-base-manifest",
                    str(train_manifest),
                    "--mixed-output-manifest",
                    str(mixed_manifest),
                    "--mixed-replay-fraction",
                    "0.25",
                    "--summary-json",
                    str(summary_json),
                    "--report-html",
                    str(report_html),
                    "--max-per-pattern",
                    "2",
                    "--seed",
                    "11",
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                replay_rows = list(csv.DictReader(handle))
            self.assertEqual(len(replay_rows), 2)
            self.assertEqual(replay_rows[0]["split"], "train")
            self.assertEqual(replay_rows[0]["false_cluster_target_variant"], "extreme_03")
            self.assertTrue(mixed_manifest.exists())
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["patterns"], 1)
            self.assertEqual(summary["replay_rows"], 2)
            self.assertGreaterEqual(summary["mixed_replay_rows"] / summary["mixed_rows"], 0.25)
            self.assertIn("False-cluster replay manifest", report_html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
