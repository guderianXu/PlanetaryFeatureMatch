import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class FailureBucketReplayManifestTest(unittest.TestCase):
    def pair_summary_rows(self) -> list[dict[str, str]]:
        return [
            {
                "label": "PFM / all-filtered",
                "split": "formal",
                "pair_index": "0",
                "base_id": "formal_extreme",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "matches": "80",
                "correct": "50",
                "wrong": "30",
                "precision": "0.625",
                "false_cluster": "1",
                "high_confidence_wrong": "4",
                "near_miss_wrong": "8",
                "far_wrong": "3",
                "primary_buckets": "false_cluster:22;false_cluster_high_confidence:4;near_miss:4",
            },
            {
                "label": "PFM / all-filtered",
                "split": "validation",
                "pair_index": "1",
                "base_id": "validation_mid",
                "reference_variant": "nadir",
                "target_variant": "mid_01",
                "matches": "120",
                "correct": "116",
                "wrong": "4",
                "precision": "0.966667",
                "false_cluster": "0",
                "high_confidence_wrong": "0",
                "near_miss_wrong": "4",
                "far_wrong": "0",
                "primary_buckets": "near_miss:4",
            },
            {
                "label": "PFM / all-filtered",
                "split": "validation",
                "pair_index": "2",
                "base_id": "clean_pair",
                "reference_variant": "nadir",
                "target_variant": "mid_02",
                "matches": "100",
                "correct": "100",
                "wrong": "0",
                "precision": "1.0",
                "false_cluster": "0",
                "high_confidence_wrong": "0",
                "near_miss_wrong": "0",
                "far_wrong": "0",
                "primary_buckets": "",
            },
        ]

    def train_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        specs = [
            ("0", "train", "train_extreme_a", "nadir", "extreme_03"),
            ("1", "train", "train_extreme_b", "nadir", "extreme_03"),
            ("2", "train", "train_mid", "nadir", "mid_01"),
            ("3", "test", "not_train", "nadir", "extreme_03"),
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
                    "true_geometry_positive_matches": "64",
                    "true_geometry_filtered_matches": "64",
                    "true_geometry_wrong_matches": "0",
                    "true_geometry_supervision_weight": "1.000000",
                    "true_geometry_supervision_source": "unit_test",
                    "true_geometry_supervision_reason": "clean_overlap",
                    "pair_accept_label": "1",
                    "pair_accept_weight": "1.000000",
                }
            )
        return rows

    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_collects_patterns_and_samples_train_rows_only(self) -> None:
        import build_failure_bucket_replay_manifest as replay

        patterns = replay.collect_failure_bucket_patterns(
            self.pair_summary_rows(),
            source_name="phase7h_formal_validation",
            config=replay.FailureBucketReplayConfig(min_wrong=2, min_bucket_wrong=2),
        )
        self.assertEqual([pattern.target_variant for pattern in patterns], ["extreme_03", "mid_01"])
        self.assertTrue(patterns[0].has_false_cluster)
        self.assertFalse(patterns[1].has_false_cluster)

        sampled = replay.sample_train_rows_by_failure_bucket_patterns(
            self.train_rows(),
            patterns,
            max_per_pattern=4,
            seed=17,
        )

        self.assertEqual(len(sampled), 3)
        self.assertTrue(all(row["split"] == "train" for row in sampled))
        extreme_rows = [row for row in sampled if row["target_variant"] == "extreme_03"]
        mid_rows = [row for row in sampled if row["target_variant"] == "mid_01"]
        self.assertTrue(all(row["false_cluster_reasons"] for row in extreme_rows))
        self.assertTrue(all(row["failure_bucket_reasons"] for row in sampled))
        self.assertEqual(mid_rows[0]["false_cluster_reasons"], "")
        self.assertIn("near_miss", mid_rows[0]["failure_bucket_reasons"])

    def test_rejects_fresh_heldout_sources_by_default(self) -> None:
        import build_failure_bucket_replay_manifest as replay

        with self.assertRaisesRegex(ValueError, "heldout"):
            replay.collect_failure_bucket_patterns(
                self.pair_summary_rows(),
                source_name="phase10_fresh_heldout",
                config=replay.FailureBucketReplayConfig(),
            )

    def test_merge_patterns_combines_sources_by_training_key(self) -> None:
        import build_failure_bucket_replay_manifest as replay

        formal = replay.collect_failure_bucket_patterns(
            self.pair_summary_rows()[:1],
            source_name="phase7h_formal",
            config=replay.FailureBucketReplayConfig(),
        )
        validation_rows = [dict(self.pair_summary_rows()[0])]
        validation_rows[0]["wrong"] = "2"
        validation_rows[0]["primary_buckets"] = "near_miss:2"
        validation = replay.collect_failure_bucket_patterns(
            validation_rows,
            source_name="phase7h_validation",
            config=replay.FailureBucketReplayConfig(),
        )

        merged = replay.merge_failure_bucket_patterns([*formal, *validation])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].wrong_sum, formal[0].wrong_sum + validation[0].wrong_sum)
        self.assertEqual(merged[0].source_rows, 2)
        self.assertTrue(any("phase7h_formal" in source for source in merged[0].sources))
        self.assertTrue(any("phase7h_validation" in source for source in merged[0].sources))

    def test_cli_writes_replay_mixed_manifest_summary_and_report(self) -> None:
        import build_failure_bucket_replay_manifest as replay

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_summary = root / "pair_failure_summary.csv"
            train_manifest = root / "overlap_edges_train.csv"
            output_manifest = root / "failure_bucket_replay.csv"
            mixed_manifest = root / "failure_bucket_mix.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"
            self.write_csv(pair_summary, self.pair_summary_rows())
            self.write_csv(train_manifest, self.train_rows())

            exit_code = replay.main(
                [
                    "--pair-failure-summary",
                    f"phase7h_formal,{pair_summary}",
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
                    "5",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_manifest.exists())
            self.assertTrue(mixed_manifest.exists())
            self.assertTrue(report_html.exists())
            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                replay_rows = list(csv.DictReader(handle))
            self.assertEqual(len(replay_rows), 3)
            self.assertTrue(any(row["failure_bucket_target_variant"] == "extreme_03" for row in replay_rows))
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["patterns"], 2)
            self.assertEqual(summary["replay_rows"], 3)
            self.assertGreater(summary["mixed_replay_fraction_actual"], 0.0)

    def test_cli_preserves_true_geometry_supervision_fields(self) -> None:
        import build_failure_bucket_replay_manifest as replay

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_summary = root / "pair_failure_summary.csv"
            train_manifest = root / "true_geometry_train.csv"
            output_manifest = root / "failure_bucket_replay.csv"
            mixed_manifest = root / "failure_bucket_mix.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"
            self.write_csv(pair_summary, self.pair_summary_rows())
            self.write_csv(train_manifest, self.train_rows())

            replay.main(
                [
                    "--pair-failure-summary",
                    f"phase70_dev,{pair_summary}",
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
                    "5",
                ]
            )

            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                replay_rows = list(csv.DictReader(handle))
            with mixed_manifest.open("r", encoding="utf-8", newline="") as handle:
                mixed_reader = csv.DictReader(handle)
                mixed_fieldnames = mixed_reader.fieldnames or []
                mixed_rows = list(mixed_reader)

            self.assertTrue(replay_rows)
            self.assertIn("true_geometry_positive_matches", replay_rows[0])
            self.assertEqual(replay_rows[0]["true_geometry_positive_matches"], "64")
            self.assertIn("true_geometry_supervision_reason", replay_rows[0])
            self.assertIn("pair_accept_label", replay_rows[0])
            self.assertIn("true_geometry_positive_matches", mixed_fieldnames)
            replay_mixed_rows = [row for row in mixed_rows if row.get("failure_bucket_reasons")]
            self.assertTrue(replay_mixed_rows)
            self.assertEqual(replay_mixed_rows[0]["true_geometry_positive_matches"], "64")


if __name__ == "__main__":
    unittest.main()
