import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class AnalyzePfmFailureBucketsTest(unittest.TestCase):
    def write_match_details(self, path: Path) -> None:
        fieldnames = [
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
            "raw_margin",
            "accept_probability",
            "error_px",
            "correct",
            "valid_fraction",
        ]
        rows = [
            ("0", "cluster_pair", "0", "0", "5", "5", "21.0", "0.60", "0.96", "1.0", "1"),
            ("0", "cluster_pair", "10", "0", "15", "5", "20.0", "0.55", "0.94", "1.5", "1"),
            ("0", "cluster_pair", "20", "0", "50", "30", "19.5", "0.50", "0.93", "35.0", "0"),
            ("0", "cluster_pair", "25", "5", "55", "35", "18.5", "0.45", "0.91", "34.0", "0"),
            ("1", "near_pair", "0", "0", "4", "4", "17.0", "0.40", "0.90", "1.0", "1"),
            ("1", "near_pair", "10", "10", "19", "14", "8.0", "0.02", "0.30", "6.0", "0"),
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                pair_index, base_id, ax, ay, bx, by, score, raw_margin, accept, error_px, correct = row
                writer.writerow(
                    {
                        "label": "PFM / all-filtered",
                        "pair_index": pair_index,
                        "base_id": base_id,
                        "reference_variant": "nadir",
                        "target_variant": "mid_01",
                        "split": "fresh",
                        "match_index": str(index),
                        "point_a_x_px": ax,
                        "point_a_y_px": ay,
                        "point_b_x_px": bx,
                        "point_b_y_px": by,
                        "score": score,
                        "raw_margin": raw_margin,
                        "accept_probability": accept,
                        "error_px": error_px,
                        "correct": correct,
                        "valid_fraction": "0.8",
                    }
                )

    def test_cli_buckets_clustered_and_near_miss_wrong_matches(self) -> None:
        import analyze_pfm_failure_buckets as analyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            match_details = root / "all_filtered_match_details.csv"
            output_dir = root / "failure_buckets"
            self.write_match_details(match_details)

            exit_code = analyzer.main(
                [
                    "--match-details",
                    str(match_details),
                    "--output-dir",
                    str(output_dir),
                    "--cluster-wrong-min",
                    "2",
                    "--false-cluster-mad-px",
                    "3",
                    "--high-score-min",
                    "15",
                    "--high-accept-min",
                    "0.8",
                    "--high-raw-margin-min",
                    "0.2",
                    "--near-miss-px",
                    "8",
                    "--far-error-px",
                    "20",
                ]
            )

            self.assertEqual(exit_code, 0)
            for relative_path in [
                "wrong_match_buckets.csv",
                "pair_failure_summary.csv",
                "bucket_summary.csv",
                "error_bin_summary.csv",
                "summary.json",
                "index.html",
            ]:
                self.assertTrue((output_dir / relative_path).exists(), relative_path)

            with (output_dir / "wrong_match_buckets.csv").open(newline="", encoding="utf-8") as handle:
                wrong_rows = list(csv.DictReader(handle))
            self.assertEqual(len(wrong_rows), 3)
            by_pair = {}
            for row in wrong_rows:
                by_pair.setdefault(row["base_id"], []).append(row)
            self.assertTrue(all(row["primary_bucket"] == "false_cluster_high_confidence" for row in by_pair["cluster_pair"]))
            self.assertEqual(by_pair["near_pair"][0]["primary_bucket"], "near_miss")
            self.assertIn("use false-cluster replay", by_pair["cluster_pair"][0]["training_action"])

            with (output_dir / "bucket_summary.csv").open(newline="", encoding="utf-8") as handle:
                bucket_rows = {row["bucket"]: row for row in csv.DictReader(handle)}
            self.assertEqual(bucket_rows["false_cluster_high_confidence"]["wrong_matches"], "2")
            self.assertEqual(bucket_rows["near_miss"]["wrong_matches"], "1")

            with (output_dir / "error_bin_summary.csv").open(newline="", encoding="utf-8") as handle:
                error_bins = {row["error_bin"]: row for row in csv.DictReader(handle)}
            self.assertEqual(error_bins["5-6"]["wrong_matches"], "1")
            self.assertEqual(error_bins[">20"]["wrong_matches"], "2")

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["matches"], 6)
            self.assertEqual(summary["wrong"], 3)
            self.assertEqual(summary["bucket_counts"]["false_cluster_high_confidence"], 2)
            self.assertEqual(summary["bucket_counts"]["near_miss"], 1)
            self.assertEqual(summary["error_bin_counts"]["5-6"], 1)
            self.assertIn("PFM failure bucket analysis", (output_dir / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
