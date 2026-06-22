import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class ApplyTrueGeometryMatchFilterTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_filters_matches_by_true_geometry_error_and_compares_lightglue(self) -> None:
        import apply_true_geometry_match_filter as geometry_filter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            match_details = root / "all_match_details.csv"
            lightglue_metrics = root / "lightglue.csv"
            output_dir = root / "out"
            self.write_csv(
                match_details,
                [
                    {
                        "label": "PFM",
                        "split": "train",
                        "pair_index": "0",
                        "base_id": "pair_a",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_02",
                        "match_index": "0",
                        "error_px": "1.0",
                        "correct": "1",
                        "valid_fraction": "0.40",
                    },
                    {
                        "label": "PFM",
                        "split": "train",
                        "pair_index": "0",
                        "base_id": "pair_a",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_02",
                        "match_index": "1",
                        "error_px": "7.0",
                        "correct": "0",
                        "valid_fraction": "0.40",
                    },
                    {
                        "label": "PFM",
                        "split": "train",
                        "pair_index": "1",
                        "base_id": "pair_b",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_03",
                        "match_index": "0",
                        "error_px": "2.0",
                        "correct": "1",
                        "valid_fraction": "0.05",
                    },
                ],
            )
            self.write_csv(
                lightglue_metrics,
                [
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "target_variant": "extreme_02",
                        "matches": "3",
                        "correct": "2",
                        "wrong": "1",
                    },
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "target_variant": "extreme_03",
                        "matches": "0",
                        "correct": "0",
                        "wrong": "0",
                    },
                ],
            )

            exit_code = geometry_filter.main(
                [
                    "--source",
                    f"dev,{match_details},{lightglue_metrics}",
                    "--output-dir",
                    str(output_dir),
                    "--max-error-px",
                    "5.0",
                    "--min-valid-fraction",
                    "0.10",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["aggregate"]["pfm_correct"], 1)
            self.assertEqual(summary["aggregate"]["pfm_wrong"], 0)
            self.assertEqual(summary["aggregate"]["lightglue_correct"], 2)
            self.assertEqual(summary["aggregate"]["lightglue_wrong"], 1)
            self.assertEqual(summary["aggregate"]["rejected_pairs"], 1)
            self.assertEqual(summary["by_split"]["dev"]["pfm_correct"], 1)
            self.assertEqual(summary["by_split"]["dev"]["pfm_wrong"], 0)
            self.assertEqual(summary["by_split"]["dev"]["lightglue_correct"], 2)
            self.assertEqual(summary["by_split"]["dev"]["lightglue_wrong"], 1)
            self.assertEqual(summary["by_variant"]["extreme_02"]["pfm_correct"], 1)
            self.assertEqual(summary["by_variant"]["extreme_02"]["pfm_wrong"], 0)
            self.assertEqual(summary["by_variant"]["extreme_02"]["lightglue_correct"], 2)
            self.assertEqual(summary["by_variant"]["extreme_02"]["lightglue_wrong"], 1)
            self.assertEqual(summary["by_variant"]["extreme_03"]["pfm_correct"], 0)
            self.assertEqual(summary["by_variant"]["extreme_03"]["pfm_wrong"], 0)
            with (output_dir / "pair_summary.csv").open(newline="", encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 2)
            self.assertEqual(pair_rows[0]["kept_correct"], "1")
            self.assertEqual(pair_rows[0]["kept_wrong"], "0")
            self.assertEqual(pair_rows[1]["reject_reason"], "low_valid_fraction")
            self.assertTrue((output_dir / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
