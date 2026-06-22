import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class SweepMatchFilterThresholdsTest(unittest.TestCase):
    def write_predictions(self, path: Path, rows: list[dict[str, str]]) -> None:
        fieldnames = [
            "label",
            "split",
            "pair_index",
            "base_id",
            "reference_variant",
            "target_variant",
            "match_index",
            "correct",
            "reject_probability",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, split: str, variant: str, probability: float, correct: int, index: int) -> dict[str, str]:
        return {
            "label": "PFM / all-filtered",
            "split": split,
            "pair_index": str(index // 10),
            "base_id": f"{split}_{variant}_{index}",
            "reference_variant": "nadir",
            "target_variant": variant,
            "match_index": str(index),
            "correct": str(correct),
            "reject_probability": f"{probability:.6f}",
        }

    def test_cli_selects_per_variant_thresholds_and_reports_validation(self) -> None:
        import sweep_match_filter_thresholds as sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dev_predictions = root / "dev_predictions.csv"
            lockbox_predictions = root / "lockbox_predictions.csv"
            output_dir = root / "sweep"

            dev_rows = [
                self.row("dev", "extreme_02", 0.10, 1, 0),
                self.row("dev", "extreme_02", 0.20, 1, 1),
                self.row("dev", "extreme_02", 0.30, 1, 2),
                self.row("dev", "extreme_02", 0.70, 0, 3),
                self.row("dev", "extreme_03", 0.10, 1, 4),
                self.row("dev", "extreme_03", 0.30, 0, 5),
                self.row("dev", "extreme_03", 0.40, 1, 6),
            ]
            lockbox_rows = [
                self.row("lockbox", "extreme_02", 0.10, 1, 0),
                self.row("lockbox", "extreme_02", 0.25, 1, 1),
                self.row("lockbox", "extreme_02", 0.80, 0, 2),
                self.row("lockbox", "extreme_03", 0.05, 1, 3),
                self.row("lockbox", "extreme_03", 0.20, 0, 4),
            ]
            self.write_predictions(dev_predictions, dev_rows)
            self.write_predictions(lockbox_predictions, lockbox_rows)

            exit_code = sweep.main(
                [
                    "--select-source",
                    f"dev,{dev_predictions},3,0",
                    "--validation-source",
                    f"lockbox,{lockbox_predictions},2,0",
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "per-target-variant",
                    "--min-select-correct-delta",
                    "1",
                    "--max-select-wrong-delta",
                    "0",
                    "--max-thresholds-per-variant",
                    "8",
                    "--top-k",
                    "5",
                ]
            )

            self.assertEqual(exit_code, 0)
            for relative in ["threshold_sweep.csv", "summary.json", "index.html"]:
                self.assertTrue((output_dir / relative).exists(), relative)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            best = summary["best"]
            self.assertEqual(best["select"]["correct"], 4)
            self.assertEqual(best["select"]["wrong"], 0)
            self.assertEqual(best["select"]["correct_delta_vs_lightglue"], 1)
            self.assertEqual(best["validation"]["lockbox"]["correct"], 3)
            self.assertEqual(best["validation"]["lockbox"]["wrong"], 0)
            self.assertNotEqual(
                best["thresholds"]["extreme_02"],
                best["thresholds"]["extreme_03"],
            )

            with (output_dir / "threshold_sweep.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("validation_lockbox_correct_delta_vs_lightglue", rows[0])


if __name__ == "__main__":
    unittest.main()
