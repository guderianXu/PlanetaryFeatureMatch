import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "sweep_pair_acceptance_hybrid.py"


class SweepPairAcceptanceHybridTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_cli_sweeps_thresholds_and_uses_lightglue_for_rejected_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pfm_summary = root / "pfm_summary.csv"
            lightglue_metrics = root / "lightglue_metrics.csv"
            output_dir = root / "hybrid"

            self.write_csv(
                pfm_summary,
                [
                    "base_id",
                    "target_variant",
                    "split",
                    "matches",
                    "correct",
                    "wrong",
                    "precision",
                    "pair_accept_probability",
                ],
                [
                    {
                        "base_id": "base_a",
                        "target_variant": "mid_01",
                        "split": "val",
                        "matches": "100",
                        "correct": "98",
                        "wrong": "2",
                        "precision": "0.98",
                        "pair_accept_probability": "0.90",
                    },
                    {
                        "base_id": "base_b",
                        "target_variant": "extreme_03",
                        "split": "val",
                        "matches": "90",
                        "correct": "60",
                        "wrong": "30",
                        "precision": "0.666667",
                        "pair_accept_probability": "0.20",
                    },
                ],
            )
            self.write_csv(
                lightglue_metrics,
                [
                    "label",
                    "base_id",
                    "target_variant",
                    "split",
                    "matches",
                    "correct",
                    "wrong",
                    "precision",
                ],
                [
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "base_id": "base_a",
                        "target_variant": "mid_01",
                        "split": "val",
                        "matches": "50",
                        "correct": "48",
                        "wrong": "2",
                        "precision": "0.96",
                    },
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "base_id": "base_b",
                        "target_variant": "extreme_03",
                        "split": "val",
                        "matches": "40",
                        "correct": "40",
                        "wrong": "0",
                        "precision": "1.0",
                    },
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--pfm-summary",
                    str(pfm_summary),
                    "--lightglue-metrics",
                    str(lightglue_metrics),
                    "--output-dir",
                    str(output_dir),
                    "--split-label",
                    "dev_val",
                    "--threshold",
                    "0.1",
                    "--threshold",
                    "0.5",
                    "--threshold",
                    "0.95",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            summary = json.loads((output_dir / "summary.json").read_text())
            with (output_dir / "threshold_summary.csv").open(newline="", encoding="utf-8") as handle:
                threshold_rows = list(csv.DictReader(handle))
            with (output_dir / "hybrid_rows_threshold_0_5.csv").open(newline="", encoding="utf-8") as handle:
                hybrid_rows = list(csv.DictReader(handle))

        self.assertEqual(summary["best_threshold"]["threshold"], "0.500000")
        self.assertEqual(summary["best_threshold"]["correct"], 138)
        self.assertEqual(summary["best_threshold"]["wrong"], 2)
        self.assertEqual(summary["lightglue"]["correct"], 88)
        self.assertEqual(summary["lightglue"]["wrong"], 2)
        self.assertEqual([row["threshold"] for row in threshold_rows], ["0.100000", "0.500000", "0.950000"])
        self.assertEqual([row["source"] for row in hybrid_rows], ["pfm", "lightglue"])
        self.assertEqual([row["hybrid_correct"] for row in hybrid_rows], ["98", "40"])
        self.assertEqual([row["hybrid_wrong"] for row in hybrid_rows], ["2", "0"])


if __name__ == "__main__":
    unittest.main()
