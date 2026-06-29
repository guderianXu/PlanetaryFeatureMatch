import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class AnalyzeMatchSpatialCoverageTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def pfm_row(
        self,
        match_index: int,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        *,
        correct: int,
        reject_probability: float,
    ) -> dict[str, str]:
        return {
            "label": "PFM",
            "pair_index": "0",
            "base_id": "pair_a",
            "reference_variant": "nadir",
            "target_variant": "extreme_02",
            "split": "val",
            "match_index": str(match_index),
            "point_a_x_px": f"{ax:.3f}",
            "point_a_y_px": f"{ay:.3f}",
            "point_b_x_px": f"{bx:.3f}",
            "point_b_y_px": f"{by:.3f}",
            "score": "10.0",
            "error_px": "1.0" if correct else "20.0",
            "correct": str(correct),
            "valid_fraction": "0.8",
            "reject_probability": f"{reject_probability:.6f}",
        }

    def test_cli_reports_lg_only_and_candidate_only_coverage_cells(self) -> None:
        import analyze_match_spatial_coverage as coverage

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pfm_csv = root / "pfm_predictions.csv"
            lightglue_csv = root / "lightglue_details.csv"
            gate_json = root / "gate.json"
            output_dir = root / "coverage"

            self.write_csv(
                pfm_csv,
                [
                    self.pfm_row(0, 10, 10, 20, 20, correct=1, reject_probability=0.10),
                    self.pfm_row(1, 170, 10, 180, 20, correct=1, reject_probability=0.20),
                    self.pfm_row(2, 330, 10, 340, 20, correct=1, reject_probability=0.55),
                    self.pfm_row(3, 490, 10, 500, 20, correct=0, reject_probability=0.55),
                    self.pfm_row(4, 70, 650, 80, 660, correct=0, reject_probability=0.15),
                ],
            )
            self.write_csv(
                lightglue_csv,
                [
                    self.pfm_row(0, 10, 10, 20, 20, correct=1, reject_probability=0.0),
                    self.pfm_row(1, 650, 10, 660, 20, correct=1, reject_probability=0.0),
                ],
            )
            gate_json.write_text(
                json.dumps({"thresholds_by_target_variant": {"extreme_02": 0.5}}),
                encoding="utf-8",
            )

            exit_code = coverage.main(
                [
                    "--pfm-source",
                    f"val,{pfm_csv}",
                    "--lightglue-source",
                    f"val,{lightglue_csv}",
                    "--variant-gate-json",
                    str(gate_json),
                    "--output-dir",
                    str(output_dir),
                    "--grid-size",
                    "8",
                    "--image-size",
                    "768",
                ]
            )

            self.assertEqual(exit_code, 0)
            for relative in [
                "coverage_summary.csv",
                "selector_sweep.csv",
                "coverage_diagnostics.json",
                "wrong_by_variant_cell.csv",
                "rescue_tp_fp_analysis.csv",
                "index.html",
            ]:
                self.assertTrue((output_dir / relative).exists(), relative)

            with (output_dir / "coverage_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = {row["name"]: row for row in csv.DictReader(handle)}
            self.assertEqual(int(rows["PFM variant gate"]["correct"]), 2)
            self.assertEqual(int(rows["PFM variant gate"]["wrong"]), 1)
            self.assertEqual(int(rows["PFM variant gate"]["lg_only_correct_cells"]), 2)
            self.assertEqual(int(rows["PFM variant gate"]["candidate_only_correct_cells"]), 2)
            self.assertLess(float(rows["PFM variant gate"]["pair_mean_largest_cell_ratio"]), 1.0)
            self.assertGreater(float(rows["PFM unfiltered"]["pair_mean_coverage_mean"]), float(rows["PFM variant gate"]["pair_mean_coverage_mean"]))

            with (output_dir / "wrong_by_variant_cell.csv").open(newline="", encoding="utf-8") as handle:
                wrong_rows = list(csv.DictReader(handle))
            wrong_by_cell = {
                (row["target_variant"], row["view"], row["cell_x"], row["cell_y"]): row
                for row in wrong_rows
            }
            self.assertEqual(
                int(wrong_by_cell[("extreme_02", "a", "5", "0")]["wrong_rows"]),
                1,
            )
            self.assertEqual(
                int(wrong_by_cell[("extreme_02", "a", "5", "0")]["rescue_window_wrong_rows"]),
                1,
            )

            with (output_dir / "rescue_tp_fp_analysis.csv").open(newline="", encoding="utf-8") as handle:
                rescue_rows = list(csv.DictReader(handle))
            rescue_by_precision = {row["cell_pair_key"]: row for row in rescue_rows}
            self.assertEqual(
                int(rescue_by_precision["a:3:0|b:3:0"]["correct_rows"]),
                1,
            )
            self.assertEqual(
                int(rescue_by_precision["a:5:0|b:5:0"]["wrong_rows"]),
                1,
            )
            self.assertEqual(rescue_by_precision["a:3:0|b:3:0"]["expands_gate_cells"], "1")


if __name__ == "__main__":
    unittest.main()
