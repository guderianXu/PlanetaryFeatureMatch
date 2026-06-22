import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class GeometryOverlapGateSweepTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_sweeps_valid_fraction_threshold_and_compares_lightglue(self) -> None:
        import sweep_geometry_overlap_gate as sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pfm_summary = root / "pfm.csv"
            lightglue_metrics = root / "lg.csv"
            output_csv = root / "threshold_summary.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"
            self.write_csv(
                pfm_summary,
                [
                    {"valid_fraction": "0.05", "matches": "30", "correct": "0", "wrong": "30"},
                    {"valid_fraction": "0.20", "matches": "50", "correct": "48", "wrong": "2"},
                    {"valid_fraction": "0.70", "matches": "80", "correct": "76", "wrong": "4"},
                ],
            )
            self.write_csv(
                lightglue_metrics,
                [
                    {"label": "LightGlue-SIFT-raw", "matches": "40", "correct": "20", "wrong": "20"},
                    {"label": "LightGlue-SIFT-MAGSAC-min16", "matches": "20", "correct": "19", "wrong": "1"},
                    {"label": "LightGlue-SIFT-MAGSAC-min16", "matches": "40", "correct": "39", "wrong": "1"},
                    {"label": "LightGlue-SIFT-MAGSAC-min16", "matches": "50", "correct": "45", "wrong": "5"},
                ],
            )

            exit_code = sweep.main(
                [
                    "--source",
                    f"dev,{pfm_summary},{lightglue_metrics}",
                    "--thresholds",
                    "0.02,0.10,0.30",
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--report-html",
                    str(report_html),
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["threshold"] for row in rows], ["0.020000", "0.100000", "0.300000"])
            self.assertEqual(rows[0]["pfm_correct"], "124")
            self.assertEqual(rows[0]["pfm_wrong"], "36")
            self.assertEqual(rows[1]["pfm_correct"], "124")
            self.assertEqual(rows[1]["pfm_wrong"], "6")
            self.assertEqual(rows[2]["pfm_correct"], "76")
            self.assertEqual(rows[2]["pfm_wrong"], "4")
            self.assertEqual(rows[1]["lightglue_correct"], "103")
            self.assertEqual(rows[1]["correct_delta_vs_lightglue"], "21")
            self.assertTrue(report_html.exists())
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["best_threshold"]["threshold"], 0.1)
            self.assertEqual(summary["best_threshold"]["pfm_wrong"], 6)

    def test_writes_selected_threshold_summary_for_frozen_gate_application(self) -> None:
        import sweep_geometry_overlap_gate as sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pfm_summary = root / "pfm.csv"
            lightglue_metrics = root / "lg.csv"
            output_csv = root / "threshold_summary.csv"
            selected_json = root / "selected.json"
            selected_html = root / "selected.html"
            self.write_csv(
                pfm_summary,
                [
                    {"valid_fraction": "0.05", "matches": "30", "correct": "0", "wrong": "30"},
                    {"valid_fraction": "0.20", "matches": "50", "correct": "48", "wrong": "2"},
                    {"valid_fraction": "0.70", "matches": "80", "correct": "76", "wrong": "4"},
                ],
            )
            self.write_csv(
                lightglue_metrics,
                [
                    {"label": "LightGlue-SIFT-MAGSAC-min16", "matches": "20", "correct": "19", "wrong": "1"},
                    {"label": "LightGlue-SIFT-MAGSAC-min16", "matches": "40", "correct": "39", "wrong": "1"},
                    {"label": "LightGlue-SIFT-MAGSAC-min16", "matches": "50", "correct": "45", "wrong": "5"},
                ],
            )

            exit_code = sweep.main(
                [
                    "--source",
                    f"aggregate,{pfm_summary},{lightglue_metrics}",
                    "--thresholds",
                    "0.02,0.10,0.30",
                    "--output-csv",
                    str(output_csv),
                    "--selected-threshold",
                    "0.10",
                    "--selected-summary-json",
                    str(selected_json),
                    "--selected-report-html",
                    str(selected_html),
                ]
            )

            self.assertEqual(exit_code, 0)
            selected = json.loads(selected_json.read_text(encoding="utf-8"))
            self.assertEqual(selected["selected_threshold"], 0.1)
            self.assertEqual(selected["summary"]["pfm_correct"], 124)
            self.assertEqual(selected["summary"]["pfm_wrong"], 6)
            self.assertEqual(selected["summary"]["correct_delta_vs_lightglue"], 21)
            self.assertTrue(selected_html.exists())


if __name__ == "__main__":
    unittest.main()
