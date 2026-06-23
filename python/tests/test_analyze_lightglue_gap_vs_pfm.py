import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class AnalyzeLightGlueGapVsPfmTest(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("analyze_lightglue_gap_vs_pfm")
        except ModuleNotFoundError as exc:
            self.fail(f"missing LightGlue gap analysis module: {exc}")

    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_build_gap_rows_classifies_recall_and_wrong_risk(self) -> None:
        gap_mod = self.module()
        pfm_rows = [
            {
                "source": "dev",
                "split": "dev",
                "row_index": "0",
                "base_id": "pair_a",
                "target_variant": "extreme_01",
                "matches": "6",
                "correct": "6",
                "wrong": "0",
                "score_mean": "20.0",
                "valid_fraction": "0.20",
            },
            {
                "source": "dev",
                "split": "dev",
                "row_index": "1",
                "base_id": "pair_b",
                "target_variant": "extreme_02",
                "matches": "3",
                "correct": "3",
                "wrong": "0",
                "score_mean": "18.0",
                "valid_fraction": "0.30",
            },
            {
                "source": "dev",
                "split": "dev",
                "row_index": "2",
                "base_id": "pair_c",
                "target_variant": "extreme_03",
                "matches": "10",
                "correct": "8",
                "wrong": "2",
                "score_mean": "15.0",
                "valid_fraction": "0.40",
            },
        ]
        lightglue_rows = [
            {"label": "LightGlue-SIFT-raw", "base_id": "pair_a_raw", "target_variant": "extreme_01", "matches": "8", "correct": "7", "wrong": "1"},
            {"label": "LightGlue-SIFT-MAGSAC-min16", "base_id": "pair_a_lg", "target_variant": "extreme_01", "matches": "5", "correct": "5", "wrong": "0"},
            {"label": "LightGlue-SIFT-raw", "base_id": "pair_b_raw", "target_variant": "extreme_02", "matches": "9", "correct": "9", "wrong": "0"},
            {"label": "LightGlue-SIFT-MAGSAC-min16", "base_id": "pair_b_lg", "target_variant": "extreme_02", "matches": "9", "correct": "8", "wrong": "1"},
            {"label": "LightGlue-SIFT-raw", "base_id": "pair_c_raw", "target_variant": "extreme_03", "matches": "11", "correct": "10", "wrong": "1"},
            {"label": "LightGlue-SIFT-MAGSAC-min16", "base_id": "pair_c_lg", "target_variant": "extreme_03", "matches": "7", "correct": "7", "wrong": "0"},
        ]

        rows = gap_mod.build_gap_rows(
            pfm_rows,
            lightglue_rows,
            source="dev",
            split="dev",
            lightglue_label="LightGlue-SIFT-MAGSAC-min16",
        )

        self.assertEqual([row["gap_bucket"] for row in rows], ["pfm_clean_win", "lightglue_recall_gap", "pfm_wrong_risk"])
        self.assertEqual(rows[1]["correct_delta_vs_lightglue"], -5)
        self.assertEqual(rows[1]["lightglue_correct_gap"], 5)
        self.assertEqual(rows[2]["wrong_delta_vs_lightglue"], 2)
        self.assertEqual(rows[2]["pfm_wrong_excess"], 2)
        self.assertEqual(rows[0]["pfm_base_id"], "pair_a")
        self.assertEqual(rows[0]["lightglue_base_id"], "pair_a_lg")

        by_variant = gap_mod.summarize_by(["target_variant"], rows)
        by_variant_key = {row["target_variant"]: row for row in by_variant}
        self.assertEqual(by_variant_key["extreme_02"]["lightglue_correct_gap"], 5)
        self.assertEqual(by_variant_key["extreme_03"]["pfm_wrong_excess"], 2)

    def test_cli_writes_gap_tables_summary_and_html(self) -> None:
        gap_mod = self.module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pfm_path = root / "pfm.csv"
            lightglue_path = root / "lightglue.csv"
            self.write_csv(
                pfm_path,
                [
                    "source",
                    "split",
                    "row_index",
                    "base_id",
                    "target_variant",
                    "matches",
                    "correct",
                    "wrong",
                    "score_mean",
                    "valid_fraction",
                ],
                [
                    {
                        "source": "dev",
                        "split": "dev",
                        "row_index": "0",
                        "base_id": "pair_a",
                        "target_variant": "extreme_01",
                        "matches": "4",
                        "correct": "4",
                        "wrong": "0",
                        "score_mean": "20.0",
                        "valid_fraction": "0.2",
                    },
                    {
                        "source": "dev",
                        "split": "dev",
                        "row_index": "1",
                        "base_id": "pair_b",
                        "target_variant": "extreme_02",
                        "matches": "2",
                        "correct": "2",
                        "wrong": "0",
                        "score_mean": "19.0",
                        "valid_fraction": "0.3",
                    },
                ],
            )
            self.write_csv(
                lightglue_path,
                ["label", "base_id", "target_variant", "matches", "correct", "wrong"],
                [
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "base_id": "lg_a",
                        "target_variant": "extreme_01",
                        "matches": "4",
                        "correct": "4",
                        "wrong": "0",
                    },
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "base_id": "lg_b",
                        "target_variant": "extreme_02",
                        "matches": "10",
                        "correct": "9",
                        "wrong": "1",
                    },
                ],
            )

            output_dir = root / "out"
            exit_code = gap_mod.main(
                [
                    "--source",
                    f"dev,dev,{pfm_path},{lightglue_path}",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            for name in [
                "gap_rows.csv",
                "summary_by_split.csv",
                "summary_by_variant.csv",
                "top_lightglue_gaps.csv",
                "top_pfm_wrong_risks.csv",
                "summary.json",
                "index.html",
            ]:
                self.assertTrue((output_dir / name).exists(), name)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["aggregate"]["pfm_correct"], 6)
            self.assertEqual(summary["aggregate"]["lightglue_correct"], 13)
            self.assertEqual(summary["aggregate"]["correct_delta_vs_lightglue"], -7)
            self.assertEqual(summary["aggregate"]["wrong_delta_vs_lightglue"], -1)

            with (output_dir / "top_lightglue_gaps.csv").open("r", encoding="utf-8", newline="") as handle:
                top_rows = list(csv.DictReader(handle))
            self.assertEqual(top_rows[0]["target_variant"], "extreme_02")
            self.assertEqual(top_rows[0]["lightglue_correct_gap"], "7")


if __name__ == "__main__":
    unittest.main()
