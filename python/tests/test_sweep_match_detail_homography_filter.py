import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class SweepMatchDetailHomographyFilterTest(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("sweep_match_detail_homography_filter")
        except ModuleNotFoundError as exc:
            self.fail(f"missing homography filter sweep module: {exc}")

    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def detail_row(
        self,
        pair_index: int,
        match_index: int,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        correct: int,
        *,
        variant: str = "extreme_02",
    ) -> dict[str, object]:
        return {
            "pair_index": pair_index,
            "base_id": f"pair_{pair_index}",
            "target_variant": variant,
            "split": "dev",
            "match_index": match_index,
            "point_a_x_px": ax,
            "point_a_y_px": ay,
            "point_b_x_px": bx,
            "point_b_y_px": by,
            "score": 20.0 - match_index,
            "correct": correct,
            "valid_fraction": "0.5",
        }

    def test_sweep_filters_match_details_and_summarizes_against_lightglue(self) -> None:
        sweep_mod = self.module()
        details = [
            self.detail_row(0, 0, 0, 0, 10, 20, 1),
            self.detail_row(0, 1, 10, 0, 20, 20, 1),
            self.detail_row(0, 2, 0, 10, 10, 30, 1),
            self.detail_row(0, 3, 10, 10, 20, 30, 1),
            self.detail_row(0, 4, 20, 0, 30, 20, 1),
            self.detail_row(0, 5, 20, 10, 30, 30, 1),
            self.detail_row(0, 6, 0, 20, 10, 40, 1),
            self.detail_row(0, 7, 10, 20, 20, 40, 1),
            self.detail_row(0, 8, 20, 20, 30, 40, 1),
            self.detail_row(0, 9, 20, 30, 300, 300, 0),
            self.detail_row(1, 0, 0, 0, 0, 0, 1, variant="extreme_03"),
            self.detail_row(1, 1, 10, 0, 10, 0, 1, variant="extreme_03"),
            self.detail_row(1, 2, 0, 10, 0, 10, 1, variant="extreme_03"),
            self.detail_row(1, 3, 10, 10, 10, 10, 1, variant="extreme_03"),
            self.detail_row(1, 4, 20, 20, 20, 20, 0, variant="extreme_03"),
        ]
        lightglue = [
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "pair_index": "0",
                "base_id": "lg_0",
                "target_variant": "extreme_02",
                "matches": "3",
                "correct": "3",
                "wrong": "0",
            },
            {
                "label": "LightGlue-SIFT-MAGSAC-min16",
                "pair_index": "1",
                "base_id": "lg_1",
                "target_variant": "extreme_03",
                "matches": "3",
                "correct": "3",
                "wrong": "0",
            },
        ]

        pair_rows = sweep_mod.build_pair_rows(
            details,
            lightglue,
            source="dev",
            split="dev",
            lightglue_label="LightGlue-SIFT-MAGSAC-min16",
            threshold_px=2.0,
            min_matches=4,
            min_score=0.0,
        )

        self.assertEqual(len(pair_rows), 2)
        self.assertEqual(pair_rows[0]["pfm_matches"], 9)
        self.assertEqual(pair_rows[0]["pfm_correct"], 9)
        self.assertEqual(pair_rows[0]["pfm_wrong"], 0)
        self.assertEqual(pair_rows[1]["pfm_matches"], 5)
        self.assertEqual(pair_rows[1]["pfm_wrong"], 1)

        summary = sweep_mod.summarize_rows(pair_rows)
        self.assertEqual(summary["pfm_correct"], 13)
        self.assertEqual(summary["pfm_wrong"], 1)
        self.assertEqual(summary["lightglue_correct"], 6)
        self.assertEqual(summary["correct_delta_vs_lightglue"], 7)
        self.assertEqual(summary["wrong_delta_vs_lightglue"], 1)

        score_gated_rows = sweep_mod.build_pair_rows(
            details,
            lightglue,
            source="dev",
            split="dev",
            lightglue_label="LightGlue-SIFT-MAGSAC-min16",
            threshold_px=2.0,
            min_matches=4,
            min_score=17.0,
        )
        self.assertEqual(score_gated_rows[1]["pfm_matches"], 4)
        self.assertEqual(score_gated_rows[1]["pfm_wrong"], 0)

    def test_cli_writes_sweep_csv_json_and_html(self) -> None:
        sweep_mod = self.module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            details_path = root / "details.csv"
            lightglue_path = root / "lightglue.csv"
            detail_fields = [
                "pair_index",
                "base_id",
                "target_variant",
                "split",
                "match_index",
                "point_a_x_px",
                "point_a_y_px",
                "point_b_x_px",
                "point_b_y_px",
                "score",
                "correct",
                "valid_fraction",
            ]
            self.write_csv(
                details_path,
                detail_fields,
                [
                    self.detail_row(0, 0, 0, 0, 0, 0, 1),
                    self.detail_row(0, 1, 10, 0, 10, 0, 1),
                    self.detail_row(0, 2, 0, 10, 0, 10, 1),
                    self.detail_row(0, 3, 10, 10, 10, 10, 1),
                    self.detail_row(0, 4, 20, 0, 20, 0, 1),
                    self.detail_row(0, 5, 20, 10, 20, 10, 1),
                    self.detail_row(0, 6, 0, 20, 0, 20, 1),
                    self.detail_row(0, 7, 10, 20, 10, 20, 1),
                    self.detail_row(0, 8, 20, 20, 20, 20, 1),
                    self.detail_row(0, 9, 20, 30, 200, 200, 0),
                ],
            )
            self.write_csv(
                lightglue_path,
                ["label", "pair_index", "base_id", "target_variant", "matches", "correct", "wrong"],
                [
                    {
                        "label": "LightGlue-SIFT-MAGSAC-min16",
                        "pair_index": "0",
                        "base_id": "lg_0",
                        "target_variant": "extreme_02",
                        "matches": "3",
                        "correct": "3",
                        "wrong": "0",
                    }
                ],
            )

            output_dir = root / "out"
            exit_code = sweep_mod.main(
                [
                    "--source",
                    f"dev,dev,{details_path},{lightglue_path}",
                    "--thresholds",
                    "2.0",
                    "--min-matches",
                    "4",
                    "--min-scores",
                    "0,12",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            for name in ["sweep_summary.csv", "sweep_by_split_variant.csv", "summary.json", "index.html"]:
                self.assertTrue((output_dir / name).exists(), name)
            payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["best_aggregate"]["pfm_correct"], 9)
            self.assertEqual(payload["best_aggregate"]["wrong_delta_vs_lightglue"], 0)


if __name__ == "__main__":
    unittest.main()
