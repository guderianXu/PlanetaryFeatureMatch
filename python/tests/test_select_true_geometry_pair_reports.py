import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class SelectTrueGeometryPairReportsTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_pair_manifest(self, path: Path) -> None:
        self.write_csv(
            path,
            [
                {
                    "pair_index": "0",
                    "split": "val",
                    "pair_type": "cross_camera",
                    "reference_base_id": "base_a",
                    "reference_variant": "extreme_01",
                    "target_base_id": "base_b",
                    "target_variant": "extreme_02",
                    "valid_fraction": "0.50",
                },
                {
                    "pair_index": "1",
                    "split": "test",
                    "pair_type": "cross_camera",
                    "reference_base_id": "base_c",
                    "reference_variant": "extreme_03",
                    "target_base_id": "base_d",
                    "target_variant": "extreme_03",
                    "valid_fraction": "0.70",
                },
            ],
        )

    def write_lightglue_metrics(self, path: Path) -> None:
        self.write_csv(
            path,
            [
                {
                    "label": "LightGlue-SIFT-MAGSAC-min16",
                    "pair_index": "0",
                    "manifest_pair_index": "0",
                    "matches": "3",
                    "correct": "3",
                    "wrong": "0",
                },
                {
                    "label": "LightGlue-SIFT-MAGSAC-min16",
                    "pair_index": "1",
                    "manifest_pair_index": "1",
                    "matches": "2",
                    "correct": "1",
                    "wrong": "1",
                },
            ],
        )

    def summary_rows(self, first_matches: str, second_matches: str) -> list[dict[str, str]]:
        return [
            {
                "label": "PFM / all-filtered",
                "base_id": "base_a",
                "target_variant": "extreme_02",
                "split": "val",
                "valid_fraction": "0.50",
                "matches": first_matches,
                "correct": first_matches,
                "wrong": "0",
                "precision": "1.000000",
                "score_mean": "20.0",
            },
            {
                "label": "PFM / all-filtered",
                "base_id": "base_c",
                "target_variant": "extreme_03",
                "split": "test",
                "valid_fraction": "0.70",
                "matches": second_matches,
                "correct": second_matches,
                "wrong": "0",
                "precision": "1.000000",
                "score_mean": "21.0",
            },
        ]

    def detail_rows(self, first_count: int, second_count: int, *, score_prefix: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for pair_index, count in ((0, first_count), (1, second_count)):
            for match_index in range(count):
                rows.append(
                    {
                        "label": "PFM / all-filtered",
                        "pair_index": str(pair_index),
                        "base_id": "base_a" if pair_index == 0 else "base_c",
                        "reference_variant": "extreme_01" if pair_index == 0 else "extreme_03",
                        "target_variant": "extreme_02" if pair_index == 0 else "extreme_03",
                        "split": "val" if pair_index == 0 else "test",
                        "match_index": str(match_index),
                        "point_a_x_px": "1.0",
                        "point_a_y_px": "2.0",
                        "point_b_x_px": "3.0",
                        "point_b_y_px": "4.0",
                        "score": f"{score_prefix}{match_index}",
                        "correct": "1",
                        "valid_fraction": "0.50" if pair_index == 0 else "0.70",
                    }
                )
        return rows

    def test_cli_selects_best_candidate_per_pair(self) -> None:
        import select_true_geometry_pair_reports as selector

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pair_manifest = root / "eval" / "lockbox_pairs.csv"
            lightglue_metrics = root / "eval" / "lockbox" / "lightglue" / "lightglue_sift_metrics.csv"
            self.write_pair_manifest(pair_manifest)
            self.write_lightglue_metrics(lightglue_metrics)

            candidate_a = root / "phase45" / "lockbox" / "pfm_eval"
            candidate_b = root / "phase49b" / "lockbox" / "pfm_eval"
            self.write_csv(candidate_a / "all_filtered_summary.csv", self.summary_rows("4", "9"))
            self.write_csv(candidate_b / "all_filtered_summary.csv", self.summary_rows("7", "8"))
            self.write_csv(candidate_a / "all_filtered_match_details.csv", self.detail_rows(4, 9, score_prefix="A"))
            self.write_csv(candidate_b / "all_filtered_match_details.csv", self.detail_rows(7, 8, score_prefix="B"))

            output_dir = root / "selected"
            exit_code = selector.main(
                [
                    "--source",
                    f"lockbox,{pair_manifest},{lightglue_metrics}",
                    "--candidate",
                    f"phase45,{root / 'phase45'},pfm_eval",
                    "--candidate",
                    f"phase49b,{root / 'phase49b'},pfm_eval",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            with (output_dir / "summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["aggregate"]["selected_correct"], 16)
            self.assertEqual(summary["aggregate"]["selected_wrong"], 0)
            self.assertEqual(summary["aggregate"]["correct_delta_vs_lightglue"], 12)
            self.assertEqual(summary["by_variant"]["extreme_02"]["selected_correct"], 7)
            self.assertEqual(summary["by_variant"]["extreme_02"]["correct_delta_vs_lightglue"], 4)
            self.assertEqual(summary["by_variant"]["extreme_03"]["selected_correct"], 9)
            self.assertEqual(summary["by_variant"]["extreme_03"]["wrong_delta_vs_lightglue"], -1)
            self.assertEqual(summary["source_pair_counts"], {"phase45": 1, "phase49b": 1})

            with (output_dir / "pair_selection.csv").open(newline="", encoding="utf-8") as handle:
                selected_pairs = list(csv.DictReader(handle))
            self.assertEqual([row["selected_source"] for row in selected_pairs], ["phase49b", "phase45"])

            with (output_dir / "lockbox" / "selected_match_details.csv").open(newline="", encoding="utf-8") as handle:
                detail_rows = list(csv.DictReader(handle))
            self.assertEqual(len(detail_rows), 16)
            self.assertEqual({row["selector_source"] for row in detail_rows if row["pair_index"] == "0"}, {"phase49b"})
            self.assertEqual({row["selector_source"] for row in detail_rows if row["pair_index"] == "1"}, {"phase45"})

    def test_cli_can_select_training_reports_without_lightglue_metrics(self) -> None:
        import select_true_geometry_pair_reports as selector

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pair_manifest = root / "eval" / "train_pairs.csv"
            self.write_pair_manifest(pair_manifest)

            candidate_a = root / "phase45" / "train" / "pfm_eval"
            candidate_b = root / "phase49b" / "train" / "pfm_eval"
            self.write_csv(candidate_a / "all_filtered_summary.csv", self.summary_rows("4", "9"))
            self.write_csv(candidate_b / "all_filtered_summary.csv", self.summary_rows("7", "8"))
            self.write_csv(candidate_a / "all_filtered_match_details.csv", self.detail_rows(4, 9, score_prefix="A"))
            self.write_csv(candidate_b / "all_filtered_match_details.csv", self.detail_rows(7, 8, score_prefix="B"))

            output_dir = root / "selected_train"
            exit_code = selector.main(
                [
                    "--source",
                    f"train,{pair_manifest}",
                    "--candidate",
                    f"phase45,{root / 'phase45'},pfm_eval",
                    "--candidate",
                    f"phase49b,{root / 'phase49b'},pfm_eval",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["uses_lightglue_baseline"])
            self.assertEqual(summary["aggregate"]["selected_correct"], 16)
            self.assertEqual(summary["aggregate"]["lightglue_correct"], 0)
            self.assertEqual(summary["aggregate"]["correct_delta_vs_lightglue"], 16)

            with (output_dir / "pair_selection.csv").open(newline="", encoding="utf-8") as handle:
                selected_pairs = list(csv.DictReader(handle))
            self.assertEqual([row["selected_source"] for row in selected_pairs], ["phase49b", "phase45"])
            self.assertTrue(all(row["lightglue_correct"] == "0" for row in selected_pairs))

            with (output_dir / "train" / "selected_match_details.csv").open(newline="", encoding="utf-8") as handle:
                detail_rows = list(csv.DictReader(handle))
            self.assertEqual(len(detail_rows), 16)

    def test_default_rank_profile_does_not_use_wrong_as_tie_break(self) -> None:
        import select_true_geometry_pair_reports as selector

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pair_manifest = root / "eval" / "dev_pairs.csv"
            lightglue_metrics = root / "eval" / "dev" / "lightglue" / "lightglue_sift_metrics.csv"
            self.write_csv(
                pair_manifest,
                [
                    {
                        "pair_index": "0",
                        "split": "dev",
                        "pair_type": "cross_camera",
                        "reference_base_id": "base_a",
                        "reference_variant": "extreme_01",
                        "target_base_id": "base_b",
                        "target_variant": "extreme_03",
                        "valid_fraction": "0.40",
                    }
                ],
            )
            self.write_lightglue_metrics(lightglue_metrics)
            candidate_a = root / "phase45" / "dev" / "pfm_eval"
            candidate_b = root / "phase49b" / "dev" / "pfm_eval"
            self.write_csv(
                candidate_a / "all_filtered_summary.csv",
                [
                    {
                        "label": "PFM / all-filtered",
                        "base_id": "base_a",
                        "target_variant": "extreme_03",
                        "split": "dev",
                        "valid_fraction": "0.40",
                        "matches": "10",
                        "correct": "5",
                        "wrong": "5",
                        "precision": "0.500000",
                        "score_mean": "30.0",
                    }
                ],
            )
            self.write_csv(
                candidate_b / "all_filtered_summary.csv",
                [
                    {
                        "label": "PFM / all-filtered",
                        "base_id": "base_a",
                        "target_variant": "extreme_03",
                        "split": "dev",
                        "valid_fraction": "0.40",
                        "matches": "10",
                        "correct": "10",
                        "wrong": "0",
                        "precision": "1.000000",
                        "score_mean": "20.0",
                    }
                ],
            )
            self.write_csv(candidate_a / "all_filtered_match_details.csv", self.detail_rows(10, 0, score_prefix="A"))
            self.write_csv(candidate_b / "all_filtered_match_details.csv", self.detail_rows(10, 0, score_prefix="B"))

            output_dir = root / "selected"
            exit_code = selector.main(
                [
                    "--source",
                    f"dev,{pair_manifest},{lightglue_metrics}",
                    "--candidate",
                    f"phase45,{root / 'phase45'},pfm_eval",
                    "--candidate",
                    f"phase49b,{root / 'phase49b'},pfm_eval",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            with (output_dir / "pair_selection.csv").open(newline="", encoding="utf-8") as handle:
                selected_pairs = list(csv.DictReader(handle))
            self.assertEqual(selected_pairs[0]["selected_source"], "phase45")
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selection_rank_profile"], "inference_safe")


if __name__ == "__main__":
    unittest.main()
