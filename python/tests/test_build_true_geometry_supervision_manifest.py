import csv
import json
import tempfile
import unittest
from pathlib import Path


PAIR_FIELDS = [
    "pair_index",
    "split",
    "pair_type",
    "reference_path",
    "target_path",
    "reference_base_id",
    "target_base_id",
    "reference_variant",
    "target_variant",
    "valid_fraction",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pair_row(pair_index: str, split: str, base: str, target: str, valid_fraction: str) -> dict[str, str]:
    return {
        "pair_index": pair_index,
        "split": split,
        "pair_type": "cross_camera",
        "reference_path": f"/data/{base}_a.pt",
        "target_path": f"/data/{target}_b.pt",
        "reference_base_id": base,
        "target_base_id": target,
        "reference_variant": "extreme_01",
        "target_variant": "extreme_03",
        "valid_fraction": valid_fraction,
    }


def _selection_row(
    pair_index: str,
    base: str,
    target: str,
    selected_matches: str,
    selected_correct: str,
    selected_wrong: str,
) -> dict[str, str]:
    return {
        "eval_split": "train",
        "pair_index": pair_index,
        "manifest_pair_index": pair_index,
        "manifest_split": "train",
        "pair_type": "cross_camera",
        "reference_base_id": base,
        "target_base_id": target,
        "reference_variant": "extreme_01",
        "target_variant": "extreme_03",
        "valid_fraction": "0.35",
        "selected_source": "phase49b",
        "selected_matches": selected_matches,
        "selected_correct": selected_correct,
        "selected_wrong": selected_wrong,
        "selected_precision": "1.0",
        "lightglue_matches": "9999",
        "lightglue_correct": "9999",
        "lightglue_wrong": "9999",
        "delta_correct_vs_lightglue": "-9999",
        "delta_wrong_vs_lightglue": "-9999",
    }


def _detail_rows(pair_index: str, count: int, *, correct: str = "1") -> list[dict[str, str]]:
    return [
        {
            "label": "PFM / all-filtered",
            "pair_index": pair_index,
            "base_id": f"base_{pair_index}",
            "reference_variant": "extreme_01",
            "target_variant": "extreme_03",
            "split": "train",
            "match_index": str(match_index),
            "score": "20.0",
            "error_px": "2.0",
            "correct": correct,
            "valid_fraction": "0.35",
            "selector_source": "phase49b",
        }
        for match_index in range(count)
    ]


class TrueGeometrySupervisionManifestTest(unittest.TestCase):
    def test_builds_train_only_acceptance_manifest_without_lightglue_labels(self) -> None:
        import build_true_geometry_supervision_manifest as builder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_manifest = root / "train_pairs.csv"
            selected_pair_summary = root / "pair_selection.csv"
            selected_match_details = root / "selected_match_details.csv"
            output_csv = root / "out" / "true_geometry_supervision.csv"
            output_json = root / "out" / "summary.json"
            output_html = root / "out" / "index.html"

            _write_csv(
                pair_manifest,
                PAIR_FIELDS,
                [
                    _pair_row("0", "train", "base_accept", "target_accept", "0.35"),
                    _pair_row("1", "train", "base_reject", "target_reject", "0.02"),
                    _pair_row("2", "val", "base_val", "target_val", "0.70"),
                ],
            )
            selection_rows = [
                _selection_row("0", "base_accept", "target_accept", "30", "30", "0"),
                _selection_row("1", "base_reject", "target_reject", "30", "30", "0"),
                _selection_row("2", "base_val", "target_val", "30", "30", "0"),
            ]
            _write_csv(selected_pair_summary, list(selection_rows[0].keys()), selection_rows)
            detail_rows = [
                *_detail_rows("0", 30),
                *_detail_rows("1", 30),
                *_detail_rows("2", 30),
            ]
            _write_csv(selected_match_details, list(detail_rows[0].keys()), detail_rows)

            summary = builder.build_true_geometry_supervision_manifest(
                pair_manifest=pair_manifest,
                selected_pair_summary=selected_pair_summary,
                selected_match_details=selected_match_details,
                output_csv=output_csv,
                output_json=output_json,
                output_html=output_html,
                min_accept_valid_fraction=0.10,
                min_accept_matches=16,
                max_accept_wrong=0,
                required_split="train",
            )

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            saved = json.loads(output_json.read_text(encoding="utf-8"))
            report_text = output_html.read_text(encoding="utf-8")

        self.assertEqual([row["pair_index"] for row in rows], ["0", "1"])
        self.assertEqual(rows[0]["pair_accept_label"], "1")
        self.assertEqual(rows[0]["true_geometry_positive_matches"], "30")
        self.assertEqual(rows[0]["true_geometry_filtered_matches"], "30")
        self.assertEqual(rows[0]["true_geometry_wrong_matches"], "0")
        self.assertEqual(rows[0]["true_geometry_supervision_source"], "phase49b")
        self.assertEqual(rows[1]["pair_accept_label"], "0")
        self.assertEqual(rows[1]["true_geometry_positive_matches"], "30")
        self.assertEqual(rows[1]["true_geometry_filtered_matches"], "30")
        self.assertEqual(summary["output_rows"], 2)
        self.assertEqual(summary["accept_rows"], 1)
        self.assertEqual(summary["reject_rows"], 1)
        self.assertEqual(summary["skipped_non_required_split_rows"], 1)
        self.assertFalse(summary["uses_lightglue_labels"])
        self.assertFalse(saved["uses_lightglue_labels"])
        self.assertIn("True geometry supervision manifest", report_text)


if __name__ == "__main__":
    unittest.main()
