import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


PAIR_FIELDS = [
    "pair_index",
    "split",
    "pair_type",
    "reference_dataset_id",
    "reference_pose_id",
    "reference_base_id",
    "reference_variant",
    "target_dataset_id",
    "target_pose_id",
    "target_base_id",
    "target_variant",
    "valid_fraction",
    "valid_pixels",
]


def _pair_row(index: int, valid_fraction: float) -> dict[str, str]:
    return {
        "pair_index": str(index),
        "split": "train",
        "pair_type": "cross_camera",
        "reference_dataset_id": "h100km_fov076",
        "reference_pose_id": f"ref_{index}_extreme_01",
        "reference_base_id": f"ref_{index}",
        "reference_variant": "extreme_01",
        "target_dataset_id": "h100km_fov076",
        "target_pose_id": f"tgt_{index}_extreme_02",
        "target_base_id": f"tgt_{index}",
        "target_variant": "extreme_02",
        "valid_fraction": f"{valid_fraction:.6f}",
        "valid_pixels": str(int(valid_fraction * 2048 * 2048)),
    }


def _summary_row(index: int, *, valid_fraction: float, matches: int, wrong: int = 0) -> dict[str, str]:
    correct = matches - wrong
    precision = 0.0 if matches == 0 else correct / matches
    return {
        "base_id": f"ref_{index}",
        "target_variant": "extreme_02",
        "valid_fraction": f"{valid_fraction:.6f}",
        "matches": str(matches),
        "correct": str(correct),
        "wrong": str(wrong),
        "precision": f"{precision:.6f}",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class GeometryAcceptanceTrainingManifestTest(unittest.TestCase):
    def test_labels_low_overlap_as_reject_and_mid_high_as_accept(self) -> None:
        import build_geometry_acceptance_training_manifest as mod

        rows = [_pair_row(0, 0.04), _pair_row(1, 0.15), _pair_row(2, 0.72)]

        output = mod.build_geometry_acceptance_rows(rows)

        self.assertEqual([row["pair_accept_label"] for row in output], ["0", "1", "1"])
        self.assertEqual([row["pair_accept_weight"] for row in output], ["3.000000", "1.000000", "1.000000"])
        self.assertEqual(output[0]["geometry_accept_reason"], "low_valid_fraction")
        self.assertEqual(output[1]["geometry_accept_reason"], "observable_valid_fraction")
        self.assertEqual(output[2]["geometry_accept_source_valid_fraction"], "0.720000")

    def test_keeps_ambiguous_overlap_with_configurable_weight(self) -> None:
        import build_geometry_acceptance_training_manifest as mod

        rows = [_pair_row(0, 0.12), _pair_row(1, 0.22)]

        output = mod.build_geometry_acceptance_rows(
            rows,
            reject_below_valid_fraction=0.10,
            accept_at_valid_fraction=0.20,
            ambiguous_accept_weight=0.25,
        )

        self.assertEqual([row["pair_accept_label"] for row in output], ["1", "1"])
        self.assertEqual([row["pair_accept_weight"] for row in output], ["0.250000", "1.000000"])
        self.assertEqual(output[0]["geometry_accept_reason"], "ambiguous_valid_fraction")

    def test_true_geometry_summary_labels_by_kept_match_count_after_overlap_gate(self) -> None:
        import build_geometry_acceptance_training_manifest as mod

        rows = [_pair_row(0, 0.20), _pair_row(1, 0.25), _pair_row(2, 0.05), _pair_row(3, 0.30)]
        summary_rows = [
            _summary_row(0, valid_fraction=0.20, matches=24),
            _summary_row(1, valid_fraction=0.25, matches=5),
            _summary_row(2, valid_fraction=0.05, matches=30),
            _summary_row(3, valid_fraction=0.30, matches=25, wrong=1),
        ]

        output = mod.build_geometry_acceptance_rows(
            rows,
            true_geometry_summary_rows=summary_rows,
            reject_below_valid_fraction=0.10,
            accept_at_valid_fraction=0.10,
            min_true_geometry_matches=16,
            max_true_geometry_wrong=0,
            min_true_geometry_precision=1.0,
            reject_weight=4.0,
        )

        self.assertEqual([row["pair_accept_label"] for row in output], ["1", "0", "0", "0"])
        self.assertEqual(
            [row["geometry_accept_reason"] for row in output],
            [
                "true_geometry_match_count",
                "insufficient_true_geometry_matches",
                "low_valid_fraction",
                "true_geometry_wrong_matches",
            ],
        )
        self.assertEqual(output[0]["geometry_accept_true_geometry_matches"], "24")
        self.assertEqual(output[1]["pair_accept_weight"], "4.000000")
        self.assertEqual(output[3]["geometry_accept_true_geometry_wrong"], "1")

    def test_cli_writes_manifest_summary_and_html(self) -> None:
        import build_geometry_acceptance_training_manifest as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train_pairs.csv"
            output_manifest = root / "out" / "train_pairs_geometry_accept.csv"
            summary_json = root / "out" / "summary.json"
            report_html = root / "out" / "index.html"
            _write_csv(source, [_pair_row(0, 0.03), _pair_row(1, 0.33)])

            exit_code = mod.main(
                [
                    "--pair-manifest",
                    str(source),
                    "--output-manifest",
                    str(output_manifest),
                    "--summary-json",
                    str(summary_json),
                    "--report-html",
                    str(report_html),
                    "--reject-below-valid-fraction",
                    "0.15",
                    "--accept-at-valid-fraction",
                    "0.15",
                    "--reject-weight",
                    "4.0",
                ]
            )

            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            report = report_html.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual([row["pair_accept_label"] for row in rows], ["0", "1"])
        self.assertEqual(rows[0]["pair_accept_weight"], "4.000000")
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["accept_rows"], 1)
        self.assertEqual(summary["reject_rows"], 1)
        self.assertIn("Geometry acceptance training manifest", report)

    def test_cli_can_use_true_geometry_summary(self) -> None:
        import build_geometry_acceptance_training_manifest as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train_pairs.csv"
            summary_csv = root / "true_geometry_summary.csv"
            output_manifest = root / "out" / "train_pairs_geometry_accept.csv"
            summary_json = root / "out" / "summary.json"
            report_html = root / "out" / "index.html"
            _write_csv(source, [_pair_row(0, 0.22), _pair_row(1, 0.24)])
            with summary_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(_summary_row(0, valid_fraction=0.22, matches=32).keys()))
                writer.writeheader()
                writer.writerows(
                    [
                        _summary_row(0, valid_fraction=0.22, matches=32),
                        _summary_row(1, valid_fraction=0.24, matches=3),
                    ]
                )

            exit_code = mod.main(
                [
                    "--pair-manifest",
                    str(source),
                    "--true-geometry-summary",
                    str(summary_csv),
                    "--output-manifest",
                    str(output_manifest),
                    "--summary-json",
                    str(summary_json),
                    "--report-html",
                    str(report_html),
                    "--reject-below-valid-fraction",
                    "0.10",
                    "--accept-at-valid-fraction",
                    "0.10",
                    "--min-true-geometry-matches",
                    "16",
                ]
            )

            with output_manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(summary_json.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual([row["pair_accept_label"] for row in rows], ["1", "0"])
        self.assertEqual(summary["true_geometry_summary"], str(summary_csv))
        self.assertEqual(summary["reason_counts"]["true_geometry_match_count"], 1)
        self.assertEqual(summary["reason_counts"]["insufficient_true_geometry_matches"], 1)


if __name__ == "__main__":
    unittest.main()
