import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_gate_acceptance_training_manifest import SourceSpec, build_gate_acceptance_manifest


PAIR_FIELDS = [
    "pair_index",
    "split",
    "pair_type",
    "reference_base_id",
    "reference_variant",
    "target_base_id",
    "target_variant",
    "custom_field",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pair_row(pair_index: str, base_id: str, variant: str, custom: str) -> dict[str, str]:
    return {
        "pair_index": pair_index,
        "split": "train",
        "pair_type": "same_position_view",
        "reference_base_id": base_id,
        "reference_variant": "nadir",
        "target_base_id": base_id,
        "target_variant": variant,
        "custom_field": custom,
    }


def _hybrid_row(
    pair_index: str,
    base_id: str,
    variant: str,
    selected_pfm: str,
    wrong: str,
    precision: str,
    pfm_wrong: str,
) -> dict[str, str]:
    return {
        "source_name": "phase37_dev_source",
        "split": "dev_val",
        "pair_index": pair_index,
        "pair_type": "same_position_view",
        "base_id": base_id,
        "reference_variant": "nadir",
        "target_variant": variant,
        "gate_selected_pfm": selected_pfm,
        "chosen_source": "pfm" if selected_pfm == "1" else "lightglue",
        "matches": "100",
        "correct": str(100 - int(pfm_wrong if selected_pfm == "1" else "0")),
        "wrong": wrong,
        "precision": precision,
        "pfm_matches": "100",
        "pfm_correct": str(100 - int(pfm_wrong)),
        "pfm_wrong": pfm_wrong,
        "pfm_precision": precision,
        "lightglue_matches": "50",
        "lightglue_correct": "50",
        "lightglue_wrong": "0",
        "lightglue_precision": "1.0",
    }


class GateAcceptanceTrainingManifestTest(unittest.TestCase):
    def test_builds_labels_from_gate_hybrid_rows_and_preserves_pair_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_manifest = root / "pairs.csv"
            hybrid_rows = root / "hybrid_rows.csv"
            output_manifest = root / "out" / "gate_acceptance_pairs.csv"
            summary_json = root / "out" / "summary.json"
            report_html = root / "out" / "index.html"
            _write_csv(
                pair_manifest,
                PAIR_FIELDS,
                [
                    _pair_row("10", "base_accept", "mid_01", "first"),
                    _pair_row("11", "base_reject", "extreme_03", "second"),
                ],
            )
            hybrid_fields = list(_hybrid_row("10", "base_accept", "mid_01", "1", "0", "1.0", "0").keys())
            _write_csv(
                hybrid_rows,
                hybrid_fields,
                [
                    _hybrid_row("11", "base_reject", "extreme_03", "0", "0", "1.0", "5"),
                    _hybrid_row("10", "base_accept", "mid_01", "1", "0", "1.0", "0"),
                ],
            )

            summary = build_gate_acceptance_manifest(
                sources=[SourceSpec("dev_source", pair_manifest, hybrid_rows)],
                output_manifest=output_manifest,
                summary_json=summary_json,
                report_html=report_html,
                accept_weight=1.5,
                reject_weight=4.0,
                min_accept_precision=0.999,
                max_accept_wrong=0,
            )

            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            saved = json.loads(summary_json.read_text(encoding="utf-8"))
            report_text = report_html.read_text(encoding="utf-8")

        self.assertEqual([row["custom_field"] for row in rows], ["first", "second"])
        self.assertEqual(rows[0]["pair_accept_label"], "1")
        self.assertEqual(rows[0]["pair_accept_weight"], "1.500000")
        self.assertEqual(rows[0]["gate_accept_reason"], "gate_selected_clean_pfm")
        self.assertEqual(rows[1]["pair_accept_label"], "0")
        self.assertEqual(rows[1]["pair_accept_weight"], "4.000000")
        self.assertEqual(rows[1]["gate_accept_reason"], "gate_fallback_lightglue")
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["accept_rows"], 1)
        self.assertEqual(summary["reject_rows"], 1)
        self.assertEqual(saved["accept_rows"], 1)
        self.assertIn("Gate acceptance training manifest", report_text)

    def test_rejects_fresh_sources_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_manifest = root / "pairs.csv"
            hybrid_rows = root / "hybrid_rows.csv"
            _write_csv(pair_manifest, PAIR_FIELDS, [_pair_row("1", "base", "mid_01", "row")])
            row = _hybrid_row("1", "base", "mid_01", "1", "0", "1.0", "0")
            row["source_name"] = "phase37_fresh_holdout"
            _write_csv(hybrid_rows, list(row.keys()), [row])

            with self.assertRaisesRegex(ValueError, "fresh"):
                build_gate_acceptance_manifest(
                    sources=[SourceSpec("fresh4", pair_manifest, hybrid_rows)],
                    output_manifest=root / "out.csv",
                    summary_json=root / "summary.json",
                    report_html=root / "index.html",
                )

    def test_can_repeat_accept_rows_to_reach_target_accept_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_manifest = root / "pairs.csv"
            hybrid_rows = root / "hybrid_rows.csv"
            output_manifest = root / "balanced.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"
            pair_rows = [
                _pair_row("0", "base_accept", "mid_01", "accept"),
                _pair_row("1", "base_reject_a", "mid_01", "reject_a"),
                _pair_row("2", "base_reject_b", "mid_01", "reject_b"),
                _pair_row("3", "base_reject_c", "mid_01", "reject_c"),
            ]
            _write_csv(pair_manifest, PAIR_FIELDS, pair_rows)
            hybrid_rows_data = [
                _hybrid_row("0", "base_accept", "mid_01", "1", "0", "1.0", "0"),
                _hybrid_row("1", "base_reject_a", "mid_01", "0", "0", "1.0", "4"),
                _hybrid_row("2", "base_reject_b", "mid_01", "0", "0", "1.0", "5"),
                _hybrid_row("3", "base_reject_c", "mid_01", "0", "0", "1.0", "6"),
            ]
            _write_csv(hybrid_rows, list(hybrid_rows_data[0].keys()), hybrid_rows_data)

            summary = build_gate_acceptance_manifest(
                sources=[SourceSpec("dev_source", pair_manifest, hybrid_rows)],
                output_manifest=output_manifest,
                summary_json=summary_json,
                report_html=report_html,
                target_accept_fraction=0.5,
            )

            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        labels = [row["pair_accept_label"] for row in rows]
        self.assertEqual(labels.count("1"), 3)
        self.assertEqual(labels.count("0"), 3)
        self.assertGreaterEqual(summary["accept_fraction"], 0.5)
        self.assertEqual(summary["source_rows"], 4)
        self.assertEqual(summary["balanced_rows"], 6)
        self.assertTrue(any(row["pair_accept_label"] == "1" for row in rows[:2]))


if __name__ == "__main__":
    unittest.main()
