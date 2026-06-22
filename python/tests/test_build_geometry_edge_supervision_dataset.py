import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


MATCH_FIELDS = [
    "label",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
    "split",
    "match_index",
    "point_a_x_px",
    "point_a_y_px",
    "point_b_x_px",
    "point_b_y_px",
    "score",
    "pair_logit",
    "row_dustbin_logit",
    "col_dustbin_logit",
    "positive_vs_dustbin_margin",
    "raw_similarity",
    "raw_margin",
    "accept_logit",
    "accept_probability",
    "error_px",
    "correct",
    "valid_fraction",
]


def _match_row(
    index: int,
    *,
    error_px: str,
    valid_fraction: str,
    correct: str,
    target_variant: str = "extreme_01",
) -> dict[str, str]:
    return {
        "label": "PFM / all-filtered",
        "pair_index": "7",
        "base_id": "pose_007",
        "reference_variant": "nadir",
        "target_variant": target_variant,
        "split": "train",
        "match_index": str(index),
        "point_a_x_px": str(10.0 + index * 10.0),
        "point_a_y_px": "20.0",
        "point_b_x_px": str(13.0 + index * 10.0),
        "point_b_y_px": "22.0",
        "score": f"{0.90 - index * 0.10:.6f}",
        "pair_logit": "2.5",
        "row_dustbin_logit": "-0.5",
        "col_dustbin_logit": "-0.25",
        "positive_vs_dustbin_margin": "3.25",
        "raw_similarity": "0.88",
        "raw_margin": "0.42",
        "accept_logit": "1.7",
        "accept_probability": "0.84",
        "error_px": error_px,
        "correct": correct,
        "valid_fraction": valid_fraction,
    }


def _write_match_details(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class GeometryEdgeSupervisionDatasetTest(unittest.TestCase):
    def test_build_rows_adds_true_geometry_labels_without_feature_leakage(self) -> None:
        import build_geometry_edge_supervision_dataset as mod

        rows = [
            _match_row(0, error_px="2.0", valid_fraction="0.20", correct="1"),
            _match_row(1, error_px="12.0", valid_fraction="0.20", correct="0"),
            _match_row(2, error_px="1.0", valid_fraction="0.03", correct="1"),
        ]

        output = mod.build_geometry_edge_rows(
            [mod.MatchDetailSource(name="phase86", rows=rows)],
            max_error_px=5.0,
            min_valid_fraction=0.10,
            hard_negative_error_px=10.0,
        )
        feature_columns = [name for name in output[0] if name.startswith("feature_")]

        self.assertEqual([row["source_name"] for row in output], ["phase86", "phase86", "phase86"])
        self.assertEqual([row["geometry_valid_label"] for row in output], ["1", "0", "0"])
        self.assertEqual([row["geometry_invalid_label"] for row in output], ["0", "1", "1"])
        self.assertEqual([row["geometry_hard_negative_label"] for row in output], ["0", "1", "0"])
        self.assertEqual([row["geometry_visibility_label"] for row in output], ["1", "1", "0"])
        self.assertEqual(
            [row["geometry_reason"] for row in output],
            [
                "valid_error_le_threshold",
                "valid_error_gt_hard_negative",
                "low_valid_fraction",
            ],
        )
        self.assertEqual(output[0]["geometry_reprojection_error_px"], "2.000000")
        self.assertEqual(output[2]["geometry_valid_fraction"], "0.030000")
        self.assertEqual(output[0]["feature_score"], "0.900000")
        self.assertEqual(output[0]["feature_pair_match_count"], "3")
        self.assertEqual(output[0]["feature_target_variant_extreme_01"], "1")
        self.assertNotIn("feature_true_geometry_error_px", feature_columns)
        self.assertNotIn("feature_valid_fraction", feature_columns)
        self.assertNotIn("feature_error_px", feature_columns)
        self.assertNotIn("feature_correct", feature_columns)

    def test_cli_writes_csv_json_and_html_summary(self) -> None:
        import build_geometry_edge_supervision_dataset as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            match_details = root / "match_details.csv"
            output_csv = root / "out" / "geometry_edges.csv"
            summary_json = root / "out" / "summary.json"
            report_html = root / "out" / "index.html"
            _write_match_details(
                match_details,
                [
                    _match_row(0, error_px="2.0", valid_fraction="0.20", correct="1"),
                    _match_row(1, error_px="12.0", valid_fraction="0.20", correct="0"),
                    _match_row(2, error_px="1.0", valid_fraction="0.03", correct="1"),
                ],
            )

            exit_code = mod.main(
                [
                    "--source",
                    f"phase86,{match_details}",
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(report_html),
                    "--max-error-px",
                    "5.0",
                    "--min-valid-fraction",
                    "0.10",
                    "--hard-negative-error-px",
                    "10.0",
                ]
            )

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                output_rows = list(csv.DictReader(handle))
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            report = report_html.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(output_rows), 3)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["valid_rows"], 1)
        self.assertEqual(summary["hard_negative_rows"], 1)
        self.assertEqual(summary["low_visibility_rows"], 1)
        self.assertEqual(summary["source_counts"]["phase86"], 3)
        self.assertEqual(summary["target_variant_counts"]["extreme_01"]["rows"], 3)
        self.assertIn("Geometry edge supervision dataset", report)


if __name__ == "__main__":
    unittest.main()
