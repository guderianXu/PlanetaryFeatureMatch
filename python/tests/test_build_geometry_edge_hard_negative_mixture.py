import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


FIELDS = [
    "source_name",
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
    "accept_probability",
    "raw_margin",
    "geometry_valid_label",
    "geometry_invalid_label",
    "geometry_hard_negative_label",
    "geometry_visibility_label",
    "geometry_supervision_weight",
    "geometry_reprojection_error_px",
    "geometry_valid_fraction",
    "geometry_reason",
    "feature_score",
    "feature_accept_probability",
    "feature_pair_homography_residual_px",
]


def _row(
    index: int,
    *,
    source_name: str,
    target_variant: str,
    score: float,
    hard_negative: bool,
    valid: bool = False,
) -> dict[str, str]:
    return {
        "source_name": source_name,
        "label": "PFM / all-filtered",
        "pair_index": str(index // 3),
        "base_id": f"pose_{index // 3}",
        "reference_variant": "nadir",
        "target_variant": target_variant,
        "split": "train",
        "match_index": str(index),
        "point_a_x_px": f"{10.0 + index:.3f}",
        "point_a_y_px": "20.000",
        "point_b_x_px": f"{30.0 + index:.3f}",
        "point_b_y_px": "40.000",
        "score": f"{score:.6f}",
        "accept_probability": f"{score:.6f}",
        "raw_margin": f"{score - 0.1:.6f}",
        "geometry_valid_label": "1" if valid else "0",
        "geometry_invalid_label": "0" if valid else "1",
        "geometry_hard_negative_label": "1" if hard_negative else "0",
        "geometry_visibility_label": "1",
        "geometry_supervision_weight": "3.000000" if hard_negative else "1.000000",
        "geometry_reprojection_error_px": "18.000000" if hard_negative else "2.000000",
        "geometry_valid_fraction": "0.200000",
        "geometry_reason": "valid_error_gt_hard_negative" if hard_negative else "valid_error_le_threshold",
        "feature_score": f"{score:.6f}",
        "feature_accept_probability": f"{score:.6f}",
        "feature_pair_homography_residual_px": "12.000000" if hard_negative else "1.000000",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class GeometryEdgeHardNegativeMixtureTest(unittest.TestCase):
    def test_build_mixture_keeps_base_and_selects_filtered_hard_negatives(self) -> None:
        import build_geometry_edge_hard_negative_mixture as mod

        base_rows = [
            _row(0, source_name="filtered", target_variant="extreme_01", score=0.90, hard_negative=False, valid=True),
            _row(1, source_name="filtered", target_variant="extreme_01", score=0.70, hard_negative=True),
        ]
        hard_rows = [
            _row(3, source_name="raw", target_variant="extreme_01", score=0.95, hard_negative=True),
            _row(4, source_name="raw", target_variant="extreme_01", score=0.60, hard_negative=True),
            _row(5, source_name="raw", target_variant="extreme_02", score=0.99, hard_negative=True),
            _row(6, source_name="raw", target_variant="extreme_01", score=0.99, hard_negative=False, valid=True),
        ]

        output = mod.build_mixture_rows(
            base_rows,
            hard_rows,
            target_variants={"extreme_01"},
            min_score=0.70,
            max_hard_negatives_per_pair=1,
        )

        self.assertEqual(len(output), 3)
        self.assertEqual([row["mixture_source"] for row in output], ["base", "base", "hard_negative"])
        self.assertEqual(output[2]["source_name"], "raw")
        self.assertEqual(output[2]["target_variant"], "extreme_01")
        self.assertEqual(output[2]["score"], "0.950000")
        self.assertEqual(output[2]["geometry_hard_negative_label"], "1")

    def test_cli_writes_csv_json_and_html(self) -> None:
        import build_geometry_edge_hard_negative_mixture as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_csv = root / "base.csv"
            hard_csv = root / "raw.csv"
            output_csv = root / "out" / "mixture.csv"
            summary_json = root / "out" / "summary.json"
            report_html = root / "out" / "index.html"
            _write_csv(base_csv, [_row(0, source_name="filtered", target_variant="extreme_01", score=0.80, hard_negative=False, valid=True)])
            _write_csv(
                hard_csv,
                [
                    _row(3, source_name="raw", target_variant="extreme_01", score=0.90, hard_negative=True),
                    _row(4, source_name="raw", target_variant="extreme_02", score=0.95, hard_negative=True),
                ],
            )

            exit_code = mod.main(
                [
                    "--base-geometry-edges",
                    str(base_csv),
                    "--hard-negative-geometry-edges",
                    str(hard_csv),
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(report_html),
                    "--target-variant",
                    "extreme_01",
                    "--min-score",
                    "0.85",
                ]
            )

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            report = report_html.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["base_rows"], 1)
        self.assertEqual(summary["selected_hard_negative_rows"], 1)
        self.assertEqual(summary["output_rows"], 2)
        self.assertEqual(summary["target_variant_counts"]["extreme_01"]["selected_hard_negative_rows"], 1)
        self.assertIn("Geometry edge hard-negative mixture", report)


if __name__ == "__main__":
    unittest.main()
