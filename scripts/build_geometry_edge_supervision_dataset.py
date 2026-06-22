#!/usr/bin/env python3
"""Build per-match true-geometry supervision rows from lazy visual match details."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from train_match_detail_filter_calibrator import build_training_rows


IDENTITY_FIELDS = [
    "source_name",
    "label",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
    "split",
    "pair_type",
    "match_index",
    "point_a_x_px",
    "point_a_y_px",
    "point_b_x_px",
    "point_b_y_px",
]

OBSERVABLE_FIELDS = [
    "score",
    "pair_logit",
    "row_dustbin_logit",
    "col_dustbin_logit",
    "positive_vs_dustbin_margin",
    "raw_similarity",
    "raw_margin",
    "accept_logit",
    "accept_probability",
]

GEOMETRY_FIELDS = [
    "geometry_valid_label",
    "geometry_invalid_label",
    "geometry_hard_negative_label",
    "geometry_visibility_label",
    "geometry_supervision_weight",
    "geometry_reprojection_error_px",
    "geometry_valid_fraction",
    "geometry_reason",
]

TRUE_GEOMETRY_DIAGNOSTIC_FIELDS = [
    "error_px",
    "correct",
    "valid_fraction",
]

FORBIDDEN_FEATURE_FIELDS = {
    "feature_valid_fraction",
}


@dataclass(frozen=True)
class MatchDetailSource:
    name: str
    rows: Sequence[dict[str, str]]
    path: Path | None = None


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        rows = [
            {key: ("" if value is None else value) for key, value in row.items() if key is not None}
            for row in reader
        ]
    return list(reader.fieldnames), rows


def _optional_float(row: dict[str, str], field: str) -> float | None:
    value = _clean(row.get(field, ""))
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _validate_config(
    *,
    max_error_px: float,
    min_valid_fraction: float,
    hard_negative_error_px: float,
    positive_weight: float,
    invalid_weight: float,
    hard_negative_weight: float,
    low_visibility_weight: float,
    missing_geometry_weight: float,
) -> None:
    if not math.isfinite(max_error_px) or max_error_px < 0.0:
        raise ValueError("max_error_px must be nonnegative")
    if not math.isfinite(hard_negative_error_px) or hard_negative_error_px < max_error_px:
        raise ValueError("hard_negative_error_px must be finite and >= max_error_px")
    if not math.isfinite(min_valid_fraction) or min_valid_fraction < 0.0 or min_valid_fraction > 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1]")
    for name, value in (
        ("positive_weight", positive_weight),
        ("invalid_weight", invalid_weight),
        ("hard_negative_weight", hard_negative_weight),
        ("low_visibility_weight", low_visibility_weight),
        ("missing_geometry_weight", missing_geometry_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be nonnegative")


def _geometry_labels(
    row: dict[str, str],
    *,
    max_error_px: float,
    min_valid_fraction: float,
    hard_negative_error_px: float,
    positive_weight: float,
    invalid_weight: float,
    hard_negative_weight: float,
    low_visibility_weight: float,
    missing_geometry_weight: float,
) -> dict[str, str]:
    error_px = _optional_float(row, "error_px")
    valid_fraction = _optional_float(row, "valid_fraction")
    if error_px is None or valid_fraction is None:
        return {
            "geometry_valid_label": "0",
            "geometry_invalid_label": "1",
            "geometry_hard_negative_label": "0",
            "geometry_visibility_label": "0",
            "geometry_supervision_weight": _format_float(missing_geometry_weight),
            "geometry_reprojection_error_px": "",
            "geometry_valid_fraction": "",
            "geometry_reason": "missing_true_geometry",
        }

    if valid_fraction < min_valid_fraction:
        return {
            "geometry_valid_label": "0",
            "geometry_invalid_label": "1",
            "geometry_hard_negative_label": "0",
            "geometry_visibility_label": "0",
            "geometry_supervision_weight": _format_float(low_visibility_weight),
            "geometry_reprojection_error_px": _format_float(error_px),
            "geometry_valid_fraction": _format_float(valid_fraction),
            "geometry_reason": "low_valid_fraction",
        }

    if error_px <= max_error_px:
        return {
            "geometry_valid_label": "1",
            "geometry_invalid_label": "0",
            "geometry_hard_negative_label": "0",
            "geometry_visibility_label": "1",
            "geometry_supervision_weight": _format_float(positive_weight),
            "geometry_reprojection_error_px": _format_float(error_px),
            "geometry_valid_fraction": _format_float(valid_fraction),
            "geometry_reason": "valid_error_le_threshold",
        }

    if error_px > hard_negative_error_px:
        return {
            "geometry_valid_label": "0",
            "geometry_invalid_label": "1",
            "geometry_hard_negative_label": "1",
            "geometry_visibility_label": "1",
            "geometry_supervision_weight": _format_float(hard_negative_weight),
            "geometry_reprojection_error_px": _format_float(error_px),
            "geometry_valid_fraction": _format_float(valid_fraction),
            "geometry_reason": "valid_error_gt_hard_negative",
        }

    return {
        "geometry_valid_label": "0",
        "geometry_invalid_label": "1",
        "geometry_hard_negative_label": "0",
        "geometry_visibility_label": "1",
        "geometry_supervision_weight": _format_float(invalid_weight),
        "geometry_reprojection_error_px": _format_float(error_px),
        "geometry_valid_fraction": _format_float(valid_fraction),
        "geometry_reason": "valid_error_gt_threshold",
    }


def _copy_fields(row: dict[str, str], fields: Sequence[str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in fields}


def _normalize_feature_columns(rows: Sequence[dict[str, str]]) -> None:
    feature_fields = sorted({field for row in rows for field in row if field.startswith("feature_")})
    for row in rows:
        for field in feature_fields:
            row.setdefault(field, "0")


def build_geometry_edge_rows(
    sources: Sequence[MatchDetailSource],
    *,
    max_error_px: float = 5.0,
    min_valid_fraction: float = 0.10,
    hard_negative_error_px: float = 10.0,
    positive_weight: float = 1.0,
    invalid_weight: float = 1.0,
    hard_negative_weight: float = 3.0,
    low_visibility_weight: float = 0.25,
    missing_geometry_weight: float = 0.0,
) -> list[dict[str, str]]:
    _validate_config(
        max_error_px=max_error_px,
        min_valid_fraction=min_valid_fraction,
        hard_negative_error_px=hard_negative_error_px,
        positive_weight=positive_weight,
        invalid_weight=invalid_weight,
        hard_negative_weight=hard_negative_weight,
        low_visibility_weight=low_visibility_weight,
        missing_geometry_weight=missing_geometry_weight,
    )

    output_rows: list[dict[str, str]] = []
    for source in sources:
        if not source.name:
            raise ValueError("source name must be nonempty")
        training_rows = build_training_rows(source.rows, include_true_geometry_features=False)
        for raw_row, feature_row in zip(source.rows, training_rows):
            output_row = {"source_name": source.name}
            output_row.update(_copy_fields(raw_row, IDENTITY_FIELDS[1:]))
            output_row.update(_copy_fields(raw_row, OBSERVABLE_FIELDS))
            output_row.update(_copy_fields(raw_row, TRUE_GEOMETRY_DIAGNOSTIC_FIELDS))
            output_row.update(
                _geometry_labels(
                    raw_row,
                    max_error_px=max_error_px,
                    min_valid_fraction=min_valid_fraction,
                    hard_negative_error_px=hard_negative_error_px,
                    positive_weight=positive_weight,
                    invalid_weight=invalid_weight,
                    hard_negative_weight=hard_negative_weight,
                    low_visibility_weight=low_visibility_weight,
                    missing_geometry_weight=missing_geometry_weight,
                )
            )
            for field, value in feature_row.items():
                if field.startswith("feature_") and field not in FORBIDDEN_FEATURE_FIELDS:
                    output_row[field] = value
            output_rows.append(output_row)
    _normalize_feature_columns(output_rows)
    return output_rows


def _nested_counts(
    rows: Sequence[dict[str, str]],
    key_field: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = row.get(key_field, "") or "unknown"
        counts[key]["rows"] += 1
        counts[key]["valid_rows"] += 1 if row.get("geometry_valid_label") == "1" else 0
        counts[key]["invalid_rows"] += 1 if row.get("geometry_invalid_label") == "1" else 0
        counts[key]["hard_negative_rows"] += 1 if row.get("geometry_hard_negative_label") == "1" else 0
        counts[key]["low_visibility_rows"] += 1 if row.get("geometry_visibility_label") == "0" else 0
    return {key: dict(value) for key, value in sorted(counts.items())}


def summarize_geometry_edge_rows(
    rows: Sequence[dict[str, str]],
    *,
    sources: Sequence[MatchDetailSource],
    output_csv: Path,
    max_error_px: float,
    min_valid_fraction: float,
    hard_negative_error_px: float,
) -> dict[str, object]:
    source_counts = Counter(row.get("source_name", "") for row in rows)
    reason_counts = Counter(row.get("geometry_reason", "") for row in rows)
    return {
        "sources": [
            {
                "name": source.name,
                "path": "" if source.path is None else str(source.path),
                "rows": len(source.rows),
            }
            for source in sources
        ],
        "output_csv": str(output_csv),
        "rows": len(rows),
        "valid_rows": sum(1 for row in rows if row.get("geometry_valid_label") == "1"),
        "invalid_rows": sum(1 for row in rows if row.get("geometry_invalid_label") == "1"),
        "hard_negative_rows": sum(1 for row in rows if row.get("geometry_hard_negative_label") == "1"),
        "visible_rows": sum(1 for row in rows if row.get("geometry_visibility_label") == "1"),
        "low_visibility_rows": sum(1 for row in rows if row.get("geometry_visibility_label") == "0"),
        "missing_geometry_rows": int(reason_counts.get("missing_true_geometry", 0)),
        "max_error_px": float(max_error_px),
        "min_valid_fraction": float(min_valid_fraction),
        "hard_negative_error_px": float(hard_negative_error_px),
        "source_counts": dict(sorted(source_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "target_variant_counts": _nested_counts(rows, "target_variant"),
        "split_counts": _nested_counts(rows, "split"),
    }


def _output_fieldnames(rows: Sequence[dict[str, str]]) -> list[str]:
    fixed_fields = IDENTITY_FIELDS + OBSERVABLE_FIELDS + TRUE_GEOMETRY_DIAGNOSTIC_FIELDS + GEOMETRY_FIELDS
    feature_fields = sorted({field for row in rows for field in row if field.startswith("feature_")})
    extra_fields = sorted({field for row in rows for field in row} - set(fixed_fields) - set(feature_fields))
    return fixed_fields + extra_fields + feature_fields


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _output_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_html(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Geometry edge supervision dataset</title>",
                "<h1>Geometry edge supervision dataset</h1>",
                "<p>source=<code>true depth/camera reprojection labels from match details</code></p>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_source_spec(spec: str) -> tuple[str, Path]:
    if "," not in spec:
        raise ValueError(f"--source must use name,path.csv format: {spec}")
    name, path_text = spec.split(",", 1)
    name = name.strip()
    path = Path(path_text.strip())
    if not name:
        raise ValueError(f"--source name is empty: {spec}")
    if not path.exists():
        raise FileNotFoundError(f"--source path does not exist: {path}")
    return name, path


def build_geometry_edge_dataset(
    *,
    source_specs: Sequence[str],
    output_csv: Path,
    summary_json: Path,
    output_html: Path,
    max_error_px: float = 5.0,
    min_valid_fraction: float = 0.10,
    hard_negative_error_px: float = 10.0,
    positive_weight: float = 1.0,
    invalid_weight: float = 1.0,
    hard_negative_weight: float = 3.0,
    low_visibility_weight: float = 0.25,
    missing_geometry_weight: float = 0.0,
) -> dict[str, object]:
    sources: list[MatchDetailSource] = []
    for spec in source_specs:
        name, path = _parse_source_spec(spec)
        _, rows = _read_csv(path)
        sources.append(MatchDetailSource(name=name, rows=rows, path=path))
    output_rows = build_geometry_edge_rows(
        sources,
        max_error_px=max_error_px,
        min_valid_fraction=min_valid_fraction,
        hard_negative_error_px=hard_negative_error_px,
        positive_weight=positive_weight,
        invalid_weight=invalid_weight,
        hard_negative_weight=hard_negative_weight,
        low_visibility_weight=low_visibility_weight,
        missing_geometry_weight=missing_geometry_weight,
    )
    summary = summarize_geometry_edge_rows(
        output_rows,
        sources=sources,
        output_csv=output_csv,
        max_error_px=max_error_px,
        min_valid_fraction=min_valid_fraction,
        hard_negative_error_px=hard_negative_error_px,
    )
    _write_csv(output_csv, output_rows)
    _write_json(summary_json, summary)
    _write_html(output_html, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="Repeatable name,path.csv match details source")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--max-error-px", type=float, default=5.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.10)
    parser.add_argument("--hard-negative-error-px", type=float, default=10.0)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--invalid-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--low-visibility-weight", type=float, default=0.25)
    parser.add_argument("--missing-geometry-weight", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_geometry_edge_dataset(
        source_specs=args.source,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        output_html=args.output_html,
        max_error_px=float(args.max_error_px),
        min_valid_fraction=float(args.min_valid_fraction),
        hard_negative_error_px=float(args.hard_negative_error_px),
        positive_weight=float(args.positive_weight),
        invalid_weight=float(args.invalid_weight),
        hard_negative_weight=float(args.hard_negative_weight),
        low_visibility_weight=float(args.low_visibility_weight),
        missing_geometry_weight=float(args.missing_geometry_weight),
    )
    print(
        "geometry_edge_supervision_dataset "
        f"rows={summary['rows']} "
        f"valid_rows={summary['valid_rows']} "
        f"hard_negative_rows={summary['hard_negative_rows']} "
        f"output={args.output_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
