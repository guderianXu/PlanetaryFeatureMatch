#!/usr/bin/env python3
"""Build pair-acceptance training labels from true-geometry overlap."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence


ACCEPTANCE_FIELDS = [
    "pair_accept_label",
    "pair_accept_weight",
    "geometry_accept_source_valid_fraction",
    "geometry_accept_reason",
    "geometry_accept_true_geometry_matches",
    "geometry_accept_true_geometry_correct",
    "geometry_accept_true_geometry_wrong",
    "geometry_accept_true_geometry_precision",
]


def _clean(value: str | None) -> str:
    return "" if value is None else value.strip()


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


def _float_value(row: dict[str, str], field: str, *, default: float = 0.0) -> float:
    value = _clean(row.get(field, ""))
    if value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _validate_thresholds(
    *,
    reject_below_valid_fraction: float,
    accept_at_valid_fraction: float,
    accept_weight: float,
    reject_weight: float,
    ambiguous_accept_weight: float,
    min_true_geometry_matches: int,
    max_true_geometry_wrong: int,
    min_true_geometry_precision: float,
) -> None:
    for name, value in (
        ("reject_below_valid_fraction", reject_below_valid_fraction),
        ("accept_at_valid_fraction", accept_at_valid_fraction),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if reject_below_valid_fraction > accept_at_valid_fraction:
        raise ValueError("reject_below_valid_fraction must be <= accept_at_valid_fraction")
    for name, value in (
        ("accept_weight", accept_weight),
        ("reject_weight", reject_weight),
        ("ambiguous_accept_weight", ambiguous_accept_weight),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if min_true_geometry_matches < 0:
        raise ValueError("min_true_geometry_matches must be nonnegative")
    if max_true_geometry_wrong < 0:
        raise ValueError("max_true_geometry_wrong must be nonnegative")
    if (
        not math.isfinite(min_true_geometry_precision)
        or min_true_geometry_precision < 0.0
        or min_true_geometry_precision > 1.0
    ):
        raise ValueError("min_true_geometry_precision must be in [0, 1]")


def _label_for_valid_fraction(
    valid_fraction: float,
    *,
    reject_below_valid_fraction: float,
    accept_at_valid_fraction: float,
) -> tuple[str, str]:
    if valid_fraction < reject_below_valid_fraction:
        return "0", "low_valid_fraction"
    if valid_fraction >= accept_at_valid_fraction:
        return "1", "observable_valid_fraction"
    return "1", "ambiguous_valid_fraction"


def _int_value(row: dict[str, str], field: str, *, default: int = 0) -> int:
    return int(round(_float_value(row, field, default=float(default))))


def _true_geometry_summary_matches_pair(pair_row: dict[str, str], summary_row: dict[str, str]) -> bool:
    summary_base_id = _clean(summary_row.get("base_id", ""))
    pair_base_id = _clean(pair_row.get("reference_base_id", pair_row.get("base_id", "")))
    summary_target_variant = _clean(summary_row.get("target_variant", ""))
    pair_target_variant = _clean(pair_row.get("target_variant", ""))
    return (
        (not summary_base_id or summary_base_id == pair_base_id)
        and (not summary_target_variant or summary_target_variant == pair_target_variant)
    )


def _true_geometry_stats(summary_row: dict[str, str] | None) -> dict[str, object]:
    if summary_row is None:
        return {
            "matches": 0,
            "correct": 0,
            "wrong": 0,
            "precision": 0.0,
        }
    matches = _int_value(summary_row, "matches")
    correct = _int_value(summary_row, "correct")
    wrong = _int_value(summary_row, "wrong", default=max(0, matches - correct))
    precision = _float_value(
        summary_row,
        "precision",
        default=(float(correct) / float(matches) if matches else 0.0),
    )
    return {
        "matches": matches,
        "correct": correct,
        "wrong": wrong,
        "precision": precision,
    }


def _label_for_true_geometry(
    *,
    valid_fraction: float,
    stats: dict[str, object],
    reject_below_valid_fraction: float,
    min_true_geometry_matches: int,
    max_true_geometry_wrong: int,
    min_true_geometry_precision: float,
) -> tuple[str, str]:
    if valid_fraction < reject_below_valid_fraction:
        return "0", "low_valid_fraction"
    if int(stats["wrong"]) > max_true_geometry_wrong:
        return "0", "true_geometry_wrong_matches"
    if int(stats["matches"]) < min_true_geometry_matches:
        return "0", "insufficient_true_geometry_matches"
    if float(stats["precision"]) < min_true_geometry_precision:
        return "0", "low_true_geometry_precision"
    return "1", "true_geometry_match_count"


def build_geometry_acceptance_rows(
    pair_rows: Sequence[dict[str, str]],
    *,
    true_geometry_summary_rows: Sequence[dict[str, str]] | None = None,
    reject_below_valid_fraction: float = 0.15,
    accept_at_valid_fraction: float = 0.15,
    accept_weight: float = 1.0,
    reject_weight: float = 3.0,
    ambiguous_accept_weight: float = 0.35,
    min_true_geometry_matches: int = 16,
    max_true_geometry_wrong: int = 0,
    min_true_geometry_precision: float = 1.0,
) -> list[dict[str, str]]:
    _validate_thresholds(
        reject_below_valid_fraction=reject_below_valid_fraction,
        accept_at_valid_fraction=accept_at_valid_fraction,
        accept_weight=accept_weight,
        reject_weight=reject_weight,
        ambiguous_accept_weight=ambiguous_accept_weight,
        min_true_geometry_matches=min_true_geometry_matches,
        max_true_geometry_wrong=max_true_geometry_wrong,
        min_true_geometry_precision=min_true_geometry_precision,
    )
    if true_geometry_summary_rows is not None and len(true_geometry_summary_rows) != len(pair_rows):
        raise ValueError(
            "true_geometry_summary_rows must have the same row count as pair_rows "
            f"({len(true_geometry_summary_rows)} != {len(pair_rows)})"
        )

    output_rows: list[dict[str, str]] = []
    for index, pair_row in enumerate(pair_rows):
        valid_fraction = _float_value(pair_row, "valid_fraction", default=0.0)
        summary_row = None if true_geometry_summary_rows is None else true_geometry_summary_rows[index]
        if summary_row is not None and not _true_geometry_summary_matches_pair(pair_row, summary_row):
            raise ValueError(
                "true-geometry summary row does not align with pair manifest row "
                f"at index {index}: pair base={pair_row.get('reference_base_id', '')} "
                f"target_variant={pair_row.get('target_variant', '')}, "
                f"summary base={summary_row.get('base_id', '')} "
                f"target_variant={summary_row.get('target_variant', '')}"
            )
        true_geometry_stats = _true_geometry_stats(summary_row)
        if summary_row is None:
            label, reason = _label_for_valid_fraction(
                valid_fraction,
                reject_below_valid_fraction=reject_below_valid_fraction,
                accept_at_valid_fraction=accept_at_valid_fraction,
            )
        else:
            valid_fraction = _float_value(summary_row, "valid_fraction", default=valid_fraction)
            label, reason = _label_for_true_geometry(
                valid_fraction=valid_fraction,
                stats=true_geometry_stats,
                reject_below_valid_fraction=reject_below_valid_fraction,
                min_true_geometry_matches=min_true_geometry_matches,
                max_true_geometry_wrong=max_true_geometry_wrong,
                min_true_geometry_precision=min_true_geometry_precision,
            )
        if label == "0":
            weight = reject_weight
        elif reason == "ambiguous_valid_fraction":
            weight = ambiguous_accept_weight
        else:
            weight = accept_weight
        output_row = dict(pair_row)
        output_row.update(
            {
                "pair_accept_label": label,
                "pair_accept_weight": _format_float(weight),
                "geometry_accept_source_valid_fraction": _format_float(valid_fraction),
                "geometry_accept_reason": reason,
                "geometry_accept_true_geometry_matches": str(int(true_geometry_stats["matches"])),
                "geometry_accept_true_geometry_correct": str(int(true_geometry_stats["correct"])),
                "geometry_accept_true_geometry_wrong": str(int(true_geometry_stats["wrong"])),
                "geometry_accept_true_geometry_precision": _format_float(float(true_geometry_stats["precision"])),
            }
        )
        output_rows.append(output_row)
    return output_rows


def _summary(
    *,
    input_manifest: Path,
    output_manifest: Path,
    rows: Sequence[dict[str, str]],
    reject_below_valid_fraction: float,
    accept_at_valid_fraction: float,
    accept_weight: float,
    reject_weight: float,
    ambiguous_accept_weight: float,
    true_geometry_summary: Path | None,
    min_true_geometry_matches: int,
    max_true_geometry_wrong: int,
    min_true_geometry_precision: float,
) -> dict[str, object]:
    label_counts = Counter(row.get("pair_accept_label", "") for row in rows)
    reason_counts = Counter(row.get("geometry_accept_reason", "") for row in rows)
    valid_fractions = [
        _float_value(row, "geometry_accept_source_valid_fraction", default=0.0)
        for row in rows
    ]
    return {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "true_geometry_summary": "" if true_geometry_summary is None else str(true_geometry_summary),
        "rows": len(rows),
        "accept_rows": int(label_counts.get("1", 0)),
        "reject_rows": int(label_counts.get("0", 0)),
        "accept_fraction": int(label_counts.get("1", 0)) / len(rows) if rows else 0.0,
        "reject_below_valid_fraction": float(reject_below_valid_fraction),
        "accept_at_valid_fraction": float(accept_at_valid_fraction),
        "accept_weight": float(accept_weight),
        "reject_weight": float(reject_weight),
        "ambiguous_accept_weight": float(ambiguous_accept_weight),
        "min_true_geometry_matches": int(min_true_geometry_matches),
        "max_true_geometry_wrong": int(max_true_geometry_wrong),
        "min_true_geometry_precision": float(min_true_geometry_precision),
        "true_geometry_match_sum": sum(
            _int_value(row, "geometry_accept_true_geometry_matches")
            for row in rows
        ),
        "true_geometry_wrong_sum": sum(
            _int_value(row, "geometry_accept_true_geometry_wrong")
            for row in rows
        ),
        "valid_fraction_min": min(valid_fractions) if valid_fractions else 0.0,
        "valid_fraction_mean": sum(valid_fractions) / len(valid_fractions) if valid_fractions else 0.0,
        "valid_fraction_max": max(valid_fractions) if valid_fractions else 0.0,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
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
                "<title>Geometry acceptance training manifest</title>",
                "<h1>Geometry acceptance training manifest</h1>",
                "<p>source=<code>true dense-warp valid_fraction from depth/camera geometry</code></p>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_geometry_acceptance_manifest(
    *,
    pair_manifest: Path,
    true_geometry_summary: Path | None = None,
    output_manifest: Path,
    summary_json: Path,
    report_html: Path,
    reject_below_valid_fraction: float = 0.15,
    accept_at_valid_fraction: float = 0.15,
    accept_weight: float = 1.0,
    reject_weight: float = 3.0,
    ambiguous_accept_weight: float = 0.35,
    min_true_geometry_matches: int = 16,
    max_true_geometry_wrong: int = 0,
    min_true_geometry_precision: float = 1.0,
) -> dict[str, object]:
    fieldnames, rows = _read_csv(pair_manifest)
    _, true_geometry_rows = _read_csv(true_geometry_summary) if true_geometry_summary is not None else ([], None)
    output_rows = build_geometry_acceptance_rows(
        rows,
        true_geometry_summary_rows=true_geometry_rows,
        reject_below_valid_fraction=reject_below_valid_fraction,
        accept_at_valid_fraction=accept_at_valid_fraction,
        accept_weight=accept_weight,
        reject_weight=reject_weight,
        ambiguous_accept_weight=ambiguous_accept_weight,
        min_true_geometry_matches=min_true_geometry_matches,
        max_true_geometry_wrong=max_true_geometry_wrong,
        min_true_geometry_precision=min_true_geometry_precision,
    )
    output_fields = list(fieldnames)
    for field in ACCEPTANCE_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    summary = _summary(
        input_manifest=pair_manifest,
        output_manifest=output_manifest,
        rows=output_rows,
        reject_below_valid_fraction=reject_below_valid_fraction,
        accept_at_valid_fraction=accept_at_valid_fraction,
        accept_weight=accept_weight,
        reject_weight=reject_weight,
        ambiguous_accept_weight=ambiguous_accept_weight,
        true_geometry_summary=true_geometry_summary,
        min_true_geometry_matches=min_true_geometry_matches,
        max_true_geometry_wrong=max_true_geometry_wrong,
        min_true_geometry_precision=min_true_geometry_precision,
    )
    _write_csv(output_manifest, output_fields, output_rows)
    _write_json(summary_json, summary)
    _write_html(report_html, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--true-geometry-summary", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report-html", type=Path, required=True)
    parser.add_argument("--reject-below-valid-fraction", type=float, default=0.15)
    parser.add_argument("--accept-at-valid-fraction", type=float, default=0.15)
    parser.add_argument("--accept-weight", type=float, default=1.0)
    parser.add_argument("--reject-weight", type=float, default=3.0)
    parser.add_argument("--ambiguous-accept-weight", type=float, default=0.35)
    parser.add_argument("--min-true-geometry-matches", type=int, default=16)
    parser.add_argument("--max-true-geometry-wrong", type=int, default=0)
    parser.add_argument("--min-true-geometry-precision", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_geometry_acceptance_manifest(
        pair_manifest=args.pair_manifest,
        true_geometry_summary=args.true_geometry_summary,
        output_manifest=args.output_manifest,
        summary_json=args.summary_json,
        report_html=args.report_html,
        reject_below_valid_fraction=float(args.reject_below_valid_fraction),
        accept_at_valid_fraction=float(args.accept_at_valid_fraction),
        accept_weight=float(args.accept_weight),
        reject_weight=float(args.reject_weight),
        ambiguous_accept_weight=float(args.ambiguous_accept_weight),
        min_true_geometry_matches=int(args.min_true_geometry_matches),
        max_true_geometry_wrong=int(args.max_true_geometry_wrong),
        min_true_geometry_precision=float(args.min_true_geometry_precision),
    )
    print(
        "geometry_acceptance_manifest "
        f"rows={summary['rows']} "
        f"accept_rows={summary['accept_rows']} "
        f"reject_rows={summary['reject_rows']} "
        f"output={args.output_manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
