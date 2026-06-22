#!/usr/bin/env python3
"""Mix filtered geometry-edge rows with raw true-geometry hard negatives."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


EXTRA_FIELDS = [
    "mixture_source",
]


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


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _fmt_float(value: float) -> str:
    return f"{float(value):.6f}"


def _pair_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("split", ""),
        row.get("pair_index", ""),
        row.get("base_id", ""),
        row.get("reference_variant", ""),
        row.get("target_variant", ""),
    )


def _edge_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        *_pair_key(row),
        row.get("point_a_x_px", ""),
        row.get("point_a_y_px", ""),
        row.get("point_b_x_px", ""),
        row.get("point_b_y_px", ""),
    )


def _sort_key(row: dict[str, str]) -> tuple[float, float, float, str]:
    return (
        -_float_value(row, "score"),
        -_float_value(row, "accept_probability"),
        -_float_value(row, "raw_margin"),
        row.get("match_index", ""),
    )


def _is_candidate_hard_negative(
    row: dict[str, str],
    *,
    target_variants: set[str],
    min_score: float,
    min_accept_probability: float,
    min_raw_margin: float,
) -> bool:
    if _int_value(row, "geometry_hard_negative_label") <= 0:
        return False
    if target_variants and row.get("target_variant", "") not in target_variants:
        return False
    if _float_value(row, "score") < min_score:
        return False
    if _float_value(row, "accept_probability") < min_accept_probability:
        return False
    if _float_value(row, "raw_margin") < min_raw_margin:
        return False
    return True


def build_mixture_rows(
    base_rows: Sequence[dict[str, str]],
    hard_negative_rows: Sequence[dict[str, str]],
    *,
    target_variants: set[str] | None = None,
    min_score: float = 0.0,
    min_accept_probability: float = 0.0,
    min_raw_margin: float = -1.0e9,
    max_hard_negatives: int = 0,
    max_hard_negatives_per_pair: int = 0,
    skip_duplicate_edges: bool = True,
) -> list[dict[str, str]]:
    if max_hard_negatives < 0:
        raise ValueError("max_hard_negatives must be nonnegative")
    if max_hard_negatives_per_pair < 0:
        raise ValueError("max_hard_negatives_per_pair must be nonnegative")
    variants = set(target_variants or set())
    output_rows: list[dict[str, str]] = []
    seen_edges: set[tuple[str, ...]] = set()
    for base_row in base_rows:
        row = dict(base_row)
        row["mixture_source"] = "base"
        output_rows.append(row)
        seen_edges.add(_edge_key(base_row))

    candidates = [
        row
        for row in hard_negative_rows
        if _is_candidate_hard_negative(
            row,
            target_variants=variants,
            min_score=min_score,
            min_accept_probability=min_accept_probability,
            min_raw_margin=min_raw_margin,
        )
    ]
    candidates.sort(key=_sort_key)

    selected_by_pair: Counter[tuple[str, str, str, str, str]] = Counter()
    selected_count = 0
    for candidate in candidates:
        if max_hard_negatives and selected_count >= max_hard_negatives:
            break
        if skip_duplicate_edges and _edge_key(candidate) in seen_edges:
            continue
        pair_key = _pair_key(candidate)
        if max_hard_negatives_per_pair and selected_by_pair[pair_key] >= max_hard_negatives_per_pair:
            continue
        row = dict(candidate)
        row["mixture_source"] = "hard_negative"
        output_rows.append(row)
        selected_by_pair[pair_key] += 1
        selected_count += 1
        seen_edges.add(_edge_key(candidate))
    return output_rows


def summarize_rows(
    rows: Sequence[dict[str, str]],
    *,
    base_rows: Sequence[dict[str, str]],
    hard_negative_rows: Sequence[dict[str, str]],
    output_csv: Path,
    target_variants: set[str],
    min_score: float,
    min_accept_probability: float,
    min_raw_margin: float,
    max_hard_negatives: int,
    max_hard_negatives_per_pair: int,
) -> dict[str, object]:
    source_counts = Counter(row.get("mixture_source", "") for row in rows)
    variant_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        variant = row.get("target_variant", "") or "unknown"
        if row.get("mixture_source") == "hard_negative":
            variant_counts[variant]["selected_hard_negative_rows"] += 1
        else:
            variant_counts[variant]["base_rows"] += 1
        variant_counts[variant]["output_rows"] += 1
    return {
        "output_csv": str(output_csv),
        "base_rows": len(base_rows),
        "hard_negative_source_rows": len(hard_negative_rows),
        "selected_hard_negative_rows": int(source_counts.get("hard_negative", 0)),
        "output_rows": len(rows),
        "target_variants": sorted(target_variants),
        "min_score": float(min_score),
        "min_accept_probability": float(min_accept_probability),
        "min_raw_margin": float(min_raw_margin),
        "max_hard_negatives": int(max_hard_negatives),
        "max_hard_negatives_per_pair": int(max_hard_negatives_per_pair),
        "target_variant_counts": {
            variant: dict(counts)
            for variant, counts in sorted(variant_counts.items())
        },
    }


def _fieldnames(input_fields: Sequence[str], rows: Sequence[dict[str, str]]) -> list[str]:
    fields = list(input_fields)
    for field in EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "<title>Geometry edge hard-negative mixture</title>",
                "<h1>Geometry edge hard-negative mixture</h1>",
                "<p>source=<code>PFM geometry-edge rows with true depth/camera hard-negative labels</code></p>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_mixture_dataset(
    *,
    base_geometry_edges: Path,
    hard_negative_geometry_edges: Sequence[Path],
    output_csv: Path,
    summary_json: Path,
    output_html: Path,
    target_variants: set[str] | None = None,
    min_score: float = 0.0,
    min_accept_probability: float = 0.0,
    min_raw_margin: float = -1.0e9,
    max_hard_negatives: int = 0,
    max_hard_negatives_per_pair: int = 0,
    skip_duplicate_edges: bool = True,
) -> dict[str, object]:
    base_fields, base_rows = _read_csv(base_geometry_edges)
    hard_rows: list[dict[str, str]] = []
    hard_fields: list[str] = []
    for path in hard_negative_geometry_edges:
        fields, rows = _read_csv(path)
        hard_fields.extend(field for field in fields if field not in hard_fields)
        hard_rows.extend(rows)
    variants = set(target_variants or set())
    output_rows = build_mixture_rows(
        base_rows,
        hard_rows,
        target_variants=variants,
        min_score=min_score,
        min_accept_probability=min_accept_probability,
        min_raw_margin=min_raw_margin,
        max_hard_negatives=max_hard_negatives,
        max_hard_negatives_per_pair=max_hard_negatives_per_pair,
        skip_duplicate_edges=skip_duplicate_edges,
    )
    summary = summarize_rows(
        output_rows,
        base_rows=base_rows,
        hard_negative_rows=hard_rows,
        output_csv=output_csv,
        target_variants=variants,
        min_score=min_score,
        min_accept_probability=min_accept_probability,
        min_raw_margin=min_raw_margin,
        max_hard_negatives=max_hard_negatives,
        max_hard_negatives_per_pair=max_hard_negatives_per_pair,
    )
    _write_csv(output_csv, _fieldnames([*base_fields, *hard_fields], output_rows), output_rows)
    _write_json(summary_json, summary)
    _write_html(output_html, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-geometry-edges", type=Path, required=True)
    parser.add_argument("--hard-negative-geometry-edges", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--target-variant", action="append", default=[])
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-accept-probability", type=float, default=0.0)
    parser.add_argument("--min-raw-margin", type=float, default=-1.0e9)
    parser.add_argument("--max-hard-negatives", type=int, default=0)
    parser.add_argument("--max-hard-negatives-per-pair", type=int, default=0)
    parser.add_argument("--allow-duplicate-edges", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_mixture_dataset(
        base_geometry_edges=args.base_geometry_edges,
        hard_negative_geometry_edges=list(args.hard_negative_geometry_edges),
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        output_html=args.output_html,
        target_variants=set(args.target_variant),
        min_score=float(args.min_score),
        min_accept_probability=float(args.min_accept_probability),
        min_raw_margin=float(args.min_raw_margin),
        max_hard_negatives=int(args.max_hard_negatives),
        max_hard_negatives_per_pair=int(args.max_hard_negatives_per_pair),
        skip_duplicate_edges=not bool(args.allow_duplicate_edges),
    )
    print(
        "geometry_edge_hard_negative_mixture "
        f"base_rows={summary['base_rows']} "
        f"selected_hard_negative_rows={summary['selected_hard_negative_rows']} "
        f"output_rows={summary['output_rows']} output={args.output_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
