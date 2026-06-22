#!/usr/bin/env python3
"""Build train-only supervision manifests from PFM true-geometry selector outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence


ENSURED_PAIR_FIELDS = [
    "reference_path",
    "target_path",
    "pair_type",
    "reference_base_id",
    "target_base_id",
    "reference_variant",
    "target_variant",
    "valid_fraction",
]

TRUE_GEOMETRY_SUPERVISION_FIELDS = [
    "true_geometry_positive_matches",
    "true_geometry_filtered_matches",
    "true_geometry_wrong_matches",
    "true_geometry_supervision_weight",
    "true_geometry_supervision_source",
    "true_geometry_supervision_reason",
    "pair_accept_label",
    "pair_accept_weight",
]


PairKey = tuple[str, str, str, str]


def _clean(value: str | None) -> str:
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


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = _clean(row.get(key, ""))
    if value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _pair_key(row: dict[str, str]) -> PairKey:
    return (
        _clean(row.get("pair_index", "")),
        _clean(row.get("reference_base_id", "")),
        _clean(row.get("target_base_id", "")),
        _clean(row.get("target_variant", "")),
    )


def _selection_key(row: dict[str, str]) -> PairKey:
    return (
        _clean(row.get("manifest_pair_index", "")) or _clean(row.get("pair_index", "")),
        _clean(row.get("reference_base_id", "")),
        _clean(row.get("target_base_id", "")),
        _clean(row.get("target_variant", "")),
    )


def _index_selection_rows(rows: Sequence[dict[str, str]]) -> dict[PairKey, dict[str, str]]:
    indexed: dict[PairKey, dict[str, str]] = {}
    for row in rows:
        key = _selection_key(row)
        if key in indexed:
            raise ValueError(f"duplicate selected pair summary row for key={key}")
        indexed[key] = row
    return indexed


def _detail_rows_by_pair_index(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_clean(row.get("pair_index", ""))].append(row)
    return grouped


def _matching_details(
    selection: dict[str, str],
    details_by_pair: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    pair_index = _clean(selection.get("pair_index", ""))
    rows = list(details_by_pair.get(pair_index, []))
    selected_source = _clean(selection.get("selected_source", ""))
    if selected_source:
        source_rows = [row for row in rows if _clean(row.get("selector_source", "")) == selected_source]
        if source_rows:
            return source_rows
    return rows


def _true_geometry_counts(
    selection: dict[str, str],
    detail_rows: Sequence[dict[str, str]],
) -> tuple[int, int, int]:
    if detail_rows:
        filtered = len(detail_rows)
        positive = sum(1 for row in detail_rows if _int_value(row, "correct") > 0)
        wrong = max(0, filtered - positive)
        return positive, filtered, wrong
    return (
        _int_value(selection, "selected_correct"),
        _int_value(selection, "selected_matches"),
        _int_value(selection, "selected_wrong"),
    )


def _acceptance_label(
    *,
    valid_fraction: float,
    positive_matches: int,
    wrong_matches: int,
    min_accept_valid_fraction: float,
    min_accept_matches: int,
    max_accept_wrong: int,
) -> tuple[str, str]:
    if valid_fraction < min_accept_valid_fraction:
        return "0", "valid_fraction_below_minimum"
    if positive_matches < min_accept_matches:
        return "0", "positive_matches_below_minimum"
    if wrong_matches > max_accept_wrong:
        return "0", "wrong_matches_above_maximum"
    return "1", "true_geometry_clean_high_overlap"


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
                "<title>True geometry supervision manifest</title>",
                "<h1>True geometry supervision manifest</h1>",
                "<p>LightGlue fields are not used as training labels.</p>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _float_summary(values: Sequence[float]) -> dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(finite),
        "mean": sum(finite) / len(finite),
        "max": max(finite),
    }


def build_true_geometry_supervision_manifest(
    *,
    pair_manifest: Path,
    selected_pair_summary: Path,
    selected_match_details: Path,
    output_csv: Path,
    output_json: Path,
    output_html: Path,
    min_accept_valid_fraction: float = 0.10,
    min_accept_matches: int = 16,
    max_accept_wrong: int = 0,
    accept_weight: float = 1.0,
    reject_weight: float = 1.0,
    required_split: str = "train",
) -> dict[str, object]:
    if min_accept_matches < 0:
        raise ValueError("--min-accept-matches must be non-negative")
    if max_accept_wrong < 0:
        raise ValueError("--max-accept-wrong must be non-negative")
    if accept_weight <= 0.0 or reject_weight <= 0.0:
        raise ValueError("--accept-weight and --reject-weight must be positive")
    if not math.isfinite(min_accept_valid_fraction):
        raise ValueError("--min-accept-valid-fraction must be finite")

    pair_fields, pair_rows = _read_csv(pair_manifest)
    _, selection_rows = _read_csv(selected_pair_summary)
    _, detail_rows = _read_csv(selected_match_details)
    selection_by_key = _index_selection_rows(selection_rows)
    details_by_pair = _detail_rows_by_pair_index(detail_rows)

    output_rows: list[dict[str, str]] = []
    missing_selection_rows = 0
    skipped_non_required_split_rows = 0
    valid_fractions: list[float] = []
    for pair_row in pair_rows:
        split = _clean(pair_row.get("split", ""))
        if required_split and split != required_split:
            skipped_non_required_split_rows += 1
            continue
        selection = selection_by_key.get(_pair_key(pair_row))
        if selection is None:
            missing_selection_rows += 1
            continue
        details = _matching_details(selection, details_by_pair)
        positive_matches, filtered_matches, wrong_matches = _true_geometry_counts(selection, details)
        valid_fraction = _float_value(pair_row, "valid_fraction", _float_value(selection, "valid_fraction"))
        label, reason = _acceptance_label(
            valid_fraction=valid_fraction,
            positive_matches=positive_matches,
            wrong_matches=wrong_matches,
            min_accept_valid_fraction=min_accept_valid_fraction,
            min_accept_matches=min_accept_matches,
            max_accept_wrong=max_accept_wrong,
        )
        weight = accept_weight if label == "1" else reject_weight
        output = dict(pair_row)
        for field in ENSURED_PAIR_FIELDS:
            output.setdefault(field, "")
        output.update(
            {
                "true_geometry_positive_matches": str(positive_matches),
                "true_geometry_filtered_matches": str(filtered_matches),
                "true_geometry_wrong_matches": str(wrong_matches),
                "true_geometry_supervision_weight": f"{weight:.6f}",
                "true_geometry_supervision_source": _clean(selection.get("selected_source", "")),
                "true_geometry_supervision_reason": reason,
                "pair_accept_label": label,
                "pair_accept_weight": f"{weight:.6f}",
            }
        )
        valid_fractions.append(valid_fraction)
        output_rows.append(output)

    fieldnames = [*pair_fields]
    for field in [*ENSURED_PAIR_FIELDS, *TRUE_GEOMETRY_SUPERVISION_FIELDS]:
        if field not in fieldnames:
            fieldnames.append(field)
    _write_csv(output_csv, fieldnames, output_rows)

    accept_rows = sum(1 for row in output_rows if row.get("pair_accept_label") == "1")
    reject_rows = len(output_rows) - accept_rows
    summary: dict[str, object] = {
        "pair_manifest": str(pair_manifest),
        "selected_pair_summary": str(selected_pair_summary),
        "selected_match_details": str(selected_match_details),
        "output_csv": str(output_csv),
        "input_pair_rows": len(pair_rows),
        "selected_pair_rows": len(selection_rows),
        "selected_match_detail_rows": len(detail_rows),
        "output_rows": len(output_rows),
        "accept_rows": accept_rows,
        "reject_rows": reject_rows,
        "missing_selection_rows": missing_selection_rows,
        "skipped_non_required_split_rows": skipped_non_required_split_rows,
        "required_split": required_split,
        "min_accept_valid_fraction": min_accept_valid_fraction,
        "min_accept_matches": min_accept_matches,
        "max_accept_wrong": max_accept_wrong,
        "accept_weight": accept_weight,
        "reject_weight": reject_weight,
        "valid_fraction": _float_summary(valid_fractions),
        "uses_lightglue_labels": False,
    }
    _write_json(output_json, summary)
    _write_html(output_html, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--selected-pair-summary", type=Path, required=True)
    parser.add_argument("--selected-match-details", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--required-split", default="train")
    parser.add_argument("--min-accept-valid-fraction", type=float, default=0.10)
    parser.add_argument("--min-accept-matches", type=int, default=16)
    parser.add_argument("--max-accept-wrong", type=int, default=0)
    parser.add_argument("--accept-weight", type=float, default=1.0)
    parser.add_argument("--reject-weight", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_true_geometry_supervision_manifest(
        pair_manifest=args.pair_manifest,
        selected_pair_summary=args.selected_pair_summary,
        selected_match_details=args.selected_match_details,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_html=args.output_html,
        min_accept_valid_fraction=float(args.min_accept_valid_fraction),
        min_accept_matches=int(args.min_accept_matches),
        max_accept_wrong=int(args.max_accept_wrong),
        accept_weight=float(args.accept_weight),
        reject_weight=float(args.reject_weight),
        required_split=str(args.required_split),
    )
    print(
        f"true_geometry_supervision_rows={summary['output_rows']} "
        f"accept={summary['accept_rows']} reject={summary['reject_rows']} "
        f"output={args.output_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
