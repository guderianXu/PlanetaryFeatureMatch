#!/usr/bin/env python3
"""Apply a true-geometry per-match filter to PFM match-detail CSVs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LIGHTGLUE_LABEL = "LightGlue-SIFT-MAGSAC-min16"

PAIR_FIELDS = [
    "source",
    "split",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
]

PAIR_SUMMARY_FIELDS = [
    *PAIR_FIELDS,
    "valid_fraction",
    "raw_matches",
    "raw_correct",
    "raw_wrong",
    "kept_matches",
    "kept_correct",
    "kept_wrong",
    "kept_precision",
    "reject_reason",
]


@dataclass(frozen=True)
class Source:
    name: str
    match_details: Path
    lightglue_metrics: Path


def parse_source(value: str) -> Source:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--source must use name,match_details,lightglue_metrics")
    name, match_details, lightglue_metrics = parts
    if not name:
        raise argparse.ArgumentTypeError("source name must not be empty")
    return Source(name=name, match_details=Path(match_details), lightglue_metrics=Path(lightglue_metrics))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _pair_key(source: Source, row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        source.name,
        source.name,
        row.get("pair_index", ""),
        row.get("base_id", ""),
        row.get("reference_variant", ""),
        row.get("target_variant", ""),
    )


def _lightglue_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in _read_csv_rows(path) if row.get("label") == LIGHTGLUE_LABEL]


def _finalize(item: dict[str, object]) -> dict[str, object]:
    pfm_matches = int(item["pfm_matches"])
    lightglue_matches = int(item["lightglue_matches"])
    item["pfm_precision"] = int(item["pfm_correct"]) / pfm_matches if pfm_matches else 0.0
    item["lightglue_precision"] = (
        int(item["lightglue_correct"]) / lightglue_matches if lightglue_matches else 0.0
    )
    item["correct_delta_vs_lightglue"] = int(item["pfm_correct"]) - int(item["lightglue_correct"])
    item["wrong_delta_vs_lightglue"] = int(item["pfm_wrong"]) - int(item["lightglue_wrong"])
    item["precision_delta_vs_lightglue"] = float(item["pfm_precision"]) - float(item["lightglue_precision"])
    return item


def _empty_metrics() -> dict[str, object]:
    return {
        "rows": 0,
        "pairs_with_match_details": 0,
        "kept_pairs": 0,
        "rejected_pairs": 0,
        "pfm_matches": 0,
        "pfm_correct": 0,
        "pfm_wrong": 0,
        "lightglue_matches": 0,
        "lightglue_correct": 0,
        "lightglue_wrong": 0,
    }


def _add_int_metrics(target: dict[str, object], source: dict[str, object], keys: Sequence[str]) -> None:
    for key in keys:
        target[key] = int(target.get(key, 0)) + int(source.get(key, 0))


def _add_lightglue_metrics(target: dict[str, object], row: dict[str, str]) -> None:
    target["lightglue_matches"] = int(target["lightglue_matches"]) + _int_value(row, "matches")
    target["lightglue_correct"] = int(target["lightglue_correct"]) + _int_value(row, "correct")
    target["lightglue_wrong"] = int(target["lightglue_wrong"]) + _int_value(row, "wrong")


def summarize_source(
    source: Source,
    *,
    max_error_px: float,
    min_valid_fraction: float,
) -> tuple[list[dict[str, str]], dict[str, object], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    match_rows = _read_csv_rows(source.match_details)
    lightglue_rows = _lightglue_rows(source.lightglue_metrics)
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in match_rows:
        groups[_pair_key(source, row)].append(row)

    pair_rows: list[dict[str, str]] = []
    summary = _empty_metrics()
    summary["source"] = source.name
    summary["rows"] = len(lightglue_rows) if lightglue_rows else len(groups)
    summary["pairs_with_match_details"] = len(groups)
    by_split: dict[str, dict[str, object]] = defaultdict(_empty_metrics)
    by_variant: dict[str, dict[str, object]] = defaultdict(_empty_metrics)

    for row in lightglue_rows:
        _add_lightglue_metrics(summary, row)
        split_bucket = by_split[source.name]
        split_bucket["rows"] = max(int(split_bucket["rows"]), len(lightglue_rows))
        _add_lightglue_metrics(split_bucket, row)
        target_variant = row.get("target_variant", "")
        if target_variant:
            variant_bucket = by_variant[target_variant]
            variant_bucket["rows"] = int(variant_bucket["rows"]) + 1
            _add_lightglue_metrics(variant_bucket, row)

    for key in sorted(groups):
        rows = groups[key]
        split = key[1]
        target_variant = key[5]
        valid_fraction = _float_value(rows[0], "valid_fraction")
        raw_correct = sum(1 for row in rows if _int_value(row, "correct") > 0)
        raw_matches = len(rows)
        reject_reason = ""
        if valid_fraction < min_valid_fraction:
            kept_rows: list[dict[str, str]] = []
            reject_reason = "low_valid_fraction"
            summary["rejected_pairs"] = int(summary["rejected_pairs"]) + 1
        else:
            kept_rows = [row for row in rows if _float_value(row, "error_px", float("inf")) <= max_error_px]
            summary["kept_pairs"] = int(summary["kept_pairs"]) + 1
        kept_correct = sum(1 for row in kept_rows if _int_value(row, "correct") > 0)
        kept_matches = len(kept_rows)
        kept_wrong = kept_matches - kept_correct
        summary["pfm_matches"] = int(summary["pfm_matches"]) + kept_matches
        summary["pfm_correct"] = int(summary["pfm_correct"]) + kept_correct
        summary["pfm_wrong"] = int(summary["pfm_wrong"]) + kept_wrong
        for bucket in (by_split[split], by_variant[target_variant]):
            bucket["rows"] = int(bucket["rows"]) + 1 if bucket is by_variant[target_variant] else int(bucket["rows"])
            bucket["pairs_with_match_details"] = int(bucket["pairs_with_match_details"]) + 1
            bucket["kept_pairs"] = int(bucket["kept_pairs"]) + (1 if reject_reason == "" else 0)
            bucket["rejected_pairs"] = int(bucket["rejected_pairs"]) + (1 if reject_reason else 0)
            bucket["pfm_matches"] = int(bucket["pfm_matches"]) + kept_matches
            bucket["pfm_correct"] = int(bucket["pfm_correct"]) + kept_correct
            bucket["pfm_wrong"] = int(bucket["pfm_wrong"]) + kept_wrong
        pair_rows.append(
            {
                "source": key[0],
                "split": key[1],
                "pair_index": key[2],
                "base_id": key[3],
                "reference_variant": key[4],
                "target_variant": key[5],
                "valid_fraction": f"{valid_fraction:.6f}",
                "raw_matches": str(raw_matches),
                "raw_correct": str(raw_correct),
                "raw_wrong": str(raw_matches - raw_correct),
                "kept_matches": str(kept_matches),
                "kept_correct": str(kept_correct),
                "kept_wrong": str(kept_wrong),
                "kept_precision": f"{kept_correct / kept_matches:.6f}" if kept_matches else "0.000000",
                "reject_reason": reject_reason,
            }
        )
    return (
        pair_rows,
        _finalize(summary),
        {name: _finalize(dict(metrics)) for name, metrics in by_split.items()},
        {name: _finalize(dict(metrics)) for name, metrics in by_variant.items()},
    )


def _write_csv(path: Path, rows: Sequence[dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = payload["aggregate"]
    if not isinstance(aggregate, dict):
        raise TypeError("aggregate summary must be a dict")
    rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in aggregate.items()
    )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>True Geometry Match Filter</title>",
                "<h1>True Geometry Match Filter</h1>",
                "<p>gate=<code>valid_fraction >= min_valid_fraction and error_px <= max_error_px</code></p>",
                f"<p>max_error_px=<code>{html.escape(str(payload['max_error_px']))}</code></p>",
                f"<p>min_valid_fraction=<code>{html.escape(str(payload['min_valid_fraction']))}</code></p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>metric</th><th>value</th></tr>",
                rows,
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-error-px", type=float, default=5.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_error_px < 0.0 or not math.isfinite(args.max_error_px):
        raise ValueError("--max-error-px must be finite and nonnegative")
    if args.min_valid_fraction < 0.0 or not math.isfinite(args.min_valid_fraction):
        raise ValueError("--min-valid-fraction must be finite and nonnegative")

    output_dir = args.output_dir
    all_pair_rows: list[dict[str, str]] = []
    by_source: dict[str, dict[str, object]] = {}
    by_split: dict[str, dict[str, object]] = defaultdict(_empty_metrics)
    by_variant: dict[str, dict[str, object]] = defaultdict(_empty_metrics)
    aggregate = _empty_metrics()
    for source in args.source:
        pair_rows, summary, source_by_split, source_by_variant = summarize_source(
            source,
            max_error_px=float(args.max_error_px),
            min_valid_fraction=float(args.min_valid_fraction),
        )
        all_pair_rows.extend(pair_rows)
        by_source[source.name] = summary
        _add_int_metrics(aggregate, summary, aggregate.keys())
        for split, split_summary in source_by_split.items():
            _add_int_metrics(by_split[split], split_summary, aggregate.keys())
        for variant, variant_summary in source_by_variant.items():
            _add_int_metrics(by_variant[variant], variant_summary, aggregate.keys())
    aggregate = _finalize(aggregate)
    payload = {
        "sources": [source.name for source in args.source],
        "max_error_px": float(args.max_error_px),
        "min_valid_fraction": float(args.min_valid_fraction),
        "lightglue_label": LIGHTGLUE_LABEL,
        "filter": "valid_fraction >= min_valid_fraction and error_px <= max_error_px",
        "aggregate": aggregate,
        "by_source": by_source,
        "by_split": {name: _finalize(dict(metrics)) for name, metrics in by_split.items()},
        "by_variant": {name: _finalize(dict(metrics)) for name, metrics in by_variant.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "pair_summary.csv", all_pair_rows, PAIR_SUMMARY_FIELDS)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_html(output_dir / "index.html", payload)
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
