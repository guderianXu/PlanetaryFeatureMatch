#!/usr/bin/env python3
"""Summarize fov76 checkpoint delta reports into reusable diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _int_value(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    try:
        return int(round(float(value)))
    except ValueError:
        return 0


def _float_value(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _empty_counter() -> dict[str, int]:
    return {
        "rows": 0,
        "gain_rows": 0,
        "loss_rows": 0,
        "match_delta_sum": 0,
        "correct_delta_sum": 0,
        "wrong_delta_sum": 0,
    }


def summarize_combined_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    by_variant: dict[tuple[str, str, str], dict[str, int]] = defaultdict(_empty_counter)
    selector_reasons: Counter[str] = Counter()
    selected_models: Counter[str] = Counter()
    for row in rows:
        source = row.get("source", "unknown") or "unknown"
        split = row.get("split", "unknown") or "unknown"
        variant = row.get("target_variant", "unknown") or "unknown"
        match_delta = _int_value(row, "match_delta")
        correct_delta = _int_value(row, "correct_delta")
        wrong_delta = _int_value(row, "wrong_delta")

        source_summary = summary.setdefault(source, {})
        split_summary = source_summary.setdefault(split, _empty_counter())
        split_summary["rows"] += 1
        split_summary["gain_rows"] += int(correct_delta > 0)
        split_summary["loss_rows"] += int(correct_delta < 0)
        split_summary["match_delta_sum"] += match_delta
        split_summary["correct_delta_sum"] += correct_delta
        split_summary["wrong_delta_sum"] += wrong_delta

        variant_summary = by_variant[(source, split, variant)]
        variant_summary["rows"] += 1
        variant_summary["gain_rows"] += int(correct_delta > 0)
        variant_summary["loss_rows"] += int(correct_delta < 0)
        variant_summary["match_delta_sum"] += match_delta
        variant_summary["correct_delta_sum"] += correct_delta
        variant_summary["wrong_delta_sum"] += wrong_delta

        reason = row.get("selector_reason", "")
        if reason:
            selector_reasons[reason] += 1
        selected_model = row.get("selected_model", "")
        if selected_model:
            selected_models[selected_model] += 1

    return {
        **summary,
        "by_variant": {"|".join(key): value for key, value in sorted(by_variant.items())},
        "selector_reason_counts": dict(selector_reasons),
        "selected_model_counts": dict(selected_models),
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _top_rows(rows: list[dict[str, str]], *, reverse: bool) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _int_value(row, "correct_delta"),
            -_int_value(row, "wrong_delta"),
            _int_value(row, "match_delta"),
            _float_value(row, "precision_delta"),
        ),
        reverse=reverse,
    )
    selected: list[dict[str, object]] = []
    for row in ordered[:50]:
        selected.append(
            {
                "source": row.get("source", ""),
                "split": row.get("split", ""),
                "base_id": row.get("base_id", ""),
                "target_variant": row.get("target_variant", ""),
                "match_delta": _int_value(row, "match_delta"),
                "correct_delta": _int_value(row, "correct_delta"),
                "wrong_delta": _int_value(row, "wrong_delta"),
                "precision_delta": f"{_float_value(row, 'precision_delta'):.6f}",
                "selected_model": row.get("selected_model", ""),
                "selector_reason": row.get("selector_reason", ""),
            }
        )
    return selected


def _variant_rows(summary: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_variant = summary.get("by_variant", {})
    if not isinstance(by_variant, dict):
        return rows
    for packed_key, values in by_variant.items():
        source, split, variant = str(packed_key).split("|", 2)
        if not isinstance(values, dict):
            continue
        rows.append({"source": source, "split": split, "target_variant": variant, **values})
    return rows


def _write_html(path: Path, summary: dict[str, Any]) -> None:
    body = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    path.write_text(
        "<!doctype html>\n"
        "<html lang=\"zh-CN\">\n"
        "<head><meta charset=\"utf-8\"><title>fov76 delta analysis</title></head>\n"
        "<body><h1>fov76 delta analysis</h1><pre>"
        f"{body}"
        "</pre></body></html>\n",
        encoding="utf-8",
    )


def run_analysis(*, combined_csv: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read_rows(combined_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_combined_rows(rows)
    (output_dir / "delta_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "delta_by_variant.csv",
        _variant_rows(summary),
        [
            "source",
            "split",
            "target_variant",
            "rows",
            "gain_rows",
            "loss_rows",
            "match_delta_sum",
            "correct_delta_sum",
            "wrong_delta_sum",
        ],
    )
    top_fields = [
        "source",
        "split",
        "base_id",
        "target_variant",
        "match_delta",
        "correct_delta",
        "wrong_delta",
        "precision_delta",
        "selected_model",
        "selector_reason",
    ]
    _write_csv(output_dir / "delta_top_gains.csv", _top_rows(rows, reverse=True), top_fields)
    _write_csv(output_dir / "delta_top_losses.csv", _top_rows(rows, reverse=False), top_fields)
    _write_html(output_dir / "index.html", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(combined_csv=args.combined_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
