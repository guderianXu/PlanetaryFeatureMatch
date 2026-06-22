#!/usr/bin/env python3
"""Sweep a true-geometry overlap gate over PFM pair summaries."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LIGHTGLUE_LABEL = "LightGlue-SIFT-MAGSAC-min16"


@dataclass(frozen=True)
class Source:
    name: str
    pfm_summary: Path
    lightglue_metrics: Path


def parse_source(value: str) -> Source:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--source must use name,pfm_summary,lightglue_metrics")
    name, pfm_summary, lightglue_metrics = parts
    if not name:
        raise argparse.ArgumentTypeError("source name must not be empty")
    return Source(name=name, pfm_summary=Path(pfm_summary), lightglue_metrics=Path(lightglue_metrics))


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


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        threshold = float(item)
        if not math.isfinite(threshold):
            raise ValueError(f"invalid threshold: {item!r}")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def _lightglue_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in _read_csv_rows(path) if row.get("label") == LIGHTGLUE_LABEL]


def evaluate_threshold(sources: Sequence[Source], threshold: float) -> dict[str, object]:
    item: dict[str, object] = {
        "threshold": threshold,
        "rows": 0,
        "kept_pairs": 0,
        "rejected_pairs": 0,
        "pfm_matches": 0,
        "pfm_correct": 0,
        "pfm_wrong": 0,
        "lightglue_matches": 0,
        "lightglue_correct": 0,
        "lightglue_wrong": 0,
    }
    for source in sources:
        pfm_rows = _read_csv_rows(source.pfm_summary)
        lightglue_rows = _lightglue_rows(source.lightglue_metrics)
        if len(pfm_rows) != len(lightglue_rows):
            raise ValueError(
                f"row count mismatch for {source.name}: PFM={len(pfm_rows)} LightGlue={len(lightglue_rows)}"
            )
        item["rows"] = int(item["rows"]) + len(pfm_rows)
        item["lightglue_matches"] = int(item["lightglue_matches"]) + sum(
            _int_value(row, "matches") for row in lightglue_rows
        )
        item["lightglue_correct"] = int(item["lightglue_correct"]) + sum(
            _int_value(row, "correct") for row in lightglue_rows
        )
        item["lightglue_wrong"] = int(item["lightglue_wrong"]) + sum(
            _int_value(row, "wrong") for row in lightglue_rows
        )
        for row in pfm_rows:
            valid_fraction = _float_value(row, "valid_fraction")
            if valid_fraction >= threshold:
                item["kept_pairs"] = int(item["kept_pairs"]) + 1
                item["pfm_matches"] = int(item["pfm_matches"]) + _int_value(row, "matches")
                item["pfm_correct"] = int(item["pfm_correct"]) + _int_value(row, "correct")
                item["pfm_wrong"] = int(item["pfm_wrong"]) + _int_value(row, "wrong")
            else:
                item["rejected_pairs"] = int(item["rejected_pairs"]) + 1
    matches = int(item["pfm_matches"])
    lg_matches = int(item["lightglue_matches"])
    item["pfm_precision"] = int(item["pfm_correct"]) / matches if matches else 0.0
    item["lightglue_precision"] = int(item["lightglue_correct"]) / lg_matches if lg_matches else 0.0
    item["correct_delta_vs_lightglue"] = int(item["pfm_correct"]) - int(item["lightglue_correct"])
    item["wrong_delta_vs_lightglue"] = int(item["pfm_wrong"]) - int(item["lightglue_wrong"])
    item["precision_delta_vs_lightglue"] = float(item["pfm_precision"]) - float(item["lightglue_precision"])
    return item


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "threshold",
        "rows",
        "kept_pairs",
        "rejected_pairs",
        "pfm_matches",
        "pfm_correct",
        "pfm_wrong",
        "pfm_precision",
        "lightglue_matches",
        "lightglue_correct",
        "lightglue_wrong",
        "lightglue_precision",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "precision_delta_vs_lightglue",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["threshold"] = f"{float(row['threshold']):.6f}"
            out["pfm_precision"] = f"{float(row['pfm_precision']):.6f}"
            out["lightglue_precision"] = f"{float(row['lightglue_precision']):.6f}"
            out["precision_delta_vs_lightglue"] = f"{float(row['precision_delta_vs_lightglue']):.6f}"
            writer.writerow(out)


def write_html(path: Path, rows: Sequence[dict[str, object]], best: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(row[field]))}</td>" for field in fields) + "</tr>"
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Geometry Overlap Gate Sweep</title>",
                "<h1>Geometry Overlap Gate Sweep</h1>",
                "<p>gate=<code>keep PFM only when valid_fraction >= threshold</code></p>",
                f"<p>best_threshold=<code>{html.escape(str(best.get('threshold', '')))}</code></p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr>" + "".join(f"<th>{html.escape(field)}</th>" for field in fields) + "</tr>",
                *table_rows,
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_selected_html(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    if not isinstance(summary, dict):
        raise TypeError("selected summary must be a dictionary")
    fields = list(summary.keys())
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Selected Geometry Overlap Gate</title>",
                "<h1>Selected Geometry Overlap Gate</h1>",
                "<p>gate=<code>keep PFM only when valid_fraction >= selected_threshold</code></p>",
                f"<p>selected_threshold=<code>{html.escape(str(payload['selected_threshold']))}</code></p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr>" + "".join(f"<th>{html.escape(field)}</th>" for field in fields) + "</tr>",
                "<tr>" + "".join(f"<td>{html.escape(str(summary[field]))}</td>" for field in fields) + "</tr>",
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--report-html", type=Path)
    parser.add_argument("--selected-threshold", type=float)
    parser.add_argument("--selected-summary-json", type=Path)
    parser.add_argument("--selected-report-html", type=Path)
    args = parser.parse_args(argv)
    if (args.selected_summary_json or args.selected_report_html) and args.selected_threshold is None:
        parser.error("--selected-threshold is required when writing selected gate outputs")
    if args.selected_threshold is not None and not math.isfinite(args.selected_threshold):
        parser.error("--selected-threshold must be finite")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = parse_thresholds(args.thresholds)
    rows = [evaluate_threshold(args.source, threshold) for threshold in thresholds]
    best = max(
        rows,
        key=lambda row: (
            int(row["pfm_correct"]),
            -int(row["pfm_wrong"]),
            float(row["pfm_precision"]),
            int(row["kept_pairs"]),
        ),
    )
    write_csv(args.output_csv, rows)
    payload = {
        "sources": [source.name for source in args.source],
        "thresholds": thresholds,
        "best_threshold": best,
        "rows": rows,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report_html:
        write_html(args.report_html, rows, best)
    if args.selected_threshold is not None:
        selected = evaluate_threshold(args.source, args.selected_threshold)
        selected_payload = {
            "sources": [source.name for source in args.source],
            "lightglue_label": LIGHTGLUE_LABEL,
            "gate": "keep PFM only when valid_fraction >= selected_threshold",
            "selected_threshold": args.selected_threshold,
            "summary": selected,
        }
        if args.selected_summary_json:
            args.selected_summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.selected_summary_json.write_text(
                json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.selected_report_html:
            write_selected_html(args.selected_report_html, selected_payload)
    print(json.dumps(best, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
