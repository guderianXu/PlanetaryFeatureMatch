#!/usr/bin/env python3
"""Build pair-delta CSVs from pair manifests and filtered summary CSVs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path


PAIR_DELTA_FIELDS = [
    "split",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
    "baseline_matches",
    "baseline_correct",
    "baseline_wrong",
    "baseline_precision",
    "candidate_matches",
    "candidate_correct",
    "candidate_wrong",
    "candidate_precision",
    "match_delta",
    "correct_delta",
    "wrong_delta",
    "precision_delta",
    "source_name",
]


@dataclass(frozen=True)
class PairDeltaSummarySource:
    split: str
    pair_manifest: Path
    baseline_summary: Path
    candidate_summary: Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _base_id(pair: dict[str, str], summary: dict[str, str]) -> str:
    return pair.get("reference_base_id", "") or pair.get("base_id", "") or summary.get("base_id", "")


def build_pair_delta_rows(
    pair_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    split: str,
    source_name: str = "",
) -> list[dict[str, str]]:
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("baseline and candidate summaries must have the same row count")
    if len(baseline_rows) > len(pair_rows):
        raise ValueError("pair manifest must contain at least as many rows as the summaries")

    rows: list[dict[str, str]] = []
    for ordinal, (pair, baseline, candidate) in enumerate(zip(pair_rows, baseline_rows, candidate_rows)):
        baseline_matches = _int_value(baseline, "matches")
        baseline_correct = _int_value(baseline, "correct")
        baseline_wrong = _int_value(baseline, "wrong")
        baseline_precision = _float_value(baseline, "precision")
        candidate_matches = _int_value(candidate, "matches")
        candidate_correct = _int_value(candidate, "correct")
        candidate_wrong = _int_value(candidate, "wrong")
        candidate_precision = _float_value(candidate, "precision")
        rows.append(
            {
                "split": split,
                "pair_index": pair.get("pair_index", str(ordinal)),
                "base_id": _base_id(pair, baseline),
                "reference_variant": pair.get("reference_variant", ""),
                "target_variant": pair.get("target_variant", "") or baseline.get("target_variant", ""),
                "baseline_matches": str(baseline_matches),
                "baseline_correct": str(baseline_correct),
                "baseline_wrong": str(baseline_wrong),
                "baseline_precision": f"{baseline_precision:.6f}",
                "candidate_matches": str(candidate_matches),
                "candidate_correct": str(candidate_correct),
                "candidate_wrong": str(candidate_wrong),
                "candidate_precision": f"{candidate_precision:.6f}",
                "match_delta": str(candidate_matches - baseline_matches),
                "correct_delta": str(candidate_correct - baseline_correct),
                "wrong_delta": str(candidate_wrong - baseline_wrong),
                "precision_delta": f"{candidate_precision - baseline_precision:.6f}",
                "source_name": source_name,
            }
        )
    return rows


def build_from_sources(sources: list[PairDeltaSummarySource], *, source_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in sources:
        rows.extend(
            build_pair_delta_rows(
                _read_csv_rows(source.pair_manifest),
                _read_csv_rows(source.baseline_summary),
                _read_csv_rows(source.candidate_summary),
                split=source.split,
                source_name=source_name,
            )
        )
    for index, row in enumerate(rows):
        row["pair_index"] = str(index)
    return rows


def write_delta_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_DELTA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    by_split: dict[str, dict[str, int]] = {}
    by_variant: dict[str, dict[str, int]] = {}
    for row in rows:
        for key, bucket in (("split", by_split), ("target_variant", by_variant)):
            name = row.get(key, "")
            item = bucket.setdefault(name, {"rows": 0, "match_delta": 0, "correct_delta": 0, "wrong_delta": 0})
            item["rows"] += 1
            item["match_delta"] += _int_value(row, "match_delta")
            item["correct_delta"] += _int_value(row, "correct_delta")
            item["wrong_delta"] += _int_value(row, "wrong_delta")
    return {
        "rows": len(rows),
        "match_delta": sum(_int_value(row, "match_delta") for row in rows),
        "correct_delta": sum(_int_value(row, "correct_delta") for row in rows),
        "wrong_delta": sum(_int_value(row, "wrong_delta") for row in rows),
        "by_split": by_split,
        "by_variant": by_variant,
    }


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report_html(path: Path, *, output_csv: Path, summary: dict[str, object], sources: list[PairDeltaSummarySource]) -> None:
    source_payload = [
        {
            "split": source.split,
            "pair_manifest": str(source.pair_manifest),
            "baseline_summary": str(source.baseline_summary),
            "candidate_summary": str(source.candidate_summary),
        }
        for source in sources
    ]
    payload = {"output_csv": str(output_csv), "summary": summary, "sources": source_payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                '<head><meta charset="utf-8"><title>Pair delta summary</title></head>',
                "<body>",
                "<h1>Pair delta summary</h1>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )


def parse_source(value: str) -> PairDeltaSummarySource:
    parts = value.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--source must be split,pair_manifest,baseline_all_filtered_summary,candidate_all_filtered_summary"
        )
    split, pair_manifest, baseline_summary, candidate_summary = parts
    return PairDeltaSummarySource(
        split=split,
        pair_manifest=Path(pair_manifest),
        baseline_summary=Path(baseline_summary),
        candidate_summary=Path(candidate_summary),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=parse_source, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--source-name", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_from_sources(args.source, source_name=str(args.source_name))
    summary = summarize_rows(rows)
    write_delta_csv(args.output_csv, rows)
    write_summary_json(args.summary_json, summary)
    write_report_html(args.output_html, output_csv=args.output_csv, summary=summary, sources=args.source)
    print(f"pair_delta_rows={len(rows)} output={args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
