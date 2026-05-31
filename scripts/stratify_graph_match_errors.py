#!/usr/bin/env python3
"""Summarize GraphMatcher report CSVs by difficulty and failure bucket."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        return default if value == "" else int(float(value))
    except (TypeError, ValueError):
        return default


def _int_value_any(row: dict[str, str], keys: tuple[str, ...], default: int = 0) -> int:
    for key in keys:
        if key in row and row.get(key, "") != "":
            return _int_value(row, key, default)
    return default


def precision_bucket(precision: float) -> str:
    if precision < 0.80:
        return "precision_lt_080"
    if precision < 0.90:
        return "precision_080_090"
    if precision < 0.98:
        return "precision_090_098"
    return "precision_ge_098"


def weak_bucket(weak_matches: int, weak_precision: float) -> str:
    if weak_matches <= 0:
        return "weak_none"
    if weak_precision < 0.80:
        return "weak_low"
    if weak_precision < 0.95:
        return "weak_mid"
    return "weak_high"


def summarize_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "group_key": "",
            "dataset_group": "",
            "difficulty": "",
            "precision_bucket": "",
            "weak_bucket": "",
            "pairs": 0,
            "matches": 0,
            "correct": 0,
            "wrong": 0,
            "weak_matches": 0,
            "weak_correct": 0,
        }
    )
    for row in rows:
        matches = _int_value(row, "matches")
        correct = _int_value(row, "correct")
        wrong = max(0, matches - correct)
        precision = _float_value(row, "precision", correct / matches if matches else 0.0)
        weak_matches = _int_value_any(row, ("weak_matches", "weak_texture_matches"))
        weak_correct = _int_value_any(row, ("weak_correct", "weak_texture_correct"))
        weak_precision = weak_correct / weak_matches if weak_matches else 0.0
        dataset_group = row.get("dataset_group") or row.get("group") or "unknown"
        difficulty = row.get("difficulty") or row.get("difficulty_group") or "unknown"
        p_bucket = precision_bucket(precision)
        w_bucket = weak_bucket(weak_matches, weak_precision)
        key = f"{dataset_group}|{difficulty}|{p_bucket}|{w_bucket}"
        bucket = groups[key]
        bucket["group_key"] = key
        bucket["dataset_group"] = dataset_group
        bucket["difficulty"] = difficulty
        bucket["precision_bucket"] = p_bucket
        bucket["weak_bucket"] = w_bucket
        bucket["pairs"] = int(bucket["pairs"]) + 1
        bucket["matches"] = int(bucket["matches"]) + matches
        bucket["correct"] = int(bucket["correct"]) + correct
        bucket["wrong"] = int(bucket["wrong"]) + wrong
        bucket["weak_matches"] = int(bucket["weak_matches"]) + weak_matches
        bucket["weak_correct"] = int(bucket["weak_correct"]) + weak_correct

    summary = []
    for bucket in groups.values():
        matches = int(bucket["matches"])
        weak_matches = int(bucket["weak_matches"])
        bucket["precision"] = 0.0 if matches == 0 else int(bucket["correct"]) / matches
        bucket["weak_precision"] = 0.0 if weak_matches == 0 else int(bucket["weak_correct"]) / weak_matches
        summary.append(dict(bucket))
    summary.sort(key=lambda row: (str(row["dataset_group"]), str(row["difficulty"]), str(row["precision_bucket"])))
    return summary


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_key",
        "dataset_group",
        "difficulty",
        "precision_bucket",
        "weak_bucket",
        "pairs",
        "matches",
        "correct",
        "wrong",
        "precision",
        "weak_matches",
        "weak_correct",
        "weak_precision",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", nargs="+", type=Path, help="match_visual_summary.csv files to summarize")
    parser.add_argument("--output", type=Path, default=Path("runs/graph_match_error_strata.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_rows(read_rows(args.summary_csv))
    write_summary(summary, args.output)
    print(f"wrote {len(summary)} strata to {args.output}")


if __name__ == "__main__":
    main()
