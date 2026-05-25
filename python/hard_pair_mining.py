#!/usr/bin/env python3
"""Select hard synthetic cache pair indices from cache_match_eval summaries."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PAIR_INDEX_RE = re.compile(r"pair_(\d+)\.pt$")


@dataclass(frozen=True)
class HardPair:
    pair_index: int
    precision: float
    matches: int
    pair_pt: str


def parse_pair_index(pair_pt: str) -> int | None:
    match = PAIR_INDEX_RE.search(Path(pair_pt).name)
    if match is None:
        return None
    return int(match.group(1))


def row_to_hard_pair(row: dict[str, str]) -> HardPair | None:
    pair_pt = row.get("pair_pt", "")
    pair_index = parse_pair_index(pair_pt)
    if pair_index is None:
        return None
    try:
        matches = int(row.get("sparse_matches", "0"))
        precision = float(row.get("match_precision", "0"))
    except ValueError:
        return None
    return HardPair(pair_index=pair_index, precision=precision, matches=matches, pair_pt=pair_pt)


def select_hard_pairs(
    rows: Iterable[dict[str, str]],
    *,
    limit: int,
    min_matches: int,
    max_precision: float,
) -> list[HardPair]:
    candidates: list[HardPair] = []
    for row in rows:
        entry = row_to_hard_pair(row)
        if entry is None:
            continue
        if entry.matches < min_matches or entry.precision > max_precision:
            continue
        candidates.append(entry)

    candidates.sort(key=lambda entry: (entry.precision, -entry.matches, entry.pair_index, entry.pair_pt))
    selected: list[HardPair] = []
    seen: set[int] = set()
    for entry in candidates:
        if entry.pair_index in seen:
            continue
        selected.append(entry)
        seen.add(entry.pair_index)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def read_and_select(
    summary: Path,
    *,
    limit: int,
    min_matches: int,
    max_precision: float,
) -> list[HardPair]:
    with summary.open("r", newline="", encoding="utf-8") as handle:
        return select_hard_pairs(
            csv.DictReader(handle),
            limit=limit,
            min_matches=min_matches,
            max_precision=max_precision,
        )


def format_cli_args(entries: Iterable[HardPair]) -> str:
    return " ".join(f"--hard-synthetic-pair-cache-index {entry.pair_index}" for entry in entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard pair indices from cache_match_eval summary.csv")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--min-matches", type=int, default=40)
    parser.add_argument("--max-precision", type=float, default=0.75)
    parser.add_argument("--cli-args", action="store_true", help="Print --hard-synthetic-pair-cache-index arguments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = read_and_select(
        args.summary,
        limit=args.limit,
        min_matches=args.min_matches,
        max_precision=args.max_precision,
    )
    if args.cli_args:
        print(format_cli_args(selected))
    else:
        for entry in selected:
            print(f"{entry.pair_index},{entry.matches},{entry.precision:.6f},{entry.pair_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
