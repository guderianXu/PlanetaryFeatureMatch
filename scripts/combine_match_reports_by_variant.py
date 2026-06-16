#!/usr/bin/env python3
"""Combine two match-detail reports by target variant.

This is used to evaluate inference-time rescue policies such as keeping the
default geo5 filtered matches for most pairs while taking geo8 matches only for
extreme target variants.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MatchSummary:
    label: str
    matches: int
    correct: int
    wrong: int
    precision: float


PAIR_KEY_FIELDS = ("split", "pair_index", "base_id", "reference_variant", "target_variant")


def read_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _pair_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in PAIR_KEY_FIELDS)


def _score(row: dict[str, str]) -> float:
    try:
        return float(row.get("score", "") or 0.0)
    except ValueError:
        return 0.0


def _is_correct(row: dict[str, str]) -> bool:
    return str(row.get("correct", "")).strip().lower() in {"1", "true", "yes", "y"}


def _group_rows(rows: Iterable[dict[str, str]]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_pair_key(row), []).append(row)
    return grouped


def _sorted_pair_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def key(row: dict[str, str]) -> tuple[int, float]:
        try:
            index = int(float(row.get("match_index", "") or 0))
        except ValueError:
            index = 0
        return index, -_score(row)

    return sorted(rows, key=key)


def _with_output_label(rows: list[dict[str, str]], *, output_label: str) -> list[dict[str, str]]:
    labelled: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        source_label = copied.get("label", "")
        copied["label"] = f"{output_label}:{source_label}" if source_label else output_label
        labelled.append(copied)
    return labelled


def _mean_score(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return sum(_score(row) for row in rows) / float(len(rows))


def _rescue_pair_passes(
    *,
    baseline_rows: list[dict[str, str]],
    rescue_rows: list[dict[str, str]],
    rescue_max_baseline_matches: int | None,
    rescue_min_pair_matches: int,
    rescue_max_pair_matches: int | None,
    rescue_min_score_mean: float | None,
) -> bool:
    if rescue_max_baseline_matches is not None and len(baseline_rows) > rescue_max_baseline_matches:
        return False
    if len(rescue_rows) < rescue_min_pair_matches:
        return False
    if rescue_max_pair_matches is not None and len(rescue_rows) > rescue_max_pair_matches:
        return False
    if rescue_min_score_mean is not None and _mean_score(rescue_rows) < rescue_min_score_mean:
        return False
    return True


def combine_match_detail_rows(
    baseline_rows: list[dict[str, str]],
    rescue_rows: list[dict[str, str]],
    *,
    rescue_variants: set[str],
    rescue_min_score: float,
    fallback_if_empty: bool,
    output_label: str = "variant_rescue",
    rescue_max_baseline_matches: int | None = None,
    rescue_min_pair_matches: int = 0,
    rescue_max_pair_matches: int | None = None,
    rescue_min_score_mean: float | None = None,
) -> list[dict[str, str]]:
    baseline_by_pair = _group_rows(baseline_rows)
    rescue_by_pair = _group_rows(rescue_rows)
    pair_keys = sorted(set(baseline_by_pair) | set(rescue_by_pair))
    merged: list[dict[str, str]] = []
    for key in pair_keys:
        target_variant = key[4]
        baseline = _sorted_pair_rows(baseline_by_pair.get(key, []))
        rescue = [row for row in _sorted_pair_rows(rescue_by_pair.get(key, [])) if _score(row) >= rescue_min_score]
        if target_variant in rescue_variants:
            if rescue and _rescue_pair_passes(
                baseline_rows=baseline,
                rescue_rows=rescue,
                rescue_max_baseline_matches=rescue_max_baseline_matches,
                rescue_min_pair_matches=rescue_min_pair_matches,
                rescue_max_pair_matches=rescue_max_pair_matches,
                rescue_min_score_mean=rescue_min_score_mean,
            ):
                chosen = rescue
            else:
                chosen = []
            if fallback_if_empty and not chosen:
                chosen = baseline
        else:
            chosen = baseline
        merged.extend(_with_output_label(chosen, output_label=output_label))
    return merged


def summarize_rows(rows: list[dict[str, str]], *, label: str) -> MatchSummary:
    matches = len(rows)
    correct = sum(1 for row in rows if _is_correct(row))
    wrong = matches - correct
    precision = float(correct) / float(matches) if matches else 0.0
    return MatchSummary(label=label, matches=matches, correct=correct, wrong=wrong, precision=precision)


def summarize_by_variant(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        variant = row.get("target_variant", "") or "unknown"
        bucket = grouped.setdefault(variant, {"matches": 0, "correct": 0, "wrong": 0})
        bucket["matches"] += 1
        if _is_correct(row):
            bucket["correct"] += 1
        else:
            bucket["wrong"] += 1
    summary: list[dict[str, object]] = []
    for variant, bucket in sorted(grouped.items()):
        matches = bucket["matches"]
        correct = bucket["correct"]
        summary.append(
            {
                "variant": variant,
                "matches": matches,
                "correct": correct,
                "wrong": bucket["wrong"],
                "precision": float(correct) / float(matches) if matches else 0.0,
            }
        )
    return summary


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: MatchSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "matches", "correct", "wrong", "precision"])
        writer.writeheader()
        writer.writerow(
            {
                "label": summary.label,
                "matches": summary.matches,
                "correct": summary.correct,
                "wrong": summary.wrong,
                "precision": f"{summary.precision:.12f}",
            }
        )


def write_variant_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "matches", "correct", "wrong", "precision"])
        writer.writeheader()
        for row in rows:
            copied = dict(row)
            copied["precision"] = f"{float(copied['precision']):.12f}"
            writer.writerow(copied)


def write_html(path: Path, *, payload: dict[str, object]) -> None:
    document = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Variant Rescue Combination</title></head>
<body>
<h1>Variant Rescue Combination</h1>
<pre>{html.escape(json.dumps(payload, indent=2, ensure_ascii=False))}</pre>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument("--rescue-details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rescue-variant", action="append", default=[])
    parser.add_argument("--rescue-min-score", type=float, default=-1.0)
    parser.add_argument("--rescue-max-baseline-matches", type=int, default=None)
    parser.add_argument("--rescue-min-pair-matches", type=int, default=0)
    parser.add_argument("--rescue-max-pair-matches", type=int, default=None)
    parser.add_argument("--rescue-min-score-mean", type=float, default=None)
    parser.add_argument("--output-label", default="variant_rescue")
    parser.add_argument("--fallback-if-empty", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.rescue_variant:
        raise ValueError("at least one --rescue-variant is required")
    merged = combine_match_detail_rows(
        read_rows(args.baseline_details),
        read_rows(args.rescue_details),
        rescue_variants=set(args.rescue_variant),
        rescue_min_score=float(args.rescue_min_score),
        fallback_if_empty=bool(args.fallback_if_empty),
        output_label=args.output_label,
        rescue_max_baseline_matches=args.rescue_max_baseline_matches,
        rescue_min_pair_matches=int(args.rescue_min_pair_matches),
        rescue_max_pair_matches=args.rescue_max_pair_matches,
        rescue_min_score_mean=args.rescue_min_score_mean,
    )
    summary = summarize_rows(merged, label=args.output_label)
    variant_summary = summarize_by_variant(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "combined_match_details.csv", merged)
    write_summary(args.output_dir / "summary.csv", summary)
    write_variant_summary(args.output_dir / "variant_summary.csv", variant_summary)
    write_html(
        args.output_dir / "index.html",
        payload={
            "baseline_details": str(args.baseline_details),
            "rescue_details": str(args.rescue_details),
            "rescue_variants": list(args.rescue_variant),
            "rescue_min_score": float(args.rescue_min_score),
            "rescue_max_baseline_matches": args.rescue_max_baseline_matches,
            "rescue_min_pair_matches": int(args.rescue_min_pair_matches),
            "rescue_max_pair_matches": args.rescue_max_pair_matches,
            "rescue_min_score_mean": args.rescue_min_score_mean,
            "fallback_if_empty": bool(args.fallback_if_empty),
            "summary": summary.__dict__,
            "variant_summary": variant_summary,
        },
    )
    print(
        f"combined matches={summary.matches} correct={summary.correct} "
        f"wrong={summary.wrong} precision={summary.precision:.6f} output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
