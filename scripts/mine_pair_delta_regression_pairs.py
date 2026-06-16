#!/usr/bin/env python3
"""Mine hard pair manifests from baseline-vs-candidate pair delta reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mine_hard_failure_pairs import PAIR_MANIFEST_FIELDS


DELTA_EXTRA_FIELDS = [
    "hard_reasons",
    "hard_score",
    "source_pair_delta",
    "source_baseline_matches",
    "source_baseline_correct",
    "source_baseline_wrong",
    "source_baseline_precision",
    "source_candidate_matches",
    "source_candidate_correct",
    "source_candidate_wrong",
    "source_candidate_precision",
    "source_match_delta",
    "source_correct_delta",
    "source_wrong_delta",
    "source_precision_delta",
]


@dataclass(frozen=True)
class PairDeltaMiningConfig:
    min_precision_drop: float = 0.01
    min_correct_drop: int = 0
    min_wrong_increase: int = 0
    min_match_drop: int = 0


@dataclass(frozen=True)
class PairDeltaSource:
    pair_manifest: Path
    pair_delta_csv: Path


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


def _pair_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("split", ""),
        row.get("reference_base_id", "") or row.get("base_id", ""),
        row.get("reference_variant", ""),
        row.get("target_variant", ""),
    )


def _delta_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("split", ""),
        row.get("base_id", "") or row.get("reference_base_id", ""),
        row.get("reference_variant", ""),
        row.get("target_variant", ""),
    )


def _pair_identity(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("split", ""),
        row.get("pair_type", ""),
        row.get("reference_pose_id", ""),
        row.get("target_pose_id", ""),
        row.get("crop_a_x0", ""),
        row.get("crop_a_y0", ""),
        row.get("crop_a_x1", ""),
        row.get("crop_a_y1", ""),
        row.get("crop_b_x0", ""),
        row.get("crop_b_y0", ""),
        row.get("crop_b_x1", ""),
        row.get("crop_b_y1", ""),
    )


def classify_pair_delta_row(row: dict[str, str], *, config: PairDeltaMiningConfig) -> tuple[list[str], float]:
    correct_delta = _int_value(row, "correct_delta")
    wrong_delta = _int_value(row, "wrong_delta")
    match_delta = _int_value(row, "match_delta")
    precision_delta = _float_value(row, "precision_delta")
    baseline_matches = _int_value(row, "baseline_matches")
    candidate_wrong = _int_value(row, "candidate_wrong")

    reasons: list[str] = []
    if correct_delta < -config.min_correct_drop:
        reasons.append("correct_regression")
    if wrong_delta > config.min_wrong_increase:
        reasons.append("wrong_increase")
    if precision_delta < -config.min_precision_drop:
        reasons.append("precision_regression")
    if config.min_match_drop > 0 and match_delta < -config.min_match_drop:
        reasons.append("match_count_drop")
    if baseline_matches <= 0 and candidate_wrong > 0:
        reasons.append("candidate_wrong_from_zero")

    hard_score = 0.0
    if "correct_regression" in reasons:
        hard_score += float(max(0, -correct_delta)) * 10.0
    if "wrong_increase" in reasons:
        hard_score += float(max(0, wrong_delta)) * 5.0
    if "precision_regression" in reasons:
        hard_score += max(0.0, -precision_delta) * 100.0
    if "match_count_drop" in reasons:
        hard_score += float(max(0, -match_delta))
    if "candidate_wrong_from_zero" in reasons:
        hard_score += 20.0 + float(candidate_wrong)
    return reasons, hard_score


def _delta_extra_values(row: dict[str, str], *, reasons: list[str], hard_score: float, source: str) -> dict[str, str]:
    return {
        "hard_reasons": "|".join(dict.fromkeys(reasons)),
        "hard_score": f"{hard_score:.6f}",
        "source_pair_delta": source,
        "source_baseline_matches": str(_int_value(row, "baseline_matches")),
        "source_baseline_correct": str(_int_value(row, "baseline_correct")),
        "source_baseline_wrong": str(_int_value(row, "baseline_wrong")),
        "source_baseline_precision": f"{_float_value(row, 'baseline_precision'):.6f}",
        "source_candidate_matches": str(_int_value(row, "candidate_matches")),
        "source_candidate_correct": str(_int_value(row, "candidate_correct")),
        "source_candidate_wrong": str(_int_value(row, "candidate_wrong")),
        "source_candidate_precision": f"{_float_value(row, 'candidate_precision'):.6f}",
        "source_match_delta": str(_int_value(row, "match_delta")),
        "source_correct_delta": str(_int_value(row, "correct_delta")),
        "source_wrong_delta": str(_int_value(row, "wrong_delta")),
        "source_precision_delta": f"{_float_value(row, 'precision_delta'):.6f}",
    }


def mine_pair_delta_regression_rows(
    pair_rows: list[dict[str, str]],
    delta_rows: list[dict[str, str]],
    *,
    config: PairDeltaMiningConfig,
    source_pair_delta: str = "",
) -> list[dict[str, str]]:
    pairs_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        pairs_by_key[_pair_key(row)].append(row)

    delta_key_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    mined: list[dict[str, str]] = []
    for delta in delta_rows:
        reasons, hard_score = classify_pair_delta_row(delta, config=config)
        if not reasons:
            continue
        key = _delta_key(delta)
        occurrence_index = delta_key_counts[key]
        delta_key_counts[key] += 1
        pairs = pairs_by_key.get(key, [])
        if occurrence_index >= len(pairs):
            continue
        pair = pairs[occurrence_index]
        row = {field: pair.get(field, "") for field in PAIR_MANIFEST_FIELDS}
        row.update(_delta_extra_values(delta, reasons=reasons, hard_score=hard_score, source=source_pair_delta))
        mined.append(row)
    mined.sort(key=lambda row: (-float(row.get("hard_score") or 0.0), row.get("reference_base_id", ""), row.get("target_variant", "")))
    for index, row in enumerate(mined):
        row["pair_index"] = str(index)
    return mined


def merge_and_deduplicate_hard_rows(rows_by_source: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    for rows in rows_by_source:
        for row in rows:
            key = _pair_identity(row)
            previous = selected.get(key)
            if previous is None or float(row.get("hard_score") or 0.0) > float(previous.get("hard_score") or 0.0):
                selected[key] = row
    merged = list(selected.values())
    merged.sort(key=lambda row: (-float(row.get("hard_score") or 0.0), row.get("reference_base_id", ""), row.get("target_variant", "")))
    for index, row in enumerate(merged):
        row["pair_index"] = str(index)
    return merged


def build_mixed_manifest_rows(
    base_rows: list[dict[str, str]],
    hard_rows: list[dict[str, str]],
    *,
    target_hard_fraction: float,
) -> list[dict[str, str]]:
    if target_hard_fraction <= 0.0 or not hard_rows:
        return [_reindexed_row(row, index) for index, row in enumerate(base_rows)]
    if target_hard_fraction >= 1.0:
        raise ValueError("target_hard_fraction must be less than 1.0")
    base_count = len(base_rows)
    hard_count = len(hard_rows)
    repeat = max(1, math.ceil((target_hard_fraction * base_count) / (hard_count * (1.0 - target_hard_fraction))))
    mixed: list[dict[str, str]] = [dict(row) for row in base_rows]
    for _ in range(repeat):
        mixed.extend(dict(row) for row in hard_rows)
    return [_reindexed_row(row, index) for index, row in enumerate(mixed)]


def _reindexed_row(row: dict[str, str], index: int) -> dict[str, str]:
    copied = dict(row)
    copied["pair_index"] = str(index)
    return copied


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*PAIR_MANIFEST_FIELDS, *DELTA_EXTRA_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def filter_hard_rows_by_required_reasons(rows: list[dict[str, str]], required_reasons: list[str]) -> list[dict[str, str]]:
    required = {reason.strip() for reason in required_reasons if reason.strip()}
    if not required:
        return rows
    filtered: list[dict[str, str]] = []
    for row in rows:
        reasons = {reason for reason in row.get("hard_reasons", "").split("|") if reason}
        if required.issubset(reasons):
            filtered.append(row)
    return filtered


def expand_sources(pair_manifests: list[Path], pair_delta_csvs: list[Path]) -> list[PairDeltaSource]:
    if not pair_manifests:
        raise ValueError("at least one --pair-manifest is required")
    if not pair_delta_csvs:
        raise ValueError("at least one --pair-delta-csv is required")
    if len(pair_manifests) == 1:
        return [PairDeltaSource(pair_manifests[0], delta) for delta in pair_delta_csvs]
    if len(pair_manifests) != len(pair_delta_csvs):
        raise ValueError("--pair-manifest count must be 1 or match --pair-delta-csv count")
    return [PairDeltaSource(manifest, delta) for manifest, delta in zip(pair_manifests, pair_delta_csvs)]


def write_report(path: Path, *, args: argparse.Namespace, rows: list[dict[str, str]], sources: list[PairDeltaSource]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row.get("hard_reasons", "").split("|"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    payload = {
        "output_manifest": str(args.output_manifest),
        "mixed_output_manifest": str(args.mixed_output_manifest or ""),
        "mixed_hard_fraction": float(args.mixed_hard_fraction),
        "config": {
            "min_precision_drop": float(args.min_precision_drop),
            "min_correct_drop": int(args.min_correct_drop),
            "min_wrong_increase": int(args.min_wrong_increase),
            "min_match_drop": int(args.min_match_drop),
        },
        "required_reason": list(args.required_reason),
        "rows": len(rows),
        "reason_counts": counts,
        "sources": [
            {"pair_manifest": str(source.pair_manifest), "pair_delta_csv": str(source.pair_delta_csv)}
            for source in sources
        ],
    }
    document = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Pair delta regression mining</title></head>
<body>
<h1>Pair delta regression mining</h1>
<pre>{html.escape(json.dumps(payload, indent=2, ensure_ascii=False))}</pre>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, action="append", required=True)
    parser.add_argument("--pair-delta-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-base-manifest", type=Path, default=None)
    parser.add_argument("--mixed-output-manifest", type=Path, default=None)
    parser.add_argument("--mixed-hard-fraction", type=float, default=0.0)
    parser.add_argument("--report-html", type=Path, default=None)
    parser.add_argument("--required-reason", action="append", default=[])
    parser.add_argument("--min-precision-drop", type=float, default=0.01)
    parser.add_argument("--min-correct-drop", type=int, default=0)
    parser.add_argument("--min-wrong-increase", type=int, default=0)
    parser.add_argument("--min-match-drop", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_precision_drop < 0.0:
        raise ValueError("--min-precision-drop must be nonnegative")
    if args.min_correct_drop < 0:
        raise ValueError("--min-correct-drop must be nonnegative")
    if args.min_wrong_increase < 0:
        raise ValueError("--min-wrong-increase must be nonnegative")
    if args.min_match_drop < 0:
        raise ValueError("--min-match-drop must be nonnegative")
    if args.mixed_hard_fraction < 0.0 or args.mixed_hard_fraction >= 1.0:
        raise ValueError("--mixed-hard-fraction must be in [0, 1)")
    if args.mixed_output_manifest is not None and args.mixed_base_manifest is None:
        raise ValueError("--mixed-output-manifest requires --mixed-base-manifest")
    config = PairDeltaMiningConfig(
        min_precision_drop=float(args.min_precision_drop),
        min_correct_drop=int(args.min_correct_drop),
        min_wrong_increase=int(args.min_wrong_increase),
        min_match_drop=int(args.min_match_drop),
    )
    sources = expand_sources(args.pair_manifest, args.pair_delta_csv)
    rows_by_source: list[list[dict[str, str]]] = []
    for source in sources:
        rows_by_source.append(
            mine_pair_delta_regression_rows(
                _read_csv_rows(source.pair_manifest),
                _read_csv_rows(source.pair_delta_csv),
                config=config,
                source_pair_delta=str(source.pair_delta_csv),
            )
        )
    hard_rows = filter_hard_rows_by_required_reasons(
        merge_and_deduplicate_hard_rows(rows_by_source),
        args.required_reason,
    )
    for index, row in enumerate(hard_rows):
        row["pair_index"] = str(index)
    write_manifest(args.output_manifest, hard_rows)
    if args.mixed_output_manifest is not None and args.mixed_base_manifest is not None:
        mixed_rows = build_mixed_manifest_rows(
            _read_csv_rows(args.mixed_base_manifest),
            hard_rows,
            target_hard_fraction=float(args.mixed_hard_fraction),
        )
        write_manifest(args.mixed_output_manifest, mixed_rows)
    if args.report_html is not None:
        write_report(args.report_html, args=args, rows=hard_rows, sources=sources)
    mixed_suffix = f" mixed_manifest={args.mixed_output_manifest}" if args.mixed_output_manifest is not None else ""
    print(f"pair_delta_hard={len(hard_rows)} manifest={args.output_manifest}{mixed_suffix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
