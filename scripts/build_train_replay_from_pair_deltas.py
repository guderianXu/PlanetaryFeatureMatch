#!/usr/bin/env python3
"""Build train replay manifests from validation/test pair-delta patterns."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mine_hard_failure_pairs import PAIR_MANIFEST_FIELDS


DEFAULT_EXTREME_VARIANTS = ("extreme_02", "extreme_03")

PATTERN_REPLAY_EXTRA_FIELDS = [
    "pattern_reasons",
    "pattern_score",
    "pattern_sources",
    "pattern_reference_variant",
    "pattern_target_variant",
    "pattern_match_delta_sum",
    "pattern_correct_delta_sum",
    "pattern_wrong_delta_sum",
    "pattern_precision_delta_sum",
]


@dataclass(frozen=True)
class PatternReplayConfig:
    min_precision_drop: float = 0.01
    min_correct_drop: int = 0
    min_wrong_increase: int = 0
    min_match_drop: int = 0
    min_gain_correct: int = 1
    max_gain_wrong_increase: int = 0
    extreme_variants: tuple[str, ...] = DEFAULT_EXTREME_VARIANTS


@dataclass(frozen=True)
class DeltaPattern:
    reference_variant: str
    target_variant: str
    reasons: tuple[str, ...]
    score: float
    sources: tuple[str, ...]
    match_delta_sum: int
    correct_delta_sum: int
    wrong_delta_sum: int
    precision_delta_sum: float

    @property
    def key(self) -> tuple[str, str]:
        return self.reference_variant, self.target_variant


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


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _regression_reasons(row: dict[str, str], config: PatternReplayConfig) -> tuple[list[str], float]:
    match_delta = _int_value(row, "match_delta")
    correct_delta = _int_value(row, "correct_delta")
    wrong_delta = _int_value(row, "wrong_delta")
    precision_delta = _float_value(row, "precision_delta")
    reasons: list[str] = []
    score = 0.0
    if correct_delta < -config.min_correct_drop:
        _append_reason(reasons, "correct_regression")
        score += float(max(0, -correct_delta)) * 10.0
    if wrong_delta > config.min_wrong_increase:
        _append_reason(reasons, "wrong_increase")
        score += float(max(0, wrong_delta)) * 5.0
    if precision_delta < -config.min_precision_drop:
        _append_reason(reasons, "precision_regression")
        score += max(0.0, -precision_delta) * 100.0
    if config.min_match_drop > 0 and match_delta < -config.min_match_drop:
        _append_reason(reasons, "match_count_drop")
        score += float(max(0, -match_delta))
    if reasons:
        reasons.insert(0, "protect_regression")
    return reasons, score


def _gain_reasons(row: dict[str, str], config: PatternReplayConfig) -> tuple[list[str], float]:
    target_variant = row.get("target_variant", "")
    correct_delta = _int_value(row, "correct_delta")
    wrong_delta = _int_value(row, "wrong_delta")
    precision_delta = _float_value(row, "precision_delta")
    if target_variant not in set(config.extreme_variants):
        return [], 0.0
    if correct_delta < config.min_gain_correct:
        return [], 0.0
    if wrong_delta > config.max_gain_wrong_increase:
        return [], 0.0
    score = float(correct_delta) * 10.0 + max(0.0, precision_delta) * 100.0 + float(max(0, -wrong_delta)) * 5.0
    return ["extreme_gain"], score


def _source_name(row: dict[str, str]) -> str:
    split = row.get("split", "")
    base_id = row.get("base_id", "") or row.get("reference_base_id", "")
    return f"{split}:{base_id}" if split or base_id else "unknown"


def _collect_pattern(
    collected: dict[tuple[str, str], dict[str, object]],
    row: dict[str, str],
    *,
    reasons: list[str],
    score: float,
) -> None:
    if not reasons:
        return
    reference_variant = row.get("reference_variant", "")
    target_variant = row.get("target_variant", "")
    if not reference_variant or not target_variant:
        return
    key = (reference_variant, target_variant)
    item = collected.setdefault(
        key,
        {
            "reasons": [],
            "score": 0.0,
            "sources": [],
            "match_delta_sum": 0,
            "correct_delta_sum": 0,
            "wrong_delta_sum": 0,
            "precision_delta_sum": 0.0,
        },
    )
    for reason in reasons:
        _append_reason(item["reasons"], reason)  # type: ignore[arg-type]
    item["score"] = float(item["score"]) + score
    item["sources"].append(_source_name(row))  # type: ignore[union-attr]
    item["match_delta_sum"] = int(item["match_delta_sum"]) + _int_value(row, "match_delta")
    item["correct_delta_sum"] = int(item["correct_delta_sum"]) + _int_value(row, "correct_delta")
    item["wrong_delta_sum"] = int(item["wrong_delta_sum"]) + _int_value(row, "wrong_delta")
    item["precision_delta_sum"] = float(item["precision_delta_sum"]) + _float_value(row, "precision_delta")


def collect_delta_patterns(
    *,
    regression_delta_rows: list[dict[str, str]],
    gain_delta_rows: list[dict[str, str]],
    config: PatternReplayConfig,
) -> list[DeltaPattern]:
    collected: dict[tuple[str, str], dict[str, object]] = {}
    for row in regression_delta_rows:
        reasons, score = _regression_reasons(row, config)
        _collect_pattern(collected, row, reasons=reasons, score=score)
    for row in gain_delta_rows:
        reasons, score = _gain_reasons(row, config)
        _collect_pattern(collected, row, reasons=reasons, score=score)
    patterns: list[DeltaPattern] = []
    for (reference_variant, target_variant), item in collected.items():
        patterns.append(
            DeltaPattern(
                reference_variant=reference_variant,
                target_variant=target_variant,
                reasons=tuple(item["reasons"]),  # type: ignore[arg-type]
                score=float(item["score"]),
                sources=tuple(item["sources"]),  # type: ignore[arg-type]
                match_delta_sum=int(item["match_delta_sum"]),
                correct_delta_sum=int(item["correct_delta_sum"]),
                wrong_delta_sum=int(item["wrong_delta_sum"]),
                precision_delta_sum=float(item["precision_delta_sum"]),
            )
        )
    return sorted(patterns, key=lambda item: (-item.score, item.reference_variant, item.target_variant))


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


def _pattern_extra_values(pattern: DeltaPattern) -> dict[str, str]:
    return {
        "pattern_reasons": "|".join(pattern.reasons),
        "pattern_score": f"{pattern.score:.6f}",
        "pattern_sources": "|".join(pattern.sources[:16]),
        "pattern_reference_variant": pattern.reference_variant,
        "pattern_target_variant": pattern.target_variant,
        "pattern_match_delta_sum": str(pattern.match_delta_sum),
        "pattern_correct_delta_sum": str(pattern.correct_delta_sum),
        "pattern_wrong_delta_sum": str(pattern.wrong_delta_sum),
        "pattern_precision_delta_sum": f"{pattern.precision_delta_sum:.6f}",
    }


def _reindexed_row(row: dict[str, str], index: int) -> dict[str, str]:
    copied = dict(row)
    copied["pair_index"] = str(index)
    return copied


def sample_train_rows_by_patterns(
    train_rows: list[dict[str, str]],
    patterns: list[DeltaPattern],
    *,
    max_per_pattern: int,
    seed: int,
) -> list[dict[str, str]]:
    if max_per_pattern <= 0:
        raise ValueError("max_per_pattern must be positive")
    rows_by_pattern: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        if row.get("split") != "train":
            continue
        key = (row.get("reference_variant", ""), row.get("target_variant", ""))
        rows_by_pattern[key].append(row)
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    for pattern in patterns:
        candidates = [dict(row) for row in rows_by_pattern.get(pattern.key, [])]
        rng = random.Random(f"{seed}:{pattern.reference_variant}:{pattern.target_variant}")
        rng.shuffle(candidates)
        for row in candidates[:max_per_pattern]:
            copied = {field: row.get(field, "") for field in PAIR_MANIFEST_FIELDS}
            copied.update(_pattern_extra_values(pattern))
            key = _pair_identity(copied)
            previous = selected.get(key)
            if previous is None or float(copied["pattern_score"]) > float(previous.get("pattern_score") or 0.0):
                selected[key] = copied
    rows = list(selected.values())
    rows.sort(
        key=lambda row: (
            -float(row.get("pattern_score") or 0.0),
            row.get("reference_variant", ""),
            row.get("target_variant", ""),
            row.get("reference_base_id", ""),
        )
    )
    return [_reindexed_row(row, index) for index, row in enumerate(rows)]


def build_mixed_manifest_rows(
    base_rows: list[dict[str, str]],
    replay_rows: list[dict[str, str]],
    *,
    target_replay_fraction: float,
) -> list[dict[str, str]]:
    if target_replay_fraction <= 0.0 or not replay_rows:
        return [_reindexed_row(row, index) for index, row in enumerate(base_rows)]
    if target_replay_fraction >= 1.0:
        raise ValueError("target_replay_fraction must be less than 1.0")
    base_count = len(base_rows)
    replay_count = len(replay_rows)
    repeat = max(1, math.ceil((target_replay_fraction * base_count) / (replay_count * (1.0 - target_replay_fraction))))
    mixed = [dict(row) for row in base_rows]
    for _ in range(repeat):
        mixed.extend(dict(row) for row in replay_rows)
    return [_reindexed_row(row, index) for index, row in enumerate(mixed)]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*PAIR_MANIFEST_FIELDS, *PATTERN_REPLAY_EXTRA_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    output_manifest: Path,
    mixed_output_manifest: Path | None,
    patterns: list[DeltaPattern],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    pattern_payload = [
        {
            "reference_variant": pattern.reference_variant,
            "target_variant": pattern.target_variant,
            "reasons": list(pattern.reasons),
            "score": pattern.score,
            "sources": list(pattern.sources),
            "correct_delta_sum": pattern.correct_delta_sum,
            "wrong_delta_sum": pattern.wrong_delta_sum,
            "match_delta_sum": pattern.match_delta_sum,
            "precision_delta_sum": pattern.precision_delta_sum,
            "sampled_rows": sum(
                1
                for row in rows
                if row.get("pattern_reference_variant") == pattern.reference_variant
                and row.get("pattern_target_variant") == pattern.target_variant
            ),
        }
        for pattern in patterns
    ]
    payload = {
        "output_manifest": str(output_manifest),
        "mixed_output_manifest": str(mixed_output_manifest or ""),
        "train_manifest": str(args.train_manifest),
        "max_per_pattern": int(args.max_per_pattern),
        "mixed_replay_fraction": float(args.mixed_replay_fraction),
        "seed": int(args.seed),
        "sampled_rows": len(rows),
        "patterns": pattern_payload,
    }
    document = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Train replay from pair deltas</title></head>
<body>
<h1>Train replay from pair deltas</h1>
<pre>{html.escape(json.dumps(payload, indent=2, ensure_ascii=False))}</pre>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--regression-delta-csv", type=Path, action="append", default=[])
    parser.add_argument("--gain-delta-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-base-manifest", type=Path, default=None)
    parser.add_argument("--mixed-output-manifest", type=Path, default=None)
    parser.add_argument("--mixed-replay-fraction", type=float, default=0.0)
    parser.add_argument("--report-html", type=Path, default=None)
    parser.add_argument("--max-per-pattern", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-precision-drop", type=float, default=0.01)
    parser.add_argument("--min-correct-drop", type=int, default=0)
    parser.add_argument("--min-wrong-increase", type=int, default=0)
    parser.add_argument("--min-match-drop", type=int, default=0)
    parser.add_argument("--min-gain-correct", type=int, default=1)
    parser.add_argument("--max-gain-wrong-increase", type=int, default=0)
    parser.add_argument("--extreme-variant", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.regression_delta_csv and not args.gain_delta_csv:
        raise ValueError("at least one --regression-delta-csv or --gain-delta-csv is required")
    if args.max_per_pattern <= 0:
        raise ValueError("--max-per-pattern must be positive")
    if args.mixed_replay_fraction < 0.0 or args.mixed_replay_fraction >= 1.0:
        raise ValueError("--mixed-replay-fraction must be in [0, 1)")
    if args.mixed_output_manifest is not None and args.mixed_base_manifest is None:
        raise ValueError("--mixed-output-manifest requires --mixed-base-manifest")
    config = PatternReplayConfig(
        min_precision_drop=float(args.min_precision_drop),
        min_correct_drop=int(args.min_correct_drop),
        min_wrong_increase=int(args.min_wrong_increase),
        min_match_drop=int(args.min_match_drop),
        min_gain_correct=int(args.min_gain_correct),
        max_gain_wrong_increase=int(args.max_gain_wrong_increase),
        extreme_variants=tuple(args.extreme_variant) if args.extreme_variant else DEFAULT_EXTREME_VARIANTS,
    )
    regression_rows: list[dict[str, str]] = []
    for path in args.regression_delta_csv:
        regression_rows.extend(_read_csv_rows(path))
    gain_rows: list[dict[str, str]] = []
    for path in args.gain_delta_csv:
        gain_rows.extend(_read_csv_rows(path))
    patterns = collect_delta_patterns(
        regression_delta_rows=regression_rows,
        gain_delta_rows=gain_rows,
        config=config,
    )
    replay_rows = sample_train_rows_by_patterns(
        _read_csv_rows(args.train_manifest),
        patterns,
        max_per_pattern=int(args.max_per_pattern),
        seed=int(args.seed),
    )
    write_manifest(args.output_manifest, replay_rows)
    if args.mixed_output_manifest is not None and args.mixed_base_manifest is not None:
        mixed_rows = build_mixed_manifest_rows(
            _read_csv_rows(args.mixed_base_manifest),
            replay_rows,
            target_replay_fraction=float(args.mixed_replay_fraction),
        )
        write_manifest(args.mixed_output_manifest, mixed_rows)
    if args.report_html is not None:
        write_report(
            args.report_html,
            output_manifest=args.output_manifest,
            mixed_output_manifest=args.mixed_output_manifest,
            patterns=patterns,
            rows=replay_rows,
            args=args,
        )
    mixed_suffix = f" mixed_manifest={args.mixed_output_manifest}" if args.mixed_output_manifest else ""
    print(
        f"patterns={len(patterns)} replay_rows={len(replay_rows)} manifest={args.output_manifest}{mixed_suffix}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
