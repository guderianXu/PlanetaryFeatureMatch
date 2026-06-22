#!/usr/bin/env python3
"""Build train-only replay manifests from PFM failure-bucket patterns."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mine_hard_failure_pairs import PAIR_MANIFEST_FIELDS


FAILURE_BUCKET_EXTRA_FIELDS = [
    "failure_bucket_reasons",
    "failure_bucket_score",
    "failure_bucket_sources",
    "failure_bucket_pair_type",
    "failure_bucket_reference_variant",
    "failure_bucket_target_variant",
    "failure_bucket_source_rows",
    "failure_bucket_wrong_sum",
    "failure_bucket_near_miss_wrong_sum",
    "failure_bucket_false_cluster_wrong_sum",
    "failure_bucket_high_confidence_wrong_sum",
    "failure_bucket_far_wrong_sum",
    "failure_bucket_precision_min",
    "failure_bucket_matches_mean",
]

FALSE_CLUSTER_EXTRA_FIELDS = [
    "false_cluster_reasons",
    "false_cluster_score",
    "false_cluster_sources",
    "false_cluster_pair_type",
    "false_cluster_reference_variant",
    "false_cluster_target_variant",
    "false_cluster_source_rows",
    "false_cluster_wrong_sum",
    "false_cluster_teacher_wrong_delta_sum",
    "false_cluster_precision_min",
    "false_cluster_feature_matches_mean",
]

FORBIDDEN_SOURCE_TOKENS = ("heldout", "fresh_heldout")


@dataclass(frozen=True)
class FailureBucketReplayConfig:
    min_wrong: int = 1
    min_bucket_wrong: int = 1
    default_pair_type: str = "same_position_view"
    forbid_source_tokens: tuple[str, ...] = FORBIDDEN_SOURCE_TOKENS


@dataclass(frozen=True)
class FailureBucketPattern:
    pair_type: str
    reference_variant: str
    target_variant: str
    reasons: tuple[str, ...]
    score: float
    sources: tuple[str, ...]
    source_rows: int
    wrong_sum: int
    near_miss_wrong_sum: int
    false_cluster_wrong_sum: int
    high_confidence_wrong_sum: int
    far_wrong_sum: int
    precision_min: float
    matches_mean: float

    @property
    def key(self) -> tuple[str, str, str]:
        return self.pair_type, self.reference_variant, self.target_variant

    @property
    def has_false_cluster(self) -> bool:
        return self.false_cluster_wrong_sum > 0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


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


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _parse_bucket_counts(value: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in value.split(";"):
        stripped = item.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            counts[stripped] += 1
            continue
        name, raw_count = stripped.split(":", 1)
        try:
            counts[name] += int(round(float(raw_count)))
        except ValueError:
            counts[name] += 1
    return counts


def _assert_allowed_source(source_name: str, config: FailureBucketReplayConfig) -> None:
    source_text = source_name.lower()
    for token in config.forbid_source_tokens:
        if token and token.lower() in source_text:
            raise ValueError(
                f"refusing to build training replay from forbidden source token '{token}'; "
                "fresh held-out diagnostics must not be used for training"
            )


def _source_id(source_name: str, row: dict[str, str]) -> str:
    split = row.get("split", "")
    base_id = row.get("base_id", "")
    target_variant = row.get("target_variant", "")
    pair_index = row.get("pair_index", "")
    return ":".join(item for item in (source_name, split, base_id, target_variant, pair_index) if item)


def _row_score(row: dict[str, str], bucket_counts: Counter[str]) -> float:
    wrong = max(0, _int_value(row, "wrong"))
    precision = _float_value(row, "precision", 1.0)
    false_cluster = bucket_counts.get("false_cluster", 0) + bucket_counts.get("false_cluster_high_confidence", 0)
    high_confidence = _int_value(row, "high_confidence_wrong") + bucket_counts.get("false_cluster_high_confidence", 0)
    near_miss = _int_value(row, "near_miss_wrong") + bucket_counts.get("near_miss", 0)
    far_wrong = _int_value(row, "far_wrong")
    return (
        false_cluster * 25.0
        + high_confidence * 10.0
        + far_wrong * 12.0
        + near_miss * 2.0
        + wrong
        + max(0.0, 1.0 - precision) * 100.0
    )


def collect_failure_bucket_patterns(
    rows: Sequence[dict[str, str]],
    *,
    source_name: str,
    config: FailureBucketReplayConfig,
) -> list[FailureBucketPattern]:
    _assert_allowed_source(source_name, config)
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        wrong = _int_value(row, "wrong")
        if wrong < config.min_wrong:
            continue
        bucket_counts = _parse_bucket_counts(row.get("primary_buckets", ""))
        false_cluster_count = bucket_counts.get("false_cluster", 0) + bucket_counts.get("false_cluster_high_confidence", 0)
        near_miss_count = _int_value(row, "near_miss_wrong") or bucket_counts.get("near_miss", 0)
        far_wrong_count = _int_value(row, "far_wrong")
        high_confidence_count = _int_value(row, "high_confidence_wrong") + bucket_counts.get(
            "false_cluster_high_confidence",
            0,
        )
        if max(false_cluster_count, near_miss_count, far_wrong_count, high_confidence_count) < config.min_bucket_wrong:
            continue
        reference_variant = row.get("reference_variant", "")
        target_variant = row.get("target_variant", "")
        if not reference_variant or not target_variant:
            continue
        pair_type = row.get("pair_type", "") or config.default_pair_type
        key = (pair_type, reference_variant, target_variant)
        item = grouped.setdefault(
            key,
            {
                "reasons": [],
                "score": 0.0,
                "sources": [],
                "source_rows": 0,
                "wrong_sum": 0,
                "near_miss_wrong_sum": 0,
                "false_cluster_wrong_sum": 0,
                "high_confidence_wrong_sum": 0,
                "far_wrong_sum": 0,
                "precision_min": 1.0,
                "matches_sum": 0.0,
            },
        )
        for bucket, count in bucket_counts.items():
            if count > 0:
                _append_unique(item["reasons"], bucket)  # type: ignore[arg-type]
        if false_cluster_count > 0:
            _append_unique(item["reasons"], "false_cluster_replay")  # type: ignore[arg-type]
        if near_miss_count > 0:
            _append_unique(item["reasons"], "near_miss_boundary")  # type: ignore[arg-type]
        item["score"] = float(item["score"]) + _row_score(row, bucket_counts)
        item["sources"].append(_source_id(source_name, row))  # type: ignore[union-attr]
        item["source_rows"] = int(item["source_rows"]) + 1
        item["wrong_sum"] = int(item["wrong_sum"]) + wrong
        item["near_miss_wrong_sum"] = int(item["near_miss_wrong_sum"]) + near_miss_count
        item["false_cluster_wrong_sum"] = int(item["false_cluster_wrong_sum"]) + false_cluster_count
        item["high_confidence_wrong_sum"] = int(item["high_confidence_wrong_sum"]) + high_confidence_count
        item["far_wrong_sum"] = int(item["far_wrong_sum"]) + far_wrong_count
        item["precision_min"] = min(float(item["precision_min"]), _float_value(row, "precision", 1.0))
        item["matches_sum"] = float(item["matches_sum"]) + _float_value(row, "matches")

    patterns: list[FailureBucketPattern] = []
    for (pair_type, reference_variant, target_variant), item in grouped.items():
        source_rows = int(item["source_rows"])
        matches_mean = float(item["matches_sum"]) / source_rows if source_rows > 0 else 0.0
        patterns.append(
            FailureBucketPattern(
                pair_type=pair_type,
                reference_variant=reference_variant,
                target_variant=target_variant,
                reasons=tuple(item["reasons"]),  # type: ignore[arg-type]
                score=float(item["score"]),
                sources=tuple(item["sources"]),  # type: ignore[arg-type]
                source_rows=source_rows,
                wrong_sum=int(item["wrong_sum"]),
                near_miss_wrong_sum=int(item["near_miss_wrong_sum"]),
                false_cluster_wrong_sum=int(item["false_cluster_wrong_sum"]),
                high_confidence_wrong_sum=int(item["high_confidence_wrong_sum"]),
                far_wrong_sum=int(item["far_wrong_sum"]),
                precision_min=float(item["precision_min"]),
                matches_mean=matches_mean,
            )
        )
    return sorted(patterns, key=lambda item: (-item.score, item.pair_type, item.reference_variant, item.target_variant))


def merge_failure_bucket_patterns(patterns: Sequence[FailureBucketPattern]) -> list[FailureBucketPattern]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for pattern in patterns:
        item = grouped.setdefault(
            pattern.key,
            {
                "reasons": [],
                "score": 0.0,
                "sources": [],
                "source_rows": 0,
                "wrong_sum": 0,
                "near_miss_wrong_sum": 0,
                "false_cluster_wrong_sum": 0,
                "high_confidence_wrong_sum": 0,
                "far_wrong_sum": 0,
                "precision_min": 1.0,
                "matches_weighted_sum": 0.0,
            },
        )
        for reason in pattern.reasons:
            _append_unique(item["reasons"], reason)  # type: ignore[arg-type]
        item["score"] = float(item["score"]) + pattern.score
        item["sources"].extend(pattern.sources)  # type: ignore[union-attr]
        item["source_rows"] = int(item["source_rows"]) + pattern.source_rows
        item["wrong_sum"] = int(item["wrong_sum"]) + pattern.wrong_sum
        item["near_miss_wrong_sum"] = int(item["near_miss_wrong_sum"]) + pattern.near_miss_wrong_sum
        item["false_cluster_wrong_sum"] = int(item["false_cluster_wrong_sum"]) + pattern.false_cluster_wrong_sum
        item["high_confidence_wrong_sum"] = int(item["high_confidence_wrong_sum"]) + pattern.high_confidence_wrong_sum
        item["far_wrong_sum"] = int(item["far_wrong_sum"]) + pattern.far_wrong_sum
        item["precision_min"] = min(float(item["precision_min"]), pattern.precision_min)
        item["matches_weighted_sum"] = float(item["matches_weighted_sum"]) + pattern.matches_mean * pattern.source_rows

    merged: list[FailureBucketPattern] = []
    for (pair_type, reference_variant, target_variant), item in grouped.items():
        source_rows = int(item["source_rows"])
        merged.append(
            FailureBucketPattern(
                pair_type=pair_type,
                reference_variant=reference_variant,
                target_variant=target_variant,
                reasons=tuple(item["reasons"]),  # type: ignore[arg-type]
                score=float(item["score"]),
                sources=tuple(item["sources"]),  # type: ignore[arg-type]
                source_rows=source_rows,
                wrong_sum=int(item["wrong_sum"]),
                near_miss_wrong_sum=int(item["near_miss_wrong_sum"]),
                false_cluster_wrong_sum=int(item["false_cluster_wrong_sum"]),
                high_confidence_wrong_sum=int(item["high_confidence_wrong_sum"]),
                far_wrong_sum=int(item["far_wrong_sum"]),
                precision_min=float(item["precision_min"]),
                matches_mean=float(item["matches_weighted_sum"]) / source_rows if source_rows > 0 else 0.0,
            )
        )
    return sorted(merged, key=lambda item: (-item.score, item.pair_type, item.reference_variant, item.target_variant))


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


def _reindexed_row(row: dict[str, str], index: int) -> dict[str, str]:
    copied = dict(row)
    copied["pair_index"] = str(index)
    return copied


def _failure_bucket_values(pattern: FailureBucketPattern) -> dict[str, str]:
    return {
        "failure_bucket_reasons": "|".join(pattern.reasons),
        "failure_bucket_score": f"{pattern.score:.6f}",
        "failure_bucket_sources": "|".join(pattern.sources[:16]),
        "failure_bucket_pair_type": pattern.pair_type,
        "failure_bucket_reference_variant": pattern.reference_variant,
        "failure_bucket_target_variant": pattern.target_variant,
        "failure_bucket_source_rows": str(pattern.source_rows),
        "failure_bucket_wrong_sum": str(pattern.wrong_sum),
        "failure_bucket_near_miss_wrong_sum": str(pattern.near_miss_wrong_sum),
        "failure_bucket_false_cluster_wrong_sum": str(pattern.false_cluster_wrong_sum),
        "failure_bucket_high_confidence_wrong_sum": str(pattern.high_confidence_wrong_sum),
        "failure_bucket_far_wrong_sum": str(pattern.far_wrong_sum),
        "failure_bucket_precision_min": f"{pattern.precision_min:.6f}",
        "failure_bucket_matches_mean": f"{pattern.matches_mean:.6f}",
    }


def _false_cluster_values(pattern: FailureBucketPattern) -> dict[str, str]:
    if not pattern.has_false_cluster:
        return {field: "" for field in FALSE_CLUSTER_EXTRA_FIELDS}
    return {
        "false_cluster_reasons": "|".join(pattern.reasons),
        "false_cluster_score": f"{pattern.score:.6f}",
        "false_cluster_sources": "|".join(pattern.sources[:16]),
        "false_cluster_pair_type": pattern.pair_type,
        "false_cluster_reference_variant": pattern.reference_variant,
        "false_cluster_target_variant": pattern.target_variant,
        "false_cluster_source_rows": str(pattern.source_rows),
        "false_cluster_wrong_sum": str(pattern.false_cluster_wrong_sum),
        "false_cluster_teacher_wrong_delta_sum": "0",
        "false_cluster_precision_min": f"{pattern.precision_min:.6f}",
        "false_cluster_feature_matches_mean": f"{pattern.matches_mean:.6f}",
    }


def sample_train_rows_by_failure_bucket_patterns(
    train_rows: Sequence[dict[str, str]],
    patterns: Sequence[FailureBucketPattern],
    *,
    max_per_pattern: int,
    seed: int,
) -> list[dict[str, str]]:
    if max_per_pattern <= 0:
        raise ValueError("max_per_pattern must be positive")
    rows_by_pattern: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        if row.get("split", "") != "train":
            continue
        key = (row.get("pair_type", ""), row.get("reference_variant", ""), row.get("target_variant", ""))
        rows_by_pattern[key].append(row)

    selected: dict[tuple[str, ...], dict[str, str]] = {}
    for pattern in patterns:
        candidates = [dict(row) for row in rows_by_pattern.get(pattern.key, [])]
        rng = random.Random(f"{seed}:{pattern.pair_type}:{pattern.reference_variant}:{pattern.target_variant}")
        rng.shuffle(candidates)
        for row in candidates[:max_per_pattern]:
            copied = dict(row)
            copied.update(_failure_bucket_values(pattern))
            copied.update(_false_cluster_values(pattern))
            identity = _pair_identity(copied)
            previous = selected.get(identity)
            if previous is None or float(copied["failure_bucket_score"]) > float(
                previous.get("failure_bucket_score") or 0.0
            ):
                selected[identity] = copied

    rows = list(selected.values())
    rows.sort(
        key=lambda row: (
            -float(row.get("failure_bucket_score") or 0.0),
            row.get("reference_variant", ""),
            row.get("target_variant", ""),
            row.get("reference_base_id", ""),
        )
    )
    return [_reindexed_row(row, index) for index, row in enumerate(rows)]


def build_mixed_manifest_rows(
    base_rows: Sequence[dict[str, str]],
    replay_rows: Sequence[dict[str, str]],
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
    replay_pool = [dict(row) for _ in range(repeat) for row in replay_rows]
    replay_total = len(replay_pool)
    mixed: list[dict[str, str]] = []
    replay_index = 0
    for base_index, row in enumerate(base_rows, start=1):
        mixed.append(dict(row))
        target_replay_seen = math.floor(base_index * replay_total / base_count)
        while replay_index < target_replay_seen:
            mixed.append(dict(replay_pool[replay_index]))
            replay_index += 1
    while replay_index < replay_total:
        mixed.append(dict(replay_pool[replay_index]))
        replay_index += 1
    return [_reindexed_row(row, index) for index, row in enumerate(mixed)]


def has_failure_bucket_replay_fields(row: dict[str, str]) -> bool:
    return any(row.get(field, "") for field in FAILURE_BUCKET_EXTRA_FIELDS)


def _manifest_fieldnames(base_fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> list[str]:
    fieldnames: list[str] = []

    def append(field: str) -> None:
        if field and field not in fieldnames:
            fieldnames.append(field)

    for field in base_fieldnames:
        append(field)
    for field in PAIR_MANIFEST_FIELDS:
        append(field)
    for row in rows:
        for field in row.keys():
            append(field)
    for field in FAILURE_BUCKET_EXTRA_FIELDS:
        append(field)
    for field in FALSE_CLUSTER_EXTRA_FIELDS:
        append(field)
    return fieldnames


def _write_manifest(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
                "<title>Failure-bucket replay manifest</title>",
                "<h1>Failure-bucket replay manifest</h1>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def _source_spec(value: str) -> tuple[str, Path]:
    if "," in value:
        source_name, raw_path = value.split(",", 1)
        return source_name.strip(), Path(raw_path.strip())
    path = Path(value)
    return path.stem, path


def _summarize(patterns: Sequence[FailureBucketPattern], replay_rows: Sequence[dict[str, str]]) -> dict[str, object]:
    variant_counts: Counter[str] = Counter(row.get("target_variant", "") for row in replay_rows)
    reason_counts: Counter[str] = Counter()
    for pattern in patterns:
        for reason in pattern.reasons:
            reason_counts[reason] += pattern.source_rows
    return {
        "patterns": len(patterns),
        "replay_rows": len(replay_rows),
        "variant_counts": dict(sorted(variant_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "pattern_payload": [
            {
                "pair_type": pattern.pair_type,
                "reference_variant": pattern.reference_variant,
                "target_variant": pattern.target_variant,
                "reasons": list(pattern.reasons),
                "score": pattern.score,
                "source_rows": pattern.source_rows,
                "wrong_sum": pattern.wrong_sum,
                "near_miss_wrong_sum": pattern.near_miss_wrong_sum,
                "false_cluster_wrong_sum": pattern.false_cluster_wrong_sum,
                "high_confidence_wrong_sum": pattern.high_confidence_wrong_sum,
                "far_wrong_sum": pattern.far_wrong_sum,
                "precision_min": pattern.precision_min,
                "matches_mean": pattern.matches_mean,
                "sampled_rows": sum(
                    1
                    for row in replay_rows
                    if row.get("failure_bucket_pair_type") == pattern.pair_type
                    and row.get("failure_bucket_reference_variant") == pattern.reference_variant
                    and row.get("failure_bucket_target_variant") == pattern.target_variant
                ),
            }
            for pattern in patterns
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-failure-summary", action="append", required=True, help="source_name,path/to/pair_failure_summary.csv")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-base-manifest", type=Path)
    parser.add_argument("--mixed-output-manifest", type=Path)
    parser.add_argument("--mixed-replay-fraction", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report-html", type=Path, required=True)
    parser.add_argument("--max-per-pattern", type=int, default=64)
    parser.add_argument("--min-wrong", type=int, default=1)
    parser.add_argument("--min-bucket-wrong", type=int, default=1)
    parser.add_argument("--default-pair-type", default="same_position_view")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-forbidden-source-tokens", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_per_pattern <= 0:
        raise ValueError("--max-per-pattern must be positive")
    if args.mixed_replay_fraction < 0.0 or args.mixed_replay_fraction >= 1.0:
        raise ValueError("--mixed-replay-fraction must be in [0, 1)")
    if args.mixed_output_manifest is not None and args.mixed_base_manifest is None:
        raise ValueError("--mixed-output-manifest requires --mixed-base-manifest")
    config = FailureBucketReplayConfig(
        min_wrong=int(args.min_wrong),
        min_bucket_wrong=int(args.min_bucket_wrong),
        default_pair_type=str(args.default_pair_type),
        forbid_source_tokens=() if args.allow_forbidden_source_tokens else FORBIDDEN_SOURCE_TOKENS,
    )

    patterns: list[FailureBucketPattern] = []
    source_specs: list[tuple[str, Path]] = []
    for value in args.pair_failure_summary:
        source_name, path = _source_spec(value)
        _assert_allowed_source(f"{source_name} {path}", config)
        source_specs.append((source_name, path))
        patterns.extend(collect_failure_bucket_patterns(_read_csv_rows(path), source_name=source_name, config=config))
    patterns = merge_failure_bucket_patterns(patterns)

    train_rows = _read_csv_rows(args.train_manifest)
    train_fieldnames = _csv_fieldnames(args.train_manifest)
    replay_rows = sample_train_rows_by_failure_bucket_patterns(
        train_rows,
        patterns,
        max_per_pattern=int(args.max_per_pattern),
        seed=int(args.seed),
    )
    _write_manifest(args.output_manifest, replay_rows, _manifest_fieldnames(train_fieldnames, replay_rows))

    mixed_rows: list[dict[str, str]] = []
    if args.mixed_output_manifest is not None and args.mixed_base_manifest is not None:
        mixed_rows = build_mixed_manifest_rows(
            _read_csv_rows(args.mixed_base_manifest),
            replay_rows,
            target_replay_fraction=float(args.mixed_replay_fraction),
        )
        mixed_base_fieldnames = _csv_fieldnames(args.mixed_base_manifest)
        _write_manifest(args.mixed_output_manifest, mixed_rows, _manifest_fieldnames(mixed_base_fieldnames, mixed_rows))

    mixed_replay_rows = sum(1 for row in mixed_rows if has_failure_bucket_replay_fields(row))
    summary = {
        "pair_failure_summary": [{"source_name": name, "path": str(path)} for name, path in source_specs],
        "train_manifest": str(args.train_manifest),
        "output_manifest": str(args.output_manifest),
        "mixed_base_manifest": str(args.mixed_base_manifest or ""),
        "mixed_output_manifest": str(args.mixed_output_manifest or ""),
        "mixed_replay_fraction": float(args.mixed_replay_fraction),
        "mixed_rows": len(mixed_rows),
        "mixed_replay_rows": mixed_replay_rows,
        "mixed_replay_fraction_actual": mixed_replay_rows / len(mixed_rows) if mixed_rows else 0.0,
        "max_per_pattern": int(args.max_per_pattern),
        "seed": int(args.seed),
        **_summarize(patterns, replay_rows),
    }
    _write_json(args.summary_json, summary)
    _write_html(args.report_html, summary)
    mixed_suffix = f" mixed_manifest={args.mixed_output_manifest}" if args.mixed_output_manifest else ""
    print(
        f"patterns={len(patterns)} replay_rows={len(replay_rows)} manifest={args.output_manifest}{mixed_suffix}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
