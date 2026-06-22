#!/usr/bin/env python3
"""Build train-only replay manifests from false-cluster rejection patterns."""

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
class FalseClusterReplayConfig:
    min_source_wrong: int = 2
    min_source_rows: int = 1
    min_teacher_wrong_delta: int = 0
    forbid_source_tokens: tuple[str, ...] = FORBIDDEN_SOURCE_TOKENS


@dataclass(frozen=True)
class FalseClusterPattern:
    pair_type: str
    reference_variant: str
    target_variant: str
    reasons: tuple[str, ...]
    score: float
    sources: tuple[str, ...]
    source_rows: int
    wrong_sum: int
    teacher_wrong_delta_sum: int
    precision_min: float
    feature_matches_mean: float

    @property
    def key(self) -> tuple[str, str, str]:
        return self.pair_type, self.reference_variant, self.target_variant


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


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _source_id(row: dict[str, str]) -> str:
    split = row.get("split", "")
    base_id = row.get("base_id", "") or row.get("reference_base_id", "")
    variant = row.get("target_variant", "")
    pair_index = row.get("pair_index", "")
    return ":".join(item for item in (split, base_id, variant, pair_index) if item)


def _assert_allowed_sources(rows: Sequence[dict[str, str]], config: FalseClusterReplayConfig) -> None:
    forbidden = tuple(token.lower() for token in config.forbid_source_tokens if token)
    if not forbidden:
        return
    for row in rows:
        source_text = " ".join(
            [
                row.get("source_name", ""),
                row.get("split", ""),
                row.get("base_id", ""),
            ]
        ).lower()
        for token in forbidden:
            if token in source_text:
                raise ValueError(
                    f"refusing to build training replay from forbidden source token '{token}'; "
                    "fresh held-out diagnostics must not be used for training"
                )


def _row_score(row: dict[str, str]) -> float:
    wrong = max(0, _int_value(row, "pfm_wrong"))
    teacher_wrong_delta = max(0, _int_value(row, "teacher_wrong_delta"))
    precision = _float_value(row, "pfm_precision", 1.0)
    teacher_precision_delta = _float_value(row, "teacher_precision_delta", 0.0)
    matches = max(0, _int_value(row, "feature_matches", _int_value(row, "pfm_matches")))
    return (
        wrong * 10.0
        + teacher_wrong_delta * 5.0
        + max(0.0, 1.0 - precision) * 100.0
        + max(0.0, -teacher_precision_delta) * 100.0
        + math.log1p(matches)
    )


def collect_false_cluster_patterns(
    rows: Sequence[dict[str, str]],
    *,
    config: FalseClusterReplayConfig,
) -> list[FalseClusterPattern]:
    _assert_allowed_sources(rows, config)
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        if row.get("reject_label", "") != "1":
            continue
        wrong = _int_value(row, "pfm_wrong")
        teacher_wrong_delta = _int_value(row, "teacher_wrong_delta")
        if wrong < config.min_source_wrong:
            continue
        if teacher_wrong_delta < config.min_teacher_wrong_delta:
            continue
        pair_type = row.get("pair_type", "")
        reference_variant = row.get("reference_variant", "")
        target_variant = row.get("target_variant", "")
        if not pair_type or not reference_variant or not target_variant:
            continue
        key = (pair_type, reference_variant, target_variant)
        item = grouped.setdefault(
            key,
            {
                "reasons": [],
                "score": 0.0,
                "sources": [],
                "source_rows": 0,
                "wrong_sum": 0,
                "teacher_wrong_delta_sum": 0,
                "precision_min": 1.0,
                "feature_matches_sum": 0.0,
            },
        )
        for reason in row.get("reject_reasons", "").split("|"):
            _append_unique(item["reasons"], reason)  # type: ignore[arg-type]
        item["score"] = float(item["score"]) + _row_score(row)
        item["sources"].append(_source_id(row))  # type: ignore[union-attr]
        item["source_rows"] = int(item["source_rows"]) + 1
        item["wrong_sum"] = int(item["wrong_sum"]) + wrong
        item["teacher_wrong_delta_sum"] = int(item["teacher_wrong_delta_sum"]) + teacher_wrong_delta
        item["precision_min"] = min(float(item["precision_min"]), _float_value(row, "pfm_precision", 1.0))
        item["feature_matches_sum"] = float(item["feature_matches_sum"]) + _float_value(
            row,
            "feature_matches",
            _float_value(row, "pfm_matches"),
        )

    patterns: list[FalseClusterPattern] = []
    for (pair_type, reference_variant, target_variant), item in grouped.items():
        source_rows = int(item["source_rows"])
        if source_rows < config.min_source_rows:
            continue
        feature_matches_mean = float(item["feature_matches_sum"]) / source_rows if source_rows else 0.0
        patterns.append(
            FalseClusterPattern(
                pair_type=pair_type,
                reference_variant=reference_variant,
                target_variant=target_variant,
                reasons=tuple(item["reasons"]),  # type: ignore[arg-type]
                score=float(item["score"]),
                sources=tuple(item["sources"]),  # type: ignore[arg-type]
                source_rows=source_rows,
                wrong_sum=int(item["wrong_sum"]),
                teacher_wrong_delta_sum=int(item["teacher_wrong_delta_sum"]),
                precision_min=float(item["precision_min"]),
                feature_matches_mean=feature_matches_mean,
            )
        )
    return sorted(
        patterns,
        key=lambda item: (-item.score, item.pair_type, item.reference_variant, item.target_variant),
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


def _pattern_extra_values(pattern: FalseClusterPattern) -> dict[str, str]:
    return {
        "false_cluster_reasons": "|".join(pattern.reasons),
        "false_cluster_score": f"{pattern.score:.6f}",
        "false_cluster_sources": "|".join(pattern.sources[:16]),
        "false_cluster_pair_type": pattern.pair_type,
        "false_cluster_reference_variant": pattern.reference_variant,
        "false_cluster_target_variant": pattern.target_variant,
        "false_cluster_source_rows": str(pattern.source_rows),
        "false_cluster_wrong_sum": str(pattern.wrong_sum),
        "false_cluster_teacher_wrong_delta_sum": str(pattern.teacher_wrong_delta_sum),
        "false_cluster_precision_min": f"{pattern.precision_min:.6f}",
        "false_cluster_feature_matches_mean": f"{pattern.feature_matches_mean:.6f}",
    }


def _reindexed_row(row: dict[str, str], index: int) -> dict[str, str]:
    copied = dict(row)
    copied["pair_index"] = str(index)
    return copied


def sample_train_rows_by_false_cluster_patterns(
    train_rows: Sequence[dict[str, str]],
    patterns: Sequence[FalseClusterPattern],
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
            copied = {field: row.get(field, "") for field in PAIR_MANIFEST_FIELDS}
            copied.update(_pattern_extra_values(pattern))
            identity = _pair_identity(copied)
            previous = selected.get(identity)
            if previous is None or float(copied["false_cluster_score"]) > float(previous.get("false_cluster_score") or 0.0):
                selected[identity] = copied

    rows = list(selected.values())
    rows.sort(
        key=lambda row: (
            -float(row.get("false_cluster_score") or 0.0),
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


def write_manifest(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*PAIR_MANIFEST_FIELDS, *FALSE_CLUSTER_EXTRA_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def has_false_cluster_replay_fields(row: dict[str, str]) -> bool:
    return any(row.get(field, "") for field in FALSE_CLUSTER_EXTRA_FIELDS)


def summarize_patterns(
    patterns: Sequence[FalseClusterPattern],
    replay_rows: Sequence[dict[str, str]],
) -> dict[str, object]:
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
                "teacher_wrong_delta_sum": pattern.teacher_wrong_delta_sum,
                "precision_min": pattern.precision_min,
                "feature_matches_mean": pattern.feature_matches_mean,
                "sampled_rows": sum(
                    1
                    for row in replay_rows
                    if row.get("false_cluster_pair_type") == pattern.pair_type
                    and row.get("false_cluster_reference_variant") == pattern.reference_variant
                    and row.get("false_cluster_target_variant") == pattern.target_variant
                ),
            }
            for pattern in patterns
        ],
    }


def write_summary_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report_html(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            "<title>False-cluster replay manifest</title>",
            "<h1>False-cluster replay manifest</h1>",
            f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
        ]
    )
    path.write_text(document, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejection-dataset-csv", type=Path, action="append", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-base-manifest", type=Path)
    parser.add_argument("--mixed-output-manifest", type=Path)
    parser.add_argument("--mixed-replay-fraction", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report-html", type=Path, required=True)
    parser.add_argument("--max-per-pattern", type=int, default=64)
    parser.add_argument("--min-source-wrong", type=int, default=2)
    parser.add_argument("--min-source-rows", type=int, default=1)
    parser.add_argument("--min-teacher-wrong-delta", type=int, default=0)
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

    config = FalseClusterReplayConfig(
        min_source_wrong=int(args.min_source_wrong),
        min_source_rows=int(args.min_source_rows),
        min_teacher_wrong_delta=int(args.min_teacher_wrong_delta),
        forbid_source_tokens=() if args.allow_forbidden_source_tokens else FORBIDDEN_SOURCE_TOKENS,
    )
    rejection_rows: list[dict[str, str]] = []
    for path in args.rejection_dataset_csv:
        rejection_rows.extend(_read_csv_rows(path))
    patterns = collect_false_cluster_patterns(rejection_rows, config=config)
    train_rows = _read_csv_rows(args.train_manifest)
    replay_rows = sample_train_rows_by_false_cluster_patterns(
        train_rows,
        patterns,
        max_per_pattern=int(args.max_per_pattern),
        seed=int(args.seed),
    )
    write_manifest(args.output_manifest, replay_rows)

    mixed_rows: list[dict[str, str]] = []
    if args.mixed_output_manifest is not None and args.mixed_base_manifest is not None:
        mixed_rows = build_mixed_manifest_rows(
            _read_csv_rows(args.mixed_base_manifest),
            replay_rows,
            target_replay_fraction=float(args.mixed_replay_fraction),
        )
        write_manifest(args.mixed_output_manifest, mixed_rows)

    mixed_replay_rows = sum(1 for row in mixed_rows if has_false_cluster_replay_fields(row))
    summary = {
        "rejection_dataset_csv": [str(path) for path in args.rejection_dataset_csv],
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
        **summarize_patterns(patterns, replay_rows),
    }
    write_summary_json(args.summary_json, summary)
    write_report_html(args.report_html, summary)
    mixed_suffix = f" mixed_manifest={args.mixed_output_manifest}" if args.mixed_output_manifest else ""
    print(
        f"patterns={len(patterns)} replay_rows={len(replay_rows)} manifest={args.output_manifest}{mixed_suffix}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
