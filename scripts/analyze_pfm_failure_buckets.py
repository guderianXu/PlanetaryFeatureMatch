#!/usr/bin/env python3
"""Bucket PFM wrong matches into actionable failure categories."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence


PAIR_FIELDS = [
    "label",
    "split",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
]

WRONG_MATCH_FIELDS = [
    *PAIR_FIELDS,
    "match_index",
    "point_a_x_px",
    "point_a_y_px",
    "point_b_x_px",
    "point_b_y_px",
    "dx_px",
    "dy_px",
    "score",
    "raw_margin",
    "accept_probability",
    "error_px",
    "valid_fraction",
    "primary_bucket",
    "buckets",
    "pair_matches",
    "pair_correct",
    "pair_wrong",
    "pair_precision",
    "pair_wrong_displacement_mad_px",
    "training_action",
]

PAIR_SUMMARY_FIELDS = [
    *PAIR_FIELDS,
    "matches",
    "correct",
    "wrong",
    "precision",
    "correct_dx_median_px",
    "correct_dy_median_px",
    "wrong_dx_median_px",
    "wrong_dy_median_px",
    "wrong_displacement_mad_px",
    "false_cluster",
    "high_confidence_wrong",
    "near_miss_wrong",
    "far_wrong",
    "primary_buckets",
    "training_action",
]

BUCKET_SUMMARY_FIELDS = [
    "bucket",
    "wrong_matches",
    "unique_pairs",
    "fraction_of_wrong",
    "mean_error_px",
    "median_error_px",
    "mean_score",
    "mean_accept_probability",
    "training_action",
]

ERROR_BIN_SUMMARY_FIELDS = [
    "error_bin",
    "wrong_matches",
    "fraction_of_wrong",
    "primary_buckets",
    "mean_score",
    "mean_accept_probability",
]

TRAINING_ACTIONS = {
    "false_cluster_high_confidence": "use false-cluster replay; add in-pair hard negatives; penalize confident wrong accept logits",
    "false_cluster": "mine clustered false matches; add pair-level displacement consistency and repeated-terrain negatives",
    "high_confidence_far_wrong": "mine far hard negatives; strengthen dustbin/no-match labels for repeated terrain",
    "high_confidence_isolated": "add isolated hard-negative sampling and confidence calibration loss",
    "near_miss": "audit geometry tolerance and near-threshold label noise; train with soft boundary weights",
    "low_confidence_scattered": "lower priority for training; prefer calibration or post-filter thresholds",
}


@dataclass(frozen=True)
class BucketConfig:
    cluster_wrong_min: int = 2
    false_cluster_mad_px: float = 5.0
    high_score_min: float = 12.0
    high_accept_min: float = 0.70
    high_raw_margin_min: float = 0.05
    near_miss_px: float = 8.0
    far_error_px: float = 20.0


@dataclass(frozen=True)
class PairStats:
    key: tuple[str, str, str, str, str, str]
    matches: int
    correct: int
    wrong: int
    precision: float
    correct_dx_median: float
    correct_dy_median: float
    wrong_dx_median: float
    wrong_dy_median: float
    wrong_displacement_mad: float
    false_cluster: bool


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(median(values)) if values else 0.0


def _pair_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return tuple(row.get(field, "") for field in PAIR_FIELDS)


def _dx(row: dict[str, str]) -> float:
    return _float_value(row, "point_b_x_px") - _float_value(row, "point_a_x_px")


def _dy(row: dict[str, str]) -> float:
    return _float_value(row, "point_b_y_px") - _float_value(row, "point_a_y_px")


def _wrong(row: dict[str, str]) -> bool:
    return _int_value(row, "correct") <= 0


def _pair_stats(key: tuple[str, str, str, str, str, str], rows: Sequence[dict[str, str]], config: BucketConfig) -> PairStats:
    correct_rows = [row for row in rows if not _wrong(row)]
    wrong_rows = [row for row in rows if _wrong(row)]
    correct_dx = [_dx(row) for row in correct_rows]
    correct_dy = [_dy(row) for row in correct_rows]
    wrong_dx = [_dx(row) for row in wrong_rows]
    wrong_dy = [_dy(row) for row in wrong_rows]
    wrong_dx_median = _median(wrong_dx)
    wrong_dy_median = _median(wrong_dy)
    wrong_distances = [
        math.hypot(_dx(row) - wrong_dx_median, _dy(row) - wrong_dy_median)
        for row in wrong_rows
    ]
    wrong_displacement_mad = _median(wrong_distances)
    matches = len(rows)
    correct = len(correct_rows)
    wrong = len(wrong_rows)
    return PairStats(
        key=key,
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=correct / matches if matches > 0 else 0.0,
        correct_dx_median=_median(correct_dx),
        correct_dy_median=_median(correct_dy),
        wrong_dx_median=wrong_dx_median,
        wrong_dy_median=wrong_dy_median,
        wrong_displacement_mad=wrong_displacement_mad,
        false_cluster=wrong >= config.cluster_wrong_min and wrong_displacement_mad <= config.false_cluster_mad_px,
    )


def _high_confidence(row: dict[str, str], config: BucketConfig) -> bool:
    return (
        _float_value(row, "score") >= config.high_score_min
        and _float_value(row, "accept_probability") >= config.high_accept_min
        and _float_value(row, "raw_margin") >= config.high_raw_margin_min
    )


def _wrong_buckets(row: dict[str, str], pair_stats: PairStats, config: BucketConfig) -> tuple[str, list[str]]:
    high_confidence = _high_confidence(row, config)
    near_miss = _float_value(row, "error_px") <= config.near_miss_px
    far_wrong = _float_value(row, "error_px") >= config.far_error_px
    buckets: list[str] = []
    if pair_stats.false_cluster:
        buckets.append("false_cluster")
    if high_confidence:
        buckets.append("high_confidence")
    if near_miss:
        buckets.append("near_miss")
    if far_wrong:
        buckets.append("far_wrong")

    if pair_stats.false_cluster and high_confidence:
        primary_bucket = "false_cluster_high_confidence"
    elif pair_stats.false_cluster:
        primary_bucket = "false_cluster"
    elif near_miss:
        primary_bucket = "near_miss"
    elif high_confidence and far_wrong:
        primary_bucket = "high_confidence_far_wrong"
    elif high_confidence:
        primary_bucket = "high_confidence_isolated"
    else:
        primary_bucket = "low_confidence_scattered"
    return primary_bucket, buckets


def build_failure_analysis(
    rows: Sequence[dict[str, str]],
    *,
    config: BucketConfig,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[_pair_key(row)].append(row)
    stats_by_key = {key: _pair_stats(key, group_rows, config) for key, group_rows in groups.items()}

    wrong_rows: list[dict[str, str]] = []
    pair_bucket_counts: dict[tuple[str, str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if not _wrong(row):
            continue
        key = _pair_key(row)
        pair_stats = stats_by_key[key]
        primary_bucket, buckets = _wrong_buckets(row, pair_stats, config)
        pair_bucket_counts[key][primary_bucket] += 1
        output_row = dict(row)
        output_row.update(
            {
                "dx_px": _fmt_float(_dx(row)),
                "dy_px": _fmt_float(_dy(row)),
                "primary_bucket": primary_bucket,
                "buckets": ";".join(buckets),
                "pair_matches": str(pair_stats.matches),
                "pair_correct": str(pair_stats.correct),
                "pair_wrong": str(pair_stats.wrong),
                "pair_precision": _fmt_float(pair_stats.precision),
                "pair_wrong_displacement_mad_px": _fmt_float(pair_stats.wrong_displacement_mad),
                "training_action": TRAINING_ACTIONS[primary_bucket],
            }
        )
        wrong_rows.append(output_row)

    pair_rows: list[dict[str, str]] = []
    for key, pair_stats in sorted(stats_by_key.items(), key=lambda item: item[0]):
        group_rows = groups[key]
        wrong_group = [row for row in group_rows if _wrong(row)]
        primary_buckets = pair_bucket_counts[key]
        row = {field: value for field, value in zip(PAIR_FIELDS, key)}
        top_bucket = primary_buckets.most_common(1)[0][0] if primary_buckets else "none"
        row.update(
            {
                "matches": str(pair_stats.matches),
                "correct": str(pair_stats.correct),
                "wrong": str(pair_stats.wrong),
                "precision": _fmt_float(pair_stats.precision),
                "correct_dx_median_px": _fmt_float(pair_stats.correct_dx_median),
                "correct_dy_median_px": _fmt_float(pair_stats.correct_dy_median),
                "wrong_dx_median_px": _fmt_float(pair_stats.wrong_dx_median),
                "wrong_dy_median_px": _fmt_float(pair_stats.wrong_dy_median),
                "wrong_displacement_mad_px": _fmt_float(pair_stats.wrong_displacement_mad),
                "false_cluster": "1" if pair_stats.false_cluster else "0",
                "high_confidence_wrong": str(sum(1 for item in wrong_group if _high_confidence(item, config))),
                "near_miss_wrong": str(sum(1 for item in wrong_group if _float_value(item, "error_px") <= config.near_miss_px)),
                "far_wrong": str(sum(1 for item in wrong_group if _float_value(item, "error_px") >= config.far_error_px)),
                "primary_buckets": ";".join(f"{bucket}:{count}" for bucket, count in primary_buckets.most_common()),
                "training_action": TRAINING_ACTIONS.get(top_bucket, ""),
            }
        )
        pair_rows.append(row)

    bucket_rows, bucket_counts = _summarize_buckets(wrong_rows)
    error_bin_rows, error_bin_counts = _summarize_error_bins(wrong_rows)
    correct = sum(1 for row in rows if not _wrong(row))
    wrong = len(rows) - correct
    summary = {
        "matches": len(rows),
        "correct": correct,
        "wrong": wrong,
        "precision": correct / len(rows) if rows else 0.0,
        "pairs": len(groups),
        "pairs_with_wrong": sum(1 for pair_stats in stats_by_key.values() if pair_stats.wrong > 0),
        "false_cluster_pairs": sum(1 for pair_stats in stats_by_key.values() if pair_stats.false_cluster),
        "bucket_counts": dict(bucket_counts),
        "error_bin_counts": dict(error_bin_counts),
        "config": {
            "cluster_wrong_min": config.cluster_wrong_min,
            "false_cluster_mad_px": config.false_cluster_mad_px,
            "high_score_min": config.high_score_min,
            "high_accept_min": config.high_accept_min,
            "high_raw_margin_min": config.high_raw_margin_min,
            "near_miss_px": config.near_miss_px,
            "far_error_px": config.far_error_px,
        },
    }
    return wrong_rows, pair_rows, bucket_rows, error_bin_rows, summary


def _summarize_buckets(wrong_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], Counter[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in wrong_rows:
        grouped[row["primary_bucket"]].append(row)
    bucket_counts = Counter({bucket: len(rows) for bucket, rows in grouped.items()})
    total_wrong = len(wrong_rows)
    bucket_rows: list[dict[str, str]] = []
    for bucket, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        errors = [_float_value(row, "error_px") for row in rows]
        scores = [_float_value(row, "score") for row in rows]
        accepts = [_float_value(row, "accept_probability") for row in rows]
        pairs = {_pair_key(row) for row in rows}
        bucket_rows.append(
            {
                "bucket": bucket,
                "wrong_matches": str(len(rows)),
                "unique_pairs": str(len(pairs)),
                "fraction_of_wrong": _fmt_float(len(rows) / total_wrong if total_wrong > 0 else 0.0),
                "mean_error_px": _fmt_float(_mean(errors)),
                "median_error_px": _fmt_float(_median(errors)),
                "mean_score": _fmt_float(_mean(scores)),
                "mean_accept_probability": _fmt_float(_mean(accepts)),
                "training_action": TRAINING_ACTIONS[bucket],
            }
        )
    return bucket_rows, bucket_counts


def _error_bin(error_px: float) -> str:
    if error_px <= 5.0:
        return "<=5"
    if error_px <= 6.0:
        return "5-6"
    if error_px <= 8.0:
        return "6-8"
    if error_px <= 12.0:
        return "8-12"
    if error_px <= 20.0:
        return "12-20"
    return ">20"


def _summarize_error_bins(wrong_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], Counter[str]]:
    order = ["<=5", "5-6", "6-8", "8-12", "12-20", ">20"]
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in order}
    for row in wrong_rows:
        grouped[_error_bin(_float_value(row, "error_px"))].append(row)
    total_wrong = len(wrong_rows)
    bin_counts = Counter({name: len(rows) for name, rows in grouped.items() if rows})
    output_rows: list[dict[str, str]] = []
    for name in order:
        rows = grouped[name]
        if not rows:
            continue
        primary_buckets = Counter(row["primary_bucket"] for row in rows)
        output_rows.append(
            {
                "error_bin": name,
                "wrong_matches": str(len(rows)),
                "fraction_of_wrong": _fmt_float(len(rows) / total_wrong if total_wrong > 0 else 0.0),
                "primary_buckets": ";".join(f"{bucket}:{count}" for bucket, count in primary_buckets.most_common()),
                "mean_score": _fmt_float(_mean([_float_value(row, "score") for row in rows])),
                "mean_accept_probability": _fmt_float(_mean([_float_value(row, "accept_probability") for row in rows])),
            }
        )
    return output_rows, bin_counts


def _summary_table(rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> str:
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fieldnames)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fieldnames) + "</tr>"
        )
    return (
        '<table border="1" cellspacing="0" cellpadding="4">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _write_html(
    path: Path,
    *,
    match_details: Path,
    summary: dict[str, Any],
    bucket_rows: Sequence[dict[str, str]],
    error_bin_rows: Sequence[dict[str, str]],
    pair_rows: Sequence[dict[str, str]],
) -> None:
    top_pairs = sorted(pair_rows, key=lambda row: int(row.get("wrong", "0")), reverse=True)[:20]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>PFM failure bucket analysis</title>",
                "<h1>PFM failure bucket analysis</h1>",
                f"<p>match_details={html.escape(str(match_details))}</p>",
                "<h2>Overall</h2>",
                _summary_table(
                    [
                        {
                            "matches": str(summary["matches"]),
                            "correct": str(summary["correct"]),
                            "wrong": str(summary["wrong"]),
                            "precision": _fmt_float(float(summary["precision"])),
                            "pairs": str(summary["pairs"]),
                            "pairs_with_wrong": str(summary["pairs_with_wrong"]),
                            "false_cluster_pairs": str(summary["false_cluster_pairs"]),
                        }
                    ],
                    ["matches", "correct", "wrong", "precision", "pairs", "pairs_with_wrong", "false_cluster_pairs"],
                ),
                "<h2>Bucket summary</h2>",
                _summary_table(bucket_rows, BUCKET_SUMMARY_FIELDS),
                "<h2>Error-bin summary</h2>",
                _summary_table(error_bin_rows, ERROR_BIN_SUMMARY_FIELDS),
                "<h2>Top wrong pairs</h2>",
                _summary_table(
                    top_pairs,
                    [
                        "pair_index",
                        "base_id",
                        "target_variant",
                        "matches",
                        "correct",
                        "wrong",
                        "precision",
                        "wrong_displacement_mad_px",
                        "false_cluster",
                        "primary_buckets",
                        "training_action",
                    ],
                ),
                "<h2>Summary JSON</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-wrong-min", type=int, default=2)
    parser.add_argument("--false-cluster-mad-px", type=float, default=5.0)
    parser.add_argument("--high-score-min", type=float, default=12.0)
    parser.add_argument("--high-accept-min", type=float, default=0.70)
    parser.add_argument("--high-raw-margin-min", type=float, default=0.05)
    parser.add_argument("--near-miss-px", type=float, default=8.0)
    parser.add_argument("--far-error-px", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = BucketConfig(
        cluster_wrong_min=args.cluster_wrong_min,
        false_cluster_mad_px=args.false_cluster_mad_px,
        high_score_min=args.high_score_min,
        high_accept_min=args.high_accept_min,
        high_raw_margin_min=args.high_raw_margin_min,
        near_miss_px=args.near_miss_px,
        far_error_px=args.far_error_px,
    )
    rows = _read_csv_rows(args.match_details)
    wrong_rows, pair_rows, bucket_rows, error_bin_rows, summary = build_failure_analysis(rows, config=config)
    _write_csv(args.output_dir / "wrong_match_buckets.csv", wrong_rows, WRONG_MATCH_FIELDS)
    _write_csv(args.output_dir / "pair_failure_summary.csv", pair_rows, PAIR_SUMMARY_FIELDS)
    _write_csv(args.output_dir / "bucket_summary.csv", bucket_rows, BUCKET_SUMMARY_FIELDS)
    _write_csv(args.output_dir / "error_bin_summary.csv", error_bin_rows, ERROR_BIN_SUMMARY_FIELDS)
    _write_json(args.output_dir / "summary.json", summary)
    _write_html(
        args.output_dir / "index.html",
        match_details=args.match_details,
        summary=summary,
        bucket_rows=bucket_rows,
        error_bin_rows=error_bin_rows,
        pair_rows=pair_rows,
    )
    print(
        f"matches={summary['matches']} wrong={summary['wrong']} "
        f"false_cluster_pairs={summary['false_cluster_pairs']} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
