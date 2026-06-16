#!/usr/bin/env python3
"""Mine reusable hard failure pair manifests from lazy visual reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PAIR_MANIFEST_FIELDS = [
    "pair_index",
    "split",
    "pair_type",
    "reference_dataset_id",
    "reference_pose_id",
    "reference_base_id",
    "reference_variant",
    "target_dataset_id",
    "target_pose_id",
    "target_base_id",
    "target_variant",
    "valid_fraction",
    "valid_pixels",
    "attempts",
    "crop_a_x0",
    "crop_a_y0",
    "crop_a_x1",
    "crop_a_y1",
    "crop_b_x0",
    "crop_b_y0",
    "crop_b_x1",
    "crop_b_y1",
]

EXTRA_FIELDS = [
    "hard_reasons",
    "hard_score",
    "source_matches",
    "source_correct",
    "source_wrong",
    "source_precision",
    "source_mean_error_px",
    "source_median_error_px",
    "source_summary",
]


@dataclass(frozen=True)
class HardFailureConfig:
    low_precision_threshold: float = 0.85
    high_wrong_threshold: int = 32
    low_match_threshold: int = 4
    high_loss_threshold: float = 0.0
    extreme_variants: tuple[str, ...] = ("extreme_02", "extreme_03")
    only_extreme_variants: bool = False
    include_extreme_without_failure: bool = False
    reference_variant: str = "nadir"


def config_from_args(args: argparse.Namespace) -> HardFailureConfig:
    low_precision_threshold = float(args.low_precision_threshold)
    high_wrong_threshold = int(args.high_wrong_threshold)
    low_match_threshold = int(args.low_match_threshold)
    high_loss_threshold = float(args.high_loss_threshold)
    if args.failure_preset == "residual_filtered":
        low_precision_threshold = 0.95
        high_wrong_threshold = 1
        low_match_threshold = 8
        high_loss_threshold = 0.0
    return HardFailureConfig(
        low_precision_threshold=low_precision_threshold,
        high_wrong_threshold=high_wrong_threshold,
        low_match_threshold=low_match_threshold,
        high_loss_threshold=high_loss_threshold,
        extreme_variants=tuple(item.strip() for item in args.extreme_variants.split(",") if item.strip()),
        only_extreme_variants=args.only_extreme_variants,
        include_extreme_without_failure=args.include_extreme_without_failure,
        reference_variant=args.reference_variant,
    )


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


def _summary_key(row: dict[str, str], *, reference_variant: str) -> tuple[str, str, str, str]:
    split = row.get("split", "")
    base_id = row.get("base_id") or row.get("reference_base_id") or row.get("target_base_id") or ""
    target_variant = row.get("target_variant", "")
    return split, base_id, reference_variant, target_variant


def _pair_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("split", ""),
        row.get("reference_base_id", ""),
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


def classify_summary_row(row: dict[str, str], *, config: HardFailureConfig) -> tuple[list[str], float]:
    matches = _int_value(row, "matches")
    wrong = _int_value(row, "wrong")
    precision = _float_value(row, "precision", 1.0 if matches > 0 else 0.0)
    target_variant = row.get("target_variant", "")
    loss = max(
        _float_value(row, "loss", 0.0),
        _float_value(row, "pair_loss", 0.0),
        _float_value(row, "mean_loss", 0.0),
    )

    reasons: list[str] = []
    if matches <= config.low_match_threshold:
        reasons.append("low_match_count")
    if matches > 0 and precision < config.low_precision_threshold:
        reasons.append("low_precision")
    if wrong >= config.high_wrong_threshold:
        reasons.append("high_false")
    if config.high_loss_threshold > 0.0 and loss >= config.high_loss_threshold:
        reasons.append("high_loss")

    is_extreme = target_variant in set(config.extreme_variants)
    if config.only_extreme_variants and not is_extreme:
        return [], 0.0
    if is_extreme and (reasons or config.include_extreme_without_failure):
        reasons.append("extreme_view")

    hard_score = 0.0
    if "low_precision" in reasons:
        hard_score += max(0.0, config.low_precision_threshold - precision) * 100.0
    if "high_false" in reasons:
        hard_score += float(wrong)
    if "low_match_count" in reasons:
        hard_score += float(config.low_match_threshold - matches + 1)
    if "high_loss" in reasons:
        hard_score += loss
    if "extreme_view" in reasons:
        hard_score += 5.0
    return reasons, hard_score


def mine_hard_failure_rows(
    pair_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    *,
    config: HardFailureConfig,
    source_summary: str = "",
) -> list[dict[str, str]]:
    pairs_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        pairs_by_key[_pair_key(row)].append(row)
    summary_key_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    selected: dict[tuple[tuple[str, str, str, str], int], dict[str, str]] = {}
    for summary in summary_rows:
        reasons, hard_score = classify_summary_row(summary, config=config)
        if not reasons:
            continue
        key = _summary_key(summary, reference_variant=config.reference_variant)
        occurrence_index = summary_key_counts[key]
        summary_key_counts[key] += 1
        pairs = pairs_by_key.get(key, [])
        if occurrence_index >= len(pairs):
            continue
        pair = pairs[occurrence_index]
        row = {field: pair.get(field, "") for field in PAIR_MANIFEST_FIELDS}
        row.update(
            {
                "hard_reasons": "|".join(dict.fromkeys(reasons)),
                "hard_score": f"{hard_score:.6f}",
                "source_matches": str(_int_value(summary, "matches")),
                "source_correct": str(_int_value(summary, "correct")),
                "source_wrong": str(_int_value(summary, "wrong")),
                "source_precision": f"{_float_value(summary, 'precision'):.6f}",
                "source_mean_error_px": f"{_float_value(summary, 'mean_error_px'):.3f}",
                "source_median_error_px": f"{_float_value(summary, 'median_error_px'):.3f}",
                "source_summary": source_summary,
            }
        )
        selected_key = (key, occurrence_index)
        previous = selected.get(selected_key)
        if previous is None or float(row["hard_score"]) > float(previous["hard_score"]):
            selected[selected_key] = row
    mined = list(selected.values())
    mined.sort(key=lambda row: (-float(row.get("hard_score") or 0.0), row.get("reference_base_id", ""), row.get("target_variant", "")))
    for index, row in enumerate(mined):
        row["pair_index"] = str(index)
    return mined


def write_hard_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*PAIR_MANIFEST_FIELDS, *EXTRA_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def filter_hard_failure_rows_by_required_reasons(
    rows: list[dict[str, str]],
    required_reasons: list[str],
) -> list[dict[str, str]]:
    required = {reason.strip() for reason in required_reasons if reason.strip()}
    if not required:
        return rows
    filtered: list[dict[str, str]] = []
    for row in rows:
        row_reasons = {reason for reason in row.get("hard_reasons", "").split("|") if reason}
        if required.issubset(row_reasons):
            filtered.append(row)
    return filtered


def _reindexed_row(row: dict[str, str], index: int) -> dict[str, str]:
    copied = dict(row)
    copied["pair_index"] = str(index)
    return copied


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


def write_report(path: Path, *, args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row.get("hard_reasons", "").split("|"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    payload = {
        "output_manifest": str(args.output_manifest),
        "mixed_output_manifest": str(args.mixed_output_manifest or ""),
        "mixed_hard_fraction": float(args.mixed_hard_fraction),
        "failure_preset": str(args.failure_preset),
        "effective_config": config_from_args(args).__dict__,
        "required_reason": list(args.required_reason),
        "rows": len(rows),
        "reason_counts": counts,
        "summary_csv": [str(path) for path in args.summary_csv],
        "pair_manifest": str(args.pair_manifest),
        "mixed_base_manifest": str(args.mixed_base_manifest or ""),
    }
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Hard failure mining</title></head>
<body>
<h1>Hard failure mining</h1>
<pre>{html.escape(json.dumps(payload, indent=2, ensure_ascii=False))}</pre>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-base-manifest", type=Path, default=None)
    parser.add_argument("--mixed-output-manifest", type=Path, default=None)
    parser.add_argument("--mixed-hard-fraction", type=float, default=0.0)
    parser.add_argument("--report-html", type=Path, default=None)
    parser.add_argument("--reference-variant", default="nadir")
    parser.add_argument("--low-precision-threshold", type=float, default=0.85)
    parser.add_argument("--high-wrong-threshold", type=int, default=32)
    parser.add_argument("--low-match-threshold", type=int, default=4)
    parser.add_argument("--high-loss-threshold", type=float, default=0.0)
    parser.add_argument("--extreme-variants", default="extreme_02,extreme_03")
    parser.add_argument("--only-extreme-variants", action="store_true")
    parser.add_argument("--include-extreme-without-failure", action="store_true")
    parser.add_argument("--required-reason", action="append", default=[])
    parser.add_argument("--failure-preset", choices=["default", "residual_filtered"], default="default")
    parser.add_argument(
        "--residual-filtered",
        dest="failure_preset",
        action="store_const",
        const="residual_filtered",
        help="Use a post-MAGSAC residual failure preset: precision < 0.95, wrong >= 1, or matches <= 8.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.low_precision_threshold < 0.0 or args.low_precision_threshold > 1.0:
        raise ValueError("--low-precision-threshold must be in [0, 1]")
    if args.high_wrong_threshold < 0:
        raise ValueError("--high-wrong-threshold must be nonnegative")
    if args.low_match_threshold < 0:
        raise ValueError("--low-match-threshold must be nonnegative")
    if args.high_loss_threshold < 0.0:
        raise ValueError("--high-loss-threshold must be nonnegative")
    if args.mixed_hard_fraction < 0.0 or args.mixed_hard_fraction >= 1.0:
        raise ValueError("--mixed-hard-fraction must be in [0, 1)")
    if args.mixed_base_manifest is not None and args.mixed_output_manifest is None:
        raise ValueError("--mixed-base-manifest requires --mixed-output-manifest")
    config = config_from_args(args)
    pair_rows = _read_csv_rows(args.pair_manifest)
    all_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for summary_path in args.summary_csv:
        summary_rows = _read_csv_rows(summary_path)
        mined = mine_hard_failure_rows(pair_rows, summary_rows, config=config, source_summary=str(summary_path))
        for row in mined:
            key = _pair_identity(row)
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(row)
    all_rows.sort(key=lambda row: (-float(row.get("hard_score") or 0.0), row.get("reference_base_id", ""), row.get("target_variant", "")))
    all_rows = filter_hard_failure_rows_by_required_reasons(all_rows, args.required_reason)
    for index, row in enumerate(all_rows):
        row["pair_index"] = str(index)
    write_hard_manifest(args.output_manifest, all_rows)
    if args.mixed_output_manifest is not None:
        mixed_base_rows = _read_csv_rows(args.mixed_base_manifest) if args.mixed_base_manifest is not None else pair_rows
        mixed_rows = build_mixed_manifest_rows(
            mixed_base_rows,
            all_rows,
            target_hard_fraction=float(args.mixed_hard_fraction),
        )
        write_hard_manifest(args.mixed_output_manifest, mixed_rows)
    if args.report_html is not None:
        write_report(args.report_html, args=args, rows=all_rows)
    mixed_suffix = f" mixed_manifest={args.mixed_output_manifest}" if args.mixed_output_manifest is not None else ""
    print(f"hard_failures={len(all_rows)} manifest={args.output_manifest}{mixed_suffix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
