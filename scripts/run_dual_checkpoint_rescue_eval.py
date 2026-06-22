#!/usr/bin/env python3
"""Combine two checkpoint visual summaries with a conservative extreme rescue selector."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SelectorConfig:
    target_variants: tuple[str, ...] = ("extreme_02", "extreme_03")
    min_match_gain: int = 1
    max_match_gain: int | None = None
    min_rescue_matches: int = 8
    max_rescue_homography_p90_px: float = 3.2
    max_rescue_homography_median_px: float = 1.8
    max_rescue_homography_p90_delta_px: float = -1.0
    min_rescue_displacement_mad_px: float = -1.0
    max_rescue_displacement_mad_px: float = -1.0
    min_rescue_score_mean: float = 16.0
    require_rescue_score_mean_not_lower: bool = True
    min_valid_fraction: float | None = None
    max_valid_fraction: float | None = None
    min_rescue_score_mean_delta: float | None = None
    rescue_gate_target_variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSpec:
    name: str
    split: str
    baseline_summary: Path
    rescue_summary: Path


def safe_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def safe_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def finite_or(value: float, default: float) -> float:
    return value if math.isfinite(value) else default


def safe_float_with_fallback(
    primary_row: dict[str, object],
    fallback_row: dict[str, object],
    key: str,
    default: float = math.nan,
) -> float:
    primary = safe_float(primary_row.get(key), default=math.nan)
    if math.isfinite(primary):
        return primary
    return safe_float(fallback_row.get(key), default=default)


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing summary CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_variant_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def parse_source_spec(text: str) -> SourceSpec:
    parts = text.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--source must be formatted as name,split,baseline_summary,rescue_summary"
        )
    name, split, baseline_summary, rescue_summary = (part.strip() for part in parts)
    if not name or not split or not baseline_summary or not rescue_summary:
        raise argparse.ArgumentTypeError(
            "--source must include non-empty name, split, baseline_summary and rescue_summary"
        )
    return SourceSpec(
        name=name,
        split=split,
        baseline_summary=Path(baseline_summary),
        rescue_summary=Path(rescue_summary),
    )


def extra_rescue_gates_apply(variant: str, config: SelectorConfig) -> bool:
    return not config.rescue_gate_target_variants or variant in set(config.rescue_gate_target_variants)


def selector_reason(base_row: dict[str, object], rescue_row: dict[str, object], config: SelectorConfig) -> str:
    variant = str(rescue_row.get("target_variant", "")).strip()
    if variant not in set(config.target_variants):
        return f"blocked_target_variant:{variant or 'missing'}"

    base_matches = safe_int(base_row.get("matches"))
    rescue_matches = safe_int(rescue_row.get("matches"))
    if rescue_matches < config.min_rescue_matches:
        return f"blocked_min_rescue_matches:{rescue_matches}<{config.min_rescue_matches}"
    required_matches = base_matches + config.min_match_gain
    if rescue_matches < required_matches:
        return f"blocked_match_gain:{rescue_matches}<{required_matches}"
    match_gain = rescue_matches - base_matches
    if config.max_match_gain is not None and match_gain > config.max_match_gain:
        return f"blocked_max_match_gain:{match_gain}>{config.max_match_gain}"

    homography_valid = safe_int(rescue_row.get("homography_residual_valid"), default=1)
    if homography_valid <= 0:
        return "blocked_homography_invalid"

    rescue_p90 = safe_float(rescue_row.get("homography_residual_p90_px"))
    if (
        config.max_rescue_homography_p90_px >= 0
        and (not math.isfinite(rescue_p90) or rescue_p90 > config.max_rescue_homography_p90_px)
    ):
        return f"blocked_homography_p90:{finite_or(rescue_p90, float('inf')):.6g}>{config.max_rescue_homography_p90_px:.6g}"

    if config.max_rescue_homography_p90_delta_px >= 0:
        base_p90 = safe_float(base_row.get("homography_residual_p90_px"))
        if not (math.isfinite(base_p90) and math.isfinite(rescue_p90)):
            return "blocked_homography_p90_delta:missing"
        p90_delta = rescue_p90 - base_p90
        if p90_delta > config.max_rescue_homography_p90_delta_px:
            return (
                f"blocked_homography_p90_delta:"
                f"{p90_delta:.6g}>{config.max_rescue_homography_p90_delta_px:.6g}"
            )

    rescue_median = safe_float(rescue_row.get("homography_residual_median_px"))
    if (
        config.max_rescue_homography_median_px >= 0
        and (not math.isfinite(rescue_median) or rescue_median > config.max_rescue_homography_median_px)
    ):
        return (
            f"blocked_homography_median:"
            f"{finite_or(rescue_median, float('inf')):.6g}>{config.max_rescue_homography_median_px:.6g}"
        )

    rescue_displacement_mad = safe_float(rescue_row.get("displacement_mad_px"))
    if (
        config.min_rescue_displacement_mad_px >= 0
        and (
            not math.isfinite(rescue_displacement_mad)
            or rescue_displacement_mad < config.min_rescue_displacement_mad_px
        )
    ):
        return (
            f"blocked_min_displacement_mad:"
            f"{finite_or(rescue_displacement_mad, float('-inf')):.6g}<"
            f"{config.min_rescue_displacement_mad_px:.6g}"
        )
    if (
        config.max_rescue_displacement_mad_px >= 0
        and (
            not math.isfinite(rescue_displacement_mad)
            or rescue_displacement_mad > config.max_rescue_displacement_mad_px
        )
    ):
        return (
            f"blocked_displacement_mad:"
            f"{finite_or(rescue_displacement_mad, float('inf')):.6g}>"
            f"{config.max_rescue_displacement_mad_px:.6g}"
        )

    apply_extra_gates = extra_rescue_gates_apply(variant, config)
    if apply_extra_gates and config.min_valid_fraction is not None:
        valid_fraction = safe_float_with_fallback(rescue_row, base_row, "valid_fraction")
        if not math.isfinite(valid_fraction) or valid_fraction < config.min_valid_fraction:
            return (
                f"blocked_min_valid_fraction:"
                f"{finite_or(valid_fraction, float('-inf')):.6g}<{config.min_valid_fraction:.6g}"
            )

    if apply_extra_gates and config.max_valid_fraction is not None:
        valid_fraction = safe_float_with_fallback(rescue_row, base_row, "valid_fraction")
        if not math.isfinite(valid_fraction) or valid_fraction > config.max_valid_fraction:
            return (
                f"blocked_valid_fraction:"
                f"{finite_or(valid_fraction, float('inf')):.6g}>{config.max_valid_fraction:.6g}"
            )

    rescue_score = safe_float(rescue_row.get("score_mean"))
    if not math.isfinite(rescue_score) or rescue_score < config.min_rescue_score_mean:
        return f"blocked_score_mean:{finite_or(rescue_score, float('-inf')):.6g}<{config.min_rescue_score_mean:.6g}"

    base_score = safe_float(base_row.get("score_mean"))
    if apply_extra_gates and config.min_rescue_score_mean_delta is not None:
        if not (math.isfinite(base_score) and math.isfinite(rescue_score)):
            return "blocked_score_mean_delta:missing"
        score_delta = rescue_score - base_score
        if score_delta < config.min_rescue_score_mean_delta:
            return (
                f"blocked_score_mean_delta:"
                f"{score_delta:.6g}<{config.min_rescue_score_mean_delta:.6g}"
            )

    if config.require_rescue_score_mean_not_lower:
        if math.isfinite(base_score) and rescue_score < base_score:
            return f"blocked_score_mean_lower:{rescue_score:.6g}<{base_score:.6g}"

    return "rescue_selected"


def should_select_rescue(
    base_row: dict[str, object],
    rescue_row: dict[str, object],
    config: SelectorConfig,
) -> bool:
    return selector_reason(base_row, rescue_row, config) == "rescue_selected"


def _row_identity(row: dict[str, object], *, require_matching_split: bool = True) -> tuple[str, ...]:
    identity = (
        str(row.get("base_id", "")).strip(),
        str(row.get("target_variant", "")).strip(),
    )
    if require_matching_split:
        return (*identity, str(row.get("split", "")).strip())
    return identity


def _validate_aligned_rows(
    baseline_rows: list[dict[str, object]],
    rescue_rows: list[dict[str, object]],
    *,
    source: str,
    require_matching_split: bool = True,
) -> None:
    if len(baseline_rows) != len(rescue_rows):
        raise ValueError(
            f"{source}: baseline/rescue row count mismatch: "
            f"{len(baseline_rows)} vs {len(rescue_rows)}"
        )
    for index, (base_row, rescue_row) in enumerate(zip(baseline_rows, rescue_rows)):
        base_identity = _row_identity(base_row, require_matching_split=require_matching_split)
        rescue_identity = _row_identity(rescue_row, require_matching_split=require_matching_split)
        if base_identity != rescue_identity:
            raise ValueError(
                f"{source}: row {index} mismatch: baseline {base_identity} vs rescue {rescue_identity}"
            )


def combine_summary_rows(
    baseline_rows: list[dict[str, object]],
    rescue_rows: list[dict[str, object]],
    *,
    config: SelectorConfig,
    source: str,
    split: str,
    baseline_label: str,
    rescue_label: str,
    require_matching_split: bool = True,
) -> list[dict[str, object]]:
    _validate_aligned_rows(
        baseline_rows,
        rescue_rows,
        source=source,
        require_matching_split=require_matching_split,
    )
    combined: list[dict[str, object]] = []
    for index, (base_row, rescue_row) in enumerate(zip(baseline_rows, rescue_rows)):
        reason = selector_reason(base_row, rescue_row, config)
        selected_row = rescue_row if reason == "rescue_selected" else base_row
        selected_model = rescue_label if reason == "rescue_selected" else baseline_label
        base_matches = safe_int(base_row.get("matches"))
        rescue_matches = safe_int(rescue_row.get("matches"))
        base_correct = safe_int(base_row.get("correct"))
        rescue_correct = safe_int(rescue_row.get("correct"))
        base_wrong = safe_int(base_row.get("wrong"))
        rescue_wrong = safe_int(rescue_row.get("wrong"))
        base_score = safe_float(base_row.get("score_mean"))
        rescue_score = safe_float(rescue_row.get("score_mean"))
        base_p90 = safe_float(base_row.get("homography_residual_p90_px"))
        rescue_p90 = safe_float(rescue_row.get("homography_residual_p90_px"))
        base_displacement_mad = safe_float(base_row.get("displacement_mad_px"))
        rescue_displacement_mad = safe_float(rescue_row.get("displacement_mad_px"))

        output = dict(selected_row)
        output.update(
            {
                "source": source,
                "split": split,
                "row_index": index,
                "selected_model": selected_model,
                "selector_reason": reason,
                "baseline_model": baseline_label,
                "rescue_model": rescue_label,
                "baseline_matches": base_matches,
                "rescue_matches": rescue_matches,
                "match_delta": rescue_matches - base_matches,
                "baseline_correct": base_correct,
                "rescue_correct": rescue_correct,
                "correct_delta": rescue_correct - base_correct,
                "baseline_wrong": base_wrong,
                "rescue_wrong": rescue_wrong,
                "wrong_delta": rescue_wrong - base_wrong,
                "baseline_score_mean": "" if not math.isfinite(base_score) else f"{base_score:.6f}",
                "rescue_score_mean": "" if not math.isfinite(rescue_score) else f"{rescue_score:.6f}",
                "score_mean_delta": ""
                if not (math.isfinite(base_score) and math.isfinite(rescue_score))
                else f"{rescue_score - base_score:.6f}",
                "baseline_homography_p90_px": "" if not math.isfinite(base_p90) else f"{base_p90:.6f}",
                "rescue_homography_p90_px": "" if not math.isfinite(rescue_p90) else f"{rescue_p90:.6f}",
                "homography_p90_delta": ""
                if not (math.isfinite(base_p90) and math.isfinite(rescue_p90))
                else f"{rescue_p90 - base_p90:.6f}",
                "baseline_displacement_mad_px": ""
                if not math.isfinite(base_displacement_mad)
                else f"{base_displacement_mad:.6f}",
                "rescue_displacement_mad_px": ""
                if not math.isfinite(rescue_displacement_mad)
                else f"{rescue_displacement_mad:.6f}",
                "displacement_mad_delta": ""
                if not (math.isfinite(base_displacement_mad) and math.isfinite(rescue_displacement_mad))
                else f"{rescue_displacement_mad - base_displacement_mad:.6f}",
            }
        )
        combined.append(output)
    return combined


def summarize_rows(
    rows: Iterable[dict[str, object]],
    *,
    label: str,
    source: str,
    split: str,
) -> dict[str, object]:
    materialized = list(rows)
    matches = sum(safe_int(row.get("matches")) for row in materialized)
    correct = sum(safe_int(row.get("correct")) for row in materialized)
    wrong = sum(safe_int(row.get("wrong")) for row in materialized)
    precision = float(correct) / float(matches) if matches else 0.0
    medians = [safe_float(row.get("median_error_px")) for row in materialized]
    finite_medians = [value for value in medians if math.isfinite(value)]
    mean_median_error = sum(finite_medians) / len(finite_medians) if finite_medians else math.nan
    return {
        "model": label,
        "source": source,
        "split": split,
        "filtered_rows": len(materialized),
        "filtered_matches": matches,
        "filtered_correct": correct,
        "filtered_wrong": wrong,
        "filtered_precision": f"{precision:.6f}",
        "filtered_median_error_px": "" if not math.isfinite(mean_median_error) else f"{mean_median_error:.6f}",
    }


def summarize_by_variant(
    rows: Iterable[dict[str, object]],
    *,
    label: str,
    source: str,
    split: str,
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        variant = str(row.get("target_variant", "")).strip() or "unknown"
        groups.setdefault(variant, []).append(row)
    summaries: list[dict[str, object]] = []
    for variant in sorted(groups):
        summary = summarize_rows(groups[variant], label=label, source=source, split=split)
        summary["variant"] = variant
        summaries.append(summary)
    return summaries


def write_csv_rows(path: Path, rows: list[dict[str, object]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for field in preferred_fields or []:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_html_report(
    path: Path,
    *,
    sources: list[SourceSpec],
    config: SelectorConfig,
    summary_rows: list[dict[str, object]],
    variant_rows: list[dict[str, object]],
    combined_rows: list[dict[str, object]],
) -> None:
    def table(rows: list[dict[str, object]], columns: list[str]) -> str:
        header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body_parts = []
        for row in rows:
            body_parts.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
                + "</tr>"
            )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"

    selected_count = sum(1 for row in combined_rows if row.get("selector_reason") == "rescue_selected")
    source_items = "".join(
        "<li>"
        f"{html.escape(item.name)}/{html.escape(item.split)}: "
        f"<code>{html.escape(str(item.baseline_summary))}</code> + "
        f"<code>{html.escape(str(item.rescue_summary))}</code>"
        "</li>"
        for item in sources
    )
    selected_rows = [row for row in combined_rows if row.get("selector_reason") == "rescue_selected"]
    selected_preview = selected_rows[:50]
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>dual checkpoint rescue selector</title>
  <style>
    body {{ font-family: sans-serif; line-height: 1.45; margin: 24px; }}
    code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 5px 7px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .note {{ color: #555; }}
  </style>
</head>
<body>
  <h1>dual checkpoint rescue selector</h1>
  <p class="note">默认使用 baseline checkpoint，只在 selector 条件全部满足时使用 rescue checkpoint 行。selector 只使用推理可观测字段，不使用 correct/wrong/precision 决策。</p>
  <h2>输入</h2>
  <ul>{source_items}</ul>
  <h2>配置</h2>
  <pre>{html.escape(json.dumps(asdict(config), ensure_ascii=False, indent=2))}</pre>
  <h2>总表</h2>
  {table(summary_rows, ["source", "split", "model", "filtered_rows", "filtered_matches", "filtered_correct", "filtered_wrong", "filtered_precision"])}
  <h2>按 variant 的 selected 输出</h2>
  {table(variant_rows, ["source", "split", "variant", "model", "filtered_rows", "filtered_matches", "filtered_correct", "filtered_wrong", "filtered_precision"])}
  <h2>rescue_selected 样本</h2>
  <p>rescue_selected rows: {selected_count}</p>
  {table(selected_preview, ["source", "split", "base_id", "target_variant", "matches", "correct", "wrong", "score_mean", "homography_residual_p90_px", "selected_model", "selector_reason"])}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="phase3zn")
    parser.add_argument("--rescue-label", default="phase5d")
    parser.add_argument("--target-variants", default="extreme_02,extreme_03")
    parser.add_argument("--min-match-gain", type=int, default=1)
    parser.add_argument(
        "--max-match-gain",
        type=int,
        default=None,
        help="If set, block rescue rows whose rescue matches minus baseline matches exceeds this value.",
    )
    parser.add_argument("--min-rescue-matches", type=int, default=8)
    parser.add_argument("--max-rescue-homography-p90-px", type=float, default=3.2)
    parser.add_argument("--max-rescue-homography-median-px", type=float, default=1.8)
    parser.add_argument("--max-rescue-homography-p90-delta-px", type=float, default=-1.0)
    parser.add_argument(
        "--min-rescue-displacement-mad-px",
        type=float,
        default=-1.0,
        help="If >= 0, block rescue rows whose observable displacement_mad_px is below this threshold.",
    )
    parser.add_argument(
        "--max-rescue-displacement-mad-px",
        type=float,
        default=-1.0,
        help="If >= 0, block rescue rows whose observable displacement_mad_px exceeds this threshold.",
    )
    parser.add_argument("--min-rescue-score-mean", type=float, default=16.0)
    parser.add_argument("--allow-rescue-score-mean-drop", action="store_true")
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=None,
        help="If set, block rescue rows whose valid_fraction is below this inference-observable threshold.",
    )
    parser.add_argument(
        "--max-valid-fraction",
        type=float,
        default=None,
        help="If set, block rescue rows whose valid_fraction exceeds this inference-observable threshold.",
    )
    parser.add_argument(
        "--min-rescue-score-mean-delta",
        type=float,
        default=None,
        help="If set, block rescue rows when rescue score_mean - baseline score_mean is below this value.",
    )
    parser.add_argument(
        "--rescue-gate-target-variants",
        default="",
        help=(
            "Comma-separated target variants for --min-valid-fraction, --max-valid-fraction and "
            "--min-rescue-score-mean-delta. Empty applies those gates to all rescue variants."
        ),
    )
    parser.add_argument(
        "--ignore-row-split-for-alignment",
        action="store_true",
        help=(
            "Align baseline/rescue rows by base_id and target_variant only. Use for sequential selector "
            "replay where a previous combined CSV has source split labels."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = SelectorConfig(
        target_variants=parse_variant_list(args.target_variants),
        min_match_gain=args.min_match_gain,
        max_match_gain=args.max_match_gain,
        min_rescue_matches=args.min_rescue_matches,
        max_rescue_homography_p90_px=args.max_rescue_homography_p90_px,
        max_rescue_homography_median_px=args.max_rescue_homography_median_px,
        max_rescue_homography_p90_delta_px=args.max_rescue_homography_p90_delta_px,
        min_rescue_displacement_mad_px=args.min_rescue_displacement_mad_px,
        max_rescue_displacement_mad_px=args.max_rescue_displacement_mad_px,
        min_rescue_score_mean=args.min_rescue_score_mean,
        require_rescue_score_mean_not_lower=not args.allow_rescue_score_mean_drop,
        min_valid_fraction=args.min_valid_fraction,
        max_valid_fraction=args.max_valid_fraction,
        min_rescue_score_mean_delta=args.min_rescue_score_mean_delta,
        rescue_gate_target_variants=parse_variant_list(args.rescue_gate_target_variants),
    )

    all_combined_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    for source in args.source:
        baseline_rows = read_summary_rows(source.baseline_summary)
        rescue_rows = read_summary_rows(source.rescue_summary)
        combined_rows = combine_summary_rows(
            baseline_rows,
            rescue_rows,
            config=config,
            source=source.name,
            split=source.split,
            baseline_label=args.baseline_label,
            rescue_label=args.rescue_label,
            require_matching_split=not args.ignore_row_split_for_alignment,
        )
        all_combined_rows.extend(combined_rows)
        summary_rows.append(
            summarize_rows(baseline_rows, label=args.baseline_label, source=source.name, split=source.split)
        )
        summary_rows.append(summarize_rows(rescue_rows, label=args.rescue_label, source=source.name, split=source.split))
        summary_rows.append(summarize_rows(combined_rows, label="selected", source=source.name, split=source.split))
        variant_rows.extend(
            summarize_by_variant(combined_rows, label="selected", source=source.name, split=source.split)
        )

    preferred_combined_fields = [
        "source",
        "split",
        "row_index",
        "base_id",
        "target_variant",
        "matches",
        "correct",
        "wrong",
        "precision",
        "score_mean",
        "homography_residual_median_px",
        "homography_residual_p90_px",
        "selected_model",
        "selector_reason",
        "baseline_matches",
        "rescue_matches",
        "match_delta",
        "baseline_correct",
        "rescue_correct",
        "correct_delta",
        "baseline_wrong",
        "rescue_wrong",
        "wrong_delta",
        "baseline_score_mean",
        "rescue_score_mean",
        "score_mean_delta",
        "baseline_displacement_mad_px",
        "rescue_displacement_mad_px",
        "displacement_mad_delta",
    ]
    write_csv_rows(args.output_dir / "combined_filtered_summary.csv", all_combined_rows, preferred_combined_fields)
    write_csv_rows(
        args.output_dir / "summary.csv",
        summary_rows,
        [
            "source",
            "split",
            "model",
            "filtered_rows",
            "filtered_matches",
            "filtered_correct",
            "filtered_wrong",
            "filtered_precision",
            "filtered_median_error_px",
        ],
    )
    write_csv_rows(
        args.output_dir / "variant_summary.csv",
        variant_rows,
        [
            "source",
            "split",
            "variant",
            "model",
            "filtered_rows",
            "filtered_matches",
            "filtered_correct",
            "filtered_wrong",
            "filtered_precision",
            "filtered_median_error_px",
        ],
    )
    metadata = {
        "baseline_label": args.baseline_label,
        "rescue_label": args.rescue_label,
        "config": asdict(config),
        "require_matching_split": not args.ignore_row_split_for_alignment,
        "sources": [
            {
                "name": source.name,
                "split": source.split,
                "baseline_summary": str(source.baseline_summary),
                "rescue_summary": str(source.rescue_summary),
            }
            for source in args.source
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html_report(
        args.output_dir / "index.html",
        sources=args.source,
        config=config,
        summary_rows=summary_rows,
        variant_rows=variant_rows,
        combined_rows=all_combined_rows,
    )
    print(f"report={args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
