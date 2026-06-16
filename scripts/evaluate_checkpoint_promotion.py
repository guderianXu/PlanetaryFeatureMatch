#!/usr/bin/env python3
"""Evaluate whether a candidate checkpoint may replace the current best."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalMetrics:
    context: str
    split: str
    label: str
    matches: int
    correct: int
    wrong: int
    precision: float


@dataclass(frozen=True)
class MetricComparison:
    context: str
    split: str
    baseline: EvalMetrics
    candidate: EvalMetrics
    correct_delta: int
    wrong_delta: int
    precision_delta: float
    passed: bool
    reason: str


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    passed_reasons: list[str]
    failed_reasons: list[str]
    comparisons: list[MetricComparison]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_int(row: dict[str, str], *names: str) -> int:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return int(float(value))
    raise KeyError(f"missing integer field, expected one of: {', '.join(names)}")


def _as_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"missing float field, expected one of: {', '.join(names)}")


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _sweep_failure_reasons(
    rows: list[dict[str, str]],
    *,
    identity_column: str,
    context_column: str | None,
) -> list[str]:
    reasons: list[str] = []
    for row in rows:
        if not _as_bool(row.get("sweep_failed")):
            continue
        label = row.get(identity_column, "unknown")
        split = row.get("split", "unknown")
        context = row.get(context_column, "formal") if context_column is not None else "formal"
        error = row.get("sweep_error") or "unknown error"
        reasons.append(f"FAIL {context}/{split}: sweep_failed for {identity_column}={label!r} ({error})")
    return reasons


def _find_row(
    rows: list[dict[str, str]],
    *,
    identity_column: str,
    label: str,
    split: str,
    set_name: str | None = None,
) -> dict[str, str]:
    for row in rows:
        if row.get(identity_column) != label:
            continue
        if row.get("split") != split:
            continue
        if set_name is not None and row.get("set") != set_name:
            continue
        return row
    context = f"{set_name}/{split}" if set_name is not None else split
    raise ValueError(f"missing row for {identity_column}={label!r} context={context!r}")


def _variant_values(
    rows: list[dict[str, str]],
    *,
    identity_column: str,
    label: str,
    split: str,
    set_name: str | None = None,
    variant_column: str = "variant",
) -> set[str]:
    values: set[str] = set()
    for row in rows:
        if row.get(identity_column) != label:
            continue
        if row.get("split") != split:
            continue
        if set_name is not None and row.get("set") != set_name:
            continue
        variant = row.get(variant_column)
        if variant:
            values.add(variant)
    return values


def _find_variant_row(
    rows: list[dict[str, str]],
    *,
    identity_column: str,
    label: str,
    split: str,
    variant: str,
    set_name: str | None = None,
    variant_column: str = "variant",
) -> dict[str, str]:
    for row in rows:
        if row.get(identity_column) != label:
            continue
        if row.get("split") != split:
            continue
        if row.get(variant_column) != variant:
            continue
        if set_name is not None and row.get("set") != set_name:
            continue
        return row
    context = f"{set_name}/{split}/{variant}" if set_name is not None else f"{split}/{variant}"
    raise ValueError(f"missing row for {identity_column}={label!r} context={context!r}")


def _metrics_from_row(row: dict[str, str], *, context: str, split: str, label: str) -> EvalMetrics:
    return EvalMetrics(
        context=context,
        split=split,
        label=label,
        matches=_as_int(row, "filtered_matches", "matches"),
        correct=_as_int(row, "filtered_correct", "correct"),
        wrong=_as_int(row, "filtered_wrong", "wrong"),
        precision=_as_float(row, "filtered_precision", "precision"),
    )


def _metrics_from_rows(
    rows: list[dict[str, str]],
    *,
    context: str,
    split: str,
    label: str,
) -> EvalMetrics:
    matches = sum(_as_int(row, "filtered_matches", "matches") for row in rows)
    correct = sum(_as_int(row, "filtered_correct", "correct") for row in rows)
    wrong = sum(_as_int(row, "filtered_wrong", "wrong") for row in rows)
    precision = float(correct) / float(matches) if matches > 0 else 0.0
    return EvalMetrics(
        context=context,
        split=split,
        label=label,
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
    )


def _compare(
    *,
    context: str,
    split: str,
    baseline: EvalMetrics,
    candidate: EvalMetrics,
    max_precision_drop: float,
    max_correct_drop: int,
    max_wrong_increase: int,
    min_correct_gain: int = -10**12,
) -> MetricComparison:
    correct_delta = candidate.correct - baseline.correct
    wrong_delta = candidate.wrong - baseline.wrong
    precision_delta = candidate.precision - baseline.precision
    failures: list[str] = []
    if precision_delta < -float(max_precision_drop):
        failures.append(
            f"precision_delta={precision_delta:.6f} < -{float(max_precision_drop):.6f}"
        )
    if correct_delta < -int(max_correct_drop):
        failures.append(f"correct_delta={correct_delta} < -{int(max_correct_drop)}")
    if wrong_delta > int(max_wrong_increase):
        failures.append(f"wrong_delta={wrong_delta} > {int(max_wrong_increase)}")
    if correct_delta < int(min_correct_gain):
        failures.append(f"correct_delta={correct_delta} < required_gain={int(min_correct_gain)}")
    passed = not failures
    status = "PASS" if passed else "FAIL"
    reason = (
        f"{status} {context}/{split}: correct_delta={correct_delta}, "
        f"wrong_delta={wrong_delta}, precision_delta={precision_delta:.6f}"
    )
    if failures:
        reason = reason + " (" + "; ".join(failures) + ")"
    return MetricComparison(
        context=context,
        split=split,
        baseline=baseline,
        candidate=candidate,
        correct_delta=correct_delta,
        wrong_delta=wrong_delta,
        precision_delta=precision_delta,
        passed=passed,
        reason=reason,
    )


def evaluate_promotion(
    *,
    formal_summary: Path,
    formal_variant_summary: Path | None = None,
    guard_summary: Path,
    baseline_label: str,
    candidate_label: str,
    guard_baseline_label: str | None = None,
    guard_candidate_label: str | None = None,
    splits: list[str],
    formal_identity_column: str = "label",
    formal_variant_identity_column: str = "label",
    guard_identity_column: str = "model",
    formal_target_variants: list[str] | None = None,
    formal_protected_variants: list[str] | None = None,
    max_formal_precision_drop: float = 0.0,
    max_formal_correct_drop: int = 0,
    max_formal_wrong_increase: int = 0,
    min_formal_target_correct_gain: int = 0,
    min_formal_target_total_correct_gain: int = 0,
    max_formal_target_precision_drop: float = 0.0,
    max_formal_target_wrong_increase: int = 0,
    max_protected_variant_precision_drop: float = 0.0,
    max_protected_variant_correct_drop: int = 0,
    max_protected_variant_wrong_increase: int = 0,
    max_guard_precision_drop: float = 0.0,
    max_guard_correct_drop: int = 0,
    max_guard_wrong_increase: int = 0,
    extra_regression_guard_sets: list[str] | None = None,
    max_extra_guard_precision_drop: float = 0.0,
    max_extra_guard_correct_drop: int = 0,
    max_extra_guard_wrong_increase: int = 0,
    min_extreme_correct_gain: int = 1,
    max_extreme_precision_drop: float = 0.02,
    max_extreme_wrong_increase: int = 10**12,
) -> PromotionDecision:
    formal_rows = _read_rows(formal_summary)
    guard_rows = _read_rows(guard_summary)
    sweep_failure_reasons = [
        *_sweep_failure_reasons(formal_rows, identity_column=formal_identity_column, context_column=None),
        *_sweep_failure_reasons(guard_rows, identity_column=guard_identity_column, context_column="set"),
    ]
    if sweep_failure_reasons:
        return PromotionDecision(
            promote=False,
            passed_reasons=[],
            failed_reasons=sweep_failure_reasons,
            comparisons=[],
        )
    comparisons: list[MetricComparison] = []
    guard_baseline = baseline_label if guard_baseline_label is None else guard_baseline_label
    guard_candidate = candidate_label if guard_candidate_label is None else guard_candidate_label
    target_variants = set(formal_target_variants or [])
    protected_variants = set(formal_protected_variants or [])
    use_formal_variant_gate = formal_variant_summary is not None and bool(target_variants or protected_variants)

    if use_formal_variant_gate:
        variant_rows = _read_rows(formal_variant_summary)
        target_total_base_rows: list[dict[str, str]] = []
        target_total_cand_rows: list[dict[str, str]] = []
        for split in splits:
            variants = sorted(
                _variant_values(
                    variant_rows,
                    identity_column=formal_variant_identity_column,
                    label=baseline_label,
                    split=split,
                )
                | _variant_values(
                    variant_rows,
                    identity_column=formal_variant_identity_column,
                    label=candidate_label,
                    split=split,
                )
            )
            if not variants:
                raise ValueError(f"no formal variant rows found for split={split!r}")
            target_present = [variant for variant in variants if variant in target_variants]
            if target_present:
                context = f"formal_target_variants:{','.join(target_present)}"
                base_target_rows = [
                    _find_variant_row(
                        variant_rows,
                        identity_column=formal_variant_identity_column,
                        label=baseline_label,
                        split=split,
                        variant=variant,
                    )
                    for variant in target_present
                ]
                cand_target_rows = [
                    _find_variant_row(
                        variant_rows,
                        identity_column=formal_variant_identity_column,
                        label=candidate_label,
                        split=split,
                        variant=variant,
                    )
                    for variant in target_present
                ]
                target_total_base_rows.extend(base_target_rows)
                target_total_cand_rows.extend(cand_target_rows)
                comparisons.append(
                    _compare(
                        context=context,
                        split=split,
                        baseline=_metrics_from_rows(base_target_rows, context=context, split=split, label=baseline_label),
                        candidate=_metrics_from_rows(cand_target_rows, context=context, split=split, label=candidate_label),
                        max_precision_drop=max_formal_target_precision_drop,
                        max_correct_drop=10**12,
                        max_wrong_increase=max_formal_target_wrong_increase,
                        min_correct_gain=min_formal_target_correct_gain,
                    )
                )
            for variant in variants:
                if variant in target_variants:
                    continue
                base_row = _find_variant_row(
                    variant_rows,
                    identity_column=formal_variant_identity_column,
                    label=baseline_label,
                    split=split,
                    variant=variant,
                )
                cand_row = _find_variant_row(
                    variant_rows,
                    identity_column=formal_variant_identity_column,
                    label=candidate_label,
                    split=split,
                    variant=variant,
                )
                context = (
                    f"formal_protected_variant:{variant}"
                    if variant in protected_variants
                    else f"formal_other_variant:{variant}"
                )
                comparisons.append(
                    _compare(
                        context=context,
                        split=split,
                        baseline=_metrics_from_row(base_row, context=context, split=split, label=baseline_label),
                        candidate=_metrics_from_row(cand_row, context=context, split=split, label=candidate_label),
                        max_precision_drop=max_protected_variant_precision_drop,
                        max_correct_drop=max_protected_variant_correct_drop,
                        max_wrong_increase=max_protected_variant_wrong_increase,
                    )
                )
        if min_formal_target_total_correct_gain > 0:
            if not target_total_base_rows or not target_total_cand_rows:
                raise ValueError("formal target total gate requested but no target variant rows were found")
            comparisons.append(
                _compare(
                    context="formal_target_total",
                    split="all",
                    baseline=_metrics_from_rows(
                        target_total_base_rows,
                        context="formal_target_total",
                        split="all",
                        label=baseline_label,
                    ),
                    candidate=_metrics_from_rows(
                        target_total_cand_rows,
                        context="formal_target_total",
                        split="all",
                        label=candidate_label,
                    ),
                    max_precision_drop=max_formal_target_precision_drop,
                    max_correct_drop=10**12,
                    max_wrong_increase=max_formal_target_wrong_increase,
                    min_correct_gain=min_formal_target_total_correct_gain,
                )
            )
    else:
        for split in splits:
            base_row = _find_row(
                formal_rows,
                identity_column=formal_identity_column,
                label=baseline_label,
                split=split,
            )
            cand_row = _find_row(
                formal_rows,
                identity_column=formal_identity_column,
                label=candidate_label,
                split=split,
            )
            comparisons.append(
                _compare(
                    context="formal",
                    split=split,
                    baseline=_metrics_from_row(base_row, context="formal", split=split, label=baseline_label),
                    candidate=_metrics_from_row(cand_row, context="formal", split=split, label=candidate_label),
                    max_precision_drop=max_formal_precision_drop,
                    max_correct_drop=max_formal_correct_drop,
                    max_wrong_increase=max_formal_wrong_increase,
                )
            )

    regression_guard_sets = ["regression_guard"]
    extra_guard_sets: list[str] = []
    for set_name in extra_regression_guard_sets or []:
        if set_name and set_name not in regression_guard_sets and set_name != "extreme_gain":
            extra_guard_sets.append(set_name)

    for set_name in [*regression_guard_sets, *extra_guard_sets, "extreme_gain"]:
        for split in splits:
            base_row = _find_row(
                guard_rows,
                identity_column=guard_identity_column,
                label=guard_baseline,
                split=split,
                set_name=set_name,
            )
            cand_row = _find_row(
                guard_rows,
                identity_column=guard_identity_column,
                label=guard_candidate,
                split=split,
                set_name=set_name,
            )
            if set_name in regression_guard_sets:
                comparisons.append(
                    _compare(
                        context=set_name,
                        split=split,
                        baseline=_metrics_from_row(base_row, context=set_name, split=split, label=guard_baseline),
                        candidate=_metrics_from_row(cand_row, context=set_name, split=split, label=guard_candidate),
                        max_precision_drop=max_guard_precision_drop,
                        max_correct_drop=max_guard_correct_drop,
                        max_wrong_increase=max_guard_wrong_increase,
                    )
                )
            elif set_name in extra_guard_sets:
                comparisons.append(
                    _compare(
                        context=set_name,
                        split=split,
                        baseline=_metrics_from_row(base_row, context=set_name, split=split, label=guard_baseline),
                        candidate=_metrics_from_row(cand_row, context=set_name, split=split, label=guard_candidate),
                        max_precision_drop=max_extra_guard_precision_drop,
                        max_correct_drop=max_extra_guard_correct_drop,
                        max_wrong_increase=max_extra_guard_wrong_increase,
                    )
                )
            else:
                comparisons.append(
                    _compare(
                        context=set_name,
                        split=split,
                        baseline=_metrics_from_row(base_row, context=set_name, split=split, label=guard_baseline),
                        candidate=_metrics_from_row(cand_row, context=set_name, split=split, label=guard_candidate),
                        max_precision_drop=max_extreme_precision_drop,
                        max_correct_drop=10**12,
                        max_wrong_increase=max_extreme_wrong_increase,
                        min_correct_gain=min_extreme_correct_gain,
                    )
                )

    passed_reasons = [item.reason for item in comparisons if item.passed]
    failed_reasons = [item.reason for item in comparisons if not item.passed]
    return PromotionDecision(
        promote=not failed_reasons,
        passed_reasons=passed_reasons,
        failed_reasons=failed_reasons,
        comparisons=comparisons,
    )


def write_decision_json(decision: PromotionDecision, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_decision_html(decision: PromotionDecision, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in decision.comparisons:
        css = "pass" if item.passed else "fail"
        rows.append(
            "<tr>"
            f"<td class=\"{css}\">{'PASS' if item.passed else 'FAIL'}</td>"
            f"<td>{html.escape(item.context)}</td>"
            f"<td>{html.escape(item.split)}</td>"
            f"<td>{item.baseline.correct}</td>"
            f"<td>{item.candidate.correct}</td>"
            f"<td>{item.correct_delta}</td>"
            f"<td>{item.baseline.wrong}</td>"
            f"<td>{item.candidate.wrong}</td>"
            f"<td>{item.wrong_delta}</td>"
            f"<td>{item.baseline.precision:.6f}</td>"
            f"<td>{item.candidate.precision:.6f}</td>"
            f"<td>{item.precision_delta:.6f}</td>"
            f"<td>{html.escape(item.reason)}</td>"
            "</tr>"
        )
    verdict = "PROMOTE" if decision.promote else "REJECT"
    verdict_class = "pass" if decision.promote else "fail"
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                "<head>",
                '  <meta charset="utf-8">',
                "  <title>checkpoint promotion decision</title>",
                "  <style>",
                "    body { font-family: sans-serif; margin: 24px; line-height: 1.5; }",
                "    table { border-collapse: collapse; margin-top: 12px; }",
                "    th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: right; }",
                "    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:last-child, td:last-child { text-align: left; }",
                "    .pass { color: #126b27; font-weight: 600; }",
                "    .fail { color: #a40000; font-weight: 600; }",
                "  </style>",
                "</head>",
                "<body>",
                "  <h1>Checkpoint Promotion Decision</h1>",
                f"  <p>Verdict: <span class=\"{verdict_class}\">{verdict}</span></p>",
                "  <table>",
                "    <tr><th>Status</th><th>Context</th><th>Split</th><th>Base correct</th><th>Cand correct</th><th>Δ correct</th><th>Base wrong</th><th>Cand wrong</th><th>Δ wrong</th><th>Base precision</th><th>Cand precision</th><th>Δ precision</th><th>Reason</th></tr>",
                *rows,
                "  </table>",
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("at least one split is required")
    return splits


def _parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--formal-variant-summary", type=Path, default=None)
    parser.add_argument("--guard-summary", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--guard-baseline-label", default=None)
    parser.add_argument("--guard-candidate-label", default=None)
    parser.add_argument("--splits", type=_parse_splits, default=["val", "test"])
    parser.add_argument("--formal-identity-column", default="label")
    parser.add_argument("--formal-variant-identity-column", default="label")
    parser.add_argument("--guard-identity-column", default="model")
    parser.add_argument("--formal-target-variants", type=_parse_csv_values, default=[])
    parser.add_argument("--formal-protected-variants", type=_parse_csv_values, default=[])
    parser.add_argument("--max-formal-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-formal-correct-drop", type=int, default=0)
    parser.add_argument("--max-formal-wrong-increase", type=int, default=0)
    parser.add_argument("--min-formal-target-correct-gain", type=int, default=0)
    parser.add_argument("--min-formal-target-total-correct-gain", type=int, default=0)
    parser.add_argument("--max-formal-target-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-formal-target-wrong-increase", type=int, default=0)
    parser.add_argument("--max-protected-variant-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-protected-variant-correct-drop", type=int, default=0)
    parser.add_argument("--max-protected-variant-wrong-increase", type=int, default=0)
    parser.add_argument("--max-guard-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-guard-correct-drop", type=int, default=0)
    parser.add_argument("--max-guard-wrong-increase", type=int, default=0)
    parser.add_argument(
        "--extra-regression-guard-set",
        action="append",
        default=[],
        help="Additional guard set names. Defaults to strict regression thresholds unless --max-extra-guard-* is set.",
    )
    parser.add_argument("--max-extra-guard-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-extra-guard-correct-drop", type=int, default=0)
    parser.add_argument("--max-extra-guard-wrong-increase", type=int, default=0)
    parser.add_argument("--min-extreme-correct-gain", type=int, default=1)
    parser.add_argument("--max-extreme-precision-drop", type=float, default=0.02)
    parser.add_argument("--max-extreme-wrong-increase", type=int, default=10**12)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = evaluate_promotion(
        formal_summary=args.formal_summary,
        formal_variant_summary=args.formal_variant_summary,
        guard_summary=args.guard_summary,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        guard_baseline_label=args.guard_baseline_label,
        guard_candidate_label=args.guard_candidate_label,
        splits=args.splits,
        formal_identity_column=args.formal_identity_column,
        formal_variant_identity_column=args.formal_variant_identity_column,
        guard_identity_column=args.guard_identity_column,
        formal_target_variants=args.formal_target_variants,
        formal_protected_variants=args.formal_protected_variants,
        max_formal_precision_drop=args.max_formal_precision_drop,
        max_formal_correct_drop=args.max_formal_correct_drop,
        max_formal_wrong_increase=args.max_formal_wrong_increase,
        min_formal_target_correct_gain=args.min_formal_target_correct_gain,
        min_formal_target_total_correct_gain=args.min_formal_target_total_correct_gain,
        max_formal_target_precision_drop=args.max_formal_target_precision_drop,
        max_formal_target_wrong_increase=args.max_formal_target_wrong_increase,
        max_protected_variant_precision_drop=args.max_protected_variant_precision_drop,
        max_protected_variant_correct_drop=args.max_protected_variant_correct_drop,
        max_protected_variant_wrong_increase=args.max_protected_variant_wrong_increase,
        max_guard_precision_drop=args.max_guard_precision_drop,
        max_guard_correct_drop=args.max_guard_correct_drop,
        max_guard_wrong_increase=args.max_guard_wrong_increase,
        extra_regression_guard_sets=args.extra_regression_guard_set,
        max_extra_guard_precision_drop=args.max_extra_guard_precision_drop,
        max_extra_guard_correct_drop=args.max_extra_guard_correct_drop,
        max_extra_guard_wrong_increase=args.max_extra_guard_wrong_increase,
        min_extreme_correct_gain=args.min_extreme_correct_gain,
        max_extreme_precision_drop=args.max_extreme_precision_drop,
        max_extreme_wrong_increase=args.max_extreme_wrong_increase,
    )
    if args.output_json is not None:
        write_decision_json(decision, args.output_json)
    if args.output_html is not None:
        write_decision_html(decision, args.output_html)
    print("PROMOTE" if decision.promote else "REJECT")
    for reason in decision.failed_reasons or decision.passed_reasons:
        print(reason)
    return 0 if decision.promote else 2


if __name__ == "__main__":
    raise SystemExit(main())
