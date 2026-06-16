#!/usr/bin/env python3
"""Compare per-match detail CSVs from lazy visual reports."""

from __future__ import annotations

import argparse
import csv
import html
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MatchDetail:
    label: str
    pair_index: int
    base_id: str
    reference_variant: str
    target_variant: str
    split: str
    match_index: int
    score: float
    error_px: float
    correct: bool

    @property
    def pair_key(self) -> tuple[str, int, str, str, str]:
        return (self.split, self.pair_index, self.base_id, self.reference_variant, self.target_variant)


@dataclass(frozen=True)
class ModelSummary:
    label: str
    matches: int
    correct: int
    wrong: int
    precision: float
    score_mean: float
    correct_score_mean: float
    wrong_score_mean: float


@dataclass(frozen=True)
class PairComparison:
    split: str
    pair_index: int
    base_id: str
    reference_variant: str
    target_variant: str
    baseline_matches: int
    baseline_correct: int
    baseline_wrong: int
    baseline_precision: float
    candidate_matches: int
    candidate_correct: int
    candidate_wrong: int
    candidate_precision: float
    match_delta: int
    correct_delta: int
    wrong_delta: int
    precision_delta: float


@dataclass(frozen=True)
class ThresholdSummary:
    label: str
    threshold: float
    matches: int
    correct: int
    wrong: int
    precision: float


@dataclass(frozen=True)
class MatchDetailComparison:
    overall: list[ModelSummary]
    per_pair: list[PairComparison]
    threshold_sweep: list[ThresholdSummary]


def _parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def read_match_details(path: Path | str, *, fallback_label: str = "") -> list[MatchDetail]:
    csv_path = Path(path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    details: list[MatchDetail] = []
    for row in rows:
        details.append(
            MatchDetail(
                label=row.get("label") or fallback_label,
                pair_index=_parse_int(row.get("pair_index"), default=-1),
                base_id=row.get("base_id") or "",
                reference_variant=row.get("reference_variant") or row.get("variant_a") or "",
                target_variant=row.get("target_variant") or row.get("variant_b") or "",
                split=row.get("split") or "",
                match_index=_parse_int(row.get("match_index"), default=-1),
                score=_parse_float(row.get("score")),
                error_px=_parse_float(row.get("error_px")),
                correct=_parse_bool(row.get("correct")),
            )
        )
    return details


def _precision(correct: int, matches: int) -> float:
    if matches <= 0:
        return 0.0
    return correct / matches


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return statistics.fmean(collected)


def summarize_model(rows: Sequence[MatchDetail], label: str) -> ModelSummary:
    matches = len(rows)
    correct = sum(1 for row in rows if row.correct)
    wrong = matches - correct
    return ModelSummary(
        label=label,
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=_precision(correct, matches),
        score_mean=_mean(row.score for row in rows),
        correct_score_mean=_mean(row.score for row in rows if row.correct),
        wrong_score_mean=_mean(row.score for row in rows if not row.correct),
    )


def _group_by_pair(rows: Sequence[MatchDetail]) -> dict[tuple[str, int, str, str, str], list[MatchDetail]]:
    grouped: dict[tuple[str, int, str, str, str], list[MatchDetail]] = {}
    for row in rows:
        grouped.setdefault(row.pair_key, []).append(row)
    return grouped


def compare_pairs(baseline_rows: Sequence[MatchDetail], candidate_rows: Sequence[MatchDetail]) -> list[PairComparison]:
    baseline_by_pair = _group_by_pair(baseline_rows)
    candidate_by_pair = _group_by_pair(candidate_rows)
    pair_keys = sorted(set(baseline_by_pair) | set(candidate_by_pair), key=lambda key: (key[0], key[1], key[2], key[3], key[4]))
    comparisons: list[PairComparison] = []
    for split, pair_index, base_id, reference_variant, target_variant in pair_keys:
        baseline = baseline_by_pair.get((split, pair_index, base_id, reference_variant, target_variant), [])
        candidate = candidate_by_pair.get((split, pair_index, base_id, reference_variant, target_variant), [])
        baseline_matches = len(baseline)
        baseline_correct = sum(1 for row in baseline if row.correct)
        baseline_wrong = baseline_matches - baseline_correct
        baseline_precision = _precision(baseline_correct, baseline_matches)
        candidate_matches = len(candidate)
        candidate_correct = sum(1 for row in candidate if row.correct)
        candidate_wrong = candidate_matches - candidate_correct
        candidate_precision = _precision(candidate_correct, candidate_matches)
        comparisons.append(
            PairComparison(
                split=split,
                pair_index=pair_index,
                base_id=base_id,
                reference_variant=reference_variant,
                target_variant=target_variant,
                baseline_matches=baseline_matches,
                baseline_correct=baseline_correct,
                baseline_wrong=baseline_wrong,
                baseline_precision=baseline_precision,
                candidate_matches=candidate_matches,
                candidate_correct=candidate_correct,
                candidate_wrong=candidate_wrong,
                candidate_precision=candidate_precision,
                match_delta=candidate_matches - baseline_matches,
                correct_delta=candidate_correct - baseline_correct,
                wrong_delta=candidate_wrong - baseline_wrong,
                precision_delta=candidate_precision - baseline_precision,
            )
        )
    return comparisons


def sweep_score_thresholds(rows: Sequence[MatchDetail], *, label: str, thresholds: Sequence[float]) -> list[ThresholdSummary]:
    summaries: list[ThresholdSummary] = []
    for threshold in thresholds:
        kept = [row for row in rows if row.score >= threshold]
        matches = len(kept)
        correct = sum(1 for row in kept if row.correct)
        wrong = matches - correct
        summaries.append(
            ThresholdSummary(
                label=label,
                threshold=threshold,
                matches=matches,
                correct=correct,
                wrong=wrong,
                precision=_precision(correct, matches),
            )
        )
    return summaries


def compare_match_detail_reports(
    baseline_details: Path | str,
    candidate_details: Path | str,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    score_thresholds: Sequence[float] = (),
) -> MatchDetailComparison:
    baseline_rows = read_match_details(baseline_details, fallback_label=baseline_label)
    candidate_rows = read_match_details(candidate_details, fallback_label=candidate_label)
    thresholds = list(score_thresholds) if score_thresholds else [0.0]
    return MatchDetailComparison(
        overall=[
            summarize_model(baseline_rows, baseline_label),
            summarize_model(candidate_rows, candidate_label),
        ],
        per_pair=compare_pairs(baseline_rows, candidate_rows),
        threshold_sweep=sweep_score_thresholds(candidate_rows, label=candidate_label, thresholds=thresholds),
    )


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())  # type: ignore[attr-defined]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def write_comparison_report(comparison: MatchDetailComparison, output_dir: Path | str) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(root / "overall_summary.csv", comparison.overall)
    _write_dataclass_csv(root / "pair_delta_summary.csv", comparison.per_pair)
    _write_dataclass_csv(root / "score_threshold_sweep.csv", comparison.threshold_sweep)
    (root / "index.html").write_text(render_html_report(comparison), encoding="utf-8")


def render_html_report(comparison: MatchDetailComparison) -> str:
    def fmt(value: float) -> str:
        return f"{value:.6f}"

    def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
        head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        body = []
        for row in rows:
            body.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(fmt(value) if isinstance(value, float) else str(value))}</td>"
                    for value in row
                )
                + "</tr>"
            )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    overall_rows = [
        (
            row.label,
            row.matches,
            row.correct,
            row.wrong,
            row.precision,
            row.score_mean,
            row.correct_score_mean,
            row.wrong_score_mean,
        )
        for row in comparison.overall
    ]
    pair_rows = [
        (
            row.split,
            row.pair_index,
            row.base_id,
            row.reference_variant,
            row.target_variant,
            row.match_delta,
            row.correct_delta,
            row.wrong_delta,
            row.precision_delta,
        )
        for row in comparison.per_pair
    ]
    threshold_rows = [
        (row.label, row.threshold, row.matches, row.correct, row.wrong, row.precision)
        for row in comparison.threshold_sweep
    ]
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Match Detail Comparison</title>
<style>
body { font-family: sans-serif; margin: 24px; color: #17202a; }
table { border-collapse: collapse; margin: 16px 0 28px; font-size: 13px; }
th, td { border: 1px solid #d5d8dc; padding: 6px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #eef2f5; }
h1, h2 { margin-bottom: 8px; }
</style>
</head>
<body>
<h1>Match Detail Comparison</h1>
<h2>Overall</h2>
""" + table(
        [
            "label",
            "matches",
            "correct",
            "wrong",
            "precision",
            "score_mean",
            "correct_score_mean",
            "wrong_score_mean",
        ],
        overall_rows,
    ) + """
<h2>Pair Deltas</h2>
""" + table(
        [
            "split",
            "pair_index",
            "base_id",
            "reference_variant",
            "target_variant",
            "match_delta",
            "correct_delta",
            "wrong_delta",
            "precision_delta",
        ],
        pair_rows,
    ) + """
<h2>Candidate Score Threshold Sweep</h2>
""" + table(["label", "threshold", "matches", "correct", "wrong", "precision"], threshold_rows) + """
</body>
</html>
"""


def parse_thresholds(value: str) -> list[float]:
    if not value.strip():
        return [0.0]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-details", type=Path, required=True, help="Baseline match detail CSV.")
    parser.add_argument("--candidate-details", type=Path, required=True, help="Candidate match detail CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV and HTML comparison outputs.")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--score-thresholds",
        default="0,2,4,6,8,10,12,14,16,18,20",
        help="Comma-separated candidate score thresholds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_match_detail_reports(
        args.baseline_details,
        args.candidate_details,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        score_thresholds=parse_thresholds(args.score_thresholds),
    )
    write_comparison_report(comparison, args.output_dir)
    candidate = comparison.overall[1]
    print(
        f"wrote {args.output_dir} "
        f"candidate_matches={candidate.matches} "
        f"candidate_correct={candidate.correct} "
        f"candidate_precision={candidate.precision:.6f}"
    )


if __name__ == "__main__":
    main()
