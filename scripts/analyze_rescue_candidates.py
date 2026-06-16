#!/usr/bin/env python3
"""Analyze low-match rescue candidates between two visual summary reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RescueCandidate:
    index: int
    split: str
    base_id: str
    target_variant: str
    baseline_matches: int
    baseline_correct: int
    baseline_wrong: int
    candidate_matches: int
    candidate_correct: int
    candidate_wrong: int
    candidate_precision: float
    score_min: float
    score_mean: float
    score_median: float
    score_max: float
    span_a_x_px: float
    span_a_y_px: float
    span_b_x_px: float
    span_b_y_px: float
    bbox_area_a_px2: float
    bbox_area_b_px2: float
    displacement_mad_px: float
    homography_residual_valid: int
    homography_residual_p90_px: float
    median_error_px: float
    valid_fraction: float


@dataclass(frozen=True)
class ScoreRuleSummary:
    score_min_threshold: float
    score_mean_threshold: float
    min_bbox_area_a_px2_threshold: float
    max_homography_residual_p90_px_threshold: float
    max_displacement_mad_px_threshold: float
    rows: int
    matches: int
    correct: int
    wrong: int
    precision: float


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for item in parse_csv_list(value):
        try:
            values.append(float(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid float threshold: {item}") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return values


def _as_int(row: dict[str, str], name: str) -> int:
    value = row.get(name, "")
    return int(float(value)) if value not in ("", None) else 0


def _as_float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in ("", None) else 0.0


def _identity(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("split", ""), row.get("base_id", ""), row.get("target_variant", ""))


def find_rescue_candidates(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    target_variants: tuple[str, ...],
    min_candidate_matches: int,
    max_candidate_matches: int,
) -> list[RescueCandidate]:
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError(
            f"baseline/candidate row count mismatch: {len(baseline_rows)} != {len(candidate_rows)}"
        )
    variant_set = set(target_variants)
    candidates: list[RescueCandidate] = []
    for index, (baseline, candidate) in enumerate(zip(baseline_rows, candidate_rows)):
        if _identity(baseline) != _identity(candidate):
            raise ValueError(
                "baseline/candidate rows are not aligned at "
                f"index={index}: baseline={_identity(baseline)} candidate={_identity(candidate)}"
            )
        variant = candidate.get("target_variant", "")
        candidate_matches = _as_int(candidate, "matches")
        if variant_set and variant not in variant_set:
            continue
        if _as_int(baseline, "matches") != 0:
            continue
        if candidate_matches < min_candidate_matches or candidate_matches > max_candidate_matches:
            continue
        candidates.append(
            RescueCandidate(
                index=index,
                split=candidate.get("split", ""),
                base_id=candidate.get("base_id", ""),
                target_variant=variant,
                baseline_matches=_as_int(baseline, "matches"),
                baseline_correct=_as_int(baseline, "correct"),
                baseline_wrong=_as_int(baseline, "wrong"),
                candidate_matches=candidate_matches,
                candidate_correct=_as_int(candidate, "correct"),
                candidate_wrong=_as_int(candidate, "wrong"),
                candidate_precision=_as_float(candidate, "precision"),
                score_min=_as_float(candidate, "score_min"),
                score_mean=_as_float(candidate, "score_mean"),
                score_median=_as_float(candidate, "score_median"),
                score_max=_as_float(candidate, "score_max"),
                span_a_x_px=_as_float(candidate, "span_a_x_px"),
                span_a_y_px=_as_float(candidate, "span_a_y_px"),
                span_b_x_px=_as_float(candidate, "span_b_x_px"),
                span_b_y_px=_as_float(candidate, "span_b_y_px"),
                bbox_area_a_px2=_as_float(candidate, "bbox_area_a_px2"),
                bbox_area_b_px2=_as_float(candidate, "bbox_area_b_px2"),
                displacement_mad_px=_as_float(candidate, "displacement_mad_px"),
                homography_residual_valid=_as_int(candidate, "homography_residual_valid"),
                homography_residual_p90_px=_as_float(candidate, "homography_residual_p90_px"),
                median_error_px=_as_float(candidate, "median_error_px"),
                valid_fraction=_as_float(candidate, "valid_fraction"),
            )
        )
    return candidates


def sweep_score_rules(
    candidates: list[RescueCandidate],
    *,
    score_min_thresholds: list[float],
    score_mean_thresholds: list[float],
    min_bbox_area_a_px2_thresholds: list[float] | None = None,
    max_homography_residual_p90_px_thresholds: list[float] | None = None,
    max_displacement_mad_px_thresholds: list[float] | None = None,
) -> list[ScoreRuleSummary]:
    summaries: list[ScoreRuleSummary] = []
    bbox_thresholds = min_bbox_area_a_px2_thresholds or [0.0]
    homography_thresholds = max_homography_residual_p90_px_thresholds or [float("inf")]
    displacement_thresholds = max_displacement_mad_px_thresholds or [float("inf")]
    for score_min_threshold in score_min_thresholds:
        for score_mean_threshold in score_mean_thresholds:
            for min_bbox_area in bbox_thresholds:
                for max_homography_p90 in homography_thresholds:
                    for max_displacement_mad in displacement_thresholds:
                        kept = [
                            item
                            for item in candidates
                            if item.score_min >= score_min_threshold
                            and item.score_mean >= score_mean_threshold
                            and item.bbox_area_a_px2 >= min_bbox_area
                            and item.homography_residual_valid > 0
                            and item.homography_residual_p90_px <= max_homography_p90
                            and item.displacement_mad_px <= max_displacement_mad
                        ]
                        matches = sum(item.candidate_matches for item in kept)
                        correct = sum(item.candidate_correct for item in kept)
                        wrong = sum(item.candidate_wrong for item in kept)
                        precision = 0.0 if matches <= 0 else float(correct) / float(matches)
                        summaries.append(
                            ScoreRuleSummary(
                                score_min_threshold=score_min_threshold,
                                score_mean_threshold=score_mean_threshold,
                                min_bbox_area_a_px2_threshold=min_bbox_area,
                                max_homography_residual_p90_px_threshold=max_homography_p90,
                                max_displacement_mad_px_threshold=max_displacement_mad,
                                rows=len(kept),
                                matches=matches,
                                correct=correct,
                                wrong=wrong,
                                precision=precision,
                            )
                )
    return summaries


def write_rescue_csv(candidates: list[RescueCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RescueCandidate.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow(asdict(item))


def write_sweep_csv(summaries: list[ScoreRuleSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ScoreRuleSummary.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow(asdict(item))


def _variant_summary(candidates: list[RescueCandidate]) -> list[dict[str, object]]:
    by_variant: dict[str, dict[str, int]] = {}
    for item in candidates:
        bucket = by_variant.setdefault(
            item.target_variant,
            {"rows": 0, "matches": 0, "correct": 0, "wrong": 0},
        )
        bucket["rows"] += 1
        bucket["matches"] += item.candidate_matches
        bucket["correct"] += item.candidate_correct
        bucket["wrong"] += item.candidate_wrong
    rows: list[dict[str, object]] = []
    for variant, stats in sorted(by_variant.items()):
        matches = int(stats["matches"])
        precision = 0.0 if matches <= 0 else float(stats["correct"]) / float(matches)
        rows.append({"variant": variant, **stats, "precision": precision})
    return rows


def write_summary_json(
    *,
    candidates: list[RescueCandidate],
    sweep: list[ScoreRuleSummary],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate_count": len(candidates),
        "candidate_matches": sum(item.candidate_matches for item in candidates),
        "candidate_correct": sum(item.candidate_correct for item in candidates),
        "candidate_wrong": sum(item.candidate_wrong for item in candidates),
        "variant_summary": _variant_summary(candidates),
        "best_score_rules": [
            asdict(item)
            for item in sorted(sweep, key=lambda x: (x.precision, x.correct, -x.wrong), reverse=True)[:10]
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_html_report(
    *,
    candidates: list[RescueCandidate],
    sweep: list[ScoreRuleSummary],
    path: Path,
    baseline_summary: Path,
    candidate_summary: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variant_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['variant']))}</td>"
        f"<td>{row['rows']}</td><td>{row['matches']}</td><td>{row['correct']}</td>"
        f"<td>{row['wrong']}</td><td>{float(row['precision']):.3f}</td>"
        "</tr>"
        for row in _variant_summary(candidates)
    )
    candidate_rows = "\n".join(
        "<tr>"
        f"<td>{item.index}</td><td>{html.escape(item.split)}</td>"
        f"<td>{html.escape(item.base_id)}</td><td>{html.escape(item.target_variant)}</td>"
        f"<td>{item.candidate_matches}</td><td>{item.candidate_correct}</td>"
        f"<td>{item.candidate_wrong}</td><td>{item.candidate_precision:.3f}</td>"
        f"<td>{item.score_min:.3f}</td><td>{item.score_mean:.3f}</td>"
        f"<td>{item.score_median:.3f}</td><td>{item.score_max:.3f}</td>"
        f"<td>{item.bbox_area_a_px2:.1f}</td><td>{item.bbox_area_b_px2:.1f}</td>"
        f"<td>{item.displacement_mad_px:.3f}</td><td>{item.homography_residual_p90_px:.3f}</td>"
        f"<td>{item.median_error_px:.3f}</td><td>{item.valid_fraction:.3f}</td>"
        "</tr>"
        for item in candidates
    )
    sweep_rows = "\n".join(
        "<tr>"
        f"<td>{item.score_min_threshold:g}</td><td>{item.score_mean_threshold:g}</td>"
        f"<td>{item.min_bbox_area_a_px2_threshold:g}</td>"
        f"<td>{item.max_homography_residual_p90_px_threshold:g}</td>"
        f"<td>{item.max_displacement_mad_px_threshold:g}</td>"
        f"<td>{item.rows}</td><td>{item.matches}</td><td>{item.correct}</td>"
        f"<td>{item.wrong}</td><td>{item.precision:.3f}</td>"
        "</tr>"
        for item in sorted(sweep, key=lambda x: (x.precision, x.correct, -x.wrong), reverse=True)[:50]
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rescue Candidate Analysis</title>
<style>
body {{ margin: 24px; font-family: Arial, sans-serif; background: #0f1720; color: #e5eef7; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #2f3d4c; padding: 7px; text-align: left; }}
pre {{ background: #162232; padding: 12px; white-space: pre-wrap; }}
.warn {{ color: #fbbf24; }}
</style>
</head>
<body>
<h1>Rescue Candidate Analysis</h1>
<pre>{html.escape(json.dumps({"baseline_summary": str(baseline_summary), "candidate_summary": str(candidate_summary)}, indent=2, ensure_ascii=False))}</pre>
<p class="warn">Precision and error columns are evaluation-only labels. Score and inference-geometry columns are model/post-filter outputs and may be used for post-filter diagnostics.</p>
<h2>Variant Summary</h2>
<table><thead><tr><th>Variant</th><th>Rows</th><th>Matches</th><th>Correct</th><th>Wrong</th><th>Precision</th></tr></thead><tbody>{variant_rows}</tbody></table>
<h2>Rescue Candidates</h2>
<table><thead><tr><th>Index</th><th>Split</th><th>Base</th><th>Variant</th><th>Matches</th><th>Correct</th><th>Wrong</th><th>Precision</th><th>Score Min</th><th>Score Mean</th><th>Score Median</th><th>Score Max</th><th>Area A</th><th>Area B</th><th>Disp MAD</th><th>H P90</th><th>Median Error</th><th>Valid</th></tr></thead><tbody>{candidate_rows}</tbody></table>
<h2>Score Rule Sweep</h2>
<table><thead><tr><th>Score Min >=</th><th>Score Mean >=</th><th>Area A >=</th><th>H P90 <=</th><th>Disp MAD <=</th><th>Rows</th><th>Matches</th><th>Correct</th><th>Wrong</th><th>Precision</th></tr></thead><tbody>{sweep_rows}</tbody></table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-variants", default="extreme_02,extreme_03")
    parser.add_argument("--min-candidate-matches", type=int, default=8)
    parser.add_argument("--max-candidate-matches", type=int, default=15)
    parser.add_argument("--score-min-thresholds", type=parse_float_list, default=parse_float_list("0,10,12,14,16,18,20,24,28"))
    parser.add_argument("--score-mean-thresholds", type=parse_float_list, default=parse_float_list("0,16,18,20,22,24,26,28"))
    parser.add_argument("--min-bbox-area-a-px2-thresholds", type=parse_float_list, default=parse_float_list("0"))
    parser.add_argument("--max-homography-residual-p90-px-thresholds", type=parse_float_list, default=parse_float_list("1,2,3,4,5,8,12,1000000000"))
    parser.add_argument("--max-displacement-mad-px-thresholds", type=parse_float_list, default=parse_float_list("2,4,8,16,1000000000"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_candidate_matches < 0:
        raise ValueError("--min-candidate-matches must be nonnegative")
    if args.max_candidate_matches < args.min_candidate_matches:
        raise ValueError("--max-candidate-matches must be >= --min-candidate-matches")
    candidates = find_rescue_candidates(
        read_csv_rows(args.baseline_summary),
        read_csv_rows(args.candidate_summary),
        target_variants=tuple(parse_csv_list(args.target_variants)),
        min_candidate_matches=args.min_candidate_matches,
        max_candidate_matches=args.max_candidate_matches,
    )
    sweep = sweep_score_rules(
        candidates,
        score_min_thresholds=args.score_min_thresholds,
        score_mean_thresholds=args.score_mean_thresholds,
        min_bbox_area_a_px2_thresholds=args.min_bbox_area_a_px2_thresholds,
        max_homography_residual_p90_px_thresholds=args.max_homography_residual_p90_px_thresholds,
        max_displacement_mad_px_thresholds=args.max_displacement_mad_px_thresholds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rescue_csv(candidates, args.output_dir / "rescue_candidates.csv")
    write_sweep_csv(sweep, args.output_dir / "score_rule_sweep.csv")
    write_summary_json(candidates=candidates, sweep=sweep, path=args.output_dir / "summary.json")
    write_html_report(
        candidates=candidates,
        sweep=sweep,
        path=args.output_dir / "index.html",
        baseline_summary=args.baseline_summary,
        candidate_summary=args.candidate_summary,
    )
    print(f"report={args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
