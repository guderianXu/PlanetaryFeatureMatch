#!/usr/bin/env python3
"""Agent13 stage8 sidecar: fixed-test hybrid fallback policy sweep.

This script evaluates external matcher fallbacks without training or modifying
the pure learned PFM route.  It asks whether the classical fallback observed in
stage7 should remain limited to timestamp/compound pairs dropped by the
target-contrast gate, or whether a broader per-style/per-gate policy is useful
across the six fixed-test groups.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE7_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent13_stage7.py"
DEFAULT_BASELINE_ROUTE = PROJECT_ROOT / "runs" / "cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234"
DEFAULT_GATE_ROUTE = (
    PROJECT_ROOT
    / "runs"
    / "cross_view_1024_keypointonly_multistate_stylespecific_guard_targetcontrast_postselect_0step_seed1234"
)
DEFAULT_STAGE7_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage7"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage8"
GROUPS = [(style, gate) for style in ("numeric", "timestamp") for gate in ("rotate", "viewpoint", "compound")]

PAIR_FIELDS = [
    "style",
    "gate",
    "case_type",
    "pair_pt",
    "cache_pair_pt",
    "source_name",
    "pair_name",
    "baseline_matches",
    "baseline_correct",
    "baseline_wrong",
    "baseline_precision",
    "gate_matches",
    "gate_correct",
    "gate_wrong",
    "gate_precision",
    "lost_correct",
    "route_eligible",
]

METRIC_FIELDS = [
    *PAIR_FIELDS[:7],
    "algorithm",
    "family",
    "status",
    *PAIR_FIELDS[7:],
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "matches",
    "correct",
    "wrong",
    "precision",
    "mean_error_px",
    "median_error_px",
    "homography_threshold_px",
    "truth_threshold_px",
    "ratio",
    "min_inliers",
    "message",
]

SUMMARY_FIELDS = [
    "scope",
    "style",
    "gate",
    "algorithm",
    "family",
    "candidate_pairs",
    "route_eligible_pairs",
    "lost_correct_pairs",
    "ok_pairs",
    "covered_pairs",
    "coverage",
    "matches",
    "correct",
    "wrong",
    "precision",
    "gate_matches",
    "gate_correct",
    "gate_wrong",
    "gate_precision",
    "combined_matches",
    "combined_correct",
    "combined_wrong",
    "combined_precision",
    "combined_precision_delta",
    "combined_correct_delta",
    "recommend",
    "reason",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S7 = load_module(STAGE7_SCRIPT, "agent13_stage7_for_stage8")


@dataclass(frozen=True)
class PairRow:
    style: str
    gate: str
    case_type: str
    pair_pt: str
    cache_pair_pt: str
    source_name: str
    pair_name: str
    baseline_matches: int
    baseline_correct: int
    baseline_wrong: int
    baseline_precision: float
    gate_matches: int
    gate_correct: int
    gate_wrong: int
    gate_precision: float
    lost_correct: int
    route_eligible: int


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    case_type: str
    pair_pt: str
    cache_pair_pt: str
    source_name: str
    pair_name: str
    algorithm: str
    family: str
    status: str
    baseline_matches: int
    baseline_correct: int
    baseline_wrong: int
    baseline_precision: float
    gate_matches: int
    gate_correct: int
    gate_wrong: int
    gate_precision: float
    lost_correct: int
    route_eligible: int
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    matches: int
    correct: int
    wrong: int
    precision: float
    mean_error_px: float
    median_error_px: float
    homography_threshold_px: float
    truth_threshold_px: float
    ratio: float
    min_inliers: int
    message: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def format_value(value: object) -> object:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return value


def as_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, object], key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def precision(correct: int, matches: int) -> float:
    return correct / matches if matches else 0.0


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def group_summary(route: Path, style: str, gate: str) -> Path:
    return route / "eval" / style / gate / "summary.csv"


def cache_pair_path(pair_pt: str, gate: str) -> Path:
    path = Path(pair_pt)
    root = {"rotate": "Rotate_1024", "viewpoint": "Viewpoint_1024", "compound": "CompoundViewpoint_1024"}[gate]
    cache_path = PROJECT_ROOT / "img" / root / path.parent.name / path.name
    if cache_path.exists():
        return cache_path
    if path.exists():
        return path
    return PROJECT_ROOT / pair_pt


def load_group_pairs(args: argparse.Namespace) -> tuple[dict[tuple[str, str], list[dict[str, str]]], dict[tuple[str, str], list[dict[str, str]]], list[PairRow]]:
    baseline_by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    gate_by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    candidates: list[PairRow] = []
    for style, gate in GROUPS:
        baseline_rows = read_csv(group_summary(args.baseline_route, style, gate))
        gate_rows = read_csv(group_summary(args.gate_route, style, gate))
        gate_by_pair = {row["pair_pt"]: row for row in gate_rows}
        baseline_by_group[(style, gate)] = baseline_rows
        gate_by_group[(style, gate)] = gate_rows
        for base in baseline_rows:
            pair_pt = base["pair_pt"]
            target = gate_by_pair.get(pair_pt, {})
            baseline_matches = as_int(base, "matches")
            baseline_correct = as_int(base, "correct")
            gate_matches = as_int(target, "matches")
            gate_correct = as_int(target, "correct")
            if gate_matches > 0:
                continue
            lost_correct = max(0, baseline_correct - gate_correct)
            route_eligible = int(baseline_matches > 0)
            if route_eligible and lost_correct > 0:
                case_type = "dropped_lost_correct"
            elif route_eligible:
                case_type = "dropped_wrong_only"
            else:
                case_type = "gate_zero_baseline_zero"
            cache_path = cache_pair_path(pair_pt, gate)
            candidates.append(
                PairRow(
                    style=style,
                    gate=gate,
                    case_type=case_type,
                    pair_pt=pair_pt,
                    cache_pair_pt=rel(cache_path),
                    source_name=cache_path.parent.name,
                    pair_name=cache_path.name,
                    baseline_matches=baseline_matches,
                    baseline_correct=baseline_correct,
                    baseline_wrong=as_int(base, "wrong"),
                    baseline_precision=as_float(base, "precision"),
                    gate_matches=gate_matches,
                    gate_correct=gate_correct,
                    gate_wrong=as_int(target, "wrong"),
                    gate_precision=as_float(target, "precision"),
                    lost_correct=lost_correct,
                    route_eligible=route_eligible,
                )
            )
    return baseline_by_group, gate_by_group, candidates


def local_lightglue_reason(args: argparse.Namespace) -> str:
    if args.no_lightglue:
        return "disabled by --no-lightglue"
    if importlib.util.find_spec("lightglue") is None:
        return "module 'lightglue' unavailable in this environment"
    checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "sift_lightglue_v0-1_arxiv.pth"
    if not checkpoint.exists():
        return f"checkpoint missing: {checkpoint}"
    return ""


def make_algorithms(args: argparse.Namespace) -> tuple[list[object], list[dict[str, str]]]:
    algorithms, skipped = S7.make_algorithms(args)
    reason = local_lightglue_reason(args)
    if reason and not any(row["algorithm"] == "LightGlue-SIFT+HomographyUSAC-t3" for row in skipped):
        skipped.append({"algorithm": "LightGlue-SIFT+HomographyUSAC-t3", "reason": reason})
    return algorithms, skipped


def pair_to_stage7_target(pair: PairRow) -> dict[str, object]:
    row = asdict(pair)
    row["split"] = "fixed_test"
    return row


def evaluate_one(args: argparse.Namespace, algorithm: object, pair: PairRow) -> MetricRow:
    stage7_row = S7.evaluate_one(args, algorithm, pair_to_stage7_target(pair))
    return MetricRow(
        style=pair.style,
        gate=pair.gate,
        case_type=pair.case_type,
        pair_pt=pair.pair_pt,
        cache_pair_pt=pair.cache_pair_pt,
        source_name=pair.source_name,
        pair_name=pair.pair_name,
        algorithm=stage7_row.algorithm,
        family=stage7_row.family,
        status=stage7_row.status,
        baseline_matches=pair.baseline_matches,
        baseline_correct=pair.baseline_correct,
        baseline_wrong=pair.baseline_wrong,
        baseline_precision=pair.baseline_precision,
        gate_matches=pair.gate_matches,
        gate_correct=pair.gate_correct,
        gate_wrong=pair.gate_wrong,
        gate_precision=pair.gate_precision,
        lost_correct=pair.lost_correct,
        route_eligible=pair.route_eligible,
        keypoints_a=stage7_row.keypoints_a,
        keypoints_b=stage7_row.keypoints_b,
        raw_matches=stage7_row.raw_matches,
        matches=stage7_row.matches,
        correct=stage7_row.correct,
        wrong=stage7_row.wrong,
        precision=stage7_row.precision,
        mean_error_px=stage7_row.mean_error_px,
        median_error_px=stage7_row.median_error_px,
        homography_threshold_px=stage7_row.homography_threshold_px,
        truth_threshold_px=stage7_row.truth_threshold_px,
        ratio=stage7_row.ratio,
        min_inliers=stage7_row.min_inliers,
        message=stage7_row.message,
    )


def aggregate(rows: list[dict[str, object]] | list[MetricRow]) -> dict[str, int | float]:
    matches = sum(as_int(row if isinstance(row, dict) else asdict(row), "matches") for row in rows)
    correct = sum(as_int(row if isinstance(row, dict) else asdict(row), "correct") for row in rows)
    wrong = sum(as_int(row if isinstance(row, dict) else asdict(row), "wrong") for row in rows)
    return {"matches": matches, "correct": correct, "wrong": wrong, "precision": precision(correct, matches)}


def policy_candidates(pairs: list[PairRow], scope: str) -> list[PairRow]:
    if scope == "all_gate_zero":
        return pairs
    if scope == "dropped_only":
        return [pair for pair in pairs if pair.route_eligible]
    if scope == "timestamp_compound_dropped_only":
        return [pair for pair in pairs if pair.style == "timestamp" and pair.gate == "compound" and pair.route_eligible]
    raise ValueError(scope)


def summarize(
    *,
    gate_by_group: dict[tuple[str, str], list[dict[str, str]]],
    pairs: list[PairRow],
    metrics: list[MetricRow],
    algorithms: list[object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes = ("timestamp_compound_dropped_only", "dropped_only", "all_gate_zero")
    for scope in scopes:
        for style, gate in GROUPS:
            group_pairs = [pair for pair in pairs if pair.style == style and pair.gate == gate]
            gate_totals = aggregate(gate_by_group[(style, gate)])
            for algorithm in algorithms:
                selected_pairs = policy_candidates(group_pairs, scope)
                selected_ids = {pair.pair_pt for pair in selected_pairs}
                subset = [
                    row
                    for row in metrics
                    if row.algorithm == algorithm.name and row.style == style and row.gate == gate and row.pair_pt in selected_ids
                ]
                matches = sum(row.matches for row in subset)
                correct = sum(row.correct for row in subset)
                wrong = sum(row.wrong for row in subset)
                combined_matches = int(gate_totals["matches"]) + matches
                combined_correct = int(gate_totals["correct"]) + correct
                combined_wrong = int(gate_totals["wrong"]) + wrong
                combined_precision = precision(combined_correct, combined_matches)
                gate_precision = float(gate_totals["precision"])
                fallback_precision = precision(correct, matches)
                precision_delta = combined_precision - gate_precision
                route_pairs = sum(pair.route_eligible for pair in selected_pairs)
                covered = sum(1 for row in subset if row.matches > 0)
                recommend = int(correct > 0 and precision_delta >= 0.0 and wrong <= correct)
                if not selected_pairs:
                    reason = "no pairs selected by this policy"
                elif correct <= 0:
                    reason = "no correct fallback matches"
                elif precision_delta < 0.0:
                    reason = "recovers correct matches but lowers group precision"
                elif wrong > correct:
                    reason = "precision risk: wrong fallback matches exceed correct fallback matches"
                else:
                    reason = "recovers correct matches without lowering group precision"
                rows.append(
                    {
                        "scope": scope,
                        "style": style,
                        "gate": gate,
                        "algorithm": algorithm.name,
                        "family": algorithm.family,
                        "candidate_pairs": len(selected_pairs),
                        "route_eligible_pairs": route_pairs,
                        "lost_correct_pairs": sum(1 for pair in selected_pairs if pair.lost_correct > 0),
                        "ok_pairs": sum(1 for row in subset if row.status == "ok"),
                        "covered_pairs": covered,
                        "coverage": covered / len(selected_pairs) if selected_pairs else 0.0,
                        "matches": matches,
                        "correct": correct,
                        "wrong": wrong,
                        "precision": fallback_precision,
                        "gate_matches": gate_totals["matches"],
                        "gate_correct": gate_totals["correct"],
                        "gate_wrong": gate_totals["wrong"],
                        "gate_precision": gate_precision,
                        "combined_matches": combined_matches,
                        "combined_correct": combined_correct,
                        "combined_wrong": combined_wrong,
                        "combined_precision": combined_precision,
                        "combined_precision_delta": precision_delta,
                        "combined_correct_delta": correct,
                        "recommend": recommend,
                        "reason": reason,
                    }
                )
        for algorithm in algorithms:
            scope_rows = [row for row in rows if row["scope"] == scope and row["algorithm"] == algorithm.name]
            matches = sum(as_int(row, "matches") for row in scope_rows)
            correct = sum(as_int(row, "correct") for row in scope_rows)
            wrong = sum(as_int(row, "wrong") for row in scope_rows)
            gate_matches = sum(as_int(row, "gate_matches") for row in scope_rows)
            gate_correct = sum(as_int(row, "gate_correct") for row in scope_rows)
            combined_matches = sum(as_int(row, "combined_matches") for row in scope_rows)
            combined_correct = sum(as_int(row, "combined_correct") for row in scope_rows)
            good_groups = sum(as_int(row, "recommend") for row in scope_rows)
            rows.append(
                {
                    "scope": scope,
                    "style": "overall",
                    "gate": "all",
                    "algorithm": algorithm.name,
                    "family": algorithm.family,
                    "candidate_pairs": sum(as_int(row, "candidate_pairs") for row in scope_rows),
                    "route_eligible_pairs": sum(as_int(row, "route_eligible_pairs") for row in scope_rows),
                    "lost_correct_pairs": sum(as_int(row, "lost_correct_pairs") for row in scope_rows),
                    "ok_pairs": sum(as_int(row, "ok_pairs") for row in scope_rows),
                    "covered_pairs": sum(as_int(row, "covered_pairs") for row in scope_rows),
                    "coverage": (
                        sum(as_int(row, "covered_pairs") for row in scope_rows)
                        / sum(as_int(row, "candidate_pairs") for row in scope_rows)
                    )
                    if sum(as_int(row, "candidate_pairs") for row in scope_rows)
                    else 0.0,
                    "matches": matches,
                    "correct": correct,
                    "wrong": wrong,
                    "precision": precision(correct, matches),
                    "gate_matches": gate_matches,
                    "gate_correct": gate_correct,
                    "gate_wrong": sum(as_int(row, "gate_wrong") for row in scope_rows),
                    "gate_precision": precision(gate_correct, gate_matches),
                    "combined_matches": combined_matches,
                    "combined_correct": combined_correct,
                    "combined_wrong": sum(as_int(row, "combined_wrong") for row in scope_rows),
                    "combined_precision": precision(combined_correct, combined_matches),
                    "combined_precision_delta": precision(combined_correct, combined_matches) - precision(gate_correct, gate_matches),
                    "combined_correct_delta": correct,
                    "recommend": int(correct > 0 and good_groups > 0 and precision(combined_correct, combined_matches) >= precision(gate_correct, gate_matches)),
                    "reason": f"{good_groups}/6 groups pass per-group checks",
                }
            )
    return rows


def markdown_summary(
    *,
    args: argparse.Namespace,
    pairs: list[PairRow],
    summaries: list[dict[str, object]],
    skipped: list[dict[str, str]],
) -> str:
    lines = [
        "# Matcher Algorithm Iteration Agent13 Stage8",
        "",
        "## Scope",
        "",
        "- This is a hybrid/external matcher policy sweep, not a pure learned PFM route.",
        "- Fixed-test only; no training and no main evaluator/source edits.",
        "- Candidate set is all fixed-test pairs where the targetcontrast route emitted zero matches.",
        f"- Truth threshold: `{args.truth_threshold_px}` px; min homography inliers: `{args.min_inliers}`.",
        "",
        "## Candidate Pairs",
        "",
        "| style | gate | gate-zero pairs | dropped baseline-match pairs | lost-correct dropped pairs |",
        "|---|---|---:|---:|---:|",
    ]
    for style, gate in GROUPS:
        subset = [pair for pair in pairs if pair.style == style and pair.gate == gate]
        lines.append(
            f"| {style} | {gate} | {len(subset)} | {sum(pair.route_eligible for pair in subset)} | "
            f"{sum(1 for pair in subset if pair.lost_correct > 0)} |"
        )
    lines.extend(
        [
            "",
            "## Policy Summary",
            "",
            "| scope | style | gate | algorithm | pairs | matches | correct | wrong | fallback precision | combined precision | delta | recommend |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        if row["style"] != "overall" and not as_int(row, "recommend") and row["scope"] != "all_gate_zero":
            continue
        lines.append(
            f"| {row['scope']} | {row['style']} | {row['gate']} | {row['algorithm']} | "
            f"{row['candidate_pairs']} | {row['matches']} | {row['correct']} | {row['wrong']} | "
            f"{as_float(row, 'precision'):.6f} | {as_float(row, 'combined_precision'):.6f} | "
            f"{as_float(row, 'combined_precision_delta'):.6f} | {row['recommend']} |"
        )
    lines.extend(["", "## Recommendation", ""])
    scoped = [
        row
        for row in summaries
        if row["scope"] == "timestamp_compound_dropped_only"
        and row["style"] == "overall"
        and row["family"] == "classical"
        and as_int(row, "recommend")
    ]
    broad = [
        row
        for row in summaries
        if row["scope"] == "all_gate_zero" and row["style"] == "overall" and row["family"] == "classical" and as_int(row, "recommend")
    ]
    if broad:
        best_broad = max(broad, key=lambda row: (as_float(row, "precision"), as_float(row, "combined_precision"), -as_int(row, "wrong")))
        lines.append(
            f"Promote a broader fixed-test validation candidate: apply the hybrid/external `{best_broad['algorithm']}` fallback "
            "to all targetcontrast gate-zero pairs, not only timestamp/compound dropped pairs. It passed all six fixed-test groups "
            f"and added {best_broad['correct']} correct matches with {best_broad['wrong']} wrong fallback matches "
            f"(fallback precision {as_float(best_broad, 'precision'):.6f})."
        )
        lines.append(
            "Do not merge this as a pure learned PFM improvement; route it as a separately labeled hybrid/external policy and run "
            "the same policy on held-out/full-val artifacts before replacing the narrower timestamp/compound rescue."
        )
    elif scoped:
        best = max(scoped, key=lambda row: (as_float(row, "precision"), as_float(row, "combined_precision"), -as_int(row, "wrong")))
        lines.append(
            f"Keep the hybrid/external fallback scoped to timestamp/compound dropped pairs. `{best['algorithm']}` "
            f"recovers {best['correct']} correct matches with {best['wrong']} wrong fallback matches under that policy."
        )
    else:
        lines.append("Do not add a hybrid/external fallback from this sweep; no scoped classical policy passed the checks.")
    if not broad:
        lines.append("Do not broaden to a general all-gate-zero fallback across all six groups; aggregate or per-group precision risk remains.")
    if skipped:
        lines.extend(["", "## Skipped", "", "| algorithm | reason |", "|---|---|"])
        for row in skipped:
            lines.append(f"| {row['algorithm']} | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `per_pair_metrics.csv`",
            "- `per_group_policy_summary.csv`",
            "- `candidate_pairs.csv`",
            "- `summary.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_by_group, gate_by_group, pairs = load_group_pairs(args)
    algorithms, skipped = make_algorithms(args)
    metrics: list[MetricRow] = []
    for algorithm in algorithms:
        for index, pair in enumerate(pairs, start=1):
            row = evaluate_one(args, algorithm, pair)
            metrics.append(row)
            if args.verbose:
                print(
                    f"{algorithm.name:48s} {index:03d}/{len(pairs):03d} {pair.style}/{pair.gate:9s} "
                    f"{pair.source_name}/{pair.pair_name} m={row.matches} c={row.correct} p={row.precision:.3f}"
                )
    summaries = summarize(gate_by_group=gate_by_group, pairs=pairs, metrics=metrics, algorithms=algorithms)
    write_csv(args.output_dir / "candidate_pairs.csv", [asdict(pair) for pair in pairs], PAIR_FIELDS)
    write_csv(args.output_dir / "per_pair_metrics.csv", [asdict(row) for row in metrics], METRIC_FIELDS)
    write_csv(args.output_dir / "per_group_policy_summary.csv", summaries, SUMMARY_FIELDS)
    (args.output_dir / "summary.md").write_text(
        markdown_summary(args=args, pairs=pairs, summaries=summaries, skipped=skipped),
        encoding="utf-8",
    )
    _ = baseline_by_group


def self_test() -> None:
    assert precision(1, 2) == 0.5
    rows = [
        PairRow("timestamp", "compound", "dropped_lost_correct", "a.pt", "a.pt", "s", "a.pt", 1, 1, 0, 1.0, 0, 0, 0, 0.0, 1, 1),
        PairRow("timestamp", "compound", "gate_zero_baseline_zero", "b.pt", "b.pt", "s", "b.pt", 0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0),
    ]
    assert [row.pair_pt for row in policy_candidates(rows, "dropped_only")] == ["a.pt"]
    assert len(policy_candidates(rows, "all_gate_zero")) == 2
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-route", type=Path, default=DEFAULT_BASELINE_ROUTE)
    parser.add_argument("--gate-route", type=Path, default=DEFAULT_GATE_ROUTE)
    parser.add_argument("--stage7-dir", type=Path, default=DEFAULT_STAGE7_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--learned-max-keypoints", type=int, default=2048)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--truth-threshold-px", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-lightglue", action="store_true")
    parser.add_argument("--limit-algorithms", nargs="*", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
