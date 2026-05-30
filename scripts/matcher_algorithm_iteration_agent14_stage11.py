#!/usr/bin/env python3
"""Agent14 stage11 sidecar: fixed-test strict RootSIFT hybrid route.

This script writes only Stage11 sidecar artifacts. It does not train PFM,
does not run the PFM evaluator, and does not modify the main training/eval
source. It reads the current fixed-test pure-PFM route, evaluates an external
RootSIFT fallback on pure-PFM zero-match rows, and compares that route with the
Stage8 r0.80/H2 compatibility baseline.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE7_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent13_stage7.py"
DEFAULT_PURE_ROUTE = (
    PROJECT_ROOT
    / "runs"
    / "cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234"
)
DEFAULT_STAGE8_ROUTE = PROJECT_ROOT / "runs" / "cross_view_1024_targetcontrast_rootsift_allgatezero_fallback_route_20260526"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent14_stage11"

GROUPS = [(style, gate) for style in ("numeric", "timestamp") for gate in ("rotate", "viewpoint", "compound")]
STRICT_ALGORITHM = "RootSIFT-FLANN-r0.75+HomographyUSAC-t2"
STAGE8_BASELINE_ALGORITHM = "RootSIFT-FLANN-r0.80+HomographyUSAC-t2"
STAGE11_ROUTE_NAME = "pure_pfm_fixed_test_plus_rootsift_r075_usac_t2_gate_zero_fallback"
STAGE8_ROUTE_NAME = "targetcontrast_rootsift_all_gate_zero_hybrid"

PAIR_FIELDS = [
    "style",
    "gate",
    "split",
    "case_type",
    "pair_pt",
    "source_name",
    "pair_name",
    "pure_matches",
    "pure_correct",
    "pure_wrong",
    "pure_precision",
    "sample_rank",
    "source_pool_pairs",
]

METRIC_FIELDS = [
    *PAIR_FIELDS[:11],
    "algorithm",
    "family",
    "status",
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

GROUP_FIELDS = [
    "scope",
    "style",
    "gate",
    "algorithm",
    "family",
    "pure_pairs",
    "candidate_pairs",
    "ok_pairs",
    "covered_pairs",
    "coverage",
    "fallback_matches",
    "fallback_correct",
    "fallback_wrong",
    "fallback_precision",
    "pure_matches",
    "pure_correct",
    "pure_wrong",
    "pure_precision",
    "hybrid_matches",
    "hybrid_correct",
    "hybrid_wrong",
    "hybrid_precision",
    "hybrid_precision_delta",
    "min_pair_precision",
    "mean_pair_precision",
    "median_pair_precision",
    "mean_matches_per_candidate",
    "min_pair_matches",
    "pairs_ge_20_inliers",
    "pairs_ge_50_inliers",
    "mean_error_px",
    "median_error_px",
    "homography_threshold_px",
    "truth_threshold_px",
    "ratio",
    "recommend",
    "reason",
]

HYBRID_FIELDS = [
    "style",
    "gate",
    "route",
    "algorithm",
    "pure_pairs",
    "gate_zero_pairs",
    "fallback_pairs",
    "pure_nonzero_matches",
    "pure_nonzero_correct",
    "pure_nonzero_wrong",
    "matches",
    "correct",
    "wrong",
    "precision",
    "delta_matches",
    "delta_correct",
    "delta_wrong",
    "delta_precision",
    "pure_pfm",
    "external_fallback",
    "notes",
]

COMPARISON_FIELDS = [
    "style",
    "gate",
    "stage11_route",
    "stage11_algorithm",
    "stage11_matches",
    "stage11_correct",
    "stage11_wrong",
    "stage11_precision",
    "stage11_gate_zero_pairs",
    "stage11_fallback_pairs",
    "stage11_fallback_matches",
    "stage11_fallback_correct",
    "stage11_fallback_wrong",
    "stage11_fallback_precision",
    "baseline_route",
    "baseline_algorithm",
    "baseline_matches",
    "baseline_correct",
    "baseline_wrong",
    "baseline_precision",
    "baseline_gate_zero_pairs",
    "baseline_fallback_pairs",
    "baseline_fallback_matches",
    "baseline_fallback_correct",
    "baseline_fallback_wrong",
    "baseline_fallback_precision",
    "delta_matches_vs_stage8",
    "delta_correct_vs_stage8",
    "delta_wrong_vs_stage8",
    "delta_precision_vs_stage8",
    "delta_coverage_vs_stage8",
    "delta_fallback_matches_vs_stage8",
    "delta_fallback_correct_vs_stage8",
    "delta_fallback_wrong_vs_stage8",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S7 = load_module(STAGE7_SCRIPT, "agent13_stage7_for_stage11")
A4 = S7.A4


@dataclass(frozen=True)
class PairRow:
    style: str
    gate: str
    split: str
    case_type: str
    pair_pt: str
    source_name: str
    pair_name: str
    pure_matches: int
    pure_correct: int
    pure_wrong: int
    pure_precision: float
    sample_rank: int
    source_pool_pairs: int


@dataclass(frozen=True)
class Algorithm:
    name: str
    family: str
    matcher: object
    homography_threshold_px: float
    ratio: float
    min_inliers: int


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    split: str
    case_type: str
    pair_pt: str
    source_name: str
    pair_name: str
    pure_matches: int
    pure_correct: int
    pure_wrong: int
    pure_precision: float
    algorithm: str
    family: str
    status: str
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


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.fmean(finite)) if finite else math.nan


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.median(finite)) if finite else math.nan


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def group_summary(route: Path, style: str, gate: str) -> Path:
    return route / "eval" / style / gate / "summary.csv"


def route_case_type(matches: int) -> str:
    return "pure_pfm_gate_zero_external_fallback_candidate" if matches == 0 else "pure_pfm_nonzero_not_replayed"


def aggregate_metric_dicts(rows: list[dict[str, object]]) -> dict[str, int | float]:
    matches = sum(as_int(row, "matches") for row in rows)
    correct = sum(as_int(row, "correct") for row in rows)
    wrong = sum(as_int(row, "wrong") for row in rows)
    return {"pairs": len(rows), "matches": matches, "correct": correct, "wrong": wrong, "precision": precision(correct, matches)}


def collect_candidate_pairs(args: argparse.Namespace) -> tuple[list[PairRow], dict[tuple[str, str], list[dict[str, str]]]]:
    candidates: list[PairRow] = []
    pure_by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    for style, gate in GROUPS:
        rows = read_csv(group_summary(args.pure_route, style, gate))
        pure_by_group[(style, gate)] = rows
        zero_rows = [row for row in rows if as_int(row, "matches") == 0]
        for rank, row in enumerate(zero_rows, start=1):
            pair_path = resolve_project_path(row["pair_pt"])
            candidates.append(
                PairRow(
                    style=style,
                    gate=gate,
                    split="fixed_test",
                    case_type=route_case_type(as_int(row, "matches")),
                    pair_pt=rel(pair_path),
                    source_name=pair_path.parent.name,
                    pair_name=pair_path.name,
                    pure_matches=as_int(row, "matches"),
                    pure_correct=as_int(row, "correct"),
                    pure_wrong=as_int(row, "wrong"),
                    pure_precision=as_float(row, "precision"),
                    sample_rank=rank,
                    source_pool_pairs=len(zero_rows),
                )
            )
    return candidates, pure_by_group


def make_algorithm(args: argparse.Namespace) -> tuple[Algorithm, list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    try:
        import cv2

        if not hasattr(cv2, "SIFT_create"):
            skipped.append({"algorithm": STRICT_ALGORITHM, "reason": "cv2.SIFT_create unavailable"})
    except Exception as exc:
        skipped.append({"algorithm": STRICT_ALGORITHM, "reason": f"{type(exc).__name__}: {exc}"})
    if skipped:
        raise RuntimeError("; ".join(f"{row['algorithm']}: {row['reason']}" for row in skipped))
    return (
        Algorithm(
            name=STRICT_ALGORITHM,
            family="classical",
            matcher=S7.RootSiftFlannMatcher(
                ratio=0.75,
                max_keypoints=args.max_keypoints,
                max_matches=args.max_matches,
                sift_contrast=args.sift_contrast,
            ),
            homography_threshold_px=2.0,
            ratio=0.75,
            min_inliers=args.min_inliers,
        ),
        skipped,
    )


def evaluate_one(args: argparse.Namespace, algorithm: Algorithm, pair: PairRow) -> MetricRow:
    try:
        image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(resolve_project_path(pair.pair_pt))
        raw = algorithm.matcher.match(image_a, image_b)
        output = S7.homography_inliers(raw, algorithm.homography_threshold_px, algorithm.min_inliers)
        matches, correct, wrong, pair_precision, mean_error, median_error = A4.compute_metrics(
            output.points_a,
            output.points_b,
            warp_a_to_b,
            valid_mask,
            threshold_px=args.truth_threshold_px,
        )
        return MetricRow(
            style=pair.style,
            gate=pair.gate,
            split=pair.split,
            case_type=pair.case_type,
            pair_pt=pair.pair_pt,
            source_name=pair.source_name,
            pair_name=pair.pair_name,
            pure_matches=pair.pure_matches,
            pure_correct=pair.pure_correct,
            pure_wrong=pair.pure_wrong,
            pure_precision=pair.pure_precision,
            algorithm=algorithm.name,
            family=algorithm.family,
            status="ok",
            keypoints_a=raw.keypoints_a,
            keypoints_b=raw.keypoints_b,
            raw_matches=raw.raw_matches,
            matches=matches,
            correct=correct,
            wrong=wrong,
            precision=pair_precision,
            mean_error_px=mean_error,
            median_error_px=median_error,
            homography_threshold_px=algorithm.homography_threshold_px,
            truth_threshold_px=args.truth_threshold_px,
            ratio=algorithm.ratio,
            min_inliers=algorithm.min_inliers,
        )
    except Exception as exc:
        return MetricRow(
            style=pair.style,
            gate=pair.gate,
            split=pair.split,
            case_type=pair.case_type,
            pair_pt=pair.pair_pt,
            source_name=pair.source_name,
            pair_name=pair.pair_name,
            pure_matches=pair.pure_matches,
            pure_correct=pair.pure_correct,
            pure_wrong=pair.pure_wrong,
            pure_precision=pair.pure_precision,
            algorithm=algorithm.name,
            family=algorithm.family,
            status="error",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            matches=0,
            correct=0,
            wrong=0,
            precision=0.0,
            mean_error_px=math.nan,
            median_error_px=math.nan,
            homography_threshold_px=algorithm.homography_threshold_px,
            truth_threshold_px=args.truth_threshold_px,
            ratio=algorithm.ratio,
            min_inliers=algorithm.min_inliers,
            message=f"{type(exc).__name__}: {exc}",
        )


def summarize_groups(
    pure_by_group: dict[tuple[str, str], list[dict[str, str]]],
    candidates: list[PairRow],
    metrics: list[MetricRow],
    algorithm: Algorithm,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for style, gate in GROUPS:
        pure_totals = aggregate_metric_dicts(pure_by_group[(style, gate)])
        group_candidates = [pair for pair in candidates if pair.style == style and pair.gate == gate]
        subset = [row for row in metrics if row.style == style and row.gate == gate and row.algorithm == algorithm.name]
        fallback_matches = sum(row.matches for row in subset)
        fallback_correct = sum(row.correct for row in subset)
        fallback_wrong = sum(row.wrong for row in subset)
        hybrid_matches = int(pure_totals["matches"]) + fallback_matches
        hybrid_correct = int(pure_totals["correct"]) + fallback_correct
        hybrid_wrong = int(pure_totals["wrong"]) + fallback_wrong
        covered = sum(1 for row in subset if row.matches > 0)
        pair_precisions = [row.precision for row in subset if row.matches > 0]
        pair_matches = [row.matches for row in subset]
        fallback_precision = precision(fallback_correct, fallback_matches)
        hybrid_precision = precision(hybrid_correct, hybrid_matches)
        pure_precision = float(pure_totals["precision"])
        coverage = covered / len(group_candidates) if group_candidates else 0.0
        recommend = int(fallback_matches > 0 and fallback_precision >= 0.98 and coverage >= 0.50 and hybrid_precision >= pure_precision)
        if not group_candidates:
            reason = "no fixed-test pure-PFM gate-zero rows in this group"
        elif fallback_matches <= 0:
            reason = "no fallback support after homography filtering"
        elif fallback_precision < 0.98:
            reason = "fallback precision below 0.98 group guardrail"
        elif coverage < 0.50:
            reason = "fallback coverage below 50% of gate-zero rows"
        elif hybrid_precision < pure_precision:
            reason = "hybrid precision lower than pure-PFM fixed-test precision"
        else:
            reason = "passes fixed-test gate-zero precision, coverage, and hybrid guardrails"
        rows.append(
            {
                "scope": "fixed_test_all_gate_zero",
                "style": style,
                "gate": gate,
                "algorithm": algorithm.name,
                "family": algorithm.family,
                "pure_pairs": pure_totals["pairs"],
                "candidate_pairs": len(group_candidates),
                "ok_pairs": sum(1 for row in subset if row.status == "ok"),
                "covered_pairs": covered,
                "coverage": coverage,
                "fallback_matches": fallback_matches,
                "fallback_correct": fallback_correct,
                "fallback_wrong": fallback_wrong,
                "fallback_precision": fallback_precision,
                "pure_matches": pure_totals["matches"],
                "pure_correct": pure_totals["correct"],
                "pure_wrong": pure_totals["wrong"],
                "pure_precision": pure_precision,
                "hybrid_matches": hybrid_matches,
                "hybrid_correct": hybrid_correct,
                "hybrid_wrong": hybrid_wrong,
                "hybrid_precision": hybrid_precision,
                "hybrid_precision_delta": hybrid_precision - pure_precision,
                "min_pair_precision": min(pair_precisions) if pair_precisions else 0.0,
                "mean_pair_precision": mean(pair_precisions),
                "median_pair_precision": median(pair_precisions),
                "mean_matches_per_candidate": fallback_matches / len(group_candidates) if group_candidates else 0.0,
                "min_pair_matches": min(pair_matches) if pair_matches else 0,
                "pairs_ge_20_inliers": sum(1 for row in subset if row.matches >= 20),
                "pairs_ge_50_inliers": sum(1 for row in subset if row.matches >= 50),
                "mean_error_px": mean([row.mean_error_px for row in subset]),
                "median_error_px": median([row.median_error_px for row in subset]),
                "homography_threshold_px": algorithm.homography_threshold_px,
                "truth_threshold_px": subset[0].truth_threshold_px if subset else "",
                "ratio": algorithm.ratio,
                "recommend": recommend,
                "reason": reason,
            }
        )
    return rows


def total_hybrid_row(rows: list[dict[str, object]]) -> dict[str, object]:
    matches = sum(as_int(row, "hybrid_matches") for row in rows)
    correct = sum(as_int(row, "hybrid_correct") for row in rows)
    wrong = sum(as_int(row, "hybrid_wrong") for row in rows)
    pure_matches = sum(as_int(row, "pure_matches") for row in rows)
    pure_correct = sum(as_int(row, "pure_correct") for row in rows)
    fallback_matches = sum(as_int(row, "fallback_matches") for row in rows)
    fallback_correct = sum(as_int(row, "fallback_correct") for row in rows)
    fallback_wrong = sum(as_int(row, "fallback_wrong") for row in rows)
    candidate_pairs = sum(as_int(row, "candidate_pairs") for row in rows)
    covered_pairs = sum(as_int(row, "covered_pairs") for row in rows)
    pure_pairs = sum(as_int(row, "pure_pairs") for row in rows)
    return {
        "scope": "fixed_test_all_gate_zero",
        "style": "overall",
        "gate": "all",
        "algorithm": STRICT_ALGORITHM,
        "family": "classical",
        "pure_pairs": pure_pairs,
        "candidate_pairs": candidate_pairs,
        "ok_pairs": sum(as_int(row, "ok_pairs") for row in rows),
        "covered_pairs": covered_pairs,
        "coverage": covered_pairs / candidate_pairs if candidate_pairs else 0.0,
        "fallback_matches": fallback_matches,
        "fallback_correct": fallback_correct,
        "fallback_wrong": fallback_wrong,
        "fallback_precision": precision(fallback_correct, fallback_matches),
        "pure_matches": pure_matches,
        "pure_correct": pure_correct,
        "pure_wrong": sum(as_int(row, "pure_wrong") for row in rows),
        "pure_precision": precision(pure_correct, pure_matches),
        "hybrid_matches": matches,
        "hybrid_correct": correct,
        "hybrid_wrong": wrong,
        "hybrid_precision": precision(correct, matches),
        "hybrid_precision_delta": precision(correct, matches) - precision(pure_correct, pure_matches),
        "min_pair_precision": min((as_float(row, "min_pair_precision") for row in rows if as_int(row, "fallback_matches") > 0), default=0.0),
        "mean_pair_precision": mean([as_float(row, "mean_pair_precision") for row in rows]),
        "median_pair_precision": median([as_float(row, "median_pair_precision") for row in rows]),
        "mean_matches_per_candidate": fallback_matches / candidate_pairs if candidate_pairs else 0.0,
        "min_pair_matches": min((as_int(row, "min_pair_matches") for row in rows), default=0),
        "pairs_ge_20_inliers": sum(as_int(row, "pairs_ge_20_inliers") for row in rows),
        "pairs_ge_50_inliers": sum(as_int(row, "pairs_ge_50_inliers") for row in rows),
        "mean_error_px": mean([as_float(row, "mean_error_px") for row in rows]),
        "median_error_px": median([as_float(row, "median_error_px") for row in rows]),
        "homography_threshold_px": 2.0,
        "truth_threshold_px": 3.0,
        "ratio": 0.75,
        "recommend": int(all(as_int(row, "recommend") for row in rows)),
        "reason": f"{sum(as_int(row, 'recommend') for row in rows)}/6 groups pass fixed-test gate-zero guardrails",
    }


def hybrid_route_rows(
    group_rows: list[dict[str, object]],
    pure_by_group: dict[tuple[str, str], list[dict[str, str]]],
    algorithms: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in group_rows:
        if row["algorithm"] not in algorithms:
            continue
        style = str(row["style"])
        gate = str(row["gate"])
        pure_rows = pure_by_group.get((style, gate), [])
        nonzero = [pure for pure in pure_rows if as_int(pure, "matches") > 0]
        nonzero_totals = aggregate_metric_dicts(nonzero)
        rows.append(
            {
                "style": style,
                "gate": gate,
                "route": STAGE11_ROUTE_NAME,
                "algorithm": row["algorithm"],
                "pure_pairs": row["pure_pairs"],
                "gate_zero_pairs": row["candidate_pairs"],
                "fallback_pairs": row["covered_pairs"],
                "pure_nonzero_matches": nonzero_totals["matches"],
                "pure_nonzero_correct": nonzero_totals["correct"],
                "pure_nonzero_wrong": nonzero_totals["wrong"],
                "matches": row["hybrid_matches"],
                "correct": row["hybrid_correct"],
                "wrong": row["hybrid_wrong"],
                "precision": row["hybrid_precision"],
                "delta_matches": row["fallback_matches"],
                "delta_correct": row["fallback_correct"],
                "delta_wrong": row["fallback_wrong"],
                "delta_precision": row["hybrid_precision_delta"],
                "pure_pfm": 0,
                "external_fallback": 1,
                "notes": "Fixed-test pure-PFM nonzero rows plus r0.75/H2 external RootSIFT fallback on pure-PFM gate-zero rows",
            }
        )
    if rows:
        matches = sum(as_int(row, "matches") for row in rows)
        correct = sum(as_int(row, "correct") for row in rows)
        pure_nonzero_matches = sum(as_int(row, "pure_nonzero_matches") for row in rows)
        pure_nonzero_correct = sum(as_int(row, "pure_nonzero_correct") for row in rows)
        pure_matches = sum(as_int(row, "pure_nonzero_matches") for row in rows)
        pure_correct = sum(as_int(row, "pure_nonzero_correct") for row in rows)
        rows.append(
            {
                "style": "overall",
                "gate": "all",
                "route": STAGE11_ROUTE_NAME,
                "algorithm": STRICT_ALGORITHM,
                "pure_pairs": sum(as_int(row, "pure_pairs") for row in rows),
                "gate_zero_pairs": sum(as_int(row, "gate_zero_pairs") for row in rows),
                "fallback_pairs": sum(as_int(row, "fallback_pairs") for row in rows),
                "pure_nonzero_matches": pure_nonzero_matches,
                "pure_nonzero_correct": pure_nonzero_correct,
                "pure_nonzero_wrong": sum(as_int(row, "pure_nonzero_wrong") for row in rows),
                "matches": matches,
                "correct": correct,
                "wrong": sum(as_int(row, "wrong") for row in rows),
                "precision": precision(correct, matches),
                "delta_matches": sum(as_int(row, "delta_matches") for row in rows),
                "delta_correct": sum(as_int(row, "delta_correct") for row in rows),
                "delta_wrong": sum(as_int(row, "delta_wrong") for row in rows),
                "delta_precision": precision(correct, matches) - precision(pure_correct, pure_matches),
                "pure_pfm": 0,
                "external_fallback": 1,
                "notes": "Overall fixed-test hybrid/external route; not a pure learned PFM metric",
            }
        )
    return rows


def stage8_metric_rows(stage8_metrics_path: Path) -> list[dict[str, object]]:
    if not stage8_metrics_path.exists():
        return []
    rows: list[dict[str, object]] = [dict(row) for row in read_csv(stage8_metrics_path)]
    if rows and not any(row.get("style") == "overall" and row.get("gate") == "all" for row in rows):
        matches = sum(as_int(row, "matches") for row in rows)
        correct = sum(as_int(row, "correct") for row in rows)
        rows.append(
            {
                "style": "overall",
                "gate": "all",
                "route": STAGE8_ROUTE_NAME,
                "matches": matches,
                "correct": correct,
                "wrong": sum(as_int(row, "wrong") for row in rows),
                "precision": precision(correct, matches),
                "gate_zero_pairs": sum(as_int(row, "gate_zero_pairs") for row in rows),
                "fallback_pairs": sum(as_int(row, "fallback_pairs") for row in rows),
                "pure_pfm": 0,
                "external_fallback": 1,
                "notes": "Synthesized Stage8 overall row from group metrics",
            }
        )
    return rows


def stage8_support_rows(stage8_support_path: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not stage8_support_path.exists():
        return {}
    rows = [dict(row) for row in read_csv(stage8_support_path) if row.get("scope") == "all_gate_zero" and row.get("algorithm") == STAGE8_BASELINE_ALGORITHM]
    support = {(str(row["style"]), str(row["gate"])): row for row in rows}
    group_rows = [row for row in rows if row.get("style") != "overall"]
    if group_rows:
        matches = sum(as_int(row, "matches") for row in group_rows)
        correct = sum(as_int(row, "correct") for row in group_rows)
        support[("overall", "all")] = {
            "style": "overall",
            "gate": "all",
            "candidate_pairs": sum(as_int(row, "candidate_pairs") for row in group_rows),
            "covered_pairs": sum(as_int(row, "covered_pairs") for row in group_rows),
            "coverage": sum(as_int(row, "covered_pairs") for row in group_rows)
            / sum(as_int(row, "candidate_pairs") for row in group_rows),
            "matches": matches,
            "correct": correct,
            "wrong": sum(as_int(row, "wrong") for row in group_rows),
            "precision": precision(correct, matches),
        }
    return support


def compare_with_stage8(stage11_rows: list[dict[str, object]], stage8_metrics_path: Path) -> list[dict[str, object]]:
    support_path = stage8_metrics_path.parent / "stage8_all_gate_zero_rootsift_support.csv"
    stage8_rows = stage8_metric_rows(stage8_metrics_path)
    stage8_by_group = {(str(row["style"]), str(row["gate"])): row for row in stage8_rows}
    support_by_group = stage8_support_rows(support_path)
    rows: list[dict[str, object]] = []
    for row in stage11_rows:
        key = (str(row["style"]), str(row["gate"]))
        baseline = stage8_by_group.get(key)
        if not baseline:
            continue
        support = support_by_group.get(key, {})
        stage11_fallback_matches = as_int(row, "delta_matches")
        stage11_fallback_correct = as_int(row, "delta_correct")
        stage11_fallback_wrong = as_int(row, "delta_wrong")
        stage8_fallback_matches = as_int(support, "matches")
        stage8_fallback_correct = as_int(support, "correct")
        stage8_fallback_wrong = as_int(support, "wrong")
        rows.append(
            {
                "style": row["style"],
                "gate": row["gate"],
                "stage11_route": row["route"],
                "stage11_algorithm": row["algorithm"],
                "stage11_matches": row["matches"],
                "stage11_correct": row["correct"],
                "stage11_wrong": row["wrong"],
                "stage11_precision": row["precision"],
                "stage11_gate_zero_pairs": row["gate_zero_pairs"],
                "stage11_fallback_pairs": row["fallback_pairs"],
                "stage11_fallback_matches": stage11_fallback_matches,
                "stage11_fallback_correct": stage11_fallback_correct,
                "stage11_fallback_wrong": stage11_fallback_wrong,
                "stage11_fallback_precision": precision(stage11_fallback_correct, stage11_fallback_matches),
                "baseline_route": baseline.get("route", STAGE8_ROUTE_NAME),
                "baseline_algorithm": STAGE8_BASELINE_ALGORITHM,
                "baseline_matches": as_int(baseline, "matches"),
                "baseline_correct": as_int(baseline, "correct"),
                "baseline_wrong": as_int(baseline, "wrong"),
                "baseline_precision": as_float(baseline, "precision"),
                "baseline_gate_zero_pairs": as_int(baseline, "gate_zero_pairs"),
                "baseline_fallback_pairs": as_int(baseline, "fallback_pairs"),
                "baseline_fallback_matches": stage8_fallback_matches,
                "baseline_fallback_correct": stage8_fallback_correct,
                "baseline_fallback_wrong": stage8_fallback_wrong,
                "baseline_fallback_precision": precision(stage8_fallback_correct, stage8_fallback_matches),
                "delta_matches_vs_stage8": as_int(row, "matches") - as_int(baseline, "matches"),
                "delta_correct_vs_stage8": as_int(row, "correct") - as_int(baseline, "correct"),
                "delta_wrong_vs_stage8": as_int(row, "wrong") - as_int(baseline, "wrong"),
                "delta_precision_vs_stage8": as_float(row, "precision") - as_float(baseline, "precision"),
                "delta_coverage_vs_stage8": (as_int(row, "fallback_pairs") / as_int(row, "gate_zero_pairs"))
                - (as_int(baseline, "fallback_pairs") / as_int(baseline, "gate_zero_pairs")),
                "delta_fallback_matches_vs_stage8": stage11_fallback_matches - stage8_fallback_matches,
                "delta_fallback_correct_vs_stage8": stage11_fallback_correct - stage8_fallback_correct,
                "delta_fallback_wrong_vs_stage8": stage11_fallback_wrong - stage8_fallback_wrong,
            }
        )
    return rows


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values: list[str] = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def markdown_summary(
    *,
    args: argparse.Namespace,
    candidates: list[PairRow],
    pure_by_group: dict[tuple[str, str], list[dict[str, str]]],
    group_rows: list[dict[str, object]],
    hybrid_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
) -> str:
    overall = next(row for row in hybrid_rows if row["style"] == "overall")
    overall_group = next(row for row in group_rows if row["style"] == "overall")
    comparison_overall = next((row for row in comparison_rows if row["style"] == "overall"), {})
    recommend_switch = as_float(overall, "precision") >= as_float(comparison_overall, "baseline_precision") and as_int(overall, "wrong") <= as_int(
        comparison_overall, "baseline_wrong"
    )
    lines = [
        "# Matcher Algorithm Iteration Agent14 Stage11",
        "",
        "## Scope",
        "",
        "- This is a **hybrid/external matcher** fixed-test route, not a pure learned PFM result.",
        "- It keeps current pure-PFM fixed-test nonzero rows and applies `RootSIFT-FLANN-r0.75+HomographyUSAC-t2` only where pure PFM emitted zero matches.",
        "- It does not train, run main PFM evaluation, or modify mainline training/evaluation source.",
        f"- Pure route: `{rel(args.pure_route)}`.",
        f"- Stage8 compatibility baseline: `{rel(args.stage8_route)}` using `{STAGE8_BASELINE_ALGORITHM}`.",
        "",
        "## Candidate Rows",
        "",
        "| style | gate | pure pairs | pure matches | pure correct | pure wrong | pure precision | gate-zero pairs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for style, gate in GROUPS:
        pure_totals = aggregate_metric_dicts(pure_by_group[(style, gate)])
        lines.append(
            f"| {style} | {gate} | {pure_totals['pairs']} | {pure_totals['matches']} | {pure_totals['correct']} | "
            f"{pure_totals['wrong']} | {float(pure_totals['precision']):.6f} | "
            f"{sum(1 for pair in candidates if pair.style == style and pair.gate == gate)} |"
        )
    lines.extend(["", "## r0.75/H2 Hybrid Route", ""])
    lines.extend(
        markdown_table(
            hybrid_rows,
            ["style", "gate", "gate_zero_pairs", "fallback_pairs", "matches", "correct", "wrong", "precision", "delta_correct", "delta_wrong"],
        )
    )
    lines.extend(["", "## Stage8 r0.80/H2 Direct Comparison", ""])
    lines.extend(
        markdown_table(
            comparison_rows,
            [
                "style",
                "gate",
                "stage11_precision",
                "baseline_precision",
                "delta_precision_vs_stage8",
                "stage11_correct",
                "baseline_correct",
                "delta_correct_vs_stage8",
                "stage11_wrong",
                "baseline_wrong",
                "delta_wrong_vs_stage8",
            ],
        )
    )
    lines.extend(["", "## Recommendation", ""])
    if recommend_switch:
        lines.append(
            "Recommend changing the fixed-test broad hybrid default from Stage8 `r0.80/H2` to Stage11 `r0.75/H2`: "
            f"overall precision is {as_float(overall, 'precision'):.6f} versus Stage8 {as_float(comparison_overall, 'baseline_precision'):.6f}, "
            f"with {as_int(overall, 'wrong')} wrong matches versus Stage8 {as_int(comparison_overall, 'baseline_wrong')}."
        )
    else:
        lines.append(
            "Do not change the fixed-test broad hybrid default from Stage8 `r0.80/H2` to Stage11 `r0.75/H2` based on fixed-test alone: "
            f"Stage11 precision is {as_float(overall, 'precision'):.6f} versus Stage8 {as_float(comparison_overall, 'baseline_precision'):.6f}, "
            f"with {as_int(overall, 'wrong')} wrong matches versus Stage8 {as_int(comparison_overall, 'baseline_wrong')}."
        )
    lines.append(
        f"Stage11 fallback support: {as_int(overall_group, 'fallback_correct')}/{as_int(overall_group, 'fallback_matches')} correct, "
        f"{as_int(overall_group, 'fallback_wrong')} wrong, precision {as_float(overall_group, 'fallback_precision'):.6f}, "
        f"coverage {as_int(overall_group, 'covered_pairs')}/{as_int(overall_group, 'candidate_pairs')}."
    )
    lines.append(
        "Keep this labeled as a hybrid/external fallback route in downstream reporting and do not merge the numbers into pure-PFM metrics."
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `candidate_pairs.csv`",
            "- `per_pair_metrics.csv`",
            "- `fallback_metrics.csv`",
            "- `per_group_policy_summary.csv`",
            "- `hybrid_route_metrics.csv`",
            "- `hybrid_route_comparison.csv`",
            "- `summary.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, pure_by_group = collect_candidate_pairs(args)
    algorithm, _ = make_algorithm(args)
    metrics: list[MetricRow] = []
    for index, pair in enumerate(candidates, start=1):
        row = evaluate_one(args, algorithm, pair)
        metrics.append(row)
        if args.verbose:
            print(
                f"{algorithm.name:48s} {index:03d}/{len(candidates):03d} {pair.style}/{pair.gate:9s} "
                f"{pair.source_name}/{pair.pair_name} m={row.matches} c={row.correct} w={row.wrong} p={row.precision:.3f}"
            )
    group_rows = summarize_groups(pure_by_group, candidates, metrics, algorithm)
    group_rows_with_total = [*group_rows, total_hybrid_row(group_rows)]
    hybrid_rows = hybrid_route_rows(group_rows, pure_by_group, [STRICT_ALGORITHM])
    comparison_rows = compare_with_stage8(hybrid_rows, args.stage8_route / "hybrid_route_metrics.csv")

    write_csv(args.output_dir / "candidate_pairs.csv", [asdict(pair) for pair in candidates], PAIR_FIELDS)
    metric_dicts = [asdict(row) for row in metrics]
    write_csv(args.output_dir / "per_pair_metrics.csv", metric_dicts, METRIC_FIELDS)
    write_csv(args.output_dir / "fallback_metrics.csv", metric_dicts, METRIC_FIELDS)
    write_csv(args.output_dir / "per_group_policy_summary.csv", group_rows_with_total, GROUP_FIELDS)
    write_csv(args.output_dir / "hybrid_route_metrics.csv", hybrid_rows, HYBRID_FIELDS)
    write_csv(args.output_dir / "hybrid_route_comparison.csv", comparison_rows, COMPARISON_FIELDS)
    (args.output_dir / "summary.md").write_text(
        markdown_summary(
            args=args,
            candidates=candidates,
            pure_by_group=pure_by_group,
            group_rows=group_rows_with_total,
            hybrid_rows=hybrid_rows,
            comparison_rows=comparison_rows,
        ),
        encoding="utf-8",
    )


def self_test() -> None:
    assert precision(1, 2) == 0.5
    assert route_case_type(0) == "pure_pfm_gate_zero_external_fallback_candidate"
    assert route_case_type(3) == "pure_pfm_nonzero_not_replayed"
    rows = [{"matches": "2", "correct": "1", "wrong": "1"}, {"matches": "3", "correct": "3", "wrong": "0"}]
    totals = aggregate_metric_dicts(rows)
    assert totals["matches"] == 5
    assert totals["correct"] == 4
    assert math.isclose(float(totals["precision"]), 0.8)
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-route", type=Path, default=DEFAULT_PURE_ROUTE)
    parser.add_argument("--stage8-route", type=Path, default=DEFAULT_STAGE8_ROUTE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--truth-threshold-px", type=float, default=3.0)
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
