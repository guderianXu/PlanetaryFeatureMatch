#!/usr/bin/env python3
"""Agent13 stage7 sidecar: fallback matchers for local-contrast abstentions.

This script does not train or modify the main route. It evaluates a bounded set
of external matchers only on timestamp/compound pairs where the baseline had
matches but the target-view local-contrast gate abstained, plus any pair where a
baseline-correct match was lost by the gate.
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
AGENT4_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent4.py"
DEFAULT_ROUTE_DIR = PROJECT_ROOT / "runs" / "cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234"
DEFAULT_FULLVAL_DIR = PROJECT_ROOT / "runs" / "timestamp_compound_quality_gate_fullval_current_route_20260526"
DEFAULT_FIXED_GATE_CSV = (
    PROJECT_ROOT
    / "runs"
    / "timestamp_compound_test_quality_gate_diagnostic_20260526"
    / "eval_min_target_local_contrast_5p2.csv"
)
DEFAULT_FULLVAL_GATE_CSV = DEFAULT_FULLVAL_DIR / "eval_min_target_local_contrast_5p2.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage7"
COMPOUND_CACHE_ROOT = PROJECT_ROOT / "img" / "CompoundViewpoint_1024"

TARGET_FIELDS = [
    "split",
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
    "split",
    "case_type",
    "pair_pt",
    "cache_pair_pt",
    "source_name",
    "pair_name",
    "algorithm",
    "family",
    "status",
    "route_eligible",
    "baseline_matches",
    "baseline_correct",
    "gate_matches",
    "gate_correct",
    "lost_correct",
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
    "split",
    "algorithm",
    "family",
    "target_pairs",
    "route_eligible_pairs",
    "lost_correct_pairs",
    "ok_pairs",
    "covered_pairs",
    "coverage",
    "lost_correct_recovered_pairs",
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


A4 = load_module(AGENT4_SCRIPT, "agent4_matcher_for_agent13_stage7")


@dataclass(frozen=True)
class RawOutput:
    points_a: np.ndarray
    points_b: np.ndarray
    keypoints_a: int
    keypoints_b: int
    raw_matches: int


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
    split: str
    case_type: str
    pair_pt: str
    cache_pair_pt: str
    source_name: str
    pair_name: str
    algorithm: str
    family: str
    status: str
    route_eligible: int
    baseline_matches: int
    baseline_correct: int
    gate_matches: int
    gate_correct: int
    lost_correct: int
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


class RootSiftFlannMatcher:
    def __init__(self, *, ratio: float, max_keypoints: int, max_matches: int, sift_contrast: float, clahe: bool = False) -> None:
        import cv2

        self.ratio = ratio
        self.max_matches = max_matches
        self.clahe = clahe
        self.detector = cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=sift_contrast)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> RawOutput:
        import cv2

        if self.clahe:
            image_a = clahe_image(image_a)
            image_b = clahe_image(image_b)
        keypoints_a, descriptors_a = self.detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self.detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return RawOutput(A4.empty_points(), A4.empty_points(), len(keypoints_a or []), len(keypoints_b or []), 0)
        descriptors_a = A4.rootsift(descriptors_a.astype(np.float32, copy=False))
        descriptors_b = A4.rootsift(descriptors_b.astype(np.float32, copy=False))
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        matches = A4.ratio_filter(matcher.knnMatch(descriptors_a, descriptors_b, k=2), self.ratio)
        matches = sorted(matches, key=lambda item: item.distance)[: self.max_matches]
        output = A4.output_from_matches(keypoints_a, keypoints_b, matches)
        return RawOutput(output.points_a, output.points_b, output.keypoints_a, output.keypoints_b, len(matches))


class LightGlueSiftMatcher:
    def __init__(self, *, max_keypoints: int, max_matches: int, device: str) -> None:
        self.impl = A4.LightGlueSiftMatcher(max_keypoints=max_keypoints, device=device)
        self.max_matches = max_matches

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> RawOutput:
        output = self.impl.match(image_a, image_b)
        if output.points_a.shape[0] > self.max_matches:
            points_a = output.points_a[: self.max_matches]
            points_b = output.points_b[: self.max_matches]
        else:
            points_a = output.points_a
            points_b = output.points_b
        return RawOutput(points_a, points_b, output.keypoints_a, output.keypoints_b, int(output.points_a.shape[0]))


def clahe_image(gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def cache_pair_path(pair_pt: str) -> Path:
    path = Path(pair_pt)
    if "img" in path.parts and "CompoundViewpoint_1024" in path.parts:
        return (PROJECT_ROOT / path) if not path.is_absolute() else path
    source = path.parent.name
    return COMPOUND_CACHE_ROOT / source / path.name


def identify_dropped_pairs(
    split: str, baseline_rows: list[dict[str, str]], gate_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    gate_by_pair = {row["pair_pt"]: row for row in gate_rows}
    targets: list[dict[str, object]] = []
    for base in baseline_rows:
        pair_pt = base["pair_pt"]
        gate = gate_by_pair.get(pair_pt, {})
        baseline_matches = as_int(base, "matches")
        baseline_correct = as_int(base, "correct")
        gate_matches = as_int(gate, "matches")
        gate_correct = as_int(gate, "correct")
        route_eligible = int(baseline_matches > 0 and gate_matches == 0)
        lost_correct = max(0, baseline_correct - gate_correct)
        if not route_eligible and lost_correct <= 0:
            continue
        if route_eligible and lost_correct > 0:
            case_type = "dropped_lost_correct"
        elif route_eligible:
            case_type = "dropped_wrong_only"
        else:
            case_type = "lost_correct_partial"
        cache_path = cache_pair_path(pair_pt)
        targets.append(
            {
                "split": split,
                "case_type": case_type,
                "pair_pt": pair_pt,
                "cache_pair_pt": rel(cache_path),
                "source_name": cache_path.parent.name,
                "pair_name": cache_path.name,
                "baseline_matches": baseline_matches,
                "baseline_correct": baseline_correct,
                "baseline_wrong": as_int(base, "wrong"),
                "baseline_precision": as_float(base, "precision"),
                "gate_matches": gate_matches,
                "gate_correct": gate_correct,
                "gate_wrong": as_int(gate, "wrong"),
                "gate_precision": as_float(gate, "precision"),
                "lost_correct": lost_correct,
                "route_eligible": route_eligible,
            }
        )
    return targets


def homography_inliers(raw: RawOutput, threshold_px: float, min_inliers: int) -> RawOutput:
    inlier_a, inlier_b = A4.ransac_inliers(raw.points_a, raw.points_b, threshold_px=threshold_px)
    if inlier_a.shape[0] < min_inliers:
        inlier_a, inlier_b = A4.empty_points(), A4.empty_points()
    return RawOutput(inlier_a, inlier_b, raw.keypoints_a, raw.keypoints_b, raw.raw_matches)


def local_lightglue_sift_available() -> bool:
    if importlib.util.find_spec("lightglue") is None:
        return False
    checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "sift_lightglue_v0-1_arxiv.pth"
    return checkpoint.exists()


def make_algorithms(args: argparse.Namespace) -> tuple[list[Algorithm], list[dict[str, str]]]:
    algorithms: list[Algorithm] = []
    skipped: list[dict[str, str]] = []
    try:
        import cv2

        if not hasattr(cv2, "SIFT_create"):
            skipped.append({"algorithm": "RootSIFT-FLANN-ratio+HomographyUSAC", "reason": "cv2.SIFT_create unavailable"})
        else:
            algorithms.extend(
                [
                    Algorithm(
                        "RootSIFT-FLANN-r0.80+HomographyUSAC-t2",
                        "classical",
                        RootSiftFlannMatcher(
                            ratio=0.80,
                            max_keypoints=args.max_keypoints,
                            max_matches=args.max_matches,
                            sift_contrast=args.sift_contrast,
                        ),
                        2.0,
                        0.80,
                        args.min_inliers,
                    ),
                    Algorithm(
                        "RootSIFT-FLANN-r0.90+HomographyUSAC-t3",
                        "classical",
                        RootSiftFlannMatcher(
                            ratio=0.90,
                            max_keypoints=args.max_keypoints,
                            max_matches=args.max_matches,
                            sift_contrast=args.sift_contrast,
                        ),
                        3.0,
                        0.90,
                        args.min_inliers,
                    ),
                    Algorithm(
                        "CLAHE-RootSIFT-FLANN-r0.90+HomographyUSAC-t3",
                        "classical",
                        RootSiftFlannMatcher(
                            ratio=0.90,
                            max_keypoints=args.max_keypoints,
                            max_matches=args.max_matches,
                            sift_contrast=args.sift_contrast,
                            clahe=True,
                        ),
                        3.0,
                        0.90,
                        args.min_inliers,
                    ),
                ]
            )
    except Exception as exc:
        skipped.append({"algorithm": "OpenCV RootSIFT family", "reason": f"{type(exc).__name__}: {exc}"})

    if args.no_lightglue:
        skipped.append({"algorithm": "LightGlue-SIFT+HomographyUSAC-t3", "reason": "disabled by --no-lightglue"})
    elif not local_lightglue_sift_available():
        skipped.append({"algorithm": "LightGlue-SIFT+HomographyUSAC-t3", "reason": "module/checkpoint unavailable locally"})
    else:
        try:
            algorithms.append(
                Algorithm(
                    "LightGlue-SIFT+HomographyUSAC-t3",
                    "learned",
                    LightGlueSiftMatcher(
                        max_keypoints=args.learned_max_keypoints,
                        max_matches=args.max_matches,
                        device=choose_device(args.device),
                    ),
                    3.0,
                    math.nan,
                    args.min_inliers,
                )
            )
        except Exception as exc:
            skipped.append({"algorithm": "LightGlue-SIFT+HomographyUSAC-t3", "reason": f"{type(exc).__name__}: {exc}"})

    if args.limit_algorithms:
        keep = set(args.limit_algorithms)
        algorithms = [algorithm for algorithm in algorithms if algorithm.name in keep]
    return algorithms, skipped


def evaluate_one(args: argparse.Namespace, algorithm: Algorithm, target: dict[str, object]) -> MetricRow:
    cache_path = PROJECT_ROOT / str(target["cache_pair_pt"])
    try:
        image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(cache_path)
        raw = algorithm.matcher.match(image_a, image_b)
        output = homography_inliers(raw, algorithm.homography_threshold_px, algorithm.min_inliers)
        matches, correct, wrong, pair_precision, mean_error, median_error = A4.compute_metrics(
            output.points_a,
            output.points_b,
            warp_a_to_b,
            valid_mask,
            threshold_px=args.truth_threshold_px,
        )
        return MetricRow(
            split=str(target["split"]),
            case_type=str(target["case_type"]),
            pair_pt=str(target["pair_pt"]),
            cache_pair_pt=str(target["cache_pair_pt"]),
            source_name=str(target["source_name"]),
            pair_name=str(target["pair_name"]),
            algorithm=algorithm.name,
            family=algorithm.family,
            status="ok",
            route_eligible=as_int(target, "route_eligible"),
            baseline_matches=as_int(target, "baseline_matches"),
            baseline_correct=as_int(target, "baseline_correct"),
            gate_matches=as_int(target, "gate_matches"),
            gate_correct=as_int(target, "gate_correct"),
            lost_correct=as_int(target, "lost_correct"),
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
            split=str(target["split"]),
            case_type=str(target["case_type"]),
            pair_pt=str(target["pair_pt"]),
            cache_pair_pt=str(target["cache_pair_pt"]),
            source_name=str(target["source_name"]),
            pair_name=str(target["pair_name"]),
            algorithm=algorithm.name,
            family=algorithm.family,
            status="error",
            route_eligible=as_int(target, "route_eligible"),
            baseline_matches=as_int(target, "baseline_matches"),
            baseline_correct=as_int(target, "baseline_correct"),
            gate_matches=as_int(target, "gate_matches"),
            gate_correct=as_int(target, "gate_correct"),
            lost_correct=as_int(target, "lost_correct"),
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


def aggregate(rows: list[dict[str, str] | dict[str, object]]) -> dict[str, int | float]:
    matches = sum(as_int(row, "matches") for row in rows)
    correct = sum(as_int(row, "correct") for row in rows)
    wrong = sum(as_int(row, "wrong") for row in rows)
    return {"matches": matches, "correct": correct, "wrong": wrong, "precision": precision(correct, matches)}


def summarize_fallbacks(
    *,
    baseline_by_split: dict[str, list[dict[str, str]]],
    gate_by_split: dict[str, list[dict[str, str]]],
    targets_by_split: dict[str, list[dict[str, object]]],
    metric_rows: list[MetricRow],
    algorithms: list[Algorithm],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    splits = list(targets_by_split)
    for split in splits:
        gate_totals = aggregate(gate_by_split[split])
        baseline_totals = aggregate(baseline_by_split[split])
        summaries.append(
            {
                "scope": "baseline",
                "split": split,
                "algorithm": "baseline_current_route",
                "family": "route",
                "target_pairs": len(targets_by_split[split]),
                "route_eligible_pairs": sum(as_int(row, "route_eligible") for row in targets_by_split[split]),
                "lost_correct_pairs": sum(1 for row in targets_by_split[split] if as_int(row, "lost_correct") > 0),
                "ok_pairs": len(baseline_by_split[split]),
                "covered_pairs": sum(1 for row in baseline_by_split[split] if as_int(row, "matches") > 0),
                "coverage": 0.0,
                "lost_correct_recovered_pairs": "",
                "matches": baseline_totals["matches"],
                "correct": baseline_totals["correct"],
                "wrong": baseline_totals["wrong"],
                "precision": baseline_totals["precision"],
                "gate_matches": gate_totals["matches"],
                "gate_correct": gate_totals["correct"],
                "gate_wrong": gate_totals["wrong"],
                "gate_precision": gate_totals["precision"],
                "combined_matches": baseline_totals["matches"],
                "combined_correct": baseline_totals["correct"],
                "combined_wrong": baseline_totals["wrong"],
                "combined_precision": baseline_totals["precision"],
                "combined_precision_delta": baseline_totals["precision"] - gate_totals["precision"],
                "combined_correct_delta": baseline_totals["correct"] - gate_totals["correct"],
                "recommend": 0,
                "reason": "pre-gate reference",
            }
        )
        summaries.append(
            {
                "scope": "gate",
                "split": split,
                "algorithm": "local_contrast_gate_5.2",
                "family": "route",
                "target_pairs": len(targets_by_split[split]),
                "route_eligible_pairs": sum(as_int(row, "route_eligible") for row in targets_by_split[split]),
                "lost_correct_pairs": sum(1 for row in targets_by_split[split] if as_int(row, "lost_correct") > 0),
                "ok_pairs": len(gate_by_split[split]),
                "covered_pairs": sum(1 for row in gate_by_split[split] if as_int(row, "matches") > 0),
                "coverage": 0.0,
                "lost_correct_recovered_pairs": "",
                "matches": gate_totals["matches"],
                "correct": gate_totals["correct"],
                "wrong": gate_totals["wrong"],
                "precision": gate_totals["precision"],
                "gate_matches": gate_totals["matches"],
                "gate_correct": gate_totals["correct"],
                "gate_wrong": gate_totals["wrong"],
                "gate_precision": gate_totals["precision"],
                "combined_matches": gate_totals["matches"],
                "combined_correct": gate_totals["correct"],
                "combined_wrong": gate_totals["wrong"],
                "combined_precision": gate_totals["precision"],
                "combined_precision_delta": 0.0,
                "combined_correct_delta": 0,
                "recommend": 1,
                "reason": "current abstain-only route",
            }
        )

        for algorithm in algorithms:
            subset = [row for row in metric_rows if row.split == split and row.algorithm == algorithm.name]
            route_subset = [row for row in subset if row.route_eligible]
            matches = sum(row.matches for row in route_subset)
            correct = sum(row.correct for row in route_subset)
            wrong = sum(row.wrong for row in route_subset)
            combined_matches = int(gate_totals["matches"]) + matches
            combined_correct = int(gate_totals["correct"]) + correct
            combined_wrong = int(gate_totals["wrong"]) + wrong
            target_count = len(targets_by_split[split])
            route_count = sum(as_int(row, "route_eligible") for row in targets_by_split[split])
            covered = sum(1 for row in route_subset if row.matches > 0)
            lost_recovered = sum(1 for row in subset if row.lost_correct > 0 and row.correct > 0)
            combined_precision = precision(combined_correct, combined_matches)
            gate_precision = float(gate_totals["precision"])
            precision_delta = combined_precision - gate_precision
            recommend = int(correct > 0 and precision_delta >= 0.0)
            if correct <= 0:
                reason = "no correct fallback matches on abstained pairs"
            elif precision_delta < 0.0:
                reason = "recovers correct matches but lowers split precision"
            else:
                reason = "recovers correct matches without lowering split precision"
            summaries.append(
                {
                    "scope": "fallback_after_abstain",
                    "split": split,
                    "algorithm": algorithm.name,
                    "family": algorithm.family,
                    "target_pairs": target_count,
                    "route_eligible_pairs": route_count,
                    "lost_correct_pairs": sum(1 for row in targets_by_split[split] if as_int(row, "lost_correct") > 0),
                    "ok_pairs": sum(1 for row in subset if row.status == "ok"),
                    "covered_pairs": covered,
                    "coverage": covered / route_count if route_count else 0.0,
                    "lost_correct_recovered_pairs": lost_recovered,
                    "matches": matches,
                    "correct": correct,
                    "wrong": wrong,
                    "precision": precision(correct, matches),
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
        split_rows = [
            row
            for row in summaries
            if row["scope"] == "fallback_after_abstain" and row["algorithm"] == algorithm.name and row["split"] in splits
        ]
        if not split_rows:
            continue
        fixed = next((row for row in split_rows if row["split"] == "fixed_test"), None)
        full = next((row for row in split_rows if row["split"] == "full_val"), None)
        matches = sum(as_int(row, "matches") for row in split_rows)
        correct = sum(as_int(row, "correct") for row in split_rows)
        wrong = sum(as_int(row, "wrong") for row in split_rows)
        split_safe = bool(
            fixed
            and full
            and as_int(fixed, "recommend") == 1
            and as_int(full, "recommend") == 1
            and as_int(fixed, "correct") > 0
            and as_int(full, "correct") > 0
        )
        if split_safe:
            reason = "non-negative combined precision on both fixed-test and full-val"
        else:
            reason = "does not recover correct matches without a precision drop on both splits"
        summaries.append(
            {
                "scope": "fallback_after_abstain",
                "split": "overall",
                "algorithm": algorithm.name,
                "family": algorithm.family,
                "target_pairs": sum(as_int(row, "target_pairs") for row in split_rows),
                "route_eligible_pairs": sum(as_int(row, "route_eligible_pairs") for row in split_rows),
                "lost_correct_pairs": sum(as_int(row, "lost_correct_pairs") for row in split_rows),
                "ok_pairs": sum(as_int(row, "ok_pairs") for row in split_rows),
                "covered_pairs": sum(as_int(row, "covered_pairs") for row in split_rows),
                "coverage": (
                    sum(as_int(row, "covered_pairs") for row in split_rows)
                    / sum(as_int(row, "route_eligible_pairs") for row in split_rows)
                )
                if sum(as_int(row, "route_eligible_pairs") for row in split_rows)
                else 0.0,
                "lost_correct_recovered_pairs": sum(as_int(row, "lost_correct_recovered_pairs") for row in split_rows),
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": precision(correct, matches),
                "gate_matches": sum(as_int(row, "gate_matches") for row in split_rows),
                "gate_correct": sum(as_int(row, "gate_correct") for row in split_rows),
                "gate_wrong": sum(as_int(row, "gate_wrong") for row in split_rows),
                "gate_precision": precision(
                    sum(as_int(row, "gate_correct") for row in split_rows),
                    sum(as_int(row, "gate_matches") for row in split_rows),
                ),
                "combined_matches": sum(as_int(row, "combined_matches") for row in split_rows),
                "combined_correct": sum(as_int(row, "combined_correct") for row in split_rows),
                "combined_wrong": sum(as_int(row, "combined_wrong") for row in split_rows),
                "combined_precision": precision(
                    sum(as_int(row, "combined_correct") for row in split_rows),
                    sum(as_int(row, "combined_matches") for row in split_rows),
                ),
                "combined_precision_delta": precision(
                    sum(as_int(row, "combined_correct") for row in split_rows),
                    sum(as_int(row, "combined_matches") for row in split_rows),
                )
                - precision(
                    sum(as_int(row, "gate_correct") for row in split_rows),
                    sum(as_int(row, "gate_matches") for row in split_rows),
                ),
                "combined_correct_delta": correct,
                "recommend": int(split_safe),
                "reason": reason,
            }
        )
    return summaries


def markdown_summary(
    *,
    args: argparse.Namespace,
    targets_by_split: dict[str, list[dict[str, object]]],
    summaries: list[dict[str, object]],
    skipped: list[dict[str, str]],
) -> str:
    lines: list[str] = []
    lines.append("# Matcher Algorithm Iteration Agent13 Stage7")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Question: can external matchers rescue timestamp/compound pairs dropped by the B-view local-contrast 5.2 gate?")
    lines.append("- Evaluation is limited to dropped/lost pairs; no training or main evaluator edits.")
    lines.append(f"- Truth threshold: `{args.truth_threshold_px}` px; min homography inliers: `{args.min_inliers}`.")
    lines.append("")
    lines.append("## Dropped/Lost Set")
    lines.append("")
    lines.append("| split | target pairs | route-eligible dropped | lost-correct pairs | baseline dropped matches | baseline dropped correct |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split, rows in targets_by_split.items():
        route_rows = [row for row in rows if as_int(row, "route_eligible") == 1]
        lines.append(
            f"| {split} | {len(rows)} | {len(route_rows)} | "
            f"{sum(1 for row in rows if as_int(row, 'lost_correct') > 0)} | "
            f"{sum(as_int(row, 'baseline_matches') for row in route_rows)} | "
            f"{sum(as_int(row, 'baseline_correct') for row in route_rows)} |"
        )
    lines.append("")
    lines.append("## Fallback Route Summary")
    lines.append("")
    lines.append("| split | algorithm | coverage | fallback matches | fallback correct | fallback precision | combined precision | delta | recommend |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        if row["scope"] != "fallback_after_abstain":
            continue
        lines.append(
            f"| {row['split']} | {row['algorithm']} | {as_float(row, 'coverage'):.3f} | "
            f"{row['matches']} | {row['correct']} | {as_float(row, 'precision'):.6f} | "
            f"{as_float(row, 'combined_precision'):.6f} | {as_float(row, 'combined_precision_delta'):.6f} | "
            f"{row['recommend']} |"
        )
    lines.append("")
    lines.append("## Baseline vs Gate")
    lines.append("")
    lines.append("| split | route | matches | correct | precision |")
    lines.append("|---|---|---:|---:|---:|")
    for row in summaries:
        if row["scope"] in {"baseline", "gate"}:
            lines.append(f"| {row['split']} | {row['algorithm']} | {row['matches']} | {row['correct']} | {as_float(row, 'precision'):.6f} |")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    recommended = [row for row in summaries if row["scope"] == "fallback_after_abstain" and row["split"] == "overall" and as_int(row, "recommend") == 1]
    recommended_classical = [row for row in recommended if row["family"] == "classical"]
    if recommended_classical:
        best = max(
            recommended_classical,
            key=lambda row: (as_float(row, "coverage"), as_float(row, "precision"), as_float(row, "combined_precision_delta")),
        )
        lines.append(
            f"Combine the local-contrast gate with `{best['algorithm']}` after abstention. It covered "
            f"{best['covered_pairs']}/{best['route_eligible_pairs']} dropped pairs and recovered "
            f"{best['combined_correct_delta']} correct fallback matches overall with non-negative combined precision on both splits."
        )
        learned = [row for row in recommended if row["family"] != "classical"]
        if learned:
            light = max(learned, key=lambda row: as_float(row, "combined_precision"))
            lines.append(
                f"`{light['algorithm']}` was also viable, but keep it secondary because it is optional/learned and covered "
                f"{light['covered_pairs']}/{light['route_eligible_pairs']} dropped pairs."
            )
    elif recommended:
        best = max(recommended, key=lambda row: (as_float(row, "combined_precision_delta"), as_int(row, "combined_correct_delta")))
        lines.append(
            f"Only a non-classical candidate passed the split checks: `{best['algorithm']}` recovered "
            f"{best['combined_correct_delta']} correct fallback matches overall. That is useful diagnostically, but it does not answer "
            "the classical-fallback routing question."
        )
    else:
        lines.append(
            "Keep the local-contrast abstain-only route. In this dropped-pair set, no fallback recovered correct matches on both fixed-test "
            "and full-val without lowering combined precision."
        )
    if skipped:
        lines.append("")
        lines.append("## Skipped")
        lines.append("")
        lines.append("| algorithm | reason |")
        lines.append("|---|---|")
        for row in skipped:
            lines.append(f"| {row['algorithm']} | {row['reason']} |")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `target_pairs.csv`")
    lines.append("- `fallback_metrics.csv`")
    lines.append("- `fallback_summary.csv`")
    lines.append("- `summary.md`")
    return "\n".join(lines) + "\n"


def load_targets(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], dict[str, list[dict[str, object]]]]:
    baseline_by_split = {
        "fixed_test": read_csv(args.fixed_baseline_summary),
        "full_val": read_csv(args.fullval_baseline_summary),
    }
    gate_by_split = {
        "fixed_test": read_csv(args.fixed_gate_summary),
        "full_val": read_csv(args.fullval_gate_summary),
    }
    targets_by_split = {
        split: identify_dropped_pairs(split, baseline_by_split[split], gate_by_split[split])
        for split in ("fixed_test", "full_val")
    }
    return baseline_by_split, gate_by_split, targets_by_split


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_by_split, gate_by_split, targets_by_split = load_targets(args)
    algorithms, skipped = make_algorithms(args)
    metric_rows: list[MetricRow] = []
    all_targets = targets_by_split["fixed_test"] + targets_by_split["full_val"]
    for algorithm in algorithms:
        for index, target in enumerate(all_targets, start=1):
            row = evaluate_one(args, algorithm, target)
            metric_rows.append(row)
            if args.verbose:
                print(
                    f"{algorithm.name:48s} {index:03d}/{len(all_targets):03d} {target['split']:10s} "
                    f"{target['source_name']}/{target['pair_name']} m={row.matches} c={row.correct} p={row.precision:.3f}"
                )
    summaries = summarize_fallbacks(
        baseline_by_split=baseline_by_split,
        gate_by_split=gate_by_split,
        targets_by_split=targets_by_split,
        metric_rows=metric_rows,
        algorithms=algorithms,
    )
    write_csv(args.output_dir / "target_pairs.csv", targets_by_split["fixed_test"] + targets_by_split["full_val"], TARGET_FIELDS)
    write_csv(args.output_dir / "fallback_metrics.csv", [asdict(row) for row in metric_rows], METRIC_FIELDS)
    write_csv(args.output_dir / "fallback_summary.csv", summaries, SUMMARY_FIELDS)
    (args.output_dir / "summary.md").write_text(
        markdown_summary(args=args, targets_by_split=targets_by_split, summaries=summaries, skipped=skipped),
        encoding="utf-8",
    )


def self_test() -> None:
    baseline_rows = [
        {"pair_pt": "a.pt", "matches": "3", "correct": "1", "wrong": "2", "precision": "0.333333"},
        {"pair_pt": "b.pt", "matches": "4", "correct": "0", "wrong": "4", "precision": "0.000000"},
        {"pair_pt": "c.pt", "matches": "0", "correct": "0", "wrong": "0", "precision": "0.000000"},
    ]
    gate_rows = [
        {"pair_pt": "a.pt", "matches": "0", "correct": "0", "wrong": "0", "precision": "0.000000"},
        {"pair_pt": "b.pt", "matches": "2", "correct": "0", "wrong": "2", "precision": "0.000000"},
        {"pair_pt": "c.pt", "matches": "0", "correct": "0", "wrong": "0", "precision": "0.000000"},
    ]
    dropped = identify_dropped_pairs("fixed_test", baseline_rows, gate_rows)
    assert [row["pair_pt"] for row in dropped] == ["a.pt"]
    assert dropped[0]["lost_correct"] == 1
    assert dropped[0]["route_eligible"] == 1
    gate_totals = aggregate(gate_rows)
    assert gate_totals["matches"] == 2
    assert gate_totals["correct"] == 0
    synthetic_metrics = [
        MetricRow(
            split="fixed_test",
            case_type="dropped_lost_correct",
            pair_pt="a.pt",
            cache_pair_pt="img/CompoundViewpoint_1024/source/a.pt",
            source_name="source",
            pair_name="a.pt",
            algorithm="synthetic",
            family="test",
            status="ok",
            route_eligible=1,
            baseline_matches=3,
            baseline_correct=1,
            gate_matches=0,
            gate_correct=0,
            lost_correct=1,
            keypoints_a=5,
            keypoints_b=5,
            raw_matches=4,
            matches=2,
            correct=1,
            wrong=1,
            precision=0.5,
            mean_error_px=1.0,
            median_error_px=1.0,
            homography_threshold_px=2.0,
            truth_threshold_px=3.0,
            ratio=0.8,
            min_inliers=1,
        )
    ]
    alg = Algorithm("synthetic", "test", object(), 2.0, 0.8, 1)
    summary = summarize_fallbacks(
        baseline_by_split={"fixed_test": baseline_rows, "full_val": baseline_rows},
        gate_by_split={"fixed_test": gate_rows, "full_val": gate_rows},
        targets_by_split={"fixed_test": dropped, "full_val": []},
        metric_rows=synthetic_metrics,
        algorithms=[alg],
    )
    fixed_summary = next(row for row in summary if row["scope"] == "fallback_after_abstain" and row["split"] == "fixed_test")
    assert fixed_summary["combined_correct"] == 1
    assert fixed_summary["combined_matches"] == 4
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-baseline-summary", type=Path, default=DEFAULT_ROUTE_DIR / "eval" / "timestamp" / "compound" / "summary.csv")
    parser.add_argument("--fullval-baseline-summary", type=Path, default=DEFAULT_FULLVAL_DIR / "summary.csv")
    parser.add_argument("--fixed-gate-summary", type=Path, default=DEFAULT_FIXED_GATE_CSV)
    parser.add_argument("--fullval-gate-summary", type=Path, default=DEFAULT_FULLVAL_GATE_CSV)
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
