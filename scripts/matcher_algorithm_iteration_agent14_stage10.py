#!/usr/bin/env python3
"""Agent14 stage10 sidecar: full-val gate-zero fallback replay.

This script writes only Stage10 sidecar artifacts. It does not train PFM and
does not modify the main training/evaluation source. It first evaluates the
current selected pure-PFM route parameters on the full validation split to find
gate-zero rows, then replays external fallback matchers on those rows.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE7_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent13_stage7.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent14_stage10"
DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234"
    / "splits"
    / "val"
)
DEFAULT_SELECTED_ROUTE = (
    PROJECT_ROOT
    / "runs"
    / "cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234"
)
STAGE8_ROUTE = PROJECT_ROOT / "runs" / "cross_view_1024_targetcontrast_rootsift_allgatezero_fallback_route_20260526"
STAGE9_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent14_stage9"
PYTHON_EXE = Path("/home/xjw/anaconda3/envs/pfm-train/bin/python")

GROUPS = [(style, gate) for style in ("numeric", "timestamp") for gate in ("rotate", "viewpoint", "compound")]
REQUIRED_ALGORITHMS = {
    "RootSIFT-FLANN-r0.75+HomographyUSAC-t2",
    "RootSIFT-FLANN-r0.80+HomographyUSAC-t2",
    "RootSIFT-FLANN-r0.80+HomographyUSAC-t1.5",
    "LightGlue-SIFT+HomographyUSAC-t3",
}

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
    "route_param_label",
    "texture_blend_weight",
    "keypoint_score_mode",
    "min_margin",
    "min_target_gradient",
    "min_target_local_contrast",
    "pytorch_state_label",
    "pytorch_state",
    "sample_seed",
    "sample_rank",
    "source_pool_pairs",
]

METRIC_FIELDS = [
    *PAIR_FIELDS[:9],
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

RANKING_FIELDS = [
    "scope",
    "algorithm",
    "family",
    "groups",
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
    "min_group_fallback_precision",
    "mean_group_fallback_precision",
    "min_group_hybrid_precision",
    "wrong_per_candidate",
    "mean_matches_per_candidate",
    "rank_score",
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S7 = load_module(STAGE7_SCRIPT, "agent13_stage7_for_stage10")
A4 = S7.A4


@dataclass(frozen=True)
class RouteParams:
    style: str
    gate: str
    texture_blend_weight: float
    keypoint_score_mode: str
    min_margin: float
    min_target_gradient: float
    min_target_local_contrast: float
    pytorch_state_label: str
    pytorch_state: str


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
    route_param_label: str
    texture_blend_weight: float
    keypoint_score_mode: str
    min_margin: float
    min_target_gradient: float
    min_target_local_contrast: float
    pytorch_state_label: str
    pytorch_state: str
    sample_seed: int
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


class BinaryDescriptorMatcher:
    def __init__(self, *, detector_name: str, ratio: float, max_keypoints: int, max_matches: int) -> None:
        import cv2

        self.ratio = ratio
        self.max_matches = max_matches
        if detector_name == "ORB":
            self.detector = cv2.ORB_create(nfeatures=max_keypoints, fastThreshold=5)
        elif detector_name == "AKAZE":
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(detector_name)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> S7.RawOutput:
        import cv2

        keypoints_a, descriptors_a = self.detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self.detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return S7.RawOutput(A4.empty_points(), A4.empty_points(), len(keypoints_a or []), len(keypoints_b or []), 0)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = A4.ratio_filter(matcher.knnMatch(descriptors_a, descriptors_b, k=2), self.ratio)
        matches = sorted(matches, key=lambda item: item.distance)[: self.max_matches]
        output = A4.output_from_matches(keypoints_a, keypoints_b, matches)
        return S7.RawOutput(output.points_a, output.points_b, output.keypoints_a, output.keypoints_b, len(matches))


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


def route_case_type(matches: int) -> str:
    return "pfm_gate_zero_external_fallback_candidate" if matches == 0 else "pfm_nonzero_not_replayed"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def group_eval_csv(output_dir: Path, style: str, gate: str) -> Path:
    return output_dir / "pure_pfm_fullval" / "eval" / style / gate / "summary.csv"


def selected_weights_path(route: Path) -> Path:
    return route / "calibration" / "selected_weights.csv"


def load_route_params(route: Path) -> dict[tuple[str, str], RouteParams]:
    rows = read_csv(selected_weights_path(route))
    params: dict[tuple[str, str], RouteParams] = {}
    for row in rows:
        style = row["style"]
        gate = row["gate"]
        params[(style, gate)] = RouteParams(
            style=style,
            gate=gate,
            texture_blend_weight=as_float(row, "texture_blend_weight"),
            keypoint_score_mode=row.get("keypoint_score_mode", "texture"),
            min_margin=as_float(row, "min_margin"),
            min_target_gradient=as_float(row, "min_target_gradient"),
            min_target_local_contrast=as_float(row, "min_target_local_contrast"),
            pytorch_state_label=row.get("pytorch_state_label", "trained"),
            pytorch_state=row["pytorch_state"],
        )
    missing = [group for group in GROUPS if group not in params]
    if missing:
        raise RuntimeError(f"selected weights missing groups: {missing}")
    return params


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def pfm_eval_command(args: argparse.Namespace, style: str, gate: str, params: RouteParams, output_csv: Path) -> list[str]:
    command = [
        str(args.python_exe),
        str(PROJECT_ROOT / "python" / "pytorch_cache_match_eval.py"),
        "--cache-dir",
        str((args.split_root / style / gate).resolve()),
        "--pytorch-state",
        str(resolve_project_path(params.pytorch_state)),
        "--output",
        str(output_csv),
        "--device",
        choose_device(args.device),
        "--mode",
        "blend",
        "--texture-blend-weight",
        f"{params.texture_blend_weight:.12g}",
        "--geometry-filter",
        args.geometry_filter,
        "--descriptor-topk",
        str(args.descriptor_topk),
        "--max-keypoints",
        str(args.max_keypoints_pfm),
        "--keypoint-spatial-bins",
        str(args.keypoint_spatial_bins),
        "--keypoint-score-mode",
        params.keypoint_score_mode,
        "--mutual",
        "--exclude-self-pairs",
    ]
    if params.min_margin > 0.0:
        command.extend(["--min-margin", f"{params.min_margin:.12g}"])
    if params.min_target_gradient > 0.0:
        command.extend(["--min-target-gradient", f"{params.min_target_gradient:.12g}"])
    if params.min_target_local_contrast > 0.0:
        command.extend(["--min-target-local-contrast", f"{params.min_target_local_contrast:.12g}"])
    if args.pfm_limit_pairs > 0:
        command.extend(["--limit-pairs", str(args.pfm_limit_pairs), "--sample-seed", str(args.seed)])
    return command


def run_pfm_fullval(args: argparse.Namespace, route_params: dict[tuple[str, str], RouteParams]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "python"
    env["MKL_THREADING_LAYER"] = "GNU"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_dir = args.output_dir / "pure_pfm_fullval" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for style, gate in GROUPS:
        output_csv = group_eval_csv(args.output_dir, style, gate)
        if output_csv.exists() and not args.force_pfm_eval:
            continue
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        command = pfm_eval_command(args, style, gate, route_params[(style, gate)], output_csv)
        log_path = log_dir / f"{style}_{gate}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(command) + "\n\n")
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def route_param_label(params: RouteParams) -> str:
    if params.style == "timestamp" and params.gate == "compound" and params.pytorch_state_label == "lowcontrast":
        return "current_lowcontrast_route_params"
    return "targetcontrast_route_params"


def collect_candidate_pairs(args: argparse.Namespace, route_params: dict[tuple[str, str], RouteParams]) -> tuple[list[PairRow], dict[tuple[str, str], list[dict[str, str]]]]:
    candidates: list[PairRow] = []
    pure_by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    rng = random.Random(args.seed)
    for style, gate in GROUPS:
        rows = read_csv(group_eval_csv(args.output_dir, style, gate))
        pure_by_group[(style, gate)] = rows
        zero_rows = [row for row in rows if as_int(row, "matches") == 0]
        selected = list(zero_rows)
        if args.candidate_sample_per_group > 0 and len(selected) > args.candidate_sample_per_group:
            selected = sorted(rng.sample(selected, args.candidate_sample_per_group), key=lambda row: row["pair_pt"])
        params = route_params[(style, gate)]
        for rank, row in enumerate(selected, start=1):
            pair_path = resolve_project_path(row["pair_pt"])
            candidates.append(
                PairRow(
                    style=style,
                    gate=gate,
                    split="full_val",
                    case_type=route_case_type(as_int(row, "matches")),
                    pair_pt=rel(pair_path),
                    source_name=pair_path.parent.name,
                    pair_name=pair_path.name,
                    pure_matches=as_int(row, "matches"),
                    pure_correct=as_int(row, "correct"),
                    pure_wrong=as_int(row, "wrong"),
                    pure_precision=as_float(row, "precision"),
                    route_param_label=route_param_label(params),
                    texture_blend_weight=params.texture_blend_weight,
                    keypoint_score_mode=params.keypoint_score_mode,
                    min_margin=params.min_margin,
                    min_target_gradient=params.min_target_gradient,
                    min_target_local_contrast=params.min_target_local_contrast,
                    pytorch_state_label=params.pytorch_state_label,
                    pytorch_state=params.pytorch_state,
                    sample_seed=args.seed if args.candidate_sample_per_group > 0 else 0,
                    sample_rank=rank,
                    source_pool_pairs=len(zero_rows),
                )
            )
    return candidates, pure_by_group


def lightglue_skip_reason(args: argparse.Namespace) -> str:
    if args.no_lightglue:
        return "disabled by --no-lightglue"
    if importlib.util.find_spec("lightglue") is None:
        return "module 'lightglue' unavailable"
    checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "sift_lightglue_v0-1_arxiv.pth"
    if not checkpoint.exists():
        return f"checkpoint missing: {checkpoint}"
    return ""


def make_algorithms(args: argparse.Namespace) -> tuple[list[Algorithm], list[dict[str, str]]]:
    algorithms: list[Algorithm] = []
    skipped: list[dict[str, str]] = []
    try:
        import cv2

        if not hasattr(cv2, "SIFT_create"):
            skipped.append({"algorithm": "RootSIFT-FLANN family", "reason": "cv2.SIFT_create unavailable"})
        else:
            for ratio, threshold in ((0.75, 2.0), (0.80, 2.0), (0.80, 1.5)):
                algorithms.append(
                    Algorithm(
                        name=f"RootSIFT-FLANN-r{ratio:.2f}+HomographyUSAC-t{threshold:g}",
                        family="classical",
                        matcher=S7.RootSiftFlannMatcher(
                            ratio=ratio,
                            max_keypoints=args.max_keypoints,
                            max_matches=args.max_matches,
                            sift_contrast=args.sift_contrast,
                        ),
                        homography_threshold_px=threshold,
                        ratio=ratio,
                        min_inliers=args.min_inliers,
                    )
                )
            if args.include_secondary:
                for detector in ("ORB", "AKAZE"):
                    name = f"{detector}-BF-r0.80+HomographyUSAC-t3"
                    try:
                        algorithms.append(
                            Algorithm(
                                name=name,
                                family="classical_secondary",
                                matcher=BinaryDescriptorMatcher(
                                    detector_name=detector,
                                    ratio=0.80,
                                    max_keypoints=args.max_keypoints,
                                    max_matches=args.max_matches,
                                ),
                                homography_threshold_px=3.0,
                                ratio=0.80,
                                min_inliers=args.min_inliers,
                            )
                        )
                    except Exception as exc:
                        skipped.append({"algorithm": name, "reason": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        skipped.append({"algorithm": "OpenCV matcher families", "reason": f"{type(exc).__name__}: {exc}"})

    reason = lightglue_skip_reason(args)
    if reason:
        skipped.append({"algorithm": "LightGlue-SIFT+HomographyUSAC-t3", "reason": reason})
    else:
        try:
            algorithms.append(
                Algorithm(
                    name="LightGlue-SIFT+HomographyUSAC-t3",
                    family="learned_external",
                    matcher=S7.LightGlueSiftMatcher(
                        max_keypoints=args.learned_max_keypoints,
                        max_matches=args.max_matches,
                        device=choose_device(args.device),
                    ),
                    homography_threshold_px=3.0,
                    ratio=math.nan,
                    min_inliers=args.min_inliers,
                )
            )
        except Exception as exc:
            skipped.append({"algorithm": "LightGlue-SIFT+HomographyUSAC-t3", "reason": f"{type(exc).__name__}: {exc}"})

    if args.limit_algorithms:
        keep = set(args.limit_algorithms)
        algorithms = [algorithm for algorithm in algorithms if algorithm.name in keep]
    present = {algorithm.name for algorithm in algorithms}
    for required in sorted(REQUIRED_ALGORITHMS - present):
        if required != "LightGlue-SIFT+HomographyUSAC-t3":
            skipped.append({"algorithm": required, "reason": "required algorithm unavailable after matcher construction"})
    return algorithms, skipped


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


def aggregate_rows(rows: list[dict[str, str]]) -> dict[str, int | float]:
    matches = sum(as_int(row, "matches") for row in rows)
    correct = sum(as_int(row, "correct") for row in rows)
    wrong = sum(as_int(row, "wrong") for row in rows)
    return {"pairs": len(rows), "matches": matches, "correct": correct, "wrong": wrong, "precision": precision(correct, matches)}


def summarize_groups(
    *,
    pure_by_group: dict[tuple[str, str], list[dict[str, str]]],
    candidates: list[PairRow],
    metrics: list[MetricRow],
    algorithms: list[Algorithm],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for style, gate in GROUPS:
        pure_totals = aggregate_rows(pure_by_group[(style, gate)])
        group_candidates = [pair for pair in candidates if pair.style == style and pair.gate == gate]
        for algorithm in algorithms:
            subset = [row for row in metrics if row.style == style and row.gate == gate and row.algorithm == algorithm.name]
            fallback_matches = sum(row.matches for row in subset)
            fallback_correct = sum(row.correct for row in subset)
            fallback_wrong = sum(row.wrong for row in subset)
            hybrid_matches = int(pure_totals["matches"]) + fallback_matches
            hybrid_correct = int(pure_totals["correct"]) + fallback_correct
            hybrid_wrong = int(pure_totals["wrong"]) + fallback_wrong
            covered = sum(1 for row in subset if row.matches > 0)
            pair_precisions = [row.precision for row in subset if row.matches > 0]
            pair_matches = [float(row.matches) for row in subset]
            errors = [row.mean_error_px for row in subset]
            fallback_precision = precision(fallback_correct, fallback_matches)
            hybrid_precision = precision(hybrid_correct, hybrid_matches)
            pure_precision = float(pure_totals["precision"])
            coverage = covered / len(group_candidates) if group_candidates else 0.0
            recommend = int(
                fallback_matches > 0
                and fallback_precision >= args_group_precision_guardrail()
                and coverage >= 0.50
                and hybrid_precision >= pure_precision
            )
            if not group_candidates:
                reason = "no PFM gate-zero rows in this group"
            elif fallback_matches <= 0:
                reason = "no fallback support after homography filtering"
            elif fallback_precision < args_group_precision_guardrail():
                reason = "fallback precision below group guardrail"
            elif coverage < 0.50:
                reason = "fallback coverage below 50% of gate-zero rows"
            elif hybrid_precision < pure_precision:
                reason = "hybrid precision lower than pure PFM val precision"
            else:
                reason = "passes full-val gate-zero precision, coverage, and hybrid guardrails"
            rows.append(
                {
                    "scope": "full_val_all_gate_zero",
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
                    "min_pair_matches": min((row.matches for row in subset), default=0),
                    "pairs_ge_20_inliers": sum(1 for row in subset if row.matches >= 20),
                    "pairs_ge_50_inliers": sum(1 for row in subset if row.matches >= 50),
                    "mean_error_px": mean(errors),
                    "median_error_px": median([row.median_error_px for row in subset]),
                    "homography_threshold_px": algorithm.homography_threshold_px,
                    "truth_threshold_px": subset[0].truth_threshold_px if subset else "",
                    "ratio": algorithm.ratio,
                    "recommend": recommend,
                    "reason": reason,
                }
            )
    return rows


def args_group_precision_guardrail() -> float:
    return 0.98


def rank_policies(group_rows: list[dict[str, object]], algorithms: list[Algorithm]) -> list[dict[str, object]]:
    rankings: list[dict[str, object]] = []
    for algorithm in algorithms:
        subset = [row for row in group_rows if row["algorithm"] == algorithm.name]
        fallback_matches = sum(as_int(row, "fallback_matches") for row in subset)
        fallback_correct = sum(as_int(row, "fallback_correct") for row in subset)
        fallback_wrong = sum(as_int(row, "fallback_wrong") for row in subset)
        pure_matches = sum(as_int(row, "pure_matches") for row in subset)
        pure_correct = sum(as_int(row, "pure_correct") for row in subset)
        pure_wrong = sum(as_int(row, "pure_wrong") for row in subset)
        hybrid_matches = sum(as_int(row, "hybrid_matches") for row in subset)
        hybrid_correct = sum(as_int(row, "hybrid_correct") for row in subset)
        hybrid_wrong = sum(as_int(row, "hybrid_wrong") for row in subset)
        candidate_pairs = sum(as_int(row, "candidate_pairs") for row in subset)
        covered = sum(as_int(row, "covered_pairs") for row in subset)
        fallback_precisions = [as_float(row, "fallback_precision") for row in subset if as_int(row, "fallback_matches") > 0]
        hybrid_precisions = [as_float(row, "hybrid_precision") for row in subset if as_int(row, "hybrid_matches") > 0]
        group_recommends = sum(as_int(row, "recommend") for row in subset)
        fallback_precision = precision(fallback_correct, fallback_matches)
        pure_precision = precision(pure_correct, pure_matches)
        hybrid_precision = precision(hybrid_correct, hybrid_matches)
        wrong_per_candidate = fallback_wrong / candidate_pairs if candidate_pairs else math.inf
        mean_matches_per_candidate = fallback_matches / candidate_pairs if candidate_pairs else 0.0
        min_group_fallback = min(fallback_precisions) if fallback_precisions else 0.0
        min_group_hybrid = min(hybrid_precisions) if hybrid_precisions else 0.0
        rank_score = (
            min_group_fallback * 100.0
            + fallback_precision * 20.0
            + hybrid_precision * 10.0
            + math.log1p(fallback_correct)
            - wrong_per_candidate
        )
        recommend = int(group_recommends == len(GROUPS) and fallback_precision >= args_group_precision_guardrail())
        reason = (
            "all six groups pass full-val gate-zero guardrails"
            if recommend
            else f"{group_recommends}/6 groups pass full-val gate-zero guardrails"
        )
        rankings.append(
            {
                "scope": "full_val_all_gate_zero",
                "algorithm": algorithm.name,
                "family": algorithm.family,
                "groups": len(subset),
                "pure_pairs": sum(as_int(row, "pure_pairs") for row in subset),
                "candidate_pairs": candidate_pairs,
                "ok_pairs": sum(as_int(row, "ok_pairs") for row in subset),
                "covered_pairs": covered,
                "coverage": covered / candidate_pairs if candidate_pairs else 0.0,
                "fallback_matches": fallback_matches,
                "fallback_correct": fallback_correct,
                "fallback_wrong": fallback_wrong,
                "fallback_precision": fallback_precision,
                "pure_matches": pure_matches,
                "pure_correct": pure_correct,
                "pure_wrong": pure_wrong,
                "pure_precision": pure_precision,
                "hybrid_matches": hybrid_matches,
                "hybrid_correct": hybrid_correct,
                "hybrid_wrong": hybrid_wrong,
                "hybrid_precision": hybrid_precision,
                "hybrid_precision_delta": hybrid_precision - pure_precision,
                "min_group_fallback_precision": min_group_fallback,
                "mean_group_fallback_precision": mean(fallback_precisions),
                "min_group_hybrid_precision": min_group_hybrid,
                "wrong_per_candidate": wrong_per_candidate,
                "mean_matches_per_candidate": mean_matches_per_candidate,
                "rank_score": rank_score,
                "recommend": recommend,
                "reason": reason,
            }
        )
    return sorted(rankings, key=lambda row: (as_int(row, "recommend"), as_float(row, "rank_score")), reverse=True)


def hybrid_route_rows(group_rows: list[dict[str, object]], rankings: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    best_algorithm = rankings[0]["algorithm"] if rankings else ""
    comparison_algorithms = {
        best_algorithm,
        "RootSIFT-FLANN-r0.75+HomographyUSAC-t2",
        "RootSIFT-FLANN-r0.80+HomographyUSAC-t2",
    }
    for row in group_rows:
        if row["algorithm"] not in comparison_algorithms:
            continue
        rows.append(
            {
                "style": row["style"],
                "gate": row["gate"],
                "route": "pure_pfm_fullval_plus_external_gate_zero_fallback",
                "algorithm": row["algorithm"],
                "pure_pairs": row["pure_pairs"],
                "gate_zero_pairs": row["candidate_pairs"],
                "fallback_pairs": row["covered_pairs"],
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
                "notes": "Full-val pure-PFM nonzero rows plus external fallback on full-val pure-PFM gate-zero rows",
            }
        )
    return rows


def markdown_table(rows: list[dict[str, object]], fields: list[str], limit: int | None = None) -> list[str]:
    selected = rows[:limit] if limit is not None else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in selected:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def load_reference_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def markdown_summary(
    *,
    args: argparse.Namespace,
    candidates: list[PairRow],
    pure_by_group: dict[tuple[str, str], list[dict[str, str]]],
    group_rows: list[dict[str, object]],
    rankings: list[dict[str, object]],
    skipped: list[dict[str, str]],
) -> str:
    best = rankings[0] if rankings else {}
    strict = next((row for row in rankings if row.get("algorithm") == "RootSIFT-FLANN-r0.75+HomographyUSAC-t2"), None)
    stage8 = next(
        (
            row
            for row in load_reference_rows(STAGE8_ROUTE / "hybrid_route_metrics.csv")
            if row.get("style") == "overall" or row.get("gate") == "all"
        ),
        None,
    )
    stage9 = next(
        (
            row
            for row in load_reference_rows(STAGE9_DIR / "policy_ranking.csv")
            if row.get("algorithm") == "RootSIFT-FLANN-r0.75+HomographyUSAC-t2"
        ),
        None,
    )
    lines = [
        "# Matcher Algorithm Iteration Agent14 Stage10",
        "",
        "## Scope",
        "",
        "- This is a **hybrid/external fallback** replay on full-val pure-PFM gate-zero rows.",
        "- It is not a pure learned PFM metric and should not be reported as pure PFM improvement.",
        "- The pure-PFM route parameters come from `runs/cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234`.",
        "- Five groups use the targetcontrast selected params; `timestamp/compound` uses the current lowcontrast route params.",
        f"- Full-val split root: `{rel(args.split_root)}`.",
        f"- Gate-zero rows are evaluated without sampling."
        if args.candidate_sample_per_group <= 0
        else f"- Gate-zero rows were sampled: seed `{args.seed}`, sample_per_group `{args.candidate_sample_per_group}`.",
        "",
        "## Pure PFM Full-Val Gate-Zero Rows",
        "",
        "| style | gate | val pairs | pure matches | pure correct | pure precision | gate-zero pairs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for style, gate in GROUPS:
        pure_totals = aggregate_rows(pure_by_group[(style, gate)])
        group_candidates = [pair for pair in candidates if pair.style == style and pair.gate == gate]
        lines.append(
            f"| {style} | {gate} | {pure_totals['pairs']} | {pure_totals['matches']} | "
            f"{pure_totals['correct']} | {float(pure_totals['precision']):.6f} | {len(group_candidates)} |"
        )

    lines.extend(["", "## Policy Ranking", ""])
    lines.extend(
        markdown_table(
            rankings,
            [
                "algorithm",
                "candidate_pairs",
                "covered_pairs",
                "fallback_matches",
                "fallback_correct",
                "fallback_wrong",
                "fallback_precision",
                "min_group_fallback_precision",
                "coverage",
                "hybrid_precision",
                "recommend",
            ],
        )
    )

    lines.extend(["", "## Group-Level Risk", ""])
    risk_algorithms = {"RootSIFT-FLANN-r0.75+HomographyUSAC-t2", "RootSIFT-FLANN-r0.80+HomographyUSAC-t2"}
    if best:
        risk_algorithms.add(str(best["algorithm"]))
    risk_rows = [row for row in group_rows if row["algorithm"] in risk_algorithms]
    lines.extend(
        markdown_table(
            risk_rows,
            [
                "style",
                "gate",
                "algorithm",
                "candidate_pairs",
                "covered_pairs",
                "fallback_matches",
                "fallback_correct",
                "fallback_wrong",
                "fallback_precision",
                "hybrid_precision",
                "recommend",
            ],
        )
    )

    lines.extend(["", "## References", ""])
    lines.append(
        "Stage8 fixed-test all-gate-zero `r0.80/H2` reference: 60108/60431 fallback correct, "
        "precision 0.994655; combined route 61437/62255 = 0.986860."
    )
    if stage9:
        lines.append(
            "Stage9 cache-heldout best `r0.75/H2`: "
            f"{stage9['correct']}/{stage9['matches']} correct, {stage9['wrong']} wrong, "
            f"precision {float(stage9['precision']):.6f}, min_group_precision {float(stage9['min_group_precision']):.6f}."
        )
    if stage8:
        lines.append("Stage8 CSV reference was available; fixed values above are kept explicit for readability.")

    lines.extend(["", "## Recommendation", ""])
    if not rankings:
        lines.append("No fallback policy could be ranked.")
    else:
        lines.append(
            f"Best Stage10 full-val external policy: `{best['algorithm']}` with "
            f"{best['fallback_correct']}/{best['fallback_matches']} fallback correct, "
            f"{best['fallback_wrong']} wrong, fallback precision {as_float(best, 'fallback_precision'):.6f}, "
            f"minimum group fallback precision {as_float(best, 'min_group_fallback_precision'):.6f}, "
            f"coverage {as_float(best, 'coverage'):.6f}."
        )
        if strict:
            lines.append(
                f"`RootSIFT-FLANN-r0.75+HomographyUSAC-t2` full-val result: "
                f"{strict['fallback_correct']}/{strict['fallback_matches']} correct, {strict['fallback_wrong']} wrong, "
                f"precision {as_float(strict, 'fallback_precision'):.6f}, "
                f"min_group_precision {as_float(strict, 'min_group_fallback_precision'):.6f}, "
                f"hybrid precision {as_float(strict, 'hybrid_precision'):.6f}."
            )
        stage8_compatible = next((row for row in rankings if row.get("algorithm") == "RootSIFT-FLANN-r0.80+HomographyUSAC-t2"), None)
        if stage8_compatible:
            lines.append(
                f"Stage8-compatible `r0.80/H2` full-val result: "
                f"{stage8_compatible['fallback_correct']}/{stage8_compatible['fallback_matches']} correct, "
                f"{stage8_compatible['fallback_wrong']} wrong, precision {as_float(stage8_compatible, 'fallback_precision'):.6f}, "
                f"min_group_precision {as_float(stage8_compatible, 'min_group_fallback_precision'):.6f}."
            )
        if strict and stage8_compatible and best.get("algorithm") == strict.get("algorithm"):
            lines.append(
                "Policy call: recommend replacing broad all-gate-zero fallback default `r0.80/H2` with `r0.75/H2` "
                "for the next hybrid route integration, while retaining `r0.80/H2` as an ablation/compatibility row."
            )
        elif strict and stage8_compatible:
            lines.append(
                "Policy call: do not replace `r0.80/H2` solely from Stage9; use the Stage10 full-val ranking above. "
                "If the main agent chooses a broad fallback, integrate only the policy that wins on full-val support and group risk."
            )
        lines.append(
            "Integration must keep external fallback decisions labeled separately from pure PFM. The current pure PFM route remains "
            "`runs/cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234`."
        )

    if skipped:
        lines.extend(["", "## Skipped Algorithms", "", "| algorithm | reason |", "|---|---|"])
        for row in skipped:
            lines.append(f"| {row['algorithm']} | {row['reason']} |")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `candidate_pairs.csv`",
            "- `per_pair_metrics.csv`",
            "- `fallback_metrics.csv`",
            "- `per_group_policy_summary.csv`",
            "- `policy_ranking.csv`",
            "- `hybrid_route_fullval_comparison.csv`",
            "- `pure_pfm_fullval/eval/*/*/summary.csv`",
            "- `summary.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def append_progress(output_dir: Path, message: str) -> None:
    with (output_dir / "progress.md").open("a", encoding="utf-8") as handle:
        handle.write(f"- {message}\n")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route_params = load_route_params(args.selected_route)
    run_pfm_fullval(args, route_params)
    candidates, pure_by_group = collect_candidate_pairs(args, route_params)
    algorithms, skipped = make_algorithms(args)
    metrics: list[MetricRow] = []
    for algorithm in algorithms:
        for index, pair in enumerate(candidates, start=1):
            row = evaluate_one(args, algorithm, pair)
            metrics.append(row)
            if args.verbose or index == len(candidates) or index % args.progress_every == 0:
                print(
                    f"{algorithm.name:45s} {index:04d}/{len(candidates):04d} "
                    f"{pair.style}/{pair.gate:9s} m={row.matches} c={row.correct} w={row.wrong} p={row.precision:.3f}",
                    flush=True,
                )
    group_rows = summarize_groups(pure_by_group=pure_by_group, candidates=candidates, metrics=metrics, algorithms=algorithms)
    rankings = rank_policies(group_rows, algorithms)
    hybrid_rows = hybrid_route_rows(group_rows, rankings)

    write_csv(args.output_dir / "candidate_pairs.csv", [asdict(pair) for pair in candidates], PAIR_FIELDS)
    metric_dicts = [asdict(row) for row in metrics]
    write_csv(args.output_dir / "per_pair_metrics.csv", metric_dicts, METRIC_FIELDS)
    write_csv(args.output_dir / "fallback_metrics.csv", metric_dicts, METRIC_FIELDS)
    write_csv(args.output_dir / "per_group_policy_summary.csv", group_rows, GROUP_FIELDS)
    write_csv(args.output_dir / "policy_ranking.csv", rankings, RANKING_FIELDS)
    write_csv(args.output_dir / "hybrid_route_fullval_comparison.csv", hybrid_rows, HYBRID_FIELDS)
    (args.output_dir / "summary.md").write_text(
        markdown_summary(
            args=args,
            candidates=candidates,
            pure_by_group=pure_by_group,
            group_rows=group_rows,
            rankings=rankings,
            skipped=skipped,
        ),
        encoding="utf-8",
    )
    append_progress(
        args.output_dir,
        f"Ran Stage10: {len(candidates)} full-val gate-zero rows, {len(algorithms)} algorithms, {len(metrics)} metric rows.",
    )


def self_test() -> None:
    assert precision(2, 4) == 0.5
    assert route_case_type(0) == "pfm_gate_zero_external_fallback_candidate"
    assert route_case_type(3) == "pfm_nonzero_not_replayed"
    sample = [
        {
            "algorithm": "a",
            "fallback_matches": 10,
            "fallback_correct": 10,
            "fallback_wrong": 0,
            "pure_matches": 1,
            "pure_correct": 1,
            "pure_wrong": 0,
            "hybrid_matches": 11,
            "hybrid_correct": 11,
            "hybrid_wrong": 0,
            "candidate_pairs": 1,
            "covered_pairs": 1,
            "fallback_precision": 1.0,
            "hybrid_precision": 1.0,
            "recommend": 1,
        }
        for _ in GROUPS
    ]
    ranking = rank_policies(sample, [Algorithm("a", "classical", object(), 2.0, 0.8, 4)])
    assert ranking[0]["recommend"] == 1
    assert ranking[0]["fallback_precision"] == 1.0
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--selected-route", type=Path, default=DEFAULT_SELECTED_ROUTE)
    parser.add_argument("--python-exe", type=Path, default=PYTHON_EXE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--pfm-limit-pairs", type=int, default=0)
    parser.add_argument("--candidate-sample-per-group", type=int, default=0)
    parser.add_argument("--force-pfm-eval", action="store_true")
    parser.add_argument("--max-keypoints-pfm", type=int, default=4096)
    parser.add_argument("--descriptor-topk", type=int, default=32)
    parser.add_argument("--keypoint-spatial-bins", type=int, default=0)
    parser.add_argument("--geometry-filter", choices=["none", "local", "affine"], default="local")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--learned-max-keypoints", type=int, default=2048)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--truth-threshold-px", type=float, default=3.0)
    parser.add_argument("--no-lightglue", action="store_true")
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--limit-algorithms", nargs="*", default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.split_root = args.split_root.resolve()
    args.selected_route = args.selected_route.resolve()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
