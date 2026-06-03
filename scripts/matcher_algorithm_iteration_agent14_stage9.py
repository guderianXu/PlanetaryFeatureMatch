#!/usr/bin/env python3
"""Agent14 stage9 sidecar: held-out external matcher validation.

This script evaluates hybrid/external fallback matchers only. It does not train
PFM and does not modify the main evaluation source. The primary goal is to
stress-test the Stage8 all-gate-zero RootSIFT fallback idea on cache-held-out
rows across the six style/gate groups, then compare stricter RootSIFT variants
and optional OpenCV/LightGlue alternatives.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE7_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent13_stage7.py"
STAGE8_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage8"
DEFAULT_BASELINE_ROUTE = PROJECT_ROOT / "runs" / "cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent14_stage9"

GROUPS = [(style, gate) for style in ("numeric", "timestamp") for gate in ("rotate", "viewpoint", "compound")]
GATE_CACHE_ROOTS = {
    "rotate": PROJECT_ROOT / "img" / "Rotate_1024",
    "viewpoint": PROJECT_ROOT / "img" / "Viewpoint_1024",
    "compound": PROJECT_ROOT / "img" / "CompoundViewpoint_1024",
}

PAIR_FIELDS = [
    "style",
    "gate",
    "split",
    "case_type",
    "pair_pt",
    "cache_pair_pt",
    "source_name",
    "pair_name",
    "sample_seed",
    "sample_rank",
    "source_pool_pairs",
    "heldout_reason",
]

METRIC_FIELDS = [
    *PAIR_FIELDS[:8],
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
    "candidate_pairs",
    "ok_pairs",
    "covered_pairs",
    "coverage",
    "matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "median_pair_precision",
    "mean_matches_per_pair",
    "median_matches_per_pair",
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
    "candidate_pairs",
    "ok_pairs",
    "covered_pairs",
    "coverage",
    "matches",
    "correct",
    "wrong",
    "precision",
    "min_group_precision",
    "mean_group_precision",
    "min_group_matches",
    "wrong_per_pair",
    "mean_matches_per_pair",
    "homography_threshold_px",
    "ratio",
    "rank_score",
    "recommend",
    "reason",
]

ROTATION_FIELDS = [
    "style",
    "rotation_deg",
    "source_name",
    "pair_name",
    "cache_pair_pt",
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
    "visualization",
    "message",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S7 = load_module(STAGE7_SCRIPT, "agent13_stage7_for_stage9")
A4 = S7.A4


@dataclass(frozen=True)
class PairRow:
    style: str
    gate: str
    split: str
    case_type: str
    pair_pt: str
    cache_pair_pt: str
    source_name: str
    pair_name: str
    sample_seed: int
    sample_rank: int
    source_pool_pairs: int
    heldout_reason: str


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
    cache_pair_pt: str
    source_name: str
    pair_name: str
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


@dataclass(frozen=True)
class RotationMetricRow:
    style: str
    rotation_deg: int
    source_name: str
    pair_name: str
    cache_pair_pt: str
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
    visualization: str
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


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def source_style(source_name: str) -> str:
    return "timestamp" if "NAS" in source_name or "T" in source_name else "numeric"


def group_summary(route: Path, style: str, gate: str) -> Path:
    return route / "eval" / style / gate / "summary.csv"


def cache_key(path: Path) -> tuple[str, str, str]:
    return (path.parents[1].name, path.parent.name, path.name)


def summary_pair_to_cache(pair_pt: str, gate: str) -> Path:
    path = Path(pair_pt)
    root = GATE_CACHE_ROOTS[gate]
    return root / path.parent.name / path.name


def fixed_eval_exclusions(baseline_route: Path) -> set[tuple[str, str, str]]:
    excluded: set[tuple[str, str, str]] = set()
    for style, gate in GROUPS:
        path = group_summary(baseline_route, style, gate)
        if not path.exists():
            continue
        for row in read_csv(path):
            excluded.add(cache_key(summary_pair_to_cache(row["pair_pt"], gate)))
    stage8_pairs = STAGE8_DIR / "candidate_pairs.csv"
    if stage8_pairs.exists():
        for row in read_csv(stage8_pairs):
            excluded.add(cache_key(PROJECT_ROOT / row["cache_pair_pt"]))
    return excluded


def collect_cache_pool(style: str, gate: str, excluded: set[tuple[str, str, str]]) -> list[Path]:
    root = GATE_CACHE_ROOTS[gate]
    paths = []
    for path in sorted(root.glob("source_*/*.pt")):
        if source_style(path.parent.name) != style:
            continue
        if cache_key(path) in excluded:
            continue
        paths.append(path)
    return paths


def sample_candidates(args: argparse.Namespace) -> list[PairRow]:
    rng = random.Random(args.seed)
    excluded = fixed_eval_exclusions(args.baseline_route)
    candidates: list[PairRow] = []
    for style, gate in GROUPS:
        pool = collect_cache_pool(style, gate, excluded)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        selected = sorted(shuffled[: args.sample_per_group])
        for rank, path in enumerate(selected, start=1):
            candidates.append(
                PairRow(
                    style=style,
                    gate=gate,
                    split="cache_heldout_excluding_fixed_eval",
                    case_type="external_matcher_policy_validation",
                    pair_pt=rel(path),
                    cache_pair_pt=rel(path),
                    source_name=path.parent.name,
                    pair_name=path.name,
                    sample_seed=args.seed,
                    sample_rank=rank,
                    source_pool_pairs=len(pool),
                    heldout_reason="sampled from img cache after excluding baseline fixed-test eval and Stage8 candidate pairs",
                )
            )
    return candidates


def choose_device(requested: str) -> str:
    return S7.choose_device(requested)


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
            for ratio, threshold in ((0.80, 2.0), (0.75, 2.0), (0.80, 1.5), (0.80, 3.0)):
                name = f"RootSIFT-FLANN-r{ratio:.2f}+HomographyUSAC-t{threshold:g}"
                algorithms.append(
                    Algorithm(
                        name=name,
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
        try:
            algorithms.append(
                Algorithm(
                    name="ORB-BF-r0.80+HomographyUSAC-t3",
                    family="classical_secondary",
                    matcher=BinaryDescriptorMatcher(
                        detector_name="ORB",
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
            skipped.append({"algorithm": "ORB-BF-r0.80+HomographyUSAC-t3", "reason": f"{type(exc).__name__}: {exc}"})
        try:
            algorithms.append(
                Algorithm(
                    name="AKAZE-BF-r0.80+HomographyUSAC-t3",
                    family="classical_secondary",
                    matcher=BinaryDescriptorMatcher(
                        detector_name="AKAZE",
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
            skipped.append({"algorithm": "AKAZE-BF-r0.80+HomographyUSAC-t3", "reason": f"{type(exc).__name__}: {exc}"})
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
    return algorithms, skipped


def evaluate_arrays(
    *,
    args: argparse.Namespace,
    algorithm: Algorithm,
    image_a: np.ndarray,
    image_b: np.ndarray,
    warp_a_to_b,
    valid_mask,
) -> tuple[str, int, int, int, int, int, int, float, float, float, str]:
    try:
        raw = algorithm.matcher.match(image_a, image_b)
        output = S7.homography_inliers(raw, algorithm.homography_threshold_px, algorithm.min_inliers)
        matches, correct, wrong, pair_precision, mean_error, median_error = A4.compute_metrics(
            output.points_a,
            output.points_b,
            warp_a_to_b,
            valid_mask,
            threshold_px=args.truth_threshold_px,
        )
        return (
            "ok",
            raw.keypoints_a,
            raw.keypoints_b,
            raw.raw_matches,
            matches,
            correct,
            wrong,
            pair_precision,
            mean_error,
            median_error,
            "",
        )
    except Exception as exc:
        return ("error", 0, 0, 0, 0, 0, 0, 0.0, math.nan, math.nan, f"{type(exc).__name__}: {exc}")


def evaluate_one(args: argparse.Namespace, algorithm: Algorithm, pair: PairRow) -> MetricRow:
    image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(PROJECT_ROOT / pair.cache_pair_pt)
    status, kpa, kpb, raw, matches, correct, wrong, pair_precision, mean_error, median_error, message = evaluate_arrays(
        args=args,
        algorithm=algorithm,
        image_a=image_a,
        image_b=image_b,
        warp_a_to_b=warp_a_to_b,
        valid_mask=valid_mask,
    )
    return MetricRow(
        style=pair.style,
        gate=pair.gate,
        split=pair.split,
        case_type=pair.case_type,
        pair_pt=pair.pair_pt,
        cache_pair_pt=pair.cache_pair_pt,
        source_name=pair.source_name,
        pair_name=pair.pair_name,
        algorithm=algorithm.name,
        family=algorithm.family,
        status=status,
        keypoints_a=kpa,
        keypoints_b=kpb,
        raw_matches=raw,
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
        message=message,
    )


def summarize_groups(metrics: list[MetricRow], algorithms: list[Algorithm], pairs: list[PairRow]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for style, gate in GROUPS:
        group_pairs = [pair for pair in pairs if pair.style == style and pair.gate == gate]
        for algorithm in algorithms:
            subset = [row for row in metrics if row.style == style and row.gate == gate and row.algorithm == algorithm.name]
            matches = sum(row.matches for row in subset)
            correct = sum(row.correct for row in subset)
            wrong = sum(row.wrong for row in subset)
            covered = sum(1 for row in subset if row.matches > 0)
            pair_precisions = [row.precision for row in subset if row.matches > 0]
            pair_matches = [float(row.matches) for row in subset]
            errors = [row.mean_error_px for row in subset]
            recommend = int(matches > 0 and precision(correct, matches) >= 0.98 and covered >= max(1, len(group_pairs) // 2))
            if not subset:
                reason = "no metrics"
            elif matches <= 0:
                reason = "no homography-filtered matches"
            elif precision(correct, matches) < 0.98:
                reason = "held-out fallback precision below 0.98 guardrail"
            elif covered < max(1, len(group_pairs) // 2):
                reason = "low pair coverage"
            else:
                reason = "passes held-out precision and coverage guardrails"
            rows.append(
                {
                    "scope": "cache_heldout_excluding_fixed_eval",
                    "style": style,
                    "gate": gate,
                    "algorithm": algorithm.name,
                    "family": algorithm.family,
                    "candidate_pairs": len(group_pairs),
                    "ok_pairs": sum(1 for row in subset if row.status == "ok"),
                    "covered_pairs": covered,
                    "coverage": covered / len(group_pairs) if group_pairs else 0.0,
                    "matches": matches,
                    "correct": correct,
                    "wrong": wrong,
                    "precision": precision(correct, matches),
                    "mean_pair_precision": mean(pair_precisions),
                    "median_pair_precision": median(pair_precisions),
                    "mean_matches_per_pair": mean(pair_matches),
                    "median_matches_per_pair": median(pair_matches),
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


def rank_policies(group_rows: list[dict[str, object]], algorithms: list[Algorithm]) -> list[dict[str, object]]:
    rankings: list[dict[str, object]] = []
    for algorithm in algorithms:
        subset = [row for row in group_rows if row["algorithm"] == algorithm.name]
        matches = sum(as_int(row, "matches") for row in subset)
        correct = sum(as_int(row, "correct") for row in subset)
        wrong = sum(as_int(row, "wrong") for row in subset)
        pairs = sum(as_int(row, "candidate_pairs") for row in subset)
        covered = sum(as_int(row, "covered_pairs") for row in subset)
        precisions = [as_float(row, "precision") for row in subset if as_int(row, "matches") > 0]
        min_group_precision = min(precisions) if precisions else 0.0
        min_group_matches = min((as_int(row, "matches") for row in subset), default=0)
        group_recommends = sum(as_int(row, "recommend") for row in subset)
        overall_precision = precision(correct, matches)
        wrong_per_pair = wrong / pairs if pairs else math.inf
        mean_matches_per_pair = matches / pairs if pairs else 0.0
        rank_score = (min_group_precision * 100.0) + (overall_precision * 10.0) + math.log1p(correct) - (wrong_per_pair * 0.5)
        recommend = int(group_recommends == len(GROUPS) and overall_precision >= 0.98)
        if recommend:
            reason = "all six groups pass held-out precision and coverage guardrails"
        else:
            reason = f"{group_recommends}/6 groups pass held-out guardrails"
        rankings.append(
            {
                "scope": "cache_heldout_excluding_fixed_eval",
                "algorithm": algorithm.name,
                "family": algorithm.family,
                "groups": len(subset),
                "candidate_pairs": pairs,
                "ok_pairs": sum(as_int(row, "ok_pairs") for row in subset),
                "covered_pairs": covered,
                "coverage": covered / pairs if pairs else 0.0,
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": overall_precision,
                "min_group_precision": min_group_precision,
                "mean_group_precision": mean(precisions),
                "min_group_matches": min_group_matches,
                "wrong_per_pair": wrong_per_pair,
                "mean_matches_per_pair": mean_matches_per_pair,
                "homography_threshold_px": algorithm.homography_threshold_px,
                "ratio": algorithm.ratio,
                "rank_score": rank_score,
                "recommend": recommend,
                "reason": reason,
            }
        )
    return sorted(rankings, key=lambda row: (as_int(row, "recommend"), as_float(row, "rank_score")), reverse=True)


def rotation_warp(height: int, width: int, rotation_deg: int):
    import torch

    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij")
    if rotation_deg == 90:
        bx = float(height - 1) - yy
        by = xx
    elif rotation_deg == 180:
        bx = float(width - 1) - xx
        by = float(height - 1) - yy
    elif rotation_deg == 270:
        bx = yy
        by = float(width - 1) - xx
    else:
        raise ValueError(rotation_deg)
    warp = torch.stack([bx, by], dim=-1).contiguous()
    valid = torch.ones((height, width), dtype=torch.bool)
    return warp, valid


def rotate_image(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    return A4.rotate_image(image, rotation_deg)


def draw_rotation_visualization(
    image_a: np.ndarray,
    image_b: np.ndarray,
    algorithm: Algorithm,
    output_path: Path,
) -> None:
    import cv2

    raw = algorithm.matcher.match(image_a, image_b)
    inliers = S7.homography_inliers(raw, algorithm.homography_threshold_px, algorithm.min_inliers)
    h = max(image_a.shape[0], image_b.shape[0])
    w = image_a.shape[1] + image_b.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: image_a.shape[0], : image_a.shape[1]] = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
    canvas[: image_b.shape[0], image_a.shape[1] :] = cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(123)
    for index in range(min(80, inliers.points_a.shape[0])):
        ax, ay = inliers.points_a[index]
        bx, by = inliers.points_b[index]
        color = tuple(int(v) for v in rng.integers(64, 255, size=3))
        cv2.circle(canvas, (int(round(ax)), int(round(ay))), 2, color, -1)
        cv2.circle(canvas, (int(round(bx + image_a.shape[1])), int(round(by))), 2, color, -1)
        cv2.line(canvas, (int(round(ax)), int(round(ay))), (int(round(bx + image_a.shape[1])), int(round(by))), color, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def select_rotation_cases(pairs: list[PairRow]) -> dict[str, PairRow]:
    cases = {}
    for style in ("numeric", "timestamp"):
        style_pairs = [pair for pair in pairs if pair.style == style and pair.gate == "rotate"]
        if style_pairs:
            cases[style] = style_pairs[0]
    return cases


def evaluate_rotation_cases(args: argparse.Namespace, algorithms: list[Algorithm], pairs: list[PairRow]) -> list[RotationMetricRow]:
    cases = select_rotation_cases(pairs)
    rows: list[RotationMetricRow] = []
    root_algo = next((algo for algo in algorithms if algo.name == "RootSIFT-FLANN-r0.80+HomographyUSAC-t2"), None)
    for style, pair in cases.items():
        image_a, _, _, _ = A4.load_pair(PROJECT_ROOT / pair.cache_pair_pt)
        height, width = image_a.shape
        for rotation_deg in (90, 180, 270):
            image_b = rotate_image(image_a, rotation_deg)
            warp, valid = rotation_warp(height, width, rotation_deg)
            for algorithm in algorithms:
                status, kpa, kpb, raw, matches, correct, wrong, pair_precision, mean_error, median_error, message = evaluate_arrays(
                    args=args,
                    algorithm=algorithm,
                    image_a=image_a,
                    image_b=image_b,
                    warp_a_to_b=warp,
                    valid_mask=valid,
                )
                visualization = ""
                if root_algo is not None and algorithm.name == root_algo.name:
                    viz_path = args.output_dir / "rotation_visualizations" / f"{style}_rot{rotation_deg}_{safe_filename(algorithm.name)}.png"
                    try:
                        draw_rotation_visualization(image_a, image_b, algorithm, viz_path)
                        visualization = rel(viz_path)
                    except Exception as exc:
                        message = (message + "; " if message else "") + f"visualization_error={type(exc).__name__}: {exc}"
                rows.append(
                    RotationMetricRow(
                        style=style,
                        rotation_deg=rotation_deg,
                        source_name=pair.source_name,
                        pair_name=pair.pair_name,
                        cache_pair_pt=pair.cache_pair_pt,
                        algorithm=algorithm.name,
                        family=algorithm.family,
                        status=status,
                        keypoints_a=kpa,
                        keypoints_b=kpb,
                        raw_matches=raw,
                        matches=matches,
                        correct=correct,
                        wrong=wrong,
                        precision=pair_precision,
                        mean_error_px=mean_error,
                        median_error_px=median_error,
                        homography_threshold_px=algorithm.homography_threshold_px,
                        truth_threshold_px=args.truth_threshold_px,
                        ratio=algorithm.ratio,
                        visualization=visualization,
                        message=message,
                    )
                )
    return rows


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


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


def markdown_summary(
    *,
    args: argparse.Namespace,
    pairs: list[PairRow],
    group_rows: list[dict[str, object]],
    rankings: list[dict[str, object]],
    rotation_rows: list[RotationMetricRow],
    skipped: list[dict[str, str]],
) -> str:
    stage8_ref = next(
        (row for row in read_csv(STAGE8_DIR / "per_group_policy_summary.csv") if row.get("scope") == "all_gate_zero" and row.get("style") == "overall" and row.get("algorithm") == "RootSIFT-FLANN-r0.80+HomographyUSAC-t2"),
        None,
    ) if (STAGE8_DIR / "per_group_policy_summary.csv").exists() else None
    best = rankings[0] if rankings else {}
    stage8_policy = next((row for row in rankings if row.get("algorithm") == "RootSIFT-FLANN-r0.80+HomographyUSAC-t2"), None)
    strict = next((row for row in rankings if row.get("algorithm") == "RootSIFT-FLANN-r0.75+HomographyUSAC-t2"), None)

    lines = [
        "# Matcher Algorithm Iteration Agent14 Stage9",
        "",
        "## Scope",
        "",
        "- This is a hybrid/external matcher validation run, not a pure learned PFM route.",
        "- No training or main evaluator/source edits were performed.",
        "- Stage9 validates external matcher behavior on cache-held-out rows across all six style/gate groups.",
        "- Fixed-test eval rows and Stage8 candidate pairs are excluded from the Stage9 cache sample.",
        f"- Sample size: `{args.sample_per_group}` pairs per group requested, seed `{args.seed}`.",
        f"- Truth threshold: `{args.truth_threshold_px}` px; min homography inliers: `{args.min_inliers}`.",
        "",
        "## Candidate Pairs",
        "",
        "| style | gate | sampled pairs | source pool after exclusions |",
        "|---|---|---:|---:|",
    ]
    for style, gate in GROUPS:
        subset = [pair for pair in pairs if pair.style == style and pair.gate == gate]
        pool = subset[0].source_pool_pairs if subset else 0
        lines.append(f"| {style} | {gate} | {len(subset)} | {pool} |")

    lines.extend(["", "## Policy Ranking", ""])
    lines.extend(
        markdown_table(
            rankings,
            [
                "algorithm",
                "candidate_pairs",
                "matches",
                "correct",
                "wrong",
                "precision",
                "min_group_precision",
                "coverage",
                "recommend",
            ],
        )
    )

    lines.extend(["", "## Group Risks", ""])
    risk_rows = [
        row
        for row in group_rows
        if row["algorithm"] in {"RootSIFT-FLANN-r0.80+HomographyUSAC-t2", best.get("algorithm", "")}
    ]
    lines.extend(
        markdown_table(
            risk_rows,
            ["style", "gate", "algorithm", "candidate_pairs", "matches", "correct", "wrong", "precision", "coverage", "recommend"],
        )
    )

    lines.extend(["", "## Stage8 Reference", ""])
    if stage8_ref:
        lines.append(
            "Stage8 fixed-test all-gate-zero `RootSIFT-FLANN-r0.80+HomographyUSAC-t2` reference: "
            f"{stage8_ref['correct']}/{stage8_ref['matches']} correct, {stage8_ref['wrong']} wrong, "
            f"fallback precision {float(stage8_ref['precision']):.6f}, combined precision {float(stage8_ref['combined_precision']):.6f}."
        )
    else:
        lines.append("Stage8 fixed-test reference CSV was not found.")

    lines.extend(["", "## Recommendation", ""])
    if not rankings:
        lines.append("No external matcher policy could be ranked.")
    else:
        lines.append(
            f"Best Stage9 held-out external matcher by guardrail ranking: `{best['algorithm']}` with "
            f"{best['correct']}/{best['matches']} correct, {best['wrong']} wrong, overall precision "
            f"{as_float(best, 'precision'):.6f}, and minimum group precision {as_float(best, 'min_group_precision'):.6f}."
        )
        if stage8_policy:
            lines.append(
                f"Stage8 policy candidate on Stage9 held-out rows: `{stage8_policy['algorithm']}` with "
                f"{stage8_policy['correct']}/{stage8_policy['matches']} correct, {stage8_policy['wrong']} wrong, "
                f"overall precision {as_float(stage8_policy, 'precision'):.6f}, minimum group precision "
                f"{as_float(stage8_policy, 'min_group_precision'):.6f}, recommend={stage8_policy['recommend']}."
            )
        if strict:
            lines.append(
                f"Stricter ratio candidate: `{strict['algorithm']}` with {strict['correct']}/{strict['matches']} correct, "
                f"{strict['wrong']} wrong, precision {as_float(strict, 'precision'):.6f}, "
                f"minimum group precision {as_float(strict, 'min_group_precision'):.6f}."
            )
        if (
            stage8_policy
            and strict
            and best.get("algorithm") == strict.get("algorithm")
            and strict.get("algorithm") != stage8_policy.get("algorithm")
        ):
            lines.append(
                "Policy call: for the next route-level full-val replay, prefer the stricter "
                "`RootSIFT-FLANN-r0.75+HomographyUSAC-t2` policy as the default broad fallback candidate. "
                "It gives up some recall versus Stage8 `r0.80/H2`, but reduced wrong matches and improved the "
                "minimum group precision on this held-out cache sample."
            )
            lines.append(
                "Keep Stage8 `r0.80/H2` as a comparison/compatibility candidate, not as the only broad policy."
            )
        elif stage8_policy and as_int(stage8_policy, "recommend"):
            lines.append(
                "Recommendation: keep the Stage8 all-gate-zero policy as a hybrid/external matcher candidate, "
                "but keep it separately labeled from pure PFM. Promote only after a true route-level full-val replay "
                "generates targetcontrast gate-zero rows for all six groups."
            )
        else:
            lines.append(
                "Recommendation: do not broaden directly from Stage8 fixed-test to production routing. Use the best stricter "
                "Stage9 policy or keep the narrower Stage7 timestamp/compound rescue until six-group full-val gate-zero rows exist."
            )

    lines.extend(["", "## Rotation Cases", ""])
    if rotation_rows:
        rot_summary: list[dict[str, object]] = []
        for style in ("numeric", "timestamp"):
            for rotation_deg in (90, 180, 270):
                for algorithm in sorted({row.algorithm for row in rotation_rows}):
                    subset = [row for row in rotation_rows if row.style == style and row.rotation_deg == rotation_deg and row.algorithm == algorithm]
                    if not subset:
                        continue
                    row = subset[0]
                    rot_summary.append(
                        {
                            "style": style,
                            "rotation_deg": rotation_deg,
                            "algorithm": algorithm,
                            "matches": row.matches,
                            "correct": row.correct,
                            "wrong": row.wrong,
                            "precision": row.precision,
                        }
                    )
        lines.extend(markdown_table(rot_summary, ["style", "rotation_deg", "algorithm", "matches", "correct", "wrong", "precision"]))
    else:
        lines.append("Rotation cases were skipped because no sampled rotate pairs were available.")

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
            "- `per_group_policy_summary.csv`",
            "- `policy_ranking.csv`",
            "- `rotation_case_metrics.csv`",
            "- `rotation_visualizations/*.png`",
            "- `summary.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def update_progress(output_dir: Path, message: str) -> None:
    progress = output_dir / "progress.md"
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(f"- {message}\n")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = sample_candidates(args)
    algorithms, skipped = make_algorithms(args)
    metrics: list[MetricRow] = []
    for algorithm in algorithms:
        for index, pair in enumerate(pairs, start=1):
            row = evaluate_one(args, algorithm, pair)
            metrics.append(row)
            if args.verbose:
                print(
                    f"{algorithm.name:45s} {index:03d}/{len(pairs):03d} {pair.style}/{pair.gate:9s} "
                    f"{pair.source_name}/{pair.pair_name} m={row.matches} c={row.correct} w={row.wrong} p={row.precision:.3f}"
                )
    group_rows = summarize_groups(metrics, algorithms, pairs)
    rankings = rank_policies(group_rows, algorithms)
    rotation_rows = [] if args.skip_rotation_cases else evaluate_rotation_cases(args, algorithms, pairs)

    write_csv(args.output_dir / "candidate_pairs.csv", [asdict(pair) for pair in pairs], PAIR_FIELDS)
    write_csv(args.output_dir / "per_pair_metrics.csv", [asdict(row) for row in metrics], METRIC_FIELDS)
    write_csv(args.output_dir / "per_group_policy_summary.csv", group_rows, GROUP_FIELDS)
    write_csv(args.output_dir / "policy_ranking.csv", rankings, RANKING_FIELDS)
    write_csv(args.output_dir / "rotation_case_metrics.csv", [asdict(row) for row in rotation_rows], ROTATION_FIELDS)
    (args.output_dir / "summary.md").write_text(
        markdown_summary(
            args=args,
            pairs=pairs,
            group_rows=group_rows,
            rankings=rankings,
            rotation_rows=rotation_rows,
            skipped=skipped,
        ),
        encoding="utf-8",
    )
    update_progress(args.output_dir, f"Ran Stage9 validation: {len(pairs)} pairs, {len(algorithms)} algorithms, {len(metrics)} metric rows.")


def self_test() -> None:
    assert precision(2, 4) == 0.5
    assert source_style("source_000001_10") == "numeric"
    assert source_style("source_000108_20260514T143135909_NAS_PAN_L2b") == "timestamp"
    ranking = rank_policies(
        [
            {
                "algorithm": "a",
                "family": "classical",
                "candidate_pairs": 1,
                "ok_pairs": 1,
                "covered_pairs": 1,
                "matches": 10,
                "correct": 10,
                "wrong": 0,
                "precision": 1.0,
                "recommend": 1,
            }
            for _ in GROUPS
        ],
        [Algorithm("a", "classical", object(), 2.0, 0.8, 4)],
    )
    assert ranking[0]["recommend"] == 1
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-route", type=Path, default=DEFAULT_BASELINE_ROUTE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-per-group", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--learned-max-keypoints", type=int, default=2048)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--truth-threshold-px", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-lightglue", action="store_true")
    parser.add_argument("--limit-algorithms", nargs="*", default=None)
    parser.add_argument("--skip-rotation-cases", action="store_true")
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
