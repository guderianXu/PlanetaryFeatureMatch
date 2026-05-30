#!/usr/bin/env python3
"""Agent13 external matcher iteration on selected 1024 rotated patch pairs."""

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
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

AGENT4_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent4.py"
AGENT8_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent8.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13"
AGENT12_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent12"
DEFAULT_PFM_RUN = PROJECT_ROOT / "runs" / "cross_view_1024_rootsift_pseudo_r080t2_keypointonly_w1n002_lr1e5_viewpoint_80_seed1234"
DEFAULT_PFM_STATE = DEFAULT_PFM_RUN / "training" / "pytorch_pfm_state.pt"

METRIC_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "sample_id",
    "source_name",
    "source_hint",
    "pair_pt",
    "algorithm",
    "family",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "coverage",
    "coverage_min_inliers",
    "pass_gate",
    "min_correct",
    "mean_error_px",
    "median_error_px",
    "ransac_threshold_px",
    "ratio",
    "mode",
    "min_inliers_filter",
    "visualization",
    "message",
]

SUMMARY_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "algorithm",
    "family",
    "pairs",
    "ok_pairs",
    "covered_pairs",
    "coverage_rate",
    "pass_gate_pairs",
    "pass_gate_rate",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "mean_inliers_per_pair",
    "median_inliers_per_pair",
    "mean_correct_per_pair",
]

STYLE_GATE_FIELDS = [
    "style",
    "gate",
    "algorithm",
    "family",
    "pairs",
    "ok_pairs",
    "covered_pairs",
    "coverage_rate",
    "pass_gate_pairs",
    "pass_gate_rate",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "mean_inliers_per_pair",
    "median_inliers_per_pair",
    "mean_correct_per_pair",
]

GLOBAL_FIELDS = [
    "algorithm",
    "family",
    "pairs",
    "ok_pairs",
    "covered_pairs",
    "coverage_rate",
    "pass_gate_pairs",
    "pass_gate_rate",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "mean_inliers_per_pair",
]

SAMPLE_FIELDS = ["style", "gate", "sample_id", "source_name", "source_hint", "pair_pt"]

PFM_BASELINE_FIELDS = ["source", "label", "style", "gate", "pairs", "matches", "correct", "wrong", "precision", "csv", "note"]


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
    ransac_threshold_px: float
    ratio: float
    mode: str
    min_inliers_filter: int
    coverage_min_inliers: int


@dataclass(frozen=True)
class PairSpec:
    style: str
    gate: str
    sample_id: str
    pair_path: Path
    source_hint: str


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    rotation_deg: int
    sample_id: str
    source_name: str
    source_hint: str
    pair_pt: str
    algorithm: str
    family: str
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    inlier_matches: int
    correct: int
    wrong: int
    precision: float
    coverage: int
    coverage_min_inliers: int
    pass_gate: int
    min_correct: int
    mean_error_px: float
    median_error_px: float
    ransac_threshold_px: float
    ratio: float
    mode: str
    min_inliers_filter: int
    visualization: str = ""
    message: str = ""


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A4 = load_module(AGENT4_SCRIPT, "agent4_matcher_for_agent12")
A8 = load_module(AGENT8_SCRIPT, "agent8_matcher_for_agent12")


def empty_points() -> np.ndarray:
    return A4.empty_points()


def min_gate_labels(gate: str) -> int:
    return 8 if gate == "compound" else 20


def source_name(pair_path: Path) -> str:
    parent = pair_path.parent.name
    if parent.startswith("source_"):
        grandparent = pair_path.parent.parent.name
        if grandparent.startswith("source_"):
            return f"{grandparent}/{parent}"
        return parent
    return pair_path.parent.parent.name


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def format_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fields})


def default_pair_specs() -> list[PairSpec]:
    return [
        PairSpec(
            style="numeric",
            gate="viewpoint",
            sample_id="numeric_viewpoint_1_pair001155",
            pair_path=PROJECT_ROOT / "img" / "Viewpoint_1024" / "source_000000_1" / "pair_001155.pt",
            source_hint="numeric source 1; cache pair chosen as the closest available 1024 pair to the requested 1-155 style",
        ),
        PairSpec(
            style="numeric",
            gate="compound",
            sample_id="numeric_compound_1_to_155",
            pair_path=PROJECT_ROOT / "img" / "CompoundViewpoint_1024" / "source_000000_1" / "source_000062_155" / "pair_000062.pt",
            source_hint="numeric source 1 nested target source 155 compound 1024 cache",
        ),
        PairSpec(
            style="timestamp",
            gate="viewpoint",
            sample_id="timestamp_viewpoint_064636_pair000300",
            pair_path=PROJECT_ROOT
            / "img"
            / "Viewpoint_1024"
            / "source_000069_20260514T064636672_NAS_PAN_L2b"
            / "pair_000300.pt",
            source_hint="timestamp source 20260514T064636672_NAS_PAN_L2b viewpoint 1024 cache",
        ),
        PairSpec(
            style="timestamp",
            gate="compound",
            sample_id="timestamp_compound_064636_pair000300",
            pair_path=PROJECT_ROOT
            / "img"
            / "CompoundViewpoint_1024"
            / "source_000069_20260514T064636672_NAS_PAN_L2b"
            / "pair_000300.pt",
            source_hint="timestamp source 20260514T064636672_NAS_PAN_L2b compound 1024 cache",
        ),
    ]


class CvMatcher:
    def __init__(
        self,
        detector_name: str,
        *,
        ratio: float,
        mode: str,
        max_keypoints: int,
        max_matches: int,
        sift_contrast: float,
    ) -> None:
        import cv2

        self.detector_name = detector_name
        self.ratio = ratio
        self.mode = mode
        self.max_matches = max_matches
        if detector_name in {"SIFT", "RootSIFT"}:
            self.detector = cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=sift_contrast)
            self.descriptor_kind = "float"
        elif detector_name == "KAZE":
            self.detector = cv2.KAZE_create()
            self.descriptor_kind = "float"
        elif detector_name == "ORB":
            self.detector = cv2.ORB_create(nfeatures=max_keypoints)
            self.descriptor_kind = "binary"
        elif detector_name == "AKAZE":
            self.detector = cv2.AKAZE_create()
            self.descriptor_kind = "binary"
        else:
            raise ValueError(f"unknown cv detector: {detector_name}")

    def _prepare_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        if self.detector_name == "RootSIFT":
            descriptors = A4.rootsift(descriptors.astype(np.float32, copy=False))
        if self.descriptor_kind == "float":
            return descriptors.astype(np.float32, copy=False)
        return descriptors

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> RawOutput:
        import cv2

        keypoints_a, descriptors_a = self.detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self.detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return RawOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []), 0)
        descriptors_a = self._prepare_descriptors(descriptors_a)
        descriptors_b = self._prepare_descriptors(descriptors_b)
        norm = cv2.NORM_L2 if self.descriptor_kind == "float" else cv2.NORM_HAMMING
        if self.mode == "cross":
            matches = cv2.BFMatcher(norm, crossCheck=True).match(descriptors_a, descriptors_b)
        elif self.mode == "ratio":
            matches = A4.ratio_filter(cv2.BFMatcher(norm).knnMatch(descriptors_a, descriptors_b, k=2), self.ratio)
        elif self.mode == "ratio_mutual":
            matcher = cv2.BFMatcher(norm)
            forward = A4.ratio_filter(matcher.knnMatch(descriptors_a, descriptors_b, k=2), self.ratio)
            backward = A4.ratio_filter(matcher.knnMatch(descriptors_b, descriptors_a, k=2), self.ratio)
            backward_pairs = {(item.trainIdx, item.queryIdx) for item in backward}
            matches = [item for item in forward if (item.queryIdx, item.trainIdx) in backward_pairs]
        else:
            raise ValueError(f"unknown match mode: {self.mode}")
        matches = sorted(matches, key=lambda item: item.distance)[: self.max_matches]
        output = A4.output_from_matches(keypoints_a, keypoints_b, matches)
        return RawOutput(output.points_a, output.points_b, output.keypoints_a, output.keypoints_b, len(matches))


class LightGlueMatcher:
    def __init__(self, *, max_keypoints: int, device: str) -> None:
        self.impl = A4.LightGlueSiftMatcher(max_keypoints=max_keypoints, device=device)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> RawOutput:
        output = self.impl.match(image_a, image_b)
        return RawOutput(output.points_a, output.points_b, output.keypoints_a, output.keypoints_b, output.points_a.shape[0])


class PFMMatcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.impl = A8.PFMScoredMatcher(
            state_path=args.pfm_state,
            device=choose_device(args.device),
            max_keypoints=args.pfm_max_keypoints,
            max_matches=args.max_matches,
            min_intensity=args.pfm_min_intensity,
            min_score=args.pfm_min_score,
        )
        self.min_margin = args.pfm_min_margin

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> RawOutput:
        output = self.impl.match(image_a, image_b, min_margin=self.min_margin)
        return RawOutput(
            output.points_a,
            output.points_b_rotated,
            output.keypoints_a,
            output.keypoints_b,
            output.points_a.shape[0],
        )


def cv2_status() -> tuple[object | None, list[dict[str, str]]]:
    unavailable: list[dict[str, str]] = []
    try:
        import cv2

        checks = {"SIFT": "SIFT_create", "ORB": "ORB_create", "AKAZE": "AKAZE_create", "KAZE": "KAZE_create"}
        for label, attr in checks.items():
            if not hasattr(cv2, attr):
                unavailable.append({"algorithm": label, "reason": f"cv2.{attr} unavailable"})
        return cv2, unavailable
    except Exception as exc:
        unavailable.append({"algorithm": "OpenCV algorithms", "reason": f"{type(exc).__name__}: {exc}"})
        return None, unavailable


def add_algorithm(
    algorithms: list[Algorithm],
    name: str,
    family: str,
    matcher: object,
    *,
    ransac_threshold_px: float,
    ratio: float,
    mode: str,
    min_inliers_filter: int,
    coverage_min_inliers: int,
) -> None:
    algorithms.append(
        Algorithm(
            name=name,
            family=family,
            matcher=matcher,
            ransac_threshold_px=ransac_threshold_px,
            ratio=ratio,
            mode=mode,
            min_inliers_filter=min_inliers_filter,
            coverage_min_inliers=coverage_min_inliers,
        )
    )


def make_matchers(args: argparse.Namespace) -> tuple[list[Algorithm], list[dict[str, str]]]:
    algorithms: list[Algorithm] = []
    skipped: list[dict[str, str]] = []
    cv2, unavailable = cv2_status()
    skipped.extend(unavailable)

    def cv_algo(
        detector: str,
        *,
        ratio: float,
        threshold: float,
        min_inliers: int,
        mode: str,
        family: str = "classical_sweep",
    ) -> None:
        threshold_text = f"{threshold:g}".replace(".", "p")
        ratio_text = "nan" if math.isnan(ratio) else f"{ratio:.2f}".replace(".", "p")
        mode_text = "mutual" if mode == "ratio_mutual" else mode
        add_algorithm(
            algorithms,
            f"{detector}-{mode_text}-r{ratio_text}-Ht{threshold_text}-min{min_inliers}",
            family,
            CvMatcher(detector, ratio=ratio, mode=mode, max_keypoints=args.max_keypoints, max_matches=args.max_matches, sift_contrast=args.sift_contrast),
            ransac_threshold_px=threshold,
            ratio=ratio,
            mode=mode,
            min_inliers_filter=min_inliers,
            coverage_min_inliers=min_inliers,
        )

    if cv2 is not None and hasattr(cv2, "SIFT_create"):
        for detector in ["SIFT", "RootSIFT"]:
            for ratio in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
                for threshold in [2.0, 3.0]:
                    cv_algo(detector, ratio=ratio, threshold=threshold, min_inliers=4, mode="ratio")
            for ratio in [0.85, 0.90, 0.92]:
                for threshold in [2.0, 3.0]:
                    cv_algo(detector, ratio=ratio, threshold=threshold, min_inliers=8, mode="ratio")
        for ratio in [0.80, 0.85, 0.90, 0.92, 0.95]:
            for threshold in [2.0, 3.0]:
                cv_algo("RootSIFT", ratio=ratio, threshold=threshold, min_inliers=4, mode="ratio_mutual")
        for detector in ["SIFT", "RootSIFT"]:
            cv_algo(detector, ratio=math.nan, threshold=3.0, min_inliers=4, mode="cross")
    if cv2 is not None and hasattr(cv2, "ORB_create"):
        cv_algo("ORB", ratio=math.nan, threshold=3.0, min_inliers=4, mode="cross")
        for ratio in [0.75, 0.85, 0.95]:
            cv_algo("ORB", ratio=ratio, threshold=3.0, min_inliers=4, mode="ratio")
    if cv2 is not None and hasattr(cv2, "AKAZE_create"):
        cv_algo("AKAZE", ratio=math.nan, threshold=3.0, min_inliers=4, mode="cross")
        for ratio in [0.75, 0.85, 0.95]:
            cv_algo("AKAZE", ratio=ratio, threshold=3.0, min_inliers=4, mode="ratio")
    if cv2 is not None and hasattr(cv2, "KAZE_create"):
        cv_algo("KAZE", ratio=math.nan, threshold=3.0, min_inliers=4, mode="cross")
        for ratio in [0.75, 0.85, 0.90, 0.95]:
            cv_algo("KAZE", ratio=ratio, threshold=3.0, min_inliers=4, mode="ratio")

    if importlib.util.find_spec("lightglue") is None:
        skipped.append({"algorithm": "LightGlue-SIFT-Ht3-min4", "reason": "module 'lightglue' unavailable"})
    elif args.no_lightglue:
        skipped.append({"algorithm": "LightGlue-SIFT-Ht3-min4", "reason": "disabled by --no-lightglue"})
    else:
        try:
            add_algorithm(
                algorithms,
                "LightGlue-SIFT-Ht3-min4",
                "learned_external",
                LightGlueMatcher(max_keypoints=args.learned_max_keypoints, device=choose_device(args.device)),
                ransac_threshold_px=3.0,
                ratio=math.nan,
                mode="lightglue",
                min_inliers_filter=4,
                coverage_min_inliers=4,
            )
        except Exception as exc:
            skipped.append({"algorithm": "LightGlue-SIFT-Ht3-min4", "reason": f"{type(exc).__name__}: {exc}"})

    if importlib.util.find_spec("match_pairs") is None and importlib.util.find_spec("superglue") is None:
        skipped.append({"algorithm": "SuperGlue", "reason": "modules 'match_pairs' and 'superglue' unavailable"})
    else:
        skipped.append({"algorithm": "SuperGlue", "reason": "dependency present but no stable local runner was found for this project API"})

    if args.include_pfm_live:
        try:
            pfm = PFMMatcher(args)
            add_algorithm(
                algorithms,
                "PFM-current-raw-live",
                "pfm_live",
                pfm,
                ransac_threshold_px=math.nan,
                ratio=math.nan,
                mode="mutual_nearest",
                min_inliers_filter=0,
                coverage_min_inliers=1,
            )
            add_algorithm(
                algorithms,
                "PFM-current-Ht3-min4-live",
                "pfm_live",
                pfm,
                ransac_threshold_px=3.0,
                ratio=math.nan,
                mode="mutual_nearest",
                min_inliers_filter=4,
                coverage_min_inliers=4,
            )
        except Exception as exc:
            skipped.append({"algorithm": "PFM-current-live", "reason": f"{type(exc).__name__}: {exc}"})
    else:
        skipped.append({"algorithm": "PFM-current-live", "reason": "disabled by default; Agent13 reads existing PFM summaries instead"})

    if args.limit_algorithms:
        keep = set(args.limit_algorithms)
        algorithms = [item for item in algorithms if item.name in keep]
    return algorithms, skipped


def apply_ransac(raw: RawOutput, threshold_px: float, min_inliers: int) -> RawOutput:
    if math.isnan(threshold_px):
        return raw
    inlier_a, inlier_b = A4.ransac_inliers(raw.points_a, raw.points_b, threshold_px=threshold_px)
    if inlier_a.shape[0] < min_inliers:
        inlier_a, inlier_b = empty_points(), empty_points()
    return RawOutput(inlier_a, inlier_b, raw.keypoints_a, raw.keypoints_b, raw.raw_matches)


def metric_row(
    *,
    args: argparse.Namespace,
    algorithm: Algorithm,
    spec: PairSpec,
    rotation_deg: int,
    output: RawOutput,
    image_b_shape: tuple[int, int],
    warp_a_to_b,
    valid_mask,
    status: str = "ok",
    message: str = "",
    visualization: str = "",
) -> tuple[MetricRow, np.ndarray]:
    points_b_original = A4.unrotate_points(output.points_b, image_b_shape[0], image_b_shape[1], rotation_deg)
    matches, correct, wrong, precision, mean_error, median_error = A4.compute_metrics(
        output.points_a,
        points_b_original,
        warp_a_to_b,
        valid_mask,
        threshold_px=args.threshold_px,
    )
    min_correct = min_gate_labels(spec.gate)
    row = MetricRow(
        style=spec.style,
        gate=spec.gate,
        rotation_deg=rotation_deg,
        sample_id=spec.sample_id,
        source_name=source_name(spec.pair_path),
        source_hint=spec.source_hint,
        pair_pt=spec.pair_path.as_posix(),
        algorithm=algorithm.name,
        family=algorithm.family,
        status=status,
        keypoints_a=output.keypoints_a,
        keypoints_b=output.keypoints_b,
        raw_matches=output.raw_matches,
        inlier_matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        coverage=1 if matches >= algorithm.coverage_min_inliers else 0,
        coverage_min_inliers=algorithm.coverage_min_inliers,
        pass_gate=1 if correct >= min_correct else 0,
        min_correct=min_correct,
        mean_error_px=mean_error,
        median_error_px=median_error,
        ransac_threshold_px=algorithm.ransac_threshold_px,
        ratio=algorithm.ratio,
        mode=algorithm.mode,
        min_inliers_filter=algorithm.min_inliers_filter,
        visualization=visualization,
        message=message,
    )
    return row, points_b_original


def evaluate_one(
    args: argparse.Namespace,
    algorithm: Algorithm,
    *,
    spec: PairSpec,
    rotation_deg: int,
    image_a: np.ndarray,
    image_b: np.ndarray,
    warp_a_to_b,
    valid_mask,
    vis_budget: dict[str, int],
) -> MetricRow:
    try:
        image_b_rotated = A4.rotate_image(image_b, rotation_deg)
        raw = algorithm.matcher.match(image_a, image_b_rotated)
        output = apply_ransac(raw, algorithm.ransac_threshold_px, algorithm.min_inliers_filter)
        row, points_b_original = metric_row(
            args=args,
            algorithm=algorithm,
            spec=spec,
            rotation_deg=rotation_deg,
            output=output,
            image_b_shape=image_b.shape[:2],
            warp_a_to_b=warp_a_to_b,
            valid_mask=valid_mask,
        )
        key = f"{spec.sample_id}/rot{rotation_deg}/{algorithm.name}"
        if args.visualizations_per_algorithm > 0 and row.inlier_matches > 0 and vis_budget.get(key, 0) < args.visualizations_per_algorithm:
            vis_budget[key] = vis_budget.get(key, 0) + 1
            vis_path = (
                args.output_dir
                / "visualizations"
                / spec.style
                / spec.gate
                / f"rot{rotation_deg}"
                / f"{spec.sample_id}_{A4.safe_name(algorithm.name)}.png"
            )
            A4.draw_visualization(image_a, image_b, output.points_a, points_b_original, vis_path)
            row = MetricRow(**{**asdict(row), "visualization": vis_path.as_posix()})
        return row
    except Exception as exc:
        return MetricRow(
            style=spec.style,
            gate=spec.gate,
            rotation_deg=rotation_deg,
            sample_id=spec.sample_id,
            source_name=source_name(spec.pair_path),
            source_hint=spec.source_hint,
            pair_pt=spec.pair_path.as_posix(),
            algorithm=algorithm.name,
            family=algorithm.family,
            status="error",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            inlier_matches=0,
            correct=0,
            wrong=0,
            precision=0.0,
            coverage=0,
            coverage_min_inliers=algorithm.coverage_min_inliers,
            pass_gate=0,
            min_correct=min_gate_labels(spec.gate),
            mean_error_px=math.nan,
            median_error_px=math.nan,
            ransac_threshold_px=algorithm.ransac_threshold_px,
            ratio=algorithm.ratio,
            mode=algorithm.mode,
            min_inliers_filter=algorithm.min_inliers_filter,
            message=f"{type(exc).__name__}: {exc}",
        )


def selected_pairs(args: argparse.Namespace) -> list[PairSpec]:
    specs = default_pair_specs()
    if args.limit_samples:
        keep = set(args.limit_samples)
        specs = [item for item in specs if item.sample_id in keep]
    missing = [item.pair_path for item in specs if not item.pair_path.exists()]
    if missing:
        joined = "\n".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"selected pair paths missing:\n{joined}")
    return specs


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, object]], list[dict[str, str]]]:
    algorithms, skipped = make_matchers(args)
    specs = selected_pairs(args)
    rows: list[MetricRow] = []
    sampled: list[dict[str, object]] = [
        {
            "style": spec.style,
            "gate": spec.gate,
            "sample_id": spec.sample_id,
            "source_name": source_name(spec.pair_path),
            "source_hint": spec.source_hint,
            "pair_pt": spec.pair_path.as_posix(),
        }
        for spec in specs
    ]
    vis_budget: dict[str, int] = {}
    if not algorithms:
        skipped.append({"algorithm": "all", "reason": "no runnable algorithms after dependency checks"})
        return rows, sampled, skipped
    print(f"samples={len(specs)} rotations={len(args.rotations)} algorithms={len(algorithms)}", flush=True)
    for spec in specs:
        image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(spec.pair_path)
        print(f"sample={spec.sample_id} style={spec.style} gate={spec.gate}", flush=True)
        for rotation_deg in args.rotations:
            for algorithm in algorithms:
                rows.append(
                    evaluate_one(
                        args,
                        algorithm,
                        spec=spec,
                        rotation_deg=rotation_deg,
                        image_a=image_a,
                        image_b=image_b,
                        warp_a_to_b=warp_a_to_b,
                        valid_mask=valid_mask,
                        vis_budget=vis_budget,
                    )
                )
            print(f"{spec.sample_id:42s} rot={rotation_deg:3d} done", flush=True)
    return rows, sampled, skipped


def summarize_group(items: list[MetricRow]) -> dict[str, object]:
    ok = [row for row in items if row.status == "ok"]
    raw_matches = sum(row.raw_matches for row in ok)
    inlier_matches = sum(row.inlier_matches for row in ok)
    correct = sum(row.correct for row in ok)
    wrong = sum(row.wrong for row in ok)
    ok_pairs = len(ok)
    covered_pairs = sum(row.coverage for row in ok)
    pass_gate_pairs = sum(row.pass_gate for row in ok)
    return {
        "family": items[0].family,
        "pairs": len(items),
        "ok_pairs": ok_pairs,
        "covered_pairs": covered_pairs,
        "coverage_rate": 0.0 if ok_pairs == 0 else covered_pairs / ok_pairs,
        "pass_gate_pairs": pass_gate_pairs,
        "pass_gate_rate": 0.0 if ok_pairs == 0 else pass_gate_pairs / ok_pairs,
        "raw_matches": raw_matches,
        "inlier_matches": inlier_matches,
        "correct": correct,
        "wrong": wrong,
        "precision": 0.0 if inlier_matches == 0 else correct / inlier_matches,
        "mean_pair_precision": float(np.mean([row.precision for row in ok])) if ok else math.nan,
        "mean_inliers_per_pair": float(np.mean([row.inlier_matches for row in ok])) if ok else math.nan,
        "median_inliers_per_pair": float(np.median([row.inlier_matches for row in ok])) if ok else math.nan,
        "mean_correct_per_pair": float(np.mean([row.correct for row in ok])) if ok else math.nan,
    }


def aggregate(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.rotation_deg, row.algorithm), []).append(row)
    out: list[dict[str, object]] = []
    for (style, gate, rotation_deg, algorithm), items in sorted(grouped.items()):
        data = summarize_group(items)
        out.append({"style": style, "gate": gate, "rotation_deg": rotation_deg, "algorithm": algorithm, **data})
    return out


def aggregate_style_gate(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.algorithm), []).append(row)
    out: list[dict[str, object]] = []
    for (style, gate, algorithm), items in sorted(grouped.items()):
        data = summarize_group(items)
        out.append({"style": style, "gate": gate, "algorithm": algorithm, **data})
    return out


def aggregate_global(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.algorithm, []).append(row)
    return [{"algorithm": algorithm, **summarize_group(items)} for algorithm, items in sorted(grouped.items())]


def ranking_key(item: dict[str, object]) -> tuple[int, int, float, float, str]:
    return (
        -int(item["pass_gate_pairs"]),
        -int(item["correct"]),
        -float(item["precision"]),
        -float(item["coverage_rate"]),
        str(item["algorithm"]),
    )


def markdown_table(rows: list[dict[str, object]], *, limit: int = 16) -> list[str]:
    lines = [
        "| algorithm | family | ok | covered | coverage | pass gate | inliers | correct | precision | mean inliers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(rows, key=ranking_key)
    for row in ordered[:limit]:
        lines.append(
            f"| {row['algorithm']} | {row['family']} | {row['ok_pairs']} | {row['covered_pairs']} | "
            f"{float(row['coverage_rate']):.3f} | {row['pass_gate_pairs']} | {row['inlier_matches']} | "
            f"{row['correct']} | {float(row['precision']):.4f} | {float(row['mean_inliers_per_pair']):.2f} |"
        )
    return lines


def top_rows(global_rows: list[dict[str, object]], family_prefix: str | None = None) -> list[dict[str, object]]:
    rows = global_rows
    if family_prefix is not None:
        rows = [row for row in rows if str(row["family"]).startswith(family_prefix)]
    return sorted(rows, key=ranking_key)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pfm_baseline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    agent12_global = AGENT12_OUTPUT_DIR / "global_summary.csv"
    for row in read_csv_rows(agent12_global):
        if row.get("family") != "pfm":
            continue
        rows.append(
            {
                "source": "agent12_same12",
                "label": row["algorithm"],
                "style": "mixed",
                "gate": "viewpoint+compound",
                "pairs": row.get("ok_pairs", row.get("pairs", "")),
                "matches": row.get("inlier_matches", ""),
                "correct": row.get("correct", ""),
                "wrong": row.get("wrong", ""),
                "precision": row.get("precision", ""),
                "csv": agent12_global.as_posix(),
                "note": "Read-only Agent12 PFM rows on the same 12 sample/rotation tasks.",
            }
        )

    composite = PROJECT_ROOT / "runs" / "cross_view_1024_rootsift_pseudo_r080t2_guard_eval_0step_seed1234" / "composite_best_route_summary.csv"
    for row in read_csv_rows(composite):
        rows.append(
            {
                "source": "cross_view_composite_best_route",
                "label": row.get("source_run", "composite"),
                "style": row.get("style", ""),
                "gate": row.get("gate", ""),
                "pairs": "",
                "matches": row.get("matches", ""),
                "correct": row.get("correct", ""),
                "wrong": row.get("wrong", ""),
                "precision": row.get("precision", ""),
                "csv": row.get("summary_csv", composite.as_posix()),
                "note": row.get("note", ""),
            }
        )

    sparse = DEFAULT_PFM_RUN / "sparse_eval" / "summary.csv"
    for row in read_csv_rows(sparse):
        label = row.get("label", "")
        parts = label.split("_")
        style = parts[0] if len(parts) >= 2 else ""
        gate = parts[1] if len(parts) >= 2 else ""
        rows.append(
            {
                "source": "pfm_sparse_eval",
                "label": label,
                "style": style,
                "gate": gate,
                "pairs": row.get("pairs", ""),
                "matches": row.get("matches", ""),
                "correct": row.get("correct", ""),
                "wrong": row.get("wrong", ""),
                "precision": row.get("precision", ""),
                "csv": row.get("csv", sparse.as_posix()),
                "note": f"texture_blend_weight={row.get('texture_blend_weight', '')}; keypoint_score_mode={row.get('keypoint_score_mode', '')}",
            }
        )
    return rows


def best_by_group(rows: list[dict[str, object]], group_fields: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    return [sorted(items, key=ranking_key)[0] for _, items in sorted(grouped.items())]


def recommendation_lines(
    global_rows: list[dict[str, object]],
    style_gate_rows: list[dict[str, object]],
    skipped: list[dict[str, str]],
    pfm_rows: list[dict[str, object]],
) -> list[str]:
    external = [row for row in top_rows(global_rows) if str(row["family"]).startswith(("classical", "learned_external"))]
    tuned = [row for row in top_rows(global_rows, "classical")]
    lines = ["## Recommendations", ""]
    if external:
        best = external[0]
        lines.append(
            f"- Best teacher candidate in this run: `{best['algorithm']}` with {best['correct']}/{best['inlier_matches']} correct/inliers, "
            f"precision {float(best['precision']):.4f}, coverage {float(best['coverage_rate']):.3f}, pass-gate {best['pass_gate_pairs']}/{best['ok_pairs']}."
        )
    if tuned:
        best_tuned = tuned[0]
        lines.append(
            f"- Classical gate to prefer globally: `{best_tuned['algorithm']}`. It is the strongest classical filter by pass-gate/correct count here."
        )
    group_recs = best_by_group(style_gate_rows, ("style", "gate"))
    for row in group_recs:
        lines.append(
            f"- `{row['style']}/{row['gate']}` teacher: `{row['algorithm']}` with {row['correct']}/{row['inlier_matches']} correct/inliers, "
            f"precision {float(row['precision']):.4f}, pass-gate {row['pass_gate_pairs']}/{row['ok_pairs']}."
        )
    lines.append(
        "- Use homography-filtered external matches as pseudo-labels; keep per-pair coverage and pass-gate thresholds separate so training can reject low-texture or low-overlap pairs."
    )
    lines.append(
        "- A conservative training-data gate is at least 4 RANSAC inliers for coverage and the existing 20 correct matches for viewpoint / 8 for compound as a high-confidence pass gate."
    )
    same12_pfm = [row for row in pfm_rows if row.get("source") == "agent12_same12"]
    if same12_pfm and external:
        best_pfm = max(same12_pfm, key=lambda item: int(float(str(item.get("correct") or 0))))
        best_ext = external[0]
        gap = int(best_ext["correct"]) - int(float(str(best_pfm["correct"])))
        lines.append(
            f"- Current PFM same-12 gap: best read-only PFM row `{best_pfm['label']}` produced {best_pfm['correct']}/{best_pfm['matches']} correct/inliers "
            f"versus external best {best_ext['correct']}/{best_ext['inlier_matches']} on the same rotations, a correct-match gap of {gap}."
        )
    if skipped:
        skipped_text = "; ".join(f"{item['algorithm']}: {item['reason']}" for item in skipped)
        lines.append(f"- Unavailable/skipped dependencies recorded: {skipped_text}.")
    return lines


def write_summary(
    args: argparse.Namespace,
    rows: list[MetricRow],
    summary_rows: list[dict[str, object]],
    style_gate_rows: list[dict[str, object]],
    global_rows: list[dict[str, object]],
    pfm_rows: list[dict[str, object]],
    sampled: list[dict[str, object]],
    skipped: list[dict[str, str]],
) -> None:
    command = (
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} --device {args.device} --output-dir {args.output_dir}"
    )
    rotated_evals = len({(row.sample_id, row.rotation_deg) for row in rows})
    lines = [
        "# Matcher Algorithm Iteration Agent13",
        "",
        "## Scope",
        "",
        "- Goal: continue external/traditional matcher iteration on two style families and compare them with read-only current PFM baselines.",
        "- B image is rotated by 90/180/270 degrees before matching; B match coordinates are unrotated before `warp_a_to_b` scoring.",
        f"- Samples: `{len(sampled)}` patch pairs; rotated sample evaluations: `{rotated_evals}`.",
        f"- Runnable algorithms: `{len({row.algorithm for row in rows})}`.",
        f"- PFM checkpoint path recorded: `{args.pfm_state}`; live PFM evaluation enabled: `{args.include_pfm_live}`.",
        f"- Correct threshold: `{args.threshold_px}` px.",
        "- Pass gate: 20 correct matches for viewpoint, 8 correct matches for compound.",
        "",
        "## Command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Sampled Pairs",
        "",
        "| style | gate | sample | source | pair |",
        "|---|---|---|---|---|",
    ]
    for item in sampled:
        lines.append(
            f"| {item['style']} | {item['gate']} | {item['sample_id']} | {item['source_name']} | `{item['pair_pt']}` |"
        )
    lines.extend(["", "## Global Summary", "", *markdown_table(global_rows)])
    lines.extend(["", "## Style/Gate Recommendations", ""])
    lines.extend(
        [
            "| style | gate | algorithm | ok | pass gate | inliers | correct | precision | coverage |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_by_group(style_gate_rows, ("style", "gate")):
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['algorithm']} | {row['ok_pairs']} | {row['pass_gate_pairs']} | "
            f"{row['inlier_matches']} | {row['correct']} | {float(row['precision']):.4f} | {float(row['coverage_rate']):.3f} |"
        )
    lines.extend(["", "## Rotation-Specific Best", ""])
    lines.extend(
        [
            "| style | gate | rotation | algorithm | pass gate | inliers | correct | precision |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in best_by_group(summary_rows, ("style", "gate", "rotation_deg")):
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['rotation_deg']} | {row['algorithm']} | {row['pass_gate_pairs']} | "
            f"{row['inlier_matches']} | {row['correct']} | {float(row['precision']):.4f} |"
        )
    lines.extend(["", "## PFM Baseline Comparison", ""])
    lines.extend(
        [
            "| source | label | style | gate | pairs | matches | correct | precision |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in pfm_rows[:24]:
        precision = row.get("precision", "")
        precision_text = "" if precision == "" else f"{float(str(precision)):.4f}"
        lines.append(
            f"| {row['source']} | {row['label']} | {row['style']} | {row['gate']} | {row['pairs']} | {row['matches']} | {row['correct']} | {precision_text} |"
        )
    lines.extend(["", *recommendation_lines(global_rows, style_gate_rows, skipped, pfm_rows)])
    lines.extend(["", "## Skipped / Unavailable", ""])
    if skipped:
        for item in skipped:
            lines.append(f"- {item['algorithm']}: {item['reason']}")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `metrics.csv`: per algorithm/style/gate/rotation/sample metrics with correct, inliers, precision, coverage, and visualization path.",
            "- `summary_metrics.csv`: grouped by style/gate/rotation/algorithm.",
            "- `style_gate_summary.csv`: grouped by style/gate/algorithm for teacher selection.",
            "- `global_summary.csv`: grouped by algorithm across the run.",
            "- `pfm_baseline_summary.csv`: read-only PFM summaries from Agent12/current cross-view runs.",
            "- `sampled_pairs.csv`: selected numeric and timestamp cache pairs.",
            "- `skipped_algorithms.csv`: algorithms skipped because dependencies or local runner support were unavailable.",
            "- `visualizations/`: bounded match visualizations.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    specs = default_pair_specs()
    assert len(specs) == 4
    assert {item.style for item in specs} == {"numeric", "timestamp"}
    assert all(item.pair_path.exists() for item in specs)
    assert min_gate_labels("compound") == 8
    assert min_gate_labels("viewpoint") == 20
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert A4.rotate_image(image, 90).shape == (4, 3)
    point = np.array([[0.0, 1.0]], dtype=np.float32)
    assert np.allclose(A4.unrotate_points(point, 3, 4, 90), [[1.0, 2.0]])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pfm-state", type=Path, default=DEFAULT_PFM_STATE)
    parser.add_argument("--rotations", nargs="+", type=int, default=[90, 180, 270], choices=[90, 180, 270])
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--learned-max-keypoints", type=int, default=512)
    parser.add_argument("--pfm-max-keypoints", type=int, default=512)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--pfm-min-intensity", type=float, default=0.03)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--pfm-min-margin", type=float, default=0.0)
    parser.add_argument("--visualizations-per-algorithm", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-lightglue", action="store_true")
    parser.add_argument("--include-pfm-live", action="store_true", help="Run the PFM matcher live; default is read-only PFM summary comparison.")
    parser.add_argument("--limit-algorithms", nargs="*")
    parser.add_argument("--limit-samples", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, sampled, skipped = evaluate(args)
    summary_rows = aggregate(rows)
    style_gate_rows = aggregate_style_gate(rows)
    global_rows = aggregate_global(rows)
    pfm_rows = pfm_baseline_rows()
    write_csv(args.output_dir / "metrics.csv", [asdict(row) for row in rows], METRIC_FIELDS)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "style_gate_summary.csv", style_gate_rows, STYLE_GATE_FIELDS)
    write_csv(args.output_dir / "global_summary.csv", global_rows, GLOBAL_FIELDS)
    write_csv(args.output_dir / "pfm_baseline_summary.csv", pfm_rows, PFM_BASELINE_FIELDS)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled, SAMPLE_FIELDS)
    write_csv(args.output_dir / "skipped_algorithms.csv", skipped, ["algorithm", "reason"])
    write_summary(args, rows, summary_rows, style_gate_rows, global_rows, pfm_rows, sampled, skipped)
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
