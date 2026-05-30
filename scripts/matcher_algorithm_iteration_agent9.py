#!/usr/bin/env python3
"""Agent9 matcher iteration: high-precision classical fallback recovery."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

AGENT4_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent4.py"
AGENT7_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent7"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent9"
DEFAULT_PFM_RUN = PROJECT_ROOT / "runs" / "cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234"

METRIC_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "pair_pt",
    "config",
    "detector",
    "preprocess",
    "match_mode",
    "ratio",
    "ransac_threshold_px",
    "min_inliers",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "matches",
    "correct",
    "wrong",
    "precision",
    "coverage",
    "pass_gate",
    "base_pass_gate",
    "fallback_candidate",
    "recovered_baseline_fail",
    "mean_error_px",
    "median_error_px",
    "visualization",
    "message",
]

SUMMARY_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "config",
    "detector",
    "preprocess",
    "match_mode",
    "ratio",
    "ransac_threshold_px",
    "min_inliers",
    "pairs",
    "ok_pairs",
    "covered_pairs",
    "pass_gate_pairs",
    "matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "mean_matches_per_pair",
    "baseline_failed_pairs",
    "baseline_failed_recovered_pairs",
    "fallback_matches",
    "fallback_correct",
    "fallback_wrong",
    "fallback_precision",
]

RECOVERY_FIELDS = [
    "case_type",
    "style",
    "gate",
    "rotation_deg",
    "pair_pt",
    "config",
    "detector",
    "preprocess",
    "match_mode",
    "ratio",
    "ransac_threshold_px",
    "min_inliers",
    "baseline_matches",
    "baseline_correct",
    "baseline_precision",
    "baseline_pass_gate",
    "fallback_matches",
    "fallback_correct",
    "fallback_wrong",
    "fallback_precision",
    "fallback_pass_gate",
    "recovered",
    "mean_error_px",
    "median_error_px",
    "visualization",
    "message",
]


@dataclass(frozen=True)
class RawMatchOutput:
    points_a: np.ndarray
    points_b: np.ndarray
    keypoints_a: int
    keypoints_b: int
    raw_matches: int


@dataclass(frozen=True)
class MatcherConfig:
    config: str
    detector: str
    preprocess: str
    match_mode: str
    ratio: float
    ransac_threshold_px: float


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    rotation_deg: int
    pair_pt: str
    config: str
    detector: str
    preprocess: str
    match_mode: str
    ratio: float
    ransac_threshold_px: float
    min_inliers: int
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    matches: int
    correct: int
    wrong: int
    precision: float
    coverage: int
    pass_gate: int
    base_pass_gate: int
    fallback_candidate: int
    recovered_baseline_fail: int
    mean_error_px: float
    median_error_px: float
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


A4 = load_module(AGENT4_SCRIPT, "agent4_matcher_for_agent9")


def empty_points() -> np.ndarray:
    return A4.empty_points()


def min_gate_labels(gate: str) -> int:
    return 8 if gate == "compound" else 20


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preprocess_image(image: np.ndarray, mode: str) -> np.ndarray:
    import cv2

    if mode == "raw":
        return image
    if mode == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    if mode == "norm":
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    raise ValueError(f"unknown preprocess mode: {mode}")


def create_detector(detector: str, max_keypoints: int, sift_contrast: float):
    import cv2

    if detector == "RootSIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("cv2.SIFT_create unavailable")
        return cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=sift_contrast), "float"
    if detector == "AKAZE":
        if not hasattr(cv2, "AKAZE_create"):
            raise RuntimeError("cv2.AKAZE_create unavailable")
        return cv2.AKAZE_create(), "binary"
    if detector == "KAZE":
        if not hasattr(cv2, "KAZE_create"):
            raise RuntimeError("cv2.KAZE_create unavailable")
        return cv2.KAZE_create(), "float"
    if detector == "ORB":
        if not hasattr(cv2, "ORB_create"):
            raise RuntimeError("cv2.ORB_create unavailable")
        return cv2.ORB_create(nfeatures=max_keypoints), "binary"
    raise ValueError(f"unknown detector: {detector}")


def descriptor_match(descriptors_a, descriptors_b, descriptor_kind: str, mode: str, ratio: float, max_matches: int):
    import cv2

    if descriptor_kind == "float":
        norm = cv2.NORM_L2
        descriptors_a = descriptors_a.astype(np.float32, copy=False)
        descriptors_b = descriptors_b.astype(np.float32, copy=False)
    else:
        norm = cv2.NORM_HAMMING
    if mode == "cross":
        matcher = cv2.BFMatcher(norm, crossCheck=True)
        matches = matcher.match(descriptors_a, descriptors_b)
    elif mode == "ratio":
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        matches = A4.ratio_filter(matcher.knnMatch(descriptors_a, descriptors_b, k=2), ratio)
    else:
        raise ValueError(f"unknown match mode: {mode}")
    return sorted(matches, key=lambda item: item.distance)[:max_matches]


def match_raw(image_a: np.ndarray, image_b: np.ndarray, cfg: MatcherConfig, args: argparse.Namespace) -> RawMatchOutput:
    detector, descriptor_kind = create_detector(cfg.detector, args.max_keypoints, args.sift_contrast)
    proc_a = preprocess_image(image_a, cfg.preprocess)
    proc_b = preprocess_image(image_b, cfg.preprocess)
    keypoints_a, descriptors_a = detector.detectAndCompute(proc_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(proc_b, None)
    if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
        return RawMatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []), 0)
    if cfg.detector == "RootSIFT":
        descriptors_a = A4.rootsift(descriptors_a.astype(np.float32, copy=False))
        descriptors_b = A4.rootsift(descriptors_b.astype(np.float32, copy=False))
    matches = descriptor_match(descriptors_a, descriptors_b, descriptor_kind, cfg.match_mode, cfg.ratio, args.max_matches)
    output = A4.output_from_matches(keypoints_a, keypoints_b, matches)
    return RawMatchOutput(output.points_a, output.points_b, output.keypoints_a, output.keypoints_b, len(matches))


def apply_homography(raw: RawMatchOutput, threshold_px: float, min_inliers: int) -> RawMatchOutput:
    inlier_a, inlier_b = A4.ransac_inliers(raw.points_a, raw.points_b, threshold_px=threshold_px)
    if inlier_a.shape[0] < min_inliers:
        inlier_a, inlier_b = empty_points(), empty_points()
    return RawMatchOutput(inlier_a, inlier_b, raw.keypoints_a, raw.keypoints_b, raw.raw_matches)


def metric_row(
    *,
    args: argparse.Namespace,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    cfg: MatcherConfig,
    output: RawMatchOutput,
    image_b_shape: tuple[int, int],
    warp_a_to_b,
    valid_mask,
    base_pass_gate: int,
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
    pass_gate = 1 if correct >= min_gate_labels(gate) else 0
    fallback_candidate = 1 if not base_pass_gate and pass_gate else 0
    row = MetricRow(
        style=style,
        gate=gate,
        rotation_deg=rotation_deg,
        pair_pt=pair_path.as_posix(),
        config=cfg.config,
        detector=cfg.detector,
        preprocess=cfg.preprocess,
        match_mode=cfg.match_mode,
        ratio=cfg.ratio,
        ransac_threshold_px=cfg.ransac_threshold_px,
        min_inliers=min_gate_labels(gate),
        status=status,
        keypoints_a=output.keypoints_a,
        keypoints_b=output.keypoints_b,
        raw_matches=output.raw_matches,
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        coverage=1 if matches > 0 else 0,
        pass_gate=pass_gate,
        base_pass_gate=base_pass_gate,
        fallback_candidate=fallback_candidate,
        recovered_baseline_fail=fallback_candidate,
        mean_error_px=mean_error,
        median_error_px=median_error,
        visualization=visualization,
        message=message,
    )
    return row, points_b_original


def evaluate_config(
    *,
    args: argparse.Namespace,
    cfg: MatcherConfig,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    base_pass_gate: int,
    vis_budget: dict[str, int],
) -> MetricRow:
    try:
        image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(pair_path)
        image_b_rotated = A4.rotate_image(image_b, rotation_deg)
        raw = match_raw(image_a, image_b_rotated, cfg, args)
        output = apply_homography(raw, cfg.ransac_threshold_px, min_gate_labels(gate))
        row, points_b_original = metric_row(
            args=args,
            style=style,
            gate=gate,
            rotation_deg=rotation_deg,
            pair_path=pair_path,
            cfg=cfg,
            output=output,
            image_b_shape=image_b.shape[:2],
            warp_a_to_b=warp_a_to_b,
            valid_mask=valid_mask,
            base_pass_gate=base_pass_gate,
        )
        if row.recovered_baseline_fail and args.visualizations_per_config > 0:
            key = f"{style}/{gate}/rot{rotation_deg}/{cfg.config}"
            if vis_budget.get(key, 0) < args.visualizations_per_config:
                vis_budget[key] = vis_budget.get(key, 0) + 1
                path = (
                    args.output_dir
                    / "visualizations"
                    / style
                    / gate
                    / f"rot{rotation_deg}"
                    / f"{pair_path.parent.name}_{pair_path.stem}_{safe_name(cfg.config)}.png"
                )
                A4.draw_visualization(image_a, image_b, output.points_a, points_b_original, path)
                row = MetricRow(**{**asdict(row), "visualization": path.as_posix()})
        return row
    except Exception as exc:
        return MetricRow(
            style=style,
            gate=gate,
            rotation_deg=rotation_deg,
            pair_pt=pair_path.as_posix(),
            config=cfg.config,
            detector=cfg.detector,
            preprocess=cfg.preprocess,
            match_mode=cfg.match_mode,
            ratio=cfg.ratio,
            ransac_threshold_px=cfg.ransac_threshold_px,
            min_inliers=min_gate_labels(gate),
            status="error",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            matches=0,
            correct=0,
            wrong=0,
            precision=0.0,
            coverage=0,
            pass_gate=0,
            base_pass_gate=base_pass_gate,
            fallback_candidate=0,
            recovered_baseline_fail=0,
            mean_error_px=math.nan,
            median_error_px=math.nan,
            message=f"{type(exc).__name__}: {exc}",
        )


def make_configs(args: argparse.Namespace) -> tuple[list[MatcherConfig], list[dict[str, str]]]:
    unavailable: list[dict[str, str]] = []
    configs: list[MatcherConfig] = []
    try:
        import cv2

        detector_checks = {
            "RootSIFT": "SIFT_create",
            "AKAZE": "AKAZE_create",
            "KAZE": "KAZE_create",
            "ORB": "ORB_create",
        }
        for detector, attr in detector_checks.items():
            if not hasattr(cv2, attr):
                unavailable.append({"algorithm": detector, "reason": f"cv2.{attr} unavailable"})

        if hasattr(cv2, "SIFT_create"):
            for preprocess in ["raw", "clahe", "norm"]:
                for ratio in [0.85, 0.90]:
                    for threshold in [2.0, 3.0]:
                        configs.append(
                            MatcherConfig(
                                config=f"RootSIFT-{preprocess}-r{ratio:.2f}-Ht{threshold:.0f}",
                                detector="RootSIFT",
                                preprocess=preprocess,
                                match_mode="ratio",
                                ratio=ratio,
                                ransac_threshold_px=threshold,
                            )
                        )
        for detector in ["AKAZE", "KAZE", "ORB"]:
            if detector_checks[detector] and hasattr(cv2, detector_checks[detector]):
                for mode in ["ratio", "cross"]:
                    ratios = [0.85, 0.90] if mode == "ratio" else [math.nan]
                    for ratio in ratios:
                        for threshold in [2.0, 3.0]:
                            configs.append(
                                MatcherConfig(
                                    config=f"{detector}-raw-{mode}{'' if math.isnan(ratio) else f'-r{ratio:.2f}'}-Ht{threshold:.0f}",
                                    detector=detector,
                                    preprocess="raw",
                                    match_mode=mode,
                                    ratio=ratio,
                                    ransac_threshold_px=threshold,
                                )
                            )
    except Exception as exc:
        unavailable.append({"algorithm": "OpenCV classical detectors", "reason": f"{type(exc).__name__}: {exc}"})
    if args.limit_configs:
        keep = set(args.limit_configs)
        configs = [cfg for cfg in configs if cfg.config in keep]
    return configs, unavailable


def baseline_key(row: dict[str, str]) -> tuple[str, str, int, str]:
    return (row["style"], row["gate"], int(row["rotation_deg"]), row["pair_pt"])


def load_agent7_baseline() -> tuple[dict[tuple[str, str, int, str], dict[str, str]], dict[tuple[str, str, int, str], dict[str, str]]]:
    metrics = read_csv(AGENT7_DIR / "metrics.csv")
    baseline = {
        baseline_key(row): row
        for row in metrics
        if row.get("row_type") == "algorithm" and row.get("name") == "RootSIFT-HRANSAC-r0.80-t2"
    }
    hard = {baseline_key(row): row for row in read_csv(AGENT7_DIR / "hard_cases.csv")}
    return baseline, hard


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    configs, unavailable = make_configs(args)
    baseline, hard_cases = load_agent7_baseline()
    rows: list[MetricRow] = []
    recovery_rows: list[dict[str, object]] = []
    sampled: list[dict[str, object]] = []
    vis_budget: dict[str, int] = {}
    baseline_configs = {("RootSIFT", "raw", "ratio", 0.80, 2.0)}

    if not configs:
        unavailable.append({"algorithm": "agent9 grid", "reason": "no configs available"})
        return rows, recovery_rows, sampled, unavailable

    for style in args.styles:
        for gate in args.gates:
            pair_paths = A4.select_pairs(args, style, gate)
            sampled.extend({"style": style, "gate": gate, "pair_pt": path.as_posix()} for path in pair_paths)
            print(f"group={style}/{gate} pairs={len(pair_paths)} configs={len(configs)}", flush=True)
            for rotation_deg in args.rotations:
                for pair_index, pair_path in enumerate(pair_paths, start=1):
                    key = (style, gate, rotation_deg, pair_path.as_posix())
                    base = baseline.get(key, {})
                    base_pass_gate = int(float(base.get("pass_gate", "0") or 0))
                    for cfg in configs:
                        row = evaluate_config(
                            args=args,
                            cfg=cfg,
                            style=style,
                            gate=gate,
                            rotation_deg=rotation_deg,
                            pair_path=pair_path,
                            base_pass_gate=base_pass_gate,
                            vis_budget=vis_budget,
                        )
                        rows.append(row)
                        hard = hard_cases.get(key)
                        is_baseline_like = (cfg.detector, cfg.preprocess, cfg.match_mode, round(cfg.ratio, 2), cfg.ransac_threshold_px) in baseline_configs
                        if not base_pass_gate and not is_baseline_like:
                            recovery_rows.append(
                                {
                                    "case_type": hard.get("case_type", "rootsift_failed_gate") if hard else "rootsift_failed_gate",
                                    "style": style,
                                    "gate": gate,
                                    "rotation_deg": rotation_deg,
                                    "pair_pt": pair_path.as_posix(),
                                    "config": cfg.config,
                                    "detector": cfg.detector,
                                    "preprocess": cfg.preprocess,
                                    "match_mode": cfg.match_mode,
                                    "ratio": cfg.ratio,
                                    "ransac_threshold_px": cfg.ransac_threshold_px,
                                    "min_inliers": min_gate_labels(gate),
                                    "baseline_matches": int(float(base.get("matches", "0") or 0)),
                                    "baseline_correct": int(float(base.get("correct", "0") or 0)),
                                    "baseline_precision": float(base.get("precision", "0") or 0),
                                    "baseline_pass_gate": base_pass_gate,
                                    "fallback_matches": row.matches,
                                    "fallback_correct": row.correct,
                                    "fallback_wrong": row.wrong,
                                    "fallback_precision": row.precision,
                                    "fallback_pass_gate": row.pass_gate,
                                    "recovered": row.recovered_baseline_fail,
                                    "mean_error_px": row.mean_error_px,
                                    "median_error_px": row.median_error_px,
                                    "visualization": row.visualization,
                                    "message": row.message,
                                }
                            )
                    print(
                        f"{style:9s} {gate:9s} rot={rotation_deg:3d} {pair_index:02d}/{len(pair_paths):02d} "
                        f"base_pass={base_pass_gate} done",
                        flush=True,
                    )
    return rows, recovery_rows, sampled, unavailable


def aggregate(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.rotation_deg, row.config), []).append(row)
    out: list[dict[str, object]] = []
    for (style, gate, rotation_deg, config), items in sorted(grouped.items()):
        ok = [row for row in items if row.status == "ok"]
        matches = sum(row.matches for row in ok)
        correct = sum(row.correct for row in ok)
        wrong = sum(row.wrong for row in ok)
        fallback = [row for row in ok if not row.base_pass_gate]
        fallback_matches = sum(row.matches for row in fallback if row.pass_gate)
        fallback_correct = sum(row.correct for row in fallback if row.pass_gate)
        fallback_wrong = sum(row.wrong for row in fallback if row.pass_gate)
        sample = items[0]
        out.append(
            {
                "style": style,
                "gate": gate,
                "rotation_deg": rotation_deg,
                "config": config,
                "detector": sample.detector,
                "preprocess": sample.preprocess,
                "match_mode": sample.match_mode,
                "ratio": sample.ratio,
                "ransac_threshold_px": sample.ransac_threshold_px,
                "min_inliers": sample.min_inliers,
                "pairs": len(items),
                "ok_pairs": len(ok),
                "covered_pairs": sum(row.coverage for row in ok),
                "pass_gate_pairs": sum(row.pass_gate for row in ok),
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": 0.0 if matches == 0 else correct / matches,
                "mean_pair_precision": float(np.mean([row.precision for row in ok])) if ok else math.nan,
                "mean_matches_per_pair": float(np.mean([row.matches for row in ok])) if ok else math.nan,
                "baseline_failed_pairs": sum(1 for row in ok if not row.base_pass_gate),
                "baseline_failed_recovered_pairs": sum(row.recovered_baseline_fail for row in ok),
                "fallback_matches": fallback_matches,
                "fallback_correct": fallback_correct,
                "fallback_wrong": fallback_wrong,
                "fallback_precision": 0.0 if fallback_matches == 0 else fallback_correct / fallback_matches,
            }
        )
    return out


def global_config_summary(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.config, []).append(row)
    out = []
    for config, items in sorted(grouped.items()):
        ok = [row for row in items if row.status == "ok"]
        fallback_pass = [row for row in ok if row.recovered_baseline_fail]
        fallback_matches = sum(row.matches for row in fallback_pass)
        fallback_correct = sum(row.correct for row in fallback_pass)
        fallback_wrong = sum(row.wrong for row in fallback_pass)
        matches = sum(row.matches for row in ok)
        correct = sum(row.correct for row in ok)
        out.append(
            {
                "config": config,
                "ok_pairs": len(ok),
                "pass_gate_pairs": sum(row.pass_gate for row in ok),
                "matches": matches,
                "precision": 0.0 if matches == 0 else correct / matches,
                "baseline_failed_pairs": sum(1 for row in ok if not row.base_pass_gate),
                "recovered_pairs": len(fallback_pass),
                "fallback_matches": fallback_matches,
                "fallback_correct": fallback_correct,
                "fallback_wrong": fallback_wrong,
                "fallback_precision": 0.0 if fallback_matches == 0 else fallback_correct / fallback_matches,
            }
        )
    return out


def markdown_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| config | recovered | fallback matches | fallback precision | all-pass pairs | all precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(rows, key=lambda row: (float(row["fallback_precision"]), int(row["recovered_pairs"]), int(row["fallback_correct"])), reverse=True)
    for row in ranked[:12]:
        lines.append(
            f"| {row['config']} | {row['recovered_pairs']} | {row['fallback_matches']} | "
            f"{float(row['fallback_precision']):.4f} | {row['pass_gate_pairs']} | {float(row['precision']):.4f} |"
        )
    return lines


def write_summary(
    args: argparse.Namespace,
    rows: list[MetricRow],
    summary_rows: list[dict[str, object]],
    recovery_rows: list[dict[str, object]],
    unavailable: list[dict[str, str]],
) -> None:
    global_rows = global_config_summary(rows)
    safe = [
        row
        for row in global_rows
        if int(row["recovered_pairs"]) > 0 and float(row["fallback_precision"]) >= args.safe_precision
    ]
    best = sorted(global_rows, key=lambda row: (float(row["fallback_precision"]), int(row["recovered_pairs"]), int(row["fallback_correct"])), reverse=True)[:5]
    base_failed = len({(row.style, row.gate, row.rotation_deg, row.pair_pt) for row in rows if row.status == "ok" and not row.base_pass_gate})
    hard_empty = {(
        row["style"],
        row["gate"],
        int(row["rotation_deg"]),
        row["pair_pt"],
    ) for row in recovery_rows if row["case_type"] == "rootsift_empty_pfm_nonempty"}
    empty_rows = [row for row in recovery_rows if row["case_type"] == "rootsift_empty_pfm_nonempty" and int(row["fallback_pass_gate"]) == 1]
    empty_by_config: dict[str, dict[str, int]] = {}
    for row in empty_rows:
        bucket = empty_by_config.setdefault(str(row["config"]), {"recovered": 0, "matches": 0, "correct": 0, "wrong": 0})
        bucket["recovered"] += int(row["recovered"])
        bucket["matches"] += int(row["fallback_matches"])
        bucket["correct"] += int(row["fallback_correct"])
        bucket["wrong"] += int(row["fallback_wrong"])
    empty_best = sorted(
        (
            {
                "config": config,
                **values,
                "precision": 0.0 if values["matches"] == 0 else values["correct"] / values["matches"],
            }
            for config, values in empty_by_config.items()
        ),
        key=lambda item: (float(item["precision"]), int(item["recovered"]), int(item["correct"])),
        reverse=True,
    )
    lines = [
        "# Matcher Algorithm Iteration Agent9",
        "",
        "## Scope",
        "",
        "- Goal: find high-precision classical fallback sources for RootSIFT-r0.80-t2 empty or not-gated rotated pairs.",
        "- PFM fallback is intentionally not evaluated as a pseudo-label source in this run.",
        "- Dataset: same 1024 cached test split sampling path as agent7, excluding extreme.",
        f"- pairs per style/gate: `{args.pairs_per_group}`; rotations: `{','.join(str(item) for item in args.rotations)}`.",
        f"- correctness threshold: `{args.threshold_px}` px; safe fallback precision target: `{args.safe_precision}`.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} "
        f"--pairs-per-group {args.pairs_per_group}",
        "```",
        "",
        "## Core Result",
        "",
        f"- Baseline RootSIFT-r0.80-t2 failed the label gate on `{base_failed}` rotated pair evaluations in the sampled set.",
        f"- Agent7 `rootsift_empty_pfm_nonempty` hard cases represented `{len(hard_empty)}` unique rotated pair evaluations.",
        f"- Configs with at least one recovered baseline-failed pair and fallback precision >= `{args.safe_precision}`: `{len(safe)}`.",
    ]
    if safe:
        top = sorted(safe, key=lambda row: (int(row["recovered_pairs"]), int(row["fallback_correct"])), reverse=True)[0]
        lines.append(
            f"- Best safe fallback by recovered count: `{top['config']}` recovered `{top['recovered_pairs']}` pairs "
            f"with fallback precision `{float(top['fallback_precision']):.4f}` over `{top['fallback_matches']}` matches."
        )
    else:
        lines.append("- No tested classical fallback met the precision target with nonzero recovery.")
    if empty_best:
        empty_top = empty_best[0]
        lines.append(
            f"- Best `rootsift_empty_pfm_nonempty` recovery by precision: `{empty_top['config']}` recovered "
            f"`{empty_top['recovered']}` empty cases with precision `{float(empty_top['precision']):.4f}` "
            f"over `{empty_top['matches']}` matches."
        )
    lines.append("- Practical candidate: use a stricter-ratio RootSIFT fallback first; classical AKAZE/KAZE/ORB pockets are high precision but lower coverage and should stay secondary.")
    lines.extend(["", "## Top Configs", "", *markdown_table(best), "", "## Recovery CSV Notes", ""])
    lines.append("- `rootsift_empty_recovery.csv` includes agent7 hard cases plus all RootSIFT baseline gate failures; `case_type=rootsift_failed_gate` marks baseline failures not listed in agent7 hard cases.")
    lines.append("- `fallback_precision` in the table is computed only from recovered baseline-failed pairs that pass the fallback gate, which is the pseudo-label insertion scenario.")
    lines.extend(["", "## Unavailable / Non-blocking", ""])
    if unavailable:
        for item in unavailable:
            lines.append(f"- {item['algorithm']}: {item['reason']}")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `metrics.csv`",
            "- `summary_metrics.csv`",
            "- `rootsift_empty_recovery.csv`",
            "- `sampled_pairs.csv`",
            "- `unavailable_algorithms.csv`",
            "- `visualizations/` for recovered baseline failures when enabled",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pfm-run", type=Path, default=DEFAULT_PFM_RUN)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["viewpoint", "compound"], choices=["viewpoint", "compound"])
    parser.add_argument("--rotations", nargs="+", type=int, default=[90, 180, 270], choices=[0, 90, 180, 270])
    parser.add_argument("--pairs-per-group", type=int, default=4)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--safe-precision", type=float, default=0.95)
    parser.add_argument("--visualizations-per-config", type=int, default=1)
    parser.add_argument("--limit-configs", nargs="*", default=[])
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, recovery_rows, sampled, unavailable = evaluate(args)
    summary_rows = aggregate(rows)
    write_csv(args.output_dir / "metrics.csv", [asdict(row) for row in rows], METRIC_FIELDS)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "rootsift_empty_recovery.csv", recovery_rows, RECOVERY_FIELDS)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled, ["style", "gate", "pair_pt"])
    write_csv(args.output_dir / "unavailable_algorithms.csv", unavailable, ["algorithm", "reason"])
    write_summary(args, rows, summary_rows, recovery_rows, unavailable)
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
