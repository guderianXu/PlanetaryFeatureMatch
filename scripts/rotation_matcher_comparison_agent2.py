#!/usr/bin/env python3
"""Agent2 1024 rotation matcher comparison with post-RANSAC metrics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import rotation_matcher_benchmark as bench  # noqa: E402


CSV_FIELDS = [
    "image_style",
    "source_dir",
    "image_path",
    "angle",
    "matcher",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "raw_correct",
    "raw_wrong",
    "raw_precision",
    "raw_mean_error_px",
    "raw_median_error_px",
    "homography_matches",
    "homography_correct",
    "homography_wrong",
    "homography_precision",
    "homography_mean_error_px",
    "homography_median_error_px",
    "fundamental_matches",
    "fundamental_correct",
    "fundamental_wrong",
    "fundamental_precision",
    "fundamental_mean_error_px",
    "fundamental_median_error_px",
    "raw_visualization",
    "homography_visualization",
    "message",
]


@dataclass(frozen=True)
class ExtendedRow:
    image_style: str
    source_dir: str
    image_path: str
    angle: int
    matcher: str
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    raw_correct: int
    raw_wrong: int
    raw_precision: float
    raw_mean_error_px: float
    raw_median_error_px: float
    homography_matches: int
    homography_correct: int
    homography_wrong: int
    homography_precision: float
    homography_mean_error_px: float
    homography_median_error_px: float
    fundamental_matches: int
    fundamental_correct: int
    fundamental_wrong: int
    fundamental_precision: float
    fundamental_mean_error_px: float
    fundamental_median_error_px: float
    raw_visualization: str
    homography_visualization: str
    message: str


class SiftFlannRatioMatcher:
    def __init__(self, *, name: str, detector, max_matches: int, ratio: float, rootsift: bool) -> None:
        self.name = name
        self._detector = detector
        self._max_matches = max_matches
        self._ratio = ratio
        self._rootsift = rootsift

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> bench.MatchOutput:
        import cv2

        keypoints_a, descriptors_a = self._detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self._detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return bench.MatchOutput(bench.empty_points(), bench.empty_points(), len(keypoints_a or []), len(keypoints_b or []))
        descriptors_a = descriptors_a.astype(np.float32, copy=False)
        descriptors_b = descriptors_b.astype(np.float32, copy=False)
        if self._rootsift:
            descriptors_a = bench.normalize_sift_descriptors_to_rootsift(descriptors_a)
            descriptors_b = bench.normalize_sift_descriptors_to_rootsift(descriptors_b)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        forward = bench.ratio_filter_knn_matches(matcher.knnMatch(descriptors_a, descriptors_b, k=2), self._ratio)
        reverse = bench.ratio_filter_knn_matches(matcher.knnMatch(descriptors_b, descriptors_a, k=2), self._ratio)
        reverse_best = {item.queryIdx: item.trainIdx for item in reverse}
        mutual = [item for item in forward if reverse_best.get(item.trainIdx) == item.queryIdx]
        mutual = sorted(mutual, key=lambda item: item.distance)[: self._max_matches]
        points_a = np.array([keypoints_a[item.queryIdx].pt for item in mutual], dtype=np.float32)
        points_b = np.array([keypoints_b[item.trainIdx].pt for item in mutual], dtype=np.float32)
        return bench.MatchOutput(points_a, points_b, len(keypoints_a), len(keypoints_b))


def unavailable(name: str, reason: str) -> bench.UnavailableMatcher:
    return bench.UnavailableMatcher(name, reason)


def make_matchers(args: argparse.Namespace) -> list[bench.Matcher]:
    matchers: list[bench.Matcher] = []
    try:
        import cv2
    except Exception as exc:
        return [unavailable("OpenCV", f"OpenCV unavailable: {exc}")]

    if hasattr(cv2, "SIFT_create"):
        matchers.append(
            bench.OpenCvFeatureMatcher(
                "SIFT-BF-crosscheck",
                cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.sift_contrast),
                cv2.NORM_L2,
                max_matches=args.max_matches,
            )
        )
        matchers.append(
            SiftFlannRatioMatcher(
                name="SIFT-FLANN-ratio",
                detector=cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.sift_contrast),
                max_matches=args.max_matches,
                ratio=args.ratio,
                rootsift=False,
            )
        )
        matchers.append(
            SiftFlannRatioMatcher(
                name="RootSIFT-FLANN-ratio",
                detector=cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.sift_contrast),
                max_matches=args.max_matches,
                ratio=args.ratio,
                rootsift=True,
            )
        )
    else:
        matchers.append(unavailable("SIFT-BF-crosscheck", "cv2.SIFT_create unavailable"))
        matchers.append(unavailable("SIFT-FLANN-ratio", "cv2.SIFT_create unavailable"))
        matchers.append(unavailable("RootSIFT-FLANN-ratio", "cv2.SIFT_create unavailable"))

    matchers.append(
        bench.OpenCvFeatureMatcher(
            "ORB-BF-crosscheck",
            cv2.ORB_create(nfeatures=args.max_keypoints, fastThreshold=args.orb_fast_threshold),
            cv2.NORM_HAMMING,
            max_matches=args.max_matches,
        )
    )
    matchers.append(
        bench.OpenCvFeatureMatcher(
            "AKAZE-BF-crosscheck",
            cv2.AKAZE_create(threshold=args.akaze_threshold),
            cv2.NORM_HAMMING,
            max_matches=args.max_matches,
        )
    )
    matchers.extend(make_optional_matchers(args))
    matchers.append(make_pfm_matcher(args))
    return matchers


def make_optional_matchers(args: argparse.Namespace) -> list[bench.Matcher]:
    selected = {"LightGlue-SIFT", "LightGlue-SuperPoint", "LightGlue-DISK", "LightGlue-ALIKED", "LoFTR", "SuperGlue"}
    return [
        matcher
        for matcher in bench.make_optional_deep_matchers(
            device=args.device,
            max_keypoints=args.max_keypoints,
            max_matches=args.max_matches,
            loftr_pretrained=args.loftr_pretrained,
        )
        if matcher.name in selected
    ]


def make_pfm_matcher(args: argparse.Namespace) -> bench.Matcher:
    state = args.pfm_pytorch_state
    if state is None:
        state = (
            PROJECT_ROOT
            / "runs"
            / "cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234"
            / "training"
            / "pytorch_pfm_state.pt"
        )
    if not state.exists():
        return unavailable("PFM", f"PFM state not found: {state}")
    if importlib.util.find_spec("torch") is None:
        return unavailable("PFM", "PyTorch is not installed")
    if importlib.util.find_spec("pfm_model") is None:
        return unavailable("PFM", "pfm_model import failed; run with PYTHONPATH=python")
    return bench.PFMPyTorchMatcher(
        state_path=state,
        device=args.device,
        max_keypoints=args.max_keypoints,
        max_matches=args.max_matches,
        min_intensity=args.min_intensity,
        min_score=args.pfm_min_score,
    )


def fixed_1024_images(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    return [
        ("numeric", args.numeric_source_dir, args.numeric_source_dir / "source_000000_view_a.png"),
        ("timestamp", args.timestamp_source_dir, args.timestamp_source_dir / "source_000069_view_a.png"),
    ]


def ransac_inliers(points_a: np.ndarray, points_b: np.ndarray, *, model: str, threshold_px: float) -> tuple[np.ndarray, np.ndarray]:
    if points_a.size == 0 or points_b.size == 0:
        return bench.empty_points(), bench.empty_points()
    try:
        import cv2
    except Exception:
        return bench.empty_points(), bench.empty_points()
    mask = None
    if model == "homography" and points_a.shape[0] >= 4:
        method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
        _, mask = cv2.findHomography(
            points_a,
            points_b,
            method=method,
            ransacReprojThreshold=threshold_px,
            maxIters=3000,
            confidence=0.995,
        )
    elif model == "fundamental" and points_a.shape[0] >= 8:
        method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC
        _, mask = cv2.findFundamentalMat(
            points_a,
            points_b,
            method=method,
            ransacReprojThreshold=threshold_px,
            confidence=0.995,
            maxIters=3000,
        )
    if mask is None:
        return bench.empty_points(), bench.empty_points()
    keep = mask.reshape(-1).astype(bool)
    return points_a[keep], points_b[keep]


def metric_tuple(points_a: np.ndarray, points_b: np.ndarray, *, width: int, height: int, angle: int, threshold_px: float):
    correct, wrong, precision, mean_error, median_error = bench.compute_metrics(
        points_a,
        points_b,
        width=width,
        height=height,
        angle=angle,
        threshold_px=threshold_px,
    )
    return int(points_a.shape[0]), correct, wrong, precision, mean_error, median_error


def evaluate_one(
    matcher: bench.Matcher,
    image: np.ndarray,
    rotated: np.ndarray,
    *,
    image_style: str,
    source_dir: Path,
    image_path: Path,
    angle: int,
    output_dir: Path,
    threshold_px: float,
    ransac_threshold_px: float,
) -> tuple[ExtendedRow, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_vis = output_dir / "visualizations" / "raw" / f"{image_style}_{angle}_{bench.safe_name(matcher.name)}.png"
    h_vis = output_dir / "visualizations" / "homography" / f"{image_style}_{angle}_{bench.safe_name(matcher.name)}.png"
    try:
        output = matcher.match(image, rotated)
        raw = metric_tuple(
            output.points_a,
            output.points_b,
            width=image.shape[1],
            height=image.shape[0],
            angle=angle,
            threshold_px=threshold_px,
        )
        h_a, h_b = ransac_inliers(output.points_a, output.points_b, model="homography", threshold_px=ransac_threshold_px)
        homography = metric_tuple(h_a, h_b, width=image.shape[1], height=image.shape[0], angle=angle, threshold_px=threshold_px)
        f_a, f_b = ransac_inliers(output.points_a, output.points_b, model="fundamental", threshold_px=ransac_threshold_px)
        fundamental = metric_tuple(f_a, f_b, width=image.shape[1], height=image.shape[0], angle=angle, threshold_px=threshold_px)
        row = ExtendedRow(
            image_style=image_style,
            source_dir=source_dir.as_posix(),
            image_path=image_path.as_posix(),
            angle=angle,
            matcher=matcher.name,
            status="ok",
            keypoints_a=output.keypoints_a,
            keypoints_b=output.keypoints_b,
            raw_matches=raw[0],
            raw_correct=raw[1],
            raw_wrong=raw[2],
            raw_precision=raw[3],
            raw_mean_error_px=raw[4],
            raw_median_error_px=raw[5],
            homography_matches=homography[0],
            homography_correct=homography[1],
            homography_wrong=homography[2],
            homography_precision=homography[3],
            homography_mean_error_px=homography[4],
            homography_median_error_px=homography[5],
            fundamental_matches=fundamental[0],
            fundamental_correct=fundamental[1],
            fundamental_wrong=fundamental[2],
            fundamental_precision=fundamental[3],
            fundamental_mean_error_px=fundamental[4],
            fundamental_median_error_px=fundamental[5],
            raw_visualization=raw_vis.as_posix() if output.points_a.size else "",
            homography_visualization=h_vis.as_posix() if h_a.size else "",
            message="",
        )
        return row, output.points_a, output.points_b, h_a, h_b
    except Exception as exc:
        status = "unavailable" if isinstance(matcher, bench.UnavailableMatcher) else "error"
        empty = bench.empty_points()
        row = ExtendedRow(
            image_style=image_style,
            source_dir=source_dir.as_posix(),
            image_path=image_path.as_posix(),
            angle=angle,
            matcher=matcher.name,
            status=status,
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            raw_correct=0,
            raw_wrong=0,
            raw_precision=0.0,
            raw_mean_error_px=math.nan,
            raw_median_error_px=math.nan,
            homography_matches=0,
            homography_correct=0,
            homography_wrong=0,
            homography_precision=0.0,
            homography_mean_error_px=math.nan,
            homography_median_error_px=math.nan,
            fundamental_matches=0,
            fundamental_correct=0,
            fundamental_wrong=0,
            fundamental_precision=0.0,
            fundamental_mean_error_px=math.nan,
            fundamental_median_error_px=math.nan,
            raw_visualization="",
            homography_visualization="",
            message=str(exc),
        )
        return row, empty, empty, empty, empty


def evaluate(args: argparse.Namespace) -> list[ExtendedRow]:
    rows: list[ExtendedRow] = []
    matchers = make_matchers(args)
    for image_style, source_dir, image_path in fixed_1024_images(args):
        image = bench.resize_long_edge(bench.read_image(image_path), args.resize_max)
        for angle in args.angles:
            rotated = bench.rotate_image(image, angle)
            for matcher in matchers:
                row, raw_a, raw_b, h_a, h_b = evaluate_one(
                    matcher,
                    image,
                    rotated,
                    image_style=image_style,
                    source_dir=source_dir,
                    image_path=image_path,
                    angle=angle,
                    output_dir=args.output_dir,
                    threshold_px=args.threshold_px,
                    ransac_threshold_px=args.ransac_threshold_px,
                )
                rows.append(row)
                if row.raw_visualization:
                    bench.draw_visualization(image, rotated, raw_a, raw_b, Path(row.raw_visualization))
                if row.homography_visualization:
                    bench.draw_visualization(image, rotated, h_a, h_b, Path(row.homography_visualization))
                print(
                    f"{image_style:9s} {angle:3d} {matcher.name:24s} {row.status:11s} "
                    f"raw={row.raw_matches:4d}/{row.raw_precision:.4f} "
                    f"H={row.homography_matches:4d}/{row.homography_precision:.4f} "
                    f"F={row.fundamental_matches:4d}/{row.fundamental_precision:.4f}",
                    flush=True,
                )
    return rows


def write_csv(path: Path, rows: Iterable[ExtendedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            for key, value in list(data.items()):
                if isinstance(value, float):
                    data[key] = "nan" if math.isnan(value) else f"{value:.6f}"
            writer.writerow(data)


def aggregate_rows(rows: list[ExtendedRow], keys: tuple[str, ...], metric_prefix: str) -> list[tuple[tuple[str, ...], dict[str, float | int]]]:
    grouped: dict[tuple[str, ...], list[ExtendedRow]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(getattr(row, key)) for key in keys), []).append(row)
    output = []
    for group_key, subset in sorted(grouped.items()):
        ok = [row for row in subset if row.status == "ok"]
        matches = sum(int(getattr(row, f"{metric_prefix}_matches")) for row in ok)
        correct = sum(int(getattr(row, f"{metric_prefix}_correct")) for row in ok)
        wrong = sum(int(getattr(row, f"{metric_prefix}_wrong")) for row in ok)
        errors = [float(getattr(row, f"{metric_prefix}_mean_error_px")) for row in ok]
        errors = [value for value in errors if not math.isnan(value)]
        output.append(
            (
                group_key,
                {
                    "ok": len(ok),
                    "non_ok": len(subset) - len(ok),
                    "matches": matches,
                    "correct": correct,
                    "wrong": wrong,
                    "precision": 0.0 if matches == 0 else correct / matches,
                    "mean_error_px": math.nan if not errors else sum(errors) / len(errors),
                },
            )
        )
    return output


def summary_table(rows: list[ExtendedRow], keys: tuple[str, ...], metric_prefix: str) -> list[str]:
    label = " | ".join(keys)
    lines = [
        f"| {label} | ok | non_ok | matches | correct | wrong | precision | mean_error_px |",
        f"| {' | '.join(['---'] * len(keys))} |---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_key, values in aggregate_rows(rows, keys, metric_prefix):
        err = values["mean_error_px"]
        err_text = "nan" if isinstance(err, float) and math.isnan(err) else f"{float(err):.3f}"
        lines.append(
            "| "
            + " | ".join(group_key)
            + f" | {values['ok']} | {values['non_ok']} | {values['matches']} | {values['correct']} | "
            + f"{values['wrong']} | {float(values['precision']):.6f} | {err_text} |"
        )
    return lines


def write_summary(path: Path, rows: list[ExtendedRow], args: argparse.Namespace) -> None:
    lines = [
        "# Rotation Matcher Comparison Agent2",
        "",
        "## Setup",
        "",
        f"- dataset: `img/Rotate_1024`",
        f"- numeric source: `{args.numeric_source_dir}`",
        f"- timestamp source: `{args.timestamp_source_dir}`",
        f"- angles: `{','.join(str(angle) for angle in args.angles)}`",
        f"- PFM state: `{args.pfm_pytorch_state or (PROJECT_ROOT / 'runs' / 'cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234' / 'training' / 'pytorch_pfm_state.pt')}`",
        f"- thresholds: correctness `{args.threshold_px}` px, RANSAC `{args.ransac_threshold_px}` px",
        "",
        "## Raw Aggregate By Matcher",
        "",
        *summary_table(rows, ("matcher",), "raw"),
        "",
        "## Homography RANSAC Aggregate By Matcher",
        "",
        *summary_table(rows, ("matcher",), "homography"),
        "",
        "## Raw Split By Style And Angle",
        "",
        *summary_table(rows, ("image_style", "angle", "matcher"), "raw"),
        "",
        "## Homography Split By Style And Angle",
        "",
        *summary_table(rows, ("image_style", "angle", "matcher"), "homography"),
    ]
    unavailable = sorted({(row.matcher, row.message) for row in rows if row.status != "ok" and row.message})
    if unavailable:
        lines.extend(["", "## Unavailable Or Error", ""])
        for matcher, message in unavailable:
            lines.append(f"- {matcher}: {message}")
    lines.extend(["", "metrics: `metrics.csv`", "visualizations: `visualizations/raw`, `visualizations/homography`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    points = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [4, 5], [20, 20]], dtype=np.float32)
    rotated = bench.rotate_points(points, width=32, height=32, angle=180)
    rotated[-1] = np.array([3, 29], dtype=np.float32)
    h_a, h_b = ransac_inliers(points, rotated, model="homography", threshold_px=1.0)
    assert 4 <= h_a.shape[0] <= 5, h_a.shape[0]
    matches, correct, wrong, precision, _, _ = metric_tuple(h_a, h_b, width=32, height=32, angle=180, threshold_px=1.0)
    assert matches == correct and wrong == 0 and precision == 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-source-dir", type=Path, default=PROJECT_ROOT / "img" / "Rotate_1024" / "source_000000_1")
    parser.add_argument(
        "--timestamp-source-dir",
        type=Path,
        default=PROJECT_ROOT / "img" / "Rotate_1024" / "source_000069_20260514T064636672_NAS_PAN_L2b",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "rotation_matcher_comparison_agent2")
    parser.add_argument("--angles", type=int, nargs="+", default=[90, 180, 270], choices=[90, 180, 270])
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--resize-max", type=int, default=1024)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--orb-fast-threshold", type=int, default=5)
    parser.add_argument("--akaze-threshold", type=float, default=0.0003)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--pfm-pytorch-state", type=Path, default=None)
    parser.add_argument("--loftr-pretrained", default="outdoor", choices=["outdoor", "indoor", "indoor_new"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate(args)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_summary(args.output_dir / "summary.md", rows, args)
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

