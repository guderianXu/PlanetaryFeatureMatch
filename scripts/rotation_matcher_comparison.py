#!/usr/bin/env python3
"""Run a two-pass rotation matcher comparison without touching training code."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import rotation_matcher_benchmark as bench  # noqa: E402


COMPARISON_FIELDS = ["phase", *bench.CSV_FIELDS]


class GeometricFilterMatcher:
    """Filter a matcher output with a robust affine model estimated from matches."""

    def __init__(self, wrapped: bench.Matcher, *, threshold_px: float, min_matches: int = 3) -> None:
        self._wrapped = wrapped
        self._threshold_px = threshold_px
        self._min_matches = min_matches
        self.name = f"{wrapped.name}-AffineRANSAC"

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> bench.MatchOutput:
        import cv2

        output = self._wrapped.match(image_a, image_b)
        if output.points_a.shape[0] < self._min_matches:
            return output
        _, inliers = cv2.estimateAffinePartial2D(
            output.points_a,
            output.points_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._threshold_px,
            maxIters=3000,
            confidence=0.995,
        )
        if inliers is None:
            return output
        mask = inliers.reshape(-1).astype(bool)
        return bench.MatchOutput(
            output.points_a[mask],
            output.points_b[mask],
            output.keypoints_a,
            output.keypoints_b,
        )


class ClaheMatcher:
    """Apply local contrast normalization before running the wrapped matcher."""

    def __init__(self, wrapped: bench.Matcher, *, clip_limit: float = 2.0, tile_grid_size: int = 8) -> None:
        self._wrapped = wrapped
        self._clip_limit = clip_limit
        self._tile_grid_size = tile_grid_size
        self.name = f"{wrapped.name}-CLAHE"

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> bench.MatchOutput:
        return self._wrapped.match(self._clahe(image_a), self._clahe(image_b))

    def _clahe(self, image: np.ndarray) -> np.ndarray:
        import cv2

        if image.ndim == 3:
            image = image.mean(axis=2).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=(self._tile_grid_size, self._tile_grid_size),
        )
        return clahe.apply(image.astype(np.uint8, copy=False))


def self_test() -> None:
    points = np.array([[0, 0], [7, 0], [0, 5], [7, 5], [3, 2]], dtype=np.float32)
    np.testing.assert_allclose(
        bench.rotate_points(points, width=8, height=6, angle=180),
        np.array([[7, 5], [0, 5], [7, 0], [0, 0], [4, 3]], dtype=np.float32),
    )
    correct, wrong, precision, mean_error, median_error = bench.compute_metrics(
        points,
        bench.rotate_points(points, width=8, height=6, angle=180),
        width=8,
        height=6,
        angle=180,
        threshold_px=0.01,
    )
    assert (correct, wrong, precision) == (5, 0, 1.0)
    assert mean_error == 0.0 and median_error == 0.0


def unavailable(name: str, reason: str) -> bench.UnavailableMatcher:
    return bench.UnavailableMatcher(name, reason)


def make_baseline_matchers(args: argparse.Namespace) -> list[bench.Matcher]:
    matchers: list[bench.Matcher] = []
    try:
        import cv2
    except Exception as exc:
        reason = f"OpenCV unavailable: {exc}"
        return [
            unavailable("SIFT", reason),
            unavailable("ORB", reason),
            unavailable("AKAZE", reason),
            unavailable("LightGlue-SIFT", reason),
            unavailable("LightGlue-SuperPoint", reason),
            unavailable("SuperGlue", reason),
            unavailable("PFM", reason),
        ]

    if hasattr(cv2, "SIFT_create"):
        matchers.append(
            bench.OpenCvFeatureMatcher(
                "SIFT",
                cv2.SIFT_create(nfeatures=args.max_keypoints),
                cv2.NORM_L2,
                max_matches=args.max_matches,
            )
        )
    else:
        matchers.append(unavailable("SIFT", "cv2.SIFT_create unavailable"))

    matchers.append(
        bench.OpenCvFeatureMatcher(
            "ORB",
            cv2.ORB_create(nfeatures=args.max_keypoints),
            cv2.NORM_HAMMING,
            max_matches=args.max_matches,
        )
    )
    matchers.append(
        bench.OpenCvFeatureMatcher(
            "AKAZE",
            cv2.AKAZE_create(),
            cv2.NORM_HAMMING,
            max_matches=args.max_matches,
        )
    )

    matchers.extend(make_selected_optional_matchers(args))
    matchers.append(make_pfm_matcher(args))
    return matchers


def make_iterated_matchers(args: argparse.Namespace) -> list[bench.Matcher]:
    matchers: list[bench.Matcher] = []
    try:
        import cv2
    except Exception as exc:
        return [unavailable("iteration", f"OpenCV unavailable: {exc}")]

    if hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.iter_sift_contrast)
        matchers.append(
            bench.RootSiftFlannRansacMatcher(
                sift,
                max_matches=args.max_matches,
                name="RootSIFT-FLANN-RANSAC",
                ratio=args.iter_ratio,
                ransac_threshold_px=args.iter_ransac_threshold_px,
            )
        )
        if hasattr(cv2, "USAC_MAGSAC"):
            matchers.append(
                bench.RootSiftFlannRansacMatcher(
                    cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.iter_sift_contrast),
                    max_matches=args.max_matches,
                    name="RootSIFT-FLANN-USAC-MAGSAC",
                    geometric_method=cv2.USAC_MAGSAC,
                    geometric_model="homography_usac",
                    ratio=args.iter_ratio,
                    ransac_threshold_px=args.iter_ransac_threshold_px,
                )
            )
        else:
            matchers.append(unavailable("RootSIFT-FLANN-USAC-MAGSAC", "cv2.USAC_MAGSAC unavailable"))
        matchers.append(
            ClaheMatcher(
                bench.RootSiftFlannRansacMatcher(
                    cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.iter_sift_contrast),
                    max_matches=args.max_matches,
                    name="RootSIFT-FLANN-RANSAC",
                    ratio=args.iter_ratio,
                    ransac_threshold_px=args.iter_ransac_threshold_px,
                )
            )
        )
    else:
        matchers.append(unavailable("RootSIFT-FLANN-RANSAC", "cv2.SIFT_create unavailable"))
        matchers.append(unavailable("RootSIFT-FLANN-USAC-MAGSAC", "cv2.SIFT_create unavailable"))

    orb = bench.OpenCvFeatureMatcher(
        "ORB",
        cv2.ORB_create(nfeatures=args.max_keypoints, fastThreshold=args.iter_orb_fast_threshold),
        cv2.NORM_HAMMING,
        max_matches=args.max_matches,
    )
    akaze = bench.OpenCvFeatureMatcher(
        "AKAZE",
        cv2.AKAZE_create(threshold=args.iter_akaze_threshold),
        cv2.NORM_HAMMING,
        max_matches=args.max_matches,
    )
    matchers.append(GeometricFilterMatcher(orb, threshold_px=args.iter_ransac_threshold_px))
    matchers.append(GeometricFilterMatcher(akaze, threshold_px=args.iter_ransac_threshold_px))
    matchers.append(ClaheMatcher(GeometricFilterMatcher(orb, threshold_px=args.iter_ransac_threshold_px)))
    matchers.append(ClaheMatcher(GeometricFilterMatcher(akaze, threshold_px=args.iter_ransac_threshold_px)))

    matchers.extend(make_selected_optional_matchers(args))
    matchers.append(make_pfm_matcher(args))
    return matchers


def make_selected_optional_matchers(args: argparse.Namespace) -> list[bench.Matcher]:
    all_optional = bench.make_optional_deep_matchers(
        device=args.device,
        max_keypoints=args.max_keypoints,
        max_matches=args.max_matches,
        loftr_pretrained=args.loftr_pretrained,
    )
    selected = {
        "LightGlue-SIFT",
        "LightGlue-SuperPoint",
        "LightGlue-DISK",
        "LightGlue-ALIKED",
        "LoFTR",
        "SuperGlue",
    }
    return [matcher for matcher in all_optional if matcher.name in selected]


def make_pfm_matcher(args: argparse.Namespace) -> bench.Matcher:
    state = args.pfm_pytorch_state
    if state is None:
        state = PROJECT_ROOT / "runs" / "cross_view_1024_no_self_warp_finetune_300_seed1234" / "training" / "pytorch_pfm_state.pt"
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


def evaluate_phase(
    *,
    phase: str,
    matchers: Iterable[bench.Matcher],
    args: argparse.Namespace,
) -> list[tuple[str, bench.ResultRow]]:
    phase_dir = args.output_dir / phase
    rows: list[tuple[str, bench.ResultRow]] = []
    images = [("numeric", args.numeric_image), ("timestamp", args.timestamp_image)]
    for image_style, image_path in images:
        image = bench.resize_long_edge(bench.read_image(image_path), args.resize_max)
        for angle in args.angles:
            rotated = bench.rotate_image(image, angle)
            for matcher in matchers:
                row, points_a, points_b = bench.evaluate_matcher_on_rotation(
                    matcher,
                    image,
                    rotated,
                    image_style=image_style,
                    image_path=image_path,
                    angle=angle,
                    output_dir=phase_dir,
                    threshold_px=args.threshold_px,
                )
                rows.append((phase, row))
                if row.visualization:
                    bench.draw_visualization(image, rotated, points_a, points_b, Path(row.visualization))
                print(
                    f"{phase:9s} {image_style:9s} {angle:3d} {matcher.name:32s} "
                    f"{row.status:11s} matches={row.matches:4d} correct={row.correct:4d} "
                    f"precision={row.precision:.4f}",
                    flush=True,
                )
    return rows


def write_comparison_csv(path: Path, rows: list[tuple[str, bench.ResultRow]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for phase, row in rows:
            data = {"phase": phase, **asdict(row)}
            for key in ("precision", "mean_error_px", "median_error_px"):
                value = data[key]
                data[key] = "nan" if isinstance(value, float) and math.isnan(value) else f"{float(value):.6f}"
            writer.writerow(data)


def summarize(rows: list[tuple[str, bench.ResultRow]]) -> str:
    grouped: dict[tuple[str, str], list[bench.ResultRow]] = {}
    for phase, row in rows:
        grouped.setdefault((phase, row.matcher), []).append(row)

    lines = [
        "# Rotation Matcher Comparison Agent",
        "",
        "## Aggregate",
        "",
        "| phase | matcher | ok | non_ok | matches | correct | wrong | precision | mean_error_px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (phase, matcher), subset in sorted(grouped.items()):
        ok = [row for row in subset if row.status == "ok"]
        non_ok = len(subset) - len(ok)
        matches = sum(row.matches for row in ok)
        correct = sum(row.correct for row in ok)
        wrong = sum(row.wrong for row in ok)
        precision = 0.0 if matches == 0 else correct / matches
        errors = [row.mean_error_px for row in ok if not math.isnan(row.mean_error_px)]
        mean_error = float("nan") if not errors else sum(errors) / len(errors)
        mean_error_text = "nan" if math.isnan(mean_error) else f"{mean_error:.3f}"
        lines.append(
            f"| {phase} | {matcher} | {len(ok)} | {non_ok} | {matches} | {correct} | {wrong} | "
            f"{precision:.6f} | {mean_error_text} |"
        )

    unavailable = [
        (phase, row.matcher, row.message)
        for phase, row in rows
        if row.status != "ok" and row.message
    ]
    if unavailable:
        lines.extend(["", "## Unavailable Or Error", ""])
        for phase, matcher, message in unavailable:
            lines.append(f"- {phase} / {matcher}: {message}")
    lines.extend(["", "metrics: metrics.csv", "visualizations: baseline/visualizations and iterated/visualizations"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-image", type=Path, default=PROJECT_ROOT / "img" / "1.tif")
    parser.add_argument(
        "--timestamp-image",
        type=Path,
        default=PROJECT_ROOT / "img" / "20260514T064636672_NAS_PAN_L2b.tif",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "rotation_matcher_comparison_agent")
    parser.add_argument("--angles", type=int, nargs="+", default=[90, 180, 270], choices=[90, 180, 270])
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--resize-max", type=int, default=768)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--pfm-pytorch-state", type=Path, default=None)
    parser.add_argument("--loftr-pretrained", default="outdoor", choices=["outdoor", "indoor", "indoor_new"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iter-sift-contrast", type=float, default=0.01)
    parser.add_argument("--iter-ratio", type=float, default=0.8)
    parser.add_argument("--iter-ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--iter-orb-fast-threshold", type=int, default=5)
    parser.add_argument("--iter-akaze-threshold", type=float, default=0.0003)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate_phase(phase="baseline", matchers=make_baseline_matchers(args), args=args)
    iterated = evaluate_phase(phase="iterated", matchers=make_iterated_matchers(args), args=args)
    rows = baseline + iterated
    write_comparison_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "summary.md").write_text(summarize(rows), encoding="utf-8")
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
