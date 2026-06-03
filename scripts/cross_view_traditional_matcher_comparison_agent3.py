#!/usr/bin/env python3
"""Compare CPU OpenCV matchers on actual 1024 cross-view synthetic cache pairs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from patch_descriptor_training import load_libtorch_pair_archive

DEFAULT_PFM_RUN = (
    PROJECT_ROOT
    / "runs"
    / "cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234"
)

CSV_FIELDS = [
    "style",
    "gate",
    "pair_pt",
    "matcher",
    "stage",
    "status",
    "keypoints_a",
    "keypoints_b",
    "matches",
    "correct",
    "wrong",
    "precision",
    "mean_error_px",
    "median_error_px",
    "visualization",
    "message",
]


@dataclass(frozen=True)
class MatchOutput:
    points_a: np.ndarray
    points_b: np.ndarray
    keypoints_a: int
    keypoints_b: int


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    pair_pt: str
    matcher: str
    stage: str
    status: str
    keypoints_a: int
    keypoints_b: int
    matches: int
    correct: int
    wrong: int
    precision: float
    mean_error_px: float
    median_error_px: float
    visualization: str
    message: str


class Matcher:
    name: str

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        raise NotImplementedError


class OpenCvBfMatcher(Matcher):
    def __init__(self, name: str, detector, norm: int, *, max_matches: int) -> None:
        self.name = name
        self._detector = detector
        self._norm = norm
        self._max_matches = max_matches

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import cv2

        keypoints_a, descriptors_a = self._detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self._detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))
        matcher = cv2.BFMatcher(self._norm, crossCheck=True)
        matches = sorted(matcher.match(descriptors_a, descriptors_b), key=lambda item: item.distance)[: self._max_matches]
        return output_from_matches(keypoints_a, keypoints_b, matches)


class SiftFlannRatioMatcher(Matcher):
    def __init__(self, name: str, detector, *, max_matches: int, ratio: float, rootsift: bool) -> None:
        self.name = name
        self._detector = detector
        self._max_matches = max_matches
        self._ratio = ratio
        self._rootsift = rootsift

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import cv2

        keypoints_a, descriptors_a = self._detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self._detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))
        descriptors_a = descriptors_a.astype(np.float32, copy=False)
        descriptors_b = descriptors_b.astype(np.float32, copy=False)
        if self._rootsift:
            descriptors_a = rootsift(descriptors_a)
            descriptors_b = rootsift(descriptors_b)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        ratio_matches = ratio_filter(matcher.knnMatch(descriptors_a, descriptors_b, k=2), self._ratio)
        ratio_matches = sorted(ratio_matches, key=lambda item: item.distance)[: self._max_matches]
        return output_from_matches(keypoints_a, keypoints_b, ratio_matches)


def empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def output_from_matches(keypoints_a, keypoints_b, matches) -> MatchOutput:
    if not matches:
        return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))
    points_a = np.array([keypoints_a[item.queryIdx].pt for item in matches], dtype=np.float32)
    points_b = np.array([keypoints_b[item.trainIdx].pt for item in matches], dtype=np.float32)
    return MatchOutput(points_a, points_b, len(keypoints_a or []), len(keypoints_b or []))


def ratio_filter(knn_matches, ratio: float):
    filtered = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        first, second = candidates[:2]
        if first.distance < ratio * second.distance:
            filtered.append(first)
    return filtered


def rootsift(descriptors: np.ndarray) -> np.ndarray:
    denom = np.maximum(descriptors.sum(axis=1, keepdims=True), 1.0e-12)
    return np.sqrt(descriptors / denom).astype(np.float32, copy=False)


def make_matchers(args: argparse.Namespace) -> list[Matcher]:
    import cv2

    matchers: list[Matcher] = []
    if hasattr(cv2, "SIFT_create"):
        matchers.append(
            OpenCvBfMatcher(
                "SIFT-BF",
                cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.sift_contrast),
                cv2.NORM_L2,
                max_matches=args.max_matches,
            )
        )
        matchers.append(
            SiftFlannRatioMatcher(
                "SIFT-FLANN-ratio",
                cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.sift_contrast),
                max_matches=args.max_matches,
                ratio=args.ratio,
                rootsift=False,
            )
        )
        matchers.append(
            SiftFlannRatioMatcher(
                "RootSIFT-FLANN-ratio",
                cv2.SIFT_create(nfeatures=args.max_keypoints, contrastThreshold=args.sift_contrast),
                max_matches=args.max_matches,
                ratio=args.ratio,
                rootsift=True,
            )
        )
    matchers.append(
        OpenCvBfMatcher(
            "ORB-BF",
            cv2.ORB_create(nfeatures=args.max_keypoints, fastThreshold=args.orb_fast_threshold),
            cv2.NORM_HAMMING,
            max_matches=args.max_matches,
        )
    )
    matchers.append(
        OpenCvBfMatcher(
            "AKAZE-BF",
            cv2.AKAZE_create(threshold=args.akaze_threshold),
            cv2.NORM_HAMMING,
            max_matches=args.max_matches,
        )
    )
    return matchers


def image_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().squeeze().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    if array.max(initial=0.0) <= 1.5:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def load_pair(pair_path: Path) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    pair = load_libtorch_pair_archive(pair_path, device="cpu")
    return (
        image_from_tensor(pair.view_a),
        image_from_tensor(pair.view_b),
        pair.warp_a_to_b.detach().cpu().float().contiguous(),
        pair.valid_mask.detach().cpu().bool().contiguous(),
    )


def normalize_xy(points_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = points_xy[:, 0] * (2.0 / float(max(1, width - 1))) - 1.0
    y = points_xy[:, 1] * (2.0 / float(max(1, height - 1))) - 1.0
    return torch.stack([x, y], dim=1)


def sample_warp(warp_a_to_b: torch.Tensor, points_a: np.ndarray) -> np.ndarray:
    if points_a.size == 0:
        return empty_points()
    points = torch.from_numpy(points_a.astype(np.float32, copy=False))
    height, width = warp_a_to_b.shape[:2]
    grid = normalize_xy(points, height, width).view(1, -1, 1, 2)
    warp = warp_a_to_b.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(warp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).T.contiguous().numpy().astype(np.float32, copy=False)


def valid_at_points(valid_mask: torch.Tensor, points_a: np.ndarray) -> np.ndarray:
    if points_a.size == 0:
        return np.zeros((0,), dtype=bool)
    height, width = valid_mask.shape
    rounded = np.rint(points_a).astype(np.int64)
    x = rounded[:, 0]
    y = rounded[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid = np.zeros((points_a.shape[0],), dtype=bool)
    if inside.any():
        mask_np = valid_mask.numpy()
        valid[inside] = mask_np[y[inside], x[inside]]
    return valid


def compute_metrics(
    points_a: np.ndarray,
    points_b: np.ndarray,
    warp_a_to_b: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold_px: float,
) -> tuple[int, int, int, float, float, float]:
    total = int(points_a.shape[0])
    if total == 0:
        return 0, 0, 0, 0.0, math.nan, math.nan
    target_b = sample_warp(warp_a_to_b, points_a)
    valid = valid_at_points(valid_mask, points_a)
    errors = np.linalg.norm(target_b - points_b, axis=1)
    errors = np.where(np.isfinite(errors) & valid, errors, np.inf)
    correct = int(np.count_nonzero(errors <= threshold_px))
    wrong = total - correct
    finite = errors[np.isfinite(errors)]
    mean_error = float(finite.mean()) if finite.size else math.nan
    median_error = float(np.median(finite)) if finite.size else math.nan
    return total, correct, wrong, correct / total if total else 0.0, mean_error, median_error


def ransac_inliers(points_a: np.ndarray, points_b: np.ndarray, *, stage: str, threshold_px: float) -> tuple[np.ndarray, np.ndarray]:
    if points_a.shape[0] < 4:
        return empty_points(), empty_points()
    import cv2

    mask = None
    if stage == "homography":
        method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
        _, mask = cv2.findHomography(
            points_a,
            points_b,
            method=method,
            ransacReprojThreshold=threshold_px,
            maxIters=3000,
            confidence=0.995,
        )
    elif stage == "affine":
        _, mask = cv2.estimateAffinePartial2D(
            points_a,
            points_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold_px,
            maxIters=3000,
            confidence=0.995,
        )
    else:
        raise ValueError(f"unknown RANSAC stage: {stage}")
    if mask is None:
        return empty_points(), empty_points()
    keep = mask.reshape(-1).astype(bool)
    return points_a[keep], points_b[keep]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def draw_visualization(image_a: np.ndarray, image_b: np.ndarray, points_a: np.ndarray, points_b: np.ndarray, path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.concatenate([cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)], axis=1)
    offset = image_a.shape[1]
    limit = min(points_a.shape[0], 80)
    for index in range(limit):
        ax, ay = points_a[index]
        bx, by = points_b[index]
        color = tuple(int(v) for v in np.random.default_rng(index).integers(64, 255, size=3))
        cv2.circle(canvas, (int(round(ax)), int(round(ay))), 2, color, -1)
        cv2.circle(canvas, (int(round(bx + offset)), int(round(by))), 2, color, -1)
        cv2.line(canvas, (int(round(ax)), int(round(ay))), (int(round(bx + offset)), int(round(by))), color, 1)
    cv2.imwrite(str(path), canvas)


def read_pfm_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["pair_pt"]: row for row in csv.DictReader(handle)}


def group_summary_csv(args: argparse.Namespace, style: str, gate: str) -> Path:
    return args.pfm_run / "eval" / style / gate / "summary.csv"


def group_split_dir(args: argparse.Namespace, style: str, gate: str) -> Path:
    return args.pfm_run / "splits" / args.split / style / gate


def select_pairs(args: argparse.Namespace, style: str, gate: str) -> list[Path]:
    summary_rows = list(read_pfm_summary(group_summary_csv(args, style, gate)).values())
    if summary_rows:
        paths = [PROJECT_ROOT / row["pair_pt"] for row in summary_rows if row.get("pair_pt")]
    else:
        paths = sorted(group_split_dir(args, style, gate).glob("source_*/pair_*.pt"))
    paths = [path for path in paths if path.exists()]
    if not paths:
        cache_root = PROJECT_ROOT / "img" / ("Viewpoint_1024" if gate == "viewpoint" else "CompoundViewpoint_1024")
        paths = [
            path
            for path in sorted(cache_root.glob("source_*/pair_*.pt"))
            if (path.parent.name.split("_", 2)[-1].isdigit()) == (style == "numeric")
        ]
    if args.pairs_per_group <= 0 or len(paths) <= args.pairs_per_group:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=args.pairs_per_group, dtype=int)
    return [paths[int(index)] for index in indices]


def row_for_metric(
    *,
    style: str,
    gate: str,
    pair_path: Path,
    matcher: str,
    stage: str,
    status: str,
    keypoints_a: int,
    keypoints_b: int,
    metric: tuple[int, int, int, float, float, float],
    visualization: Path | None,
    message: str = "",
) -> MetricRow:
    matches, correct, wrong, precision, mean_error, median_error = metric
    return MetricRow(
        style=style,
        gate=gate,
        pair_pt=pair_path.as_posix(),
        matcher=matcher,
        stage=stage,
        status=status,
        keypoints_a=keypoints_a,
        keypoints_b=keypoints_b,
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        mean_error_px=mean_error,
        median_error_px=median_error,
        visualization=visualization.as_posix() if visualization else "",
        message=message,
    )


def evaluate_pair(
    args: argparse.Namespace,
    matcher: Matcher,
    *,
    style: str,
    gate: str,
    pair_path: Path,
    vis_budget: dict[str, int],
) -> list[MetricRow]:
    try:
        image_a, image_b, warp_a_to_b, valid_mask = load_pair(pair_path)
        output = matcher.match(image_a, image_b)
        rows: list[MetricRow] = []
        stages = [("raw", output.points_a, output.points_b)]
        for ransac_stage in ("homography", "affine"):
            inlier_a, inlier_b = ransac_inliers(
                output.points_a,
                output.points_b,
                stage=ransac_stage,
                threshold_px=args.ransac_threshold_px,
            )
            stages.append((ransac_stage, inlier_a, inlier_b))
        for stage, points_a, points_b in stages:
            metric = compute_metrics(points_a, points_b, warp_a_to_b, valid_mask, threshold_px=args.threshold_px)
            vis_path = None
            vis_key = f"{style}/{gate}/{matcher.name}/{stage}"
            if args.visualizations_per_group > 0 and metric[0] > 0 and vis_budget.get(vis_key, 0) < args.visualizations_per_group:
                vis_budget[vis_key] = vis_budget.get(vis_key, 0) + 1
                vis_path = (
                    args.output_dir
                    / "visualizations"
                    / style
                    / gate
                    / stage
                    / f"{pair_path.stem}_{safe_name(matcher.name)}.png"
                )
                draw_visualization(image_a, image_b, points_a, points_b, vis_path)
            rows.append(
                row_for_metric(
                    style=style,
                    gate=gate,
                    pair_path=pair_path,
                    matcher=matcher.name,
                    stage=stage,
                    status="ok",
                    keypoints_a=output.keypoints_a,
                    keypoints_b=output.keypoints_b,
                    metric=metric,
                    visualization=vis_path,
                )
            )
        return rows
    except Exception as exc:
        metric = (0, 0, 0, 0.0, math.nan, math.nan)
        return [
            row_for_metric(
                style=style,
                gate=gate,
                pair_path=pair_path,
                matcher=matcher.name,
                stage=stage,
                status="error",
                keypoints_a=0,
                keypoints_b=0,
                metric=metric,
                visualization=None,
                message=str(exc),
            )
            for stage in ("raw", "homography", "affine")
        ]


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, str]]]:
    matchers = make_matchers(args)
    rows: list[MetricRow] = []
    sampled: list[dict[str, str]] = []
    vis_budget: dict[str, int] = {}
    for style in args.styles:
        for gate in args.gates:
            pair_paths = select_pairs(args, style, gate)
            sampled.extend({"style": style, "gate": gate, "pair_pt": path.as_posix()} for path in pair_paths)
            print(f"group={style}/{gate} pairs={len(pair_paths)}", flush=True)
            for pair_index, pair_path in enumerate(pair_paths, start=1):
                for matcher in matchers:
                    group_rows = evaluate_pair(args, matcher, style=style, gate=gate, pair_path=pair_path, vis_budget=vis_budget)
                    rows.extend(group_rows)
                    raw = next(row for row in group_rows if row.stage == "raw")
                    print(
                        f"{style:9s} {gate:9s} {pair_index:02d}/{len(pair_paths):02d} "
                        f"{matcher.name:22s} matches={raw.matches:4d} correct={raw.correct:4d} "
                        f"precision={raw.precision:.4f}",
                        flush=True,
                    )
    return rows, sampled


def write_csv(path: Path, rows: Iterable[MetricRow]) -> None:
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


def write_sample_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["style", "gate", "pair_pt"])
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metric_rows(rows: list[MetricRow], *, stage: str) -> dict[tuple[str, str, str], dict[str, float | int]]:
    grouped: dict[tuple[str, str, str], list[MetricRow]] = {}
    for row in rows:
        if row.stage != stage:
            continue
        grouped.setdefault((row.style, row.gate, row.matcher), []).append(row)
    output: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for key, subset in sorted(grouped.items()):
        ok = [row for row in subset if row.status == "ok"]
        matches = sum(row.matches for row in ok)
        correct = sum(row.correct for row in ok)
        wrong = sum(row.wrong for row in ok)
        errors = [row.mean_error_px for row in ok if not math.isnan(row.mean_error_px)]
        output[key] = {
            "pairs": len(ok),
            "matches": matches,
            "correct": correct,
            "wrong": wrong,
            "precision": 0.0 if matches == 0 else correct / matches,
            "mean_error_px": math.nan if not errors else sum(errors) / len(errors),
        }
    return output


def aggregate_pfm(args: argparse.Namespace, sampled: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float | int]]:
    output: dict[tuple[str, str, str], dict[str, float | int]] = {}
    by_group: dict[tuple[str, str], list[str]] = {}
    for row in sampled:
        by_group.setdefault((row["style"], row["gate"]), []).append(row["pair_pt"])
    for (style, gate), pair_paths in sorted(by_group.items()):
        summary = read_pfm_summary(group_summary_csv(args, style, gate))
        matches = correct = wrong = pairs = 0
        for pair_path in pair_paths:
            rel = Path(pair_path).relative_to(PROJECT_ROOT).as_posix() if Path(pair_path).is_absolute() else pair_path
            pfm_row = summary.get(rel) or summary.get(pair_path)
            if pfm_row is None:
                continue
            pairs += 1
            matches += int(pfm_row.get("matches", 0))
            correct += int(pfm_row.get("correct", 0))
            wrong += int(pfm_row.get("wrong", 0))
        output[(style, gate, "PFM-guarded-summary")] = {
            "pairs": pairs,
            "matches": matches,
            "correct": correct,
            "wrong": wrong,
            "precision": 0.0 if matches == 0 else correct / matches,
            "mean_error_px": math.nan,
        }
    return output


def table_lines(title: str, aggregate: dict[tuple[str, str, str], dict[str, float | int]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| style | gate | matcher | pairs | matches | correct | wrong | precision | mean_error_px |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (style, gate, matcher), values in sorted(aggregate.items()):
        mean_error = values["mean_error_px"]
        mean_text = "nan" if isinstance(mean_error, float) and math.isnan(mean_error) else f"{float(mean_error):.3f}"
        lines.append(
            f"| {style} | {gate} | {matcher} | {values['pairs']} | {values['matches']} | "
            f"{values['correct']} | {values['wrong']} | {float(values['precision']):.6f} | {mean_text} |"
        )
    return lines


def write_summary(path: Path, rows: list[MetricRow], sampled: list[dict[str, str]], args: argparse.Namespace) -> None:
    raw = aggregate_metric_rows(rows, stage="raw")
    homography = aggregate_metric_rows(rows, stage="homography")
    affine = aggregate_metric_rows(rows, stage="affine")
    pfm = aggregate_pfm(args, sampled)
    lines = [
        "# Cross-view Traditional Matcher Comparison Agent3",
        "",
        "## Setup",
        "",
        f"- sampled pairs per group: `{args.pairs_per_group}`",
        f"- groups: styles `{','.join(args.styles)}`, gates `{','.join(args.gates)}`",
        f"- pair source: `{args.pfm_run}/splits/{args.split}` preferred, falling back to cache roots",
        f"- PFM comparison: `{args.pfm_run}/eval/*/*/summary.csv` only; no PFM rerun",
        f"- correctness threshold: `{args.threshold_px}` px",
        f"- RANSAC threshold: `{args.ransac_threshold_px}` px",
        f"- max keypoints/matches: `{args.max_keypoints}` / `{args.max_matches}`",
        "",
        *table_lines("Raw OpenCV Matches", raw),
        "",
        *table_lines("Homography RANSAC Inliers", homography),
        "",
        *table_lines("Affine RANSAC Inliers", affine),
        "",
        *table_lines("PFM Guarded Summary On Same Sample", pfm),
    ]
    errors = sorted({(row.matcher, row.message) for row in rows if row.status != "ok" and row.message})
    if errors:
        lines.extend(["", "## Errors", ""])
        for matcher, message in errors:
            lines.append(f"- {matcher}: {message}")
    lines.extend(["", "- metrics: `metrics.csv`", "- sampled pairs: `sampled_pairs.csv`", "- visualizations: `visualizations/`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    descriptors = np.array([[1.0, 3.0], [0.0, 4.0]], dtype=np.float32)
    normalized = rootsift(descriptors)
    assert normalized.shape == descriptors.shape
    assert np.allclose((normalized**2).sum(axis=1), 1.0)
    warp = torch.zeros(4, 4, 2)
    for y in range(4):
        for x in range(4):
            warp[y, x] = torch.tensor([x + 1.0, y + 2.0])
    valid = torch.ones(4, 4, dtype=torch.bool)
    points_a = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    points_b = np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float32)
    matches, correct, wrong, precision, _, _ = compute_metrics(points_a, points_b, warp, valid, threshold_px=0.1)
    assert (matches, correct, wrong) == (2, 1, 1)
    assert precision == 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "cross_view_traditional_matcher_comparison_agent3")
    parser.add_argument("--pfm-run", type=Path, default=DEFAULT_PFM_RUN)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["viewpoint", "compound"], choices=["viewpoint", "compound"])
    parser.add_argument("--pairs-per-group", type=int, default=16)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--orb-fast-threshold", type=int, default=5)
    parser.add_argument("--akaze-threshold", type=float, default=0.0003)
    parser.add_argument("--visualizations-per-group", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, sampled = evaluate(args)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_sample_csv(args.output_dir / "sampled_pairs.csv", sampled)
    write_summary(args.output_dir / "summary.md", rows, sampled, args)
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
