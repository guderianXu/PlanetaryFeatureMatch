"""Generate high-precision classical matcher pseudo labels for cache pairs."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from patch_descriptor_training import load_libtorch_pair_archive


CSV_FIELDS = ["pair_pt", "ax", "ay", "bx", "by", "matcher", "stage", "error_px", "cache_dir"]
SUMMARY_FIELDS = [
    "cache_dir",
    "pair_pt",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "ransac_matches",
    "truth_filtered_matches",
    "mean_error_px",
    "message",
]


@dataclass(frozen=True)
class MatchOutput:
    points_a: np.ndarray
    points_b: np.ndarray
    keypoints_a: int
    keypoints_b: int


@dataclass(frozen=True)
class PseudoLabelRow:
    pair_pt: str
    ax: float
    ay: float
    bx: float
    by: float
    matcher: str
    stage: str
    error_px: float
    cache_dir: str


@dataclass(frozen=True)
class PairSummaryRow:
    cache_dir: str
    pair_pt: str
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    ransac_matches: int
    truth_filtered_matches: int
    mean_error_px: float
    message: str = ""


def empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


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


def filter_matches_by_warp_truth(
    points_a: np.ndarray,
    points_b: np.ndarray,
    warp_a_to_b: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_a.size == 0:
        return empty_points(), empty_points(), np.empty((0,), dtype=np.float32)
    target_b = sample_warp(warp_a_to_b, points_a)
    valid = valid_at_points(valid_mask, points_a)
    errors = np.linalg.norm(target_b - points_b, axis=1).astype(np.float32, copy=False)
    keep = np.isfinite(errors) & valid & (errors <= float(threshold_px))
    return points_a[keep], points_b[keep], errors[keep]


def rootsift(descriptors: np.ndarray) -> np.ndarray:
    denom = np.maximum(descriptors.sum(axis=1, keepdims=True), 1.0e-12)
    return np.sqrt(descriptors / denom).astype(np.float32, copy=False)


def ratio_filter(knn_matches, ratio: float):
    filtered = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        first, second = candidates[:2]
        if first.distance < ratio * second.distance:
            filtered.append(first)
    return filtered


def output_from_matches(keypoints_a, keypoints_b, matches, *, max_matches: int) -> MatchOutput:
    if not matches:
        return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))
    selected = sorted(matches, key=lambda item: item.distance)
    if max_matches > 0:
        selected = selected[:max_matches]
    points_a = np.array([keypoints_a[item.queryIdx].pt for item in selected], dtype=np.float32)
    points_b = np.array([keypoints_b[item.trainIdx].pt for item in selected], dtype=np.float32)
    return MatchOutput(points_a, points_b, len(keypoints_a or []), len(keypoints_b or []))


def rootsift_flann_ratio_match(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    max_keypoints: int,
    max_matches: int,
    ratio: float,
    sift_contrast: float,
) -> MatchOutput:
    import cv2

    detector = cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=sift_contrast)
    keypoints_a, descriptors_a = detector.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(image_b, None)
    if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
        return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))
    descriptors_a = rootsift(descriptors_a.astype(np.float32, copy=False))
    descriptors_b = rootsift(descriptors_b.astype(np.float32, copy=False))
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    matches = ratio_filter(matcher.knnMatch(descriptors_a, descriptors_b, k=2), ratio)
    return output_from_matches(keypoints_a, keypoints_b, matches, max_matches=max_matches)


def homography_inliers(points_a: np.ndarray, points_b: np.ndarray, *, threshold_px: float) -> tuple[np.ndarray, np.ndarray]:
    if points_a.shape[0] < 4:
        return empty_points(), empty_points()
    import cv2

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    _, mask = cv2.findHomography(
        points_a,
        points_b,
        method=method,
        ransacReprojThreshold=threshold_px,
        maxIters=3000,
        confidence=0.995,
    )
    if mask is None:
        return empty_points(), empty_points()
    keep = mask.reshape(-1).astype(bool)
    return points_a[keep], points_b[keep]


def cap_matches(
    points_a: np.ndarray,
    points_b: np.ndarray,
    errors: np.ndarray,
    *,
    max_matches: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_matches <= 0 or points_a.shape[0] <= max_matches:
        return points_a, points_b, errors
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(points_a.shape[0], size=max_matches, replace=False))
    return points_a[indices], points_b[indices], errors[indices]


def rows_from_matches(
    *,
    pair_path: Path,
    points_a: np.ndarray,
    points_b: np.ndarray,
    errors: np.ndarray,
    matcher: str,
    stage: str,
    cache_dir: Path,
) -> list[PseudoLabelRow]:
    return [
        PseudoLabelRow(
            pair_pt=pair_path.as_posix(),
            ax=float(point_a[0]),
            ay=float(point_a[1]),
            bx=float(point_b[0]),
            by=float(point_b[1]),
            matcher=matcher,
            stage=stage,
            error_px=float(error),
            cache_dir=cache_dir.as_posix(),
        )
        for point_a, point_b, error in zip(points_a, points_b, errors)
    ]


def write_pseudo_label_csv(path: Path, rows: list[PseudoLabelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            for key in ("ax", "ay", "bx", "by", "error_px"):
                data[key] = f"{float(data[key]):.3f}"
            writer.writerow(data)


def write_summary_csv(path: Path, rows: list[PairSummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            if math.isnan(data["mean_error_px"]):
                data["mean_error_px"] = "nan"
            else:
                data["mean_error_px"] = f"{float(data['mean_error_px']):.3f}"
            writer.writerow(data)


def discover_pair_archives(cache_dirs: list[Path], *, limit_per_cache: int, seed: int) -> list[tuple[Path, Path]]:
    selected: list[tuple[Path, Path]] = []
    rng = random.Random(seed)
    for cache_dir in cache_dirs:
        paths = sorted(cache_dir.glob("source_*/pair_*.pt"))
        if limit_per_cache > 0 and len(paths) > limit_per_cache:
            paths = rng.sample(paths, k=limit_per_cache)
            paths.sort()
        selected.extend((cache_dir, path) for path in paths)
    return selected


def generate_for_pair(
    cache_dir: Path,
    pair_path: Path,
    *,
    max_keypoints: int,
    max_raw_matches: int,
    ratio: float,
    sift_contrast: float,
    ransac_threshold_px: float,
    truth_threshold_px: float,
    max_labels_per_pair: int,
    seed: int,
) -> tuple[list[PseudoLabelRow], PairSummaryRow]:
    image_a, image_b, warp_a_to_b, valid_mask = load_pair(pair_path)
    output = rootsift_flann_ratio_match(
        image_a,
        image_b,
        max_keypoints=max_keypoints,
        max_matches=max_raw_matches,
        ratio=ratio,
        sift_contrast=sift_contrast,
    )
    ransac_a, ransac_b = homography_inliers(output.points_a, output.points_b, threshold_px=ransac_threshold_px)
    truth_a, truth_b, errors = filter_matches_by_warp_truth(
        ransac_a,
        ransac_b,
        warp_a_to_b,
        valid_mask,
        threshold_px=truth_threshold_px,
    )
    truth_a, truth_b, errors = cap_matches(
        truth_a,
        truth_b,
        errors,
        max_matches=max_labels_per_pair,
        seed=seed,
    )
    rows = rows_from_matches(
        pair_path=pair_path,
        points_a=truth_a,
        points_b=truth_b,
        errors=errors,
        matcher="RootSIFT-FLANN-ratio",
        stage="homography_truth",
        cache_dir=cache_dir,
    )
    summary = PairSummaryRow(
        cache_dir=cache_dir.as_posix(),
        pair_pt=pair_path.as_posix(),
        status="ok",
        keypoints_a=output.keypoints_a,
        keypoints_b=output.keypoints_b,
        raw_matches=int(output.points_a.shape[0]),
        ransac_matches=int(ransac_a.shape[0]),
        truth_filtered_matches=int(truth_a.shape[0]),
        mean_error_px=float(errors.mean()) if errors.size else math.nan,
    )
    return rows, summary
