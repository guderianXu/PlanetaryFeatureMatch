#!/usr/bin/env python3
"""Agent4 matcher iteration on 1024 cross-view cache pairs with B-image rotations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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

METRIC_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "pair_pt",
    "algorithm",
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
    "ransac_threshold_px",
    "ratio",
    "mutual",
    "visualization",
    "message",
]

SUMMARY_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "algorithm",
    "pairs",
    "ok_pairs",
    "matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "median_pair_precision",
    "mean_matches_per_pair",
    "median_matches_per_pair",
    "min_matches",
    "pairs_ge_20_inliers",
    "pairs_ge_50_inliers",
    "mean_error_px",
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
    rotation_deg: int
    pair_pt: str
    algorithm: str
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
    ransac_threshold_px: float
    ratio: float
    mutual: bool
    visualization: str
    message: str


class Matcher:
    name: str
    ratio: float = math.nan
    mutual: bool = False

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        raise NotImplementedError


class RootSiftFlannMatcher(Matcher):
    def __init__(
        self,
        *,
        name: str,
        max_keypoints: int,
        max_matches: int,
        ratio: float,
        mutual: bool,
        sift_contrast: float,
    ) -> None:
        import cv2

        self.name = name
        self.ratio = ratio
        self.mutual = mutual
        self._max_matches = max_matches
        self._detector = cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=sift_contrast)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import cv2

        keypoints_a, descriptors_a = self._detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self._detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))
        descriptors_a = rootsift(descriptors_a.astype(np.float32, copy=False))
        descriptors_b = rootsift(descriptors_b.astype(np.float32, copy=False))
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        forward = ratio_filter(matcher.knnMatch(descriptors_a, descriptors_b, k=2), self.ratio)
        if self.mutual:
            backward = ratio_filter(matcher.knnMatch(descriptors_b, descriptors_a, k=2), self.ratio)
            backward_pairs = {(item.trainIdx, item.queryIdx) for item in backward}
            forward = [item for item in forward if (item.queryIdx, item.trainIdx) in backward_pairs]
        matches = sorted(forward, key=lambda item: item.distance)[: self._max_matches]
        return output_from_matches(keypoints_a, keypoints_b, matches)


class HomographyRansacWrapper(Matcher):
    def __init__(self, base: Matcher, *, threshold_px: float, min_inliers: int) -> None:
        self._base = base
        self.name = f"{base.name}+HomographyRANSAC"
        self.ratio = base.ratio
        self.mutual = base.mutual
        self._threshold_px = threshold_px
        self._min_inliers = min_inliers

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        output = self._base.match(image_a, image_b)
        inlier_a, inlier_b = ransac_inliers(output.points_a, output.points_b, threshold_px=self._threshold_px)
        if inlier_a.shape[0] < self._min_inliers:
            inlier_a, inlier_b = empty_points(), empty_points()
        return MatchOutput(inlier_a, inlier_b, output.keypoints_a, output.keypoints_b)


class LightGlueSiftMatcher(Matcher):
    def __init__(self, *, max_keypoints: int, device: str) -> None:
        self.name = "LightGlue-SIFT"
        self.ratio = math.nan
        self.mutual = True
        from lightglue import LightGlue, SIFT

        self._device = torch.device(device)
        self._extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(self._device)
        self._matcher = LightGlue(features="sift").eval().to(self._device)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        from lightglue.utils import numpy_image_to_torch, rbd

        with torch.inference_mode():
            tensor_a = numpy_image_to_torch(image_a).to(self._device)
            tensor_b = numpy_image_to_torch(image_b).to(self._device)
            feats_a = self._extractor.extract(tensor_a)
            feats_b = self._extractor.extract(tensor_b)
            pred = self._matcher({"image0": feats_a, "image1": feats_b})
            feats_a, feats_b, pred = [rbd(item) for item in (feats_a, feats_b, pred)]
            matches = pred["matches"].detach().cpu().numpy()
            keypoints_a = feats_a["keypoints"].detach().cpu().numpy()
            keypoints_b = feats_b["keypoints"].detach().cpu().numpy()
        if matches.size == 0:
            return MatchOutput(empty_points(), empty_points(), int(keypoints_a.shape[0]), int(keypoints_b.shape[0]))
        points_a = keypoints_a[matches[:, 0]].astype(np.float32, copy=False)
        points_b = keypoints_b[matches[:, 1]].astype(np.float32, copy=False)
        return MatchOutput(points_a, points_b, int(keypoints_a.shape[0]), int(keypoints_b.shape[0]))


def empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


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


def output_from_matches(keypoints_a, keypoints_b, matches) -> MatchOutput:
    if not matches:
        return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))
    points_a = np.array([keypoints_a[item.queryIdx].pt for item in matches], dtype=np.float32)
    points_b = np.array([keypoints_b[item.trainIdx].pt for item in matches], dtype=np.float32)
    return MatchOutput(points_a, points_b, len(keypoints_a or []), len(keypoints_b or []))


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


def rotate_image(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    import cv2

    if rotation_deg == 0:
        return image
    if rotation_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation: {rotation_deg}")


def unrotate_points(points_xy: np.ndarray, original_height: int, original_width: int, rotation_deg: int) -> np.ndarray:
    if points_xy.size == 0 or rotation_deg == 0:
        return points_xy.astype(np.float32, copy=False)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    if rotation_deg == 90:
        original_x = y
        original_y = original_height - 1 - x
    elif rotation_deg == 180:
        original_x = original_width - 1 - x
        original_y = original_height - 1 - y
    elif rotation_deg == 270:
        original_x = original_width - 1 - y
        original_y = x
    else:
        raise ValueError(f"unsupported rotation: {rotation_deg}")
    return np.stack([original_x, original_y], axis=1).astype(np.float32, copy=False)


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
    points_b_original: np.ndarray,
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
    errors = np.linalg.norm(target_b - points_b_original, axis=1)
    errors = np.where(np.isfinite(errors) & valid, errors, np.inf)
    correct = int(np.count_nonzero(errors <= threshold_px))
    wrong = total - correct
    finite = errors[np.isfinite(errors)]
    mean_error = float(finite.mean()) if finite.size else math.nan
    median_error = float(np.median(finite)) if finite.size else math.nan
    return total, correct, wrong, correct / total if total else 0.0, mean_error, median_error


def ransac_inliers(points_a: np.ndarray, points_b: np.ndarray, *, threshold_px: float) -> tuple[np.ndarray, np.ndarray]:
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def draw_visualization(
    image_a: np.ndarray,
    image_b_original: np.ndarray,
    points_a: np.ndarray,
    points_b_original: np.ndarray,
    path: Path,
) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.concatenate([cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(image_b_original, cv2.COLOR_GRAY2BGR)], axis=1)
    offset = image_a.shape[1]
    limit = min(points_a.shape[0], 80)
    for index in range(limit):
        ax, ay = points_a[index]
        bx, by = points_b_original[index]
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


def cache_root_for_gate(gate: str) -> Path:
    if gate == "rotate":
        return PROJECT_ROOT / "img" / "Rotate_1024"
    if gate == "viewpoint":
        return PROJECT_ROOT / "img" / "Viewpoint_1024"
    if gate == "compound":
        return PROJECT_ROOT / "img" / "CompoundViewpoint_1024"
    raise ValueError(f"unknown gate: {gate}")


def is_style_path(path: Path, style: str) -> bool:
    suffix = path.parent.name.split("_", 2)[-1]
    return suffix.isdigit() == (style == "numeric")


def select_pairs(args: argparse.Namespace, style: str, gate: str) -> list[Path]:
    summary_rows = list(read_pfm_summary(group_summary_csv(args, style, gate)).values())
    if summary_rows:
        paths = [PROJECT_ROOT / row["pair_pt"] for row in summary_rows if row.get("pair_pt")]
    else:
        paths = sorted(group_split_dir(args, style, gate).glob("source_*/pair_*.pt"))
    paths = [path for path in paths if path.exists()]
    if not paths:
        paths = [path for path in sorted(cache_root_for_gate(gate).glob("source_*/pair_*.pt")) if is_style_path(path, style)]
    if args.pairs_per_group <= 0 or len(paths) <= args.pairs_per_group:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=args.pairs_per_group, dtype=int)
    return [paths[int(index)] for index in indices]


def make_matchers(args: argparse.Namespace) -> tuple[list[Matcher], list[dict[str, str]]]:
    unavailable: list[dict[str, str]] = []
    matchers: list[Matcher] = []
    try:
        import cv2

        if not hasattr(cv2, "SIFT_create"):
            unavailable.append({"algorithm": "RootSIFT/OpenCV-SIFT", "reason": "cv2.SIFT_create unavailable"})
        else:
            ratio = RootSiftFlannMatcher(
                name="RootSIFT-FLANN-ratio",
                max_keypoints=args.max_keypoints,
                max_matches=args.max_matches,
                ratio=args.ratio,
                mutual=False,
                sift_contrast=args.sift_contrast,
            )
            mutual = RootSiftFlannMatcher(
                name="RootSIFT-FLANN-ratio+mutual",
                max_keypoints=args.max_keypoints,
                max_matches=args.max_matches,
                ratio=args.ratio,
                mutual=True,
                sift_contrast=args.sift_contrast,
            )
            matchers.extend(
                [
                    ratio,
                    mutual,
                    HomographyRansacWrapper(ratio, threshold_px=args.ransac_threshold_px, min_inliers=args.min_inliers),
                    HomographyRansacWrapper(mutual, threshold_px=args.ransac_threshold_px, min_inliers=args.min_inliers),
                ]
            )
    except Exception as exc:
        unavailable.append({"algorithm": "OpenCV RootSIFT family", "reason": f"{type(exc).__name__}: {exc}"})

    for module_name, label in [
        ("kornia", "kornia.feature APIs"),
        ("lightglue", "LightGlue package"),
        ("match_pairs", "SuperGlue match_pairs"),
    ]:
        if importlib.util.find_spec(module_name) is None:
            unavailable.append({"algorithm": label, "reason": f"module {module_name!r} unavailable"})

    if args.try_lightglue:
        try:
            device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
            matchers.append(LightGlueSiftMatcher(max_keypoints=args.learned_max_keypoints, device=device))
        except Exception as exc:
            unavailable.append({"algorithm": "LightGlue-SIFT", "reason": f"{type(exc).__name__}: {exc}"})
    else:
        unavailable.append({"algorithm": "LightGlue-SIFT", "reason": "not requested; pass --try-lightglue"})
    unavailable.extend(
        [
            {"algorithm": "LoFTR", "reason": "not run in this sidecar; keep optional to avoid checkpoint/download blocking"},
            {"algorithm": "ALIKED/DISK LightGlue", "reason": "not run by default; SIFT-backed LightGlue is the learned smoke path"},
            {"algorithm": "SuperGlue", "reason": "match_pairs module/API not present in this repo session"},
        ]
    )
    return matchers, unavailable


def row_for_metric(
    *,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    matcher: Matcher,
    status: str,
    raw_matches: int,
    keypoints_a: int,
    keypoints_b: int,
    metric: tuple[int, int, int, float, float, float],
    visualization: Path | None,
    message: str = "",
    ransac_threshold_px: float = math.nan,
) -> MetricRow:
    matches, correct, wrong, precision, mean_error, median_error = metric
    return MetricRow(
        style=style,
        gate=gate,
        rotation_deg=rotation_deg,
        pair_pt=pair_path.as_posix(),
        algorithm=matcher.name,
        status=status,
        keypoints_a=keypoints_a,
        keypoints_b=keypoints_b,
        raw_matches=raw_matches,
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        mean_error_px=mean_error,
        median_error_px=median_error,
        ransac_threshold_px=ransac_threshold_px,
        ratio=matcher.ratio,
        mutual=matcher.mutual,
        visualization=visualization.as_posix() if visualization else "",
        message=message,
    )


def evaluate_pair(
    args: argparse.Namespace,
    matcher: Matcher,
    *,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    vis_budget: dict[str, int],
) -> MetricRow:
    try:
        image_a, image_b, warp_a_to_b, valid_mask = load_pair(pair_path)
        image_b_rotated = rotate_image(image_b, rotation_deg)
        output = matcher.match(image_a, image_b_rotated)
        points_b_original = unrotate_points(output.points_b, image_b.shape[0], image_b.shape[1], rotation_deg)
        metric = compute_metrics(output.points_a, points_b_original, warp_a_to_b, valid_mask, threshold_px=args.threshold_px)
        vis_path = None
        vis_key = f"{style}/{gate}/{rotation_deg}/{matcher.name}"
        if args.visualizations_per_group > 0 and metric[0] > 0 and vis_budget.get(vis_key, 0) < args.visualizations_per_group:
            vis_budget[vis_key] = vis_budget.get(vis_key, 0) + 1
            vis_path = (
                args.output_dir
                / "visualizations"
                / style
                / gate
                / f"rot{rotation_deg}"
                / f"{pair_path.stem}_{safe_name(matcher.name)}.png"
            )
            draw_visualization(image_a, image_b, output.points_a, points_b_original, vis_path)
        return row_for_metric(
            style=style,
            gate=gate,
            rotation_deg=rotation_deg,
            pair_path=pair_path,
            matcher=matcher,
            status="ok",
            raw_matches=output.points_a.shape[0],
            keypoints_a=output.keypoints_a,
            keypoints_b=output.keypoints_b,
            metric=metric,
            visualization=vis_path,
            ransac_threshold_px=args.ransac_threshold_px if "HomographyRANSAC" in matcher.name else math.nan,
        )
    except Exception as exc:
        return row_for_metric(
            style=style,
            gate=gate,
            rotation_deg=rotation_deg,
            pair_path=pair_path,
            matcher=matcher,
            status="error",
            raw_matches=0,
            keypoints_a=0,
            keypoints_b=0,
            metric=(0, 0, 0, 0.0, math.nan, math.nan),
            visualization=None,
            message=f"{type(exc).__name__}: {exc}",
        )


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, str]], list[dict[str, str]]]:
    matchers, unavailable = make_matchers(args)
    rows: list[MetricRow] = []
    sampled: list[dict[str, str]] = []
    vis_budget: dict[str, int] = {}
    for style in args.styles:
        for gate in args.gates:
            pair_paths = select_pairs(args, style, gate)
            sampled.extend({"style": style, "gate": gate, "pair_pt": path.as_posix()} for path in pair_paths)
            print(f"group={style}/{gate} pairs={len(pair_paths)}", flush=True)
            for rotation_deg in args.rotations:
                for pair_index, pair_path in enumerate(pair_paths, start=1):
                    for matcher in matchers:
                        row = evaluate_pair(
                            args,
                            matcher,
                            style=style,
                            gate=gate,
                            rotation_deg=rotation_deg,
                            pair_path=pair_path,
                            vis_budget=vis_budget,
                        )
                        rows.append(row)
                        print(
                            f"{style:9s} {gate:9s} rot={rotation_deg:3d} {pair_index:02d}/{len(pair_paths):02d} "
                            f"{matcher.name:36s} matches={row.matches:4d} correct={row.correct:4d} precision={row.precision:.4f}",
                            flush=True,
                        )
    return rows, sampled, unavailable


def write_metric_csv(path: Path, rows: Iterable[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            for key, value in list(data.items()):
                if isinstance(value, float):
                    data[key] = "nan" if math.isnan(value) else f"{value:.6f}"
            writer.writerow(data)


def write_dict_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.rotation_deg, row.algorithm), []).append(row)
    summary: list[dict[str, object]] = []
    for (style, gate, rotation_deg, algorithm), subset in sorted(grouped.items()):
        ok = [row for row in subset if row.status == "ok"]
        matches = sum(row.matches for row in ok)
        correct = sum(row.correct for row in ok)
        wrong = sum(row.wrong for row in ok)
        precisions = np.array([row.precision for row in ok], dtype=np.float64)
        match_counts = np.array([row.matches for row in ok], dtype=np.float64)
        errors = [row.mean_error_px for row in ok if not math.isnan(row.mean_error_px)]
        summary.append(
            {
                "style": style,
                "gate": gate,
                "rotation_deg": rotation_deg,
                "algorithm": algorithm,
                "pairs": len(subset),
                "ok_pairs": len(ok),
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": f"{(0.0 if matches == 0 else correct / matches):.6f}",
                "mean_pair_precision": f"{(float(precisions.mean()) if precisions.size else 0.0):.6f}",
                "median_pair_precision": f"{(float(np.median(precisions)) if precisions.size else 0.0):.6f}",
                "mean_matches_per_pair": f"{(float(match_counts.mean()) if match_counts.size else 0.0):.3f}",
                "median_matches_per_pair": f"{(float(np.median(match_counts)) if match_counts.size else 0.0):.3f}",
                "min_matches": int(match_counts.min()) if match_counts.size else 0,
                "pairs_ge_20_inliers": sum(row.matches >= 20 for row in ok),
                "pairs_ge_50_inliers": sum(row.matches >= 50 for row in ok),
                "mean_error_px": "nan" if not errors else f"{(sum(errors) / len(errors)):.3f}",
            }
        )
    return summary


def markdown_table(rows: list[dict[str, object]], *, gate_filter: set[str], algorithm_contains: str) -> list[str]:
    selected = [row for row in rows if row["gate"] in gate_filter and algorithm_contains in str(row["algorithm"])]
    lines = [
        "| style | gate | rot | algorithm | ok_pairs | matches | precision | mean_matches_pair | min_matches | pairs_ge_20 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['rotation_deg']} | {row['algorithm']} | "
            f"{row['ok_pairs']} | {row['matches']} | {row['precision']} | {row['mean_matches_per_pair']} | "
            f"{row['min_matches']} | {row['pairs_ge_20_inliers']} |"
        )
    return lines


def write_readme(
    path: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, object]],
    sampled: list[dict[str, str]],
    unavailable: list[dict[str, str]],
) -> None:
    command = (
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} "
        f"--pairs-per-group {args.pairs_per_group} --visualizations-per-group {args.visualizations_per_group}"
    )
    lines = [
        "# Matcher Algorithm Iteration Agent4",
        "",
        "## Setup",
        "",
        f"- output dir: `{args.output_dir}`",
        f"- split: `{args.split}`",
        f"- styles: `{','.join(args.styles)}`",
        f"- gates: `{','.join(args.gates)}`",
        f"- rotations of image B: `{','.join(str(item) for item in args.rotations)}`",
        f"- pairs per style/gate: `{args.pairs_per_group}`",
        f"- correctness threshold: `{args.threshold_px}` px",
        f"- RootSIFT ratio: `{args.ratio}`",
        f"- Homography RANSAC threshold/min inliers: `{args.ransac_threshold_px}` px / `{args.min_inliers}`",
        f"- sampled pairs: `{len(sampled)}` unique style/gate rows before rotation expansion",
        "",
        "Command:",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Focus: Homography RANSAC Pseudo-label Candidate",
        "",
        *markdown_table(summary_rows, gate_filter={"viewpoint", "compound"}, algorithm_contains="HomographyRANSAC"),
        "",
        "## Raw/Mutual RootSIFT",
        "",
        *markdown_table(summary_rows, gate_filter={"viewpoint", "compound"}, algorithm_contains="RootSIFT-FLANN-ratio"),
        "",
        "## Recommendation",
        "",
        "- Use `RootSIFT-FLANN-ratio+HomographyRANSAC` as the first pseudo-label sidecar source when precision is the priority.",
        "- Keep `ratio=0.80`, `ransacReprojThreshold=4 px`, and require at least `20` homography inliers per pair for training labels.",
        "- Prefer `50+` inliers when mining high-confidence viewpoint/compound examples; this run records `pairs_ge_50_inliers` for that stricter filter.",
        "- For 90/180/270 robustness, match against the rotated B image, then inverse-map B keypoints back to original B coordinates before evaluating or writing labels.",
        "",
        "## Unavailable / Not Blocking",
        "",
    ]
    for item in unavailable:
        lines.append(f"- {item['algorithm']}: {item['reason']}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `per_pair_metrics.csv`",
            "- `summary_metrics.csv`",
            "- `sampled_pairs.csv`",
            "- `unavailable_algorithms.json`",
            "- `visualizations/`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    point = np.array([[1.0, 2.0]], dtype=np.float32)
    assert np.allclose(unrotate_points(point, 3, 4, 0), point)
    assert np.allclose(unrotate_points(np.array([[0.0, 1.0]], dtype=np.float32), 3, 4, 90), [[1.0, 2.0]])
    assert np.allclose(unrotate_points(np.array([[2.0, 1.0]], dtype=np.float32), 3, 4, 180), [[1.0, 1.0]])
    assert np.allclose(unrotate_points(np.array([[1.0, 1.0]], dtype=np.float32), 3, 4, 270), [[2.0, 1.0]])
    assert rotate_image(image, 90).shape == (4, 3)
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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent4")
    parser.add_argument("--pfm-run", type=Path, default=DEFAULT_PFM_RUN)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["rotate", "viewpoint", "compound"], choices=["rotate", "viewpoint", "compound"])
    parser.add_argument("--rotations", nargs="+", type=int, default=[0, 90, 180, 270], choices=[0, 90, 180, 270])
    parser.add_argument("--pairs-per-group", type=int, default=8)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--visualizations-per-group", type=int, default=1)
    parser.add_argument("--try-lightglue", action="store_true")
    parser.add_argument("--learned-max-keypoints", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, sampled, unavailable = evaluate(args)
    summary_rows = aggregate_rows(rows)
    write_metric_csv(args.output_dir / "per_pair_metrics.csv", rows)
    write_dict_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_dict_csv(args.output_dir / "sampled_pairs.csv", sampled, ["style", "gate", "pair_pt"])
    (args.output_dir / "unavailable_algorithms.json").write_text(json.dumps(unavailable, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(args.output_dir / "README.md", args, summary_rows, sampled, unavailable)
    print(f"output_dir={args.output_dir}")
    print(f"summary={args.output_dir / 'summary_metrics.csv'}")
    print(f"readme={args.output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
