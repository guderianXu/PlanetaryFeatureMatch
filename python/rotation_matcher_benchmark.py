#!/usr/bin/env python3
"""Benchmark rotation matching baselines on same-image 90/180/270 degree pairs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


CSV_FIELDS = [
    "image_style",
    "image_path",
    "angle",
    "matcher",
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
class ResultRow:
    image_style: str
    image_path: str
    angle: int
    matcher: str
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


class Matcher(Protocol):
    name: str

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        ...


class UnavailableMatcher:
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        raise RuntimeError(self.reason)


class OpenCvFeatureMatcher:
    def __init__(self, name: str, detector, norm_type: int, *, max_matches: int) -> None:
        self.name = name
        self._detector = detector
        self._norm_type = norm_type
        self._max_matches = max_matches

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import cv2

        keypoints_a, descriptors_a = self._detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self._detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))
        matcher = cv2.BFMatcher(self._norm_type, crossCheck=True)
        matches = sorted(matcher.match(descriptors_a, descriptors_b), key=lambda item: item.distance)
        matches = matches[: self._max_matches]
        points_a = np.array([keypoints_a[item.queryIdx].pt for item in matches], dtype=np.float32)
        points_b = np.array([keypoints_b[item.trainIdx].pt for item in matches], dtype=np.float32)
        return MatchOutput(points_a, points_b, len(keypoints_a), len(keypoints_b))


class RootSiftFlannRansacMatcher:
    def __init__(
        self,
        detector,
        *,
        max_matches: int,
        name: str = "RootSIFT-FLANN-RANSAC",
        geometric_method: int | None = None,
        geometric_model: str = "affine",
        ratio: float = 0.75,
        ransac_threshold_px: float = 4.0,
    ) -> None:
        self.name = name
        self._detector = detector
        self._max_matches = max_matches
        self._geometric_method = geometric_method
        self._geometric_model = geometric_model
        self._ratio = ratio
        self._ransac_threshold_px = ransac_threshold_px

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import cv2

        keypoints_a, descriptors_a = self._detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self._detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return MatchOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []))

        descriptors_a = normalize_sift_descriptors_to_rootsift(descriptors_a)
        descriptors_b = normalize_sift_descriptors_to_rootsift(descriptors_b)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        forward = ratio_filter_knn_matches(matcher.knnMatch(descriptors_a, descriptors_b, k=2), self._ratio)
        reverse = ratio_filter_knn_matches(matcher.knnMatch(descriptors_b, descriptors_a, k=2), self._ratio)
        reverse_best = {item.queryIdx: item.trainIdx for item in reverse}
        mutual = [item for item in forward if reverse_best.get(item.trainIdx) == item.queryIdx]
        mutual = sorted(mutual, key=lambda item: item.distance)
        if not mutual:
            return MatchOutput(empty_points(), empty_points(), len(keypoints_a), len(keypoints_b))

        points_a = np.array([keypoints_a[item.queryIdx].pt for item in mutual], dtype=np.float32)
        points_b = np.array([keypoints_b[item.trainIdx].pt for item in mutual], dtype=np.float32)
        if self._geometric_model == "homography_usac":
            points_a, points_b = filter_points_with_homography_usac(
                points_a,
                points_b,
                method=self._geometric_method,
                reprojection_threshold_px=self._ransac_threshold_px,
                max_matches=self._max_matches,
            )
            return MatchOutput(points_a, points_b, len(keypoints_a), len(keypoints_b))
        if points_a.shape[0] >= 3:
            method = cv2.RANSAC if self._geometric_method is None else self._geometric_method
            _, inliers = cv2.estimateAffinePartial2D(
                points_a,
                points_b,
                method=method,
                ransacReprojThreshold=self._ransac_threshold_px,
                maxIters=2000,
                confidence=0.995,
            )
            if inliers is not None:
                inlier_mask = inliers.reshape(-1).astype(bool)
                points_a = points_a[inlier_mask]
                points_b = points_b[inlier_mask]

        points_a = points_a[: self._max_matches]
        points_b = points_b[: self._max_matches]
        return MatchOutput(points_a, points_b, len(keypoints_a), len(keypoints_b))


class PFMPyTorchMatcher:
    def __init__(
        self,
        *,
        state_path: Path,
        device: str,
        max_keypoints: int,
        max_matches: int,
        min_intensity: float,
        min_score: float,
    ) -> None:
        self.name = "PFM"
        self._state_path = state_path
        self._device = device
        self._max_keypoints = max_keypoints
        self._max_matches = max_matches
        self._min_intensity = min_intensity
        self._min_score = min_score
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        import pfm_model

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for PFM but torch.cuda.is_available() is false")
        self._model, _ = pfm_model.load_pytorch_state(self._state_path, device=self._device)
        self._model.eval()
        return self._model

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import torch
        import pytorch_cache_match_eval as eval_py

        model = self._load()
        tensor_a = image_to_tensor(image_a, device=self._device)
        tensor_b = image_to_tensor(image_b, device=self._device)
        with torch.no_grad():
            descriptors_a = model.descriptor_map_single(tensor_a)
            descriptors_b = model.descriptor_map_single(tensor_b)
            keypoints_a, selected_a = eval_py.select_descriptor_keypoints(
                tensor_a.squeeze(0),
                descriptors_a,
                max_keypoints=self._max_keypoints,
                min_intensity=self._min_intensity,
            )
            keypoints_b, selected_b = eval_py.select_descriptor_keypoints(
                tensor_b.squeeze(0),
                descriptors_b,
                max_keypoints=self._max_keypoints,
                min_intensity=self._min_intensity,
            )
            rows_a = eval_py.gather_descriptor_rows(descriptors_a, selected_a)
            rows_b = eval_py.gather_descriptor_rows(descriptors_b, selected_b)
            matches, _ = eval_py.mutual_nearest_matches(
                rows_a,
                rows_b,
                max_matches=self._max_matches,
                min_score=self._min_score,
            )
            if matches.numel() == 0:
                return MatchOutput(empty_points(), empty_points(), int(keypoints_a.size(0)), int(keypoints_b.size(0)))
            points_a = eval_py._feature_to_image_points(
                keypoints_a.index_select(0, matches[:, 0].to(keypoints_a.device)),
                feature_height=descriptors_a.size(2),
                feature_width=descriptors_a.size(3),
                image_height=image_a.shape[0],
                image_width=image_a.shape[1],
            )
            points_b = eval_py._feature_to_image_points(
                keypoints_b.index_select(0, matches[:, 1].to(keypoints_b.device)),
                feature_height=descriptors_b.size(2),
                feature_width=descriptors_b.size(3),
                image_height=image_b.shape[0],
                image_width=image_b.shape[1],
            )
        return MatchOutput(
            points_a.detach().cpu().numpy().astype(np.float32),
            points_b.detach().cpu().numpy().astype(np.float32),
            int(keypoints_a.size(0)),
            int(keypoints_b.size(0)),
        )


class LightGlueFeatureMatcher:
    def __init__(
        self,
        *,
        name: str,
        feature_name: str,
        extractor_name: str,
        device: str,
        max_keypoints: int,
        max_matches: int,
    ) -> None:
        self.name = name
        self._feature_name = feature_name
        self._extractor_name = extractor_name
        self._device = device
        self._max_keypoints = max_keypoints
        self._max_matches = max_matches
        self._extractor = None
        self._matcher = None

    def _load(self):
        if self._extractor is not None and self._matcher is not None:
            return self._extractor, self._matcher
        import torch
        import lightglue

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for LightGlue but torch.cuda.is_available() is false")
        try:
            extractor_class = getattr(lightglue, self._extractor_name)
        except AttributeError as exc:
            raise RuntimeError(f"lightglue.{self._extractor_name} unavailable") from exc
        self._extractor = extractor_class(max_num_keypoints=self._max_keypoints).eval().to(self._device)
        self._matcher = lightglue.LightGlue(features=self._feature_name).eval().to(self._device)
        return self._extractor, self._matcher

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import torch
        from lightglue.utils import numpy_image_to_torch

        extractor, matcher = self._load()
        image0 = numpy_image_to_torch(image_a).to(self._device)
        image1 = numpy_image_to_torch(image_b).to(self._device)
        with torch.no_grad():
            features0 = extractor.extract(image0)
            features1 = extractor.extract(image1)
            prediction = matcher({"image0": features0, "image1": features1})

        keypoints0 = features0["keypoints"][0]
        keypoints1 = features1["keypoints"][0]
        matches = prediction["matches"][0]
        if matches.numel() == 0:
            return MatchOutput(empty_points(), empty_points(), int(keypoints0.size(0)), int(keypoints1.size(0)))

        if "scores" in prediction and prediction["scores"]:
            scores = prediction["scores"][0]
            order = torch.argsort(scores, descending=True)[: self._max_matches]
            matches = matches.index_select(0, order.to(matches.device))
        else:
            matches = matches[: self._max_matches]
        points_a = keypoints0.index_select(0, matches[:, 0].to(keypoints0.device))
        points_b = keypoints1.index_select(0, matches[:, 1].to(keypoints1.device))
        return MatchOutput(
            points_a.detach().cpu().numpy().astype(np.float32),
            points_b.detach().cpu().numpy().astype(np.float32),
            int(keypoints0.size(0)),
            int(keypoints1.size(0)),
        )


class LightGlueSiftMatcher(LightGlueFeatureMatcher):
    def __init__(self, *, device: str, max_keypoints: int, max_matches: int) -> None:
        super().__init__(
            name="LightGlue-SIFT",
            feature_name="sift",
            extractor_name="SIFT",
            device=device,
            max_keypoints=max_keypoints,
            max_matches=max_matches,
        )


class LightGlueSuperPointMatcher(LightGlueFeatureMatcher):
    def __init__(self, *, device: str, max_keypoints: int, max_matches: int) -> None:
        super().__init__(
            name="LightGlue-SuperPoint",
            feature_name="superpoint",
            extractor_name="SuperPoint",
            device=device,
            max_keypoints=max_keypoints,
            max_matches=max_matches,
        )


class LightGlueDiskMatcher(LightGlueFeatureMatcher):
    def __init__(self, *, device: str, max_keypoints: int, max_matches: int) -> None:
        super().__init__(
            name="LightGlue-DISK",
            feature_name="disk",
            extractor_name="DISK",
            device=device,
            max_keypoints=max_keypoints,
            max_matches=max_matches,
        )


class LightGlueAlikedMatcher(LightGlueFeatureMatcher):
    def __init__(self, *, device: str, max_keypoints: int, max_matches: int) -> None:
        super().__init__(
            name="LightGlue-ALIKED",
            feature_name="aliked",
            extractor_name="ALIKED",
            device=device,
            max_keypoints=max_keypoints,
            max_matches=max_matches,
        )


class KorniaLoFTRMatcher:
    def __init__(self, *, device: str, max_matches: int, pretrained: str) -> None:
        self.name = "LoFTR"
        self._device = device
        self._max_matches = max_matches
        self._pretrained = pretrained
        self._matcher = None

    def _load(self):
        if self._matcher is not None:
            return self._matcher
        import torch
        from kornia.feature import LoFTR

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for LoFTR but torch.cuda.is_available() is false")
        self._matcher = LoFTR(pretrained=self._pretrained).eval().to(self._device)
        return self._matcher

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import torch

        matcher = self._load()
        tensor_a = image_to_tensor(image_a, device=self._device)
        tensor_b = image_to_tensor(image_b, device=self._device)
        if tensor_a.size(1) != 1:
            tensor_a = tensor_a.mean(dim=1, keepdim=True)
        if tensor_b.size(1) != 1:
            tensor_b = tensor_b.mean(dim=1, keepdim=True)
        with torch.no_grad():
            prediction = matcher({"image0": tensor_a, "image1": tensor_b})

        keypoints0 = prediction["keypoints0"]
        keypoints1 = prediction["keypoints1"]
        confidence = prediction.get("confidence")
        if "batch_indexes" in prediction:
            batch_mask = prediction["batch_indexes"] == 0
            keypoints0 = keypoints0[batch_mask]
            keypoints1 = keypoints1[batch_mask]
            if confidence is not None:
                confidence = confidence[batch_mask]

        match_count = int(keypoints0.size(0))
        if match_count == 0:
            return MatchOutput(empty_points(), empty_points(), 0, 0)

        if confidence is not None and int(confidence.numel()) == match_count:
            order = torch.argsort(confidence, descending=True)[: self._max_matches]
        else:
            order = torch.arange(match_count, device=keypoints0.device)[: self._max_matches]
        points_a = keypoints0.index_select(0, order.to(keypoints0.device))
        points_b = keypoints1.index_select(0, order.to(keypoints1.device))
        return MatchOutput(
            points_a.detach().cpu().numpy().astype(np.float32),
            points_b.detach().cpu().numpy().astype(np.float32),
            match_count,
            match_count,
        )


def empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def normalize_sift_descriptors_to_rootsift(descriptors: np.ndarray) -> np.ndarray:
    descriptors = descriptors.astype(np.float32, copy=False)
    l1 = descriptors.sum(axis=1, keepdims=True)
    normalized = np.divide(descriptors, l1, out=np.zeros_like(descriptors), where=l1 > 0.0)
    return np.sqrt(normalized, out=normalized)


def ratio_filter_knn_matches(knn_matches, ratio: float):
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair[0], pair[1]
        if first.distance < ratio * second.distance:
            good.append(first)
    return good


def filter_points_with_homography_usac(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    method: int | None,
    reprojection_threshold_px: float,
    max_matches: int,
) -> tuple[np.ndarray, np.ndarray]:
    if points_a.shape[0] < 4:
        return points_a[:max_matches], points_b[:max_matches]
    import cv2

    robust_method = cv2.USAC_MAGSAC if method is None else method
    _, inliers = cv2.findHomography(
        points_a,
        points_b,
        method=robust_method,
        ransacReprojThreshold=reprojection_threshold_px,
        maxIters=2000,
        confidence=0.995,
    )
    if inliers is not None:
        inlier_mask = inliers.reshape(-1).astype(bool)
        points_a = points_a[inlier_mask]
        points_b = points_b[inlier_mask]
    return points_a[:max_matches], points_b[:max_matches]


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle not in (90, 180, 270):
        raise ValueError("angle must be one of 90, 180, 270")
    return np.ascontiguousarray(np.rot90(image, k=angle // 90))


def rotate_points(points: np.ndarray, *, width: int, height: int, angle: int) -> np.ndarray:
    if points.size == 0:
        return empty_points()
    x = points[:, 0]
    y = points[:, 1]
    if angle == 90:
        rotated = np.stack([y, width - 1 - x], axis=1)
    elif angle == 180:
        rotated = np.stack([width - 1 - x, height - 1 - y], axis=1)
    elif angle == 270:
        rotated = np.stack([height - 1 - y, x], axis=1)
    else:
        raise ValueError("angle must be one of 90, 180, 270")
    return rotated.astype(np.float32, copy=False)


def image_to_tensor(image: np.ndarray, *, device: str):
    import torch

    array = image.astype(np.float32)
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    if array.ndim == 2:
        array = array[None, None, :, :]
    else:
        array = array.transpose(2, 0, 1)[None, :, :, :]
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def normalize_image_for_matching(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = image.mean(axis=2)
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    valid = image[finite]
    low = float(np.percentile(valid, 1.0))
    high = float(np.percentile(valid, 99.0))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    image = None
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception:
        image = None
    if image is None:
        try:
            import tifffile

            image = tifffile.imread(path)
        except Exception:
            image = None
    if image is None:
        try:
            from PIL import Image

            image = np.asarray(Image.open(path))
        except Exception as exc:
            raise FileNotFoundError(f"failed to read image: {path}") from exc
    return normalize_image_for_matching(image)


def resize_long_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return image
    height, width = image.shape[:2]
    current = max(height, width)
    if current <= max_edge:
        return image
    scale = float(max_edge) / float(current)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    try:
        import cv2

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BILINEAR))


def compute_metrics(points_a: np.ndarray, points_b: np.ndarray, *, width: int, height: int, angle: int, threshold_px: float):
    if points_a.size == 0 or points_b.size == 0:
        return 0, 0, 0.0, math.nan, math.nan
    expected_b = rotate_points(points_a, width=width, height=height, angle=angle)
    errors = np.linalg.norm(expected_b - points_b, axis=1)
    correct = int(np.count_nonzero(errors <= threshold_px))
    total = int(errors.shape[0])
    wrong = total - correct
    precision = 0.0 if total == 0 else correct / total
    return correct, wrong, precision, float(errors.mean()), float(np.median(errors))


def evaluate_matcher_on_rotation(
    matcher: Matcher,
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    image_style: str,
    image_path: Path,
    angle: int,
    output_dir: Path,
    threshold_px: float,
) -> tuple[ResultRow, np.ndarray, np.ndarray]:
    visualization = output_dir / "visualizations" / f"{image_style}_{angle}_{safe_name(matcher.name)}.png"
    try:
        output = matcher.match(image_a, image_b)
        correct, wrong, precision, mean_error, median_error = compute_metrics(
            output.points_a,
            output.points_b,
            width=image_a.shape[1],
            height=image_a.shape[0],
            angle=angle,
            threshold_px=threshold_px,
        )
        status = "ok"
        message = ""
        points_a = output.points_a
        points_b = output.points_b
        keypoints_a = output.keypoints_a
        keypoints_b = output.keypoints_b
    except Exception as exc:  # Keep benchmark rows complete when optional matchers fail.
        status = "unavailable" if isinstance(matcher, UnavailableMatcher) else "error"
        message = str(exc)
        points_a = empty_points()
        points_b = empty_points()
        keypoints_a = 0
        keypoints_b = 0
        correct = 0
        wrong = 0
        precision = 0.0
        mean_error = math.nan
        median_error = math.nan
    row = ResultRow(
        image_style=image_style,
        image_path=image_path.as_posix(),
        angle=angle,
        matcher=matcher.name,
        status=status,
        keypoints_a=keypoints_a,
        keypoints_b=keypoints_b,
        matches=int(points_a.shape[0]),
        correct=correct,
        wrong=wrong,
        precision=precision,
        mean_error_px=mean_error,
        median_error_px=median_error,
        visualization=visualization.as_posix() if points_a.size else "",
        message=message,
    )
    return row, points_a, points_b


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def write_metrics_csv(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            for key in ("precision", "mean_error_px", "median_error_px"):
                value = data[key]
                data[key] = "nan" if isinstance(value, float) and math.isnan(value) else f"{float(value):.6f}"
            writer.writerow(data)


def draw_visualization(image_a: np.ndarray, image_b: np.ndarray, points_a: np.ndarray, points_b: np.ndarray, path: Path) -> None:
    if points_a.size == 0 or points_b.size == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    height = max(image_a.shape[0], image_b.shape[0])
    width_a = image_a.shape[1]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.imshow(image_a, cmap="gray", extent=(0, width_a, height, 0))
    ax.imshow(image_b, cmap="gray", extent=(width_a, width_a + image_b.shape[1], height, 0))
    count = min(256, points_a.shape[0], points_b.shape[0])
    for index in range(count):
        x_a, y_a = points_a[index]
        x_b, y_b = points_b[index]
        ax.plot([x_a, width_a + x_b], [y_a, y_b], linewidth=0.55, alpha=0.65)
    ax.scatter(points_a[:count, 0], points_a[:count, 1], s=5)
    ax.scatter(width_a + points_b[:count, 0], points_b[:count, 1], s=5)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_opencv_matchers(max_matches: int) -> list[Matcher]:
    matchers: list[Matcher] = []
    try:
        import cv2
    except Exception as exc:
        reason = f"OpenCV unavailable: {exc}"
        return [
            UnavailableMatcher("SIFT", reason),
            UnavailableMatcher("RootSIFT-FLANN-RANSAC", reason),
            UnavailableMatcher("AffineSIFT-BF", reason),
            UnavailableMatcher("ORB", reason),
            UnavailableMatcher("AKAZE", reason),
        ]
    if hasattr(cv2, "SIFT_create"):
        matchers.append(OpenCvFeatureMatcher("SIFT", cv2.SIFT_create(), cv2.NORM_L2, max_matches=max_matches))
        matchers.append(RootSiftFlannRansacMatcher(cv2.SIFT_create(), max_matches=max_matches))
        if hasattr(cv2, "USAC_MAGSAC"):
            matchers.append(
                RootSiftFlannRansacMatcher(
                    cv2.SIFT_create(),
                    max_matches=max_matches,
                    name="RootSIFT-FLANN-USAC-MAGSAC",
                    geometric_method=cv2.USAC_MAGSAC,
                    geometric_model="homography_usac",
                )
            )
        else:
            matchers.append(UnavailableMatcher("RootSIFT-FLANN-USAC-MAGSAC", "cv2.USAC_MAGSAC unavailable"))
        if hasattr(cv2, "USAC_PROSAC"):
            matchers.append(
                RootSiftFlannRansacMatcher(
                    cv2.SIFT_create(),
                    max_matches=max_matches,
                    name="RootSIFT-FLANN-USAC-PROSAC",
                    geometric_method=cv2.USAC_PROSAC,
                    geometric_model="homography_usac",
                )
            )
        else:
            matchers.append(UnavailableMatcher("RootSIFT-FLANN-USAC-PROSAC", "cv2.USAC_PROSAC unavailable"))
        if hasattr(cv2, "AffineFeature_create"):
            affine_sift = cv2.AffineFeature_create(cv2.SIFT_create())
            matchers.append(OpenCvFeatureMatcher("AffineSIFT-BF", affine_sift, cv2.NORM_L2, max_matches=max_matches))
        else:
            matchers.append(UnavailableMatcher("AffineSIFT-BF", "cv2.AffineFeature_create unavailable"))
    else:
        matchers.append(UnavailableMatcher("SIFT", "cv2.SIFT_create unavailable"))
        matchers.append(UnavailableMatcher("RootSIFT-FLANN-RANSAC", "cv2.SIFT_create unavailable"))
        matchers.append(UnavailableMatcher("RootSIFT-FLANN-USAC-MAGSAC", "cv2.SIFT_create unavailable"))
        matchers.append(UnavailableMatcher("RootSIFT-FLANN-USAC-PROSAC", "cv2.SIFT_create unavailable"))
        matchers.append(UnavailableMatcher("AffineSIFT-BF", "cv2.SIFT_create unavailable"))
    matchers.append(OpenCvFeatureMatcher("ORB", cv2.ORB_create(nfeatures=max_matches * 4), cv2.NORM_HAMMING, max_matches=max_matches))
    matchers.append(OpenCvFeatureMatcher("AKAZE", cv2.AKAZE_create(), cv2.NORM_HAMMING, max_matches=max_matches))
    return matchers


def make_optional_deep_matchers(
    *, device: str, max_keypoints: int, max_matches: int, loftr_pretrained: str = "outdoor"
) -> list[Matcher]:
    matchers: list[Matcher] = []
    if importlib.util.find_spec("lightglue") is None:
        matchers.append(UnavailableMatcher("LightGlue-SIFT", "optional dependency not found: lightglue"))
        matchers.append(UnavailableMatcher("LightGlue-SuperPoint", "optional dependency not found: lightglue"))
        matchers.append(UnavailableMatcher("LightGlue-DISK", "optional dependency not found: lightglue"))
        matchers.append(UnavailableMatcher("LightGlue-ALIKED", "optional dependency not found: lightglue"))
    else:
        matchers.append(LightGlueSiftMatcher(device=device, max_keypoints=max_keypoints, max_matches=max_matches))
        matchers.append(LightGlueSuperPointMatcher(device=device, max_keypoints=max_keypoints, max_matches=max_matches))
        matchers.append(LightGlueDiskMatcher(device=device, max_keypoints=max_keypoints, max_matches=max_matches))
        matchers.append(LightGlueAlikedMatcher(device=device, max_keypoints=max_keypoints, max_matches=max_matches))
    if importlib.util.find_spec("kornia.feature") is None:
        matchers.append(UnavailableMatcher("LoFTR", "optional dependency not found: kornia.feature"))
    else:
        matchers.append(KorniaLoFTRMatcher(device=device, max_matches=max_matches, pretrained=loftr_pretrained))
    if importlib.util.find_spec("match_pairs") is None:
        matchers.append(UnavailableMatcher("SuperGlue", "optional dependency not found: match_pairs"))
    else:
        matchers.append(UnavailableMatcher("SuperGlue", "optional dependency detected, but no local adapter is configured"))
    return matchers


def find_latest_pfm_state(root: Path) -> Path | None:
    candidates = list((root / "runs").glob("**/pytorch_pfm_state.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_matchers(args: argparse.Namespace, project_root: Path) -> list[Matcher]:
    matchers = make_opencv_matchers(args.max_matches)
    matchers.extend(
        make_optional_deep_matchers(
            device=args.device,
            max_keypoints=args.max_keypoints,
            max_matches=args.max_matches,
            loftr_pretrained=args.loftr_pretrained,
        )
    )
    pfm_state = args.pfm_pytorch_state
    if pfm_state is None and args.auto_pfm:
        pfm_state = find_latest_pfm_state(project_root)
    if pfm_state is None:
        matchers.append(UnavailableMatcher("PFM", "no --pfm-pytorch-state found; pass a state or enable runs autodiscovery"))
    elif importlib.util.find_spec("torch") is None:
        matchers.append(UnavailableMatcher("PFM", "PyTorch is not installed in this Python environment"))
    elif importlib.util.find_spec("pfm_model") is None:
        matchers.append(UnavailableMatcher("PFM", "pfm_model import failed; run with PYTHONPATH=python"))
    else:
        matchers.append(
            PFMPyTorchMatcher(
                state_path=pfm_state,
                device=args.device,
                max_keypoints=args.max_keypoints,
                max_matches=args.max_matches,
                min_intensity=args.min_intensity,
                min_score=args.pfm_min_score,
            )
        )
    return matchers


def run_benchmark(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = args.output_dir or project_root / "runs" / f"rotation_matcher_benchmark_{timestamp}"
    images = [("numeric", args.numeric_image), ("timestamp", args.timestamp_image)]
    matchers = build_matchers(args, project_root)
    rows: list[ResultRow] = []
    for image_style, image_path in images:
        image = resize_long_edge(read_image(image_path), args.resize_max)
        for angle in args.angles:
            rotated = rotate_image(image, angle)
            for matcher in matchers:
                row, points_a, points_b = evaluate_matcher_on_rotation(
                    matcher,
                    image,
                    rotated,
                    image_style=image_style,
                    image_path=image_path,
                    angle=angle,
                    output_dir=output_dir,
                    threshold_px=args.threshold_px,
                )
                rows.append(row)
                if row.visualization:
                    draw_visualization(image, rotated, points_a, points_b, Path(row.visualization))
                print(
                    f"{image_style} {angle:3d} {matcher.name:10s} {row.status:11s} "
                    f"matches={row.matches} correct={row.correct} precision={row.precision:.4f}",
                    flush=True,
                )
    write_metrics_csv(output_dir / "metrics.csv", rows)
    write_summary(output_dir / "summary.txt", rows)
    return output_dir


def write_summary(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    available = sorted({row.matcher for row in rows if row.status == "ok"})
    unavailable = sorted({row.matcher for row in rows if row.status != "ok"})
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"available={','.join(available) if available else 'none'}\n")
        handle.write(f"unavailable_or_error={','.join(unavailable) if unavailable else 'none'}\n")
        handle.write("metrics=metrics.csv\n")
        handle.write("visualizations=visualizations/\n")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-image", type=Path, default=project_root / "img" / "1.tif")
    parser.add_argument("--timestamp-image", type=Path, default=project_root / "img" / "20260514T064636672_NAS_PAN_L2b.tif")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--angles", type=int, nargs="+", default=[90, 180, 270], choices=[90, 180, 270])
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--resize-max", type=int, default=1024, help="Resize long edge before rotation; use 0 for original size")
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--pfm-pytorch-state", type=Path, default=None)
    parser.add_argument("--loftr-pretrained", default="outdoor", choices=["outdoor", "indoor", "indoor_new"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-auto-pfm", dest="auto_pfm", action="store_false")
    parser.set_defaults(auto_pfm=True)
    return parser.parse_args()


def main() -> int:
    output_dir = run_benchmark(parse_args())
    print(f"output_dir={output_dir}")
    print(f"metrics={output_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
