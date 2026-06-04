#!/usr/bin/env python3
"""PyTorch fine-tuning loop for the current PFM feature extractor."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import math
import os
import random
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

import hard_pair_mining
import pfm_model
import pose_pair_metadata
from patch_descriptor_training import (
    SyntheticPair,
    discover_pair_archives,
    load_libtorch_pair_archive,
    paired_descriptor_loss,
    paired_descriptor_metrics,
)


@dataclass(frozen=True)
class PseudoLabelMatches:
    points_a_xy: torch.Tensor
    points_b_xy: torch.Tensor


class PairArchiveCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[Path, SyntheticPair] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.prefetch_inserts = 0

    def get(self, path: Path, *, device: torch.device) -> SyntheticPair:
        key = path.resolve(strict=False)
        if self.max_items > 0 and key in self._items:
            self.hits += 1
            pair = self._items.pop(key)
            self._items[key] = pair
        else:
            self.misses += 1
            pair = load_libtorch_pair_archive(key, device="cpu")
            if self.max_items > 0:
                self._items[key] = pair
                while len(self._items) > self.max_items:
                    self._items.popitem(last=False)
        return move_pair_to_device(pair, device=device)

    def put(self, path: Path, pair: SyntheticPair) -> None:
        if self.max_items <= 0:
            return
        key = path.resolve(strict=False)
        if key in self._items:
            self._items.pop(key)
        self._items[key] = pair
        self.prefetch_inserts += 1
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def put_batch(self, pairs: dict[Path, SyntheticPair]) -> None:
        for path, pair in pairs.items():
            self.put(path, pair)

    @property
    def size(self) -> int:
        return len(self._items)


def move_pair_to_device(pair: SyntheticPair, *, device: torch.device) -> SyntheticPair:
    return SyntheticPair(
        view_a=pair.view_a.to(device=device, non_blocking=True),
        view_b=pair.view_b.to(device=device, non_blocking=True),
        warp_a_to_b=pair.warp_a_to_b.to(device=device, non_blocking=True),
        valid_mask=pair.valid_mask.to(device=device, non_blocking=True),
    )


def load_pair_for_training(path: Path, *, device: torch.device, pair_cache: PairArchiveCache | None) -> SyntheticPair:
    if pair_cache is not None:
        return pair_cache.get(path, device=device)
    return load_libtorch_pair_archive(path, device=device)


def load_pair_batch_cpu(paths: list[Path]) -> dict[Path, SyntheticPair]:
    return {path.resolve(strict=False): load_libtorch_pair_archive(path, device="cpu") for path in paths}


@dataclass(frozen=True)
class FalseMatchLabels:
    points_a_xy: torch.Tensor
    points_b_xy: torch.Tensor


def _normalize_xy(points_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = points_xy[:, 0] * (2.0 / float(max(1, width - 1))) - 1.0
    y = points_xy[:, 1] * (2.0 / float(max(1, height - 1))) - 1.0
    return torch.stack([x, y], dim=1)


def _center_intensity(image: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    if points_xy.numel() == 0:
        return image.new_empty((0,))
    _, height, width = image.shape
    xy = points_xy.round().to(torch.long)
    x = xy[:, 0].clamp(0, width - 1)
    y = xy[:, 1].clamp(0, height - 1)
    return image.mean(dim=0)[y, x]


def _scale_points_to_feature_grid(
    points_xy: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    if points_xy.numel() == 0:
        return points_xy.new_empty((0, 2))
    x = points_xy[:, 0] * float(max(1, feature_width - 1)) / float(max(1, image_width - 1))
    y = points_xy[:, 1] * float(max(1, feature_height - 1)) / float(max(1, image_height - 1))
    return torch.stack([x, y], dim=1)


def _resized_hw(height: int, width: int, *, max_image_size: int) -> tuple[int, int]:
    if max_image_size <= 0 or max(height, width) <= max_image_size:
        return height, width
    scale = float(max_image_size) / float(max(height, width))
    return max(2, int(round(height * scale))), max(2, int(round(width * scale)))


def _clamped_crop_origin(center: torch.Tensor, *, crop_size: int, full_size: int) -> int:
    if crop_size >= full_size:
        return 0
    origin = int(torch.round(center - float(crop_size - 1) * 0.5).detach().cpu())
    return max(0, min(origin, full_size - crop_size))


def _uniform_crop_origin(
    full_size: int,
    crop_size: int,
    *,
    generator: torch.Generator | None,
    device: torch.device,
) -> int:
    if crop_size >= full_size:
        return 0
    if generator is None:
        return (full_size - crop_size) // 2
    limit = full_size - crop_size + 1
    return int(torch.randint(limit, (1,), generator=generator, device=device).detach().cpu()[0])


def crop_pair_for_training(
    pair: SyntheticPair,
    *,
    crop_size: int,
    generator: torch.Generator | None = None,
) -> SyntheticPair:
    if crop_size <= 0:
        return pair
    _, height_a, width_a = pair.view_a.shape
    _, height_b, width_b = pair.view_b.shape
    crop_h_a = min(int(crop_size), height_a)
    crop_w_a = min(int(crop_size), width_a)
    crop_h_b = min(int(crop_size), height_b)
    crop_w_b = min(int(crop_size), width_b)
    if (crop_h_a, crop_w_a, crop_h_b, crop_w_b) == (height_a, width_a, height_b, width_b):
        return pair

    finite_full_warp = torch.isfinite(pair.warp_a_to_b).all(dim=-1)
    valid_full = pair.valid_mask & finite_full_warp
    if bool(valid_full.any()):
        valid_yx = torch.nonzero(valid_full, as_tuple=False)
        if generator is None:
            selected_yx = valid_yx.to(dtype=torch.float32).mean(dim=0)
        else:
            selected_index = torch.randint(
                valid_yx.size(0),
                (1,),
                generator=generator,
                device=valid_yx.device,
            )[0]
            selected_yx = valid_yx[selected_index]
        ax0 = _clamped_crop_origin(selected_yx[1].to(torch.float32), crop_size=crop_w_a, full_size=width_a)
        ay0 = _clamped_crop_origin(selected_yx[0].to(torch.float32), crop_size=crop_h_a, full_size=height_a)
    else:
        ax0 = _uniform_crop_origin(width_a, crop_w_a, generator=generator, device=pair.view_a.device)
        ay0 = _uniform_crop_origin(height_a, crop_h_a, generator=generator, device=pair.view_a.device)
    ax1 = ax0 + crop_w_a
    ay1 = ay0 + crop_h_a

    warp_crop_full_b = pair.warp_a_to_b[ay0:ay1, ax0:ax1]
    valid_crop = pair.valid_mask[ay0:ay1, ax0:ax1].clone()
    finite_warp = torch.isfinite(warp_crop_full_b).all(dim=-1)
    valid_for_center = valid_crop & finite_warp
    if bool(valid_for_center.any()):
        center_b = warp_crop_full_b[valid_for_center].mean(dim=0)
        bx0 = _clamped_crop_origin(center_b[0], crop_size=crop_w_b, full_size=width_b)
        by0 = _clamped_crop_origin(center_b[1], crop_size=crop_h_b, full_size=height_b)
    else:
        bx0 = max(0, min(ax0, width_b - crop_w_b))
        by0 = max(0, min(ay0, height_b - crop_h_b))
    bx1 = bx0 + crop_w_b
    by1 = by0 + crop_h_b

    warp = warp_crop_full_b.clone()
    warp[..., 0] -= float(bx0)
    warp[..., 1] -= float(by0)
    valid_crop &= finite_warp
    valid_crop &= warp[..., 0] >= 0.0
    valid_crop &= warp[..., 0] <= float(crop_w_b - 1)
    valid_crop &= warp[..., 1] >= 0.0
    valid_crop &= warp[..., 1] <= float(crop_h_b - 1)
    return SyntheticPair(
        view_a=pair.view_a[:, ay0:ay1, ax0:ax1].contiguous(),
        view_b=pair.view_b[:, by0:by1, bx0:bx1].contiguous(),
        warp_a_to_b=warp.contiguous(),
        valid_mask=valid_crop.contiguous(),
    )


def resize_pair_for_training(pair: SyntheticPair, *, max_image_size: int) -> SyntheticPair:
    if max_image_size <= 0:
        return pair
    _, height_a, width_a = pair.view_a.shape
    _, height_b, width_b = pair.view_b.shape
    resized_a = _resized_hw(height_a, width_a, max_image_size=max_image_size)
    resized_b = _resized_hw(height_b, width_b, max_image_size=max_image_size)
    if resized_a == (height_a, width_a) and resized_b == (height_b, width_b):
        return pair
    new_height_a, new_width_a = resized_a
    new_height_b, new_width_b = resized_b
    view_a = F.interpolate(
        pair.view_a.unsqueeze(0),
        size=(new_height_a, new_width_a),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    view_b = F.interpolate(
        pair.view_b.unsqueeze(0),
        size=(new_height_b, new_width_b),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    warp = F.interpolate(
        pair.warp_a_to_b.permute(2, 0, 1).unsqueeze(0),
        size=(new_height_a, new_width_a),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0).permute(1, 2, 0).contiguous()
    scale_x_b = float(new_width_b - 1) / float(max(1, width_b - 1))
    scale_y_b = float(new_height_b - 1) / float(max(1, height_b - 1))
    warp = warp.clone()
    warp[..., 0] *= scale_x_b
    warp[..., 1] *= scale_y_b
    valid_mask = F.interpolate(
        pair.valid_mask.to(dtype=torch.float32).view(1, 1, height_a, width_a),
        size=(new_height_a, new_width_a),
        mode="area",
    ).view(new_height_a, new_width_a) > 0.0
    return SyntheticPair(
        view_a=view_a.contiguous(),
        view_b=view_b.contiguous(),
        warp_a_to_b=warp,
        valid_mask=valid_mask.contiguous(),
    )


def _pseudo_label_path_keys(path: Path) -> list[str]:
    keys: list[str] = []

    def add(value: Path) -> None:
        text = value.as_posix()
        if text not in keys:
            keys.append(text)

    add(path)
    try:
        add(path.resolve(strict=False))
    except OSError:
        pass
    try:
        cwd = Path.cwd().resolve()
        absolute = path if path.is_absolute() else cwd / path
        add(absolute.resolve(strict=False).relative_to(cwd))
    except (OSError, ValueError):
        pass
    return keys


def read_pseudo_label_matches(paths: list[Path]) -> dict[str, PseudoLabelMatches]:
    grouped_a: dict[str, list[list[float]]] = {}
    grouped_b: dict[str, list[list[float]]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pair_pt = (row.get("pair_pt") or "").strip()
                if not pair_pt:
                    continue
                try:
                    ax = float(row["ax"])
                    ay = float(row["ay"])
                    bx = float(row["bx"])
                    by = float(row["by"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in (ax, ay, bx, by)):
                    continue
                key = Path(pair_pt).as_posix()
                grouped_a.setdefault(key, []).append([ax, ay])
                grouped_b.setdefault(key, []).append([bx, by])
    return {
        key: PseudoLabelMatches(
            points_a_xy=torch.tensor(grouped_a[key], dtype=torch.float32),
            points_b_xy=torch.tensor(grouped_b[key], dtype=torch.float32),
        )
        for key in sorted(grouped_a)
    }


def read_false_match_labels(paths: list[Path]) -> dict[str, FalseMatchLabels]:
    grouped_a: dict[str, list[list[float]]] = {}
    grouped_b: dict[str, list[list[float]]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pair_pt = (row.get("pair_pt") or "").strip()
                if not pair_pt:
                    continue
                try:
                    ax = float(row["ax"])
                    ay = float(row["ay"])
                    bx = float(row["bx"])
                    by = float(row["by"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in (ax, ay, bx, by)):
                    continue
                key = Path(pair_pt).as_posix()
                grouped_a.setdefault(key, []).append([ax, ay])
                grouped_b.setdefault(key, []).append([bx, by])
    return {
        key: FalseMatchLabels(
            points_a_xy=torch.tensor(grouped_a[key], dtype=torch.float32),
            points_b_xy=torch.tensor(grouped_b[key], dtype=torch.float32),
        )
        for key in sorted(grouped_a)
    }


def pseudo_label_feature_correspondences(
    pair_path: Path,
    pair: SyntheticPair,
    labels_by_pair: dict[str, PseudoLabelMatches],
    *,
    feature_height: int,
    feature_width: int,
    max_points: int,
    generator: torch.Generator | None = None,
    min_intensity: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = None
    for key in _pseudo_label_path_keys(pair_path):
        labels = labels_by_pair.get(key)
        if labels is not None:
            break
    if labels is None:
        return pair.view_a.new_empty((0, 2)), pair.view_a.new_empty((0, 2))

    _, height_a, width_a = pair.view_a.shape
    _, height_b, width_b = pair.view_b.shape
    points_a = labels.points_a_xy.to(device=pair.view_a.device, dtype=torch.float32)
    points_b = labels.points_b_xy.to(device=pair.view_b.device, dtype=torch.float32)
    if points_a.shape != points_b.shape or points_a.dim() != 2 or points_a.size(1) != 2:
        return pair.view_a.new_empty((0, 2)), pair.view_a.new_empty((0, 2))

    valid = torch.isfinite(points_a).all(dim=1) & torch.isfinite(points_b).all(dim=1)
    valid &= points_a[:, 0] >= 0.0
    valid &= points_a[:, 0] <= float(width_a - 1)
    valid &= points_a[:, 1] >= 0.0
    valid &= points_a[:, 1] <= float(height_a - 1)
    valid &= points_b[:, 0] >= 0.0
    valid &= points_b[:, 0] <= float(width_b - 1)
    valid &= points_b[:, 1] >= 0.0
    valid &= points_b[:, 1] <= float(height_b - 1)
    points_a = points_a[valid]
    points_b = points_b[valid]
    if min_intensity > 0.0 and points_a.numel() > 0:
        textured = (_center_intensity(pair.view_a, points_a) > min_intensity) & (
            _center_intensity(pair.view_b, points_b) > min_intensity
        )
        points_a = points_a[textured]
        points_b = points_b[textured]
    if points_a.numel() == 0:
        return pair.view_a.new_empty((0, 2)), pair.view_a.new_empty((0, 2))

    take = points_a.size(0) if max_points <= 0 else min(max_points, points_a.size(0))
    if take < points_a.size(0):
        order = torch.randperm(points_a.size(0), generator=generator, device=points_a.device)[:take]
        points_a = points_a.index_select(0, order)
        points_b = points_b.index_select(0, order)
    feature_a = _scale_points_to_feature_grid(
        points_a,
        image_height=height_a,
        image_width=width_a,
        feature_height=feature_height,
        feature_width=feature_width,
    )
    feature_b = _scale_points_to_feature_grid(
        points_b,
        image_height=height_b,
        image_width=width_b,
        feature_height=feature_height,
        feature_width=feature_width,
    )
    return feature_a, feature_b


def false_match_feature_correspondences(
    pair_path: Path,
    pair: SyntheticPair,
    labels_by_pair: dict[str, FalseMatchLabels],
    *,
    feature_height: int,
    feature_width: int,
    max_points: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = None
    for key in _pseudo_label_path_keys(pair_path):
        labels = labels_by_pair.get(key)
        if labels is not None:
            break
    if labels is None:
        return pair.view_a.new_empty((0, 2)), pair.view_a.new_empty((0, 2))

    _, height_a, width_a = pair.view_a.shape
    _, height_b, width_b = pair.view_b.shape
    points_a = labels.points_a_xy.to(device=pair.view_a.device, dtype=torch.float32)
    points_b = labels.points_b_xy.to(device=pair.view_b.device, dtype=torch.float32)
    if points_a.shape != points_b.shape or points_a.dim() != 2 or points_a.size(1) != 2:
        return pair.view_a.new_empty((0, 2)), pair.view_a.new_empty((0, 2))

    valid = torch.isfinite(points_a).all(dim=1) & torch.isfinite(points_b).all(dim=1)
    valid &= points_a[:, 0] >= 0.0
    valid &= points_a[:, 0] <= float(width_a - 1)
    valid &= points_a[:, 1] >= 0.0
    valid &= points_a[:, 1] <= float(height_a - 1)
    valid &= points_b[:, 0] >= 0.0
    valid &= points_b[:, 0] <= float(width_b - 1)
    valid &= points_b[:, 1] >= 0.0
    valid &= points_b[:, 1] <= float(height_b - 1)
    points_a = points_a[valid]
    points_b = points_b[valid]
    if points_a.numel() == 0:
        return pair.view_a.new_empty((0, 2)), pair.view_a.new_empty((0, 2))

    take = points_a.size(0) if max_points <= 0 else min(max_points, points_a.size(0))
    if take < points_a.size(0):
        order = torch.randperm(points_a.size(0), generator=generator, device=points_a.device)[:take]
        points_a = points_a.index_select(0, order)
        points_b = points_b.index_select(0, order)
    feature_a = _scale_points_to_feature_grid(
        points_a,
        image_height=height_a,
        image_width=width_a,
        feature_height=feature_height,
        feature_width=feature_width,
    )
    feature_b = _scale_points_to_feature_grid(
        points_b,
        image_height=height_b,
        image_width=width_b,
        feature_height=feature_height,
        feature_width=feature_width,
    )
    return feature_a, feature_b


def select_pseudo_labeled_training_pairs(
    pair_paths: list[Path],
    labels_by_pair: dict[str, PseudoLabelMatches],
) -> list[Path]:
    selected: list[Path] = []
    for pair_path in pair_paths:
        if any(key in labels_by_pair for key in _pseudo_label_path_keys(pair_path)):
            selected.append(pair_path)
    return selected


def select_false_match_training_pairs(
    pair_paths: list[Path],
    labels_by_pair: dict[str, FalseMatchLabels],
) -> list[Path]:
    selected: list[Path] = []
    for pair_path in pair_paths:
        if any(key in labels_by_pair for key in _pseudo_label_path_keys(pair_path)):
            selected.append(pair_path)
    return selected


def sample_feature_correspondences(
    pair: SyntheticPair,
    *,
    feature_height: int,
    feature_width: int,
    count: int,
    min_intensity: float,
    weak_texture_fraction: float = 0.0,
    spatial_bins: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("count must be positive")
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("feature size must be positive")
    if weak_texture_fraction < 0.0 or weak_texture_fraction > 1.0:
        raise ValueError("weak_texture_fraction must be in [0, 1]")
    if spatial_bins < 0:
        raise ValueError("spatial_bins must be non-negative")
    _, height_a, width_a = pair.view_a.shape
    _, height_b, width_b = pair.view_b.shape
    yy, xx = torch.meshgrid(
        torch.arange(height_a, device=pair.view_a.device),
        torch.arange(width_a, device=pair.view_a.device),
        indexing="ij",
    )
    points_a_all = torch.stack([xx.to(torch.float32), yy.to(torch.float32)], dim=-1)
    points_b_all = pair.warp_a_to_b
    valid = pair.valid_mask.clone()
    valid &= points_b_all[..., 0] >= 0.0
    valid &= points_b_all[..., 0] <= float(width_b - 1)
    valid &= points_b_all[..., 1] >= 0.0
    valid &= points_b_all[..., 1] <= float(height_b - 1)
    points_a = points_a_all[valid]
    points_b = points_b_all[valid]
    if min_intensity > 0.0 and points_a.numel() > 0:
        textured = (_center_intensity(pair.view_a, points_a) > min_intensity) & (
            _center_intensity(pair.view_b, points_b) > min_intensity
        )
        points_a = points_a[textured]
        points_b = points_b[textured]
    if points_a.numel() == 0:
        return points_a.new_empty((0, 2)), points_b.new_empty((0, 2))

    take = min(count, points_a.size(0))
    if weak_texture_fraction > 0.0 and take > 1:
        order = weak_texture_spatially_balanced_order(
            pair.view_a,
            pair.view_b,
            points_a,
            points_b,
            take=take,
            weak_fraction=weak_texture_fraction,
            spatial_bins=spatial_bins,
            generator=generator,
        )
    elif spatial_bins > 0:
        order = spatially_balanced_order(
            points_a,
            take=take,
            spatial_bins=spatial_bins,
            image_height=height_a,
            image_width=width_a,
            generator=generator,
        )
    else:
        order = torch.randperm(points_a.size(0), generator=generator, device=points_a.device)[:take]
    points_a = points_a.index_select(0, order)
    points_b = points_b.index_select(0, order)
    feature_a = _scale_points_to_feature_grid(
        points_a,
        image_height=height_a,
        image_width=width_a,
        feature_height=feature_height,
        feature_width=feature_width,
    )
    feature_b = _scale_points_to_feature_grid(
        points_b,
        image_height=height_b,
        image_width=width_b,
        feature_height=feature_height,
        feature_width=feature_width,
    )
    return feature_a, feature_b


def image_local_texture_scores(image: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    if points_xy.dim() != 2 or points_xy.size(1) != 2:
        raise ValueError("points_xy must have shape Nx2")
    if points_xy.numel() == 0:
        return image.new_empty((0,))
    gray = image.to(torch.float32).mean(dim=0, keepdim=True).unsqueeze(0)
    local_mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3, count_include_pad=False)
    contrast = (gray - local_mean).abs()
    dx = (gray - torch.roll(gray, shifts=1, dims=3)).abs()
    dy = (gray - torch.roll(gray, shifts=1, dims=2)).abs()
    texture = contrast + dx + dy
    _, _, height, width = texture.shape
    rounded = points_xy.round().to(torch.long)
    x = rounded[:, 0].clamp(0, width - 1)
    y = rounded[:, 1].clamp(0, height - 1)
    return texture[0, 0, y, x].to(points_xy.device)


def weak_texture_balanced_order(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    *,
    take: int,
    weak_fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if take <= 0:
        return torch.empty(0, dtype=torch.long, device=points_a.device)
    total = points_a.size(0)
    if total <= take:
        return torch.arange(total, dtype=torch.long, device=points_a.device)
    weak_target = min(take, int(round(float(take) * float(weak_fraction))))
    if weak_target <= 0:
        return torch.randperm(total, generator=generator, device=points_a.device)[:take]
    texture = 0.5 * (
        image_local_texture_scores(image_a, points_a) + image_local_texture_scores(image_b, points_b)
    )
    weak_pool_size = min(total, max(weak_target, int(math.ceil(float(total) * 0.25))))
    weak_pool = torch.argsort(texture, stable=True)[:weak_pool_size]
    weak_take = min(weak_target, weak_pool.numel())
    weak_perm = torch.randperm(weak_pool.numel(), generator=generator, device=weak_pool.device)[:weak_take]
    selected = weak_pool.index_select(0, weak_perm)
    if selected.numel() < take:
        used = torch.zeros(total, dtype=torch.bool, device=points_a.device)
        used[selected] = True
        remaining = torch.nonzero(~used, as_tuple=False).reshape(-1)
        fill = torch.randperm(remaining.numel(), generator=generator, device=remaining.device)[: take - selected.numel()]
        selected = torch.cat([selected, remaining.index_select(0, fill)], dim=0)
    order = torch.randperm(selected.numel(), generator=generator, device=selected.device)
    return selected.index_select(0, order)[:take].contiguous()


def spatially_balanced_order(
    points_xy: torch.Tensor,
    *,
    take: int,
    spatial_bins: int,
    image_height: int,
    image_width: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if take <= 0:
        return torch.empty(0, dtype=torch.long, device=points_xy.device)
    total = points_xy.size(0)
    if total <= take:
        return torch.arange(total, dtype=torch.long, device=points_xy.device)
    if spatial_bins <= 0:
        return torch.randperm(total, generator=generator, device=points_xy.device)[:take]
    bins = int(spatial_bins)
    x_bin = torch.clamp((points_xy[:, 0] * bins / max(1, image_width)).floor().to(torch.long), 0, bins - 1)
    y_bin = torch.clamp((points_xy[:, 1] * bins / max(1, image_height)).floor().to(torch.long), 0, bins - 1)
    cell_ids = y_bin * bins + x_bin
    cell_order = torch.randperm(bins * bins, generator=generator, device=points_xy.device)
    chosen: list[torch.Tensor] = []
    leftovers: list[torch.Tensor] = []
    for cell_id in cell_order:
        members = torch.nonzero(cell_ids == cell_id, as_tuple=False).reshape(-1)
        if members.numel() == 0:
            continue
        perm = torch.randperm(members.numel(), generator=generator, device=members.device)
        shuffled = members.index_select(0, perm)
        chosen.append(shuffled[:1])
        if shuffled.numel() > 1:
            leftovers.append(shuffled[1:])
        if len(chosen) >= take:
            break
    if not chosen:
        return torch.randperm(total, generator=generator, device=points_xy.device)[:take]
    selected = torch.cat(chosen, dim=0)
    if selected.numel() < take:
        if leftovers:
            remaining = torch.cat(leftovers, dim=0)
        else:
            used = torch.zeros(total, dtype=torch.bool, device=points_xy.device)
            used[selected] = True
            remaining = torch.nonzero(~used, as_tuple=False).reshape(-1)
        if remaining.numel() > 0:
            fill_order = torch.randperm(remaining.numel(), generator=generator, device=remaining.device)
            fill = remaining.index_select(0, fill_order[: take - selected.numel()])
            selected = torch.cat([selected, fill], dim=0)
    order = torch.randperm(selected.numel(), generator=generator, device=selected.device)
    return selected.index_select(0, order)[:take].contiguous()


def weak_texture_spatially_balanced_order(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    *,
    take: int,
    weak_fraction: float,
    spatial_bins: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if spatial_bins <= 0:
        return weak_texture_balanced_order(
            image_a,
            image_b,
            points_a,
            points_b,
            take=take,
            weak_fraction=weak_fraction,
            generator=generator,
        )
    total = points_a.size(0)
    if total <= take:
        return torch.arange(total, dtype=torch.long, device=points_a.device)
    weak_target = min(take, int(round(float(take) * float(weak_fraction))))
    if weak_target <= 0:
        return spatially_balanced_order(
            points_a,
            take=take,
            spatial_bins=spatial_bins,
            image_height=image_a.size(1),
            image_width=image_a.size(2),
            generator=generator,
        )
    texture = 0.5 * (
        image_local_texture_scores(image_a, points_a) + image_local_texture_scores(image_b, points_b)
    )
    weak_pool_size = min(total, max(weak_target, int(math.ceil(float(total) * 0.25))))
    weak_pool = torch.argsort(texture, stable=True)[:weak_pool_size]
    weak_local_order = spatially_balanced_order(
        points_a.index_select(0, weak_pool),
        take=min(weak_target, weak_pool.numel()),
        spatial_bins=spatial_bins,
        image_height=image_a.size(1),
        image_width=image_a.size(2),
        generator=generator,
    )
    selected = weak_pool.index_select(0, weak_local_order)
    if selected.numel() < take:
        used = torch.zeros(total, dtype=torch.bool, device=points_a.device)
        used[selected] = True
        remaining = torch.nonzero(~used, as_tuple=False).reshape(-1)
        fill_local_order = spatially_balanced_order(
            points_a.index_select(0, remaining),
            take=min(take - selected.numel(), remaining.numel()),
            spatial_bins=spatial_bins,
            image_height=image_a.size(1),
            image_width=image_a.size(2),
            generator=generator,
        )
        selected = torch.cat([selected, remaining.index_select(0, fill_local_order)], dim=0)
    order = torch.randperm(selected.numel(), generator=generator, device=selected.device)
    return selected.index_select(0, order)[:take].contiguous()


def sample_descriptors(descriptor_map: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    if descriptor_map.dim() != 4 or descriptor_map.size(0) != 1:
        raise ValueError("descriptor_map must have shape 1xDxHxW")
    if points_xy.dim() != 2 or points_xy.size(1) != 2:
        raise ValueError("points_xy must have shape Nx2")
    if points_xy.size(0) == 0:
        return descriptor_map.new_empty((0, descriptor_map.size(1)))
    height = descriptor_map.size(2)
    width = descriptor_map.size(3)
    grid = _normalize_xy(points_xy, height, width).view(1, -1, 1, 2)
    sampled = F.grid_sample(descriptor_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def normalize_descriptor_batch(descriptors: torch.Tensor, *, eps: float = 1.0e-3) -> torch.Tensor:
    if descriptors.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    finite = torch.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    norm = finite.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return finite / norm


def apply_graph_metadata_mode(metadata: torch.Tensor, mode: str) -> torch.Tensor:
    adjusted = metadata.clone()
    if mode == "full":
        return adjusted
    if mode == "descriptor_only":
        return adjusted.zero_()
    if mode == "no_xy":
        adjusted[:, : min(adjusted.size(1), 4)] = 0.0
        return adjusted
    if mode == "no_geometry":
        if adjusted.size(1) > 5:
            adjusted[:, 5 : min(adjusted.size(1), 12)] = 0.0
        return adjusted
    if mode == "no_quality":
        if adjusted.size(1) > 12:
            adjusted[:, 12:] = 0.0
        return adjusted
    raise ValueError(f"unsupported graph metadata mode: {mode}")


def sample_unmatched_feature_points(
    *,
    feature_height: int,
    feature_width: int,
    reference_points: torch.Tensor,
    count: int,
    min_distance: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if count <= 0:
        return reference_points.new_empty((0, 2))
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("feature size must be positive")
    out_device = device if device is not None else reference_points.device
    total = feature_height * feature_width
    if total <= 0:
        return torch.empty((0, 2), dtype=torch.float32, device=out_device)
    candidate_count = min(total, max(count * 32, count + 128))
    flat = torch.randperm(total, generator=generator, device=out_device)[:candidate_count]
    y = torch.div(flat, feature_width, rounding_mode="floor").to(torch.float32)
    x = (flat % feature_width).to(torch.float32)
    candidates = torch.stack([x, y], dim=1)
    if reference_points.numel() > 0 and min_distance > 0.0:
        refs = reference_points.to(device=out_device, dtype=torch.float32)
        distances = torch.cdist(candidates, refs)
        keep = distances.min(dim=1).values.ge(float(min_distance))
        candidates = candidates[keep]
    return candidates[:count].contiguous()


def _candidate_grid_descriptors(
    descriptor_map: torch.Tensor,
    *,
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if descriptor_map.dim() != 4 or descriptor_map.size(0) != 1:
        raise ValueError("descriptor_map must have shape 1xDxHxW")
    height = descriptor_map.size(2)
    width = descriptor_map.size(3)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=descriptor_map.device),
        torch.arange(width, device=descriptor_map.device),
        indexing="ij",
    )
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(dtype=descriptor_map.dtype)
    descriptors = descriptor_map.squeeze(0).flatten(1).transpose(0, 1).contiguous()
    total = descriptors.size(0)
    if max_candidates > 0 and total > max_candidates:
        indices = torch.linspace(0, total - 1, steps=max_candidates, device=descriptor_map.device)
        indices = indices.round().to(torch.long).unique(sorted=True)
        descriptors = descriptors.index_select(0, indices)
        coords = coords.index_select(0, indices)
    return descriptors, coords


def warp_aware_hard_negative_loss(
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    points_a_xy: torch.Tensor,
    points_b_xy: torch.Tensor,
    *,
    negative_radius: float = 2.0,
    margin: float = 0.2,
    max_candidates: int = 4096,
) -> torch.Tensor:
    if points_a_xy.dim() != 2 or points_a_xy.size(1) != 2:
        raise ValueError("points_a_xy must have shape Nx2")
    if points_b_xy.dim() != 2 or points_b_xy.size(1) != 2:
        raise ValueError("points_b_xy must have shape Nx2")
    if points_a_xy.shape != points_b_xy.shape:
        raise ValueError("point tensors must have the same shape")
    if points_a_xy.size(0) == 0:
        return descriptors_a.new_zeros(())
    query = normalize_descriptor_batch(sample_descriptors(descriptors_a, points_a_xy))
    positive = normalize_descriptor_batch(sample_descriptors(descriptors_b, points_b_xy))
    candidates, candidate_xy = _candidate_grid_descriptors(descriptors_b, max_candidates=max_candidates)
    candidates = normalize_descriptor_batch(candidates)

    similarity = query @ candidates.T
    positive_similarity = (query * positive).sum(dim=1)
    target_xy = points_b_xy.to(device=candidate_xy.device, dtype=candidate_xy.dtype)
    distance_sq = (candidate_xy.unsqueeze(0) - target_xy.unsqueeze(1)).pow(2).sum(dim=2)
    valid_negative = distance_sq > float(negative_radius) * float(negative_radius)
    has_negative = valid_negative.any(dim=1)
    if not bool(has_negative.any()):
        return descriptors_a.new_zeros(())

    masked_similarity = similarity.masked_fill(~valid_negative, -float("inf"))
    hardest_wrong = masked_similarity.max(dim=1).values
    penalty = hardest_wrong[has_negative] - positive_similarity[has_negative] + float(margin)
    return penalty.clamp_min(0.0).pow(2).mean()


def descriptor_false_match_suppression_loss(
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    points_a_xy: torch.Tensor,
    points_b_xy: torch.Tensor,
    *,
    negative_radius: float = 2.0,
    max_false_score: float = 0.35,
    topk: int = 8,
    max_candidates: int = 4096,
) -> torch.Tensor:
    if points_a_xy.dim() != 2 or points_a_xy.size(1) != 2:
        raise ValueError("points_a_xy must have shape Nx2")
    if points_b_xy.dim() != 2 or points_b_xy.size(1) != 2:
        raise ValueError("points_b_xy must have shape Nx2")
    if points_a_xy.shape != points_b_xy.shape:
        raise ValueError("point tensors must have the same shape")
    if negative_radius < 0.0:
        raise ValueError("negative_radius must be nonnegative")
    if max_false_score < -1.0 or max_false_score > 1.0:
        raise ValueError("max_false_score must be in [-1, 1]")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative")
    if points_a_xy.size(0) == 0:
        return descriptors_a.new_zeros(())

    query = normalize_descriptor_batch(sample_descriptors(descriptors_a, points_a_xy))
    candidates, candidate_xy = _candidate_grid_descriptors(descriptors_b, max_candidates=max_candidates)
    candidates = normalize_descriptor_batch(candidates)

    similarity = query @ candidates.T
    target_xy = points_b_xy.to(device=candidate_xy.device, dtype=candidate_xy.dtype)
    distance_sq = (candidate_xy.unsqueeze(0) - target_xy.unsqueeze(1)).pow(2).sum(dim=2)
    valid_negative = distance_sq > float(negative_radius) * float(negative_radius)
    if not bool(valid_negative.any()):
        return descriptors_a.new_zeros(())

    masked_similarity = similarity.masked_fill(~valid_negative, -float("inf"))
    k = min(int(topk), masked_similarity.size(1))
    top_values = torch.topk(masked_similarity, k=k, dim=1).values
    finite = torch.isfinite(top_values)
    if not bool(finite.any()):
        return descriptors_a.new_zeros(())
    excess = top_values[finite] - float(max_false_score)
    return excess.clamp_min(0.0).pow(2).mean()


def paired_cyclic_similarity(desc_a: torch.Tensor, desc_b: torch.Tensor) -> torch.Tensor:
    if desc_a.dim() != 2 or desc_b.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if desc_a.shape != desc_b.shape:
        raise ValueError("descriptor tensors must have the same shape")
    desc_a = normalize_descriptor_batch(desc_a)
    desc_b = normalize_descriptor_batch(desc_b)
    channels = desc_a.size(1)
    if channels < 4 or channels % 4 != 0:
        return (desc_a * desc_b).sum(dim=1)
    group = channels // 4
    scores = [(desc_a * torch.roll(desc_b, shifts=turns * group, dims=1)).sum(dim=1) for turns in range(4)]
    return torch.stack(scores, dim=0).max(dim=0).values


def false_match_negative_loss(
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    points_a_xy: torch.Tensor,
    points_b_xy: torch.Tensor,
    *,
    max_false_score: float = 0.25,
) -> torch.Tensor:
    if points_a_xy.dim() != 2 or points_a_xy.size(1) != 2:
        raise ValueError("points_a_xy must have shape Nx2")
    if points_b_xy.dim() != 2 or points_b_xy.size(1) != 2:
        raise ValueError("points_b_xy must have shape Nx2")
    if points_a_xy.shape != points_b_xy.shape:
        raise ValueError("point tensors must have the same shape")
    if max_false_score < -1.0 or max_false_score > 1.0:
        raise ValueError("max_false_score must be in [-1, 1]")
    if points_a_xy.size(0) == 0:
        return descriptors_a.new_zeros(())
    desc_a = sample_descriptors(descriptors_a, points_a_xy)
    desc_b = sample_descriptors(descriptors_b, points_b_xy)
    similarity = paired_cyclic_similarity(desc_a, desc_b)
    return (similarity - float(max_false_score)).clamp_min(0.0).pow(2).mean()


def descriptor_map_pair_loss(
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    points_a_xy: torch.Tensor,
    points_b_xy: torch.Tensor,
    *,
    temperature: float = 0.07,
    teacher_descriptors_a: torch.Tensor | None = None,
    teacher_descriptors_b: torch.Tensor | None = None,
    teacher_weight: float = 1.0,
    hard_negative_weight: float = 0.5,
    diversity_weight: float = 0.10,
    warp_hard_negative_weight: float = 0.0,
    warp_hard_negative_radius: float = 2.0,
    warp_hard_negative_margin: float = 0.2,
    warp_hard_negative_candidates: int = 4096,
    abstention_weight: float = 0.0,
    abstention_negative_radius: float = 2.0,
    abstention_max_false_score: float = 0.35,
    abstention_topk: int = 8,
    abstention_candidates: int = 4096,
) -> tuple[torch.Tensor, dict[str, float]]:
    if points_a_xy.size(0) == 0:
        raise ValueError("descriptor map pair loss requires at least one correspondence")
    desc_a = normalize_descriptor_batch(sample_descriptors(descriptors_a, points_a_xy))
    desc_b = normalize_descriptor_batch(sample_descriptors(descriptors_b, points_b_xy))
    loss = paired_descriptor_loss(desc_a, desc_b, temperature=temperature, diversity_weight=diversity_weight)
    if hard_negative_weight > 0.0:
        loss = loss + hard_negative_weight * hard_negative_margin_loss(desc_a, desc_b, margin=0.20)
    if warp_hard_negative_weight > 0.0:
        loss = loss + warp_hard_negative_weight * warp_aware_hard_negative_loss(
            descriptors_a,
            descriptors_b,
            points_a_xy,
            points_b_xy,
            negative_radius=warp_hard_negative_radius,
            margin=warp_hard_negative_margin,
            max_candidates=warp_hard_negative_candidates,
        )
    if abstention_weight > 0.0:
        loss = loss + abstention_weight * descriptor_false_match_suppression_loss(
            descriptors_a,
            descriptors_b,
            points_a_xy,
            points_b_xy,
            negative_radius=abstention_negative_radius,
            max_false_score=abstention_max_false_score,
            topk=abstention_topk,
            max_candidates=abstention_candidates,
        )
    if teacher_descriptors_a is not None and teacher_descriptors_b is not None and teacher_weight > 0.0:
        teacher_a = normalize_descriptor_batch(sample_descriptors(teacher_descriptors_a, points_a_xy))
        teacher_b = normalize_descriptor_batch(sample_descriptors(teacher_descriptors_b, points_b_xy))
        teacher_loss = 0.5 * (
            teacher_guided_descriptor_loss(desc_a, teacher_b, temperature=temperature)
            + teacher_guided_descriptor_loss(desc_b, teacher_a, temperature=temperature)
        )
        loss = loss + teacher_weight * teacher_loss
    return loss, paired_descriptor_metrics(desc_a, desc_b)


def graph_matcher_correspondence_loss(
    model: pfm_model.PlanetaryFeatureMatcher,
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    points_a_xy: torch.Tensor,
    points_b_xy: torch.Tensor,
    *,
    metadata_mode: str = "full",
    no_match_points: int = 0,
    no_match_weight: float = 0.0,
    no_match_min_distance: float = 4.0,
    accept_weight: float = 0.0,
    accept_negative_topk: int = 8,
    raw_preservation_weight: float = 0.0,
    raw_preservation_margin: float = 1.0,
    raw_preservation_raw_margin: float = 0.05,
    hard_negative_dustbin_weight: float = 0.0,
    hard_negative_dustbin_topk: int = 8,
    hard_negative_dustbin_margin: float = 0.25,
    hard_negative_dustbin_spatial_min_distance: float = 0.0,
    semi_dense_no_match_points: int = 0,
    semi_dense_min_score: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if points_a_xy.size(0) == 0 or points_b_xy.size(0) == 0:
        return descriptors_a.new_tensor(0.0)
    count = min(points_a_xy.size(0), points_b_xy.size(0))
    points_a_xy = points_a_xy[:count]
    points_b_xy = points_b_xy[:count]
    positive_points_a_xy = points_a_xy
    positive_points_b_xy = points_b_xy
    desc_a = normalize_descriptor_batch(sample_descriptors(descriptors_a, points_a_xy))
    desc_b = normalize_descriptor_batch(sample_descriptors(descriptors_b, points_b_xy))
    if no_match_points > 0 and no_match_weight > 0.0:
        neg_a_points = sample_unmatched_feature_points(
            feature_height=descriptors_a.size(2),
            feature_width=descriptors_a.size(3),
            reference_points=points_a_xy,
            count=no_match_points,
            min_distance=no_match_min_distance,
            generator=generator,
            device=descriptors_a.device,
        )
        neg_b_points = sample_unmatched_feature_points(
            feature_height=descriptors_b.size(2),
            feature_width=descriptors_b.size(3),
            reference_points=points_b_xy,
            count=no_match_points,
            min_distance=no_match_min_distance,
            generator=generator,
            device=descriptors_b.device,
        )
        if neg_a_points.numel() > 0:
            desc_a = torch.cat([desc_a, normalize_descriptor_batch(sample_descriptors(descriptors_a, neg_a_points))], dim=0)
            points_a_xy = torch.cat([points_a_xy, neg_a_points.to(points_a_xy.device)], dim=0)
        if neg_b_points.numel() > 0:
            desc_b = torch.cat([desc_b, normalize_descriptor_batch(sample_descriptors(descriptors_b, neg_b_points))], dim=0)
            points_b_xy = torch.cat([points_b_xy, neg_b_points.to(points_b_xy.device)], dim=0)
    if semi_dense_no_match_points > 0 and no_match_weight > 0.0 and hasattr(model, "semi_dense_branch"):
        semi_dense = model.semi_dense_branch(
            descriptors_a,
            descriptors_b,
            max_candidates=semi_dense_no_match_points,
            min_score=semi_dense_min_score,
        )
        semi_a_points = semi_dense.keypoints_a.to(device=descriptors_a.device, dtype=points_a_xy.dtype)
        semi_b_points = semi_dense.keypoints_b.to(device=descriptors_b.device, dtype=points_b_xy.dtype)
        if semi_a_points.numel() > 0:
            keep = torch.ones(semi_a_points.size(0), dtype=torch.bool, device=semi_a_points.device)
            if no_match_min_distance > 0.0 and positive_points_a_xy.numel() > 0:
                keep &= torch.cdist(semi_a_points, positive_points_a_xy.to(semi_a_points.device)).amin(dim=1) >= float(
                    no_match_min_distance
                )
            if no_match_min_distance > 0.0 and positive_points_b_xy.numel() > 0:
                keep &= torch.cdist(semi_b_points, positive_points_b_xy.to(semi_b_points.device)).amin(dim=1) >= float(
                    no_match_min_distance
                )
            semi_a_points = semi_a_points[keep]
            semi_b_points = semi_b_points[keep]
        if semi_a_points.numel() > 0:
            desc_a = torch.cat(
                [desc_a, normalize_descriptor_batch(sample_descriptors(descriptors_a, semi_a_points))],
                dim=0,
            )
            desc_b = torch.cat(
                [desc_b, normalize_descriptor_batch(sample_descriptors(descriptors_b, semi_b_points))],
                dim=0,
            )
            points_a_xy = torch.cat([points_a_xy, semi_a_points.to(points_a_xy.device)], dim=0)
            points_b_xy = torch.cat([points_b_xy, semi_b_points.to(points_b_xy.device)], dim=0)
    meta_a = pfm_model.prepare_graph_keypoint_metadata(
        points_a_xy,
        meta_dim=model.config.graph_keypoint_meta_dim,
    ).to(desc_a.device)
    meta_b = pfm_model.prepare_graph_keypoint_metadata(
        points_b_xy,
        meta_dim=model.config.graph_keypoint_meta_dim,
    ).to(desc_b.device)
    meta_a = apply_graph_metadata_mode(meta_a, metadata_mode)
    meta_b = apply_graph_metadata_mode(meta_b, metadata_mode)
    output = model.graph_matcher(desc_a, meta_a, desc_b, meta_b, apply_candidate_mask=False)
    targets = torch.arange(count, dtype=torch.long, device=output.logits.device)
    row_loss = F.cross_entropy(output.logits[:count, :], targets)
    col_loss = F.cross_entropy(output.logits[:, :count].T, targets)
    loss = 0.5 * (row_loss + col_loss)
    total_a = desc_a.size(0)
    total_b = desc_b.size(0)
    if no_match_weight > 0.0 and (total_a > count or total_b > count):
        no_match_terms: list[torch.Tensor] = []
        if total_a > count:
            dustbin_col = torch.full((total_a - count,), total_b, dtype=torch.long, device=output.logits.device)
            no_match_terms.append(F.cross_entropy(output.logits[count:total_a, :], dustbin_col))
        if total_b > count:
            dustbin_row = torch.full((total_b - count,), total_a, dtype=torch.long, device=output.logits.device)
            no_match_terms.append(F.cross_entropy(output.logits[:, count:total_b].T, dustbin_row))
        if no_match_terms:
            loss = loss + float(no_match_weight) * torch.stack(no_match_terms).mean()
    if accept_weight > 0.0:
        loss = loss + float(accept_weight) * graph_matcher_acceptance_loss(
            output,
            desc_a,
            desc_b,
            positive_count=count,
            negative_topk=accept_negative_topk,
        )
    if raw_preservation_weight > 0.0:
        loss = loss + float(raw_preservation_weight) * graph_matcher_raw_preservation_loss(
            output.logits,
            desc_a[:count],
            desc_b[:count],
            target_margin=raw_preservation_margin,
            raw_margin_threshold=raw_preservation_raw_margin,
        )
    if hard_negative_dustbin_weight > 0.0:
        loss = loss + float(hard_negative_dustbin_weight) * graph_matcher_hard_negative_dustbin_loss(
            output.logits,
            desc_a[:count],
            desc_b[:count],
            positive_count=count,
            negative_topk=hard_negative_dustbin_topk,
            margin=hard_negative_dustbin_margin,
            points_b_xy=points_b_xy[:count],
            spatial_min_distance=hard_negative_dustbin_spatial_min_distance,
        )
    return loss


def graph_matcher_acceptance_loss(
    output: pfm_model.GraphMatcherOutput,
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    positive_count: int,
    negative_topk: int = 8,
) -> torch.Tensor:
    if output.accept_logits is None:
        return output.logits.new_zeros(())
    if positive_count <= 0:
        return output.logits.new_zeros(())
    accept_logits = output.accept_logits
    count = min(int(positive_count), accept_logits.size(0), accept_logits.size(1))
    if count <= 0:
        return output.logits.new_zeros(())
    terms: list[torch.Tensor] = []
    diag_logits = accept_logits[:count, :count].diagonal()
    terms.append(F.binary_cross_entropy_with_logits(diag_logits, torch.ones_like(diag_logits)))
    if count > 1 and negative_topk > 0:
        similarity = normalize_descriptor_batch(desc_a[:count]) @ normalize_descriptor_batch(desc_b[:count]).T
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
        masked_similarity = similarity.masked_fill(~off_diagonal, -float("inf"))
        k = min(int(negative_topk), count - 1)
        hard_negative_indices = masked_similarity.topk(k, dim=1).indices
        hard_negative_logits = accept_logits[:count, :count].gather(1, hard_negative_indices)
        terms.append(F.binary_cross_entropy_with_logits(hard_negative_logits, torch.zeros_like(hard_negative_logits)))
    if accept_logits.size(1) > count:
        no_match_ab = accept_logits[:count, count:]
        if no_match_ab.numel() > 0:
            terms.append(F.binary_cross_entropy_with_logits(no_match_ab, torch.zeros_like(no_match_ab)))
    if accept_logits.size(0) > count:
        no_match_ba = accept_logits[count:, :count]
        if no_match_ba.numel() > 0:
            terms.append(F.binary_cross_entropy_with_logits(no_match_ba, torch.zeros_like(no_match_ba)))
    return torch.stack(terms).mean()


def graph_matcher_raw_preservation_loss(
    logits: torch.Tensor,
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    target_margin: float = 1.0,
    raw_margin_threshold: float = 0.05,
) -> torch.Tensor:
    count = min(desc_a.size(0), desc_b.size(0), logits.size(0) - 1, logits.size(1) - 1)
    if count <= 1:
        return logits.new_zeros(())
    raw_similarity = normalize_descriptor_batch(desc_a[:count]) @ normalize_descriptor_batch(desc_b[:count]).T
    pair_logits = logits[:count, :count]
    diagonal_mask = torch.eye(count, dtype=torch.bool, device=pair_logits.device)
    raw_diag = raw_similarity.diagonal()
    raw_row_hard = raw_similarity.masked_fill(diagonal_mask, -float("inf")).max(dim=1).values
    raw_col_hard = raw_similarity.masked_fill(diagonal_mask, -float("inf")).max(dim=0).values
    confident = (raw_diag - raw_row_hard).ge(float(raw_margin_threshold)) & (
        raw_diag - raw_col_hard
    ).ge(float(raw_margin_threshold))
    if not bool(confident.any()):
        return logits.new_zeros(())
    logit_diag = pair_logits.diagonal()
    row_hard = pair_logits.masked_fill(diagonal_mask, -float("inf")).max(dim=1).values
    col_hard = pair_logits.masked_fill(diagonal_mask, -float("inf")).max(dim=0).values
    row_loss = (float(target_margin) - (logit_diag - row_hard)).clamp_min(0.0).pow(2)
    col_loss = (float(target_margin) - (logit_diag - col_hard)).clamp_min(0.0).pow(2)
    return 0.5 * (row_loss[confident].mean() + col_loss[confident].mean())


def graph_matcher_hard_negative_dustbin_loss(
    logits: torch.Tensor,
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    positive_count: int,
    negative_topk: int = 8,
    margin: float = 0.25,
    points_b_xy: torch.Tensor | None = None,
    spatial_min_distance: float = 0.0,
) -> torch.Tensor:
    count = min(int(positive_count), desc_a.size(0), desc_b.size(0), logits.size(0) - 1, logits.size(1) - 1)
    if count <= 1 or negative_topk <= 0:
        return logits.new_zeros(())
    raw_similarity = normalize_descriptor_batch(desc_a[:count]) @ normalize_descriptor_batch(desc_b[:count]).T
    diagonal = torch.eye(count, dtype=torch.bool, device=raw_similarity.device)
    masked_similarity = raw_similarity.masked_fill(diagonal, -float("inf"))
    if points_b_xy is not None and spatial_min_distance > 0.0:
        if points_b_xy.dim() != 2 or points_b_xy.size(1) != 2:
            raise ValueError("points_b_xy must have shape Nx2")
        if points_b_xy.size(0) < count:
            raise ValueError("points_b_xy must contain at least positive_count rows")
        target_points = points_b_xy[:count].to(device=raw_similarity.device, dtype=raw_similarity.dtype)
        target_distance = torch.cdist(target_points, target_points, p=2.0)
        far_enough = target_distance.ge(float(spatial_min_distance))
        masked_similarity = masked_similarity.masked_fill(~far_enough, -float("inf"))
        valid_rows = torch.isfinite(masked_similarity).any(dim=1)
        if not bool(valid_rows.any()):
            return logits.new_zeros(())
        row_indices = torch.nonzero(valid_rows, as_tuple=False).reshape(-1)
        masked_similarity = masked_similarity.index_select(0, row_indices)
    else:
        row_indices = torch.arange(count, dtype=torch.long, device=raw_similarity.device)
    k = min(int(negative_topk), count - 1)
    finite_per_row = torch.isfinite(masked_similarity).sum(dim=1)
    k = min(k, int(finite_per_row.max().item()))
    if k <= 0:
        return logits.new_zeros(())
    hard_indices = masked_similarity.topk(k, dim=1).indices
    pair_logits = logits[:count, :count]
    selected_pair_logits = pair_logits.index_select(0, row_indices)
    hard_logits = selected_pair_logits.gather(1, hard_indices)
    finite_hard = torch.isfinite(masked_similarity.gather(1, hard_indices))
    row_dustbin = logits[:count, -1].index_select(0, row_indices).unsqueeze(1).expand_as(hard_logits)
    col_dustbin = logits[-1, :count].index_select(0, hard_indices.reshape(-1)).view_as(hard_logits)
    dustbin_floor = torch.minimum(row_dustbin, col_dustbin)
    loss = (hard_logits - dustbin_floor + float(margin)).clamp_min(0.0).pow(2)
    if not bool(finite_hard.any()):
        return logits.new_zeros(())
    return loss[finite_hard].mean()


def hard_negative_margin_loss(desc_a: torch.Tensor, desc_b: torch.Tensor, *, margin: float = 0.2) -> torch.Tensor:
    if desc_a.dim() != 2 or desc_b.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if desc_a.shape != desc_b.shape:
        raise ValueError("descriptor tensors must have the same shape")
    if desc_a.size(0) <= 1:
        return desc_a.new_zeros(())
    desc_a = normalize_descriptor_batch(desc_a)
    desc_b = normalize_descriptor_batch(desc_b)
    similarity = desc_a @ desc_b.T
    positive = similarity.diagonal()
    off_diagonal = ~torch.eye(desc_a.size(0), device=desc_a.device, dtype=torch.bool)
    hardest_ab = similarity.masked_fill(~off_diagonal, -float("inf")).max(dim=1).values
    hardest_ba = similarity.masked_fill(~off_diagonal, -float("inf")).max(dim=0).values
    loss_ab = (hardest_ab - positive + margin).clamp_min(0.0).pow(2)
    loss_ba = (hardest_ba - positive + margin).clamp_min(0.0).pow(2)
    return 0.5 * (loss_ab.mean() + loss_ba.mean())


def teacher_guided_descriptor_loss(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    if student_descriptors.dim() != 2 or teacher_descriptors.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if student_descriptors.shape != teacher_descriptors.shape:
        raise ValueError("student and teacher descriptors must have the same shape")
    if student_descriptors.size(0) == 0:
        raise ValueError("descriptor batch must not be empty")
    student = normalize_descriptor_batch(student_descriptors)
    teacher = normalize_descriptor_batch(teacher_descriptors.detach())
    logits = student @ teacher.T / temperature
    target = torch.arange(student.size(0), device=student.device)
    return F.cross_entropy(logits, target)


def heatmap_point_loss(
    heatmap: torch.Tensor,
    points_xy: torch.Tensor,
    *,
    negative_weight: float = 0.01,
) -> torch.Tensor:
    if heatmap.dim() != 4 or heatmap.size(0) != 1 or heatmap.size(1) != 1:
        raise ValueError("heatmap must have shape 1x1xHxW")
    if points_xy.dim() != 2 or points_xy.size(1) != 2:
        raise ValueError("points_xy must have shape Nx2")
    if negative_weight < 0.0:
        raise ValueError("negative_weight must be nonnegative")
    if points_xy.numel() == 0:
        return heatmap.sum() * 0.0
    height, width = heatmap.shape[-2:]
    rounded = points_xy.round().to(device=heatmap.device, dtype=torch.long)
    x = rounded[:, 0].clamp(0, width - 1)
    y = rounded[:, 1].clamp(0, height - 1)
    scores = heatmap[0, 0, y, x].to(torch.float32).clamp(1.0e-6, 1.0 - 1.0e-6)
    positive_loss = -scores.log().mean()
    return positive_loss + float(negative_weight) * heatmap.to(torch.float32).mean()


def compute_descriptor_maps(model, pair: SyntheticPair) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        model.descriptor_map_single(pair.view_a.unsqueeze(0)),
        model.descriptor_map_single(pair.view_b.unsqueeze(0)),
    )


def learned_descriptor_and_heatmap_single(
    model: pfm_model.PlanetaryFeatureMatcher,
    image: torch.Tensor,
    *,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.dim() != 4:
        raise ValueError("image must have shape BxCxHxW")
    features = model.backbone(image)
    if hasattr(model, "dual_fpn"):
        p2_keypoint, p2_descriptor = model.dual_fpn(features)
        sparse = model.sparse_head(p2_keypoint, p2_descriptor)
    else:
        sparse = model.sparse_head(features[1])
    descriptors = sparse.descriptors
    if train_blended_descriptors:
        descriptors = model.fuse_descriptor_maps(descriptors, image, texture_blend_weight=texture_blend_weight)
    return descriptors, sparse.heatmap


def compute_student_teacher_descriptor_maps(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair: SyntheticPair,
    *,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    include_heatmaps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    student_a, heatmap_a = learned_descriptor_and_heatmap_single(
        model,
        pair.view_a.unsqueeze(0),
        train_blended_descriptors=train_blended_descriptors,
        texture_blend_weight=texture_blend_weight,
    )
    student_b, heatmap_b = learned_descriptor_and_heatmap_single(
        model,
        pair.view_b.unsqueeze(0),
        train_blended_descriptors=train_blended_descriptors,
        texture_blend_weight=texture_blend_weight,
    )
    with torch.no_grad():
        teacher_a = model.texture_descriptor_map_single(pair.view_a.unsqueeze(0))
        teacher_b = model.texture_descriptor_map_single(pair.view_b.unsqueeze(0))
    if not include_heatmaps:
        return student_a, student_b, teacher_a, teacher_b
    return student_a, student_b, teacher_a, teacher_b, heatmap_a, heatmap_b


def descriptor_parameters(
    model: pfm_model.PlanetaryFeatureMatcher,
    *,
    train_backbone: bool = False,
    train_dual_fpn: bool = False,
    train_descriptor_head: bool = True,
    train_sparse_context: bool = False,
    train_keypoint_head: bool = False,
    train_geometry_head: bool = False,
    train_texture_adapter: bool = False,
    train_descriptor_fusion: bool = False,
    train_quality_head: bool = False,
    train_graph_matcher: bool = False,
) -> list[torch.nn.Parameter]:
    selected: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        trainable = train_backbone and name.startswith("backbone.")
        trainable = trainable or (train_dual_fpn and name.startswith("dual_fpn."))
        trainable = trainable or (train_descriptor_head and (
            name.startswith("sparse_head.descriptor") or name.startswith("sparse_head.descriptors")
        ))
        trainable = trainable or (
            train_sparse_context
            and (
                name.startswith("sparse_head.context")
                or name.startswith("sparse_head.descriptor_context")
                or name.startswith("sparse_head.geometry_context")
            )
        )
        trainable = trainable or (
            train_keypoint_head
            and (
                name.startswith("sparse_head.heatmap")
                or name.startswith("sparse_head.keypoint_context")
                or name.startswith("sparse_head.keypoint_offsets")
            )
        )
        trainable = trainable or (
            train_geometry_head
            and (
                name.startswith("sparse_head.scale")
                or name.startswith("sparse_head.orientation")
                or name.startswith("sparse_head.affine")
                or name.startswith("sparse_head.geometry_context")
            )
        )
        trainable = trainable or (train_texture_adapter and name.startswith("texture_adapter."))
        trainable = trainable or (train_descriptor_fusion and name.startswith("descriptor_fusion."))
        trainable = trainable or (train_quality_head and name.startswith("quality_head."))
        trainable = trainable or (train_graph_matcher and name.startswith("graph_matcher."))
        if trainable:
            parameter.requires_grad_(True)
            selected.append(parameter)
        else:
            parameter.requires_grad_(False)
    return selected


def gradient_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def require_finite_scalar(value: torch.Tensor, *, name: str) -> None:
    if value.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"non-finite {name}")


def clip_and_measure_gradients(parameters: list[torch.nn.Parameter], *, max_grad_norm: float = 0.0) -> float:
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    grad_norm = gradient_l2_norm(parameters)
    if not math.isfinite(grad_norm):
        raise FloatingPointError("non-finite gradient norm")
    return grad_norm


def averaged_step_metrics(metric_rows: list[dict[str, float]], sampled_count: int) -> dict[str, float]:
    if not metric_rows:
        return {
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "mean_positive_rank": 0.0,
            "mean_positive_score": 0.0,
            "mean_negative_score": 0.0,
            "points": float(sampled_count),
        }
    return {
        "top1_accuracy": sum(row["top1_accuracy"] for row in metric_rows) / len(metric_rows),
        "top5_accuracy": sum(row["top5_accuracy"] for row in metric_rows) / len(metric_rows),
        "top10_accuracy": sum(row["top10_accuracy"] for row in metric_rows) / len(metric_rows),
        "mean_positive_rank": sum(row["mean_positive_rank"] for row in metric_rows) / len(metric_rows),
        "mean_positive_score": sum(row["mean_positive_score"] for row in metric_rows) / len(metric_rows),
        "mean_negative_score": sum(row["mean_negative_score"] for row in metric_rows) / len(metric_rows),
        "points": float(sampled_count),
    }


def skipped_step_metrics(
    loss: torch.Tensor,
    metric_rows: list[dict[str, float]],
    *,
    sampled_count: int,
) -> dict[str, float]:
    metrics = averaged_step_metrics(metric_rows, sampled_count)
    return {
        "loss": float(loss.detach().cpu()) if bool(torch.isfinite(loss.detach()).all()) else float("nan"),
        "grad_l2": 0.0,
        "skipped": 1.0,
        **metrics,
    }


def pose_metadata_for_pair(
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None,
    pair_path: Path,
) -> pose_pair_metadata.PosePairMetadata | None:
    return pose_pair_metadata.lookup_pose_metadata(pose_metadata, pair_path)


def pose_difficulty_loss_multiplier(
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None,
    pair_path: Path,
    *,
    strength: float,
) -> float:
    if strength <= 0.0:
        return 1.0
    metadata = pose_metadata_for_pair(pose_metadata, pair_path)
    if metadata is None:
        return 1.0
    score = min(1.0, max(0.0, float(metadata.difficulty_score)))
    return 1.0 + float(strength) * score


def record_pose_training_metrics(
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None,
    pair_path: Path,
    *,
    loss_multiplier: float,
    counts: dict[str, float],
) -> None:
    metadata = pose_metadata_for_pair(pose_metadata, pair_path)
    difficulty = metadata.difficulty if metadata is not None else "unknown"
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "unknown"
    counts[f"pose_{difficulty}_pairs"] += 1.0
    counts["pose_weight_sum"] += float(loss_multiplier)
    counts["pose_weight_pairs"] += 1.0


def train_step(
    model: pfm_model.PlanetaryFeatureMatcher,
    optimizer: torch.optim.Optimizer,
    pair_paths: list[Path],
    *,
    device: torch.device,
    batch_pairs: int,
    samples_per_pair: int,
    min_intensity: float,
    generator: torch.Generator,
    training_weak_texture_fraction: float = 0.0,
    temperature: float,
    teacher_weight: float,
    synthetic_loss_weight: float = 1.0,
    hard_pair_paths: list[Path] | None = None,
    hard_probability: float = 0.0,
    hard_negative_weight: float = 0.5,
    diversity_weight: float = 0.10,
    warp_hard_negative_weight: float = 0.0,
    warp_hard_negative_radius: float = 2.0,
    warp_hard_negative_margin: float = 0.2,
    warp_hard_negative_candidates: int = 4096,
    abstention_weight: float = 0.0,
    abstention_negative_radius: float = 2.0,
    abstention_max_false_score: float = 0.35,
    abstention_topk: int = 8,
    abstention_candidates: int = 4096,
    max_grad_norm: float = 0.0,
    skip_nonfinite_steps: bool = False,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    balanced_cache_sampling: bool = False,
    gradient_accumulation_steps: int = 1,
    pseudo_labels: dict[str, PseudoLabelMatches] | None = None,
    pseudo_label_weight: float = 0.0,
    pseudo_keypoint_weight: float = 0.0,
    pseudo_keypoint_negative_weight: float = 0.01,
    pseudo_label_max_points: int = 0,
    pseudo_label_pair_paths: list[Path] | None = None,
    pseudo_label_probability: float = 0.0,
    false_matches: dict[str, FalseMatchLabels] | None = None,
    false_match_weight: float = 0.0,
    false_match_max_points: int = 0,
    false_match_max_score: float = 0.25,
    false_match_pair_paths: list[Path] | None = None,
    false_match_probability: float = 0.0,
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None = None,
    pose_balanced_sampling: bool = False,
    pose_difficulty_loss_weight: float = 0.0,
    graph_matcher_loss_weight: float = 0.0,
    graph_matcher_metadata_mode: str = "full",
    graph_matcher_no_match_points: int = 0,
    graph_matcher_no_match_weight: float = 0.0,
    graph_matcher_no_match_min_distance: float = 4.0,
    graph_matcher_accept_weight: float = 0.0,
    graph_matcher_accept_negative_topk: int = 8,
    graph_matcher_raw_preservation_weight: float = 0.0,
    graph_matcher_raw_preservation_margin: float = 1.0,
    graph_matcher_raw_preservation_raw_margin: float = 0.05,
    graph_matcher_hard_negative_dustbin_weight: float = 0.0,
    graph_matcher_hard_negative_dustbin_topk: int = 8,
    graph_matcher_hard_negative_dustbin_margin: float = 0.25,
    graph_matcher_hard_negative_dustbin_spatial_min_distance: float = 0.0,
    graph_matcher_semi_dense_no_match_points: int = 0,
    graph_matcher_semi_dense_min_score: float = 0.0,
    training_spatial_bins: int = 0,
    training_crop_size: int = 0,
    training_max_image_size: int = 0,
    forced_pair_paths: list[Path] | None = None,
    prefetched_pairs: dict[Path, SyntheticPair] | None = None,
    pair_cache: PairArchiveCache | None = None,
) -> dict[str, float]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    optimizer.zero_grad(set_to_none=True)
    metric_rows: list[dict[str, float]] = []
    sampled_count = 0
    pseudo_label_points = 0
    pseudo_keypoint_points = 0
    pseudo_label_pairs = 0
    false_match_points = 0
    false_match_pairs = 0
    pose_counts = {
        "pose_easy_pairs": 0.0,
        "pose_medium_pairs": 0.0,
        "pose_hard_pairs": 0.0,
        "pose_unknown_pairs": 0.0,
        "pose_weight_sum": 0.0,
        "pose_weight_pairs": 0.0,
    }
    loss_values: list[float] = []
    valid_micro_batches = 0
    parameters = [group_param for group in optimizer.param_groups for group_param in group["params"]]
    try:
        for _ in range(gradient_accumulation_steps):
            if forced_pair_paths is not None:
                selected = forced_pair_paths
            else:
                selected = sample_training_pairs_with_pseudo_labels(
                    base_pair_paths=pair_paths,
                    hard_pair_paths=hard_pair_paths or [],
                    pseudo_label_pair_paths=pseudo_label_pair_paths or [],
                    batch_pairs=batch_pairs,
                    hard_probability=hard_probability,
                    pseudo_label_probability=pseudo_label_probability if pseudo_label_pair_paths else 0.0,
                    false_match_pair_paths=false_match_pair_paths or [],
                    false_match_probability=false_match_probability if false_match_pair_paths else 0.0,
                    rng=random,
                    balanced_cache_sampling=balanced_cache_sampling,
                    pose_metadata=pose_metadata,
                    pose_balanced_sampling=pose_balanced_sampling,
                )
            losses = []
            for pair_path in selected:
                pair_key = pair_path.resolve(strict=False)
                if prefetched_pairs is not None and pair_key in prefetched_pairs:
                    pair = move_pair_to_device(prefetched_pairs[pair_key], device=device)
                else:
                    pair = load_pair_for_training(pair_path, device=device, pair_cache=pair_cache)
                pair = crop_pair_for_training(pair, crop_size=training_crop_size, generator=generator)
                pair = resize_pair_for_training(pair, max_image_size=training_max_image_size)
                descriptor_maps = compute_student_teacher_descriptor_maps(
                    model,
                    pair,
                    train_blended_descriptors=train_blended_descriptors,
                    texture_blend_weight=texture_blend_weight,
                    include_heatmaps=True,
                )
                if len(descriptor_maps) == 4:
                    descriptors_a, descriptors_b, teacher_a, teacher_b = descriptor_maps
                    heatmap_a = None
                    heatmap_b = None
                elif len(descriptor_maps) == 6:
                    descriptors_a, descriptors_b, teacher_a, teacher_b, heatmap_a, heatmap_b = descriptor_maps
                else:
                    raise ValueError("compute_student_teacher_descriptor_maps returned an unsupported tuple length")
                points_a, points_b = sample_feature_correspondences(
                    pair,
                    feature_height=descriptors_a.size(2),
                    feature_width=descriptors_a.size(3),
                    count=samples_per_pair,
                    min_intensity=min_intensity,
                    weak_texture_fraction=training_weak_texture_fraction,
                    spatial_bins=training_spatial_bins,
                    generator=generator,
                )
                pair_losses: list[torch.Tensor] = []
                if points_a.size(0) > 0 and (synthetic_loss_weight > 0.0 or graph_matcher_loss_weight > 0.0):
                    pose_multiplier = pose_difficulty_loss_multiplier(
                        pose_metadata,
                        pair_path,
                        strength=pose_difficulty_loss_weight,
                    )
                    if synthetic_loss_weight > 0.0:
                        loss, metrics = descriptor_map_pair_loss(
                            descriptors_a,
                            descriptors_b,
                            points_a,
                            points_b,
                            temperature=temperature,
                            teacher_descriptors_a=teacher_a,
                            teacher_descriptors_b=teacher_b,
                            teacher_weight=teacher_weight,
                            hard_negative_weight=hard_negative_weight,
                            diversity_weight=diversity_weight,
                            warp_hard_negative_weight=warp_hard_negative_weight,
                            warp_hard_negative_radius=warp_hard_negative_radius,
                            warp_hard_negative_margin=warp_hard_negative_margin,
                            warp_hard_negative_candidates=warp_hard_negative_candidates,
                            abstention_weight=abstention_weight,
                            abstention_negative_radius=abstention_negative_radius,
                            abstention_max_false_score=abstention_max_false_score,
                            abstention_topk=abstention_topk,
                            abstention_candidates=abstention_candidates,
                        )
                        pair_losses.append(float(synthetic_loss_weight) * float(pose_multiplier) * loss)
                    else:
                        desc_a = normalize_descriptor_batch(sample_descriptors(descriptors_a, points_a))
                        desc_b = normalize_descriptor_batch(sample_descriptors(descriptors_b, points_b))
                        metrics = paired_descriptor_metrics(desc_a.detach(), desc_b.detach())
                    if graph_matcher_loss_weight > 0.0:
                        graph_loss = graph_matcher_correspondence_loss(
                            model,
                            descriptors_a,
                            descriptors_b,
                            points_a,
                            points_b,
                            metadata_mode=graph_matcher_metadata_mode,
                            no_match_points=graph_matcher_no_match_points,
                            no_match_weight=graph_matcher_no_match_weight,
                            no_match_min_distance=graph_matcher_no_match_min_distance,
                            accept_weight=graph_matcher_accept_weight,
                            accept_negative_topk=graph_matcher_accept_negative_topk,
                            raw_preservation_weight=graph_matcher_raw_preservation_weight,
                            raw_preservation_margin=graph_matcher_raw_preservation_margin,
                            raw_preservation_raw_margin=graph_matcher_raw_preservation_raw_margin,
                            hard_negative_dustbin_weight=graph_matcher_hard_negative_dustbin_weight,
                            hard_negative_dustbin_topk=graph_matcher_hard_negative_dustbin_topk,
                            hard_negative_dustbin_margin=graph_matcher_hard_negative_dustbin_margin,
                            hard_negative_dustbin_spatial_min_distance=graph_matcher_hard_negative_dustbin_spatial_min_distance,
                            semi_dense_no_match_points=graph_matcher_semi_dense_no_match_points,
                            semi_dense_min_score=graph_matcher_semi_dense_min_score,
                            generator=generator,
                        )
                        pair_losses.append(float(graph_matcher_loss_weight) * float(pose_multiplier) * graph_loss)
                    record_pose_training_metrics(
                        pose_metadata,
                        pair_path,
                        loss_multiplier=pose_multiplier,
                        counts=pose_counts,
                    )
                    metric_rows.append(metrics)
                    sampled_count += points_a.size(0)
                if pseudo_labels and (pseudo_label_weight > 0.0 or pseudo_keypoint_weight > 0.0):
                    pseudo_a, pseudo_b = pseudo_label_feature_correspondences(
                        pair_path,
                        pair,
                        pseudo_labels,
                        feature_height=descriptors_a.size(2),
                        feature_width=descriptors_a.size(3),
                        max_points=pseudo_label_max_points,
                        generator=generator,
                        min_intensity=min_intensity,
                    )
                    if pseudo_a.size(0) > 0:
                        if pseudo_label_weight > 0.0:
                            pseudo_loss, pseudo_metrics = descriptor_map_pair_loss(
                                descriptors_a,
                                descriptors_b,
                                pseudo_a,
                                pseudo_b,
                                temperature=temperature,
                                teacher_weight=0.0,
                                hard_negative_weight=hard_negative_weight,
                                diversity_weight=diversity_weight,
                                warp_hard_negative_weight=warp_hard_negative_weight,
                                warp_hard_negative_radius=warp_hard_negative_radius,
                                warp_hard_negative_margin=warp_hard_negative_margin,
                                warp_hard_negative_candidates=warp_hard_negative_candidates,
                                abstention_weight=abstention_weight,
                                abstention_negative_radius=abstention_negative_radius,
                                abstention_max_false_score=abstention_max_false_score,
                                abstention_topk=abstention_topk,
                                abstention_candidates=abstention_candidates,
                            )
                            pair_losses.append(float(pseudo_label_weight) * pseudo_loss)
                            metric_rows.append(pseudo_metrics)
                            sampled_count += pseudo_a.size(0)
                            pseudo_label_points += pseudo_a.size(0)
                            pseudo_label_pairs += 1
                        if pseudo_keypoint_weight > 0.0 and heatmap_a is not None and heatmap_b is not None:
                            keypoint_loss = heatmap_point_loss(
                                heatmap_a,
                                pseudo_a,
                                negative_weight=pseudo_keypoint_negative_weight,
                            ) + heatmap_point_loss(
                                heatmap_b,
                                pseudo_b,
                                negative_weight=pseudo_keypoint_negative_weight,
                            )
                            pair_losses.append(float(pseudo_keypoint_weight) * keypoint_loss)
                            pseudo_keypoint_points += pseudo_a.size(0) + pseudo_b.size(0)
                if false_matches and false_match_weight > 0.0:
                    false_a, false_b = false_match_feature_correspondences(
                        pair_path,
                        pair,
                        false_matches,
                        feature_height=descriptors_a.size(2),
                        feature_width=descriptors_a.size(3),
                        max_points=false_match_max_points,
                        generator=generator,
                    )
                    if false_a.size(0) > 0:
                        negative_loss = false_match_negative_loss(
                            descriptors_a,
                            descriptors_b,
                            false_a,
                            false_b,
                            max_false_score=false_match_max_score,
                        )
                        pair_losses.append(float(false_match_weight) * negative_loss)
                        false_match_points += false_a.size(0)
                        false_match_pairs += 1
                if pair_losses:
                    losses.append(torch.stack(pair_losses).sum())
            if not losses:
                continue
            micro_loss = torch.stack(losses).mean()
            loss_values.append(float(micro_loss.detach().cpu()) if bool(torch.isfinite(micro_loss.detach()).all()) else float("nan"))
            require_finite_scalar(micro_loss, name="training loss")
            (micro_loss / float(gradient_accumulation_steps)).backward()
            valid_micro_batches += 1
        if valid_micro_batches == 0:
            raise RuntimeError("no valid correspondences sampled")
        grad_norm = clip_and_measure_gradients(parameters, max_grad_norm=max_grad_norm)
    except FloatingPointError:
        optimizer.zero_grad(set_to_none=True)
        if not skip_nonfinite_steps:
            raise
        if not metric_rows:
            raise RuntimeError("no valid correspondences sampled")
        reported_loss = sum(loss_values) / float(len(loss_values)) if loss_values else float("nan")
        metrics = skipped_step_metrics(
            torch.tensor(reported_loss, device=device),
            metric_rows,
            sampled_count=sampled_count,
        )
        metrics["pseudo_label_points"] = float(pseudo_label_points)
        metrics["pseudo_keypoint_points"] = float(pseudo_keypoint_points)
        metrics["pseudo_label_pairs"] = float(pseudo_label_pairs)
        metrics["false_match_points"] = float(false_match_points)
        metrics["false_match_pairs"] = float(false_match_pairs)
        metrics["pose_easy_pairs"] = pose_counts["pose_easy_pairs"]
        metrics["pose_medium_pairs"] = pose_counts["pose_medium_pairs"]
        metrics["pose_hard_pairs"] = pose_counts["pose_hard_pairs"]
        metrics["pose_unknown_pairs"] = pose_counts["pose_unknown_pairs"]
        metrics["pose_mean_loss_weight"] = (
            pose_counts["pose_weight_sum"] / pose_counts["pose_weight_pairs"]
            if pose_counts["pose_weight_pairs"] > 0.0
            else 1.0
        )
        return metrics
    optimizer.step()
    return {
        "loss": sum(loss_values) / float(len(loss_values)),
        "grad_l2": grad_norm,
        "skipped": 0.0,
        **averaged_step_metrics(metric_rows, sampled_count),
        "pseudo_label_points": float(pseudo_label_points),
        "pseudo_keypoint_points": float(pseudo_keypoint_points),
        "pseudo_label_pairs": float(pseudo_label_pairs),
        "false_match_points": float(false_match_points),
        "false_match_pairs": float(false_match_pairs),
        "pose_easy_pairs": pose_counts["pose_easy_pairs"],
        "pose_medium_pairs": pose_counts["pose_medium_pairs"],
        "pose_hard_pairs": pose_counts["pose_hard_pairs"],
        "pose_unknown_pairs": pose_counts["pose_unknown_pairs"],
        "pose_mean_loss_weight": (
            pose_counts["pose_weight_sum"] / pose_counts["pose_weight_pairs"]
            if pose_counts["pose_weight_pairs"] > 0.0
            else 1.0
        ),
    }


def aggregate_descriptor_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "loss": 0.0,
            "top1_accuracy": 0.0,
            "mean_positive_score": 0.0,
            "mean_negative_score": 0.0,
            "points": 0.0,
        }
    total_points = sum(max(0.0, float(row.get("points", 0.0))) for row in rows)
    if total_points <= 0.0:
        total_points = float(len(rows))
        weights = [1.0 for _ in rows]
    else:
        weights = [max(0.0, float(row.get("points", 0.0))) for row in rows]
    result: dict[str, float] = {"points": sum(weights)}
    for key in (
        "loss",
        "top1_accuracy",
        "top5_accuracy",
        "top10_accuracy",
        "mean_positive_rank",
        "mean_positive_score",
        "mean_negative_score",
    ):
        if key not in rows[0]:
            continue
        result[key] = sum(float(row[key]) * weight for row, weight in zip(rows, weights)) / total_points
    return result


def split_train_eval_pairs(pair_paths: list[Path], *, eval_pairs: int) -> tuple[list[Path], list[Path]]:
    if eval_pairs <= 0 or len(pair_paths) <= 1:
        return pair_paths, []
    eval_count = min(eval_pairs, len(pair_paths) - 1)
    groups: dict[Path, list[Path]] = {}
    for path in pair_paths:
        cache_root = path.parent.parent if len(path.parents) >= 2 else Path(".")
        groups.setdefault(cache_root, []).append(path)
    if len(groups) == 1:
        return pair_paths[:-eval_count], pair_paths[-eval_count:]

    quotas = {cache_root: 0 for cache_root in groups}
    remaining = eval_count
    eligible = [cache_root for cache_root, paths in groups.items() if len(paths) > 1]
    while remaining > 0 and eligible:
        assigned_this_round = False
        for cache_root in eligible:
            capacity = len(groups[cache_root]) - 1
            if quotas[cache_root] >= capacity:
                continue
            quotas[cache_root] += 1
            remaining -= 1
            assigned_this_round = True
            if remaining == 0:
                break
        if not assigned_this_round:
            break

    train_paths: list[Path] = []
    eval_paths: list[Path] = []
    for cache_root, paths in groups.items():
        quota = quotas[cache_root]
        if quota <= 0:
            train_paths.extend(paths)
            continue
        train_paths.extend(paths[:-quota])
        eval_paths.extend(paths[-quota:])
    return train_paths, eval_paths


def resolve_training_and_eval_pair_paths(
    cache_dirs: list[Path],
    validation_cache_dirs: list[Path],
    *,
    limit_pairs: int,
    eval_pairs: int,
    exclude_self_pairs: bool = False,
) -> tuple[list[Path], list[Path]]:
    train_paths = discover_pair_archives(
        cache_dirs,
        limit_pairs=limit_pairs,
        exclude_self_pairs=exclude_self_pairs,
    )
    if validation_cache_dirs:
        eval_limit = eval_pairs if eval_pairs > 0 else 0
        eval_paths = discover_pair_archives(
            validation_cache_dirs,
            limit_pairs=eval_limit,
            exclude_self_pairs=exclude_self_pairs,
        )
        if eval_pairs > 0:
            eval_paths = eval_paths[:eval_pairs]
        return train_paths, eval_paths
    return split_train_eval_pairs(train_paths, eval_pairs=eval_pairs)


def filter_pair_paths_by_pose_overlap(
    pair_paths: list[Path],
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None,
    *,
    min_overlap: float,
) -> list[Path]:
    if min_overlap <= 0.0 or not pose_metadata:
        return pair_paths
    filtered: list[Path] = []
    for path in pair_paths:
        metadata = pose_metadata_for_pair(pose_metadata, path)
        if metadata is None or metadata.overlap_fraction >= min_overlap:
            filtered.append(path)
    return filtered


def repeat_hard_training_pairs(
    pair_paths: list[Path],
    summaries: list[Path],
    *,
    limit: int,
    min_matches: int,
    max_precision: float,
    repeat: int,
) -> tuple[list[Path], list[Path]]:
    selected = select_hard_training_pairs(
        pair_paths,
        summaries,
        limit=limit,
        min_matches=min_matches,
        max_precision=max_precision,
    )
    if repeat <= 0:
        return pair_paths, selected
    return pair_paths + selected * repeat, selected


def select_hard_training_pairs(
    pair_paths: list[Path],
    summaries: list[Path],
    *,
    limit: int,
    min_matches: int,
    max_precision: float,
) -> list[Path]:
    if not summaries:
        return []
    by_text = {path.as_posix(): path for path in pair_paths}
    by_name = {path.name: path for path in pair_paths}
    selected: list[Path] = []
    seen: set[str] = set()
    for summary in summaries:
        for hard_pair in hard_pair_mining.read_and_select(
            summary,
            limit=limit,
            min_matches=min_matches,
            max_precision=max_precision,
        ):
            path = by_text.get(Path(hard_pair.pair_pt).as_posix()) or by_name.get(Path(hard_pair.pair_pt).name)
            if path is None:
                continue
            key = path.as_posix()
            if key in seen:
                continue
            selected.append(path)
            seen.add(key)
    return selected


def select_hard_training_pairs_by_glob(pair_paths: list[Path], patterns: list[str]) -> list[Path]:
    if not patterns:
        return []
    selected: list[Path] = []
    for pair_path in pair_paths:
        text = pair_path.as_posix()
        if any(fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(pair_path.name, pattern) for pattern in patterns):
            selected.append(pair_path)
    return sorted(dict.fromkeys(selected))


def hard_pair_probability(step: int, *, max_probability: float, warmup_steps: int) -> float:
    if max_probability <= 0.0 or step <= 0:
        return 0.0
    capped = min(1.0, max(0.0, float(max_probability)))
    if warmup_steps <= 0:
        return capped
    progress = min(1.0, float(step) / float(warmup_steps))
    return capped * progress


def scheduled_value(step: int, *, start: float, final: float, schedule_steps: int) -> float:
    if schedule_steps <= 0:
        return float(final)
    progress = min(1.0, max(0.0, float(step) / float(schedule_steps)))
    return float(start) + (float(final) - float(start)) * progress


def _cache_root_for_pair(path: Path) -> Path:
    return path.parent.parent if len(path.parents) >= 2 else Path(".")


def sample_pose_balanced_training_pairs(
    pair_paths: list[Path],
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None,
    *,
    batch_pairs: int,
    rng,
) -> list[Path]:
    if batch_pairs <= 0:
        raise ValueError("batch_pairs must be positive")
    if not pair_paths:
        raise ValueError("pair_paths must not be empty")
    buckets: dict[str, list[Path]] = {"easy": [], "medium": [], "hard": [], "unknown": []}
    for path in pair_paths:
        metadata = pose_metadata_for_pair(pose_metadata, path)
        difficulty = metadata.difficulty if metadata is not None else "unknown"
        if difficulty not in buckets:
            difficulty = "unknown"
        buckets[difficulty].append(path)
    for paths in buckets.values():
        rng.shuffle(paths)
    selected: list[Path] = []
    selected_set: set[Path] = set()
    target_count = min(batch_pairs, len(pair_paths))
    while len(selected) < target_count:
        added = False
        order = ["easy", "medium", "hard", "unknown"]
        rng.shuffle(order)
        for difficulty in order:
            while buckets[difficulty] and buckets[difficulty][0] in selected_set:
                buckets[difficulty].pop(0)
            if not buckets[difficulty]:
                continue
            path = buckets[difficulty].pop(0)
            selected.append(path)
            selected_set.add(path)
            added = True
            if len(selected) >= target_count:
                break
        if not added:
            break
    if len(selected) < target_count:
        remaining = [path for path in pair_paths if path not in selected_set]
        selected.extend(rng.sample(remaining, k=min(target_count - len(selected), len(remaining))))
    rng.shuffle(selected)
    return selected


def sample_cache_balanced_training_pairs(pair_paths: list[Path], *, batch_pairs: int, rng) -> list[Path]:
    if batch_pairs <= 0:
        raise ValueError("batch_pairs must be positive")
    if not pair_paths:
        raise ValueError("pair_paths must not be empty")
    groups: dict[Path, list[Path]] = {}
    for path in pair_paths:
        groups.setdefault(_cache_root_for_pair(path), []).append(path)

    roots = list(groups)
    rng.shuffle(roots)
    selected: list[Path] = []
    selected_set: set[Path] = set()
    target_count = min(batch_pairs, len(pair_paths))
    while roots and len(selected) < target_count:
        for root in roots:
            candidates = [path for path in groups[root] if path not in selected_set]
            if not candidates:
                continue
            choice = rng.choice(candidates)
            selected.append(choice)
            selected_set.add(choice)
            if len(selected) >= target_count:
                break
        roots = [root for root in roots if any(path not in selected_set for path in groups[root])]
        rng.shuffle(roots)
    return selected


def sample_curriculum_training_pairs(
    base_pair_paths: list[Path],
    hard_pair_paths: list[Path],
    *,
    batch_pairs: int,
    hard_probability: float,
    rng,
    balanced_cache_sampling: bool = False,
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None = None,
    pose_balanced_sampling: bool = False,
) -> list[Path]:
    if batch_pairs <= 0:
        raise ValueError("batch_pairs must be positive")
    if not base_pair_paths and not hard_pair_paths:
        raise ValueError("at least one training pair path is required")
    probability = min(1.0, max(0.0, float(hard_probability)))
    hard_count = 0
    if hard_pair_paths and probability > 0.0:
        expected = float(batch_pairs) * probability
        hard_count = min(batch_pairs, int(math.floor(expected)))
        if hard_count < batch_pairs and rng.random() < expected - float(hard_count):
            hard_count += 1
        hard_count = min(hard_count, len(hard_pair_paths))
    base_count = max(0, batch_pairs - hard_count)

    selected: list[Path] = []
    if base_count > 0 and base_pair_paths:
        if pose_balanced_sampling and pose_metadata:
            selected.extend(
                sample_pose_balanced_training_pairs(
                    base_pair_paths,
                    pose_metadata,
                    batch_pairs=base_count,
                    rng=rng,
                )
            )
        elif balanced_cache_sampling:
            selected.extend(sample_cache_balanced_training_pairs(base_pair_paths, batch_pairs=base_count, rng=rng))
        else:
            selected.extend(rng.sample(base_pair_paths, k=min(base_count, len(base_pair_paths))))
    if hard_count > 0:
        selected.extend(rng.sample(hard_pair_paths, k=hard_count))
    if len(selected) < min(batch_pairs, len(base_pair_paths) + len(hard_pair_paths)):
        used = set(selected)
        remaining = [path for path in base_pair_paths + hard_pair_paths if path not in used]
        fill_count = min(batch_pairs - len(selected), len(remaining))
        selected.extend(rng.sample(remaining, k=fill_count))
    rng.shuffle(selected)
    return selected


def sample_training_pairs_with_pseudo_labels(
    base_pair_paths: list[Path],
    hard_pair_paths: list[Path],
    pseudo_label_pair_paths: list[Path],
    *,
    batch_pairs: int,
    hard_probability: float,
    pseudo_label_probability: float,
    false_match_pair_paths: list[Path] | None = None,
    false_match_probability: float = 0.0,
    rng,
    balanced_cache_sampling: bool = False,
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None = None,
    pose_balanced_sampling: bool = False,
) -> list[Path]:
    if batch_pairs <= 0:
        raise ValueError("batch_pairs must be positive")
    false_match_pair_paths = list(false_match_pair_paths or [])
    pseudo_probability = min(1.0, max(0.0, float(pseudo_label_probability)))
    false_probability = min(1.0, max(0.0, float(false_match_probability)))
    selected: list[Path] = []
    used: set[Path] = set()

    def stochastic_count(probability: float, limit: int) -> int:
        if probability <= 0.0 or limit <= 0:
            return 0
        expected = float(batch_pairs) * probability
        count = min(batch_pairs, int(math.floor(expected)))
        if count < batch_pairs and rng.random() < expected - float(count):
            count += 1
        return min(count, limit)

    pseudo_active = bool(pseudo_label_pair_paths) and pseudo_probability > 0.0
    false_active = bool(false_match_pair_paths) and false_probability > 0.0
    supervised_target = 0
    if pseudo_active and false_active:
        supervised_probability = min(1.0, pseudo_probability + false_probability)
        supervised_target = stochastic_count(
            supervised_probability,
            min(batch_pairs, len(set(pseudo_label_pair_paths).union(false_match_pair_paths))),
        )
        if supervised_target <= 0:
            pseudo_count = 0
            false_count = 0
        elif supervised_target == 1:
            pseudo_count = 1 if pseudo_probability >= false_probability else 0
            false_count = 1 - pseudo_count
        else:
            total_probability = pseudo_probability + false_probability
            pseudo_count = int(round(float(supervised_target) * pseudo_probability / total_probability))
            pseudo_count = max(1, min(supervised_target - 1, pseudo_count))
            false_count = supervised_target - pseudo_count
        pseudo_count = min(pseudo_count, len(pseudo_label_pair_paths))
        false_count = min(false_count, len(false_match_pair_paths))
    else:
        pseudo_count = stochastic_count(pseudo_probability, len(pseudo_label_pair_paths)) if pseudo_active else 0
        false_count = stochastic_count(false_probability, len(false_match_pair_paths)) if false_active else 0
        supervised_target = pseudo_count + false_count

    def extend_unique(pool: list[Path], count: int) -> int:
        if count <= 0:
            return 0
        available = [path for path in pool if path not in used]
        take = min(count, len(available), batch_pairs - len(selected))
        if take <= 0:
            return 0
        chosen = rng.sample(available, k=take)
        selected.extend(chosen)
        used.update(chosen)
        return take

    extend_unique(list(pseudo_label_pair_paths), pseudo_count)
    extend_unique(false_match_pair_paths, false_count)
    if pseudo_active and false_active and len(selected) < supervised_target:
        supervised_remaining = list(dict.fromkeys(list(pseudo_label_pair_paths) + false_match_pair_paths))
        extend_unique(supervised_remaining, supervised_target - len(selected))

    remaining_count = batch_pairs - len(selected)
    if remaining_count > 0:
        remaining_base = [path for path in base_pair_paths if path not in used]
        remaining_hard = [path for path in hard_pair_paths if path not in used]
        if remaining_base or remaining_hard:
            selected.extend(
                sample_curriculum_training_pairs(
                    remaining_base,
                    remaining_hard,
                    batch_pairs=remaining_count,
                    hard_probability=hard_probability,
                    rng=rng,
                    balanced_cache_sampling=balanced_cache_sampling,
                    pose_metadata=pose_metadata,
                    pose_balanced_sampling=pose_balanced_sampling,
                )
            )
    rng.shuffle(selected)
    return selected


def make_torch_generator(device: torch.device, *, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


class EpochShuffleSampler:
    def __init__(self, pair_paths: list[Path], *, batch_pairs: int, seed: int) -> None:
        if batch_pairs <= 0:
            raise ValueError("batch_pairs must be positive")
        if not pair_paths:
            raise ValueError("pair_paths must not be empty")
        self.pair_paths = list(pair_paths)
        self.batch_pairs = int(batch_pairs)
        self.rng = random.Random(seed)
        self.order: list[Path] = []
        self.epoch_index = -1

    def batch_for_step(self, step: int) -> list[Path]:
        zero_based = max(0, int(step) - 1)
        absolute_start = zero_based * self.batch_pairs
        epoch_index = absolute_start // len(self.pair_paths)
        offset = absolute_start % len(self.pair_paths)
        if epoch_index != self.epoch_index:
            self.order = list(self.pair_paths)
            self.rng.shuffle(self.order)
            self.epoch_index = epoch_index
        batch = self.order[offset : offset + self.batch_pairs]
        if len(batch) < self.batch_pairs:
            next_order = list(self.pair_paths)
            self.rng.shuffle(next_order)
            batch.extend(next_order[: self.batch_pairs - len(batch)])
        return batch


@torch.no_grad()
def evaluate_descriptor_retrieval(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair_paths: list[Path],
    *,
    device: torch.device,
    samples_per_pair: int,
    min_intensity: float,
    generator: torch.Generator,
    temperature: float = 0.07,
    training_crop_size: int = 0,
    training_max_image_size: int = 0,
    pair_cache: PairArchiveCache | None = None,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for pair_path in pair_paths:
        pair = load_pair_for_training(pair_path, device=device, pair_cache=pair_cache)
        pair = crop_pair_for_training(pair, crop_size=training_crop_size, generator=generator)
        pair = resize_pair_for_training(pair, max_image_size=training_max_image_size)
        descriptors_a, descriptors_b = compute_descriptor_maps(model, pair)
        points_a, points_b = sample_feature_correspondences(
            pair,
            feature_height=descriptors_a.size(2),
            feature_width=descriptors_a.size(3),
            count=samples_per_pair,
            min_intensity=min_intensity,
            generator=generator,
        )
        if points_a.size(0) == 0:
            continue
        loss, metrics = descriptor_map_pair_loss(
            descriptors_a,
            descriptors_b,
            points_a,
            points_b,
            temperature=temperature,
        )
        rows.append({"loss": float(loss.detach().cpu()), "points": float(points_a.size(0)), **metrics})
    return aggregate_descriptor_metrics(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the current PFM model in PyTorch from warp correspondences")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--init-pytorch-state", type=Path, default=None)
    parser.add_argument("--init-random", action="store_true")
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--validation-cache-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pytorch_pfm_finetune"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--eval-pairs", type=int, default=0)
    parser.add_argument("--exclude-self-pairs", action="store_true")
    parser.add_argument("--batch-pairs", type=int, default=2)
    parser.add_argument("--samples-per-pair", type=int, default=256)
    parser.add_argument("--training-weak-texture-fraction", type=float, default=0.0)
    parser.add_argument("--training-spatial-bins", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.10)
    parser.add_argument("--teacher-final-weight", type=float, default=None)
    parser.add_argument("--hard-negative-final-weight", type=float, default=None)
    parser.add_argument("--diversity-final-weight", type=float, default=None)
    parser.add_argument("--loss-schedule-steps", type=int, default=0)
    parser.add_argument("--warp-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--warp-hard-negative-radius", type=float, default=2.0)
    parser.add_argument("--warp-hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--warp-hard-negative-candidates", type=int, default=4096)
    parser.add_argument("--abstention-weight", type=float, default=0.0)
    parser.add_argument("--abstention-negative-radius", type=float, default=2.0)
    parser.add_argument("--abstention-max-false-score", type=float, default=0.35)
    parser.add_argument("--abstention-topk", type=int, default=8)
    parser.add_argument("--abstention-candidates", type=int, default=4096)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--skip-nonfinite-steps", action="store_true")
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--hard-summary", action="append", type=Path, default=[])
    parser.add_argument("--hard-pair-glob", action="append", default=[])
    parser.add_argument("--hard-limit", type=int, default=64)
    parser.add_argument("--hard-min-matches", type=int, default=4)
    parser.add_argument("--hard-max-precision", type=float, default=0.9)
    parser.add_argument("--hard-repeat", type=int, default=3)
    parser.add_argument("--hard-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--hard-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--balanced-cache-sampling", action="store_true")
    parser.add_argument("--epoch-shuffle-sampling", action="store_true")
    parser.add_argument("--pair-cache-size", type=int, default=0)
    parser.add_argument(
        "--memory-cache-items",
        type=int,
        default=None,
        help="Alias for --pair-cache-size; number of CPU pair archives kept in the LRU memory pool.",
    )
    parser.add_argument("--prefetch-batches", type=int, default=0)
    parser.add_argument("--prefetch-workers", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--pose-metadata-root", action="append", type=Path, default=[])
    parser.add_argument("--pose-balanced-sampling", action="store_true")
    parser.add_argument("--pose-min-overlap", type=float, default=0.0)
    parser.add_argument("--pose-difficulty-loss-weight", type=float, default=0.0)
    parser.add_argument("--training-crop-size", type=int, default=0)
    parser.add_argument("--training-max-image-size", type=int, default=0)
    parser.add_argument("--pseudo-label-csv", action="append", type=Path, default=[])
    parser.add_argument("--pseudo-label-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-keypoint-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-keypoint-negative-weight", type=float, default=0.01)
    parser.add_argument("--pseudo-label-max-points", type=int, default=128)
    parser.add_argument("--pseudo-label-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--pseudo-label-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--false-match-csv", action="append", type=Path, default=[])
    parser.add_argument("--false-match-weight", type=float, default=0.0)
    parser.add_argument("--false-match-max-points", type=int, default=128)
    parser.add_argument("--false-match-max-score", type=float, default=0.25)
    parser.add_argument("--false-match-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--false-match-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--train-dual-fpn", action="store_true")
    parser.add_argument("--train-sparse-context", action="store_true")
    parser.add_argument("--train-geometry-head", action="store_true")
    parser.add_argument("--train-blended-descriptors", action="store_true")
    parser.add_argument("--train-texture-adapter", action="store_true")
    parser.add_argument("--train-descriptor-fusion", action="store_true")
    parser.add_argument("--train-quality-head", action="store_true")
    parser.add_argument("--train-graph-matcher", action="store_true")
    parser.add_argument("--graph-matcher-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--graph-matcher-metadata-mode",
        choices=["full", "descriptor_only", "no_xy", "no_geometry", "no_quality"],
        default="full",
    )
    parser.add_argument("--graph-matcher-no-match-points", type=int, default=0)
    parser.add_argument("--graph-matcher-no-match-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-no-match-min-distance", type=float, default=4.0)
    parser.add_argument("--graph-matcher-accept-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-accept-negative-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-raw-preservation-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-raw-preservation-margin", type=float, default=1.0)
    parser.add_argument("--graph-matcher-raw-preservation-raw-margin", type=float, default=0.05)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-spatial-min-distance", type=float, default=0.0)
    parser.add_argument("--graph-matcher-semi-dense-no-match-points", type=int, default=0)
    parser.add_argument("--graph-matcher-semi-dense-min-score", type=float, default=0.0)
    parser.add_argument("--freeze-descriptor-head", action="store_true")
    parser.add_argument("--training-texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--generate-training-report", action="store_true")
    parser.add_argument("--report-output-dir", type=Path, default=None)
    parser.add_argument("--report-sample-count", type=int, default=16)
    parser.add_argument("--report-max-keypoints", type=int, default=2048)
    parser.add_argument("--report-max-matches", type=int, default=512)
    parser.add_argument("--report-draw-matches", type=int, default=160)
    parser.add_argument("--report-min-margin", type=float, default=0.0)
    parser.add_argument("--report-matcher-mode", choices=["raw_descriptor", "graph_matcher", "both"], default="raw_descriptor")
    parser.add_argument("--report-texture-keypoint-fraction", type=float, default=1.0)
    parser.add_argument("--report-weak-texture-keypoint-fraction", type=float, default=0.0)
    parser.add_argument("--report-keypoint-spatial-bins", type=int, default=8)
    parser.add_argument("--report-keypoint-cell-cap", type=int, default=0)
    parser.add_argument("--report-coverage-bins", type=int, default=8)
    parser.add_argument("--report-required-sample-glob", action="append", default=[])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.init_random and (args.checkpoint is not None or args.init_pytorch_state is not None):
        parser.error("--init-random cannot be combined with --checkpoint or --init-pytorch-state")
    if args.checkpoint is None and args.init_pytorch_state is None and not args.init_random:
        parser.error("one of --checkpoint, --init-pytorch-state, or --init-random is required")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.epochs < 0:
        parser.error("--epochs must be nonnegative")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.training_weak_texture_fraction < 0.0 or args.training_weak_texture_fraction > 1.0:
        parser.error("--training-weak-texture-fraction must be in [0, 1]")
    if args.training_spatial_bins < 0:
        parser.error("--training-spatial-bins must be nonnegative")
    if args.memory_cache_items is not None:
        if args.memory_cache_items < 0:
            parser.error("--memory-cache-items must be nonnegative")
        if args.pair_cache_size != 0 and args.pair_cache_size != args.memory_cache_items:
            parser.error("--pair-cache-size and --memory-cache-items disagree")
        args.pair_cache_size = args.memory_cache_items
    if args.pair_cache_size < 0:
        parser.error("--pair-cache-size must be nonnegative")
    if args.prefetch_batches < 0:
        parser.error("--prefetch-batches must be nonnegative")
    if args.prefetch_workers <= 0:
        parser.error("--prefetch-workers must be positive")
    if args.pose_min_overlap < 0.0 or args.pose_min_overlap > 1.0:
        parser.error("--pose-min-overlap must be in [0, 1]")
    if args.pose_difficulty_loss_weight < 0.0:
        parser.error("--pose-difficulty-loss-weight must be nonnegative")
    if args.training_max_image_size < 0:
        parser.error("--training-max-image-size must be nonnegative")
    if args.training_crop_size < 0:
        parser.error("--training-crop-size must be nonnegative")
    if args.pseudo_label_weight < 0.0:
        parser.error("--pseudo-label-weight must be nonnegative")
    if args.pseudo_keypoint_weight < 0.0:
        parser.error("--pseudo-keypoint-weight must be nonnegative")
    if args.pseudo_keypoint_negative_weight < 0.0:
        parser.error("--pseudo-keypoint-negative-weight must be nonnegative")
    if args.pseudo_label_max_points < 0:
        parser.error("--pseudo-label-max-points must be nonnegative")
    if args.pseudo_label_curriculum_max_probability < 0.0:
        parser.error("--pseudo-label-curriculum-max-probability must be nonnegative")
    if args.synthetic_loss_weight < 0.0:
        parser.error("--synthetic-loss-weight must be nonnegative")
    if args.false_match_weight < 0.0:
        parser.error("--false-match-weight must be nonnegative")
    if args.false_match_max_points < 0:
        parser.error("--false-match-max-points must be nonnegative")
    if args.false_match_max_score < -1.0 or args.false_match_max_score > 1.0:
        parser.error("--false-match-max-score must be in [-1, 1]")
    if args.false_match_curriculum_max_probability < 0.0:
        parser.error("--false-match-curriculum-max-probability must be nonnegative")
    if args.graph_matcher_loss_weight < 0.0:
        parser.error("--graph-matcher-loss-weight must be nonnegative")
    if args.graph_matcher_no_match_points < 0:
        parser.error("--graph-matcher-no-match-points must be nonnegative")
    if args.graph_matcher_no_match_weight < 0.0:
        parser.error("--graph-matcher-no-match-weight must be nonnegative")
    if args.graph_matcher_no_match_min_distance < 0.0:
        parser.error("--graph-matcher-no-match-min-distance must be nonnegative")
    if args.graph_matcher_accept_weight < 0.0:
        parser.error("--graph-matcher-accept-weight must be nonnegative")
    if args.graph_matcher_accept_negative_topk < 0:
        parser.error("--graph-matcher-accept-negative-topk must be nonnegative")
    if args.graph_matcher_raw_preservation_weight < 0.0:
        parser.error("--graph-matcher-raw-preservation-weight must be nonnegative")
    if args.graph_matcher_raw_preservation_margin < 0.0:
        parser.error("--graph-matcher-raw-preservation-margin must be nonnegative")
    if args.graph_matcher_raw_preservation_raw_margin < 0.0:
        parser.error("--graph-matcher-raw-preservation-raw-margin must be nonnegative")
    if args.graph_matcher_hard_negative_dustbin_weight < 0.0:
        parser.error("--graph-matcher-hard-negative-dustbin-weight must be nonnegative")
    if args.graph_matcher_hard_negative_dustbin_topk < 0:
        parser.error("--graph-matcher-hard-negative-dustbin-topk must be nonnegative")
    if args.graph_matcher_hard_negative_dustbin_margin < 0.0:
        parser.error("--graph-matcher-hard-negative-dustbin-margin must be nonnegative")
    if args.graph_matcher_hard_negative_dustbin_spatial_min_distance < 0.0:
        parser.error("--graph-matcher-hard-negative-dustbin-spatial-min-distance must be nonnegative")
    if args.graph_matcher_semi_dense_no_match_points < 0:
        parser.error("--graph-matcher-semi-dense-no-match-points must be nonnegative")
    if args.graph_matcher_semi_dense_min_score < 0.0:
        parser.error("--graph-matcher-semi-dense-min-score must be nonnegative")
    if args.abstention_weight < 0.0:
        parser.error("--abstention-weight must be nonnegative")
    if args.abstention_negative_radius < 0.0:
        parser.error("--abstention-negative-radius must be nonnegative")
    if args.abstention_max_false_score < -1.0 or args.abstention_max_false_score > 1.0:
        parser.error("--abstention-max-false-score must be in [-1, 1]")
    if args.abstention_topk <= 0:
        parser.error("--abstention-topk must be positive")
    if args.abstention_candidates < 0:
        parser.error("--abstention-candidates must be nonnegative")
    if args.report_sample_count < 0:
        parser.error("--report-sample-count must be nonnegative")
    if args.report_max_keypoints <= 0:
        parser.error("--report-max-keypoints must be positive")
    if args.report_max_matches <= 0:
        parser.error("--report-max-matches must be positive")
    if args.report_draw_matches <= 0:
        parser.error("--report-draw-matches must be positive")
    if args.report_min_margin < 0.0:
        parser.error("--report-min-margin must be nonnegative")
    if args.report_texture_keypoint_fraction < 0.0 or args.report_texture_keypoint_fraction > 1.0:
        parser.error("--report-texture-keypoint-fraction must be in [0, 1]")
    if args.report_weak_texture_keypoint_fraction < 0.0 or args.report_weak_texture_keypoint_fraction > 1.0:
        parser.error("--report-weak-texture-keypoint-fraction must be in [0, 1]")
    if args.report_texture_keypoint_fraction + args.report_weak_texture_keypoint_fraction > 1.0:
        parser.error("--report-texture-keypoint-fraction + --report-weak-texture-keypoint-fraction must be <= 1")
    if args.report_keypoint_spatial_bins < 0:
        parser.error("--report-keypoint-spatial-bins must be nonnegative")
    if args.report_keypoint_cell_cap < 0:
        parser.error("--report-keypoint-cell-cap must be nonnegative")
    if args.report_coverage_bins <= 0:
        parser.error("--report-coverage-bins must be positive")
    return args


def load_training_model(args: argparse.Namespace) -> tuple[pfm_model.PlanetaryFeatureMatcher, pfm_model.CheckpointConfig]:
    if getattr(args, "init_random", False):
        model = pfm_model.PlanetaryFeatureMatcher().to(args.device)
        model.train()
        return model, model.config
    if getattr(args, "init_pytorch_state", None) is not None:
        return pfm_model.load_pytorch_state(args.init_pytorch_state, device=args.device)
    if args.checkpoint is None:
        raise ValueError("checkpoint is required unless init_pytorch_state or init_random is set")
    return pfm_model.load_libtorch_checkpoint(args.checkpoint, device=args.device)


def run_training_report(args: argparse.Namespace, *, pytorch_state: Path) -> None:
    if not args.validation_cache_dir:
        print("training_report_skipped=no_validation_cache_dir", flush=True)
        return
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "training_visual_report.py"
    output_dir = args.report_output_dir or args.output_dir / "visual_report"
    matcher_modes = ["raw_descriptor", "graph_matcher"] if args.report_matcher_mode == "both" else [args.report_matcher_mode]
    for matcher_mode in matcher_modes:
        mode_output_dir = output_dir / matcher_mode if args.report_matcher_mode == "both" else output_dir
        command = [
            sys.executable,
            str(script_path),
            "--run-dir",
            str(args.output_dir),
            "--pytorch-state",
            str(pytorch_state),
            "--output-dir",
            str(mode_output_dir),
            "--device",
            args.device,
            "--sample-count",
            str(args.report_sample_count),
            "--training-crop-size",
            str(args.training_crop_size),
            "--training-max-image-size",
            str(args.training_max_image_size),
            "--max-keypoints",
            str(args.report_max_keypoints),
            "--max-matches",
            str(args.report_max_matches),
            "--draw-matches",
            str(args.report_draw_matches),
            "--texture-keypoint-fraction",
            str(args.report_texture_keypoint_fraction),
            "--weak-texture-keypoint-fraction",
            str(args.report_weak_texture_keypoint_fraction),
            "--keypoint-spatial-bins",
            str(args.report_keypoint_spatial_bins),
            "--keypoint-cell-cap",
            str(args.report_keypoint_cell_cap),
            "--coverage-bins",
            str(args.report_coverage_bins),
            "--min-margin",
            str(args.report_min_margin),
            "--matcher-mode",
            matcher_mode,
            "--min-intensity",
            str(args.min_intensity),
        ]
        for pattern in args.report_required_sample_glob:
            command.extend(["--required-sample-glob", pattern])
        for cache_dir in args.validation_cache_dir:
            command.extend(["--validation-cache-dir", str(cache_dir)])
        for root in args.pose_metadata_root:
            command.extend(["--pose-metadata-root", str(root)])
        env = os.environ.copy()
        python_path = str(project_root / "python")
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = python_path + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = python_path
        print("training_report_command=" + " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)


def save_pytorch_training_state(
    path: Path,
    *,
    model,
    config,
    args: argparse.Namespace,
    step: int,
    epoch_progress: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.__dict__,
            "model": model.state_dict(),
            "source_checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "source_pytorch_state": str(args.init_pytorch_state) if args.init_pytorch_state is not None else None,
            "training_step": int(step),
            "epoch_progress": float(epoch_progress),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model, config = load_training_model(args)
    trainable = descriptor_parameters(
        model,
        train_backbone=args.train_backbone,
        train_dual_fpn=args.train_dual_fpn,
        train_descriptor_head=not args.freeze_descriptor_head,
        train_sparse_context=args.train_sparse_context,
        train_keypoint_head=args.pseudo_keypoint_weight > 0.0,
        train_geometry_head=args.train_geometry_head,
        train_texture_adapter=args.train_texture_adapter,
        train_descriptor_fusion=args.train_descriptor_fusion,
        train_quality_head=args.train_quality_head,
        train_graph_matcher=args.train_graph_matcher,
    )
    if not trainable:
        raise RuntimeError("no trainable parameters selected")
    if not trainable:
        raise RuntimeError("no descriptor parameters selected")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1.0e-4)
    pair_paths, eval_paths = resolve_training_and_eval_pair_paths(
        args.cache_dir,
        args.validation_cache_dir,
        limit_pairs=args.limit_pairs,
        eval_pairs=args.eval_pairs,
        exclude_self_pairs=args.exclude_self_pairs,
    )
    pose_metadata: pose_pair_metadata.PoseMetadataIndex = {}
    pose_roots = pose_pair_metadata.infer_pose_metadata_roots(
        list(args.cache_dir) + list(args.validation_cache_dir),
        args.pose_metadata_root,
    )
    for pose_root in pose_roots:
        pose_metadata.update(pose_pair_metadata.load_pose_metadata_index(pose_root))
    if pose_roots:
        print(
            f"pose_metadata_roots={len(pose_roots)} pose_metadata_pairs={len(pose_metadata)} "
            f"pose_balanced_sampling={int(args.pose_balanced_sampling)} "
            f"pose_min_overlap={args.pose_min_overlap:.3f} "
            f"pose_difficulty_loss_weight={args.pose_difficulty_loss_weight:.3f} "
            f"training_crop_size={args.training_crop_size}",
            flush=True,
        )
    pair_paths = filter_pair_paths_by_pose_overlap(
        pair_paths,
        pose_metadata,
        min_overlap=args.pose_min_overlap,
    )
    eval_paths = filter_pair_paths_by_pose_overlap(
        eval_paths,
        pose_metadata,
        min_overlap=args.pose_min_overlap,
    )
    if not pair_paths:
        raise RuntimeError("no pair_*.pt archives found")
    if not pair_paths:
        raise RuntimeError("no training pair_*.pt archives left after eval split")
    steps_per_epoch = max(1, math.ceil(len(pair_paths) / float(args.batch_pairs)))
    total_epochs = args.epochs if args.epochs > 0 else args.steps / float(steps_per_epoch)
    if args.epochs > 0:
        args.steps = args.epochs * steps_per_epoch
    print(
        f"training_pairs={len(pair_paths)} batch_pairs={args.batch_pairs} "
        f"steps_per_epoch={steps_per_epoch} epochs={total_epochs:.4f} total_steps={args.steps} "
        f"epoch_shuffle_sampling={int(args.epoch_shuffle_sampling)} pair_cache_size={args.pair_cache_size} "
        f"prefetch_batches={args.prefetch_batches} prefetch_workers={args.prefetch_workers}",
        flush=True,
    )
    epoch_sampler = (
        EpochShuffleSampler(pair_paths, batch_pairs=args.batch_pairs, seed=args.seed + 1701)
        if args.epoch_shuffle_sampling
        else None
    )
    hard_paths = select_hard_training_pairs(
        pair_paths,
        args.hard_summary,
        limit=args.hard_limit,
        min_matches=args.hard_min_matches,
        max_precision=args.hard_max_precision,
    )
    hard_paths = sorted(dict.fromkeys(hard_paths + select_hard_training_pairs_by_glob(pair_paths, args.hard_pair_glob)))
    if hard_paths and args.hard_curriculum_max_probability <= 0.0:
        pair_paths = pair_paths + hard_paths * max(0, args.hard_repeat)
    if hard_paths:
        print(
            f"hard_training_pairs={len(hard_paths)} hard_repeat={args.hard_repeat} "
            f"hard_curriculum_max_probability={args.hard_curriculum_max_probability:.3f} "
            f"effective_train_pairs={len(pair_paths)}",
            flush=True,
        )
    pseudo_labels = read_pseudo_label_matches(args.pseudo_label_csv) if args.pseudo_label_csv else {}
    pseudo_label_paths = select_pseudo_labeled_training_pairs(pair_paths, pseudo_labels) if pseudo_labels else []
    false_matches = read_false_match_labels(args.false_match_csv) if args.false_match_csv else {}
    false_match_paths = select_false_match_training_pairs(pair_paths, false_matches) if false_matches else []
    if args.pseudo_label_csv:
        pseudo_match_count = sum(label.points_a_xy.size(0) for label in pseudo_labels.values())
        print(
            f"pseudo_label_pairs={len(pseudo_labels)} pseudo_label_matches={pseudo_match_count} "
            f"pseudo_label_training_pairs={len(pseudo_label_paths)} "
            f"pseudo_label_weight={args.pseudo_label_weight:.3f} "
            f"pseudo_keypoint_weight={args.pseudo_keypoint_weight:.3f} "
            f"pseudo_label_max_points={args.pseudo_label_max_points} "
            f"pseudo_label_curriculum_max_probability={args.pseudo_label_curriculum_max_probability:.3f}",
            flush=True,
        )
    if args.false_match_csv:
        false_match_count = sum(label.points_a_xy.size(0) for label in false_matches.values())
        print(
            f"false_match_pairs={len(false_matches)} false_matches={false_match_count} "
            f"false_match_training_pairs={len(false_match_paths)} "
            f"false_match_weight={args.false_match_weight:.3f} "
            f"false_match_max_points={args.false_match_max_points} "
            f"false_match_curriculum_max_probability={args.false_match_curriculum_max_probability:.3f}",
            flush=True,
        )
    train_generator = make_torch_generator(device, seed=args.seed)
    eval_seed = args.seed + 1000003
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    eval_summary_path = args.output_dir / "eval_summary.csv"
    pair_cache = PairArchiveCache(args.pair_cache_size) if args.pair_cache_size > 0 else None
    if eval_paths:
        eval_before = evaluate_descriptor_retrieval(
            model,
            eval_paths,
            device=device,
            samples_per_pair=args.samples_per_pair,
            min_intensity=args.min_intensity,
            generator=make_torch_generator(device, seed=eval_seed),
            temperature=args.temperature,
            training_crop_size=args.training_crop_size,
            training_max_image_size=args.training_max_image_size,
            pair_cache=pair_cache,
        )
        print(
            f"eval_before loss={eval_before['loss']:.6f} top1={eval_before['top1_accuracy']:.4f} "
            f"top5={eval_before['top5_accuracy']:.4f} rank={eval_before['mean_positive_rank']:.2f} "
            f"pos={eval_before['mean_positive_score']:.6f} neg={eval_before['mean_negative_score']:.6f} "
            f"points={int(eval_before['points'])}",
            flush=True,
        )
    prefetch_executor: ThreadPoolExecutor | None = None
    prefetch_futures: dict[int, Future] = {}

    def schedule_prefetch(prefetch_step: int) -> None:
        if prefetch_executor is None or epoch_sampler is None:
            return
        if prefetch_step < 1 or prefetch_step > args.steps or prefetch_step in prefetch_futures:
            return
        prefetch_futures[prefetch_step] = prefetch_executor.submit(
            load_pair_batch_cpu,
            epoch_sampler.batch_for_step(prefetch_step),
        )

    if args.prefetch_batches > 0:
        if epoch_sampler is None:
            print("prefetch_disabled=requires_epoch_shuffle_sampling", flush=True)
        else:
            prefetch_executor = ThreadPoolExecutor(max_workers=args.prefetch_workers)
            for prefetch_step in range(1, min(args.steps, args.prefetch_batches) + 1):
                schedule_prefetch(prefetch_step)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "epoch",
                "epoch_progress",
                "loss",
                "grad_l2",
                "skipped",
                "teacher_weight",
                "synthetic_loss_weight",
                "hard_negative_weight",
                "diversity_weight",
                "abstention_weight",
                "top1_accuracy",
                "top5_accuracy",
                "top10_accuracy",
                "mean_positive_rank",
                "mean_positive_score",
                "mean_negative_score",
                "points",
                "pseudo_label_points",
                "pseudo_keypoint_points",
                "pseudo_label_pairs",
                "false_match_points",
                "false_match_pairs",
                "pose_easy_pairs",
                "pose_medium_pairs",
                "pose_hard_pairs",
                "pose_unknown_pairs",
                "pose_mean_loss_weight",
            ],
        )
        writer.writeheader()
        for step in range(1, args.steps + 1):
            epoch_float = step / float(steps_per_epoch)
            epoch_index = min(int((step - 1) // steps_per_epoch) + 1, max(1, math.ceil(total_epochs)))
            teacher_weight = scheduled_value(
                step,
                start=args.teacher_weight,
                final=args.teacher_final_weight if args.teacher_final_weight is not None else args.teacher_weight,
                schedule_steps=args.loss_schedule_steps,
            )
            hard_negative_weight = scheduled_value(
                step,
                start=args.hard_negative_weight,
                final=(
                    args.hard_negative_final_weight
                    if args.hard_negative_final_weight is not None
                    else args.hard_negative_weight
                ),
                schedule_steps=args.loss_schedule_steps,
            )
            diversity_weight = scheduled_value(
                step,
                start=args.diversity_weight,
                final=args.diversity_final_weight if args.diversity_final_weight is not None else args.diversity_weight,
                schedule_steps=args.loss_schedule_steps,
            )
            forced_pair_paths = epoch_sampler.batch_for_step(step) if epoch_sampler is not None else None
            prefetched_pairs = None
            if prefetch_executor is not None:
                future = prefetch_futures.pop(step, None)
                if future is not None:
                    prefetched_pairs = future.result()
                    if pair_cache is not None:
                        pair_cache.put_batch(prefetched_pairs)
                schedule_prefetch(step + args.prefetch_batches)
            metrics = train_step(
                model,
                optimizer,
                pair_paths,
                device=device,
                batch_pairs=args.batch_pairs,
                samples_per_pair=args.samples_per_pair,
                min_intensity=args.min_intensity,
                generator=train_generator,
                training_weak_texture_fraction=args.training_weak_texture_fraction,
                temperature=args.temperature,
                teacher_weight=teacher_weight,
                synthetic_loss_weight=args.synthetic_loss_weight,
                hard_pair_paths=hard_paths if args.hard_curriculum_max_probability > 0.0 else None,
                hard_probability=hard_pair_probability(
                    step,
                    max_probability=args.hard_curriculum_max_probability,
                    warmup_steps=args.hard_curriculum_warmup_steps,
                ),
                hard_negative_weight=hard_negative_weight,
                diversity_weight=diversity_weight,
                warp_hard_negative_weight=args.warp_hard_negative_weight,
                warp_hard_negative_radius=args.warp_hard_negative_radius,
                warp_hard_negative_margin=args.warp_hard_negative_margin,
                warp_hard_negative_candidates=args.warp_hard_negative_candidates,
                abstention_weight=args.abstention_weight,
                abstention_negative_radius=args.abstention_negative_radius,
                abstention_max_false_score=args.abstention_max_false_score,
                abstention_topk=args.abstention_topk,
                abstention_candidates=args.abstention_candidates,
                max_grad_norm=args.max_grad_norm,
                skip_nonfinite_steps=args.skip_nonfinite_steps,
                train_blended_descriptors=args.train_blended_descriptors,
                texture_blend_weight=args.training_texture_blend_weight,
                balanced_cache_sampling=args.balanced_cache_sampling,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                pseudo_labels=pseudo_labels,
                pseudo_label_weight=args.pseudo_label_weight,
                pseudo_keypoint_weight=args.pseudo_keypoint_weight,
                pseudo_keypoint_negative_weight=args.pseudo_keypoint_negative_weight,
                pseudo_label_max_points=args.pseudo_label_max_points,
                pseudo_label_pair_paths=pseudo_label_paths,
                pseudo_label_probability=hard_pair_probability(
                    step,
                    max_probability=args.pseudo_label_curriculum_max_probability,
                    warmup_steps=args.pseudo_label_curriculum_warmup_steps,
                ),
                false_matches=false_matches,
                false_match_weight=args.false_match_weight,
                false_match_max_points=args.false_match_max_points,
                false_match_max_score=args.false_match_max_score,
                false_match_pair_paths=false_match_paths,
                false_match_probability=hard_pair_probability(
                    step,
                    max_probability=args.false_match_curriculum_max_probability,
                    warmup_steps=args.false_match_curriculum_warmup_steps,
                ),
                pose_metadata=pose_metadata,
                pose_balanced_sampling=args.pose_balanced_sampling,
                pose_difficulty_loss_weight=args.pose_difficulty_loss_weight,
                graph_matcher_loss_weight=args.graph_matcher_loss_weight if args.train_graph_matcher else 0.0,
                graph_matcher_metadata_mode=args.graph_matcher_metadata_mode,
                graph_matcher_no_match_points=args.graph_matcher_no_match_points,
                graph_matcher_no_match_weight=args.graph_matcher_no_match_weight,
                graph_matcher_no_match_min_distance=args.graph_matcher_no_match_min_distance,
                graph_matcher_accept_weight=args.graph_matcher_accept_weight,
                graph_matcher_accept_negative_topk=args.graph_matcher_accept_negative_topk,
                graph_matcher_raw_preservation_weight=args.graph_matcher_raw_preservation_weight,
                graph_matcher_raw_preservation_margin=args.graph_matcher_raw_preservation_margin,
                graph_matcher_raw_preservation_raw_margin=args.graph_matcher_raw_preservation_raw_margin,
                graph_matcher_hard_negative_dustbin_weight=args.graph_matcher_hard_negative_dustbin_weight,
                graph_matcher_hard_negative_dustbin_topk=args.graph_matcher_hard_negative_dustbin_topk,
                graph_matcher_hard_negative_dustbin_margin=args.graph_matcher_hard_negative_dustbin_margin,
                graph_matcher_hard_negative_dustbin_spatial_min_distance=args.graph_matcher_hard_negative_dustbin_spatial_min_distance,
                graph_matcher_semi_dense_no_match_points=args.graph_matcher_semi_dense_no_match_points,
                graph_matcher_semi_dense_min_score=args.graph_matcher_semi_dense_min_score,
                training_spatial_bins=args.training_spatial_bins,
                training_crop_size=args.training_crop_size,
                training_max_image_size=args.training_max_image_size,
                forced_pair_paths=forced_pair_paths,
                prefetched_pairs=prefetched_pairs,
                pair_cache=pair_cache,
            )
            writer.writerow(
                {
                    "step": step,
                    "epoch": epoch_index,
                    "epoch_progress": epoch_float,
                    "teacher_weight": teacher_weight,
                    "synthetic_loss_weight": args.synthetic_loss_weight,
                    "hard_negative_weight": hard_negative_weight,
                    "diversity_weight": diversity_weight,
                    "abstention_weight": args.abstention_weight,
                    **metrics,
                }
            )
            handle.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                cache_text = (
                    f" cache={pair_cache.hits}/{pair_cache.misses}/{pair_cache.size}"
                    f"/{pair_cache.max_items} prefetch_cached={pair_cache.prefetch_inserts}"
                    if pair_cache is not None
                    else ""
                )
                print(
                    f"step={step}/{args.steps} epoch={epoch_float:.3f}/{total_epochs:.3f} "
                    f"loss={metrics['loss']:.6f} grad={metrics['grad_l2']:.6f} "
                    f"tw={teacher_weight:.3f} syn={args.synthetic_loss_weight:.3f} "
                    f"hn={hard_negative_weight:.3f} div={diversity_weight:.3f} "
                    f"abst={args.abstention_weight:.3f} "
                    f"skip={int(metrics['skipped'])} "
                    f"top1={metrics['top1_accuracy']:.4f} top5={metrics['top5_accuracy']:.4f} "
                    f"rank={metrics['mean_positive_rank']:.2f} "
                    f"pos={metrics['mean_positive_score']:.6f} neg={metrics['mean_negative_score']:.6f} "
                    f"points={int(metrics['points'])} pseudo={int(metrics.get('pseudo_label_points', 0.0))} "
                    f"pseudo_kp={int(metrics.get('pseudo_keypoint_points', 0.0))} "
                    f"false={int(metrics.get('false_match_points', 0.0))} "
                    f"pose_w={metrics.get('pose_mean_loss_weight', 1.0):.3f}{cache_text}",
                    flush=True,
                )
            if args.save_every_epoch and step % steps_per_epoch == 0:
                epoch_checkpoint = args.output_dir / "checkpoints" / f"epoch_{epoch_index:03d}_pytorch_pfm_state.pt"
                save_pytorch_training_state(
                    epoch_checkpoint,
                    model=model,
                    config=config,
                    args=args,
                    step=step,
                    epoch_progress=epoch_float,
                )
                latest_checkpoint = args.output_dir / "checkpoints" / "latest_pytorch_pfm_state.pt"
                save_pytorch_training_state(
                    latest_checkpoint,
                    model=model,
                    config=config,
                    args=args,
                    step=step,
                    epoch_progress=epoch_float,
                )
                print(f"epoch_checkpoint={epoch_checkpoint}", flush=True)
    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=True)
    if eval_paths:
        eval_after = evaluate_descriptor_retrieval(
            model,
            eval_paths,
            device=device,
            samples_per_pair=args.samples_per_pair,
            min_intensity=args.min_intensity,
            generator=make_torch_generator(device, seed=eval_seed),
            temperature=args.temperature,
            training_crop_size=args.training_crop_size,
            training_max_image_size=args.training_max_image_size,
            pair_cache=pair_cache,
        )
        with eval_summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "phase",
                    "loss",
                    "top1_accuracy",
                    "top5_accuracy",
                    "top10_accuracy",
                    "mean_positive_rank",
                    "mean_positive_score",
                    "mean_negative_score",
                    "points",
                ],
            )
            writer.writeheader()
            writer.writerow({"phase": "before", **eval_before})
            writer.writerow({"phase": "after", **eval_after})
        print(
            f"eval_after loss={eval_after['loss']:.6f} top1={eval_after['top1_accuracy']:.4f} "
            f"top5={eval_after['top5_accuracy']:.4f} rank={eval_after['mean_positive_rank']:.2f} "
            f"pos={eval_after['mean_positive_score']:.6f} neg={eval_after['mean_negative_score']:.6f} "
            f"points={int(eval_after['points'])}",
            flush=True,
        )
    output_path = args.output_dir / "pytorch_pfm_state.pt"
    save_pytorch_training_state(
        output_path,
        model=model,
        config=config,
        args=args,
        step=args.steps,
        epoch_progress=args.steps / float(steps_per_epoch),
    )
    print(f"checkpoint={output_path}")
    print(f"metrics={metrics_path}")
    if args.generate_training_report:
        del optimizer
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        run_training_report(args, pytorch_state=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
