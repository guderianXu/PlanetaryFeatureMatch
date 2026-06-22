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
from contextlib import nullcontext
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


GRAPH_INFERENCE_PRESET_CHOICES = ("off", "fast", "high_precision")
AMP_DTYPE_CHOICES = ("float16", "bfloat16")
REJECTION_TRAINING_DEFAULTS = {
    "graph_matcher_loss_weight": 0.75,
    "graph_matcher_no_match_points": 128,
    "graph_matcher_no_match_weight": 0.45,
    "graph_matcher_assignment_weight": 0.35,
    "graph_matcher_accept_weight": 0.30,
    "graph_matcher_prune_ranking_weight": 0.10,
    "graph_matcher_stop_confidence_weight": 0.05,
    "graph_matcher_hard_negative_dustbin_weight": 0.075,
    "graph_matcher_hard_negative_dustbin_topk": 16,
    "graph_matcher_hard_negative_dustbin_margin": 0.35,
    "graph_matcher_semi_dense_no_match_points": 128,
    "false_match_weight": 0.15,
    "false_match_max_points": 192,
    "false_match_curriculum_max_probability": 1.0,
    "keypoint_weight": 0.05,
    "keypoint_negative_weight": 0.02,
    "matchability_weight": 0.08,
    "descriptor_uncertainty_weight": 0.05,
    "no_match_prior_weight": 0.08,
    "reliability_negative_points": 128,
    "rotation_descriptor_consistency_weight": 0.03,
}
REJECTION_TRAINING_BASE_DEFAULTS = {
    "graph_matcher_loss_weight": 1.0,
    "graph_matcher_accept_weight": 0.2,
    "graph_matcher_hard_negative_dustbin_topk": 8,
    "graph_matcher_hard_negative_dustbin_margin": 0.25,
    "false_match_max_points": 128,
    "keypoint_negative_weight": 0.01,
}
GRAPH_MATCHER_LOSS_METRIC_KEYS = (
    "graph_matcher_total_loss",
    "graph_matcher_ce_loss",
    "graph_matcher_assignment_loss",
    "graph_matcher_no_match_loss",
    "graph_matcher_accept_loss",
    "graph_matcher_prune_ranking_loss",
    "graph_matcher_stop_confidence_loss",
    "graph_matcher_raw_preservation_loss",
    "graph_matcher_hard_negative_dustbin_loss",
    "graph_matcher_positive_dustbin_margin_loss",
    "graph_matcher_true_match_margin_loss",
    "graph_matcher_true_match_margin_violations",
    "graph_matcher_true_match_margin_mean",
    "graph_matcher_true_geometry_match_count_floor_loss",
    "graph_matcher_true_geometry_match_count_floor_target_count",
    "graph_matcher_true_geometry_match_count_floor_student_count",
    "graph_matcher_true_geometry_match_count_floor_count_deficit",
    "graph_matcher_true_geometry_match_count_floor_topk_score_mean",
    "graph_matcher_true_geometry_match_count_floor_violations",
    "graph_matcher_final_false_match_loss",
    "graph_matcher_final_false_match_edges",
    "graph_matcher_final_false_match_score_mean",
    "graph_matcher_final_false_match_accept_mean",
    "graph_matcher_mined_false_match_loss",
    "graph_matcher_mined_false_match_edges",
    "graph_matcher_mined_false_match_reference_filtered_edges",
    "graph_matcher_mined_false_match_score_mean",
    "graph_matcher_mined_false_match_logit_mean",
    "graph_matcher_mined_false_match_accept_mean",
    "graph_matcher_raw_false_match_loss",
    "graph_matcher_raw_false_match_edges",
    "graph_matcher_raw_false_match_similarity_mean",
    "graph_matcher_raw_false_match_margin_mean",
    "graph_matcher_ransac_consistency_loss",
    "graph_matcher_ransac_consistency_edges",
    "graph_matcher_ransac_consistency_score_mean",
    "graph_matcher_ransac_consistency_residual_mean_px",
    "graph_matcher_ransac_consistency_accept_mean",
    "graph_matcher_warp_outlier_loss",
    "graph_matcher_warp_outlier_edges",
    "graph_matcher_warp_outlier_residual_mean_px",
    "graph_matcher_warp_outlier_accept_mean",
    "graph_matcher_warp_outlier_accept_loss",
    "graph_matcher_warp_outlier_accept_edges",
    "graph_matcher_warp_outlier_accept_score_mean",
    "graph_matcher_warp_outlier_accept_residual_mean_px",
    "graph_matcher_warp_outlier_accept_probability_mean",
    "graph_matcher_warp_soft_boundary_loss",
    "graph_matcher_warp_soft_boundary_edges",
    "graph_matcher_warp_soft_boundary_residual_mean_px",
    "graph_matcher_warp_soft_boundary_target_mean",
    "graph_matcher_warp_soft_boundary_score_probability_mean",
    "graph_matcher_warp_soft_boundary_accept_probability_mean",
    "graph_matcher_pair_acceptance_loss",
    "graph_matcher_pair_acceptance_target",
    "graph_matcher_pair_acceptance_weight",
    "graph_matcher_pair_acceptance_probability",
    "graph_matcher_deep_supervision_loss",
    "graph_matcher_depth_distillation_loss",
    "graph_matcher_depth_distillation_teacher_layers",
    "graph_matcher_teacher_distillation_loss",
    "graph_matcher_teacher_guard_loss",
    "graph_matcher_teacher_guard_positive_margin_loss",
    "graph_matcher_teacher_guard_false_edge_loss",
    "graph_matcher_teacher_guard_positive_violations",
    "graph_matcher_teacher_guard_false_edges",
    "graph_matcher_teacher_score_floor_loss",
    "graph_matcher_teacher_score_floor_violations",
    "graph_matcher_teacher_score_floor_delta_mean",
    "graph_matcher_teacher_score_floor_teacher_score_mean",
    "graph_matcher_teacher_match_count_floor_loss",
    "graph_matcher_teacher_match_count_floor_teacher_count",
    "graph_matcher_teacher_match_count_floor_student_count",
    "graph_matcher_teacher_match_count_floor_count_deficit",
    "graph_matcher_teacher_match_count_floor_topk_score_mean",
    "graph_matcher_teacher_match_count_ceiling_loss",
    "graph_matcher_teacher_match_count_ceiling_teacher_count",
    "graph_matcher_teacher_match_count_ceiling_student_count",
    "graph_matcher_teacher_match_count_ceiling_count_excess",
    "graph_matcher_teacher_match_count_ceiling_excess_score_mean",
    "graph_matcher_executed_attention_layers",
    "graph_matcher_attention_work_fraction",
    "graph_matcher_positive_pairs",
    "graph_matcher_extra_no_match_points",
    "graph_matcher_extra_false_match_pairs",
    "graph_matcher_train_candidate_topk",
    "graph_matcher_effective_no_match_weight",
    "graph_matcher_effective_hard_negative_dustbin_weight",
    "graph_matcher_dustbin_guard_active",
    "graph_matcher_guarded_no_match_weight",
    "graph_matcher_guarded_hard_negative_dustbin_weight",
    "true_match_rejected_by_dustbin_ratio",
    "positive_pair_logit_mean",
    "positive_dustbin_logit_mean",
    "dustbin_logit_mean",
    "dustbin_logit_for_true_match_mean",
    "positive_vs_dustbin_margin_mean",
    "positive_vs_dustbin_margin_median",
    "positive_vs_dustbin_margin_p10",
    "positive_vs_dustbin_margin_below0_ratio",
    "false_match_accepted_ratio",
    "accept_logit_mean",
    "true_pair_prob_mean",
    "dustbin_prob_for_true_match_mean",
    "true_match_in_topk@64",
    "true_match_in_topk@256",
)
RELIABILITY_LOSS_METRIC_KEYS = (
    "matchability_loss",
    "descriptor_uncertainty_loss",
    "no_match_prior_loss",
    "reliability_points",
    "rotation_descriptor_consistency_loss",
    "orientation_consistency_loss",
    "scale_consistency_loss",
    "affine_consistency_loss",
    "affine_regularization_loss",
    "affine_det_mean",
    "affine_det_std",
    "affine_condition_mean",
    "affine_condition_max",
    "rotation_consistency_points",
    "rotation_consistency_pairs",
)


@dataclass(frozen=True)
class PseudoLabelMatches:
    points_a_xy: torch.Tensor
    points_b_xy: torch.Tensor


@dataclass(frozen=True)
class TrainingSparseMaps:
    descriptors: torch.Tensor
    heatmap: torch.Tensor
    keypoint_offsets: torch.Tensor
    matchability: torch.Tensor
    descriptor_uncertainty: torch.Tensor
    no_match_prior: torch.Tensor
    scale: torch.Tensor
    orientation: torch.Tensor
    affine: torch.Tensor
    quality: torch.Tensor | None = None


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


def _scale_feature_to_image_grid(
    points_xy: torch.Tensor,
    *,
    feature_height: int,
    feature_width: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    if points_xy.numel() == 0:
        return points_xy.new_empty((0, 2))
    x = points_xy[:, 0] * float(max(1, image_width - 1)) / float(max(1, feature_width - 1))
    y = points_xy[:, 1] * float(max(1, image_height - 1)) / float(max(1, feature_height - 1))
    return torch.stack([x, y], dim=1)


def _sample_warp_points(warp_a_to_b: torch.Tensor, points_a_xy: torch.Tensor) -> torch.Tensor:
    if points_a_xy.numel() == 0:
        return points_a_xy.new_empty((0, 2))
    if warp_a_to_b.dim() != 3 or warp_a_to_b.size(2) != 2:
        raise ValueError("warp_a_to_b must have shape HxWx2")
    height, width, _ = warp_a_to_b.shape
    grid = _normalize_xy(points_a_xy.to(warp_a_to_b.device, torch.float32), height, width).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        warp_a_to_b.permute(2, 0, 1).unsqueeze(0).to(torch.float32),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).T.contiguous()


def _valid_source_mask(valid_mask: torch.Tensor, points_a_xy: torch.Tensor) -> torch.Tensor:
    if points_a_xy.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=valid_mask.device)
    height, width = valid_mask.shape
    points = points_a_xy.to(valid_mask.device, torch.float32)
    in_bounds = (
        torch.isfinite(points).all(dim=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= float(width - 1))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= float(height - 1))
    )
    rounded = points.round().to(torch.long)
    x = rounded[:, 0].clamp(0, width - 1)
    y = rounded[:, 1].clamp(0, height - 1)
    return in_bounds & valid_mask.to(torch.bool)[y, x]


def _descriptor_rows_at_indices(descriptors: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
    if descriptors.dim() != 4 or descriptors.size(0) != 1:
        raise ValueError("descriptors must have shape 1xDxHxW")
    flat = descriptors.squeeze(0).permute(1, 2, 0).reshape(-1, descriptors.size(1))
    if selected_indices.numel() == 0:
        return flat.new_empty((0, descriptors.size(1)))
    return flat.index_select(0, selected_indices.to(descriptors.device)).contiguous()


def _cyclic_similarity_matrix(desc_a: torch.Tensor, desc_b: torch.Tensor) -> torch.Tensor:
    """Legacy entry point for descriptor similarity; now uses strict cosine only."""
    if desc_a.dim() != 2 or desc_b.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if desc_a.size(1) != desc_b.size(1):
        raise ValueError("descriptor dimensions must match")
    desc_a = normalize_descriptor_batch(desc_a)
    desc_b = normalize_descriptor_batch(desc_b)
    return desc_a @ desc_b.T


def _mutual_nearest_descriptor_matches(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    max_matches: int,
    min_score: float,
    min_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_matches < 0:
        raise ValueError("max_matches must be nonnegative; use 0 to keep all matches")
    if min_margin < 0.0:
        raise ValueError("min_margin must be non-negative")
    if desc_a.size(0) == 0 or desc_b.size(0) == 0:
        empty_matches = torch.empty(0, 2, dtype=torch.long, device=desc_a.device)
        return empty_matches, torch.empty(0, dtype=torch.float32, device=desc_a.device)
    similarity = _cyclic_similarity_matrix(desc_a, desc_b)
    best_scores, best_targets = similarity.max(dim=1)
    best_sources = similarity.max(dim=0).indices
    source_indices = torch.arange(similarity.size(0), dtype=torch.long, device=similarity.device)
    keep = best_sources.index_select(0, best_targets) == source_indices
    keep &= best_scores >= float(min_score)
    if min_margin > 0.0 and similarity.size(1) > 1:
        top2 = similarity.topk(2, dim=1).values
        keep &= (top2[:, 0] - top2[:, 1]) >= float(min_margin)
    kept_sources = torch.nonzero(keep, as_tuple=False).reshape(-1)
    if kept_sources.numel() == 0:
        empty_matches = torch.empty(0, 2, dtype=torch.long, device=desc_a.device)
        return empty_matches, torch.empty(0, dtype=torch.float32, device=desc_a.device)
    kept_scores = best_scores.index_select(0, kept_sources)
    order = kept_scores.argsort(descending=True, stable=True)
    limit = order.numel() if max_matches == 0 else min(max_matches, order.numel())
    order = order[:limit]
    sources = kept_sources.index_select(0, order)
    targets = best_targets.index_select(0, sources)
    matches = torch.stack([sources, targets], dim=1)
    return matches.contiguous(), kept_scores.index_select(0, order).contiguous()


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


def _descriptor_keypoints_for_online_false_mining(
    image: torch.Tensor,
    descriptors: torch.Tensor,
    *,
    max_keypoints: int,
    min_intensity: float,
    keypoint_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    if descriptors.dim() != 4 or descriptors.size(0) != 1:
        raise ValueError("descriptors must have shape 1xDxHxW")
    if max_keypoints <= 0:
        raise ValueError("max_keypoints must be positive")
    _, image_height, image_width = image.shape
    descriptor_height = descriptors.size(2)
    descriptor_width = descriptors.size(3)
    yy, xx = torch.meshgrid(
        torch.arange(descriptor_height, device=descriptors.device),
        torch.arange(descriptor_width, device=descriptors.device),
        indexing="ij",
    )
    keypoints = torch.stack([xx.to(torch.float32), yy.to(torch.float32)], dim=-1).reshape(-1, 2)
    image_points = _scale_feature_to_image_grid(
        keypoints,
        feature_height=descriptor_height,
        feature_width=descriptor_width,
        image_height=image_height,
        image_width=image_width,
    )
    rounded = image_points.round().to(torch.long)
    x = rounded[:, 0].clamp(0, image_width - 1)
    y = rounded[:, 1].clamp(0, image_height - 1)
    intensity = image.to(descriptors.device, torch.float32).mean(dim=0)[y, x]
    valid = intensity > float(min_intensity) if min_intensity > 0.0 else torch.ones_like(intensity, dtype=torch.bool)
    selected = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        return keypoints.new_empty((0, 2)), selected
    if selected.numel() > max_keypoints:
        if keypoint_scores is not None:
            if keypoint_scores.dim() == 4:
                scores = keypoint_scores[0, 0].to(descriptors.device, torch.float32)
            elif keypoint_scores.dim() == 2:
                scores = keypoint_scores.to(descriptors.device, torch.float32)
            else:
                raise ValueError("keypoint_scores must have shape 1x1xHxW or HxW")
            if tuple(scores.shape) != (descriptor_height, descriptor_width):
                scores = F.interpolate(
                    scores.view(1, 1, scores.size(0), scores.size(1)),
                    size=(descriptor_height, descriptor_width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            flat_scores = scores.reshape(-1)
        else:
            flat_scores = image_local_texture_scores(image.to(descriptors.device), image_points)
        selected_scores = flat_scores.index_select(0, selected)
        order = selected_scores.argsort(descending=True, stable=True)[:max_keypoints]
        selected = selected.index_select(0, order)
    return keypoints.index_select(0, selected).contiguous(), selected.to(torch.long).contiguous()


def _feature_points_in_bounds(points_xy: torch.Tensor, *, feature_height: int, feature_width: int) -> torch.Tensor:
    if points_xy.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=points_xy.device)
    return (
        torch.isfinite(points_xy).all(dim=1)
        & (points_xy[:, 0] >= 0.0)
        & (points_xy[:, 0] <= float(max(0, feature_width - 1)))
        & (points_xy[:, 1] >= 0.0)
        & (points_xy[:, 1] <= float(max(0, feature_height - 1)))
    )


def _warp_feature_points_a_to_b(
    pair: SyntheticPair,
    points_a_feature: torch.Tensor,
    *,
    feature_height_a: int,
    feature_width_a: int,
    feature_height_b: int,
    feature_width_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points_a_feature.numel() == 0:
        empty = points_a_feature.new_empty((0, 2))
        return empty, empty
    _, image_height_a, image_width_a = pair.view_a.shape
    _, image_height_b, image_width_b = pair.view_b.shape
    points_a_image = _scale_feature_to_image_grid(
        points_a_feature,
        feature_height=feature_height_a,
        feature_width=feature_width_a,
        image_height=image_height_a,
        image_width=image_width_a,
    )
    points_b_image = _sample_warp_points(pair.warp_a_to_b, points_a_image)
    valid_source = _valid_source_mask(pair.valid_mask, points_a_image).to(points_a_feature.device)
    valid_target_image = (
        torch.isfinite(points_b_image).all(dim=1).to(points_a_feature.device)
        & (points_b_image[:, 0].to(points_a_feature.device) >= 0.0)
        & (points_b_image[:, 0].to(points_a_feature.device) <= float(max(0, image_width_b - 1)))
        & (points_b_image[:, 1].to(points_a_feature.device) >= 0.0)
        & (points_b_image[:, 1].to(points_a_feature.device) <= float(max(0, image_height_b - 1)))
    )
    valid = valid_source & valid_target_image
    if not bool(valid.any()):
        empty = points_a_feature.new_empty((0, 2))
        return empty, empty
    selected_a = points_a_feature.index_select(0, torch.nonzero(valid, as_tuple=False).reshape(-1))
    selected_b_image = points_b_image.to(points_a_feature.device).index_select(
        0,
        torch.nonzero(valid, as_tuple=False).reshape(-1),
    )
    selected_b = _scale_points_to_feature_grid(
        selected_b_image,
        image_height=image_height_b,
        image_width=image_width_b,
        feature_height=feature_height_b,
        feature_width=feature_width_b,
    )
    in_bounds = _feature_points_in_bounds(
        selected_b,
        feature_height=feature_height_b,
        feature_width=feature_width_b,
    )
    if not bool(in_bounds.any()):
        empty = points_a_feature.new_empty((0, 2))
        return empty, empty
    keep = torch.nonzero(in_bounds, as_tuple=False).reshape(-1)
    return selected_a.index_select(0, keep).contiguous(), selected_b.index_select(0, keep).contiguous()


def _subcell_feature_grid(
    *,
    feature_height: int,
    feature_width: int,
    device: torch.device,
    dtype: torch.dtype,
    max_candidates: int,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(feature_height, device=device, dtype=dtype),
        torch.arange(feature_width, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    offsets = torch.tensor(
        [
            [0.0, 0.0],
            [-0.25, 0.0],
            [0.25, 0.0],
            [0.0, -0.25],
            [0.0, 0.25],
            [-0.25, -0.25],
            [-0.25, 0.25],
            [0.25, -0.25],
            [0.25, 0.25],
        ],
        device=device,
        dtype=dtype,
    )
    candidates = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    candidates[:, 0].clamp_(0.0, float(max(0, feature_width - 1)))
    candidates[:, 1].clamp_(0.0, float(max(0, feature_height - 1)))
    if max_candidates > 0 and candidates.size(0) > max_candidates:
        order = torch.linspace(0, candidates.size(0) - 1, steps=max_candidates, device=device)
        candidates = candidates.index_select(0, order.round().to(torch.long).unique(sorted=True))
    return candidates.contiguous()


def _selected_keypoint_forward_targets(
    pair: SyntheticPair,
    sparse_maps_a: TrainingSparseMaps,
    sparse_maps_b: TrainingSparseMaps,
    *,
    max_points: int,
    min_intensity: float,
) -> torch.Tensor:
    keypoints_a, _ = _descriptor_keypoints_for_online_false_mining(
        pair.view_a,
        sparse_maps_a.descriptors,
        max_keypoints=max(1, int(max_points)),
        min_intensity=min_intensity,
        keypoint_scores=sparse_maps_a.heatmap,
    )
    _, targets_b = _warp_feature_points_a_to_b(
        pair,
        keypoints_a,
        feature_height_a=sparse_maps_a.keypoint_offsets.size(2),
        feature_width_a=sparse_maps_a.keypoint_offsets.size(3),
        feature_height_b=sparse_maps_b.keypoint_offsets.size(2),
        feature_width_b=sparse_maps_b.keypoint_offsets.size(3),
    )
    return targets_b


def _selected_keypoint_reverse_targets(
    pair: SyntheticPair,
    sparse_maps_a: TrainingSparseMaps,
    sparse_maps_b: TrainingSparseMaps,
    *,
    max_points: int,
    min_intensity: float,
    inverse_radius_px: float,
) -> torch.Tensor:
    keypoints_b, _ = _descriptor_keypoints_for_online_false_mining(
        pair.view_b,
        sparse_maps_b.descriptors,
        max_keypoints=max(1, int(max_points)),
        min_intensity=min_intensity,
        keypoint_scores=sparse_maps_b.heatmap,
    )
    if keypoints_b.numel() == 0:
        return sparse_maps_a.keypoint_offsets.new_empty((0, 2))
    max_candidates = max(4096, max(1, int(max_points)) * 256)
    candidate_a = _subcell_feature_grid(
        feature_height=sparse_maps_a.keypoint_offsets.size(2),
        feature_width=sparse_maps_a.keypoint_offsets.size(3),
        device=sparse_maps_a.keypoint_offsets.device,
        dtype=torch.float32,
        max_candidates=max_candidates,
    )
    candidate_a, candidate_b = _warp_feature_points_a_to_b(
        pair,
        candidate_a,
        feature_height_a=sparse_maps_a.keypoint_offsets.size(2),
        feature_width_a=sparse_maps_a.keypoint_offsets.size(3),
        feature_height_b=sparse_maps_b.keypoint_offsets.size(2),
        feature_width_b=sparse_maps_b.keypoint_offsets.size(3),
    )
    if candidate_a.numel() == 0 or candidate_b.numel() == 0:
        return sparse_maps_a.keypoint_offsets.new_empty((0, 2))
    distances = torch.cdist(keypoints_b.to(candidate_b.device, torch.float32), candidate_b.to(torch.float32))
    nearest_distance, nearest_index = distances.min(dim=1)
    keep = nearest_distance <= float(inverse_radius_px)
    if not bool(keep.any()):
        return sparse_maps_a.keypoint_offsets.new_empty((0, 2))
    selected_indices = nearest_index.index_select(0, torch.nonzero(keep, as_tuple=False).reshape(-1))
    return candidate_a.index_select(0, selected_indices).contiguous()


def selected_keypoint_offset_supervision_loss(
    pair: SyntheticPair,
    sparse_maps_a: TrainingSparseMaps,
    sparse_maps_b: TrainingSparseMaps,
    *,
    max_points: int,
    min_intensity: float,
    inverse_radius_px: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if max_points < 0:
        raise ValueError("max_points must be nonnegative")
    if inverse_radius_px < 0.0:
        raise ValueError("inverse_radius_px must be nonnegative")
    zero = sparse_maps_a.keypoint_offsets.sum() * 0.0 + sparse_maps_b.keypoint_offsets.sum() * 0.0
    if max_points == 0:
        return zero, {
            "loss": zero.detach(),
            "points": zero.detach(),
            "forward_points": zero.detach(),
            "reverse_points": zero.detach(),
        }
    forward_targets_b = _selected_keypoint_forward_targets(
        pair,
        sparse_maps_a,
        sparse_maps_b,
        max_points=max_points,
        min_intensity=min_intensity,
    )
    reverse_targets_a = _selected_keypoint_reverse_targets(
        pair,
        sparse_maps_a,
        sparse_maps_b,
        max_points=max_points,
        min_intensity=min_intensity,
        inverse_radius_px=inverse_radius_px,
    )
    terms: list[torch.Tensor] = []
    if forward_targets_b.numel() > 0:
        terms.append(keypoint_offset_supervision_loss(sparse_maps_b.keypoint_offsets, forward_targets_b))
    if reverse_targets_a.numel() > 0:
        terms.append(keypoint_offset_supervision_loss(sparse_maps_a.keypoint_offsets, reverse_targets_a))
    loss = torch.stack(terms).mean() if terms else zero
    forward_points = sparse_maps_a.keypoint_offsets.new_tensor(float(forward_targets_b.size(0)))
    reverse_points = sparse_maps_a.keypoint_offsets.new_tensor(float(reverse_targets_a.size(0)))
    return loss, {
        "loss": loss.detach(),
        "points": (forward_points + reverse_points).detach(),
        "forward_points": forward_points.detach(),
        "reverse_points": reverse_points.detach(),
    }


def online_false_match_feature_correspondences(
    pair: SyntheticPair,
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    *,
    max_keypoints: int,
    max_matches: int,
    min_intensity: float,
    min_score: float,
    min_margin: float,
    threshold_px: float,
    max_points: int,
    generator: torch.Generator | None = None,
    keypoint_scores_a: torch.Tensor | None = None,
    keypoint_scores_b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if threshold_px < 0.0:
        raise ValueError("threshold_px must be non-negative")
    if max_points < 0:
        raise ValueError("max_points must be non-negative")
    keypoints_a, selected_a = _descriptor_keypoints_for_online_false_mining(
        pair.view_a,
        descriptors_a,
        max_keypoints=max_keypoints,
        min_intensity=min_intensity,
        keypoint_scores=keypoint_scores_a,
    )
    keypoints_b, selected_b = _descriptor_keypoints_for_online_false_mining(
        pair.view_b,
        descriptors_b,
        max_keypoints=max_keypoints,
        min_intensity=min_intensity,
        keypoint_scores=keypoint_scores_b,
    )
    rows_a = _descriptor_rows_at_indices(descriptors_a, selected_a)
    rows_b = _descriptor_rows_at_indices(descriptors_b, selected_b)
    matches, _ = _mutual_nearest_descriptor_matches(
        rows_a,
        rows_b,
        max_matches=max_matches,
        min_score=min_score,
        min_margin=min_margin,
    )
    if matches.numel() == 0:
        return keypoints_a.new_empty((0, 2)), keypoints_b.new_empty((0, 2))
    _, image_height_a, image_width_a = pair.view_a.shape
    _, image_height_b, image_width_b = pair.view_b.shape
    points_a_feature = keypoints_a.index_select(0, matches[:, 0].to(keypoints_a.device))
    points_b_feature = keypoints_b.index_select(0, matches[:, 1].to(keypoints_b.device))
    points_a_image = _scale_feature_to_image_grid(
        points_a_feature,
        feature_height=descriptors_a.size(2),
        feature_width=descriptors_a.size(3),
        image_height=image_height_a,
        image_width=image_width_a,
    )
    points_b_image = _scale_feature_to_image_grid(
        points_b_feature,
        feature_height=descriptors_b.size(2),
        feature_width=descriptors_b.size(3),
        image_height=image_height_b,
        image_width=image_width_b,
    )
    target_b = _sample_warp_points(pair.warp_a_to_b, points_a_image)
    errors = (target_b.to(points_b_image.device) - points_b_image).norm(dim=1)
    valid_source = _valid_source_mask(pair.valid_mask, points_a_image).to(errors.device)
    wrong = (~valid_source) | (~torch.isfinite(target_b).all(dim=1).to(errors.device)) | errors.gt(float(threshold_px))
    false_indices = torch.nonzero(wrong, as_tuple=False).reshape(-1)
    if false_indices.numel() == 0:
        return keypoints_a.new_empty((0, 2)), keypoints_b.new_empty((0, 2))
    take = false_indices.numel() if max_points <= 0 else min(max_points, false_indices.numel())
    if take < false_indices.numel():
        order = torch.randperm(false_indices.numel(), generator=generator, device=false_indices.device)[:take]
        false_indices = false_indices.index_select(0, order)
    return (
        points_a_feature.index_select(0, false_indices).contiguous(),
        points_b_feature.index_select(0, false_indices).contiguous(),
    )


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
    grid = _normalize_xy(points_xy, height, width).to(dtype=descriptor_map.dtype).view(1, -1, 1, 2)
    sampled = F.grid_sample(descriptor_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def sample_optional_map_values(feature_map: torch.Tensor | None, points_xy: torch.Tensor) -> torch.Tensor | None:
    if feature_map is None:
        return None
    return sample_descriptors(feature_map, points_xy)


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
    if mode == "calibrated":
        if adjusted.size(1) > 12:
            adjusted[:, 12:13] = 0.0
        if adjusted.size(1) > 14:
            adjusted[:, 14 : min(adjusted.size(1), 16)] = 0.0
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
    """Legacy entry point for paired descriptor similarity; now uses strict cosine only."""
    if desc_a.dim() != 2 or desc_b.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if desc_a.shape != desc_b.shape:
        raise ValueError("descriptor tensors must have the same shape")
    desc_a = normalize_descriptor_batch(desc_a)
    desc_b = normalize_descriptor_batch(desc_b)
    return (desc_a * desc_b).sum(dim=1)


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


def _limited_points(
    points_xy: torch.Tensor,
    *,
    max_points: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if max_points <= 0 or points_xy.size(0) <= max_points:
        return points_xy
    order = torch.randperm(points_xy.size(0), generator=generator, device=points_xy.device)[:max_points]
    return points_xy.index_select(0, order).contiguous()


def descriptor_consistency_loss(
    descriptors_reference: torch.Tensor,
    descriptors_changed_light: torch.Tensor,
    points_xy: torch.Tensor,
    *,
    max_points: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if descriptors_reference.shape != descriptors_changed_light.shape:
        raise ValueError("descriptor maps must have the same shape")
    if points_xy.dim() != 2 or points_xy.size(1) != 2:
        raise ValueError("points_xy must have shape Nx2")
    if points_xy.size(0) == 0:
        return descriptors_reference.new_zeros(())
    selected = _limited_points(points_xy, max_points=max_points, generator=generator)
    reference = normalize_descriptor_batch(sample_descriptors(descriptors_reference, selected))
    changed = normalize_descriptor_batch(sample_descriptors(descriptors_changed_light, selected))
    similarity = paired_cyclic_similarity(reference, changed)
    return (1.0 - similarity).clamp_min(0.0).pow(2).mean()


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
    metadata_mode: str = "calibrated",
    no_match_points: int = 0,
    no_match_weight: float = 0.0,
    no_match_min_distance: float = 4.0,
    assignment_weight: float = 0.0,
    accept_weight: float = 0.0,
    accept_negative_topk: int = 8,
    prune_ranking_weight: float = 0.0,
    prune_ranking_margin: float = 0.25,
    stop_confidence_weight: float = 0.0,
    stop_confidence_margin: float = 0.5,
    raw_preservation_weight: float = 0.0,
    raw_preservation_margin: float = 1.0,
    raw_preservation_raw_margin: float = 0.05,
    hard_negative_dustbin_weight: float = 0.0,
    hard_negative_dustbin_topk: int = 8,
    hard_negative_dustbin_margin: float = 0.25,
    hard_negative_dustbin_spatial_min_distance: float = 0.0,
    positive_dustbin_margin_weight: float = 0.0,
    positive_dustbin_margin: float = 0.0,
    true_match_margin_weight: float = 0.0,
    true_match_margin: float = 0.0,
    true_geometry_match_count_floor_weight: float = 0.0,
    true_geometry_match_count_floor_target_count: float | None = None,
    true_geometry_match_count_floor_threshold: float = 0.0,
    true_geometry_match_count_floor_margin: float = 0.0,
    final_false_match_weight: float = 0.0,
    mined_false_match_weight: float = 0.0,
    mined_false_match_loss_cap: float = 0.0,
    mined_false_match_reference_margin: float = -1.0,
    final_false_match_topk: int = 8,
    final_false_match_min_score: float = 0.0,
    final_false_match_margin: float = 0.25,
    final_false_match_spatial_min_distance: float = 0.0,
    raw_false_match_weight: float = 0.0,
    raw_false_match_topk: int = 8,
    raw_false_match_min_similarity: float = 0.75,
    raw_false_match_margin: float = 0.25,
    raw_false_match_spatial_min_distance: float = 0.0,
    ransac_consistency_weight: float = 0.0,
    ransac_consistency_topk: int = 8,
    ransac_consistency_residual_threshold_px: float = 3.0,
    ransac_consistency_min_score: float = 0.0,
    ransac_consistency_margin: float = 0.25,
    warp_outlier_weight: float = 0.0,
    warp_outlier_topk: int = 8,
    warp_outlier_residual_threshold_px: float = 3.0,
    warp_outlier_min_score: float = 0.0,
    warp_outlier_margin: float = 0.25,
    warp_outlier_accept_weight: float = 0.0,
    warp_outlier_accept_topk: int = 8,
    warp_outlier_accept_residual_threshold_px: float = 3.0,
    warp_outlier_accept_min_score: float = 0.0,
    warp_soft_boundary_weight: float = 0.0,
    warp_soft_boundary_topk: int = 8,
    warp_soft_boundary_lower_residual_px: float = 5.0,
    warp_soft_boundary_upper_residual_px: float = 8.0,
    warp_soft_boundary_min_score: float = 0.0,
    pair_acceptance_target: float | None = None,
    pair_acceptance_weight: float = 1.0,
    pair_acceptance_loss_weight: float = 0.0,
    train_candidate_topk: int = 0,
    semi_dense_no_match_points: int = 0,
    semi_dense_min_score: float = 0.0,
    extra_no_match_points_a_xy: torch.Tensor | None = None,
    extra_no_match_points_b_xy: torch.Tensor | None = None,
    extra_false_match_points_a_xy: torch.Tensor | None = None,
    extra_false_match_points_b_xy: torch.Tensor | None = None,
    max_attention_layers: int = 0,
    random_attention_layers: bool = False,
    max_attention_work_fraction: float = 1.0,
    width_keep_ratio: float = 1.0,
    deep_supervision_depths: list[int] | tuple[int, ...] | None = None,
    deep_supervision_weight: float = 0.0,
    depth_distillation_weight: float = 0.0,
    depth_distillation_teacher_layers: int = 0,
    depth_distillation_temperature: float = 1.0,
    teacher_guard_output: pfm_model.GraphMatcherOutput | None = None,
    teacher_guard_model: pfm_model.PlanetaryFeatureMatcher | None = None,
    teacher_guard_weight: float = 0.0,
    teacher_guard_positive_margin_tolerance: float = 0.0,
    teacher_guard_false_margin_tolerance: float = 0.0,
    teacher_score_floor_weight: float = 0.0,
    teacher_score_floor_tolerance: float = 0.0,
    teacher_score_floor_min_score: float = 0.0,
    teacher_match_count_floor_weight: float = 0.0,
    teacher_match_count_floor_threshold: float = 0.0,
    teacher_match_count_floor_margin: float = 0.0,
    teacher_match_count_ceiling_weight: float = 0.0,
    teacher_match_count_ceiling_threshold: float = 0.0,
    teacher_match_count_ceiling_margin: float = 0.0,
    teacher_distillation_weight: float = 0.0,
    teacher_distillation_temperature: float = 1.0,
    positive_dustbin_guard_reject_threshold: float = 1.1,
    positive_dustbin_guard_margin_threshold: float = -float("inf"),
    matchability_a: torch.Tensor | None = None,
    matchability_b: torch.Tensor | None = None,
    descriptor_uncertainty_a: torch.Tensor | None = None,
    descriptor_uncertainty_b: torch.Tensor | None = None,
    no_match_prior_a: torch.Tensor | None = None,
    no_match_prior_b: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def components(
        total_loss: torch.Tensor,
        ce_loss: torch.Tensor,
        assignment_loss: torch.Tensor,
        no_match_loss: torch.Tensor,
        accept_loss: torch.Tensor,
        prune_ranking_loss: torch.Tensor,
        stop_confidence_loss: torch.Tensor,
        raw_preservation_loss: torch.Tensor,
        hard_negative_dustbin_loss: torch.Tensor,
        positive_dustbin_margin_loss: torch.Tensor,
        true_match_margin_loss: torch.Tensor,
        true_match_margin_violations: torch.Tensor | None,
        true_match_margin_mean: torch.Tensor | None,
        true_geometry_match_count_floor_loss: torch.Tensor | None,
        true_geometry_match_count_floor_metrics: dict[str, torch.Tensor] | None,
        final_false_loss: torch.Tensor | None,
        final_false_edges: torch.Tensor | None,
        final_false_score_mean: torch.Tensor | None,
        final_false_accept_mean: torch.Tensor | None,
        mined_false_loss: torch.Tensor | None,
        mined_false_edges: torch.Tensor | None,
        mined_false_reference_filtered_edges: torch.Tensor | None,
        mined_false_score_mean: torch.Tensor | None,
        mined_false_logit_mean: torch.Tensor | None,
        mined_false_accept_mean: torch.Tensor | None,
        raw_false_loss: torch.Tensor | None,
        raw_false_edges: torch.Tensor | None,
        raw_false_similarity_mean: torch.Tensor | None,
        raw_false_margin_mean: torch.Tensor | None,
        deep_supervision_loss: torch.Tensor,
        executed_attention_layers: torch.Tensor,
        attention_work_fraction: torch.Tensor,
        positive_pairs: torch.Tensor,
        extra_no_match_points: torch.Tensor,
        extra_false_match_pairs: torch.Tensor,
        dustbin_diagnostics: dict[str, float] | None = None,
        depth_distillation_loss: torch.Tensor | None = None,
        depth_distillation_teacher_layers_tensor: torch.Tensor | None = None,
        teacher_distillation_loss: torch.Tensor | None = None,
        dustbin_guard_active: torch.Tensor | None = None,
        guarded_no_match_weight: torch.Tensor | None = None,
        guarded_hard_negative_dustbin_weight: torch.Tensor | None = None,
        candidate_topk_diagnostics: dict[str, float] | None = None,
        teacher_guard_loss: torch.Tensor | None = None,
        teacher_guard_metrics: dict[str, torch.Tensor] | None = None,
        teacher_score_floor_loss: torch.Tensor | None = None,
        teacher_score_floor_metrics: dict[str, torch.Tensor] | None = None,
        teacher_match_count_floor_loss: torch.Tensor | None = None,
        teacher_match_count_floor_metrics: dict[str, torch.Tensor] | None = None,
        teacher_match_count_ceiling_loss: torch.Tensor | None = None,
        teacher_match_count_ceiling_metrics: dict[str, torch.Tensor] | None = None,
        ransac_consistency_loss: torch.Tensor | None = None,
        ransac_consistency_metrics: dict[str, torch.Tensor] | None = None,
        warp_outlier_loss: torch.Tensor | None = None,
        warp_outlier_metrics: dict[str, torch.Tensor] | None = None,
        warp_outlier_accept_loss: torch.Tensor | None = None,
        warp_outlier_accept_metrics: dict[str, torch.Tensor] | None = None,
        warp_soft_boundary_loss: torch.Tensor | None = None,
        warp_soft_boundary_metrics: dict[str, torch.Tensor] | None = None,
        pair_acceptance_loss: torch.Tensor | None = None,
        pair_acceptance_metrics: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        result = {
            "graph_matcher_total_loss": total_loss,
            "graph_matcher_ce_loss": ce_loss,
            "graph_matcher_assignment_loss": assignment_loss,
            "graph_matcher_no_match_loss": no_match_loss,
            "graph_matcher_accept_loss": accept_loss,
            "graph_matcher_prune_ranking_loss": prune_ranking_loss,
            "graph_matcher_stop_confidence_loss": stop_confidence_loss,
            "graph_matcher_raw_preservation_loss": raw_preservation_loss,
            "graph_matcher_hard_negative_dustbin_loss": hard_negative_dustbin_loss,
            "graph_matcher_positive_dustbin_margin_loss": positive_dustbin_margin_loss,
            "graph_matcher_true_match_margin_loss": true_match_margin_loss,
            "graph_matcher_true_match_margin_violations": (
                total_loss.new_zeros(()) if true_match_margin_violations is None else true_match_margin_violations
            ),
            "graph_matcher_true_match_margin_mean": (
                total_loss.new_zeros(()) if true_match_margin_mean is None else true_match_margin_mean
            ),
            "graph_matcher_true_geometry_match_count_floor_loss": (
                total_loss.new_zeros(())
                if true_geometry_match_count_floor_loss is None
                else true_geometry_match_count_floor_loss
            ),
            "graph_matcher_true_geometry_match_count_floor_target_count": (
                total_loss.new_zeros(())
                if true_geometry_match_count_floor_metrics is None
                else true_geometry_match_count_floor_metrics["target_count"]
            ),
            "graph_matcher_true_geometry_match_count_floor_student_count": (
                total_loss.new_zeros(())
                if true_geometry_match_count_floor_metrics is None
                else true_geometry_match_count_floor_metrics["student_count"]
            ),
            "graph_matcher_true_geometry_match_count_floor_count_deficit": (
                total_loss.new_zeros(())
                if true_geometry_match_count_floor_metrics is None
                else true_geometry_match_count_floor_metrics["count_deficit"]
            ),
            "graph_matcher_true_geometry_match_count_floor_topk_score_mean": (
                total_loss.new_zeros(())
                if true_geometry_match_count_floor_metrics is None
                else true_geometry_match_count_floor_metrics["topk_score_mean"]
            ),
            "graph_matcher_true_geometry_match_count_floor_violations": (
                total_loss.new_zeros(())
                if true_geometry_match_count_floor_metrics is None
                else true_geometry_match_count_floor_metrics["violations"]
            ),
            "graph_matcher_final_false_match_loss": (
                total_loss.new_zeros(()) if final_false_loss is None else final_false_loss
            ),
            "graph_matcher_final_false_match_edges": (
                total_loss.new_zeros(()) if final_false_edges is None else final_false_edges
            ),
            "graph_matcher_final_false_match_score_mean": (
                total_loss.new_zeros(()) if final_false_score_mean is None else final_false_score_mean
            ),
            "graph_matcher_final_false_match_accept_mean": (
                total_loss.new_zeros(()) if final_false_accept_mean is None else final_false_accept_mean
            ),
            "graph_matcher_mined_false_match_loss": (
                total_loss.new_zeros(()) if mined_false_loss is None else mined_false_loss
            ),
            "graph_matcher_mined_false_match_edges": (
                total_loss.new_zeros(()) if mined_false_edges is None else mined_false_edges
            ),
            "graph_matcher_mined_false_match_reference_filtered_edges": (
                total_loss.new_zeros(())
                if mined_false_reference_filtered_edges is None
                else mined_false_reference_filtered_edges
            ),
            "graph_matcher_mined_false_match_score_mean": (
                total_loss.new_zeros(()) if mined_false_score_mean is None else mined_false_score_mean
            ),
            "graph_matcher_mined_false_match_logit_mean": (
                total_loss.new_zeros(()) if mined_false_logit_mean is None else mined_false_logit_mean
            ),
            "graph_matcher_mined_false_match_accept_mean": (
                total_loss.new_zeros(()) if mined_false_accept_mean is None else mined_false_accept_mean
            ),
            "graph_matcher_raw_false_match_loss": (
                total_loss.new_zeros(()) if raw_false_loss is None else raw_false_loss
            ),
            "graph_matcher_raw_false_match_edges": (
                total_loss.new_zeros(()) if raw_false_edges is None else raw_false_edges
            ),
            "graph_matcher_raw_false_match_similarity_mean": (
                total_loss.new_zeros(()) if raw_false_similarity_mean is None else raw_false_similarity_mean
            ),
            "graph_matcher_raw_false_match_margin_mean": (
                total_loss.new_zeros(()) if raw_false_margin_mean is None else raw_false_margin_mean
            ),
            "graph_matcher_ransac_consistency_loss": (
                total_loss.new_zeros(()) if ransac_consistency_loss is None else ransac_consistency_loss
            ),
            "graph_matcher_ransac_consistency_edges": (
                total_loss.new_zeros(())
                if ransac_consistency_metrics is None
                else ransac_consistency_metrics["edges"]
            ),
            "graph_matcher_ransac_consistency_score_mean": (
                total_loss.new_zeros(())
                if ransac_consistency_metrics is None
                else ransac_consistency_metrics["score_mean"]
            ),
            "graph_matcher_ransac_consistency_residual_mean_px": (
                total_loss.new_zeros(())
                if ransac_consistency_metrics is None
                else ransac_consistency_metrics["residual_mean_px"]
            ),
            "graph_matcher_ransac_consistency_accept_mean": (
                total_loss.new_zeros(())
                if ransac_consistency_metrics is None
                else ransac_consistency_metrics["accept_mean"]
            ),
            "graph_matcher_warp_outlier_loss": (
                total_loss.new_zeros(()) if warp_outlier_loss is None else warp_outlier_loss
            ),
            "graph_matcher_warp_outlier_edges": (
                total_loss.new_zeros(())
                if warp_outlier_metrics is None
                else warp_outlier_metrics["edges"]
            ),
            "graph_matcher_warp_outlier_residual_mean_px": (
                total_loss.new_zeros(())
                if warp_outlier_metrics is None
                else warp_outlier_metrics["residual_mean_px"]
            ),
            "graph_matcher_warp_outlier_accept_mean": (
                total_loss.new_zeros(())
                if warp_outlier_metrics is None
                else warp_outlier_metrics["accept_mean"]
            ),
            "graph_matcher_warp_outlier_accept_loss": (
                total_loss.new_zeros(()) if warp_outlier_accept_loss is None else warp_outlier_accept_loss
            ),
            "graph_matcher_warp_outlier_accept_edges": (
                total_loss.new_zeros(())
                if warp_outlier_accept_metrics is None
                else warp_outlier_accept_metrics["edges"]
            ),
            "graph_matcher_warp_outlier_accept_score_mean": (
                total_loss.new_zeros(())
                if warp_outlier_accept_metrics is None
                else warp_outlier_accept_metrics["score_mean"]
            ),
            "graph_matcher_warp_outlier_accept_residual_mean_px": (
                total_loss.new_zeros(())
                if warp_outlier_accept_metrics is None
                else warp_outlier_accept_metrics["residual_mean_px"]
            ),
            "graph_matcher_warp_outlier_accept_probability_mean": (
                total_loss.new_zeros(())
                if warp_outlier_accept_metrics is None
                else warp_outlier_accept_metrics["probability_mean"]
            ),
            "graph_matcher_warp_soft_boundary_loss": (
                total_loss.new_zeros(()) if warp_soft_boundary_loss is None else warp_soft_boundary_loss
            ),
            "graph_matcher_warp_soft_boundary_edges": (
                total_loss.new_zeros(())
                if warp_soft_boundary_metrics is None
                else warp_soft_boundary_metrics["edges"]
            ),
            "graph_matcher_warp_soft_boundary_residual_mean_px": (
                total_loss.new_zeros(())
                if warp_soft_boundary_metrics is None
                else warp_soft_boundary_metrics["residual_mean_px"]
            ),
            "graph_matcher_warp_soft_boundary_target_mean": (
                total_loss.new_zeros(())
                if warp_soft_boundary_metrics is None
                else warp_soft_boundary_metrics["target_mean"]
            ),
            "graph_matcher_warp_soft_boundary_score_probability_mean": (
                total_loss.new_zeros(())
                if warp_soft_boundary_metrics is None
                else warp_soft_boundary_metrics["score_probability_mean"]
            ),
            "graph_matcher_warp_soft_boundary_accept_probability_mean": (
                total_loss.new_zeros(())
                if warp_soft_boundary_metrics is None
                else warp_soft_boundary_metrics["accept_probability_mean"]
            ),
            "graph_matcher_pair_acceptance_loss": (
                total_loss.new_zeros(()) if pair_acceptance_loss is None else pair_acceptance_loss
            ),
            "graph_matcher_pair_acceptance_target": total_loss.new_tensor(
                float((pair_acceptance_metrics or {}).get("target", 0.0))
            ),
            "graph_matcher_pair_acceptance_weight": total_loss.new_tensor(
                float((pair_acceptance_metrics or {}).get("weight", 0.0))
            ),
            "graph_matcher_pair_acceptance_probability": total_loss.new_tensor(
                float((pair_acceptance_metrics or {}).get("probability", 0.0))
            ),
            "graph_matcher_deep_supervision_loss": deep_supervision_loss,
            "graph_matcher_depth_distillation_loss": (
                total_loss.new_zeros(()) if depth_distillation_loss is None else depth_distillation_loss
            ),
            "graph_matcher_depth_distillation_teacher_layers": (
                total_loss.new_zeros(())
                if depth_distillation_teacher_layers_tensor is None
                else depth_distillation_teacher_layers_tensor
            ),
            "graph_matcher_teacher_distillation_loss": (
                total_loss.new_zeros(()) if teacher_distillation_loss is None else teacher_distillation_loss
            ),
            "graph_matcher_teacher_guard_loss": (
                total_loss.new_zeros(()) if teacher_guard_loss is None else teacher_guard_loss
            ),
            "graph_matcher_teacher_guard_positive_margin_loss": (
                total_loss.new_zeros(())
                if teacher_guard_metrics is None
                else teacher_guard_metrics["positive_margin_loss"]
            ),
            "graph_matcher_teacher_guard_false_edge_loss": (
                total_loss.new_zeros(())
                if teacher_guard_metrics is None
                else teacher_guard_metrics["false_edge_loss"]
            ),
            "graph_matcher_teacher_guard_positive_violations": (
                total_loss.new_zeros(())
                if teacher_guard_metrics is None
                else teacher_guard_metrics["positive_violations"]
            ),
            "graph_matcher_teacher_guard_false_edges": (
                total_loss.new_zeros(())
                if teacher_guard_metrics is None
                else teacher_guard_metrics["false_edges"]
            ),
            "graph_matcher_teacher_score_floor_loss": (
                total_loss.new_zeros(())
                if teacher_score_floor_loss is None
                else teacher_score_floor_loss
            ),
            "graph_matcher_teacher_score_floor_violations": (
                total_loss.new_zeros(())
                if teacher_score_floor_metrics is None
                else teacher_score_floor_metrics["violations"]
            ),
            "graph_matcher_teacher_score_floor_delta_mean": (
                total_loss.new_zeros(())
                if teacher_score_floor_metrics is None
                else teacher_score_floor_metrics["score_delta_mean"]
            ),
            "graph_matcher_teacher_score_floor_teacher_score_mean": (
                total_loss.new_zeros(())
                if teacher_score_floor_metrics is None
                else teacher_score_floor_metrics["teacher_score_mean"]
            ),
            "graph_matcher_teacher_match_count_floor_loss": (
                total_loss.new_zeros(())
                if teacher_match_count_floor_loss is None
                else teacher_match_count_floor_loss
            ),
            "graph_matcher_teacher_match_count_floor_teacher_count": (
                total_loss.new_zeros(())
                if teacher_match_count_floor_metrics is None
                else teacher_match_count_floor_metrics["teacher_count"]
            ),
            "graph_matcher_teacher_match_count_floor_student_count": (
                total_loss.new_zeros(())
                if teacher_match_count_floor_metrics is None
                else teacher_match_count_floor_metrics["student_count"]
            ),
            "graph_matcher_teacher_match_count_floor_count_deficit": (
                total_loss.new_zeros(())
                if teacher_match_count_floor_metrics is None
                else teacher_match_count_floor_metrics["count_deficit"]
            ),
            "graph_matcher_teacher_match_count_floor_topk_score_mean": (
                total_loss.new_zeros(())
                if teacher_match_count_floor_metrics is None
                else teacher_match_count_floor_metrics["topk_score_mean"]
            ),
            "graph_matcher_teacher_match_count_ceiling_loss": (
                total_loss.new_zeros(())
                if teacher_match_count_ceiling_loss is None
                else teacher_match_count_ceiling_loss
            ),
            "graph_matcher_teacher_match_count_ceiling_teacher_count": (
                total_loss.new_zeros(())
                if teacher_match_count_ceiling_metrics is None
                else teacher_match_count_ceiling_metrics["teacher_count"]
            ),
            "graph_matcher_teacher_match_count_ceiling_student_count": (
                total_loss.new_zeros(())
                if teacher_match_count_ceiling_metrics is None
                else teacher_match_count_ceiling_metrics["student_count"]
            ),
            "graph_matcher_teacher_match_count_ceiling_count_excess": (
                total_loss.new_zeros(())
                if teacher_match_count_ceiling_metrics is None
                else teacher_match_count_ceiling_metrics["count_excess"]
            ),
            "graph_matcher_teacher_match_count_ceiling_excess_score_mean": (
                total_loss.new_zeros(())
                if teacher_match_count_ceiling_metrics is None
                else teacher_match_count_ceiling_metrics["excess_score_mean"]
            ),
            "graph_matcher_executed_attention_layers": executed_attention_layers,
            "graph_matcher_attention_work_fraction": attention_work_fraction,
            "graph_matcher_positive_pairs": positive_pairs,
            "graph_matcher_extra_no_match_points": extra_no_match_points,
            "graph_matcher_extra_false_match_pairs": extra_false_match_pairs,
            "graph_matcher_train_candidate_topk": total_loss.new_tensor(float(train_candidate_topk)),
            "graph_matcher_dustbin_guard_active": (
                total_loss.new_zeros(()) if dustbin_guard_active is None else dustbin_guard_active
            ),
            "graph_matcher_guarded_no_match_weight": (
                total_loss.new_tensor(float(no_match_weight))
                if guarded_no_match_weight is None
                else guarded_no_match_weight
            ),
            "graph_matcher_guarded_hard_negative_dustbin_weight": (
                total_loss.new_tensor(float(hard_negative_dustbin_weight))
                if guarded_hard_negative_dustbin_weight is None
                else guarded_hard_negative_dustbin_weight
            ),
        }
        for key in (
            "true_match_rejected_by_dustbin_ratio",
            "positive_pair_logit_mean",
            "positive_dustbin_logit_mean",
            "dustbin_logit_mean",
            "dustbin_logit_for_true_match_mean",
            "positive_vs_dustbin_margin_mean",
            "positive_vs_dustbin_margin_median",
            "positive_vs_dustbin_margin_p10",
            "positive_vs_dustbin_margin_below0_ratio",
            "false_match_accepted_ratio",
            "accept_logit_mean",
            "true_pair_prob_mean",
            "dustbin_prob_for_true_match_mean",
        ):
            result[key] = total_loss.new_tensor(float((dustbin_diagnostics or {}).get(key, 0.0)))
        for key in ("true_match_in_topk@64", "true_match_in_topk@256"):
            result[key] = total_loss.new_tensor(float((candidate_topk_diagnostics or {}).get(key, 0.0)))
        return result

    if (
        not math.isfinite(float(max_attention_work_fraction))
        or max_attention_work_fraction < 0.0
        or max_attention_work_fraction > 1.0
    ):
        raise ValueError("max_attention_work_fraction must be in [0, 1]")
    if not math.isfinite(float(width_keep_ratio)) or width_keep_ratio <= 0.0 or width_keep_ratio > 1.0:
        raise ValueError("width_keep_ratio must be in (0, 1]")
    if deep_supervision_weight < 0.0:
        raise ValueError("deep_supervision_weight must be nonnegative")
    if depth_distillation_weight < 0.0:
        raise ValueError("depth_distillation_weight must be nonnegative")
    if depth_distillation_teacher_layers < 0:
        raise ValueError("depth_distillation_teacher_layers must be nonnegative")
    if not math.isfinite(float(depth_distillation_temperature)) or depth_distillation_temperature <= 0.0:
        raise ValueError("depth_distillation_temperature must be positive and finite")
    if teacher_guard_weight < 0.0:
        raise ValueError("teacher_guard_weight must be nonnegative")
    if teacher_score_floor_weight < 0.0:
        raise ValueError("teacher_score_floor_weight must be nonnegative")
    if teacher_match_count_floor_weight < 0.0:
        raise ValueError("teacher_match_count_floor_weight must be nonnegative")
    if teacher_match_count_ceiling_weight < 0.0:
        raise ValueError("teacher_match_count_ceiling_weight must be nonnegative")
    if teacher_distillation_weight < 0.0:
        raise ValueError("teacher_distillation_weight must be nonnegative")
    if not math.isfinite(float(teacher_distillation_temperature)) or teacher_distillation_temperature <= 0.0:
        raise ValueError("teacher_distillation_temperature must be positive and finite")
    if teacher_guard_positive_margin_tolerance < 0.0:
        raise ValueError("teacher_guard_positive_margin_tolerance must be nonnegative")
    if teacher_guard_false_margin_tolerance < 0.0:
        raise ValueError("teacher_guard_false_margin_tolerance must be nonnegative")
    if teacher_score_floor_tolerance < 0.0:
        raise ValueError("teacher_score_floor_tolerance must be nonnegative")
    if not math.isfinite(float(teacher_score_floor_min_score)):
        raise ValueError("teacher_score_floor_min_score must be finite")
    if not math.isfinite(float(teacher_match_count_floor_threshold)):
        raise ValueError("teacher_match_count_floor_threshold must be finite")
    if teacher_match_count_floor_margin < 0.0:
        raise ValueError("teacher_match_count_floor_margin must be nonnegative")
    if not math.isfinite(float(teacher_match_count_ceiling_threshold)):
        raise ValueError("teacher_match_count_ceiling_threshold must be finite")
    if teacher_match_count_ceiling_margin < 0.0:
        raise ValueError("teacher_match_count_ceiling_margin must be nonnegative")
    if (
        teacher_guard_model is not None
        and teacher_guard_model.config.graph_keypoint_meta_dim != model.config.graph_keypoint_meta_dim
    ):
        raise ValueError("teacher_guard_model graph_keypoint_meta_dim must match the student model")
    if not math.isfinite(float(positive_dustbin_guard_reject_threshold)):
        raise ValueError("positive_dustbin_guard_reject_threshold must be finite")
    if positive_dustbin_margin_weight < 0.0:
        raise ValueError("positive_dustbin_margin_weight must be nonnegative")
    if true_match_margin_weight < 0.0:
        raise ValueError("true_match_margin_weight must be nonnegative")
    if true_match_margin < 0.0:
        raise ValueError("true_match_margin must be nonnegative")
    if true_geometry_match_count_floor_weight < 0.0:
        raise ValueError("true_geometry_match_count_floor_weight must be nonnegative")
    if true_geometry_match_count_floor_target_count is not None and (
        not math.isfinite(float(true_geometry_match_count_floor_target_count))
        or true_geometry_match_count_floor_target_count < 0.0
    ):
        raise ValueError("true_geometry_match_count_floor_target_count must be finite and nonnegative")
    if not math.isfinite(float(true_geometry_match_count_floor_threshold)):
        raise ValueError("true_geometry_match_count_floor_threshold must be finite")
    if true_geometry_match_count_floor_margin < 0.0:
        raise ValueError("true_geometry_match_count_floor_margin must be nonnegative")
    if final_false_match_weight < 0.0:
        raise ValueError("final_false_match_weight must be nonnegative")
    if mined_false_match_weight < 0.0:
        raise ValueError("mined_false_match_weight must be nonnegative")
    if mined_false_match_loss_cap < 0.0:
        raise ValueError("mined_false_match_loss_cap must be nonnegative")
    if (
        not math.isfinite(float(mined_false_match_reference_margin))
        or mined_false_match_reference_margin < -1.0
    ):
        raise ValueError("mined_false_match_reference_margin must be finite and >= -1")
    if final_false_match_topk < 0:
        raise ValueError("final_false_match_topk must be nonnegative")
    if final_false_match_min_score < 0.0:
        raise ValueError("final_false_match_min_score must be nonnegative")
    if final_false_match_margin < 0.0:
        raise ValueError("final_false_match_margin must be nonnegative")
    if final_false_match_spatial_min_distance < 0.0:
        raise ValueError("final_false_match_spatial_min_distance must be nonnegative")
    if raw_false_match_weight < 0.0:
        raise ValueError("raw_false_match_weight must be nonnegative")
    if raw_false_match_topk < 0:
        raise ValueError("raw_false_match_topk must be nonnegative")
    if raw_false_match_min_similarity < -1.0 or raw_false_match_min_similarity > 1.0:
        raise ValueError("raw_false_match_min_similarity must be in [-1, 1]")
    if raw_false_match_margin < 0.0:
        raise ValueError("raw_false_match_margin must be nonnegative")
    if raw_false_match_spatial_min_distance < 0.0:
        raise ValueError("raw_false_match_spatial_min_distance must be nonnegative")
    if ransac_consistency_weight < 0.0:
        raise ValueError("ransac_consistency_weight must be nonnegative")
    if ransac_consistency_topk < 0:
        raise ValueError("ransac_consistency_topk must be nonnegative")
    if ransac_consistency_residual_threshold_px < 0.0:
        raise ValueError("ransac_consistency_residual_threshold_px must be nonnegative")
    if ransac_consistency_min_score < 0.0:
        raise ValueError("ransac_consistency_min_score must be nonnegative")
    if ransac_consistency_margin < 0.0:
        raise ValueError("ransac_consistency_margin must be nonnegative")
    if warp_outlier_weight < 0.0:
        raise ValueError("warp_outlier_weight must be nonnegative")
    if warp_outlier_topk < 0:
        raise ValueError("warp_outlier_topk must be nonnegative")
    if warp_outlier_residual_threshold_px < 0.0:
        raise ValueError("warp_outlier_residual_threshold_px must be nonnegative")
    if warp_outlier_min_score < 0.0:
        raise ValueError("warp_outlier_min_score must be nonnegative")
    if warp_outlier_margin < 0.0:
        raise ValueError("warp_outlier_margin must be nonnegative")
    if warp_outlier_accept_weight < 0.0:
        raise ValueError("warp_outlier_accept_weight must be nonnegative")
    if warp_outlier_accept_topk < 0:
        raise ValueError("warp_outlier_accept_topk must be nonnegative")
    if warp_outlier_accept_residual_threshold_px < 0.0:
        raise ValueError("warp_outlier_accept_residual_threshold_px must be nonnegative")
    if warp_outlier_accept_min_score < 0.0:
        raise ValueError("warp_outlier_accept_min_score must be nonnegative")
    if warp_soft_boundary_weight < 0.0:
        raise ValueError("warp_soft_boundary_weight must be nonnegative")
    if warp_soft_boundary_topk < 0:
        raise ValueError("warp_soft_boundary_topk must be nonnegative")
    if (
        not math.isfinite(float(warp_soft_boundary_lower_residual_px))
        or warp_soft_boundary_lower_residual_px < 0.0
    ):
        raise ValueError("warp_soft_boundary_lower_residual_px must be finite and nonnegative")
    if (
        not math.isfinite(float(warp_soft_boundary_upper_residual_px))
        or warp_soft_boundary_upper_residual_px <= warp_soft_boundary_lower_residual_px
    ):
        raise ValueError("warp_soft_boundary_upper_residual_px must be finite and greater than lower")
    if warp_soft_boundary_min_score < 0.0:
        raise ValueError("warp_soft_boundary_min_score must be nonnegative")
    if pair_acceptance_weight <= 0.0 or not math.isfinite(float(pair_acceptance_weight)):
        raise ValueError("pair_acceptance_weight must be positive and finite")
    if pair_acceptance_loss_weight < 0.0 or not math.isfinite(float(pair_acceptance_loss_weight)):
        raise ValueError("pair_acceptance_loss_weight must be nonnegative and finite")
    if pair_acceptance_target is not None and float(pair_acceptance_target) not in (0.0, 1.0):
        raise ValueError("pair_acceptance_target must be 0, 1, or None")
    if train_candidate_topk < 0:
        raise ValueError("train_candidate_topk must be nonnegative")
    supervision_depths = [int(depth) for depth in (deep_supervision_depths or [])]
    if any(depth <= 0 for depth in supervision_depths):
        raise ValueError("deep supervision depths must be positive")
    if points_a_xy.size(0) == 0 or points_b_xy.size(0) == 0:
        zero = descriptors_a.new_tensor(0.0)
        if return_components:
            return zero, components(
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
            )
        return zero
    count = min(points_a_xy.size(0), points_b_xy.size(0))
    points_a_xy = points_a_xy[:count]
    points_b_xy = points_b_xy[:count]
    if width_keep_ratio < 1.0 and count > 1:
        keep_count = max(1, min(count, int(math.ceil(float(count) * float(width_keep_ratio)))))
        if keep_count < count:
            keep_indices = torch.randperm(count, device=points_a_xy.device, generator=generator)[
                :keep_count
            ].sort().values
            points_a_xy = points_a_xy.index_select(0, keep_indices)
            points_b_xy = points_b_xy.index_select(0, keep_indices)
            count = keep_count
    positive_points_a_xy = points_a_xy
    positive_points_b_xy = points_b_xy
    desc_a = normalize_descriptor_batch(sample_descriptors(descriptors_a, points_a_xy))
    desc_b = normalize_descriptor_batch(sample_descriptors(descriptors_b, points_b_xy))
    if no_match_points > 0:
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
    if semi_dense_no_match_points > 0 and hasattr(model, "semi_dense_branch"):
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
    extra_no_match_count = 0
    if extra_no_match_points_a_xy is not None:
        if extra_no_match_points_a_xy.dim() != 2 or extra_no_match_points_a_xy.size(1) != 2:
            raise ValueError("extra_no_match_points_a_xy must have shape Nx2")
        extra_a = extra_no_match_points_a_xy.to(device=descriptors_a.device, dtype=points_a_xy.dtype)
        if extra_a.numel() > 0:
            desc_a = torch.cat(
                [desc_a, normalize_descriptor_batch(sample_descriptors(descriptors_a, extra_a))],
                dim=0,
            )
            points_a_xy = torch.cat([points_a_xy, extra_a.to(points_a_xy.device)], dim=0)
            extra_no_match_count += int(extra_a.size(0))
    if extra_no_match_points_b_xy is not None:
        if extra_no_match_points_b_xy.dim() != 2 or extra_no_match_points_b_xy.size(1) != 2:
            raise ValueError("extra_no_match_points_b_xy must have shape Nx2")
        extra_b = extra_no_match_points_b_xy.to(device=descriptors_b.device, dtype=points_b_xy.dtype)
        if extra_b.numel() > 0:
            desc_b = torch.cat(
                [desc_b, normalize_descriptor_batch(sample_descriptors(descriptors_b, extra_b))],
                dim=0,
            )
            points_b_xy = torch.cat([points_b_xy, extra_b.to(points_b_xy.device)], dim=0)
            extra_no_match_count += int(extra_b.size(0))
    extra_false_a_start = int(points_a_xy.size(0))
    extra_false_b_start = int(points_b_xy.size(0))
    extra_false_match_pair_count = 0
    if extra_false_match_points_a_xy is not None or extra_false_match_points_b_xy is not None:
        if extra_false_match_points_a_xy is None or extra_false_match_points_b_xy is None:
            raise ValueError("extra_false_match_points_a_xy and extra_false_match_points_b_xy must be provided together")
        if extra_false_match_points_a_xy.dim() != 2 or extra_false_match_points_a_xy.size(1) != 2:
            raise ValueError("extra_false_match_points_a_xy must have shape Nx2")
        if extra_false_match_points_b_xy.dim() != 2 or extra_false_match_points_b_xy.size(1) != 2:
            raise ValueError("extra_false_match_points_b_xy must have shape Nx2")
        if extra_false_match_points_a_xy.shape != extra_false_match_points_b_xy.shape:
            raise ValueError("extra_false_match_points tensors must have the same shape")
        false_a = extra_false_match_points_a_xy.to(device=descriptors_a.device, dtype=points_a_xy.dtype)
        false_b = extra_false_match_points_b_xy.to(device=descriptors_b.device, dtype=points_b_xy.dtype)
        if false_a.numel() > 0:
            desc_a = torch.cat(
                [desc_a, normalize_descriptor_batch(sample_descriptors(descriptors_a, false_a))],
                dim=0,
            )
            desc_b = torch.cat(
                [desc_b, normalize_descriptor_batch(sample_descriptors(descriptors_b, false_b))],
                dim=0,
            )
            points_a_xy = torch.cat([points_a_xy, false_a.to(points_a_xy.device)], dim=0)
            points_b_xy = torch.cat([points_b_xy, false_b.to(points_b_xy.device)], dim=0)
            extra_false_match_pair_count = int(false_a.size(0))
    meta_a = pfm_model.prepare_graph_keypoint_metadata(
        points_a_xy,
        meta_dim=model.config.graph_keypoint_meta_dim,
        matchability=sample_optional_map_values(matchability_a, points_a_xy),
        descriptor_uncertainty=sample_optional_map_values(descriptor_uncertainty_a, points_a_xy),
        no_match_prior=sample_optional_map_values(no_match_prior_a, points_a_xy),
    ).to(desc_a.device)
    meta_b = pfm_model.prepare_graph_keypoint_metadata(
        points_b_xy,
        meta_dim=model.config.graph_keypoint_meta_dim,
        matchability=sample_optional_map_values(matchability_b, points_b_xy),
        descriptor_uncertainty=sample_optional_map_values(descriptor_uncertainty_b, points_b_xy),
        no_match_prior=sample_optional_map_values(no_match_prior_b, points_b_xy),
    ).to(desc_b.device)
    meta_a = apply_graph_metadata_mode(meta_a, metadata_mode)
    meta_b = apply_graph_metadata_mode(meta_b, metadata_mode)
    candidate_topk_diagnostics = graph_matcher_candidate_topk_diagnostics(
        model,
        desc_a,
        desc_b,
        meta_a,
        meta_b,
        positive_count=count,
        topk_values=(64, 256),
    )
    attention_layer_budget = int(max_attention_layers)
    if random_attention_layers:
        layer_limit = attention_layer_budget if attention_layer_budget > 0 else len(model.graph_matcher.attention_layers)
        layer_limit = max(1, int(layer_limit))
        attention_layer_budget = int(
            torch.randint(
                1,
                layer_limit + 1,
                (1,),
                device=desc_a.device,
                generator=generator,
            ).item()
        )
    def run_matcher(layer_budget: int) -> pfm_model.GraphMatcherOutput:
        return model.graph_matcher(
            desc_a,
            meta_a,
            desc_b,
            meta_b,
            apply_candidate_mask=train_candidate_topk > 0,
            max_attention_layers=int(layer_budget),
            max_attention_work_fraction=max_attention_work_fraction,
            candidate_topk=train_candidate_topk if train_candidate_topk > 0 else None,
            positive_pair_count_for_mask=count if train_candidate_topk > 0 else 0,
        )

    def run_teacher_guard_matcher(layer_budget: int) -> pfm_model.GraphMatcherOutput | None:
        if teacher_guard_model is None:
            return None
        was_training = bool(teacher_guard_model.training)
        teacher_guard_model.eval()
        try:
            with torch.no_grad():
                return teacher_guard_model.graph_matcher(
                    desc_a.detach(),
                    meta_a.detach(),
                    desc_b.detach(),
                    meta_b.detach(),
                    apply_candidate_mask=train_candidate_topk > 0,
                    max_attention_layers=int(layer_budget),
                    max_attention_work_fraction=max_attention_work_fraction,
                    candidate_topk=train_candidate_topk if train_candidate_topk > 0 else None,
                    positive_pair_count_for_mask=count if train_candidate_topk > 0 else 0,
                )
        finally:
            if was_training:
                teacher_guard_model.train()

    def positive_ce(output: pfm_model.GraphMatcherOutput) -> torch.Tensor:
        targets = torch.arange(count, dtype=torch.long, device=output.logits.device)
        row_loss = F.cross_entropy(output.logits[:count, :], targets)
        col_loss = F.cross_entropy(output.logits[:, :count].T, targets)
        return 0.5 * (row_loss + col_loss)

    deep_supervision_loss = desc_a.new_zeros(())
    if deep_supervision_weight > 0.0 and supervision_depths:
        deep_terms: list[torch.Tensor] = []
        for depth in supervision_depths:
            if attention_layer_budget > 0 and depth >= attention_layer_budget:
                continue
            if depth > len(model.graph_matcher.attention_layers):
                continue
            deep_output = run_matcher(depth)
            deep_terms.append(positive_ce(deep_output))
        if deep_terms:
            deep_supervision_loss = torch.stack(deep_terms).mean()
    depth_distillation_loss = desc_a.new_zeros(())
    depth_distillation_teacher_depth = 0
    effective_student_depth = (
        int(attention_layer_budget)
        if int(attention_layer_budget) > 0
        else len(model.graph_matcher.attention_layers)
    )
    teacher_output: pfm_model.GraphMatcherOutput | None = None
    if (
        depth_distillation_weight > 0.0
        and depth_distillation_teacher_layers > 0
        and depth_distillation_teacher_layers < effective_student_depth
        and depth_distillation_teacher_layers <= len(model.graph_matcher.attention_layers)
    ):
        depth_distillation_teacher_depth = int(depth_distillation_teacher_layers)
        with torch.no_grad():
            teacher_output = run_matcher(depth_distillation_teacher_depth)
    output = run_matcher(attention_layer_budget)
    resolved_teacher_guard_output = teacher_guard_output
    if (
        teacher_guard_weight > 0.0
        or teacher_score_floor_weight > 0.0
        or teacher_match_count_floor_weight > 0.0
        or teacher_match_count_ceiling_weight > 0.0
        or teacher_distillation_weight > 0.0
    ) and resolved_teacher_guard_output is None:
        resolved_teacher_guard_output = run_teacher_guard_matcher(attention_layer_budget)
    teacher_guard_loss = output.logits.new_zeros(())
    teacher_score_floor_loss = output.logits.new_zeros(())
    teacher_match_count_floor_loss = output.logits.new_zeros(())
    teacher_match_count_ceiling_loss = output.logits.new_zeros(())
    teacher_distillation_loss = output.logits.new_zeros(())
    teacher_guard_metrics = {
        "positive_margin_loss": output.logits.new_zeros(()),
        "false_edge_loss": output.logits.new_zeros(()),
        "positive_violations": output.logits.new_zeros(()),
        "false_edges": output.logits.new_zeros(()),
    }
    teacher_score_floor_metrics = {
        "violations": output.logits.new_zeros(()),
        "score_delta_mean": output.logits.new_zeros(()),
        "teacher_score_mean": output.logits.new_zeros(()),
    }
    teacher_match_count_floor_metrics = {
        "teacher_count": output.logits.new_zeros(()),
        "student_count": output.logits.new_zeros(()),
        "count_deficit": output.logits.new_zeros(()),
        "topk_score_mean": output.logits.new_zeros(()),
        "violations": output.logits.new_zeros(()),
    }
    teacher_match_count_ceiling_metrics = {
        "teacher_count": output.logits.new_zeros(()),
        "student_count": output.logits.new_zeros(()),
        "count_excess": output.logits.new_zeros(()),
        "excess_score_mean": output.logits.new_zeros(()),
        "violations": output.logits.new_zeros(()),
    }
    dustbin_diagnostics = graph_matcher_dustbin_diagnostics(output, positive_count=count)
    dustbin_guard_enabled = should_apply_positive_dustbin_guard(
        dustbin_diagnostics,
        reject_threshold=positive_dustbin_guard_reject_threshold,
        margin_threshold=positive_dustbin_guard_margin_threshold,
    )
    guarded_no_match_weight = 0.0 if dustbin_guard_enabled else float(no_match_weight)
    guarded_hard_negative_dustbin_weight = (
        0.0 if dustbin_guard_enabled else float(hard_negative_dustbin_weight)
    )
    match_ce_loss = positive_ce(output)
    assignment_loss = output.logits.new_zeros(())
    no_match_loss = output.logits.new_zeros(())
    accept_loss = output.logits.new_zeros(())
    prune_ranking_loss = output.logits.new_zeros(())
    stop_confidence_loss = output.logits.new_zeros(())
    raw_preservation_loss = output.logits.new_zeros(())
    hard_negative_dustbin_loss = output.logits.new_zeros(())
    positive_dustbin_margin_loss = output.logits.new_zeros(())
    true_match_margin_loss = output.logits.new_zeros(())
    true_match_margin_metrics = {
        "violations": output.logits.new_zeros(()),
        "margin_mean": output.logits.new_zeros(()),
    }
    true_geometry_match_count_floor_loss = output.logits.new_zeros(())
    true_geometry_match_count_floor_metrics = {
        "target_count": output.logits.new_zeros(()),
        "student_count": output.logits.new_zeros(()),
        "count_deficit": output.logits.new_zeros(()),
        "topk_score_mean": output.logits.new_zeros(()),
        "violations": output.logits.new_zeros(()),
    }
    final_false_loss = output.logits.new_zeros(())
    final_false_metrics = {
        "edges": output.logits.new_zeros(()),
        "score_mean": output.logits.new_zeros(()),
        "accept_mean": output.logits.new_zeros(()),
    }
    mined_false_loss = output.logits.new_zeros(())
    mined_false_metrics = {
        "edges": output.logits.new_zeros(()),
        "reference_filtered_edges": output.logits.new_zeros(()),
        "score_mean": output.logits.new_zeros(()),
        "logit_mean": output.logits.new_zeros(()),
        "accept_mean": output.logits.new_zeros(()),
    }
    raw_false_loss = output.logits.new_zeros(())
    raw_false_metrics = {
        "edges": output.logits.new_zeros(()),
        "raw_similarity_mean": output.logits.new_zeros(()),
        "margin_mean": output.logits.new_zeros(()),
    }
    ransac_consistency_loss = output.logits.new_zeros(())
    ransac_consistency_metrics = {
        "edges": output.logits.new_zeros(()),
        "score_mean": output.logits.new_zeros(()),
        "residual_mean_px": output.logits.new_zeros(()),
        "accept_mean": output.logits.new_zeros(()),
    }
    warp_outlier_loss = output.logits.new_zeros(())
    warp_outlier_metrics = {
        "edges": output.logits.new_zeros(()),
        "residual_mean_px": output.logits.new_zeros(()),
        "accept_mean": output.logits.new_zeros(()),
    }
    warp_outlier_accept_loss = output.logits.new_zeros(())
    warp_outlier_accept_metrics = {
        "edges": output.logits.new_zeros(()),
        "score_mean": output.logits.new_zeros(()),
        "residual_mean_px": output.logits.new_zeros(()),
        "probability_mean": output.logits.new_zeros(()),
    }
    warp_soft_boundary_loss = output.logits.new_zeros(())
    warp_soft_boundary_metrics = {
        "edges": output.logits.new_zeros(()),
        "residual_mean_px": output.logits.new_zeros(()),
        "target_mean": output.logits.new_zeros(()),
        "score_probability_mean": output.logits.new_zeros(()),
        "accept_probability_mean": output.logits.new_zeros(()),
    }
    pair_acceptance_loss = output.logits.new_zeros(())
    pair_acceptance_metrics = {
        "target": 0.0,
        "weight": 0.0,
        "probability": 0.0,
        "raw_loss": 0.0,
    }
    loss = match_ce_loss
    if deep_supervision_weight > 0.0:
        loss = loss + float(deep_supervision_weight) * deep_supervision_loss
    if depth_distillation_weight > 0.0 and teacher_output is not None:
        depth_distillation_loss = graph_matcher_depth_distillation_loss(
            output,
            teacher_output,
            positive_count=count,
            temperature=depth_distillation_temperature,
        )
        loss = loss + float(depth_distillation_weight) * depth_distillation_loss
    if teacher_guard_weight > 0.0 and resolved_teacher_guard_output is not None:
        teacher_guard_loss, teacher_guard_metrics = graph_matcher_teacher_guard_loss(
            output,
            resolved_teacher_guard_output,
            positive_count=count,
            positive_margin_tolerance=teacher_guard_positive_margin_tolerance,
            false_margin_tolerance=teacher_guard_false_margin_tolerance,
        )
        loss = loss + float(teacher_guard_weight) * teacher_guard_loss
    if teacher_score_floor_weight > 0.0 and resolved_teacher_guard_output is not None:
        teacher_score_floor_loss, teacher_score_floor_metrics = graph_matcher_teacher_score_floor_loss(
            output,
            resolved_teacher_guard_output,
            positive_count=count,
            tolerance=teacher_score_floor_tolerance,
            min_teacher_score=teacher_score_floor_min_score,
        )
        loss = loss + float(teacher_score_floor_weight) * teacher_score_floor_loss
    if teacher_match_count_floor_weight > 0.0 and resolved_teacher_guard_output is not None:
        (
            teacher_match_count_floor_loss,
            teacher_match_count_floor_metrics,
        ) = graph_matcher_teacher_match_count_floor_loss(
            output,
            resolved_teacher_guard_output,
            positive_count=count,
            score_threshold=teacher_match_count_floor_threshold,
            margin=teacher_match_count_floor_margin,
        )
        loss = loss + float(teacher_match_count_floor_weight) * teacher_match_count_floor_loss
    if teacher_match_count_ceiling_weight > 0.0 and resolved_teacher_guard_output is not None:
        (
            teacher_match_count_ceiling_loss,
            teacher_match_count_ceiling_metrics,
        ) = graph_matcher_teacher_match_count_ceiling_loss(
            output,
            resolved_teacher_guard_output,
            positive_count=count,
            score_threshold=teacher_match_count_ceiling_threshold,
            margin=teacher_match_count_ceiling_margin,
        )
        loss = loss + float(teacher_match_count_ceiling_weight) * teacher_match_count_ceiling_loss
    if teacher_distillation_weight > 0.0 and resolved_teacher_guard_output is not None:
        teacher_distillation_loss = graph_matcher_depth_distillation_loss(
            output,
            resolved_teacher_guard_output,
            positive_count=count,
            temperature=teacher_distillation_temperature,
        )
        loss = loss + float(teacher_distillation_weight) * teacher_distillation_loss
    total_a = desc_a.size(0)
    total_b = desc_b.size(0)
    unmatched_total_a = extra_false_a_start if extra_false_match_pair_count > 0 else total_a
    unmatched_total_b = extra_false_b_start if extra_false_match_pair_count > 0 else total_b
    if guarded_no_match_weight > 0.0 and (unmatched_total_a > count or unmatched_total_b > count):
        no_match_terms: list[torch.Tensor] = []
        if unmatched_total_a > count:
            dustbin_col = torch.full(
                (unmatched_total_a - count,),
                total_b,
                dtype=torch.long,
                device=output.logits.device,
            )
            no_match_terms.append(F.cross_entropy(output.logits[count:unmatched_total_a, :], dustbin_col))
        if unmatched_total_b > count:
            dustbin_row = torch.full(
                (unmatched_total_b - count,),
                total_a,
                dtype=torch.long,
                device=output.logits.device,
            )
            no_match_terms.append(F.cross_entropy(output.logits[:, count:unmatched_total_b].T, dustbin_row))
        if no_match_terms:
            no_match_loss = torch.stack(no_match_terms).mean()
            loss = loss + float(guarded_no_match_weight) * no_match_loss
    if assignment_weight > 0.0:
        assignment_loss = graph_matcher_assignment_loss(
            output,
            positive_count=count,
            unmatched_total_a=unmatched_total_a,
            unmatched_total_b=unmatched_total_b,
        )
        loss = loss + float(assignment_weight) * assignment_loss
    if accept_weight > 0.0:
        accept_loss = graph_matcher_acceptance_loss(
            output,
            desc_a,
            desc_b,
            positive_count=count,
            negative_topk=accept_negative_topk,
        )
        loss = loss + float(accept_weight) * accept_loss
    if prune_ranking_weight > 0.0:
        prune_ranking_loss = graph_matcher_prune_ranking_loss(
            output,
            positive_count=count,
            margin=prune_ranking_margin,
        )
        loss = loss + float(prune_ranking_weight) * prune_ranking_loss
    if stop_confidence_weight > 0.0:
        stop_confidence_loss = graph_matcher_stop_confidence_loss(
            output,
            positive_count=count,
            safe_margin=stop_confidence_margin,
        )
        loss = loss + float(stop_confidence_weight) * stop_confidence_loss
    if raw_preservation_weight > 0.0:
        raw_preservation_loss = graph_matcher_raw_preservation_loss(
            output.logits,
            desc_a[:count],
            desc_b[:count],
            target_margin=raw_preservation_margin,
            raw_margin_threshold=raw_preservation_raw_margin,
        )
        loss = loss + float(raw_preservation_weight) * raw_preservation_loss
    if guarded_hard_negative_dustbin_weight > 0.0:
        hard_negative_dustbin_loss = graph_matcher_hard_negative_dustbin_loss(
            output.logits,
            desc_a[:count],
            desc_b[:count],
            positive_count=count,
            negative_topk=hard_negative_dustbin_topk,
            margin=hard_negative_dustbin_margin,
            points_b_xy=points_b_xy[:count],
            spatial_min_distance=hard_negative_dustbin_spatial_min_distance,
        )
        loss = loss + float(guarded_hard_negative_dustbin_weight) * hard_negative_dustbin_loss
    if positive_dustbin_margin_weight > 0.0:
        positive_dustbin_margin_loss = graph_matcher_positive_dustbin_margin_loss(
            output,
            positive_count=count,
            margin=positive_dustbin_margin,
        )
        loss = loss + float(positive_dustbin_margin_weight) * positive_dustbin_margin_loss
    if true_match_margin_weight > 0.0:
        true_match_margin_loss, true_match_margin_metrics = graph_matcher_true_match_margin_loss(
            output,
            positive_count=count,
            margin=true_match_margin,
        )
        loss = loss + float(true_match_margin_weight) * true_match_margin_loss
    if true_geometry_match_count_floor_weight > 0.0 and true_geometry_match_count_floor_target_count is not None:
        (
            true_geometry_match_count_floor_loss,
            true_geometry_match_count_floor_metrics,
        ) = graph_matcher_true_geometry_match_count_floor_loss(
            output,
            positive_count=count,
            target_count=float(true_geometry_match_count_floor_target_count),
            score_threshold=true_geometry_match_count_floor_threshold,
            margin=true_geometry_match_count_floor_margin,
        )
        loss = loss + float(true_geometry_match_count_floor_weight) * true_geometry_match_count_floor_loss
    if final_false_match_weight > 0.0 and not dustbin_guard_enabled:
        final_false_loss, final_false_metrics = graph_matcher_final_false_match_loss(
            output,
            positive_count=count,
            points_b_xy=points_b_xy[:count],
            topk=final_false_match_topk,
            min_score=final_false_match_min_score,
            margin=final_false_match_margin,
            spatial_min_distance=final_false_match_spatial_min_distance,
        )
        loss = loss + float(final_false_match_weight) * final_false_loss
    effective_mined_false_match_weight = float(mined_false_match_weight)
    if effective_mined_false_match_weight <= 0.0 and final_false_match_weight > 0.0:
        effective_mined_false_match_weight = float(final_false_match_weight)
    if (
        effective_mined_false_match_weight > 0.0
        and extra_false_match_pair_count > 0
        and not dustbin_guard_enabled
    ):
        mined_false_loss, mined_false_metrics = graph_matcher_mined_false_match_loss(
            output,
            positive_count=count,
            false_a_start=extra_false_a_start,
            false_b_start=extra_false_b_start,
            false_pair_count=extra_false_match_pair_count,
            topk=final_false_match_topk,
            min_score=final_false_match_min_score,
            margin=final_false_match_margin,
            loss_cap=mined_false_match_loss_cap,
            reference_margin=mined_false_match_reference_margin,
        )
        loss = loss + effective_mined_false_match_weight * mined_false_loss
    if raw_false_match_weight > 0.0:
        raw_false_loss, raw_false_metrics = graph_matcher_raw_false_match_loss(
            output.logits,
            desc_a[:count],
            desc_b[:count],
            positive_count=count,
            negative_topk=raw_false_match_topk,
            min_raw_similarity=raw_false_match_min_similarity,
            margin=raw_false_match_margin,
            points_b_xy=points_b_xy[:count],
            spatial_min_distance=raw_false_match_spatial_min_distance,
        )
        loss = loss + float(raw_false_match_weight) * raw_false_loss
    if ransac_consistency_weight > 0.0:
        ransac_consistency_loss, ransac_consistency_metrics = graph_matcher_ransac_consistency_loss(
            output,
            positive_count=count,
            points_a_xy=positive_points_a_xy,
            points_b_xy=positive_points_b_xy,
            topk=ransac_consistency_topk,
            residual_threshold_px=ransac_consistency_residual_threshold_px,
            min_score=ransac_consistency_min_score,
            margin=ransac_consistency_margin,
        )
        loss = loss + float(ransac_consistency_weight) * ransac_consistency_loss
    if warp_outlier_weight > 0.0:
        warp_outlier_loss, warp_outlier_metrics = graph_matcher_warp_outlier_loss(
            output,
            positive_count=count,
            points_b_xy=positive_points_b_xy,
            topk=warp_outlier_topk,
            residual_threshold_px=warp_outlier_residual_threshold_px,
            min_score=warp_outlier_min_score,
            margin=warp_outlier_margin,
        )
        loss = loss + float(warp_outlier_weight) * warp_outlier_loss
    if warp_outlier_accept_weight > 0.0:
        warp_outlier_accept_loss, warp_outlier_accept_metrics = graph_matcher_warp_outlier_accept_loss(
            output,
            positive_count=count,
            points_b_xy=positive_points_b_xy,
            topk=warp_outlier_accept_topk,
            residual_threshold_px=warp_outlier_accept_residual_threshold_px,
            min_score=warp_outlier_accept_min_score,
        )
        loss = loss + float(warp_outlier_accept_weight) * warp_outlier_accept_loss
    if warp_soft_boundary_weight > 0.0:
        warp_soft_boundary_loss, warp_soft_boundary_metrics = graph_matcher_warp_soft_boundary_loss(
            output,
            positive_count=count,
            points_b_xy=positive_points_b_xy,
            topk=warp_soft_boundary_topk,
            lower_residual_px=warp_soft_boundary_lower_residual_px,
            upper_residual_px=warp_soft_boundary_upper_residual_px,
            min_score=warp_soft_boundary_min_score,
        )
        loss = loss + float(warp_soft_boundary_weight) * warp_soft_boundary_loss
    if pair_acceptance_loss_weight > 0.0 and pair_acceptance_target is not None:
        pair_acceptance_loss, pair_acceptance_metrics = graph_matcher_pair_acceptance_loss(
            output,
            target=float(pair_acceptance_target),
            weight=float(pair_acceptance_weight),
        )
        loss = loss + float(pair_acceptance_loss_weight) * pair_acceptance_loss
    if return_components:
        executed_attention_layers = output.logits.new_tensor(float(model.graph_matcher.last_executed_attention_layers))
        attention_work_fraction = output.logits.new_tensor(float(getattr(output, "attention_work_fraction", 0.0)))
        positive_pairs = output.logits.new_tensor(float(count))
        extra_no_match_points = output.logits.new_tensor(float(extra_no_match_count))
        extra_false_match_pairs = output.logits.new_tensor(float(extra_false_match_pair_count))
        return loss, components(
            loss,
            match_ce_loss,
            assignment_loss,
            no_match_loss,
            accept_loss,
            prune_ranking_loss,
            stop_confidence_loss,
            raw_preservation_loss,
            hard_negative_dustbin_loss,
            positive_dustbin_margin_loss,
            true_match_margin_loss,
            true_match_margin_metrics["violations"],
            true_match_margin_metrics["margin_mean"],
            true_geometry_match_count_floor_loss,
            true_geometry_match_count_floor_metrics,
            final_false_loss,
            final_false_metrics["edges"],
            final_false_metrics["score_mean"],
            final_false_metrics["accept_mean"],
            mined_false_loss,
            mined_false_metrics["edges"],
            mined_false_metrics["reference_filtered_edges"],
            mined_false_metrics["score_mean"],
            mined_false_metrics["logit_mean"],
            mined_false_metrics["accept_mean"],
            raw_false_loss,
            raw_false_metrics["edges"],
            raw_false_metrics["raw_similarity_mean"],
            raw_false_metrics["margin_mean"],
            deep_supervision_loss,
            executed_attention_layers,
            attention_work_fraction,
            positive_pairs,
            extra_no_match_points,
            extra_false_match_pairs,
            dustbin_diagnostics,
            depth_distillation_loss,
            output.logits.new_tensor(float(depth_distillation_teacher_depth)),
            teacher_distillation_loss,
            output.logits.new_tensor(1.0 if dustbin_guard_enabled else 0.0),
            output.logits.new_tensor(float(guarded_no_match_weight)),
            output.logits.new_tensor(float(guarded_hard_negative_dustbin_weight)),
            candidate_topk_diagnostics,
            teacher_guard_loss=teacher_guard_loss,
            teacher_guard_metrics=teacher_guard_metrics,
            teacher_score_floor_loss=teacher_score_floor_loss,
            teacher_score_floor_metrics=teacher_score_floor_metrics,
            teacher_match_count_floor_loss=teacher_match_count_floor_loss,
            teacher_match_count_floor_metrics=teacher_match_count_floor_metrics,
            teacher_match_count_ceiling_loss=teacher_match_count_ceiling_loss,
            teacher_match_count_ceiling_metrics=teacher_match_count_ceiling_metrics,
            ransac_consistency_loss=ransac_consistency_loss,
            ransac_consistency_metrics=ransac_consistency_metrics,
            warp_outlier_loss=warp_outlier_loss,
            warp_outlier_metrics=warp_outlier_metrics,
            warp_outlier_accept_loss=warp_outlier_accept_loss,
            warp_outlier_accept_metrics=warp_outlier_accept_metrics,
            warp_soft_boundary_loss=warp_soft_boundary_loss,
            warp_soft_boundary_metrics=warp_soft_boundary_metrics,
            pair_acceptance_loss=pair_acceptance_loss,
            pair_acceptance_metrics=pair_acceptance_metrics,
        )
    return loss


def graph_matcher_assignment_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    unmatched_total_a: int | None = None,
    unmatched_total_b: int | None = None,
) -> torch.Tensor:
    """按推理时的双向 soft assignment 训练匹配和 dustbin 拒配。"""

    logits = output.logits
    total_a = logits.size(0) - 1
    total_b = logits.size(1) - 1
    count = min(int(positive_count), total_a, total_b)
    if count <= 0:
        return logits.new_zeros(())
    unmatched_total_a = total_a if unmatched_total_a is None else min(max(int(unmatched_total_a), count), total_a)
    unmatched_total_b = total_b if unmatched_total_b is None else min(max(int(unmatched_total_b), count), total_b)

    eps = torch.finfo(logits.dtype).eps
    row_prob_full = torch.softmax(logits[:total_a, :], dim=1)
    col_prob_full = torch.softmax(logits[:, :total_b], dim=0)
    dual_prob = (row_prob_full[:, :total_b] * col_prob_full[:total_a, :]).clamp_min(eps)
    terms: list[torch.Tensor] = [
        -dual_prob[:count, :count].diagonal().log().mean(),
    ]
    if unmatched_total_a > count:
        terms.append(-row_prob_full[count:unmatched_total_a, total_b].clamp_min(eps).log().mean())
    if unmatched_total_b > count:
        terms.append(-col_prob_full[total_a, count:unmatched_total_b].clamp_min(eps).log().mean())
    return torch.stack(terms).mean()


def scheduled_graph_matcher_weight(
    base_weight: float,
    *,
    step: int,
    warmup_steps: int = 0,
    ramp_steps: int = 0,
) -> float:
    if base_weight < 0.0:
        raise ValueError("base_weight must be nonnegative")
    if warmup_steps < 0 or ramp_steps < 0:
        raise ValueError("warmup_steps and ramp_steps must be nonnegative")
    if base_weight == 0.0:
        return 0.0
    step = max(0, int(step))
    if step <= int(warmup_steps):
        return 0.0
    if ramp_steps <= 0:
        return float(base_weight)
    progress = min(1.0, max(0.0, (step - int(warmup_steps)) / float(ramp_steps)))
    return float(base_weight) * progress


def graph_matcher_positive_dustbin_margin_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    margin: float = 0.0,
) -> torch.Tensor:
    count = min(int(positive_count), output.logits.size(0) - 1, output.logits.size(1) - 1)
    if count <= 0:
        return output.logits.new_zeros(())
    device = output.logits.device
    indices = torch.arange(count, device=device)
    true_logits = output.logits[:count, :count][indices, indices]
    row_dustbin = output.logits[:count, output.logits.size(1) - 1]
    col_dustbin = output.logits[output.logits.size(0) - 1, :count]
    strongest_dustbin = torch.maximum(row_dustbin, col_dustbin)
    positive_margin = true_logits - strongest_dustbin
    return (float(margin) - positive_margin).clamp_min(0.0).pow(2).mean()


def graph_matcher_true_match_margin_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize true pairs that lose under the final logit-minus-dustbin score."""

    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    logits = output.logits
    count = min(int(positive_count), logits.size(0) - 1, logits.size(1) - 1)
    zero = logits.new_zeros(())
    metrics = {
        "violations": zero,
        "margin_mean": zero,
    }
    if count <= 1:
        return zero, metrics

    pair_logits = logits[:count, :count]
    row_dustbin = logits[:count, logits.size(1) - 1].unsqueeze(1)
    col_dustbin = logits[logits.size(0) - 1, :count].unsqueeze(0)
    final_scores = pair_logits - row_dustbin - col_dustbin
    diagonal_mask = torch.eye(count, dtype=torch.bool, device=logits.device)
    true_scores = final_scores.diagonal()
    negative_scores = final_scores.masked_fill(diagonal_mask, -float("inf"))
    row_hard = negative_scores.max(dim=1).values
    col_hard = negative_scores.max(dim=0).values
    hardest_competitor = torch.maximum(row_hard, col_hard)
    true_margin = true_scores - hardest_competitor
    terms = (float(margin) - true_margin).clamp_min(0.0)
    active = terms.gt(0.0)
    if not bool(active.any()):
        return zero, metrics
    metrics = {
        "violations": logits.new_tensor(float(active.sum().detach().cpu().item())),
        "margin_mean": true_margin.detach().mean(),
    }
    return terms[active].pow(2).mean(), metrics


def graph_matcher_true_geometry_match_count_floor_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    target_count: float,
    score_threshold: float = 0.0,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Require enough true-geometry positive pairs to clear the final acceptance score."""

    if not math.isfinite(float(target_count)) or target_count < 0.0:
        raise ValueError("target_count must be finite and nonnegative")
    if not math.isfinite(float(score_threshold)):
        raise ValueError("score_threshold must be finite")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    logits = output.logits
    count = min(int(positive_count), logits.size(0) - 1, logits.size(1) - 1)
    zero = logits.new_zeros(())
    metrics = {
        "target_count": zero,
        "student_count": zero,
        "count_deficit": zero,
        "topk_score_mean": zero,
        "violations": zero,
    }
    if count <= 0:
        return zero, metrics

    required_count = min(count, int(math.ceil(float(target_count))))
    if required_count <= 0:
        return zero, metrics

    device = logits.device
    indices = torch.arange(count, device=device)
    true_logits = logits[:count, :count][indices, indices]
    row_dustbin = logits[:count, logits.size(1) - 1]
    col_dustbin = logits[logits.size(0) - 1, :count]
    true_scores = true_logits - row_dustbin - col_dustbin
    required_score = float(score_threshold) + float(margin)
    student_count_tensor = true_scores.ge(required_score).to(logits.dtype).sum()
    target_count_tensor = logits.new_tensor(float(required_count))
    selected_scores = torch.topk(true_scores, k=required_count).values
    deficit = (required_score - selected_scores).clamp_min(0.0)
    count_deficit = (target_count_tensor - student_count_tensor).clamp_min(0.0)
    metrics = {
        "target_count": target_count_tensor.detach(),
        "student_count": student_count_tensor.detach(),
        "count_deficit": count_deficit.detach(),
        "topk_score_mean": selected_scores.detach().mean(),
        "violations": deficit.gt(0.0).to(logits.dtype).sum().detach(),
    }
    return deficit.pow(2).mean(), metrics


def graph_matcher_final_false_match_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    points_b_xy: torch.Tensor | None = None,
    topk: int = 8,
    min_score: float = 0.0,
    margin: float = 0.25,
    spatial_min_distance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize final high-confidence off-diagonal matches in the positive block."""

    if topk < 0:
        raise ValueError("topk must be nonnegative")
    if min_score < 0.0:
        raise ValueError("min_score must be nonnegative")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    if spatial_min_distance < 0.0:
        raise ValueError("spatial_min_distance must be nonnegative")
    logits = output.logits
    zero = logits.new_zeros(())
    metrics = {
        "edges": zero,
        "score_mean": zero,
        "accept_mean": zero,
    }
    count = min(int(positive_count), logits.size(0) - 1, logits.size(1) - 1)
    if count <= 1 or topk <= 0:
        return zero, metrics
    pair_logits = logits[:count, :count]
    row_dustbin = logits[:count, logits.size(1) - 1].detach().unsqueeze(1)
    col_dustbin = logits[logits.size(0) - 1, :count].detach().unsqueeze(0)
    final_scores = pair_logits - row_dustbin - col_dustbin
    candidate_mask = ~torch.eye(count, dtype=torch.bool, device=logits.device)
    if spatial_min_distance > 0.0 and points_b_xy is not None and points_b_xy.size(0) >= count:
        points_b = points_b_xy[:count].to(device=logits.device, dtype=logits.dtype)
        candidate_mask &= torch.cdist(points_b, points_b) >= float(spatial_min_distance)
    candidate_mask &= final_scores.detach() >= float(min_score)
    if not bool(candidate_mask.any()):
        return zero, metrics
    candidate_indices = candidate_mask.nonzero(as_tuple=False)
    candidate_scores = final_scores[candidate_mask]
    keep_count = min(int(topk), int(candidate_scores.numel()))
    selected_order = candidate_scores.detach().topk(keep_count).indices
    selected = candidate_indices.index_select(0, selected_order)
    selected_scores = candidate_scores.index_select(0, selected_order)
    rows = selected[:, 0]
    cols = selected[:, 1]
    wrong_scores = final_scores[rows, cols]
    row_true_scores = final_scores[rows, rows]
    col_true_scores = final_scores[cols, cols]
    true_reference_scores = torch.minimum(row_true_scores, col_true_scores)
    ranking_loss = (float(margin) - (true_reference_scores - wrong_scores)).clamp_min(0.0).pow(2).mean()
    terms = [ranking_loss]
    accept_mean = zero
    if output.accept_logits is not None:
        accept_logits = output.accept_logits
        if accept_logits.size(0) >= count and accept_logits.size(1) >= count:
            selected_accept_logits = accept_logits[rows, cols]
            terms.append(
                F.binary_cross_entropy_with_logits(
                    selected_accept_logits,
                    torch.zeros_like(selected_accept_logits),
                )
            )
            accept_mean = torch.sigmoid(selected_accept_logits.detach()).mean()
    loss = torch.stack(terms).mean()
    metrics = {
        "edges": logits.new_tensor(float(keep_count)),
        "score_mean": selected_scores.detach().mean(),
        "accept_mean": accept_mean,
    }
    return loss, metrics


def graph_matcher_ransac_consistency_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    points_a_xy: torch.Tensor,
    points_b_xy: torch.Tensor,
    topk: int = 8,
    residual_threshold_px: float = 3.0,
    min_score: float = 0.0,
    margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize high-scoring edges that are inconsistent with the positive-pair geometry."""

    if topk < 0:
        raise ValueError("topk must be nonnegative")
    if residual_threshold_px < 0.0:
        raise ValueError("residual_threshold_px must be nonnegative")
    if min_score < 0.0:
        raise ValueError("min_score must be nonnegative")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    logits = output.logits
    zero = logits.new_zeros(())
    metrics = {
        "edges": zero,
        "score_mean": zero,
        "residual_mean_px": zero,
        "accept_mean": zero,
    }
    count = min(
        int(positive_count),
        logits.size(0) - 1,
        logits.size(1) - 1,
        points_a_xy.size(0),
        points_b_xy.size(0),
    )
    if count < 3 or topk <= 0:
        return zero, metrics

    points_a = points_a_xy[:count].to(device=logits.device, dtype=logits.dtype)
    points_b = points_b_xy[:count].to(device=logits.device, dtype=logits.dtype)
    design = torch.cat([points_a.detach(), torch.ones((count, 1), device=logits.device, dtype=logits.dtype)], dim=1)
    try:
        affine = torch.linalg.lstsq(design, points_b.detach()).solution
    except RuntimeError:
        affine = torch.linalg.pinv(design).matmul(points_b.detach())
    predicted_b = design.matmul(affine)
    residuals = torch.cdist(predicted_b, points_b.detach())

    pair_logits = logits[:count, :count]
    row_dustbin = logits[:count, logits.size(1) - 1].detach().unsqueeze(1)
    col_dustbin = logits[logits.size(0) - 1, :count].detach().unsqueeze(0)
    final_scores = pair_logits - row_dustbin - col_dustbin
    candidate_mask = ~torch.eye(count, dtype=torch.bool, device=logits.device)
    candidate_mask &= residuals > float(residual_threshold_px)
    candidate_mask &= final_scores.detach() >= float(min_score)
    if not bool(candidate_mask.any()):
        return zero, metrics

    candidate_indices = candidate_mask.nonzero(as_tuple=False)
    candidate_scores = final_scores[candidate_mask]
    keep_count = min(int(topk), int(candidate_scores.numel()))
    selected_order = candidate_scores.detach().topk(keep_count).indices
    selected = candidate_indices.index_select(0, selected_order)
    selected_scores = candidate_scores.index_select(0, selected_order)
    rows = selected[:, 0]
    cols = selected[:, 1]
    wrong_scores = final_scores[rows, cols]
    row_true_scores = final_scores[rows, rows]
    col_true_scores = final_scores[cols, cols]
    true_reference_scores = torch.minimum(row_true_scores, col_true_scores)
    ranking_loss = (float(margin) - (true_reference_scores - wrong_scores)).clamp_min(0.0).pow(2).mean()
    terms = [ranking_loss]
    accept_mean = zero
    if output.accept_logits is not None:
        accept_logits = output.accept_logits
        if accept_logits.size(0) >= count and accept_logits.size(1) >= count:
            selected_accept_logits = accept_logits[rows, cols]
            terms.append(
                F.binary_cross_entropy_with_logits(
                    selected_accept_logits,
                    torch.zeros_like(selected_accept_logits),
                )
            )
            accept_mean = torch.sigmoid(selected_accept_logits.detach()).mean()
    loss = torch.stack(terms).mean()
    metrics = {
        "edges": logits.new_tensor(float(keep_count)),
        "score_mean": selected_scores.detach().mean(),
        "residual_mean_px": residuals[rows, cols].detach().mean(),
        "accept_mean": accept_mean,
    }
    return loss, metrics


def graph_matcher_warp_outlier_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    points_b_xy: torch.Tensor,
    topk: int = 8,
    residual_threshold_px: float = 3.0,
    min_score: float = 0.0,
    margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize high-scoring edges whose target is far from the exact warped point."""

    if topk < 0:
        raise ValueError("topk must be nonnegative")
    if residual_threshold_px < 0.0:
        raise ValueError("residual_threshold_px must be nonnegative")
    if min_score < 0.0:
        raise ValueError("min_score must be nonnegative")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    logits = output.logits
    zero = logits.new_zeros(())
    metrics = {
        "edges": zero,
        "residual_mean_px": zero,
        "accept_mean": zero,
    }
    count = min(
        int(positive_count),
        logits.size(0) - 1,
        logits.size(1) - 1,
        points_b_xy.size(0),
    )
    if count <= 1 or topk <= 0:
        return zero, metrics

    target_points = points_b_xy[:count].to(device=logits.device, dtype=logits.dtype)
    residuals = torch.cdist(target_points.detach(), target_points.detach(), p=2.0)
    pair_logits = logits[:count, :count]
    row_dustbin = logits[:count, logits.size(1) - 1].detach().unsqueeze(1)
    col_dustbin = logits[logits.size(0) - 1, :count].detach().unsqueeze(0)
    final_scores = pair_logits - row_dustbin - col_dustbin
    candidate_mask = ~torch.eye(count, dtype=torch.bool, device=logits.device)
    candidate_mask &= residuals > float(residual_threshold_px)
    candidate_mask &= final_scores.detach() >= float(min_score)
    if not bool(candidate_mask.any()):
        return zero, metrics

    candidate_indices = candidate_mask.nonzero(as_tuple=False)
    candidate_scores = final_scores[candidate_mask]
    keep_count = min(int(topk), int(candidate_scores.numel()))
    selected_order = candidate_scores.detach().topk(keep_count).indices
    selected = candidate_indices.index_select(0, selected_order)
    rows = selected[:, 0]
    cols = selected[:, 1]
    wrong_scores = final_scores[rows, cols]
    row_true_scores = final_scores[rows, rows]
    col_true_scores = final_scores[cols, cols]
    true_reference_scores = torch.minimum(row_true_scores, col_true_scores)
    ranking_loss = (float(margin) - (true_reference_scores - wrong_scores)).clamp_min(0.0).pow(2).mean()
    terms = [ranking_loss]
    accept_mean = zero
    if output.accept_logits is not None:
        accept_logits = output.accept_logits
        if accept_logits.size(0) >= count and accept_logits.size(1) >= count:
            selected_accept_logits = accept_logits[rows, cols]
            terms.append(
                F.binary_cross_entropy_with_logits(
                    selected_accept_logits,
                    torch.zeros_like(selected_accept_logits),
                )
            )
            accept_mean = torch.sigmoid(selected_accept_logits.detach()).mean()
    loss = torch.stack(terms).mean()
    metrics = {
        "edges": logits.new_tensor(float(keep_count)),
        "residual_mean_px": residuals[rows, cols].detach().mean(),
        "accept_mean": accept_mean,
    }
    return loss, metrics


def graph_matcher_warp_outlier_accept_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    points_b_xy: torch.Tensor,
    topk: int = 8,
    residual_threshold_px: float = 3.0,
    min_score: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the accept head to reject high-scoring exact-warp outlier edges."""

    if topk < 0:
        raise ValueError("topk must be nonnegative")
    if residual_threshold_px < 0.0:
        raise ValueError("residual_threshold_px must be nonnegative")
    if min_score < 0.0:
        raise ValueError("min_score must be nonnegative")
    logits = output.logits
    zero = logits.new_zeros(())
    metrics = {
        "edges": zero,
        "score_mean": zero,
        "residual_mean_px": zero,
        "probability_mean": zero,
    }
    accept_logits = output.accept_logits
    if accept_logits is None:
        return zero, metrics
    count = min(
        int(positive_count),
        logits.size(0) - 1,
        logits.size(1) - 1,
        accept_logits.size(0),
        accept_logits.size(1),
        points_b_xy.size(0),
    )
    if count <= 1 or topk <= 0:
        return zero, metrics

    target_points = points_b_xy[:count].to(device=logits.device, dtype=logits.dtype)
    residuals = torch.cdist(target_points.detach(), target_points.detach(), p=2.0)
    pair_logits = logits[:count, :count].detach()
    row_dustbin = logits[:count, logits.size(1) - 1].detach().unsqueeze(1)
    col_dustbin = logits[logits.size(0) - 1, :count].detach().unsqueeze(0)
    final_scores = pair_logits - row_dustbin - col_dustbin
    candidate_mask = ~torch.eye(count, dtype=torch.bool, device=logits.device)
    candidate_mask &= residuals > float(residual_threshold_px)
    candidate_mask &= final_scores >= float(min_score)
    if not bool(candidate_mask.any()):
        return zero, metrics

    candidate_indices = candidate_mask.nonzero(as_tuple=False)
    candidate_scores = final_scores[candidate_mask]
    keep_count = min(int(topk), int(candidate_scores.numel()))
    selected_order = candidate_scores.topk(keep_count).indices
    selected = candidate_indices.index_select(0, selected_order)
    rows = selected[:, 0]
    cols = selected[:, 1]
    selected_accept_logits = accept_logits[rows, cols]
    loss = F.binary_cross_entropy_with_logits(
        selected_accept_logits,
        torch.zeros_like(selected_accept_logits),
    )
    metrics = {
        "edges": logits.new_tensor(float(keep_count)),
        "score_mean": candidate_scores.index_select(0, selected_order).mean(),
        "residual_mean_px": residuals[rows, cols].detach().mean(),
        "probability_mean": torch.sigmoid(selected_accept_logits.detach()).mean(),
    }
    return loss, metrics


def graph_matcher_warp_soft_boundary_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    points_b_xy: torch.Tensor,
    topk: int = 8,
    lower_residual_px: float = 5.0,
    upper_residual_px: float = 8.0,
    min_score: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply soft accept targets to near-boundary exact-warp residual edges."""

    if topk < 0:
        raise ValueError("topk must be nonnegative")
    if not math.isfinite(float(lower_residual_px)) or lower_residual_px < 0.0:
        raise ValueError("lower_residual_px must be finite and nonnegative")
    if not math.isfinite(float(upper_residual_px)) or upper_residual_px <= lower_residual_px:
        raise ValueError("upper_residual_px must be finite and greater than lower_residual_px")
    if min_score < 0.0:
        raise ValueError("min_score must be nonnegative")
    logits = output.logits
    zero = logits.new_zeros(())
    metrics = {
        "edges": zero,
        "residual_mean_px": zero,
        "target_mean": zero,
        "score_probability_mean": zero,
        "accept_probability_mean": zero,
    }
    count = min(
        int(positive_count),
        logits.size(0) - 1,
        logits.size(1) - 1,
        points_b_xy.size(0),
    )
    if count <= 1 or topk <= 0:
        return zero, metrics

    target_points = points_b_xy[:count].to(device=logits.device, dtype=logits.dtype)
    residuals = torch.cdist(target_points.detach(), target_points.detach(), p=2.0)
    pair_logits = logits[:count, :count]
    row_dustbin = logits[:count, logits.size(1) - 1].detach().unsqueeze(1)
    col_dustbin = logits[logits.size(0) - 1, :count].detach().unsqueeze(0)
    final_scores = pair_logits - row_dustbin - col_dustbin
    candidate_mask = ~torch.eye(count, dtype=torch.bool, device=logits.device)
    candidate_mask &= residuals > float(lower_residual_px)
    candidate_mask &= residuals <= float(upper_residual_px)
    candidate_mask &= final_scores.detach() >= float(min_score)
    if not bool(candidate_mask.any()):
        return zero, metrics

    candidate_indices = candidate_mask.nonzero(as_tuple=False)
    candidate_scores = final_scores[candidate_mask]
    keep_count = min(int(topk), int(candidate_scores.numel()))
    selected_order = candidate_scores.detach().topk(keep_count).indices
    selected = candidate_indices.index_select(0, selected_order)
    rows = selected[:, 0]
    cols = selected[:, 1]
    selected_scores = final_scores[rows, cols]
    selected_residuals = residuals[rows, cols]
    span = float(upper_residual_px) - float(lower_residual_px)
    soft_targets = ((float(upper_residual_px) - selected_residuals) / span).clamp(0.0, 1.0)
    terms = [F.binary_cross_entropy_with_logits(selected_scores, soft_targets)]
    accept_probability_mean = zero
    if output.accept_logits is not None:
        accept_logits = output.accept_logits
        if accept_logits.size(0) >= count and accept_logits.size(1) >= count:
            selected_accept_logits = accept_logits[rows, cols]
            terms.append(F.binary_cross_entropy_with_logits(selected_accept_logits, soft_targets))
            accept_probability_mean = torch.sigmoid(selected_accept_logits.detach()).mean()
    loss = torch.stack(terms).mean()
    metrics = {
        "edges": logits.new_tensor(float(keep_count)),
        "residual_mean_px": selected_residuals.detach().mean(),
        "target_mean": soft_targets.detach().mean(),
        "score_probability_mean": torch.sigmoid(selected_scores.detach()).mean(),
        "accept_probability_mean": accept_probability_mean,
    }
    return loss, metrics


def graph_matcher_mined_false_match_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    false_a_start: int,
    false_b_start: int,
    false_pair_count: int,
    topk: int = 8,
    min_score: float = 0.0,
    margin: float = 0.25,
    loss_cap: float = 0.0,
    reference_margin: float = -1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize mined false matcher edges without training dustbin logits."""

    if false_a_start < 0 or false_b_start < 0:
        raise ValueError("false starts must be nonnegative")
    if false_pair_count < 0:
        raise ValueError("false_pair_count must be nonnegative")
    if topk < 0:
        raise ValueError("topk must be nonnegative")
    if min_score < 0.0:
        raise ValueError("min_score must be nonnegative")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    if loss_cap < 0.0:
        raise ValueError("loss_cap must be nonnegative")
    if not math.isfinite(float(reference_margin)) or reference_margin < -1.0:
        raise ValueError("reference_margin must be finite and >= -1")

    logits = output.logits
    zero = logits.new_zeros(())
    metrics = {
        "edges": zero,
        "reference_filtered_edges": zero,
        "score_mean": zero,
        "logit_mean": zero,
        "accept_mean": zero,
    }
    total_a = logits.size(0) - 1
    total_b = logits.size(1) - 1
    count = min(
        int(false_pair_count),
        max(0, total_a - int(false_a_start)),
        max(0, total_b - int(false_b_start)),
    )
    if count <= 0 or topk <= 0:
        return zero, metrics

    rows = torch.arange(int(false_a_start), int(false_a_start) + count, device=logits.device)
    cols = torch.arange(int(false_b_start), int(false_b_start) + count, device=logits.device)
    pair_logits = logits[rows, cols]
    row_dustbin = logits[rows, total_b].detach()
    col_dustbin = logits[total_a, cols].detach()
    final_scores = pair_logits - row_dustbin - col_dustbin

    positive_limit = min(int(positive_count), total_a, total_b)
    if positive_limit > 0:
        positive_rows = torch.arange(positive_limit, device=logits.device)
        positive_pair_logits = logits[:positive_limit, :positive_limit].diagonal()
        positive_row_dustbin = logits[positive_rows, total_b].detach()
        positive_col_dustbin = logits[total_a, positive_rows].detach()
        positive_reference = (
            positive_pair_logits - positive_row_dustbin - positive_col_dustbin
        ).detach().mean()
    else:
        positive_reference = pair_logits.detach().new_zeros(())

    candidate_mask = final_scores.detach() >= float(min_score)
    reference_filtered_edges = zero
    if positive_limit > 0 and reference_margin >= 0.0:
        candidate_before_reference = candidate_mask
        candidate_mask = candidate_mask & (
            final_scores.detach() >= positive_reference - float(reference_margin)
        )
        reference_filtered_edges = (candidate_before_reference & ~candidate_mask).to(dtype=logits.dtype).sum()
    if not bool(candidate_mask.any()):
        metrics["reference_filtered_edges"] = reference_filtered_edges
        return zero, metrics

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).reshape(-1)
    candidate_scores = final_scores.index_select(0, candidate_indices)
    keep_count = min(int(topk), int(candidate_scores.numel()))
    selected_order = candidate_scores.detach().topk(keep_count).indices
    selected_indices = candidate_indices.index_select(0, selected_order)
    selected_scores = candidate_scores.index_select(0, selected_order)
    selected_pair_logits = pair_logits.index_select(0, selected_indices)

    ranking_loss = (selected_scores - positive_reference + float(margin)).clamp_min(0.0).pow(2).mean()
    terms = [ranking_loss]

    accept_mean = zero
    if output.accept_logits is not None:
        accept_logits = output.accept_logits
        if accept_logits.size(0) > int(false_a_start) and accept_logits.size(1) > int(false_b_start):
            selected_rows = rows.index_select(0, selected_indices)
            selected_cols = cols.index_select(0, selected_indices)
            valid = (selected_rows < accept_logits.size(0)) & (selected_cols < accept_logits.size(1))
            if bool(valid.any()):
                selected_accept_logits = accept_logits[selected_rows[valid], selected_cols[valid]]
                terms.append(
                    F.binary_cross_entropy_with_logits(
                        selected_accept_logits,
                        torch.zeros_like(selected_accept_logits),
                    )
                )
                accept_mean = torch.sigmoid(selected_accept_logits.detach()).mean()

    loss = torch.stack(terms).mean()
    if loss_cap > 0.0:
        loss = loss.clamp_max(float(loss_cap))
    metrics = {
        "edges": logits.new_tensor(float(keep_count)),
        "reference_filtered_edges": reference_filtered_edges.detach(),
        "score_mean": selected_scores.detach().mean(),
        "logit_mean": selected_pair_logits.detach().mean(),
        "accept_mean": accept_mean,
    }
    return loss, metrics


def graph_matcher_depth_distillation_loss(
    student: pfm_model.GraphMatcherOutput,
    teacher: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Keep a deeper matcher assignment distribution close to a detached shallow teacher."""

    count = min(
        int(positive_count),
        student.logits.size(0) - 1,
        student.logits.size(1) - 1,
        teacher.logits.size(0) - 1,
        teacher.logits.size(1) - 1,
    )
    if count <= 1:
        return student.logits.new_zeros(())
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    scale = float(temperature)
    student_logits = student.logits[:count, :count]
    teacher_logits = teacher.logits[:count, :count].detach().to(device=student_logits.device, dtype=student_logits.dtype)
    student_row_log_prob = F.log_softmax(student_logits / scale, dim=1)
    teacher_row_prob = F.softmax(teacher_logits / scale, dim=1)
    student_col_log_prob = F.log_softmax(student_logits.T / scale, dim=1)
    teacher_col_prob = F.softmax(teacher_logits.T / scale, dim=1)
    row_loss = F.kl_div(student_row_log_prob, teacher_row_prob, reduction="batchmean")
    col_loss = F.kl_div(student_col_log_prob, teacher_col_prob, reduction="batchmean")
    return 0.5 * (row_loss + col_loss) * scale * scale


def graph_matcher_teacher_guard_loss(
    student: pfm_model.GraphMatcherOutput,
    teacher: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    positive_margin_tolerance: float = 0.0,
    false_margin_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve a frozen teacher's positive margins while discouraging stronger false edges."""

    if positive_margin_tolerance < 0.0:
        raise ValueError("positive_margin_tolerance must be nonnegative")
    if false_margin_tolerance < 0.0:
        raise ValueError("false_margin_tolerance must be nonnegative")
    count = min(
        int(positive_count),
        student.logits.size(0) - 1,
        student.logits.size(1) - 1,
        teacher.logits.size(0) - 1,
        teacher.logits.size(1) - 1,
    )
    zero = student.logits.new_zeros(())
    if count <= 0:
        return zero, {
            "positive_margin_loss": zero,
            "false_edge_loss": zero,
            "positive_violations": zero,
            "false_edges": zero,
        }
    device = student.logits.device
    indices = torch.arange(count, device=device)
    student_logits = student.logits
    teacher_logits = teacher.logits.detach().to(device=device, dtype=student_logits.dtype)

    def positive_margin(logits: torch.Tensor) -> torch.Tensor:
        true_logits = logits[:count, :count][indices, indices]
        row_block = logits[:count, :count].clone()
        col_block = logits[:count, :count].clone()
        row_block[indices, indices] = -torch.inf
        col_block[indices, indices] = -torch.inf
        row_competitor = row_block.max(dim=1).values
        col_competitor = col_block.max(dim=0).values
        row_dustbin = logits[:count, logits.size(1) - 1]
        col_dustbin = logits[logits.size(0) - 1, :count]
        strongest_competitor = torch.stack(
            [row_competitor, col_competitor, row_dustbin, col_dustbin],
            dim=0,
        ).max(dim=0).values
        return true_logits - strongest_competitor

    teacher_positive_margin = positive_margin(teacher_logits)
    student_positive_margin = positive_margin(student_logits)
    positive_deficit = (
        teacher_positive_margin - student_positive_margin - float(positive_margin_tolerance)
    ).clamp_min(0.0)
    positive_margin_loss = positive_deficit.pow(2).mean()
    positive_violations = positive_deficit.gt(0.0).to(student_logits.dtype).sum()

    if count <= 1:
        false_edge_loss = zero
        false_edges = zero
    else:
        mask = torch.ones((count, count), dtype=torch.bool, device=device)
        mask[indices, indices] = False
        student_true = student_logits[:count, :count][indices, indices]
        teacher_true = teacher_logits[:count, :count][indices, indices]
        student_false_row_margin = student_logits[:count, :count] - student_true[:, None]
        teacher_false_row_margin = teacher_logits[:count, :count] - teacher_true[:, None]
        student_false_col_margin = student_logits[:count, :count] - student_true[None, :]
        teacher_false_col_margin = teacher_logits[:count, :count] - teacher_true[None, :]
        false_excess = torch.maximum(
            student_false_row_margin - teacher_false_row_margin,
            student_false_col_margin - teacher_false_col_margin,
        )
        selected_excess = (false_excess[mask] - float(false_margin_tolerance)).clamp_min(0.0)
        false_edge_loss = selected_excess.pow(2).mean() if selected_excess.numel() else zero
        false_edges = selected_excess.gt(0.0).to(student_logits.dtype).sum()

    loss = 0.5 * (positive_margin_loss + false_edge_loss)
    return loss, {
        "positive_margin_loss": positive_margin_loss.detach(),
        "false_edge_loss": false_edge_loss.detach(),
        "positive_violations": positive_violations.detach(),
        "false_edges": false_edges.detach(),
    }


def graph_matcher_teacher_score_floor_loss(
    student: pfm_model.GraphMatcherOutput,
    teacher: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    tolerance: float = 0.0,
    min_teacher_score: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve teacher true-pair final scores to avoid conservative recall collapse."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if not math.isfinite(float(min_teacher_score)):
        raise ValueError("min_teacher_score must be finite")
    count = min(
        int(positive_count),
        student.logits.size(0) - 1,
        student.logits.size(1) - 1,
        teacher.logits.size(0) - 1,
        teacher.logits.size(1) - 1,
    )
    zero = student.logits.new_zeros(())
    metrics = {
        "violations": zero,
        "score_delta_mean": zero,
        "teacher_score_mean": zero,
    }
    if count <= 0:
        return zero, metrics

    device = student.logits.device
    indices = torch.arange(count, device=device)
    student_logits = student.logits
    teacher_logits = teacher.logits.detach().to(device=device, dtype=student_logits.dtype)

    def true_final_scores(logits: torch.Tensor) -> torch.Tensor:
        true_logits = logits[:count, :count][indices, indices]
        row_dustbin = logits[:count, logits.size(1) - 1]
        col_dustbin = logits[logits.size(0) - 1, :count]
        return true_logits - row_dustbin - col_dustbin

    teacher_scores = true_final_scores(teacher_logits)
    protected = teacher_scores >= float(min_teacher_score)
    if not bool(protected.any()):
        return zero, metrics

    student_scores = true_final_scores(student_logits)
    selected_teacher = teacher_scores[protected]
    selected_student = student_scores[protected]
    score_delta = selected_student - selected_teacher
    deficit = (selected_teacher - selected_student - float(tolerance)).clamp_min(0.0)
    loss = deficit.pow(2).mean()
    metrics = {
        "violations": deficit.gt(0.0).to(student_logits.dtype).sum().detach(),
        "score_delta_mean": score_delta.detach().mean(),
        "teacher_score_mean": selected_teacher.detach().mean(),
    }
    return loss, metrics


def graph_matcher_teacher_match_count_floor_loss(
    student: pfm_model.GraphMatcherOutput,
    teacher: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    score_threshold: float = 0.0,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep at least the teacher's high-confidence true-pair count."""

    if not math.isfinite(float(score_threshold)):
        raise ValueError("score_threshold must be finite")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    count = min(
        int(positive_count),
        student.logits.size(0) - 1,
        student.logits.size(1) - 1,
        teacher.logits.size(0) - 1,
        teacher.logits.size(1) - 1,
    )
    zero = student.logits.new_zeros(())
    metrics = {
        "teacher_count": zero,
        "student_count": zero,
        "count_deficit": zero,
        "topk_score_mean": zero,
        "violations": zero,
    }
    if count <= 0:
        return zero, metrics

    device = student.logits.device
    indices = torch.arange(count, device=device)
    student_logits = student.logits
    teacher_logits = teacher.logits.detach().to(device=device, dtype=student_logits.dtype)

    def true_final_scores(logits: torch.Tensor) -> torch.Tensor:
        true_logits = logits[:count, :count][indices, indices]
        row_dustbin = logits[:count, logits.size(1) - 1]
        col_dustbin = logits[logits.size(0) - 1, :count]
        return true_logits - row_dustbin - col_dustbin

    teacher_scores = true_final_scores(teacher_logits)
    teacher_count_tensor = teacher_scores.ge(float(score_threshold)).to(student_logits.dtype).sum()
    teacher_count = int(teacher_count_tensor.detach().cpu().item())
    if teacher_count <= 0:
        return zero, metrics

    student_scores = true_final_scores(student_logits)
    required_score = float(score_threshold) + float(margin)
    student_count_tensor = student_scores.ge(required_score).to(student_logits.dtype).sum()
    selected_student = torch.topk(student_scores, k=min(teacher_count, student_scores.numel())).values
    deficit = (required_score - selected_student).clamp_min(0.0)
    loss = deficit.pow(2).mean() if deficit.numel() else zero
    count_deficit = (teacher_count_tensor - student_count_tensor).clamp_min(0.0)
    metrics = {
        "teacher_count": teacher_count_tensor.detach(),
        "student_count": student_count_tensor.detach(),
        "count_deficit": count_deficit.detach(),
        "topk_score_mean": selected_student.detach().mean(),
        "violations": deficit.gt(0.0).to(student_logits.dtype).sum().detach(),
    }
    return loss, metrics


def graph_matcher_teacher_match_count_ceiling_loss(
    student: pfm_model.GraphMatcherOutput,
    teacher: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    score_threshold: float = 0.0,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prevent the student from accepting more aligned high-confidence pairs than the teacher."""

    if not math.isfinite(float(score_threshold)):
        raise ValueError("score_threshold must be finite")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    count = min(
        int(positive_count),
        student.logits.size(0) - 1,
        student.logits.size(1) - 1,
        teacher.logits.size(0) - 1,
        teacher.logits.size(1) - 1,
    )
    zero = student.logits.new_zeros(())
    metrics = {
        "teacher_count": zero,
        "student_count": zero,
        "count_excess": zero,
        "excess_score_mean": zero,
        "violations": zero,
    }
    if count <= 0:
        return zero, metrics

    device = student.logits.device
    indices = torch.arange(count, device=device)
    student_logits = student.logits
    teacher_logits = teacher.logits.detach().to(device=device, dtype=student_logits.dtype)

    def true_final_scores(logits: torch.Tensor) -> torch.Tensor:
        true_logits = logits[:count, :count][indices, indices]
        row_dustbin = logits[:count, logits.size(1) - 1]
        col_dustbin = logits[logits.size(0) - 1, :count]
        return true_logits - row_dustbin - col_dustbin

    teacher_scores = true_final_scores(teacher_logits)
    teacher_count_tensor = teacher_scores.ge(float(score_threshold)).to(student_logits.dtype).sum()
    teacher_count = int(teacher_count_tensor.detach().cpu().item())
    student_scores = true_final_scores(student_logits)
    ceiling_score = float(score_threshold) + float(margin)
    student_count_tensor = student_scores.ge(ceiling_score).to(student_logits.dtype).sum()
    count_excess = (student_count_tensor - teacher_count_tensor).clamp_min(0.0)
    if teacher_count >= student_scores.numel():
        return zero, {
            "teacher_count": teacher_count_tensor.detach(),
            "student_count": student_count_tensor.detach(),
            "count_excess": count_excess.detach(),
            "excess_score_mean": zero,
            "violations": zero,
        }

    sorted_student = student_scores.sort(descending=True).values
    excess_scores = sorted_student[teacher_count:]
    excess = (excess_scores - ceiling_score).clamp_min(0.0)
    loss = excess.pow(2).mean() if excess.numel() else zero
    violating_scores = excess_scores[excess.gt(0.0)]
    metrics = {
        "teacher_count": teacher_count_tensor.detach(),
        "student_count": student_count_tensor.detach(),
        "count_excess": count_excess.detach(),
        "excess_score_mean": violating_scores.detach().mean() if violating_scores.numel() else zero,
        "violations": excess.gt(0.0).to(student_logits.dtype).sum().detach(),
    }
    return loss, metrics


def should_apply_positive_dustbin_guard(
    diagnostics: dict[str, float],
    *,
    reject_threshold: float = 1.1,
    margin_threshold: float = -float("inf"),
) -> bool:
    """Return true when positive matches are already being overpowered by dustbin."""

    rejected = float(diagnostics.get("true_match_rejected_by_dustbin_ratio", 0.0))
    margin = float(diagnostics.get("positive_vs_dustbin_margin_mean", 0.0))
    reject_trigger = math.isfinite(float(reject_threshold)) and 0.0 <= float(reject_threshold) <= 1.0
    margin_trigger = math.isfinite(float(margin_threshold))
    return (reject_trigger and rejected > float(reject_threshold)) or (
        margin_trigger and margin < float(margin_threshold)
    )


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


def graph_matcher_pair_acceptance_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    target: float,
    weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if output.pair_accept_logit is None:
        raise ValueError("pair acceptance loss requires output.pair_accept_logit")
    if float(target) not in (0.0, 1.0):
        raise ValueError("target must be 0 or 1")
    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    target_tensor = output.pair_accept_logit.new_tensor(float(target))
    loss = F.binary_cross_entropy_with_logits(output.pair_accept_logit, target_tensor)
    weighted = float(weight) * loss
    probability = float(torch.sigmoid(output.pair_accept_logit.detach()).cpu())
    return weighted, {
        "target": float(target),
        "weight": float(weight),
        "probability": probability,
        "raw_loss": float(loss.detach().cpu()),
    }


def graph_matcher_prune_ranking_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    margin: float = 0.25,
) -> torch.Tensor:
    if output.accept_logits is None:
        return output.logits.new_zeros(())
    accept_logits = output.accept_logits
    count = min(int(positive_count), accept_logits.size(0), accept_logits.size(1))
    if count <= 0:
        return output.logits.new_zeros(())
    positive_logits = accept_logits[:count, :count].diagonal()
    terms: list[torch.Tensor] = []
    if count > 1:
        positive_square = accept_logits[:count, :count]
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=accept_logits.device)
        row_hard = positive_square.masked_fill(~off_diagonal, -float("inf")).max(dim=1).values
        col_hard = positive_square.masked_fill(~off_diagonal, -float("inf")).max(dim=0).values
        terms.append(F.relu(float(margin) - positive_logits + row_hard).mean())
        terms.append(F.relu(float(margin) - positive_logits + col_hard).mean())
    positive_anchor = positive_logits.mean()
    if accept_logits.size(0) > count:
        negative_row_scores = accept_logits[count:, :count].max(dim=1).values
        if negative_row_scores.numel() > 0:
            terms.append(F.relu(float(margin) - positive_anchor + negative_row_scores).mean())
    if accept_logits.size(1) > count:
        negative_col_scores = accept_logits[:count, count:].max(dim=0).values
        if negative_col_scores.numel() > 0:
            terms.append(F.relu(float(margin) - positive_anchor + negative_col_scores).mean())
    if not terms:
        return output.logits.new_zeros(())
    return torch.stack(terms).mean()


def graph_matcher_stop_confidence_loss(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    safe_margin: float = 0.5,
) -> torch.Tensor:
    count = min(int(positive_count), output.logits.size(0) - 1, output.logits.size(1) - 1)
    if count <= 1:
        return output.logits.new_zeros(())
    pair_logits = output.logits[:count, :count]
    diagonal = torch.eye(count, dtype=torch.bool, device=pair_logits.device)
    positive_logits = pair_logits.diagonal()
    row_hard = pair_logits.masked_fill(diagonal, -float("inf")).max(dim=1).values
    col_hard = pair_logits.masked_fill(diagonal, -float("inf")).max(dim=0).values
    safe_assignments = ((positive_logits - row_hard) >= float(safe_margin)) & (
        (positive_logits - col_hard) >= float(safe_margin)
    )
    target = safe_assignments.to(dtype=pair_logits.dtype).mean().detach()
    row_confidence = torch.softmax(pair_logits, dim=1).max(dim=1).values.mean()
    column_confidence = torch.softmax(pair_logits, dim=0).max(dim=0).values.mean()
    confidence = torch.minimum(row_confidence, column_confidence).clamp(1.0e-6, 1.0 - 1.0e-6)
    return _binary_cross_entropy_from_probabilities(confidence, target)


def graph_matcher_dustbin_diagnostics(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
) -> dict[str, float]:
    count = min(int(positive_count), output.logits.size(0) - 1, output.logits.size(1) - 1)
    if count <= 0:
        return {
            "true_match_rejected_by_dustbin_ratio": 0.0,
            "positive_pair_logit_mean": 0.0,
            "positive_dustbin_logit_mean": 0.0,
            "dustbin_logit_mean": 0.0,
            "dustbin_logit_for_true_match_mean": 0.0,
            "positive_vs_dustbin_margin_mean": 0.0,
            "positive_vs_dustbin_margin_median": 0.0,
            "positive_vs_dustbin_margin_p10": 0.0,
            "positive_vs_dustbin_margin_below0_ratio": 0.0,
            "false_match_accepted_ratio": 0.0,
            "accept_logit_mean": 0.0,
            "true_pair_prob_mean": 0.0,
            "dustbin_prob_for_true_match_mean": 0.0,
        }
    device = output.logits.device
    indices = torch.arange(count, device=device)
    pair_logits = output.logits[:count, :count]
    true_logits = pair_logits.diagonal().to(torch.float32)
    row_dustbin = output.logits[:count, output.logits.size(1) - 1].to(torch.float32)
    col_dustbin = output.logits[output.logits.size(0) - 1, :count].to(torch.float32)
    two_sided_dustbin = row_dustbin + col_dustbin
    margin = true_logits - two_sided_dustbin
    row_prob = torch.softmax(output.logits[:count, :], dim=1)
    col_prob = torch.softmax(output.logits[:, :count], dim=0)
    true_pair_prob = row_prob[indices, indices].to(torch.float32)
    dustbin_prob = torch.maximum(row_prob[:, -1], col_prob[-1, :]).to(torch.float32)
    sorted_margin = margin.sort().values
    p10_index = min(sorted_margin.numel() - 1, max(0, int(math.floor((sorted_margin.numel() - 1) * 0.10))))
    rejected = margin.lt(0.0).to(torch.float32)
    if count > 1:
        false_mask = ~torch.eye(count, dtype=torch.bool, device=device)
        false_scores = (pair_logits - row_dustbin[:, None] - col_dustbin[None, :])[false_mask].to(torch.float32)
        false_match_accepted_ratio = float(false_scores.gt(0.0).to(torch.float32).mean().detach().cpu())
    else:
        false_match_accepted_ratio = 0.0
    accept_logit_mean = 0.0
    if output.accept_logits is not None:
        accept_count = min(count, output.accept_logits.size(0), output.accept_logits.size(1))
        if accept_count > 0:
            accept_logit_mean = float(
                output.accept_logits[:accept_count, :accept_count].diagonal().to(torch.float32).mean().detach().cpu()
            )
    return {
        "true_match_rejected_by_dustbin_ratio": float(rejected.mean().detach().cpu()),
        "positive_pair_logit_mean": float(true_logits.mean().detach().cpu()),
        "positive_dustbin_logit_mean": float(two_sided_dustbin.mean().detach().cpu()),
        "dustbin_logit_mean": float(two_sided_dustbin.mean().detach().cpu()),
        "dustbin_logit_for_true_match_mean": float(two_sided_dustbin.mean().detach().cpu()),
        "positive_vs_dustbin_margin_mean": float(margin.mean().detach().cpu()),
        "positive_vs_dustbin_margin_median": float(margin.median().detach().cpu()),
        "positive_vs_dustbin_margin_p10": float(sorted_margin[p10_index].detach().cpu()),
        "positive_vs_dustbin_margin_below0_ratio": float(rejected.mean().detach().cpu()),
        "false_match_accepted_ratio": false_match_accepted_ratio,
        "accept_logit_mean": accept_logit_mean,
        "true_pair_prob_mean": float(true_pair_prob.mean().detach().cpu()),
        "dustbin_prob_for_true_match_mean": float(dustbin_prob.mean().detach().cpu()),
    }


def graph_matcher_candidate_topk_diagnostics(
    model: pfm_model.PlanetaryFeatureMatcher,
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    meta_a: torch.Tensor,
    meta_b: torch.Tensor,
    *,
    positive_count: int,
    topk_values: tuple[int, ...] = (64, 256),
) -> dict[str, float]:
    count = min(int(positive_count), desc_a.size(0), desc_b.size(0))
    result = {f"true_match_in_topk@{int(topk)}": 0.0 for topk in topk_values}
    if count <= 0:
        return result
    indices = torch.arange(count, device=desc_a.device)
    with torch.no_grad():
        for topk in topk_values:
            mask = model.graph_matcher._candidate_mask(
                desc_a.detach(),
                desc_b.detach(),
                meta_a.detach(),
                meta_b.detach(),
                candidate_topk=int(topk),
                positive_pair_count=0,
            )
            if mask.numel() == 0:
                hit_rate = 0.0
            else:
                hit_rate = float(mask[indices, indices].to(torch.float32).mean().detach().cpu())
            result[f"true_match_in_topk@{int(topk)}"] = hit_rate
    return result


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


def graph_matcher_raw_false_match_loss(
    logits: torch.Tensor,
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    positive_count: int,
    negative_topk: int = 8,
    min_raw_similarity: float = 0.75,
    margin: float = 0.25,
    points_b_xy: torch.Tensor | None = None,
    spatial_min_distance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize raw-descriptor-confusable off-diagonal graph logits without boosting dustbin."""

    if negative_topk < 0:
        raise ValueError("negative_topk must be nonnegative")
    if min_raw_similarity < -1.0 or min_raw_similarity > 1.0:
        raise ValueError("min_raw_similarity must be in [-1, 1]")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    if spatial_min_distance < 0.0:
        raise ValueError("spatial_min_distance must be nonnegative")
    count = min(int(positive_count), desc_a.size(0), desc_b.size(0), logits.size(0) - 1, logits.size(1) - 1)
    zero = logits.new_zeros(())
    metrics = {
        "edges": zero,
        "raw_similarity_mean": zero,
        "margin_mean": zero,
    }
    if count <= 1 or negative_topk <= 0:
        return zero, metrics
    raw_similarity = normalize_descriptor_batch(desc_a[:count]) @ normalize_descriptor_batch(desc_b[:count]).T
    diagonal = torch.eye(count, dtype=torch.bool, device=raw_similarity.device)
    candidate_mask = ~diagonal
    if points_b_xy is not None and spatial_min_distance > 0.0:
        if points_b_xy.dim() != 2 or points_b_xy.size(1) != 2:
            raise ValueError("points_b_xy must have shape Nx2")
        if points_b_xy.size(0) < count:
            raise ValueError("points_b_xy must contain at least positive_count rows")
        target_points = points_b_xy[:count].to(device=raw_similarity.device, dtype=raw_similarity.dtype)
        candidate_mask &= torch.cdist(target_points, target_points, p=2.0).ge(float(spatial_min_distance))
    candidate_mask &= raw_similarity.ge(float(min_raw_similarity))
    if not bool(candidate_mask.any()):
        return zero, metrics
    candidate_indices = candidate_mask.nonzero(as_tuple=False)
    candidate_scores = raw_similarity[candidate_mask]
    keep_count = min(int(negative_topk), int(candidate_scores.numel()))
    selected_order = candidate_scores.topk(keep_count).indices
    selected = candidate_indices.index_select(0, selected_order)
    selected_scores = candidate_scores.index_select(0, selected_order)
    rows = selected[:, 0]
    cols = selected[:, 1]
    pair_logits = logits[:count, :count]
    wrong_logits = pair_logits[rows, cols]
    row_true_logits = pair_logits[rows, rows]
    col_true_logits = pair_logits[cols, cols]
    true_reference_logits = torch.minimum(row_true_logits, col_true_logits)
    selected_margin = true_reference_logits - wrong_logits
    loss = (float(margin) - selected_margin).clamp_min(0.0).pow(2).mean()
    metrics = {
        "edges": logits.new_tensor(float(keep_count)),
        "raw_similarity_mean": selected_scores.detach().mean(),
        "margin_mean": selected_margin.detach().mean(),
    }
    return loss, metrics


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


def keypoint_offset_supervision_loss(keypoint_offsets: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    if keypoint_offsets.dim() != 4 or keypoint_offsets.size(0) != 1 or keypoint_offsets.size(1) != 2:
        raise ValueError("keypoint_offsets must have shape 1x2xHxW")
    if points_xy.dim() != 2 or points_xy.size(1) != 2:
        raise ValueError("points_xy must have shape Nx2")
    if points_xy.numel() == 0:
        return keypoint_offsets.sum() * 0.0
    height, width = keypoint_offsets.shape[-2:]
    rounded = points_xy.round().to(device=keypoint_offsets.device, dtype=torch.long)
    x = rounded[:, 0].clamp(0, width - 1)
    y = rounded[:, 1].clamp(0, height - 1)
    centers = torch.stack([x, y], dim=1).to(device=keypoint_offsets.device, dtype=torch.float32)
    targets = points_xy.to(device=keypoint_offsets.device, dtype=torch.float32) - centers
    targets = targets.clamp(-0.5, 0.5)
    predictions = keypoint_offsets[0].permute(1, 2, 0)[y, x].to(torch.float32)
    return F.smooth_l1_loss(predictions, targets)


def _binary_cross_entropy_from_probabilities(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities_f32 = probabilities.to(torch.float32).clamp(1.0e-6, 1.0 - 1.0e-6)
    targets_f32 = targets.to(device=probabilities_f32.device, dtype=torch.float32)
    probabilities_f32, targets_f32 = torch.broadcast_tensors(probabilities_f32, targets_f32)
    loss = -(targets_f32 * probabilities_f32.log() + (1.0 - targets_f32) * torch.log1p(-probabilities_f32))
    return loss.mean()


def _binary_map_point_loss(
    score_map: torch.Tensor,
    positive_points_xy: torch.Tensor,
    negative_points_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    if score_map.dim() != 4 or score_map.size(0) != 1 or score_map.size(1) != 1:
        raise ValueError("score_map must have shape 1x1xHxW")
    terms: list[torch.Tensor] = []
    if positive_points_xy.numel() > 0:
        positive_scores = sample_descriptors(score_map, positive_points_xy).reshape(-1).to(torch.float32)
        terms.append(
            _binary_cross_entropy_from_probabilities(positive_scores, torch.ones_like(positive_scores))
        )
    if negative_points_xy is not None and negative_points_xy.numel() > 0:
        negative_scores = sample_descriptors(score_map, negative_points_xy).reshape(-1).to(torch.float32)
        terms.append(
            _binary_cross_entropy_from_probabilities(negative_scores, torch.zeros_like(negative_scores))
        )
    if not terms:
        return score_map.sum() * 0.0
    return torch.stack(terms).mean()


def matchability_supervision_loss(
    matchability: torch.Tensor,
    positive_points_xy: torch.Tensor,
    negative_points_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    return _binary_map_point_loss(matchability, positive_points_xy, negative_points_xy)


def no_match_prior_supervision_loss(
    no_match_prior: torch.Tensor,
    no_match_points_xy: torch.Tensor,
    positive_points_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    positive_points = no_match_points_xy
    negative_points = positive_points_xy if positive_points_xy is not None else None
    return _binary_map_point_loss(no_match_prior, positive_points, negative_points)


def descriptor_uncertainty_supervision_loss(
    descriptor_uncertainty: torch.Tensor,
    false_or_no_match_points_xy: torch.Tensor,
    positive_points_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    positive_points = false_or_no_match_points_xy
    negative_points = positive_points_xy if positive_points_xy is not None else None
    return _binary_map_point_loss(descriptor_uncertainty, positive_points, negative_points)


def _rotation_k(rotation_degrees: int) -> int:
    if int(rotation_degrees) % 90 != 0:
        raise ValueError("rotation_degrees must be a multiple of 90")
    return (int(rotation_degrees) // 90) % 4


def rotate_points_xy_90(points_xy: torch.Tensor, height: int, width: int, rotation_degrees: int) -> torch.Tensor:
    if points_xy.dim() != 2 or points_xy.size(1) != 2:
        raise ValueError("points_xy must have shape Nx2")
    k = _rotation_k(rotation_degrees)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    if k == 0:
        return points_xy.clone()
    if k == 1:
        return torch.stack([y, points_xy.new_tensor(float(width - 1)) - x], dim=1)
    if k == 2:
        return torch.stack(
            [
                points_xy.new_tensor(float(width - 1)) - x,
                points_xy.new_tensor(float(height - 1)) - y,
            ],
            dim=1,
        )
    return torch.stack([points_xy.new_tensor(float(height - 1)) - y, x], dim=1)


def rotate_orientation_vectors(vectors: torch.Tensor, rotation_degrees: int) -> torch.Tensor:
    if vectors.dim() != 2 or vectors.size(1) != 2:
        raise ValueError("orientation vectors must have shape Nx2")
    k = _rotation_k(rotation_degrees)
    if k == 0:
        return vectors
    x = vectors[:, 0]
    y = vectors[:, 1]
    if k == 1:
        return torch.stack([-y, x], dim=1)
    if k == 2:
        return -vectors
    return torch.stack([y, -x], dim=1)


def _rotation_matrix_2d(rotation_degrees: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = _rotation_k(rotation_degrees)
    if k == 0:
        values = [1.0, 0.0, 0.0, 1.0]
    elif k == 1:
        values = [0.0, -1.0, 1.0, 0.0]
    elif k == 2:
        values = [-1.0, 0.0, 0.0, -1.0]
    else:
        values = [0.0, 1.0, -1.0, 0.0]
    return torch.tensor(values, dtype=dtype, device=device).view(2, 2)


def rotation_descriptor_consistency_loss(
    descriptors_reference: torch.Tensor,
    descriptors_rotated: torch.Tensor,
    points_xy: torch.Tensor,
    rotation_degrees: int,
    *,
    max_points: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    selected = _limited_points(points_xy, max_points=max_points, generator=generator)
    if selected.numel() == 0:
        return descriptors_reference.sum() * 0.0
    rotated_points = rotate_points_xy_90(selected, descriptors_reference.size(2), descriptors_reference.size(3), rotation_degrees)
    reference = normalize_descriptor_batch(sample_descriptors(descriptors_reference, selected))
    rotated = normalize_descriptor_batch(sample_descriptors(descriptors_rotated, rotated_points))
    return (1.0 - (reference * rotated).sum(dim=1).clamp(-1.0, 1.0)).mean()


def orientation_consistency_loss(
    orientation_reference: torch.Tensor,
    orientation_rotated: torch.Tensor,
    points_xy: torch.Tensor,
    rotation_degrees: int,
    *,
    max_points: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    selected = _limited_points(points_xy, max_points=max_points, generator=generator)
    if selected.numel() == 0:
        return orientation_reference.sum() * 0.0
    rotated_points = rotate_points_xy_90(selected, orientation_reference.size(2), orientation_reference.size(3), rotation_degrees)
    reference = normalize_descriptor_batch(sample_descriptors(orientation_reference, selected), eps=1.0e-6)
    rotated = normalize_descriptor_batch(sample_descriptors(orientation_rotated, rotated_points), eps=1.0e-6)
    target = rotate_orientation_vectors(reference, rotation_degrees)
    return (1.0 - (target * rotated).sum(dim=1).clamp(-1.0, 1.0)).mean()


def scale_consistency_loss(
    scale_reference: torch.Tensor,
    scale_rotated: torch.Tensor,
    points_xy: torch.Tensor,
    rotation_degrees: int,
    *,
    max_points: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    selected = _limited_points(points_xy, max_points=max_points, generator=generator)
    if selected.numel() == 0:
        return scale_reference.sum() * 0.0
    rotated_points = rotate_points_xy_90(selected, scale_reference.size(2), scale_reference.size(3), rotation_degrees)
    reference = sample_descriptors(scale_reference.clamp_min(1.0e-6).log(), selected)
    rotated = sample_descriptors(scale_rotated.clamp_min(1.0e-6).log(), rotated_points)
    return F.smooth_l1_loss(rotated, reference)


def affine_consistency_loss(
    affine_reference: torch.Tensor,
    affine_rotated: torch.Tensor,
    points_xy: torch.Tensor,
    rotation_degrees: int,
    *,
    max_points: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    selected = _limited_points(points_xy, max_points=max_points, generator=generator)
    if selected.numel() == 0:
        return affine_reference.sum() * 0.0
    rotated_points = rotate_points_xy_90(selected, affine_reference.size(2), affine_reference.size(3), rotation_degrees)
    reference = sample_descriptors(affine_reference, selected).view(-1, 2, 2)
    rotated = sample_descriptors(affine_rotated, rotated_points).view(-1, 2, 2)
    rotation = _rotation_matrix_2d(rotation_degrees, device=reference.device, dtype=reference.dtype)
    target = rotation @ reference @ rotation.transpose(0, 1)
    return F.smooth_l1_loss(rotated, target)


def affine_regularization_loss(
    affine: torch.Tensor,
    *,
    identity_weight: float = 1.0,
    determinant_weight: float = 1.0,
    condition_weight: float = 0.25,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if affine.dim() != 4 or affine.size(1) != 4:
        raise ValueError("affine must have shape Bx4xHxW")
    matrix = affine.to(dtype=torch.float32).permute(0, 2, 3, 1).reshape(-1, 2, 2)
    identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
    det = torch.linalg.det(matrix)
    singular_values = torch.linalg.svdvals(matrix)
    condition = singular_values[:, 0] / singular_values[:, 1].clamp_min(1.0e-4)
    identity_loss = (matrix - identity).pow(2).mean()
    determinant_loss = (det - 1.0).pow(2).mean()
    condition_loss = condition.clamp_min(1.0).log().pow(2).mean()
    loss = (
        float(identity_weight) * identity_loss
        + float(determinant_weight) * determinant_loss
        + float(condition_weight) * condition_loss
    )
    if not return_metrics:
        return loss
    metrics = {
        "affine_det_mean": det.mean(),
        "affine_det_std": det.std(unbiased=False),
        "affine_condition_mean": condition.mean(),
        "affine_condition_max": condition.max(),
    }
    return loss, metrics


def compute_descriptor_maps(model, pair: SyntheticPair) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        model.descriptor_map_single(pair.view_a.unsqueeze(0)),
        model.descriptor_map_single(pair.view_b.unsqueeze(0)),
    )


def learned_training_sparse_maps_single(
    model: pfm_model.PlanetaryFeatureMatcher,
    image: torch.Tensor,
    *,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    activation_checkpointing: bool = False,
) -> TrainingSparseMaps:
    if image.dim() != 4:
        raise ValueError("image must have shape BxCxHxW")
    features = (
        model.backbone(image, activation_checkpointing=True)
        if activation_checkpointing
        else model.backbone(image)
    )
    if hasattr(model, "dual_fpn"):
        p2_keypoint, p2_descriptor = (
            model.dual_fpn(features, activation_checkpointing=True)
            if activation_checkpointing
            else model.dual_fpn(features)
        )
        sparse = (
            model.sparse_head(p2_keypoint, p2_descriptor, activation_checkpointing=True)
            if activation_checkpointing
            else model.sparse_head(p2_keypoint, p2_descriptor)
        )
    else:
        sparse = (
            model.sparse_head(features[1], activation_checkpointing=True)
            if activation_checkpointing
            else model.sparse_head(features[1])
        )
    descriptors = sparse.descriptors
    if train_blended_descriptors:
        descriptors = model.fuse_descriptor_maps(descriptors, image, texture_blend_weight=texture_blend_weight)
    heatmap = sparse.heatmap
    quality = None
    quality_score_mode = getattr(model.config, "quality_score_mode", "soft")
    if quality_score_mode != "raw" and hasattr(model, "quality_head") and hasattr(model, "dense_head"):
        texture_saliency = pfm_model.make_rotation_invariant_texture_saliency(
            image,
            sparse.heatmap.size(2),
            sparse.heatmap.size(3),
        )
        dense = model.dense_head(features[0], features[0])
        dense_confidence = F.interpolate(dense.confidence, size=sparse.heatmap.shape[-2:], mode="nearest")
        quality = model.quality_head(descriptors, sparse.heatmap, texture_saliency, dense_confidence)
        heatmap = pfm_model.apply_quality_score_mode(
            sparse.heatmap,
            quality,
            mode=quality_score_mode,
        )
    return TrainingSparseMaps(
        descriptors=descriptors,
        heatmap=heatmap,
        keypoint_offsets=sparse.keypoint_offsets,
        matchability=sparse.matchability,
        descriptor_uncertainty=sparse.descriptor_uncertainty,
        no_match_prior=sparse.no_match_prior,
        scale=sparse.scale,
        orientation=sparse.orientation,
        affine=sparse.affine,
        quality=quality,
    )


def learned_descriptor_and_heatmap_single(
    model: pfm_model.PlanetaryFeatureMatcher,
    image: torch.Tensor,
    *,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    activation_checkpointing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    sparse_maps = learned_training_sparse_maps_single(
        model,
        image,
        train_blended_descriptors=train_blended_descriptors,
        texture_blend_weight=texture_blend_weight,
        activation_checkpointing=activation_checkpointing,
    )
    return sparse_maps.descriptors, sparse_maps.heatmap


def compute_training_descriptor_map(
    model: pfm_model.PlanetaryFeatureMatcher,
    image: torch.Tensor,
    *,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    if image.dim() != 4:
        raise ValueError("image must have shape BxCxHxW")
    if train_blended_descriptors:
        if hasattr(model, "descriptor_map_single"):
            return (
                model.descriptor_map_single(
                    image,
                    texture_blend_weight=texture_blend_weight,
                    activation_checkpointing=True,
                )
                if activation_checkpointing
                else model.descriptor_map_single(image, texture_blend_weight=texture_blend_weight)
            )
        descriptors, _ = learned_descriptor_and_heatmap_single(
            model,
            image,
            train_blended_descriptors=True,
            texture_blend_weight=texture_blend_weight,
            activation_checkpointing=activation_checkpointing,
        )
        return descriptors
    if hasattr(model, "learned_descriptor_map_single"):
        return (
            model.learned_descriptor_map_single(image, activation_checkpointing=True)
            if activation_checkpointing
            else model.learned_descriptor_map_single(image)
        )
    descriptors, _ = learned_descriptor_and_heatmap_single(
        model,
        image,
        train_blended_descriptors=False,
        texture_blend_weight=texture_blend_weight,
        activation_checkpointing=activation_checkpointing,
    )
    return descriptors


def compute_student_teacher_descriptor_map_single(
    model: pfm_model.PlanetaryFeatureMatcher,
    image: torch.Tensor,
    *,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    activation_checkpointing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    student = compute_training_descriptor_map(
        model,
        image,
        train_blended_descriptors=train_blended_descriptors,
        texture_blend_weight=texture_blend_weight,
        activation_checkpointing=activation_checkpointing,
    )
    with torch.no_grad():
        teacher = model.texture_descriptor_map_single(image)
    return student, teacher


def compute_student_teacher_descriptor_maps(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair: SyntheticPair,
    *,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    include_heatmaps: bool = False,
    activation_checkpointing: bool = False,
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
        activation_checkpointing=activation_checkpointing,
    )
    student_b, heatmap_b = learned_descriptor_and_heatmap_single(
        model,
        pair.view_b.unsqueeze(0),
        train_blended_descriptors=train_blended_descriptors,
        texture_blend_weight=texture_blend_weight,
        activation_checkpointing=activation_checkpointing,
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
    train_keypoint_offset_head: bool = False,
    train_geometry_head: bool = False,
    train_texture_adapter: bool = False,
    train_descriptor_fusion: bool = False,
    train_quality_head: bool = False,
    train_reliability_head: bool = False,
    train_graph_matcher: bool = False,
    train_graph_calibration_only: bool = False,
    train_pair_accept_head_only: bool = False,
) -> list[torch.nn.Parameter]:
    selected: list[torch.nn.Parameter] = []

    def graph_matcher_trainable(name: str) -> bool:
        if not name.startswith("graph_matcher."):
            return False
        if train_pair_accept_head_only:
            return name.startswith("graph_matcher.pair_accept_head.") or name.startswith(
                "graph_matcher.pair_accept_context_head."
            )
        if not train_graph_calibration_only:
            return True
        return (
            name.startswith("graph_matcher.geometry_bias.")
            or name.startswith("graph_matcher.accept_head.")
            or name.startswith("graph_matcher.pair_accept_head.")
            or name.startswith("graph_matcher.pair_accept_context_head.")
            or name
            in {
                "graph_matcher.logit_scale",
                "graph_matcher.raw_score_temperature",
                "graph_matcher.graph_delta_scale",
                "graph_matcher.accept_logit_scale",
                "graph_matcher.dustbin_bias",
            }
        )

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
            train_keypoint_offset_head and name.startswith("sparse_head.keypoint_offsets")
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
        trainable = trainable or (
            train_reliability_head
            and (
                name.startswith("sparse_head.matchability")
                or name.startswith("sparse_head.descriptor_uncertainty")
                or name.startswith("sparse_head.no_match_prior")
            )
        )
        trainable = trainable or ((train_graph_matcher or train_pair_accept_head_only) and graph_matcher_trainable(name))
        if trainable:
            parameter.requires_grad_(True)
            selected.append(parameter)
        else:
            parameter.requires_grad_(False)
    return selected


def freeze_non_trainable_batch_norm_statistics(model: torch.nn.Module) -> None:
    """Keep BatchNorm running statistics fixed for modules whose affine parameters are frozen."""

    batch_norm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    for module in model.modules():
        if not isinstance(module, batch_norm_types):
            continue
        parameters = list(module.parameters(recurse=False))
        if not parameters or not any(parameter.requires_grad for parameter in parameters):
            module.eval()


def should_freeze_non_trainable_batch_norm_statistics(args: argparse.Namespace) -> bool:
    """Return whether head-only training should keep frozen BatchNorm statistics fixed."""

    return bool(
        getattr(args, "train_keypoint_offset_head_only", False)
        or getattr(args, "train_pair_accept_head_only", False)
    )


def apply_extractor_freeze_warmup(
    model: pfm_model.PlanetaryFeatureMatcher,
    original_requires_grad: dict[str, bool],
    *,
    freeze_extractor: bool,
) -> None:
    """Temporarily freeze non-GraphMatcher parameters while preserving the selected trainable mask."""

    for name, parameter in model.named_parameters():
        original = bool(original_requires_grad.get(name, parameter.requires_grad))
        if freeze_extractor and not name.startswith("graph_matcher."):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(original)


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


def amp_dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported AMP dtype: {name}")


def autocast_context(device: torch.device, *, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_grad_scaler(device: torch.device, *, enabled: bool, dtype: torch.dtype):
    if not enabled or device.type != "cuda" or dtype is not torch.float16:
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


def grad_scaler_scale(grad_scaler) -> float:
    if grad_scaler is None or not hasattr(grad_scaler, "get_scale"):
        return 0.0
    return float(grad_scaler.get_scale())


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
    keypoint_weight: float = 0.0,
    keypoint_negative_weight: float = 0.01,
    false_matches: dict[str, FalseMatchLabels] | None = None,
    false_match_weight: float = 0.0,
    false_match_max_points: int = 0,
    false_match_max_score: float = 0.25,
    false_match_pair_paths: list[Path] | None = None,
    false_match_probability: float = 0.0,
    online_false_match_weight: float = 0.0,
    online_false_match_max_points: int = 0,
    online_false_match_max_score: float = 0.25,
    online_false_match_max_keypoints: int = 256,
    online_false_match_max_matches: int = 0,
    online_false_match_min_score: float = -1.0,
    online_false_match_min_margin: float = 0.02,
    online_false_match_threshold_px: float = 5.0,
    illumination_consistency_pairs: dict[Path, SyntheticPair] | None = None,
    illumination_consistency_weight: float = 0.0,
    illumination_consistency_max_points: int = 0,
    illumination_consistency_probability: float = 1.0,
    illumination_match_pairs: dict[Path, SyntheticPair] | None = None,
    illumination_match_weight: float = 0.0,
    illumination_match_probability: float = 1.0,
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None = None,
    pose_balanced_sampling: bool = False,
    pose_difficulty_loss_weight: float = 0.0,
    keypoint_offset_weight: float = 0.0,
    selected_keypoint_offset_weight: float = 0.0,
    selected_keypoint_offset_max_points: int = 256,
    selected_keypoint_offset_inverse_radius_px: float = 1.5,
    graph_matcher_loss_weight: float = 0.0,
    graph_matcher_metadata_mode: str = "calibrated",
    graph_matcher_no_match_points: int = 0,
    graph_matcher_no_match_weight: float = 0.0,
    graph_matcher_no_match_min_distance: float = 4.0,
    graph_matcher_assignment_weight: float = 0.0,
    graph_matcher_accept_weight: float = 0.0,
    graph_matcher_accept_negative_topk: int = 8,
    graph_matcher_prune_ranking_weight: float = 0.0,
    graph_matcher_prune_ranking_margin: float = 0.25,
    graph_matcher_stop_confidence_weight: float = 0.0,
    graph_matcher_stop_confidence_margin: float = 0.5,
    graph_matcher_raw_preservation_weight: float = 0.0,
    graph_matcher_raw_preservation_margin: float = 1.0,
    graph_matcher_raw_preservation_raw_margin: float = 0.05,
    graph_matcher_hard_negative_dustbin_weight: float = 0.0,
    graph_matcher_hard_negative_dustbin_topk: int = 8,
    graph_matcher_hard_negative_dustbin_margin: float = 0.25,
    graph_matcher_hard_negative_dustbin_spatial_min_distance: float = 0.0,
    graph_matcher_dustbin_warmup_steps: int = 0,
    graph_matcher_dustbin_ramp_steps: int = 0,
    graph_matcher_positive_dustbin_margin_weight: float = 0.0,
    graph_matcher_positive_dustbin_margin: float = 0.0,
    graph_matcher_true_match_margin_weight: float = 0.0,
    graph_matcher_true_match_margin: float = 0.25,
    graph_matcher_true_geometry_match_count_floor_weight: float = 0.0,
    graph_matcher_true_geometry_match_count_floor_threshold: float = 0.0,
    graph_matcher_true_geometry_match_count_floor_margin: float = 0.0,
    graph_matcher_final_false_match_weight: float = 0.0,
    graph_matcher_mined_false_match_weight: float = 0.0,
    graph_matcher_mined_false_match_loss_cap: float = 0.0,
    graph_matcher_mined_false_match_reference_margin: float = -1.0,
    graph_matcher_final_false_match_topk: int = 8,
    graph_matcher_final_false_match_min_score: float = 0.0,
    graph_matcher_final_false_match_margin: float = 0.25,
    graph_matcher_final_false_match_spatial_min_distance: float = 0.0,
    graph_matcher_raw_false_match_weight: float = 0.0,
    graph_matcher_raw_false_match_topk: int = 8,
    graph_matcher_raw_false_match_min_similarity: float = 0.75,
    graph_matcher_raw_false_match_margin: float = 0.25,
    graph_matcher_raw_false_match_spatial_min_distance: float = 0.0,
    graph_matcher_ransac_consistency_weight: float = 0.0,
    graph_matcher_ransac_consistency_topk: int = 8,
    graph_matcher_ransac_consistency_residual_threshold_px: float = 3.0,
    graph_matcher_ransac_consistency_min_score: float = 0.0,
    graph_matcher_ransac_consistency_margin: float = 0.25,
    graph_matcher_warp_outlier_weight: float = 0.0,
    graph_matcher_warp_outlier_topk: int = 8,
    graph_matcher_warp_outlier_residual_threshold_px: float = 3.0,
    graph_matcher_warp_outlier_min_score: float = 0.0,
    graph_matcher_warp_outlier_margin: float = 0.25,
    graph_matcher_warp_outlier_accept_weight: float = 0.0,
    graph_matcher_warp_outlier_accept_topk: int = 8,
    graph_matcher_warp_outlier_accept_residual_threshold_px: float = 3.0,
    graph_matcher_warp_outlier_accept_min_score: float = 0.0,
    graph_matcher_warp_soft_boundary_weight: float = 0.0,
    graph_matcher_warp_soft_boundary_topk: int = 8,
    graph_matcher_warp_soft_boundary_lower_residual_px: float = 5.0,
    graph_matcher_warp_soft_boundary_upper_residual_px: float = 8.0,
    graph_matcher_warp_soft_boundary_min_score: float = 0.0,
    graph_matcher_pair_acceptance_loss_weight: float = 0.0,
    graph_matcher_train_candidate_topk: int = 0,
    graph_matcher_semi_dense_no_match_points: int = 0,
    graph_matcher_semi_dense_min_score: float = 0.0,
    graph_matcher_online_false_no_match: bool = False,
    graph_matcher_train_max_attention_layers: int = 0,
    graph_matcher_train_random_attention_layers: bool = False,
    graph_matcher_train_max_attention_work_fraction: float = 1.0,
    graph_matcher_train_width_keep_ratio: float = 1.0,
    graph_matcher_deep_supervision_depths: list[int] | None = None,
    graph_matcher_deep_supervision_weight: float = 0.0,
    graph_matcher_depth_distillation_weight: float = 0.0,
    graph_matcher_depth_distillation_teacher_layers: int = 0,
    graph_matcher_depth_distillation_temperature: float = 1.0,
    graph_matcher_teacher_guard_model: pfm_model.PlanetaryFeatureMatcher | None = None,
    graph_matcher_teacher_guard_weight: float = 0.0,
    graph_matcher_teacher_guard_positive_margin_tolerance: float = 0.0,
    graph_matcher_teacher_guard_false_margin_tolerance: float = 0.0,
    graph_matcher_teacher_score_floor_weight: float = 0.0,
    graph_matcher_teacher_score_floor_tolerance: float = 0.0,
    graph_matcher_teacher_score_floor_min_score: float = 0.0,
    graph_matcher_teacher_match_count_floor_weight: float = 0.0,
    graph_matcher_teacher_match_count_floor_threshold: float = 0.0,
    graph_matcher_teacher_match_count_floor_margin: float = 0.0,
    graph_matcher_teacher_match_count_ceiling_weight: float = 0.0,
    graph_matcher_teacher_match_count_ceiling_threshold: float = 0.0,
    graph_matcher_teacher_match_count_ceiling_margin: float = 0.0,
    graph_matcher_teacher_distillation_weight: float = 0.0,
    graph_matcher_teacher_distillation_temperature: float = 1.0,
    graph_matcher_positive_dustbin_guard_reject_threshold: float = 1.1,
    graph_matcher_positive_dustbin_guard_margin_threshold: float = -float("inf"),
    matchability_weight: float = 0.0,
    descriptor_uncertainty_weight: float = 0.0,
    no_match_prior_weight: float = 0.0,
    reliability_negative_points: int = 0,
    reliability_negative_min_distance: float = 4.0,
    rotation_descriptor_consistency_weight: float = 0.0,
    orientation_consistency_weight: float = 0.0,
    scale_consistency_weight: float = 0.0,
    affine_consistency_weight: float = 0.0,
    affine_regularization_weight: float = 0.0,
    rotation_consistency_degrees: list[int] | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    grad_scaler=None,
    activation_checkpointing: bool = False,
    training_spatial_bins: int = 0,
    training_crop_size: int = 0,
    training_max_image_size: int = 0,
    forced_pair_paths: list[Path] | None = None,
    prefetched_pairs: dict[Path, SyntheticPair] | None = None,
    pair_acceptance_targets: dict[Path, tuple[float, float]] | None = None,
    true_geometry_match_count_targets: dict[Path, tuple[float, float]] | None = None,
    pair_cache: PairArchiveCache | None = None,
    training_step: int = 0,
    freeze_extractor_warmup_active: bool = False,
) -> dict[str, float]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if selected_keypoint_offset_max_points < 0:
        raise ValueError("selected_keypoint_offset_max_points must be nonnegative")
    for name, value in (
        ("matchability_weight", matchability_weight),
        ("descriptor_uncertainty_weight", descriptor_uncertainty_weight),
        ("no_match_prior_weight", no_match_prior_weight),
        ("keypoint_weight", keypoint_weight),
        ("keypoint_negative_weight", keypoint_negative_weight),
        ("keypoint_offset_weight", keypoint_offset_weight),
        ("selected_keypoint_offset_weight", selected_keypoint_offset_weight),
        ("selected_keypoint_offset_inverse_radius_px", selected_keypoint_offset_inverse_radius_px),
        ("rotation_descriptor_consistency_weight", rotation_descriptor_consistency_weight),
        ("orientation_consistency_weight", orientation_consistency_weight),
        ("scale_consistency_weight", scale_consistency_weight),
        ("affine_consistency_weight", affine_consistency_weight),
        ("affine_regularization_weight", affine_regularization_weight),
        ("graph_matcher_deep_supervision_weight", graph_matcher_deep_supervision_weight),
        ("graph_matcher_depth_distillation_weight", graph_matcher_depth_distillation_weight),
        ("graph_matcher_positive_dustbin_margin_weight", graph_matcher_positive_dustbin_margin_weight),
        ("graph_matcher_true_match_margin_weight", graph_matcher_true_match_margin_weight),
        (
            "graph_matcher_true_geometry_match_count_floor_weight",
            graph_matcher_true_geometry_match_count_floor_weight,
        ),
        ("graph_matcher_final_false_match_weight", graph_matcher_final_false_match_weight),
        ("graph_matcher_mined_false_match_weight", graph_matcher_mined_false_match_weight),
        ("graph_matcher_mined_false_match_loss_cap", graph_matcher_mined_false_match_loss_cap),
        ("graph_matcher_raw_false_match_weight", graph_matcher_raw_false_match_weight),
        ("graph_matcher_ransac_consistency_weight", graph_matcher_ransac_consistency_weight),
        ("graph_matcher_warp_outlier_weight", graph_matcher_warp_outlier_weight),
        ("graph_matcher_warp_outlier_accept_weight", graph_matcher_warp_outlier_accept_weight),
        ("graph_matcher_warp_soft_boundary_weight", graph_matcher_warp_soft_boundary_weight),
        ("graph_matcher_pair_acceptance_loss_weight", graph_matcher_pair_acceptance_loss_weight),
        ("graph_matcher_teacher_guard_weight", graph_matcher_teacher_guard_weight),
        ("graph_matcher_teacher_score_floor_weight", graph_matcher_teacher_score_floor_weight),
        ("graph_matcher_teacher_match_count_floor_weight", graph_matcher_teacher_match_count_floor_weight),
        ("graph_matcher_teacher_match_count_ceiling_weight", graph_matcher_teacher_match_count_ceiling_weight),
        ("graph_matcher_teacher_distillation_weight", graph_matcher_teacher_distillation_weight),
        (
            "graph_matcher_teacher_guard_positive_margin_tolerance",
            graph_matcher_teacher_guard_positive_margin_tolerance,
        ),
        (
            "graph_matcher_teacher_guard_false_margin_tolerance",
            graph_matcher_teacher_guard_false_margin_tolerance,
        ),
        (
            "graph_matcher_teacher_score_floor_tolerance",
            graph_matcher_teacher_score_floor_tolerance,
        ),
        (
            "graph_matcher_teacher_match_count_floor_margin",
            graph_matcher_teacher_match_count_floor_margin,
        ),
        (
            "graph_matcher_teacher_match_count_ceiling_margin",
            graph_matcher_teacher_match_count_ceiling_margin,
        ),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if not math.isfinite(float(graph_matcher_teacher_score_floor_min_score)):
        raise ValueError("graph_matcher_teacher_score_floor_min_score must be finite")
    if not math.isfinite(float(graph_matcher_teacher_match_count_floor_threshold)):
        raise ValueError("graph_matcher_teacher_match_count_floor_threshold must be finite")
    if not math.isfinite(float(graph_matcher_teacher_match_count_ceiling_threshold)):
        raise ValueError("graph_matcher_teacher_match_count_ceiling_threshold must be finite")
    if (
        not math.isfinite(float(graph_matcher_mined_false_match_reference_margin))
        or graph_matcher_mined_false_match_reference_margin < -1.0
    ):
        raise ValueError("graph_matcher_mined_false_match_reference_margin must be finite and >= -1")
    if graph_matcher_true_match_margin < 0.0:
        raise ValueError("graph_matcher_true_match_margin must be nonnegative")
    if not math.isfinite(float(graph_matcher_true_geometry_match_count_floor_threshold)):
        raise ValueError("graph_matcher_true_geometry_match_count_floor_threshold must be finite")
    if graph_matcher_true_geometry_match_count_floor_margin < 0.0:
        raise ValueError("graph_matcher_true_geometry_match_count_floor_margin must be nonnegative")
    if graph_matcher_final_false_match_topk < 0:
        raise ValueError("graph_matcher_final_false_match_topk must be nonnegative")
    if graph_matcher_final_false_match_min_score < 0.0:
        raise ValueError("graph_matcher_final_false_match_min_score must be nonnegative")
    if graph_matcher_final_false_match_margin < 0.0:
        raise ValueError("graph_matcher_final_false_match_margin must be nonnegative")
    if graph_matcher_final_false_match_spatial_min_distance < 0.0:
        raise ValueError("graph_matcher_final_false_match_spatial_min_distance must be nonnegative")
    if graph_matcher_raw_false_match_topk < 0:
        raise ValueError("graph_matcher_raw_false_match_topk must be nonnegative")
    if graph_matcher_raw_false_match_min_similarity < -1.0 or graph_matcher_raw_false_match_min_similarity > 1.0:
        raise ValueError("graph_matcher_raw_false_match_min_similarity must be in [-1, 1]")
    if graph_matcher_raw_false_match_margin < 0.0:
        raise ValueError("graph_matcher_raw_false_match_margin must be nonnegative")
    if graph_matcher_raw_false_match_spatial_min_distance < 0.0:
        raise ValueError("graph_matcher_raw_false_match_spatial_min_distance must be nonnegative")
    if graph_matcher_ransac_consistency_topk < 0:
        raise ValueError("graph_matcher_ransac_consistency_topk must be nonnegative")
    if graph_matcher_ransac_consistency_residual_threshold_px < 0.0:
        raise ValueError("graph_matcher_ransac_consistency_residual_threshold_px must be nonnegative")
    if graph_matcher_ransac_consistency_min_score < 0.0:
        raise ValueError("graph_matcher_ransac_consistency_min_score must be nonnegative")
    if graph_matcher_ransac_consistency_margin < 0.0:
        raise ValueError("graph_matcher_ransac_consistency_margin must be nonnegative")
    if graph_matcher_warp_outlier_topk < 0:
        raise ValueError("graph_matcher_warp_outlier_topk must be nonnegative")
    if graph_matcher_warp_outlier_residual_threshold_px < 0.0:
        raise ValueError("graph_matcher_warp_outlier_residual_threshold_px must be nonnegative")
    if graph_matcher_warp_outlier_min_score < 0.0:
        raise ValueError("graph_matcher_warp_outlier_min_score must be nonnegative")
    if graph_matcher_warp_outlier_margin < 0.0:
        raise ValueError("graph_matcher_warp_outlier_margin must be nonnegative")
    if graph_matcher_warp_outlier_accept_topk < 0:
        raise ValueError("graph_matcher_warp_outlier_accept_topk must be nonnegative")
    if graph_matcher_warp_outlier_accept_residual_threshold_px < 0.0:
        raise ValueError("graph_matcher_warp_outlier_accept_residual_threshold_px must be nonnegative")
    if graph_matcher_warp_outlier_accept_min_score < 0.0:
        raise ValueError("graph_matcher_warp_outlier_accept_min_score must be nonnegative")
    if graph_matcher_warp_soft_boundary_topk < 0:
        raise ValueError("graph_matcher_warp_soft_boundary_topk must be nonnegative")
    if (
        not math.isfinite(float(graph_matcher_warp_soft_boundary_lower_residual_px))
        or graph_matcher_warp_soft_boundary_lower_residual_px < 0.0
    ):
        raise ValueError("graph_matcher_warp_soft_boundary_lower_residual_px must be finite and nonnegative")
    if (
        not math.isfinite(float(graph_matcher_warp_soft_boundary_upper_residual_px))
        or graph_matcher_warp_soft_boundary_upper_residual_px
        <= graph_matcher_warp_soft_boundary_lower_residual_px
    ):
        raise ValueError(
            "graph_matcher_warp_soft_boundary_upper_residual_px must be finite and greater than lower"
        )
    if graph_matcher_warp_soft_boundary_min_score < 0.0:
        raise ValueError("graph_matcher_warp_soft_boundary_min_score must be nonnegative")
    if graph_matcher_depth_distillation_teacher_layers < 0:
        raise ValueError("graph_matcher_depth_distillation_teacher_layers must be nonnegative")
    if (
        not math.isfinite(float(graph_matcher_depth_distillation_temperature))
        or graph_matcher_depth_distillation_temperature <= 0.0
    ):
        raise ValueError("graph_matcher_depth_distillation_temperature must be positive and finite")
    if (
        not math.isfinite(float(graph_matcher_teacher_distillation_temperature))
        or graph_matcher_teacher_distillation_temperature <= 0.0
    ):
        raise ValueError("graph_matcher_teacher_distillation_temperature must be positive and finite")
    if not math.isfinite(float(graph_matcher_positive_dustbin_guard_reject_threshold)):
        raise ValueError("graph_matcher_positive_dustbin_guard_reject_threshold must be finite")
    if graph_matcher_dustbin_warmup_steps < 0 or graph_matcher_dustbin_ramp_steps < 0:
        raise ValueError("graph matcher dustbin schedule steps must be nonnegative")
    if graph_matcher_train_candidate_topk < 0:
        raise ValueError("graph_matcher_train_candidate_topk must be nonnegative")
    effective_graph_matcher_no_match_weight = scheduled_graph_matcher_weight(
        graph_matcher_no_match_weight,
        step=training_step,
        warmup_steps=graph_matcher_dustbin_warmup_steps,
        ramp_steps=graph_matcher_dustbin_ramp_steps,
    )
    effective_graph_matcher_hard_negative_dustbin_weight = scheduled_graph_matcher_weight(
        graph_matcher_hard_negative_dustbin_weight,
        step=training_step,
        warmup_steps=graph_matcher_dustbin_warmup_steps,
        ramp_steps=graph_matcher_dustbin_ramp_steps,
    )
    if reliability_negative_points < 0:
        raise ValueError("reliability_negative_points must be nonnegative")
    if reliability_negative_min_distance < 0.0:
        raise ValueError("reliability_negative_min_distance must be nonnegative")
    rotation_consistency_degrees = rotation_consistency_degrees or [90, 180, 270]
    for degree in rotation_consistency_degrees:
        _rotation_k(degree)
    optimizer.zero_grad(set_to_none=True)
    metric_rows: list[dict[str, float]] = []
    graph_metric_rows: list[dict[str, float]] = []
    sampled_count = 0
    pseudo_label_points = 0
    pseudo_keypoint_points = 0
    pseudo_label_pairs = 0
    keypoint_loss_sum = 0.0
    keypoint_loss_count = 0
    keypoint_points = 0
    keypoint_offset_loss_sum = 0.0
    keypoint_offset_loss_count = 0
    keypoint_offset_points = 0
    selected_keypoint_offset_loss_sum = 0.0
    selected_keypoint_offset_loss_count = 0
    selected_keypoint_offset_points = 0.0
    selected_keypoint_offset_forward_points = 0.0
    selected_keypoint_offset_reverse_points = 0.0
    false_match_points = 0
    false_match_pairs = 0
    online_false_match_points = 0
    online_false_match_pairs = 0
    illumination_consistency_points = 0
    illumination_consistency_pairs_used = 0
    illumination_match_points = 0
    illumination_match_pairs_used = 0
    matchability_loss_sum = 0.0
    descriptor_uncertainty_loss_sum = 0.0
    no_match_prior_loss_sum = 0.0
    reliability_loss_count = 0
    reliability_points = 0
    rotation_descriptor_loss_sum = 0.0
    orientation_loss_sum = 0.0
    scale_loss_sum = 0.0
    affine_loss_sum = 0.0
    affine_regularization_loss_sum = 0.0
    affine_det_mean_sum = 0.0
    affine_det_std_sum = 0.0
    affine_condition_mean_sum = 0.0
    affine_condition_max_sum = 0.0
    affine_regularization_count = 0
    rotation_loss_count = 0
    rotation_consistency_points = 0
    rotation_consistency_pairs_used = 0
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
    reliability_loss_enabled = (
        matchability_weight > 0.0 or descriptor_uncertainty_weight > 0.0 or no_match_prior_weight > 0.0
    )
    rotation_loss_enabled = (
        rotation_descriptor_consistency_weight > 0.0
        or orientation_consistency_weight > 0.0
        or scale_consistency_weight > 0.0
        or affine_consistency_weight > 0.0
    )
    affine_regularization_enabled = affine_regularization_weight > 0.0
    sparse_maps_required = reliability_loss_enabled or rotation_loss_enabled or affine_regularization_enabled
    keypoint_loss_enabled = keypoint_weight > 0.0
    keypoint_offset_loss_enabled = keypoint_offset_weight > 0.0
    selected_keypoint_offset_loss_enabled = selected_keypoint_offset_weight > 0.0
    sparse_maps_required = sparse_maps_required or keypoint_offset_loss_enabled or selected_keypoint_offset_loss_enabled
    use_amp = bool(amp_enabled)
    use_grad_scaler = use_amp and grad_scaler is not None
    use_activation_checkpointing = bool(activation_checkpointing)
    grad_scaler_unscaled = False

    def auxiliary_loss_metrics() -> dict[str, float]:
        reliability_denominator = float(max(1, reliability_loss_count))
        rotation_denominator = float(max(1, rotation_loss_count))
        return {
            "matchability_loss": matchability_loss_sum / reliability_denominator,
            "descriptor_uncertainty_loss": descriptor_uncertainty_loss_sum / reliability_denominator,
            "no_match_prior_loss": no_match_prior_loss_sum / reliability_denominator,
            "reliability_points": float(reliability_points),
            "rotation_descriptor_consistency_loss": rotation_descriptor_loss_sum / rotation_denominator,
            "orientation_consistency_loss": orientation_loss_sum / rotation_denominator,
            "scale_consistency_loss": scale_loss_sum / rotation_denominator,
            "affine_consistency_loss": affine_loss_sum / rotation_denominator,
            "affine_regularization_loss": affine_regularization_loss_sum / float(max(1, affine_regularization_count)),
            "affine_det_mean": affine_det_mean_sum / float(max(1, affine_regularization_count)),
            "affine_det_std": affine_det_std_sum / float(max(1, affine_regularization_count)),
            "affine_condition_mean": affine_condition_mean_sum / float(max(1, affine_regularization_count)),
            "affine_condition_max": affine_condition_max_sum / float(max(1, affine_regularization_count)),
            "rotation_consistency_points": float(rotation_consistency_points),
            "rotation_consistency_pairs": float(rotation_consistency_pairs_used),
            "keypoint_loss": keypoint_loss_sum / float(max(1, keypoint_loss_count)),
            "keypoint_points": float(keypoint_points),
            "keypoint_offset_loss": keypoint_offset_loss_sum / float(max(1, keypoint_offset_loss_count)),
            "keypoint_offset_points": float(keypoint_offset_points),
            "selected_keypoint_offset_loss": selected_keypoint_offset_loss_sum
            / float(max(1, selected_keypoint_offset_loss_count)),
            "selected_keypoint_offset_points": float(selected_keypoint_offset_points),
            "selected_keypoint_offset_forward_points": float(selected_keypoint_offset_forward_points),
            "selected_keypoint_offset_reverse_points": float(selected_keypoint_offset_reverse_points),
            "amp_enabled": 1.0 if use_amp else 0.0,
            "amp_scale": grad_scaler_scale(grad_scaler) if use_grad_scaler else 0.0,
            "activation_checkpointing": 1.0 if use_activation_checkpointing else 0.0,
            "freeze_extractor_warmup_active": 1.0 if freeze_extractor_warmup_active else 0.0,
        }

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
                pair_acceptance = None if pair_acceptance_targets is None else pair_acceptance_targets.get(pair_key)
                true_geometry_match_count = (
                    None
                    if true_geometry_match_count_targets is None
                    else true_geometry_match_count_targets.get(pair_key)
                )
                if prefetched_pairs is not None and pair_key in prefetched_pairs:
                    pair = move_pair_to_device(prefetched_pairs[pair_key], device=device)
                else:
                    pair = load_pair_for_training(pair_path, device=device, pair_cache=pair_cache)
                pair = crop_pair_for_training(pair, crop_size=training_crop_size, generator=generator)
                pair = resize_pair_for_training(pair, max_image_size=training_max_image_size)
                sparse_maps_a: TrainingSparseMaps | None = None
                sparse_maps_b: TrainingSparseMaps | None = None
                if sparse_maps_required:
                    with autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                        sparse_maps_a = learned_training_sparse_maps_single(
                            model,
                            pair.view_a.unsqueeze(0),
                            train_blended_descriptors=train_blended_descriptors,
                            texture_blend_weight=texture_blend_weight,
                            activation_checkpointing=use_activation_checkpointing,
                        )
                        sparse_maps_b = learned_training_sparse_maps_single(
                            model,
                            pair.view_b.unsqueeze(0),
                            train_blended_descriptors=train_blended_descriptors,
                            texture_blend_weight=texture_blend_weight,
                            activation_checkpointing=use_activation_checkpointing,
                        )
                    descriptors_a = sparse_maps_a.descriptors
                    descriptors_b = sparse_maps_b.descriptors
                    heatmap_a = sparse_maps_a.heatmap
                    heatmap_b = sparse_maps_b.heatmap
                    with torch.no_grad(), autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                        teacher_a = model.texture_descriptor_map_single(pair.view_a.unsqueeze(0))
                        teacher_b = model.texture_descriptor_map_single(pair.view_b.unsqueeze(0))
                else:
                    with autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                        descriptor_maps = compute_student_teacher_descriptor_maps(
                            model,
                            pair,
                            train_blended_descriptors=train_blended_descriptors,
                            texture_blend_weight=texture_blend_weight,
                            include_heatmaps=True,
                            activation_checkpointing=use_activation_checkpointing,
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
                online_false_a = descriptors_a.new_empty((0, 2))
                online_false_b = descriptors_b.new_empty((0, 2))
                static_false_a = descriptors_a.new_empty((0, 2))
                static_false_b = descriptors_b.new_empty((0, 2))
                graph_online_false_can_train = (
                    graph_matcher_online_false_no_match
                    and graph_matcher_loss_weight > 0.0
                    and (
                        effective_graph_matcher_no_match_weight > 0.0
                        or graph_matcher_assignment_weight > 0.0
                        or graph_matcher_accept_weight > 0.0
                        or graph_matcher_prune_ranking_weight > 0.0
                    )
                )
                needs_online_false_mining = online_false_match_weight > 0.0 or graph_online_false_can_train
                if needs_online_false_mining:
                    online_false_a, online_false_b = online_false_match_feature_correspondences(
                        pair,
                        descriptors_a,
                        descriptors_b,
                        max_keypoints=online_false_match_max_keypoints,
                        max_matches=online_false_match_max_matches,
                        min_intensity=min_intensity,
                        min_score=online_false_match_min_score,
                        min_margin=online_false_match_min_margin,
                        threshold_px=online_false_match_threshold_px,
                        max_points=online_false_match_max_points,
                        generator=generator,
                        keypoint_scores_a=heatmap_a,
                        keypoint_scores_b=heatmap_b,
                    )
                    if online_false_a.size(0) > 0:
                        online_false_match_points += online_false_a.size(0)
                        online_false_match_pairs += 1
                        false_match_points += online_false_a.size(0)
                        false_match_pairs += 1
                needs_static_false_matches = bool(false_matches) and (
                    false_match_weight > 0.0
                    or (
                        graph_matcher_loss_weight > 0.0
                        and (
                            graph_matcher_final_false_match_weight > 0.0
                            or graph_matcher_mined_false_match_weight > 0.0
                        )
                    )
                )
                if needs_static_false_matches:
                    static_false_a, static_false_b = false_match_feature_correspondences(
                        pair_path,
                        pair,
                        false_matches,
                        feature_height=descriptors_a.size(2),
                        feature_width=descriptors_a.size(3),
                        max_points=false_match_max_points,
                        generator=generator,
                    )
                pair_losses: list[torch.Tensor] = []
                if points_a.size(0) > 0 and (
                    synthetic_loss_weight > 0.0
                    or graph_matcher_loss_weight > 0.0
                    or reliability_loss_enabled
                    or rotation_loss_enabled
                    or keypoint_loss_enabled
                    or keypoint_offset_loss_enabled
                    or selected_keypoint_offset_loss_enabled
                ):
                    pose_multiplier = pose_difficulty_loss_multiplier(
                        pose_metadata,
                        pair_path,
                        strength=pose_difficulty_loss_weight,
                    )
                    if reliability_loss_enabled and sparse_maps_a is not None and sparse_maps_b is not None:
                        negative_count = int(reliability_negative_points)
                        if negative_count <= 0:
                            negative_count = max(
                                int(graph_matcher_no_match_points),
                                min(int(samples_per_pair), int(points_a.size(0))),
                            )
                        neg_a_points = sample_unmatched_feature_points(
                            feature_height=descriptors_a.size(2),
                            feature_width=descriptors_a.size(3),
                            reference_points=points_a,
                            count=negative_count,
                            min_distance=reliability_negative_min_distance,
                            generator=generator,
                            device=descriptors_a.device,
                        )
                        neg_b_points = sample_unmatched_feature_points(
                            feature_height=descriptors_b.size(2),
                            feature_width=descriptors_b.size(3),
                            reference_points=points_b,
                            count=negative_count,
                            min_distance=reliability_negative_min_distance,
                            generator=generator,
                            device=descriptors_b.device,
                        )
                        if matchability_weight > 0.0:
                            matchability_loss = 0.5 * (
                                matchability_supervision_loss(sparse_maps_a.matchability, points_a, neg_a_points)
                                + matchability_supervision_loss(sparse_maps_b.matchability, points_b, neg_b_points)
                            )
                            pair_losses.append(float(matchability_weight) * matchability_loss)
                            matchability_loss_sum += float(matchability_loss.detach().cpu())
                        if descriptor_uncertainty_weight > 0.0:
                            uncertainty_loss = 0.5 * (
                                descriptor_uncertainty_supervision_loss(
                                    sparse_maps_a.descriptor_uncertainty,
                                    neg_a_points,
                                    points_a,
                                )
                                + descriptor_uncertainty_supervision_loss(
                                    sparse_maps_b.descriptor_uncertainty,
                                    neg_b_points,
                                    points_b,
                                )
                            )
                            pair_losses.append(float(descriptor_uncertainty_weight) * uncertainty_loss)
                            descriptor_uncertainty_loss_sum += float(uncertainty_loss.detach().cpu())
                        if no_match_prior_weight > 0.0:
                            no_match_loss = 0.5 * (
                                no_match_prior_supervision_loss(
                                    sparse_maps_a.no_match_prior,
                                    neg_a_points,
                                    points_a,
                                )
                                + no_match_prior_supervision_loss(
                                    sparse_maps_b.no_match_prior,
                                    neg_b_points,
                                    points_b,
                                )
                            )
                            pair_losses.append(float(no_match_prior_weight) * no_match_loss)
                            no_match_prior_loss_sum += float(no_match_loss.detach().cpu())
                        reliability_loss_count += 1
                        reliability_points += int(
                            points_a.size(0) + points_b.size(0) + neg_a_points.size(0) + neg_b_points.size(0)
                        )
                    if affine_regularization_enabled and sparse_maps_a is not None and sparse_maps_b is not None:
                        affine_reg_a, affine_metrics_a = affine_regularization_loss(
                            sparse_maps_a.affine,
                            return_metrics=True,
                        )
                        affine_reg_b, affine_metrics_b = affine_regularization_loss(
                            sparse_maps_b.affine,
                            return_metrics=True,
                        )
                        affine_reg_loss = 0.5 * (affine_reg_a + affine_reg_b)
                        pair_losses.append(float(affine_regularization_weight) * affine_reg_loss)
                        affine_regularization_loss_sum += float(affine_reg_loss.detach().cpu())
                        for metric_name, metric_value_a in affine_metrics_a.items():
                            metric_value = 0.5 * (metric_value_a + affine_metrics_b[metric_name])
                            if metric_name == "affine_det_mean":
                                affine_det_mean_sum += float(metric_value.detach().cpu())
                            elif metric_name == "affine_det_std":
                                affine_det_std_sum += float(metric_value.detach().cpu())
                            elif metric_name == "affine_condition_mean":
                                affine_condition_mean_sum += float(metric_value.detach().cpu())
                            elif metric_name == "affine_condition_max":
                                affine_condition_max_sum += float(metric_value.detach().cpu())
                        affine_regularization_count += 1
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
                    if keypoint_loss_enabled and heatmap_a is not None and heatmap_b is not None:
                        keypoint_loss = heatmap_point_loss(
                            heatmap_a,
                            points_a,
                            negative_weight=keypoint_negative_weight,
                        ) + heatmap_point_loss(
                            heatmap_b,
                            points_b,
                            negative_weight=keypoint_negative_weight,
                        )
                        pair_losses.append(float(keypoint_weight) * keypoint_loss)
                        keypoint_loss_sum += float(keypoint_loss.detach().cpu())
                        keypoint_loss_count += 1
                        keypoint_points += int(points_a.size(0) + points_b.size(0))
                    if keypoint_offset_loss_enabled and sparse_maps_a is not None and sparse_maps_b is not None:
                        keypoint_offset_loss = 0.5 * (
                            keypoint_offset_supervision_loss(sparse_maps_a.keypoint_offsets, points_a)
                            + keypoint_offset_supervision_loss(sparse_maps_b.keypoint_offsets, points_b)
                        )
                        pair_losses.append(float(keypoint_offset_weight) * keypoint_offset_loss)
                        keypoint_offset_loss_sum += float(keypoint_offset_loss.detach().cpu())
                        keypoint_offset_loss_count += 1
                        keypoint_offset_points += int(points_a.size(0) + points_b.size(0))
                    if (
                        selected_keypoint_offset_loss_enabled
                        and sparse_maps_a is not None
                        and sparse_maps_b is not None
                    ):
                        selected_offset_loss, selected_offset_metrics = selected_keypoint_offset_supervision_loss(
                            pair,
                            sparse_maps_a,
                            sparse_maps_b,
                            max_points=selected_keypoint_offset_max_points,
                            min_intensity=min_intensity,
                            inverse_radius_px=selected_keypoint_offset_inverse_radius_px,
                        )
                        pair_losses.append(float(selected_keypoint_offset_weight) * selected_offset_loss)
                        selected_keypoint_offset_loss_sum += float(selected_offset_loss.detach().cpu())
                        selected_keypoint_offset_loss_count += 1
                        selected_keypoint_offset_points += float(selected_offset_metrics["points"].detach().cpu())
                        selected_keypoint_offset_forward_points += float(
                            selected_offset_metrics["forward_points"].detach().cpu()
                        )
                        selected_keypoint_offset_reverse_points += float(
                            selected_offset_metrics["reverse_points"].detach().cpu()
                        )
                    if graph_matcher_loss_weight > 0.0:
                        graph_online_false_enabled = graph_online_false_can_train and online_false_a.size(0) > 0
                        true_geometry_floor_target_count = None
                        true_geometry_floor_weight = 0.0
                        if true_geometry_match_count is not None:
                            true_geometry_floor_target_count, true_geometry_floor_pair_weight = true_geometry_match_count
                            true_geometry_floor_weight = (
                                float(graph_matcher_true_geometry_match_count_floor_weight)
                                * float(true_geometry_floor_pair_weight)
                            )
                        with autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                            graph_loss, graph_components = graph_matcher_correspondence_loss(
                                model,
                                descriptors_a,
                                descriptors_b,
                                points_a,
                                points_b,
                                metadata_mode=graph_matcher_metadata_mode,
                                no_match_points=graph_matcher_no_match_points,
                                no_match_weight=effective_graph_matcher_no_match_weight,
                                no_match_min_distance=graph_matcher_no_match_min_distance,
                                assignment_weight=graph_matcher_assignment_weight,
                                accept_weight=graph_matcher_accept_weight,
                                accept_negative_topk=graph_matcher_accept_negative_topk,
                                prune_ranking_weight=graph_matcher_prune_ranking_weight,
                                prune_ranking_margin=graph_matcher_prune_ranking_margin,
                                stop_confidence_weight=graph_matcher_stop_confidence_weight,
                                stop_confidence_margin=graph_matcher_stop_confidence_margin,
                                raw_preservation_weight=graph_matcher_raw_preservation_weight,
                                raw_preservation_margin=graph_matcher_raw_preservation_margin,
                                raw_preservation_raw_margin=graph_matcher_raw_preservation_raw_margin,
                                hard_negative_dustbin_weight=effective_graph_matcher_hard_negative_dustbin_weight,
                                hard_negative_dustbin_topk=graph_matcher_hard_negative_dustbin_topk,
                                hard_negative_dustbin_margin=graph_matcher_hard_negative_dustbin_margin,
                                hard_negative_dustbin_spatial_min_distance=(
                                    graph_matcher_hard_negative_dustbin_spatial_min_distance
                                ),
                                positive_dustbin_margin_weight=graph_matcher_positive_dustbin_margin_weight,
                                positive_dustbin_margin=graph_matcher_positive_dustbin_margin,
                                true_match_margin_weight=graph_matcher_true_match_margin_weight,
                                true_match_margin=graph_matcher_true_match_margin,
                                true_geometry_match_count_floor_weight=true_geometry_floor_weight,
                                true_geometry_match_count_floor_target_count=true_geometry_floor_target_count,
                                true_geometry_match_count_floor_threshold=(
                                    graph_matcher_true_geometry_match_count_floor_threshold
                                ),
                                true_geometry_match_count_floor_margin=(
                                    graph_matcher_true_geometry_match_count_floor_margin
                                ),
                                final_false_match_weight=graph_matcher_final_false_match_weight,
                                mined_false_match_weight=graph_matcher_mined_false_match_weight,
                                mined_false_match_loss_cap=graph_matcher_mined_false_match_loss_cap,
                                mined_false_match_reference_margin=(
                                    graph_matcher_mined_false_match_reference_margin
                                ),
                                final_false_match_topk=graph_matcher_final_false_match_topk,
                                final_false_match_min_score=graph_matcher_final_false_match_min_score,
                                final_false_match_margin=graph_matcher_final_false_match_margin,
                                final_false_match_spatial_min_distance=(
                                    graph_matcher_final_false_match_spatial_min_distance
                                ),
                                raw_false_match_weight=graph_matcher_raw_false_match_weight,
                                raw_false_match_topk=graph_matcher_raw_false_match_topk,
                                raw_false_match_min_similarity=graph_matcher_raw_false_match_min_similarity,
                                raw_false_match_margin=graph_matcher_raw_false_match_margin,
                                raw_false_match_spatial_min_distance=graph_matcher_raw_false_match_spatial_min_distance,
                                ransac_consistency_weight=graph_matcher_ransac_consistency_weight,
                                ransac_consistency_topk=graph_matcher_ransac_consistency_topk,
                                ransac_consistency_residual_threshold_px=(
                                    graph_matcher_ransac_consistency_residual_threshold_px
                                ),
                                ransac_consistency_min_score=graph_matcher_ransac_consistency_min_score,
                                ransac_consistency_margin=graph_matcher_ransac_consistency_margin,
                                warp_outlier_weight=graph_matcher_warp_outlier_weight,
                                warp_outlier_topk=graph_matcher_warp_outlier_topk,
                                warp_outlier_residual_threshold_px=(
                                    graph_matcher_warp_outlier_residual_threshold_px
                                ),
                                warp_outlier_min_score=graph_matcher_warp_outlier_min_score,
                                warp_outlier_margin=graph_matcher_warp_outlier_margin,
                                warp_outlier_accept_weight=graph_matcher_warp_outlier_accept_weight,
                                warp_outlier_accept_topk=graph_matcher_warp_outlier_accept_topk,
                                warp_outlier_accept_residual_threshold_px=(
                                    graph_matcher_warp_outlier_accept_residual_threshold_px
                                ),
                                warp_outlier_accept_min_score=graph_matcher_warp_outlier_accept_min_score,
                                warp_soft_boundary_weight=graph_matcher_warp_soft_boundary_weight,
                                warp_soft_boundary_topk=graph_matcher_warp_soft_boundary_topk,
                                warp_soft_boundary_lower_residual_px=(
                                    graph_matcher_warp_soft_boundary_lower_residual_px
                                ),
                                warp_soft_boundary_upper_residual_px=(
                                    graph_matcher_warp_soft_boundary_upper_residual_px
                                ),
                                warp_soft_boundary_min_score=graph_matcher_warp_soft_boundary_min_score,
                                pair_acceptance_target=None if pair_acceptance is None else pair_acceptance[0],
                                pair_acceptance_weight=1.0 if pair_acceptance is None else pair_acceptance[1],
                                pair_acceptance_loss_weight=graph_matcher_pair_acceptance_loss_weight,
                                train_candidate_topk=graph_matcher_train_candidate_topk,
                                semi_dense_no_match_points=graph_matcher_semi_dense_no_match_points,
                                semi_dense_min_score=graph_matcher_semi_dense_min_score,
                                extra_no_match_points_a_xy=online_false_a if graph_online_false_enabled else None,
                                extra_no_match_points_b_xy=online_false_b if graph_online_false_enabled else None,
                                extra_false_match_points_a_xy=(
                                    static_false_a if static_false_a.size(0) > 0 else None
                                ),
                                extra_false_match_points_b_xy=(
                                    static_false_b if static_false_b.size(0) > 0 else None
                                ),
                                max_attention_layers=graph_matcher_train_max_attention_layers,
                                random_attention_layers=graph_matcher_train_random_attention_layers,
                                max_attention_work_fraction=graph_matcher_train_max_attention_work_fraction,
                                width_keep_ratio=graph_matcher_train_width_keep_ratio,
                                deep_supervision_depths=graph_matcher_deep_supervision_depths,
                                deep_supervision_weight=graph_matcher_deep_supervision_weight,
                                depth_distillation_weight=graph_matcher_depth_distillation_weight,
                                depth_distillation_teacher_layers=graph_matcher_depth_distillation_teacher_layers,
                                depth_distillation_temperature=graph_matcher_depth_distillation_temperature,
                                teacher_guard_model=graph_matcher_teacher_guard_model,
                                teacher_guard_weight=graph_matcher_teacher_guard_weight,
                                teacher_guard_positive_margin_tolerance=(
                                    graph_matcher_teacher_guard_positive_margin_tolerance
                                ),
                                teacher_guard_false_margin_tolerance=(
                                    graph_matcher_teacher_guard_false_margin_tolerance
                                ),
                                teacher_score_floor_weight=graph_matcher_teacher_score_floor_weight,
                                teacher_score_floor_tolerance=graph_matcher_teacher_score_floor_tolerance,
                                teacher_score_floor_min_score=graph_matcher_teacher_score_floor_min_score,
                                teacher_match_count_floor_weight=graph_matcher_teacher_match_count_floor_weight,
                                teacher_match_count_floor_threshold=(
                                    graph_matcher_teacher_match_count_floor_threshold
                                ),
                                teacher_match_count_floor_margin=graph_matcher_teacher_match_count_floor_margin,
                                teacher_match_count_ceiling_weight=(
                                    graph_matcher_teacher_match_count_ceiling_weight
                                ),
                                teacher_match_count_ceiling_threshold=(
                                    graph_matcher_teacher_match_count_ceiling_threshold
                                ),
                                teacher_match_count_ceiling_margin=(
                                    graph_matcher_teacher_match_count_ceiling_margin
                                ),
                                teacher_distillation_weight=graph_matcher_teacher_distillation_weight,
                                teacher_distillation_temperature=(
                                    graph_matcher_teacher_distillation_temperature
                                ),
                                positive_dustbin_guard_reject_threshold=(
                                    graph_matcher_positive_dustbin_guard_reject_threshold
                                ),
                                positive_dustbin_guard_margin_threshold=(
                                    graph_matcher_positive_dustbin_guard_margin_threshold
                                ),
                                matchability_a=sparse_maps_a.matchability if sparse_maps_a is not None else None,
                                matchability_b=sparse_maps_b.matchability if sparse_maps_b is not None else None,
                                descriptor_uncertainty_a=(
                                    sparse_maps_a.descriptor_uncertainty if sparse_maps_a is not None else None
                                ),
                                descriptor_uncertainty_b=(
                                    sparse_maps_b.descriptor_uncertainty if sparse_maps_b is not None else None
                                ),
                                no_match_prior_a=sparse_maps_a.no_match_prior if sparse_maps_a is not None else None,
                                no_match_prior_b=sparse_maps_b.no_match_prior if sparse_maps_b is not None else None,
                                generator=generator,
                                return_components=True,
                            )
                        graph_metric_rows.append(
                            {
                                key: float(value.detach().cpu())
                                if bool(torch.isfinite(value.detach()).all())
                                else float("nan")
                                for key, value in graph_components.items()
                            }
                            | {
                                "points": float(points_a.size(0)),
                                "graph_matcher_effective_no_match_weight": float(
                                    effective_graph_matcher_no_match_weight
                                ),
                                "graph_matcher_effective_hard_negative_dustbin_weight": float(
                                    effective_graph_matcher_hard_negative_dustbin_weight
                                ),
                            }
                        )
                        pair_losses.append(float(graph_matcher_loss_weight) * float(pose_multiplier) * graph_loss)
                    if rotation_loss_enabled and sparse_maps_a is not None and sparse_maps_b is not None:
                        degree_index = int(
                            torch.randint(
                                0,
                                len(rotation_consistency_degrees),
                                (1,),
                                device=points_a.device,
                                generator=generator,
                            ).item()
                        )
                        rotation_degrees = int(rotation_consistency_degrees[degree_index])
                        k = _rotation_k(rotation_degrees)
                        with autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                            rotated_sparse_a = learned_training_sparse_maps_single(
                                model,
                                torch.rot90(pair.view_a.unsqueeze(0), k=k, dims=(-2, -1)),
                                train_blended_descriptors=train_blended_descriptors,
                                texture_blend_weight=texture_blend_weight,
                                activation_checkpointing=use_activation_checkpointing,
                            )
                            rotated_sparse_b = learned_training_sparse_maps_single(
                                model,
                                torch.rot90(pair.view_b.unsqueeze(0), k=k, dims=(-2, -1)),
                                train_blended_descriptors=train_blended_descriptors,
                                texture_blend_weight=texture_blend_weight,
                                activation_checkpointing=use_activation_checkpointing,
                            )
                        if rotation_descriptor_consistency_weight > 0.0:
                            descriptor_rotation_loss = 0.5 * (
                                rotation_descriptor_consistency_loss(
                                    sparse_maps_a.descriptors,
                                    rotated_sparse_a.descriptors,
                                    points_a,
                                    rotation_degrees,
                                    generator=generator,
                                )
                                + rotation_descriptor_consistency_loss(
                                    sparse_maps_b.descriptors,
                                    rotated_sparse_b.descriptors,
                                    points_b,
                                    rotation_degrees,
                                    generator=generator,
                                )
                            )
                            pair_losses.append(
                                float(rotation_descriptor_consistency_weight) * descriptor_rotation_loss
                            )
                            rotation_descriptor_loss_sum += float(descriptor_rotation_loss.detach().cpu())
                        if orientation_consistency_weight > 0.0:
                            orientation_loss = 0.5 * (
                                orientation_consistency_loss(
                                    sparse_maps_a.orientation,
                                    rotated_sparse_a.orientation,
                                    points_a,
                                    rotation_degrees,
                                    generator=generator,
                                )
                                + orientation_consistency_loss(
                                    sparse_maps_b.orientation,
                                    rotated_sparse_b.orientation,
                                    points_b,
                                    rotation_degrees,
                                    generator=generator,
                                )
                            )
                            pair_losses.append(float(orientation_consistency_weight) * orientation_loss)
                            orientation_loss_sum += float(orientation_loss.detach().cpu())
                        if scale_consistency_weight > 0.0:
                            scale_loss = 0.5 * (
                                scale_consistency_loss(
                                    sparse_maps_a.scale,
                                    rotated_sparse_a.scale,
                                    points_a,
                                    rotation_degrees,
                                    generator=generator,
                                )
                                + scale_consistency_loss(
                                    sparse_maps_b.scale,
                                    rotated_sparse_b.scale,
                                    points_b,
                                    rotation_degrees,
                                    generator=generator,
                                )
                            )
                            pair_losses.append(float(scale_consistency_weight) * scale_loss)
                            scale_loss_sum += float(scale_loss.detach().cpu())
                        if affine_consistency_weight > 0.0:
                            affine_loss = 0.5 * (
                                affine_consistency_loss(
                                    sparse_maps_a.affine,
                                    rotated_sparse_a.affine,
                                    points_a,
                                    rotation_degrees,
                                    generator=generator,
                                )
                                + affine_consistency_loss(
                                    sparse_maps_b.affine,
                                    rotated_sparse_b.affine,
                                    points_b,
                                    rotation_degrees,
                                    generator=generator,
                                )
                            )
                            pair_losses.append(float(affine_consistency_weight) * affine_loss)
                            affine_loss_sum += float(affine_loss.detach().cpu())
                        rotation_loss_count += 1
                        rotation_consistency_points += int(points_a.size(0) + points_b.size(0))
                        rotation_consistency_pairs_used += 1
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
                    if static_false_a.size(0) > 0:
                        negative_loss = false_match_negative_loss(
                            descriptors_a,
                            descriptors_b,
                            static_false_a,
                            static_false_b,
                            max_false_score=false_match_max_score,
                        )
                        pair_losses.append(float(false_match_weight) * negative_loss)
                        false_match_points += static_false_a.size(0)
                        false_match_pairs += 1
                if online_false_match_weight > 0.0 and online_false_a.size(0) > 0:
                    online_negative_loss = false_match_negative_loss(
                        descriptors_a,
                        descriptors_b,
                        online_false_a,
                        online_false_b,
                        max_false_score=online_false_match_max_score,
                    )
                    pair_losses.append(float(online_false_match_weight) * online_negative_loss)
                if (
                    illumination_consistency_pairs
                    and illumination_consistency_weight > 0.0
                    and points_a.size(0) > 0
                    and random.random() <= max(0.0, min(1.0, float(illumination_consistency_probability)))
                ):
                    changed_pair = illumination_consistency_pairs.get(pair_key)
                    if changed_pair is not None:
                        changed_pair = move_pair_to_device(changed_pair, device=device)
                        with autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                            changed_descriptors_a = compute_training_descriptor_map(
                                model,
                                changed_pair.view_a.unsqueeze(0),
                                train_blended_descriptors=train_blended_descriptors,
                                texture_blend_weight=texture_blend_weight,
                                activation_checkpointing=use_activation_checkpointing,
                            )
                            changed_descriptors_b = compute_training_descriptor_map(
                                model,
                                changed_pair.view_b.unsqueeze(0),
                                train_blended_descriptors=train_blended_descriptors,
                                texture_blend_weight=texture_blend_weight,
                                activation_checkpointing=use_activation_checkpointing,
                            )
                        consistency_loss_a = descriptor_consistency_loss(
                            descriptors_a,
                            changed_descriptors_a,
                            points_a,
                            max_points=illumination_consistency_max_points,
                            generator=generator,
                        )
                        consistency_loss_b = descriptor_consistency_loss(
                            descriptors_b,
                            changed_descriptors_b,
                            points_b,
                            max_points=illumination_consistency_max_points,
                            generator=generator,
                        )
                        consistency_loss = torch.stack([consistency_loss_a, consistency_loss_b]).mean()
                        pair_losses.append(float(illumination_consistency_weight) * consistency_loss)
                        per_view_points = (
                            min(points_a.size(0), int(illumination_consistency_max_points))
                            if illumination_consistency_max_points > 0
                            else points_a.size(0)
                        )
                        illumination_consistency_points += int(per_view_points) * 2
                        illumination_consistency_pairs_used += 1
                if (
                    illumination_match_pairs
                    and illumination_match_weight > 0.0
                    and points_a.size(0) > 0
                    and random.random() <= max(0.0, min(1.0, float(illumination_match_probability)))
                ):
                    changed_match_pair = illumination_match_pairs.get(pair_key)
                    if changed_match_pair is not None:
                        changed_match_pair = move_pair_to_device(changed_match_pair, device=device)
                        # 单侧光照扰动时复用未变化视图，避免重复 full forward 占满显存。
                        view_a_unchanged = changed_match_pair.view_a.shape == pair.view_a.shape and torch.equal(
                            changed_match_pair.view_a,
                            pair.view_a,
                        )
                        view_b_unchanged = changed_match_pair.view_b.shape == pair.view_b.shape and torch.equal(
                            changed_match_pair.view_b,
                            pair.view_b,
                        )
                        if view_a_unchanged and view_b_unchanged:
                            changed_descriptors_a, changed_descriptors_b = descriptors_a, descriptors_b
                            changed_teacher_a, changed_teacher_b = teacher_a, teacher_b
                        elif view_a_unchanged:
                            changed_descriptors_a, changed_teacher_a = descriptors_a, teacher_a
                            with autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                                changed_descriptors_b, changed_teacher_b = compute_student_teacher_descriptor_map_single(
                                    model,
                                    changed_match_pair.view_b.unsqueeze(0),
                                    train_blended_descriptors=train_blended_descriptors,
                                    texture_blend_weight=texture_blend_weight,
                                    activation_checkpointing=use_activation_checkpointing,
                                )
                        elif view_b_unchanged:
                            changed_descriptors_b, changed_teacher_b = descriptors_b, teacher_b
                            with autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                                changed_descriptors_a, changed_teacher_a = compute_student_teacher_descriptor_map_single(
                                    model,
                                    changed_match_pair.view_a.unsqueeze(0),
                                    train_blended_descriptors=train_blended_descriptors,
                                    texture_blend_weight=texture_blend_weight,
                                    activation_checkpointing=use_activation_checkpointing,
                                )
                        else:
                            with autocast_context(device, enabled=use_amp, dtype=amp_dtype):
                                changed_maps = compute_student_teacher_descriptor_maps(
                                    model,
                                    changed_match_pair,
                                    train_blended_descriptors=train_blended_descriptors,
                                    texture_blend_weight=texture_blend_weight,
                                    include_heatmaps=False,
                                    activation_checkpointing=use_activation_checkpointing,
                                )
                            changed_descriptors_a, changed_descriptors_b, changed_teacher_a, changed_teacher_b = (
                                changed_maps[:4]
                            )
                        match_loss, match_metrics = descriptor_map_pair_loss(
                            changed_descriptors_a,
                            changed_descriptors_b,
                            points_a,
                            points_b,
                            temperature=temperature,
                            teacher_descriptors_a=changed_teacher_a,
                            teacher_descriptors_b=changed_teacher_b,
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
                        pair_losses.append(float(illumination_match_weight) * match_loss)
                        metric_rows.append(match_metrics)
                        sampled_count += points_a.size(0)
                        illumination_match_points += points_a.size(0)
                        illumination_match_pairs_used += 1
                if pair_losses:
                    losses.append(torch.stack(pair_losses).sum())
            if not losses:
                continue
            micro_loss = torch.stack(losses).mean()
            loss_values.append(float(micro_loss.detach().cpu()) if bool(torch.isfinite(micro_loss.detach()).all()) else float("nan"))
            require_finite_scalar(micro_loss, name="training loss")
            scaled_micro_loss = micro_loss / float(gradient_accumulation_steps)
            if use_grad_scaler:
                grad_scaler.scale(scaled_micro_loss).backward()
            else:
                scaled_micro_loss.backward()
            valid_micro_batches += 1
        if valid_micro_batches == 0:
            raise RuntimeError("no valid correspondences sampled")
        if use_grad_scaler:
            grad_scaler.unscale_(optimizer)
            grad_scaler_unscaled = True
        grad_norm = clip_and_measure_gradients(parameters, max_grad_norm=max_grad_norm)
    except FloatingPointError:
        optimizer.zero_grad(set_to_none=True)
        if use_grad_scaler and grad_scaler_unscaled:
            grad_scaler.update()
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
        metrics.update(aggregate_graph_matcher_loss_metrics(graph_metric_rows))
        metrics.update(auxiliary_loss_metrics())
        metrics["pseudo_label_points"] = float(pseudo_label_points)
        metrics["pseudo_keypoint_points"] = float(pseudo_keypoint_points)
        metrics["pseudo_label_pairs"] = float(pseudo_label_pairs)
        metrics["false_match_points"] = float(false_match_points)
        metrics["false_match_pairs"] = float(false_match_pairs)
        metrics["online_false_match_points"] = float(online_false_match_points)
        metrics["online_false_match_pairs"] = float(online_false_match_pairs)
        metrics["illumination_consistency_points"] = float(illumination_consistency_points)
        metrics["illumination_consistency_pairs"] = float(illumination_consistency_pairs_used)
        metrics["illumination_match_points"] = float(illumination_match_points)
        metrics["illumination_match_pairs"] = float(illumination_match_pairs_used)
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
    if use_grad_scaler:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
    return {
        "loss": sum(loss_values) / float(len(loss_values)),
        "grad_l2": grad_norm,
        "skipped": 0.0,
        **averaged_step_metrics(metric_rows, sampled_count),
        **aggregate_graph_matcher_loss_metrics(graph_metric_rows),
        **auxiliary_loss_metrics(),
        "pseudo_label_points": float(pseudo_label_points),
        "pseudo_keypoint_points": float(pseudo_keypoint_points),
        "pseudo_label_pairs": float(pseudo_label_pairs),
        "false_match_points": float(false_match_points),
        "false_match_pairs": float(false_match_pairs),
        "online_false_match_points": float(online_false_match_points),
        "online_false_match_pairs": float(online_false_match_pairs),
        "illumination_consistency_points": float(illumination_consistency_points),
        "illumination_consistency_pairs": float(illumination_consistency_pairs_used),
        "illumination_match_points": float(illumination_match_points),
        "illumination_match_pairs": float(illumination_match_pairs_used),
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


def _set_rejection_training_default(args: argparse.Namespace, name: str) -> None:
    value = getattr(args, name)
    if value <= 0 or value == REJECTION_TRAINING_BASE_DEFAULTS.get(name):
        setattr(args, name, REJECTION_TRAINING_DEFAULTS[name])


def apply_rejection_training_defaults(args: argparse.Namespace) -> None:
    """把拒配训练开关展开成实际会参与 loss 的参数。"""

    if not getattr(args, "enable_rejection_training", False):
        return
    args.train_graph_matcher = True
    args.graph_matcher_online_false_no_match = True
    args.report_matcher_mode = "graph_matcher" if args.report_matcher_mode == "raw_descriptor" else args.report_matcher_mode
    args.report_graph_inference_preset = (
        "fast" if args.report_graph_inference_preset == "off" else args.report_graph_inference_preset
    )
    for name in REJECTION_TRAINING_DEFAULTS:
        _set_rejection_training_default(args, name)


def aggregate_graph_matcher_loss_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in GRAPH_MATCHER_LOSS_METRIC_KEYS}
    total_points = sum(max(0.0, float(row.get("points", 0.0))) for row in rows)
    if total_points <= 0.0:
        total_points = float(len(rows))
        weights = [1.0 for _ in rows]
    else:
        weights = [max(0.0, float(row.get("points", 0.0))) for row in rows]
    return {
        key: sum(float(row.get(key, 0.0)) * weight for row, weight in zip(rows, weights)) / total_points
        for key in GRAPH_MATCHER_LOSS_METRIC_KEYS
    }


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


def parse_rotation_consistency_degrees(value: str) -> list[int]:
    degrees: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        degree = int(item)
        _rotation_k(degree)
        degrees.append(degree)
    if not degrees:
        raise argparse.ArgumentTypeError("rotation consistency degrees must not be empty")
    return degrees


def parse_graph_supervision_depths(value: str) -> list[int]:
    depths: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            depth = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("graph supervision depths must be comma-separated integers") from exc
        if depth <= 0:
            raise argparse.ArgumentTypeError("graph supervision depths must be positive")
        depths.append(depth)
    return sorted(set(depths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the current PFM model in PyTorch from warp correspondences")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--init-pytorch-state", type=Path, default=None)
    parser.add_argument("--init-random", action="store_true")
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--validation-cache-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pytorch_pfm_finetune"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp-dtype", choices=AMP_DTYPE_CHOICES, default="float16")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--skip-nonfinite-steps", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--keypoint-weight", type=float, default=0.0)
    parser.add_argument("--keypoint-negative-weight", type=float, default=0.01)
    parser.add_argument("--keypoint-offset-weight", type=float, default=0.0)
    parser.add_argument("--selected-keypoint-offset-weight", type=float, default=0.0)
    parser.add_argument("--selected-keypoint-offset-max-points", type=int, default=256)
    parser.add_argument("--selected-keypoint-offset-inverse-radius-px", type=float, default=1.5)
    parser.add_argument("--train-keypoint-offset-head-only", action="store_true")
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
    parser.add_argument("--train-reliability-head", action="store_true")
    parser.add_argument("--train-graph-matcher", action="store_true")
    parser.add_argument("--train-graph-calibration-only", action="store_true")
    parser.add_argument("--train-pair-accept-head-only", action="store_true")
    parser.add_argument("--descriptor-geometry-mode", choices=pfm_model.DESCRIPTOR_GEOMETRY_MODES, default="full")
    parser.add_argument("--descriptor-geometry-blend-weight", type=float, default=1.0)
    parser.add_argument("--descriptor-scale-log-clamp-min", type=float, default=-2.0)
    parser.add_argument("--descriptor-scale-log-clamp-max", type=float, default=2.0)
    parser.add_argument(
        "--descriptor-geometry-safety-schedule",
        choices=pfm_model.DESCRIPTOR_GEOMETRY_SAFETY_SCHEDULES,
        default="off",
    )
    parser.add_argument("--quality-score-mode", choices=pfm_model.QUALITY_SCORE_MODES, default="soft")
    parser.add_argument("--graph-matcher-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--graph-matcher-metadata-mode",
        choices=["full", "calibrated", "descriptor_only", "no_xy", "no_geometry", "no_quality"],
        default="calibrated",
    )
    parser.add_argument("--graph-matcher-no-match-points", type=int, default=0)
    parser.add_argument("--graph-matcher-no-match-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-no-match-min-distance", type=float, default=4.0)
    parser.add_argument("--graph-matcher-assignment-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-train-max-attention-layers", type=int, default=0)
    parser.add_argument("--graph-matcher-train-random-attention-layers", action="store_true")
    parser.add_argument("--graph-matcher-train-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--graph-matcher-train-width-keep-ratio", type=float, default=1.0)
    parser.add_argument("--graph-matcher-deep-supervision-depths", type=parse_graph_supervision_depths, default=[])
    parser.add_argument("--graph-matcher-deep-supervision-weight", type=float, default=0.0)
    parser.add_argument(
        "--matcher-reliability-pair-bias",
        choices=pfm_model.MATCHER_RELIABILITY_PAIR_BIAS_MODES,
        default="off",
    )
    parser.add_argument(
        "--matcher-reliability-dustbin-bias",
        choices=pfm_model.MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES,
        default="off",
    )
    parser.add_argument(
        "--matcher-final-accept-score-mode",
        choices=pfm_model.MATCHER_FINAL_ACCEPT_SCORE_MODES,
        default="none",
    )
    parser.add_argument(
        "--matcher-accept-assignment-mode",
        choices=pfm_model.MATCHER_ACCEPT_ASSIGNMENT_MODES,
        default="add",
    )
    parser.add_argument("--matcher-final-accept-score-alpha", type=float, default=0.05)
    parser.add_argument("--matcher-geometry-bias-scale", type=float, default=1.0)
    parser.add_argument("--matcher-geometry-bias-clamp", type=float, default=2.0)
    parser.add_argument("--matcher-attention-residual-gate-init", type=float, default=None)
    parser.add_argument("--matcher-attention-residual-gate-start-layer", type=int, default=1)
    parser.add_argument("--matcher-candidate-topk", type=int, default=256)
    parser.add_argument("--graph-matcher-accept-weight", type=float, default=0.2)
    parser.add_argument("--graph-matcher-accept-negative-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-prune-ranking-weight", type=float, default=0.1)
    parser.add_argument("--graph-matcher-prune-ranking-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-stop-confidence-weight", type=float, default=0.05)
    parser.add_argument("--graph-matcher-stop-confidence-margin", type=float, default=0.5)
    parser.add_argument("--graph-matcher-raw-preservation-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-raw-preservation-margin", type=float, default=1.0)
    parser.add_argument("--graph-matcher-raw-preservation-raw-margin", type=float, default=0.05)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-spatial-min-distance", type=float, default=0.0)
    parser.add_argument("--graph-matcher-dustbin-warmup-steps", type=int, default=0)
    parser.add_argument("--graph-matcher-dustbin-ramp-steps", type=int, default=0)
    parser.add_argument("--graph-matcher-positive-dustbin-margin-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-positive-dustbin-margin", type=float, default=0.0)
    parser.add_argument("--graph-matcher-true-match-margin-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-true-match-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-true-geometry-match-count-floor-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-true-geometry-match-count-floor-threshold", type=float, default=0.0)
    parser.add_argument("--graph-matcher-true-geometry-match-count-floor-margin", type=float, default=0.0)
    parser.add_argument("--graph-matcher-final-false-match-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-mined-false-match-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-mined-false-match-loss-cap", type=float, default=0.0)
    parser.add_argument("--graph-matcher-mined-false-match-reference-margin", type=float, default=-1.0)
    parser.add_argument("--graph-matcher-final-false-match-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-final-false-match-min-score", type=float, default=0.0)
    parser.add_argument("--graph-matcher-final-false-match-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-final-false-match-spatial-min-distance", type=float, default=0.0)
    parser.add_argument("--graph-matcher-raw-false-match-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-raw-false-match-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-raw-false-match-min-similarity", type=float, default=0.75)
    parser.add_argument("--graph-matcher-raw-false-match-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-raw-false-match-spatial-min-distance", type=float, default=0.0)
    parser.add_argument("--graph-matcher-ransac-consistency-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-ransac-consistency-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-ransac-consistency-residual-threshold-px", type=float, default=3.0)
    parser.add_argument("--graph-matcher-ransac-consistency-min-score", type=float, default=0.0)
    parser.add_argument("--graph-matcher-ransac-consistency-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-warp-outlier-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-warp-outlier-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-warp-outlier-residual-threshold-px", type=float, default=3.0)
    parser.add_argument("--graph-matcher-warp-outlier-min-score", type=float, default=0.0)
    parser.add_argument("--graph-matcher-warp-outlier-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-warp-outlier-accept-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-warp-outlier-accept-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-warp-outlier-accept-residual-threshold-px", type=float, default=3.0)
    parser.add_argument("--graph-matcher-warp-outlier-accept-min-score", type=float, default=0.0)
    parser.add_argument("--graph-matcher-warp-soft-boundary-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-warp-soft-boundary-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-warp-soft-boundary-lower-residual-px", type=float, default=5.0)
    parser.add_argument("--graph-matcher-warp-soft-boundary-upper-residual-px", type=float, default=8.0)
    parser.add_argument("--graph-matcher-warp-soft-boundary-min-score", type=float, default=0.0)
    parser.add_argument("--graph-matcher-pair-acceptance-loss-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-depth-distillation-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-depth-distillation-teacher-layers", type=int, default=0)
    parser.add_argument("--graph-matcher-depth-distillation-temperature", type=float, default=1.0)
    parser.add_argument("--graph-matcher-teacher-guard-state", type=Path, default=None)
    parser.add_argument("--graph-matcher-teacher-guard-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-guard-positive-margin-tolerance", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-guard-false-margin-tolerance", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-score-floor-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-score-floor-tolerance", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-score-floor-min-score", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-match-count-floor-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-match-count-floor-threshold", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-match-count-floor-margin", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-match-count-ceiling-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-match-count-ceiling-threshold", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-match-count-ceiling-margin", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-distillation-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-teacher-distillation-temperature", type=float, default=1.0)
    parser.add_argument("--graph-matcher-positive-dustbin-guard-reject-threshold", type=float, default=1.1)
    parser.add_argument(
        "--graph-matcher-positive-dustbin-guard-margin-threshold",
        type=float,
        default=-float("inf"),
    )
    parser.add_argument("--graph-matcher-train-candidate-topk", type=int, default=0)
    parser.add_argument("--graph-matcher-semi-dense-no-match-points", type=int, default=0)
    parser.add_argument("--graph-matcher-semi-dense-min-score", type=float, default=0.0)
    parser.add_argument("--graph-matcher-online-false-no-match", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--enable-rejection-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a safe no-match/dustbin training preset for descriptor false matches and GraphMatcher rejection.",
    )
    parser.add_argument("--matchability-weight", type=float, default=0.0)
    parser.add_argument("--descriptor-uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--no-match-prior-weight", type=float, default=0.0)
    parser.add_argument("--reliability-negative-points", type=int, default=0)
    parser.add_argument("--reliability-negative-min-distance", type=float, default=4.0)
    parser.add_argument("--rotation-descriptor-consistency-weight", type=float, default=0.0)
    parser.add_argument("--orientation-consistency-weight", type=float, default=0.0)
    parser.add_argument("--scale-consistency-weight", type=float, default=0.0)
    parser.add_argument("--affine-consistency-weight", type=float, default=0.0)
    parser.add_argument("--affine-regularization-weight", type=float, default=0.0)
    parser.add_argument("--rotation-consistency-degrees", type=parse_rotation_consistency_degrees, default=[90, 180, 270])
    parser.add_argument("--freeze-descriptor-head", action="store_true")
    parser.add_argument("--freeze-extractor-warmup-steps", type=int, default=0)
    parser.add_argument("--training-texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--generate-training-report", action="store_true")
    parser.add_argument("--report-output-dir", type=Path, default=None)
    parser.add_argument("--report-sample-count", type=int, default=16)
    parser.add_argument("--report-max-keypoints", type=int, default=2048)
    parser.add_argument("--report-max-matches", type=int, default=0)
    parser.add_argument("--report-draw-matches", type=int, default=0)
    parser.add_argument("--report-min-margin", type=float, default=0.0)
    parser.add_argument("--report-matcher-mode", choices=["raw_descriptor", "graph_matcher", "both"], default="raw_descriptor")
    parser.add_argument("--report-graph-inference-preset", choices=GRAPH_INFERENCE_PRESET_CHOICES, default="off")
    parser.add_argument("--report-graph-width-prune-min-score", type=float, default=-1.0)
    parser.add_argument("--report-graph-early-stop-min-confidence", type=float, default=-1.0)
    parser.add_argument("--report-graph-min-accept-probability", type=float, default=-1.0)
    parser.add_argument("--report-graph-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--report-graph-width-prune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--report-texture-keypoint-fraction", type=float, default=1.0)
    parser.add_argument("--report-weak-texture-keypoint-fraction", type=float, default=0.0)
    parser.add_argument("--report-keypoint-spatial-bins", type=int, default=8)
    parser.add_argument("--report-keypoint-cell-cap", type=int, default=0)
    parser.add_argument("--report-coverage-bins", type=int, default=8)
    parser.add_argument("--report-required-sample-glob", action="append", default=[])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    apply_rejection_training_defaults(args)
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
    if args.keypoint_weight < 0.0:
        parser.error("--keypoint-weight must be nonnegative")
    if args.keypoint_negative_weight < 0.0:
        parser.error("--keypoint-negative-weight must be nonnegative")
    if args.keypoint_offset_weight < 0.0:
        parser.error("--keypoint-offset-weight must be nonnegative")
    if args.selected_keypoint_offset_weight < 0.0:
        parser.error("--selected-keypoint-offset-weight must be nonnegative")
    if args.selected_keypoint_offset_max_points < 0:
        parser.error("--selected-keypoint-offset-max-points must be nonnegative")
    if args.selected_keypoint_offset_inverse_radius_px < 0.0:
        parser.error("--selected-keypoint-offset-inverse-radius-px must be nonnegative")
    if args.train_keypoint_offset_head_only and (
        args.pseudo_keypoint_weight > 0.0 or args.keypoint_weight > 0.0
    ):
        parser.error("--train-keypoint-offset-head-only conflicts with keypoint heatmap supervision")
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
    if args.graph_matcher_assignment_weight < 0.0:
        parser.error("--graph-matcher-assignment-weight must be nonnegative")
    if args.graph_matcher_train_max_attention_layers < 0:
        parser.error("--graph-matcher-train-max-attention-layers must be nonnegative")
    if (
        not math.isfinite(float(args.graph_matcher_train_max_attention_work_fraction))
        or args.graph_matcher_train_max_attention_work_fraction < 0.0
        or args.graph_matcher_train_max_attention_work_fraction > 1.0
    ):
        parser.error("--graph-matcher-train-max-attention-work-fraction must be in [0, 1]")
    if (
        not math.isfinite(float(args.graph_matcher_train_width_keep_ratio))
        or args.graph_matcher_train_width_keep_ratio <= 0.0
        or args.graph_matcher_train_width_keep_ratio > 1.0
    ):
        parser.error("--graph-matcher-train-width-keep-ratio must be in (0, 1]")
    if args.graph_matcher_deep_supervision_weight < 0.0:
        parser.error("--graph-matcher-deep-supervision-weight must be nonnegative")
    if args.matcher_candidate_topk < 0:
        parser.error("--matcher-candidate-topk must be nonnegative")
    if args.matcher_final_accept_score_alpha < 0.0:
        parser.error("--matcher-final-accept-score-alpha must be nonnegative")
    if not math.isfinite(float(args.matcher_geometry_bias_scale)):
        parser.error("--matcher-geometry-bias-scale must be finite")
    if not math.isfinite(float(args.matcher_final_accept_score_alpha)):
        parser.error("--matcher-final-accept-score-alpha must be finite")
    if not math.isfinite(float(args.matcher_geometry_bias_clamp)) or args.matcher_geometry_bias_clamp < 0.0:
        parser.error("--matcher-geometry-bias-clamp must be finite and nonnegative")
    if args.matcher_attention_residual_gate_init is not None and not math.isfinite(
        float(args.matcher_attention_residual_gate_init)
    ):
        parser.error("--matcher-attention-residual-gate-init must be finite")
    if args.matcher_attention_residual_gate_start_layer < 1:
        parser.error("--matcher-attention-residual-gate-start-layer must be at least 1")
    if args.graph_matcher_accept_weight < 0.0:
        parser.error("--graph-matcher-accept-weight must be nonnegative")
    if args.graph_matcher_accept_negative_topk < 0:
        parser.error("--graph-matcher-accept-negative-topk must be nonnegative")
    if args.graph_matcher_prune_ranking_weight < 0.0:
        parser.error("--graph-matcher-prune-ranking-weight must be nonnegative")
    if args.graph_matcher_prune_ranking_margin < 0.0:
        parser.error("--graph-matcher-prune-ranking-margin must be nonnegative")
    if args.graph_matcher_stop_confidence_weight < 0.0:
        parser.error("--graph-matcher-stop-confidence-weight must be nonnegative")
    if args.graph_matcher_stop_confidence_margin < 0.0:
        parser.error("--graph-matcher-stop-confidence-margin must be nonnegative")
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
    if args.graph_matcher_dustbin_warmup_steps < 0:
        parser.error("--graph-matcher-dustbin-warmup-steps must be nonnegative")
    if args.graph_matcher_dustbin_ramp_steps < 0:
        parser.error("--graph-matcher-dustbin-ramp-steps must be nonnegative")
    if args.graph_matcher_positive_dustbin_margin_weight < 0.0:
        parser.error("--graph-matcher-positive-dustbin-margin-weight must be nonnegative")
    if args.graph_matcher_positive_dustbin_margin < 0.0:
        parser.error("--graph-matcher-positive-dustbin-margin must be nonnegative")
    if args.graph_matcher_true_match_margin_weight < 0.0:
        parser.error("--graph-matcher-true-match-margin-weight must be nonnegative")
    if args.graph_matcher_true_match_margin < 0.0:
        parser.error("--graph-matcher-true-match-margin must be nonnegative")
    if args.graph_matcher_true_geometry_match_count_floor_weight < 0.0:
        parser.error("--graph-matcher-true-geometry-match-count-floor-weight must be nonnegative")
    if not math.isfinite(float(args.graph_matcher_true_geometry_match_count_floor_threshold)):
        parser.error("--graph-matcher-true-geometry-match-count-floor-threshold must be finite")
    if args.graph_matcher_true_geometry_match_count_floor_margin < 0.0:
        parser.error("--graph-matcher-true-geometry-match-count-floor-margin must be nonnegative")
    if args.graph_matcher_final_false_match_weight < 0.0:
        parser.error("--graph-matcher-final-false-match-weight must be nonnegative")
    if args.graph_matcher_mined_false_match_weight < 0.0:
        parser.error("--graph-matcher-mined-false-match-weight must be nonnegative")
    if args.graph_matcher_mined_false_match_loss_cap < 0.0:
        parser.error("--graph-matcher-mined-false-match-loss-cap must be nonnegative")
    if (
        not math.isfinite(float(args.graph_matcher_mined_false_match_reference_margin))
        or args.graph_matcher_mined_false_match_reference_margin < -1.0
    ):
        parser.error("--graph-matcher-mined-false-match-reference-margin must be finite and >= -1")
    if args.graph_matcher_final_false_match_topk < 0:
        parser.error("--graph-matcher-final-false-match-topk must be nonnegative")
    if args.graph_matcher_final_false_match_min_score < 0.0:
        parser.error("--graph-matcher-final-false-match-min-score must be nonnegative")
    if args.graph_matcher_final_false_match_margin < 0.0:
        parser.error("--graph-matcher-final-false-match-margin must be nonnegative")
    if args.graph_matcher_final_false_match_spatial_min_distance < 0.0:
        parser.error("--graph-matcher-final-false-match-spatial-min-distance must be nonnegative")
    if args.graph_matcher_raw_false_match_weight < 0.0:
        parser.error("--graph-matcher-raw-false-match-weight must be nonnegative")
    if args.graph_matcher_raw_false_match_topk < 0:
        parser.error("--graph-matcher-raw-false-match-topk must be nonnegative")
    if args.graph_matcher_raw_false_match_min_similarity < -1.0 or args.graph_matcher_raw_false_match_min_similarity > 1.0:
        parser.error("--graph-matcher-raw-false-match-min-similarity must be in [-1, 1]")
    if args.graph_matcher_raw_false_match_margin < 0.0:
        parser.error("--graph-matcher-raw-false-match-margin must be nonnegative")
    if args.graph_matcher_raw_false_match_spatial_min_distance < 0.0:
        parser.error("--graph-matcher-raw-false-match-spatial-min-distance must be nonnegative")
    if args.graph_matcher_ransac_consistency_weight < 0.0:
        parser.error("--graph-matcher-ransac-consistency-weight must be nonnegative")
    if args.graph_matcher_ransac_consistency_topk < 0:
        parser.error("--graph-matcher-ransac-consistency-topk must be nonnegative")
    if args.graph_matcher_ransac_consistency_residual_threshold_px < 0.0:
        parser.error("--graph-matcher-ransac-consistency-residual-threshold-px must be nonnegative")
    if args.graph_matcher_ransac_consistency_min_score < 0.0:
        parser.error("--graph-matcher-ransac-consistency-min-score must be nonnegative")
    if args.graph_matcher_ransac_consistency_margin < 0.0:
        parser.error("--graph-matcher-ransac-consistency-margin must be nonnegative")
    if args.graph_matcher_warp_outlier_weight < 0.0:
        parser.error("--graph-matcher-warp-outlier-weight must be nonnegative")
    if args.graph_matcher_warp_outlier_topk < 0:
        parser.error("--graph-matcher-warp-outlier-topk must be nonnegative")
    if args.graph_matcher_warp_outlier_residual_threshold_px < 0.0:
        parser.error("--graph-matcher-warp-outlier-residual-threshold-px must be nonnegative")
    if args.graph_matcher_warp_outlier_min_score < 0.0:
        parser.error("--graph-matcher-warp-outlier-min-score must be nonnegative")
    if args.graph_matcher_warp_outlier_margin < 0.0:
        parser.error("--graph-matcher-warp-outlier-margin must be nonnegative")
    if args.graph_matcher_warp_outlier_accept_weight < 0.0:
        parser.error("--graph-matcher-warp-outlier-accept-weight must be nonnegative")
    if args.graph_matcher_warp_outlier_accept_topk < 0:
        parser.error("--graph-matcher-warp-outlier-accept-topk must be nonnegative")
    if args.graph_matcher_warp_outlier_accept_residual_threshold_px < 0.0:
        parser.error("--graph-matcher-warp-outlier-accept-residual-threshold-px must be nonnegative")
    if args.graph_matcher_warp_outlier_accept_min_score < 0.0:
        parser.error("--graph-matcher-warp-outlier-accept-min-score must be nonnegative")
    if args.graph_matcher_warp_soft_boundary_weight < 0.0:
        parser.error("--graph-matcher-warp-soft-boundary-weight must be nonnegative")
    if args.graph_matcher_warp_soft_boundary_topk < 0:
        parser.error("--graph-matcher-warp-soft-boundary-topk must be nonnegative")
    if (
        not math.isfinite(float(args.graph_matcher_warp_soft_boundary_lower_residual_px))
        or args.graph_matcher_warp_soft_boundary_lower_residual_px < 0.0
    ):
        parser.error("--graph-matcher-warp-soft-boundary-lower-residual-px must be finite and nonnegative")
    if (
        not math.isfinite(float(args.graph_matcher_warp_soft_boundary_upper_residual_px))
        or args.graph_matcher_warp_soft_boundary_upper_residual_px
        <= args.graph_matcher_warp_soft_boundary_lower_residual_px
    ):
        parser.error(
            "--graph-matcher-warp-soft-boundary-upper-residual-px must be finite and greater than lower"
        )
    if args.graph_matcher_warp_soft_boundary_min_score < 0.0:
        parser.error("--graph-matcher-warp-soft-boundary-min-score must be nonnegative")
    if args.graph_matcher_pair_acceptance_loss_weight < 0.0:
        parser.error("--graph-matcher-pair-acceptance-loss-weight must be nonnegative")
    if args.graph_matcher_depth_distillation_weight < 0.0:
        parser.error("--graph-matcher-depth-distillation-weight must be nonnegative")
    if args.graph_matcher_depth_distillation_teacher_layers < 0:
        parser.error("--graph-matcher-depth-distillation-teacher-layers must be nonnegative")
    if (
        not math.isfinite(float(args.graph_matcher_depth_distillation_temperature))
        or args.graph_matcher_depth_distillation_temperature <= 0.0
    ):
        parser.error("--graph-matcher-depth-distillation-temperature must be positive and finite")
    if args.graph_matcher_teacher_guard_weight < 0.0:
        parser.error("--graph-matcher-teacher-guard-weight must be nonnegative")
    if args.graph_matcher_teacher_guard_positive_margin_tolerance < 0.0:
        parser.error("--graph-matcher-teacher-guard-positive-margin-tolerance must be nonnegative")
    if args.graph_matcher_teacher_guard_false_margin_tolerance < 0.0:
        parser.error("--graph-matcher-teacher-guard-false-margin-tolerance must be nonnegative")
    if args.graph_matcher_teacher_score_floor_weight < 0.0:
        parser.error("--graph-matcher-teacher-score-floor-weight must be nonnegative")
    if args.graph_matcher_teacher_score_floor_tolerance < 0.0:
        parser.error("--graph-matcher-teacher-score-floor-tolerance must be nonnegative")
    if not math.isfinite(float(args.graph_matcher_teacher_score_floor_min_score)):
        parser.error("--graph-matcher-teacher-score-floor-min-score must be finite")
    if args.graph_matcher_teacher_match_count_floor_weight < 0.0:
        parser.error("--graph-matcher-teacher-match-count-floor-weight must be nonnegative")
    if not math.isfinite(float(args.graph_matcher_teacher_match_count_floor_threshold)):
        parser.error("--graph-matcher-teacher-match-count-floor-threshold must be finite")
    if args.graph_matcher_teacher_match_count_floor_margin < 0.0:
        parser.error("--graph-matcher-teacher-match-count-floor-margin must be nonnegative")
    if args.graph_matcher_teacher_match_count_ceiling_weight < 0.0:
        parser.error("--graph-matcher-teacher-match-count-ceiling-weight must be nonnegative")
    if not math.isfinite(float(args.graph_matcher_teacher_match_count_ceiling_threshold)):
        parser.error("--graph-matcher-teacher-match-count-ceiling-threshold must be finite")
    if args.graph_matcher_teacher_match_count_ceiling_margin < 0.0:
        parser.error("--graph-matcher-teacher-match-count-ceiling-margin must be nonnegative")
    if args.graph_matcher_teacher_distillation_weight < 0.0:
        parser.error("--graph-matcher-teacher-distillation-weight must be nonnegative")
    if (
        not math.isfinite(float(args.graph_matcher_teacher_distillation_temperature))
        or args.graph_matcher_teacher_distillation_temperature <= 0.0
    ):
        parser.error("--graph-matcher-teacher-distillation-temperature must be positive and finite")
    if (
        args.graph_matcher_teacher_guard_weight > 0.0
        or args.graph_matcher_teacher_score_floor_weight > 0.0
        or args.graph_matcher_teacher_match_count_floor_weight > 0.0
        or args.graph_matcher_teacher_match_count_ceiling_weight > 0.0
        or args.graph_matcher_teacher_distillation_weight > 0.0
    ) and args.graph_matcher_teacher_guard_state is None:
        parser.error(
            "--graph-matcher-teacher-guard-state is required when teacher guard, score floor, match-count floor, match-count ceiling, or distillation weight is positive"
        )
    if not math.isfinite(float(args.graph_matcher_positive_dustbin_guard_reject_threshold)):
        parser.error("--graph-matcher-positive-dustbin-guard-reject-threshold must be finite")
    if args.graph_matcher_train_candidate_topk < 0:
        parser.error("--graph-matcher-train-candidate-topk must be nonnegative")
    if args.graph_matcher_semi_dense_no_match_points < 0:
        parser.error("--graph-matcher-semi-dense-no-match-points must be nonnegative")
    if args.graph_matcher_semi_dense_min_score < 0.0:
        parser.error("--graph-matcher-semi-dense-min-score must be nonnegative")
    if (
        not math.isfinite(float(args.descriptor_geometry_blend_weight))
        or args.descriptor_geometry_blend_weight < 0.0
        or args.descriptor_geometry_blend_weight > 1.0
    ):
        parser.error("--descriptor-geometry-blend-weight must be in [0, 1]")
    if (
        not math.isfinite(float(args.descriptor_scale_log_clamp_min))
        or not math.isfinite(float(args.descriptor_scale_log_clamp_max))
        or args.descriptor_scale_log_clamp_min > args.descriptor_scale_log_clamp_max
    ):
        parser.error("--descriptor-scale-log-clamp-min/max must be finite and ordered")
    if args.matchability_weight < 0.0:
        parser.error("--matchability-weight must be nonnegative")
    if args.descriptor_uncertainty_weight < 0.0:
        parser.error("--descriptor-uncertainty-weight must be nonnegative")
    if args.no_match_prior_weight < 0.0:
        parser.error("--no-match-prior-weight must be nonnegative")
    if args.reliability_negative_points < 0:
        parser.error("--reliability-negative-points must be nonnegative")
    if args.reliability_negative_min_distance < 0.0:
        parser.error("--reliability-negative-min-distance must be nonnegative")
    if args.rotation_descriptor_consistency_weight < 0.0:
        parser.error("--rotation-descriptor-consistency-weight must be nonnegative")
    if args.orientation_consistency_weight < 0.0:
        parser.error("--orientation-consistency-weight must be nonnegative")
    if args.scale_consistency_weight < 0.0:
        parser.error("--scale-consistency-weight must be nonnegative")
    if args.affine_consistency_weight < 0.0:
        parser.error("--affine-consistency-weight must be nonnegative")
    if args.affine_regularization_weight < 0.0:
        parser.error("--affine-regularization-weight must be nonnegative")
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
    if args.report_max_matches < 0:
        parser.error("--report-max-matches must be nonnegative; use 0 to keep all matches")
    if args.report_draw_matches < 0:
        parser.error("--report-draw-matches must be nonnegative; use 0 to draw all matches")
    if args.report_min_margin < 0.0:
        parser.error("--report-min-margin must be nonnegative")
    if args.report_graph_width_prune_min_score < -1.0 or args.report_graph_width_prune_min_score > 1.0:
        parser.error("--report-graph-width-prune-min-score must be in [-1, 1]")
    if args.report_graph_early_stop_min_confidence < -1.0 or args.report_graph_early_stop_min_confidence > 1.0:
        parser.error("--report-graph-early-stop-min-confidence must be in [-1, 1]")
    if args.report_graph_min_accept_probability < -1.0 or args.report_graph_min_accept_probability > 1.0:
        parser.error("--report-graph-min-accept-probability must be in [-1, 1]")
    if args.report_graph_max_attention_work_fraction < 0.0 or args.report_graph_max_attention_work_fraction > 1.0:
        parser.error("--report-graph-max-attention-work-fraction must be in [0, 1]")
    if args.report_graph_width_prune_keep_ratio < 0.0 or args.report_graph_width_prune_keep_ratio > 1.0:
        parser.error("--report-graph-width-prune-keep-ratio must be in [0, 1]")
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
    if args.freeze_extractor_warmup_steps < 0:
        parser.error("--freeze-extractor-warmup-steps must be nonnegative")
    return args


def load_training_model(args: argparse.Namespace) -> tuple[pfm_model.PlanetaryFeatureMatcher, pfm_model.CheckpointConfig]:
    if getattr(args, "init_random", False):
        model = pfm_model.PlanetaryFeatureMatcher().to(args.device)
        model.train()
    elif getattr(args, "init_pytorch_state", None) is not None:
        model, _ = pfm_model.load_pytorch_state(args.init_pytorch_state, device=args.device)
    elif args.checkpoint is None:
        raise ValueError("checkpoint is required unless init_pytorch_state or init_random is set")
    else:
        model, _ = pfm_model.load_libtorch_checkpoint(args.checkpoint, device=args.device)
    model.set_matcher_calibration(
        reliability_pair_bias_mode=getattr(args, "matcher_reliability_pair_bias", "off"),
        reliability_dustbin_bias_mode=getattr(args, "matcher_reliability_dustbin_bias", "off"),
        final_accept_score_mode=getattr(args, "matcher_final_accept_score_mode", "none"),
        geometry_bias_scale=getattr(args, "matcher_geometry_bias_scale", 1.0),
        accept_assignment_mode=getattr(args, "matcher_accept_assignment_mode", "add"),
        final_accept_score_alpha=getattr(args, "matcher_final_accept_score_alpha", 0.05),
        geometry_bias_clamp=getattr(args, "matcher_geometry_bias_clamp", 2.0),
        attention_residual_gate_init=getattr(args, "matcher_attention_residual_gate_init", None),
        attention_residual_gate_start_layer=getattr(args, "matcher_attention_residual_gate_start_layer", 1),
        candidate_topk=getattr(args, "matcher_candidate_topk", 256),
    )
    model.train()
    return model, model.config


def load_graph_matcher_teacher_guard_model(
    args: argparse.Namespace,
) -> pfm_model.PlanetaryFeatureMatcher | None:
    state_path = getattr(args, "graph_matcher_teacher_guard_state", None)
    if state_path is None:
        return None
    teacher, _ = pfm_model.load_pytorch_state(state_path, device=getattr(args, "device", "cpu"))
    teacher.set_matcher_calibration(
        reliability_pair_bias_mode=getattr(args, "matcher_reliability_pair_bias", "off"),
        reliability_dustbin_bias_mode=getattr(args, "matcher_reliability_dustbin_bias", "off"),
        final_accept_score_mode=getattr(args, "matcher_final_accept_score_mode", "none"),
        geometry_bias_scale=getattr(args, "matcher_geometry_bias_scale", 1.0),
        accept_assignment_mode=getattr(args, "matcher_accept_assignment_mode", "add"),
        final_accept_score_alpha=getattr(args, "matcher_final_accept_score_alpha", 0.05),
        geometry_bias_clamp=getattr(args, "matcher_geometry_bias_clamp", 2.0),
        attention_residual_gate_init=getattr(args, "matcher_attention_residual_gate_init", None),
        attention_residual_gate_start_layer=getattr(args, "matcher_attention_residual_gate_start_layer", 1),
        candidate_topk=getattr(args, "matcher_candidate_topk", 256),
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


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
            "--graph-inference-preset",
            args.report_graph_inference_preset,
            "--graph-width-prune-min-score",
            str(args.report_graph_width_prune_min_score),
            "--graph-early-stop-min-confidence",
            str(args.report_graph_early_stop_min_confidence),
            "--graph-min-accept-probability",
            str(args.report_graph_min_accept_probability),
            "--graph-max-attention-work-fraction",
            str(args.report_graph_max_attention_work_fraction),
            "--graph-width-prune-keep-ratio",
            str(args.report_graph_width_prune_keep_ratio),
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
            "amp": bool(args.amp),
            "amp_dtype": str(args.amp_dtype),
            "activation_checkpointing": bool(args.activation_checkpointing),
            "descriptor_geometry_safety_schedule": str(args.descriptor_geometry_safety_schedule),
            "keypoint_offset_weight": float(getattr(args, "keypoint_offset_weight", 0.0)),
            "selected_keypoint_offset_weight": float(getattr(args, "selected_keypoint_offset_weight", 0.0)),
            "selected_keypoint_offset_max_points": int(getattr(args, "selected_keypoint_offset_max_points", 0)),
            "selected_keypoint_offset_inverse_radius_px": float(
                getattr(args, "selected_keypoint_offset_inverse_radius_px", 0.0)
            ),
            "train_keypoint_offset_head_only": bool(getattr(args, "train_keypoint_offset_head_only", False)),
            "graph_matcher_true_geometry_match_count_floor_weight": float(
                getattr(args, "graph_matcher_true_geometry_match_count_floor_weight", 0.0)
            ),
            "graph_matcher_true_geometry_match_count_floor_threshold": float(
                getattr(args, "graph_matcher_true_geometry_match_count_floor_threshold", 0.0)
            ),
            "graph_matcher_true_geometry_match_count_floor_margin": float(
                getattr(args, "graph_matcher_true_geometry_match_count_floor_margin", 0.0)
            ),
            "graph_matcher_teacher_guard_state": (
                str(args.graph_matcher_teacher_guard_state)
                if getattr(args, "graph_matcher_teacher_guard_state", None) is not None
                else None
            ),
            "graph_matcher_teacher_guard_weight": float(
                getattr(args, "graph_matcher_teacher_guard_weight", 0.0)
            ),
            "graph_matcher_teacher_score_floor_weight": float(
                getattr(args, "graph_matcher_teacher_score_floor_weight", 0.0)
            ),
            "graph_matcher_teacher_score_floor_tolerance": float(
                getattr(args, "graph_matcher_teacher_score_floor_tolerance", 0.0)
            ),
            "graph_matcher_teacher_score_floor_min_score": float(
                getattr(args, "graph_matcher_teacher_score_floor_min_score", 0.0)
            ),
            "graph_matcher_teacher_match_count_floor_weight": float(
                getattr(args, "graph_matcher_teacher_match_count_floor_weight", 0.0)
            ),
            "graph_matcher_teacher_match_count_floor_threshold": float(
                getattr(args, "graph_matcher_teacher_match_count_floor_threshold", 0.0)
            ),
            "graph_matcher_teacher_match_count_floor_margin": float(
                getattr(args, "graph_matcher_teacher_match_count_floor_margin", 0.0)
            ),
            "graph_matcher_teacher_match_count_ceiling_weight": float(
                getattr(args, "graph_matcher_teacher_match_count_ceiling_weight", 0.0)
            ),
            "graph_matcher_teacher_match_count_ceiling_threshold": float(
                getattr(args, "graph_matcher_teacher_match_count_ceiling_threshold", 0.0)
            ),
            "graph_matcher_teacher_match_count_ceiling_margin": float(
                getattr(args, "graph_matcher_teacher_match_count_ceiling_margin", 0.0)
            ),
            "graph_matcher_teacher_distillation_weight": float(
                getattr(args, "graph_matcher_teacher_distillation_weight", 0.0)
            ),
            "graph_matcher_teacher_distillation_temperature": float(
                getattr(args, "graph_matcher_teacher_distillation_temperature", 1.0)
            ),
            "graph_matcher_warp_outlier_weight": float(
                getattr(args, "graph_matcher_warp_outlier_weight", 0.0)
            ),
            "graph_matcher_warp_outlier_topk": int(
                getattr(args, "graph_matcher_warp_outlier_topk", 0)
            ),
            "graph_matcher_warp_outlier_residual_threshold_px": float(
                getattr(args, "graph_matcher_warp_outlier_residual_threshold_px", 0.0)
            ),
            "graph_matcher_warp_outlier_min_score": float(
                getattr(args, "graph_matcher_warp_outlier_min_score", 0.0)
            ),
            "graph_matcher_warp_outlier_margin": float(
                getattr(args, "graph_matcher_warp_outlier_margin", 0.0)
            ),
            "graph_matcher_warp_outlier_accept_weight": float(
                getattr(args, "graph_matcher_warp_outlier_accept_weight", 0.0)
            ),
            "graph_matcher_warp_outlier_accept_topk": int(
                getattr(args, "graph_matcher_warp_outlier_accept_topk", 0)
            ),
            "graph_matcher_warp_outlier_accept_residual_threshold_px": float(
                getattr(args, "graph_matcher_warp_outlier_accept_residual_threshold_px", 0.0)
            ),
            "graph_matcher_warp_outlier_accept_min_score": float(
                getattr(args, "graph_matcher_warp_outlier_accept_min_score", 0.0)
            ),
            "graph_matcher_warp_soft_boundary_weight": float(
                getattr(args, "graph_matcher_warp_soft_boundary_weight", 0.0)
            ),
            "graph_matcher_warp_soft_boundary_topk": int(
                getattr(args, "graph_matcher_warp_soft_boundary_topk", 0)
            ),
            "graph_matcher_warp_soft_boundary_lower_residual_px": float(
                getattr(args, "graph_matcher_warp_soft_boundary_lower_residual_px", 0.0)
            ),
            "graph_matcher_warp_soft_boundary_upper_residual_px": float(
                getattr(args, "graph_matcher_warp_soft_boundary_upper_residual_px", 0.0)
            ),
            "graph_matcher_warp_soft_boundary_min_score": float(
                getattr(args, "graph_matcher_warp_soft_boundary_min_score", 0.0)
            ),
            "graph_matcher_pair_acceptance_loss_weight": float(
                getattr(args, "graph_matcher_pair_acceptance_loss_weight", 0.0)
            ),
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
    amp_dtype = amp_dtype_from_name(args.amp_dtype)
    grad_scaler = make_grad_scaler(device, enabled=args.amp, dtype=amp_dtype)
    model, config = load_training_model(args)
    graph_matcher_teacher_guard_model = load_graph_matcher_teacher_guard_model(args)
    model.set_descriptor_geometry_mode(args.descriptor_geometry_mode)
    model.set_descriptor_geometry_safety(
        blend_weight=args.descriptor_geometry_blend_weight,
        scale_log_clamp_min=args.descriptor_scale_log_clamp_min,
        scale_log_clamp_max=args.descriptor_scale_log_clamp_max,
    )
    model.set_quality_score_mode(args.quality_score_mode)
    model.set_matcher_calibration(
        reliability_pair_bias_mode=args.matcher_reliability_pair_bias,
        reliability_dustbin_bias_mode=args.matcher_reliability_dustbin_bias,
        final_accept_score_mode=args.matcher_final_accept_score_mode,
        geometry_bias_scale=args.matcher_geometry_bias_scale,
    )
    config = model.config
    if graph_matcher_teacher_guard_model is not None:
        if graph_matcher_teacher_guard_model.config.graph_keypoint_meta_dim != model.config.graph_keypoint_meta_dim:
            raise RuntimeError("teacher guard model graph_keypoint_meta_dim does not match the student model")
        print(
            f"graph_matcher_teacher_guard_state={args.graph_matcher_teacher_guard_state} "
            f"weight={args.graph_matcher_teacher_guard_weight:.3f} "
            f"positive_margin_tolerance={args.graph_matcher_teacher_guard_positive_margin_tolerance:.3f} "
            f"false_margin_tolerance={args.graph_matcher_teacher_guard_false_margin_tolerance:.3f} "
            f"score_floor_weight={args.graph_matcher_teacher_score_floor_weight:.3f} "
            f"score_floor_tolerance={args.graph_matcher_teacher_score_floor_tolerance:.3f} "
            f"score_floor_min_score={args.graph_matcher_teacher_score_floor_min_score:.3f} "
            f"match_count_floor_weight={args.graph_matcher_teacher_match_count_floor_weight:.3f} "
            f"match_count_floor_threshold={args.graph_matcher_teacher_match_count_floor_threshold:.3f} "
            f"match_count_floor_margin={args.graph_matcher_teacher_match_count_floor_margin:.3f} "
            f"match_count_ceiling_weight={args.graph_matcher_teacher_match_count_ceiling_weight:.3f} "
            f"match_count_ceiling_threshold={args.graph_matcher_teacher_match_count_ceiling_threshold:.3f} "
            f"match_count_ceiling_margin={args.graph_matcher_teacher_match_count_ceiling_margin:.3f}",
            flush=True,
        )
    trainable = descriptor_parameters(
        model,
        train_backbone=args.train_backbone,
        train_dual_fpn=args.train_dual_fpn,
        train_descriptor_head=not args.freeze_descriptor_head,
        train_sparse_context=args.train_sparse_context,
        train_keypoint_head=(
            args.pseudo_keypoint_weight > 0.0
            or args.keypoint_weight > 0.0
            or (args.keypoint_offset_weight > 0.0 and not args.train_keypoint_offset_head_only)
            or (args.selected_keypoint_offset_weight > 0.0 and not args.train_keypoint_offset_head_only)
        ),
        train_keypoint_offset_head=args.train_keypoint_offset_head_only,
        train_geometry_head=(
            args.train_geometry_head
            or args.orientation_consistency_weight > 0.0
            or args.scale_consistency_weight > 0.0
            or args.affine_consistency_weight > 0.0
            or args.affine_regularization_weight > 0.0
        ),
        train_texture_adapter=args.train_texture_adapter,
        train_descriptor_fusion=args.train_descriptor_fusion,
        train_quality_head=args.train_quality_head,
        train_reliability_head=(
            args.train_reliability_head
            or args.matchability_weight > 0.0
            or args.descriptor_uncertainty_weight > 0.0
            or args.no_match_prior_weight > 0.0
        ),
        train_graph_matcher=args.train_graph_matcher,
        train_graph_calibration_only=args.train_graph_calibration_only,
        train_pair_accept_head_only=args.train_pair_accept_head_only,
    )
    if not trainable:
        raise RuntimeError("no trainable parameters selected")
    if not trainable:
        raise RuntimeError("no descriptor parameters selected")
    if should_freeze_non_trainable_batch_norm_statistics(args):
        freeze_non_trainable_batch_norm_statistics(model)
    original_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
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
                "batch",
                "total_batches",
                "total_iterations",
                "epoch",
                "epoch_progress",
                "total_epochs",
                "loss",
                "grad_l2",
                "skipped",
                "amp_enabled",
                "amp_scale",
                "activation_checkpointing",
                "freeze_extractor_warmup_active",
                "descriptor_geometry_safety_schedule",
                "descriptor_geometry_blend_weight",
                "descriptor_scale_log_clamp_min",
                "descriptor_scale_log_clamp_max",
                "teacher_weight",
                "synthetic_loss_weight",
                "keypoint_weight",
                "keypoint_loss",
                "keypoint_points",
                "keypoint_offset_weight",
                "keypoint_offset_loss",
                "keypoint_offset_points",
                "selected_keypoint_offset_weight",
                "selected_keypoint_offset_loss",
                "selected_keypoint_offset_points",
                "selected_keypoint_offset_forward_points",
                "selected_keypoint_offset_reverse_points",
                "hard_negative_weight",
                "diversity_weight",
                "abstention_weight",
                "graph_matcher_loss_weight",
                "graph_matcher_assignment_weight",
                "graph_matcher_accept_weight",
                "graph_matcher_prune_ranking_weight",
                "graph_matcher_stop_confidence_weight",
                "graph_matcher_train_max_attention_layers",
                "graph_matcher_train_random_attention_layers",
                "graph_matcher_train_max_attention_work_fraction",
                "graph_matcher_train_width_keep_ratio",
                "graph_matcher_deep_supervision_depths",
                "graph_matcher_deep_supervision_weight",
                "graph_matcher_depth_distillation_weight",
                "graph_matcher_depth_distillation_target_layers",
                "graph_matcher_depth_distillation_temperature",
                "graph_matcher_teacher_guard_state",
                "graph_matcher_teacher_guard_weight",
                "graph_matcher_teacher_guard_positive_margin_tolerance",
                "graph_matcher_teacher_guard_false_margin_tolerance",
                "graph_matcher_teacher_score_floor_weight",
                "graph_matcher_teacher_score_floor_tolerance",
                "graph_matcher_teacher_score_floor_min_score",
                "graph_matcher_teacher_match_count_floor_weight",
                "graph_matcher_teacher_match_count_floor_threshold",
                "graph_matcher_teacher_match_count_floor_margin",
                "graph_matcher_teacher_match_count_ceiling_weight",
                "graph_matcher_teacher_match_count_ceiling_threshold",
                "graph_matcher_teacher_match_count_ceiling_margin",
                "graph_matcher_teacher_distillation_weight",
                "graph_matcher_teacher_distillation_temperature",
                "graph_matcher_positive_dustbin_guard_reject_threshold",
                "graph_matcher_positive_dustbin_guard_margin_threshold",
                "matcher_reliability_pair_bias",
                "matcher_reliability_dustbin_bias",
                "matcher_final_accept_score_mode",
                "matcher_geometry_bias_scale",
                "matcher_accept_assignment_mode",
                "matcher_final_accept_score_alpha",
                "matcher_geometry_bias_clamp",
                "matcher_attention_residual_gate_init",
                "matcher_attention_residual_gate_start_layer",
                "matcher_candidate_topk",
                "graph_matcher_online_false_no_match",
                "graph_matcher_train_candidate_topk",
                "graph_matcher_dustbin_warmup_steps",
                "graph_matcher_dustbin_ramp_steps",
                "graph_matcher_positive_dustbin_margin_weight",
                "graph_matcher_positive_dustbin_margin",
                "graph_matcher_true_match_margin_weight",
                "graph_matcher_true_match_margin",
                "graph_matcher_true_geometry_match_count_floor_weight",
                "graph_matcher_true_geometry_match_count_floor_threshold",
                "graph_matcher_true_geometry_match_count_floor_margin",
                "graph_matcher_final_false_match_weight",
                "graph_matcher_mined_false_match_weight",
                "graph_matcher_mined_false_match_loss_cap",
                "graph_matcher_mined_false_match_reference_margin",
                "graph_matcher_raw_false_match_weight",
                "graph_matcher_ransac_consistency_weight",
                "graph_matcher_ransac_consistency_topk",
                "graph_matcher_ransac_consistency_residual_threshold_px",
                "graph_matcher_ransac_consistency_min_score",
                "graph_matcher_ransac_consistency_margin",
                "graph_matcher_warp_outlier_weight",
                "graph_matcher_warp_outlier_topk",
                "graph_matcher_warp_outlier_residual_threshold_px",
                "graph_matcher_warp_outlier_min_score",
                "graph_matcher_warp_outlier_margin",
                "graph_matcher_warp_outlier_accept_weight",
                "graph_matcher_warp_outlier_accept_topk",
                "graph_matcher_warp_outlier_accept_residual_threshold_px",
                "graph_matcher_warp_outlier_accept_min_score",
                "graph_matcher_warp_soft_boundary_weight",
                "graph_matcher_warp_soft_boundary_topk",
                "graph_matcher_warp_soft_boundary_lower_residual_px",
                "graph_matcher_warp_soft_boundary_upper_residual_px",
                "graph_matcher_warp_soft_boundary_min_score",
                "graph_matcher_pair_acceptance_loss_weight",
                "matchability_weight",
                "descriptor_uncertainty_weight",
                "no_match_prior_weight",
                "rotation_descriptor_consistency_weight",
                "orientation_consistency_weight",
                "scale_consistency_weight",
                "affine_consistency_weight",
                "affine_regularization_weight",
                "top1_accuracy",
                "top5_accuracy",
                "top10_accuracy",
                "mean_positive_rank",
                "mean_positive_score",
                "mean_negative_score",
                "graph_matcher_total_loss",
                "graph_matcher_ce_loss",
                "graph_matcher_assignment_loss",
                "graph_matcher_no_match_loss",
                "graph_matcher_accept_loss",
                "graph_matcher_prune_ranking_loss",
                "graph_matcher_stop_confidence_loss",
                "graph_matcher_raw_preservation_loss",
                "graph_matcher_hard_negative_dustbin_loss",
                "graph_matcher_positive_dustbin_margin_loss",
                "graph_matcher_true_match_margin_loss",
                "graph_matcher_true_match_margin_violations",
                "graph_matcher_true_match_margin_mean",
                "graph_matcher_true_geometry_match_count_floor_loss",
                "graph_matcher_true_geometry_match_count_floor_target_count",
                "graph_matcher_true_geometry_match_count_floor_student_count",
                "graph_matcher_true_geometry_match_count_floor_count_deficit",
                "graph_matcher_true_geometry_match_count_floor_topk_score_mean",
                "graph_matcher_true_geometry_match_count_floor_violations",
                "graph_matcher_final_false_match_loss",
                "graph_matcher_final_false_match_edges",
                "graph_matcher_final_false_match_score_mean",
                "graph_matcher_final_false_match_accept_mean",
                "graph_matcher_mined_false_match_loss",
                "graph_matcher_mined_false_match_edges",
                "graph_matcher_mined_false_match_reference_filtered_edges",
                "graph_matcher_mined_false_match_score_mean",
                "graph_matcher_mined_false_match_logit_mean",
                "graph_matcher_mined_false_match_accept_mean",
                "graph_matcher_raw_false_match_loss",
                "graph_matcher_raw_false_match_edges",
                "graph_matcher_raw_false_match_similarity_mean",
                "graph_matcher_raw_false_match_margin_mean",
                "graph_matcher_ransac_consistency_loss",
                "graph_matcher_ransac_consistency_edges",
                "graph_matcher_ransac_consistency_score_mean",
                "graph_matcher_ransac_consistency_residual_mean_px",
                "graph_matcher_ransac_consistency_accept_mean",
                "graph_matcher_warp_outlier_loss",
                "graph_matcher_warp_outlier_edges",
                "graph_matcher_warp_outlier_residual_mean_px",
                "graph_matcher_warp_outlier_accept_mean",
                "graph_matcher_warp_outlier_accept_loss",
                "graph_matcher_warp_outlier_accept_edges",
                "graph_matcher_warp_outlier_accept_score_mean",
                "graph_matcher_warp_outlier_accept_residual_mean_px",
                "graph_matcher_warp_outlier_accept_probability_mean",
                "graph_matcher_warp_soft_boundary_loss",
                "graph_matcher_warp_soft_boundary_edges",
                "graph_matcher_warp_soft_boundary_residual_mean_px",
                "graph_matcher_warp_soft_boundary_target_mean",
                "graph_matcher_warp_soft_boundary_score_probability_mean",
                "graph_matcher_warp_soft_boundary_accept_probability_mean",
                "graph_matcher_pair_acceptance_loss",
                "graph_matcher_pair_acceptance_target",
                "graph_matcher_pair_acceptance_weight",
                "graph_matcher_pair_acceptance_probability",
                "graph_matcher_deep_supervision_loss",
                "graph_matcher_depth_distillation_loss",
                "graph_matcher_depth_distillation_teacher_layers",
                "graph_matcher_teacher_distillation_loss",
                "graph_matcher_teacher_guard_loss",
                "graph_matcher_teacher_guard_positive_margin_loss",
                "graph_matcher_teacher_guard_false_edge_loss",
                "graph_matcher_teacher_guard_positive_violations",
                "graph_matcher_teacher_guard_false_edges",
                "graph_matcher_teacher_score_floor_loss",
                "graph_matcher_teacher_score_floor_violations",
                "graph_matcher_teacher_score_floor_delta_mean",
                "graph_matcher_teacher_score_floor_teacher_score_mean",
                "graph_matcher_teacher_match_count_floor_loss",
                "graph_matcher_teacher_match_count_floor_teacher_count",
                "graph_matcher_teacher_match_count_floor_student_count",
                "graph_matcher_teacher_match_count_floor_count_deficit",
                "graph_matcher_teacher_match_count_floor_topk_score_mean",
                "graph_matcher_teacher_match_count_ceiling_loss",
                "graph_matcher_teacher_match_count_ceiling_teacher_count",
                "graph_matcher_teacher_match_count_ceiling_student_count",
                "graph_matcher_teacher_match_count_ceiling_count_excess",
                "graph_matcher_teacher_match_count_ceiling_excess_score_mean",
                "graph_matcher_executed_attention_layers",
                "graph_matcher_attention_work_fraction",
                "graph_matcher_positive_pairs",
                "graph_matcher_extra_no_match_points",
                "graph_matcher_extra_false_match_pairs",
                "graph_matcher_effective_no_match_weight",
                "graph_matcher_effective_hard_negative_dustbin_weight",
                "graph_matcher_dustbin_guard_active",
                "graph_matcher_guarded_no_match_weight",
                "graph_matcher_guarded_hard_negative_dustbin_weight",
                "true_match_rejected_by_dustbin_ratio",
                "positive_pair_logit_mean",
                "positive_dustbin_logit_mean",
                "dustbin_logit_mean",
                "dustbin_logit_for_true_match_mean",
                "positive_vs_dustbin_margin_mean",
                "positive_vs_dustbin_margin_median",
                "positive_vs_dustbin_margin_p10",
                "positive_vs_dustbin_margin_below0_ratio",
                "false_match_accepted_ratio",
                "accept_logit_mean",
                "true_pair_prob_mean",
                "dustbin_prob_for_true_match_mean",
                "true_match_in_topk@64",
                "true_match_in_topk@256",
                *RELIABILITY_LOSS_METRIC_KEYS,
                "points",
                "pseudo_label_points",
                "pseudo_keypoint_points",
                "pseudo_label_pairs",
                "false_match_points",
                "false_match_pairs",
                "online_false_match_points",
                "online_false_match_pairs",
                "illumination_consistency_points",
                "illumination_consistency_pairs",
                "illumination_match_points",
                "illumination_match_pairs",
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
            descriptor_geometry_schedule_progress = (
                0.0 if args.steps <= 1 else float(step - 1) / float(args.steps - 1)
            )
            descriptor_geometry_safety = pfm_model.descriptor_geometry_safety_for_progress(
                args.descriptor_geometry_safety_schedule,
                descriptor_geometry_schedule_progress,
            )
            if descriptor_geometry_safety is not None:
                blend_weight, clamp_min, clamp_max = descriptor_geometry_safety
                model.set_descriptor_geometry_safety(
                    blend_weight=blend_weight,
                    scale_log_clamp_min=clamp_min,
                    scale_log_clamp_max=clamp_max,
                )
            epoch_index = min(int((step - 1) // steps_per_epoch) + 1, max(1, math.ceil(total_epochs)))
            batch_index = ((step - 1) % steps_per_epoch) + 1
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
            freeze_extractor_warmup_active = (
                args.freeze_extractor_warmup_steps > 0 and step <= args.freeze_extractor_warmup_steps
            )
            apply_extractor_freeze_warmup(
                model,
                original_requires_grad,
                freeze_extractor=freeze_extractor_warmup_active,
            )
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
                keypoint_weight=args.keypoint_weight,
                keypoint_negative_weight=args.keypoint_negative_weight,
                keypoint_offset_weight=args.keypoint_offset_weight,
                selected_keypoint_offset_weight=args.selected_keypoint_offset_weight,
                selected_keypoint_offset_max_points=args.selected_keypoint_offset_max_points,
                selected_keypoint_offset_inverse_radius_px=args.selected_keypoint_offset_inverse_radius_px,
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
                online_false_match_max_points=args.false_match_max_points,
                online_false_match_max_score=args.false_match_max_score,
                pose_metadata=pose_metadata,
                pose_balanced_sampling=args.pose_balanced_sampling,
                pose_difficulty_loss_weight=args.pose_difficulty_loss_weight,
                graph_matcher_loss_weight=args.graph_matcher_loss_weight if args.train_graph_matcher else 0.0,
                graph_matcher_metadata_mode=args.graph_matcher_metadata_mode,
                graph_matcher_no_match_points=args.graph_matcher_no_match_points,
                graph_matcher_no_match_weight=args.graph_matcher_no_match_weight,
                graph_matcher_no_match_min_distance=args.graph_matcher_no_match_min_distance,
                graph_matcher_assignment_weight=args.graph_matcher_assignment_weight,
                graph_matcher_accept_weight=args.graph_matcher_accept_weight,
                graph_matcher_accept_negative_topk=args.graph_matcher_accept_negative_topk,
                graph_matcher_prune_ranking_weight=args.graph_matcher_prune_ranking_weight,
                graph_matcher_prune_ranking_margin=args.graph_matcher_prune_ranking_margin,
                graph_matcher_stop_confidence_weight=args.graph_matcher_stop_confidence_weight,
                graph_matcher_stop_confidence_margin=args.graph_matcher_stop_confidence_margin,
                graph_matcher_raw_preservation_weight=args.graph_matcher_raw_preservation_weight,
                graph_matcher_raw_preservation_margin=args.graph_matcher_raw_preservation_margin,
                graph_matcher_raw_preservation_raw_margin=args.graph_matcher_raw_preservation_raw_margin,
                graph_matcher_hard_negative_dustbin_weight=args.graph_matcher_hard_negative_dustbin_weight,
                graph_matcher_hard_negative_dustbin_topk=args.graph_matcher_hard_negative_dustbin_topk,
                graph_matcher_hard_negative_dustbin_margin=args.graph_matcher_hard_negative_dustbin_margin,
                graph_matcher_hard_negative_dustbin_spatial_min_distance=args.graph_matcher_hard_negative_dustbin_spatial_min_distance,
                graph_matcher_dustbin_warmup_steps=args.graph_matcher_dustbin_warmup_steps,
                graph_matcher_dustbin_ramp_steps=args.graph_matcher_dustbin_ramp_steps,
                graph_matcher_positive_dustbin_margin_weight=args.graph_matcher_positive_dustbin_margin_weight,
                graph_matcher_positive_dustbin_margin=args.graph_matcher_positive_dustbin_margin,
                graph_matcher_true_match_margin_weight=args.graph_matcher_true_match_margin_weight,
                graph_matcher_true_match_margin=args.graph_matcher_true_match_margin,
                graph_matcher_true_geometry_match_count_floor_weight=(
                    args.graph_matcher_true_geometry_match_count_floor_weight
                ),
                graph_matcher_true_geometry_match_count_floor_threshold=(
                    args.graph_matcher_true_geometry_match_count_floor_threshold
                ),
                graph_matcher_true_geometry_match_count_floor_margin=(
                    args.graph_matcher_true_geometry_match_count_floor_margin
                ),
                graph_matcher_final_false_match_weight=args.graph_matcher_final_false_match_weight,
                graph_matcher_mined_false_match_weight=args.graph_matcher_mined_false_match_weight,
                graph_matcher_mined_false_match_loss_cap=args.graph_matcher_mined_false_match_loss_cap,
                graph_matcher_mined_false_match_reference_margin=(
                    args.graph_matcher_mined_false_match_reference_margin
                ),
                graph_matcher_final_false_match_topk=args.graph_matcher_final_false_match_topk,
                graph_matcher_final_false_match_min_score=args.graph_matcher_final_false_match_min_score,
                graph_matcher_final_false_match_margin=args.graph_matcher_final_false_match_margin,
                graph_matcher_final_false_match_spatial_min_distance=(
                    args.graph_matcher_final_false_match_spatial_min_distance
                ),
                graph_matcher_raw_false_match_weight=args.graph_matcher_raw_false_match_weight,
                graph_matcher_raw_false_match_topk=args.graph_matcher_raw_false_match_topk,
                graph_matcher_raw_false_match_min_similarity=args.graph_matcher_raw_false_match_min_similarity,
                graph_matcher_raw_false_match_margin=args.graph_matcher_raw_false_match_margin,
                graph_matcher_raw_false_match_spatial_min_distance=(
                    args.graph_matcher_raw_false_match_spatial_min_distance
                ),
                graph_matcher_ransac_consistency_weight=args.graph_matcher_ransac_consistency_weight,
                graph_matcher_ransac_consistency_topk=args.graph_matcher_ransac_consistency_topk,
                graph_matcher_ransac_consistency_residual_threshold_px=(
                    args.graph_matcher_ransac_consistency_residual_threshold_px
                ),
                graph_matcher_ransac_consistency_min_score=args.graph_matcher_ransac_consistency_min_score,
                graph_matcher_ransac_consistency_margin=args.graph_matcher_ransac_consistency_margin,
                graph_matcher_warp_outlier_weight=args.graph_matcher_warp_outlier_weight,
                graph_matcher_warp_outlier_topk=args.graph_matcher_warp_outlier_topk,
                graph_matcher_warp_outlier_residual_threshold_px=(
                    args.graph_matcher_warp_outlier_residual_threshold_px
                ),
                graph_matcher_warp_outlier_min_score=args.graph_matcher_warp_outlier_min_score,
                graph_matcher_warp_outlier_margin=args.graph_matcher_warp_outlier_margin,
                graph_matcher_warp_outlier_accept_weight=args.graph_matcher_warp_outlier_accept_weight,
                graph_matcher_warp_outlier_accept_topk=args.graph_matcher_warp_outlier_accept_topk,
                graph_matcher_warp_outlier_accept_residual_threshold_px=(
                    args.graph_matcher_warp_outlier_accept_residual_threshold_px
                ),
                graph_matcher_warp_outlier_accept_min_score=args.graph_matcher_warp_outlier_accept_min_score,
                graph_matcher_warp_soft_boundary_weight=args.graph_matcher_warp_soft_boundary_weight,
                graph_matcher_warp_soft_boundary_topk=args.graph_matcher_warp_soft_boundary_topk,
                graph_matcher_warp_soft_boundary_lower_residual_px=(
                    args.graph_matcher_warp_soft_boundary_lower_residual_px
                ),
                graph_matcher_warp_soft_boundary_upper_residual_px=(
                    args.graph_matcher_warp_soft_boundary_upper_residual_px
                ),
                graph_matcher_warp_soft_boundary_min_score=args.graph_matcher_warp_soft_boundary_min_score,
                graph_matcher_pair_acceptance_loss_weight=args.graph_matcher_pair_acceptance_loss_weight,
                graph_matcher_train_candidate_topk=args.graph_matcher_train_candidate_topk,
                graph_matcher_semi_dense_no_match_points=args.graph_matcher_semi_dense_no_match_points,
                graph_matcher_semi_dense_min_score=args.graph_matcher_semi_dense_min_score,
                graph_matcher_online_false_no_match=args.graph_matcher_online_false_no_match,
                graph_matcher_train_max_attention_layers=args.graph_matcher_train_max_attention_layers,
                graph_matcher_train_random_attention_layers=args.graph_matcher_train_random_attention_layers,
                graph_matcher_train_max_attention_work_fraction=args.graph_matcher_train_max_attention_work_fraction,
                graph_matcher_train_width_keep_ratio=args.graph_matcher_train_width_keep_ratio,
                graph_matcher_deep_supervision_depths=args.graph_matcher_deep_supervision_depths,
                graph_matcher_deep_supervision_weight=args.graph_matcher_deep_supervision_weight,
                graph_matcher_depth_distillation_weight=args.graph_matcher_depth_distillation_weight,
                graph_matcher_depth_distillation_teacher_layers=args.graph_matcher_depth_distillation_teacher_layers,
                graph_matcher_depth_distillation_temperature=args.graph_matcher_depth_distillation_temperature,
                graph_matcher_teacher_guard_model=graph_matcher_teacher_guard_model,
                graph_matcher_teacher_guard_weight=args.graph_matcher_teacher_guard_weight,
                graph_matcher_teacher_guard_positive_margin_tolerance=(
                    args.graph_matcher_teacher_guard_positive_margin_tolerance
                ),
                graph_matcher_teacher_guard_false_margin_tolerance=(
                    args.graph_matcher_teacher_guard_false_margin_tolerance
                ),
                graph_matcher_teacher_score_floor_weight=args.graph_matcher_teacher_score_floor_weight,
                graph_matcher_teacher_score_floor_tolerance=args.graph_matcher_teacher_score_floor_tolerance,
                graph_matcher_teacher_score_floor_min_score=args.graph_matcher_teacher_score_floor_min_score,
                graph_matcher_teacher_match_count_floor_weight=(
                    args.graph_matcher_teacher_match_count_floor_weight
                ),
                graph_matcher_teacher_match_count_floor_threshold=(
                    args.graph_matcher_teacher_match_count_floor_threshold
                ),
                graph_matcher_teacher_match_count_floor_margin=args.graph_matcher_teacher_match_count_floor_margin,
                graph_matcher_teacher_match_count_ceiling_weight=(
                    args.graph_matcher_teacher_match_count_ceiling_weight
                ),
                graph_matcher_teacher_match_count_ceiling_threshold=(
                    args.graph_matcher_teacher_match_count_ceiling_threshold
                ),
                graph_matcher_teacher_match_count_ceiling_margin=(
                    args.graph_matcher_teacher_match_count_ceiling_margin
                ),
                graph_matcher_teacher_distillation_weight=args.graph_matcher_teacher_distillation_weight,
                graph_matcher_teacher_distillation_temperature=(
                    args.graph_matcher_teacher_distillation_temperature
                ),
                graph_matcher_positive_dustbin_guard_reject_threshold=(
                    args.graph_matcher_positive_dustbin_guard_reject_threshold
                ),
                graph_matcher_positive_dustbin_guard_margin_threshold=(
                    args.graph_matcher_positive_dustbin_guard_margin_threshold
                ),
                matchability_weight=args.matchability_weight,
                descriptor_uncertainty_weight=args.descriptor_uncertainty_weight,
                no_match_prior_weight=args.no_match_prior_weight,
                reliability_negative_points=args.reliability_negative_points,
                reliability_negative_min_distance=args.reliability_negative_min_distance,
                rotation_descriptor_consistency_weight=args.rotation_descriptor_consistency_weight,
                orientation_consistency_weight=args.orientation_consistency_weight,
                scale_consistency_weight=args.scale_consistency_weight,
                affine_consistency_weight=args.affine_consistency_weight,
                affine_regularization_weight=args.affine_regularization_weight,
                rotation_consistency_degrees=args.rotation_consistency_degrees,
                training_spatial_bins=args.training_spatial_bins,
                training_crop_size=args.training_crop_size,
                training_max_image_size=args.training_max_image_size,
                forced_pair_paths=forced_pair_paths,
                prefetched_pairs=prefetched_pairs,
                pair_cache=pair_cache,
                amp_enabled=args.amp,
                amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
                activation_checkpointing=args.activation_checkpointing,
                freeze_extractor_warmup_active=freeze_extractor_warmup_active,
            )
            writer.writerow(
                {
                    "step": step,
                    "batch": batch_index,
                    "total_batches": steps_per_epoch,
                    "total_iterations": args.steps,
                    "epoch": epoch_index,
                    "epoch_progress": epoch_float,
                    "total_epochs": total_epochs,
                    "freeze_extractor_warmup_active": int(freeze_extractor_warmup_active),
                    "descriptor_geometry_safety_schedule": args.descriptor_geometry_safety_schedule,
                    "descriptor_geometry_blend_weight": model.config.descriptor_geometry_blend_weight,
                    "descriptor_scale_log_clamp_min": model.config.descriptor_scale_log_clamp_min,
                    "descriptor_scale_log_clamp_max": model.config.descriptor_scale_log_clamp_max,
                    "teacher_weight": teacher_weight,
                    "synthetic_loss_weight": args.synthetic_loss_weight,
                    "keypoint_weight": args.keypoint_weight,
                    "keypoint_offset_weight": args.keypoint_offset_weight,
                    "selected_keypoint_offset_weight": args.selected_keypoint_offset_weight,
                    "hard_negative_weight": hard_negative_weight,
                    "diversity_weight": diversity_weight,
                    "abstention_weight": args.abstention_weight,
                    "graph_matcher_loss_weight": args.graph_matcher_loss_weight if args.train_graph_matcher else 0.0,
                    "graph_matcher_assignment_weight": args.graph_matcher_assignment_weight
                    if args.train_graph_matcher
                    else 0.0,
                    "graph_matcher_accept_weight": args.graph_matcher_accept_weight if args.train_graph_matcher else 0.0,
                    "graph_matcher_prune_ranking_weight": args.graph_matcher_prune_ranking_weight
                    if args.train_graph_matcher
                    else 0.0,
                    "graph_matcher_stop_confidence_weight": args.graph_matcher_stop_confidence_weight
                    if args.train_graph_matcher
                    else 0.0,
                    "graph_matcher_train_max_attention_layers": args.graph_matcher_train_max_attention_layers
                    if args.train_graph_matcher
                    else 0,
                    "graph_matcher_train_random_attention_layers": int(
                        bool(args.graph_matcher_train_random_attention_layers and args.train_graph_matcher)
                    ),
                    "graph_matcher_train_max_attention_work_fraction": args.graph_matcher_train_max_attention_work_fraction
                    if args.train_graph_matcher
                    else 1.0,
                    "graph_matcher_train_width_keep_ratio": args.graph_matcher_train_width_keep_ratio
                    if args.train_graph_matcher
                    else 1.0,
                    "graph_matcher_deep_supervision_depths": ",".join(
                        str(depth) for depth in args.graph_matcher_deep_supervision_depths
                    )
                    if args.train_graph_matcher
                    else "",
                    "graph_matcher_deep_supervision_weight": args.graph_matcher_deep_supervision_weight
                    if args.train_graph_matcher
                    else 0.0,
                    "graph_matcher_depth_distillation_weight": args.graph_matcher_depth_distillation_weight
                    if args.train_graph_matcher
                    else 0.0,
                    "graph_matcher_depth_distillation_target_layers": args.graph_matcher_depth_distillation_teacher_layers
                    if args.train_graph_matcher
                    else 0,
                    "graph_matcher_depth_distillation_temperature": args.graph_matcher_depth_distillation_temperature
                    if args.train_graph_matcher
                    else 1.0,
                    "graph_matcher_teacher_guard_state": (
                        "" if args.graph_matcher_teacher_guard_state is None else str(args.graph_matcher_teacher_guard_state)
                    ),
                    "graph_matcher_teacher_guard_weight": args.graph_matcher_teacher_guard_weight
                    if args.train_graph_matcher
                    else 0.0,
                    "graph_matcher_teacher_guard_positive_margin_tolerance": (
                        args.graph_matcher_teacher_guard_positive_margin_tolerance
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_teacher_guard_false_margin_tolerance": (
                        args.graph_matcher_teacher_guard_false_margin_tolerance
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_teacher_score_floor_weight": (
                        args.graph_matcher_teacher_score_floor_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_score_floor_tolerance": (
                        args.graph_matcher_teacher_score_floor_tolerance if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_score_floor_min_score": (
                        args.graph_matcher_teacher_score_floor_min_score if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_match_count_floor_weight": (
                        args.graph_matcher_teacher_match_count_floor_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_match_count_floor_threshold": (
                        args.graph_matcher_teacher_match_count_floor_threshold if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_match_count_floor_margin": (
                        args.graph_matcher_teacher_match_count_floor_margin if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_match_count_ceiling_weight": (
                        args.graph_matcher_teacher_match_count_ceiling_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_match_count_ceiling_threshold": (
                        args.graph_matcher_teacher_match_count_ceiling_threshold if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_match_count_ceiling_margin": (
                        args.graph_matcher_teacher_match_count_ceiling_margin if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_distillation_weight": (
                        args.graph_matcher_teacher_distillation_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_teacher_distillation_temperature": (
                        args.graph_matcher_teacher_distillation_temperature if args.train_graph_matcher else 1.0
                    ),
                    "graph_matcher_positive_dustbin_guard_reject_threshold": (
                        args.graph_matcher_positive_dustbin_guard_reject_threshold if args.train_graph_matcher else 1.1
                    ),
                    "graph_matcher_positive_dustbin_guard_margin_threshold": (
                        args.graph_matcher_positive_dustbin_guard_margin_threshold
                        if args.train_graph_matcher
                        else -float("inf")
                    ),
                    "matcher_reliability_pair_bias": args.matcher_reliability_pair_bias,
                    "matcher_reliability_dustbin_bias": args.matcher_reliability_dustbin_bias,
                    "matcher_final_accept_score_mode": args.matcher_final_accept_score_mode,
                    "matcher_geometry_bias_scale": args.matcher_geometry_bias_scale,
                    "matcher_accept_assignment_mode": args.matcher_accept_assignment_mode,
                    "matcher_final_accept_score_alpha": args.matcher_final_accept_score_alpha,
                    "matcher_geometry_bias_clamp": args.matcher_geometry_bias_clamp,
                    "matcher_attention_residual_gate_init": (
                        "" if args.matcher_attention_residual_gate_init is None else args.matcher_attention_residual_gate_init
                    ),
                    "matcher_attention_residual_gate_start_layer": args.matcher_attention_residual_gate_start_layer,
                    "matcher_candidate_topk": args.matcher_candidate_topk,
                    "graph_matcher_online_false_no_match": int(
                        bool(args.graph_matcher_online_false_no_match and args.train_graph_matcher)
                    ),
                    "graph_matcher_train_candidate_topk": args.graph_matcher_train_candidate_topk
                    if args.train_graph_matcher
                    else 0,
                    "graph_matcher_dustbin_warmup_steps": args.graph_matcher_dustbin_warmup_steps
                    if args.train_graph_matcher
                    else 0,
                    "graph_matcher_dustbin_ramp_steps": args.graph_matcher_dustbin_ramp_steps
                    if args.train_graph_matcher
                    else 0,
                    "graph_matcher_positive_dustbin_margin_weight": (
                        args.graph_matcher_positive_dustbin_margin_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_positive_dustbin_margin": (
                        args.graph_matcher_positive_dustbin_margin if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_true_match_margin_weight": (
                        args.graph_matcher_true_match_margin_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_true_match_margin": (
                        args.graph_matcher_true_match_margin if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_true_geometry_match_count_floor_weight": (
                        args.graph_matcher_true_geometry_match_count_floor_weight
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_true_geometry_match_count_floor_threshold": (
                        args.graph_matcher_true_geometry_match_count_floor_threshold
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_true_geometry_match_count_floor_margin": (
                        args.graph_matcher_true_geometry_match_count_floor_margin
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_final_false_match_weight": (
                        args.graph_matcher_final_false_match_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_mined_false_match_weight": (
                        args.graph_matcher_mined_false_match_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_mined_false_match_loss_cap": (
                        args.graph_matcher_mined_false_match_loss_cap if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_mined_false_match_reference_margin": (
                        args.graph_matcher_mined_false_match_reference_margin
                        if args.train_graph_matcher
                        else -1.0
                    ),
                    "graph_matcher_raw_false_match_weight": (
                        args.graph_matcher_raw_false_match_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_ransac_consistency_weight": (
                        args.graph_matcher_ransac_consistency_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_ransac_consistency_topk": (
                        args.graph_matcher_ransac_consistency_topk if args.train_graph_matcher else 0
                    ),
                    "graph_matcher_ransac_consistency_residual_threshold_px": (
                        args.graph_matcher_ransac_consistency_residual_threshold_px
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_ransac_consistency_min_score": (
                        args.graph_matcher_ransac_consistency_min_score if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_ransac_consistency_margin": (
                        args.graph_matcher_ransac_consistency_margin if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_warp_outlier_weight": (
                        args.graph_matcher_warp_outlier_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_warp_outlier_topk": (
                        args.graph_matcher_warp_outlier_topk if args.train_graph_matcher else 0
                    ),
                    "graph_matcher_warp_outlier_residual_threshold_px": (
                        args.graph_matcher_warp_outlier_residual_threshold_px
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_warp_outlier_min_score": (
                        args.graph_matcher_warp_outlier_min_score if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_warp_outlier_margin": (
                        args.graph_matcher_warp_outlier_margin if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_warp_outlier_accept_weight": (
                        args.graph_matcher_warp_outlier_accept_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_warp_outlier_accept_topk": (
                        args.graph_matcher_warp_outlier_accept_topk if args.train_graph_matcher else 0
                    ),
                    "graph_matcher_warp_outlier_accept_residual_threshold_px": (
                        args.graph_matcher_warp_outlier_accept_residual_threshold_px
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_warp_outlier_accept_min_score": (
                        args.graph_matcher_warp_outlier_accept_min_score if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_warp_soft_boundary_weight": (
                        args.graph_matcher_warp_soft_boundary_weight if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_warp_soft_boundary_topk": (
                        args.graph_matcher_warp_soft_boundary_topk if args.train_graph_matcher else 0
                    ),
                    "graph_matcher_warp_soft_boundary_lower_residual_px": (
                        args.graph_matcher_warp_soft_boundary_lower_residual_px
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_warp_soft_boundary_upper_residual_px": (
                        args.graph_matcher_warp_soft_boundary_upper_residual_px
                        if args.train_graph_matcher
                        else 0.0
                    ),
                    "graph_matcher_warp_soft_boundary_min_score": (
                        args.graph_matcher_warp_soft_boundary_min_score if args.train_graph_matcher else 0.0
                    ),
                    "graph_matcher_pair_acceptance_loss_weight": (
                        args.graph_matcher_pair_acceptance_loss_weight if args.train_graph_matcher else 0.0
                    ),
                    "matchability_weight": args.matchability_weight,
                    "descriptor_uncertainty_weight": args.descriptor_uncertainty_weight,
                    "no_match_prior_weight": args.no_match_prior_weight,
                    "rotation_descriptor_consistency_weight": args.rotation_descriptor_consistency_weight,
                    "orientation_consistency_weight": args.orientation_consistency_weight,
                    "scale_consistency_weight": args.scale_consistency_weight,
                    "affine_consistency_weight": args.affine_consistency_weight,
                    "affine_regularization_weight": args.affine_regularization_weight,
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
                    f"kploss={metrics.get('keypoint_loss', 0.0):.6f} "
                    f"kpoff={metrics.get('keypoint_offset_loss', 0.0):.6f} "
                    f"gce={metrics.get('graph_matcher_ce_loss', 0.0):.6f} "
                    f"gassign={metrics.get('graph_matcher_assignment_loss', 0.0):.6f} "
                    f"gnomatch={metrics.get('graph_matcher_no_match_loss', 0.0):.6f} "
                    f"gacc={metrics.get('graph_matcher_accept_loss', 0.0):.6f} "
                    f"gprune={metrics.get('graph_matcher_prune_ranking_loss', 0.0):.6f} "
                    f"gstop={metrics.get('graph_matcher_stop_confidence_loss', 0.0):.6f} "
                    f"matchab={metrics.get('matchability_loss', 0.0):.6f} "
                    f"nomprior={metrics.get('no_match_prior_loss', 0.0):.6f} "
                    f"rotdesc={metrics.get('rotation_descriptor_consistency_loss', 0.0):.6f} "
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
                training_step=step,
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
