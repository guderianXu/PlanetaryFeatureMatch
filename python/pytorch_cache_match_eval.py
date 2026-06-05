#!/usr/bin/env python3
"""Lightweight PyTorch cache matching evaluation for PFM descriptor states."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import pfm_model
import pfm_pytorch_training
from patch_descriptor_training import SyntheticPair, discover_pair_archives, load_libtorch_pair_archive


GRAPH_INFERENCE_PRESETS: dict[str, tuple[float, float]] = {
    "off": (-1.0, -1.0),
    "fast": (0.25, 0.85),
    "high_precision": (0.5, 0.85),
}


def graph_inference_thresholds(
    preset: str,
    graph_width_prune_min_score: float,
    graph_early_stop_min_confidence: float,
) -> tuple[float, float]:
    """解析 LightGlue 风格命名预设，并允许显式数字阈值覆盖。"""

    try:
        width_prune_min_score, early_stop_min_confidence = GRAPH_INFERENCE_PRESETS[preset]
    except KeyError as exc:
        allowed = ", ".join(sorted(GRAPH_INFERENCE_PRESETS))
        raise ValueError(f"graph_inference_preset must be one of: {allowed}") from exc
    if graph_width_prune_min_score < -1.0 or graph_width_prune_min_score > 1.0:
        raise ValueError("graph_width_prune_min_score must be in [-1, 1]")
    if graph_early_stop_min_confidence < -1.0 or graph_early_stop_min_confidence > 1.0:
        raise ValueError("graph_early_stop_min_confidence must be in [-1, 1]")
    if graph_width_prune_min_score > -1.0:
        width_prune_min_score = float(graph_width_prune_min_score)
    if graph_early_stop_min_confidence > -1.0:
        early_stop_min_confidence = float(graph_early_stop_min_confidence)
    return width_prune_min_score, early_stop_min_confidence


@dataclass(frozen=True)
class MatchEvalResult:
    matches: int
    correct: int
    wrong: int
    precision: float
    graph_executed_layers: int = 0
    graph_input_keypoints_a: int = 0
    graph_input_keypoints_b: int = 0
    graph_kept_keypoints_a: int = 0
    graph_kept_keypoints_b: int = 0
    graph_pruned_keypoints_a: int = 0
    graph_pruned_keypoints_b: int = 0
    graph_attention_work_units: int = 0
    graph_full_attention_work_units: int = 0
    graph_attention_work_fraction: float = 0.0


EVAL_CSV_FIELDNAMES = [
    "pair_pt",
    "matches",
    "correct",
    "wrong",
    "precision",
    "graph_executed_layers",
    "graph_input_keypoints_a",
    "graph_input_keypoints_b",
    "graph_kept_keypoints_a",
    "graph_kept_keypoints_b",
    "graph_pruned_keypoints_a",
    "graph_pruned_keypoints_b",
    "graph_attention_work_units",
    "graph_full_attention_work_units",
    "graph_attention_work_fraction",
]


def match_eval_result(
    *,
    matches: int,
    correct: int,
    wrong: int,
    precision: float,
    graph_stats: dict[str, float | int] | None = None,
) -> MatchEvalResult:
    stats = graph_stats or {}
    return MatchEvalResult(
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        graph_executed_layers=int(stats.get("graph_executed_layers", 0)),
        graph_input_keypoints_a=int(stats.get("graph_input_keypoints_a", 0)),
        graph_input_keypoints_b=int(stats.get("graph_input_keypoints_b", 0)),
        graph_kept_keypoints_a=int(stats.get("graph_kept_keypoints_a", 0)),
        graph_kept_keypoints_b=int(stats.get("graph_kept_keypoints_b", 0)),
        graph_pruned_keypoints_a=int(stats.get("graph_pruned_keypoints_a", 0)),
        graph_pruned_keypoints_b=int(stats.get("graph_pruned_keypoints_b", 0)),
        graph_attention_work_units=int(stats.get("graph_attention_work_units", 0)),
        graph_full_attention_work_units=int(stats.get("graph_full_attention_work_units", 0)),
        graph_attention_work_fraction=float(stats.get("graph_attention_work_fraction", 0.0)),
    )


def _feature_to_image_points(
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


def image_texture_scores(image: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    if points_xy.dim() != 2 or points_xy.size(1) != 2:
        raise ValueError("points_xy must have shape Nx2")
    if points_xy.numel() == 0:
        return image.new_empty((0,))
    base = image.to(torch.float32).mean(dim=0, keepdim=True).unsqueeze(0)
    local_mean = F.avg_pool2d(base, kernel_size=5, stride=1, padding=2, count_include_pad=False)
    contrast = (base - local_mean).abs()
    dx = (base - torch.roll(base, shifts=1, dims=3)).abs()
    dy = (base - torch.roll(base, shifts=1, dims=2)).abs()
    texture = contrast + dx + dy
    _, _, height, width = texture.shape
    rounded = points_xy.round().to(torch.long)
    x = rounded[:, 0].clamp(0, width - 1)
    y = rounded[:, 1].clamp(0, height - 1)
    return texture[0, 0, y, x].to(image.device)


def target_texture_gradient_mean(image: torch.Tensor) -> float:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    if image.numel() == 0:
        return 0.0
    base = image.detach().to(dtype=torch.float32).mean(dim=0).cpu().numpy()
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if float(base.max(initial=0.0)) <= 1.5:
        base = base * 255.0
    uint8_image = np.clip(base, 0.0, 255.0).astype(np.uint8, copy=False)
    grad_x = cv2.Sobel(uint8_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(uint8_image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return float(magnitude.mean())


def target_local_contrast_mean(image: torch.Tensor) -> float:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    if image.numel() == 0:
        return 0.0
    base = image.detach().to(dtype=torch.float32).mean(dim=0).cpu().numpy()
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if float(base.max(initial=0.0)) <= 1.5:
        base = base * 255.0
    gray = np.clip(base, 0.0, 255.0).astype(np.float32, copy=False)
    local_mean = cv2.blur(gray, (9, 9))
    local_square_mean = cv2.blur(np.square(gray), (9, 9))
    local_variance = np.maximum(local_square_mean - np.square(local_mean), 0.0)
    return float(np.mean(np.sqrt(local_variance)))


def select_spatially_distributed_indices(
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    *,
    max_keypoints: int,
    spatial_bins: int,
    descriptor_height: int,
    descriptor_width: int,
) -> torch.Tensor:
    if max_keypoints <= 0:
        raise ValueError("max_keypoints must be positive")
    if spatial_bins <= 0:
        raise ValueError("spatial_bins must be positive")
    if keypoints.dim() != 2 or keypoints.size(1) != 2:
        raise ValueError("keypoints must have shape Nx2")
    if scores.dim() != 1 or scores.size(0) != keypoints.size(0):
        raise ValueError("scores must have shape N")
    if keypoints.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=keypoints.device)

    bins = int(spatial_bins)
    x_bin = torch.clamp((keypoints[:, 0] * bins / max(1, descriptor_width)).floor().to(torch.long), 0, bins - 1)
    y_bin = torch.clamp((keypoints[:, 1] * bins / max(1, descriptor_height)).floor().to(torch.long), 0, bins - 1)
    cell_ids = y_bin * bins + x_bin
    chosen: list[torch.Tensor] = []
    for cell_id in range(bins * bins):
        members = torch.nonzero(cell_ids == cell_id, as_tuple=False).reshape(-1)
        if members.numel() == 0:
            continue
        best = members.index_select(0, scores.index_select(0, members).argmax().reshape(1))[0]
        chosen.append(best)
    if not chosen:
        return torch.empty(0, dtype=torch.long, device=keypoints.device)
    selected = torch.stack(chosen)
    order = scores.index_select(0, selected).argsort(descending=True, stable=True)
    selected = selected.index_select(0, order)
    if selected.numel() < min(max_keypoints, keypoints.size(0)):
        used = torch.zeros(keypoints.size(0), dtype=torch.bool, device=keypoints.device)
        used[selected] = True
        remaining = torch.nonzero(~used, as_tuple=False).reshape(-1)
        if remaining.numel() > 0:
            fill_order = scores.index_select(0, remaining).argsort(descending=True, stable=True)
            fill_count = min(max_keypoints - selected.numel(), remaining.numel())
            selected = torch.cat([selected, remaining.index_select(0, fill_order[:fill_count])])
    return selected[:max_keypoints].contiguous()


def apply_keypoint_cell_cap(
    ordered_indices: torch.Tensor,
    *,
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    max_keypoints: int,
    spatial_bins: int,
    descriptor_height: int,
    descriptor_width: int,
    keypoint_cell_cap: int,
) -> torch.Tensor:
    if keypoint_cell_cap <= 0 or spatial_bins <= 0 or ordered_indices.numel() == 0:
        return ordered_indices[:max_keypoints].contiguous()
    bins = int(spatial_bins)
    x_bin = torch.clamp((keypoints[:, 0] * bins / max(1, descriptor_width)).floor().to(torch.long), 0, bins - 1)
    y_bin = torch.clamp((keypoints[:, 1] * bins / max(1, descriptor_height)).floor().to(torch.long), 0, bins - 1)
    cell_ids = y_bin * bins + x_bin
    counts = torch.zeros(bins * bins, dtype=torch.long, device=keypoints.device)
    used = torch.zeros(keypoints.size(0), dtype=torch.bool, device=keypoints.device)
    chosen: list[torch.Tensor] = []
    for index in ordered_indices:
        cell_id = int(cell_ids[index].detach().cpu())
        if int(counts[cell_id]) >= keypoint_cell_cap:
            continue
        counts[cell_id] += 1
        used[index] = True
        chosen.append(index)
        if len(chosen) >= max_keypoints:
            break
    if len(chosen) < max_keypoints:
        remaining = candidate_indices[~used.index_select(0, candidate_indices)]
        if remaining.numel() > 0:
            order = scores.index_select(0, remaining).argsort(descending=True, stable=True)
            for index in remaining.index_select(0, order):
                cell_id = int(cell_ids[index].detach().cpu())
                if int(counts[cell_id]) >= keypoint_cell_cap:
                    continue
                counts[cell_id] += 1
                used[index] = True
                chosen.append(index)
                if len(chosen) >= max_keypoints:
                    break
    if len(chosen) < max_keypoints:
        remaining = candidate_indices[~used.index_select(0, candidate_indices)]
        if remaining.numel() > 0:
            order = scores.index_select(0, remaining).argsort(descending=True, stable=True)
            for index in remaining.index_select(0, order):
                chosen.append(index)
                if len(chosen) >= max_keypoints:
                    break
    if not chosen:
        return ordered_indices[:max_keypoints].contiguous()
    return torch.stack(chosen)[:max_keypoints].contiguous()


def select_descriptor_keypoints(
    image: torch.Tensor,
    descriptors: torch.Tensor,
    *,
    max_keypoints: int,
    min_intensity: float,
    texture_fraction: float = 1.0,
    weak_texture_fraction: float = 0.0,
    spatial_bins: int = 0,
    keypoint_cell_cap: int = 0,
    keypoint_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    if descriptors.dim() != 4 or descriptors.size(0) != 1:
        raise ValueError("descriptors must have shape 1xDxHxW")
    if max_keypoints <= 0:
        raise ValueError("max_keypoints must be positive")
    if texture_fraction < 0.0 or texture_fraction > 1.0:
        raise ValueError("texture_fraction must be in [0, 1]")
    if weak_texture_fraction < 0.0 or weak_texture_fraction > 1.0:
        raise ValueError("weak_texture_fraction must be in [0, 1]")
    if texture_fraction + weak_texture_fraction > 1.0:
        raise ValueError("texture_fraction + weak_texture_fraction must be <= 1")
    if spatial_bins < 0:
        raise ValueError("spatial_bins must be non-negative")
    if keypoint_cell_cap < 0:
        raise ValueError("keypoint_cell_cap must be non-negative")
    _, image_height, image_width = image.shape
    descriptor_height = descriptors.size(2)
    descriptor_width = descriptors.size(3)
    score_map: torch.Tensor | None = None
    if keypoint_scores is not None:
        if keypoint_scores.dim() == 4:
            if keypoint_scores.size(0) != 1 or keypoint_scores.size(1) != 1:
                raise ValueError("keypoint_scores must have shape 1x1xHxW or HxW")
            score_map = keypoint_scores[0, 0].to(descriptors.device, torch.float32)
        elif keypoint_scores.dim() == 2:
            score_map = keypoint_scores.to(descriptors.device, torch.float32)
        else:
            raise ValueError("keypoint_scores must have shape 1x1xHxW or HxW")
        if tuple(score_map.shape) != (descriptor_height, descriptor_width):
            score_map = F.interpolate(
                score_map.view(1, 1, score_map.size(0), score_map.size(1)),
                size=(descriptor_height, descriptor_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
    yy, xx = torch.meshgrid(
        torch.arange(descriptor_height, device=descriptors.device),
        torch.arange(descriptor_width, device=descriptors.device),
        indexing="ij",
    )
    keypoints = torch.stack([xx.to(torch.float32), yy.to(torch.float32)], dim=-1).reshape(-1, 2)
    image_points = _feature_to_image_points(
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
    valid = intensity > min_intensity if min_intensity > 0.0 else torch.ones_like(intensity, dtype=torch.bool)
    selected = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if selected.numel() > max_keypoints:
        texture_count = min(max_keypoints, int(round(float(max_keypoints) * float(texture_fraction))))
        weak_count = min(max_keypoints - texture_count, int(round(float(max_keypoints) * float(weak_texture_fraction))))
        uniform_count = max_keypoints - texture_count - weak_count
        chosen_parts: list[torch.Tensor] = []
        chosen_mask = torch.zeros(keypoints.size(0), dtype=torch.bool, device=keypoints.device)
        scores = score_map.reshape(-1) if score_map is not None else image_texture_scores(image.to(descriptors.device), image_points)
        if texture_count > 0:
            selected_scores = scores.index_select(0, selected)
            selected_keypoints = keypoints.index_select(0, selected)
            if spatial_bins > 0:
                order = select_spatially_distributed_indices(
                    selected_keypoints,
                    selected_scores,
                    max_keypoints=texture_count,
                    spatial_bins=spatial_bins,
                    descriptor_height=descriptor_height,
                    descriptor_width=descriptor_width,
                )
                chosen = selected.index_select(0, order)
                chosen_mask[chosen] = True
                chosen_parts.append(chosen)
            else:
                order = selected_scores.argsort(descending=True, stable=True)[:texture_count]
                chosen = selected.index_select(0, order)
                chosen_mask[chosen] = True
                chosen_parts.append(chosen)
        if weak_count > 0:
            remaining = selected[~chosen_mask.index_select(0, selected)]
            if remaining.numel() > 0:
                weak_scores = (-scores).index_select(0, remaining)
                if spatial_bins > 0:
                    weak_keypoints = keypoints.index_select(0, remaining)
                    order = select_spatially_distributed_indices(
                        weak_keypoints,
                        weak_scores,
                        max_keypoints=weak_count,
                        spatial_bins=spatial_bins,
                        descriptor_height=descriptor_height,
                        descriptor_width=descriptor_width,
                    )
                else:
                    order = weak_scores.argsort(descending=True, stable=True)[:weak_count]
                chosen = remaining.index_select(0, order)
                chosen_mask[chosen] = True
                chosen_parts.append(chosen)
        if uniform_count > 0:
            remaining = selected[~chosen_mask.index_select(0, selected)]
            if remaining.numel() > 0:
                sample = torch.linspace(
                    0,
                    remaining.numel() - 1,
                    steps=min(uniform_count, remaining.numel()),
                    device=remaining.device,
                ).round().to(torch.long)
                chosen_parts.append(remaining.index_select(0, sample))
        selected = torch.cat(chosen_parts) if chosen_parts else selected[:max_keypoints]
        if keypoint_cell_cap > 0:
            selected = apply_keypoint_cell_cap(
                selected,
                keypoints=keypoints,
                scores=scores,
                candidate_indices=torch.nonzero(valid, as_tuple=False).reshape(-1),
                max_keypoints=max_keypoints,
                spatial_bins=spatial_bins,
                descriptor_height=descriptor_height,
                descriptor_width=descriptor_width,
                keypoint_cell_cap=keypoint_cell_cap,
            )
    return keypoints.index_select(0, selected).contiguous(), selected.to(torch.long).contiguous()


def gather_descriptor_rows(descriptors: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
    if descriptors.dim() != 4 or descriptors.size(0) != 1:
        raise ValueError("descriptors must have shape 1xDxHxW")
    flat = descriptors.squeeze(0).permute(1, 2, 0).reshape(-1, descriptors.size(1))
    if selected_indices.numel() == 0:
        return flat.new_empty((0, descriptors.size(1)))
    return flat.index_select(0, selected_indices.to(descriptors.device)).contiguous()


def sample_descriptor_rows_at_keypoints(descriptors: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    if descriptors.dim() != 4 or descriptors.size(0) != 1:
        raise ValueError("descriptors must have shape 1xDxHxW")
    if keypoints.dim() != 2 or keypoints.size(1) != 2:
        raise ValueError("keypoints must have shape Nx2")
    if keypoints.numel() == 0:
        return descriptors.new_empty((0, descriptors.size(1)))
    height, width = descriptors.shape[-2:]
    grid_x = keypoints[:, 0].to(descriptors.device, torch.float32) * (2.0 / float(max(1, width - 1))) - 1.0
    grid_y = keypoints[:, 1].to(descriptors.device, torch.float32) * (2.0 / float(max(1, height - 1))) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(descriptors.to(torch.float32), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).T.contiguous()


def sample_map_rows_at_keypoints(map_tensor: torch.Tensor | None, keypoints: torch.Tensor, *, width: int = 1) -> torch.Tensor | None:
    if map_tensor is None:
        return None
    if map_tensor.dim() != 4 or map_tensor.size(0) != 1:
        raise ValueError("map_tensor must have shape 1xCxHxW")
    sampled = sample_descriptor_rows_at_keypoints(map_tensor, keypoints)
    if sampled.size(1) < width:
        sampled = torch.cat([sampled, sampled.new_zeros((sampled.size(0), width - sampled.size(1)))], dim=1)
    return sampled[:, :width].contiguous()


def cyclic_descriptor_similarity(desc_a: torch.Tensor, desc_b: torch.Tensor) -> torch.Tensor:
    if desc_a.dim() != 2 or desc_b.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if desc_a.size(1) != desc_b.size(1):
        raise ValueError("descriptor dimensions must match")
    desc_a = F.normalize(desc_a.to(torch.float32), p=2, dim=1, eps=1.0e-12)
    desc_b = F.normalize(desc_b.to(torch.float32), p=2, dim=1, eps=1.0e-12)
    channels = desc_a.size(1)
    if channels < 4 or channels % 4 != 0:
        return desc_a @ desc_b.T
    group = channels // 4
    scores = [desc_a @ torch.roll(desc_b, shifts=turns * group, dims=1).T for turns in range(4)]
    return torch.stack(scores, dim=0).max(dim=0).values


def greedy_unique_matches(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    topk: int = 1,
    max_matches: int = 512,
    min_score: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if topk <= 0:
        raise ValueError("topk must be positive")
    if max_matches < 0:
        raise ValueError("max_matches must be nonnegative; use 0 to keep all matches")
    if desc_a.size(0) == 0 or desc_b.size(0) == 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=desc_a.device),
            torch.empty(0, dtype=torch.float32, device=desc_a.device),
        )
    similarity = cyclic_descriptor_similarity(desc_a, desc_b)
    k = min(topk, similarity.size(1))
    top_scores, top_targets = similarity.topk(k, dim=1)
    candidates: list[tuple[float, int, int]] = []
    for source in range(similarity.size(0)):
        for rank in range(k):
            score = float(top_scores[source, rank].detach().cpu())
            if score >= min_score:
                candidates.append((score, source, int(top_targets[source, rank].detach().cpu())))
    candidates.sort(key=lambda row: row[0], reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[list[int]] = []
    scores: list[float] = []
    limit = min(desc_a.size(0), desc_b.size(0)) if max_matches == 0 else max_matches
    for score, source, target in candidates:
        if source in used_a or target in used_b:
            continue
        used_a.add(source)
        used_b.add(target)
        matches.append([source, target])
        scores.append(score)
        if len(matches) >= limit:
            break
    device = desc_a.device
    if not matches:
        return torch.empty(0, 2, dtype=torch.long, device=device), torch.empty(0, dtype=torch.float32, device=device)
    return (
        torch.tensor(matches, dtype=torch.long, device=device),
        torch.tensor(scores, dtype=torch.float32, device=device),
    )


def mutual_nearest_matches(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    max_matches: int = 512,
    min_score: float = -1.0,
    min_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_matches < 0:
        raise ValueError("max_matches must be nonnegative; use 0 to keep all matches")
    if min_margin < 0.0:
        raise ValueError("min_margin must be non-negative")
    if desc_a.size(0) == 0 or desc_b.size(0) == 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=desc_a.device),
            torch.empty(0, dtype=torch.float32, device=desc_a.device),
        )
    similarity = cyclic_descriptor_similarity(desc_a, desc_b)
    best_scores, best_targets = similarity.max(dim=1)
    if min_margin > 0.0 and similarity.size(1) > 1:
        top2 = similarity.topk(2, dim=1).values
        row_margins = top2[:, 0] - top2[:, 1]
    else:
        row_margins = torch.full((similarity.size(0),), float("inf"), dtype=torch.float32, device=similarity.device)
    best_sources = similarity.max(dim=0).indices
    matches: list[list[int]] = []
    scores: list[float] = []
    for source in range(similarity.size(0)):
        target = int(best_targets[source].detach().cpu())
        score = float(best_scores[source].detach().cpu())
        if score < min_score:
            continue
        if float(row_margins[source].detach().cpu()) < min_margin:
            continue
        if int(best_sources[target].detach().cpu()) == source:
            matches.append([source, target])
            scores.append(score)
    limit = len(scores) if max_matches == 0 else max_matches
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:limit]
    device = desc_a.device
    if not order:
        return torch.empty(0, 2, dtype=torch.long, device=device), torch.empty(0, dtype=torch.float32, device=device)
    return (
        torch.tensor([matches[index] for index in order], dtype=torch.long, device=device),
        torch.tensor([scores[index] for index in order], dtype=torch.float32, device=device),
    )


def graph_matcher_matches(
    model: pfm_model.PlanetaryFeatureMatcher,
    desc_a: torch.Tensor,
    keypoints_a: torch.Tensor,
    desc_b: torch.Tensor,
    keypoints_b: torch.Tensor,
    *,
    max_matches: int = 512,
    min_score: float = -1.0,
    graph_dustbin_delta: float = 0.0,
    graph_acceptance_margin: float = 0.0,
    graph_min_raw_score: float = -1.0,
    graph_min_raw_margin: float = 0.0,
    graph_min_accept_probability: float = -1.0,
    graph_width_prune_min_score: float = -1.0,
    graph_early_stop_min_confidence: float = -1.0,
    scores_a: torch.Tensor | None = None,
    scores_b: torch.Tensor | None = None,
    metadata_a: torch.Tensor | None = None,
    metadata_b: torch.Tensor | None = None,
    graph_stats: dict[str, float | int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_matches < 0:
        raise ValueError("max_matches must be nonnegative; use 0 to keep all matches")
    if graph_acceptance_margin < 0.0:
        raise ValueError("graph_acceptance_margin must be non-negative")
    if graph_min_raw_margin < 0.0:
        raise ValueError("graph_min_raw_margin must be non-negative")
    if graph_min_accept_probability < -1.0 or graph_min_accept_probability > 1.0:
        raise ValueError("graph_min_accept_probability must be in [-1, 1]")
    if graph_width_prune_min_score < -1.0:
        raise ValueError("graph_width_prune_min_score must be at least -1.0; -1 disables pruning")
    if graph_early_stop_min_confidence < -1.0:
        raise ValueError("graph_early_stop_min_confidence must be at least -1.0; -1 disables early stopping")
    if desc_a.size(0) == 0 or desc_b.size(0) == 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=desc_a.device),
            torch.empty(0, dtype=torch.float32, device=desc_a.device),
        )
    model_device = next(model.parameters()).device
    meta_dim = int(getattr(getattr(model, "config", None), "graph_keypoint_meta_dim", 2))
    if metadata_a is None:
        meta_a = pfm_model.prepare_graph_keypoint_metadata(
            keypoints_a.to(model_device, torch.float32),
            meta_dim=meta_dim,
            scores=scores_a.to(model_device, torch.float32) if scores_a is not None else None,
            quality=scores_a.to(model_device, torch.float32) if scores_a is not None else None,
        )
    else:
        meta_a = metadata_a.to(model_device, torch.float32)
    if metadata_b is None:
        meta_b = pfm_model.prepare_graph_keypoint_metadata(
            keypoints_b.to(model_device, torch.float32),
            meta_dim=meta_dim,
            scores=scores_b.to(model_device, torch.float32) if scores_b is not None else None,
            quality=scores_b.to(model_device, torch.float32) if scores_b is not None else None,
        )
    else:
        meta_b = metadata_b.to(model_device, torch.float32)
    graph_kwargs = {}
    if graph_width_prune_min_score > -1.0:
        graph_kwargs["width_prune_min_score"] = float(graph_width_prune_min_score)
    if graph_early_stop_min_confidence > -1.0:
        graph_kwargs["early_stop_min_confidence"] = float(graph_early_stop_min_confidence)
    output = model.graph_matcher(
        desc_a.to(model_device, torch.float32),
        meta_a,
        desc_b.to(model_device, torch.float32),
        meta_b,
        **graph_kwargs,
    )
    if graph_stats is not None:
        graph_stats.clear()
        graph_stats.update(
            {
                "graph_executed_layers": int(getattr(output, "executed_layers", 0)),
                "graph_input_keypoints_a": int(getattr(output, "input_keypoints_a", desc_a.size(0))),
                "graph_input_keypoints_b": int(getattr(output, "input_keypoints_b", desc_b.size(0))),
                "graph_kept_keypoints_a": int(getattr(output, "kept_keypoints_a", desc_a.size(0))),
                "graph_kept_keypoints_b": int(getattr(output, "kept_keypoints_b", desc_b.size(0))),
                "graph_pruned_keypoints_a": int(getattr(output, "pruned_keypoints_a", 0)),
                "graph_pruned_keypoints_b": int(getattr(output, "pruned_keypoints_b", 0)),
                "graph_attention_work_units": int(getattr(output, "attention_work_units", 0)),
                "graph_full_attention_work_units": int(getattr(output, "full_attention_work_units", 0)),
                "graph_attention_work_fraction": float(getattr(output, "attention_work_fraction", 0.0)),
            }
        )
    use_calibrated_logits = (
        abs(float(graph_dustbin_delta)) > 0.0
        or float(graph_acceptance_margin) > 0.0
        or float(graph_min_raw_score) > -1.0
        or float(graph_min_raw_margin) > 0.0
    )
    if use_calibrated_logits:
        matches, scores = calibrated_graph_matches_from_logits(
            output.logits,
            count_a=desc_a.size(0),
            count_b=desc_b.size(0),
            dustbin_delta=graph_dustbin_delta,
            acceptance_margin=graph_acceptance_margin,
        )
        matches = matches.to(device=desc_a.device)
        scores = scores.to(device=desc_a.device)
    else:
        matches = output.matches.to(device=desc_a.device)
        scores = output.scores.to(device=desc_a.device)
    if scores.numel() == 0:
        return matches, scores
    keep = scores >= float(min_score)
    if graph_min_raw_score > -1.0 or graph_min_raw_margin > 0.0:
        raw_similarity = cyclic_descriptor_similarity(desc_a, desc_b)
        raw_scores = raw_similarity[matches[:, 0].to(raw_similarity.device), matches[:, 1].to(raw_similarity.device)]
        if graph_min_raw_score > -1.0:
            keep = keep & raw_scores.to(keep.device).ge(float(graph_min_raw_score))
        if graph_min_raw_margin > 0.0:
            if raw_similarity.size(1) > 1:
                top2 = raw_similarity.topk(2, dim=1).values
                raw_margins = top2[:, 0] - top2[:, 1]
            else:
                raw_margins = torch.full(
                    (raw_similarity.size(0),),
                    float("inf"),
                    dtype=torch.float32,
                    device=raw_similarity.device,
                )
            keep = keep & raw_margins.index_select(0, matches[:, 0].to(raw_similarity.device)).to(keep.device).ge(
                float(graph_min_raw_margin)
            )
    if graph_min_accept_probability > -1.0:
        if output.accept_logits is None:
            raise ValueError("graph_min_accept_probability requires graph matcher accept_logits")
        accept_logits = output.accept_logits.to(device=matches.device, dtype=torch.float32)
        accept_prob = torch.sigmoid(accept_logits[matches[:, 0], matches[:, 1]])
        keep = keep & accept_prob.ge(float(graph_min_accept_probability))
    matches = matches[keep]
    scores = scores[keep]
    limit = scores.numel() if max_matches == 0 else max_matches
    order = torch.argsort(scores, descending=True)[:limit]
    return matches.index_select(0, order), scores.index_select(0, order)


def calibrated_graph_matches_from_logits(
    logits: torch.Tensor,
    *,
    count_a: int,
    count_b: int,
    dustbin_delta: float = 0.0,
    acceptance_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count_a < 0 or count_b < 0:
        raise ValueError("count_a and count_b must be non-negative")
    if acceptance_margin < 0.0:
        raise ValueError("acceptance_margin must be non-negative")
    if count_a == 0 or count_b == 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=logits.device),
            torch.empty(0, dtype=torch.float32, device=logits.device),
        )
    if logits.dim() != 2 or logits.size(0) < count_a + 1 or logits.size(1) < count_b + 1:
        raise ValueError("logits must include pair scores plus dustbin row and column")
    adjusted = logits[: count_a + 1, : count_b + 1].to(torch.float32).clone()
    if dustbin_delta != 0.0:
        adjusted[:count_a, count_b] += float(dustbin_delta)
        adjusted[count_a, :count_b] += float(dustbin_delta)
        adjusted[count_a, count_b] += float(dustbin_delta)
    row_logits = adjusted[:count_a, :]
    row_prob = torch.softmax(row_logits, dim=1)[:, :count_b]
    col_prob = torch.softmax(adjusted[:, :count_b], dim=0)[:count_a, :]
    dual_scores = row_prob * col_prob
    best_values, best_indices = dual_scores.max(dim=1)
    dustbin_prob = torch.softmax(row_logits, dim=1)[:, -1]
    inlier_mask = best_values.gt(dustbin_prob + float(acceptance_margin))
    best_sources = dual_scores.max(dim=0).indices
    source_indices = torch.arange(count_a, device=logits.device)
    mutual_mask = best_sources.index_select(0, best_indices).eq(source_indices)
    keep = inlier_mask & mutual_mask
    if not bool(keep.any()):
        return (
            torch.empty(0, 2, dtype=torch.long, device=logits.device),
            torch.empty(0, dtype=torch.float32, device=logits.device),
        )
    kept_sources = source_indices[keep]
    kept_targets = best_indices[keep]
    matches = torch.stack([kept_sources, kept_targets], dim=1).to(torch.long)
    scores = best_values[keep].to(torch.float32)
    order = torch.argsort(scores, descending=True)
    return matches.index_select(0, order), scores.index_select(0, order)


def gather_score_rows(score_map: torch.Tensor | None, selected: torch.Tensor) -> torch.Tensor | None:
    if score_map is None:
        return None
    if score_map.dim() == 4:
        flat = score_map[0, 0].reshape(-1)
    elif score_map.dim() == 2:
        flat = score_map.reshape(-1)
    else:
        raise ValueError("score_map must have shape 1x1xHxW or HxW")
    flat = flat.to(selected.device, torch.float32)
    return flat.index_select(0, selected.to(selected.device))


def graph_metadata_from_raw_features(
    raw: pfm_model.RawFeatureMaps | None,
    keypoints: torch.Tensor,
    *,
    meta_dim: int,
    fallback_scores: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if raw is None:
        return None
    scores = sample_map_rows_at_keypoints(raw.heatmap, keypoints, width=1)
    if scores is None:
        scores = fallback_scores.reshape(-1, 1) if fallback_scores is not None else None
    scale = sample_map_rows_at_keypoints(raw.scale, keypoints, width=1)
    orientation = sample_map_rows_at_keypoints(raw.orientation, keypoints, width=2)
    affine = sample_map_rows_at_keypoints(raw.affine, keypoints, width=4)
    quality = sample_map_rows_at_keypoints(raw.quality, keypoints, width=1)
    local_contrast = sample_map_rows_at_keypoints(raw.local_contrast, keypoints, width=1)
    return pfm_model.prepare_graph_keypoint_metadata(
        keypoints,
        meta_dim=meta_dim,
        scores=scores.squeeze(1) if scores is not None else None,
        scale=scale.squeeze(1) if scale is not None else None,
        orientation=orientation,
        affine=affine,
        quality=quality.squeeze(1) if quality is not None else None,
        local_contrast=local_contrast.squeeze(1) if local_contrast is not None else None,
    )


def _fit_affine(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if source.size(0) < 3:
        return None
    ones = torch.ones(source.size(0), 1, dtype=torch.float32, device=source.device)
    zeros = torch.zeros(source.size(0), 3, dtype=torch.float32, device=source.device)
    base = torch.cat([source.to(torch.float32), ones], dim=1)
    lhs = torch.cat([torch.cat([base, zeros], dim=1), torch.cat([zeros, base], dim=1)], dim=0)
    rhs = torch.cat([target[:, 0], target[:, 1]], dim=0).to(torch.float32)
    try:
        solution = torch.linalg.lstsq(lhs, rhs).solution
    except RuntimeError:
        return None
    if not torch.isfinite(solution).all():
        return None
    return solution.reshape(2, 3)


def _apply_affine(affine: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(points.size(0), 1, dtype=torch.float32, device=points.device)
    homogeneous = torch.cat([points.to(torch.float32), ones], dim=1)
    return homogeneous @ affine.T


def filter_affine_consistent_matches(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    matches: torch.Tensor,
    scores: torch.Tensor,
    *,
    threshold_px: float,
    iterations: int = 128,
    min_inliers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if matches.size(0) < min_inliers or matches.size(0) < 3:
        return matches, scores
    matched_a = points_a.index_select(0, matches[:, 0].to(points_a.device)).to(torch.float32)
    matched_b = points_b.index_select(0, matches[:, 1].to(points_b.device)).to(torch.float32)
    count = matches.size(0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260525)
    candidate_sets: list[torch.Tensor] = []
    if count <= 12:
        for first in range(count - 2):
            for second in range(first + 1, count - 1):
                for third in range(second + 1, count):
                    candidate_sets.append(torch.tensor([first, second, third], dtype=torch.long))
    else:
        score_order = scores.detach().cpu().to(torch.float32).argsort(descending=True)
        if score_order.numel() >= 3:
            candidate_sets.append(score_order[:3].to(torch.long))
        for _ in range(max(0, iterations - len(candidate_sets))):
            candidate_sets.append(torch.randperm(count, generator=generator)[:3].to(torch.long))

    best_keep: torch.Tensor | None = None
    best_score = -1.0
    cpu_scores = scores.detach().cpu().to(torch.float32)
    for candidate in candidate_sets[: max(1, iterations)]:
        sample_a = matched_a.index_select(0, candidate.to(matched_a.device))
        sample_b = matched_b.index_select(0, candidate.to(matched_b.device))
        affine = _fit_affine(sample_a, sample_b)
        if affine is None:
            continue
        predicted = _apply_affine(affine.to(matched_a.device), matched_a)
        residual = (predicted - matched_b).norm(dim=1).detach().cpu()
        keep = residual <= threshold_px
        inliers = int(keep.sum())
        if inliers < min_inliers:
            continue
        score = float(cpu_scores[keep].mean()) + float(inliers)
        if score > best_score:
            best_score = score
            best_keep = keep
    if best_keep is None:
        return (
            torch.empty(0, 2, dtype=torch.long, device=matches.device),
            torch.empty(0, dtype=torch.float32, device=scores.device),
        )
    keep_indices = torch.nonzero(best_keep, as_tuple=False).reshape(-1).to(matches.device)
    kept_matches = matches.index_select(0, keep_indices)
    kept_scores = scores.index_select(0, keep_indices.to(scores.device))
    order = kept_scores.detach().cpu().argsort(descending=True).to(kept_scores.device)
    return kept_matches.index_select(0, order.to(kept_matches.device)), kept_scores.index_select(0, order)


def filter_local_displacement_consistent_matches(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    matches: torch.Tensor,
    scores: torch.Tensor,
    *,
    threshold_px: float,
    neighbors: int = 12,
    min_inliers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if matches.size(0) < min_inliers:
        return matches, scores
    matched_a = points_a.index_select(0, matches[:, 0].to(points_a.device)).to(torch.float32)
    matched_b = points_b.index_select(0, matches[:, 1].to(points_b.device)).to(torch.float32)
    displacement = matched_b - matched_a
    neighbor_count = min(max(2, int(neighbors)), matches.size(0))
    distances = torch.cdist(matched_a, matched_a)
    nearest = distances.argsort(dim=1)[:, :neighbor_count]
    local_displacement = displacement.index_select(0, nearest.reshape(-1))
    local_displacement = local_displacement.reshape(matches.size(0), neighbor_count, 2)
    median = local_displacement.median(dim=1).values
    residual = (displacement - median).norm(dim=1)
    keep = residual <= threshold_px
    if int(keep.sum()) < min_inliers:
        return (
            torch.empty(0, 2, dtype=torch.long, device=matches.device),
            torch.empty(0, dtype=torch.float32, device=scores.device),
        )
    keep_indices = torch.nonzero(keep.detach().cpu(), as_tuple=False).reshape(-1).to(matches.device)
    kept_matches = matches.index_select(0, keep_indices)
    kept_scores = scores.index_select(0, keep_indices.to(scores.device))
    order = kept_scores.detach().cpu().argsort(descending=True).to(kept_scores.device)
    return kept_matches.index_select(0, order.to(kept_matches.device)), kept_scores.index_select(0, order)


def _normalize_xy(points_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = points_xy[:, 0] * (2.0 / float(max(1, width - 1))) - 1.0
    y = points_xy[:, 1] * (2.0 / float(max(1, height - 1))) - 1.0
    return torch.stack([x, y], dim=1)


def sample_warp(warp_a_to_b: torch.Tensor, points_a_xy: torch.Tensor) -> torch.Tensor:
    if warp_a_to_b.dim() != 3 or warp_a_to_b.size(-1) != 2:
        raise ValueError("warp_a_to_b must have shape HxWx2")
    if points_a_xy.numel() == 0:
        return points_a_xy.new_empty((0, 2))
    height, width = warp_a_to_b.shape[:2]
    grid = _normalize_xy(points_a_xy.to(warp_a_to_b.device), height, width).view(1, -1, 1, 2)
    warp = warp_a_to_b.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    sampled = F.grid_sample(warp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).T.contiguous()


def match_pair_descriptor_maps(
    pair: SyntheticPair,
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    *,
    model: pfm_model.PlanetaryFeatureMatcher | None = None,
    matcher_mode: str = "raw_descriptor",
    graph_fallback_mode: str = "mutual",
    max_keypoints: int,
    min_intensity: float,
    threshold_px: float,
    topk: int = 1,
    max_matches: int = 512,
    min_score: float = -1.0,
    min_margin: float = 0.0,
    graph_dustbin_delta: float = 0.0,
    graph_acceptance_margin: float = 0.0,
    graph_min_raw_score: float = -1.0,
    graph_min_raw_margin: float = 0.0,
    graph_min_accept_probability: float = -1.0,
    graph_width_prune_min_score: float = -1.0,
    graph_early_stop_min_confidence: float = -1.0,
    mutual: bool = False,
    geometry_filter: str = "none",
    texture_fraction: float = 1.0,
    weak_texture_fraction: float = 0.0,
    keypoint_spatial_bins: int = 0,
    keypoint_cell_cap: int = 0,
    keypoint_scores_a: torch.Tensor | None = None,
    keypoint_scores_b: torch.Tensor | None = None,
    raw_features_a: pfm_model.RawFeatureMaps | None = None,
    raw_features_b: pfm_model.RawFeatureMaps | None = None,
) -> MatchEvalResult:
    if graph_fallback_mode not in {"mutual", "none"}:
        raise ValueError("graph_fallback_mode must be mutual or none")
    graph_stats: dict[str, float | int] = {}
    keypoints_a, selected_a = select_descriptor_keypoints(
        pair.view_a,
        descriptors_a,
        max_keypoints=max_keypoints,
        min_intensity=min_intensity,
        texture_fraction=texture_fraction,
        weak_texture_fraction=weak_texture_fraction,
        spatial_bins=keypoint_spatial_bins,
        keypoint_cell_cap=keypoint_cell_cap,
        keypoint_scores=keypoint_scores_a,
    )
    keypoints_b, selected_b = select_descriptor_keypoints(
        pair.view_b,
        descriptors_b,
        max_keypoints=max_keypoints,
        min_intensity=min_intensity,
        texture_fraction=texture_fraction,
        weak_texture_fraction=weak_texture_fraction,
        spatial_bins=keypoint_spatial_bins,
        keypoint_cell_cap=keypoint_cell_cap,
        keypoint_scores=keypoint_scores_b,
    )
    rows_a = gather_descriptor_rows(descriptors_a, selected_a)
    rows_b = gather_descriptor_rows(descriptors_b, selected_b)
    row_scores_a = gather_score_rows(keypoint_scores_a, selected_a)
    row_scores_b = gather_score_rows(keypoint_scores_b, selected_b)
    if matcher_mode == "graph_matcher":
        if model is None:
            raise ValueError("model is required for graph_matcher mode")
        if hasattr(model, "semi_dense_branch") and raw_features_a is not None and raw_features_b is not None:
            semi_dense = model.semi_dense_branch(
                descriptors_a,
                descriptors_b,
                max_candidates=max(1, max_matches // 2),
                min_score=max(0.0, min_score),
            )
            if semi_dense.scores.numel() > 0:
                keypoints_a = torch.cat([keypoints_a, semi_dense.keypoints_a.to(keypoints_a.device)], dim=0)
                keypoints_b = torch.cat([keypoints_b, semi_dense.keypoints_b.to(keypoints_b.device)], dim=0)
                rows_a = torch.cat(
                    [rows_a, sample_descriptor_rows_at_keypoints(descriptors_a, semi_dense.keypoints_a.to(descriptors_a.device))],
                    dim=0,
                )
                rows_b = torch.cat(
                    [rows_b, sample_descriptor_rows_at_keypoints(descriptors_b, semi_dense.keypoints_b.to(descriptors_b.device))],
                    dim=0,
                )
                row_scores_a = (
                    torch.cat([row_scores_a, semi_dense.scores.to(row_scores_a.device)], dim=0)
                    if row_scores_a is not None
                    else None
                )
                row_scores_b = (
                    torch.cat([row_scores_b, semi_dense.scores.to(row_scores_b.device)], dim=0)
                    if row_scores_b is not None
                    else None
                )
        metadata_a = graph_metadata_from_raw_features(
            raw_features_a,
            keypoints_a,
            meta_dim=getattr(getattr(model, "config", None), "graph_keypoint_meta_dim", 2),
            fallback_scores=row_scores_a,
        )
        metadata_b = graph_metadata_from_raw_features(
            raw_features_b,
            keypoints_b,
            meta_dim=getattr(getattr(model, "config", None), "graph_keypoint_meta_dim", 2),
            fallback_scores=row_scores_b,
        )
        matches, _ = graph_matcher_matches(
            model,
            rows_a,
            keypoints_a,
            rows_b,
            keypoints_b,
            max_matches=max_matches,
            min_score=min_score,
            graph_dustbin_delta=graph_dustbin_delta,
            graph_acceptance_margin=graph_acceptance_margin,
            graph_min_raw_score=graph_min_raw_score,
            graph_min_raw_margin=graph_min_raw_margin,
            graph_min_accept_probability=graph_min_accept_probability,
            graph_width_prune_min_score=graph_width_prune_min_score,
            graph_early_stop_min_confidence=graph_early_stop_min_confidence,
            scores_a=row_scores_a,
            scores_b=row_scores_b,
            metadata_a=metadata_a,
            metadata_b=metadata_b,
            graph_stats=graph_stats,
        )
        if (
            graph_fallback_mode == "mutual"
            and matches.size(0) < max_matches
            and rows_a.size(0) > 0
            and rows_b.size(0) > 0
        ):
            fallback_matches, fallback_scores = mutual_nearest_matches(
                rows_a,
                rows_b,
                max_matches=max_matches,
                min_score=min_score,
                min_margin=min_margin,
            )
            if fallback_matches.numel() > 0:
                seen = {(int(a), int(b)) for a, b in matches.detach().cpu().tolist()}
                additions = []
                addition_scores = []
                for index, (source, target) in enumerate(fallback_matches.detach().cpu().tolist()):
                    key = (int(source), int(target))
                    if key in seen:
                        continue
                    seen.add(key)
                    additions.append([source, target])
                    addition_scores.append(float(fallback_scores[index].detach().cpu()))
                    if len(additions) + matches.size(0) >= max_matches:
                        break
                if additions:
                    add_matches = torch.tensor(additions, dtype=torch.long, device=matches.device)
                    matches = torch.cat([matches, add_matches], dim=0)
    elif matcher_mode == "raw_descriptor" and mutual:
        matches, _ = mutual_nearest_matches(
            rows_a,
            rows_b,
            max_matches=max_matches,
            min_score=min_score,
            min_margin=min_margin,
        )
    elif matcher_mode == "raw_descriptor":
        matches, _ = greedy_unique_matches(
            rows_a,
            rows_b,
            topk=topk,
            max_matches=max_matches,
            min_score=min_score,
        )
    else:
        raise ValueError(f"unsupported matcher_mode: {matcher_mode}")
    if matches.numel() == 0:
        return match_eval_result(matches=0, correct=0, wrong=0, precision=0.0, graph_stats=graph_stats)

    _, image_height_a, image_width_a = pair.view_a.shape
    _, image_height_b, image_width_b = pair.view_b.shape
    points_a = _feature_to_image_points(
        keypoints_a.index_select(0, matches[:, 0].to(keypoints_a.device)),
        feature_height=descriptors_a.size(2),
        feature_width=descriptors_a.size(3),
        image_height=image_height_a,
        image_width=image_width_a,
    )
    points_b = _feature_to_image_points(
        keypoints_b.index_select(0, matches[:, 1].to(keypoints_b.device)),
        feature_height=descriptors_b.size(2),
        feature_width=descriptors_b.size(3),
        image_height=image_height_b,
        image_width=image_width_b,
    )
    if geometry_filter == "affine":
        local_indices = torch.arange(matches.size(0), dtype=torch.long, device=matches.device)
        local_matches = torch.stack([local_indices, local_indices], dim=1)
        local_scores = torch.arange(matches.size(0), 0, -1, dtype=torch.float32, device=matches.device)
        kept_local, _ = filter_affine_consistent_matches(
            points_a,
            points_b,
            local_matches,
            local_scores,
            threshold_px=threshold_px,
            min_inliers=4,
        )
        if kept_local.numel() == 0:
            return match_eval_result(matches=0, correct=0, wrong=0, precision=0.0, graph_stats=graph_stats)
        keep = kept_local[:, 0].to(points_a.device)
        matches = matches.index_select(0, keep.to(matches.device))
        points_a = points_a.index_select(0, keep)
        points_b = points_b.index_select(0, keep.to(points_b.device))
    elif geometry_filter == "local":
        local_indices = torch.arange(matches.size(0), dtype=torch.long, device=matches.device)
        local_matches = torch.stack([local_indices, local_indices], dim=1)
        local_scores = torch.arange(matches.size(0), 0, -1, dtype=torch.float32, device=matches.device)
        kept_local, _ = filter_local_displacement_consistent_matches(
            points_a,
            points_b,
            local_matches,
            local_scores,
            threshold_px=threshold_px,
            min_inliers=4,
        )
        if kept_local.numel() == 0:
            return match_eval_result(matches=0, correct=0, wrong=0, precision=0.0, graph_stats=graph_stats)
        keep = kept_local[:, 0].to(points_a.device)
        matches = matches.index_select(0, keep.to(matches.device))
        points_a = points_a.index_select(0, keep)
        points_b = points_b.index_select(0, keep.to(points_b.device))
    elif geometry_filter != "none":
        raise ValueError(f"unsupported geometry filter: {geometry_filter}")
    target_b = sample_warp(pair.warp_a_to_b, points_a)
    errors = (target_b.to(points_b.device) - points_b).norm(dim=1)
    correct = int(errors.le(threshold_px).sum().detach().cpu())
    total = int(matches.size(0))
    wrong = total - correct
    precision = 0.0 if total == 0 else correct / total
    return match_eval_result(matches=total, correct=correct, wrong=wrong, precision=precision, graph_stats=graph_stats)


def descriptor_maps_for_pair(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair: SyntheticPair,
    *,
    mode: str,
    texture_blend_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_a = pair.view_a.unsqueeze(0)
    image_b = pair.view_b.unsqueeze(0)
    if mode == "learned":
        return model.learned_descriptor_map_single(image_a), model.learned_descriptor_map_single(image_b)
    if mode == "texture":
        return model.texture_descriptor_map_single(image_a), model.texture_descriptor_map_single(image_b)
    if mode == "blend":
        return (
            model.descriptor_map_single(image_a, texture_blend_weight=texture_blend_weight),
            model.descriptor_map_single(image_b, texture_blend_weight=texture_blend_weight),
        )
    raise ValueError(f"unsupported descriptor mode: {mode}")


def descriptor_maps_and_keypoint_scores_for_pair(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair: SyntheticPair,
    *,
    mode: str,
    texture_blend_weight: float,
    keypoint_score_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if keypoint_score_mode == "texture":
        descriptors_a, descriptors_b = descriptor_maps_for_pair(
            model,
            pair,
            mode=mode,
            texture_blend_weight=texture_blend_weight,
        )
        return descriptors_a, descriptors_b, None, None
    if keypoint_score_mode != "learned":
        raise ValueError(f"unsupported keypoint score mode: {keypoint_score_mode}")

    def single(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_batch = image.unsqueeze(0)
        raw = model.forward_single(image_batch, texture_blend_weight=texture_blend_weight)
        if mode == "learned":
            descriptors = model.learned_descriptor_map_single(image_batch)
        elif mode == "texture":
            descriptors = model.texture_descriptor_map_single(image_batch)
        elif mode == "blend":
            descriptors = raw.descriptors
        else:
            raise ValueError(f"unsupported descriptor mode: {mode}")
        return descriptors, raw.heatmap

    descriptors_a, keypoint_scores_a = single(pair.view_a)
    descriptors_b, keypoint_scores_b = single(pair.view_b)
    return descriptors_a, descriptors_b, keypoint_scores_a, keypoint_scores_b


def feature_maps_and_keypoint_scores_for_pair(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair: SyntheticPair,
    *,
    mode: str,
    texture_blend_weight: float,
    keypoint_score_mode: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    pfm_model.RawFeatureMaps | None,
    pfm_model.RawFeatureMaps | None,
]:
    if keypoint_score_mode != "learned" and mode != "blend":
        descriptors_a, descriptors_b, scores_a, scores_b = descriptor_maps_and_keypoint_scores_for_pair(
            model,
            pair,
            mode=mode,
            texture_blend_weight=texture_blend_weight,
            keypoint_score_mode=keypoint_score_mode,
        )
        return descriptors_a, descriptors_b, scores_a, scores_b, None, None

    def single(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, pfm_model.RawFeatureMaps]:
        image_batch = image.unsqueeze(0)
        raw = model.forward_single(image_batch, texture_blend_weight=texture_blend_weight)
        if mode == "blend":
            descriptors = raw.descriptors
        elif mode == "learned":
            descriptors = model.learned_descriptor_map_single(image_batch)
        elif mode == "texture":
            descriptors = model.texture_descriptor_map_single(image_batch)
        else:
            raise ValueError(f"unsupported descriptor mode: {mode}")
        score = raw.heatmap if keypoint_score_mode == "learned" else None
        return descriptors, score, raw

    descriptors_a, scores_a, raw_a = single(pair.view_a)
    descriptors_b, scores_b, raw_b = single(pair.view_b)
    return descriptors_a, descriptors_b, scores_a, scores_b, raw_a, raw_b


def evaluate_pair_path(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair_path: Path,
    *,
    device: torch.device,
    mode: str,
    texture_blend_weight: float,
    max_keypoints: int,
    min_intensity: float,
    texture_fraction: float,
    threshold_px: float,
    topk: int,
    max_matches: int,
    min_score: float,
    min_margin: float,
    min_target_gradient: float,
    min_target_local_contrast: float,
    mutual: bool,
    geometry_filter: str,
    keypoint_spatial_bins: int,
    weak_texture_fraction: float = 0.0,
    keypoint_cell_cap: int = 0,
    keypoint_score_mode: str = "texture",
    matcher_mode: str = "raw_descriptor",
    graph_fallback_mode: str = "mutual",
    graph_dustbin_delta: float = 0.0,
    graph_acceptance_margin: float = 0.0,
    graph_min_raw_score: float = -1.0,
    graph_min_raw_margin: float = 0.0,
    graph_min_accept_probability: float = -1.0,
    graph_width_prune_min_score: float = -1.0,
    graph_early_stop_min_confidence: float = -1.0,
) -> MatchEvalResult:
    if min_target_gradient < 0.0:
        raise ValueError("min_target_gradient must be non-negative")
    if min_target_local_contrast < 0.0:
        raise ValueError("min_target_local_contrast must be non-negative")
    pair = load_libtorch_pair_archive(pair_path, device=device)
    if min_target_gradient > 0.0 and target_texture_gradient_mean(pair.view_b) < min_target_gradient:
        return MatchEvalResult(matches=0, correct=0, wrong=0, precision=0.0)
    if min_target_local_contrast > 0.0 and target_local_contrast_mean(pair.view_b) < min_target_local_contrast:
        return MatchEvalResult(matches=0, correct=0, wrong=0, precision=0.0)
    with torch.no_grad():
        (
            descriptors_a,
            descriptors_b,
            keypoint_scores_a,
            keypoint_scores_b,
            raw_features_a,
            raw_features_b,
        ) = feature_maps_and_keypoint_scores_for_pair(
            model,
            pair,
            mode=mode,
            texture_blend_weight=texture_blend_weight,
            keypoint_score_mode=keypoint_score_mode,
        )
        return match_pair_descriptor_maps(
            pair,
            descriptors_a,
            descriptors_b,
            model=model,
            matcher_mode=matcher_mode,
            graph_fallback_mode=graph_fallback_mode,
            max_keypoints=max_keypoints,
            min_intensity=min_intensity,
            texture_fraction=texture_fraction,
            weak_texture_fraction=weak_texture_fraction,
            threshold_px=threshold_px,
            topk=topk,
            max_matches=max_matches,
            min_score=min_score,
            min_margin=min_margin,
            graph_dustbin_delta=graph_dustbin_delta,
            graph_acceptance_margin=graph_acceptance_margin,
            graph_min_raw_score=graph_min_raw_score,
            graph_min_raw_margin=graph_min_raw_margin,
            graph_min_accept_probability=graph_min_accept_probability,
            graph_width_prune_min_score=graph_width_prune_min_score,
            graph_early_stop_min_confidence=graph_early_stop_min_confidence,
            mutual=mutual,
            geometry_filter=geometry_filter,
            keypoint_spatial_bins=keypoint_spatial_bins,
            keypoint_cell_cap=keypoint_cell_cap,
            keypoint_scores_a=keypoint_scores_a,
            keypoint_scores_b=keypoint_scores_b,
            raw_features_a=raw_features_a,
            raw_features_b=raw_features_b,
        )


def summarize(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "pairs=0"
    total = sum(int(row["matches"]) for row in rows)
    correct = sum(int(row["correct"]) for row in rows)
    precision = 0.0 if total == 0 else correct / total
    low = sum(1 for row in rows if float(row["precision"]) < 0.9)
    return f"pairs={len(rows)} total_matches={total} correct={correct} precision={precision:.6f} low_precision_pairs={low}"


def limit_pair_paths(pair_paths: list[Path], *, limit_pairs: int, sample_seed: int | None = None) -> list[Path]:
    paths = sorted(dict.fromkeys(pair_paths))
    if limit_pairs <= 0 or len(paths) <= limit_pairs:
        return paths
    if sample_seed is None:
        return paths[:limit_pairs]
    return sorted(random.Random(sample_seed).sample(paths, limit_pairs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch PFM descriptor states on synthetic pair caches")
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--pytorch-state", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["learned", "texture", "blend"], default="blend")
    parser.add_argument("--texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--texture-keypoint-fraction", type=float, default=1.0)
    parser.add_argument("--weak-texture-keypoint-fraction", type=float, default=0.0)
    parser.add_argument("--keypoint-spatial-bins", type=int, default=0)
    parser.add_argument("--keypoint-cell-cap", type=int, default=0)
    parser.add_argument("--keypoint-score-mode", choices=["texture", "learned"], default="texture")
    parser.add_argument("--matcher-mode", choices=["raw_descriptor", "graph_matcher"], default="raw_descriptor")
    parser.add_argument("--graph-fallback-mode", choices=["mutual", "none"], default="mutual")
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--descriptor-topk", type=int, default=1)
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--geometry-filter", choices=["none", "affine", "local"], default="none")
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--graph-dustbin-delta", type=float, default=0.0)
    parser.add_argument("--graph-acceptance-margin", type=float, default=0.0)
    parser.add_argument("--graph-min-raw-score", type=float, default=-1.0)
    parser.add_argument("--graph-min-raw-margin", type=float, default=0.0)
    parser.add_argument("--graph-min-accept-probability", type=float, default=-1.0)
    parser.add_argument("--graph-inference-preset", choices=sorted(GRAPH_INFERENCE_PRESETS), default="off")
    parser.add_argument("--graph-width-prune-min-score", type=float, default=-1.0)
    parser.add_argument("--graph-early-stop-min-confidence", type=float, default=-1.0)
    parser.add_argument("--min-target-gradient", type=float, default=0.0)
    parser.add_argument("--min-target-local-contrast", type=float, default=0.0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--exclude-self-pairs", action="store_true")
    parser.add_argument("--hard-summary", action="append", type=Path, default=[])
    parser.add_argument("--hard-limit", type=int, default=64)
    parser.add_argument("--hard-min-matches", type=int, default=4)
    parser.add_argument("--hard-max-precision", type=float, default=0.9)
    args = parser.parse_args()
    if args.graph_min_accept_probability < -1.0 or args.graph_min_accept_probability > 1.0:
        parser.error("--graph-min-accept-probability must be in [-1, 1]")
    return args


def load_model(args: argparse.Namespace) -> pfm_model.PlanetaryFeatureMatcher:
    if args.pytorch_state is None and args.checkpoint is None:
        raise ValueError("either --pytorch-state or --checkpoint is required")
    if args.pytorch_state is not None and args.checkpoint is not None:
        raise ValueError("use only one of --pytorch-state or --checkpoint")
    if args.pytorch_state is not None:
        model, _ = pfm_model.load_pytorch_state(args.pytorch_state, device=args.device)
        return model
    model, _ = pfm_model.load_libtorch_checkpoint(args.checkpoint, device=args.device)
    return model


def selected_pair_paths(args: argparse.Namespace) -> list[Path]:
    sample_seed = getattr(args, "sample_seed", None)
    discover_limit = 0 if sample_seed is not None else args.limit_pairs
    pair_paths = discover_pair_archives(
        args.cache_dir,
        limit_pairs=discover_limit,
        exclude_self_pairs=getattr(args, "exclude_self_pairs", False),
    )
    if not args.hard_summary:
        return limit_pair_paths(pair_paths, limit_pairs=args.limit_pairs, sample_seed=sample_seed)
    selected = pfm_pytorch_training.select_hard_training_pairs(
        pair_paths,
        args.hard_summary,
        limit=args.hard_limit,
        min_matches=args.hard_min_matches,
        max_precision=args.hard_max_precision,
    )
    return limit_pair_paths(selected, limit_pairs=args.limit_pairs, sample_seed=sample_seed)


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    args.graph_width_prune_min_score, args.graph_early_stop_min_confidence = graph_inference_thresholds(
        args.graph_inference_preset,
        args.graph_width_prune_min_score,
        args.graph_early_stop_min_confidence,
    )
    model = load_model(args)
    model.eval()
    device = torch.device(args.device)
    pairs = selected_pair_paths(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_CSV_FIELDNAMES)
        writer.writeheader()
        for index, pair_path in enumerate(pairs, 1):
            result = evaluate_pair_path(
                model,
                pair_path,
                device=device,
                mode=args.mode,
                texture_blend_weight=args.texture_blend_weight,
                max_keypoints=args.max_keypoints,
                min_intensity=args.min_intensity,
                texture_fraction=args.texture_keypoint_fraction,
                weak_texture_fraction=args.weak_texture_keypoint_fraction,
                threshold_px=args.threshold_px,
                topk=args.descriptor_topk,
                max_matches=args.max_matches,
                min_score=args.min_score,
                min_margin=args.min_margin,
                min_target_gradient=args.min_target_gradient,
                min_target_local_contrast=args.min_target_local_contrast,
                mutual=args.mutual,
                geometry_filter=args.geometry_filter,
                keypoint_spatial_bins=args.keypoint_spatial_bins,
                keypoint_cell_cap=args.keypoint_cell_cap,
                keypoint_score_mode=args.keypoint_score_mode,
                matcher_mode=args.matcher_mode,
                graph_fallback_mode=args.graph_fallback_mode,
                graph_dustbin_delta=args.graph_dustbin_delta,
                graph_acceptance_margin=args.graph_acceptance_margin,
                graph_min_raw_score=args.graph_min_raw_score,
                graph_min_raw_margin=args.graph_min_raw_margin,
                graph_min_accept_probability=args.graph_min_accept_probability,
                graph_width_prune_min_score=args.graph_width_prune_min_score,
                graph_early_stop_min_confidence=args.graph_early_stop_min_confidence,
            )
            row = {
                "pair_pt": pair_path.as_posix(),
                "matches": str(result.matches),
                "correct": str(result.correct),
                "wrong": str(result.wrong),
                "precision": f"{result.precision:.6f}",
                "graph_executed_layers": str(result.graph_executed_layers),
                "graph_input_keypoints_a": str(result.graph_input_keypoints_a),
                "graph_input_keypoints_b": str(result.graph_input_keypoints_b),
                "graph_kept_keypoints_a": str(result.graph_kept_keypoints_a),
                "graph_kept_keypoints_b": str(result.graph_kept_keypoints_b),
                "graph_pruned_keypoints_a": str(result.graph_pruned_keypoints_a),
                "graph_pruned_keypoints_b": str(result.graph_pruned_keypoints_b),
                "graph_attention_work_units": str(result.graph_attention_work_units),
                "graph_full_attention_work_units": str(result.graph_full_attention_work_units),
                "graph_attention_work_fraction": f"{result.graph_attention_work_fraction:.6f}",
            }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(
                f"[{index}/{len(pairs)}] {pair_path} matches={result.matches} "
                f"correct={result.correct} precision={result.precision:.6f}",
                flush=True,
            )
    print(summarize(rows))
    print(f"summary={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
