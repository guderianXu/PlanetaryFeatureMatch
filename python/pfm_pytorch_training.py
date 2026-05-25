#!/usr/bin/env python3
"""PyTorch fine-tuning loop for the current PFM feature extractor."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import hard_pair_mining
import pfm_model
from patch_descriptor_training import (
    SyntheticPair,
    discover_pair_archives,
    load_libtorch_pair_archive,
    paired_descriptor_loss,
    paired_descriptor_metrics,
)


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


def sample_feature_correspondences(
    pair: SyntheticPair,
    *,
    feature_height: int,
    feature_width: int,
    count: int,
    min_intensity: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("count must be positive")
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("feature size must be positive")
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
    if teacher_descriptors_a is not None and teacher_descriptors_b is not None and teacher_weight > 0.0:
        teacher_a = normalize_descriptor_batch(sample_descriptors(teacher_descriptors_a, points_a_xy))
        teacher_b = normalize_descriptor_batch(sample_descriptors(teacher_descriptors_b, points_b_xy))
        teacher_loss = 0.5 * (
            teacher_guided_descriptor_loss(desc_a, teacher_b, temperature=temperature)
            + teacher_guided_descriptor_loss(desc_b, teacher_a, temperature=temperature)
        )
        loss = loss + teacher_weight * teacher_loss
    return loss, paired_descriptor_metrics(desc_a, desc_b)


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


def compute_descriptor_maps(model, pair: SyntheticPair) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        model.descriptor_map_single(pair.view_a.unsqueeze(0)),
        model.descriptor_map_single(pair.view_b.unsqueeze(0)),
    )


def compute_student_teacher_descriptor_maps(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair: SyntheticPair,
    *,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_blended_descriptors:
        student_a = model.descriptor_map_single(pair.view_a.unsqueeze(0), texture_blend_weight=texture_blend_weight)
        student_b = model.descriptor_map_single(pair.view_b.unsqueeze(0), texture_blend_weight=texture_blend_weight)
    else:
        student_a = model.learned_descriptor_map_single(pair.view_a.unsqueeze(0))
        student_b = model.learned_descriptor_map_single(pair.view_b.unsqueeze(0))
    with torch.no_grad():
        teacher_a = model.texture_descriptor_map_single(pair.view_a.unsqueeze(0))
        teacher_b = model.texture_descriptor_map_single(pair.view_b.unsqueeze(0))
    return student_a, student_b, teacher_a, teacher_b


def descriptor_parameters(
    model: pfm_model.PlanetaryFeatureMatcher,
    *,
    train_sparse_context: bool = False,
) -> list[torch.nn.Parameter]:
    selected: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith("sparse_head.descriptor") or name.startswith("sparse_head.descriptors")
        trainable = trainable or (train_sparse_context and name.startswith("sparse_head.context"))
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
    temperature: float,
    teacher_weight: float,
    hard_pair_paths: list[Path] | None = None,
    hard_probability: float = 0.0,
    hard_negative_weight: float = 0.5,
    diversity_weight: float = 0.10,
    warp_hard_negative_weight: float = 0.0,
    warp_hard_negative_radius: float = 2.0,
    warp_hard_negative_margin: float = 0.2,
    warp_hard_negative_candidates: int = 4096,
    max_grad_norm: float = 0.0,
    skip_nonfinite_steps: bool = False,
    train_blended_descriptors: bool = False,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
) -> dict[str, float]:
    selected = sample_curriculum_training_pairs(
        pair_paths,
        hard_pair_paths or [],
        batch_pairs=batch_pairs,
        hard_probability=hard_probability,
        rng=random,
    )
    optimizer.zero_grad(set_to_none=True)
    losses = []
    metric_rows: list[dict[str, float]] = []
    sampled_count = 0
    for pair_path in selected:
        pair = load_libtorch_pair_archive(pair_path, device=device)
        descriptors_a, descriptors_b, teacher_a, teacher_b = compute_student_teacher_descriptor_maps(
            model,
            pair,
            train_blended_descriptors=train_blended_descriptors,
            texture_blend_weight=texture_blend_weight,
        )
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
            teacher_descriptors_a=teacher_a,
            teacher_descriptors_b=teacher_b,
            teacher_weight=teacher_weight,
            hard_negative_weight=hard_negative_weight,
            diversity_weight=diversity_weight,
            warp_hard_negative_weight=warp_hard_negative_weight,
            warp_hard_negative_radius=warp_hard_negative_radius,
            warp_hard_negative_margin=warp_hard_negative_margin,
            warp_hard_negative_candidates=warp_hard_negative_candidates,
        )
        losses.append(loss)
        metric_rows.append(metrics)
        sampled_count += points_a.size(0)
    if not losses:
        raise RuntimeError("no valid correspondences sampled")
    loss = torch.stack(losses).mean()
    parameters = [group_param for group in optimizer.param_groups for group_param in group["params"]]
    try:
        require_finite_scalar(loss, name="training loss")
        loss.backward()
        grad_norm = clip_and_measure_gradients(parameters, max_grad_norm=max_grad_norm)
    except FloatingPointError:
        optimizer.zero_grad(set_to_none=True)
        if not skip_nonfinite_steps:
            raise
        return skipped_step_metrics(loss, metric_rows, sampled_count=sampled_count)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "grad_l2": grad_norm,
        "skipped": 0.0,
        **averaged_step_metrics(metric_rows, sampled_count),
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


def sample_curriculum_training_pairs(
    base_pair_paths: list[Path],
    hard_pair_paths: list[Path],
    *,
    batch_pairs: int,
    hard_probability: float,
    rng,
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


def make_torch_generator(device: torch.device, *, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


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
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for pair_path in pair_paths:
        pair = load_libtorch_pair_archive(pair_path, device=device)
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
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--init-pytorch-state", type=Path, default=None)
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pytorch_pfm_finetune"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--eval-pairs", type=int, default=0)
    parser.add_argument("--batch-pairs", type=int, default=2)
    parser.add_argument("--samples-per-pair", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
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
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--skip-nonfinite-steps", action="store_true")
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--hard-summary", action="append", type=Path, default=[])
    parser.add_argument("--hard-limit", type=int, default=64)
    parser.add_argument("--hard-min-matches", type=int, default=4)
    parser.add_argument("--hard-max-precision", type=float, default=0.9)
    parser.add_argument("--hard-repeat", type=int, default=3)
    parser.add_argument("--hard-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--hard-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--train-sparse-context", action="store_true")
    parser.add_argument("--train-blended-descriptors", action="store_true")
    parser.add_argument("--training-texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_training_model(args: argparse.Namespace) -> tuple[pfm_model.PlanetaryFeatureMatcher, pfm_model.CheckpointConfig]:
    if getattr(args, "init_pytorch_state", None) is not None:
        return pfm_model.load_pytorch_state(args.init_pytorch_state, device=args.device)
    return pfm_model.load_libtorch_checkpoint(args.checkpoint, device=args.device)


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model, config = load_training_model(args)
    trainable = descriptor_parameters(model, train_sparse_context=args.train_sparse_context)
    if not trainable:
        raise RuntimeError("no descriptor parameters selected")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1.0e-4)
    pair_paths = discover_pair_archives(args.cache_dir, limit_pairs=args.limit_pairs)
    if not pair_paths:
        raise RuntimeError("no pair_*.pt archives found")
    pair_paths, eval_paths = split_train_eval_pairs(pair_paths, eval_pairs=args.eval_pairs)
    if not pair_paths:
        raise RuntimeError("no training pair_*.pt archives left after eval split")
    hard_paths = select_hard_training_pairs(
        pair_paths,
        args.hard_summary,
        limit=args.hard_limit,
        min_matches=args.hard_min_matches,
        max_precision=args.hard_max_precision,
    )
    if hard_paths and args.hard_curriculum_max_probability <= 0.0:
        pair_paths = pair_paths + hard_paths * max(0, args.hard_repeat)
    if hard_paths:
        print(
            f"hard_training_pairs={len(hard_paths)} hard_repeat={args.hard_repeat} "
            f"hard_curriculum_max_probability={args.hard_curriculum_max_probability:.3f} "
            f"effective_train_pairs={len(pair_paths)}",
            flush=True,
        )
    train_generator = make_torch_generator(device, seed=args.seed)
    eval_seed = args.seed + 1000003
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    eval_summary_path = args.output_dir / "eval_summary.csv"
    if eval_paths:
        eval_before = evaluate_descriptor_retrieval(
            model,
            eval_paths,
            device=device,
            samples_per_pair=args.samples_per_pair,
            min_intensity=args.min_intensity,
            generator=make_torch_generator(device, seed=eval_seed),
            temperature=args.temperature,
        )
        print(
            f"eval_before loss={eval_before['loss']:.6f} top1={eval_before['top1_accuracy']:.4f} "
            f"top5={eval_before['top5_accuracy']:.4f} rank={eval_before['mean_positive_rank']:.2f} "
            f"pos={eval_before['mean_positive_score']:.6f} neg={eval_before['mean_negative_score']:.6f} "
            f"points={int(eval_before['points'])}",
            flush=True,
        )
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "loss",
                "grad_l2",
                "skipped",
                "teacher_weight",
                "hard_negative_weight",
                "diversity_weight",
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
        for step in range(1, args.steps + 1):
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
            metrics = train_step(
                model,
                optimizer,
                pair_paths,
                device=device,
                batch_pairs=args.batch_pairs,
                samples_per_pair=args.samples_per_pair,
                min_intensity=args.min_intensity,
                generator=train_generator,
                temperature=args.temperature,
                teacher_weight=teacher_weight,
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
                max_grad_norm=args.max_grad_norm,
                skip_nonfinite_steps=args.skip_nonfinite_steps,
                train_blended_descriptors=args.train_blended_descriptors,
                texture_blend_weight=args.training_texture_blend_weight,
            )
            writer.writerow(
                {
                    "step": step,
                    "teacher_weight": teacher_weight,
                    "hard_negative_weight": hard_negative_weight,
                    "diversity_weight": diversity_weight,
                    **metrics,
                }
            )
            handle.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(
                    f"step={step} loss={metrics['loss']:.6f} grad={metrics['grad_l2']:.6f} "
                    f"tw={teacher_weight:.3f} hn={hard_negative_weight:.3f} div={diversity_weight:.3f} "
                    f"skip={int(metrics['skipped'])} "
                    f"top1={metrics['top1_accuracy']:.4f} top5={metrics['top5_accuracy']:.4f} "
                    f"rank={metrics['mean_positive_rank']:.2f} "
                    f"pos={metrics['mean_positive_score']:.6f} neg={metrics['mean_negative_score']:.6f} "
                    f"points={int(metrics['points'])}",
                    flush=True,
                )
    if eval_paths:
        eval_after = evaluate_descriptor_retrieval(
            model,
            eval_paths,
            device=device,
            samples_per_pair=args.samples_per_pair,
            min_intensity=args.min_intensity,
            generator=make_torch_generator(device, seed=eval_seed),
            temperature=args.temperature,
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
    torch.save(
        {
            "config": config.__dict__,
            "model": model.state_dict(),
            "source_checkpoint": str(args.checkpoint),
        },
        output_path,
    )
    print(f"checkpoint={output_path}")
    print(f"metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
