#!/usr/bin/env python3
"""Lightweight PyTorch cache matching evaluation for PFM descriptor states."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

import pfm_model
import pfm_pytorch_training
from patch_descriptor_training import SyntheticPair, discover_pair_archives, load_libtorch_pair_archive


@dataclass(frozen=True)
class MatchEvalResult:
    matches: int
    correct: int
    wrong: int
    precision: float


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


def select_descriptor_keypoints(
    image: torch.Tensor,
    descriptors: torch.Tensor,
    *,
    max_keypoints: int,
    min_intensity: float,
    texture_fraction: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    if descriptors.dim() != 4 or descriptors.size(0) != 1:
        raise ValueError("descriptors must have shape 1xDxHxW")
    if max_keypoints <= 0:
        raise ValueError("max_keypoints must be positive")
    if texture_fraction < 0.0 or texture_fraction > 1.0:
        raise ValueError("texture_fraction must be in [0, 1]")
    _, image_height, image_width = image.shape
    descriptor_height = descriptors.size(2)
    descriptor_width = descriptors.size(3)
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
        uniform_count = max_keypoints - texture_count
        chosen_parts: list[torch.Tensor] = []
        if texture_count > 0:
            texture = image_texture_scores(image.to(descriptors.device), image_points)
            selected_scores = texture.index_select(0, selected)
            order = selected_scores.argsort(descending=True, stable=True)[:texture_count]
            chosen_parts.append(selected.index_select(0, order))
        if uniform_count > 0:
            already = torch.cat(chosen_parts) if chosen_parts else selected.new_empty((0,))
            keep = torch.ones(selected.size(0), dtype=torch.bool, device=selected.device)
            if already.numel() > 0:
                keep &= ~torch.isin(selected, already)
            remaining = selected[keep]
            if remaining.numel() > 0:
                sample = torch.linspace(
                    0,
                    remaining.numel() - 1,
                    steps=min(uniform_count, remaining.numel()),
                    device=remaining.device,
                ).round().to(torch.long)
                chosen_parts.append(remaining.index_select(0, sample))
        selected = torch.cat(chosen_parts) if chosen_parts else selected[:max_keypoints]
    return keypoints.index_select(0, selected).contiguous(), selected.to(torch.long).contiguous()


def gather_descriptor_rows(descriptors: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
    if descriptors.dim() != 4 or descriptors.size(0) != 1:
        raise ValueError("descriptors must have shape 1xDxHxW")
    flat = descriptors.squeeze(0).permute(1, 2, 0).reshape(-1, descriptors.size(1))
    if selected_indices.numel() == 0:
        return flat.new_empty((0, descriptors.size(1)))
    return flat.index_select(0, selected_indices.to(descriptors.device)).contiguous()


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
    if max_matches <= 0:
        raise ValueError("max_matches must be positive")
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
    for score, source, target in candidates:
        if source in used_a or target in used_b:
            continue
        used_a.add(source)
        used_b.add(target)
        matches.append([source, target])
        scores.append(score)
        if len(matches) >= max_matches:
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
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_matches <= 0:
        raise ValueError("max_matches must be positive")
    if desc_a.size(0) == 0 or desc_b.size(0) == 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=desc_a.device),
            torch.empty(0, dtype=torch.float32, device=desc_a.device),
        )
    similarity = cyclic_descriptor_similarity(desc_a, desc_b)
    best_scores, best_targets = similarity.max(dim=1)
    best_sources = similarity.max(dim=0).indices
    matches: list[list[int]] = []
    scores: list[float] = []
    for source in range(similarity.size(0)):
        target = int(best_targets[source].detach().cpu())
        score = float(best_scores[source].detach().cpu())
        if score < min_score:
            continue
        if int(best_sources[target].detach().cpu()) == source:
            matches.append([source, target])
            scores.append(score)
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:max_matches]
    device = desc_a.device
    if not order:
        return torch.empty(0, 2, dtype=torch.long, device=device), torch.empty(0, dtype=torch.float32, device=device)
    return (
        torch.tensor([matches[index] for index in order], dtype=torch.long, device=device),
        torch.tensor([scores[index] for index in order], dtype=torch.float32, device=device),
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
    max_keypoints: int,
    min_intensity: float,
    threshold_px: float,
    topk: int = 1,
    max_matches: int = 512,
    min_score: float = -1.0,
    mutual: bool = False,
    geometry_filter: str = "none",
    texture_fraction: float = 1.0,
) -> MatchEvalResult:
    keypoints_a, selected_a = select_descriptor_keypoints(
        pair.view_a,
        descriptors_a,
        max_keypoints=max_keypoints,
        min_intensity=min_intensity,
        texture_fraction=texture_fraction,
    )
    keypoints_b, selected_b = select_descriptor_keypoints(
        pair.view_b,
        descriptors_b,
        max_keypoints=max_keypoints,
        min_intensity=min_intensity,
        texture_fraction=texture_fraction,
    )
    rows_a = gather_descriptor_rows(descriptors_a, selected_a)
    rows_b = gather_descriptor_rows(descriptors_b, selected_b)
    if mutual:
        matches, _ = mutual_nearest_matches(rows_a, rows_b, max_matches=max_matches, min_score=min_score)
    else:
        matches, _ = greedy_unique_matches(
            rows_a,
            rows_b,
            topk=topk,
            max_matches=max_matches,
            min_score=min_score,
        )
    if matches.numel() == 0:
        return MatchEvalResult(matches=0, correct=0, wrong=0, precision=0.0)

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
            return MatchEvalResult(matches=0, correct=0, wrong=0, precision=0.0)
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
    return MatchEvalResult(matches=total, correct=correct, wrong=wrong, precision=precision)


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
    mutual: bool,
    geometry_filter: str,
) -> MatchEvalResult:
    pair = load_libtorch_pair_archive(pair_path, device=device)
    with torch.no_grad():
        descriptors_a, descriptors_b = descriptor_maps_for_pair(
            model,
            pair,
            mode=mode,
            texture_blend_weight=texture_blend_weight,
        )
        return match_pair_descriptor_maps(
            pair,
            descriptors_a,
            descriptors_b,
            max_keypoints=max_keypoints,
            min_intensity=min_intensity,
            texture_fraction=texture_fraction,
            threshold_px=threshold_px,
            topk=topk,
            max_matches=max_matches,
            min_score=min_score,
            mutual=mutual,
            geometry_filter=geometry_filter,
        )


def summarize(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "pairs=0"
    total = sum(int(row["matches"]) for row in rows)
    correct = sum(int(row["correct"]) for row in rows)
    precision = 0.0 if total == 0 else correct / total
    low = sum(1 for row in rows if float(row["precision"]) < 0.9)
    return f"pairs={len(rows)} total_matches={total} correct={correct} precision={precision:.6f} low_precision_pairs={low}"


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
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--descriptor-topk", type=int, default=1)
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--geometry-filter", choices=["none", "affine"], default="none")
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--hard-summary", action="append", type=Path, default=[])
    parser.add_argument("--hard-limit", type=int, default=64)
    parser.add_argument("--hard-min-matches", type=int, default=4)
    parser.add_argument("--hard-max-precision", type=float, default=0.9)
    return parser.parse_args()


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
    pair_paths = discover_pair_archives(args.cache_dir, limit_pairs=args.limit_pairs)
    if not args.hard_summary:
        return pair_paths
    return pfm_pytorch_training.select_hard_training_pairs(
        pair_paths,
        args.hard_summary,
        limit=args.hard_limit,
        min_matches=args.hard_min_matches,
        max_precision=args.hard_max_precision,
    )


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = load_model(args)
    model.eval()
    device = torch.device(args.device)
    pairs = selected_pair_paths(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_pt", "matches", "correct", "wrong", "precision"])
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
                threshold_px=args.threshold_px,
                topk=args.descriptor_topk,
                max_matches=args.max_matches,
                min_score=args.min_score,
                mutual=args.mutual,
                geometry_filter=args.geometry_filter,
            )
            row = {
                "pair_pt": pair_path.as_posix(),
                "matches": str(result.matches),
                "correct": str(result.correct),
                "wrong": str(result.wrong),
                "precision": f"{result.precision:.6f}",
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
