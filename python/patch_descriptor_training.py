#!/usr/bin/env python3
"""PyTorch patch-level descriptor training for synthetic warp pairs.

This is the fast Python/PyTorch prototype path. It trains a local descriptor
from known A-to-B warp correspondences before the validated ideas are ported
back to the C++/LibTorch inference path.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from compact_pair_cache import (
    is_compact_pair_payload,
    load_shared_image,
    resolve_compact_image_path,
)

DESCRIPTOR_NORMALIZE_EPS = 1.0e-3


@dataclass(frozen=True)
class SyntheticPair:
    view_a: torch.Tensor
    view_b: torch.Tensor
    warp_a_to_b: torch.Tensor
    valid_mask: torch.Tensor


def load_libtorch_pair_archive(path: Path | str, *, device: str | torch.device = "cpu") -> SyntheticPair:
    pair_path = Path(path)
    try:
        payload = torch.load(pair_path, map_location=device)
    except RuntimeError as exc:
        message = str(exc)
        if "TorchScript archive" not in message and "weights_only" not in message:
            raise
    else:
        if is_compact_pair_payload(payload):
            image_a_path = resolve_compact_image_path(pair_path, payload["image_a"])
            image_b_path = resolve_compact_image_path(pair_path, payload["image_b"])
            return SyntheticPair(
                view_a=load_shared_image(image_a_path, device=device),
                view_b=load_shared_image(image_b_path, device=device),
                warp_a_to_b=payload["warp_a_to_b"].to(device=device, dtype=torch.float32).contiguous(),
                valid_mask=payload["valid_mask"].to(device=device, dtype=torch.bool).contiguous(),
            )
    module = torch.jit.load(str(pair_path), map_location=device)
    return SyntheticPair(
        view_a=module.view_a.to(device=device, dtype=torch.float32).contiguous(),
        view_b=module.view_b.to(device=device, dtype=torch.float32).contiguous(),
        warp_a_to_b=module.warp_a_to_b.to(device=device, dtype=torch.float32).contiguous(),
        valid_mask=module.valid_mask.to(device=device, dtype=torch.bool).contiguous(),
    )


def _normalize_xy(points_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if width <= 1 or height <= 1:
        raise ValueError("image height and width must be greater than 1")
    x = points_xy[..., 0] * (2.0 / float(width - 1)) - 1.0
    y = points_xy[..., 1] * (2.0 / float(height - 1)) - 1.0
    return torch.stack((x, y), dim=-1)


def extract_patches(image: torch.Tensor, points_xy: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    if points_xy.dim() != 2 or points_xy.size(1) != 2:
        raise ValueError("points_xy must have shape Nx2")
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")

    channels, height, width = image.shape
    count = points_xy.size(0)
    if count == 0:
        return image.new_empty((0, channels, patch_size, patch_size))

    radius = patch_size // 2
    offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=points_xy.dtype)
    off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")
    patch_xy = points_xy[:, None, None, :] + torch.stack((off_x, off_y), dim=-1)
    grid = _normalize_xy(patch_xy, height, width)
    expanded = image.unsqueeze(0).expand(count, -1, -1, -1)
    return F.grid_sample(expanded, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _center_intensity(image: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    if points_xy.numel() == 0:
        return image.new_empty((0,))
    _, height, width = image.shape
    xy = points_xy.round().to(torch.long)
    x = xy[:, 0].clamp(0, width - 1)
    y = xy[:, 1].clamp(0, height - 1)
    return image.mean(dim=0)[y, x]


def sample_valid_correspondences(
    pair: SyntheticPair,
    *,
    count: int,
    patch_size: int,
    min_intensity: float = 0.01,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("count must be positive")
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    if pair.view_a.dim() != 3 or pair.view_b.dim() != 3:
        raise ValueError("views must have shape CxHxW")
    if pair.warp_a_to_b.dim() != 3 or pair.warp_a_to_b.size(-1) != 2:
        raise ValueError("warp_a_to_b must have shape HxWx2")

    _, height_a, width_a = pair.view_a.shape
    _, height_b, width_b = pair.view_b.shape
    if pair.valid_mask.shape != (height_a, width_a):
        raise ValueError("valid_mask shape must match view_a")
    if pair.warp_a_to_b.shape[:2] != (height_a, width_a):
        raise ValueError("warp_a_to_b spatial shape must match view_a")

    radius = patch_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(height_a, device=pair.view_a.device),
        torch.arange(width_a, device=pair.view_a.device),
        indexing="ij",
    )
    points_a_all = torch.stack((xx.to(torch.float32), yy.to(torch.float32)), dim=-1)
    points_b_all = pair.warp_a_to_b
    valid = pair.valid_mask.clone()
    valid &= xx >= radius
    valid &= xx < width_a - radius
    valid &= yy >= radius
    valid &= yy < height_a - radius
    valid &= points_b_all[..., 0] >= radius
    valid &= points_b_all[..., 0] < width_b - radius
    valid &= points_b_all[..., 1] >= radius
    valid &= points_b_all[..., 1] < height_b - radius

    candidate_a = points_a_all[valid]
    candidate_b = points_b_all[valid]
    if candidate_a.numel() == 0:
        return (
            pair.view_a.new_empty((0, 2)),
            pair.view_b.new_empty((0, 2)),
        )

    if min_intensity > 0.0:
        intensity_a = _center_intensity(pair.view_a, candidate_a)
        intensity_b = _center_intensity(pair.view_b, candidate_b)
        textured = (intensity_a > min_intensity) & (intensity_b > min_intensity)
        candidate_a = candidate_a[textured]
        candidate_b = candidate_b[textured]
    if candidate_a.numel() == 0:
        return (
            pair.view_a.new_empty((0, 2)),
            pair.view_b.new_empty((0, 2)),
        )

    take = min(count, candidate_a.size(0))
    order = torch.randperm(candidate_a.size(0), generator=generator, device=candidate_a.device)[:take]
    return candidate_a.index_select(0, order), candidate_b.index_select(0, order)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv2(F.relu(self.conv1(x))), inplace=False)


def normalize_patches(patches: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    if patches.dim() != 4:
        raise ValueError("patches must have shape NxCxHxW")
    mean = patches.mean(dim=(1, 2, 3), keepdim=True)
    std = patches.std(dim=(1, 2, 3), unbiased=False, keepdim=True)
    return torch.where(std > eps, (patches - mean) / std.clamp_min(eps), torch.zeros_like(patches))


def normalize_descriptors(desc: torch.Tensor, *, eps: float = DESCRIPTOR_NORMALIZE_EPS) -> torch.Tensor:
    if desc.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    finite = torch.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    return finite / finite.norm(p=2, dim=1, keepdim=True).clamp_min(eps)


class RotationInvariantPatchDescriptor(nn.Module):
    def __init__(self, *, descriptor_dim: int = 128, base_channels: int = 32) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            _ResidualBlock(base_channels),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            _ResidualBlock(base_channels * 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(base_channels * 2, descriptor_dim)

    def _encode_once(self, patches: torch.Tensor) -> torch.Tensor:
        features = self.stem(patches).flatten(1)
        return self.projection(features)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4 or patches.size(1) != 1:
            raise ValueError("patches must have shape Nx1xHxW")
        patches = normalize_patches(patches)
        descriptors = []
        for turns in range(4):
            rotated = patches if turns == 0 else torch.rot90(patches, turns, dims=(-2, -1)).contiguous()
            descriptors.append(normalize_descriptors(self._encode_once(rotated)))
        return normalize_descriptors(torch.stack(descriptors, dim=0).mean(dim=0))


def descriptor_diversity_loss(desc: torch.Tensor, *, margin: float = 0.2) -> torch.Tensor:
    if desc.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if desc.size(0) <= 1:
        return desc.new_zeros(())
    if margin < -1.0 or margin > 1.0:
        raise ValueError("margin must be in [-1, 1]")
    desc = normalize_descriptors(desc)
    similarity = desc @ desc.T
    off_diagonal = ~torch.eye(desc.size(0), device=desc.device, dtype=torch.bool)
    excess = (similarity[off_diagonal] - margin).clamp_min(0.0)
    return excess.pow(2).mean()


def paired_descriptor_metrics(desc_a: torch.Tensor, desc_b: torch.Tensor) -> dict[str, float]:
    if desc_a.dim() != 2 or desc_b.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if desc_a.shape != desc_b.shape:
        raise ValueError("descriptor tensors must have the same shape")
    if desc_a.size(0) == 0:
        raise ValueError("descriptor batch must not be empty")
    desc_a = normalize_descriptors(desc_a)
    desc_b = normalize_descriptors(desc_b)
    similarity = desc_a @ desc_b.T
    target = torch.arange(desc_a.size(0), device=desc_a.device)
    top1 = similarity.argmax(dim=1).eq(target).to(torch.float32).mean()
    sorted_indices = similarity.argsort(dim=1, descending=True)
    ranks = sorted_indices.eq(target[:, None]).to(torch.int64).argmax(dim=1).to(torch.float32) + 1.0
    top5 = ranks.le(min(5, desc_a.size(0))).to(torch.float32).mean()
    top10 = ranks.le(min(10, desc_a.size(0))).to(torch.float32).mean()
    positive = similarity.diagonal().mean()
    if desc_a.size(0) == 1:
        negative = similarity.new_tensor(0.0)
    else:
        off_diagonal = ~torch.eye(desc_a.size(0), device=desc_a.device, dtype=torch.bool)
        negative = similarity[off_diagonal].mean()
    return {
        "top1_accuracy": float(top1.detach().cpu()),
        "top5_accuracy": float(top5.detach().cpu()),
        "top10_accuracy": float(top10.detach().cpu()),
        "mean_positive_rank": float(ranks.mean().detach().cpu()),
        "mean_positive_score": float(positive.detach().cpu()),
        "mean_negative_score": float(negative.detach().cpu()),
    }


def paired_descriptor_loss(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    temperature: float = 0.07,
    diversity_weight: float = 0.10,
    diversity_margin: float = 0.2,
) -> torch.Tensor:
    if desc_a.dim() != 2 or desc_b.dim() != 2:
        raise ValueError("descriptors must have shape NxD")
    if desc_a.shape != desc_b.shape:
        raise ValueError("descriptor tensors must have the same shape")
    if desc_a.size(0) == 0:
        raise ValueError("descriptor batch must not be empty")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    desc_a = normalize_descriptors(desc_a)
    desc_b = normalize_descriptors(desc_b)
    logits = desc_a @ desc_b.T / temperature
    target = torch.arange(desc_a.size(0), device=desc_a.device)
    contrastive = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    if diversity_weight <= 0.0:
        return contrastive
    diversity = descriptor_diversity_loss(desc_a, margin=diversity_margin)
    diversity = diversity + descriptor_diversity_loss(desc_b, margin=diversity_margin)
    return contrastive + diversity_weight * diversity


_SOURCE_INDEX_RE = re.compile(r"^source_(\d+)(?:_|$)")
_PAIR_INDEX_RE = re.compile(r"^pair_(\d+)$")


def is_self_pair_archive(path: Path | str) -> bool:
    pair_path = Path(path)
    source_match = _SOURCE_INDEX_RE.match(pair_path.parent.name)
    pair_match = _PAIR_INDEX_RE.match(pair_path.stem)
    if source_match is None or pair_match is None:
        return False
    return int(source_match.group(1)) == int(pair_match.group(1))


def discover_pair_archives(
    cache_dirs: Iterable[Path | str],
    *,
    limit_pairs: int = 0,
    exclude_self_pairs: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for cache_dir in cache_dirs:
        root = Path(cache_dir)
        if not root.exists():
            continue
        paths.extend(sorted(root.glob("source_*/pair_*.pt")))
    unique = sorted(dict.fromkeys(paths))
    if exclude_self_pairs:
        unique = [path for path in unique if not is_self_pair_archive(path)]
    if limit_pairs > 0:
        return unique[:limit_pairs]
    return unique


def train_step(
    model: RotationInvariantPatchDescriptor,
    optimizer: torch.optim.Optimizer,
    pair_paths: list[Path],
    *,
    device: str | torch.device,
    batch_pairs: int,
    samples_per_pair: int,
    patch_size: int,
    min_intensity: float,
    generator: torch.Generator,
) -> dict[str, float]:
    if not pair_paths:
        raise ValueError("pair_paths must not be empty")
    selected = random.sample(pair_paths, k=min(batch_pairs, len(pair_paths)))
    patches_a: list[torch.Tensor] = []
    patches_b: list[torch.Tensor] = []
    sampled_points = 0
    for pair_path in selected:
        pair = load_libtorch_pair_archive(pair_path, device=device)
        points_a, points_b = sample_valid_correspondences(
            pair,
            count=samples_per_pair,
            patch_size=patch_size,
            min_intensity=min_intensity,
            generator=generator,
        )
        if points_a.size(0) == 0:
            continue
        patches_a.append(extract_patches(pair.view_a, points_a, patch_size=patch_size))
        patches_b.append(extract_patches(pair.view_b, points_b, patch_size=patch_size))
        sampled_points += points_a.size(0)

    if not patches_a:
        raise RuntimeError("no valid correspondences sampled")

    batch_a = torch.cat(patches_a, dim=0)
    batch_b = torch.cat(patches_b, dim=0)
    optimizer.zero_grad(set_to_none=True)
    desc_a = model(batch_a)
    desc_b = model(batch_b)
    loss = paired_descriptor_loss(desc_a, desc_b)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        metrics = paired_descriptor_metrics(desc_a, desc_b)
    return {
        "loss": float(loss.detach().cpu()),
        **metrics,
        "points": float(sampled_points),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch patch descriptor from synthetic warp pairs")
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pytorch_patch_descriptor"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--samples-per-pair", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=33)
    parser.add_argument("--descriptor-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    pair_paths = discover_pair_archives(args.cache_dir, limit_pairs=args.limit_pairs)
    if not pair_paths:
        raise RuntimeError("no pair_*.pt archives found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    model = RotationInvariantPatchDescriptor(
        descriptor_dim=args.descriptor_dim,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
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
        for step in range(1, args.steps + 1):
            metrics = train_step(
                model,
                optimizer,
                pair_paths,
                device=device,
                batch_pairs=args.batch_pairs,
                samples_per_pair=args.samples_per_pair,
                patch_size=args.patch_size,
                min_intensity=args.min_intensity,
                generator=generator,
            )
            row = {"step": step, **metrics}
            writer.writerow(row)
            handle.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(
                    f"step={step} loss={metrics['loss']:.6f} "
                    f"top1={metrics['top1_accuracy']:.4f} top5={metrics['top5_accuracy']:.4f} "
                    f"pos={metrics['mean_positive_score']:.6f} "
                    f"neg={metrics['mean_negative_score']:.6f} "
                    f"points={int(metrics['points'])}",
                    flush=True,
                )

    checkpoint_path = args.output_dir / "patch_descriptor.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "descriptor_dim": args.descriptor_dim,
            "base_channels": args.base_channels,
            "patch_size": args.patch_size,
        },
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
