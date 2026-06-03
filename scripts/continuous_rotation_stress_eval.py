#!/usr/bin/env python3
"""Evaluate PFM matching on synthetic continuous-angle rotations."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import pfm_model
from patch_descriptor_training import SyntheticPair, load_libtorch_pair_archive
from illumination_stress_eval import evaluate_pair


def _normalize_grid_xy(x: torch.Tensor, y: torch.Tensor, height: int, width: int) -> torch.Tensor:
    gx = x * (2.0 / float(max(1, width - 1))) - 1.0
    gy = y * (2.0 / float(max(1, height - 1))) - 1.0
    return torch.stack([gx, gy], dim=-1)


def rotate_image(image: torch.Tensor, *, angle_deg: float) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    _, height, width = image.shape
    dtype = image.dtype
    device = image.device
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    dx = xx - cx
    dy = yy - cy
    src_x = cos_a * dx + sin_a * dy + cx
    src_y = -sin_a * dx + cos_a * dy + cy
    grid = _normalize_grid_xy(src_x, src_y, height, width).unsqueeze(0)
    rotated = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return rotated.squeeze(0).contiguous()


def rotate_pair_from_view(image: torch.Tensor, *, angle_deg: float) -> SyntheticPair:
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    _, height, width = image.shape
    dtype = image.dtype
    device = image.device
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    dx = xx - cx
    dy = yy - cy
    dst_x = cos_a * dx - sin_a * dy + cx
    dst_y = sin_a * dx + cos_a * dy + cy
    warp = torch.stack([dst_x, dst_y], dim=-1).to(torch.float32).contiguous()
    valid = (dst_x >= 0.0) & (dst_x <= width - 1) & (dst_y >= 0.0) & (dst_y <= height - 1)
    return SyntheticPair(
        view_a=image.contiguous(),
        view_b=rotate_image(image, angle_deg=angle_deg),
        warp_a_to_b=warp,
        valid_mask=valid.contiguous(),
    )


def parse_angles(text: str) -> list[float]:
    angles = []
    for item in text.split(","):
        stripped = item.strip()
        if stripped:
            angles.append(float(stripped))
    if not angles:
        raise ValueError("at least one angle is required")
    return angles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True, help="A pair archive; view_a is used as the source image")
    parser.add_argument("--pytorch-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", default="0,30,45,60,90,120,135,150,180,270")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--descriptor-mode", choices=["learned", "texture", "blend"], default="blend")
    parser.add_argument("--texture-blend-weight", type=float, default=0.35)
    parser.add_argument("--keypoint-score-mode", choices=["texture", "learned"], default="learned")
    parser.add_argument("--matcher-mode", choices=["raw_descriptor", "graph_matcher"], default="graph_matcher")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--texture-fraction", type=float, default=0.4)
    parser.add_argument("--weak-texture-fraction", type=float, default=0.4)
    parser.add_argument("--threshold-px", type=float, default=8.0)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--graph-dustbin-delta", type=float, default=0.0)
    parser.add_argument("--graph-acceptance-margin", type=float, default=0.0)
    parser.add_argument("--graph-min-raw-score", type=float, default=-1.0)
    parser.add_argument("--graph-min-raw-margin", type=float, default=0.0)
    parser.add_argument("--spatial-bins", type=int, default=8)
    parser.add_argument("--keypoint-cell-cap", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = pfm_model.load_pytorch_state(args.pytorch_state, device=device)
    model.eval()
    source_pair = load_libtorch_pair_archive(args.pair, device=device)
    rows = []
    for angle in parse_angles(args.angles):
        pair = rotate_pair_from_view(source_pair.view_a, angle_deg=angle)
        matches, correct, precision = evaluate_pair(
            model,
            pair,
            mode=args.descriptor_mode,
            texture_blend_weight=args.texture_blend_weight,
            keypoint_score_mode=args.keypoint_score_mode,
            matcher_mode=args.matcher_mode,
            max_keypoints=args.max_keypoints,
            min_intensity=args.min_intensity,
            texture_fraction=args.texture_fraction,
            weak_texture_fraction=args.weak_texture_fraction,
            threshold_px=args.threshold_px,
            max_matches=args.max_matches,
            min_score=args.min_score,
            min_margin=args.min_margin,
            graph_dustbin_delta=args.graph_dustbin_delta,
            graph_acceptance_margin=args.graph_acceptance_margin,
            graph_min_raw_score=args.graph_min_raw_score,
            graph_min_raw_margin=args.graph_min_raw_margin,
            spatial_bins=args.spatial_bins,
            keypoint_cell_cap=args.keypoint_cell_cap,
        )
        rows.append({"angle_deg": angle, "matches": matches, "correct": correct, "precision": precision})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["angle_deg", "matches", "correct", "precision"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote continuous rotation stress report to {args.output}")


if __name__ == "__main__":
    main()
