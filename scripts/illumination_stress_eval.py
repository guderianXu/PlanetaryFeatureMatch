#!/usr/bin/env python3
"""Evaluate PFM matching under deterministic illumination changes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import pfm_model
from patch_descriptor_training import SyntheticPair, load_libtorch_pair_archive
from pytorch_cache_match_eval import feature_maps_and_keypoint_scores_for_pair, match_pair_descriptor_maps


def _clamp_image(image: torch.Tensor) -> torch.Tensor:
    return image.to(torch.float32).clamp(0.0, 1.0)


def make_illumination_variants(image: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Return deterministic photometric variants with unchanged geometry."""
    if image.dim() != 3:
        raise ValueError("image must have shape CxHxW")
    base = _clamp_image(image)
    _, height, width = base.shape
    xx = torch.linspace(0.0, 1.0, width, dtype=base.dtype, device=base.device).view(1, 1, width)
    yy = torch.linspace(0.0, 1.0, height, dtype=base.dtype, device=base.device).view(1, height, 1)
    center_shadow = 0.35 + 0.65 * torch.clamp((yy - 0.5).abs() * 3.0, 0.0, 1.0)
    side_shading = 0.45 + 0.55 * xx
    local_mean = torch.nn.functional.avg_pool2d(
        base.unsqueeze(0), kernel_size=9, stride=1, padding=4, count_include_pad=False
    ).squeeze(0)
    low_contrast = torch.clamp((base - local_mean) * 0.35 + local_mean, 0.0, 1.0)
    high_contrast = torch.clamp((base - 0.5) * 1.6 + 0.5, 0.0, 1.0)
    return [
        ("original", base),
        ("gamma_dark", torch.pow(base, 1.8)),
        ("gamma_bright", torch.pow(base, 0.55)),
        ("contrast_low", low_contrast),
        ("contrast_high", high_contrast),
        ("shadow_band", torch.clamp(base * center_shadow, 0.0, 1.0)),
        ("side_shading", torch.clamp(base * side_shading, 0.0, 1.0)),
    ]


def evaluate_pair(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair: SyntheticPair,
    *,
    mode: str,
    texture_blend_weight: float,
    keypoint_score_mode: str,
    matcher_mode: str,
    max_keypoints: int,
    min_intensity: float,
    texture_fraction: float,
    weak_texture_fraction: float,
    threshold_px: float,
    max_matches: int,
    min_score: float,
    min_margin: float,
    graph_dustbin_delta: float,
    graph_acceptance_margin: float,
    graph_min_raw_score: float,
    graph_min_raw_margin: float,
    spatial_bins: int,
    keypoint_cell_cap: int,
) -> tuple[int, int, float]:
    with torch.no_grad():
        descriptors_a, descriptors_b, scores_a, scores_b, raw_a, raw_b = feature_maps_and_keypoint_scores_for_pair(
            model,
            pair,
            mode=mode,
            texture_blend_weight=texture_blend_weight,
            keypoint_score_mode=keypoint_score_mode,
        )
        result = match_pair_descriptor_maps(
            pair,
            descriptors_a,
            descriptors_b,
            model=model,
            matcher_mode=matcher_mode,
            max_keypoints=max_keypoints,
            min_intensity=min_intensity,
            threshold_px=threshold_px,
            topk=1,
            max_matches=max_matches,
            min_score=min_score,
            min_margin=min_margin,
            texture_fraction=texture_fraction,
            weak_texture_fraction=weak_texture_fraction,
            keypoint_spatial_bins=spatial_bins,
            keypoint_cell_cap=keypoint_cell_cap,
            keypoint_scores_a=scores_a,
            keypoint_scores_b=scores_b,
            raw_features_a=raw_a,
            raw_features_b=raw_b,
            graph_dustbin_delta=graph_dustbin_delta,
            graph_acceptance_margin=graph_acceptance_margin,
            graph_min_raw_score=graph_min_raw_score,
            graph_min_raw_margin=graph_min_raw_margin,
        )
    return result.matches, result.correct, result.precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--pytorch-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    base_pair = load_libtorch_pair_archive(args.pair, device=device)
    rows = []
    for name, variant_b in make_illumination_variants(base_pair.view_b):
        pair = SyntheticPair(
            view_a=base_pair.view_a,
            view_b=variant_b,
            warp_a_to_b=base_pair.warp_a_to_b,
            valid_mask=base_pair.valid_mask,
        )
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
        rows.append({"variant": name, "matches": matches, "correct": correct, "precision": precision})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "matches", "correct", "precision"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote illumination stress report to {args.output}")


if __name__ == "__main__":
    main()
