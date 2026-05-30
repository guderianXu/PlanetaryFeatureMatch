#!/usr/bin/env python3
"""Mine raw PFM false matches from synthetic cache pairs."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import pfm_model
import pytorch_cache_match_eval as eval_py
from patch_descriptor_training import SyntheticPair, discover_pair_archives, load_libtorch_pair_archive


CSV_FIELDS = ["pair_pt", "ax", "ay", "bx", "by", "error_px", "score", "margin", "style", "gate"]


@dataclass(frozen=True)
class FalseMatchRow:
    pair_pt: str
    ax: float
    ay: float
    bx: float
    by: float
    error_px: float
    score: float
    margin: float
    style: str
    gate: str

    def csv_dict(self) -> dict[str, str]:
        row = asdict(self)
        return {
            "pair_pt": str(row["pair_pt"]),
            "ax": f"{float(row['ax']):.3f}",
            "ay": f"{float(row['ay']):.3f}",
            "bx": f"{float(row['bx']):.3f}",
            "by": f"{float(row['by']):.3f}",
            "error_px": f"{float(row['error_px']):.3f}",
            "score": f"{float(row['score']):.6f}",
            "margin": f"{float(row['margin']):.6f}",
            "style": str(row["style"]),
            "gate": str(row["gate"]),
        }


def infer_style_gate(pair_path: Path) -> tuple[str, str]:
    lower_parts = [part.lower() for part in pair_path.parts]
    style = "timestamp" if "timestamp" in lower_parts else "numeric" if "numeric" in lower_parts else ""
    gate = ""
    for candidate in ("rotate", "viewpoint", "compound"):
        if candidate in lower_parts:
            gate = candidate
            break
    if not gate:
        parent_text = pair_path.parent.parent.name.lower()
        if "compound" in parent_text:
            gate = "compound"
        elif "viewpoint" in parent_text:
            gate = "viewpoint"
        elif "rotate" in parent_text:
            gate = "rotate"
    if not style:
        style = "timestamp" if "nas_pan" in pair_path.parent.name.lower() else "numeric"
    if not gate:
        gate = "unknown"
    return style, gate


def _valid_source_mask(valid_mask: torch.Tensor, points_a: torch.Tensor) -> torch.Tensor:
    if points_a.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=points_a.device)
    height, width = valid_mask.shape
    in_bounds = (
        (points_a[:, 0] >= 0.0)
        & (points_a[:, 0] <= float(width - 1))
        & (points_a[:, 1] >= 0.0)
        & (points_a[:, 1] <= float(height - 1))
    )
    rounded = points_a.round().to(torch.long)
    x = rounded[:, 0].clamp(0, width - 1)
    y = rounded[:, 1].clamp(0, height - 1)
    return in_bounds.to(valid_mask.device) & valid_mask.to(points_a.device)[y, x]


def false_match_rows_from_tensors(
    *,
    pair_pt: str,
    style: str,
    gate: str,
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    scores: torch.Tensor,
    margins: torch.Tensor,
    warp_a_to_b: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold_px: float,
) -> list[FalseMatchRow]:
    if points_a.numel() == 0:
        return []
    target_b = eval_py.sample_warp(warp_a_to_b, points_a)
    errors = (target_b.to(points_b.device) - points_b).norm(dim=1)
    valid = _valid_source_mask(valid_mask, points_a)
    wrong = (~valid.to(errors.device)) | errors.gt(float(threshold_px))
    rows: list[FalseMatchRow] = []
    for index in torch.nonzero(wrong, as_tuple=False).reshape(-1).tolist():
        rows.append(
            FalseMatchRow(
                pair_pt=pair_pt,
                ax=float(points_a[index, 0].detach().cpu()),
                ay=float(points_a[index, 1].detach().cpu()),
                bx=float(points_b[index, 0].detach().cpu()),
                by=float(points_b[index, 1].detach().cpu()),
                error_px=float(errors[index].detach().cpu()),
                score=float(scores[index].detach().cpu()),
                margin=float(margins[index].detach().cpu()),
                style=style,
                gate=gate,
            )
        )
    return rows


def mutual_matches_with_margins(
    rows_a: torch.Tensor,
    rows_b: torch.Tensor,
    *,
    max_matches: int,
    min_score: float,
    min_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rows_a.size(0) == 0 or rows_b.size(0) == 0:
        device = rows_a.device
        return (
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
    similarity = eval_py.cyclic_descriptor_similarity(rows_a, rows_b)
    best_scores, best_targets = similarity.max(dim=1)
    if similarity.size(1) > 1:
        top2 = similarity.topk(2, dim=1).values
        row_margins = top2[:, 0] - top2[:, 1]
    else:
        row_margins = torch.full((similarity.size(0),), float("inf"), dtype=torch.float32, device=similarity.device)
    best_sources = similarity.max(dim=0).indices

    matches: list[list[int]] = []
    scores: list[float] = []
    margins: list[float] = []
    for source in range(similarity.size(0)):
        target = int(best_targets[source].detach().cpu())
        score = float(best_scores[source].detach().cpu())
        margin = float(row_margins[source].detach().cpu())
        if score < min_score or margin < min_margin:
            continue
        if int(best_sources[target].detach().cpu()) == source:
            matches.append([source, target])
            scores.append(score)
            margins.append(margin)
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:max_matches]
    device = rows_a.device
    if not order:
        return (
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
    return (
        torch.tensor([matches[index] for index in order], dtype=torch.long, device=device),
        torch.tensor([scores[index] for index in order], dtype=torch.float32, device=device),
        torch.tensor([margins[index] for index in order], dtype=torch.float32, device=device),
    )


@torch.no_grad()
def mine_pair(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair_path: Path,
    *,
    device: torch.device,
    mode: str,
    texture_blend_weight: float,
    max_keypoints: int,
    max_matches: int,
    min_intensity: float,
    min_score: float,
    min_margin: float,
    threshold_px: float,
    keypoint_score_mode: str,
) -> list[FalseMatchRow]:
    pair = load_libtorch_pair_archive(pair_path, device=device)
    descriptors_a, descriptors_b, scores_a, scores_b = eval_py.descriptor_maps_and_keypoint_scores_for_pair(
        model,
        pair,
        mode=mode,
        texture_blend_weight=texture_blend_weight,
        keypoint_score_mode=keypoint_score_mode,
    )
    keypoints_a, selected_a = eval_py.select_descriptor_keypoints(
        pair.view_a,
        descriptors_a,
        max_keypoints=max_keypoints,
        min_intensity=min_intensity,
        keypoint_scores=scores_a,
    )
    keypoints_b, selected_b = eval_py.select_descriptor_keypoints(
        pair.view_b,
        descriptors_b,
        max_keypoints=max_keypoints,
        min_intensity=min_intensity,
        keypoint_scores=scores_b,
    )
    rows_a = eval_py.gather_descriptor_rows(descriptors_a, selected_a)
    rows_b = eval_py.gather_descriptor_rows(descriptors_b, selected_b)
    matches, match_scores, match_margins = mutual_matches_with_margins(
        rows_a,
        rows_b,
        max_matches=max_matches,
        min_score=min_score,
        min_margin=min_margin,
    )
    if matches.numel() == 0:
        return []

    _, image_height_a, image_width_a = pair.view_a.shape
    _, image_height_b, image_width_b = pair.view_b.shape
    points_a = eval_py._feature_to_image_points(
        keypoints_a.index_select(0, matches[:, 0].to(keypoints_a.device)),
        feature_height=descriptors_a.size(2),
        feature_width=descriptors_a.size(3),
        image_height=image_height_a,
        image_width=image_width_a,
    )
    points_b = eval_py._feature_to_image_points(
        keypoints_b.index_select(0, matches[:, 1].to(keypoints_b.device)),
        feature_height=descriptors_b.size(2),
        feature_width=descriptors_b.size(3),
        image_height=image_height_b,
        image_width=image_width_b,
    )
    style, gate = infer_style_gate(pair_path)
    return false_match_rows_from_tensors(
        pair_pt=pair_path.as_posix(),
        style=style,
        gate=gate,
        points_a=points_a,
        points_b=points_b,
        scores=match_scores,
        margins=match_margins,
        warp_a_to_b=pair.warp_a_to_b,
        valid_mask=pair.valid_mask,
        threshold_px=threshold_px,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine raw PFM false matches from synthetic cache pairs")
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--pytorch-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["learned", "texture", "blend"], default="blend")
    parser.add_argument("--texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--keypoint-score-mode", choices=["texture", "learned"], default="texture")
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--exclude-self-pairs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model, _ = pfm_model.load_pytorch_state(args.pytorch_state, device=args.device)
    model.eval()
    pair_paths = discover_pair_archives(args.cache_dir, limit_pairs=0, exclude_self_pairs=args.exclude_self_pairs)
    pair_paths = eval_py.limit_pair_paths(pair_paths, limit_pairs=args.limit_pairs, sample_seed=args.sample_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, pair_path in enumerate(pair_paths, 1):
            rows = mine_pair(
                model,
                pair_path,
                device=device,
                mode=args.mode,
                texture_blend_weight=args.texture_blend_weight,
                max_keypoints=args.max_keypoints,
                max_matches=args.max_matches,
                min_intensity=args.min_intensity,
                min_score=args.min_score,
                min_margin=args.min_margin,
                threshold_px=args.threshold_px,
                keypoint_score_mode=args.keypoint_score_mode,
            )
            for row in rows:
                writer.writerow(row.csv_dict())
            total_rows += len(rows)
            handle.flush()
            print(f"[{index}/{len(pair_paths)}] {pair_path} false_matches={len(rows)}", flush=True)
    print(f"pairs={len(pair_paths)} false_matches={total_rows}")
    print(f"false_match_csv={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
