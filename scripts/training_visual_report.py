#!/usr/bin/env python3
"""Generate YOLO-style visual artifacts for a PyTorch PFM training run."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import math
import random
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

import pfm_model
import pfm_pytorch_training
import pose_pair_metadata
import pytorch_cache_match_eval as match_eval
from patch_descriptor_training import SyntheticPair, discover_pair_archives, load_libtorch_pair_archive


CHINESE_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]


def configure_matplotlib_fonts() -> None:
    for font_path in CHINESE_FONT_CANDIDATES:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(path.as_posix())
            family = font_manager.FontProperties(fname=path.as_posix()).get_name()
            plt.rcParams["font.sans-serif"] = [family, "Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


@dataclass(frozen=True)
class VisualMatchResult:
    pair_path: Path
    pair: SyntheticPair
    points_a: np.ndarray
    points_b: np.ndarray
    scores: np.ndarray
    errors: np.ndarray
    correct: np.ndarray
    difficulty: str = "unknown"
    view_angle_deg: float = float("nan")
    overlap_fraction: float = float("nan")
    focal_ratio: float = float("nan")
    coverage_bins: int = 8
    coverage_occupied_fraction: float = 0.0
    max_cell_match_fraction: float = 0.0
    coverage_entropy: float = 0.0
    weak_texture_match_fraction: float = 0.0
    weak_texture_matches: int = 0
    weak_texture_correct: int = 0
    weak_texture_wrong: int = 0
    weak_texture_precision: float = 0.0
    dataset_group: str = "unknown"
    selected_a_count: int = 0
    selected_b_count: int = 0
    selected_a_weak_count: int = 0
    graph_rejected_count: int = 0
    raw_top1_recall: float = 0.0
    raw_top5_recall: float = 0.0
    raw_top16_recall: float = 0.0
    raw_top32_recall: float = 0.0
    raw_top64_recall: float = 0.0
    raw_topk_valid_targets: int = 0

    @property
    def matches(self) -> int:
        return int(self.points_a.shape[0])

    @property
    def correct_count(self) -> int:
        return int(self.correct.sum())

    @property
    def wrong_count(self) -> int:
        return self.matches - self.correct_count

    @property
    def precision(self) -> float:
        return 0.0 if self.matches == 0 else float(self.correct_count) / float(self.matches)

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors.size else 0.0

    @property
    def median_error(self) -> float:
        return float(np.median(self.errors)) if self.errors.size else 0.0

    @property
    def p90_error(self) -> float:
        return float(np.percentile(self.errors, 90.0)) if self.errors.size else 0.0

    @property
    def correct_score_mean(self) -> float:
        if self.scores.size == 0 or not bool(self.correct.any()):
            return 0.0
        return float(np.mean(self.scores[self.correct]))

    @property
    def wrong_score_mean(self) -> float:
        if self.scores.size == 0 or not bool((~self.correct).any()):
            return 0.0
        return float(np.mean(self.scores[~self.correct]))

    @property
    def score_gap(self) -> float:
        return self.correct_score_mean - self.wrong_score_mean


def spatial_coverage_metrics(points_xy: np.ndarray, *, image_height: int, image_width: int, bins: int) -> tuple[float, float, float]:
    if points_xy.size == 0 or bins <= 0:
        return 0.0, 0.0, 0.0
    x_bin = np.clip(np.floor(points_xy[:, 0] * bins / max(1, image_width)).astype(np.int64), 0, bins - 1)
    y_bin = np.clip(np.floor(points_xy[:, 1] * bins / max(1, image_height)).astype(np.int64), 0, bins - 1)
    cell_ids = y_bin * bins + x_bin
    counts = np.bincount(cell_ids, minlength=bins * bins).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    occupied = float(np.count_nonzero(counts)) / float(bins * bins)
    max_cell = float(counts.max()) / total
    probabilities = counts[counts > 0.0] / total
    entropy = float(-(probabilities * np.log(probabilities)).sum() / max(1.0, math.log(float(bins * bins))))
    return occupied, max_cell, entropy


def weak_texture_mask(image: torch.Tensor, points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0,), dtype=bool)
    _, height, width = image.shape
    stride = max(1, min(height, width) // 256)
    yy, xx = torch.meshgrid(torch.arange(0, height, stride), torch.arange(0, width, stride), indexing="ij")
    background_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.float32)
    background_scores = match_eval.image_texture_scores(image.to(torch.float32), background_points)
    if background_scores.numel() == 0:
        return np.zeros((points_xy.shape[0],), dtype=bool)
    threshold = torch.quantile(background_scores, 0.25)
    match_points = torch.from_numpy(points_xy[:, :2]).to(torch.float32)
    match_scores = match_eval.image_texture_scores(image.to(torch.float32), match_points)
    if match_scores.numel() == 0:
        return np.zeros((points_xy.shape[0],), dtype=bool)
    return (match_scores <= threshold).detach().cpu().numpy().astype(bool, copy=False)


def weak_texture_match_fraction(image: torch.Tensor, points_xy: np.ndarray) -> float:
    mask = weak_texture_mask(image, points_xy)
    return 0.0 if mask.size == 0 else float(mask.mean())


def raw_descriptor_topk_recall(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    keypoints_a: torch.Tensor,
    keypoints_b: torch.Tensor,
    pair: SyntheticPair,
    *,
    feature_height_a: int,
    feature_width_a: int,
    feature_height_b: int,
    feature_width_b: int,
    threshold_px: float,
    topks: tuple[int, ...] = (1, 5, 16, 32, 64),
) -> tuple[dict[int, float], int]:
    if desc_a.size(0) == 0 or desc_b.size(0) == 0:
        return {topk: 0.0 for topk in topks}, 0
    _, image_height_a, image_width_a = pair.view_a.shape
    _, image_height_b, image_width_b = pair.view_b.shape
    points_a = image_points_from_feature_points(
        keypoints_a,
        feature_height=feature_height_a,
        feature_width=feature_width_a,
        image_height=image_height_a,
        image_width=image_width_a,
    )
    points_b = image_points_from_feature_points(
        keypoints_b,
        feature_height=feature_height_b,
        feature_width=feature_width_b,
        image_height=image_height_b,
        image_width=image_width_b,
    )
    target_b = match_eval.sample_warp(pair.warp_a_to_b, points_a)
    distances = (points_b.to(target_b.device).unsqueeze(0) - target_b.unsqueeze(1)).norm(dim=2)
    nearest_distance, nearest_target = distances.min(dim=1)
    valid = nearest_distance <= float(threshold_px)
    valid_count = int(valid.sum().detach().cpu().item())
    if valid_count == 0:
        return {topk: 0.0 for topk in topks}, 0
    similarity = match_eval.cyclic_descriptor_similarity(desc_a, desc_b)
    recalls: dict[int, float] = {}
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    for topk in topks:
        k = min(int(topk), similarity.size(1))
        top_targets = similarity.topk(k, dim=1).indices
        hit = top_targets.index_select(0, valid_indices).eq(nearest_target.index_select(0, valid_indices).unsqueeze(1)).any(dim=1)
        recalls[int(topk)] = float(hit.to(torch.float32).mean().detach().cpu().item())
    return recalls, valid_count


def apply_graph_metadata_mode(metadata: torch.Tensor, mode: str) -> torch.Tensor:
    adjusted = metadata.clone()
    if mode == "full":
        return adjusted
    if mode == "descriptor_only":
        return adjusted.zero_()
    if mode == "no_xy":
        adjusted[:, :4] = 0.0
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


def infer_dataset_group(pair_path: Path) -> str:
    text = pair_path.as_posix()
    if "cross_position_rendered_extreme" in text:
        return "extreme_cross_position"
    if "cross_position_rendered" in text or "_cross_" in pair_path.name:
        return "cross_position"
    if "pose_sim_2048" in text:
        return "same_position"
    return "unknown"


def attach_coverage_metrics(result: VisualMatchResult, *, bins: int) -> VisualMatchResult:
    _, image_height, image_width = result.pair.view_a.shape
    occupied, max_cell, entropy = spatial_coverage_metrics(
        result.points_a,
        image_height=image_height,
        image_width=image_width,
        bins=bins,
    )
    weak_mask = weak_texture_mask(result.pair.view_a, result.points_a)
    weak_matches = int(weak_mask.sum())
    if weak_matches > 0:
        weak_correct = int(result.correct[weak_mask].sum())
        weak_wrong = weak_matches - weak_correct
        weak_precision = float(weak_correct) / float(weak_matches)
    else:
        weak_correct = 0
        weak_wrong = 0
        weak_precision = 0.0
    weak_fraction = 0.0 if weak_mask.size == 0 else float(weak_mask.mean())
    return VisualMatchResult(
        pair_path=result.pair_path,
        pair=result.pair,
        points_a=result.points_a,
        points_b=result.points_b,
        scores=result.scores,
        errors=result.errors,
        correct=result.correct,
        difficulty=result.difficulty,
        view_angle_deg=result.view_angle_deg,
        overlap_fraction=result.overlap_fraction,
        focal_ratio=result.focal_ratio,
        coverage_bins=bins,
        coverage_occupied_fraction=occupied,
        max_cell_match_fraction=max_cell,
        coverage_entropy=entropy,
        weak_texture_match_fraction=weak_fraction,
        weak_texture_matches=weak_matches,
        weak_texture_correct=weak_correct,
        weak_texture_wrong=weak_wrong,
        weak_texture_precision=weak_precision,
        dataset_group=result.dataset_group,
        selected_a_count=result.selected_a_count,
        selected_b_count=result.selected_b_count,
        selected_a_weak_count=result.selected_a_weak_count,
        graph_rejected_count=result.graph_rejected_count,
        raw_top1_recall=result.raw_top1_recall,
        raw_top5_recall=result.raw_top5_recall,
        raw_top16_recall=result.raw_top16_recall,
        raw_top32_recall=result.raw_top32_recall,
        raw_top64_recall=result.raw_top64_recall,
        raw_topk_valid_targets=result.raw_topk_valid_targets,
    )


def read_float_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = ""
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def series(rows: list[dict[str, float | str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    finite = np.isfinite(values)
    cleaned = np.where(finite, values, 0.0)
    weights = np.convolve(finite.astype(np.float64), np.ones(window), mode="same")
    summed = np.convolve(cleaned, np.ones(window), mode="same")
    return np.divide(summed, np.maximum(weights, 1.0))


def plot_training_curves(run_dir: Path, output_dir: Path) -> None:
    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval_summary.csv"
    rows = read_float_csv(metrics_path)
    eval_rows = read_float_csv(eval_path) if eval_path.exists() else []
    steps = series(rows, "step")
    if not np.isfinite(steps).any():
        steps = np.arange(1, len(rows) + 1, dtype=np.float64)
    window = max(5, min(75, len(rows) // 20 if len(rows) >= 100 else 5))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(run_dir.name, fontsize=14)

    def draw_line(ax, key: str, label: str, *, raw: bool = True) -> None:
        values = series(rows, key)
        if raw:
            ax.plot(steps, values, color="#94a3b8", linewidth=0.6, alpha=0.35)
        ax.plot(steps, smooth(values, window), linewidth=1.8, label=label)

    ax = axes[0, 0]
    draw_line(ax, "loss", "loss")
    ax.set_title("训练损失")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    for key, label in (
        ("top1_accuracy", "top1"),
        ("top5_accuracy", "top5"),
        ("top10_accuracy", "top10"),
    ):
        draw_line(ax, key, label, raw=False)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("描述子检索准确率")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[0, 2]
    draw_line(ax, "mean_positive_rank", "rank")
    ax.set_title("真实匹配平均排名")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    draw_line(ax, "mean_positive_score", "positive", raw=False)
    draw_line(ax, "mean_negative_score", "negative", raw=False)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("正负样本相似度")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    draw_line(ax, "grad_l2", "grad_l2")
    skipped = series(rows, "skipped")
    if np.nanmax(np.where(np.isfinite(skipped), skipped, 0.0)) > 0.0:
        ax.plot(steps, skipped, color="#ef4444", linewidth=1.0, label="skipped")
        ax.legend()
    ax.set_title("梯度范数")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 2]
    if eval_rows:
        labels = [str(row.get("phase", "")) for row in eval_rows]
        x = np.arange(len(labels))
        width = 0.25
        for offset, key, label in (
            (-width, "top1_accuracy", "top1"),
            (0.0, "top5_accuracy", "top5"),
            (width, "top10_accuracy", "top10"),
        ):
            values = [float(row.get(key, 0.0)) for row in eval_rows]
            ax.bar(x + offset, values, width=width, label=label)
        ax.set_xticks(x, labels)
        ax.set_ylim(0.0, 1.02)
        ax.legend()
    ax.set_title("验证集训练前后")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def image_to_array(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().to(torch.float32).mean(dim=0).numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(arr, 1.0))
    high = float(np.percentile(arr, 99.0))
    if high <= low:
        high = low + 1.0
    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return arr


def image_points_from_feature_points(
    points_xy: torch.Tensor,
    *,
    feature_height: int,
    feature_width: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    x = points_xy[:, 0] * float(max(1, image_width - 1)) / float(max(1, feature_width - 1))
    y = points_xy[:, 1] * float(max(1, image_height - 1)) / float(max(1, feature_height - 1))
    return torch.stack([x, y], dim=1)


def compute_visual_matches(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair_path: Path,
    *,
    device: torch.device,
    pose_metadata: pose_pair_metadata.PoseMetadataIndex | None,
    training_crop_size: int,
    training_max_image_size: int,
    mode: str,
    texture_blend_weight: float,
    keypoint_score_mode: str,
    max_keypoints: int,
    max_matches: int,
    min_intensity: float,
    texture_fraction: float,
    weak_texture_fraction: float,
    keypoint_spatial_bins: int,
    keypoint_cell_cap: int,
    threshold_px: float,
    descriptor_topk: int,
    min_score: float,
    min_margin: float,
    graph_dustbin_delta: float,
    graph_acceptance_margin: float,
    graph_min_raw_score: float,
    graph_min_raw_margin: float,
    graph_min_accept_probability: float,
    graph_width_prune_min_score: float,
    graph_early_stop_min_confidence: float,
    mutual: bool,
    matcher_mode: str,
    graph_metadata_mode: str,
) -> VisualMatchResult:
    pair = load_libtorch_pair_archive(pair_path, device=device)
    pair = pfm_pytorch_training.crop_pair_for_training(pair, crop_size=training_crop_size, generator=None)
    pair = pfm_pytorch_training.resize_pair_for_training(pair, max_image_size=training_max_image_size)
    with torch.no_grad():
        descriptors_a, descriptors_b, score_a, score_b, raw_a, raw_b = match_eval.feature_maps_and_keypoint_scores_for_pair(
            model,
            pair,
            mode=mode,
            texture_blend_weight=texture_blend_weight,
            keypoint_score_mode=keypoint_score_mode,
        )
        keypoints_a, selected_a = match_eval.select_descriptor_keypoints(
            pair.view_a,
            descriptors_a,
            max_keypoints=max_keypoints,
            min_intensity=min_intensity,
            texture_fraction=texture_fraction,
            weak_texture_fraction=weak_texture_fraction,
            spatial_bins=keypoint_spatial_bins,
            keypoint_cell_cap=keypoint_cell_cap,
            keypoint_scores=score_a,
        )
        keypoints_b, selected_b = match_eval.select_descriptor_keypoints(
            pair.view_b,
            descriptors_b,
            max_keypoints=max_keypoints,
            min_intensity=min_intensity,
            texture_fraction=texture_fraction,
            weak_texture_fraction=weak_texture_fraction,
            spatial_bins=keypoint_spatial_bins,
            keypoint_cell_cap=keypoint_cell_cap,
            keypoint_scores=score_b,
        )
        rows_a = match_eval.gather_descriptor_rows(descriptors_a, selected_a)
        rows_b = match_eval.gather_descriptor_rows(descriptors_b, selected_b)
        row_scores_a = match_eval.gather_score_rows(score_a, selected_a)
        row_scores_b = match_eval.gather_score_rows(score_b, selected_b)
        topk_recalls, topk_valid_targets = raw_descriptor_topk_recall(
            rows_a,
            rows_b,
            keypoints_a,
            keypoints_b,
            pair,
            feature_height_a=descriptors_a.size(2),
            feature_width_a=descriptors_a.size(3),
            feature_height_b=descriptors_b.size(2),
            feature_width_b=descriptors_b.size(3),
            threshold_px=threshold_px,
        )
        _, image_height_a, image_width_a = pair.view_a.shape
        selected_a_image_points = image_points_from_feature_points(
            keypoints_a,
            feature_height=descriptors_a.size(2),
            feature_width=descriptors_a.size(3),
            image_height=image_height_a,
            image_width=image_width_a,
        )
        selected_a_weak_count = int(weak_texture_mask(pair.view_a, selected_a_image_points.detach().cpu().numpy()).sum())
        if matcher_mode == "graph_matcher":
            meta_a = match_eval.graph_metadata_from_raw_features(
                raw_a,
                keypoints_a,
                meta_dim=getattr(model.config, "graph_keypoint_meta_dim", 2),
                fallback_scores=row_scores_a,
            )
            meta_b = match_eval.graph_metadata_from_raw_features(
                raw_b,
                keypoints_b,
                meta_dim=getattr(model.config, "graph_keypoint_meta_dim", 2),
                fallback_scores=row_scores_b,
            )
            meta_a = apply_graph_metadata_mode(meta_a, graph_metadata_mode)
            meta_b = apply_graph_metadata_mode(meta_b, graph_metadata_mode)
            matches, scores = match_eval.graph_matcher_matches(
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
                metadata_a=meta_a,
                metadata_b=meta_b,
            )
        elif matcher_mode == "raw_descriptor" and mutual:
            matches, scores = match_eval.mutual_nearest_matches(
                rows_a,
                rows_b,
                max_matches=max_matches,
                min_score=min_score,
                min_margin=min_margin,
            )
        elif matcher_mode == "raw_descriptor":
            matches, scores = match_eval.greedy_unique_matches(
                rows_a,
                rows_b,
                topk=descriptor_topk,
                max_matches=max_matches,
                min_score=min_score,
            )
        else:
            raise ValueError(f"unsupported matcher_mode: {matcher_mode}")
        if matches.numel() == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            metadata = pose_pair_metadata.lookup_pose_metadata(pose_metadata, pair_path)
            return VisualMatchResult(
                pair_path,
                cpu_pair(pair),
                empty,
                empty,
                np.empty((0,)),
                np.empty((0,)),
                np.empty((0,), bool),
                difficulty=metadata.difficulty if metadata is not None else "unknown",
                view_angle_deg=metadata.view_angle_deg if metadata is not None else float("nan"),
                overlap_fraction=metadata.overlap_fraction if metadata is not None else float("nan"),
                focal_ratio=metadata.focal_ratio if metadata is not None else float("nan"),
                dataset_group=infer_dataset_group(pair_path),
                selected_a_count=int(keypoints_a.size(0)),
                selected_b_count=int(keypoints_b.size(0)),
                selected_a_weak_count=selected_a_weak_count,
                graph_rejected_count=int(max(0, keypoints_a.size(0) - matches.size(0))) if matcher_mode == "graph_matcher" else 0,
                raw_top1_recall=topk_recalls[1],
                raw_top5_recall=topk_recalls[5],
                raw_top16_recall=topk_recalls[16],
                raw_top32_recall=topk_recalls[32],
                raw_top64_recall=topk_recalls[64],
                raw_topk_valid_targets=topk_valid_targets,
            )

        _, image_height_b, image_width_b = pair.view_b.shape
        points_a = image_points_from_feature_points(
            keypoints_a.index_select(0, matches[:, 0].to(keypoints_a.device)),
            feature_height=descriptors_a.size(2),
            feature_width=descriptors_a.size(3),
            image_height=image_height_a,
            image_width=image_width_a,
        )
        points_b = image_points_from_feature_points(
            keypoints_b.index_select(0, matches[:, 1].to(keypoints_b.device)),
            feature_height=descriptors_b.size(2),
            feature_width=descriptors_b.size(3),
            image_height=image_height_b,
            image_width=image_width_b,
        )
        target_b = match_eval.sample_warp(pair.warp_a_to_b, points_a)
        errors = (target_b.to(points_b.device) - points_b).norm(dim=1)
        correct = errors <= float(threshold_px)
        metadata = pose_pair_metadata.lookup_pose_metadata(pose_metadata, pair_path)
        return VisualMatchResult(
            pair_path=pair_path,
            pair=cpu_pair(pair),
            points_a=points_a.detach().cpu().numpy(),
            points_b=points_b.detach().cpu().numpy(),
            scores=scores.detach().cpu().numpy(),
            errors=errors.detach().cpu().numpy(),
            correct=correct.detach().cpu().numpy(),
            difficulty=metadata.difficulty if metadata is not None else "unknown",
            view_angle_deg=metadata.view_angle_deg if metadata is not None else float("nan"),
            overlap_fraction=metadata.overlap_fraction if metadata is not None else float("nan"),
            focal_ratio=metadata.focal_ratio if metadata is not None else float("nan"),
            dataset_group=infer_dataset_group(pair_path),
            selected_a_count=int(keypoints_a.size(0)),
            selected_b_count=int(keypoints_b.size(0)),
            selected_a_weak_count=selected_a_weak_count,
            graph_rejected_count=int(max(0, keypoints_a.size(0) - matches.size(0))) if matcher_mode == "graph_matcher" else 0,
            raw_top1_recall=topk_recalls[1],
            raw_top5_recall=topk_recalls[5],
            raw_top16_recall=topk_recalls[16],
            raw_top32_recall=topk_recalls[32],
            raw_top64_recall=topk_recalls[64],
            raw_topk_valid_targets=topk_valid_targets,
        )


def cpu_pair(pair: SyntheticPair) -> SyntheticPair:
    return SyntheticPair(
        view_a=pair.view_a.detach().cpu(),
        view_b=pair.view_b.detach().cpu(),
        warp_a_to_b=pair.warp_a_to_b.detach().cpu(),
        valid_mask=pair.valid_mask.detach().cpu(),
    )


def selected_draw_indices(result: VisualMatchResult, draw_matches: int) -> np.ndarray:
    if draw_matches <= 0 or result.matches <= draw_matches:
        return np.arange(result.matches, dtype=np.int64)
    order = np.argsort(-result.scores)
    wrong = order[~result.correct[order]]
    correct = order[result.correct[order]]
    wrong_take = min(len(wrong), max(4, draw_matches // 3))
    correct_take = min(len(correct), draw_matches - wrong_take)
    chosen = np.concatenate([correct[:correct_take], wrong[:wrong_take]])
    if chosen.size < draw_matches:
        used = set(int(index) for index in chosen)
        fill = [int(index) for index in order if int(index) not in used]
        chosen = np.concatenate([chosen, np.asarray(fill[: draw_matches - chosen.size], dtype=np.int64)])
    return chosen[np.argsort(-result.scores[chosen])]


def draw_match_image(result: VisualMatchResult, output_path: Path, *, draw_matches: int) -> None:
    image_a = image_to_array(result.pair.view_a)
    image_b = image_to_array(result.pair.view_b)
    height = max(image_a.shape[0], image_b.shape[0])
    width_a = image_a.shape[1]
    width_b = image_b.shape[1]
    canvas = np.zeros((height, width_a + width_b), dtype=np.float32)
    canvas[: image_a.shape[0], :width_a] = image_a
    canvas[: image_b.shape[0], width_a : width_a + width_b] = image_b

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    if result.matches > 0:
        indices = selected_draw_indices(result, draw_matches)
        segments = [
            [
                (float(result.points_a[i, 0]), float(result.points_a[i, 1])),
                (float(result.points_b[i, 0] + width_a), float(result.points_b[i, 1])),
            ]
            for i in indices
        ]
        colors = ["#22c55e" if bool(result.correct[i]) else "#ef4444" for i in indices]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.8, alpha=0.75))
        ax.scatter(result.points_a[indices, 0], result.points_a[indices, 1], s=8, c=colors, linewidths=0)
        ax.scatter(result.points_b[indices, 0] + width_a, result.points_b[indices, 1], s=8, c=colors, linewidths=0)
    title = (
        f"{result.pair_path.name} | matches={result.matches} correct={result.correct_count} "
        f"wrong={result.wrong_count} precision={result.precision:.3f} | "
        f"{result.difficulty} angle={result.view_angle_deg:.1f} overlap={result.overlap_fraction:.2f}"
    )
    ax.set_title(title, fontsize=11)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_match_summary(results: list[VisualMatchResult], output_path: Path, *, requested_matches: int) -> None:
    fields = [
        "pair_pt",
        "difficulty",
        "dataset_group",
        "view_angle_deg",
        "overlap_fraction",
        "focal_ratio",
        "matches",
        "correct",
        "wrong",
        "precision",
        "correct_per_requested_match",
        "selected_a_count",
        "selected_b_count",
        "graph_rejected_count",
        "mean_error_px",
        "median_error_px",
        "p90_error_px",
        "correct_score_mean",
        "wrong_score_mean",
        "score_gap",
        "coverage_bins",
        "coverage_occupied_fraction",
        "max_cell_match_fraction",
        "coverage_entropy",
        "weak_texture_match_fraction",
        "weak_texture_matches",
        "weak_texture_correct",
        "weak_texture_wrong",
        "weak_texture_precision",
        "selected_a_weak_count",
        "weak_texture_correct_per_selected_weak",
        "raw_top1_recall",
        "raw_top5_recall",
        "raw_top16_recall",
        "raw_top32_recall",
        "raw_top64_recall",
        "raw_topk_valid_targets",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "pair_pt": result.pair_path.as_posix(),
                    "difficulty": result.difficulty,
                    "dataset_group": result.dataset_group,
                    "view_angle_deg": f"{result.view_angle_deg:.6f}" if math.isfinite(result.view_angle_deg) else "",
                    "overlap_fraction": f"{result.overlap_fraction:.6f}"
                    if math.isfinite(result.overlap_fraction)
                    else "",
                    "focal_ratio": f"{result.focal_ratio:.6f}" if math.isfinite(result.focal_ratio) else "",
                    "matches": result.matches,
                    "correct": result.correct_count,
                    "wrong": result.wrong_count,
                    "precision": f"{result.precision:.6f}",
                    "correct_per_requested_match": f"{result.correct_count / max(1, requested_matches):.6f}",
                    "selected_a_count": result.selected_a_count,
                    "selected_b_count": result.selected_b_count,
                    "graph_rejected_count": result.graph_rejected_count,
                    "mean_error_px": f"{result.mean_error:.3f}",
                    "median_error_px": f"{result.median_error:.3f}",
                    "p90_error_px": f"{result.p90_error:.3f}",
                    "correct_score_mean": f"{result.correct_score_mean:.6f}",
                    "wrong_score_mean": f"{result.wrong_score_mean:.6f}",
                    "score_gap": f"{result.score_gap:.6f}",
                    "coverage_bins": result.coverage_bins,
                    "coverage_occupied_fraction": f"{result.coverage_occupied_fraction:.6f}",
                    "max_cell_match_fraction": f"{result.max_cell_match_fraction:.6f}",
                    "coverage_entropy": f"{result.coverage_entropy:.6f}",
                    "weak_texture_match_fraction": f"{result.weak_texture_match_fraction:.6f}",
                    "weak_texture_matches": result.weak_texture_matches,
                    "weak_texture_correct": result.weak_texture_correct,
                    "weak_texture_wrong": result.weak_texture_wrong,
                    "weak_texture_precision": f"{result.weak_texture_precision:.6f}",
                    "selected_a_weak_count": result.selected_a_weak_count,
                    "weak_texture_correct_per_selected_weak": f"{result.weak_texture_correct / max(1, result.selected_a_weak_count):.6f}",
                    "raw_top1_recall": f"{result.raw_top1_recall:.6f}",
                    "raw_top5_recall": f"{result.raw_top5_recall:.6f}",
                    "raw_top16_recall": f"{result.raw_top16_recall:.6f}",
                    "raw_top32_recall": f"{result.raw_top32_recall:.6f}",
                    "raw_top64_recall": f"{result.raw_top64_recall:.6f}",
                    "raw_topk_valid_targets": result.raw_topk_valid_targets,
                }
            )


def plot_match_summary(results: list[VisualMatchResult], output_path: Path) -> None:
    if not results:
        return
    labels = [f"{index + 1}" for index in range(len(results))]
    correct = np.asarray([result.correct_count for result in results], dtype=np.float64)
    wrong = np.asarray([result.wrong_count for result in results], dtype=np.float64)
    precision = np.asarray([result.precision for result in results], dtype=np.float64)
    x = np.arange(len(results))
    fig, ax1 = plt.subplots(figsize=(max(8, len(results) * 1.1), 5), constrained_layout=True)
    ax1.bar(x, correct, color="#22c55e", label="correct")
    ax1.bar(x, wrong, bottom=correct, color="#ef4444", label="wrong")
    ax1.set_xticks(x, labels)
    ax1.set_xlabel("sample")
    ax1.set_ylabel("匹配点数量")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, precision, color="#2563eb", marker="o", label="precision")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_ylabel("正确率")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("抽样验证对 raw 匹配正确/错误数量")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _concat(values: list[np.ndarray]) -> np.ndarray:
    usable = [value for value in values if value.size > 0]
    if not usable:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(usable).astype(np.float64, copy=False)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def aggregate_match_stats(results: list[VisualMatchResult], *, requested_matches: int) -> dict[str, float]:
    total = sum(result.matches for result in results)
    correct = sum(result.correct_count for result in results)
    wrong = total - correct
    precisions = np.asarray([result.precision for result in results], dtype=np.float64)
    errors = _finite(_concat([result.errors for result in results]))
    score_gaps = np.asarray([result.score_gap for result in results], dtype=np.float64)
    return {
        "samples": float(len(results)),
        "matches": float(total),
        "correct": float(correct),
        "wrong": float(wrong),
        "precision": 0.0 if total == 0 else float(correct) / float(total),
        "mean_matches_per_sample": 0.0 if not results else float(total) / float(len(results)),
        "mean_correct_per_sample": 0.0 if not results else float(correct) / float(len(results)),
        "mean_correct_per_requested": 0.0
        if not results
        else float(correct) / float(max(1, requested_matches) * len(results)),
        "median_precision": float(np.nanmedian(precisions)) if precisions.size else 0.0,
        "mean_error": float(np.mean(errors)) if errors.size else 0.0,
        "median_error": float(np.median(errors)) if errors.size else 0.0,
        "p90_error": float(np.percentile(errors, 90.0)) if errors.size else 0.0,
        "mean_score_gap": float(np.nanmean(score_gaps)) if score_gaps.size else 0.0,
        "mean_coverage_occupied": float(np.mean([result.coverage_occupied_fraction for result in results]))
        if results
        else 0.0,
        "mean_max_cell_fraction": float(np.mean([result.max_cell_match_fraction for result in results])) if results else 0.0,
        "mean_coverage_entropy": float(np.mean([result.coverage_entropy for result in results])) if results else 0.0,
        "mean_weak_texture_fraction": float(np.mean([result.weak_texture_match_fraction for result in results]))
        if results
        else 0.0,
        "mean_weak_texture_precision": float(np.mean([result.weak_texture_precision for result in results])) if results else 0.0,
        "mean_raw_top1_recall": float(np.mean([result.raw_top1_recall for result in results])) if results else 0.0,
        "mean_raw_top5_recall": float(np.mean([result.raw_top5_recall for result in results])) if results else 0.0,
        "mean_raw_top16_recall": float(np.mean([result.raw_top16_recall for result in results])) if results else 0.0,
        "mean_raw_top32_recall": float(np.mean([result.raw_top32_recall for result in results])) if results else 0.0,
        "mean_raw_top64_recall": float(np.mean([result.raw_top64_recall for result in results])) if results else 0.0,
    }


def difficulty_rows(results: list[VisualMatchResult]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for difficulty in ("easy", "medium", "hard", "unknown"):
        group = [result for result in results if result.difficulty == difficulty]
        if not group:
            continue
        matches = sum(result.matches for result in group)
        correct = sum(result.correct_count for result in group)
        rows.append(
            {
                "difficulty": difficulty,
                "samples": float(len(group)),
                "matches": float(matches),
                "correct": float(correct),
                "precision": 0.0 if matches == 0 else float(correct) / float(matches),
                "mean_view_angle": float(np.nanmean([result.view_angle_deg for result in group])),
                "mean_overlap": float(np.nanmean([result.overlap_fraction for result in group])),
            }
        )
    return rows


def plot_match_diagnostics(results: list[VisualMatchResult], output_path: Path) -> None:
    if not results:
        return
    correct_errors = _finite(_concat([result.errors[result.correct] for result in results]))
    wrong_errors = _finite(_concat([result.errors[~result.correct] for result in results]))
    correct_scores = _finite(_concat([result.scores[result.correct] for result in results]))
    wrong_scores = _finite(_concat([result.scores[~result.correct] for result in results]))
    rows = difficulty_rows(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("匹配质量诊断", fontsize=15)

    ax = axes[0, 0]
    error_arrays = []
    labels = []
    colors = []
    if correct_errors.size:
        error_arrays.append(np.clip(correct_errors, 0.0, 100.0))
        labels.append("正确")
        colors.append("#22c55e")
    if wrong_errors.size:
        error_arrays.append(np.clip(wrong_errors, 0.0, 100.0))
        labels.append("错误")
        colors.append("#ef4444")
    if error_arrays:
        ax.hist(error_arrays, bins=30, label=labels, color=colors, alpha=0.75, stacked=False)
    ax.axvline(5.0, color="#111827", linestyle="--", linewidth=1.0, label="5px 阈值")
    ax.set_title("几何误差分布（超过100px截断显示）")
    ax.set_xlabel("误差 / px")
    ax.set_ylabel("匹配数")
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    score_arrays = []
    labels = []
    colors = []
    if correct_scores.size:
        score_arrays.append(correct_scores)
        labels.append("正确")
        colors.append("#22c55e")
    if wrong_scores.size:
        score_arrays.append(wrong_scores)
        labels.append("错误")
        colors.append("#ef4444")
    if score_arrays:
        ax.hist(score_arrays, bins=30, label=labels, color=colors, alpha=0.72)
    ax.set_title("匹配 score 分布")
    ax.set_xlabel("descriptor score")
    ax.set_ylabel("匹配数")
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    if rows:
        labels = [str(row["difficulty"]) for row in rows]
        precision = [float(row["precision"]) for row in rows]
        samples = [float(row["samples"]) for row in rows]
        bars = ax.bar(labels, precision, color=["#16a34a", "#f59e0b", "#dc2626", "#64748b"][: len(rows)])
        for bar, sample_count in zip(bars, samples):
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                bar.get_height() + 0.02,
                f"n={int(sample_count)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylim(0.0, 1.05)
    ax.set_title("难度分组匹配 precision")
    ax.set_xlabel("pose 难度")
    ax.set_ylabel("正确率")
    ax.grid(True, axis="y", alpha=0.2)

    ax = axes[1, 1]
    x = np.asarray([result.correct_count for result in results], dtype=np.float64)
    y = np.asarray([result.precision for result in results], dtype=np.float64)
    size = np.asarray([max(20, result.matches) for result in results], dtype=np.float64)
    colors = {"easy": "#16a34a", "medium": "#f59e0b", "hard": "#dc2626", "unknown": "#64748b"}
    for difficulty, color in colors.items():
        mask = np.asarray([result.difficulty == difficulty for result in results])
        if bool(mask.any()):
            ax.scatter(x[mask], y[mask], s=np.sqrt(size[mask]) * 10.0, c=color, alpha=0.75, label=difficulty)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("每对图像：正确点数量 vs precision")
    ax.set_xlabel("正确匹配点数量")
    ax.set_ylabel("正确率")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_coverage_diagnostics(results: list[VisualMatchResult], output_path: Path) -> None:
    if not results:
        return
    bins = max(1, results[0].coverage_bins)
    aggregate = np.zeros((bins, bins), dtype=np.float64)
    for result in results:
        if result.points_a.size == 0:
            continue
        _, image_height, image_width = result.pair.view_a.shape
        x_bin = np.clip(np.floor(result.points_a[:, 0] * bins / max(1, image_width)).astype(np.int64), 0, bins - 1)
        y_bin = np.clip(np.floor(result.points_a[:, 1] * bins / max(1, image_height)).astype(np.int64), 0, bins - 1)
        for x, y in zip(x_bin, y_bin):
            aggregate[y, x] += 1.0

    labels = [f"{index + 1}" for index in range(len(results))]
    occupied = np.asarray([result.coverage_occupied_fraction for result in results], dtype=np.float64)
    max_cell = np.asarray([result.max_cell_match_fraction for result in results], dtype=np.float64)
    entropy = np.asarray([result.coverage_entropy for result in results], dtype=np.float64)
    weak = np.asarray([result.weak_texture_match_fraction for result in results], dtype=np.float64)
    x = np.arange(len(results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("匹配点覆盖与弱纹理诊断", fontsize=15)

    ax = axes[0, 0]
    image = ax.imshow(aggregate, cmap="magma")
    ax.set_title(f"{bins}x{bins} 网格累计匹配点热力图")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    ax.plot(x, occupied, marker="o", color="#2563eb", label="occupied cells")
    ax.plot(x, entropy, marker="s", color="#16a34a", label="entropy")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("每个样本的空间覆盖")
    ax.set_xlabel("sample")
    ax.set_ylabel("覆盖比例 / 熵")
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    ax.bar(x, max_cell, color="#f59e0b")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, max(0.1, float(max_cell.max(initial=0.0)) * 1.2))
    ax.set_title("单个最密集网格占全部匹配比例")
    ax.set_xlabel("sample")
    ax.set_ylabel("max-cell fraction")
    ax.grid(True, axis="y", alpha=0.2)

    ax = axes[1, 1]
    ax.bar(x, weak, color="#64748b")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("弱纹理区域匹配占比")
    ax.set_xlabel("sample")
    ax.set_ylabel("weak texture match fraction")
    ax.grid(True, axis="y", alpha=0.2)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _format_eval_delta(eval_rows: list[dict[str, float | str]]) -> list[str]:
    if len(eval_rows) < 2:
        return ["没有找到完整 eval_summary.csv，无法比较训练前后验证指标。"]
    before = eval_rows[0]
    after = eval_rows[-1]

    def val(row: dict[str, float | str], key: str) -> float:
        value = row.get(key, 0.0)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0

    return [
        f"验证 loss：{val(before, 'loss'):.4f} -> {val(after, 'loss'):.4f}，变化 {val(after, 'loss') - val(before, 'loss'):+.4f}。",
        f"Top1：{val(before, 'top1_accuracy'):.4f} -> {val(after, 'top1_accuracy'):.4f}，变化 {val(after, 'top1_accuracy') - val(before, 'top1_accuracy'):+.4f}。",
        f"Top5：{val(before, 'top5_accuracy'):.4f} -> {val(after, 'top5_accuracy'):.4f}，变化 {val(after, 'top5_accuracy') - val(before, 'top5_accuracy'):+.4f}。",
        f"负样本平均 score：{val(before, 'mean_negative_score'):.4f} -> {val(after, 'mean_negative_score'):.4f}。",
    ]


def _latest_metric_summary(run_dir: Path) -> list[str]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return ["没有找到 metrics.csv。"]
    rows = read_float_csv(metrics_path)
    if not rows:
        return ["metrics.csv 为空。"]
    loss = series(rows, "loss")
    grad = series(rows, "grad_l2")
    top1 = series(rows, "top1_accuracy")
    finite_loss = _finite(loss)
    finite_grad = _finite(grad)
    finite_top1 = _finite(top1)
    clipped_ratio = 0.0
    if finite_grad.size:
        clipped_ratio = float(np.mean(finite_grad >= np.nanmax(finite_grad) * 0.99))
    return [
        f"训练步数：{len(rows)}。",
        f"训练 loss：首步 {finite_loss[0]:.4f}，末步 {finite_loss[-1]:.4f}，最小 {np.min(finite_loss):.4f}。"
        if finite_loss.size
        else "训练 loss 无有效数值。",
        f"训练 Top1：首步 {finite_top1[0]:.4f}，末步 {finite_top1[-1]:.4f}。"
        if finite_top1.size
        else "训练 Top1 无有效数值。",
        f"梯度范数：中位数 {np.median(finite_grad):.3f}，最大 {np.max(finite_grad):.3f}，接近最大值比例 {clipped_ratio:.2%}。"
        if finite_grad.size
        else "梯度范数无有效数值。",
    ]


def _add_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, list[str]]]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=False)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.axis("off")
    y = 0.98
    ax.text(0.0, y, title, fontsize=22, fontweight="bold", va="top")
    y -= 0.09
    for heading, lines in sections:
        ax.text(0.0, y, heading, fontsize=14, fontweight="bold", va="top")
        y -= 0.045
        for line in lines:
            wrapped = textwrap.wrap(line, width=78, break_long_words=True, replace_whitespace=False) or [""]
            for wrapped_line in wrapped:
                ax.text(0.02, y, wrapped_line, fontsize=10.5, va="top")
                y -= 0.034
            y -= 0.006
        y -= 0.025
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, fontsize=15)
    pdf.savefig(fig)
    plt.close(fig)


def _add_match_gallery(pdf: PdfPages, image_paths: list[Path], *, title: str, per_page: int = 4) -> None:
    for start in range(0, len(image_paths), per_page):
        page_paths = image_paths[start : start + per_page]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
        fig.suptitle(f"{title} {start + 1}-{start + len(page_paths)}", fontsize=15)
        flat_axes = list(axes.flat)
        for ax, path in zip(flat_axes, page_paths):
            ax.imshow(plt.imread(path))
            ax.set_title(path.stem, fontsize=9)
            ax.axis("off")
        for ax in flat_axes[len(page_paths) :]:
            ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def create_pdf_report(
    run_dir: Path,
    output_dir: Path,
    results: list[VisualMatchResult],
    *,
    match_summary_path: Path,
    requested_matches: int,
    pdf_path: Path,
    config_lines: list[str],
) -> None:
    eval_rows = read_float_csv(run_dir / "eval_summary.csv") if (run_dir / "eval_summary.csv").exists() else []
    stats = aggregate_match_stats(results, requested_matches=requested_matches)
    matching_lines = [
        f"抽样图像对：{int(stats['samples'])}；未做几何修复的 mutual 匹配总数：{int(stats['matches'])}；正确：{int(stats['correct'])}；错误：{int(stats['wrong'])}。",
        f"总体 precision：{stats['precision']:.4f}；每对平均匹配点：{stats['mean_matches_per_sample']:.1f}；每对平均正确点：{stats['mean_correct_per_sample']:.1f}。",
        f"每对最多请求 {requested_matches} 个匹配，正确点/请求上限：{stats['mean_correct_per_requested']:.4f}。",
        f"误差中位数：{stats['median_error']:.2f}px；P90：{stats['p90_error']:.2f}px；正确/错误 score 均值差：{stats['mean_score_gap']:.4f}。",
        f"平均网格覆盖率：{stats['mean_coverage_occupied']:.4f}；覆盖熵：{stats['mean_coverage_entropy']:.4f}；最密集网格占比：{stats['mean_max_cell_fraction']:.4f}。",
        f"弱纹理匹配占比：{stats['mean_weak_texture_fraction']:.4f}。",
    ]
    difficulty_lines = []
    for row in difficulty_rows(results):
        difficulty_lines.append(
            f"{row['difficulty']}：样本 {int(float(row['samples']))}，匹配 {int(float(row['matches']))}，"
            f"正确率 {float(row['precision']):.4f}，平均视角差 {float(row['mean_view_angle']):.2f} deg，"
            f"平均重叠 {float(row['mean_overlap']):.3f}。"
        )
    if not difficulty_lines:
        difficulty_lines = ["没有 pose 难度元数据。"]
    diagnosis_lines = [
        "当前模型还不能视为合格版本。训练前后验证 retrieval 有提升，但 raw sparse matching 的正确点数量和稳定性仍不足。",
        "如果新版报告中 matches 已经明显增多但红线仍多，主要问题是 descriptor 区分度和错误匹配抑制；如果 matches 本身也少，说明 keypoint/互近邻过滤后的可用支撑不足。",
        "loss 居高不下且梯度长期贴近裁剪上限时，优先检查学习率、loss 权重、hard negative 强度、样本难度分布，而不是只延长训练。",
    ]
    file_lines = [
        f"PNG/CSV/PDF 目录：{output_dir}",
        f"匹配明细 CSV：{match_summary_path.name}",
        "图中绿色线表示 warp 真值下小于阈值的正确匹配，红色线表示错误匹配；没有使用 RANSAC/Homography 做几何修复。",
    ]
    with PdfPages(pdf_path) as pdf:
        _add_text_page(
            pdf,
            "PlanetaryFeatureMatch 训练报告",
            [
                ("训练指标", _latest_metric_summary(run_dir) + _format_eval_delta(eval_rows)),
                ("匹配质量", matching_lines),
                ("匹配参数", config_lines),
                ("难度分组", difficulty_lines),
                ("诊断", diagnosis_lines),
                ("文件", file_lines),
            ],
        )
        _add_image_page(pdf, output_dir / "training_curves.png", "训练曲线与验证指标")
        _add_image_page(pdf, output_dir / "matching_summary.png", "抽样匹配正确/错误数量")
        _add_image_page(pdf, output_dir / "matching_diagnostics.png", "匹配质量分布诊断")
        _add_image_page(pdf, output_dir / "coverage_diagnostics.png", "匹配点覆盖与弱纹理诊断")
        _add_match_gallery(pdf, sorted((output_dir / "matches").glob("sample_*.png"))[:8], title="匹配样例")


def make_markdown_report(
    run_dir: Path,
    output_dir: Path,
    results: list[VisualMatchResult],
    *,
    match_summary_path: Path,
    pdf_path: Path,
    requested_matches: int,
    config_lines: list[str],
) -> None:
    eval_path = run_dir / "eval_summary.csv"
    stats = aggregate_match_stats(results, requested_matches=requested_matches)
    lines = [
        "# 训练可视化报告",
        "",
        f"Run: `{run_dir}`",
        "",
        "## 文件",
        "",
        f"- `{pdf_path.name}`：中文 PDF 报告。",
        "- `training_curves.png`：训练曲线和训练前后验证指标。",
        "- `matching_summary.png`：抽样验证对的正确/错误匹配数量。",
        "- `matching_diagnostics.png`：误差、score、难度分组和正确点数量诊断。",
        "- `coverage_diagnostics.png`：匹配点空间覆盖、局部热点和弱纹理占比。",
        f"- `{match_summary_path.name}`：逐样本数值指标。",
        "- `matches/*.png`：绿色线为正确匹配，红色线为错误匹配；未使用 RANSAC/Homography 修复，可能包含 score/margin 过滤。",
        "",
    ]
    lines.extend(["## 匹配参数", ""])
    lines.extend(f"- {line}" for line in config_lines)
    lines.append("")
    if eval_path.exists():
        lines.extend(["## 验证指标", ""])
        eval_rows = read_float_csv(eval_path)
        lines.append("| phase | loss | top1 | top5 | top10 | neg score |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in eval_rows:
            lines.append(
                f"| {row.get('phase', '')} | {float(row.get('loss', 0.0)):.4f} | "
                f"{float(row.get('top1_accuracy', 0.0)):.4f} | {float(row.get('top5_accuracy', 0.0)):.4f} | "
                f"{float(row.get('top10_accuracy', 0.0)):.4f} | {float(row.get('mean_negative_score', 0.0)):.4f} |"
            )
        lines.append("")
    if results:
        lines.extend(
            [
                "## 抽样匹配",
                "",
                f"Samples: `{len(results)}`",
                f"Total matches: `{int(stats['matches'])}`",
                f"Correct matches: `{int(stats['correct'])}`",
                f"Precision: `{stats['precision']:.4f}`",
                f"Mean correct/sample: `{stats['mean_correct_per_sample']:.2f}`",
                f"P90 error: `{stats['p90_error']:.2f}px`",
                f"Mean grid coverage: `{stats['mean_coverage_occupied']:.4f}`",
                f"Mean weak-texture match fraction: `{stats['mean_weak_texture_fraction']:.4f}`",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def sample_pair_paths(
    cache_dirs: list[Path],
    *,
    sample_count: int,
    seed: int,
    required_globs: list[str],
) -> list[Path]:
    paths = discover_pair_archives(cache_dirs, limit_pairs=0, exclude_self_pairs=False)
    paths = sorted(dict.fromkeys(paths))
    required: list[Path] = []
    for pattern in required_globs:
        matched = [
            path
            for path in paths
            if fnmatch.fnmatch(str(path), pattern) or fnmatch.fnmatch(path.name, pattern)
        ]
        required.extend(matched)
    required = sorted(dict.fromkeys(required))
    if sample_count <= 0 or len(paths) <= sample_count:
        return paths
    if len(required) >= sample_count:
        return sorted(random.Random(seed).sample(required, sample_count))
    remaining = [path for path in paths if path not in set(required)]
    random_count = max(0, sample_count - len(required))
    if random_count <= 0:
        return required
    return sorted(required + random.Random(seed).sample(remaining, min(random_count, len(remaining))))


def matching_config_lines(args: argparse.Namespace) -> list[str]:
    return [
        f"mode={args.mode}, matcher_mode={args.matcher_mode}, graph_metadata_mode={args.graph_metadata_mode}, keypoint_score_mode={args.keypoint_score_mode}, mutual={int(not args.non_mutual)}。",
        f"max_keypoints={args.max_keypoints}, max_matches={args.max_matches}, draw_matches={args.draw_matches}。",
        f"texture_fraction={args.texture_keypoint_fraction:.3f}, weak_texture_fraction={args.weak_texture_keypoint_fraction:.3f}, spatial_bins={args.keypoint_spatial_bins}, cell_cap={args.keypoint_cell_cap}。",
        f"min_score={args.min_score:.6f}, min_margin={args.min_margin:.6f}, threshold_px={args.threshold_px:.2f}。",
        f"graph_dustbin_delta={args.graph_dustbin_delta:.6f}, graph_acceptance_margin={args.graph_acceptance_margin:.6f}, graph_min_raw_score={args.graph_min_raw_score:.6f}, graph_min_raw_margin={args.graph_min_raw_margin:.6f}。",
        f"graph_inference_preset={args.graph_inference_preset}, graph_min_accept_probability={args.graph_min_accept_probability:.6f}, graph_width_prune_min_score={args.graph_width_prune_min_score:.6f}, graph_early_stop_min_confidence={args.graph_early_stop_min_confidence:.6f}。",
        f"training_crop_size={args.training_crop_size}, training_max_image_size={args.training_max_image_size}。",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual charts and match examples for a PFM training run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", action="append", type=Path, required=True)
    parser.add_argument("--pose-metadata-root", action="append", type=Path, default=[])
    parser.add_argument("--pytorch-state", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pdf-report-name", default="training_report_zh.pdf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=20260527)
    parser.add_argument(
        "--required-sample-glob",
        action="append",
        default=[],
        help="Glob matched against full pair path or filename; matched pairs are always included in the report sample.",
    )
    parser.add_argument("--training-crop-size", type=int, default=1024)
    parser.add_argument("--training-max-image-size", type=int, default=0)
    parser.add_argument("--mode", choices=["learned", "texture", "blend"], default="blend")
    parser.add_argument("--texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--keypoint-score-mode", choices=["texture", "learned"], default="texture")
    parser.add_argument("--matcher-mode", choices=["raw_descriptor", "graph_matcher"], default="raw_descriptor")
    parser.add_argument(
        "--graph-metadata-mode",
        choices=["full", "descriptor_only", "no_xy", "no_geometry", "no_quality"],
        default="full",
    )
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--draw-matches", type=int, default=0)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--texture-keypoint-fraction", type=float, default=1.0)
    parser.add_argument("--weak-texture-keypoint-fraction", type=float, default=0.0)
    parser.add_argument("--keypoint-spatial-bins", type=int, default=8)
    parser.add_argument("--keypoint-cell-cap", type=int, default=0)
    parser.add_argument("--coverage-bins", type=int, default=8)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--descriptor-topk", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--graph-dustbin-delta", type=float, default=0.0)
    parser.add_argument("--graph-acceptance-margin", type=float, default=0.0)
    parser.add_argument("--graph-min-raw-score", type=float, default=-1.0)
    parser.add_argument("--graph-min-raw-margin", type=float, default=0.0)
    parser.add_argument("--graph-min-accept-probability", type=float, default=-1.0)
    parser.add_argument("--graph-inference-preset", choices=sorted(match_eval.GRAPH_INFERENCE_PRESETS), default="off")
    parser.add_argument("--graph-width-prune-min-score", type=float, default=-1.0)
    parser.add_argument("--graph-early-stop-min-confidence", type=float, default=-1.0)
    parser.add_argument("--non-mutual", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    args.pytorch_state = args.pytorch_state or args.run_dir / "pytorch_pfm_state.pt"
    args.output_dir = args.output_dir or args.run_dir / "visual_report"
    if args.sample_count < 0:
        parser.error("--sample-count must be nonnegative")
    if args.max_keypoints <= 0:
        parser.error("--max-keypoints must be positive")
    if args.max_matches < 0:
        parser.error("--max-matches must be nonnegative; use 0 to keep all matches")
    if args.draw_matches < 0:
        parser.error("--draw-matches must be nonnegative; use 0 to draw all matches")
    if args.texture_keypoint_fraction < 0.0 or args.texture_keypoint_fraction > 1.0:
        parser.error("--texture-keypoint-fraction must be in [0, 1]")
    if args.weak_texture_keypoint_fraction < 0.0 or args.weak_texture_keypoint_fraction > 1.0:
        parser.error("--weak-texture-keypoint-fraction must be in [0, 1]")
    if args.texture_keypoint_fraction + args.weak_texture_keypoint_fraction > 1.0:
        parser.error("--texture-keypoint-fraction + --weak-texture-keypoint-fraction must be <= 1")
    if args.keypoint_cell_cap < 0:
        parser.error("--keypoint-cell-cap must be nonnegative")
    if args.coverage_bins <= 0:
        parser.error("--coverage-bins must be positive")
    if args.graph_width_prune_min_score < -1.0 or args.graph_width_prune_min_score > 1.0:
        parser.error("--graph-width-prune-min-score must be in [-1, 1]")
    if args.graph_early_stop_min_confidence < -1.0 or args.graph_early_stop_min_confidence > 1.0:
        parser.error("--graph-early-stop-min-confidence must be in [-1, 1]")
    if args.graph_min_accept_probability < -1.0 or args.graph_min_accept_probability > 1.0:
        parser.error("--graph-min-accept-probability must be in [-1, 1]")
    return args


def main() -> int:
    args = parse_args()
    args.graph_width_prune_min_score, args.graph_early_stop_min_confidence = match_eval.graph_inference_thresholds(
        args.graph_inference_preset,
        args.graph_width_prune_min_score,
        args.graph_early_stop_min_confidence,
    )
    configure_matplotlib_fonts()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    match_dir = args.output_dir / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(args.run_dir, args.output_dir)
    model, _ = pfm_model.load_pytorch_state(args.pytorch_state, device=args.device)
    model.eval()
    device = torch.device(args.device)
    pose_metadata: pose_pair_metadata.PoseMetadataIndex = {}
    pose_roots = pose_pair_metadata.infer_pose_metadata_roots(args.validation_cache_dir, args.pose_metadata_root)
    for pose_root in pose_roots:
        pose_metadata.update(pose_pair_metadata.load_pose_metadata_index(pose_root))
    if pose_roots:
        print(f"pose_metadata_roots={len(pose_roots)} pose_metadata_pairs={len(pose_metadata)}", flush=True)
    pair_paths = sample_pair_paths(
        args.validation_cache_dir,
        sample_count=args.sample_count,
        seed=args.sample_seed,
        required_globs=args.required_sample_glob,
    )
    results: list[VisualMatchResult] = []
    for index, pair_path in enumerate(pair_paths, 1):
        result = compute_visual_matches(
            model,
            pair_path,
            device=device,
            pose_metadata=pose_metadata,
            training_crop_size=args.training_crop_size,
            training_max_image_size=args.training_max_image_size,
            mode=args.mode,
            texture_blend_weight=args.texture_blend_weight,
            keypoint_score_mode=args.keypoint_score_mode,
            max_keypoints=args.max_keypoints,
            max_matches=args.max_matches,
            min_intensity=args.min_intensity,
            texture_fraction=args.texture_keypoint_fraction,
            weak_texture_fraction=args.weak_texture_keypoint_fraction,
            keypoint_spatial_bins=args.keypoint_spatial_bins,
            keypoint_cell_cap=args.keypoint_cell_cap,
            threshold_px=args.threshold_px,
            descriptor_topk=args.descriptor_topk,
            min_score=args.min_score,
            min_margin=args.min_margin,
            graph_dustbin_delta=args.graph_dustbin_delta,
            graph_acceptance_margin=args.graph_acceptance_margin,
            graph_min_raw_score=args.graph_min_raw_score,
            graph_min_raw_margin=args.graph_min_raw_margin,
            graph_min_accept_probability=args.graph_min_accept_probability,
            graph_width_prune_min_score=args.graph_width_prune_min_score,
            graph_early_stop_min_confidence=args.graph_early_stop_min_confidence,
            mutual=not args.non_mutual,
            matcher_mode=args.matcher_mode,
            graph_metadata_mode=args.graph_metadata_mode,
        )
        result = attach_coverage_metrics(result, bins=args.coverage_bins)
        results.append(result)
        draw_match_image(result, match_dir / f"sample_{index:02d}.png", draw_matches=args.draw_matches)
        print(
            f"[{index}/{len(pair_paths)}] {pair_path.name} matches={result.matches} "
            f"correct={result.correct_count} precision={result.precision:.4f}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    match_summary_path = args.output_dir / "match_visual_summary.csv"
    config_lines = matching_config_lines(args)
    write_match_summary(results, match_summary_path, requested_matches=args.max_matches)
    plot_match_summary(results, args.output_dir / "matching_summary.png")
    plot_match_diagnostics(results, args.output_dir / "matching_diagnostics.png")
    plot_coverage_diagnostics(results, args.output_dir / "coverage_diagnostics.png")
    (args.output_dir / "report_config.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    pdf_path = args.output_dir / args.pdf_report_name
    if not args.no_pdf:
        create_pdf_report(
            args.run_dir,
            args.output_dir,
            results,
            match_summary_path=match_summary_path,
            requested_matches=args.max_matches,
            pdf_path=pdf_path,
            config_lines=config_lines,
        )
    make_markdown_report(
        args.run_dir,
        args.output_dir,
        results,
        match_summary_path=match_summary_path,
        pdf_path=pdf_path,
        requested_matches=args.max_matches,
        config_lines=config_lines,
    )
    print(f"visual_report={args.output_dir}")
    if not args.no_pdf:
        print(f"pdf_report={pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
