#!/usr/bin/env python3
"""Visualize representative matches from lazy pose-manifest pairs."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for candidate in (PYTHON_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pfm_model  # noqa: E402
import pfm_pytorch_training  # noqa: E402
import pytorch_cache_match_eval as match_eval  # noqa: E402
from benchmark_lazy_pose_pairs import (  # noqa: E402
    CropWindow,
    DEFAULT_PAIR_TYPE_WEIGHTS,
    DEFAULT_PLANET_RADIUS_M,
    DEFAULT_TARGET_VARIANTS,
    LazyPairResult,
    LazyPairSpec,
    RenderRecord,
    _effective_pair_type_weights,
    _read_all_render_records,
    apply_local_contrast_normalization,
    build_lazy_pair_specs,
    count_pair_types,
    generate_lazy_pair,
    parse_int_list,
    parse_pair_type_weights,
    read_pair_spec_manifest,
)
from illumination_stress_eval import make_illumination_variants  # noqa: E402
from patch_descriptor_training import SyntheticPair  # noqa: E402
from training_visual_report import configure_matplotlib_fonts  # noqa: E402


@dataclass(frozen=True)
class LazyMatchVisual:
    label: str
    spec: LazyPairSpec
    pair: SyntheticPair
    valid_fraction: float
    points_a: np.ndarray
    points_b: np.ndarray
    scores: np.ndarray
    errors: np.ndarray
    correct: np.ndarray
    image_name: str = ""
    crop_a: CropWindow | None = None
    crop_b: CropWindow | None = None

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


@dataclass(frozen=True)
class IlluminationStressLazyResult:
    label: str
    variant_name: str
    result: LazyPairResult


def image_points_from_feature_points(
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


def image_to_array(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().to(torch.float32).mean(dim=0).numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(arr, 1.0))
    high = float(np.percentile(arr, 99.0))
    if high <= low:
        high = low + 1.0
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def cpu_pair(pair: SyntheticPair) -> SyntheticPair:
    return SyntheticPair(
        view_a=pair.view_a.detach().cpu(),
        view_b=pair.view_b.detach().cpu(),
        warp_a_to_b=pair.warp_a_to_b.detach().cpu(),
        valid_mask=pair.valid_mask.detach().cpu(),
    )


def move_pair_to_device(pair: SyntheticPair, *, device: torch.device) -> SyntheticPair:
    return SyntheticPair(
        view_a=pair.view_a.to(device=device, non_blocking=True),
        view_b=pair.view_b.to(device=device, non_blocking=True),
        warp_a_to_b=pair.warp_a_to_b.to(device=device, non_blocking=True),
        valid_mask=pair.valid_mask.to(device=device, non_blocking=True),
    )


def compute_visual(
    model: pfm_model.PlanetaryFeatureMatcher,
    result: LazyPairResult,
    *,
    label: str,
    device: torch.device,
    max_image_size: int,
    descriptor_mode: str,
    texture_blend_weight: float,
    keypoint_score_mode: str,
    max_keypoints: int,
    min_intensity: float,
    matcher_mode: str,
    texture_fraction: float,
    weak_texture_fraction: float,
    keypoint_spatial_bins: int,
    keypoint_cell_cap: int,
    topk: int,
    max_matches: int,
    min_score: float,
    min_margin: float,
    graph_dustbin_delta: float,
    graph_acceptance_margin: float,
    graph_min_raw_score: float,
    graph_min_raw_margin: float,
    graph_min_accept_probability: float,
    graph_width_prune_min_score: float,
    graph_early_stop_min_confidence: float,
    graph_max_attention_layers: int,
    graph_max_attention_work_fraction: float,
    graph_width_prune_keep_ratio: float,
    mutual: bool,
    geometry_filter: str,
    input_local_contrast: bool,
    input_local_contrast_strength: float,
    input_local_contrast_kernel: int,
    threshold_px: float,
) -> LazyMatchVisual:
    pair = move_pair_to_device(result.pair, device=device)
    if input_local_contrast:
        pair = apply_local_contrast_normalization(
            pair,
            strength=input_local_contrast_strength,
            kernel_size=input_local_contrast_kernel,
        )
    pair = pfm_pytorch_training.resize_pair_for_training(pair, max_image_size=max_image_size)
    with torch.no_grad():
        descriptors_a, descriptors_b, score_a, score_b, raw_a, raw_b = match_eval.feature_maps_and_keypoint_scores_for_pair(
            model,
            pair,
            mode=descriptor_mode,
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
        if matcher_mode == "graph_matcher":
            row_scores_a = match_eval.gather_score_rows(score_a, selected_a)
            row_scores_b = match_eval.gather_score_rows(score_b, selected_b)
            meta_dim = getattr(getattr(model, "config", None), "graph_keypoint_meta_dim", 2)
            metadata_a = match_eval.graph_metadata_from_raw_features(
                raw_a,
                keypoints_a,
                meta_dim=meta_dim,
                fallback_scores=row_scores_a,
            )
            metadata_b = match_eval.graph_metadata_from_raw_features(
                raw_b,
                keypoints_b,
                meta_dim=meta_dim,
                fallback_scores=row_scores_b,
            )
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
                graph_max_attention_layers=graph_max_attention_layers,
                graph_max_attention_work_fraction=graph_max_attention_work_fraction,
                graph_width_prune_keep_ratio=graph_width_prune_keep_ratio,
                scores_a=row_scores_a,
                scores_b=row_scores_b,
                metadata_a=metadata_a,
                metadata_b=metadata_b,
            )
        elif mutual:
            matches, scores = match_eval.mutual_nearest_matches(
                rows_a,
                rows_b,
                max_matches=max_matches,
                min_score=min_score,
                min_margin=min_margin,
            )
        else:
            matches, scores = match_eval.greedy_unique_matches(
                rows_a,
                rows_b,
                topk=topk,
                max_matches=max_matches,
                min_score=min_score,
            )
        if matches.numel() == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return LazyMatchVisual(
                label=label,
                spec=result.spec,
                pair=cpu_pair(pair),
                valid_fraction=result.valid_fraction,
                points_a=empty,
                points_b=empty,
                scores=np.empty((0,), dtype=np.float32),
                errors=np.empty((0,), dtype=np.float32),
                correct=np.empty((0,), dtype=bool),
                crop_a=result.crop_a,
                crop_b=result.crop_b,
            )
        _, image_height_a, image_width_a = pair.view_a.shape
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
        visual = LazyMatchVisual(
            label=label,
            spec=result.spec,
            pair=cpu_pair(pair),
            valid_fraction=result.valid_fraction,
            points_a=points_a.detach().cpu().numpy(),
            points_b=points_b.detach().cpu().numpy(),
            scores=scores.detach().cpu().numpy(),
            errors=errors.detach().cpu().numpy(),
            correct=correct.detach().cpu().numpy().astype(bool, copy=False),
            crop_a=result.crop_a,
            crop_b=result.crop_b,
        )
        if geometry_filter != "none":
            return filter_visual_matches(visual, geometry_filter=geometry_filter, threshold_px=threshold_px, label=label)
        return visual


def filter_visual_matches(
    result: LazyMatchVisual,
    *,
    geometry_filter: str,
    threshold_px: float,
    label: str | None = None,
) -> LazyMatchVisual:
    if geometry_filter == "none" or result.matches == 0:
        return LazyMatchVisual(
            label=label or f"{result.label} / filtered",
            spec=result.spec,
            pair=result.pair,
            valid_fraction=result.valid_fraction,
            points_a=result.points_a.copy(),
            points_b=result.points_b.copy(),
            scores=result.scores.copy(),
            errors=result.errors.copy(),
            correct=result.correct.copy(),
            image_name=result.image_name,
            crop_a=result.crop_a,
            crop_b=result.crop_b,
        )
    points_a = torch.from_numpy(result.points_a).to(torch.float32)
    points_b = torch.from_numpy(result.points_b).to(torch.float32)
    scores = torch.from_numpy(result.scores).to(torch.float32)
    local_indices = torch.arange(result.matches, dtype=torch.long)
    local_matches = torch.stack([local_indices, local_indices], dim=1)
    if geometry_filter == "affine":
        kept, _ = match_eval.filter_affine_consistent_matches(
            points_a,
            points_b,
            local_matches,
            scores,
            threshold_px=threshold_px,
            min_inliers=4,
        )
    elif geometry_filter == "local":
        kept, _ = match_eval.filter_local_displacement_consistent_matches(
            points_a,
            points_b,
            local_matches,
            scores,
            threshold_px=threshold_px,
            min_inliers=4,
        )
    else:
        raise ValueError(f"unsupported geometry filter: {geometry_filter}")
    keep = kept[:, 0].detach().cpu().numpy().astype(np.int64, copy=False) if kept.numel() > 0 else np.empty(0, dtype=np.int64)
    return LazyMatchVisual(
        label=label or f"{result.label} / filtered",
        spec=result.spec,
        pair=result.pair,
        valid_fraction=result.valid_fraction,
        points_a=result.points_a[keep],
        points_b=result.points_b[keep],
        scores=result.scores[keep],
        errors=result.errors[keep],
        correct=result.correct[keep],
        image_name=result.image_name,
        crop_a=result.crop_a,
        crop_b=result.crop_b,
    )


def selected_draw_indices(result: LazyMatchVisual, draw_matches: int) -> np.ndarray:
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


def draw_match_image(result: LazyMatchVisual, output_path: Path, *, draw_matches: int) -> None:
    image_a = image_to_array(result.pair.view_a)
    image_b = image_to_array(result.pair.view_b)
    height = max(image_a.shape[0], image_b.shape[0])
    width_a = image_a.shape[1]
    width_b = image_b.shape[1]
    canvas = np.zeros((height, width_a + width_b), dtype=np.float32)
    canvas[: image_a.shape[0], :width_a] = image_a
    canvas[: image_b.shape[0], width_a : width_a + width_b] = image_b

    fig, ax = plt.subplots(figsize=(15, 7.5), constrained_layout=True)
    ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    if result.matches > 0:
        indices = selected_draw_indices(result, draw_matches)
        segments = [
            [
                (float(result.points_a[index, 0]), float(result.points_a[index, 1])),
                (float(result.points_b[index, 0] + width_a), float(result.points_b[index, 1])),
            ]
            for index in indices
        ]
        colors = ["#22c55e" if bool(result.correct[index]) else "#ef4444" for index in indices]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.9, alpha=0.76))
        ax.scatter(result.points_a[indices, 0], result.points_a[indices, 1], s=9, c=colors, linewidths=0)
        ax.scatter(result.points_b[indices, 0] + width_a, result.points_b[indices, 1], s=9, c=colors, linewidths=0)
    title = (
        f"{result.label} | {result.spec.reference.base_id} -> {result.spec.target.variant} | "
        f"匹配={result.matches} 正确={result.correct_count} 错误={result.wrong_count} "
        f"precision={result.precision:.3f} 中位误差={result.median_error:.2f}px"
    )
    ax.set_title(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=165)
    plt.close(fig)


def choose_representatives(results: list[LazyMatchVisual], count: int) -> list[LazyMatchVisual]:
    if len(results) <= count:
        return results
    ranked = sorted(results, key=lambda item: (item.precision, item.correct_count, -item.wrong_count), reverse=True)
    picks: list[LazyMatchVisual] = []
    bands = [
        ("高正确率", [0, 1]),
        ("中位样本", [len(ranked) // 2 - 1, len(ranked) // 2]),
        ("困难/失败", [len(ranked) - 2, len(ranked) - 1]),
    ]
    used: set[int] = set()
    for label, indices in bands:
        for index in indices:
            if len(picks) >= count:
                break
            clamped = max(0, min(len(ranked) - 1, index))
            result = ranked[clamped]
            identity = id(result)
            if identity in used:
                continue
            used.add(identity)
            picks.append(
                LazyMatchVisual(
                    label=label,
                    spec=result.spec,
                    pair=result.pair,
                    valid_fraction=result.valid_fraction,
                    points_a=result.points_a,
                    points_b=result.points_b,
                    scores=result.scores,
                    errors=result.errors,
                    correct=result.correct,
                    crop_a=result.crop_a,
                    crop_b=result.crop_b,
                )
            )
    return picks


def make_illumination_stress_lazy_results(selected: list[LazyMatchVisual]) -> list[IlluminationStressLazyResult]:
    stress_results: list[IlluminationStressLazyResult] = []
    for visual in selected:
        for variant_name, view_b in make_illumination_variants(visual.pair.view_b):
            pair = SyntheticPair(
                view_a=visual.pair.view_a.detach().cpu().contiguous(),
                view_b=view_b.detach().cpu().contiguous(),
                warp_a_to_b=visual.pair.warp_a_to_b.detach().cpu().contiguous(),
                valid_mask=visual.pair.valid_mask.detach().cpu().contiguous(),
            )
            stress_results.append(
                IlluminationStressLazyResult(
                    label=f"{visual.label}/{variant_name}",
                    variant_name=variant_name,
                    result=LazyPairResult(
                        spec=visual.spec,
                        pair=pair,
                        valid_fraction=visual.valid_fraction,
                        valid_pixels=int(pair.valid_mask.sum().item()),
                        attempt_count=1,
                        elapsed_ms=0.0,
                        crop_a=visual.crop_a,
                        crop_b=visual.crop_b,
                    ),
                )
            )
    return stress_results


def write_summary_csv(results: list[LazyMatchVisual], path: Path) -> None:
    fields = [
        "label",
        "base_id",
        "target_variant",
        "split",
        "valid_fraction",
        "matches",
        "correct",
        "wrong",
        "precision",
        "mean_error_px",
        "median_error_px",
        "image",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "label": result.label,
                    "base_id": result.spec.reference.base_id,
                    "target_variant": result.spec.target.variant,
                    "split": result.spec.split,
                    "valid_fraction": f"{result.valid_fraction:.6f}",
                    "matches": result.matches,
                    "correct": result.correct_count,
                    "wrong": result.wrong_count,
                    "precision": f"{result.precision:.6f}",
                    "mean_error_px": f"{result.mean_error:.3f}",
                    "median_error_px": f"{result.median_error:.3f}",
                    "image": result.image_name,
                }
            )


def image_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def display_image_path(record) -> str:
    path = record.uint8_path if record.uint8_path is not None else record.image_path
    return str(path)


def format_crop_window(window: CropWindow | None) -> str:
    if window is None:
        return "未记录"
    width = max(0, int(window.x1) - int(window.x0))
    height = max(0, int(window.y1) - int(window.y0))
    return (
        f"x={int(window.x0)}, y={int(window.y0)}, w={width}, h={height} "
        f"(x1={int(window.x1)}, y1={int(window.y1)}, 右下开区间)"
    )


def read_metric_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def metric_series(rows: list[dict[str, float]], *names: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = None
        for name in names:
            if name in row:
                value = row[name]
                break
        values.append(float("nan") if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    window = min(int(window), int(values.size))
    if window <= 1:
        return values
    finite = np.isfinite(values)
    cleaned = np.where(finite, values, 0.0)
    weights = np.convolve(finite.astype(np.float64), np.ones(window), mode="same")
    summed = np.convolve(cleaned, np.ones(window), mode="same")
    return np.divide(summed, np.maximum(weights, 1.0))


def write_training_metric_artifacts(metrics_path: Path, output_dir: Path) -> dict[str, object]:
    rows = read_metric_rows(metrics_path)
    summary: dict[str, object] = {"metrics_path": str(metrics_path), "rows": len(rows)}
    if not rows:
        return summary

    steps = metric_series(rows, "step", "global_step", "iteration", "batch")
    if not np.isfinite(steps).any():
        steps = np.arange(1, len(rows) + 1, dtype=np.float64)
    window = max(5, min(101, len(rows) // 40 if len(rows) >= 200 else 5))
    curves = [
        ("loss", ("loss", "loss_total", "total_loss", "train_loss"), "损失"),
        ("top1", ("descriptor_accuracy", "top1_accuracy", "top1", "mean_top1"), "Top1"),
        ("rank", ("descriptor_positive_rank", "mean_positive_rank", "mean_rank"), "正样本排名"),
        ("gpu", ("gpu_mem_used_mib", "gpu_memory_used_mb", "gpu_mem_used_mb"), "显存"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, (_, names, title) in zip(axes.reshape(-1), curves):
        values = metric_series(rows, *names)
        if np.isfinite(values).any():
            ax.plot(steps, values, color="#7aa2c7", linewidth=0.45, alpha=0.28, label="每 batch")
            ax.plot(steps, smooth_series(values, window), color="#47d5df", linewidth=1.7, label=f"平滑 {window}")
            latest = values[np.isfinite(values)][-1]
            ax.scatter([steps[np.isfinite(values)][-1]], [latest], color="#47d5df", s=22, zorder=3)
            summary[f"latest_{title}"] = float(latest)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.22)
    axes[0, 0].legend(loc="best")
    fig.suptitle("训练指标曲线", fontsize=15)
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)

    loss = metric_series(rows, "loss", "loss_total", "total_loss", "train_loss")
    finite_loss = loss[np.isfinite(loss)]
    if finite_loss.size:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.hist(finite_loss, bins=40, color="#47d5df", alpha=0.78)
        ax.axvline(float(finite_loss[-1]), color="#f59e0b", linewidth=1.4, label="最后一个 batch")
        ax.set_title("训练 loss 分布")
        ax.set_xlabel("loss")
        ax.set_ylabel("batch 数")
        ax.grid(True, axis="y", alpha=0.22)
        ax.legend()
        fig.savefig(output_dir / "loss_histogram.png", dpi=180)
        plt.close(fig)
        summary["loss_mean"] = float(np.mean(finite_loss))
        summary["loss_last"] = float(finite_loss[-1])
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def write_match_error_histogram(results: list[LazyMatchVisual], output_path: Path) -> None:
    correct_errors = []
    wrong_errors = []
    for result in results:
        if result.errors.size == 0:
            continue
        correct_errors.extend(result.errors[result.correct].tolist())
        wrong_errors.extend(result.errors[~result.correct].tolist())
    if not correct_errors and not wrong_errors:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if correct_errors:
        ax.hist(np.clip(correct_errors, 0.0, 100.0), bins=36, color="#22c55e", alpha=0.70, label="正确")
    if wrong_errors:
        ax.hist(np.clip(wrong_errors, 0.0, 100.0), bins=36, color="#ef4444", alpha=0.62, label="错误")
    ax.axvline(5.0, color="#f59e0b", linestyle="--", linewidth=1.2, label="5px 阈值")
    ax.set_title("匹配几何误差直方图")
    ax.set_xlabel("误差 / px，超过 100px 截断显示")
    ax.set_ylabel("匹配数")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_illumination_stress_csv(results: list[LazyMatchVisual], path: Path) -> None:
    fields = [
        "label",
        "base_id",
        "target_variant",
        "matches",
        "correct",
        "wrong",
        "precision",
        "median_error_px",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "label": result.label,
                    "base_id": result.spec.reference.base_id,
                    "target_variant": result.spec.target.variant,
                    "matches": result.matches,
                    "correct": result.correct_count,
                    "wrong": result.wrong_count,
                    "precision": f"{result.precision:.6f}",
                    "median_error_px": f"{result.median_error:.3f}",
                }
            )


def write_illumination_stress_plot(results: list[LazyMatchVisual], output_path: Path) -> None:
    if not results:
        return
    labels = [f"{result.spec.reference.base_id}\n{result.spec.target.variant}\n{result.label}" for result in results]
    values = [result.precision for result in results]
    colors = ["#22c55e" if value >= 0.5 else "#ef4444" for value in values]
    fig_height = max(5.0, min(18.0, 0.34 * len(results)))
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    positions = np.arange(len(results), dtype=np.float32)
    ax.barh(positions, values, color=colors, alpha=0.78)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("正确率")
    ax.set_title("光照压力测试：同一几何 pair 的匹配正确率")
    ax.grid(True, axis="x", alpha=0.22)
    ax.invert_yaxis()
    for position, value in zip(positions, values):
        ax.text(min(0.98, value + 0.015), position, f"{value:.3f}", va="center", fontsize=8, color="#dcebf7")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_html_report(
    path: Path,
    *,
    args: argparse.Namespace,
    all_results: list[LazyMatchVisual],
    selected: list[LazyMatchVisual],
    image_paths: dict[str, Path],
    artifact_paths: dict[str, Path],
    elapsed_s: float,
) -> None:
    total_matches = sum(item.matches for item in all_results)
    total_correct = sum(item.correct_count for item in all_results)
    precision = 0.0 if total_matches == 0 else total_correct / total_matches
    summary = {
        "候选样本数": len(all_results),
        "展示样本数": len(selected),
        "总匹配数": total_matches,
        "正确匹配数": total_correct,
        "整体正确率": round(precision, 6),
        "耗时秒": round(elapsed_s, 2),
        "绿色": "正确匹配",
        "红色": "错误匹配",
    }
    cards = []
    for result in selected:
        image_path = image_paths[result.image_name]
        reference_path = display_image_path(result.spec.reference)
        target_path = display_image_path(result.spec.target)
        reference_crop = format_crop_window(result.crop_a)
        target_crop = format_crop_window(result.crop_b)
        cards.append(
            f"""
<article class="card">
  <h2>{html.escape(result.label)}：{html.escape(result.spec.reference.base_id)} -> {html.escape(result.spec.target.variant)}</h2>
  <p>匹配 {result.matches}，正确 {result.correct_count}，错误 {result.wrong_count}，正确率 {result.precision:.3f}，中位误差 {result.median_error:.2f}px，有效重叠 {result.valid_fraction:.3f}</p>
  <dl class="paths">
    <div><dt>A图文件</dt><dd><code>{html.escape(reference_path)}</code></dd></div>
    <div><dt>A图 crop</dt><dd><code>{html.escape(reference_crop)}</code></dd></div>
    <div><dt>B图文件</dt><dd><code>{html.escape(target_path)}</code></dd></div>
    <div><dt>B图 crop</dt><dd><code>{html.escape(target_crop)}</code></dd></div>
  </dl>
  <img src="{image_data_uri(image_path)}" alt="{html.escape(result.image_name)}">
</article>
"""
        )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item.label)}</td>"
        f"<td>{html.escape(item.spec.reference.base_id)}</td>"
        f"<td>{html.escape(item.spec.target.variant)}</td>"
        f"<td>{item.matches}</td>"
        f"<td>{item.correct_count}</td>"
        f"<td>{item.wrong_count}</td>"
        f"<td>{item.precision:.3f}</td>"
        f"<td>{item.median_error:.2f}</td>"
        "</tr>"
        for item in selected
    )
    artifact_cards = []
    for title, artifact_path in artifact_paths.items():
        if artifact_path.exists():
            artifact_cards.append(
                f"""
<article class="card">
  <h2>{html.escape(title)}</h2>
  <img src="{image_data_uri(artifact_path)}" alt="{html.escape(title)}">
</article>
"""
            )
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>懒加载位姿训练匹配可视化</title>
<style>
body {{ margin: 0; background: #081017; color: #dcebf7; font-family: Arial, "Noto Sans CJK SC", sans-serif; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
.muted {{ color: #91a3b3; }}
.legend {{ display: flex; gap: 18px; margin: 14px 0 20px; color: #b8c8d6; }}
.legend span {{ display: inline-flex; align-items: center; gap: 8px; }}
.dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.green {{ background: #22c55e; }}
.red {{ background: #ef4444; }}
pre {{ background: #101b24; border: 1px solid #223545; border-radius: 8px; padding: 14px; white-space: pre-wrap; }}
.card {{ margin: 18px 0; padding: 16px; background: #101923; border: 1px solid #26394a; border-radius: 8px; }}
.card h2 {{ margin: 0 0 8px; font-size: 18px; }}
.card p {{ margin: 0 0 12px; color: #a9bac8; }}
.paths {{ display: grid; gap: 6px; margin: 0 0 14px; color: #b8c8d6; font-size: 13px; }}
.paths div {{ display: grid; grid-template-columns: 86px minmax(0, 1fr); gap: 10px; align-items: start; }}
.paths dt {{ color: #72dce3; font-weight: 700; }}
.paths dd {{ margin: 0; min-width: 0; }}
.paths code {{ display: block; padding: 6px 8px; border: 1px solid #203242; border-radius: 4px; background: #0a121a; color: #dcebf7; white-space: normal; overflow-wrap: anywhere; word-break: break-all; }}
img {{ display: block; width: 100%; border-radius: 6px; border: 1px solid #1f303e; background: #050a0f; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
th, td {{ border-bottom: 1px solid #223545; padding: 8px; text-align: left; }}
th {{ color: #72dce3; }}
</style>
</head>
<body>
<main>
<h1>懒加载位姿训练匹配可视化</h1>
<p class="muted">checkpoint: {html.escape(str(args.pytorch_state))}</p>
<div class="legend"><span><i class="dot green"></i>绿色：正确匹配</span><span><i class="dot red"></i>红色：错误匹配</span></div>
<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
<details>
<summary>运行参数</summary>
<pre>{html.escape(json.dumps(vars(args), indent=2, ensure_ascii=False, default=str))}</pre>
</details>
<table>
<thead><tr><th>类型</th><th>base</th><th>扰动</th><th>匹配</th><th>正确</th><th>错误</th><th>正确率</th><th>中位误差 px</th></tr></thead>
<tbody>{rows}</tbody>
</table>
{''.join(artifact_cards)}
{''.join(cards)}
</main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def select_visual_pair_specs(
    args: argparse.Namespace,
    records: list[RenderRecord],
) -> tuple[list[LazyPairSpec], str, dict[str, int]]:
    if args.pair_spec_manifest is not None:
        specs = read_pair_spec_manifest(args.pair_spec_manifest, records)
        if args.shuffle:
            rng = random.Random(args.seed)
            rng.shuffle(specs)
        if args.limit_pairs > 0:
            specs = specs[: args.limit_pairs]
        return specs, "pair_spec_manifest", count_pair_types(specs)

    target_variants = tuple(args.target_variant) if args.target_variant else DEFAULT_TARGET_VARIANTS
    cross_variants = tuple(dict.fromkeys(args.cross_pair_variant or (args.reference_variant, *target_variants)))
    effective_pair_type_weights = _effective_pair_type_weights(args.pair_mode, args.pair_type_weights)
    specs, pair_type_counts = build_lazy_pair_specs(
        records,
        split=args.split,
        pair_mode=args.pair_mode,
        reference_variant=args.reference_variant,
        target_variants=target_variants,
        cross_variants=cross_variants,
        cross_camera_offsets=tuple(args.cross_camera_offsets),
        cross_fov_offsets=tuple(args.cross_fov_offsets),
        image_source=args.image_source,
        limit_pairs=args.limit_pairs,
        seed=args.seed,
        shuffle=args.shuffle,
        pair_type_weights=effective_pair_type_weights,
        spatial_index_planet_radius_m=args.spatial_index_planet_radius_m,
        spatial_index_footprint_samples=args.spatial_index_footprint_samples,
        spatial_index_margin_m=args.spatial_index_margin_m,
        spatial_index_height_km=tuple(args.spatial_index_height_km),
    )
    return specs, args.pair_mode, pair_type_counts


def read_visual_records(render_manifest: Path, uint8_manifest: Path) -> list[RenderRecord]:
    return _read_all_render_records([render_manifest], [uint8_manifest])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--uint8-manifest", type=Path, required=True)
    parser.add_argument("--pytorch-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--reference-variant", default="nadir")
    parser.add_argument("--target-variant", action="append", default=[])
    parser.add_argument("--pair-spec-manifest", type=Path, default=None)
    parser.add_argument(
        "--pair-mode",
        choices=["same-position", "cross-camera", "cross-fov", "mixed", "spatial-index"],
        default="same-position",
    )
    parser.add_argument("--cross-camera-offsets", type=parse_int_list, default=parse_int_list("1,2,4,8"))
    parser.add_argument("--cross-fov-offsets", type=parse_int_list, default=parse_int_list("0,1,2,4"))
    parser.add_argument("--cross-pair-variant", action="append", default=[])
    parser.add_argument("--pair-type-weights", type=parse_pair_type_weights, default=DEFAULT_PAIR_TYPE_WEIGHTS.copy())
    parser.add_argument("--spatial-index-planet-radius-m", type=float, default=DEFAULT_PLANET_RADIUS_M)
    parser.add_argument("--spatial-index-footprint-samples", type=int, default=5)
    parser.add_argument("--spatial-index-margin-m", type=float, default=2000.0)
    parser.add_argument("--spatial-index-height-km", type=parse_int_list, default=[])
    parser.add_argument("--image-source", choices=["uint8", "render"], default="uint8")
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-pairs", type=int, default=36)
    parser.add_argument("--select-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--max-image-size", type=int, default=768)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--min-valid-fraction", type=float, default=0.02)
    parser.add_argument("--absolute-depth-tolerance-m", type=float, default=100.0)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--descriptor-mode", choices=["learned", "texture", "blend"], default="learned")
    parser.add_argument("--texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--keypoint-score-mode", choices=["texture", "learned"], default="texture")
    parser.add_argument("--matcher-mode", choices=["raw_descriptor", "graph_matcher"], default="raw_descriptor")
    parser.add_argument("--max-keypoints", type=int, default=512)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--texture-fraction", type=float, default=0.85)
    parser.add_argument("--weak-texture-fraction", type=float, default=0.05)
    parser.add_argument("--keypoint-spatial-bins", type=int, default=12)
    parser.add_argument("--keypoint-cell-cap", type=int, default=6)
    parser.add_argument("--input-local-contrast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-local-contrast-strength", type=float, default=0.0)
    parser.add_argument("--input-local-contrast-kernel", type=int, default=31)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--draw-matches", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--graph-dustbin-delta", type=float, default=0.0)
    parser.add_argument("--graph-acceptance-margin", type=float, default=0.0)
    parser.add_argument("--graph-min-raw-score", type=float, default=-1.0)
    parser.add_argument("--graph-min-raw-margin", type=float, default=0.0)
    parser.add_argument("--graph-min-accept-probability", type=float, default=-1.0)
    parser.add_argument("--graph-width-prune-min-score", type=float, default=-1.0)
    parser.add_argument("--graph-early-stop-min-confidence", type=float, default=-1.0)
    parser.add_argument("--graph-max-attention-layers", type=int, default=0)
    parser.add_argument("--graph-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--graph-width-prune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--mutual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--geometry-filter", choices=["none", "affine", "local"], default="none")
    parser.add_argument("--filtered-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filtered-mutual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filtered-geometry-filter", choices=["none", "affine", "local"], default="local")
    parser.add_argument("--filtered-max-matches", type=int, default=0)
    parser.add_argument("--filtered-draw-matches", type=int, default=0)
    parser.add_argument("--filtered-min-score", type=float, default=-1.0)
    parser.add_argument("--filtered-min-margin", type=float, default=0.02)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--illumination-stress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--illumination-stress-limit", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_matches < 0:
        raise ValueError("--max-matches must be nonnegative; use 0 to keep all matches")
    if args.draw_matches < 0:
        raise ValueError("--draw-matches must be nonnegative; use 0 to draw all matches")
    if args.filtered_max_matches < 0:
        raise ValueError("--filtered-max-matches must be nonnegative; use 0 to keep all matches")
    if args.filtered_draw_matches < 0:
        raise ValueError("--filtered-draw-matches must be nonnegative; use 0 to draw all matches")
    if args.graph_acceptance_margin < 0.0:
        raise ValueError("--graph-acceptance-margin must be nonnegative")
    if args.graph_min_raw_score < -1.0:
        raise ValueError("--graph-min-raw-score must be at least -1.0; -1 disables this filter")
    if args.graph_min_raw_margin < 0.0:
        raise ValueError("--graph-min-raw-margin must be nonnegative")
    if args.graph_min_accept_probability < -1.0 or args.graph_min_accept_probability > 1.0:
        raise ValueError("--graph-min-accept-probability must be in [-1, 1]")
    if args.graph_width_prune_min_score < -1.0:
        raise ValueError("--graph-width-prune-min-score must be at least -1.0; -1 disables pruning")
    if args.graph_early_stop_min_confidence < -1.0:
        raise ValueError("--graph-early-stop-min-confidence must be at least -1.0; -1 disables early stopping")
    if args.graph_max_attention_layers < 0:
        raise ValueError("--graph-max-attention-layers must be nonnegative; use 0 to keep all graph layers")
    if args.graph_max_attention_work_fraction < 0.0 or args.graph_max_attention_work_fraction > 1.0:
        raise ValueError("--graph-max-attention-work-fraction must be in [0, 1]")
    if args.graph_width_prune_keep_ratio < 0.0 or args.graph_width_prune_keep_ratio > 1.0:
        raise ValueError("--graph-width-prune-keep-ratio must be in [0, 1]")
    if args.filtered_min_margin < 0.0:
        raise ValueError("--filtered-min-margin must be nonnegative")
    if args.input_local_contrast_strength < 0.0 or args.input_local_contrast_strength > 1.0:
        raise ValueError("--input-local-contrast-strength must be in [0, 1]")
    start = time.perf_counter()
    configure_matplotlib_fonts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = read_visual_records(args.render_manifest, args.uint8_manifest)
    specs, pair_source, pair_type_counts = select_visual_pair_specs(args, records)
    if not specs:
        raise RuntimeError("no lazy pair specs found")
    specs = specs[: max(args.candidate_pairs, args.select_count)]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, config = pfm_model.load_pytorch_state(args.pytorch_state, device=device, strict=True)
    model.eval()

    all_results: list[LazyMatchVisual] = []
    skipped = 0
    for index, spec in enumerate(specs, 1):
        try:
            lazy_result = generate_lazy_pair(
                spec,
                crop_size=args.crop_size,
                image_source=args.image_source,
                max_attempts=args.max_attempts,
                min_valid_fraction=args.min_valid_fraction,
                absolute_depth_tolerance_m=args.absolute_depth_tolerance_m,
                relative_depth_tolerance=args.relative_depth_tolerance,
                seed=args.seed + index * 97,
            )
            visual = compute_visual(
                model,
                lazy_result,
                label="候选样本",
                device=device,
                max_image_size=args.max_image_size,
                descriptor_mode=args.descriptor_mode,
                texture_blend_weight=args.texture_blend_weight,
                keypoint_score_mode=args.keypoint_score_mode,
                max_keypoints=args.max_keypoints,
                min_intensity=args.min_intensity,
                matcher_mode=args.matcher_mode,
                texture_fraction=args.texture_fraction,
                weak_texture_fraction=args.weak_texture_fraction,
                keypoint_spatial_bins=args.keypoint_spatial_bins,
                keypoint_cell_cap=args.keypoint_cell_cap,
                topk=args.topk,
                max_matches=args.max_matches,
                min_score=args.min_score,
                min_margin=args.min_margin,
                graph_dustbin_delta=args.graph_dustbin_delta,
                graph_acceptance_margin=args.graph_acceptance_margin,
                graph_min_raw_score=args.graph_min_raw_score,
                graph_min_raw_margin=args.graph_min_raw_margin,
                graph_min_accept_probability=args.graph_min_accept_probability,
                graph_width_prune_min_score=args.graph_width_prune_min_score,
                graph_early_stop_min_confidence=args.graph_early_stop_min_confidence,
                graph_max_attention_layers=args.graph_max_attention_layers,
                graph_max_attention_work_fraction=args.graph_max_attention_work_fraction,
                graph_width_prune_keep_ratio=args.graph_width_prune_keep_ratio,
                mutual=args.mutual,
                geometry_filter=args.geometry_filter,
                input_local_contrast=args.input_local_contrast,
                input_local_contrast_strength=args.input_local_contrast_strength,
                input_local_contrast_kernel=args.input_local_contrast_kernel,
                threshold_px=args.threshold_px,
            )
        except Exception as exc:
            skipped += 1
            print(f"skip index={index} reason={exc}", flush=True)
            continue
        all_results.append(visual)
        print(
            f"eval {len(all_results)}/{args.candidate_pairs} "
            f"{spec.reference.base_id}->{spec.target.variant} "
            f"matches={visual.matches} correct={visual.correct_count} precision={visual.precision:.3f}",
            flush=True,
        )
    if not all_results:
        raise RuntimeError(f"all candidate pairs failed, skipped={skipped}")

    raw_selected = choose_representatives(all_results, args.select_count)
    filtered_selected: list[LazyMatchVisual] = []
    if args.filtered_report:
        for result in raw_selected:
            pair = result.pair
            filtered_input = LazyPairResult(
                spec=result.spec,
                pair=pair,
                valid_fraction=result.valid_fraction,
                valid_pixels=int(pair.valid_mask.sum().item()),
                attempt_count=1,
                elapsed_ms=0.0,
                crop_a=result.crop_a,
                crop_b=result.crop_b,
            )
            filtered_selected.append(
                compute_visual(
                    model,
                    filtered_input,
                    label=f"{result.label} / filtered",
                    device=device,
                    max_image_size=args.max_image_size,
                    descriptor_mode=args.descriptor_mode,
                    texture_blend_weight=args.texture_blend_weight,
                    keypoint_score_mode=args.keypoint_score_mode,
                    max_keypoints=args.max_keypoints,
                    min_intensity=args.min_intensity,
                    matcher_mode=args.matcher_mode,
                    texture_fraction=args.texture_fraction,
                    weak_texture_fraction=args.weak_texture_fraction,
                    keypoint_spatial_bins=args.keypoint_spatial_bins,
                    keypoint_cell_cap=args.keypoint_cell_cap,
                    topk=args.topk,
                    max_matches=args.filtered_max_matches,
                    min_score=args.filtered_min_score,
                    min_margin=args.filtered_min_margin,
                    graph_dustbin_delta=args.graph_dustbin_delta,
                    graph_acceptance_margin=args.graph_acceptance_margin,
                    graph_min_raw_score=args.graph_min_raw_score,
                    graph_min_raw_margin=args.graph_min_raw_margin,
                    graph_min_accept_probability=args.graph_min_accept_probability,
                    graph_width_prune_min_score=args.graph_width_prune_min_score,
                    graph_early_stop_min_confidence=args.graph_early_stop_min_confidence,
                    graph_max_attention_layers=args.graph_max_attention_layers,
                    graph_max_attention_work_fraction=args.graph_max_attention_work_fraction,
                    graph_width_prune_keep_ratio=args.graph_width_prune_keep_ratio,
                    mutual=args.filtered_mutual,
                    geometry_filter=args.filtered_geometry_filter,
                    input_local_contrast=args.input_local_contrast,
                    input_local_contrast_strength=args.input_local_contrast_strength,
                    input_local_contrast_kernel=args.input_local_contrast_kernel,
                    threshold_px=args.threshold_px,
                )
            )
    selected = raw_selected + filtered_selected
    image_paths: dict[str, Path] = {}
    for index, result in enumerate(selected, 1):
        image_name = f"{index:02d}_{result.label}_{result.spec.reference.base_id}_{result.spec.target.variant}.png"
        image_name = image_name.replace("/", "_")
        image_path = figures_dir / image_name
        result_with_name = LazyMatchVisual(
            label=result.label,
            spec=result.spec,
            pair=result.pair,
            valid_fraction=result.valid_fraction,
            points_a=result.points_a,
            points_b=result.points_b,
            scores=result.scores,
            errors=result.errors,
            correct=result.correct,
            image_name=image_name,
            crop_a=result.crop_a,
            crop_b=result.crop_b,
        )
        selected[index - 1] = result_with_name
        draw_match_image(result_with_name, image_path, draw_matches=args.draw_matches)
        image_paths[image_name] = image_path

    write_summary_csv(selected, args.output_dir / "summary.csv")
    if filtered_selected:
        write_summary_csv(filtered_selected, args.output_dir / "filtered_summary.csv")
    metadata = {
        "config": {
            "input_channels": config.input_channels,
            "base_channels": config.base_channels,
            "descriptor_dim": config.descriptor_dim,
            "graph_hidden_dim": config.graph_hidden_dim,
            "graph_attention_layers": config.graph_attention_layers,
            "graph_keypoint_meta_dim": config.graph_keypoint_meta_dim,
        },
        "records": len(records),
        "candidate_specs": len(specs),
        "pair_source": pair_source,
        "pair_spec_manifest": str(args.pair_spec_manifest or ""),
        "pair_mode": args.pair_mode,
        "pair_type_counts": pair_type_counts,
        "evaluated": len(all_results),
        "skipped": skipped,
        "graph_inference": {
            "graph_dustbin_delta": float(args.graph_dustbin_delta),
            "graph_acceptance_margin": float(args.graph_acceptance_margin),
            "graph_min_raw_score": float(args.graph_min_raw_score),
            "graph_min_raw_margin": float(args.graph_min_raw_margin),
            "graph_min_accept_probability": float(args.graph_min_accept_probability),
            "graph_width_prune_min_score": float(args.graph_width_prune_min_score),
            "graph_early_stop_min_confidence": float(args.graph_early_stop_min_confidence),
            "graph_max_attention_layers": int(args.graph_max_attention_layers),
            "graph_max_attention_work_fraction": float(args.graph_max_attention_work_fraction),
            "graph_width_prune_keep_ratio": float(args.graph_width_prune_keep_ratio),
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elapsed_s = time.perf_counter() - start
    artifact_paths: dict[str, Path] = {}
    metrics_path = args.metrics_csv
    if metrics_path is None and args.run_dir is not None:
        for candidate in (args.run_dir / "train_metrics.csv", args.run_dir / "metrics.csv"):
            if candidate.exists():
                metrics_path = candidate
                break
    if metrics_path is not None:
        write_training_metric_artifacts(metrics_path, args.output_dir)
        artifact_paths["训练指标曲线"] = args.output_dir / "training_curves.png"
        artifact_paths["Loss 分布"] = args.output_dir / "loss_histogram.png"
    write_match_error_histogram(all_results, args.output_dir / "match_error_histogram.png")
    artifact_paths["匹配误差直方图"] = args.output_dir / "match_error_histogram.png"
    if args.illumination_stress:
        stress_visuals: list[LazyMatchVisual] = []
        for stress_item in make_illumination_stress_lazy_results(raw_selected[: max(0, args.illumination_stress_limit)]):
            try:
                stress_visuals.append(
                    compute_visual(
                        model,
                        stress_item.result,
                        label=stress_item.label,
                        device=device,
                        max_image_size=args.max_image_size,
                        descriptor_mode=args.descriptor_mode,
                        texture_blend_weight=args.texture_blend_weight,
                        keypoint_score_mode=args.keypoint_score_mode,
                        max_keypoints=args.max_keypoints,
                        min_intensity=args.min_intensity,
                        matcher_mode=args.matcher_mode,
                        texture_fraction=args.texture_fraction,
                        weak_texture_fraction=args.weak_texture_fraction,
                        keypoint_spatial_bins=args.keypoint_spatial_bins,
                        keypoint_cell_cap=args.keypoint_cell_cap,
                        topk=args.topk,
                        max_matches=args.max_matches,
                        min_score=args.min_score,
                        min_margin=args.min_margin,
                        graph_dustbin_delta=args.graph_dustbin_delta,
                        graph_acceptance_margin=args.graph_acceptance_margin,
                        graph_min_raw_score=args.graph_min_raw_score,
                        graph_min_raw_margin=args.graph_min_raw_margin,
                        graph_min_accept_probability=args.graph_min_accept_probability,
                        graph_width_prune_min_score=args.graph_width_prune_min_score,
                        graph_early_stop_min_confidence=args.graph_early_stop_min_confidence,
                        graph_max_attention_layers=args.graph_max_attention_layers,
                        graph_max_attention_work_fraction=args.graph_max_attention_work_fraction,
                        graph_width_prune_keep_ratio=args.graph_width_prune_keep_ratio,
                        mutual=args.mutual,
                        geometry_filter=args.geometry_filter,
                        input_local_contrast=args.input_local_contrast,
                        input_local_contrast_strength=args.input_local_contrast_strength,
                        input_local_contrast_kernel=args.input_local_contrast_kernel,
                        threshold_px=args.threshold_px,
                    )
                )
            except Exception as exc:
                print(f"skip illumination stress {stress_item.label} reason={exc}", flush=True)
        if stress_visuals:
            write_illumination_stress_csv(stress_visuals, args.output_dir / "illumination_stress.csv")
            write_illumination_stress_plot(stress_visuals, args.output_dir / "illumination_stress.png")
            artifact_paths["光照压力测试"] = args.output_dir / "illumination_stress.png"
    write_html_report(
        args.output_dir / "index.html",
        args=args,
        all_results=all_results,
        selected=selected,
        image_paths=image_paths,
        artifact_paths=artifact_paths,
        elapsed_s=elapsed_s,
    )
    (args.output_dir / "run.html").write_text((args.output_dir / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"report={args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
