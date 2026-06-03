#!/usr/bin/env python3
"""Orchestrate 1024-cache cross-view PFM training and grouped evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import torch  # noqa: E402

import cache_split  # noqa: E402
import pfm_model  # noqa: E402
import pytorch_cache_match_eval as eval_py  # noqa: E402
from patch_descriptor_training import discover_pair_archives, load_libtorch_pair_archive  # noqa: E402


STYLES = ("numeric", "timestamp")
GATES = ("rotate", "viewpoint", "compound")
DEFAULT_BLEND_WEIGHT_CANDIDATES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0)
DEFAULT_MATCH_MARGIN_CANDIDATES = (0.0, 0.01, 0.02)
DEFAULT_TARGET_GRADIENT_CANDIDATES = (0.0, 20.0, 22.0, 24.0)
DEFAULT_TARGET_LOCAL_CONTRAST_CANDIDATES = (0.0, 5.0, 5.32, 6.0)
DEFAULT_KEYPOINT_SCORE_MODE_CANDIDATES = ("texture", "learned")
KEYPOINT_SCORE_MODES = ("texture", "learned")


@dataclass(frozen=True)
class EvalGroup:
    style: str
    gate: str
    cache_dir: Path


@dataclass(frozen=True)
class BlendWeightSummary:
    style: str
    gate: str
    texture_blend_weight: float
    matches: int
    correct: int
    precision: float
    summary_csv: Path
    min_margin: float = 0.0
    min_target_gradient: float = 0.0
    min_target_local_contrast: float = 0.0
    keypoint_score_mode: str = "texture"
    pytorch_state_label: str = "trained"
    pytorch_state: Path | None = None


def build_training_command(
    *,
    python_exe: Path,
    project_root: Path,
    train_cache_dirs: Iterable[Path],
    validation_cache_dirs: Iterable[Path],
    output_dir: Path,
    checkpoint: Path | None,
    init_pytorch_state: Path | None,
    init_random: bool,
    device: str,
    steps: int,
    batch_pairs: int,
    samples_per_pair: int,
    learning_rate: float,
    gradient_accumulation_steps: int = 1,
    balanced_cache_sampling: bool = False,
    training_texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    training_eval_pairs: int = 0,
    synthetic_loss_weight: float = 1.0,
    warp_hard_negative_weight: float = 0.0,
    warp_hard_negative_radius: float = 2.0,
    warp_hard_negative_margin: float = 0.2,
    warp_hard_negative_candidates: int = 4096,
    abstention_weight: float = 0.0,
    abstention_negative_radius: float = 2.0,
    abstention_max_false_score: float = 0.35,
    abstention_topk: int = 8,
    abstention_candidates: int = 4096,
    hard_summaries: Iterable[Path] | None = None,
    hard_limit: int = 64,
    hard_min_matches: int = 4,
    hard_max_precision: float = 0.9,
    hard_repeat: int = 3,
    hard_curriculum_max_probability: float = 0.0,
    hard_curriculum_warmup_steps: int = 100,
    pseudo_label_csvs: Iterable[Path] | None = None,
    pseudo_label_weight: float = 0.0,
    pseudo_keypoint_weight: float = 0.0,
    pseudo_keypoint_negative_weight: float = 0.01,
    pseudo_label_max_points: int = 128,
    pseudo_label_curriculum_max_probability: float = 0.0,
    pseudo_label_curriculum_warmup_steps: int = 100,
    false_match_csvs: Iterable[Path] | None = None,
    false_match_weight: float = 0.0,
    false_match_max_points: int = 128,
    false_match_max_score: float = 0.25,
    false_match_curriculum_max_probability: float = 0.0,
    false_match_curriculum_warmup_steps: int = 100,
) -> list[str]:
    if checkpoint is None and init_pytorch_state is None and not init_random:
        raise ValueError("checkpoint, init_pytorch_state, or init_random is required")
    command = [str(python_exe), str(project_root / "python" / "pfm_pytorch_training.py")]
    if init_random:
        command.append("--init-random")
    elif init_pytorch_state is not None:
        command.extend(["--checkpoint", str(checkpoint if checkpoint is not None else init_pytorch_state)])
        command.extend(["--init-pytorch-state", str(init_pytorch_state)])
    else:
        command.extend(["--checkpoint", str(checkpoint)])
    for cache_dir in train_cache_dirs:
        command.extend(["--cache-dir", str(cache_dir)])
    for cache_dir in validation_cache_dirs:
        command.extend(["--validation-cache-dir", str(cache_dir)])
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--device",
            device,
            "--steps",
            str(steps),
            "--batch-pairs",
            str(batch_pairs),
            "--gradient-accumulation-steps",
            str(gradient_accumulation_steps),
            "--samples-per-pair",
            str(samples_per_pair),
            "--learning-rate",
            f"{learning_rate:.12g}",
            "--synthetic-loss-weight",
            f"{synthetic_loss_weight:.12g}",
            "--exclude-self-pairs",
            "--train-blended-descriptors",
            "--training-texture-blend-weight",
            f"{training_texture_blend_weight:.12g}",
            "--skip-nonfinite-steps",
        ]
    )
    if balanced_cache_sampling:
        command.append("--balanced-cache-sampling")
    if training_eval_pairs > 0:
        command.extend(["--eval-pairs", str(training_eval_pairs)])
    if warp_hard_negative_weight > 0.0:
        command.extend(["--warp-hard-negative-weight", f"{warp_hard_negative_weight:.12g}"])
        command.extend(["--warp-hard-negative-radius", f"{warp_hard_negative_radius:.12g}"])
        command.extend(["--warp-hard-negative-margin", f"{warp_hard_negative_margin:.12g}"])
        command.extend(["--warp-hard-negative-candidates", str(warp_hard_negative_candidates)])
    if abstention_weight > 0.0:
        command.extend(["--abstention-weight", f"{abstention_weight:.12g}"])
        command.extend(["--abstention-negative-radius", f"{abstention_negative_radius:.12g}"])
        command.extend(["--abstention-max-false-score", f"{abstention_max_false_score:.12g}"])
        command.extend(["--abstention-topk", str(abstention_topk)])
        command.extend(["--abstention-candidates", str(abstention_candidates)])
    summary_paths = list(hard_summaries or [])
    if summary_paths:
        for summary in summary_paths:
            command.extend(["--hard-summary", str(summary)])
        command.extend(["--hard-limit", str(hard_limit)])
        command.extend(["--hard-min-matches", str(hard_min_matches)])
        command.extend(["--hard-max-precision", f"{hard_max_precision:.12g}"])
        command.extend(["--hard-repeat", str(hard_repeat)])
        command.extend(["--hard-curriculum-max-probability", f"{hard_curriculum_max_probability:.12g}"])
        command.extend(["--hard-curriculum-warmup-steps", str(hard_curriculum_warmup_steps)])
    pseudo_paths = list(pseudo_label_csvs or [])
    if pseudo_paths:
        for pseudo_csv in pseudo_paths:
            command.extend(["--pseudo-label-csv", str(pseudo_csv)])
        command.extend(["--pseudo-label-weight", f"{pseudo_label_weight:.12g}"])
        command.extend(["--pseudo-keypoint-weight", f"{pseudo_keypoint_weight:.12g}"])
        command.extend(["--pseudo-keypoint-negative-weight", f"{pseudo_keypoint_negative_weight:.12g}"])
        command.extend(["--pseudo-label-max-points", str(pseudo_label_max_points)])
        command.extend(["--pseudo-label-curriculum-max-probability", f"{pseudo_label_curriculum_max_probability:.12g}"])
        command.extend(["--pseudo-label-curriculum-warmup-steps", str(pseudo_label_curriculum_warmup_steps)])
    false_paths = list(false_match_csvs or [])
    if false_paths:
        for false_csv in false_paths:
            command.extend(["--false-match-csv", str(false_csv)])
        command.extend(["--false-match-weight", f"{false_match_weight:.12g}"])
        command.extend(["--false-match-max-points", str(false_match_max_points)])
        command.extend(["--false-match-max-score", f"{false_match_max_score:.12g}"])
        command.extend(["--false-match-curriculum-max-probability", f"{false_match_curriculum_max_probability:.12g}"])
        command.extend(["--false-match-curriculum-warmup-steps", str(false_match_curriculum_warmup_steps)])
    return command


def parse_blend_weight_candidates(text: str) -> list[float]:
    candidates: list[float] = []
    seen: set[float] = set()
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0.0:
            raise ValueError("blend weight candidates must be non-negative")
        if value in seen:
            continue
        candidates.append(value)
        seen.add(value)
    if not candidates:
        raise ValueError("at least one blend weight candidate is required")
    return candidates


def parse_match_margin_candidates(text: str) -> list[float]:
    candidates: list[float] = []
    seen: set[float] = set()
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0.0:
            raise ValueError("match margin candidates must be non-negative")
        if value in seen:
            continue
        candidates.append(value)
        seen.add(value)
    if not candidates:
        raise ValueError("at least one match margin candidate is required")
    return candidates


def parse_target_gradient_candidates(text: str) -> list[float]:
    candidates: list[float] = []
    seen: set[float] = set()
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0.0:
            raise ValueError("target gradient candidates must be non-negative")
        if value in seen:
            continue
        candidates.append(value)
        seen.add(value)
    if not candidates:
        raise ValueError("at least one target gradient candidate is required")
    return candidates


def parse_target_local_contrast_candidates(text: str) -> list[float]:
    candidates: list[float] = []
    seen: set[float] = set()
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0.0:
            raise ValueError("target local contrast candidates must be non-negative")
        if value in seen:
            continue
        candidates.append(value)
        seen.add(value)
    if not candidates:
        raise ValueError("at least one target local contrast candidate is required")
    return candidates


def parse_keypoint_score_mode_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for item in text.split(","):
        value = item.strip()
        if not value:
            continue
        if value not in KEYPOINT_SCORE_MODES:
            raise ValueError(f"unsupported keypoint score mode: {value}")
        if value in seen:
            continue
        candidates.append(value)
        seen.add(value)
    if not candidates:
        raise ValueError("at least one keypoint score mode candidate is required")
    return candidates


def parse_sample_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value in seen:
            continue
        seeds.append(value)
        seen.add(value)
    if not seeds:
        raise ValueError("at least one sample seed is required")
    return seeds


def parse_group_keys(text: str) -> set[tuple[str, str]]:
    groups: set[tuple[str, str]] = set()
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parts = stripped.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("group keys must use style/gate")
        groups.add((parts[0], parts[1]))
    if not groups:
        raise ValueError("at least one group key is required")
    return groups


def parse_calibration_pytorch_state_entries(entries: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for entry in entries:
        if "=" not in entry:
            raise ValueError("calibration pytorch states must use label=path")
        label, path_text = entry.split("=", 1)
        label = label.strip()
        path_text = path_text.strip()
        if not label or not path_text:
            raise ValueError("calibration pytorch states must use non-empty label=path")
        if any(char in label for char in "/\\"):
            raise ValueError("calibration pytorch state labels must not contain path separators")
        if label in seen:
            raise ValueError(f"duplicate calibration pytorch state label: {label}")
        parsed.append((label, Path(path_text)))
        seen.add(label)
    return parsed


def select_created_cache_dirs(
    created: dict[tuple[str, str, str], Path],
    *,
    split: str,
    groups: set[tuple[str, str]] | None = None,
) -> list[Path]:
    return [
        path
        for (entry_split, style, gate), path in sorted(created.items())
        if entry_split == split and (groups is None or (style, gate) in groups)
    ]


def blend_weight_label(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def match_margin_label(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def target_gradient_label(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def target_local_contrast_label(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def keypoint_score_mode_label(value: str) -> str:
    return value.replace(".", "p").replace(" ", "_")


def state_label(value: str) -> str:
    return value.replace(".", "p").replace(" ", "_")


def read_match_summary_csv(path: Path) -> tuple[int, int, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = sum(int(row["matches"]) for row in rows)
    correct = sum(int(row["correct"]) for row in rows)
    precision = 0.0 if matches == 0 else correct / matches
    return matches, correct, precision


def select_best_blend_weight_summaries(
    summaries: Iterable[BlendWeightSummary],
    *,
    min_matches: int = 0,
    min_match_fraction: float = 0.0,
    state_switch_reference_label: str = "trained",
    state_switch_min_precision_gain: float = 0.0,
    state_switch_min_match_ratio: float = 0.0,
) -> dict[tuple[str, str], BlendWeightSummary]:
    if min_matches < 0:
        raise ValueError("min_matches must be non-negative")
    if min_match_fraction < 0.0:
        raise ValueError("min_match_fraction must be non-negative")
    if state_switch_min_precision_gain < 0.0:
        raise ValueError("state_switch_min_precision_gain must be non-negative")
    if state_switch_min_match_ratio < 0.0:
        raise ValueError("state_switch_min_match_ratio must be non-negative")
    grouped: dict[tuple[str, str], list[BlendWeightSummary]] = {}
    for summary in summaries:
        grouped.setdefault((summary.style, summary.gate), []).append(summary)
    selected: dict[tuple[str, str], BlendWeightSummary] = {}
    for key, group_summaries in grouped.items():
        max_matches = max(summary.matches for summary in group_summaries)
        support_floor = max(min_matches, math.ceil(max_matches * min_match_fraction))
        eligible = [summary for summary in group_summaries if summary.matches >= support_floor]
        if not eligible:
            eligible = group_summaries
        best = max(eligible, key=lambda summary: (summary.precision, summary.correct, summary.matches))
        if (
            best.pytorch_state_label != state_switch_reference_label
            and (state_switch_min_precision_gain > 0.0 or state_switch_min_match_ratio > 0.0)
        ):
            reference_candidates = [
                summary for summary in eligible if summary.pytorch_state_label == state_switch_reference_label
            ]
            if reference_candidates:
                reference = max(
                    reference_candidates,
                    key=lambda summary: (summary.precision, summary.correct, summary.matches),
                )
                enough_precision = best.precision >= reference.precision + state_switch_min_precision_gain
                enough_support = best.matches >= math.ceil(reference.matches * state_switch_min_match_ratio)
                if not (enough_precision and enough_support):
                    best = reference
        selected[key] = best
    return selected


def write_blend_weight_summaries(path: Path, summaries: Iterable[BlendWeightSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "style",
                "gate",
                "texture_blend_weight",
                "keypoint_score_mode",
                "min_margin",
                "min_target_gradient",
                "min_target_local_contrast",
                "pytorch_state_label",
                "pytorch_state",
                "matches",
                "correct",
                "precision",
                "summary_csv",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "style": summary.style,
                    "gate": summary.gate,
                    "texture_blend_weight": f"{summary.texture_blend_weight:.12g}",
                    "keypoint_score_mode": summary.keypoint_score_mode,
                    "min_margin": f"{summary.min_margin:.12g}",
                    "min_target_gradient": f"{summary.min_target_gradient:.12g}",
                    "min_target_local_contrast": f"{summary.min_target_local_contrast:.12g}",
                    "pytorch_state_label": summary.pytorch_state_label,
                    "pytorch_state": "" if summary.pytorch_state is None else summary.pytorch_state.as_posix(),
                    "matches": summary.matches,
                    "correct": summary.correct,
                    "precision": f"{summary.precision:.12g}",
                    "summary_csv": summary.summary_csv.as_posix(),
                }
            )


def calibrate_texture_blend_weights(
    *,
    python_exe: Path,
    project_root: Path,
    validation_groups: Iterable[EvalGroup],
    output_dir: Path,
    pytorch_state: Path,
    calibration_pytorch_states: Iterable[tuple[str, Path]] | None = None,
    device: str,
    candidates: Iterable[float],
    limit_pairs: int,
    sample_seed: int | None,
    match_margin_candidates: Iterable[float] | None = None,
    target_gradient_candidates: Iterable[float] | None = None,
    target_local_contrast_candidates: Iterable[float] | None = None,
    keypoint_score_mode_candidates: Iterable[str] | None = None,
    calibration_sample_seeds: Iterable[int] | None = None,
    multiseed_groups: set[tuple[str, str]] | None = None,
    max_keypoints: int,
    descriptor_topk: int,
    keypoint_spatial_bins: int = 0,
    geometry_filter: str = "local",
    calibration_min_matches: int = 0,
    calibration_min_match_fraction: float = 0.0,
    calibration_state_switch_reference_label: str = "trained",
    calibration_state_switch_min_precision_gain: float = 0.0,
    calibration_state_switch_min_match_ratio: float = 0.0,
) -> dict[tuple[str, str], BlendWeightSummary]:
    calibration_dir = output_dir / "calibration"
    candidate_values = list(candidates)
    margin_values = list(match_margin_candidates) if match_margin_candidates is not None else [0.0]
    target_gradient_values = list(target_gradient_candidates) if target_gradient_candidates is not None else [0.0]
    target_local_contrast_values = (
        list(target_local_contrast_candidates) if target_local_contrast_candidates is not None else [0.0]
    )
    keypoint_score_values = list(keypoint_score_mode_candidates) if keypoint_score_mode_candidates is not None else ["texture"]
    state_values = [("trained", pytorch_state)]
    if calibration_pytorch_states is not None:
        state_values.extend(list(calibration_pytorch_states))
    seed_values: list[int | None] = list(calibration_sample_seeds) if calibration_sample_seeds is not None else [sample_seed]
    summaries: list[BlendWeightSummary] = []
    for group in validation_groups:
        group_key = (group.style, group.gate)
        group_seed_values = (
            seed_values
            if calibration_sample_seeds is not None and (multiseed_groups is None or group_key in multiseed_groups)
            else [sample_seed]
        )
        for current_state_label, current_state in state_values:
            for weight in candidate_values:
                for keypoint_score_mode in keypoint_score_values:
                    for margin in margin_values:
                        for target_gradient in target_gradient_values:
                            for target_local_contrast in target_local_contrast_values:
                                matches = 0
                                correct = 0
                                candidate_dir_name = (
                                    f"weight_{blend_weight_label(weight)}"
                                    f"_score_{keypoint_score_mode_label(keypoint_score_mode)}"
                                    f"_margin_{match_margin_label(margin)}"
                                    f"_targetgrad_{target_gradient_label(target_gradient)}"
                                    f"_targetcontrast_{target_local_contrast_label(target_local_contrast)}"
                                )
                                summary_csv = (
                                    calibration_dir
                                    / group.style
                                    / group.gate
                                    / f"state_{state_label(current_state_label)}"
                                    / f"{candidate_dir_name}.csv"
                                )
                                for current_seed in group_seed_values:
                                    if len(group_seed_values) > 1:
                                        seed_label = "none" if current_seed is None else str(current_seed)
                                        output_csv = (
                                            calibration_dir
                                            / group.style
                                            / group.gate
                                            / f"state_{state_label(current_state_label)}"
                                            / candidate_dir_name
                                            / f"seed_{seed_label}.csv"
                                        )
                                    else:
                                        output_csv = summary_csv
                                    command = build_eval_command(
                                        python_exe=python_exe,
                                        project_root=project_root,
                                        group=group,
                                        output_csv=output_csv,
                                        pytorch_state=current_state,
                                        device=device,
                                        limit_pairs=limit_pairs,
                                        sample_seed=current_seed,
                                        max_keypoints=max_keypoints,
                                        descriptor_topk=descriptor_topk,
                                        texture_blend_weight=weight,
                                        keypoint_spatial_bins=keypoint_spatial_bins,
                                        keypoint_score_mode=keypoint_score_mode,
                                        min_margin=margin,
                                        min_target_gradient=target_gradient,
                                        min_target_local_contrast=target_local_contrast,
                                        geometry_filter=geometry_filter,
                                    )
                                    run_command(command, cwd=project_root, quiet=True)
                                    seed_matches, seed_correct, _ = read_match_summary_csv(output_csv)
                                    matches += seed_matches
                                    correct += seed_correct
                                precision = 0.0 if matches == 0 else correct / matches
                                summaries.append(
                                    BlendWeightSummary(
                                        group.style,
                                        group.gate,
                                        weight,
                                        min_margin=margin,
                                        min_target_gradient=target_gradient,
                                        min_target_local_contrast=target_local_contrast,
                                        keypoint_score_mode=keypoint_score_mode,
                                        matches=matches,
                                        correct=correct,
                                        precision=precision,
                                        summary_csv=summary_csv,
                                        pytorch_state_label=current_state_label,
                                        pytorch_state=current_state,
                                    )
                                )
    selected = select_best_blend_weight_summaries(
        summaries,
        min_matches=calibration_min_matches,
        min_match_fraction=calibration_min_match_fraction,
        state_switch_reference_label=calibration_state_switch_reference_label,
        state_switch_min_precision_gain=calibration_state_switch_min_precision_gain,
        state_switch_min_match_ratio=calibration_state_switch_min_match_ratio,
    )
    write_blend_weight_summaries(calibration_dir / "blend_weight_summary.csv", summaries)
    write_blend_weight_summaries(calibration_dir / "selected_weights.csv", selected.values())
    return selected


def evaluation_groups(split_root: Path | str, *, split: str = "test") -> list[EvalGroup]:
    root = Path(split_root)
    groups: list[EvalGroup] = []
    for style in STYLES:
        for gate in GATES:
            cache_dir = root / split / style / gate
            if cache_dir.exists():
                groups.append(EvalGroup(style=style, gate=gate, cache_dir=cache_dir))
    return groups


def select_visualization_pairs(pair_paths: Iterable[Path], *, count: int, seed: int) -> list[Path]:
    paths = sorted(dict.fromkeys(Path(path) for path in pair_paths))
    if count <= 0 or not paths:
        return []
    sample_count = min(count, len(paths))
    return random.Random(seed).sample(paths, sample_count)


def _image_for_plot(view: torch.Tensor) -> torch.Tensor:
    if view.dim() != 3:
        raise ValueError("view tensor must have shape CxHxW")
    image = view.detach().to(torch.float32).cpu().mean(dim=0)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    return image


def draw_match_visualization(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    output_path: Path | str,
    *,
    max_lines: int = 256,
) -> None:
    image_a = _image_for_plot(view_a)
    image_b = _image_for_plot(view_b)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height = max(image_a.shape[0], image_b.shape[0])
    width_a = image_a.shape[1]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.imshow(image_a, cmap="gray", extent=(0, width_a, height, 0))
    ax.imshow(image_b, cmap="gray", extent=(width_a, width_a + image_b.shape[1], height, 0))
    if points_a.numel() and points_b.numel():
        count = min(max_lines, points_a.size(0), points_b.size(0))
        for index in range(count):
            x_a, y_a = points_a[index].detach().cpu().tolist()
            x_b, y_b = points_b[index].detach().cpu().tolist()
            ax.plot([x_a, width_a + x_b], [y_a, y_b], linewidth=0.6, alpha=0.65)
        ax.scatter(points_a[:count, 0].detach().cpu(), points_a[:count, 1].detach().cpu(), s=5)
        ax.scatter(width_a + points_b[:count, 0].detach().cpu(), points_b[:count, 1].detach().cpu(), s=5)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_sample_visualizations(
    pair_paths: Iterable[Path],
    output_dir: Path | str,
    *,
    count: int,
    seed: int,
    load_matches,
) -> list[Path]:
    output = Path(output_dir)
    written: list[Path] = []
    for index, pair_path in enumerate(select_visualization_pairs(pair_paths, count=count, seed=seed), 1):
        view_a, view_b, points_a, points_b = load_matches(pair_path)
        image_path = output / f"{index:02d}_{pair_path.parent.name}_{pair_path.stem}.png"
        draw_match_visualization(view_a, view_b, points_a, points_b, image_path)
        written.append(image_path)
    return written


def evaluated_match_tensors(
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
    keypoint_spatial_bins: int = 0,
    keypoint_score_mode: str = "texture",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if min_target_gradient < 0.0:
        raise ValueError("min_target_gradient must be non-negative")
    if min_target_local_contrast < 0.0:
        raise ValueError("min_target_local_contrast must be non-negative")
    pair = load_libtorch_pair_archive(pair_path, device=device)
    if min_target_gradient > 0.0 and eval_py.target_texture_gradient_mean(pair.view_b) < min_target_gradient:
        return pair.view_a.cpu(), pair.view_b.cpu(), torch.empty(0, 2), torch.empty(0, 2)
    if min_target_local_contrast > 0.0 and eval_py.target_local_contrast_mean(pair.view_b) < min_target_local_contrast:
        return pair.view_a.cpu(), pair.view_b.cpu(), torch.empty(0, 2), torch.empty(0, 2)
    with torch.no_grad():
        descriptors_a, descriptors_b, keypoint_scores_a, keypoint_scores_b = eval_py.descriptor_maps_and_keypoint_scores_for_pair(
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
            texture_fraction=texture_fraction,
            spatial_bins=keypoint_spatial_bins,
            keypoint_scores=keypoint_scores_a,
        )
        keypoints_b, selected_b = eval_py.select_descriptor_keypoints(
            pair.view_b,
            descriptors_b,
            max_keypoints=max_keypoints,
            min_intensity=min_intensity,
            texture_fraction=texture_fraction,
            spatial_bins=keypoint_spatial_bins,
            keypoint_scores=keypoint_scores_b,
        )
        rows_a = eval_py.gather_descriptor_rows(descriptors_a, selected_a)
        rows_b = eval_py.gather_descriptor_rows(descriptors_b, selected_b)
        if mutual:
            matches, scores = eval_py.mutual_nearest_matches(
                rows_a,
                rows_b,
                max_matches=max_matches,
                min_score=min_score,
                min_margin=min_margin,
            )
        else:
            matches, scores = eval_py.greedy_unique_matches(
                rows_a,
                rows_b,
                topk=topk,
                max_matches=max_matches,
                min_score=min_score,
            )
        if matches.numel() == 0:
            return pair.view_a.cpu(), pair.view_b.cpu(), torch.empty(0, 2), torch.empty(0, 2)

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
        if geometry_filter in {"affine", "local"}:
            local_indices = torch.arange(matches.size(0), dtype=torch.long, device=matches.device)
            local_matches = torch.stack([local_indices, local_indices], dim=1)
            local_scores = scores if scores.numel() else torch.arange(matches.size(0), 0, -1, dtype=torch.float32, device=matches.device)
            if geometry_filter == "affine":
                kept_local, _ = eval_py.filter_affine_consistent_matches(
                    points_a,
                    points_b,
                    local_matches,
                    local_scores,
                    threshold_px=threshold_px,
                    min_inliers=4,
                )
            else:
                kept_local, _ = eval_py.filter_local_displacement_consistent_matches(
                    points_a,
                    points_b,
                    local_matches,
                    local_scores,
                    threshold_px=threshold_px,
                    min_inliers=4,
                )
            if kept_local.numel() == 0:
                return pair.view_a.cpu(), pair.view_b.cpu(), torch.empty(0, 2), torch.empty(0, 2)
            keep = kept_local[:, 0].to(points_a.device)
            points_a = points_a.index_select(0, keep)
            points_b = points_b.index_select(0, keep.to(points_b.device))
        elif geometry_filter != "none":
            raise ValueError(f"unsupported geometry filter: {geometry_filter}")
    return pair.view_a.cpu(), pair.view_b.cpu(), points_a.cpu(), points_b.cpu()


def build_eval_command(
    *,
    python_exe: Path,
    project_root: Path,
    group: EvalGroup,
    output_csv: Path,
    pytorch_state: Path,
    device: str,
    limit_pairs: int,
    sample_seed: int | None,
    max_keypoints: int,
    descriptor_topk: int,
    texture_blend_weight: float = pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT,
    keypoint_spatial_bins: int = 0,
    keypoint_score_mode: str = "texture",
    min_margin: float = 0.0,
    min_target_gradient: float = 0.0,
    min_target_local_contrast: float = 0.0,
    geometry_filter: str = "local",
) -> list[str]:
    command = [
        str(python_exe),
        str(project_root / "python" / "pytorch_cache_match_eval.py"),
        "--cache-dir",
        str(group.cache_dir),
        "--pytorch-state",
        str(pytorch_state),
        "--output",
        str(output_csv),
        "--device",
        device,
        "--mode",
        "blend",
        "--texture-blend-weight",
        f"{texture_blend_weight:.12g}",
        "--geometry-filter",
        geometry_filter,
        "--descriptor-topk",
        str(descriptor_topk),
        "--max-keypoints",
        str(max_keypoints),
        "--keypoint-spatial-bins",
        str(keypoint_spatial_bins),
        "--keypoint-score-mode",
        keypoint_score_mode,
        "--mutual",
        "--exclude-self-pairs",
    ]
    if min_margin > 0.0:
        command.extend(["--min-margin", f"{min_margin:.12g}"])
    if min_target_gradient > 0.0:
        command.extend(["--min-target-gradient", f"{min_target_gradient:.12g}"])
    if min_target_local_contrast > 0.0:
        command.extend(["--min-target-local-contrast", f"{min_target_local_contrast:.12g}"])
    if limit_pairs > 0:
        command.extend(["--limit-pairs", str(limit_pairs)])
    if sample_seed is not None:
        command.extend(["--sample-seed", str(sample_seed)])
    return command


def run_command(command: list[str], *, cwd: Path, quiet: bool = False) -> None:
    env = dict(os.environ)
    env["MKL_THREADING_LAYER"] = "GNU"
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT} if quiet else {}
    subprocess.run(command, cwd=cwd, check=True, env=env, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1024-cache cross-view training and grouped evaluation")
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--init-pytorch-state", type=Path, default=None)
    parser.add_argument("--init-random", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-pairs", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--balanced-cache-sampling", action="store_true")
    parser.add_argument("--samples-per-pair", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-limit-pairs", type=int, default=0)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--keypoint-spatial-bins", type=int, default=0)
    parser.add_argument("--keypoint-score-mode", choices=KEYPOINT_SCORE_MODES, default="texture")
    parser.add_argument("--descriptor-topk", type=int, default=32)
    parser.add_argument("--texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--match-min-margin", type=float, default=0.0)
    parser.add_argument("--match-min-target-gradient", type=float, default=0.0)
    parser.add_argument("--match-min-target-local-contrast", type=float, default=0.0)
    parser.add_argument("--geometry-filter", choices=["none", "local", "affine"], default="local")
    parser.add_argument("--training-texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--training-eval-pairs", type=int, default=0)
    parser.add_argument("--synthetic-loss-weight", type=float, default=1.0)
    parser.add_argument("--training-groups", default=None)
    parser.add_argument("--hard-summary", action="append", type=Path, default=[])
    parser.add_argument("--mine-hard-training-pairs", action="store_true")
    parser.add_argument("--hard-mine-limit-pairs", type=int, default=128)
    parser.add_argument("--hard-limit", type=int, default=64)
    parser.add_argument("--hard-min-matches", type=int, default=4)
    parser.add_argument("--hard-max-precision", type=float, default=0.9)
    parser.add_argument("--hard-repeat", type=int, default=3)
    parser.add_argument("--hard-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--hard-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--pseudo-label-csv", action="append", type=Path, default=[])
    parser.add_argument("--pseudo-label-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-keypoint-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-keypoint-negative-weight", type=float, default=0.01)
    parser.add_argument("--pseudo-label-max-points", type=int, default=128)
    parser.add_argument("--pseudo-label-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--pseudo-label-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--false-match-csv", action="append", type=Path, default=[])
    parser.add_argument("--false-match-weight", type=float, default=0.0)
    parser.add_argument("--false-match-max-points", type=int, default=128)
    parser.add_argument("--false-match-max-score", type=float, default=0.25)
    parser.add_argument("--false-match-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--false-match-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--warp-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--warp-hard-negative-radius", type=float, default=2.0)
    parser.add_argument("--warp-hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--warp-hard-negative-candidates", type=int, default=4096)
    parser.add_argument("--abstention-weight", type=float, default=0.0)
    parser.add_argument("--abstention-negative-radius", type=float, default=2.0)
    parser.add_argument("--abstention-max-false-score", type=float, default=0.35)
    parser.add_argument("--abstention-topk", type=int, default=8)
    parser.add_argument("--abstention-candidates", type=int, default=4096)
    parser.add_argument("--calibrate-texture-blend-weights", action="store_true")
    parser.add_argument("--calibrate-match-min-margins", action="store_true")
    parser.add_argument("--calibrate-target-gradients", action="store_true")
    parser.add_argument("--calibrate-target-local-contrasts", action="store_true")
    parser.add_argument("--calibrate-keypoint-score-modes", action="store_true")
    parser.add_argument(
        "--texture-blend-candidates",
        default=",".join(f"{value:.12g}" for value in DEFAULT_BLEND_WEIGHT_CANDIDATES),
    )
    parser.add_argument(
        "--match-min-margin-candidates",
        default=",".join(f"{value:.12g}" for value in DEFAULT_MATCH_MARGIN_CANDIDATES),
    )
    parser.add_argument(
        "--target-gradient-candidates",
        default=",".join(f"{value:.12g}" for value in DEFAULT_TARGET_GRADIENT_CANDIDATES),
    )
    parser.add_argument(
        "--target-local-contrast-candidates",
        default=",".join(f"{value:.12g}" for value in DEFAULT_TARGET_LOCAL_CONTRAST_CANDIDATES),
    )
    parser.add_argument("--keypoint-score-mode-candidates", default=",".join(DEFAULT_KEYPOINT_SCORE_MODE_CANDIDATES))
    parser.add_argument("--calibration-limit-pairs", type=int, default=None)
    parser.add_argument("--calibration-sample-seeds", default=None)
    parser.add_argument("--calibration-multiseed-groups", default=None)
    parser.add_argument("--calibration-pytorch-state", action="append", default=[])
    parser.add_argument("--calibration-min-matches", type=int, default=0)
    parser.add_argument("--calibration-min-match-fraction", type=float, default=0.0)
    parser.add_argument("--calibration-state-switch-reference-label", default="trained")
    parser.add_argument("--calibration-state-switch-min-precision-gain", type=float, default=0.0)
    parser.add_argument("--calibration-state-switch-min-match-ratio", type=float, default=0.0)
    parser.add_argument("--visualization-samples", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable)
    output_dir = args.output_dir
    sources = cache_split.discover_cache_sources(args.cache_dir)
    assignments = cache_split.split_source_names(
        [source.source_name for source in sources],
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
    )
    split_root = output_dir / "splits"
    created = cache_split.create_split_cache_dirs(sources, assignments, split_root)
    training_groups = parse_group_keys(args.training_groups) if args.training_groups is not None else None
    train_cache_dirs = select_created_cache_dirs(created, split="train", groups=training_groups)
    validation_cache_dirs = select_created_cache_dirs(created, split="val", groups=training_groups)
    if not train_cache_dirs:
        raise RuntimeError("no training cache directories selected")
    hard_summaries = list(args.hard_summary)
    if args.mine_hard_training_pairs:
        if args.init_pytorch_state is None:
            raise RuntimeError("--mine-hard-training-pairs requires --init-pytorch-state")
        train_groups = [
            group
            for group in evaluation_groups(split_root, split="train")
            if training_groups is None or (group.style, group.gate) in training_groups
        ]
        hard_mining_dir = output_dir / "hard_mining"
        for group in train_groups:
            output_csv = hard_mining_dir / group.style / group.gate / "summary.csv"
            hard_command = build_eval_command(
                python_exe=python_exe,
                project_root=project_root,
                group=group,
                output_csv=output_csv,
                pytorch_state=args.init_pytorch_state,
                device=args.device,
                limit_pairs=args.hard_mine_limit_pairs,
                sample_seed=args.seed,
                max_keypoints=args.max_keypoints,
                descriptor_topk=args.descriptor_topk,
                texture_blend_weight=args.texture_blend_weight,
                keypoint_spatial_bins=args.keypoint_spatial_bins,
                keypoint_score_mode=args.keypoint_score_mode,
                min_margin=args.match_min_margin,
                min_target_gradient=args.match_min_target_gradient,
                min_target_local_contrast=args.match_min_target_local_contrast,
                geometry_filter=args.geometry_filter,
            )
            run_command(hard_command, cwd=project_root)
            hard_summaries.append(output_csv)
    training_dir = output_dir / "training"
    train_command = build_training_command(
        python_exe=python_exe,
        project_root=project_root,
        train_cache_dirs=train_cache_dirs,
        validation_cache_dirs=validation_cache_dirs,
        output_dir=training_dir,
        checkpoint=args.checkpoint,
        init_pytorch_state=args.init_pytorch_state,
        init_random=args.init_random,
        device=args.device,
        steps=args.steps,
        batch_pairs=args.batch_pairs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        samples_per_pair=args.samples_per_pair,
        learning_rate=args.learning_rate,
        balanced_cache_sampling=args.balanced_cache_sampling,
        training_texture_blend_weight=args.training_texture_blend_weight,
        training_eval_pairs=args.training_eval_pairs,
        synthetic_loss_weight=args.synthetic_loss_weight,
        warp_hard_negative_weight=args.warp_hard_negative_weight,
        warp_hard_negative_radius=args.warp_hard_negative_radius,
        warp_hard_negative_margin=args.warp_hard_negative_margin,
        warp_hard_negative_candidates=args.warp_hard_negative_candidates,
        abstention_weight=args.abstention_weight,
        abstention_negative_radius=args.abstention_negative_radius,
        abstention_max_false_score=args.abstention_max_false_score,
        abstention_topk=args.abstention_topk,
        abstention_candidates=args.abstention_candidates,
        hard_summaries=hard_summaries,
        hard_limit=args.hard_limit,
        hard_min_matches=args.hard_min_matches,
        hard_max_precision=args.hard_max_precision,
        hard_repeat=args.hard_repeat,
        hard_curriculum_max_probability=args.hard_curriculum_max_probability,
        hard_curriculum_warmup_steps=args.hard_curriculum_warmup_steps,
        pseudo_label_csvs=args.pseudo_label_csv,
        pseudo_label_weight=args.pseudo_label_weight,
        pseudo_keypoint_weight=args.pseudo_keypoint_weight,
        pseudo_keypoint_negative_weight=args.pseudo_keypoint_negative_weight,
        pseudo_label_max_points=args.pseudo_label_max_points,
        pseudo_label_curriculum_max_probability=args.pseudo_label_curriculum_max_probability,
        pseudo_label_curriculum_warmup_steps=args.pseudo_label_curriculum_warmup_steps,
        false_match_csvs=args.false_match_csv,
        false_match_weight=args.false_match_weight,
        false_match_max_points=args.false_match_max_points,
        false_match_max_score=args.false_match_max_score,
        false_match_curriculum_max_probability=args.false_match_curriculum_max_probability,
        false_match_curriculum_warmup_steps=args.false_match_curriculum_warmup_steps,
    )
    run_command(train_command, cwd=project_root)
    state_path = training_dir / "pytorch_pfm_state.pt"
    device = torch.device(args.device)
    model_cache: dict[Path, pfm_model.PlanetaryFeatureMatcher] = {}

    def model_for_state(path: Path) -> pfm_model.PlanetaryFeatureMatcher:
        cached = model_cache.get(path)
        if cached is not None:
            return cached
        loaded, _ = pfm_model.load_pytorch_state(path, device=args.device)
        loaded.eval()
        model_cache[path] = loaded
        return loaded

    blend_weights: dict[tuple[str, str], float] = {}
    match_margins: dict[tuple[str, str], float] = {}
    target_gradients: dict[tuple[str, str], float] = {}
    target_local_contrasts: dict[tuple[str, str], float] = {}
    keypoint_score_modes: dict[tuple[str, str], str] = {}
    state_paths: dict[tuple[str, str], Path] = {}
    if (
        args.calibrate_texture_blend_weights
        or args.calibrate_match_min_margins
        or args.calibrate_target_gradients
        or args.calibrate_target_local_contrasts
        or args.calibrate_keypoint_score_modes
    ):
        selected = calibrate_texture_blend_weights(
            python_exe=python_exe,
            project_root=project_root,
            validation_groups=evaluation_groups(split_root, split="val"),
            output_dir=output_dir,
            pytorch_state=state_path,
            calibration_pytorch_states=parse_calibration_pytorch_state_entries(args.calibration_pytorch_state),
            device=args.device,
            candidates=(
                parse_blend_weight_candidates(args.texture_blend_candidates)
                if args.calibrate_texture_blend_weights
                else [args.texture_blend_weight]
            ),
            match_margin_candidates=(
                parse_match_margin_candidates(args.match_min_margin_candidates)
                if args.calibrate_match_min_margins
                else [args.match_min_margin]
            ),
            target_gradient_candidates=(
                parse_target_gradient_candidates(args.target_gradient_candidates)
                if args.calibrate_target_gradients
                else [args.match_min_target_gradient]
            ),
            target_local_contrast_candidates=(
                parse_target_local_contrast_candidates(args.target_local_contrast_candidates)
                if args.calibrate_target_local_contrasts
                else [args.match_min_target_local_contrast]
            ),
            keypoint_score_mode_candidates=(
                parse_keypoint_score_mode_candidates(args.keypoint_score_mode_candidates)
                if args.calibrate_keypoint_score_modes
                else [args.keypoint_score_mode]
            ),
            limit_pairs=args.eval_limit_pairs if args.calibration_limit_pairs is None else args.calibration_limit_pairs,
            sample_seed=args.seed,
            calibration_sample_seeds=(
                parse_sample_seeds(args.calibration_sample_seeds)
                if args.calibration_sample_seeds is not None
                else None
            ),
            multiseed_groups=(
                parse_group_keys(args.calibration_multiseed_groups)
                if args.calibration_multiseed_groups is not None
                else None
            ),
            max_keypoints=args.max_keypoints,
            descriptor_topk=args.descriptor_topk,
            keypoint_spatial_bins=args.keypoint_spatial_bins,
            geometry_filter=args.geometry_filter,
            calibration_min_matches=args.calibration_min_matches,
            calibration_min_match_fraction=args.calibration_min_match_fraction,
            calibration_state_switch_reference_label=args.calibration_state_switch_reference_label,
            calibration_state_switch_min_precision_gain=args.calibration_state_switch_min_precision_gain,
            calibration_state_switch_min_match_ratio=args.calibration_state_switch_min_match_ratio,
        )
        blend_weights = {key: summary.texture_blend_weight for key, summary in selected.items()}
        match_margins = {key: summary.min_margin for key, summary in selected.items()}
        target_gradients = {key: summary.min_target_gradient for key, summary in selected.items()}
        target_local_contrasts = {key: summary.min_target_local_contrast for key, summary in selected.items()}
        keypoint_score_modes = {key: summary.keypoint_score_mode for key, summary in selected.items()}
        state_paths = {key: summary.pytorch_state or state_path for key, summary in selected.items()}
    for group in evaluation_groups(split_root, split="test"):
        group_key = (group.style, group.gate)
        texture_blend_weight = blend_weights.get(group_key, args.texture_blend_weight)
        min_margin = match_margins.get(group_key, args.match_min_margin)
        min_target_gradient = target_gradients.get(group_key, args.match_min_target_gradient)
        min_target_local_contrast = target_local_contrasts.get(group_key, args.match_min_target_local_contrast)
        keypoint_score_mode = keypoint_score_modes.get(group_key, args.keypoint_score_mode)
        selected_state_path = state_paths.get(group_key, state_path)
        group_dir = output_dir / "eval" / group.style / group.gate
        group_dir.mkdir(parents=True, exist_ok=True)
        eval_command = build_eval_command(
            python_exe=python_exe,
            project_root=project_root,
            group=group,
            output_csv=group_dir / "summary.csv",
            pytorch_state=selected_state_path,
            device=args.device,
            limit_pairs=args.eval_limit_pairs,
            sample_seed=args.seed,
            max_keypoints=args.max_keypoints,
            descriptor_topk=args.descriptor_topk,
            texture_blend_weight=texture_blend_weight,
            keypoint_spatial_bins=args.keypoint_spatial_bins,
            keypoint_score_mode=keypoint_score_mode,
            min_margin=min_margin,
            min_target_gradient=min_target_gradient,
            min_target_local_contrast=min_target_local_contrast,
            geometry_filter=args.geometry_filter,
        )
        run_command(eval_command, cwd=project_root)
        pair_paths = eval_py.limit_pair_paths(
            discover_pair_archives([group.cache_dir], limit_pairs=0, exclude_self_pairs=True),
            limit_pairs=args.eval_limit_pairs,
            sample_seed=args.seed,
        )
        selected_model = model_for_state(selected_state_path)
        render_sample_visualizations(
            pair_paths,
            output_dir / "visualizations" / group.style / group.gate,
            count=args.visualization_samples,
            seed=args.seed,
            load_matches=lambda pair_path, texture_blend_weight=texture_blend_weight, keypoint_score_mode=keypoint_score_mode, selected_model=selected_model: evaluated_match_tensors(
                selected_model,
                pair_path,
                device=device,
                mode="blend",
                texture_blend_weight=texture_blend_weight,
                max_keypoints=args.max_keypoints,
                min_intensity=0.01,
                texture_fraction=1.0,
                keypoint_spatial_bins=args.keypoint_spatial_bins,
                keypoint_score_mode=keypoint_score_mode,
                threshold_px=5.0,
                topk=args.descriptor_topk,
                max_matches=512,
                min_score=-1.0,
                min_margin=min_margin,
                min_target_gradient=min_target_gradient,
                min_target_local_contrast=min_target_local_contrast,
                mutual=True,
                geometry_filter=args.geometry_filter,
            ),
        )
    print(f"experiment={output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
