#!/usr/bin/env python3
"""Run the fov76 formal/guard evaluation and promotion gate for a checkpoint."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import run_dual_checkpoint_rescue_eval as dual_rescue_mod


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_SWEEP_SCRIPT = PROJECT_ROOT / "scripts" / "run_graph_filter_sweep.py"
PROMOTION_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_checkpoint_promotion.py"
DUAL_CHECKPOINT_SELECTOR_SCRIPT = PROJECT_ROOT / "scripts" / "run_dual_checkpoint_rescue_eval.py"
FOV76_GEO5_GEO10_EXTREME_RESCUE_PROFILE = "fov76_geo5_geo10_extreme_rescue"
FOV76_GEO5_GEO10_EXTREME_RESCUE_LOW_MATCH_GUARD_PROFILE = (
    "fov76_geo5_geo10_extreme_rescue_lowmatch_guard"
)
FOV76_POST_FILTER_PROFILES = (
    FOV76_GEO5_GEO10_EXTREME_RESCUE_PROFILE,
    FOV76_GEO5_GEO10_EXTREME_RESCUE_LOW_MATCH_GUARD_PROFILE,
)
FOV76_RANSAC_MINMATCH16_DUAL_RESCUE_PROFILE = "fov76_ransac_minmatch16"
FOV76_DUAL_CHECKPOINT_RESCUE_PROFILES = (
    FOV76_RANSAC_MINMATCH16_DUAL_RESCUE_PROFILE,
)


@dataclass(frozen=True)
class EvalModel:
    label: str
    guard_label: str
    state: Path
    run_dir: Path


@dataclass(frozen=True)
class SelectorPromotionInputs:
    selector_output_dir: Path
    formal_summary: Path
    formal_variant_summary: Path
    guard_summary: Path
    guard_variant_summary: Path


def _filtered_min_matches_by_variant(args: argparse.Namespace, *, model: EvalModel) -> list[str]:
    values = list(getattr(args, "filtered_min_matches_by_variant", []))
    if model.label == getattr(args, "baseline_label", None) or model.guard_label == getattr(args, "guard_baseline_label", None):
        values.extend(getattr(args, "baseline_filtered_min_matches_by_variant", []))
    if model.label == getattr(args, "candidate_label", None) or model.guard_label == getattr(args, "guard_candidate_label", None):
        values.extend(getattr(args, "candidate_filtered_min_matches_by_variant", []))
    return values


def _split_variant_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _join_variant_csv(values: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        for item in _split_variant_csv(value):
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
    return ",".join(ordered)


def _adaptive_geometry_rescue_variants(args: argparse.Namespace, *, model: EvalModel) -> str:
    values = [getattr(args, "adaptive_geometry_rescue_variants", "")]
    if model.label == getattr(args, "baseline_label", None) or model.guard_label == getattr(args, "guard_baseline_label", None):
        values.append(getattr(args, "baseline_adaptive_geometry_rescue_variants", ""))
    if model.label == getattr(args, "candidate_label", None) or model.guard_label == getattr(args, "guard_candidate_label", None):
        values.append(getattr(args, "candidate_adaptive_geometry_rescue_variants", ""))
    return _join_variant_csv(values)


def _low_match_geometry_guard_variants(args: argparse.Namespace, *, model: EvalModel) -> str:
    values = [getattr(args, "low_match_geometry_guard_variants", "")]
    if model.label == getattr(args, "baseline_label", None) or model.guard_label == getattr(args, "guard_baseline_label", None):
        values.append(getattr(args, "baseline_low_match_geometry_guard_variants", ""))
    if model.label == getattr(args, "candidate_label", None) or model.guard_label == getattr(args, "guard_candidate_label", None):
        values.append(getattr(args, "candidate_low_match_geometry_guard_variants", ""))
    return _join_variant_csv(values)


def _matches_baseline_model(args: argparse.Namespace, *, model: EvalModel) -> bool:
    return model.label == getattr(args, "baseline_label", None) or model.guard_label == getattr(args, "guard_baseline_label", None)


def _matches_candidate_model(args: argparse.Namespace, *, model: EvalModel) -> bool:
    return model.label == getattr(args, "candidate_label", None) or model.guard_label == getattr(args, "guard_candidate_label", None)


def _side_adaptive_geometry_rescue_value(args: argparse.Namespace, *, model: EvalModel, name: str) -> object:
    value = getattr(args, name)
    if _matches_baseline_model(args, model=model):
        baseline_value = getattr(args, f"baseline_{name}", None)
        if baseline_value is not None:
            value = baseline_value
    if _matches_candidate_model(args, model=model):
        candidate_value = getattr(args, f"candidate_{name}", None)
        if candidate_value is not None:
            value = candidate_value
    return value


def _side_low_match_geometry_guard_value(args: argparse.Namespace, *, model: EvalModel, name: str) -> object:
    value = getattr(args, name)
    if _matches_baseline_model(args, model=model):
        baseline_value = getattr(args, f"baseline_{name}", None)
        if baseline_value is not None:
            value = baseline_value
    if _matches_candidate_model(args, model=model):
        candidate_value = getattr(args, f"candidate_{name}", None)
        if candidate_value is not None:
            value = candidate_value
    return value


def _set_profile_default(args: argparse.Namespace, name: str, value: object, default: object) -> None:
    if getattr(args, name) == default:
        setattr(args, name, value)


def apply_post_filter_profile(args: argparse.Namespace) -> None:
    profile = getattr(args, "post_filter_profile", "")
    if not profile:
        return
    if profile not in FOV76_POST_FILTER_PROFILES:
        raise ValueError(f"unknown post-filter profile: {profile}")
    if profile == FOV76_GEO5_GEO10_EXTREME_RESCUE_PROFILE:
        _set_profile_default(args, "geometry_threshold_px", 5.0, 10.0)
        _set_profile_default(args, "baseline_adaptive_geometry_rescue_variants", "extreme_02,extreme_03", "")
        _set_profile_default(args, "candidate_adaptive_geometry_rescue_variants", "extreme_02,extreme_03", "")
        _set_profile_default(args, "adaptive_geometry_rescue_threshold_px", 10.0, 0.0)
        _set_profile_default(args, "adaptive_geometry_rescue_min_match_gain", 20, 0)
        _set_profile_default(args, "adaptive_geometry_rescue_max_base_matches", 0, -1)
        _set_profile_default(args, "adaptive_geometry_rescue_max_homography_p90_px", 4.5, -1.0)
        _set_profile_default(args, "adaptive_geometry_rescue_max_homography_median_px", 1.8, -1.0)
        _set_profile_default(args, "adaptive_geometry_rescue_require_score_mean_not_lower", True, False)
        _set_profile_default(args, "candidate_adaptive_geometry_rescue_min_match_gain", 5, None)
        _set_profile_default(args, "candidate_adaptive_geometry_rescue_max_base_matches", 16, None)
        _set_profile_default(args, "candidate_adaptive_geometry_rescue_max_homography_p90_px", 4.2, None)
        _set_profile_default(args, "candidate_adaptive_geometry_rescue_max_homography_median_px", 2.3, None)
        _set_profile_default(args, "candidate_adaptive_geometry_rescue_require_score_mean_not_lower", False, None)
    if profile == FOV76_GEO5_GEO10_EXTREME_RESCUE_LOW_MATCH_GUARD_PROFILE:
        _set_profile_default(args, "geometry_threshold_px", 5.0, 10.0)
        _set_profile_default(args, "adaptive_geometry_rescue_variants", "extreme_02,extreme_03", "")
        _set_profile_default(args, "adaptive_geometry_rescue_threshold_px", 10.0, 0.0)
        _set_profile_default(args, "adaptive_geometry_rescue_min_match_gain", 5, 0)
        _set_profile_default(args, "adaptive_geometry_rescue_max_base_matches", 16, -1)
        _set_profile_default(args, "adaptive_geometry_rescue_max_homography_p90_px", 4.2, -1.0)
        _set_profile_default(args, "adaptive_geometry_rescue_max_homography_median_px", 2.3, -1.0)
        _set_profile_default(args, "low_match_geometry_guard_variants", "extreme_02,extreme_03", "")
        _set_profile_default(args, "low_match_geometry_guard_min_matches", 12, 0)
        _set_profile_default(args, "low_match_geometry_guard_max_matches", 15, -1)
        _set_profile_default(args, "low_match_geometry_guard_max_homography_p90_px", 2.8, -1.0)
        _set_profile_default(args, "low_match_geometry_guard_max_homography_median_px", 1.5, -1.0)
        _set_profile_default(args, "low_match_geometry_guard_min_score_mean", 19.0, float("-inf"))
        _set_profile_default_unless_explicit(
            args,
            "dual_checkpoint_rescue_min_match_gain",
            3,
            1,
            "--dual-checkpoint-rescue-min-match-gain",
        )
        _set_profile_default_unless_explicit(
            args,
            "dual_checkpoint_rescue_min_rescue_matches",
            16,
            8,
            "--dual-checkpoint-rescue-min-rescue-matches",
        )
    _set_profile_default(args, "formal_target_variants", "extreme_02,extreme_03", "")
    _set_profile_default(args, "formal_protected_variants", "mid_01,mid_02,extreme_01,nadir", "")
    _set_profile_default(args, "min_formal_target_total_correct_gain", 1, 0)
    _set_profile_default_unless_explicit(
        args,
        "max_formal_precision_drop",
        0.0,
        0.0,
        "--max-formal-precision-drop",
    )
    _set_profile_default_unless_explicit(
        args,
        "max_formal_wrong_increase",
        1,
        0,
        "--max-formal-wrong-increase",
    )
    _set_profile_default_unless_explicit(
        args,
        "max_formal_target_precision_drop",
        0.0,
        0.0,
        "--max-formal-target-precision-drop",
    )
    _set_profile_default_unless_explicit(
        args,
        "max_formal_target_wrong_increase",
        1,
        0,
        "--max-formal-target-wrong-increase",
    )
    _set_profile_default_unless_explicit(
        args,
        "max_guard_precision_drop",
        0.0,
        0.0,
        "--max-guard-precision-drop",
    )
    _set_profile_default_unless_explicit(
        args,
        "max_guard_wrong_increase",
        0,
        0,
        "--max-guard-wrong-increase",
    )
    _set_profile_default(args, "min_extreme_correct_gain", 0, 1)
    _set_profile_default(args, "max_extreme_wrong_increase", 20, 10**12)


def _explicit_cli_option(args: argparse.Namespace, option_name: str) -> bool:
    explicit_options = getattr(args, "_explicit_cli_options", set())
    return option_name in explicit_options


def _set_profile_default_unless_explicit(
    args: argparse.Namespace,
    name: str,
    value: object,
    default: object,
    option_name: str,
) -> None:
    if _explicit_cli_option(args, option_name):
        return
    _set_profile_default(args, name, value, default)


def _extra_regression_guard_sets(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for value in getattr(args, "extra_regression_guard_set", []):
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    return values


def _guard_set_names(args: argparse.Namespace) -> list[str]:
    values = ["regression_guard", "extreme_gain"]
    for set_name in _extra_regression_guard_sets(args):
        if set_name not in values:
            values.append(set_name)
    return values


def apply_dual_checkpoint_rescue_profile(args: argparse.Namespace) -> None:
    profile = getattr(args, "dual_checkpoint_rescue_profile", "")
    if not profile:
        return
    if profile not in FOV76_DUAL_CHECKPOINT_RESCUE_PROFILES:
        raise ValueError(f"unknown dual-checkpoint rescue profile: {profile}")
    if profile == FOV76_RANSAC_MINMATCH16_DUAL_RESCUE_PROFILE:
        _set_profile_default_unless_explicit(
            args,
            "dual_checkpoint_rescue_min_match_gain",
            3,
            1,
            "--dual-checkpoint-rescue-min-match-gain",
        )
        _set_profile_default_unless_explicit(
            args,
            "dual_checkpoint_rescue_min_rescue_matches",
            16,
            8,
            "--dual-checkpoint-rescue-min-rescue-matches",
        )


def _add_common_sweep_options(
    command: list[str],
    args: argparse.Namespace,
    *,
    model: EvalModel,
    candidate_pairs: int,
) -> None:
    command.extend(
        [
            "--pair-mode",
            "same-position",
            "--image-source",
            "uint8",
            "--candidate-pairs",
            str(candidate_pairs),
            "--select-count",
            "0",
            "--seed",
            str(args.seed),
            "--crop-size",
            str(args.crop_size),
            "--max-image-size",
            str(args.max_image_size),
            "--device",
            args.device,
            "--descriptor-mode",
            "learned",
            "--keypoint-score-mode",
            "learned",
            "--max-keypoints",
            str(args.max_keypoints),
            "--matcher-candidate-topk",
            str(args.matcher_candidate_topk),
            "--max-matches",
            "0",
            "--draw-matches",
            "0",
            "--threshold-px",
            "5.0",
            "--geometry-filter",
            "local",
            "--geometry-threshold-px",
            str(args.geometry_threshold_px),
            "--geometry-threshold-px-values",
            str(args.geometry_threshold_px),
            "--min-score-values",
            "-1",
            "--graph-dustbin-delta-values",
            "0",
            "--graph-acceptance-margin-values",
            "0",
            "--graph-min-raw-score-values",
            "-1",
            "--graph-min-raw-margin-values",
            "0",
            "--graph-min-accept-probability-values",
            "-1",
            "--graph-max-attention-layers",
            str(args.graph_layers),
            "--graph-max-attention-work-fraction",
            "1.0",
            "--graph-width-prune-keep-ratio",
            "1.0",
            "--graph-width-prune-min-score",
            "-1.0",
            "--graph-early-stop-min-confidence",
            "-1.0",
            "--filtered-geometry-filter",
            "magsac",
            "--filtered-min-margin",
            "0.0",
            "--filtered-max-matches",
            "0",
            "--filtered-draw-matches",
            "0",
            "--filtered-min-matches",
            str(args.filtered_min_matches),
            "--filtered-min-matches-values",
            str(args.filtered_min_matches),
        ]
    )
    for value in _filtered_min_matches_by_variant(args, model=model):
        command.extend(["--filtered-min-matches-by-variant", value])
    low_match_guard_variants = _low_match_geometry_guard_variants(args, model=model)
    if low_match_guard_variants:
        command.extend(["--low-match-geometry-guard-variants", low_match_guard_variants])
        command.extend(
            [
                "--low-match-geometry-guard-min-matches",
                str(_side_low_match_geometry_guard_value(args, model=model, name="low_match_geometry_guard_min_matches")),
            ]
        )
        command.extend(
            [
                "--low-match-geometry-guard-max-matches",
                str(_side_low_match_geometry_guard_value(args, model=model, name="low_match_geometry_guard_max_matches")),
            ]
        )
        command.extend(
            [
                "--low-match-geometry-guard-max-homography-p90-px",
                str(
                    _side_low_match_geometry_guard_value(
                        args,
                        model=model,
                        name="low_match_geometry_guard_max_homography_p90_px",
                    )
                ),
            ]
        )
        command.extend(
            [
                "--low-match-geometry-guard-max-homography-median-px",
                str(
                    _side_low_match_geometry_guard_value(
                        args,
                        model=model,
                        name="low_match_geometry_guard_max_homography_median_px",
                    )
                ),
            ]
        )
        command.extend(
            [
                "--low-match-geometry-guard-min-score-mean",
                str(_side_low_match_geometry_guard_value(args, model=model, name="low_match_geometry_guard_min_score_mean")),
            ]
        )
    adaptive_variants = _adaptive_geometry_rescue_variants(args, model=model)
    if adaptive_variants:
        command.extend(["--adaptive-geometry-rescue-variants", adaptive_variants])
        command.extend(
            [
                "--adaptive-geometry-rescue-threshold-px",
                str(_side_adaptive_geometry_rescue_value(args, model=model, name="adaptive_geometry_rescue_threshold_px")),
            ]
        )
        command.extend(
            [
                "--adaptive-geometry-rescue-min-match-gain",
                str(_side_adaptive_geometry_rescue_value(args, model=model, name="adaptive_geometry_rescue_min_match_gain")),
            ]
        )
        command.extend(
            [
                "--adaptive-geometry-rescue-max-base-matches",
                str(_side_adaptive_geometry_rescue_value(args, model=model, name="adaptive_geometry_rescue_max_base_matches")),
            ]
        )
        command.extend(
            [
                "--adaptive-geometry-rescue-max-homography-p90-px",
                str(_side_adaptive_geometry_rescue_value(args, model=model, name="adaptive_geometry_rescue_max_homography_p90_px")),
            ]
        )
        command.extend(
            [
                "--adaptive-geometry-rescue-max-homography-median-px",
                str(_side_adaptive_geometry_rescue_value(args, model=model, name="adaptive_geometry_rescue_max_homography_median_px")),
            ]
        )
        if _side_adaptive_geometry_rescue_value(
            args,
            model=model,
            name="adaptive_geometry_rescue_require_score_mean_not_lower",
        ):
            command.append("--adaptive-geometry-rescue-require-score-mean-not-lower")
    command.extend(
        [
            "--max-configs",
            "1",
            "--write-all-summary",
            "--no-shuffle",
            "--no-illumination-stress",
            "--input-local-contrast",
            "--input-local-contrast-strength",
            "0.35",
            "--input-local-contrast-kernel",
            "31",
        ]
    )
    if getattr(args, "write_match_details", False):
        command.append("--write-match-details")


def _base_sweep_command(args: argparse.Namespace, *, model: EvalModel, split: str, output_dir: Path) -> list[str]:
    return [
        str(args.python_executable),
        str(GRAPH_SWEEP_SCRIPT),
        "--render-manifest",
        str(args.pair_root / "manifests" / "h100km_fov076_render_manifest.csv"),
        "--uint8-manifest",
        str(args.pair_root / "manifests" / "h100km_fov076_uint8_manifest.csv"),
        "--pytorch-state",
        str(model.state),
        "--output-dir",
        str(output_dir),
        "--run-dir",
        str(model.run_dir),
        "--metrics-csv",
        str(model.run_dir / "train_metrics.csv"),
        "--split",
        split,
        "--reference-variant",
        "nadir",
    ]


def build_formal_sweep_command(args: argparse.Namespace, *, model: EvalModel, split: str) -> list[str]:
    output_dir = args.output_dir / "formal" / f"{model.label}_{split}_geo10_minmatch16"
    command = _base_sweep_command(args, model=model, split=split, output_dir=output_dir)
    command.extend(["--pair-spec-manifest", str(args.pair_root / f"overlap_edges_{split}.csv")])
    _add_common_sweep_options(
        command,
        args,
        model=model,
        candidate_pairs=int(getattr(args, "formal_candidate_pairs", 60)),
    )
    return command


def build_guard_sweep_command(
    args: argparse.Namespace,
    *,
    model: EvalModel,
    set_name: str,
    split: str,
) -> list[str]:
    output_dir = args.output_dir / "guard" / f"{model.guard_label}_{set_name}_{split}_geo10_minmatch16"
    command = _base_sweep_command(args, model=model, split=split, output_dir=output_dir)
    command.extend(["--pair-spec-manifest", str(args.guard_root / f"{set_name}_{split}.csv")])
    _add_common_sweep_options(
        command,
        args,
        model=model,
        candidate_pairs=int(getattr(args, "guard_candidate_pairs", 100)),
    )
    return command


def build_promotion_command(
    args: argparse.Namespace,
    *,
    formal_summary: Path,
    guard_summary: Path,
    formal_variant_summary: Path | None = None,
) -> list[str]:
    command = [
        str(args.python_executable),
        str(PROMOTION_SCRIPT),
        "--formal-summary",
        str(formal_summary),
        "--guard-summary",
        str(guard_summary),
        "--baseline-label",
        args.baseline_label,
        "--candidate-label",
        args.candidate_label,
        "--guard-baseline-label",
        args.guard_baseline_label,
        "--guard-candidate-label",
        args.guard_candidate_label,
        "--splits",
        ",".join(args.splits),
        "--max-formal-precision-drop",
        str(args.max_formal_precision_drop),
        "--max-formal-correct-drop",
        str(args.max_formal_correct_drop),
        "--max-formal-wrong-increase",
        str(args.max_formal_wrong_increase),
        "--max-guard-precision-drop",
        str(args.max_guard_precision_drop),
        "--max-guard-correct-drop",
        str(args.max_guard_correct_drop),
        "--max-guard-wrong-increase",
        str(args.max_guard_wrong_increase),
        "--max-extra-guard-precision-drop",
        str(getattr(args, "max_extra_guard_precision_drop", 0.0)),
        "--max-extra-guard-correct-drop",
        str(getattr(args, "max_extra_guard_correct_drop", 0)),
        "--max-extra-guard-wrong-increase",
        str(getattr(args, "max_extra_guard_wrong_increase", 0)),
        "--min-extreme-correct-gain",
        str(args.min_extreme_correct_gain),
        "--max-extreme-precision-drop",
        str(args.max_extreme_precision_drop),
        "--max-extreme-wrong-increase",
        str(args.max_extreme_wrong_increase),
        "--output-json",
        str(args.output_dir / "promotion_decision.json"),
        "--output-html",
        str(args.output_dir / "promotion_decision.html"),
    ]
    for set_name in _extra_regression_guard_sets(args):
        command.extend(["--extra-regression-guard-set", set_name])
    if formal_variant_summary is not None:
        command.extend(["--formal-variant-summary", str(formal_variant_summary)])
    formal_target_variants = getattr(args, "formal_target_variants", "")
    if formal_target_variants:
        command.extend(["--formal-target-variants", str(formal_target_variants)])
    formal_protected_variants = getattr(args, "formal_protected_variants", "")
    if formal_protected_variants:
        command.extend(["--formal-protected-variants", str(formal_protected_variants)])
    command.extend(
        [
            "--min-formal-target-correct-gain",
            str(getattr(args, "min_formal_target_correct_gain", 0)),
            "--min-formal-target-total-correct-gain",
            str(getattr(args, "min_formal_target_total_correct_gain", 0)),
            "--max-formal-target-precision-drop",
            str(getattr(args, "max_formal_target_precision_drop", 0.0)),
            "--max-formal-target-wrong-increase",
            str(getattr(args, "max_formal_target_wrong_increase", 0)),
            "--max-protected-variant-precision-drop",
            str(getattr(args, "max_protected_variant_precision_drop", 0.0)),
            "--max-protected-variant-correct-drop",
            str(getattr(args, "max_protected_variant_correct_drop", 0)),
            "--max-protected-variant-wrong-increase",
            str(getattr(args, "max_protected_variant_wrong_increase", 0)),
        ]
    )
    return command


def required_input_paths(args: argparse.Namespace) -> list[Path]:
    paths = [
        args.baseline_state,
        args.candidate_state,
        args.baseline_run_dir / "train_metrics.csv",
        args.candidate_run_dir / "train_metrics.csv",
        args.pair_root / "manifests" / "h100km_fov076_render_manifest.csv",
        args.pair_root / "manifests" / "h100km_fov076_uint8_manifest.csv",
    ]
    for split in args.splits:
        paths.append(args.pair_root / f"overlap_edges_{split}.csv")
    for set_name in _guard_set_names(args):
        for split in args.splits:
            paths.append(args.guard_root / f"{set_name}_{split}.csv")
    return paths


def validate_inputs(args: argparse.Namespace) -> None:
    if getattr(args, "low_match_geometry_guard_min_matches", 0) < 0:
        raise ValueError("--low-match-geometry-guard-min-matches must be nonnegative")
    if getattr(args, "low_match_geometry_guard_max_matches", -1) < -1:
        raise ValueError("--low-match-geometry-guard-max-matches must be >= -1")
    if getattr(args, "low_match_geometry_guard_max_homography_p90_px", -1.0) < -1.0:
        raise ValueError("--low-match-geometry-guard-max-homography-p90-px must be >= -1")
    if getattr(args, "low_match_geometry_guard_max_homography_median_px", -1.0) < -1.0:
        raise ValueError("--low-match-geometry-guard-max-homography-median-px must be >= -1")
    for prefix in ("baseline", "candidate"):
        min_matches = getattr(args, f"{prefix}_low_match_geometry_guard_min_matches", None)
        max_matches = getattr(args, f"{prefix}_low_match_geometry_guard_max_matches", None)
        max_p90 = getattr(args, f"{prefix}_low_match_geometry_guard_max_homography_p90_px", None)
        max_median = getattr(args, f"{prefix}_low_match_geometry_guard_max_homography_median_px", None)
        if min_matches is not None and min_matches < 0:
            raise ValueError(f"--{prefix}-low-match-geometry-guard-min-matches must be nonnegative")
        if max_matches is not None and max_matches < -1:
            raise ValueError(f"--{prefix}-low-match-geometry-guard-max-matches must be >= -1")
        if max_p90 is not None and max_p90 < -1.0:
            raise ValueError(f"--{prefix}-low-match-geometry-guard-max-homography-p90-px must be >= -1")
        if max_median is not None and max_median < -1.0:
            raise ValueError(f"--{prefix}-low-match-geometry-guard-max-homography-median-px must be >= -1")
    missing = [path for path in required_input_paths(args) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing required fov76 promotion inputs:\n{formatted}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _command_option(command: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, item in enumerate(command):
        if item == name and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def record_failed_sweep_command(command: list[str], *, returncode: int, error: str) -> Path:
    output_dir_value = _command_option(command, "--output-dir")
    if not output_dir_value:
        raise ValueError("failed sweep command does not contain --output-dir")
    output_dir = Path(output_dir_value)
    report_dir = output_dir / "failed_sweep"
    report_dir.mkdir(parents=True, exist_ok=True)
    failure_text = f"returncode={int(returncode)}: {error}"
    (output_dir / "sweep_failure.json").write_text(
        json.dumps(
            {
                "returncode": int(returncode),
                "error": error,
                "command": command,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv_rows(
        report_dir / "all_filtered_summary.csv",
        [
            "label",
            "base_id",
            "target_variant",
            "split",
            "matches",
            "correct",
            "wrong",
            "precision",
            "score_mean",
            "median_error_px",
        ],
        [
            {
                "label": "failed_sweep",
                "base_id": "failed_sweep",
                "target_variant": "sweep_failed",
                "split": "",
                "matches": 0,
                "correct": 0,
                "wrong": 0,
                "precision": 0.0,
                "score_mean": 0.0,
                "median_error_px": 0.0,
            }
        ],
    )
    write_csv_rows(
        output_dir / "graph_filter_sweep_summary.csv",
        [
            "min_score",
            "dustbin_delta",
            "acceptance_margin",
            "min_raw_score",
            "min_raw_margin",
            "min_accept_probability",
            "geometry_threshold_px",
            "filtered_min_matches",
            "raw_rows",
            "raw_matches",
            "raw_correct",
            "raw_wrong",
            "raw_precision",
            "raw_median_error_px",
            "filtered_rows",
            "filtered_matches",
            "filtered_correct",
            "filtered_wrong",
            "filtered_precision",
            "filtered_median_error_px",
            "report_dir",
            "sweep_failed",
            "sweep_error",
        ],
        [
            {
                "min_score": -1,
                "dustbin_delta": 0,
                "acceptance_margin": 0,
                "min_raw_score": -1,
                "min_raw_margin": 0,
                "min_accept_probability": -1,
                "geometry_threshold_px": 0,
                "filtered_min_matches": 0,
                "raw_rows": 0,
                "raw_matches": 0,
                "raw_correct": 0,
                "raw_wrong": 0,
                "raw_precision": 0.0,
                "raw_median_error_px": 0.0,
                "filtered_rows": 0,
                "filtered_matches": 0,
                "filtered_correct": 0,
                "filtered_wrong": 0,
                "filtered_precision": 0.0,
                "filtered_median_error_px": 0.0,
                "report_dir": "failed_sweep",
                "sweep_failed": 1,
                "sweep_error": failure_text,
            }
        ],
    )
    return output_dir


def _run_sweep_command(command: list[str]) -> bool:
    try:
        result = subprocess.run(command, check=False)
    except Exception as exc:
        record_failed_sweep_command(command, returncode=1, error=repr(exc))
        return False
    if result.returncode != 0:
        record_failed_sweep_command(command, returncode=int(result.returncode), error="subprocess failed")
        return False
    return True


def _summary_has_sweep_failures(path: Path) -> bool:
    if not path.exists():
        return False
    for row in read_csv_rows(path):
        if str(row.get("sweep_failed") or "").strip().lower() in {"1", "true", "yes", "y"}:
            return True
    return False


def _parse_formal_sweep_name(name: str) -> tuple[str, str]:
    suffix = "_geo10_minmatch16"
    base = name[: -len(suffix)] if name.endswith(suffix) else name
    for split in ("val", "test"):
        split_suffix = f"_{split}"
        if base.endswith(split_suffix):
            return base[: -len(split_suffix)], split
    return base, "unknown"


def _parse_guard_sweep_name(name: str, labels: list[str]) -> tuple[str, str, str]:
    suffix = "_geo10_minmatch16"
    base = name[: -len(suffix)] if name.endswith(suffix) else name
    for label in labels:
        prefix = f"{label}_"
        if not base.startswith(prefix):
            continue
        rest = base[len(prefix) :]
        for split in ("val", "test"):
            split_suffix = f"_{split}"
            if rest.endswith(split_suffix):
                return label, rest[: -len(split_suffix)], split
    return "unknown", base, "unknown"


def _variant_rows_from_report(report_dir: Path, *, label_key: str, label: str, split: str, set_name: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped: dict[str, dict[str, int]] = {}
    for item in read_csv_rows(report_dir / "all_filtered_summary.csv"):
        variant = item.get("target_variant", "unknown")
        matches = int(float(item.get("matches") or 0))
        correct = int(float(item.get("correct") or 0))
        wrong = int(float(item.get("wrong") or 0))
        bucket = grouped.setdefault(variant, {"rows": 0, "matches": 0, "correct": 0, "wrong": 0, "zero": 0})
        bucket["rows"] += 1
        bucket["matches"] += matches
        bucket["correct"] += correct
        bucket["wrong"] += wrong
        if matches == 0:
            bucket["zero"] += 1
    for variant, bucket in sorted(grouped.items()):
        row: dict[str, object] = {
            label_key: label,
            "split": split,
            "variant": variant,
            **bucket,
            "precision": 0.0 if bucket["matches"] <= 0 else bucket["correct"] / bucket["matches"],
        }
        if set_name is not None:
            row["set"] = set_name
        rows.append(row)
    return rows


def combine_formal_summaries(output_dir: Path) -> tuple[Path, Path]:
    root = output_dir / "formal"
    summary_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    for sweep_dir in sorted(root.glob("*_geo10_minmatch16")):
        label, split = _parse_formal_sweep_name(sweep_dir.name)
        for row in read_csv_rows(sweep_dir / "graph_filter_sweep_summary.csv"):
            item = dict(row)
            item["label"] = label
            item["split"] = split
            summary_rows.append(item)
            report_dir = Path(item["report_dir"])
            if not report_dir.is_absolute():
                report_dir = sweep_dir / report_dir
            variant_rows.extend(_variant_rows_from_report(report_dir, label_key="label", label=label, split=split, set_name=None))
    summary_fields = [
        "label",
        "split",
        "geometry_threshold_px",
        "filtered_min_matches",
        "raw_rows",
        "raw_matches",
        "raw_correct",
        "raw_wrong",
        "raw_precision",
        "raw_median_error_px",
        "filtered_rows",
        "filtered_matches",
        "filtered_correct",
        "filtered_wrong",
        "filtered_precision",
        "filtered_median_error_px",
        "report_dir",
        "sweep_failed",
        "sweep_error",
    ]
    variant_fields = ["label", "split", "variant", "rows", "matches", "correct", "wrong", "precision", "zero"]
    summary_path = output_dir / "formal_summary.csv"
    variant_path = output_dir / "formal_variant_summary.csv"
    write_csv_rows(summary_path, summary_fields, summary_rows)
    write_csv_rows(variant_path, variant_fields, variant_rows)
    return summary_path, variant_path


def combine_guard_summaries(output_dir: Path, *, guard_labels: list[str]) -> tuple[Path, Path]:
    root = output_dir / "guard"
    summary_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    for sweep_dir in sorted(root.glob("*_geo10_minmatch16")):
        label, set_name, split = _parse_guard_sweep_name(sweep_dir.name, guard_labels)
        for row in read_csv_rows(sweep_dir / "graph_filter_sweep_summary.csv"):
            item = dict(row)
            item["model"] = label
            item["set"] = set_name
            item["split"] = split
            summary_rows.append(item)
            report_dir = Path(item["report_dir"])
            if not report_dir.is_absolute():
                report_dir = sweep_dir / report_dir
            variant_rows.extend(
                _variant_rows_from_report(
                    report_dir,
                    label_key="model",
                    label=label,
                    split=split,
                    set_name=set_name,
                )
            )
    summary_fields = [
        "model",
        "set",
        "split",
        "filtered_rows",
        "filtered_matches",
        "filtered_correct",
        "filtered_wrong",
        "filtered_precision",
        "filtered_median_error_px",
        "report_dir",
        "sweep_failed",
        "sweep_error",
    ]
    variant_fields = ["model", "set", "split", "variant", "rows", "matches", "correct", "wrong", "precision", "zero"]
    summary_path = output_dir / "guard_summary.csv"
    variant_path = output_dir / "guard_variant_summary.csv"
    write_csv_rows(summary_path, summary_fields, summary_rows)
    write_csv_rows(variant_path, variant_fields, variant_rows)
    return summary_path, variant_path


def _first_all_filtered_summary_from_sweep(sweep_dir: Path) -> Path:
    rows = read_csv_rows(sweep_dir / "graph_filter_sweep_summary.csv")
    if not rows:
        raise ValueError(f"empty graph filter sweep summary: {sweep_dir / 'graph_filter_sweep_summary.csv'}")
    report_dir = Path(rows[0].get("report_dir") or "")
    if not report_dir.is_absolute():
        report_dir = sweep_dir / report_dir
    summary_path = report_dir / "all_filtered_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing selector all_filtered_summary: {summary_path}")
    return summary_path


def build_dual_checkpoint_selector_sources(args: argparse.Namespace) -> list[dual_rescue_mod.SourceSpec]:
    sources: list[dual_rescue_mod.SourceSpec] = []
    formal_root = args.output_dir / "formal"
    for baseline_dir in sorted(formal_root.glob("*_geo10_minmatch16")):
        label, split = _parse_formal_sweep_name(baseline_dir.name)
        if label != args.baseline_label:
            continue
        candidate_dir = formal_root / f"{args.candidate_label}_{split}_geo10_minmatch16"
        if not candidate_dir.exists():
            raise FileNotFoundError(f"missing candidate formal sweep dir for selector: {candidate_dir}")
        sources.append(
            dual_rescue_mod.SourceSpec(
                name="formal",
                split=split,
                baseline_summary=_first_all_filtered_summary_from_sweep(baseline_dir),
                rescue_summary=_first_all_filtered_summary_from_sweep(candidate_dir),
            )
        )

    guard_root = args.output_dir / "guard"
    guard_labels = [args.guard_baseline_label, args.guard_candidate_label]
    for baseline_dir in sorted(guard_root.glob("*_geo10_minmatch16")):
        label, set_name, split = _parse_guard_sweep_name(baseline_dir.name, guard_labels)
        if label != args.guard_baseline_label:
            continue
        candidate_dir = guard_root / f"{args.guard_candidate_label}_{set_name}_{split}_geo10_minmatch16"
        if not candidate_dir.exists():
            raise FileNotFoundError(f"missing candidate guard sweep dir for selector: {candidate_dir}")
        sources.append(
            dual_rescue_mod.SourceSpec(
                name=set_name,
                split=split,
                baseline_summary=_first_all_filtered_summary_from_sweep(baseline_dir),
                rescue_summary=_first_all_filtered_summary_from_sweep(candidate_dir),
            )
        )

    if not sources:
        raise ValueError(f"no baseline/candidate sweep reports found for selector under {args.output_dir}")
    return sources


def _dual_selector_config_from_args(args: argparse.Namespace) -> dual_rescue_mod.SelectorConfig:
    apply_dual_checkpoint_rescue_profile(args)
    return dual_rescue_mod.SelectorConfig(
        target_variants=dual_rescue_mod.parse_variant_list(args.dual_checkpoint_rescue_target_variants),
        min_match_gain=args.dual_checkpoint_rescue_min_match_gain,
        min_rescue_matches=args.dual_checkpoint_rescue_min_rescue_matches,
        max_rescue_homography_p90_px=args.dual_checkpoint_rescue_max_homography_p90_px,
        max_rescue_homography_median_px=args.dual_checkpoint_rescue_max_homography_median_px,
        min_rescue_score_mean=args.dual_checkpoint_rescue_min_score_mean,
        require_rescue_score_mean_not_lower=not args.dual_checkpoint_rescue_allow_score_mean_drop,
    )


def _dual_selector_metadata(args: argparse.Namespace) -> dict[str, object]:
    enabled = bool(getattr(args, "dual_checkpoint_rescue_selector", False))
    metadata: dict[str, object] = {
        "enabled": enabled,
        "profile": getattr(args, "dual_checkpoint_rescue_profile", ""),
    }
    if not enabled:
        return metadata
    config = _dual_selector_config_from_args(args)
    metadata.update(
        {
            "baseline_label": args.baseline_label,
            "rescue_label": args.candidate_label,
            "selected_label": args.dual_checkpoint_rescue_label,
            "config": {
                "target_variants": list(config.target_variants),
                "min_match_gain": config.min_match_gain,
                "min_rescue_matches": config.min_rescue_matches,
                "max_rescue_homography_p90_px": config.max_rescue_homography_p90_px,
                "max_rescue_homography_median_px": config.max_rescue_homography_median_px,
                "min_rescue_score_mean": config.min_rescue_score_mean,
                "require_rescue_score_mean_not_lower": config.require_rescue_score_mean_not_lower,
            },
        }
    )
    return metadata


def write_pipeline_metadata(args: argparse.Namespace, commands: list[list[str]]) -> None:
    metadata = {
        "pair_root": str(args.pair_root),
        "guard_root": str(args.guard_root),
        "splits": list(args.splits),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "post_filter_profile": getattr(args, "post_filter_profile", ""),
        "extra_regression_guard_sets": _extra_regression_guard_sets(args),
        "extra_guard_thresholds": {
            "max_precision_drop": getattr(args, "max_extra_guard_precision_drop", 0.0),
            "max_correct_drop": getattr(args, "max_extra_guard_correct_drop", 0),
            "max_wrong_increase": getattr(args, "max_extra_guard_wrong_increase", 0),
        },
        "planned_command_count": len(commands),
        "dual_checkpoint_rescue": _dual_selector_metadata(args),
    }
    (args.output_dir / "promotion_pipeline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_dual_checkpoint_selector(args: argparse.Namespace) -> Path:
    selector_output_dir = args.output_dir / "dual_checkpoint_rescue_selector"
    config = _dual_selector_config_from_args(args)
    sources = build_dual_checkpoint_selector_sources(args)
    all_combined_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    for source in sources:
        baseline_rows = dual_rescue_mod.read_summary_rows(source.baseline_summary)
        rescue_rows = dual_rescue_mod.read_summary_rows(source.rescue_summary)
        combined_rows = dual_rescue_mod.combine_summary_rows(
            baseline_rows,
            rescue_rows,
            config=config,
            source=source.name,
            split=source.split,
            baseline_label=args.baseline_label,
            rescue_label=args.candidate_label,
        )
        all_combined_rows.extend(combined_rows)
        summary_rows.append(
            dual_rescue_mod.summarize_rows(
                baseline_rows,
                label=args.baseline_label,
                source=source.name,
                split=source.split,
            )
        )
        summary_rows.append(
            dual_rescue_mod.summarize_rows(
                rescue_rows,
                label=args.candidate_label,
                source=source.name,
                split=source.split,
            )
        )
        summary_rows.append(
            dual_rescue_mod.summarize_rows(
                combined_rows,
                label="selected",
                source=source.name,
                split=source.split,
            )
        )
        variant_rows.extend(
            dual_rescue_mod.summarize_by_variant(
                combined_rows,
                label="selected",
                source=source.name,
                split=source.split,
            )
        )

    dual_rescue_mod.write_csv_rows(selector_output_dir / "combined_filtered_summary.csv", all_combined_rows)
    dual_rescue_mod.write_csv_rows(selector_output_dir / "summary.csv", summary_rows)
    dual_rescue_mod.write_csv_rows(selector_output_dir / "variant_summary.csv", variant_rows)
    metadata = {
        "baseline_label": args.baseline_label,
        "rescue_label": args.candidate_label,
        "selected_label": args.dual_checkpoint_rescue_label,
        "config": {
            "target_variants": list(config.target_variants),
            "min_match_gain": config.min_match_gain,
            "min_rescue_matches": config.min_rescue_matches,
            "max_rescue_homography_p90_px": config.max_rescue_homography_p90_px,
            "max_rescue_homography_median_px": config.max_rescue_homography_median_px,
            "min_rescue_score_mean": config.min_rescue_score_mean,
            "require_rescue_score_mean_not_lower": config.require_rescue_score_mean_not_lower,
        },
        "sources": [
            {
                "name": source.name,
                "split": source.split,
                "baseline_summary": str(source.baseline_summary),
                "rescue_summary": str(source.rescue_summary),
            }
            for source in sources
        ],
    }
    selector_output_dir.mkdir(parents=True, exist_ok=True)
    (selector_output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dual_rescue_mod.write_html_report(
        selector_output_dir / "index.html",
        sources=sources,
        config=config,
        summary_rows=summary_rows,
        variant_rows=variant_rows,
        combined_rows=all_combined_rows,
    )
    return selector_output_dir


def _selector_formal_summary_rows(
    *,
    original_formal_summary: Path,
    selector_summary: Path,
    baseline_label: str,
    selected_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv_rows(original_formal_summary):
        if row.get("label") == baseline_label:
            item = dict(row)
            item["filtered_precision"] = _precision_from_counts(
                item,
                matches_key="filtered_matches",
                correct_key="filtered_correct",
            )
            rows.append(item)
    for row in read_csv_rows(selector_summary):
        if row.get("source") != "formal" or row.get("model") != "selected":
            continue
        item = dict(row)
        item["label"] = selected_label
        item["report_dir"] = "dual_checkpoint_rescue_selector"
        item["filtered_precision"] = _precision_from_counts(
            item,
            matches_key="filtered_matches",
            correct_key="filtered_correct",
        )
        rows.append(item)
    return rows


def _precision_from_counts(row: dict[str, str], *, matches_key: str, correct_key: str) -> str:
    matches = int(float(row.get(matches_key) or 0))
    correct = int(float(row.get(correct_key) or 0))
    if matches <= 0:
        return "0"
    return f"{(float(correct) / float(matches)):.17g}"


def _selector_formal_variant_rows(
    *,
    original_formal_variant_summary: Path,
    selector_variant_summary: Path,
    baseline_label: str,
    selected_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv_rows(original_formal_variant_summary):
        if row.get("label") == baseline_label:
            item = dict(row)
            item["precision"] = _precision_from_counts(item, matches_key="matches", correct_key="correct")
            rows.append(item)
    for row in read_csv_rows(selector_variant_summary):
        if row.get("source") != "formal" or row.get("model") != "selected":
            continue
        rows.append(
            {
                "label": selected_label,
                "split": row.get("split", ""),
                "variant": row.get("variant", ""),
                "rows": row.get("filtered_rows", ""),
                "matches": row.get("filtered_matches", ""),
                "correct": row.get("filtered_correct", ""),
                "wrong": row.get("filtered_wrong", ""),
                "precision": _precision_from_counts(row, matches_key="filtered_matches", correct_key="filtered_correct"),
                "zero": "",
            }
        )
    return rows


def _selector_guard_summary_rows(
    *,
    original_guard_summary: Path,
    selector_summary: Path,
    baseline_label: str,
    selected_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv_rows(original_guard_summary):
        if row.get("model") == baseline_label:
            item = dict(row)
            item["filtered_precision"] = _precision_from_counts(
                item,
                matches_key="filtered_matches",
                correct_key="filtered_correct",
            )
            rows.append(item)
    for row in read_csv_rows(selector_summary):
        source = row.get("source", "")
        if source == "formal" or row.get("model") != "selected":
            continue
        item = dict(row)
        item["model"] = selected_label
        item["set"] = source
        item["report_dir"] = "dual_checkpoint_rescue_selector"
        item["filtered_precision"] = _precision_from_counts(
            item,
            matches_key="filtered_matches",
            correct_key="filtered_correct",
        )
        rows.append(item)
    return rows


def _selector_guard_variant_rows(
    *,
    original_guard_variant_summary: Path,
    selector_variant_summary: Path,
    baseline_label: str,
    selected_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv_rows(original_guard_variant_summary):
        if row.get("model") == baseline_label:
            item = dict(row)
            item["precision"] = _precision_from_counts(item, matches_key="matches", correct_key="correct")
            rows.append(item)
    for row in read_csv_rows(selector_variant_summary):
        source = row.get("source", "")
        if source == "formal" or row.get("model") != "selected":
            continue
        rows.append(
            {
                "model": selected_label,
                "set": source,
                "split": row.get("split", ""),
                "variant": row.get("variant", ""),
                "rows": row.get("filtered_rows", ""),
                "matches": row.get("filtered_matches", ""),
                "correct": row.get("filtered_correct", ""),
                "wrong": row.get("filtered_wrong", ""),
                "precision": _precision_from_counts(row, matches_key="filtered_matches", correct_key="filtered_correct"),
                "zero": "",
            }
        )
    return rows


def build_dual_checkpoint_selector_promotion_inputs(
    args: argparse.Namespace,
    *,
    formal_summary: Path,
    formal_variant_summary: Path,
    guard_summary: Path,
    guard_variant_summary: Path,
) -> SelectorPromotionInputs:
    selector_output_dir = run_dual_checkpoint_selector(args)
    selected_label = args.dual_checkpoint_rescue_label

    formal_fields = [
        "label",
        "split",
        "geometry_threshold_px",
        "filtered_min_matches",
        "raw_rows",
        "raw_matches",
        "raw_correct",
        "raw_wrong",
        "raw_precision",
        "raw_median_error_px",
        "filtered_rows",
        "filtered_matches",
        "filtered_correct",
        "filtered_wrong",
        "filtered_precision",
        "filtered_median_error_px",
        "report_dir",
    ]
    formal_variant_fields = ["label", "split", "variant", "rows", "matches", "correct", "wrong", "precision", "zero"]
    guard_fields = [
        "model",
        "set",
        "split",
        "filtered_rows",
        "filtered_matches",
        "filtered_correct",
        "filtered_wrong",
        "filtered_precision",
        "filtered_median_error_px",
        "report_dir",
    ]
    guard_variant_fields = ["model", "set", "split", "variant", "rows", "matches", "correct", "wrong", "precision", "zero"]

    selected_formal_summary = selector_output_dir / "promotion_formal_summary.csv"
    selected_formal_variant_summary = selector_output_dir / "promotion_formal_variant_summary.csv"
    selected_guard_summary = selector_output_dir / "promotion_guard_summary.csv"
    selected_guard_variant_summary = selector_output_dir / "promotion_guard_variant_summary.csv"
    write_csv_rows(
        selected_formal_summary,
        formal_fields,
        _selector_formal_summary_rows(
            original_formal_summary=formal_summary,
            selector_summary=selector_output_dir / "summary.csv",
            baseline_label=args.baseline_label,
            selected_label=selected_label,
        ),
    )
    write_csv_rows(
        selected_formal_variant_summary,
        formal_variant_fields,
        _selector_formal_variant_rows(
            original_formal_variant_summary=formal_variant_summary,
            selector_variant_summary=selector_output_dir / "variant_summary.csv",
            baseline_label=args.baseline_label,
            selected_label=selected_label,
        ),
    )
    write_csv_rows(
        selected_guard_summary,
        guard_fields,
        _selector_guard_summary_rows(
            original_guard_summary=guard_summary,
            selector_summary=selector_output_dir / "summary.csv",
            baseline_label=args.guard_baseline_label,
            selected_label=selected_label,
        ),
    )
    write_csv_rows(
        selected_guard_variant_summary,
        guard_variant_fields,
        _selector_guard_variant_rows(
            original_guard_variant_summary=guard_variant_summary,
            selector_variant_summary=selector_output_dir / "variant_summary.csv",
            baseline_label=args.guard_baseline_label,
            selected_label=selected_label,
        ),
    )
    return SelectorPromotionInputs(
        selector_output_dir=selector_output_dir,
        formal_summary=selected_formal_summary,
        formal_variant_summary=selected_formal_variant_summary,
        guard_summary=selected_guard_summary,
        guard_variant_summary=selected_guard_variant_summary,
    )


def write_index_html(output_dir: Path, *, formal_summary: Path, guard_summary: Path, promotion_html: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                "<head><meta charset=\"utf-8\"><title>fov76 checkpoint promotion pipeline</title></head>",
                "<body>",
                "<h1>fov76 checkpoint promotion pipeline</h1>",
                "<ul>",
                f"<li>Formal summary: <code>{html.escape(str(formal_summary))}</code></li>",
                f"<li>Guard summary: <code>{html.escape(str(guard_summary))}</code></li>",
                f"<li>Promotion decision: <code>{html.escape(str(promotion_html))}</code></li>",
                "</ul>",
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _dual_selector_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / "dual_checkpoint_rescue_selector"


def _dual_selector_promotion_args(args: argparse.Namespace) -> argparse.Namespace:
    promotion_args = argparse.Namespace(**vars(args))
    promotion_args.candidate_label = args.dual_checkpoint_rescue_label
    promotion_args.guard_candidate_label = args.dual_checkpoint_rescue_label
    return promotion_args


def planned_commands(args: argparse.Namespace) -> list[list[str]]:
    baseline = EvalModel(
        label=args.baseline_label,
        guard_label=args.guard_baseline_label,
        state=args.baseline_state,
        run_dir=args.baseline_run_dir,
    )
    candidate = EvalModel(
        label=args.candidate_label,
        guard_label=args.guard_candidate_label,
        state=args.candidate_state,
        run_dir=args.candidate_run_dir,
    )
    commands: list[list[str]] = []
    for model in (baseline, candidate):
        for split in args.splits:
            commands.append(build_formal_sweep_command(args, model=model, split=split))
    for model in (baseline, candidate):
        for set_name in _guard_set_names(args):
            for split in args.splits:
                commands.append(build_guard_sweep_command(args, model=model, set_name=set_name, split=split))
    promotion_args = args
    formal_summary = args.output_dir / "formal_summary.csv"
    formal_variant_summary = args.output_dir / "formal_variant_summary.csv"
    guard_summary = args.output_dir / "guard_summary.csv"
    if getattr(args, "dual_checkpoint_rescue_selector", False):
        selector_output_dir = _dual_selector_output_dir(args)
        promotion_args = _dual_selector_promotion_args(args)
        formal_summary = selector_output_dir / "promotion_formal_summary.csv"
        formal_variant_summary = selector_output_dir / "promotion_formal_variant_summary.csv"
        guard_summary = selector_output_dir / "promotion_guard_summary.csv"
    commands.append(
        build_promotion_command(
            promotion_args,
            formal_summary=formal_summary,
            formal_variant_summary=formal_variant_summary,
            guard_summary=guard_summary,
        )
    )
    return commands


def run_pipeline(args: argparse.Namespace) -> int:
    apply_post_filter_profile(args)
    apply_dual_checkpoint_rescue_profile(args)
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = planned_commands(args)
    (args.output_dir / "planned_commands.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_pipeline_metadata(args, commands)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0
    for command in commands[:-1]:
        print(" ".join(command), flush=True)
        _run_sweep_command(command)
    formal_summary, formal_variant_summary = combine_formal_summaries(args.output_dir)
    guard_summary, guard_variant_summary = combine_guard_summaries(
        args.output_dir,
        guard_labels=[args.guard_baseline_label, args.guard_candidate_label],
    )
    promotion_args = args
    has_sweep_failures = _summary_has_sweep_failures(formal_summary) or _summary_has_sweep_failures(guard_summary)
    if getattr(args, "dual_checkpoint_rescue_selector", False) and not has_sweep_failures:
        selector_inputs = build_dual_checkpoint_selector_promotion_inputs(
            args,
            formal_summary=formal_summary,
            formal_variant_summary=formal_variant_summary,
            guard_summary=guard_summary,
            guard_variant_summary=guard_variant_summary,
        )
        formal_summary = selector_inputs.formal_summary
        formal_variant_summary = selector_inputs.formal_variant_summary
        guard_summary = selector_inputs.guard_summary
        promotion_args = _dual_selector_promotion_args(args)
    promotion_command = build_promotion_command(
        promotion_args,
        formal_summary=formal_summary,
        formal_variant_summary=formal_variant_summary,
        guard_summary=guard_summary,
    )
    print(" ".join(promotion_command), flush=True)
    result = subprocess.run(promotion_command, check=False)
    write_index_html(
        args.output_dir,
        formal_summary=formal_summary,
        guard_summary=guard_summary,
        promotion_html=args.output_dir / "promotion_decision.html",
    )
    return int(result.returncode)


def _parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("at least one split is required")
    return splits


def _collect_explicit_cli_options(argv: list[str]) -> set[str]:
    return {item.split("=", 1)[0] for item in argv if item.startswith("--")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--guard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-state", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-state", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="phase2h_ransac")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--guard-baseline-label", default="phase2h")
    parser.add_argument("--guard-candidate-label", required=True)
    parser.add_argument("--splits", type=_parse_splits, default=["val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--crop-size", type=int, default=2048)
    parser.add_argument("--max-image-size", type=int, default=768)
    parser.add_argument("--max-keypoints", type=int, default=512)
    parser.add_argument("--matcher-candidate-topk", type=int, default=256)
    parser.add_argument("--graph-layers", type=int, default=4)
    parser.add_argument("--formal-candidate-pairs", type=int, default=60)
    parser.add_argument("--guard-candidate-pairs", type=int, default=100)
    parser.add_argument(
        "--post-filter-profile",
        choices=FOV76_POST_FILTER_PROFILES,
        default="",
        help="Apply a named post-filter/promotion profile before building sweep commands.",
    )
    parser.add_argument("--geometry-threshold-px", type=float, default=10.0)
    parser.add_argument("--filtered-min-matches", type=int, default=16)
    parser.add_argument("--filtered-min-matches-by-variant", action="append", default=[])
    parser.add_argument("--baseline-filtered-min-matches-by-variant", action="append", default=[])
    parser.add_argument("--candidate-filtered-min-matches-by-variant", action="append", default=[])
    parser.add_argument("--adaptive-geometry-rescue-variants", default="")
    parser.add_argument("--baseline-adaptive-geometry-rescue-variants", default="")
    parser.add_argument("--candidate-adaptive-geometry-rescue-variants", default="")
    parser.add_argument("--adaptive-geometry-rescue-threshold-px", type=float, default=0.0)
    parser.add_argument("--adaptive-geometry-rescue-min-match-gain", type=int, default=0)
    parser.add_argument("--adaptive-geometry-rescue-max-base-matches", type=int, default=-1)
    parser.add_argument("--adaptive-geometry-rescue-max-homography-p90-px", type=float, default=-1.0)
    parser.add_argument("--adaptive-geometry-rescue-max-homography-median-px", type=float, default=-1.0)
    parser.add_argument("--adaptive-geometry-rescue-require-score-mean-not-lower", action="store_true")
    parser.add_argument("--baseline-adaptive-geometry-rescue-threshold-px", type=float, default=None)
    parser.add_argument("--baseline-adaptive-geometry-rescue-min-match-gain", type=int, default=None)
    parser.add_argument("--baseline-adaptive-geometry-rescue-max-base-matches", type=int, default=None)
    parser.add_argument("--baseline-adaptive-geometry-rescue-max-homography-p90-px", type=float, default=None)
    parser.add_argument("--baseline-adaptive-geometry-rescue-max-homography-median-px", type=float, default=None)
    parser.add_argument(
        "--baseline-adaptive-geometry-rescue-require-score-mean-not-lower",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--candidate-adaptive-geometry-rescue-threshold-px", type=float, default=None)
    parser.add_argument("--candidate-adaptive-geometry-rescue-min-match-gain", type=int, default=None)
    parser.add_argument("--candidate-adaptive-geometry-rescue-max-base-matches", type=int, default=None)
    parser.add_argument("--candidate-adaptive-geometry-rescue-max-homography-p90-px", type=float, default=None)
    parser.add_argument("--candidate-adaptive-geometry-rescue-max-homography-median-px", type=float, default=None)
    parser.add_argument(
        "--candidate-adaptive-geometry-rescue-require-score-mean-not-lower",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--low-match-geometry-guard-variants", default="")
    parser.add_argument("--baseline-low-match-geometry-guard-variants", default="")
    parser.add_argument("--candidate-low-match-geometry-guard-variants", default="")
    parser.add_argument("--low-match-geometry-guard-min-matches", type=int, default=0)
    parser.add_argument("--low-match-geometry-guard-max-matches", type=int, default=-1)
    parser.add_argument("--low-match-geometry-guard-max-homography-p90-px", type=float, default=-1.0)
    parser.add_argument("--low-match-geometry-guard-max-homography-median-px", type=float, default=-1.0)
    parser.add_argument("--low-match-geometry-guard-min-score-mean", type=float, default=float("-inf"))
    parser.add_argument("--baseline-low-match-geometry-guard-min-matches", type=int, default=None)
    parser.add_argument("--baseline-low-match-geometry-guard-max-matches", type=int, default=None)
    parser.add_argument("--baseline-low-match-geometry-guard-max-homography-p90-px", type=float, default=None)
    parser.add_argument("--baseline-low-match-geometry-guard-max-homography-median-px", type=float, default=None)
    parser.add_argument("--baseline-low-match-geometry-guard-min-score-mean", type=float, default=None)
    parser.add_argument("--candidate-low-match-geometry-guard-min-matches", type=int, default=None)
    parser.add_argument("--candidate-low-match-geometry-guard-max-matches", type=int, default=None)
    parser.add_argument("--candidate-low-match-geometry-guard-max-homography-p90-px", type=float, default=None)
    parser.add_argument("--candidate-low-match-geometry-guard-max-homography-median-px", type=float, default=None)
    parser.add_argument("--candidate-low-match-geometry-guard-min-score-mean", type=float, default=None)
    parser.add_argument("--max-formal-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-formal-correct-drop", type=int, default=0)
    parser.add_argument("--max-formal-wrong-increase", type=int, default=0)
    parser.add_argument("--formal-target-variants", default="")
    parser.add_argument("--formal-protected-variants", default="")
    parser.add_argument("--min-formal-target-correct-gain", type=int, default=0)
    parser.add_argument("--min-formal-target-total-correct-gain", type=int, default=0)
    parser.add_argument("--max-formal-target-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-formal-target-wrong-increase", type=int, default=0)
    parser.add_argument("--max-protected-variant-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-protected-variant-correct-drop", type=int, default=0)
    parser.add_argument("--max-protected-variant-wrong-increase", type=int, default=0)
    parser.add_argument("--max-guard-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-guard-correct-drop", type=int, default=0)
    parser.add_argument("--max-guard-wrong-increase", type=int, default=0)
    parser.add_argument(
        "--extra-regression-guard-set",
        action="append",
        default=[],
        help=(
            "Additional guard set under --guard-root. Defaults to strict regression thresholds unless "
            "--max-extra-guard-* is set. Use once per set or pass a comma-separated list."
        ),
    )
    parser.add_argument("--max-extra-guard-precision-drop", type=float, default=0.0)
    parser.add_argument("--max-extra-guard-correct-drop", type=int, default=0)
    parser.add_argument("--max-extra-guard-wrong-increase", type=int, default=0)
    parser.add_argument("--min-extreme-correct-gain", type=int, default=1)
    parser.add_argument("--max-extreme-precision-drop", type=float, default=0.02)
    parser.add_argument("--max-extreme-wrong-increase", type=int, default=10**12)
    parser.add_argument(
        "--dual-checkpoint-rescue-selector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After baseline/candidate sweeps, build a phase3zn-default + candidate-extreme rescue selector and promote that selected output.",
    )
    parser.add_argument(
        "--dual-checkpoint-rescue-profile",
        choices=FOV76_DUAL_CHECKPOINT_RESCUE_PROFILES,
        default="",
        help=(
            "Apply a named dual-checkpoint selector profile. "
            "fov76_ransac_minmatch16 sets min rescue matches to 16 for RANSAC-consistency candidates unless explicitly overridden."
        ),
    )
    parser.add_argument("--dual-checkpoint-rescue-label", default="dual_checkpoint_rescue_selected")
    parser.add_argument("--dual-checkpoint-rescue-target-variants", default="extreme_02,extreme_03")
    parser.add_argument("--dual-checkpoint-rescue-min-match-gain", type=int, default=1)
    parser.add_argument("--dual-checkpoint-rescue-min-rescue-matches", type=int, default=8)
    parser.add_argument("--dual-checkpoint-rescue-max-homography-p90-px", type=float, default=3.2)
    parser.add_argument("--dual-checkpoint-rescue-max-homography-median-px", type=float, default=1.8)
    parser.add_argument("--dual-checkpoint-rescue-min-score-mean", type=float, default=16.0)
    parser.add_argument("--dual-checkpoint-rescue-allow-score-mean-drop", action="store_true")
    parser.add_argument("--write-match-details", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    args._explicit_cli_options = _collect_explicit_cli_options(sys.argv[1:])
    return args


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
