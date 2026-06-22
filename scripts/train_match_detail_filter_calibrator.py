#!/usr/bin/env python3
"""Train a lightweight per-match filter calibrator from match detail CSVs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from train_match_set_rejection_calibrator import (
    evaluate_threshold,
    score_rows,
    select_feature_columns,
    train_model,
)


MATCH_FEATURE_SOURCE_COLUMNS = [
    "score",
    "pair_logit",
    "row_dustbin_logit",
    "col_dustbin_logit",
    "positive_vs_dustbin_margin",
    "raw_similarity",
    "raw_margin",
    "accept_logit",
    "accept_probability",
    "valid_fraction",
]


DEFAULT_MATCH_IMAGE_SIZE_PX = 2048.0


PREDICTION_FIELDS = [
    "label",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
    "split",
    "match_index",
    "score",
    "raw_margin",
    "accept_probability",
    "error_px",
    "correct",
    "reject_probability",
    "predicted_reject",
]


SWEEP_FIELDS = [
    "threshold",
    "train_kept_correct",
    "train_kept_wrong",
    "train_precision",
    "train_correct_retention",
    "train_wrong_reduction",
    "eval_kept_correct",
    "eval_kept_wrong",
    "eval_precision",
    "eval_correct_retention",
    "eval_wrong_reduction",
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_csv_rows_many(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_csv_rows(path))
    return rows


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _first_positive_float(row: dict[str, str], keys: Sequence[str], default: float) -> float:
    for key in keys:
        value = _float_value(row, key, default=-1.0)
        if value > 0.0:
            return value
    return default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _safe_feature_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    stddev = math.sqrt(variance)
    return stddev if stddev > 1e-12 else 1.0


def _rank_fractions(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    sorted_values = sorted(values)
    denominator = float(len(values) - 1)
    ranks: list[float] = []
    for value in values:
        positions = [index for index, sorted_value in enumerate(sorted_values) if sorted_value == value]
        ranks.append(_mean([float(position) for position in positions]) / denominator)
    return ranks


def _pair_key(match: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        match.get("split", ""),
        match.get("pair_index", ""),
        match.get("base_id", ""),
        match.get("reference_variant", ""),
        match.get("target_variant", ""),
    )


def _norm_distance(x: float, y: float, center_x: float, center_y: float, scale: float) -> float:
    return math.hypot(x - center_x, y - center_y) / (scale if scale > 1e-12 else 1.0)


def _consensus_fraction(distances: Sequence[float], threshold_px: float) -> float:
    if not distances:
        return 0.0
    return sum(1 for value in distances if value <= threshold_px) / len(distances)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, fraction)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _homography_residual_distances(
    ax: Sequence[float],
    ay: Sequence[float],
    bx: Sequence[float],
    by: Sequence[float],
) -> tuple[list[float], bool]:
    if len(ax) < 4:
        return [0.0] * len(ax), False
    points_a = np.asarray([[x_value, y_value] for x_value, y_value in zip(ax, ay)], dtype=np.float32)
    points_b = np.asarray([[x_value, y_value] for x_value, y_value in zip(bx, by)], dtype=np.float32)
    homography, _mask = cv2.findHomography(
        points_a,
        points_b,
        cv2.USAC_MAGSAC,
        5.0,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None:
        return [0.0] * len(ax), False
    homogeneous = np.concatenate([points_a, np.ones((points_a.shape[0], 1), dtype=np.float32)], axis=1)
    projected = (homography @ homogeneous.T).T
    denominator = projected[:, 2:3]
    if np.any(np.abs(denominator) < 1e-12):
        return [0.0] * len(ax), False
    projected_xy = projected[:, :2] / denominator
    residuals = np.linalg.norm(projected_xy - points_b, axis=1)
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in residuals], True


def _local_displacement_distances(
    ax: Sequence[float],
    ay: Sequence[float],
    dx: Sequence[float],
    dy: Sequence[float],
    *,
    neighbor_count: int = 8,
) -> list[float]:
    distances: list[float] = []
    for index, (x_value, y_value, dx_value, dy_value) in enumerate(zip(ax, ay, dx, dy)):
        neighbors: list[tuple[float, int]] = []
        for other_index, (other_x, other_y) in enumerate(zip(ax, ay)):
            if other_index == index:
                continue
            distance = math.hypot(other_x - x_value, other_y - y_value)
            neighbors.append((distance, other_index))
        neighbors.sort()
        selected = [other_index for _, other_index in neighbors[:neighbor_count]]
        if not selected:
            distances.append(0.0)
            continue
        local_dx = statistics.median(dx[other_index] for other_index in selected)
        local_dy = statistics.median(dy[other_index] for other_index in selected)
        distances.append(math.hypot(dx_value - local_dx, dy_value - local_dy))
    return distances


def _pair_context_features(match_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, match in enumerate(match_rows):
        groups[_pair_key(match)].append((index, match))

    contexts: list[dict[str, str]] = [{} for _ in match_rows]
    for group in groups.values():
        indices = [index for index, _ in group]
        matches = [match for _, match in group]
        scores = [_float_value(match, "score") for match in matches]
        raw_margins = [_float_value(match, "raw_margin") for match in matches]
        accept_probabilities = [_float_value(match, "accept_probability") for match in matches]
        ax = [_float_value(match, "point_a_x_px") for match in matches]
        ay = [_float_value(match, "point_a_y_px") for match in matches]
        bx = [_float_value(match, "point_b_x_px") for match in matches]
        by = [_float_value(match, "point_b_y_px") for match in matches]
        dx = [bx_value - ax_value for ax_value, bx_value in zip(ax, bx)]
        dy = [by_value - ay_value for ay_value, by_value in zip(ay, by)]

        score_mean = _mean(scores)
        score_std = _std(scores)
        raw_margin_mean = _mean(raw_margins)
        raw_margin_std = _std(raw_margins)
        accept_probability_mean = _mean(accept_probabilities)
        accept_probability_std = _std(accept_probabilities)
        score_ranks = _rank_fractions(scores)

        center_a_x = _mean(ax)
        center_a_y = _mean(ay)
        center_b_x = _mean(bx)
        center_b_y = _mean(by)
        scale_a = math.hypot(max(ax) - min(ax), max(ay) - min(ay)) if ax and ay else 1.0
        scale_b = math.hypot(max(bx) - min(bx), max(by) - min(by)) if bx and by else 1.0
        median_dx = statistics.median(dx) if dx else 0.0
        median_dy = statistics.median(dy) if dy else 0.0
        displacement_distances = [
            math.hypot(dx_value - median_dx, dy_value - median_dy)
            for dx_value, dy_value in zip(dx, dy)
        ]
        displacement_scale = statistics.median(displacement_distances) if displacement_distances else 1.0
        if displacement_scale < 1.0:
            displacement_scale = 1.0
        raw_displacement_mad = statistics.median(displacement_distances) if displacement_distances else 0.0
        consensus_fraction_5px = _consensus_fraction(displacement_distances, 5.0)
        consensus_fraction_10px = _consensus_fraction(displacement_distances, 10.0)
        local_displacement_distances = _local_displacement_distances(ax, ay, dx, dy)
        local_displacement_scale = (
            statistics.median(local_displacement_distances)
            if local_displacement_distances
            else 1.0
        )
        if local_displacement_scale < 1.0:
            local_displacement_scale = 1.0
        homography_residual_distances, homography_residual_valid = _homography_residual_distances(ax, ay, bx, by)
        homography_residual_median = (
            statistics.median(homography_residual_distances)
            if homography_residual_distances
            else 0.0
        )
        homography_residual_p90 = _percentile(homography_residual_distances, 0.90)
        homography_residual_scale = homography_residual_median
        if homography_residual_scale < 1.0:
            homography_residual_scale = 1.0

        for local_index, source_index in enumerate(indices):
            contexts[source_index] = {
                "feature_pair_match_count": str(len(matches)),
                "feature_pair_score_mean": _fmt_float(score_mean),
                "feature_pair_score_zscore": _fmt_float((scores[local_index] - score_mean) / score_std),
                "feature_pair_score_rank_fraction": _fmt_float(score_ranks[local_index]),
                "feature_pair_raw_margin_mean": _fmt_float(raw_margin_mean),
                "feature_pair_raw_margin_zscore": _fmt_float(
                    (raw_margins[local_index] - raw_margin_mean) / raw_margin_std
                ),
                "feature_pair_accept_probability_mean": _fmt_float(accept_probability_mean),
                "feature_pair_accept_probability_zscore": _fmt_float(
                    (accept_probabilities[local_index] - accept_probability_mean) / accept_probability_std
                ),
                "feature_pair_a_center_distance_norm": _fmt_float(
                    _norm_distance(ax[local_index], ay[local_index], center_a_x, center_a_y, scale_a)
                ),
                "feature_pair_b_center_distance_norm": _fmt_float(
                    _norm_distance(bx[local_index], by[local_index], center_b_x, center_b_y, scale_b)
                ),
                "feature_pair_displacement_median_distance_px": _fmt_float(
                    displacement_distances[local_index]
                ),
                "feature_pair_displacement_median_distance_norm": _fmt_float(
                    displacement_distances[local_index] / displacement_scale
                ),
                "feature_pair_displacement_mad_px": _fmt_float(raw_displacement_mad),
                "feature_pair_displacement_consensus_fraction_5px": _fmt_float(consensus_fraction_5px),
                "feature_pair_displacement_consensus_fraction_10px": _fmt_float(consensus_fraction_10px),
                "feature_pair_is_displacement_consensus_5px": (
                    "1" if displacement_distances[local_index] <= 5.0 else "0"
                ),
                "feature_pair_is_displacement_consensus_10px": (
                    "1" if displacement_distances[local_index] <= 10.0 else "0"
                ),
                "feature_pair_local_displacement_median_distance_px": _fmt_float(
                    local_displacement_distances[local_index]
                ),
                "feature_pair_local_displacement_median_distance_norm": _fmt_float(
                    local_displacement_distances[local_index] / local_displacement_scale
                ),
                "feature_pair_homography_residual_valid": "1" if homography_residual_valid else "0",
                "feature_pair_homography_residual_px": _fmt_float(
                    homography_residual_distances[local_index]
                ),
                "feature_pair_homography_residual_norm": _fmt_float(
                    homography_residual_distances[local_index] / homography_residual_scale
                ),
                "feature_pair_homography_residual_median_px": _fmt_float(homography_residual_median),
                "feature_pair_homography_residual_p90_px": _fmt_float(homography_residual_p90),
                "feature_pair_homography_consensus_fraction_4px": _fmt_float(
                    _consensus_fraction(homography_residual_distances, 4.0)
                ),
                "feature_pair_is_homography_consensus_4px": (
                    "1" if homography_residual_distances[local_index] <= 4.0 else "0"
                ),
            }
    return contexts


def _variant_feature_values(match: dict[str, str]) -> dict[str, str]:
    reference_variant = match.get("reference_variant", "")
    target_variant = match.get("target_variant", "")
    pair_type = match.get("pair_type", "")
    reference_token = _safe_feature_token(reference_variant)
    target_token = _safe_feature_token(target_variant)
    features = {
        "feature_reference_is_extreme": "1" if reference_variant.startswith("extreme") else "0",
        "feature_target_is_extreme": "1" if target_variant.startswith("extreme") else "0",
        "feature_variant_changed": "1" if reference_variant != target_variant else "0",
        f"feature_reference_variant_{reference_token}": "1",
        f"feature_target_variant_{target_token}": "1",
        f"feature_variant_transition_{reference_token}_to_{target_token}": "1",
        f"feature_score_x_target_variant_{target_token}": match.get("score", "0"),
        f"feature_raw_margin_x_target_variant_{target_token}": match.get("raw_margin", "0"),
        f"feature_accept_probability_x_target_variant_{target_token}": match.get("accept_probability", "0"),
    }
    if pair_type:
        features[f"feature_pair_type_{_safe_feature_token(pair_type)}"] = "1"
    return features


def _true_geometry_feature_values(match: dict[str, str]) -> dict[str, str]:
    error_px = _float_value(match, "error_px")
    valid_fraction = _float_value(match, "valid_fraction")
    error_le_5px = error_px <= 5.0
    valid_ge_010 = valid_fraction >= 0.10
    return {
        "feature_true_geometry_error_px": _fmt_float(error_px),
        "feature_true_geometry_error_sq_px": _fmt_float(error_px * error_px),
        "feature_true_geometry_error_le_5px": "1" if error_le_5px else "0",
        "feature_true_geometry_valid_ge_0_10": "1" if valid_ge_010 else "0",
        "feature_true_geometry_rule_error5_valid010": "1" if error_le_5px and valid_ge_010 else "0",
        "feature_true_geometry_valid_fraction": _fmt_float(valid_fraction),
    }


def _match_coordinate_feature_values(match: dict[str, str]) -> dict[str, str]:
    ax = _float_value(match, "point_a_x_px")
    ay = _float_value(match, "point_a_y_px")
    bx = _float_value(match, "point_b_x_px")
    by = _float_value(match, "point_b_y_px")
    dx = bx - ax
    dy = by - ay
    displacement_magnitude = math.hypot(dx, dy)
    angle_cos = dx / displacement_magnitude if displacement_magnitude > 1e-12 else 0.0
    angle_sin = dy / displacement_magnitude if displacement_magnitude > 1e-12 else 0.0
    a_width = _first_positive_float(
        match,
        ("image_a_width_px", "view_a_width_px", "width_a_px", "image_width_a_px"),
        DEFAULT_MATCH_IMAGE_SIZE_PX,
    )
    a_height = _first_positive_float(
        match,
        ("image_a_height_px", "view_a_height_px", "height_a_px", "image_height_a_px"),
        DEFAULT_MATCH_IMAGE_SIZE_PX,
    )
    b_width = _first_positive_float(
        match,
        ("image_b_width_px", "view_b_width_px", "width_b_px", "image_width_b_px"),
        DEFAULT_MATCH_IMAGE_SIZE_PX,
    )
    b_height = _first_positive_float(
        match,
        ("image_b_height_px", "view_b_height_px", "height_b_px", "image_height_b_px"),
        DEFAULT_MATCH_IMAGE_SIZE_PX,
    )
    displacement_width_scale = max(a_width, b_width, 1.0)
    displacement_height_scale = max(a_height, b_height, 1.0)
    displacement_magnitude_scale = math.hypot(displacement_width_scale, displacement_height_scale)
    if displacement_magnitude_scale < 1.0:
        displacement_magnitude_scale = 1.0
    return {
        "feature_point_a_x_norm": _fmt_float(ax / max(a_width, 1.0)),
        "feature_point_a_y_norm": _fmt_float(ay / max(a_height, 1.0)),
        "feature_point_b_x_norm": _fmt_float(bx / max(b_width, 1.0)),
        "feature_point_b_y_norm": _fmt_float(by / max(b_height, 1.0)),
        "feature_displacement_dx_px": _fmt_float(dx),
        "feature_displacement_dy_px": _fmt_float(dy),
        "feature_displacement_magnitude_px": _fmt_float(displacement_magnitude),
        "feature_displacement_dx_norm": _fmt_float(dx / displacement_width_scale),
        "feature_displacement_dy_norm": _fmt_float(dy / displacement_height_scale),
        "feature_displacement_magnitude_norm": _fmt_float(displacement_magnitude / displacement_magnitude_scale),
        "feature_displacement_angle_cos": _fmt_float(angle_cos),
        "feature_displacement_angle_sin": _fmt_float(angle_sin),
    }


def build_training_rows(
    match_rows: Sequence[dict[str, str]],
    *,
    include_true_geometry_features: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    context_rows = _pair_context_features(match_rows)
    variant_feature_names = sorted(
        {
            feature_name
            for match in match_rows
            for feature_name in _variant_feature_values(match)
        }
    )
    for match, context in zip(match_rows, context_rows):
        correct = 1 if _int_value(match, "correct") > 0 else 0
        variant_features = _variant_feature_values(match)
        row = {
            "label": match.get("label", ""),
            "pair_index": match.get("pair_index", ""),
            "base_id": match.get("base_id", ""),
            "reference_variant": match.get("reference_variant", ""),
            "target_variant": match.get("target_variant", ""),
            "split": match.get("split", ""),
            "match_index": match.get("match_index", ""),
            "score": match.get("score", ""),
            "raw_margin": match.get("raw_margin", ""),
            "accept_probability": match.get("accept_probability", ""),
            "error_px": match.get("error_px", ""),
            "correct": str(correct),
            "reject_label": "0" if correct else "1",
            "pfm_matches": "1",
            "pfm_correct": str(correct),
            "pfm_wrong": "0" if correct else "1",
            "pfm_precision": "1.0" if correct else "0.0",
        }
        for feature_name in variant_feature_names:
            row[feature_name] = variant_features.get(feature_name, "0")
        if include_true_geometry_features:
            row.update(_true_geometry_feature_values(match))
        row.update(_match_coordinate_feature_values(match))
        for name in MATCH_FEATURE_SOURCE_COLUMNS:
            row[f"feature_{name}"] = match.get(name, "0")
        row.update(context)
        rows.append(row)
    return rows


def select_match_feature_columns(rows: Sequence[dict[str, str]]) -> list[str]:
    return select_feature_columns(rows)


def filter_feature_columns(feature_columns: Sequence[str], pattern: str) -> list[str]:
    if not pattern:
        return list(feature_columns)
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid --feature-name-regex: {exc}") from exc
    filtered = [name for name in feature_columns if compiled.search(name)]
    if not filtered:
        raise ValueError(f"--feature-name-regex matched no feature columns: {pattern}")
    return filtered


def _evenly_sample(rows: Sequence[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    if limit == 1:
        return [rows[0]]
    last_index = len(rows) - 1
    sampled: list[dict[str, str]] = []
    used: set[int] = set()
    for item_index in range(limit):
        source_index = round(item_index * last_index / (limit - 1))
        while source_index in used and source_index < last_index:
            source_index += 1
        while source_index in used and source_index > 0:
            source_index -= 1
        used.add(source_index)
        sampled.append(rows[source_index])
    return sampled


def _limit_training_rows_by_key(
    rows: Sequence[dict[str, str]],
    max_train_rows: int,
    *,
    balance_key: str,
) -> list[dict[str, str]]:
    if max_train_rows <= 0 or len(rows) <= max_train_rows:
        return list(rows)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get(balance_key, ""), row.get("reject_label", "0"))].append(row)
    if not groups:
        return []

    selected: list[dict[str, str]] = []
    selected_ids: set[int] = set()
    sorted_keys = sorted(groups)
    base_quota = max_train_rows // len(sorted_keys)
    extra = max_train_rows % len(sorted_keys)
    for key_index, key in enumerate(sorted_keys):
        quota = base_quota + (1 if key_index < extra else 0)
        if quota <= 0:
            continue
        sampled = _evenly_sample(groups[key], quota)
        for row in sampled:
            if id(row) not in selected_ids and len(selected) < max_train_rows:
                selected.append(row)
                selected_ids.add(id(row))

    if len(selected) >= max_train_rows:
        return selected

    for key in sorted_keys:
        remaining = [row for row in groups[key] if id(row) not in selected_ids]
        for row in _evenly_sample(remaining, max_train_rows - len(selected)):
            if id(row) not in selected_ids and len(selected) < max_train_rows:
                selected.append(row)
                selected_ids.add(id(row))
        if len(selected) >= max_train_rows:
            break
    return selected


def limit_training_rows(
    rows: Sequence[dict[str, str]],
    max_train_rows: int,
    *,
    balance_key: str = "",
) -> list[dict[str, str]]:
    if max_train_rows <= 0 or len(rows) <= max_train_rows:
        return list(rows)
    if balance_key:
        return _limit_training_rows_by_key(rows, max_train_rows, balance_key=balance_key)
    reject_rows = [row for row in rows if row.get("reject_label") == "1"]
    keep_rows = [row for row in rows if row.get("reject_label") != "1"]
    if not reject_rows or not keep_rows:
        return _evenly_sample(rows, max_train_rows)

    min_class_quota = max(1, max_train_rows // 2)
    reject_quota = min(len(reject_rows), min_class_quota)
    keep_quota = min(len(keep_rows), max_train_rows - reject_quota)
    if keep_quota == len(keep_rows) and reject_quota < len(reject_rows):
        reject_quota = min(len(reject_rows), max_train_rows - keep_quota)
    elif reject_quota == len(reject_rows) and keep_quota < len(keep_rows):
        keep_quota = min(len(keep_rows), max_train_rows - reject_quota)

    sampled = _evenly_sample(reject_rows, reject_quota) + _evenly_sample(keep_rows, keep_quota)
    return sampled[:max_train_rows]


def _summarize_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "matches": metrics["pfm_matches"],
        "correct": metrics["pfm_correct"],
        "wrong": metrics["pfm_wrong"],
        "precision": metrics["pfm_precision"],
        "kept_matches": metrics["kept_pfm_matches"],
        "kept_correct": metrics["kept_pfm_correct"],
        "kept_wrong": metrics["kept_pfm_wrong"],
        "kept_precision": metrics["kept_precision"],
        "rejected_correct": metrics["rejected_pfm_correct"],
        "rejected_wrong": metrics["rejected_pfm_wrong"],
        "correct_retention": metrics["correct_retention"],
        "wrong_reduction": metrics["wrong_reduction"],
        "threshold": metrics["threshold"],
        "predicted_reject_rows": metrics["predicted_reject_rows"],
        "label_accuracy": metrics["label_accuracy"],
    }


def summarize_threshold(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    threshold: float,
) -> dict[str, object]:
    return _summarize_metrics(
        evaluate_threshold(
            rows,
            scores,
            threshold=threshold,
            label_column="reject_label",
        )
    )


def _candidate_thresholds(scores: Sequence[float], *, max_thresholds: int = 0) -> list[float]:
    if not scores:
        return [1.000001]
    values = sorted({max(0.0, min(1.0, score)) for score in scores})
    thresholds = [0.0, *values, min(1.000001, values[-1] + 1e-6)]
    if max_thresholds <= 0 or len(thresholds) <= max_thresholds:
        return thresholds
    if max_thresholds == 1:
        return [thresholds[-1]]
    sampled: list[float] = []
    last_index = len(thresholds) - 1
    for index in range(max_thresholds):
        sampled.append(thresholds[round(index * last_index / (max_thresholds - 1))])
    return sorted(set(sampled))


def choose_match_threshold(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    min_kept_correct_ratio: float,
    max_thresholds: int,
) -> float:
    if min_kept_correct_ratio < 0.0 or min_kept_correct_ratio > 1.0:
        raise ValueError("min_kept_correct_ratio must be in [0, 1]")
    candidates: list[tuple[float, float, int, float, float]] = []
    for threshold in _candidate_thresholds(scores, max_thresholds=max_thresholds):
        metrics = summarize_threshold(rows, scores, threshold=threshold)
        correct_retention = float(metrics["correct_retention"])
        if correct_retention < min_kept_correct_ratio:
            continue
        candidates.append(
            (
                float(metrics["wrong_reduction"]),
                float(metrics["kept_precision"]),
                -int(metrics["kept_wrong"]),
                correct_retention,
                threshold,
            )
        )
    if not candidates:
        return 1.000001
    return max(candidates)[-1]


def choose_match_wrong_cap_threshold(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    max_kept_wrong: int,
    max_thresholds: int,
) -> float:
    if max_kept_wrong < 0:
        raise ValueError("max_kept_wrong must be nonnegative")
    candidates: list[tuple[int, float, int, float]] = []
    for threshold in _candidate_thresholds(scores, max_thresholds=max_thresholds):
        metrics = summarize_threshold(rows, scores, threshold=threshold)
        kept_wrong = int(metrics["kept_wrong"])
        if kept_wrong > max_kept_wrong:
            continue
        candidates.append(
            (
                int(metrics["kept_correct"]),
                float(metrics["kept_precision"]),
                -int(metrics["predicted_reject_rows"]),
                threshold,
            )
        )
    if not candidates:
        return 1.000001
    return max(candidates)[-1]


def build_threshold_sweep(
    train_rows: Sequence[dict[str, str]],
    train_scores: Sequence[float],
    eval_rows: Sequence[dict[str, str]],
    eval_scores: Sequence[float],
    *,
    max_thresholds: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in _candidate_thresholds([*train_scores, *eval_scores], max_thresholds=max_thresholds):
        train_metrics = summarize_threshold(train_rows, train_scores, threshold=threshold)
        eval_metrics = summarize_threshold(eval_rows, eval_scores, threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "train_kept_correct": train_metrics["kept_correct"],
                "train_kept_wrong": train_metrics["kept_wrong"],
                "train_precision": train_metrics["kept_precision"],
                "train_correct_retention": train_metrics["correct_retention"],
                "train_wrong_reduction": train_metrics["wrong_reduction"],
                "eval_kept_correct": eval_metrics["kept_correct"],
                "eval_kept_wrong": eval_metrics["kept_wrong"],
                "eval_precision": eval_metrics["kept_precision"],
                "eval_correct_retention": eval_metrics["correct_retention"],
                "eval_wrong_reduction": eval_metrics["wrong_reduction"],
            }
        )
    return rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_predictions(path: Path, rows: Sequence[dict[str, str]], scores: Sequence[float], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row, score in zip(rows, scores):
            payload = {field: row.get(field, "") for field in PREDICTION_FIELDS}
            payload["reject_probability"] = _fmt_float(score)
            payload["predicted_reject"] = "1" if score >= threshold else "0"
            writer.writerow(payload)


def _write_sweep(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key, value in list(formatted.items()):
                if isinstance(value, float):
                    formatted[key] = _fmt_float(value)
            writer.writerow(formatted)


def _prediction_name(eval_match_details: Path) -> str:
    stem = eval_match_details.stem.replace("details", "predictions")
    return f"{stem}.csv"


def _prediction_name_for_paths(eval_match_details: Sequence[Path]) -> str:
    if len(eval_match_details) == 1:
        return _prediction_name(eval_match_details[0])
    return "all_match_predictions.csv"


def _path_payload(paths: Sequence[Path]) -> str | list[str]:
    if len(paths) == 1:
        return str(paths[0])
    return [str(path) for path in paths]


def _write_report_html(
    path: Path,
    *,
    train_match_details: Sequence[Path],
    eval_match_details: Sequence[Path],
    predictions_csv: Path,
    sweep_csv: Path,
    summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Match-detail filter calibrator</title>",
                "<h1>Match-detail filter calibrator</h1>",
                f"<p>train_match_details={html.escape(json.dumps(_path_payload(train_match_details), ensure_ascii=False))}</p>",
                f"<p>eval_match_details={html.escape(json.dumps(_path_payload(eval_match_details), ensure_ascii=False))}</p>",
                f"<p>predictions_csv={html.escape(str(predictions_csv))}</p>",
                f"<p>threshold_sweep_csv={html.escape(str(sweep_csv))}</p>",
                "<h2>Summary</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-match-details", type=Path, action="append", required=True)
    parser.add_argument("--eval-match-details", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--min-kept-correct-ratio", type=float, default=0.99)
    parser.add_argument(
        "--threshold-objective",
        choices=["kept_correct_ratio", "pfm_wrong_cap"],
        default="kept_correct_ratio",
    )
    parser.add_argument(
        "--threshold-selection-source",
        choices=["train", "eval"],
        default="train",
    )
    parser.add_argument("--max-kept-wrong", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--balance-sampling-key", default="")
    parser.add_argument("--max-thresholds", type=int, default=500)
    parser.add_argument("--include-true-geometry-features", action="store_true")
    parser.add_argument(
        "--feature-name-regex",
        default="",
        help="Optional regular expression used to keep only matching feature columns.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_match_details = list(args.train_match_details)
    eval_match_details = list(args.eval_match_details)
    train_source_rows = build_training_rows(
        _read_csv_rows_many(train_match_details),
        include_true_geometry_features=bool(args.include_true_geometry_features),
    )
    train_rows = limit_training_rows(
        train_source_rows,
        int(args.max_train_rows),
        balance_key=str(args.balance_sampling_key),
    )
    eval_rows = build_training_rows(
        _read_csv_rows_many(eval_match_details),
        include_true_geometry_features=bool(args.include_true_geometry_features),
    )
    feature_columns = filter_feature_columns(
        select_match_feature_columns(train_rows),
        str(args.feature_name_regex),
    )
    model = train_model(
        train_rows,
        feature_columns=feature_columns,
        label_column="reject_label",
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
    )
    train_scores = score_rows(model, train_rows)
    eval_scores = score_rows(model, eval_rows)
    if args.threshold_selection_source == "eval":
        threshold_rows = eval_rows
        threshold_scores = eval_scores
    else:
        threshold_rows = train_rows
        threshold_scores = train_scores
    max_kept_wrong = (
        int(args.max_kept_wrong)
        if args.max_kept_wrong is not None
        else sum(_int_value(row, "pfm_wrong") for row in threshold_rows)
    )
    if max_kept_wrong < 0:
        raise ValueError("max_kept_wrong must be nonnegative")
    if args.threshold_objective == "pfm_wrong_cap":
        threshold = choose_match_wrong_cap_threshold(
            threshold_rows,
            threshold_scores,
            max_kept_wrong=max_kept_wrong,
            max_thresholds=int(args.max_thresholds),
        )
    else:
        threshold = choose_match_threshold(
            threshold_rows,
            threshold_scores,
            min_kept_correct_ratio=float(args.min_kept_correct_ratio),
            max_thresholds=int(args.max_thresholds),
        )
    train_summary = summarize_threshold(train_rows, train_scores, threshold=threshold)
    eval_summary = summarize_threshold(eval_rows, eval_scores, threshold=threshold)
    threshold_selection_summary = summarize_threshold(threshold_rows, threshold_scores, threshold=threshold)
    sweep = build_threshold_sweep(
        train_rows,
        train_scores,
        eval_rows,
        eval_scores,
        max_thresholds=int(args.max_thresholds),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / _prediction_name_for_paths(eval_match_details)
    sweep_csv = output_dir / "threshold_sweep.csv"
    summary_json = output_dir / "summary.json"
    model_json = output_dir / "model.json"
    output_html = output_dir / "index.html"

    model_payload = {
        "type": "standardized_logistic_regression_match_filter",
        "feature_columns": model.feature_columns,
        "means": model.means,
        "scales": model.scales,
        "weights": model.weights,
        "bias": model.bias,
        "label_column": model.label_column,
        "threshold": threshold,
        "include_true_geometry_features": bool(args.include_true_geometry_features),
    }
    summary = {
        "train_match_details": _path_payload(train_match_details),
        "eval_match_details": _path_payload(eval_match_details),
        "train_source_matches": len(train_source_rows),
        "max_train_rows": int(args.max_train_rows),
        "balance_sampling_key": str(args.balance_sampling_key),
        "feature_name_regex": str(args.feature_name_regex),
        "max_thresholds": int(args.max_thresholds),
        "feature_columns": model.feature_columns,
        "include_true_geometry_features": bool(args.include_true_geometry_features),
        "threshold_objective": str(args.threshold_objective),
        "threshold_selection_source": str(args.threshold_selection_source),
        "max_kept_wrong": max_kept_wrong,
        "threshold": threshold,
        "threshold_selection": threshold_selection_summary,
        "train": train_summary,
        "eval": eval_summary,
    }
    _write_json(model_json, model_payload)
    _write_json(summary_json, summary)
    _write_predictions(predictions_csv, eval_rows, eval_scores, threshold)
    _write_sweep(sweep_csv, sweep)
    _write_report_html(
        output_html,
        train_match_details=train_match_details,
        eval_match_details=eval_match_details,
        predictions_csv=predictions_csv,
        sweep_csv=sweep_csv,
        summary=summary,
    )
    print(
        "match_filter "
        f"train_matches={train_summary['matches']} eval_matches={eval_summary['matches']} "
        f"threshold={threshold:.6f} eval_kept_correct={eval_summary['kept_correct']} "
        f"eval_kept_wrong={eval_summary['kept_wrong']} output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
