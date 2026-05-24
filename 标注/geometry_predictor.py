from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Prediction:
    point: dict[str, float]
    method: str


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clamp_point(point: dict[str, float], size: tuple[int, int] | list[int]) -> dict[str, float]:
    width = max(1.0, float(size[0]))
    height = max(1.0, float(size[1]))
    return {
        "x": max(0.0, min(width - 1.0, float(point["x"]))),
        "y": max(0.0, min(height - 1.0, float(point["y"]))),
    }


def clean_matches(matches: Iterable[dict[str, Any]]) -> list[dict[str, dict[str, float]]]:
    result: list[dict[str, dict[str, float]]] = []
    for match in matches:
        point_a = match.get("a") if isinstance(match, dict) else None
        point_b = match.get("b") if isinstance(match, dict) else None
        if not isinstance(point_a, dict) or not isinstance(point_b, dict):
            continue
        if not all(_is_finite(point.get(axis)) for point in (point_a, point_b) for axis in ("x", "y")):
            continue
        result.append(
            {
                "a": {"x": float(point_a["x"]), "y": float(point_a["y"])},
                "b": {"x": float(point_b["x"]), "y": float(point_b["y"])},
            }
        )
    return result


def _local_matches(point: dict[str, float], matches: list[dict[str, dict[str, float]]], count: int) -> list[dict[str, dict[str, float]]]:
    return sorted(
        matches,
        key=lambda match: math.hypot(match["a"]["x"] - point["x"], match["a"]["y"] - point["y"]),
    )[: min(count, len(matches))]


def predict_by_scale(
    point: dict[str, float],
    size_a: tuple[int, int] | list[int],
    size_b: tuple[int, int] | list[int],
) -> Prediction:
    width_a = max(1.0, float(size_a[0]))
    height_a = max(1.0, float(size_a[1]))
    width_b = max(1.0, float(size_b[0]))
    height_b = max(1.0, float(size_b[1]))
    return Prediction(
        clamp_point({"x": point["x"] * width_b / width_a, "y": point["y"] * height_b / height_a}, size_b),
        "scale",
    )


def predict_by_translation(
    point: dict[str, float],
    matches: list[dict[str, dict[str, float]]],
    size_b: tuple[int, int] | list[int],
) -> Prediction:
    dx = sum(match["b"]["x"] - match["a"]["x"] for match in matches) / len(matches)
    dy = sum(match["b"]["y"] - match["a"]["y"] for match in matches) / len(matches)
    return Prediction(clamp_point({"x": point["x"] + dx, "y": point["y"] + dy}, size_b), "translation")


def predict_by_similarity(
    point: dict[str, float],
    matches: list[dict[str, dict[str, float]]],
    size_b: tuple[int, int] | list[int],
) -> Prediction:
    if len(matches) < 2:
        return predict_by_translation(point, matches, size_b)

    mean_ax = sum(match["a"]["x"] for match in matches) / len(matches)
    mean_ay = sum(match["a"]["y"] for match in matches) / len(matches)
    mean_bx = sum(match["b"]["x"] for match in matches) / len(matches)
    mean_by = sum(match["b"]["y"] for match in matches) / len(matches)
    denominator = 0.0
    real = 0.0
    imag = 0.0
    for match in matches:
        ax = match["a"]["x"] - mean_ax
        ay = match["a"]["y"] - mean_ay
        bx = match["b"]["x"] - mean_bx
        by = match["b"]["y"] - mean_by
        denominator += ax * ax + ay * ay
        real += ax * bx + ay * by
        imag += ax * by - ay * bx
    if denominator < 1.0e-8:
        return predict_by_translation(point, matches, size_b)

    a = real / denominator
    b = imag / denominator
    tx = mean_bx - a * mean_ax + b * mean_ay
    ty = mean_by - b * mean_ax - a * mean_ay
    return Prediction(
        clamp_point({"x": a * point["x"] - b * point["y"] + tx, "y": b * point["x"] + a * point["y"] + ty}, size_b),
        "similarity",
    )


def predict_by_affine(
    point: dict[str, float],
    matches: list[dict[str, dict[str, float]]],
    size_b: tuple[int, int] | list[int],
) -> Prediction | None:
    if len(matches) < 3:
        return None
    rows: list[list[float]] = []
    values: list[float] = []
    for match in matches:
        x = match["a"]["x"]
        y = match["a"]["y"]
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0])
        values.append(match["b"]["x"])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0])
        values.append(match["b"]["y"])
    try:
        solution, *_ = np.linalg.lstsq(np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return None
    prediction = {
        "x": float(solution[0] * point["x"] + solution[1] * point["y"] + solution[2]),
        "y": float(solution[3] * point["x"] + solution[4] * point["y"] + solution[5]),
    }
    if not _is_finite(prediction["x"]) or not _is_finite(prediction["y"]):
        return None
    return Prediction(clamp_point(prediction, size_b), "affine")


def predict_by_homography(
    point: dict[str, float],
    matches: list[dict[str, dict[str, float]]],
    size_b: tuple[int, int] | list[int],
) -> Prediction | None:
    if len(matches) < 4:
        return None
    rows: list[list[float]] = []
    values: list[float] = []
    for match in matches:
        x = match["a"]["x"]
        y = match["a"]["y"]
        u = match["b"]["x"]
        v = match["b"]["y"]
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    try:
        h, *_ = np.linalg.lstsq(np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return None
    denominator = float(h[6] * point["x"] + h[7] * point["y"] + 1.0)
    if abs(denominator) < 1.0e-8:
        return None
    prediction = {
        "x": float((h[0] * point["x"] + h[1] * point["y"] + h[2]) / denominator),
        "y": float((h[3] * point["x"] + h[4] * point["y"] + h[5]) / denominator),
    }
    if not _is_finite(prediction["x"]) or not _is_finite(prediction["y"]):
        return None
    return Prediction(clamp_point(prediction, size_b), "homography")


def predict_match_point(
    point: dict[str, float],
    matches: Iterable[dict[str, Any]],
    size_a: tuple[int, int] | list[int],
    size_b: tuple[int, int] | list[int],
) -> Prediction:
    point = {"x": float(point["x"]), "y": float(point["y"])}
    clean = clean_matches(matches)
    if not clean:
        return predict_by_scale(point, size_a, size_b)
    if len(clean) == 1:
        return predict_by_translation(point, clean, size_b)
    if len(clean) == 2:
        return predict_by_similarity(point, clean, size_b)

    local_affine = predict_by_affine(point, _local_matches(point, clean, 8), size_b)
    if len(clean) >= 4:
        local_homography = predict_by_homography(point, _local_matches(point, clean, 12), size_b)
        if local_homography is not None:
            return local_homography
    if local_affine is not None:
        return local_affine
    return predict_by_similarity(point, clean, size_b)
