#!/usr/bin/env python3
"""Analyze spatial coverage of match-detail CSVs and sweep simple coverage selectors."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class CoverageConfig:
    grid_size: int = 8
    image_size: float = 768.0
    stage_min_correct: int = 0
    stage_max_wrong: int = 10**12
    stage_min_precision: float = 0.0
    stage_min_coverage: float = 0.0
    stage_max_lg_only_cells: int = 10**12


@dataclass(frozen=True)
class MatchRow:
    split: str
    pair_index: int
    base_id: str
    reference_variant: str
    target_variant: str
    x_a: float
    y_a: float
    x_b: float
    y_b: float
    score: float
    correct: bool
    reject_probability: float = 0.0
    valid_fraction: float = 0.0

    @property
    def key(self) -> tuple[str, int]:
        return self.split, self.pair_index


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = (row.get(key) or "").strip()
    if value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _parse_source(value: str) -> SourceSpec:
    parts = [part.strip() for part in value.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("source must be formatted as name,path")
    return SourceSpec(name=parts[0], path=Path(parts[1]))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_from_csv(row: dict[str, str]) -> MatchRow:
    return MatchRow(
        split=row.get("split", ""),
        pair_index=int(round(_float(row, "pair_index"))),
        base_id=row.get("base_id", ""),
        reference_variant=row.get("reference_variant", ""),
        target_variant=row.get("target_variant", ""),
        x_a=_float(row, "point_a_x_px"),
        y_a=_float(row, "point_a_y_px"),
        x_b=_float(row, "point_b_x_px"),
        y_b=_float(row, "point_b_y_px"),
        score=_float(row, "score"),
        correct=_float(row, "correct") > 0.0,
        reject_probability=_float(row, "reject_probability", 0.0),
        valid_fraction=_float(row, "valid_fraction", 0.0),
    )


def load_match_rows(sources: Sequence[SourceSpec]) -> list[MatchRow]:
    rows: list[MatchRow] = []
    for source in sources:
        rows.extend(_row_from_csv(row) for row in _read_csv(source.path))
    return rows


def load_variant_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("thresholds_by_target_variant") or payload.get("thresholds") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"missing thresholds_by_target_variant in {path}")
    return {str(key): float(value) for key, value in raw.items()}


def group_rows(rows: Iterable[MatchRow]) -> dict[tuple[str, int], list[MatchRow]]:
    grouped: dict[tuple[str, int], list[MatchRow]] = {}
    for row in rows:
        grouped.setdefault(row.key, []).append(row)
    return grouped


def cell_for_point(x: float, y: float, config: CoverageConfig) -> tuple[int, int]:
    ix = max(0, min(config.grid_size - 1, int(math.floor(x / config.image_size * config.grid_size))))
    iy = max(0, min(config.grid_size - 1, int(math.floor(y / config.image_size * config.grid_size))))
    return ix, iy


def view_cells(
    rows: Sequence[MatchRow],
    *,
    config: CoverageConfig,
    correct_only: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells_a: set[tuple[int, int]] = set()
    cells_b: set[tuple[int, int]] = set()
    for row in rows:
        if correct_only and not row.correct:
            continue
        cells_a.add(cell_for_point(row.x_a, row.y_a, config))
        cells_b.add(cell_for_point(row.x_b, row.y_b, config))
    return cells_a, cells_b


def union_cells(
    rows: Sequence[MatchRow],
    *,
    config: CoverageConfig,
    correct_only: bool,
) -> set[tuple[str, int, int]]:
    cells: set[tuple[str, int, int]] = set()
    cells_a, cells_b = view_cells(rows, config=config, correct_only=correct_only)
    cells.update(("a", x, y) for x, y in cells_a)
    cells.update(("b", x, y) for x, y in cells_b)
    return cells


def _bbox_area(points: np.ndarray, config: CoverageConfig) -> float:
    if points.shape[0] < 2:
        return 0.0
    span = np.ptp(points, axis=0)
    return float(max(0.0, span[0]) * max(0.0, span[1]) / (config.image_size * config.image_size))


def _entropy(rows: Sequence[MatchRow], *, config: CoverageConfig, correct_only: bool) -> float:
    counts = np.zeros((config.grid_size, config.grid_size), dtype=np.float64)
    for row in rows:
        if correct_only and not row.correct:
            continue
        for x, y in ((row.x_a, row.y_a), (row.x_b, row.y_b)):
            ix, iy = cell_for_point(x, y, config)
            counts[iy, ix] += 1.0
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts.reshape(-1) / total
    probs = probs[probs > 0.0]
    return float(-(probs * np.log(probs)).sum() / math.log(config.grid_size * config.grid_size))


def _largest_cell_ratio(rows: Sequence[MatchRow], *, config: CoverageConfig, correct_only: bool) -> float:
    counts: dict[tuple[str, int, int], int] = {}
    total = 0
    for row in rows:
        if correct_only and not row.correct:
            continue
        total += 1
        ax, ay = cell_for_point(row.x_a, row.y_a, config)
        bx, by = cell_for_point(row.x_b, row.y_b, config)
        counts[("a", ax, ay)] = counts.get(("a", ax, ay), 0) + 1
        counts[("b", bx, by)] = counts.get(("b", bx, by), 0) + 1
    if total <= 0 or not counts:
        return 0.0
    return float(max(counts.values()) / total)


def _quadrant_min_correct(rows: Sequence[MatchRow], config: CoverageConfig) -> int:
    counts = [0, 0, 0, 0]
    for row in rows:
        if not row.correct:
            continue
        for x, y in ((row.x_a, row.y_a), (row.x_b, row.y_b)):
            qx = 1 if x >= config.image_size / 2 else 0
            qy = 1 if y >= config.image_size / 2 else 0
            counts[qy * 2 + qx] += 1
    return min(counts)


def coverage_metrics(rows: Sequence[MatchRow], config: CoverageConfig) -> dict[str, float | int]:
    matches = len(rows)
    correct = sum(1 for row in rows if row.correct)
    wrong = matches - correct
    correct_rows = [row for row in rows if row.correct]
    cells_a, cells_b = view_cells(rows, config=config, correct_only=True)
    all_cells = union_cells(rows, config=config, correct_only=True)
    points_a = np.asarray([[row.x_a, row.y_a] for row in correct_rows], dtype=np.float32)
    points_b = np.asarray([[row.x_b, row.y_b] for row in correct_rows], dtype=np.float32)
    total_cells = config.grid_size * config.grid_size
    return {
        "matches": matches,
        "correct": correct,
        "wrong": wrong,
        "precision": correct / matches if matches else 0.0,
        "coverage_a": len(cells_a) / total_cells,
        "coverage_b": len(cells_b) / total_cells,
        "coverage_mean": (len(cells_a) + len(cells_b)) / (2 * total_cells),
        "correct_cell_union": len(all_cells),
        "entropy": _entropy(rows, config=config, correct_only=True),
        "bbox_area_a": _bbox_area(points_a, config),
        "bbox_area_b": _bbox_area(points_b, config),
        "bbox_area_mean": (_bbox_area(points_a, config) + _bbox_area(points_b, config)) / 2.0,
        "largest_cell_ratio": _largest_cell_ratio(rows, config=config, correct_only=True),
        "quadrant_min_correct": _quadrant_min_correct(rows, config),
    }


def aggregate_pair_metrics(
    grouped: dict[tuple[str, int], list[MatchRow]],
    config: CoverageConfig,
) -> dict[str, float | int]:
    rows = list(itertools.chain.from_iterable(grouped.values()))
    base = coverage_metrics(rows, config)
    pair_metrics = [coverage_metrics(items, config) for items in grouped.values()]
    for key in [
        "coverage_mean",
        "entropy",
        "bbox_area_mean",
        "largest_cell_ratio",
        "quadrant_min_correct",
        "correct_cell_union",
    ]:
        values = [float(item[key]) for item in pair_metrics]
        base[f"pair_mean_{key}"] = float(np.mean(values)) if values else 0.0
        base[f"pair_median_{key}"] = float(np.median(values)) if values else 0.0
        base[f"pair_p10_{key}"] = float(np.percentile(values, 10)) if values else 0.0
    base["pair_count"] = len(grouped)
    return base


def compare_to_lightglue(
    lightglue: dict[tuple[str, int], list[MatchRow]],
    candidate: dict[tuple[str, int], list[MatchRow]],
    config: CoverageConfig,
) -> dict[str, float | int]:
    keys = sorted(set(lightglue) & set(candidate))
    lg_only_cells = 0
    candidate_only_cells = 0
    jaccards: list[float] = []
    coverage_deltas: list[float] = []
    rows_with_lg_only = 0
    for key in keys:
        lg_cells = union_cells(lightglue[key], config=config, correct_only=True)
        candidate_cells = union_cells(candidate[key], config=config, correct_only=True)
        if lg_cells - candidate_cells:
            rows_with_lg_only += 1
        lg_only_cells += len(lg_cells - candidate_cells)
        candidate_only_cells += len(candidate_cells - lg_cells)
        union = lg_cells | candidate_cells
        jaccards.append(len(lg_cells & candidate_cells) / len(union) if union else 1.0)
        coverage_deltas.append(
            float(coverage_metrics(candidate[key], config)["coverage_mean"])
            - float(coverage_metrics(lightglue[key], config)["coverage_mean"])
        )
    return {
        "common_pairs": len(keys),
        "lg_only_correct_cells": lg_only_cells,
        "candidate_only_correct_cells": candidate_only_cells,
        "pairs_with_lg_only_cells": rows_with_lg_only,
        "mean_cell_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "median_cell_jaccard": float(np.median(jaccards)) if jaccards else 0.0,
        "mean_coverage_delta_vs_lightglue": float(np.mean(coverage_deltas)) if coverage_deltas else 0.0,
        "median_coverage_delta_vs_lightglue": float(np.median(coverage_deltas)) if coverage_deltas else 0.0,
    }


def select_variant_gate(rows: Sequence[MatchRow], thresholds: dict[str, float]) -> list[MatchRow]:
    return [
        row
        for row in rows
        if row.reject_probability < thresholds.get(row.target_variant, 0.0)
    ]


def coverage_selector(
    rows: Sequence[MatchRow],
    *,
    thresholds: dict[str, float],
    config: CoverageConfig,
    cap_per_view_cell: int,
    rescue_slack: float,
    rescue_per_pair: int,
    rescue_cell_limit: int,
) -> list[MatchRow]:
    grouped = group_rows(rows)
    selected: list[MatchRow] = []
    for items in grouped.values():
        base = [row for row in items if row.reject_probability < thresholds.get(row.target_variant, 0.0)]
        base.sort(key=lambda row: (row.reject_probability, -row.score))
        kept: list[MatchRow] = []
        counts_a: dict[tuple[int, int], int] = {}
        counts_b: dict[tuple[int, int], int] = {}
        cap = int(cap_per_view_cell)
        for row in base:
            cell_a = cell_for_point(row.x_a, row.y_a, config)
            cell_b = cell_for_point(row.x_b, row.y_b, config)
            if cap > 0 and (counts_a.get(cell_a, 0) >= cap or counts_b.get(cell_b, 0) >= cap):
                continue
            kept.append(row)
            counts_a[cell_a] = counts_a.get(cell_a, 0) + 1
            counts_b[cell_b] = counts_b.get(cell_b, 0) + 1

        selected_ids = {id(row) for row in kept}
        occupied = union_cells(kept, config=config, correct_only=False)
        rescued = 0
        rescue_cell_counts: dict[tuple[str, int, int], int] = {}
        if rescue_per_pair > 0 and rescue_slack > 0.0:
            rescue_pool: list[MatchRow] = []
            for row in items:
                if id(row) in selected_ids:
                    continue
                threshold = thresholds.get(row.target_variant, 0.0)
                if row.reject_probability >= threshold + rescue_slack:
                    continue
                ax, ay = cell_for_point(row.x_a, row.y_a, config)
                bx, by = cell_for_point(row.x_b, row.y_b, config)
                row_cells = {("a", ax, ay), ("b", bx, by)}
                if row_cells <= occupied:
                    continue
                rescue_pool.append(row)
            rescue_pool.sort(key=lambda row: (row.reject_probability, -row.score))
            for row in rescue_pool:
                if rescued >= rescue_per_pair:
                    break
                ax, ay = cell_for_point(row.x_a, row.y_a, config)
                bx, by = cell_for_point(row.x_b, row.y_b, config)
                row_cells = [("a", ax, ay), ("b", bx, by)]
                if all(rescue_cell_counts.get(cell, 0) >= rescue_cell_limit for cell in row_cells):
                    continue
                kept.append(row)
                rescued += 1
                for cell in row_cells:
                    occupied.add(cell)
                    rescue_cell_counts[cell] = rescue_cell_counts.get(cell, 0) + 1
        selected.extend(kept)
    return selected


def coverage_score(summary: dict[str, float | int], compare: dict[str, float | int]) -> float:
    return (
        float(summary["pair_mean_coverage_mean"])
        + 0.35 * float(summary["pair_mean_entropy"])
        + 0.20 * float(summary["pair_mean_bbox_area_mean"])
        + 0.0015 * float(compare["candidate_only_correct_cells"])
        - 0.0015 * float(compare["lg_only_correct_cells"])
        - 0.20 * float(summary["pair_mean_largest_cell_ratio"])
    )


def _row_cells(row: MatchRow, config: CoverageConfig) -> set[tuple[str, int, int]]:
    ax, ay = cell_for_point(row.x_a, row.y_a, config)
    bx, by = cell_for_point(row.x_b, row.y_b, config)
    return {("a", ax, ay), ("b", bx, by)}


def _cell_pair_key(row: MatchRow, config: CoverageConfig) -> str:
    ax, ay = cell_for_point(row.x_a, row.y_a, config)
    bx, by = cell_for_point(row.x_b, row.y_b, config)
    return f"a:{ax}:{ay}|b:{bx}:{by}"


def _is_rescue_window(row: MatchRow, thresholds: dict[str, float], rescue_slack: float) -> bool:
    threshold = thresholds.get(row.target_variant, 0.0)
    return row.reject_probability >= threshold and row.reject_probability < threshold + rescue_slack


def wrong_by_variant_cell(
    rows: Sequence[MatchRow],
    *,
    thresholds: dict[str, float],
    config: CoverageConfig,
    rescue_slack: float,
) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, int, int], dict[str, object]] = {}
    score_sums: dict[tuple[str, str, int, int], float] = {}
    reject_sums: dict[tuple[str, str, int, int], float] = {}
    for row in rows:
        threshold = thresholds.get(row.target_variant, 0.0)
        rescue_window = row.reject_probability >= threshold and row.reject_probability < threshold + rescue_slack
        for view, cell_x, cell_y in _row_cells(row, config):
            key = (row.target_variant, view, cell_x, cell_y)
            bucket = buckets.setdefault(
                key,
                {
                    "target_variant": row.target_variant,
                    "view": view,
                    "cell_x": cell_x,
                    "cell_y": cell_y,
                    "total_rows": 0,
                    "correct_rows": 0,
                    "wrong_rows": 0,
                    "gate_kept_rows": 0,
                    "gate_kept_correct_rows": 0,
                    "gate_kept_wrong_rows": 0,
                    "rescue_window_rows": 0,
                    "rescue_window_correct_rows": 0,
                    "rescue_window_wrong_rows": 0,
                    "avg_score": 0.0,
                    "avg_reject_probability": 0.0,
                },
            )
            bucket["total_rows"] = int(bucket["total_rows"]) + 1
            if row.correct:
                bucket["correct_rows"] = int(bucket["correct_rows"]) + 1
            else:
                bucket["wrong_rows"] = int(bucket["wrong_rows"]) + 1
            if row.reject_probability < threshold:
                bucket["gate_kept_rows"] = int(bucket["gate_kept_rows"]) + 1
                if row.correct:
                    bucket["gate_kept_correct_rows"] = int(bucket["gate_kept_correct_rows"]) + 1
                else:
                    bucket["gate_kept_wrong_rows"] = int(bucket["gate_kept_wrong_rows"]) + 1
            if rescue_window:
                bucket["rescue_window_rows"] = int(bucket["rescue_window_rows"]) + 1
                if row.correct:
                    bucket["rescue_window_correct_rows"] = int(bucket["rescue_window_correct_rows"]) + 1
                else:
                    bucket["rescue_window_wrong_rows"] = int(bucket["rescue_window_wrong_rows"]) + 1
            score_sums[key] = score_sums.get(key, 0.0) + row.score
            reject_sums[key] = reject_sums.get(key, 0.0) + row.reject_probability

    output: list[dict[str, object]] = []
    for key, row in buckets.items():
        total = max(1, int(row["total_rows"]))
        item = dict(row)
        item["wrong_rate"] = int(row["wrong_rows"]) / total
        item["rescue_window_wrong_rate"] = (
            int(row["rescue_window_wrong_rows"]) / max(1, int(row["rescue_window_rows"]))
        )
        item["avg_score"] = score_sums[key] / total
        item["avg_reject_probability"] = reject_sums[key] / total
        output.append(item)
    output.sort(
        key=lambda row: (
            int(row["rescue_window_wrong_rows"]),
            int(row["wrong_rows"]),
            float(row["wrong_rate"]),
            str(row["target_variant"]),
            str(row["view"]),
            int(row["cell_y"]),
            int(row["cell_x"]),
        ),
        reverse=True,
    )
    return output


def rescue_tp_fp_analysis(
    rows: Sequence[MatchRow],
    lightglue_rows: Sequence[MatchRow],
    *,
    thresholds: dict[str, float],
    config: CoverageConfig,
    rescue_slack: float,
) -> list[dict[str, object]]:
    gate_cells_by_pair = {
        key: union_cells(select_variant_gate(items, thresholds), config=config, correct_only=True)
        for key, items in group_rows(rows).items()
    }
    lightglue_cells_by_pair = {
        key: union_cells(items, config=config, correct_only=True)
        for key, items in group_rows(lightglue_rows).items()
    }
    buckets: dict[tuple[str, str, int, int], dict[str, object]] = {}
    score_sums: dict[tuple[str, str, int, int], float] = {}
    reject_sums: dict[tuple[str, str, int, int], float] = {}
    pair_sets: dict[tuple[str, str, int, int], set[tuple[str, int]]] = {}
    for row in rows:
        if not _is_rescue_window(row, thresholds, rescue_slack):
            continue
        row_cells = _row_cells(row, config)
        gate_cells = gate_cells_by_pair.get(row.key, set())
        lg_only_cells = lightglue_cells_by_pair.get(row.key, set()) - gate_cells
        expands_gate_cells = not row_cells <= gate_cells
        overlaps_lg_only_cells = bool(row_cells & lg_only_cells)
        ax, ay = cell_for_point(row.x_a, row.y_a, config)
        key = (row.target_variant, _cell_pair_key(row, config), ax, ay)
        bucket = buckets.setdefault(
            key,
            {
                "target_variant": row.target_variant,
                "cell_pair_key": _cell_pair_key(row, config),
                "cell_a_x": ax,
                "cell_a_y": ay,
                "rows": 0,
                "correct_rows": 0,
                "wrong_rows": 0,
                "precision": 0.0,
                "pair_count": 0,
                "expands_gate_cells": 0,
                "overlaps_lightglue_only_cells": 0,
                "avg_score": 0.0,
                "avg_reject_probability": 0.0,
            },
        )
        bucket["rows"] = int(bucket["rows"]) + 1
        if row.correct:
            bucket["correct_rows"] = int(bucket["correct_rows"]) + 1
        else:
            bucket["wrong_rows"] = int(bucket["wrong_rows"]) + 1
        if expands_gate_cells:
            bucket["expands_gate_cells"] = 1
        if overlaps_lg_only_cells:
            bucket["overlaps_lightglue_only_cells"] = 1
        score_sums[key] = score_sums.get(key, 0.0) + row.score
        reject_sums[key] = reject_sums.get(key, 0.0) + row.reject_probability
        pair_sets.setdefault(key, set()).add(row.key)

    output: list[dict[str, object]] = []
    for key, row in buckets.items():
        total = max(1, int(row["rows"]))
        item = dict(row)
        item["precision"] = int(row["correct_rows"]) / total
        item["pair_count"] = len(pair_sets.get(key, set()))
        item["avg_score"] = score_sums[key] / total
        item["avg_reject_probability"] = reject_sums[key] / total
        output.append(item)
    output.sort(
        key=lambda row: (
            int(row["wrong_rows"]),
            int(row["correct_rows"]),
            int(row["overlaps_lightglue_only_cells"]),
            int(row["expands_gate_cells"]),
            str(row["target_variant"]),
        ),
        reverse=True,
    )
    return output


def summarize_source(
    name: str,
    rows: Sequence[MatchRow],
    *,
    lightglue_grouped: dict[tuple[str, int], list[MatchRow]],
    config: CoverageConfig,
) -> dict[str, float | int | str | bool]:
    grouped = group_rows(rows)
    summary = aggregate_pair_metrics(grouped, config)
    compare = compare_to_lightglue(lightglue_grouped, grouped, config)
    payload: dict[str, float | int | str | bool] = {"name": name}
    payload.update(summary)
    payload.update(compare)
    payload["coverage_score"] = coverage_score(summary, compare)
    payload["passes_stage"] = (
        int(payload["correct"]) >= config.stage_min_correct
        and int(payload["wrong"]) <= config.stage_max_wrong
        and float(payload["precision"]) >= config.stage_min_precision
        and float(payload["pair_mean_coverage_mean"]) >= config.stage_min_coverage
        and int(payload["lg_only_correct_cells"]) <= config.stage_max_lg_only_cells
    )
    return payload


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.6f}"
    return str(value)


def _plot_summary(rows: Sequence[dict[str, object]], figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    labels = [str(row["name"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar(labels, [float(row["pair_mean_coverage_mean"]) for row in rows], color="#2878b5")
    axes[0].set_title("Mean per-pair correct-cell coverage")
    axes[0].set_ylabel("coverage")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [float(row["pair_mean_entropy"]) for row in rows], color="#55a868")
    axes[1].set_title("Mean spatial entropy")
    axes[1].set_ylabel("normalized entropy")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    name = "coverage_entropy_comparison.png"
    fig.savefig(figures_dir / name, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    names.append(name)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, [float(row["lg_only_correct_cells"]) for row in rows], width, label="LG-only cells", color="#c44e52")
    ax.bar(x + width / 2, [float(row["candidate_only_correct_cells"]) for row in rows], width, label="candidate-only cells", color="#2878b5")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title("Correct-cell coverage gaps vs LightGlue")
    ax.set_ylabel("cell count over common pairs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    name = "lg_gap_cells_comparison.png"
    fig.savefig(figures_dir / name, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    names.append(name)
    return names


def _plot_selector_sweep(rows: Sequence[dict[str, object]], figures_dir: Path, config: CoverageConfig) -> str:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    eligible = [row for row in rows if row.get("passes_stage")]
    ineligible = [row for row in rows if not row.get("passes_stage")]
    if ineligible:
        ax.scatter(
            [float(row["wrong"]) for row in ineligible],
            [float(row["pair_mean_coverage_mean"]) for row in ineligible],
            s=18,
            c="#999999",
            alpha=0.45,
            label="ineligible",
        )
    if eligible:
        ax.scatter(
            [float(row["wrong"]) for row in eligible],
            [float(row["pair_mean_coverage_mean"]) for row in eligible],
            s=34,
            c="#2878b5",
            alpha=0.85,
            label="passes stage",
        )
    if config.stage_max_wrong < 10**12:
        ax.axvline(config.stage_max_wrong, color="#c44e52", linestyle="--", linewidth=1)
    ax.set_title("Coverage-aware selector sweep")
    ax.set_xlabel("wrong matches")
    ax.set_ylabel("mean per-pair correct-cell coverage")
    ax.grid(alpha=0.25)
    ax.legend()
    name = "selector_sweep_coverage_vs_wrong.png"
    fig.savefig(figures_dir / name, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return name


def _write_html(path: Path, payload: dict[str, object]) -> None:
    summary_rows = payload["summary_rows"]
    top_selector_rows = payload["top_selector_rows"]
    fields = [
        "name",
        "matches",
        "correct",
        "wrong",
        "precision",
        "pair_mean_coverage_mean",
        "pair_mean_entropy",
        "pair_mean_bbox_area_mean",
        "pair_mean_largest_cell_ratio",
        "lg_only_correct_cells",
        "candidate_only_correct_cells",
        "mean_cell_jaccard",
        "coverage_score",
        "passes_stage",
    ]
    selector_fields = [
        "name",
        "cap_per_view_cell",
        "rescue_slack",
        "rescue_per_pair",
        "rescue_cell_limit",
        "matches",
        "correct",
        "wrong",
        "precision",
        "pair_mean_coverage_mean",
        "lg_only_correct_cells",
        "coverage_score",
        "passes_stage",
    ]

    def table(rows: object, table_fields: Sequence[str]) -> str:
        assert isinstance(rows, list)
        body = []
        for row in rows:
            assert isinstance(row, dict)
            body.append(
                "<tr>"
                + "".join(f"<td>{html.escape(_format_cell(row.get(field, '')))}</td>" for field in table_fields)
                + "</tr>"
            )
        return "<table><tr>" + "".join(f"<th>{field}</th>" for field in table_fields) + "</tr>" + "".join(body) + "</table>"

    figures = payload.get("figures", [])
    assert isinstance(figures, list)
    figure_cards = [f'<img src="figures/{html.escape(str(figure))}" alt="{html.escape(str(figure))}">' for figure in figures]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>PFM Spatial Coverage Diagnostics</title>",
                (
                    "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.45}"
                    "table{border-collapse:collapse;margin:12px 0 24px;font-size:13px}"
                    "th,td{border:1px solid #bbb;padding:5px 7px;text-align:right}"
                    "th:first-child,td:first-child{text-align:left}"
                    "img{max-width:100%;display:block;margin:14px 0;border:1px solid #ddd}"
                    "pre{background:#f6f8fa;padding:12px;overflow:auto}</style>"
                ),
                "<h1>PFM Spatial Coverage Diagnostics</h1>",
                "<p>Coverage is computed on correct matches using an image-space grid over both views.</p>",
                "<h2>Source Comparison</h2>",
                table(summary_rows, fields),
                "<h2>Top Selector Candidates</h2>",
                table(top_selector_rows, selector_fields),
                "<h2>Figures</h2>",
                *figure_cards,
                "<h2>Conclusion JSON</h2>",
                f"<pre>{html.escape(json.dumps(payload['conclusion'], ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def analyze(
    *,
    pfm_rows: Sequence[MatchRow],
    lightglue_rows: Sequence[MatchRow],
    thresholds: dict[str, float],
    config: CoverageConfig,
) -> dict[str, object]:
    lightglue_grouped = group_rows(lightglue_rows)
    variant_gate = select_variant_gate(pfm_rows, thresholds)
    summary_rows: list[dict[str, object]] = [
        summarize_source("LightGlue", lightglue_rows, lightglue_grouped=lightglue_grouped, config=config),
        summarize_source("PFM unfiltered", pfm_rows, lightglue_grouped=lightglue_grouped, config=config),
        summarize_source("PFM variant gate", variant_gate, lightglue_grouped=lightglue_grouped, config=config),
    ]

    selector_rows: list[dict[str, object]] = []
    for cap, slack, rescue_count, cell_limit in itertools.product(
        [0, 16, 12, 8, 6],
        [0.0, 0.02, 0.05, 0.08, 0.12],
        [0, 4, 8, 16, 32],
        [1, 2],
    ):
        if slack == 0.0 and rescue_count > 0:
            continue
        selected = coverage_selector(
            pfm_rows,
            thresholds=thresholds,
            config=config,
            cap_per_view_cell=cap,
            rescue_slack=slack,
            rescue_per_pair=rescue_count,
            rescue_cell_limit=cell_limit,
        )
        name = f"selector_cap{cap}_slack{slack:.2f}_rescue{rescue_count}_cell{cell_limit}"
        row = summarize_source(name, selected, lightglue_grouped=lightglue_grouped, config=config)
        row.update(
            {
                "cap_per_view_cell": cap,
                "rescue_slack": slack,
                "rescue_per_pair": rescue_count,
                "rescue_cell_limit": cell_limit,
            }
        )
        selector_rows.append(row)

    eligible_selectors = [row for row in selector_rows if row.get("passes_stage")]
    ranked_selectors = sorted(
        eligible_selectors or selector_rows,
        key=lambda row: (
            bool(row.get("passes_stage")),
            float(row["coverage_score"]),
            float(row["pair_mean_coverage_mean"]),
            int(row["correct"]),
            -int(row["wrong"]),
        ),
        reverse=True,
    )
    best_selector = ranked_selectors[0] if ranked_selectors else summary_rows[2]
    summary_rows.append({**best_selector, "name": "Best coverage selector"})
    strict_beats_variant = (
        bool(best_selector.get("passes_stage"))
        and float(best_selector["pair_mean_coverage_mean"]) > float(summary_rows[2]["pair_mean_coverage_mean"])
        and int(best_selector["correct"]) >= int(summary_rows[2]["correct"])
        and int(best_selector["wrong"]) <= int(summary_rows[2]["wrong"])
    )
    conclusion = {
        "best_selector": {
            key: best_selector.get(key)
            for key in [
                "name",
                "cap_per_view_cell",
                "rescue_slack",
                "rescue_per_pair",
                "rescue_cell_limit",
                "matches",
                "correct",
                "wrong",
                "precision",
                "pair_mean_coverage_mean",
                "lg_only_correct_cells",
                "candidate_only_correct_cells",
                "passes_stage",
            ]
        },
        "strict_beats_variant": strict_beats_variant,
        "availability": {
            "lightglue_only_cells_after_unfiltered": summary_rows[1]["lg_only_correct_cells"],
            "lightglue_only_cells_after_variant_gate": summary_rows[2]["lg_only_correct_cells"],
            "lightglue_only_cells_after_best_selector": best_selector["lg_only_correct_cells"],
            "unfiltered_candidate_only_cells": summary_rows[1]["candidate_only_correct_cells"],
            "variant_candidate_only_cells": summary_rows[2]["candidate_only_correct_cells"],
            "best_selector_candidate_only_cells": best_selector["candidate_only_correct_cells"],
        },
    }
    return {
        "grid_size": config.grid_size,
        "image_size": config.image_size,
        "stage_targets": {
            "min_correct": config.stage_min_correct,
            "max_wrong": config.stage_max_wrong,
            "min_precision": config.stage_min_precision,
            "min_coverage": config.stage_min_coverage,
            "max_lg_only_cells": config.stage_max_lg_only_cells,
        },
        "summary_rows": summary_rows,
        "selector_rows": selector_rows,
        "top_selector_rows": ranked_selectors[:20],
        "conclusion": conclusion,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfm-source", action="append", required=True, help="name,path")
    parser.add_argument("--lightglue-source", action="append", required=True, help="name,path")
    parser.add_argument("--variant-gate-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=float, default=768.0)
    parser.add_argument("--stage-min-correct", type=int, default=0)
    parser.add_argument("--stage-max-wrong", type=int, default=10**12)
    parser.add_argument("--stage-min-precision", type=float, default=0.0)
    parser.add_argument("--stage-min-coverage", type=float, default=0.0)
    parser.add_argument("--stage-max-lg-only-cells", type=int, default=10**12)
    parser.add_argument("--diagnostic-rescue-slack", type=float, default=0.12)
    args = parser.parse_args(argv)
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    if args.image_size <= 0.0 or not math.isfinite(args.image_size):
        raise ValueError("--image-size must be finite and positive")
    if args.diagnostic_rescue_slack < 0.0 or not math.isfinite(args.diagnostic_rescue_slack):
        raise ValueError("--diagnostic-rescue-slack must be finite and nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = CoverageConfig(
        grid_size=args.grid_size,
        image_size=args.image_size,
        stage_min_correct=args.stage_min_correct,
        stage_max_wrong=args.stage_max_wrong,
        stage_min_precision=args.stage_min_precision,
        stage_min_coverage=args.stage_min_coverage,
        stage_max_lg_only_cells=args.stage_max_lg_only_cells,
    )
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pfm_rows = load_match_rows([_parse_source(value) for value in args.pfm_source])
    lightglue_rows = load_match_rows([_parse_source(value) for value in args.lightglue_source])
    thresholds = load_variant_thresholds(args.variant_gate_json)
    payload = analyze(pfm_rows=pfm_rows, lightglue_rows=lightglue_rows, thresholds=thresholds, config=config)
    wrong_cell_rows = wrong_by_variant_cell(
        pfm_rows,
        thresholds=thresholds,
        config=config,
        rescue_slack=args.diagnostic_rescue_slack,
    )
    rescue_rows = rescue_tp_fp_analysis(
        pfm_rows,
        lightglue_rows,
        thresholds=thresholds,
        config=config,
        rescue_slack=args.diagnostic_rescue_slack,
    )
    payload["diagnostics"] = {
        "diagnostic_rescue_slack": args.diagnostic_rescue_slack,
        "wrong_by_variant_cell_rows": len(wrong_cell_rows),
        "rescue_tp_fp_rows": len(rescue_rows),
        "top_wrong_by_variant_cell": wrong_cell_rows[:20],
        "top_rescue_tp_fp": rescue_rows[:20],
    }
    summary_rows = payload["summary_rows"]
    selector_rows = payload["selector_rows"]
    top_selector_rows = payload["top_selector_rows"]
    assert isinstance(summary_rows, list)
    assert isinstance(selector_rows, list)
    assert isinstance(top_selector_rows, list)
    figures = _plot_summary(summary_rows[:3] + [summary_rows[-1]], figures_dir)
    figures.append(_plot_selector_sweep(selector_rows, figures_dir, config))
    payload["figures"] = figures
    (output_dir / "coverage_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "coverage_summary.csv", summary_rows)
    _write_csv(output_dir / "selector_sweep.csv", selector_rows)
    _write_csv(output_dir / "top_selector_candidates.csv", top_selector_rows)
    _write_csv(output_dir / "wrong_by_variant_cell.csv", wrong_cell_rows)
    _write_csv(output_dir / "rescue_tp_fp_analysis.csv", rescue_rows)
    _write_html(output_dir / "index.html", payload)
    print(
        json.dumps(
            {
                "output": str(output_dir / "index.html"),
                "best_selector": payload["conclusion"]["best_selector"],
                "strict_beats_variant": payload["conclusion"]["strict_beats_variant"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
