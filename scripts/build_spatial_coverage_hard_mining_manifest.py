#!/usr/bin/env python3
"""Build train-only replay manifests from spatial coverage hard-match patterns."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from analyze_match_spatial_coverage import (
    CoverageConfig,
    MatchRow,
    SourceSpec,
    cell_for_point,
    group_rows,
    load_match_rows,
    load_variant_thresholds,
    select_variant_gate,
    union_cells,
)


HARD_BASE_TRUE = "base_true_positive"
HARD_CANDIDATE_RESCUE_TRUE = "candidate_rescue_true_positive"
HARD_GATE_DROPPED_TRUE = "gate_dropped_true_positive"
HARD_LIGHTGLUE_ONLY_CELL = "lightglue_only_cell_gap"
HARD_COVERAGE_RESCUE_FALSE = "coverage_rescue_false_positive"
DEFAULT_TRUE_GEOMETRY_REPLAY_LABELS = (
    HARD_CANDIDATE_RESCUE_TRUE,
    HARD_GATE_DROPPED_TRUE,
    HARD_LIGHTGLUE_ONLY_CELL,
)


@dataclass(frozen=True)
class TrueGeometryReplayConfig:
    labels: tuple[str, ...]
    target_scale: float
    target_min: int
    target_max: int
    supervision_weight: float
    pair_accept_weight: float


@dataclass(frozen=True)
class TrueGeometryReplayProfile:
    target_count: int
    source_count: int
    source_pair_count: int
    label_counts: dict[str, int]
    supervision_weight: float
    pair_accept_weight: float

    def to_json_row(self) -> dict[str, object]:
        return {
            "target_count": self.target_count,
            "source_count": self.source_count,
            "source_pair_count": self.source_pair_count,
            "label_counts": dict(sorted(self.label_counts.items())),
            "supervision_weight": self.supervision_weight,
            "pair_accept_weight": self.pair_accept_weight,
        }


def _parse_source(value: str) -> SourceSpec:
    parts = [part.strip() for part in value.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("source must be formatted as name,path")
    return SourceSpec(parts[0], Path(parts[1]))


def _valid_bucket(value: float) -> str:
    if value < 0.20:
        return "low"
    if value < 0.50:
        return "mid"
    return "high"


def _row_cells(row: MatchRow, config: CoverageConfig) -> set[tuple[str, int, int]]:
    ax, ay = cell_for_point(row.x_a, row.y_a, config)
    bx, by = cell_for_point(row.x_b, row.y_b, config)
    return {("a", ax, ay), ("b", bx, by)}


def _cell_pair_key(row: MatchRow, config: CoverageConfig) -> str:
    ax, ay = cell_for_point(row.x_a, row.y_a, config)
    bx, by = cell_for_point(row.x_b, row.y_b, config)
    return f"a:{ax}:{ay}|b:{bx}:{by}"


def _hard_row(
    row: MatchRow,
    label: str,
    config: CoverageConfig,
    *,
    expands_gate_cells: bool = False,
    overlaps_lightglue_only_cells: bool = False,
) -> dict[str, str]:
    ax_cell = cell_for_point(row.x_a, row.y_a, config)
    bx_cell = cell_for_point(row.x_b, row.y_b, config)
    return {
        "hard_label": label,
        "split": row.split,
        "pair_index": str(row.pair_index),
        "base_id": row.base_id,
        "reference_variant": row.reference_variant,
        "target_variant": row.target_variant,
        "ax": f"{row.x_a:.6f}",
        "ay": f"{row.y_a:.6f}",
        "bx": f"{row.x_b:.6f}",
        "by": f"{row.y_b:.6f}",
        "score": f"{row.score:.9f}",
        "correct": "1" if row.correct else "0",
        "reject_probability": f"{row.reject_probability:.9f}",
        "valid_fraction": f"{row.valid_fraction:.9f}",
        "valid_bucket": _valid_bucket(row.valid_fraction),
        "cell_a_x": str(ax_cell[0]),
        "cell_a_y": str(ax_cell[1]),
        "cell_b_x": str(bx_cell[0]),
        "cell_b_y": str(bx_cell[1]),
        "cell_pair_key": _cell_pair_key(row, config),
        "expands_gate_cells": "1" if expands_gate_cells else "0",
        "overlaps_lightglue_only_cells": "1" if overlaps_lightglue_only_cells else "0",
    }


def mine_hard_rows(
    *,
    pfm_rows: Sequence[MatchRow],
    lightglue_rows: Sequence[MatchRow],
    thresholds: dict[str, float],
    config: CoverageConfig,
    rescue_slack: float,
) -> list[dict[str, str]]:
    hard_rows: list[dict[str, str]] = []
    pfm_gate_rows = select_variant_gate(pfm_rows, thresholds)
    pfm_gate_by_pair = group_rows(pfm_gate_rows)
    pfm_gate_cells_by_pair = {
        key: union_cells(rows, config=config, correct_only=True)
        for key, rows in pfm_gate_by_pair.items()
    }
    lightglue_cells_by_pair = {
        key: union_cells(rows, config=config, correct_only=True)
        for key, rows in group_rows(lightglue_rows).items()
    }

    for row in pfm_rows:
        threshold = thresholds.get(row.target_variant, 0.0)
        row_cells = _row_cells(row, config)
        gate_cells = pfm_gate_cells_by_pair.get(row.key, set())
        lightglue_only_cells = lightglue_cells_by_pair.get(row.key, set()) - gate_cells
        expands_gate_cells = not row_cells <= gate_cells
        overlaps_lightglue_only_cells = bool(row_cells & lightglue_only_cells)
        in_rescue_window = row.reject_probability >= threshold and row.reject_probability < threshold + rescue_slack
        if row.correct and row.reject_probability < threshold:
            hard_rows.append(
                _hard_row(
                    row,
                    HARD_BASE_TRUE,
                    config,
                    expands_gate_cells=False,
                    overlaps_lightglue_only_cells=overlaps_lightglue_only_cells,
                )
            )
        if row.correct and row.reject_probability >= threshold:
            hard_rows.append(
                _hard_row(
                    row,
                    HARD_GATE_DROPPED_TRUE,
                    config,
                    expands_gate_cells=expands_gate_cells,
                    overlaps_lightglue_only_cells=overlaps_lightglue_only_cells,
                )
            )
        if row.correct and in_rescue_window and (expands_gate_cells or overlaps_lightglue_only_cells):
            hard_rows.append(
                _hard_row(
                    row,
                    HARD_CANDIDATE_RESCUE_TRUE,
                    config,
                    expands_gate_cells=expands_gate_cells,
                    overlaps_lightglue_only_cells=overlaps_lightglue_only_cells,
                )
            )
        if not row.correct and in_rescue_window:
            hard_rows.append(
                _hard_row(
                    row,
                    HARD_COVERAGE_RESCUE_FALSE,
                    config,
                    expands_gate_cells=expands_gate_cells,
                    overlaps_lightglue_only_cells=overlaps_lightglue_only_cells,
                )
            )

    for row in lightglue_rows:
        if not row.correct:
            continue
        candidate_cells = pfm_gate_cells_by_pair.get(row.key, set())
        ax, ay = cell_for_point(row.x_a, row.y_a, config)
        bx, by = cell_for_point(row.x_b, row.y_b, config)
        lightglue_cells = {("a", ax, ay), ("b", bx, by)}
        if not lightglue_cells <= candidate_cells:
            hard_rows.append(_hard_row(row, HARD_LIGHTGLUE_ONLY_CELL, config))

    hard_rows.sort(
        key=lambda row: (
            row["target_variant"],
            row["hard_label"],
            row["split"],
            int(row["pair_index"]),
            int(row["cell_a_y"]),
            int(row["cell_a_x"]),
        )
    )
    return hard_rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
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


def _row_base_tokens(row: dict[str, str]) -> set[str]:
    tokens = set()
    for key in ("base_id", "reference_base_id", "target_base_id", "reference_raw_base_id", "target_raw_base_id"):
        value = (row.get(key) or "").strip()
        if value:
            tokens.add(value)
    return tokens


def _hard_source_pair_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("split", ""),
        row.get("pair_index", ""),
        row.get("base_id", ""),
        row.get("reference_variant", ""),
        row.get("target_variant", ""),
    )


def build_true_geometry_replay_profiles(
    hard_rows: Sequence[dict[str, str]],
    config: TrueGeometryReplayConfig,
) -> dict[str, TrueGeometryReplayProfile]:
    if config.target_scale <= 0.0 or not config.labels:
        return {}
    label_set = set(config.labels)
    label_counts_by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    source_pairs_by_variant: dict[str, set[tuple[str, str, str, str, str]]] = defaultdict(set)
    for row in hard_rows:
        label = row.get("hard_label", "")
        if label not in label_set:
            continue
        variant = row.get("target_variant", "")
        if not variant:
            continue
        label_counts_by_variant[variant][label] += 1
        source_pairs_by_variant[variant].add(_hard_source_pair_key(row))

    profiles: dict[str, TrueGeometryReplayProfile] = {}
    for variant in sorted(label_counts_by_variant):
        source_count = sum(label_counts_by_variant[variant].values())
        source_pair_count = max(1, len(source_pairs_by_variant[variant]))
        raw_target = (float(source_count) / float(source_pair_count)) * float(config.target_scale)
        target_count = int(math.ceil(raw_target))
        if target_count <= 0:
            continue
        target_count = max(int(config.target_min), target_count)
        target_count = min(int(config.target_max), target_count)
        if target_count <= 0:
            continue
        profiles[variant] = TrueGeometryReplayProfile(
            target_count=target_count,
            source_count=source_count,
            source_pair_count=source_pair_count,
            label_counts=dict(label_counts_by_variant[variant]),
            supervision_weight=float(config.supervision_weight),
            pair_accept_weight=float(config.pair_accept_weight),
        )
    return profiles


def _eligible_train_rows(
    train_rows: Sequence[dict[str, str]],
    hard_rows: Sequence[dict[str, str]],
    *,
    forbidden_base_ids: set[str],
) -> dict[str, list[dict[str, str]]]:
    variants = {row["target_variant"] for row in hard_rows}
    eligible: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        if (row.get("split") or "").strip() != "train":
            continue
        variant = (row.get("target_variant") or "").strip()
        if variant not in variants:
            continue
        if _row_base_tokens(row) & forbidden_base_ids:
            continue
        eligible[variant].append(row)
    return eligible


def build_replay_rows(
    *,
    train_rows: Sequence[dict[str, str]],
    hard_rows: Sequence[dict[str, str]],
    true_geometry_profiles: dict[str, TrueGeometryReplayProfile] | None = None,
    repeat: int,
    max_per_variant: int,
    seed: int,
) -> list[dict[str, str]]:
    forbidden = {row["base_id"] for row in hard_rows if row.get("base_id")}
    eligible_by_variant = _eligible_train_rows(train_rows, hard_rows, forbidden_base_ids=forbidden)
    labels_by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    buckets_by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    for row in hard_rows:
        labels_by_variant[row["target_variant"]][row["hard_label"]] += 1
        buckets_by_variant[row["target_variant"]][row["valid_bucket"]] += 1

    rng = random.Random(seed)
    replay_rows: list[dict[str, str]] = []
    for variant in sorted(labels_by_variant):
        candidates = list(eligible_by_variant.get(variant, []))
        if max_per_variant > 0 and len(candidates) > max_per_variant:
            candidates = rng.sample(candidates, max_per_variant)
            candidates.sort(key=lambda row: (row.get("target_base_id", ""), row.get("reference_base_id", "")))
        true_geometry = (true_geometry_profiles or {}).get(variant)
        for replay_round in range(max(1, repeat)):
            for row in candidates:
                replay = dict(row)
                replay["spatial_coverage_hard_reason"] = ",".join(sorted(labels_by_variant[variant]))
                replay["spatial_coverage_source_labels_json"] = json.dumps(labels_by_variant[variant], sort_keys=True)
                replay["spatial_coverage_valid_buckets_json"] = json.dumps(buckets_by_variant[variant], sort_keys=True)
                replay["spatial_coverage_replay_round"] = str(replay_round)
                replay["spatial_coverage_source_count"] = str(sum(labels_by_variant[variant].values()))
                if true_geometry is not None:
                    replay["pair_accept_label"] = "1"
                    replay["pair_accept_weight"] = f"{true_geometry.pair_accept_weight:.6f}"
                    replay["true_geometry_positive_matches"] = str(true_geometry.target_count)
                    replay["true_geometry_supervision_weight"] = f"{true_geometry.supervision_weight:.6f}"
                    replay["true_geometry_source_count"] = str(true_geometry.source_count)
                    replay["true_geometry_source_pair_count"] = str(true_geometry.source_pair_count)
                    replay["true_geometry_source_labels_json"] = json.dumps(
                        true_geometry.label_counts,
                        sort_keys=True,
                    )
                replay_rows.append(replay)
    return replay_rows


def false_match_rows(hard_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in hard_rows:
        if row["hard_label"] != HARD_COVERAGE_RESCUE_FALSE:
            continue
        base_id = row["base_id"]
        rows.append(
            {
                "pair_type": "same-position",
                "reference_base_id": base_id,
                "reference_variant": row["reference_variant"],
                "target_base_id": base_id,
                "target_variant": row["target_variant"],
                "ax": row["ax"],
                "ay": row["ay"],
                "bx": row["bx"],
                "by": row["by"],
                "hard_label": row["hard_label"],
                "split": row["split"],
                "pair_index": row["pair_index"],
                "base_id": base_id,
                "reject_probability": row["reject_probability"],
            }
        )
    return rows


def _write_html(path: Path, summary: dict[str, object], hard_rows: Sequence[dict[str, str]]) -> None:
    counts = summary.get("hard_label_counts", {})
    assert isinstance(counts, dict)
    top_rows = list(hard_rows[:200])
    fieldnames = list(top_rows[0].keys()) if top_rows else []
    table = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(row.get(field, ''))}</td>" for field in fieldnames) + "</tr>"
        for row in top_rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Spatial Coverage Hard Mining</title>",
                "<h1>Spatial Coverage Hard Mining</h1>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
                "<h2>Top Hard Rows</h2>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr>" + "".join(f"<th>{html.escape(field)}</th>" for field in fieldnames) + "</tr>",
                table,
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfm-source", action="append", required=True, help="name,path")
    parser.add_argument("--lightglue-source", action="append", required=True, help="name,path")
    parser.add_argument("--variant-gate-json", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-base-manifest", type=Path)
    parser.add_argument("--mixed-output-manifest", type=Path)
    parser.add_argument("--mixed-replay-fraction", type=float, default=1.0)
    parser.add_argument("--mining-csv", type=Path, required=True)
    parser.add_argument("--false-match-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report-html", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=float, default=768.0)
    parser.add_argument("--rescue-slack", type=float, default=0.12)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-per-variant", type=int, default=5000)
    parser.add_argument("--true-geometry-label", action="append", default=[])
    parser.add_argument("--true-geometry-target-scale", type=float, default=0.35)
    parser.add_argument("--true-geometry-target-min", type=int, default=16)
    parser.add_argument("--true-geometry-target-max", type=int, default=96)
    parser.add_argument("--true-geometry-supervision-weight", type=float, default=1.0)
    parser.add_argument("--pair-accept-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    if args.image_size <= 0.0 or not math.isfinite(args.image_size):
        raise ValueError("--image-size must be finite and positive")
    if args.rescue_slack < 0.0:
        raise ValueError("--rescue-slack must be nonnegative")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.max_per_variant < 0:
        raise ValueError("--max-per-variant must be nonnegative")
    if not math.isfinite(float(args.true_geometry_target_scale)) or args.true_geometry_target_scale < 0.0:
        raise ValueError("--true-geometry-target-scale must be finite and nonnegative")
    if args.true_geometry_target_min < 0:
        raise ValueError("--true-geometry-target-min must be nonnegative")
    if args.true_geometry_target_max < 0:
        raise ValueError("--true-geometry-target-max must be nonnegative")
    if args.true_geometry_target_max < args.true_geometry_target_min:
        raise ValueError("--true-geometry-target-max must be >= --true-geometry-target-min")
    if (
        not math.isfinite(float(args.true_geometry_supervision_weight))
        or args.true_geometry_supervision_weight <= 0.0
    ):
        raise ValueError("--true-geometry-supervision-weight must be finite and positive")
    if not math.isfinite(float(args.pair_accept_weight)) or args.pair_accept_weight <= 0.0:
        raise ValueError("--pair-accept-weight must be finite and positive")
    if (args.mixed_base_manifest is None) != (args.mixed_output_manifest is None):
        raise ValueError("--mixed-base-manifest and --mixed-output-manifest must be provided together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = CoverageConfig(grid_size=args.grid_size, image_size=args.image_size)
    pfm_rows = load_match_rows([_parse_source(value) for value in args.pfm_source])
    lightglue_rows = load_match_rows([_parse_source(value) for value in args.lightglue_source])
    thresholds = load_variant_thresholds(args.variant_gate_json)
    hard_rows = mine_hard_rows(
        pfm_rows=pfm_rows,
        lightglue_rows=lightglue_rows,
        thresholds=thresholds,
        config=config,
        rescue_slack=args.rescue_slack,
    )
    train_rows = _read_csv(args.train_manifest)
    true_geometry_labels = tuple(args.true_geometry_label or DEFAULT_TRUE_GEOMETRY_REPLAY_LABELS)
    true_geometry_config = TrueGeometryReplayConfig(
        labels=true_geometry_labels,
        target_scale=float(args.true_geometry_target_scale),
        target_min=int(args.true_geometry_target_min),
        target_max=int(args.true_geometry_target_max),
        supervision_weight=float(args.true_geometry_supervision_weight),
        pair_accept_weight=float(args.pair_accept_weight),
    )
    true_geometry_profiles = build_true_geometry_replay_profiles(hard_rows, true_geometry_config)
    replay_rows = build_replay_rows(
        train_rows=train_rows,
        hard_rows=hard_rows,
        true_geometry_profiles=true_geometry_profiles,
        repeat=args.repeat,
        max_per_variant=args.max_per_variant,
        seed=args.seed,
    )
    false_rows = false_match_rows(hard_rows)
    _write_csv(args.mining_csv, hard_rows)
    _write_csv(args.false_match_csv, false_rows)
    _write_csv(args.output_manifest, replay_rows)

    mixed_rows: list[dict[str, str]] = []
    if args.mixed_base_manifest is not None and args.mixed_output_manifest is not None:
        base_rows = _read_csv(args.mixed_base_manifest)
        replay_for_mix = replay_rows
        if 0.0 <= args.mixed_replay_fraction < 1.0:
            keep = int(round(len(replay_rows) * args.mixed_replay_fraction))
            replay_for_mix = replay_rows[:keep]
        mixed_rows = [*base_rows, *replay_for_mix]
        _write_csv(args.mixed_output_manifest, mixed_rows)

    hard_counts = Counter(row["hard_label"] for row in hard_rows)
    variant_counts = Counter(row["target_variant"] for row in hard_rows)
    summary: dict[str, object] = {
        "pfm_rows": len(pfm_rows),
        "lightglue_rows": len(lightglue_rows),
        "hard_rows": len(hard_rows),
        "hard_label_counts": dict(sorted(hard_counts.items())),
        "target_variant_counts": dict(sorted(variant_counts.items())),
        "train_manifest_rows": len(train_rows),
        "replay_rows": len(replay_rows),
        "mixed_rows": len(mixed_rows),
        "false_match_rows": len(false_rows),
        "grid_size": args.grid_size,
        "image_size": args.image_size,
        "rescue_slack": args.rescue_slack,
        "repeat": args.repeat,
        "max_per_variant": args.max_per_variant,
        "true_geometry_labels": list(true_geometry_labels),
        "true_geometry_target_scale": args.true_geometry_target_scale,
        "true_geometry_target_min": args.true_geometry_target_min,
        "true_geometry_target_max": args.true_geometry_target_max,
        "true_geometry_replay": {
            variant: profile.to_json_row()
            for variant, profile in sorted(true_geometry_profiles.items())
        },
        "seed": args.seed,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_html(args.report_html, summary, hard_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
