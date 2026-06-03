#!/usr/bin/env python3
"""Agent13 stage3 expanded teacher coverage and timestamp/compound diagnosis."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import pseudo_label_generation as plg


AGENT13_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent13.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage3"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "runs" / "cross_view_1024_hard_mined_weakgates_80_seed1234" / "splits" / "train"
STAGE2_SUMMARY = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage2" / "summary_metrics.csv"


PAIR_FIELDS = [
    "style",
    "gate",
    "sample_index",
    "source_name",
    "pair_name",
    "pair_pt",
    "profile",
    "algorithm",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "homography_inliers",
    "truth_labels",
    "wrong_inliers",
    "truth_precision",
    "kept_pair",
    "kept_labels",
    "homography_pass",
    "mean_label_error_px",
    "median_label_error_px",
    "grid_cells_4x4",
    "grid_coverage_4x4",
    "edge_label_fraction",
    "center_label_fraction",
    "x_mean",
    "y_mean",
    "x_std",
    "y_std",
    "runtime_ms",
    "failure_reason",
    "message",
]

SUMMARY_FIELDS = [
    "style",
    "gate",
    "profile",
    "algorithm",
    "sampled_pairs",
    "ok_pairs",
    "homography_pass_pairs",
    "homography_pass_rate",
    "kept_pairs",
    "kept_pair_rate",
    "kept_sources",
    "top_source_fraction",
    "source_entropy",
    "labels",
    "homography_inliers",
    "truth_labels",
    "wrong_inliers",
    "truth_precision",
    "mean_inliers_per_pair",
    "median_inliers_per_pair",
    "p10_inliers_per_pair",
    "p90_inliers_per_pair",
    "mean_labels_per_pair",
    "median_labels_per_pair",
    "p10_labels_per_pair",
    "p90_labels_per_pair",
    "mean_grid_coverage_4x4",
    "median_grid_coverage_4x4",
    "low_spatial_coverage_pairs",
    "mean_edge_label_fraction",
    "mean_center_label_fraction",
    "failure_counts",
]

OVERLAP_FIELDS = [
    "style",
    "gate",
    "profile_a",
    "profile_b",
    "sampled_pairs",
    "kept_a",
    "kept_b",
    "both_kept",
    "a_only",
    "b_only",
    "neither",
    "kept_jaccard",
    "sources_a",
    "sources_b",
    "sources_both",
    "source_jaccard",
    "median_inliers_a_only",
    "median_inliers_b_only",
    "median_inliers_both_a",
    "median_inliers_both_b",
    "median_labels_a_only",
    "median_labels_b_only",
    "median_labels_both_a",
    "median_labels_both_b",
]

SOURCE_FIELDS = ["style", "gate", "profile", "source_name", "sampled_pairs", "kept_pairs", "kept_labels", "mean_grid_coverage_4x4"]
SAMPLED_FIELDS = ["style", "gate", "sample_index", "source_name", "pair_name", "pair_pt"]
SKIPPED_FIELDS = ["profile", "algorithm", "reason"]


@dataclass(frozen=True)
class TeacherConfig:
    profile: str
    algorithm: str
    detector: str
    mode: str
    ratio: float
    ransac_threshold_px: float
    min_inliers: int
    style: str
    gate: str


@dataclass(frozen=True)
class PairDiag:
    style: str
    gate: str
    sample_index: int
    source_name: str
    pair_name: str
    pair_pt: str
    profile: str
    algorithm: str
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    homography_inliers: int
    truth_labels: int
    wrong_inliers: int
    truth_precision: float
    kept_pair: int
    kept_labels: int
    homography_pass: int
    mean_label_error_px: float
    median_label_error_px: float
    grid_cells_4x4: int
    grid_coverage_4x4: float
    edge_label_fraction: float
    center_label_fraction: float
    x_mean: float
    y_mean: float
    x_std: float
    y_std: float
    runtime_ms: float
    failure_reason: str
    message: str = ""


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A13 = load_module(AGENT13_SCRIPT, "agent13_matcher_for_stage3")
A4 = A13.A4


def format_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fields})


def min_labels_for_gate(gate: str) -> int:
    return 8 if gate == "compound" else 20


def source_name(pair_path: Path) -> str:
    return pair_path.parent.name


def discover_pairs(split_root: Path, style: str, gate: str, *, limit: int, seed: int) -> list[Path]:
    root = split_root / style / gate
    paths: list[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if filename.startswith("pair_") and filename.endswith(".pt"):
                paths.append(Path(dirpath) / filename)
    paths = sorted(paths)
    if limit > 0 and len(paths) > limit:
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, k=limit))
    return paths


def teacher_configs() -> list[TeacherConfig]:
    return [
        TeacherConfig("style_numeric_viewpoint", "RootSIFT-mutual-r0p95-Ht3-min4", "RootSIFT", "ratio_mutual", 0.95, 3.0, 4, "numeric", "viewpoint"),
        TeacherConfig("style_numeric_compound", "LightGlue-SIFT-Ht3-min4", "LightGlue-SIFT", "lightglue", math.nan, 3.0, 4, "numeric", "compound"),
        TeacherConfig("style_timestamp_viewpoint", "RootSIFT-ratio-r0p92-Ht3-min4", "RootSIFT", "ratio", 0.92, 3.0, 4, "timestamp", "viewpoint"),
        TeacherConfig("baseline_r080_ht2", "RootSIFT-ratio-r0p80-Ht2-min4", "RootSIFT", "ratio", 0.80, 2.0, 4, "timestamp", "compound"),
        TeacherConfig("style_timestamp_compound", "RootSIFT-ratio-r0p88-Ht3-min4", "RootSIFT", "ratio", 0.88, 3.0, 4, "timestamp", "compound"),
        TeacherConfig("baseline_r090_ht3", "RootSIFT-ratio-r0p90-Ht3-min4", "RootSIFT", "ratio", 0.90, 3.0, 4, "timestamp", "compound"),
    ]


class MatcherFactory:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cache: dict[str, object] = {}
        self.skipped: list[dict[str, str]] = []

    def matcher(self, config: TeacherConfig) -> object | None:
        if config.profile in self.cache:
            return self.cache[config.profile]
        try:
            if config.detector == "LightGlue-SIFT":
                if importlib.util.find_spec("lightglue") is None:
                    self.skipped.append({"profile": config.profile, "algorithm": config.algorithm, "reason": "module 'lightglue' unavailable"})
                    return None
                matcher = A13.LightGlueMatcher(max_keypoints=self.args.learned_max_keypoints, device=self.args.device)
            else:
                matcher = A13.CvMatcher(
                    config.detector,
                    ratio=config.ratio,
                    mode=config.mode,
                    max_keypoints=self.args.max_keypoints,
                    max_matches=self.args.max_raw_matches,
                    sift_contrast=self.args.sift_contrast,
                )
            self.cache[config.profile] = matcher
            return matcher
        except Exception as exc:
            self.skipped.append({"profile": config.profile, "algorithm": config.algorithm, "reason": f"{type(exc).__name__}: {exc}"})
            return None


def spatial_stats(points_a: np.ndarray, height: int, width: int) -> dict[str, float | int]:
    if points_a.size == 0:
        return {
            "grid_cells_4x4": 0,
            "grid_coverage_4x4": 0.0,
            "edge_label_fraction": 0.0,
            "center_label_fraction": 0.0,
            "x_mean": math.nan,
            "y_mean": math.nan,
            "x_std": math.nan,
            "y_std": math.nan,
        }
    x = np.clip(points_a[:, 0], 0, max(1, width - 1))
    y = np.clip(points_a[:, 1], 0, max(1, height - 1))
    gx = np.minimum((x / max(1, width) * 4).astype(int), 3)
    gy = np.minimum((y / max(1, height) * 4).astype(int), 3)
    cells = set(zip(gx.tolist(), gy.tolist()))
    edge = (x < width * 0.10) | (x > width * 0.90) | (y < height * 0.10) | (y > height * 0.90)
    center = (x > width * 0.25) & (x < width * 0.75) & (y > height * 0.25) & (y < height * 0.75)
    return {
        "grid_cells_4x4": len(cells),
        "grid_coverage_4x4": len(cells) / 16.0,
        "edge_label_fraction": float(edge.mean()),
        "center_label_fraction": float(center.mean()),
        "x_mean": float(x.mean()),
        "y_mean": float(y.mean()),
        "x_std": float(x.std()),
        "y_std": float(y.std()),
    }


def evaluate_pair(args: argparse.Namespace, config: TeacherConfig, matcher: object, sample_index: int, pair_path: Path) -> PairDiag:
    start = time.perf_counter()
    min_labels = min_labels_for_gate(config.gate)
    try:
        image_a, image_b, warp_a_to_b, valid_mask = plg.load_pair(pair_path)
        raw = matcher.match(image_a, image_b)
        inlier_a, inlier_b = A4.ransac_inliers(raw.points_a, raw.points_b, threshold_px=config.ransac_threshold_px)
        homography_pass = int(inlier_a.shape[0] >= config.min_inliers)
        if not homography_pass:
            inlier_a, inlier_b = plg.empty_points(), plg.empty_points()
        truth_a, truth_b, errors = plg.filter_matches_by_warp_truth(
            inlier_a,
            inlier_b,
            warp_a_to_b,
            valid_mask,
            threshold_px=args.truth_threshold_px,
        )
        capped_a, _, capped_errors = plg.cap_matches(
            truth_a,
            truth_b,
            errors,
            max_matches=args.max_labels_per_pair,
            seed=args.seed + sample_index,
        )
        truth_labels = int(truth_a.shape[0])
        inliers = int(inlier_a.shape[0])
        wrong = max(0, inliers - truth_labels)
        precision = 0.0 if inliers == 0 else truth_labels / inliers
        kept = int(homography_pass and truth_labels >= min_labels and precision >= args.min_truth_precision)
        if raw.raw_matches < 4:
            reason = "too_few_raw_matches"
        elif not homography_pass:
            reason = "homography_fail"
        elif truth_labels < min_labels:
            reason = "too_few_truth_labels"
        elif precision < args.min_truth_precision:
            reason = "low_truth_precision"
        else:
            reason = "kept"
        stats = spatial_stats(capped_a if kept else plg.empty_points(), image_a.shape[0], image_a.shape[1])
        return PairDiag(
            style=config.style,
            gate=config.gate,
            sample_index=sample_index,
            source_name=source_name(pair_path),
            pair_name=pair_path.name,
            pair_pt=pair_path.as_posix(),
            profile=config.profile,
            algorithm=config.algorithm,
            status="ok",
            keypoints_a=raw.keypoints_a,
            keypoints_b=raw.keypoints_b,
            raw_matches=raw.raw_matches,
            homography_inliers=inliers,
            truth_labels=truth_labels,
            wrong_inliers=wrong,
            truth_precision=precision,
            kept_pair=kept,
            kept_labels=int(capped_a.shape[0]) if kept else 0,
            homography_pass=homography_pass,
            mean_label_error_px=float(errors.mean()) if errors.size else math.nan,
            median_label_error_px=float(np.median(errors)) if errors.size else math.nan,
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            failure_reason=reason,
            **stats,
        )
    except Exception as exc:
        return PairDiag(
            style=config.style,
            gate=config.gate,
            sample_index=sample_index,
            source_name=source_name(pair_path),
            pair_name=pair_path.name,
            pair_pt=pair_path.as_posix(),
            profile=config.profile,
            algorithm=config.algorithm,
            status="error",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            homography_inliers=0,
            truth_labels=0,
            wrong_inliers=0,
            truth_precision=0.0,
            kept_pair=0,
            kept_labels=0,
            homography_pass=0,
            mean_label_error_px=math.nan,
            median_label_error_px=math.nan,
            grid_cells_4x4=0,
            grid_coverage_4x4=0.0,
            edge_label_fraction=0.0,
            center_label_fraction=0.0,
            x_mean=math.nan,
            y_mean=math.nan,
            x_std=math.nan,
            y_std=math.nan,
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            failure_reason="error",
            message=f"{type(exc).__name__}: {exc}",
        )


def pct(values: list[int], q: float) -> float:
    return float(np.percentile(values, q)) if values else math.nan


def source_entropy(sources: list[str]) -> float:
    if not sources:
        return 0.0
    counts: dict[str, int] = {}
    for source in sources:
        counts[source] = counts.get(source, 0) + 1
    probs = np.array(list(counts.values()), dtype=np.float64) / len(sources)
    return float(-(probs * np.log2(probs)).sum())


def summarize_group(items: list[PairDiag]) -> dict[str, object]:
    ok = [row for row in items if row.status == "ok"]
    kept = [row for row in ok if row.kept_pair]
    sources = [row.source_name for row in kept]
    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    top_source_fraction = 0.0 if not sources else max(source_counts.values()) / len(sources)
    failures: dict[str, int] = {}
    for row in items:
        failures[row.failure_reason] = failures.get(row.failure_reason, 0) + 1
    inliers = [row.homography_inliers for row in ok]
    labels = [row.truth_labels for row in ok]
    kept_grid = [row.grid_coverage_4x4 for row in kept]
    kept_edge = [row.edge_label_fraction for row in kept]
    kept_center = [row.center_label_fraction for row in kept]
    total_inliers = sum(row.homography_inliers for row in ok)
    total_truth = sum(row.truth_labels for row in ok)
    return {
        "algorithm": items[0].algorithm,
        "sampled_pairs": len(items),
        "ok_pairs": len(ok),
        "homography_pass_pairs": sum(row.homography_pass for row in ok),
        "homography_pass_rate": 0.0 if not ok else sum(row.homography_pass for row in ok) / len(ok),
        "kept_pairs": len(kept),
        "kept_pair_rate": 0.0 if not ok else len(kept) / len(ok),
        "kept_sources": len(set(sources)),
        "top_source_fraction": top_source_fraction,
        "source_entropy": source_entropy(sources),
        "labels": sum(row.kept_labels for row in kept),
        "homography_inliers": total_inliers,
        "truth_labels": total_truth,
        "wrong_inliers": sum(row.wrong_inliers for row in ok),
        "truth_precision": 0.0 if total_inliers == 0 else total_truth / total_inliers,
        "mean_inliers_per_pair": float(np.mean(inliers)) if inliers else math.nan,
        "median_inliers_per_pair": float(np.median(inliers)) if inliers else math.nan,
        "p10_inliers_per_pair": pct(inliers, 10),
        "p90_inliers_per_pair": pct(inliers, 90),
        "mean_labels_per_pair": float(np.mean(labels)) if labels else math.nan,
        "median_labels_per_pair": float(np.median(labels)) if labels else math.nan,
        "p10_labels_per_pair": pct(labels, 10),
        "p90_labels_per_pair": pct(labels, 90),
        "mean_grid_coverage_4x4": float(np.mean(kept_grid)) if kept_grid else 0.0,
        "median_grid_coverage_4x4": float(np.median(kept_grid)) if kept_grid else 0.0,
        "low_spatial_coverage_pairs": sum(value < 0.25 for value in kept_grid),
        "mean_edge_label_fraction": float(np.mean(kept_edge)) if kept_edge else 0.0,
        "mean_center_label_fraction": float(np.mean(kept_center)) if kept_center else 0.0,
        "failure_counts": ";".join(f"{key}:{value}" for key, value in sorted(failures.items())),
    }


def aggregate(rows: list[PairDiag]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[PairDiag]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.profile), []).append(row)
    out = []
    for (style, gate, profile), items in sorted(grouped.items()):
        out.append({"style": style, "gate": gate, "profile": profile, **summarize_group(items)})
    return out


def overlap_rows(rows: list[PairDiag]) -> list[dict[str, object]]:
    tc = [row for row in rows if row.style == "timestamp" and row.gate == "compound"]
    by_profile: dict[str, dict[str, PairDiag]] = {}
    for row in tc:
        by_profile.setdefault(row.profile, {})[row.pair_pt] = row
    pairs = [
        ("baseline_r080_ht2", "style_timestamp_compound"),
        ("baseline_r090_ht3", "style_timestamp_compound"),
        ("baseline_r080_ht2", "baseline_r090_ht3"),
    ]
    out = []
    for a, b in pairs:
        map_a = by_profile.get(a, {})
        map_b = by_profile.get(b, {})
        all_pairs = set(map_a) | set(map_b)
        kept_a = {pair for pair, row in map_a.items() if row.kept_pair}
        kept_b = {pair for pair, row in map_b.items() if row.kept_pair}
        both = kept_a & kept_b
        a_only = kept_a - kept_b
        b_only = kept_b - kept_a
        src_a = {map_a[pair].source_name for pair in kept_a}
        src_b = {map_b[pair].source_name for pair in kept_b}

        def med(profile_map: dict[str, PairDiag], pairs_set: set[str], attr: str) -> float:
            values = [getattr(profile_map[pair], attr) for pair in pairs_set if pair in profile_map]
            return float(np.median(values)) if values else math.nan

        out.append(
            {
                "style": "timestamp",
                "gate": "compound",
                "profile_a": a,
                "profile_b": b,
                "sampled_pairs": len(all_pairs),
                "kept_a": len(kept_a),
                "kept_b": len(kept_b),
                "both_kept": len(both),
                "a_only": len(a_only),
                "b_only": len(b_only),
                "neither": len(all_pairs - (kept_a | kept_b)),
                "kept_jaccard": 0.0 if not (kept_a | kept_b) else len(both) / len(kept_a | kept_b),
                "sources_a": len(src_a),
                "sources_b": len(src_b),
                "sources_both": len(src_a & src_b),
                "source_jaccard": 0.0 if not (src_a | src_b) else len(src_a & src_b) / len(src_a | src_b),
                "median_inliers_a_only": med(map_a, a_only, "homography_inliers"),
                "median_inliers_b_only": med(map_b, b_only, "homography_inliers"),
                "median_inliers_both_a": med(map_a, both, "homography_inliers"),
                "median_inliers_both_b": med(map_b, both, "homography_inliers"),
                "median_labels_a_only": med(map_a, a_only, "truth_labels"),
                "median_labels_b_only": med(map_b, b_only, "truth_labels"),
                "median_labels_both_a": med(map_a, both, "truth_labels"),
                "median_labels_both_b": med(map_b, both, "truth_labels"),
            }
        )
    return out


def source_rows(rows: list[PairDiag]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[PairDiag]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.profile, row.source_name), []).append(row)
    out = []
    for (style, gate, profile, source), items in sorted(grouped.items()):
        kept = [row for row in items if row.kept_pair]
        out.append(
            {
                "style": style,
                "gate": gate,
                "profile": profile,
                "source_name": source,
                "sampled_pairs": len(items),
                "kept_pairs": len(kept),
                "kept_labels": sum(row.kept_labels for row in kept),
                "mean_grid_coverage_4x4": float(np.mean([row.grid_coverage_4x4 for row in kept])) if kept else 0.0,
            }
        )
    return out


def ranking(row: dict[str, object]) -> tuple[int, int, float]:
    return (-int(row["kept_pairs"]), -int(row["labels"]), -float(row["truth_precision"]))


def write_summary(args: argparse.Namespace, summary: list[dict[str, object]], overlaps: list[dict[str, object]], skipped: list[dict[str, str]]) -> None:
    tc = [row for row in summary if row["style"] == "timestamp" and row["gate"] == "compound"]
    tc_sorted = sorted(tc, key=ranking)
    lines = [
        "# Agent13 Stage3 Coverage Diagnosis",
        "",
        "## Scope",
        "",
        f"- Split root: `{args.split_root}`.",
        f"- Pairs per group: `{args.pairs_per_group}`; seed `{args.seed}`.",
        "- Expanded stage2 style-specific teachers on train split; timestamp/compound additionally compares r0.80/t2, r0.88/Ht3, and r0.90/Ht3.",
        "- No PFM training was run.",
        "",
        "## Summary Metrics",
        "",
        "| style | gate | profile | kept | labels | precision | kept sources | top source frac | grid cov | edge frac | median inliers | failures |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(summary, key=lambda item: (str(item["style"]), str(item["gate"]), ranking(item))):
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['profile']} | {row['kept_pairs']} | {row['labels']} | "
            f"{float(row['truth_precision']):.4f} | {row['kept_sources']} | {float(row['top_source_fraction']):.3f} | "
            f"{float(row['mean_grid_coverage_4x4']):.3f} | {float(row['mean_edge_label_fraction']):.3f} | "
            f"{float(row['median_inliers_per_pair']):.1f} | {row['failure_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Timestamp/Compound Overlap",
            "",
            "| profile A | profile B | kept A | kept B | both | A only | B only | Jaccard | source Jaccard |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overlaps:
        lines.append(
            f"| {row['profile_a']} | {row['profile_b']} | {row['kept_a']} | {row['kept_b']} | {row['both_kept']} | "
            f"{row['a_only']} | {row['b_only']} | {float(row['kept_jaccard']):.3f} | {float(row['source_jaccard']):.3f} |"
        )
    lines.extend(["", "## Diagnosis", ""])
    if tc_sorted:
        best = tc_sorted[0]
        r088 = next((row for row in tc if row["profile"] == "style_timestamp_compound"), None)
        r080 = next((row for row in tc if row["profile"] == "baseline_r080_ht2"), None)
        r090 = next((row for row in tc if row["profile"] == "baseline_r090_ht3"), None)
        lines.append(
            f"- Timestamp/compound best by kept/labels is `{best['profile']}` with {best['kept_pairs']} kept pairs and {best['labels']} capped labels."
        )
        if r088 and r080 and r090:
            lines.append(
                f"- r0.88/Ht3 keeps {r088['kept_pairs']} pairs, versus r0.80/t2 {r080['kept_pairs']} and r0.90/Ht3 {r090['kept_pairs']}; "
                f"precision is {float(r088['truth_precision']):.4f}, {float(r080['truth_precision']):.4f}, and {float(r090['truth_precision']):.4f} respectively."
            )
            lines.append(
                f"- r0.88/Ht3 median inliers are {float(r088['median_inliers_per_pair']):.1f}, between r0.80/t2 {float(r080['median_inliers_per_pair']):.1f} "
                f"and r0.90/Ht3 {float(r090['median_inliers_per_pair']):.1f}; this points to coverage/route distribution rather than raw precision as the likely transfer issue."
            )
            lines.append(
                f"- r0.88/Ht3 source concentration top fraction is {float(r088['top_source_fraction']):.3f} across {r088['kept_sources']} kept sources; "
                f"mean 4x4 grid coverage is {float(r088['mean_grid_coverage_4x4']):.3f} and edge fraction is {float(r088['mean_edge_label_fraction']):.3f}."
            )
    lines.extend(
        [
            "",
            "## Next Teacher Recommendation",
            "",
            "- Do not use timestamp/compound r0.88/Ht3 as the only heatmap pseudo-label route. Keep r0.80/t2 as a fallback branch because its kept-pair set is not identical and parent route eval still needs that fallback.",
            "- For timestamp/compound, prefer a union teacher: r0.88/Ht3 high-confidence labels plus r0.80/t2-only kept pairs that pass the same truth/gate filters; optionally add r0.90/Ht3 where it contributes unique kept pairs.",
            "- Add source-balanced sampling or per-source caps for timestamp/compound pseudo labels before the next heatmap-only run; high train precision alone is not enough if kept labels are concentrated in a subset of sources/pairs.",
            "- Keep numeric/compound on LightGlue-SIFT or a LightGlue-first/RootSIFT fallback route; RootSIFT global r0.88 is weaker there.",
            "",
            "## Skipped / Unavailable",
            "",
        ]
    )
    if skipped:
        for item in skipped:
            lines.append(f"- {item['profile']} `{item['algorithm']}`: {item['reason']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `summary_metrics.csv`: expanded style/gate teacher evidence with source and spatial coverage.",
            "- `overlap_metrics.csv`: timestamp/compound kept-pair overlap across r0.80/t2, r0.88/Ht3, and r0.90/Ht3.",
            "- `pair_metrics.csv`: per sampled pair diagnostics.",
            "- `source_metrics.csv`: per-source kept pair and label distribution.",
            "- `sampled_pairs.csv`: fixed sampled train pairs.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    configs = teacher_configs()
    assert any(item.profile == "style_timestamp_compound" for item in configs)
    assert any(item.profile == "baseline_r080_ht2" for item in configs)
    assert DEFAULT_SPLIT_ROOT.exists()
    assert STAGE2_SUMMARY.exists()
    stats = spatial_stats(np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32), 200, 200)
    assert stats["grid_cells_4x4"] == 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--pairs-per-group", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2234)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--learned-max-keypoints", type=int, default=512)
    parser.add_argument("--max-raw-matches", type=int, default=256)
    parser.add_argument("--max-labels-per-pair", type=int, default=128)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--truth-threshold-px", type=float, default=5.0)
    parser.add_argument("--min-truth-precision", type=float, default=0.95)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = teacher_configs()
    groups = sorted({(item.style, item.gate) for item in configs})
    samples: dict[tuple[str, str], list[Path]] = {}
    sampled_rows: list[dict[str, object]] = []
    for group_index, (style, gate) in enumerate(groups):
        paths = discover_pairs(args.split_root, style, gate, limit=args.pairs_per_group, seed=args.seed + group_index)
        samples[(style, gate)] = paths
        print(f"group={style}/{gate} sampled={len(paths)}", flush=True)
        for index, path in enumerate(paths, start=1):
            sampled_rows.append(
                {
                    "style": style,
                    "gate": gate,
                    "sample_index": index,
                    "source_name": source_name(path),
                    "pair_name": path.name,
                    "pair_pt": path.as_posix(),
                }
            )

    factory = MatcherFactory(args)
    pair_rows: list[PairDiag] = []
    for config in configs:
        matcher = factory.matcher(config)
        if matcher is None:
            continue
        for index, path in enumerate(samples[(config.style, config.gate)], start=1):
            pair_rows.append(evaluate_pair(args, config, matcher, index, path))
        print(f"{config.style}/{config.gate} {config.profile} done", flush=True)

    summary = aggregate(pair_rows)
    overlaps = overlap_rows(pair_rows)
    sources = source_rows(pair_rows)
    write_csv(args.output_dir / "pair_metrics.csv", [asdict(row) for row in pair_rows], PAIR_FIELDS)
    write_csv(args.output_dir / "summary_metrics.csv", summary, SUMMARY_FIELDS)
    write_csv(args.output_dir / "overlap_metrics.csv", overlaps, OVERLAP_FIELDS)
    write_csv(args.output_dir / "source_metrics.csv", sources, SOURCE_FIELDS)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled_rows, SAMPLED_FIELDS)
    write_csv(args.output_dir / "skipped_teachers.csv", factory.skipped, SKIPPED_FIELDS)
    write_summary(args, summary, overlaps, factory.skipped)
    print(f"output_dir={args.output_dir}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
