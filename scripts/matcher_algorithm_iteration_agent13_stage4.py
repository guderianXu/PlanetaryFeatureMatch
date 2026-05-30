#!/usr/bin/env python3
"""Agent13 stage4 timestamp/compound hard-tail matcher mining."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage4"
STAGE3_PAIR_METRICS = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage3" / "pair_metrics.csv"


PAIR_FIELDS = [
    "source_name",
    "pair_name",
    "pair_rel",
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
    "diagnostic_pair",
    "candidate_labels",
    "homography_pass",
    "min_labels",
    "min_truth_precision",
    "mean_error_px",
    "median_error_px",
    "grid_cells_4x4",
    "grid_coverage_4x4",
    "edge_label_fraction",
    "center_label_fraction",
    "runtime_ms",
    "failure_reason",
    "message",
]

SUMMARY_FIELDS = [
    "profile",
    "algorithm",
    "teacher_use",
    "hardtail_pairs",
    "ok_pairs",
    "homography_pass_pairs",
    "homography_pass_rate",
    "kept_pairs",
    "kept_pair_rate",
    "diagnostic_pairs",
    "candidate_labels",
    "kept_truth_labels",
    "kept_homography_inliers",
    "kept_truth_precision",
    "truth_labels",
    "homography_inliers",
    "wrong_inliers",
    "truth_precision",
    "median_inliers",
    "median_truth_labels",
    "p90_truth_labels",
    "kept_sources",
    "top_source_fraction",
    "mean_grid_coverage_4x4",
    "mean_edge_label_fraction",
    "failure_counts",
]

LABEL_FIELDS = ["profile", "algorithm", "teacher_use", "source_name", "pair_name", "pair_rel", "ax", "ay", "bx", "by", "error_px"]
HARDTAIL_FIELDS = ["source_name", "pair_name", "pair_rel", "stage3_truth_labels", "stage3_homography_inliers", "stage3_truth_precision"]
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
    teacher_use: str
    clahe: bool = False


@dataclass(frozen=True)
class PairRow:
    source_name: str
    pair_name: str
    pair_rel: str
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
    diagnostic_pair: int
    candidate_labels: int
    homography_pass: int
    min_labels: int
    min_truth_precision: float
    mean_error_px: float
    median_error_px: float
    grid_cells_4x4: int
    grid_coverage_4x4: float
    edge_label_fraction: float
    center_label_fraction: float
    runtime_ms: float
    failure_reason: str
    message: str = ""


@dataclass(frozen=True)
class LabelRow:
    profile: str
    algorithm: str
    teacher_use: str
    source_name: str
    pair_name: str
    pair_rel: str
    ax: float
    ay: float
    bx: float
    by: float
    error_px: float


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A13 = load_module(AGENT13_SCRIPT, "agent13_matcher_for_stage4")
A4 = A13.A4


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def recover_hardtail(stage3_pair_metrics: Path, hardtail_profile: str) -> list[dict[str, object]]:
    rows = read_csv(stage3_pair_metrics)
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("style") != "timestamp" or row.get("gate") != "compound":
            continue
        if row.get("profile") != hardtail_profile:
            continue
        if row.get("failure_reason") != "too_few_truth_labels":
            continue
        pair_path = Path(row["pair_pt"])
        key = rel(pair_path)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "path": pair_path,
                "source_name": row.get("source_name", pair_path.parent.name),
                "pair_name": row.get("pair_name", pair_path.name),
                "pair_rel": key,
                "stage3_truth_labels": row.get("truth_labels", ""),
                "stage3_homography_inliers": row.get("homography_inliers", ""),
                "stage3_truth_precision": row.get("truth_precision", ""),
            }
        )
    return out


def teacher_configs() -> list[TeacherConfig]:
    return [
        TeacherConfig("lightglue_sift_ht3", "LightGlue-SIFT-Ht3-min4", "LightGlue-SIFT", "lightglue", math.nan, 3.0, 4, "train_candidate"),
        TeacherConfig("rootsift_r088_ht3", "RootSIFT-ratio-r0p88-Ht3-min4", "RootSIFT", "ratio", 0.88, 3.0, 4, "train_candidate"),
        TeacherConfig("rootsift_r092_ht3", "RootSIFT-ratio-r0p92-Ht3-min4", "RootSIFT", "ratio", 0.92, 3.0, 4, "train_candidate"),
        TeacherConfig("rootsift_r095_ht3", "RootSIFT-ratio-r0p95-Ht3-min4", "RootSIFT", "ratio", 0.95, 3.0, 4, "diagnostic"),
        TeacherConfig("rootsift_r095_ht5_diag", "RootSIFT-ratio-r0p95-Ht5-min4", "RootSIFT", "ratio", 0.95, 5.0, 4, "diagnostic"),
        TeacherConfig("clahe_rootsift_r092_ht3", "CLAHE-RootSIFT-ratio-r0p92-Ht3-min4", "RootSIFT", "ratio", 0.92, 3.0, 4, "train_candidate", clahe=True),
        TeacherConfig("clahe_rootsift_r095_ht3", "CLAHE-RootSIFT-ratio-r0p95-Ht3-min4", "RootSIFT", "ratio", 0.95, 3.0, 4, "diagnostic", clahe=True),
        TeacherConfig("akaze_cross_ht3", "AKAZE-cross-Ht3-min4", "AKAZE", "cross", math.nan, 3.0, 4, "fallback"),
        TeacherConfig("akaze_ratio_r095_ht3", "AKAZE-ratio-r0p95-Ht3-min4", "AKAZE", "ratio", 0.95, 3.0, 4, "fallback"),
        TeacherConfig("orb_cross_ht3", "ORB-cross-Ht3-min4", "ORB", "cross", math.nan, 3.0, 4, "fallback"),
    ]


class ClaheMatcher:
    def __init__(self, base: object) -> None:
        import cv2

        self.base = base
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def match(self, image_a: np.ndarray, image_b: np.ndarray):
        return self.base.match(self.clahe.apply(image_a), self.clahe.apply(image_b))


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
                if config.clahe:
                    matcher = ClaheMatcher(matcher)
            self.cache[config.profile] = matcher
            return matcher
        except Exception as exc:
            self.skipped.append({"profile": config.profile, "algorithm": config.algorithm, "reason": f"{type(exc).__name__}: {exc}"})
            return None


def spatial_stats(points_a: np.ndarray, height: int, width: int) -> dict[str, float | int]:
    if points_a.size == 0:
        return {"grid_cells_4x4": 0, "grid_coverage_4x4": 0.0, "edge_label_fraction": 0.0, "center_label_fraction": 0.0}
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
    }


def evaluate_pair(args: argparse.Namespace, config: TeacherConfig, matcher: object, pair_info: dict[str, object]) -> tuple[PairRow, list[LabelRow]]:
    start = time.perf_counter()
    pair_path = Path(pair_info["path"])
    min_labels = args.min_labels
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
        capped_a, capped_b, capped_errors = plg.cap_matches(
            truth_a,
            truth_b,
            errors,
            max_matches=args.max_labels_per_pair,
            seed=args.seed + hash((config.profile, str(pair_path))) % 100000,
        )
        truth_labels = int(truth_a.shape[0])
        inliers = int(inlier_a.shape[0])
        wrong = max(0, inliers - truth_labels)
        precision = 0.0 if inliers == 0 else truth_labels / inliers
        kept = int(homography_pass and truth_labels >= min_labels and precision >= args.min_truth_precision)
        diagnostic = int(homography_pass and truth_labels >= min_labels and not kept)
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
        labels: list[LabelRow] = []
        if kept and config.teacher_use != "diagnostic":
            labels = [
                LabelRow(
                    profile=config.profile,
                    algorithm=config.algorithm,
                    teacher_use=config.teacher_use,
                    source_name=str(pair_info["source_name"]),
                    pair_name=str(pair_info["pair_name"]),
                    pair_rel=str(pair_info["pair_rel"]),
                    ax=float(point_a[0]),
                    ay=float(point_a[1]),
                    bx=float(point_b[0]),
                    by=float(point_b[1]),
                    error_px=float(error),
                )
                for point_a, point_b, error in zip(capped_a, capped_b, capped_errors)
            ]
        return (
            PairRow(
                source_name=str(pair_info["source_name"]),
                pair_name=str(pair_info["pair_name"]),
                pair_rel=str(pair_info["pair_rel"]),
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
                diagnostic_pair=diagnostic,
                candidate_labels=len(labels),
                homography_pass=homography_pass,
                min_labels=min_labels,
                min_truth_precision=args.min_truth_precision,
                mean_error_px=float(errors.mean()) if errors.size else math.nan,
                median_error_px=float(np.median(errors)) if errors.size else math.nan,
                runtime_ms=(time.perf_counter() - start) * 1000.0,
                failure_reason=reason,
                **stats,
            ),
            labels,
        )
    except Exception as exc:
        return (
            PairRow(
                source_name=str(pair_info["source_name"]),
                pair_name=str(pair_info["pair_name"]),
                pair_rel=str(pair_info["pair_rel"]),
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
                diagnostic_pair=0,
                candidate_labels=0,
                homography_pass=0,
                min_labels=min_labels,
                min_truth_precision=args.min_truth_precision,
                mean_error_px=math.nan,
                median_error_px=math.nan,
                grid_cells_4x4=0,
                grid_coverage_4x4=0.0,
                edge_label_fraction=0.0,
                center_label_fraction=0.0,
                runtime_ms=(time.perf_counter() - start) * 1000.0,
                failure_reason="error",
                message=f"{type(exc).__name__}: {exc}",
            ),
            [],
        )


def pct(values: list[int], q: float) -> float:
    return float(np.percentile(values, q)) if values else math.nan


def summarize(rows: list[PairRow], configs: list[TeacherConfig]) -> list[dict[str, object]]:
    config_by_profile = {config.profile: config for config in configs}
    out = []
    grouped: dict[str, list[PairRow]] = {}
    for row in rows:
        grouped.setdefault(row.profile, []).append(row)
    for profile, items in sorted(grouped.items()):
        ok = [row for row in items if row.status == "ok"]
        kept = [row for row in ok if row.kept_pair]
        sources: dict[str, int] = {}
        for row in kept:
            sources[row.source_name] = sources.get(row.source_name, 0) + 1
        failures: dict[str, int] = {}
        for row in items:
            failures[row.failure_reason] = failures.get(row.failure_reason, 0) + 1
        inliers = [row.homography_inliers for row in ok]
        truth = [row.truth_labels for row in ok]
        total_inliers = sum(inliers)
        total_truth = sum(truth)
        config = config_by_profile[profile]
        out.append(
            {
                "profile": profile,
                "algorithm": config.algorithm,
                "teacher_use": config.teacher_use,
                "hardtail_pairs": len(items),
                "ok_pairs": len(ok),
                "homography_pass_pairs": sum(row.homography_pass for row in ok),
                "homography_pass_rate": 0.0 if not ok else sum(row.homography_pass for row in ok) / len(ok),
                "kept_pairs": len(kept),
                "kept_pair_rate": 0.0 if not ok else len(kept) / len(ok),
        "diagnostic_pairs": sum(row.diagnostic_pair for row in ok),
        "candidate_labels": sum(row.candidate_labels for row in ok),
        "kept_truth_labels": sum(row.truth_labels for row in kept),
        "kept_homography_inliers": sum(row.homography_inliers for row in kept),
        "kept_truth_precision": 0.0 if not kept or sum(row.homography_inliers for row in kept) == 0 else sum(row.truth_labels for row in kept) / sum(row.homography_inliers for row in kept),
        "truth_labels": total_truth,
        "homography_inliers": total_inliers,
                "wrong_inliers": sum(row.wrong_inliers for row in ok),
                "truth_precision": 0.0 if total_inliers == 0 else total_truth / total_inliers,
                "median_inliers": float(np.median(inliers)) if inliers else math.nan,
                "median_truth_labels": float(np.median(truth)) if truth else math.nan,
                "p90_truth_labels": pct(truth, 90),
                "kept_sources": len(sources),
                "top_source_fraction": 0.0 if not sources else max(sources.values()) / len(kept),
                "mean_grid_coverage_4x4": float(np.mean([row.grid_coverage_4x4 for row in kept])) if kept else 0.0,
                "mean_edge_label_fraction": float(np.mean([row.edge_label_fraction for row in kept])) if kept else 0.0,
                "failure_counts": ";".join(f"{key}:{value}" for key, value in sorted(failures.items())),
            }
        )
    return out


def ranking(row: dict[str, object]) -> tuple[int, int, float, str]:
    return (-int(row["kept_pairs"]), -int(row["candidate_labels"]), -float(row["truth_precision"]), str(row["profile"]))


def write_summary(args: argparse.Namespace, summary_rows: list[dict[str, object]], skipped: list[dict[str, str]], hardtail_count: int) -> None:
    ordered = sorted(summary_rows, key=ranking)
    train_candidates = [row for row in ordered if row["teacher_use"] != "diagnostic" and int(row["candidate_labels"]) > 0]
    lines = [
        "# Agent13 Stage4 Timestamp/Compound Hard-Tail Mining",
        "",
        "## Scope",
        "",
        f"- Stage3 pair metrics: `{args.stage3_pair_metrics}`.",
        f"- Hard-tail profile: `{args.hardtail_profile}`; recovered pairs: `{hardtail_count}`.",
        f"- Truth threshold: `{args.truth_threshold_px}` px; kept requires at least `{args.min_labels}` truth labels and precision `{args.min_truth_precision}`.",
        "- No PFM training was run.",
        "",
        "## Summary Metrics",
        "",
        "| profile | use | kept | diagnostic | candidate labels | kept precision | all-pair precision | median inliers | median truth labels | kept sources | failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['profile']} | {row['teacher_use']} | {row['kept_pairs']} | {row['diagnostic_pairs']} | {row['candidate_labels']} | "
            f"{float(row['kept_truth_precision']):.4f} | {float(row['truth_precision']):.4f} | {float(row['median_inliers']):.1f} | {float(row['median_truth_labels']):.1f} | "
            f"{row['kept_sources']} | {row['failure_counts']} |"
        )
    lines.extend(["", "## Findings", ""])
    if ordered:
        best = ordered[0]
        lines.append(
            f"- Best hard-tail coverage by kept pairs is `{best['profile']}`: {best['kept_pairs']}/{hardtail_count} kept, "
            f"{best['candidate_labels']} candidate labels, kept-pair precision {float(best['kept_truth_precision']):.4f}."
        )
    if train_candidates:
        best_train = train_candidates[0]
        lines.append(
            f"- Best train-candidate output is `{best_train['profile']}` with {best_train['candidate_labels']} labels. "
            "Only non-diagnostic teachers are written to `candidate_labels.csv`."
        )
    light = next((row for row in summary_rows if row["profile"] == "lightglue_sift_ht3"), None)
    if light:
        lines.append(
            f"- LightGlue-SIFT on the hard-tail kept {light['kept_pairs']} pairs and produced {light['candidate_labels']} labels "
            f"at kept-pair precision {float(light['kept_truth_precision']):.4f}; all-pair precision was {float(light['truth_precision']):.4f}."
        )
    lines.extend(
        [
            "- Wide RootSIFT/CLAHE/AKAZE/ORB rows are useful only if they pass the same truth precision gate; otherwise treat them as diagnostics for why pair geometry is hard.",
            "",
            "## Next Recommendation",
            "",
            "- If no teacher keeps a meaningful fraction of the hard-tail with high precision, do not force pseudo labels from these pairs into heatmap training.",
            "- Prefer hard-tail candidates only from teachers with high truth precision and non-trivial source spread; keep diagnostic-only rows out of training labels.",
            "- For the next training-side experiment, combine Stage2/3 balanced labels with Stage4 `candidate_labels.csv` only if Stage4 adds unique kept hard-tail pairs beyond r0.88/Ht3.",
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
            "- `summary_metrics.csv`: per-teacher hard-tail coverage and precision.",
            "- `pair_metrics.csv`: per hard-tail pair and teacher.",
            "- `candidate_labels.csv`: high-precision non-diagnostic labels only, using repo-relative pair paths.",
            "- `hardtail_pairs.csv`: recovered Stage3 hard-tail list.",
            "- `skipped_teachers.csv`: unavailable or failed matcher setup.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    assert STAGE3_PAIR_METRICS.exists()
    configs = teacher_configs()
    assert any(config.profile == "lightglue_sift_ht3" for config in configs)
    assert any(config.clahe for config in configs)
    hardtail = recover_hardtail(STAGE3_PAIR_METRICS, "style_timestamp_compound")
    assert hardtail
    assert not str(hardtail[0]["pair_rel"]).startswith("/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage3-pair-metrics", type=Path, default=STAGE3_PAIR_METRICS)
    parser.add_argument("--hardtail-profile", default="style_timestamp_compound")
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--learned-max-keypoints", type=int, default=512)
    parser.add_argument("--max-raw-matches", type=int, default=256)
    parser.add_argument("--max-labels-per-pair", type=int, default=128)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--truth-threshold-px", type=float, default=5.0)
    parser.add_argument("--min-truth-precision", type=float, default=0.95)
    parser.add_argument("--min-labels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4234)
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
    hardtail = recover_hardtail(args.stage3_pair_metrics, args.hardtail_profile)
    configs = teacher_configs()
    factory = MatcherFactory(args)
    pair_rows: list[PairRow] = []
    label_rows: list[LabelRow] = []
    for config in configs:
        matcher = factory.matcher(config)
        if matcher is None:
            continue
        for item in hardtail:
            row, labels = evaluate_pair(args, config, matcher, item)
            pair_rows.append(row)
            label_rows.extend(labels)
        print(f"{config.profile} done", flush=True)
    summary_rows = summarize(pair_rows, configs)
    hardtail_rows = [
        {
            "source_name": item["source_name"],
            "pair_name": item["pair_name"],
            "pair_rel": item["pair_rel"],
            "stage3_truth_labels": item["stage3_truth_labels"],
            "stage3_homography_inliers": item["stage3_homography_inliers"],
            "stage3_truth_precision": item["stage3_truth_precision"],
        }
        for item in hardtail
    ]
    write_csv(args.output_dir / "hardtail_pairs.csv", hardtail_rows, HARDTAIL_FIELDS)
    write_csv(args.output_dir / "pair_metrics.csv", [asdict(row) for row in pair_rows], PAIR_FIELDS)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "candidate_labels.csv", [asdict(row) for row in label_rows], LABEL_FIELDS)
    write_csv(args.output_dir / "skipped_teachers.csv", factory.skipped, SKIPPED_FIELDS)
    write_summary(args, summary_rows, factory.skipped, len(hardtail))
    print(f"output_dir={args.output_dir}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
