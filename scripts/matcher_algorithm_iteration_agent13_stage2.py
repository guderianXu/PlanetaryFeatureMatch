#!/usr/bin/env python3
"""Agent13 stage2 train-split teacher mining evidence."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage2"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "runs" / "cross_view_1024_hard_mined_weakgates_80_seed1234" / "splits" / "train"
AGENT13_RUN = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13"


PAIR_FIELDS = [
    "style",
    "gate",
    "sample_index",
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
    "min_homography_inliers",
    "min_labels",
    "min_truth_precision",
    "mean_label_error_px",
    "median_label_error_px",
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
    "labels",
    "raw_matches",
    "homography_inliers",
    "truth_labels",
    "wrong_inliers",
    "truth_precision",
    "mean_inliers_per_pair",
    "median_inliers_per_pair",
    "mean_labels_per_pair",
    "median_labels_per_pair",
    "mean_runtime_ms",
    "failure_counts",
]

LABEL_FIELDS = ["style", "gate", "profile", "algorithm", "pair_pt", "ax", "ay", "bx", "by", "error_px"]
SAMPLED_FIELDS = ["style", "gate", "sample_index", "pair_pt"]
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
    style: str | None = None
    gate: str | None = None


@dataclass(frozen=True)
class PairRow:
    style: str
    gate: str
    sample_index: int
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
    min_homography_inliers: int
    min_labels: int
    min_truth_precision: float
    mean_label_error_px: float
    median_label_error_px: float
    runtime_ms: float
    failure_reason: str
    message: str = ""


@dataclass(frozen=True)
class LabelRow:
    style: str
    gate: str
    profile: str
    algorithm: str
    pair_pt: str
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


A13 = load_module(AGENT13_SCRIPT, "agent13_matcher_for_stage2")
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


def discover_pairs(split_root: Path, style: str, gate: str, *, limit: int, seed: int) -> list[Path]:
    root = split_root / style / gate
    if not root.exists():
        return []
    paths: list[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if filename.endswith(".pt") and filename.startswith("pair_"):
                paths.append(Path(dirpath) / filename)
    paths = sorted(paths)
    if limit > 0 and len(paths) > limit:
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, k=limit))
    return paths


def all_teacher_configs() -> list[TeacherConfig]:
    return [
        TeacherConfig(
            profile="baseline_r080_ht2",
            algorithm="RootSIFT-ratio-r0p80-Ht2-min4",
            detector="RootSIFT",
            mode="ratio",
            ratio=0.80,
            ransac_threshold_px=2.0,
            min_inliers=4,
        ),
        TeacherConfig(
            profile="baseline_r090_ht3",
            algorithm="RootSIFT-ratio-r0p90-Ht3-min4",
            detector="RootSIFT",
            mode="ratio",
            ratio=0.90,
            ransac_threshold_px=3.0,
            min_inliers=4,
        ),
        TeacherConfig(
            profile="global_r088_ht3",
            algorithm="RootSIFT-ratio-r0p88-Ht3-min4",
            detector="RootSIFT",
            mode="ratio",
            ratio=0.88,
            ransac_threshold_px=3.0,
            min_inliers=4,
        ),
        TeacherConfig(
            profile="style_numeric_viewpoint",
            algorithm="RootSIFT-mutual-r0p95-Ht3-min4",
            detector="RootSIFT",
            mode="ratio_mutual",
            ratio=0.95,
            ransac_threshold_px=3.0,
            min_inliers=4,
            style="numeric",
            gate="viewpoint",
        ),
        TeacherConfig(
            profile="style_numeric_compound",
            algorithm="LightGlue-SIFT-Ht3-min4",
            detector="LightGlue-SIFT",
            mode="lightglue",
            ratio=math.nan,
            ransac_threshold_px=3.0,
            min_inliers=4,
            style="numeric",
            gate="compound",
        ),
        TeacherConfig(
            profile="style_timestamp_viewpoint",
            algorithm="RootSIFT-ratio-r0p92-Ht3-min4",
            detector="RootSIFT",
            mode="ratio",
            ratio=0.92,
            ransac_threshold_px=3.0,
            min_inliers=4,
            style="timestamp",
            gate="viewpoint",
        ),
        TeacherConfig(
            profile="style_timestamp_compound",
            algorithm="RootSIFT-ratio-r0p88-Ht3-min4",
            detector="RootSIFT",
            mode="ratio",
            ratio=0.88,
            ransac_threshold_px=3.0,
            min_inliers=4,
            style="timestamp",
            gate="compound",
        ),
    ]


class MatcherFactory:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._cache: dict[str, object] = {}
        self.skipped: list[dict[str, str]] = []

    def applies(self, config: TeacherConfig, style: str, gate: str) -> bool:
        return (config.style is None or config.style == style) and (config.gate is None or config.gate == gate)

    def matcher(self, config: TeacherConfig) -> object | None:
        if config.profile in self._cache:
            return self._cache[config.profile]
        try:
            if config.detector == "LightGlue-SIFT":
                if importlib.util.find_spec("lightglue") is None:
                    self.skipped.append(
                        {"profile": config.profile, "algorithm": config.algorithm, "reason": "module 'lightglue' unavailable"}
                    )
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
            self._cache[config.profile] = matcher
            return matcher
        except Exception as exc:
            self.skipped.append({"profile": config.profile, "algorithm": config.algorithm, "reason": f"{type(exc).__name__}: {exc}"})
            return None


def evaluate_pair(
    args: argparse.Namespace,
    config: TeacherConfig,
    matcher: object,
    *,
    style: str,
    gate: str,
    sample_index: int,
    pair_path: Path,
) -> tuple[PairRow, list[LabelRow]]:
    start = time.perf_counter()
    min_labels = min_labels_for_gate(gate)
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
            seed=args.seed + sample_index,
        )
        truth_labels = int(truth_a.shape[0])
        homography_inliers = int(inlier_a.shape[0])
        wrong_inliers = max(0, homography_inliers - truth_labels)
        precision = 0.0 if homography_inliers == 0 else truth_labels / homography_inliers
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
        label_rows = []
        if kept:
            label_rows = [
                LabelRow(
                    style=style,
                    gate=gate,
                    profile=config.profile,
                    algorithm=config.algorithm,
                    pair_pt=pair_path.as_posix(),
                    ax=float(point_a[0]),
                    ay=float(point_a[1]),
                    bx=float(point_b[0]),
                    by=float(point_b[1]),
                    error_px=float(error),
                )
                for point_a, point_b, error in zip(capped_a, capped_b, capped_errors)
            ]
        row = PairRow(
            style=style,
            gate=gate,
            sample_index=sample_index,
            pair_pt=pair_path.as_posix(),
            profile=config.profile,
            algorithm=config.algorithm,
            status="ok",
            keypoints_a=raw.keypoints_a,
            keypoints_b=raw.keypoints_b,
            raw_matches=raw.raw_matches,
            homography_inliers=homography_inliers,
            truth_labels=truth_labels,
            wrong_inliers=wrong_inliers,
            truth_precision=precision,
            kept_pair=kept,
            kept_labels=int(capped_a.shape[0]) if kept else 0,
            homography_pass=homography_pass,
            min_homography_inliers=config.min_inliers,
            min_labels=min_labels,
            min_truth_precision=args.min_truth_precision,
            mean_label_error_px=float(errors.mean()) if errors.size else math.nan,
            median_label_error_px=float(np.median(errors)) if errors.size else math.nan,
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            failure_reason=reason,
        )
        return row, label_rows
    except Exception as exc:
        row = PairRow(
            style=style,
            gate=gate,
            sample_index=sample_index,
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
            min_homography_inliers=config.min_inliers,
            min_labels=min_labels,
            min_truth_precision=args.min_truth_precision,
            mean_label_error_px=math.nan,
            median_label_error_px=math.nan,
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            failure_reason="error",
            message=f"{type(exc).__name__}: {exc}",
        )
        return row, []


def summarize_group(rows: list[PairRow]) -> dict[str, object]:
    ok = [row for row in rows if row.status == "ok"]
    labels = sum(row.kept_labels for row in ok)
    raw = sum(row.raw_matches for row in ok)
    inliers = sum(row.homography_inliers for row in ok)
    correct = sum(row.truth_labels for row in ok)
    wrong = sum(row.wrong_inliers for row in ok)
    failure_counts: dict[str, int] = {}
    for row in rows:
        failure_counts[row.failure_reason] = failure_counts.get(row.failure_reason, 0) + 1
    failure_text = ";".join(f"{key}:{value}" for key, value in sorted(failure_counts.items()))
    return {
        "algorithm": rows[0].algorithm,
        "sampled_pairs": len(rows),
        "ok_pairs": len(ok),
        "homography_pass_pairs": sum(row.homography_pass for row in ok),
        "homography_pass_rate": 0.0 if not ok else sum(row.homography_pass for row in ok) / len(ok),
        "kept_pairs": sum(row.kept_pair for row in ok),
        "kept_pair_rate": 0.0 if not ok else sum(row.kept_pair for row in ok) / len(ok),
        "labels": labels,
        "raw_matches": raw,
        "homography_inliers": inliers,
        "truth_labels": correct,
        "wrong_inliers": wrong,
        "truth_precision": 0.0 if inliers == 0 else correct / inliers,
        "mean_inliers_per_pair": float(np.mean([row.homography_inliers for row in ok])) if ok else math.nan,
        "median_inliers_per_pair": float(np.median([row.homography_inliers for row in ok])) if ok else math.nan,
        "mean_labels_per_pair": float(np.mean([row.truth_labels for row in ok])) if ok else math.nan,
        "median_labels_per_pair": float(np.median([row.truth_labels for row in ok])) if ok else math.nan,
        "mean_runtime_ms": float(np.mean([row.runtime_ms for row in ok])) if ok else math.nan,
        "failure_counts": failure_text,
    }


def aggregate(rows: list[PairRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[PairRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.profile), []).append(row)
    out: list[dict[str, object]] = []
    for (style, gate, profile), items in sorted(grouped.items()):
        out.append({"style": style, "gate": gate, "profile": profile, **summarize_group(items)})
    return out


def ranking_key(row: dict[str, object]) -> tuple[int, int, float, float, str]:
    return (
        -int(row["kept_pairs"]),
        -int(row["labels"]),
        -float(row["truth_precision"]),
        -float(row["homography_pass_rate"]),
        str(row["profile"]),
    )


def best_by_group(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["style"]), str(row["gate"])), []).append(row)
    return [sorted(items, key=ranking_key)[0] for _, items in sorted(grouped.items())]


def write_summary(
    args: argparse.Namespace,
    pair_rows: list[PairRow],
    summary_rows: list[dict[str, object]],
    sampled_rows: list[dict[str, object]],
    skipped: list[dict[str, str]],
) -> None:
    lines = [
        "# Agent13 Stage2 Teacher Mining",
        "",
        "## Scope",
        "",
        f"- Split root: `{args.split_root}`.",
        f"- Samples per style/gate: `{args.pairs_per_group}` with seed `{args.seed}`.",
        f"- Truth threshold: `{args.truth_threshold_px}` px; min truth precision for kept pairs: `{args.min_truth_precision}`.",
        "- Kept pair definition: homography pass, gate-specific truth labels reached, and truth precision above threshold.",
        "- Gate-specific label thresholds: 20 for viewpoint, 8 for compound.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python "
        f"scripts/{Path(__file__).name} --device {args.device} --output-dir {args.output_dir}",
        "```",
        "",
        "## Group Summary",
        "",
        "| style | gate | profile | algorithm | sampled | homo pass | kept | labels | truth precision | mean inliers | median inliers | failures |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(summary_rows, key=lambda item: (str(item["style"]), str(item["gate"]), ranking_key(item))):
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['profile']} | {row['algorithm']} | {row['sampled_pairs']} | "
            f"{row['homography_pass_pairs']} | {row['kept_pairs']} | {row['labels']} | {float(row['truth_precision']):.4f} | "
            f"{float(row['mean_inliers_per_pair']):.2f} | {float(row['median_inliers_per_pair']):.2f} | {row['failure_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Best Per Style/Gate",
            "",
            "| style | gate | profile | algorithm | kept | labels | truth precision | recommendation |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in best_by_group(summary_rows):
        recommendation = "use"
        if int(row["kept_pairs"]) == 0:
            recommendation = "do not use alone"
        elif "style_numeric_compound" == row["profile"]:
            recommendation = "use only with fallback/gating"
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['profile']} | {row['algorithm']} | {row['kept_pairs']} | "
            f"{row['labels']} | {float(row['truth_precision']):.4f} | {recommendation} |"
        )
    lines.extend(["", "## Skipped / Unavailable", ""])
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
            "- `pair_metrics.csv`: per sampled pair and teacher profile.",
            "- `summary_metrics.csv`: grouped evidence for teacher selection.",
            "- `mined_labels.csv`: capped truth-filtered labels for kept pairs only, for evidence inspection.",
            "- `sampled_pairs.csv`: fixed-seed train pairs used.",
            "- `skipped_teachers.csv`: unavailable teacher profiles.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    configs = all_teacher_configs()
    assert any(item.profile == "global_r088_ht3" for item in configs)
    assert min_labels_for_gate("viewpoint") == 20
    assert min_labels_for_gate("compound") == 8
    assert DEFAULT_SPLIT_ROOT.exists()
    assert AGENT13_RUN.joinpath("metrics.csv").exists()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--pairs-per-group", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--learned-max-keypoints", type=int, default=512)
    parser.add_argument("--max-raw-matches", type=int, default=256)
    parser.add_argument("--max-labels-per-pair", type=int, default=128)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--truth-threshold-px", type=float, default=5.0)
    parser.add_argument("--min-truth-precision", type=float, default=0.95)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit-groups", nargs="*", help="Optional style/gate filters like numeric/viewpoint.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = all_teacher_configs()
    factory = MatcherFactory(args)
    groups = [("numeric", "viewpoint"), ("numeric", "compound"), ("timestamp", "viewpoint"), ("timestamp", "compound")]
    if args.limit_groups:
        keep = set(args.limit_groups)
        groups = [group for group in groups if f"{group[0]}/{group[1]}" in keep]

    pair_rows: list[PairRow] = []
    label_rows: list[LabelRow] = []
    sampled_rows: list[dict[str, object]] = []
    for style, gate in groups:
        paths = discover_pairs(args.split_root, style, gate, limit=args.pairs_per_group, seed=args.seed + len(sampled_rows))
        print(f"group={style}/{gate} sampled={len(paths)}", flush=True)
        for index, pair_path in enumerate(paths, start=1):
            sampled_rows.append({"style": style, "gate": gate, "sample_index": index, "pair_pt": pair_path.as_posix()})
        for config in configs:
            if not factory.applies(config, style, gate):
                continue
            matcher = factory.matcher(config)
            if matcher is None:
                continue
            for index, pair_path in enumerate(paths, start=1):
                row, labels = evaluate_pair(args, config, matcher, style=style, gate=gate, sample_index=index, pair_path=pair_path)
                pair_rows.append(row)
                label_rows.extend(labels)
            print(f"{style}/{gate} {config.profile} done", flush=True)

    summary_rows = aggregate(pair_rows)
    write_csv(args.output_dir / "pair_metrics.csv", [asdict(row) for row in pair_rows], PAIR_FIELDS)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "mined_labels.csv", [asdict(row) for row in label_rows], LABEL_FIELDS)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled_rows, SAMPLED_FIELDS)
    write_csv(args.output_dir / "skipped_teachers.csv", factory.skipped, SKIPPED_FIELDS)
    write_summary(args, pair_rows, summary_rows, sampled_rows, factory.skipped)
    print(f"output_dir={args.output_dir}")
    print(f"pair_metrics={args.output_dir / 'pair_metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
