#!/usr/bin/env python3
"""Agent6 pair-level RootSIFT mining report for cleaner pseudo-label candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import pseudo_label_generation as plg


DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "runs" / "cross_view_1024_hard_mined_weakgates_80_seed1234" / "splits" / "train"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent6"
DEFAULT_PREVIOUS_SUMMARY = PROJECT_ROOT / "runs" / "rootsift_pseudo_labels_weakgroups_seed1234" / "pseudo_label_summary.csv"

CANDIDATE_FIELDS = [
    "pair_pt",
    "labels",
    "mean_error_px",
    "style",
    "gate",
    "ratio",
    "ransac_threshold_px",
    "recommended",
    "precision",
    "ransac_inliers",
    "raw_matches",
    "status",
]

PAIR_FIELDS = [
    "style",
    "gate",
    "pair_pt",
    "ratio",
    "ransac_threshold_px",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "ransac_inliers",
    "labels",
    "wrong",
    "precision",
    "mean_error_px",
    "median_error_px",
    "p90_error_px",
    "p95_error_px",
    "max_error_px",
    "message",
]

GROUP_FIELDS = [
    "style",
    "gate",
    "ratio",
    "ransac_threshold_px",
    "sampled_pairs",
    "ok_pairs",
    "kept_pairs",
    "recommended_pairs",
    "labels",
    "ransac_inliers",
    "wrong",
    "precision",
    "mean_labels_per_ok_pair",
    "median_labels_per_ok_pair",
    "mean_ransac_inliers_per_ok_pair",
    "median_ransac_inliers_per_ok_pair",
    "mean_error_px",
    "median_error_px",
    "p90_error_px",
    "p95_error_px",
    "pairs_ge_8_labels",
    "pairs_ge_20_labels",
    "pairs_ge_50_labels",
    "recommended_min_inliers",
    "recommended_limit",
]


@dataclass(frozen=True)
class PairReport:
    style: str
    gate: str
    pair_pt: str
    ratio: float
    ransac_threshold_px: float
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    ransac_inliers: int
    labels: int
    wrong: int
    precision: float
    mean_error_px: float
    median_error_px: float
    p90_error_px: float
    p95_error_px: float
    max_error_px: float
    message: str = ""


def discover_group_pairs(split_root: Path, style: str, gate: str) -> list[Path]:
    return sorted((split_root / style / gate).glob("source_*/pair_*.pt"))


def sample_pairs(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    rng = random.Random(seed)
    selected = rng.sample(paths, k=limit)
    selected.sort()
    return selected


def finite_stats(errors: np.ndarray) -> tuple[float, float, float, float, float]:
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    return (
        float(finite.mean()),
        float(np.median(finite)),
        float(np.percentile(finite, 90)),
        float(np.percentile(finite, 95)),
        float(finite.max()),
    )


def truth_errors(points_a: np.ndarray, points_b: np.ndarray, warp_a_to_b, valid_mask) -> np.ndarray:
    if points_a.size == 0:
        return np.empty((0,), dtype=np.float32)
    target_b = plg.sample_warp(warp_a_to_b, points_a)
    valid = plg.valid_at_points(valid_mask, points_a)
    errors = np.linalg.norm(target_b - points_b, axis=1).astype(np.float32, copy=False)
    return np.where(np.isfinite(errors) & valid, errors, np.inf).astype(np.float32, copy=False)


def evaluate_pair(
    *,
    style: str,
    gate: str,
    pair_path: Path,
    ratio: float,
    ransac_threshold_px: float,
    truth_threshold_px: float,
    max_keypoints: int,
    max_raw_matches: int,
    sift_contrast: float,
) -> PairReport:
    try:
        image_a, image_b, warp_a_to_b, valid_mask = plg.load_pair(pair_path)
        raw = plg.rootsift_flann_ratio_match(
            image_a,
            image_b,
            max_keypoints=max_keypoints,
            max_matches=max_raw_matches,
            ratio=ratio,
            sift_contrast=sift_contrast,
        )
        ransac_a, ransac_b = plg.homography_inliers(raw.points_a, raw.points_b, threshold_px=ransac_threshold_px)
        errors = truth_errors(ransac_a, ransac_b, warp_a_to_b, valid_mask)
        labels = int(np.count_nonzero(errors <= truth_threshold_px))
        ransac_inliers = int(errors.shape[0])
        wrong = ransac_inliers - labels
        precision = labels / ransac_inliers if ransac_inliers else 0.0
        correct_errors = errors[errors <= truth_threshold_px]
        mean_error, median_error, p90_error, p95_error, max_error = finite_stats(correct_errors)
        status = "ok" if labels > 0 else "no_truth_labels"
        return PairReport(
            style=style,
            gate=gate,
            pair_pt=pair_path.as_posix(),
            ratio=ratio,
            ransac_threshold_px=ransac_threshold_px,
            status=status,
            keypoints_a=raw.keypoints_a,
            keypoints_b=raw.keypoints_b,
            raw_matches=int(raw.points_a.shape[0]),
            ransac_inliers=ransac_inliers,
            labels=labels,
            wrong=wrong,
            precision=precision,
            mean_error_px=mean_error,
            median_error_px=median_error,
            p90_error_px=p90_error,
            p95_error_px=p95_error,
            max_error_px=max_error,
        )
    except Exception as exc:
        return PairReport(
            style=style,
            gate=gate,
            pair_pt=pair_path.as_posix(),
            ratio=ratio,
            ransac_threshold_px=ransac_threshold_px,
            status="error",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            ransac_inliers=0,
            labels=0,
            wrong=0,
            precision=0.0,
            mean_error_px=math.nan,
            median_error_px=math.nan,
            p90_error_px=math.nan,
            p95_error_px=math.nan,
            max_error_px=math.nan,
            message=f"{type(exc).__name__}: {exc}",
        )


def recommendation_threshold(gate: str) -> tuple[int, float]:
    if gate == "compound":
        return 8, 0.98
    return 20, 0.98


def is_recommended(row: PairReport) -> bool:
    min_labels, min_precision = recommendation_threshold(row.gate)
    return row.status == "ok" and row.labels >= min_labels and row.precision >= min_precision


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


def aggregate_group(rows: list[PairReport]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float, float], list[PairReport]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.ratio, row.ransac_threshold_px), []).append(row)

    summaries: list[dict[str, object]] = []
    for (style, gate, ratio, threshold), subset in sorted(grouped.items()):
        ok = [row for row in subset if row.status != "error"]
        kept = [row for row in ok if row.labels > 0]
        recommended = [row for row in ok if is_recommended(row)]
        labels = sum(row.labels for row in ok)
        ransac_inliers = sum(row.ransac_inliers for row in ok)
        wrong = sum(row.wrong for row in ok)
        label_counts = np.array([row.labels for row in ok], dtype=np.float64)
        inlier_counts = np.array([row.ransac_inliers for row in ok], dtype=np.float64)
        mean_errors = np.array([row.mean_error_px for row in ok if not math.isnan(row.mean_error_px)], dtype=np.float64)
        min_inliers, limit = recommend_group_policy(gate, threshold, ok)
        summaries.append(
            {
                "style": style,
                "gate": gate,
                "ratio": ratio,
                "ransac_threshold_px": threshold,
                "sampled_pairs": len(subset),
                "ok_pairs": len(ok),
                "kept_pairs": len(kept),
                "recommended_pairs": len(recommended),
                "labels": labels,
                "ransac_inliers": ransac_inliers,
                "wrong": wrong,
                "precision": labels / ransac_inliers if ransac_inliers else 0.0,
                "mean_labels_per_ok_pair": float(label_counts.mean()) if label_counts.size else 0.0,
                "median_labels_per_ok_pair": float(np.median(label_counts)) if label_counts.size else 0.0,
                "mean_ransac_inliers_per_ok_pair": float(inlier_counts.mean()) if inlier_counts.size else 0.0,
                "median_ransac_inliers_per_ok_pair": float(np.median(inlier_counts)) if inlier_counts.size else 0.0,
                "mean_error_px": float(mean_errors.mean()) if mean_errors.size else math.nan,
                "median_error_px": float(np.median(mean_errors)) if mean_errors.size else math.nan,
                "p90_error_px": float(np.percentile(mean_errors, 90)) if mean_errors.size else math.nan,
                "p95_error_px": float(np.percentile(mean_errors, 95)) if mean_errors.size else math.nan,
                "pairs_ge_8_labels": sum(row.labels >= 8 for row in ok),
                "pairs_ge_20_labels": sum(row.labels >= 20 for row in ok),
                "pairs_ge_50_labels": sum(row.labels >= 50 for row in ok),
                "recommended_min_inliers": min_inliers,
                "recommended_limit": limit,
            }
        )
    return summaries


def recommend_group_policy(gate: str, threshold: float, rows: list[PairReport]) -> tuple[int, int]:
    if not rows:
        return 20, 0
    if gate == "compound":
        min_inliers = 8 if threshold <= 2.0 else 12
    else:
        min_inliers = 20
    clean = [row for row in rows if row.labels >= min_inliers and row.precision >= 0.98]
    limit = max(8, len(clean)) if clean else 0
    return min_inliers, limit


def candidate_rows(rows: list[PairReport]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in rows:
        candidates.append(
            {
                "pair_pt": row.pair_pt,
                "labels": row.labels,
                "mean_error_px": row.mean_error_px,
                "style": row.style,
                "gate": row.gate,
                "ratio": row.ratio,
                "ransac_threshold_px": row.ransac_threshold_px,
                "recommended": is_recommended(row),
                "precision": row.precision,
                "ransac_inliers": row.ransac_inliers,
                "raw_matches": row.raw_matches,
                "status": row.status,
            }
        )
    return sorted(candidates, key=lambda item: (item["style"], item["gate"], -int(item["recommended"]), -int(item["labels"]), item["pair_pt"]))


def read_previous_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def previous_comparison(path: Path, rows: list[PairReport]) -> list[dict[str, object]]:
    previous = read_previous_summary(path)
    previous_pairs = {row["pair_pt"]: row for row in previous}
    current_pairs = {row.pair_pt for row in rows if is_recommended(row)}
    previous_ok = {row["pair_pt"] for row in previous if row.get("status") == "ok"}
    return [
        {"metric": "previous_rows", "value": len(previous)},
        {"metric": "previous_ok_pairs", "value": len(previous_ok)},
        {"metric": "current_recommended_pairs", "value": len(current_pairs)},
        {"metric": "recommended_also_previous_ok", "value": len(current_pairs & previous_ok)},
        {"metric": "recommended_not_in_previous_sample", "value": len(current_pairs - set(previous_pairs))},
        {"metric": "previous_ok_not_recommended_or_not_sampled", "value": len(previous_ok - current_pairs)},
    ]


def write_readme(
    args: argparse.Namespace,
    group_rows: list[dict[str, object]],
    comparison: list[dict[str, object]],
    sampled: list[dict[str, object]],
) -> None:
    best = [
        row
        for row in group_rows
        if float(row["ratio"]) == args.ratio and float(row["ransac_threshold_px"]) in set(args.ransac_thresholds)
    ]
    lines = [
        "# Matcher Algorithm Iteration Agent6",
        "",
        "## Scope",
        "",
        "- Purpose: pair-level mining report for cleaner RootSIFT-HRANSAC pseudo-label candidates.",
        f"- Split root: `{args.split_root}`",
        f"- Output dir: `{args.output_dir}`",
        f"- Groups: `{','.join(args.styles)}` x `{','.join(args.gates)}`",
        f"- Sampled pairs per group: `{args.pairs_per_group}`",
        f"- Ratio: `{args.ratio:.2f}`",
        f"- RANSAC thresholds: `{','.join(str(item) for item in args.ransac_thresholds)}` px",
        f"- Warp-truth threshold: `{args.truth_threshold_px}` px",
        "",
        "## Group Recommendations",
        "",
        "| style | gate | ransac | sampled | kept | recommended | labels | precision | mean error | min_inliers | limit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['style']} | {row['gate']} | {float(row['ransac_threshold_px']):.0f} | "
            f"{row['sampled_pairs']} | {row['kept_pairs']} | {row['recommended_pairs']} | {row['labels']} | "
            f"{float(row['precision']):.4f} | {format_value(row['mean_error_px'])} | "
            f"{row['recommended_min_inliers']} | {row['recommended_limit']} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Use `ratio=0.80` as the primary mining ratio.",
            "- Prefer `RANSAC=2px` when precision is the priority; use `3px` only for expansion if a group needs more pairs after filtering.",
            "- Default pair filters: `precision >= 0.98`; `labels >= 20` for viewpoint, `labels >= 8` for compound.",
            "- Treat `recommended=true` rows in `recommended_pair_candidates.csv` as the clean pair-level candidate list; the existing generator remains responsible for point-level pseudo labels.",
            "",
            "## Previous Summary Comparison",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
    )
    for row in comparison:
        lines.append(f"| {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `sampled_pairs.csv`",
            "- `pair_mining_report.csv`",
            "- `group_summary.csv`",
            "- `recommended_pair_candidates.csv`",
            "- `previous_summary_comparison.csv`",
            "- `config.json`",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(args.output_dir / "sampled_pairs.csv", sampled, ["style", "gate", "pair_pt"])


def self_test() -> None:
    assert recommendation_threshold("viewpoint") == (20, 0.98)
    assert recommendation_threshold("compound") == (8, 0.98)
    rows = [
        PairReport("numeric", "viewpoint", "a.pt", 0.8, 2.0, "ok", 1, 1, 30, 30, 30, 0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5),
        PairReport("numeric", "viewpoint", "b.pt", 0.8, 2.0, "ok", 1, 1, 30, 30, 19, 11, 19 / 30, 0.5, 0.5, 0.5, 0.5, 0.5),
        PairReport("numeric", "compound", "c.pt", 0.8, 2.0, "ok", 1, 1, 10, 10, 8, 0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5),
    ]
    assert is_recommended(rows[0])
    assert not is_recommended(rows[1])
    assert is_recommended(rows[2])
    grouped = aggregate_group(rows)
    assert len(grouped) == 2
    assert sum(int(row["recommended_pairs"]) for row in grouped) == 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--previous-summary", type=Path, default=DEFAULT_PREVIOUS_SUMMARY)
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["viewpoint", "compound"], choices=["viewpoint", "compound"])
    parser.add_argument("--pairs-per-group", type=int, default=32)
    parser.add_argument("--ratio", type=float, default=0.80)
    parser.add_argument("--ransac-thresholds", nargs="+", type=float, default=[2.0, 3.0])
    parser.add_argument("--truth-threshold-px", type=float, default=3.0)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--max-raw-matches", type=int, default=512)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[PairReport] = []
    sampled_rows: list[dict[str, object]] = []
    for style in args.styles:
        for gate in args.gates:
            paths = sample_pairs(discover_group_pairs(args.split_root, style, gate), args.pairs_per_group, args.seed + len(sampled_rows))
            if len(paths) < args.pairs_per_group:
                print(f"warning group={style}/{gate} only found {len(paths)} pairs", flush=True)
            sampled_rows.extend({"style": style, "gate": gate, "pair_pt": path.as_posix()} for path in paths)
            for threshold in args.ransac_thresholds:
                for index, pair_path in enumerate(paths, start=1):
                    row = evaluate_pair(
                        style=style,
                        gate=gate,
                        pair_path=pair_path,
                        ratio=args.ratio,
                        ransac_threshold_px=threshold,
                        truth_threshold_px=args.truth_threshold_px,
                        max_keypoints=args.max_keypoints,
                        max_raw_matches=args.max_raw_matches,
                        sift_contrast=args.sift_contrast,
                    )
                    all_rows.append(row)
                    print(
                        f"{style:9s} {gate:9s} t={threshold:.0f} {index:02d}/{len(paths):02d} "
                        f"labels={row.labels:4d} inliers={row.ransac_inliers:4d} precision={row.precision:.4f} status={row.status}",
                        flush=True,
                    )

    pair_dicts = [asdict(row) for row in all_rows]
    group_rows = aggregate_group(all_rows)
    candidates = candidate_rows(all_rows)
    comparison = previous_comparison(args.previous_summary, all_rows)
    write_csv(args.output_dir / "pair_mining_report.csv", pair_dicts, PAIR_FIELDS)
    write_csv(args.output_dir / "group_summary.csv", group_rows, GROUP_FIELDS)
    write_csv(args.output_dir / "recommended_pair_candidates.csv", candidates, CANDIDATE_FIELDS)
    write_csv(args.output_dir / "previous_summary_comparison.csv", comparison, ["metric", "value"])
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    write_readme(args, group_rows, comparison, sampled_rows)
    print(f"output_dir={args.output_dir}")
    print(f"pair_report={args.output_dir / 'pair_mining_report.csv'}")
    print(f"group_summary={args.output_dir / 'group_summary.csv'}")
    print(f"candidates={args.output_dir / 'recommended_pair_candidates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
