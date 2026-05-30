#!/usr/bin/env python3
"""Agent10 larger-sample classical fallback validation for pseudo-label expansion."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT9_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent9.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent10"
DEFAULT_PFM_RUN = PROJECT_ROOT / "runs" / "cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234"

BASELINE_CONFIG = "RootSIFT-raw-r0.80-Ht2"
TARGET_CONFIG = "RootSIFT-raw-r0.90-Ht2"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A9 = load_module(AGENT9_SCRIPT, "agent9_matcher_for_agent10")
A4 = A9.A4

MetricRow = A9.MetricRow
MatcherConfig = A9.MatcherConfig


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


def make_configs() -> list[MatcherConfig]:
    return [
        MatcherConfig(BASELINE_CONFIG, "RootSIFT", "raw", "ratio", 0.80, 2.0),
        MatcherConfig(TARGET_CONFIG, "RootSIFT", "raw", "ratio", 0.90, 2.0),
        MatcherConfig("RootSIFT-clahe-r0.90-Ht2", "RootSIFT", "clahe", "ratio", 0.90, 2.0),
        MatcherConfig("RootSIFT-raw-r0.90-Ht3", "RootSIFT", "raw", "ratio", 0.90, 3.0),
    ]


def cv2_unavailable() -> list[dict[str, str]]:
    try:
        import cv2

        if not hasattr(cv2, "SIFT_create"):
            return [{"algorithm": "RootSIFT/OpenCV-SIFT", "reason": "cv2.SIFT_create unavailable"}]
        return []
    except Exception as exc:
        return [{"algorithm": "OpenCV", "reason": f"{type(exc).__name__}: {exc}"}]


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    configs = make_configs()
    unavailable = cv2_unavailable()
    if unavailable:
        return [], [], [], unavailable

    baseline_cfg = configs[0]
    candidate_configs = configs[1:]
    rows: list[MetricRow] = []
    recovery_rows: list[dict[str, object]] = []
    sampled: list[dict[str, object]] = []
    vis_budget: dict[str, int] = {}

    for style in args.styles:
        for gate in args.gates:
            pair_paths = A4.select_pairs(args, style, gate)
            sampled.extend({"style": style, "gate": gate, "pair_pt": path.as_posix()} for path in pair_paths)
            print(
                f"group={style}/{gate} pairs={len(pair_paths)} rotations={len(args.rotations)} configs={len(configs)}",
                flush=True,
            )
            for rotation_deg in args.rotations:
                for pair_index, pair_path in enumerate(pair_paths, start=1):
                    baseline_row = A9.evaluate_config(
                        args=args,
                        cfg=baseline_cfg,
                        style=style,
                        gate=gate,
                        rotation_deg=rotation_deg,
                        pair_path=pair_path,
                        base_pass_gate=1,
                        vis_budget=vis_budget,
                    )
                    baseline_row = MetricRow(
                        **{
                            **asdict(baseline_row),
                            "base_pass_gate": baseline_row.pass_gate,
                            "fallback_candidate": 0,
                            "recovered_baseline_fail": 0,
                        }
                    )
                    rows.append(baseline_row)

                    for cfg in candidate_configs:
                        row = A9.evaluate_config(
                            args=args,
                            cfg=cfg,
                            style=style,
                            gate=gate,
                            rotation_deg=rotation_deg,
                            pair_path=pair_path,
                            base_pass_gate=baseline_row.pass_gate,
                            vis_budget=vis_budget,
                        )
                        rows.append(row)
                        if baseline_row.status == "ok" and not baseline_row.pass_gate:
                            recovery_rows.append(
                                {
                                    "case_type": "baseline_r080t2_failed_gate",
                                    "style": style,
                                    "gate": gate,
                                    "rotation_deg": rotation_deg,
                                    "pair_pt": pair_path.as_posix(),
                                    "config": cfg.config,
                                    "detector": cfg.detector,
                                    "preprocess": cfg.preprocess,
                                    "match_mode": cfg.match_mode,
                                    "ratio": cfg.ratio,
                                    "ransac_threshold_px": cfg.ransac_threshold_px,
                                    "min_inliers": A9.min_gate_labels(gate),
                                    "baseline_matches": baseline_row.matches,
                                    "baseline_correct": baseline_row.correct,
                                    "baseline_precision": baseline_row.precision,
                                    "baseline_pass_gate": baseline_row.pass_gate,
                                    "fallback_matches": row.matches,
                                    "fallback_correct": row.correct,
                                    "fallback_wrong": row.wrong,
                                    "fallback_precision": row.precision,
                                    "fallback_pass_gate": row.pass_gate,
                                    "recovered": row.recovered_baseline_fail,
                                    "mean_error_px": row.mean_error_px,
                                    "median_error_px": row.median_error_px,
                                    "visualization": row.visualization,
                                    "message": row.message,
                                }
                            )
                    print(
                        f"{style:9s} {gate:9s} rot={rotation_deg:3d} {pair_index:02d}/{len(pair_paths):02d} "
                        f"base_pass={baseline_row.pass_gate} base_correct={baseline_row.correct}",
                        flush=True,
                    )
    return rows, recovery_rows, sampled, unavailable


def aggregate(rows: list[MetricRow]) -> list[dict[str, object]]:
    return A9.aggregate(rows)


def global_config_summary(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.config, []).append(row)
    out: list[dict[str, object]] = []
    for config, items in sorted(grouped.items()):
        ok = [row for row in items if row.status == "ok"]
        matches = sum(row.matches for row in ok)
        correct = sum(row.correct for row in ok)
        wrong = sum(row.wrong for row in ok)
        recovered = [row for row in ok if row.recovered_baseline_fail]
        fallback_matches = sum(row.matches for row in recovered)
        fallback_correct = sum(row.correct for row in recovered)
        fallback_wrong = sum(row.wrong for row in recovered)
        out.append(
            {
                "config": config,
                "pairs": len(items),
                "ok_pairs": len(ok),
                "pass_gate_pairs": sum(row.pass_gate for row in ok),
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": 0.0 if matches == 0 else correct / matches,
                "baseline_failed_pairs": sum(1 for row in ok if not row.base_pass_gate),
                "recovered_pairs": len(recovered),
                "fallback_matches": fallback_matches,
                "fallback_correct": fallback_correct,
                "fallback_wrong": fallback_wrong,
                "fallback_precision": 0.0 if fallback_matches == 0 else fallback_correct / fallback_matches,
            }
        )
    return out


def target_recovery_summary(recovery_rows: list[dict[str, object]], config: str = TARGET_CONFIG) -> dict[str, object]:
    rows = [row for row in recovery_rows if row["config"] == config]
    recovered = [row for row in rows if int(row["recovered"]) == 1]
    matches = sum(int(row["fallback_matches"]) for row in recovered)
    correct = sum(int(row["fallback_correct"]) for row in recovered)
    wrong = sum(int(row["fallback_wrong"]) for row in recovered)
    return {
        "config": config,
        "baseline_failed_pairs": len(rows),
        "recovery_count": len(recovered),
        "fallback_matches": matches,
        "fallback_correct": correct,
        "fallback_wrong": wrong,
        "fallback_precision": 0.0 if matches == 0 else correct / matches,
    }


def markdown_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| config | ok pairs | pass gate | all matches | all precision | recovered | fallback matches | fallback precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: str(item["config"])):
        lines.append(
            f"| {row['config']} | {row['ok_pairs']} | {row['pass_gate_pairs']} | {row['matches']} | "
            f"{float(row['precision']):.4f} | {row['recovered_pairs']} | {row['fallback_matches']} | "
            f"{float(row['fallback_precision']):.4f} |"
        )
    return lines


def write_summary(
    args: argparse.Namespace,
    rows: list[MetricRow],
    summary_rows: list[dict[str, object]],
    recovery_rows: list[dict[str, object]],
    unavailable: list[dict[str, str]],
) -> None:
    global_rows = global_config_summary(rows)
    target = target_recovery_summary(recovery_rows)
    base_failed = target["baseline_failed_pairs"]
    sample_pairs = len({(row.style, row.gate, row.pair_pt) for row in rows})
    rotated_evals = len({(row.style, row.gate, row.rotation_deg, row.pair_pt) for row in rows})
    target_all = next((row for row in global_rows if row["config"] == TARGET_CONFIG), None)
    safe = bool(
        target_all
        and float(target_all["precision"]) >= args.safe_precision
        and (int(target["fallback_matches"]) == 0 or float(target["fallback_precision"]) >= args.safe_precision)
    )

    lines = [
        "# Matcher Algorithm Iteration Agent10",
        "",
        "## Scope",
        "",
        "- Goal: larger-sample CPU-only validation of `RootSIFT raw r0.90 + Homography RANSAC 2px` as a pseudo-label expansion source.",
        "- Baseline is recomputed in this run as `RootSIFT raw r0.80 + Homography RANSAC 2px`.",
        f"- Sample target: `{args.pairs_per_group}` pairs per style/gate; styles: `{','.join(args.styles)}`; gates: `{','.join(args.gates)}`; rotations: `{','.join(str(item) for item in args.rotations)}`.",
        f"- Correctness threshold: `{args.threshold_px}` px; safe precision target: `{args.safe_precision}`.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} "
        f"--pairs-per-group {args.pairs_per_group} --visualizations-per-config {args.visualizations_per_config}",
        "```",
        "",
        "## Core Result",
        "",
        f"- Unique sampled source pairs: `{sample_pairs}`; rotated pair evaluations: `{rotated_evals}`.",
        f"- Baseline `{BASELINE_CONFIG}` failed gate on `{base_failed}` rotated pair evaluations.",
        f"- Target `{TARGET_CONFIG}` recovered `{target['recovery_count']}` of those failures.",
        f"- Target fallback insertion matches: `{target['fallback_correct']}/{target['fallback_matches']}` correct, precision `{float(target['fallback_precision']):.4f}`.",
    ]
    if target_all:
        lines.append(
            f"- Target all-sample precision/pass_gate/matches: `{float(target_all['precision']):.4f}` / "
            f"`{target_all['pass_gate_pairs']}/{target_all['ok_pairs']}` / `{target_all['matches']}`."
        )
    lines.append(
        "- Recommendation: "
        + (
            f"`{TARGET_CONFIG}` is supported as a pseudo-label expansion source under the tested gate."
            if safe
            else f"`{TARGET_CONFIG}` should not be used broadly without an additional guard under the tested gate."
        )
    )
    lines.extend(["", "## Global Config Metrics", "", *markdown_table(global_rows)])
    lines.extend(["", "## Recovery Metric Definition", ""])
    lines.append("- Recovery is counted only where the recomputed baseline `r0.80/t2` failed the style/gate label threshold and the candidate passed it.")
    lines.append("- Fallback precision is computed only over recovered candidate rows, matching the pseudo-label insertion scenario.")
    lines.extend(["", "## Unavailable / Non-blocking", ""])
    if unavailable:
        for item in unavailable:
            lines.append(f"- {item['algorithm']}: {item['reason']}")
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Outputs", ""])
    for name in ["metrics.csv", "summary_metrics.csv", "recovery_metrics.csv", "sampled_pairs.csv", "unavailable_algorithms.csv"]:
        lines.append(f"- `{name}`")
    if args.visualizations_per_config > 0:
        lines.append("- `visualizations/` for recovered baseline failures within the configured budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pfm-run", type=Path, default=DEFAULT_PFM_RUN)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["viewpoint", "compound"], choices=["viewpoint", "compound"])
    parser.add_argument("--rotations", nargs="+", type=int, default=[90, 180, 270], choices=[90, 180, 270])
    parser.add_argument("--pairs-per-group", type=int, default=16)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--safe-precision", type=float, default=0.95)
    parser.add_argument("--visualizations-per-config", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, recovery_rows, sampled, unavailable = evaluate(args)
    summary_rows = aggregate(rows)
    write_csv(args.output_dir / "metrics.csv", [asdict(row) for row in rows], A9.METRIC_FIELDS)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, A9.SUMMARY_FIELDS)
    write_csv(args.output_dir / "recovery_metrics.csv", recovery_rows, A9.RECOVERY_FIELDS)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled, ["style", "gate", "pair_pt"])
    write_csv(args.output_dir / "unavailable_algorithms.csv", unavailable, ["algorithm", "reason"])
    write_summary(args, rows, summary_rows, recovery_rows, unavailable)
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
