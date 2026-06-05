#!/usr/bin/env python3
"""Wait for a lazy-training checkpoint and run the lazy match visual report."""

from __future__ import annotations

import argparse
import csv
import html
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_last_step(metrics_csv: Path) -> int:
    if not metrics_csv.exists():
        return 0
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0
    try:
        return int(float(rows[-1].get("step", "0")))
    except ValueError:
        return 0


def checkpoint_is_stable(checkpoint_path: Path, *, stable_seconds: float) -> bool:
    if not checkpoint_path.exists():
        return False
    size_before = checkpoint_path.stat().st_size
    if size_before <= 1_000_000:
        return False
    time.sleep(max(0.0, float(stable_seconds)))
    return checkpoint_path.exists() and checkpoint_path.stat().st_size == size_before


def write_status_html(
    path: Path,
    *,
    title: str,
    status: str,
    detail: str,
    run_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 24px; background: #081017; color: #dcebf7; font-family: Arial, "Noto Sans CJK SC", sans-serif; }}
section {{ border: 1px solid #203546; border-radius: 8px; background: #111b24; padding: 16px; margin: 16px 0; }}
code, pre {{ background: #071018; border: 1px solid #1d2d3a; border-radius: 6px; }}
pre {{ padding: 12px; white-space: pre-wrap; overflow: auto; }}
.ok {{ color: #5eead4; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<section>
<h2>状态</h2>
<p class="ok">{html.escape(status)}</p>
<pre>{html.escape(detail)}</pre>
</section>
<section>
<h2>路径</h2>
<ul>
<li>训练目录：<code>{html.escape(str(run_dir))}</code></li>
<li>Checkpoint：<code>{html.escape(str(checkpoint_path))}</code></li>
<li>报告目录：<code>{html.escape(str(output_dir))}</code></li>
<li>报告首页：<code>{html.escape(str(output_dir / "index.html"))}</code></li>
</ul>
</section>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def build_visual_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "visualize_lazy_pose_matches.py"),
        "--render-manifest",
        str(args.render_manifest),
        "--uint8-manifest",
        str(args.uint8_manifest),
        "--pytorch-state",
        str(args.checkpoint),
        "--output-dir",
        str(args.output_dir),
        "--run-dir",
        str(args.run_dir),
        "--metrics-csv",
        str(args.metrics_csv),
        "--split",
        args.split,
        "--reference-variant",
        args.reference_variant,
        "--candidate-pairs",
        str(args.candidate_pairs),
        "--select-count",
        str(args.select_count),
        "--seed",
        str(args.seed),
        "--crop-size",
        str(args.crop_size),
        "--max-image-size",
        str(args.max_image_size),
        "--max-attempts",
        str(args.max_attempts),
        "--min-valid-fraction",
        str(args.min_valid_fraction),
        "--absolute-depth-tolerance-m",
        str(args.absolute_depth_tolerance_m),
        "--relative-depth-tolerance",
        str(args.relative_depth_tolerance),
        "--device",
        args.device,
        "--descriptor-mode",
        args.descriptor_mode,
        "--keypoint-score-mode",
        args.keypoint_score_mode,
        "--max-keypoints",
        str(args.max_keypoints),
        "--max-matches",
        str(args.max_matches),
        "--draw-matches",
        str(args.draw_matches),
        "--threshold-px",
        str(args.threshold_px),
        "--filtered-geometry-filter",
        args.filtered_geometry_filter,
        "--filtered-min-score",
        str(args.filtered_min_score),
        "--filtered-min-margin",
        str(args.filtered_min_margin),
        "--filtered-max-matches",
        str(args.filtered_max_matches),
        "--filtered-draw-matches",
        str(args.filtered_draw_matches),
    ]
    command.append("--filtered-report" if args.filtered_report else "--no-filtered-report")
    command.append("--illumination-stress" if args.illumination_stress else "--no-illumination-stress")
    if args.input_local_contrast:
        command.extend(
            [
                "--input-local-contrast",
                "--input-local-contrast-strength",
                str(args.input_local_contrast_strength),
                "--input-local-contrast-kernel",
                str(args.input_local_contrast_kernel),
            ]
        )
    return command


def wait_until_ready(args: argparse.Namespace) -> int:
    started = time.monotonic()
    while True:
        step = read_last_step(args.metrics_csv)
        if step >= args.target_step and checkpoint_is_stable(args.checkpoint, stable_seconds=args.stable_seconds):
            return step

        detail = (
            f"当前时间：{time.strftime('%F %T')}\n"
            f"当前 step：{step}\n"
            f"目标 step：{args.target_step}\n"
            f"checkpoint：{args.checkpoint}"
        )
        write_status_html(
            args.status_html,
            title=args.title,
            status="等待 checkpoint",
            detail=detail,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
        if args.timeout_seconds > 0 and time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError(f"timeout waiting for step {args.target_step}: last_step={step}")
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-html", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=3000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stable-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--title", default="Lazy 训练中间可视化 watcher")

    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--uint8-manifest", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--reference-variant", default="nadir")
    parser.add_argument("--candidate-pairs", type=int, default=24)
    parser.add_argument("--select-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--max-image-size", type=int, default=768)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--min-valid-fraction", type=float, default=0.10)
    parser.add_argument("--absolute-depth-tolerance-m", type=float, default=100.0)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--descriptor-mode", choices=["learned", "texture", "blend"], default="learned")
    parser.add_argument("--keypoint-score-mode", choices=["texture", "learned"], default="texture")
    parser.add_argument("--max-keypoints", type=int, default=384)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--draw-matches", type=int, default=0)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--filtered-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filtered-geometry-filter", choices=["none", "affine", "local"], default="local")
    parser.add_argument("--filtered-min-score", type=float, default=-1.0)
    parser.add_argument("--filtered-min-margin", type=float, default=0.02)
    parser.add_argument("--filtered-max-matches", type=int, default=0)
    parser.add_argument("--filtered-draw-matches", type=int, default=0)
    parser.add_argument("--illumination-stress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-local-contrast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-local-contrast-strength", type=float, default=0.0)
    parser.add_argument("--input-local-contrast-kernel", type=int, default=31)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_step <= 0:
        raise ValueError("--target-step must be positive")
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    if args.stable_seconds < 0.0:
        raise ValueError("--stable-seconds must be non-negative")

    step = wait_until_ready(args)
    write_status_html(
        args.status_html,
        title=args.title,
        status="开始生成可视化",
        detail=f"当前时间：{time.strftime('%F %T')}\n当前 step：{step}",
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    command = build_visual_command(args)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    write_status_html(
        args.status_html,
        title=args.title,
        status="可视化完成",
        detail=f"当前时间：{time.strftime('%F %T')}\n报告首页：{args.output_dir / 'index.html'}",
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
