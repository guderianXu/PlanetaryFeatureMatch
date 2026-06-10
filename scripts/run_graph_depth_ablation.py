#!/usr/bin/env python3
"""Run lazy visual reports with different GraphMatcher inference depths."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL_SCRIPT = PROJECT_ROOT / "scripts" / "visualize_lazy_pose_matches.py"


@dataclass(frozen=True)
class DepthSummary:
    depth: int
    report_dir: Path
    rows: int
    matches: int
    correct: int
    wrong: int
    precision: float
    median_error_px: float


def parse_depths(value: str) -> list[int]:
    depths: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            depth = int(item)
        except ValueError as exc:
            raise ValueError(f"invalid graph depth: {item}") from exc
        if depth <= 0:
            raise ValueError("graph depths must be positive")
        depths.append(depth)
    if not depths:
        raise ValueError("at least one graph depth is required")
    return list(dict.fromkeys(depths))


def _add_repeated(command: list[str], option: str, values: list[str]) -> None:
    for value in values:
        command.extend([option, value])


def build_visual_command(args: argparse.Namespace, *, depth: int, report_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(VISUAL_SCRIPT),
        "--render-manifest",
        str(args.render_manifest),
        "--uint8-manifest",
        str(args.uint8_manifest),
        "--pytorch-state",
        str(args.pytorch_state),
        "--output-dir",
        str(report_dir),
        "--split",
        args.split,
        "--reference-variant",
        args.reference_variant,
        "--pair-mode",
        args.pair_mode,
        "--image-source",
        args.image_source,
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
        "--device",
        args.device,
        "--descriptor-mode",
        args.descriptor_mode,
        "--keypoint-score-mode",
        args.keypoint_score_mode,
        "--matcher-mode",
        "graph_matcher",
        "--max-keypoints",
        str(args.max_keypoints),
        "--max-matches",
        str(args.max_matches),
        "--draw-matches",
        str(args.draw_matches),
        "--threshold-px",
        str(args.threshold_px),
        "--graph-max-attention-layers",
        str(depth),
        "--graph-max-attention-work-fraction",
        str(args.graph_max_attention_work_fraction),
        "--graph-width-prune-keep-ratio",
        str(args.graph_width_prune_keep_ratio),
        "--graph-width-prune-min-score",
        str(args.graph_width_prune_min_score),
        "--graph-early-stop-min-confidence",
        str(args.graph_early_stop_min_confidence),
        "--filtered-geometry-filter",
        args.filtered_geometry_filter,
        "--filtered-min-margin",
        str(args.filtered_min_margin),
        "--filtered-min-score",
        str(args.filtered_min_score),
        "--filtered-max-matches",
        str(args.filtered_max_matches),
        "--filtered-draw-matches",
        str(args.filtered_draw_matches),
    ]
    if args.run_dir is not None:
        command.extend(["--run-dir", str(args.run_dir)])
    if args.metrics_csv is not None:
        command.extend(["--metrics-csv", str(args.metrics_csv)])
    if args.pair_spec_manifest is not None:
        command.extend(["--pair-spec-manifest", str(args.pair_spec_manifest)])
    _add_repeated(command, "--target-variant", args.target_variant)
    _add_repeated(command, "--cross-pair-variant", args.cross_pair_variant)
    command.extend(["--cross-camera-offsets", args.cross_camera_offsets])
    command.extend(["--cross-fov-offsets", args.cross_fov_offsets])
    command.extend(["--pair-type-weights", args.pair_type_weights])
    if args.spatial_index_height_km:
        command.extend(["--spatial-index-height-km", args.spatial_index_height_km])
    command.extend(["--spatial-index-planet-radius-m", str(args.spatial_index_planet_radius_m)])
    command.extend(["--spatial-index-footprint-samples", str(args.spatial_index_footprint_samples)])
    command.extend(["--spatial-index-margin-m", str(args.spatial_index_margin_m)])
    command.append("--shuffle" if args.shuffle else "--no-shuffle")
    command.append("--filtered-report" if args.filtered_report else "--no-filtered-report")
    command.append("--filtered-mutual" if args.filtered_mutual else "--no-filtered-mutual")
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


def summarize_visual_csv(path: Path, *, depth: int, report_dir: Path) -> DepthSummary:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    matches = sum(int(float(row.get("matches", 0) or 0)) for row in rows)
    correct = sum(int(float(row.get("correct", 0) or 0)) for row in rows)
    wrong = sum(int(float(row.get("wrong", 0) or 0)) for row in rows)
    precision = 0.0 if matches <= 0 else float(correct) / float(matches)
    median_values = [float(row["median_error_px"]) for row in rows if row.get("median_error_px")]
    median_error = statistics.median(median_values) if median_values else float("nan")
    return DepthSummary(
        depth=depth,
        report_dir=report_dir,
        rows=len(rows),
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        median_error_px=median_error,
    )


def write_summary_csv(summaries: list[DepthSummary], path: Path) -> None:
    fields = ["depth", "report_dir", "rows", "matches", "correct", "wrong", "precision", "median_error_px"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "depth": item.depth,
                    "report_dir": str(item.report_dir),
                    "rows": item.rows,
                    "matches": item.matches,
                    "correct": item.correct,
                    "wrong": item.wrong,
                    "precision": f"{item.precision:.6f}",
                    "median_error_px": f"{item.median_error_px:.3f}",
                }
            )


def write_html_report(args: argparse.Namespace, summaries: list[DepthSummary], path: Path) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{item.depth}</td>"
        f"<td>{item.rows}</td>"
        f"<td>{item.matches}</td>"
        f"<td>{item.correct}</td>"
        f"<td>{item.wrong}</td>"
        f"<td>{item.precision:.3f}</td>"
        f"<td>{item.median_error_px:.2f}</td>"
        f"<td><a href=\"{html.escape(item.report_dir.name)}/index.html\">report</a></td>"
        "</tr>"
        for item in summaries
    )
    metadata = {
        "pytorch_state": str(args.pytorch_state),
        "render_manifest": str(args.render_manifest),
        "uint8_manifest": str(args.uint8_manifest),
        "depths": [item.depth for item in summaries],
    }
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Graph Depth Ablation</title>
<style>
body {{ margin: 24px; font-family: Arial, "Noto Sans CJK SC", sans-serif; background: #091018; color: #e5eef7; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
th, td {{ border-bottom: 1px solid #293748; padding: 8px; text-align: left; }}
a {{ color: #7dd3fc; }}
pre {{ background: #101a24; padding: 12px; white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>Graph Depth Ablation</h1>
<pre>{html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre>
<table>
<thead>
<tr><th>Depth</th><th>Rows</th><th>Matches</th><th>Correct</th><th>Wrong</th><th>Precision</th><th>Median error</th><th>Report</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--uint8-manifest", type=Path, required=True)
    parser.add_argument("--pytorch-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depths", type=parse_depths, default=parse_depths("1,2,4,6,8"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--reference-variant", default="nadir")
    parser.add_argument("--target-variant", action="append", default=[])
    parser.add_argument("--pair-spec-manifest", type=Path, default=None)
    parser.add_argument("--pair-mode", choices=["same-position", "cross-camera", "cross-fov", "mixed", "spatial-index"], default="same-position")
    parser.add_argument("--cross-camera-offsets", default="1,2,4,8")
    parser.add_argument("--cross-fov-offsets", default="0,1,2,4")
    parser.add_argument("--cross-pair-variant", action="append", default=[])
    parser.add_argument("--pair-type-weights", default="same_position_view=0.4,cross_camera=0.35,cross_fov=0.25")
    parser.add_argument("--spatial-index-planet-radius-m", type=float, default=3_396_190.0)
    parser.add_argument("--spatial-index-footprint-samples", type=int, default=5)
    parser.add_argument("--spatial-index-margin-m", type=float, default=2000.0)
    parser.add_argument("--spatial-index-height-km", default="")
    parser.add_argument("--image-source", choices=["uint8", "render"], default="uint8")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-pairs", type=int, default=24)
    parser.add_argument("--select-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--crop-size", type=int, default=2048)
    parser.add_argument("--max-image-size", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--descriptor-mode", choices=["learned", "texture", "blend"], default="learned")
    parser.add_argument("--keypoint-score-mode", choices=["texture", "learned"], default="learned")
    parser.add_argument("--max-keypoints", type=int, default=512)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--draw-matches", type=int, default=0)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--graph-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--graph-width-prune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--graph-width-prune-min-score", type=float, default=-1.0)
    parser.add_argument("--graph-early-stop-min-confidence", type=float, default=-1.0)
    parser.add_argument("--filtered-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filtered-mutual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filtered-geometry-filter", choices=["none", "affine", "local"], default="local")
    parser.add_argument("--filtered-max-matches", type=int, default=0)
    parser.add_argument("--filtered-draw-matches", type=int, default=0)
    parser.add_argument("--filtered-min-score", type=float, default=-1.0)
    parser.add_argument("--filtered-min-margin", type=float, default=0.02)
    parser.add_argument("--input-local-contrast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-local-contrast-strength", type=float, default=0.0)
    parser.add_argument("--input-local-contrast-kernel", type=int, default=31)
    parser.add_argument("--illumination-stress", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[DepthSummary] = []
    for depth in args.depths:
        report_dir = args.output_dir / f"layers_{depth}"
        command = build_visual_command(args, depth=depth, report_dir=report_dir)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        summaries.append(summarize_visual_csv(report_dir / "summary.csv", depth=depth, report_dir=report_dir))
    write_summary_csv(summaries, args.output_dir / "depth_ablation_summary.csv")
    write_html_report(args, summaries, args.output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
