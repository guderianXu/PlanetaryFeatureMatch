#!/usr/bin/env python3
"""Run lazy visual reports with different GraphMatcher filter thresholds."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL_SCRIPT = PROJECT_ROOT / "scripts" / "visualize_lazy_pose_matches.py"


@dataclass(frozen=True)
class GraphFilterConfig:
    min_score: float
    dustbin_delta: float
    acceptance_margin: float
    min_raw_score: float
    min_raw_margin: float
    min_accept_probability: float


@dataclass(frozen=True)
class FilterSweepSummary:
    config: GraphFilterConfig
    report_dir: Path
    raw_rows: int
    raw_matches: int
    raw_correct: int
    raw_wrong: int
    raw_precision: float
    raw_median_error_px: float
    filtered_rows: int
    filtered_matches: int
    filtered_correct: int
    filtered_wrong: int
    filtered_precision: float
    filtered_median_error_px: float


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise ValueError(f"invalid float value: {item}") from exc
    if not values:
        raise ValueError("at least one float value is required")
    return values


def _format_slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def slug_for_config(config: GraphFilterConfig) -> str:
    return (
        f"score{_format_slug_float(config.min_score)}"
        f"_dust{_format_slug_float(config.dustbin_delta)}"
        f"_accept{_format_slug_float(config.acceptance_margin)}"
        f"_raw{_format_slug_float(config.min_raw_score)}"
        f"_margin{_format_slug_float(config.min_raw_margin)}"
        f"_prob{_format_slug_float(config.min_accept_probability)}"
    )


def iter_sweep_configs(args: argparse.Namespace) -> list[GraphFilterConfig]:
    configs = [
        GraphFilterConfig(
            min_score=min_score,
            dustbin_delta=dustbin_delta,
            acceptance_margin=acceptance_margin,
            min_raw_score=min_raw_score,
            min_raw_margin=min_raw_margin,
            min_accept_probability=min_accept_probability,
        )
        for min_score, dustbin_delta, acceptance_margin, min_raw_score, min_raw_margin, min_accept_probability in product(
            args.min_score_values,
            args.graph_dustbin_delta_values,
            args.graph_acceptance_margin_values,
            args.graph_min_raw_score_values,
            args.graph_min_raw_margin_values,
            args.graph_min_accept_probability_values,
        )
    ]
    return list(dict.fromkeys(configs))


def validate_config(config: GraphFilterConfig) -> None:
    if config.min_score < -1.0:
        raise ValueError("min_score must be at least -1.0; -1 disables this filter")
    if config.acceptance_margin < 0.0:
        raise ValueError("acceptance_margin must be nonnegative")
    if config.min_raw_score < -1.0:
        raise ValueError("min_raw_score must be at least -1.0; -1 disables this filter")
    if config.min_raw_margin < 0.0:
        raise ValueError("min_raw_margin must be nonnegative")
    if config.min_accept_probability < -1.0 or config.min_accept_probability > 1.0:
        raise ValueError("min_accept_probability must be in [-1, 1]")


def _add_repeated(command: list[str], option: str, values: list[str]) -> None:
    for value in values:
        command.extend([option, value])


def build_visual_command(args: argparse.Namespace, *, config: GraphFilterConfig, report_dir: Path) -> list[str]:
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
        "--min-score",
        str(config.min_score),
        "--graph-dustbin-delta",
        str(config.dustbin_delta),
        "--graph-acceptance-margin",
        str(config.acceptance_margin),
        "--graph-min-raw-score",
        str(config.min_raw_score),
        "--graph-min-raw-margin",
        str(config.min_raw_margin),
        "--graph-min-accept-probability",
        str(config.min_accept_probability),
        "--graph-max-attention-layers",
        str(args.graph_max_attention_layers),
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
        str(config.min_score),
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


def _summarize_csv(path: Path, *, exclude_filtered_labels: bool) -> tuple[int, int, int, int, float, float]:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                label = row.get("label", "")
                if exclude_filtered_labels and "filtered" in label:
                    continue
                rows.append(row)
    matches = sum(int(float(row.get("matches", 0) or 0)) for row in rows)
    correct = sum(int(float(row.get("correct", 0) or 0)) for row in rows)
    wrong = sum(int(float(row.get("wrong", 0) or 0)) for row in rows)
    precision = 0.0 if matches <= 0 else float(correct) / float(matches)
    median_values = [float(row["median_error_px"]) for row in rows if row.get("median_error_px")]
    median_error = statistics.median(median_values) if median_values else float("nan")
    return len(rows), matches, correct, wrong, precision, median_error


def summarize_report(report_dir: Path, *, config: GraphFilterConfig) -> FilterSweepSummary:
    raw_rows, raw_matches, raw_correct, raw_wrong, raw_precision, raw_median_error = _summarize_csv(
        report_dir / "summary.csv",
        exclude_filtered_labels=True,
    )
    filtered_rows, filtered_matches, filtered_correct, filtered_wrong, filtered_precision, filtered_median_error = (
        _summarize_csv(report_dir / "filtered_summary.csv", exclude_filtered_labels=False)
    )
    return FilterSweepSummary(
        config=config,
        report_dir=report_dir,
        raw_rows=raw_rows,
        raw_matches=raw_matches,
        raw_correct=raw_correct,
        raw_wrong=raw_wrong,
        raw_precision=raw_precision,
        raw_median_error_px=raw_median_error,
        filtered_rows=filtered_rows,
        filtered_matches=filtered_matches,
        filtered_correct=filtered_correct,
        filtered_wrong=filtered_wrong,
        filtered_precision=filtered_precision,
        filtered_median_error_px=filtered_median_error,
    )


def write_summary_csv(summaries: list[FilterSweepSummary], path: Path) -> None:
    fields = [
        "min_score",
        "dustbin_delta",
        "acceptance_margin",
        "min_raw_score",
        "min_raw_margin",
        "min_accept_probability",
        "raw_rows",
        "raw_matches",
        "raw_correct",
        "raw_wrong",
        "raw_precision",
        "raw_median_error_px",
        "filtered_rows",
        "filtered_matches",
        "filtered_correct",
        "filtered_wrong",
        "filtered_precision",
        "filtered_median_error_px",
        "report_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            cfg = item.config
            writer.writerow(
                {
                    "min_score": f"{cfg.min_score:g}",
                    "dustbin_delta": f"{cfg.dustbin_delta:g}",
                    "acceptance_margin": f"{cfg.acceptance_margin:g}",
                    "min_raw_score": f"{cfg.min_raw_score:g}",
                    "min_raw_margin": f"{cfg.min_raw_margin:g}",
                    "min_accept_probability": f"{cfg.min_accept_probability:g}",
                    "raw_rows": item.raw_rows,
                    "raw_matches": item.raw_matches,
                    "raw_correct": item.raw_correct,
                    "raw_wrong": item.raw_wrong,
                    "raw_precision": f"{item.raw_precision:.6f}",
                    "raw_median_error_px": f"{item.raw_median_error_px:.3f}",
                    "filtered_rows": item.filtered_rows,
                    "filtered_matches": item.filtered_matches,
                    "filtered_correct": item.filtered_correct,
                    "filtered_wrong": item.filtered_wrong,
                    "filtered_precision": f"{item.filtered_precision:.6f}",
                    "filtered_median_error_px": f"{item.filtered_median_error_px:.3f}",
                    "report_dir": str(item.report_dir),
                }
            )


def write_html_report(args: argparse.Namespace, summaries: list[FilterSweepSummary], path: Path) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{item.config.min_score:g}</td>"
        f"<td>{item.config.dustbin_delta:g}</td>"
        f"<td>{item.config.acceptance_margin:g}</td>"
        f"<td>{item.config.min_raw_score:g}</td>"
        f"<td>{item.config.min_raw_margin:g}</td>"
        f"<td>{item.config.min_accept_probability:g}</td>"
        f"<td>{item.raw_matches}</td>"
        f"<td>{item.raw_correct}</td>"
        f"<td>{item.raw_precision:.3f}</td>"
        f"<td>{item.raw_median_error_px:.2f}</td>"
        f"<td>{item.filtered_matches}</td>"
        f"<td>{item.filtered_correct}</td>"
        f"<td>{item.filtered_precision:.3f}</td>"
        f"<td>{item.filtered_median_error_px:.2f}</td>"
        f"<td><a href=\"{html.escape(item.report_dir.name)}/index.html\">report</a></td>"
        "</tr>"
        for item in summaries
    )
    metadata = {
        "pytorch_state": str(args.pytorch_state),
        "render_manifest": str(args.render_manifest),
        "uint8_manifest": str(args.uint8_manifest),
        "pair_spec_manifest": str(args.pair_spec_manifest or ""),
        "config_count": len(summaries),
    }
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Graph Filter Sweep</title>
<style>
body {{ margin: 24px; font-family: Arial, "Noto Sans CJK SC", sans-serif; background: #091018; color: #e5eef7; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #293748; padding: 7px; text-align: left; }}
a {{ color: #7dd3fc; }}
pre {{ background: #101a24; padding: 12px; white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>Graph Filter Sweep</h1>
<pre>{html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre>
<table>
<thead>
<tr><th>min score</th><th>dustbin</th><th>accept margin</th><th>raw score</th><th>raw margin</th><th>accept prob</th><th>raw matches</th><th>raw correct</th><th>raw precision</th><th>raw median px</th><th>filtered matches</th><th>filtered correct</th><th>filtered precision</th><th>filtered median px</th><th>Report</th></tr>
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
    parser.add_argument("--select-count", type=int, default=24)
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
    parser.add_argument("--graph-max-attention-layers", type=int, default=0)
    parser.add_argument("--graph-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--graph-width-prune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--graph-width-prune-min-score", type=float, default=-1.0)
    parser.add_argument("--graph-early-stop-min-confidence", type=float, default=-1.0)
    parser.add_argument("--min-score-values", type=parse_float_list, default=parse_float_list("-1"))
    parser.add_argument("--graph-dustbin-delta-values", type=parse_float_list, default=parse_float_list("0"))
    parser.add_argument("--graph-acceptance-margin-values", type=parse_float_list, default=parse_float_list("0"))
    parser.add_argument("--graph-min-raw-score-values", type=parse_float_list, default=parse_float_list("-1"))
    parser.add_argument("--graph-min-raw-margin-values", type=parse_float_list, default=parse_float_list("0"))
    parser.add_argument("--graph-min-accept-probability-values", type=parse_float_list, default=parse_float_list("-1"))
    parser.add_argument("--max-configs", type=int, default=64)
    parser.add_argument("--filtered-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filtered-mutual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filtered-geometry-filter", choices=["none", "affine", "local"], default="local")
    parser.add_argument("--filtered-max-matches", type=int, default=0)
    parser.add_argument("--filtered-draw-matches", type=int, default=0)
    parser.add_argument("--filtered-min-margin", type=float, default=0.02)
    parser.add_argument("--input-local-contrast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-local-contrast-strength", type=float, default=0.0)
    parser.add_argument("--input-local-contrast-kernel", type=int, default=31)
    parser.add_argument("--illumination-stress", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = iter_sweep_configs(args)
    for config in configs:
        validate_config(config)
    if args.max_configs <= 0:
        raise ValueError("--max-configs must be positive")
    if len(configs) > args.max_configs:
        raise ValueError(f"refusing to run {len(configs)} configs; increase --max-configs to allow this sweep")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[FilterSweepSummary] = []
    for index, config in enumerate(configs, 1):
        report_dir = args.output_dir / f"{index:02d}_{slug_for_config(config)}"
        command = build_visual_command(args, config=config, report_dir=report_dir)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        summaries.append(summarize_report(report_dir, config=config))
    write_summary_csv(summaries, args.output_dir / "graph_filter_sweep_summary.csv")
    write_html_report(args, summaries, args.output_dir / "index.html")
    print(f"report={args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
