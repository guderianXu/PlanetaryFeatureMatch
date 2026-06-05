#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EVAL = PROJECT_ROOT / "python" / "pytorch_cache_match_eval.py"
GRAPH_PRESETS = {"off", "fast", "high_precision"}
PYTHON_FALLBACK_MODES = {"mutual", "none"}
RECOMMEND_PRECISION_TOLERANCE = 0.002


@dataclass(frozen=True)
class GraphSweepConfig:
    preset: str
    accept_probability: float
    fallback_mode: str


@dataclass(frozen=True)
class EvalSummary:
    config: GraphSweepConfig
    output_csv: Path
    pairs: int
    matches: int
    correct: int
    wrong: int
    precision: float
    low_precision_pairs: int
    avg_executed_layers: float = 0.0
    avg_input_keypoints_a: float = 0.0
    avg_input_keypoints_b: float = 0.0
    avg_kept_keypoints_a: float = 0.0
    avg_kept_keypoints_b: float = 0.0
    pruned_keypoint_fraction: float = 0.0
    attention_work_fraction: float = 0.0


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError as exc:
            raise ValueError(f"invalid float value: {item}") from exc
        if parsed < -1.0 or parsed > 1.0:
            raise ValueError("accept probabilities must be in [-1, 1]")
        values.append(parsed)
    if not values:
        raise ValueError("at least one accept probability is required")
    return values


def parse_choice_list(value: str, *, allowed: set[str], label: str) -> list[str]:
    choices: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(f"{label} must be one of: {allowed_text}")
        choices.append(item)
    if not choices:
        raise ValueError(f"at least one {label} is required")
    return choices


def iter_sweep_configs(
    presets: list[str],
    accept_probabilities: list[float],
    fallback_modes: list[str],
) -> list[GraphSweepConfig]:
    return [
        GraphSweepConfig(preset=preset, accept_probability=accept_probability, fallback_mode=fallback_mode)
        for preset in presets
        for accept_probability in accept_probabilities
        for fallback_mode in fallback_modes
    ]


def slug_for_config(config: GraphSweepConfig) -> str:
    probability = f"{config.accept_probability:g}".replace("-", "neg").replace(".", "p")
    return f"{config.preset}_accept{probability}_fallback{config.fallback_mode}"


def _int_from_row(row: dict[str, str], key: str) -> int:
    try:
        return int(row.get(key, "0") or 0)
    except ValueError:
        return 0


def summarize_eval_csv(path: Path, config: GraphSweepConfig) -> EvalSummary:
    pairs = 0
    matches = 0
    correct = 0
    wrong = 0
    low_precision_pairs = 0
    executed_layers = 0
    input_keypoints_a = 0
    input_keypoints_b = 0
    kept_keypoints_a = 0
    kept_keypoints_b = 0
    pruned_keypoints_a = 0
    pruned_keypoints_b = 0
    attention_work_units = 0
    full_attention_work_units = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pairs += 1
            row_matches = int(row.get("matches", "0") or 0)
            row_correct = int(row.get("correct", "0") or 0)
            row_wrong = int(row.get("wrong", "0") or 0)
            row_precision = float(row.get("precision", "0") or 0.0)
            matches += row_matches
            correct += row_correct
            wrong += row_wrong
            if row_precision < 0.9:
                low_precision_pairs += 1
            executed_layers += _int_from_row(row, "graph_executed_layers")
            input_keypoints_a += _int_from_row(row, "graph_input_keypoints_a")
            input_keypoints_b += _int_from_row(row, "graph_input_keypoints_b")
            kept_keypoints_a += _int_from_row(row, "graph_kept_keypoints_a")
            kept_keypoints_b += _int_from_row(row, "graph_kept_keypoints_b")
            pruned_keypoints_a += _int_from_row(row, "graph_pruned_keypoints_a")
            pruned_keypoints_b += _int_from_row(row, "graph_pruned_keypoints_b")
            attention_work_units += _int_from_row(row, "graph_attention_work_units")
            full_attention_work_units += _int_from_row(row, "graph_full_attention_work_units")
    precision = 0.0 if matches == 0 else correct / matches
    pair_count = max(1, pairs)
    total_input_keypoints = input_keypoints_a + input_keypoints_b
    total_pruned_keypoints = pruned_keypoints_a + pruned_keypoints_b
    pruned_keypoint_fraction = (
        0.0 if total_input_keypoints == 0 else total_pruned_keypoints / total_input_keypoints
    )
    attention_work_fraction = (
        0.0 if full_attention_work_units == 0 else attention_work_units / full_attention_work_units
    )
    return EvalSummary(
        config=config,
        output_csv=path,
        pairs=pairs,
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        low_precision_pairs=low_precision_pairs,
        avg_executed_layers=executed_layers / pair_count,
        avg_input_keypoints_a=input_keypoints_a / pair_count,
        avg_input_keypoints_b=input_keypoints_b / pair_count,
        avg_kept_keypoints_a=kept_keypoints_a / pair_count,
        avg_kept_keypoints_b=kept_keypoints_b / pair_count,
        pruned_keypoint_fraction=pruned_keypoint_fraction,
        attention_work_fraction=attention_work_fraction,
    )


def build_eval_command(args: argparse.Namespace, config: GraphSweepConfig, output_csv: Path) -> list[str]:
    command = [
        sys.executable,
        str(PYTHON_EVAL),
    ]
    for cache_dir in args.cache_dir:
        command.extend(["--cache-dir", str(cache_dir)])
    if args.pytorch_state is not None:
        command.extend(["--pytorch-state", str(args.pytorch_state)])
    if args.checkpoint is not None:
        command.extend(["--checkpoint", str(args.checkpoint)])
    command.extend(
        [
            "--output",
            str(output_csv),
            "--device",
            args.device,
            "--mode",
            args.mode,
            "--texture-blend-weight",
            f"{args.texture_blend_weight:g}",
            "--max-keypoints",
            str(args.max_keypoints),
            "--max-matches",
            str(args.max_matches),
            "--min-intensity",
            f"{args.min_intensity:g}",
            "--texture-keypoint-fraction",
            f"{args.texture_keypoint_fraction:g}",
            "--weak-texture-keypoint-fraction",
            f"{args.weak_texture_keypoint_fraction:g}",
            "--keypoint-spatial-bins",
            str(args.keypoint_spatial_bins),
            "--keypoint-cell-cap",
            str(args.keypoint_cell_cap),
            "--keypoint-score-mode",
            args.keypoint_score_mode,
            "--matcher-mode",
            "graph_matcher",
            "--graph-fallback-mode",
            config.fallback_mode,
            "--threshold-px",
            f"{args.threshold_px:g}",
            "--descriptor-topk",
            str(args.descriptor_topk),
            "--geometry-filter",
            args.geometry_filter,
            "--min-score",
            f"{args.min_score:g}",
            "--min-margin",
            f"{args.min_margin:g}",
            "--graph-dustbin-delta",
            f"{args.graph_dustbin_delta:g}",
            "--graph-acceptance-margin",
            f"{args.graph_acceptance_margin:g}",
            "--graph-min-raw-score",
            f"{args.graph_min_raw_score:g}",
            "--graph-min-raw-margin",
            f"{args.graph_min_raw_margin:g}",
            "--graph-max-attention-layers",
            str(args.graph_max_attention_layers),
            "--graph-max-attention-work-fraction",
            f"{args.graph_max_attention_work_fraction:g}",
            "--graph-width-prune-keep-ratio",
            f"{args.graph_width_prune_keep_ratio:g}",
            "--graph-min-accept-probability",
            f"{config.accept_probability:g}",
            "--graph-inference-preset",
            config.preset,
            "--min-target-gradient",
            f"{args.min_target_gradient:g}",
            "--min-target-local-contrast",
            f"{args.min_target_local_contrast:g}",
            "--limit-pairs",
            str(args.limit_pairs),
            "--hard-limit",
            str(args.hard_limit),
            "--hard-min-matches",
            str(args.hard_min_matches),
            "--hard-max-precision",
            f"{args.hard_max_precision:g}",
        ]
    )
    if args.mutual:
        command.append("--mutual")
    if args.sample_seed is not None:
        command.extend(["--sample-seed", str(args.sample_seed)])
    if args.exclude_self_pairs:
        command.append("--exclude-self-pairs")
    for hard_summary in args.hard_summary:
        command.extend(["--hard-summary", str(hard_summary)])
    return command


def write_summary_csv(summaries: list[EvalSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "preset",
                "accept_probability",
                "fallback_mode",
                "pairs",
                "matches",
                "correct",
                "wrong",
                "precision",
                "low_precision_pairs",
                "avg_executed_layers",
                "avg_input_keypoints_a",
                "avg_input_keypoints_b",
                "avg_kept_keypoints_a",
                "avg_kept_keypoints_b",
                "pruned_keypoint_fraction",
                "attention_work_fraction",
                "output_csv",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "preset": summary.config.preset,
                    "accept_probability": f"{summary.config.accept_probability:g}",
                    "fallback_mode": summary.config.fallback_mode,
                    "pairs": summary.pairs,
                    "matches": summary.matches,
                    "correct": summary.correct,
                    "wrong": summary.wrong,
                    "precision": f"{summary.precision:.6f}",
                    "low_precision_pairs": summary.low_precision_pairs,
                    "avg_executed_layers": f"{summary.avg_executed_layers:.3f}",
                    "avg_input_keypoints_a": f"{summary.avg_input_keypoints_a:.1f}",
                    "avg_input_keypoints_b": f"{summary.avg_input_keypoints_b:.1f}",
                    "avg_kept_keypoints_a": f"{summary.avg_kept_keypoints_a:.1f}",
                    "avg_kept_keypoints_b": f"{summary.avg_kept_keypoints_b:.1f}",
                    "pruned_keypoint_fraction": f"{summary.pruned_keypoint_fraction:.6f}",
                    "attention_work_fraction": f"{summary.attention_work_fraction:.6f}",
                    "output_csv": summary.output_csv.as_posix(),
                }
            )


def _effective_attention_work_fraction(summary: EvalSummary) -> float:
    return summary.attention_work_fraction if summary.attention_work_fraction > 0.0 else 1.0


def _best_summary(
    summaries: list[EvalSummary],
    *,
    max_attention_work_fraction: float = 1.0,
) -> EvalSummary | None:
    if not summaries:
        return None
    budget_candidates = [
        item
        for item in summaries
        if _effective_attention_work_fraction(item) <= max_attention_work_fraction
    ]
    pool = budget_candidates or summaries
    best_precision = max(item.precision for item in pool)
    candidates = [
        item for item in pool if item.precision >= best_precision - RECOMMEND_PRECISION_TOLERANCE
    ]
    return min(
        candidates,
        key=lambda item: (
            _effective_attention_work_fraction(item),
            -item.precision,
            -item.correct,
            -item.matches,
        ),
    )


def write_report_html(
    summaries: list[EvalSummary],
    path: Path,
    *,
    max_attention_work_fraction: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = _best_summary(summaries, max_attention_work_fraction=max_attention_work_fraction)
    budget_has_candidate = any(
        _effective_attention_work_fraction(summary) <= max_attention_work_fraction for summary in summaries
    )
    rows = []
    for summary in sorted(summaries, key=lambda item: (item.precision, item.correct, item.matches), reverse=True):
        is_best = summary is best
        rows.append(
            "<tr class=\"best\" if-best>"
            f"<td>{html.escape(summary.config.preset)}</td>"
            f"<td>{summary.config.accept_probability:g}</td>"
            f"<td>{html.escape(summary.config.fallback_mode)}</td>"
            f"<td>{summary.pairs}</td>"
            f"<td>{summary.matches}</td>"
            f"<td>{summary.correct}</td>"
            f"<td>{summary.wrong}</td>"
            f"<td>{summary.precision:.6f}</td>"
            f"<td>{summary.low_precision_pairs}</td>"
            f"<td>{summary.avg_executed_layers:.3f}</td>"
            f"<td>{summary.avg_kept_keypoints_a:.1f} / {summary.avg_kept_keypoints_b:.1f}</td>"
            f"<td>{summary.pruned_keypoint_fraction:.2%}</td>"
            f"<td>{summary.attention_work_fraction:.2%}</td>"
            f"<td><code>{html.escape(summary.output_csv.as_posix())}</code></td>"
            "</tr>"
        )
        if is_best:
            rows[-1] = rows[-1].replace(" if-best", "")
        else:
            rows[-1] = rows[-1].replace(" class=\"best\" if-best", "")

    best_text = "暂无结果"
    if best is not None:
        budget_text = (
            f"预算内推荐，max_attention_work_fraction={max_attention_work_fraction:.2%}"
            if budget_has_candidate
            else f"没有配置满足 max_attention_work_fraction={max_attention_work_fraction:.2%}，已退回全量候选推荐"
        )
        best_text = (
            f"{best.config.preset} / accept={best.config.accept_probability:g} / "
            f"fallback={best.config.fallback_mode}，precision={best.precision:.6f}，correct={best.correct}，"
            f"平均执行层数={best.avg_executed_layers:.3f}，剪枝比例={best.pruned_keypoint_fraction:.2%}，"
            f"计算量比例={best.attention_work_fraction:.2%}。{budget_text}"
        )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Graph 推理配置 Sweep 报告</title>
  <style>
    body {{
      margin: 0;
      background: #071014;
      color: #d9edf2;
      font-family: Arial, "Microsoft YaHei", sans-serif;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
    }}
    .meta {{
      color: #8ca3ad;
      margin-bottom: 20px;
    }}
    .summary {{
      border: 1px solid #21414b;
      background: #101b22;
      border-radius: 8px;
      padding: 16px 18px;
      margin-bottom: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #0c171d;
      border: 1px solid #21414b;
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #1b3038;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #14232b;
      color: #7fe7ee;
      font-weight: 700;
    }}
    tr.best {{
      background: #123034;
    }}
    code {{
      color: #9bd3ff;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
<main>
  <h1>Graph 推理配置 Sweep 报告</h1>
  <div class="meta">生成时间：{html.escape(generated_at)}。本报告用于比较严格图匹配、置信门控、早停、宽度剪枝和 fallback 策略；推荐配置会优先满足 max_attention_work_fraction={max_attention_work_fraction:.2%} 的计算预算，再在预算内选择质量和计算量更均衡的配置。</div>
  <section class="summary">
    <strong>推荐配置：</strong>{html.escape(best_text)}
  </section>
  <table>
    <thead>
      <tr>
        <th>Preset</th>
        <th>Accept 概率</th>
        <th>Fallback</th>
        <th>Pair 数</th>
        <th>匹配数</th>
        <th>正确数</th>
        <th>错误数</th>
        <th>Precision</th>
        <th>低精度 Pair</th>
        <th>平均执行层数</th>
        <th>平均保留点 A/B</th>
        <th>剪枝比例</th>
        <th>计算量比例</th>
        <th>明细 CSV</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _pythonpath_with_project(existing: str | None) -> str:
    paths = [str(PROJECT_ROOT / "python"), str(PROJECT_ROOT / "scripts")]
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def run_sweep(args: argparse.Namespace) -> list[EvalSummary]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    presets = parse_choice_list(args.presets, allowed=GRAPH_PRESETS, label="graph preset")
    accept_probabilities = parse_float_list(args.accept_probabilities)
    fallback_modes = parse_choice_list(args.fallback_modes, allowed=PYTHON_FALLBACK_MODES, label="fallback mode")
    configs = iter_sweep_configs(presets, accept_probabilities, fallback_modes)
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_project(env.get("PYTHONPATH"))

    command_log = args.output_dir / "commands.sh"
    summaries: list[EvalSummary] = []
    with command_log.open("w", encoding="utf-8") as command_handle:
        command_handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for config in configs:
            output_csv = args.output_dir / f"{slug_for_config(config)}.csv"
            command = build_eval_command(args, config, output_csv)
            command_handle.write(shlex.join(command) + "\n")
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
            if result.returncode != 0:
                if args.continue_on_error:
                    continue
                raise RuntimeError(
                    f"eval failed for {slug_for_config(config)} with exit code {result.returncode}"
                )
            summaries.append(summarize_eval_csv(output_csv, config))

    write_summary_csv(summaries, args.output_dir / "summary.csv")
    write_report_html(
        summaries,
        args.output_dir / "report.html",
        max_attention_work_fraction=args.max_attention_work_fraction,
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep graph matcher inference configs and write CSV/HTML reports")
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--pytorch-state", type=Path)
    model_group.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--presets", default="off,fast,high_precision")
    parser.add_argument("--accept-probabilities", default="-1,0.5,0.7,0.85")
    parser.add_argument("--fallback-modes", default="none,mutual")
    parser.add_argument("--mode", choices=["learned", "texture", "blend"], default="blend")
    parser.add_argument("--texture-blend-weight", type=float, default=1.0)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--texture-keypoint-fraction", type=float, default=1.0)
    parser.add_argument("--weak-texture-keypoint-fraction", type=float, default=0.0)
    parser.add_argument("--keypoint-spatial-bins", type=int, default=0)
    parser.add_argument("--keypoint-cell-cap", type=int, default=0)
    parser.add_argument("--keypoint-score-mode", choices=["texture", "learned"], default="texture")
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--descriptor-topk", type=int, default=1)
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--geometry-filter", choices=["none", "affine", "local"], default="none")
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--graph-dustbin-delta", type=float, default=0.0)
    parser.add_argument("--graph-acceptance-margin", type=float, default=0.0)
    parser.add_argument("--graph-min-raw-score", type=float, default=-1.0)
    parser.add_argument("--graph-min-raw-margin", type=float, default=0.0)
    parser.add_argument("--graph-max-attention-layers", type=int, default=0)
    parser.add_argument("--graph-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--graph-width-prune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--min-target-gradient", type=float, default=0.0)
    parser.add_argument("--min-target-local-contrast", type=float, default=0.0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--exclude-self-pairs", action="store_true")
    parser.add_argument("--hard-summary", action="append", type=Path, default=[])
    parser.add_argument("--hard-limit", type=int, default=64)
    parser.add_argument("--hard-min-matches", type=int, default=4)
    parser.add_argument("--hard-max-precision", type=float, default=0.9)
    parser.add_argument(
        "--max-attention-work-fraction",
        type=float,
        default=1.0,
        help="Only recommend configs at or below this graph attention work fraction when possible.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.max_attention_work_fraction < 0.0 or args.max_attention_work_fraction > 1.0:
        parser.error("--max-attention-work-fraction must be in [0, 1]")
    if args.graph_max_attention_layers < 0:
        parser.error("--graph-max-attention-layers must be nonnegative")
    if args.graph_max_attention_work_fraction < 0.0 or args.graph_max_attention_work_fraction > 1.0:
        parser.error("--graph-max-attention-work-fraction must be in [0, 1]")
    if args.graph_width_prune_keep_ratio < 0.0 or args.graph_width_prune_keep_ratio > 1.0:
        parser.error("--graph-width-prune-keep-ratio must be in [0, 1]")
    try:
        parse_choice_list(args.presets, allowed=GRAPH_PRESETS, label="graph preset")
        parse_float_list(args.accept_probabilities)
        parse_choice_list(args.fallback_modes, allowed=PYTHON_FALLBACK_MODES, label="fallback mode")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    summaries = run_sweep(parse_args())
    print(f"summary={len(summaries)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
