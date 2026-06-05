from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .commands import TrainingRequest, create_training_runs, start_generated_run
from .models import RunSummary
from .services import (
    active_training_processes,
    discover_runs,
    read_metrics_csv,
    run_metrics_path,
    delete_run,
    start_run_script,
    stop_run,
    summarize_dataset,
    tail_text,
)


DEFAULT_DATASETS = [
    Path("/media/xjw/PCIE5_8T/tmp/PlanetaryFeatureMatch/训练数据_regen/samepos_hx1_10view_2048_721/cache"),
    Path("/media/xjw/PCIE5_8T/tmp/PlanetaryFeatureMatch/训练数据_regen/crossres_lowdom_hx1_10view_2048_1200/cache"),
    Path("/media/xjw/PCIE5_8T/tmp/PlanetaryFeatureMatch/训练数据_regen/focal_narrow_hx1_10view_2048_900/cache"),
    Path("/media/xjw/PCIE5_8T/tmp/PlanetaryFeatureMatch/训练数据_regen/focal_wide_hx1_10view_2048_600/cache"),
    Path("/media/xjw/PCIE5_8T/tmp/PlanetaryFeatureMatch/训练数据_regen/crossres_lowdom_focal_narrow_hx1_10view_2048_600/cache"),
]


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return value


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _metric(summary: RunSummary, *names: str) -> str:
    for name in names:
        value = summary.latest_metrics.get(name)
        if isinstance(value, float):
            return f"{value:.6g}"
        if value not in (None, ""):
            return str(value)
    return "-"


def _metric_number(summary: RunSummary, *names: str) -> float | None:
    for name in names:
        value = summary.latest_metrics.get(name)
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return float(value)
    return None


def _format_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def _format_optional_time(timestamp: float | None) -> str:
    return _format_time(timestamp) if timestamp is not None else "-"


def _status_class(status: str) -> str:
    if status == "running":
        return "status-running"
    if status in {"logged", "stopped"}:
        return "status-done"
    if status in {"invalid", "unknown"}:
        return "status-warn"
    return "status-muted"


def _status_label(status: str) -> str:
    labels = {
        "running": "运行中",
        "logged": "有日志",
        "stopped": "已停止",
        "missing": "未启动",
        "invalid": "PID 异常",
        "unknown": "未知",
    }
    return labels.get(status, status)


def _backend_label(backend: str) -> str:
    labels = {
        "python": "Python",
        "cpp": "C++",
        "unknown": "未知",
    }
    return labels.get(backend, backend)


def _nav_item(path: str, label: str, active: str) -> str:
    selected = " active" if active == path else ""
    return f'<a class="nav-item{selected}" href="{path}">{label}</a>'


def _layout(title: str, body: str, active: str = "/") -> str:
    nav = "".join(
        [
            _nav_item("/", "总览", active),
            _nav_item("/train", "训练", active),
            _nav_item("/history", "历史训练", active),
            _nav_item("/runs", "任务", active),
            _nav_item("/compare", "对比", active),
            _nav_item("/datasets", "数据集", active),
        ]
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - PFM Lab</title>
  <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark"><span>PFM</span></div>
        <div>
          <div class="brand-title">PFM Lab</div>
          <div class="brand-subtitle">行星匹配训练台</div>
        </div>
      </div>
      <nav>{nav}</nav>
      <div class="sidebar-note">
        <strong>本地控制节点</strong>
        <span>127.0.0.1:7860 · Python/C++ 训练编排。</span>
      </div>
    </aside>
    <div class="page">
      <header class="topbar">
        <div>
          <h1>{title}</h1>
          <p>行星影像特征匹配训练、仿真检查、实验对比和数据集状态。</p>
        </div>
        <div class="topbar-actions">
          <span class="system-chip">CUDA 实验机</span>
          <a class="button secondary" href="/runs">查看任务</a>
          <a class="button primary" href="/train">新建训练</a>
        </div>
      </header>
      <main>{body}</main>
    </div>
  </div>
  <script src="/static/dashboard.js"></script>
</body>
</html>"""


def _runs_table(runs: list[RunSummary]) -> str:
    rows = []
    for run in runs[:80]:
        escaped_name = html.escape(run.name)
        encoded_name = quote(run.name)
        report = f'<a href="/runs/{encoded_name}/report">报告</a>' if run.has_report else "-"
        log = f'<a href="/runs/{encoded_name}/log">日志</a>' if run.has_log else "-"
        loss = _metric_number(run, "loss", "total_loss")
        top1 = _metric_number(run, "descriptor_accuracy", "top1_accuracy", "top1", "mean_top1")
        quality = top1 if top1 is not None else loss
        quality_width = 0
        if quality is not None:
            quality_width = max(4, min(100, int((1.0 - quality) * 100 if quality == loss else quality * 100)))
        start_button = (
            f'<form class="action-form" method="post" action="/runs/{encoded_name}/start">'
            '<button class="button small" type="submit">开始</button></form>'
            if run.can_start
            else ""
        )
        stop_button = (
            f'<form class="action-form" method="post" action="/runs/{encoded_name}/stop">'
            '<button class="button small danger" type="submit">停止</button></form>'
            if run.can_stop
            else ""
        )
        delete_button = (
            f'<form class="action-form" method="post" action="/runs/{encoded_name}/delete">'
            f'<button class="button small danger" type="submit" data-confirm="确认删除任务 {escaped_name}？任务会移动到 runs/.trash/">删除</button></form>'
            if run.can_delete
            else ""
        )
        actions = start_button + stop_button
        if not actions:
            actions = '<span class="muted">空闲</span>'
        delete_action = delete_button or '<span class="muted">-</span>'
        rows.append(
            "<tr>"
            f"<td><a class=\"run-link\" href=\"/compare?runs={encoded_name}\">{escaped_name}</a>"
            f"<span class=\"run-time\">更新 {_format_time(run.updated_at)}</span></td>"
            f"<td><span class=\"backend backend-{html.escape(run.backend)}\">{html.escape(_backend_label(run.backend))}</span></td>"
            f"<td><span class=\"status-pill {_status_class(run.status)}\">{html.escape(_status_label(run.status))}</span></td>"
            f"<td><span class=\"time-cell\">{_format_time(run.created_at)}</span></td>"
            f"<td><span class=\"time-cell\">{_format_optional_time(run.completed_at)}</span></td>"
            f"<td><div class=\"progress-cell\"><div class=\"progress-track\"><span style=\"width:{run.progress_percent:.1f}%\"></span></div>"
            f"<small>{html.escape(run.progress_label)}</small></div></td>"
            f"<td>{_metric(run, 'loss', 'total_loss')}</td>"
            f"<td>{_metric(run, 'descriptor_accuracy', 'top1_accuracy', 'top1', 'mean_top1')}</td>"
            f"<td>{_metric(run, 'descriptor_positive_rank', 'mean_positive_rank', 'mean_rank')}</td>"
            f"<td><div class=\"quality-bar\"><span style=\"width:{quality_width}%\"></span></div></td>"
            f"<td>{run.checkpoint_count}</td>"
            f"<td class=\"row-actions\">{log} {report}</td>"
            f"<td class=\"run-actions\">{actions}</td>"
            f"<td class=\"run-actions\">{delete_action}</td>"
            "</tr>"
        )
    return (
        "<div class=\"table-wrap\"><table><thead><tr><th>任务</th><th>后端</th><th>状态</th><th>创建时间</th><th>完成时间</th><th>进度</th><th>损失</th>"
        "<th>Top1</th><th>排名</th><th>信号</th><th>检查点</th><th>产物</th><th>控制</th><th>删除</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</div>"
    )


def _visual_report_path(run_path: Path) -> Path | None:
    for candidate in (run_path / "visual_report" / "index.html", run_path / "visual_report" / "run.html"):
        if candidate.exists():
            return candidate
    report_dirs = sorted(
        (path for path in run_path.glob("*_visual_report") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for report_dir in report_dirs:
        for candidate in (report_dir / "index.html", report_dir / "run.html"):
            if candidate.exists():
                return candidate
    return None


def _duration_label(run: RunSummary) -> str:
    if run.completed_at is None:
        return "未完成" if run.status != "running" else "运行中"
    seconds = max(0.0, run.completed_at - run.created_at)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}小时 {minutes}分"
    if minutes:
        return f"{minutes}分 {sec}秒"
    return f"{sec}秒"


def _row_number(row: dict[str, object], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, (float, int)):
            number = float(value)
            if math.isfinite(number):
                return number
    return None


def _metric_points(metrics, names: tuple[str, ...]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index, row in enumerate(metrics.rows):
        x = _row_number(row, "step", "global_step", "iteration", "batch")
        y = _row_number(row, *names)
        if y is None:
            continue
        points.append((float(index + 1 if x is None else x), y))
    return points


def _smooth_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 5:
        return points
    radius = max(2, min(16, len(points) // 36))
    smoothed: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        avg = sum(item[1] for item in points[start:end]) / max(1, end - start)
        smoothed.append((point[0], avg))
    return smoothed


def _svg_path(points: list[tuple[float, float]], x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    width = 520.0
    height = 190.0
    left = 48.0
    right = 16.0
    top = 16.0
    bottom = 30.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_scale(value: float) -> float:
        return left + ((value - x_min) / max(1.0e-9, x_max - x_min)) * plot_w

    def y_scale(value: float) -> float:
        return top + (1.0 - ((value - y_min) / max(1.0e-9, y_max - y_min))) * plot_h

    commands = []
    for index, (x_value, y_value) in enumerate(points):
        command = "M" if index == 0 else "L"
        commands.append(f"{command}{x_scale(x_value):.1f},{y_scale(y_value):.1f}")
    return " ".join(commands)


def _line_chart_from_points_svg(points: list[tuple[float, float]], title: str) -> str:
    if not points:
        return f'<div class="history-chart empty"><strong>{html.escape(title)}</strong><span>暂无指标</span></div>'
    max_points = 420
    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride] + ([points[-1]] if points[-1] != points[::stride][-1] else [])
    y_values = [point[1] for point in points]
    x_values = [point[0] for point in points]
    y_min = min(y_values)
    y_max = max(y_values)
    pad = max(1.0e-9, (y_max - y_min) * 0.08)
    y_min -= pad
    y_max += pad
    raw_path = _svg_path(points, min(x_values), max(x_values), y_min, y_max)
    smooth_path = _svg_path(_smooth_points(points), min(x_values), max(x_values), y_min, y_max)
    latest = points[-1]
    return f"""
<div class="history-chart">
  <div><strong>{html.escape(title)}</strong><span>当前 {latest[1]:.6g}</span></div>
  <svg viewBox="0 0 520 190" preserveAspectRatio="none">
    <line class="chart-guide" x1="48" y1="52" x2="504" y2="52"></line>
    <line class="chart-guide" x1="48" y1="98" x2="504" y2="98"></line>
    <line class="chart-guide" x1="48" y1="144" x2="504" y2="144"></line>
    <path class="chart-raw" d="{raw_path}"></path>
    <path class="chart-smooth" d="{smooth_path}"></path>
    <text class="chart-label" x="4" y="20">{y_max:.4g}</text>
    <text class="chart-label" x="4" y="158">{y_min:.4g}</text>
    <text class="chart-label" x="48" y="182">step {min(x_values):.0f}</text>
    <text class="chart-label" x="430" y="182">step {max(x_values):.0f}</text>
  </svg>
</div>
"""


def _line_chart_svg(metrics, title: str, names: tuple[str, ...]) -> str:
    return _line_chart_from_points_svg(_metric_points(metrics, names), title)


def _derived_line_chart_svg(metrics, title: str, value_fn) -> str:
    points: list[tuple[float, float]] = []
    for index, row in enumerate(metrics.rows):
        x = _row_number(row, "step", "global_step", "iteration", "batch")
        value = value_fn(row)
        if value is None:
            continue
        points.append((float(index + 1 if x is None else x), value))
    return _line_chart_from_points_svg(points, title)


def _histogram_svg(metrics, title: str, names: tuple[str, ...]) -> str:
    values = [point[1] for point in _metric_points(metrics, names)]
    if not values:
        return f'<div class="history-chart empty"><strong>{html.escape(title)}</strong><span>暂无指标</span></div>'
    bins = 28
    low = min(values)
    high = max(values)
    if high <= low:
        high = low + 1.0
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, max(0, int((value - low) / (high - low) * bins)))
        counts[index] += 1
    max_count = max(counts) or 1
    bars = []
    for index, count in enumerate(counts):
        x = 48 + index * (456 / bins)
        height = (count / max_count) * 130
        y = 154 - height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(2.0, 456 / bins - 2):.1f}" height="{height:.1f}"></rect>')
    return f"""
<div class="history-chart">
  <div><strong>{html.escape(title)}</strong><span>{len(values)} 个 batch</span></div>
  <svg class="histogram" viewBox="0 0 520 190" preserveAspectRatio="none">
    <line class="chart-guide" x1="48" y1="154" x2="504" y2="154"></line>
    {''.join(bars)}
    <text class="chart-label" x="4" y="20">{max_count}</text>
    <text class="chart-label" x="48" y="182">{low:.4g}</text>
    <text class="chart-label" x="430" y="182">{high:.4g}</text>
  </svg>
</div>
"""


def _read_visual_summary(run_path: Path) -> list[dict[str, str]]:
    report_path = _visual_report_path(run_path)
    summary = report_path.parent / "summary.csv" if report_path is not None else run_path / "visual_report" / "summary.csv"
    if not summary.exists():
        return []
    with summary.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_visual_match_summary(run_path: Path) -> list[dict[str, str]]:
    report_path = _visual_report_path(run_path)
    summary = (
        report_path.parent / "match_visual_summary.csv"
        if report_path is not None
        else run_path / "visual_report" / "match_visual_summary.csv"
    )
    if not summary.exists():
        return []
    with summary.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_value(value: object) -> float | None:
    if isinstance(value, (float, int)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _mapping_number(row: dict[str, object], *names: str) -> float | None:
    for name in names:
        number = _float_value(row.get(name))
        if number is not None:
            return number
    return None


def _metric_values(metrics, *names: str) -> list[float]:
    return [
        value
        for row in metrics.rows
        for value in [_mapping_number(row, *names)]
        if value is not None
    ]


def _summary_values(rows: list[dict[str, str]], *names: str) -> list[float]:
    return [
        value
        for row in rows
        for value in [_mapping_number(row, *names)]
        if value is not None
    ]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100.0:.1f}%"


def _format_float(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _graph_pruned_fraction_from_row(row: dict[str, object]) -> float | None:
    explicit = _mapping_number(row, "graph_pruned_keypoint_fraction", "pruned_keypoint_fraction", "graph_prune_fraction")
    if explicit is not None:
        return explicit
    pruned_a = _mapping_number(row, "graph_pruned_keypoints_a", "pruned_keypoints_a")
    pruned_b = _mapping_number(row, "graph_pruned_keypoints_b", "pruned_keypoints_b")
    input_a = _mapping_number(row, "graph_input_keypoints_a", "input_keypoints_a")
    input_b = _mapping_number(row, "graph_input_keypoints_b", "input_keypoints_b")
    if None in (pruned_a, pruned_b, input_a, input_b):
        return None
    total = float(input_a or 0.0) + float(input_b or 0.0)
    if total <= 0.0:
        return None
    return (float(pruned_a or 0.0) + float(pruned_b or 0.0)) / total


def _graph_pruned_fraction_values(metrics, summary_rows: list[dict[str, str]]) -> list[float]:
    values = [value for row in metrics.rows for value in [_graph_pruned_fraction_from_row(row)] if value is not None]
    if values:
        return values
    return [value for row in summary_rows for value in [_graph_pruned_fraction_from_row(row)] if value is not None]


def _graph_efficiency_section(metrics, summary_rows: list[dict[str, str]]) -> str:
    work_values = _metric_values(
        metrics,
        "graph_attention_work_fraction",
        "attention_work_fraction",
        "graph_work",
        "average_graph_attention_work_fraction",
    )
    if not work_values:
        work_values = _summary_values(summary_rows, "graph_attention_work_fraction", "attention_work_fraction", "graph_work")
    layer_values = _metric_values(metrics, "graph_executed_layers", "executed_layers", "average_graph_executed_layers")
    if not layer_values:
        layer_values = _summary_values(summary_rows, "graph_executed_layers", "executed_layers")
    pruned_values = _graph_pruned_fraction_values(metrics, summary_rows)
    if not work_values and not layer_values and not pruned_values:
        return ""
    mean_work = _mean(work_values)
    mean_layers = _mean(layer_values)
    mean_pruned = _mean(pruned_values)
    saved = None if mean_work is None else max(0.0, 1.0 - mean_work)
    return f"""
  <section class="panel history-charts">
    <div class="panel-head"><div><h2>LightGlue 自适应推理</h2><p>展示 GraphMatcher 的早停、剪枝和注意力计算量占比，用来判断自适应推理是否真的省算力。</p></div></div>
    <div class="metric-grid history-metrics">
      <article class="metric-card"><span>平均计算量占比</span><strong>{_format_percent(mean_work)}</strong><small>越低越省 attention</small></article>
      <article class="metric-card"><span>平均节省计算量</span><strong>{_format_percent(saved)}</strong><small>相对满层全宽度</small></article>
      <article class="metric-card"><span>平均执行层数</span><strong>{_format_float(mean_layers)}</strong><small>早停后层数</small></article>
      <article class="metric-card"><span>平均剪枝比例</span><strong>{_format_percent(mean_pruned)}</strong><small>宽度剪枝点占比</small></article>
    </div>
    <div class="history-chart-grid">
      {_line_chart_svg(metrics, '计算量占比', ('graph_attention_work_fraction', 'attention_work_fraction', 'graph_work', 'average_graph_attention_work_fraction'))}
      {_line_chart_svg(metrics, '执行层数', ('graph_executed_layers', 'executed_layers', 'average_graph_executed_layers'))}
      {_derived_line_chart_svg(metrics, '剪枝比例', _graph_pruned_fraction_from_row)}
      {_histogram_svg(metrics, '计算量占比分布', ('graph_attention_work_fraction', 'attention_work_fraction', 'graph_work', 'average_graph_attention_work_fraction'))}
    </div>
  </section>
"""


def render_history(project_root: Path, query: dict[str, list[str]]) -> str:
    runs = discover_runs(project_root / "runs")
    selected_name = query.get("run", [runs[0].name if runs else ""])[0]
    selected = next((run for run in runs if run.name == selected_name), runs[0] if runs else None)
    run_links = []
    for run in runs[:120]:
        active = " active" if selected and run.name == selected.name else ""
        visual_badge = "有图" if _visual_report_path(run.path) is not None else "无图"
        run_links.append(
            f'<a class="history-run{active}" href="/history?run={quote(run.name)}">'
            f'<strong>{html.escape(run.name)}</strong><span>{_format_time(run.created_at)} · {visual_badge}</span></a>'
        )
    if selected is None:
        detail = '<section class="panel"><h2>暂无历史训练</h2><p class="muted">runs/ 下还没有可展示的训练。</p></section>'
    else:
        metrics = read_metrics_csv(run_metrics_path(selected.path))
        visual_report = _visual_report_path(selected.path)
        summary_rows = _read_visual_summary(selected.path)
        match_summary_rows = _read_visual_match_summary(selected.path)
        summary_table = "".join(
            "<tr>"
            f"<td>{html.escape(row.get('label', '-'))}</td>"
            f"<td>{html.escape(row.get('target_variant', '-'))}</td>"
            f"<td>{html.escape(row.get('matches', '-'))}</td>"
            f"<td>{html.escape(row.get('correct', '-'))}</td>"
            f"<td>{html.escape(row.get('wrong', '-'))}</td>"
            f"<td>{html.escape(row.get('precision', '-'))}</td>"
            "</tr>"
            for row in summary_rows[:12]
        )
        visual_block = (
            f'<iframe class="history-report-frame" src="/runs/{quote(selected.name)}/visual-report"></iframe>'
            if visual_report is not None
            else '<div class="history-empty-report">这次训练还没有 visual_report。新训练结束后会自动生成；旧 run 可以手动补跑可视化脚本。</div>'
        )
        detail = f"""
<section class="history-detail">
  <div class="history-head panel">
    <div>
      <h2>{html.escape(selected.name)}</h2>
      <p>创建 {_format_time(selected.created_at)} · 完成 {_format_optional_time(selected.completed_at)} · 用时 {_duration_label(selected)}</p>
    </div>
    <div class="history-actions">
      <a class="button secondary" href="/runs/{quote(selected.name)}/log">日志</a>
      <a class="button secondary" href="/runs/{quote(selected.name)}/report">训练报告</a>
      {f'<a class="button primary" href="/runs/{quote(selected.name)}/visual-report">匹配报告</a>' if visual_report is not None else ''}
    </div>
  </div>
  <div class="metric-grid history-metrics">
    <article class="metric-card"><span>训练用时</span><strong>{_duration_label(selected)}</strong><small>{selected.progress_label}</small></article>
    <article class="metric-card"><span>最新 Loss</span><strong>{_metric(selected, 'loss', 'total_loss', 'train_loss')}</strong><small>{len(metrics.rows)} 行指标</small></article>
    <article class="metric-card"><span>最新 Top1</span><strong>{_metric(selected, 'descriptor_accuracy', 'top1_accuracy', 'top1', 'mean_top1')}</strong><small>训练 batch 指标</small></article>
    <article class="metric-card"><span>检查点</span><strong>{selected.checkpoint_count}</strong><small>模型产物数量</small></article>
  </div>
  <section class="panel history-charts">
    <div class="panel-head"><div><h2>训练指标</h2><p>每个 batch 的原始指标和趋势，来自 metrics.csv / train_metrics.csv。</p></div></div>
    <div class="history-chart-grid">
      {_line_chart_svg(metrics, 'Loss', ('loss', 'loss_total', 'total_loss', 'train_loss'))}
      {_line_chart_svg(metrics, 'Top1', ('descriptor_accuracy', 'top1_accuracy', 'top1', 'mean_top1'))}
      {_line_chart_svg(metrics, '正样本排名', ('descriptor_positive_rank', 'mean_positive_rank', 'mean_rank'))}
      {_histogram_svg(metrics, 'Loss 直方图', ('loss', 'loss_total', 'total_loss', 'train_loss'))}
    </div>
  </section>
  {_graph_efficiency_section(metrics, match_summary_rows)}
  <section class="panel">
    <div class="panel-head"><div><h2>匹配样本摘要</h2><p>绿色正确、红色错误；下方完整报告内含连线图和误差直方图。</p></div></div>
    <div class="table-wrap"><table><thead><tr><th>类型</th><th>扰动</th><th>匹配</th><th>正确</th><th>错误</th><th>正确率</th></tr></thead><tbody>{summary_table}</tbody></table></div>
    {visual_block}
  </section>
</section>
"""
    body = f"""
<section class="history-layout">
  <aside class="panel history-list">
    <div class="panel-head"><div><h2>历史训练</h2><p>按 runs/ 目录更新时间排序。</p></div></div>
    <div class="history-run-list">{''.join(run_links) or '<p class="muted">暂无 run</p>'}</div>
  </aside>
  {detail}
</section>
"""
    return _layout("历史训练", body, active="/history")


def render_index(project_root: Path) -> str:
    runs = discover_runs(project_root / "runs")
    active = active_training_processes()
    disk = shutil.disk_usage(project_root)
    active_rows = "".join(f"<li><code>{html.escape(line)}</code></li>" for line in active) or "<li>没有发现活动训练进程</li>"
    running_count = sum(1 for run in runs if run.status == "running")
    checkpoint_count = sum(run.checkpoint_count for run in runs)
    latest_loss = _metric(runs[0], "loss", "total_loss") if runs else "-"
    disk_used_percent = int((disk.used / disk.total) * 100) if disk.total else 0
    body = f"""
<section class="hero-panel">
  <div>
    <h2>训练作业控制台</h2>
    <p>启动 Python/C++ 对比实验，查看实时进度、指标曲线、日志和 HTML 留档。</p>
  </div>
  <div class="hero-actions">
    <a class="button primary" href="/train">启动对比训练</a>
    <a class="button secondary" href="/compare">打开对比</a>
  </div>
</section>
<section class="metric-grid">
  <article class="metric-card"><span>训练任务</span><strong>{len(runs)}</strong><small>{running_count} 个运行中</small></article>
  <article class="metric-card"><span>最新损失</span><strong>{latest_loss}</strong><small>来自最新任务</small></article>
  <article class="metric-card"><span>检查点</span><strong>{checkpoint_count}</strong><small>已发现模型产物</small></article>
  <article class="metric-card"><span>磁盘剩余</span><strong>{_format_bytes(disk.free)}</strong><small>已用 {disk_used_percent}%</small></article>
</section>
<section class="content-grid overview-runs-grid">
  <article class="panel wide">
    <div class="panel-head"><div><h2>近期任务</h2><p>最新训练实验、进度和模型产物。</p></div><a href="/runs">全部任务</a></div>
    {_runs_table(runs)}
  </article>
  <article class="panel">
    <div class="panel-head"><div><h2>活动进程</h2><p>当前可见的仿真和训练进程。</p></div></div>
    <ul class="processes">{active_rows}</ul>
  </article>
</section>
"""
    return _layout("总览", body, active="/")


def render_train(project_root: Path, message: str = "") -> str:
    default_cache = "\n".join(str(path / "train") for path in DEFAULT_DATASETS if (path / "train").exists())
    notice = f'<p class="notice">{html.escape(message)}</p>' if message else ""
    body = f"""
{notice}
<section class="panel live-training-panel" data-live-training>
  <div class="panel-head">
    <div><h2>实时训练进度</h2><p>自动刷新当前训练任务、进度和关键指标曲线。</p></div>
    <div class="live-refresh">
      <span data-live-updated>等待刷新</span>
      <button class="button small" type="button" data-live-refresh>立即刷新</button>
    </div>
  </div>
  <div class="live-grid">
    <div class="live-run-list" data-live-runs>
      <p class="muted">正在读取训练任务...</p>
    </div>
    <div class="live-metrics">
      <div class="live-stat-grid">
        <article><span>运行中</span><strong data-live-running>0</strong></article>
        <article><span>当前 Epoch</span><strong data-live-epoch>-</strong></article>
        <article><span>当前 Batch</span><strong data-live-batch>-</strong></article>
        <article><span>最新损失</span><strong data-live-loss>-</strong></article>
        <article><span>最新 Top1</span><strong data-live-top1>-</strong></article>
        <article><span>指标行数</span><strong data-live-rows>0</strong></article>
      </div>
    </div>
  </div>
  <div class="live-chart-section">
    <div class="section-caption">
      <h3>训练指标曲线</h3>
      <span>独立 2x2 区域，默认显示当前任务最近 300 个 batch。</span>
      <span class="chart-legend"><i class="legend-raw"></i>浅线：每 batch 原始值 <i class="legend-smooth"></i>亮线：平滑趋势 <i class="legend-current"></i>圆点：当前 batch</span>
    </div>
    <div class="live-chart-grid">
      <div class="live-chart-card"><div><strong>损失</strong><span data-live-chart-meta="loss">最近 300 batch</span></div><svg data-live-chart="loss" viewBox="0 0 520 220" preserveAspectRatio="none"></svg></div>
      <div class="live-chart-card"><div><strong>Top1</strong><span data-live-chart-meta="top1">最近 300 batch</span></div><svg data-live-chart="top1" viewBox="0 0 520 220" preserveAspectRatio="none"></svg></div>
      <div class="live-chart-card"><div><strong>早停置信</strong><span data-live-chart-meta="stop_confidence">最近 300 batch</span></div><svg data-live-chart="stop_confidence" viewBox="0 0 520 220" preserveAspectRatio="none"></svg></div>
      <div class="live-chart-card"><div><strong>图剪枝</strong><span data-live-chart-meta="graph_prune">最近 300 batch</span></div><svg data-live-chart="graph_prune" viewBox="0 0 520 220" preserveAspectRatio="none"></svg></div>
    </div>
  </div>
</section>
<form method="post" action="/train" class="train-workbench">
  <section class="panel launch-panel">
    <div class="panel-head"><div><h2>实验配置</h2><p>默认启动 C++ 训练；Python 只作为单独验证入口。</p></div></div>
    <div class="form-grid two">
      <label>实验名称 <input name="experiment_name" value="dashboard_cpp"></label>
      <label>训练后端
        <select name="backend">
          <option value="cpp">C++ 训练</option>
          <option value="python">Python 验证</option>
        </select>
      </label>
      <label>设备 <input name="device" value="cuda"></label>
      <label>初始检查点 <input name="init_checkpoint" placeholder="可选路径"></label>
      <label>GraphMatcher 推理预设
        <select name="graph_inference_preset">
          <option value="fast">LightGlue 快速剪枝</option>
          <option value="high_precision">LightGlue 高精度过滤</option>
          <option value="off">关闭自适应剪枝</option>
        </select>
      </label>
      <label>匹配接受概率 <input type="number" name="graph_min_accept_probability" value="-1" min="-1" max="1" step="0.01"></label>
      <label>计算量预算 <input type="number" name="graph_max_attention_work_fraction" value="1" min="0" max="1" step="0.01"></label>
      <label>宽度保留比例 <input type="number" name="graph_width_prune_keep_ratio" value="1" min="0" max="1" step="0.01"></label>
      <label>不可匹配点数 <input type="number" name="graph_matcher_no_match_points" value="0" min="0"></label>
      <label>不可匹配权重 <input type="number" name="graph_matcher_no_match_weight" value="0" min="0" step="0.01"></label>
      <label>不可匹配最小距离 <input type="number" name="graph_matcher_no_match_min_distance" value="4" min="0" step="0.5"></label>
      <label>早停置信权重 <input type="number" name="graph_matcher_stop_confidence_weight" value="0.05" min="0" step="0.01"></label>
      <label>早停安全间隔 <input type="number" name="graph_matcher_stop_confidence_margin" value="0.5" min="0" step="0.05"></label>
    </div>
    <div class="quick-presets">
      <button type="button" data-preset="smoke">冒烟测试</button>
      <button type="button" data-preset="balanced">均衡训练</button>
      <button type="button" data-preset="long">长训练</button>
    </div>
  </section>
  <section class="panel data-panel">
    <div class="panel-head"><div><h2>训练数据</h2><p>缓存目录会写入当前后端的启动脚本。</p></div><a href="/datasets">查看数据集</a></div>
    <label>训练缓存目录<textarea name="cache_dirs" rows="7">{html.escape(default_cache)}</textarea></label>
    <label>验证缓存目录<textarea name="validation_cache_dirs" rows="3" placeholder="可选，每行一个路径"></textarea></label>
  </section>
  <section class="panel">
    <div class="panel-head"><div><h2>训练参数</h2><p>优化器、裁剪和采样相关核心参数。</p></div></div>
    <div class="form-grid three">
      <label>训练轮数 <input type="number" name="epochs" value="1" min="1"></label>
      <label>批大小 <input type="number" name="batch_size" value="1" min="1"></label>
      <label>训练裁剪 <input type="number" name="training_crop_size" value="512" min="0"></label>
      <label>输入缩放 <input type="number" name="resize" value="512" min="0"></label>
      <label>每对采样点 <input type="number" name="samples_per_pair" value="512" min="1"></label>
      <label>学习率 <input name="learning_rate" value="3e-5"></label>
      <label>最大批次数 <input type="number" name="max_train_batches" value="0" min="0"></label>
    </div>
    <p class="hint">C++ 的完整训练定义已经默认与 Python 对齐，不再需要手动选择 profile。</p>
  </section>
  <section class="panel">
    <div class="panel-head"><div><h2>缓存与加载器</h2><p>提前把 pair tensor 加载到内存，减少 GPU 等 IO。</p></div></div>
    <div class="form-grid three">
      <label>内存缓存条数 <input type="number" name="memory_cache_items" value="64" min="0"></label>
      <label>预取批次数 <input type="number" name="prefetch_batches" value="4" min="1"></label>
      <label>Python 加载线程 <input type="number" name="prefetch_workers" value="2" min="0"></label>
      <label>C++ 加载线程 <input type="number" name="dataloader_workers" value="2" min="0"></label>
    </div>
  </section>
  <div class="sticky-submit">
    <span>启动脚本、日志和 HTML 记录会写入 <code>runs/</code>。</span>
    <button class="button primary" type="submit">启动训练</button>
  </div>
</form>
"""
    return _layout("训练", body, active="/train")


def render_runs(project_root: Path, message: str = "") -> str:
    runs = discover_runs(project_root / "runs")
    notice = f'<p class="notice">{html.escape(message)}</p>' if message else ""
    body = f"""
{notice}
<section class="panel">
  <div class="panel-head"><div><h2>训练任务</h2><p><code>runs/</code> 下已发现的训练目录。</p></div><a href="/train">新建训练</a></div>
  {_runs_table(runs)}
</section>
"""
    return _layout("任务", body, active="/runs")


def render_compare(project_root: Path, query: dict[str, list[str]]) -> str:
    selected = query.get("runs", [])
    runs = discover_runs(project_root / "runs")
    options = "".join(
        f'<option value="{html.escape(run.name)}" {"selected" if run.name in selected else ""}>{html.escape(run.name)}</option>' for run in runs
    )
    body = f"""
<section class="compare-layout">
  <form method="get" action="/compare" class="panel compare-picker">
    <div class="panel-head"><div><h2>任务对比</h2><p>选择 Python 和 C++ 任务，叠加查看指标曲线。</p></div></div>
    <select name="runs" multiple size="16">{options}</select>
    <button class="button primary" type="submit">加载曲线</button>
  </form>
  <div class="panel chart-panel">
    <div class="panel-head"><div><h2>指标曲线</h2><p>损失曲线来自各任务的 metrics.csv 或 train_metrics.csv。</p></div></div>
    <canvas id="metricChart" data-runs="{html.escape(','.join(selected))}"></canvas>
  </div>
</section>
"""
    return _layout("对比", body, active="/compare")


def render_datasets() -> str:
    rows = []
    for path in DEFAULT_DATASETS:
        summary = summarize_dataset(path)
        exists = path.exists()
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(path))}</code><span class=\"run-time\">{'可用' if exists else '缺失'}</span></td>"
            f"<td>{summary.counts['train']}</td>"
            f"<td>{summary.counts['val']}</td>"
            f"<td>{summary.counts['test']}</td>"
            f"<td>{summary.counts['total']}</td>"
            f"<td>{_format_bytes(summary.bytes_used)}</td>"
            "</tr>"
        )
    body = (
        "<section class=\"panel\"><div class=\"panel-head\"><div><h2>数据集</h2>"
        "<p>训练面板默认使用的 pair-cache 根目录。</p></div></div>"
        "<div class=\"table-wrap\"><table><thead><tr><th>路径</th><th>训练</th><th>验证</th>"
        "<th>测试</th><th>总数</th><th>大小</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )
    return _layout("数据集", body, active="/datasets")


class DashboardHandler(BaseHTTPRequestHandler):
    project_root: Path = Path.cwd()

    def _send_html(self, html_text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str) -> None:
        self._send_payload(text, "text/plain; charset=utf-8")

    def _send_payload(self, text: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        root = self.project_root
        if parsed.path == "/":
            self._send_html(render_index(root))
        elif parsed.path == "/train":
            self._send_html(render_train(root))
        elif parsed.path == "/history":
            self._send_html(render_history(root, query))
        elif parsed.path == "/runs":
            self._send_html(render_runs(root))
        elif parsed.path == "/compare":
            self._send_html(render_compare(root, query))
        elif parsed.path == "/datasets":
            self._send_html(render_datasets())
        elif parsed.path == "/api/runs":
            self._send_json({"runs": [run.__dict__ for run in discover_runs(root / "runs")]})
        elif parsed.path == "/api/metrics":
            names = query.get("runs", [])
            metrics = {
                name: read_metrics_csv(run_metrics_path(root / "runs" / name)).__dict__
                for name in names
                if (root / "runs" / name).exists()
            }
            self._send_json({"metrics": metrics})
        elif parsed.path.startswith("/runs/") and parsed.path.endswith("/log"):
            name = unquote(parsed.path.split("/")[2])
            self._send_text(tail_text(root / "runs" / name / "train.log", lines=200))
        elif parsed.path.startswith("/runs/") and parsed.path.endswith("/report"):
            name = unquote(parsed.path.split("/")[2])
            report = root / "runs" / name / "run.html"
            self._send_html(report.read_text(encoding="utf-8") if report.exists() else "报告缺失")
        elif parsed.path.startswith("/runs/") and parsed.path.endswith("/visual-report"):
            name = unquote(parsed.path.split("/")[2])
            run_path = root / "runs" / name
            report = _visual_report_path(run_path)
            self._send_html(report.read_text(encoding="utf-8") if report is not None else "匹配可视化报告缺失")
        elif parsed.path == "/static/dashboard.css":
            self._send_payload(STYLE, "text/css; charset=utf-8")
        elif parsed.path == "/static/dashboard.js":
            self._send_payload(SCRIPT, "application/javascript; charset=utf-8")
        else:
            self._send_html(_layout("未找到", "<p>页面不存在</p>"), HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/runs/") and (
            parsed.path.endswith("/start") or parsed.path.endswith("/stop") or parsed.path.endswith("/delete")
        ):
            self._handle_run_action(parsed.path)
            return
        if parsed.path != "/train":
            self._send_html(_layout("未找到", "<p>页面不存在</p>"), HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        fields = parse_qs(self.rfile.read(length).decode("utf-8"))
        value = lambda name, default="": fields.get(name, [default])[0]
        lines = lambda name: [line.strip() for line in value(name).splitlines() if line.strip()]
        request = TrainingRequest(
            experiment_name=value("experiment_name", "dashboard_run"),
            backend=value("backend", "cpp"),
            cache_dirs=lines("cache_dirs"),
            validation_cache_dirs=lines("validation_cache_dirs"),
            output_root=self.project_root / "runs",
            init_checkpoint=value("init_checkpoint", ""),
            device=value("device", "cuda"),
            epochs=int(value("epochs", "1")),
            batch_size=int(value("batch_size", "1")),
            resize=int(value("resize", "512")),
            training_crop_size=int(value("training_crop_size", "512")),
            samples_per_pair=int(value("samples_per_pair", "512")),
            learning_rate=float(value("learning_rate", "3e-5")),
            profile="full",
            memory_cache_items=int(value("memory_cache_items", "64")),
            prefetch_batches=int(value("prefetch_batches", "4")),
            prefetch_workers=int(value("prefetch_workers", "2")),
            dataloader_workers=int(value("dataloader_workers", "2")),
            max_train_batches=int(value("max_train_batches", "0")),
            graph_inference_preset=value("graph_inference_preset", "fast"),
            graph_min_accept_probability=float(value("graph_min_accept_probability", "-1")),
            graph_max_attention_work_fraction=float(value("graph_max_attention_work_fraction", "1")),
            graph_width_prune_keep_ratio=float(value("graph_width_prune_keep_ratio", "1")),
            graph_matcher_no_match_points=int(value("graph_matcher_no_match_points", "0")),
            graph_matcher_no_match_weight=float(value("graph_matcher_no_match_weight", "0")),
            graph_matcher_no_match_min_distance=float(value("graph_matcher_no_match_min_distance", "4")),
            graph_matcher_stop_confidence_weight=float(value("graph_matcher_stop_confidence_weight", "0.05")),
            graph_matcher_stop_confidence_margin=float(value("graph_matcher_stop_confidence_margin", "0.5")),
        )
        try:
            generated = create_training_runs(request)
            pids = [start_generated_run(run) for run in generated]
            message = "已启动：" + "，".join(f"{run.run_dir.name} pid={pid}" for run, pid in zip(generated, pids))
        except Exception as exc:
            message = f"启动失败：{exc}"
        self._send_html(render_train(self.project_root, message=message))

    def _handle_run_action(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3:
            self._send_html(_layout("未找到", "<p>页面不存在</p>"), HTTPStatus.NOT_FOUND)
            return
        name = unquote(parts[1])
        action = parts[2]
        run_path = (self.project_root / "runs" / name).resolve()
        runs_root = (self.project_root / "runs").resolve()
        if runs_root not in run_path.parents or not run_path.exists():
            self._send_html(_layout("未找到", "<p>页面不存在</p>"), HTTPStatus.NOT_FOUND)
            return
        try:
            if action == "start":
                pid = start_run_script(run_path)
                message = f"已启动 {name}，pid={pid}"
            elif action == "stop":
                pid = stop_run(run_path)
                message = f"已向 {name} 发送停止信号，pid={pid}"
            elif action == "delete":
                target = delete_run(run_path)
                message = f"已删除 {name}，已移动到 {target}"
            else:
                self._send_html(_layout("未找到", "<p>页面不存在</p>"), HTTPStatus.NOT_FOUND)
                return
        except Exception as exc:
            message = f"{name} 执行 {action} 失败：{exc}"
        self._send_html(render_runs(self.project_root, message=message))


STYLE = """
:root {
  --bg: #0b1015;
  --sidebar: #0e151b;
  --sidebar-line: #22303b;
  --surface: #141c24;
  --surface-raised: #18232c;
  --surface-soft: #10171e;
  --surface-hot: #172728;
  --line: #273542;
  --line-strong: #354656;
  --text: #dfe7ee;
  --muted: #82909f;
  --muted-strong: #a8b5c2;
  --accent: #48bfc1;
  --accent-strong: #7bd6d4;
  --blue: #83a8cf;
  --green: #79bf70;
  --amber: #d37b49;
  --red: #cf664f;
  --shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
  --inner: inset 0 1px 0 rgba(255, 255, 255, 0.035);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 72% 8%, rgba(72, 191, 193, 0.05), transparent 30%),
    linear-gradient(135deg, #0b1015 0%, #101821 54%, #0b1015 100%);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--accent-strong); text-decoration: none; }
a:hover { text-decoration: underline; }
.app-shell { display: grid; grid-template-columns: 248px minmax(0, 1fr); min-height: 100vh; }
.sidebar {
  background:
    linear-gradient(180deg, rgba(72, 191, 193, 0.055), transparent 260px),
    var(--sidebar);
  color: var(--text);
  padding: 22px 18px;
  display: flex;
  flex-direction: column;
  gap: 22px;
  position: sticky;
  top: 0;
  height: 100vh;
  border-right: 1px solid var(--sidebar-line);
  box-shadow: 18px 0 60px rgba(0, 0, 0, 0.28);
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
  width: 44px;
  height: 44px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background:
    linear-gradient(135deg, rgba(72, 191, 193, 0.15), rgba(131, 168, 207, 0.07)),
    #141c24;
  border: 1px solid rgba(123, 214, 212, 0.22);
  color: var(--accent-strong);
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0;
  box-shadow: 0 10px 26px rgba(0, 0, 0, 0.24);
}
.brand-mark span { transform: translateY(-1px); }
.brand-title { font-size: 17px; font-weight: 750; }
.brand-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
nav { display: grid; gap: 7px; }
.nav-item {
  color: #cbd7e5;
  padding: 11px 12px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 650;
  border: 1px solid transparent;
}
.nav-item:hover { background: rgba(255, 255, 255, 0.06); border-color: var(--line); text-decoration: none; }
.nav-item.active {
  background: linear-gradient(135deg, rgba(72, 191, 193, 0.13), rgba(131, 168, 207, 0.06));
  border-color: rgba(123, 214, 212, 0.24);
  color: #f7fffd;
  box-shadow: var(--inner);
}
.sidebar-note {
  margin-top: auto;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.035);
  box-shadow: var(--inner);
}
.sidebar-note strong { display: block; font-size: 12px; margin-bottom: 5px; }
.sidebar-note span { display: block; color: var(--muted); font-size: 12px; line-height: 1.45; }
.page { min-width: 0; }
.topbar {
  min-height: 104px;
  padding: 22px 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  background: rgba(11, 16, 21, 0.84);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(10px);
  position: sticky;
  top: 0;
  z-index: 10;
}
h1 { margin: 0; font-size: 24px; line-height: 1.15; }
.topbar p, .panel-head p, .hero-panel p { margin: 5px 0 0; color: var(--muted); font-size: 13px; line-height: 1.45; }
.topbar-actions, .hero-actions { display: flex; gap: 10px; flex-wrap: wrap; }
main { padding: 22px 28px 36px; }
.system-chip {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  padding: 0 12px;
  border-radius: 6px;
  color: var(--accent-strong);
  background: rgba(72, 191, 193, 0.08);
  border: 1px solid rgba(123, 214, 212, 0.2);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}
.button, button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 36px;
  padding: 8px 13px;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 700;
  line-height: 1;
}
.button.primary, button[type="submit"] {
  background: linear-gradient(135deg, #4fbec0, #2f898b);
  color: #071113;
  border-color: rgba(123, 214, 212, 0.25);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
}
.button.primary:hover, button[type="submit"]:hover { filter: brightness(1.08); text-decoration: none; }
.button.secondary { background: rgba(255, 255, 255, 0.04); color: var(--text); border-color: var(--line-strong); }
.button.small {
  min-height: 28px;
  padding: 5px 9px;
  font-size: 11px;
  background: rgba(255, 255, 255, 0.05);
  color: var(--accent-strong);
  border-color: rgba(114, 242, 223, 0.24);
  box-shadow: none;
}
.button.danger {
  color: #ffd0c8;
  background: rgba(255, 122, 102, 0.13);
  border-color: rgba(255, 122, 102, 0.34);
}
.hero-panel {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 24px;
  border: 1px solid rgba(123, 214, 212, 0.16);
  border-radius: 8px;
  background:
    linear-gradient(90deg, rgba(72, 191, 193, 0.09), rgba(131, 168, 207, 0.05) 48%, rgba(211, 123, 73, 0.055)),
    linear-gradient(180deg, rgba(255, 255, 255, 0.06), transparent),
    #101923;
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
}
.hero-panel::after {
  content: "";
  position: absolute;
  inset: auto 0 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), transparent 42%, var(--amber));
  opacity: 0.52;
}
.hero-panel h2 { margin: 0; font-size: 26px; line-height: 1.15; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin: 16px 0;
}
.metric-card, .panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.metric-card {
  padding: 16px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.01)),
    var(--surface);
  box-shadow: var(--shadow), var(--inner);
  position: relative;
  overflow: hidden;
}
.metric-card::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), rgba(131, 168, 207, 0.26));
}
.metric-card span { display: block; color: var(--muted-strong); font-size: 12px; font-weight: 750; text-transform: uppercase; }
.metric-card strong { display: block; margin-top: 8px; font-size: 27px; line-height: 1; color: #ffffff; }
.metric-card small { display: block; margin-top: 8px; color: var(--muted); font-size: 12px; }
.content-grid { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.8fr); gap: 16px; }
.overview-runs-grid {
  grid-template-columns: minmax(0, 1fr);
}
.overview-runs-grid .wide {
  min-width: 0;
}
.panel {
  padding: 16px;
  min-width: 0;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.012)),
    var(--surface);
  box-shadow: var(--shadow), var(--inner);
}
.panel + .panel { margin-top: 16px; }
.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 13px;
}
.panel-head h2 { margin: 0; font-size: 17px; line-height: 1.2; color: #ffffff; }
.table-wrap {
  max-width: 100%;
  overflow-x: auto;
  overflow-y: visible;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-soft);
}
table { width: 100%; min-width: 1500px; border-collapse: collapse; font-size: 13px; background: transparent; }
th, td { border-bottom: 1px solid rgba(169, 190, 212, 0.11); padding: 10px 11px; text-align: left; vertical-align: middle; white-space: nowrap; }
th { background: rgba(255, 255, 255, 0.035); color: var(--muted-strong); font-size: 11px; text-transform: uppercase; font-weight: 800; }
td { color: #dce6f2; }
tbody tr:hover { background: rgba(29, 214, 195, 0.045); }
tbody tr:last-child td { border-bottom: 0; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
.run-link { display: block; color: #ffffff; font-weight: 750; }
.run-time { display: block; margin-top: 3px; color: var(--muted); font-size: 11px; }
.time-cell { color: #cbd6e3; font-size: 12px; }
.backend, .status-pill {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}
.backend-python { background: rgba(131, 168, 207, 0.12); color: #b8cde4; border: 1px solid rgba(131, 168, 207, 0.24); }
.backend-cpp { background: rgba(121, 191, 112, 0.11); color: #b9ddb2; border: 1px solid rgba(121, 191, 112, 0.24); }
.backend-unknown { background: rgba(145, 161, 180, 0.12); color: #c5d1df; border: 1px solid rgba(145, 161, 180, 0.22); }
.status-running { background: rgba(72, 191, 193, 0.12); color: var(--accent-strong); border: 1px solid rgba(123, 214, 212, 0.28); }
.status-done { background: rgba(145, 161, 180, 0.12); color: #cbd6e3; border: 1px solid rgba(145, 161, 180, 0.2); }
.status-warn { background: rgba(211, 123, 73, 0.13); color: #e8a978; border: 1px solid rgba(211, 123, 73, 0.26); }
.status-muted { background: rgba(145, 161, 180, 0.09); color: #9dadbf; border: 1px solid rgba(145, 161, 180, 0.16); }
.quality-bar { width: 96px; height: 7px; background: rgba(255, 255, 255, 0.08); border-radius: 999px; overflow: hidden; }
.quality-bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--amber)); border-radius: inherit; opacity: 0.9; }
.progress-cell {
  min-width: 142px;
  display: grid;
  gap: 5px;
}
.progress-cell small, .muted {
  color: var(--muted);
  font-size: 11px;
}
.progress-track {
  width: 142px;
  height: 8px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.04);
}
.progress-track span {
  display: block;
  height: 100%;
  min-width: 2px;
  background: linear-gradient(90deg, #48bfc1, #83a8cf);
  box-shadow: none;
  border-radius: inherit;
}
.row-actions { color: var(--muted); }
.row-actions a { margin-right: 8px; font-weight: 700; }
.run-actions {
  min-width: 96px;
}
.action-form {
  display: inline-flex;
  margin: 0 6px 0 0;
}
.processes { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.processes li { padding: 10px; background: rgba(255, 255, 255, 0.035); border: 1px solid var(--line); border-radius: 6px; line-height: 1.45; overflow-wrap: anywhere; }
.live-training-panel {
  margin-bottom: 16px;
}
.live-refresh {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}
.live-grid {
  display: grid;
  grid-template-columns: minmax(360px, 0.9fr) minmax(360px, 1.1fr);
  gap: 14px;
  align-items: start;
}
.live-chart-section {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
}
.section-caption {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}
.section-caption h3 {
  margin: 0;
  color: #ffffff;
  font-size: 15px;
}
.section-caption span {
  color: var(--muted);
  font-size: 12px;
}
.live-run-list {
  display: grid;
  gap: 10px;
  align-content: start;
  max-height: 440px;
  overflow: auto;
  padding-right: 4px;
}
.live-run-card {
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(5, 10, 16, 0.55);
}
.live-run-card header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.live-run-name {
  color: #ffffff;
  font-size: 13px;
  font-weight: 800;
  overflow-wrap: anywhere;
}
.live-run-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}
.live-run-card .progress-track {
  width: 100%;
}
.live-run-footer {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 10px;
}
.live-run-footer span {
  display: block;
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
  font-weight: 800;
}
.live-run-footer strong {
  display: block;
  margin-top: 3px;
  color: var(--text);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.live-metrics {
  min-width: 0;
  display: grid;
  gap: 12px;
}
.live-stat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}
.live-stat-grid article {
  display: grid;
  grid-template-columns: minmax(104px, 0.8fr) minmax(0, 1fr);
  align-items: center;
  gap: 10px;
  min-height: 46px;
  padding: 9px 11px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.035);
}
.live-stat-grid span {
  display: block;
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}
.live-stat-grid strong {
  display: block;
  margin-top: 0;
  color: #ffffff;
  font-size: 18px;
  line-height: 1;
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.live-chart-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.live-chart-card {
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-soft);
}
.live-chart-card > div {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 8px;
}
.live-chart-card strong {
  flex: 0 0 auto;
  color: #ffffff;
  font-size: 13px;
}
.live-chart-card span {
  min-width: 0;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.35;
  text-align: right;
}
.live-chart-card svg {
  display: block;
  width: 100%;
  height: 190px;
  border-radius: 6px;
  background:
    linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    rgba(4, 8, 13, 0.48);
  background-size: 100% 38px, 72px 100%, auto;
}
.live-chart-axis {
  stroke: rgba(168, 181, 194, 0.32);
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}
.live-chart-line {
  fill: none;
  stroke-width: 2.4;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
}
.live-chart-line-raw {
  opacity: 0.26;
  stroke-width: 1.2;
}
.live-chart-line-smooth {
  opacity: 0.96;
  stroke-width: 2.6;
}
.live-chart-dot {
  opacity: 0.38;
}
.live-chart-endpoint {
  stroke: #071018;
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
.live-chart-guide {
  stroke: rgba(168, 181, 194, 0.14);
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}
.live-chart-label {
  fill: #9dadbf;
  font-size: 10px;
}
.chart-legend {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: 8px;
}
.chart-legend i {
  display: inline-block;
  flex: 0 0 auto;
}
.legend-raw {
  width: 20px;
  height: 2px;
  background: rgba(72, 191, 193, 0.34);
}
.legend-smooth {
  width: 20px;
  height: 3px;
  background: #48bfc1;
}
.legend-current {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #48bfc1;
  box-shadow: 0 0 0 2px rgba(72, 191, 193, 0.18);
}
.history-layout {
  display: grid;
  grid-template-columns: minmax(280px, 0.28fr) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
.history-list {
  position: sticky;
  top: 126px;
  max-height: calc(100vh - 154px);
  overflow: hidden;
}
.history-run-list {
  display: grid;
  gap: 8px;
  max-height: calc(100vh - 244px);
  overflow: auto;
  padding-right: 4px;
}
.history-run {
  display: block;
  padding: 10px 11px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(5, 10, 16, 0.42);
}
.history-run:hover {
  border-color: rgba(123, 214, 212, 0.34);
  text-decoration: none;
}
.history-run.active {
  background: rgba(72, 191, 193, 0.10);
  border-color: rgba(123, 214, 212, 0.36);
}
.history-run strong {
  display: block;
  color: #ffffff;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.history-run span {
  display: block;
  margin-top: 5px;
  color: var(--muted);
  font-size: 11px;
}
.history-detail {
  display: grid;
  gap: 16px;
  min-width: 0;
}
.history-head {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: flex-start;
}
.history-head h2 {
  margin: 0;
  font-size: 19px;
  overflow-wrap: anywhere;
}
.history-head p {
  margin: 6px 0 0;
  color: var(--muted);
}
.history-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}
.history-metrics {
  margin: 0;
}
.history-chart-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.history-chart {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-soft);
  padding: 12px;
}
.history-chart > div {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}
.history-chart strong {
  color: #ffffff;
  font-size: 13px;
}
.history-chart span {
  color: var(--muted);
  font-size: 11px;
}
.history-chart svg {
  display: block;
  width: 100%;
  height: 190px;
  border-radius: 6px;
  background:
    linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    rgba(4, 8, 13, 0.48);
  background-size: 100% 38px, 72px 100%, auto;
}
.history-chart.empty {
  min-height: 140px;
  display: grid;
  align-content: center;
  gap: 6px;
}
.chart-guide {
  stroke: rgba(168, 181, 194, 0.14);
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}
.chart-raw {
  fill: none;
  stroke: rgba(72, 191, 193, 0.30);
  stroke-width: 1.1;
  vector-effect: non-scaling-stroke;
}
.chart-smooth {
  fill: none;
  stroke: #48bfc1;
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
}
.chart-label {
  fill: #9dadbf;
  font-size: 10px;
}
.histogram rect {
  fill: rgba(72, 191, 193, 0.72);
}
.history-report-frame {
  display: block;
  width: 100%;
  height: 860px;
  margin-top: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #081017;
}
.history-empty-report {
  margin-top: 14px;
  padding: 18px;
  border: 1px dashed var(--line-strong);
  border-radius: 8px;
  color: var(--muted);
  background: rgba(5, 10, 16, 0.42);
}
.train-workbench { display: grid; grid-template-columns: minmax(340px, 0.9fr) minmax(0, 1.1fr); gap: 16px; align-items: start; }
.launch-panel, .data-panel { grid-column: auto; }
.form-grid { display: grid; gap: 12px; }
.form-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.form-grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
label { display: block; color: var(--muted-strong); font-size: 12px; font-weight: 800; }
.check-row {
  margin-top: 14px;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  column-gap: 10px;
  row-gap: 3px;
  align-items: start;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.035);
}
.check-row input {
  width: auto;
  margin: 2px 0 0;
}
.check-row span {
  color: var(--text);
  font-size: 13px;
}
.check-row small {
  grid-column: 2;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.45;
}
input, select, textarea {
  width: 100%;
  margin-top: 6px;
  padding: 9px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(4, 8, 13, 0.72);
  color: var(--text);
  font: inherit;
  font-size: 13px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
textarea { resize: vertical; min-height: 88px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
input::placeholder, textarea::placeholder { color: #617085; }
input:focus, select:focus, textarea:focus { outline: 2px solid rgba(72, 191, 193, 0.16); border-color: rgba(123, 214, 212, 0.45); }
.quick-presets { display: flex; gap: 8px; margin-top: 14px; }
.quick-presets button { background: rgba(255, 255, 255, 0.04); color: var(--text); border-color: var(--line); }
.quick-presets button:hover { border-color: rgba(123, 214, 212, 0.28); color: var(--accent-strong); }
.sticky-submit {
  grid-column: 1 / -1;
  position: sticky;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  background: rgba(13, 20, 29, 0.92);
  border: 1px solid rgba(123, 214, 212, 0.18);
  border-radius: 8px;
  box-shadow: 0 -18px 44px rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(10px);
}
.sticky-submit span { color: var(--muted); font-size: 13px; }
.notice { padding: 12px 14px; margin: 0 0 16px; background: rgba(72, 191, 193, 0.1); border: 1px solid rgba(123, 214, 212, 0.22); color: var(--accent-strong); border-radius: 8px; font-weight: 650; }
.compare-layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 16px; }
.compare-picker select { min-height: 420px; }
.chart-panel { height: 580px; }
.chart-panel canvas { min-height: 500px; }
select option { background: #0d141d; color: var(--text); }
@media (max-width: 1180px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .content-grid, .train-workbench, .compare-layout, .live-grid, .history-layout { grid-template-columns: 1fr; }
  .history-list { position: static; max-height: none; }
  .history-run-list { max-height: 360px; }
  .history-head { flex-direction: column; }
  .history-actions { justify-content: flex-start; }
}
@media (max-width: 760px) {
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; }
  nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .topbar, .hero-panel, .sticky-submit { align-items: stretch; flex-direction: column; }
  main { padding: 16px; }
  .topbar { position: static; padding: 18px 16px; }
  .metric-grid, .form-grid.two, .form-grid.three, .live-stat-grid, .live-chart-grid, .history-chart-grid, .live-run-footer { grid-template-columns: 1fr; }
  .history-report-frame { height: 620px; }
}
"""


SCRIPT = """
function setField(name, value) {
  const field = document.querySelector(`[name="${name}"]`);
  if (field) field.value = value;
}

function installPresets() {
  const presets = {
    smoke: {epochs: 1, batch_size: 1, training_crop_size: 512, resize: 512, samples_per_pair: 256, max_train_batches: 40, memory_cache_items: 32, prefetch_batches: 2},
    balanced: {epochs: 2, batch_size: 1, training_crop_size: 768, resize: 768, samples_per_pair: 512, max_train_batches: 0, memory_cache_items: 128, prefetch_batches: 4},
    long: {epochs: 6, batch_size: 1, training_crop_size: 1024, resize: 1024, samples_per_pair: 768, max_train_batches: 0, memory_cache_items: 128, prefetch_batches: 4}
  };
  document.querySelectorAll('[data-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const preset = presets[button.dataset.preset] || {};
      Object.entries(preset).forEach(([name, value]) => setField(name, value));
    });
  });
}

function installAutoRefresh() {
  if (!['/', '/runs'].includes(window.location.pathname)) return;
  window.setTimeout(() => {
    if (document.visibilityState === 'visible') window.location.reload();
  }, 10000);
}

function installConfirmButtons() {
  document.querySelectorAll('[data-confirm]').forEach((button) => {
    button.addEventListener('click', (event) => {
      if (!window.confirm(button.dataset.confirm || '确认执行？')) {
        event.preventDefault();
      }
    });
  });
}

function numericValue(row, names) {
  for (const name of names) {
    const value = row[name];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return null;
}

function formatMetric(value, digits = 5) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
  if (Math.abs(value) >= 1000) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(3);
  return value.toPrecision(digits).replace(/0+$/, '').replace(/\\.$/, '');
}

function statusLabel(status) {
  return {
    running: '运行中',
    logged: '有日志',
    stopped: '已停止',
    missing: '未启动',
    invalid: 'PID 异常',
    unknown: '未知'
  }[status] || status || '未知';
}

function backendLabel(backend) {
  return {python: 'Python', cpp: 'C++', unknown: '未知'}[backend] || backend || '未知';
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[char]));
}

function metricFromLatest(latest, names) {
  return numericValue(latest || {}, names);
}

function integerMetric(latest, names) {
  const value = metricFromLatest(latest, names);
  return value === null ? null : Math.max(0, Math.round(value));
}

function formatProgressPart(current, total, unit) {
  if (current === null && total === null) return '-';
  if (current !== null && total !== null && total > 0) return `${current}/${total} ${unit}`;
  if (current !== null) return `${current} ${unit}`;
  return `-/${total} ${unit}`;
}

function runEpochLabel(run) {
  const latest = run.latest_metrics || {};
  const current = integerMetric(latest, ['epoch']);
  const total = integerMetric(latest, ['total_epochs']);
  return formatProgressPart(current, total, '轮');
}

function runBatchLabel(run) {
  const latest = run.latest_metrics || {};
  const current = integerMetric(latest, ['iteration', 'batch', 'step', 'global_step']);
  const total = integerMetric(latest, ['total_batches', 'total_iterations']);
  return formatProgressPart(current, total, '批');
}

function rowStep(row, index) {
  return numericValue(row, ['step', 'global_step', 'batch', 'iteration']) || index + 1;
}

function runMetricRows(metricsPayload, runName) {
  return (((metricsPayload.metrics || {})[runName] || {}).rows || []);
}

const LIVE_CHART_WINDOW_BATCHES = 300;
const LIVE_CHART_MAX_DOTS = 80;
const LIVE_CHART_METRICS = {
  loss: ['loss', 'loss_total', 'total_loss', 'train_loss'],
  top1: ['descriptor_accuracy', 'top1_accuracy', 'top1', 'mean_top1'],
  stop_confidence: ['graph_matcher_stop_confidence_loss', 'stop_confidence_loss'],
  graph_prune: ['graph_matcher_prune_ranking_loss', 'prune_ranking_loss']
};

function visibleMetricRows(rows) {
  return rows.slice(Math.max(0, rows.length - LIVE_CHART_WINDOW_BATCHES));
}

function chartPoints(metricsPayload, selectedRuns, names) {
  return selectedRuns.map((run) => {
    const rows = runMetricRows(metricsPayload, run.name);
    const startIndex = Math.max(0, rows.length - LIVE_CHART_WINDOW_BATCHES);
    const visibleRows = rows.slice(startIndex);
    return {
      name: run.name,
      color: run.backend === 'cpp' ? '#48bfc1' : '#83a8cf',
      points: visibleRows.map((row, index) => ({
        x: startIndex + index + 1,
        y: numericValue(row, names)
      })).filter((point) => point.y !== null)
    };
  }).filter((series) => series.points.length);
}

function movingAveragePoints(points) {
  const radius = Math.max(2, Math.min(12, Math.floor(points.length / 28)));
  return points.map((point, index) => {
    const start = Math.max(0, index - radius);
    const end = Math.min(points.length, index + radius + 1);
    const window = points.slice(start, end);
    const average = window.reduce((sum, item) => sum + item.y, 0) / Math.max(1, window.length);
    return {x: point.x, y: average};
  });
}

function pathForPoints(points, xScale, yScale) {
  return points.map((point, index) => {
    const command = index === 0 ? 'M' : 'L';
    return `${command}${xScale(point.x).toFixed(1)},${yScale(point.y).toFixed(1)}`;
  }).join(' ');
}

function renderLiveChart(svg, seriesList) {
  if (!svg) return;
  const width = 520;
  const height = 220;
  const pad = {left: 52, right: 18, top: 18, bottom: 34};
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const points = seriesList.flatMap((series) => series.points);
  if (!points.length) {
    svg.innerHTML = '<text class="live-chart-label" x="18" y="84">暂无可绘制指标</text>';
    return;
  }
  const xMin = Math.min(...points.map((point) => point.x));
  const xMax = Math.max(...points.map((point) => point.x));
  const rawYMin = Math.min(...points.map((point) => point.y));
  const rawYMax = Math.max(...points.map((point) => point.y));
  const yPadding = Math.max(1.0e-9, (rawYMax - rawYMin) * 0.08);
  const yMin = rawYMin - yPadding;
  const yMax = rawYMax + yPadding;
  const xScale = (value) => pad.left + ((value - xMin) / Math.max(1, xMax - xMin)) * plotW;
  const yScale = (value) => pad.top + (1 - ((value - yMin) / Math.max(1e-9, yMax - yMin))) * plotH;
  const guides = [0.25, 0.5, 0.75].map((ratio) => {
    const y = pad.top + ratio * plotH;
    return `<line class="live-chart-guide" x1="${pad.left}" y1="${y.toFixed(1)}" x2="${pad.left + plotW}" y2="${y.toFixed(1)}"></line>`;
  }).join('');
  const lines = seriesList.map((series) => {
    const rawPath = pathForPoints(series.points, xScale, yScale);
    const smoothPath = pathForPoints(movingAveragePoints(series.points), xScale, yScale);
    return `<path class="live-chart-line live-chart-line-raw" d="${rawPath}" stroke="${series.color}"></path>
      <path class="live-chart-line live-chart-line-smooth" d="${smoothPath}" stroke="${series.color}"></path>`;
  }).join('');
  const dots = seriesList.map((series) => {
    const stride = Math.max(1, Math.ceil(series.points.length / LIVE_CHART_MAX_DOTS));
    return series.points
      .filter((point, index) => index % stride === 0 || index === series.points.length - 1)
      .map((point) => (
        `<circle class="live-chart-dot" cx="${xScale(point.x).toFixed(1)}" cy="${yScale(point.y).toFixed(1)}" r="2.2" fill="${series.color}"></circle>`
      )).join('');
  }).join('');
  const endpoints = seriesList.map((series) => {
    const point = series.points[series.points.length - 1];
    if (!point) return '';
    return `<circle class="live-chart-endpoint" cx="${xScale(point.x).toFixed(1)}" cy="${yScale(point.y).toFixed(1)}" r="4.2" fill="${series.color}"></circle>
      <text class="live-chart-label" x="${Math.min(width - 78, xScale(point.x) + 8).toFixed(1)}" y="${Math.max(14, yScale(point.y) - 8).toFixed(1)}">${formatMetric(point.y, 4)}</text>`;
  }).join('');
  svg.innerHTML = `
    ${guides}
    <line class="live-chart-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotH}"></line>
    <line class="live-chart-axis" x1="${pad.left}" y1="${pad.top + plotH}" x2="${pad.left + plotW}" y2="${pad.top + plotH}"></line>
    <text class="live-chart-label" x="4" y="${pad.top + 4}">${formatMetric(yMax, 4)}</text>
    <text class="live-chart-label" x="4" y="${pad.top + plotH}">${formatMetric(yMin, 4)}</text>
    <text class="live-chart-label" x="${pad.left}" y="${height - 10}">batch ${formatMetric(xMin, 4)}</text>
    <text class="live-chart-label" x="${pad.left + plotW - 58}" y="${height - 10}">batch ${formatMetric(xMax, 4)}</text>
    ${lines}
    ${dots}
    ${endpoints}
  `;
}

function liveChartRun(selectedRuns) {
  return selectedRuns.find((run) => run.status === 'running') || selectedRuns[0] || null;
}

function updateChartMeta(chartKey, run, metricsPayload, names) {
  const element = document.querySelector(`[data-live-chart-meta="${chartKey}"]`);
  if (!element) return;
  if (!run) {
    element.textContent = '等待任务';
    return;
  }
  const series = chartPoints(metricsPayload, [run], names)[0];
  if (!series || !series.points.length) {
    const rows = runMetricRows(metricsPayload, run.name);
    const visibleCount = visibleMetricRows(rows).length;
    element.textContent = `${run.name.slice(0, 24)} · 最近 ${visibleCount} batch · 无当前指标`;
    return;
  }
  const current = series.points[series.points.length - 1];
  const smoothPoints = movingAveragePoints(series.points);
  const smooth = smoothPoints[smoothPoints.length - 1] || current;
  element.textContent = `batch ${Math.round(current.x)} · 当前 ${formatMetric(current.y, 4)} · 平滑 ${formatMetric(smooth.y, 4)}`;
}

function renderLiveRuns(container, runs) {
  if (!container) return;
  if (!runs.length) {
    container.innerHTML = '<p class="muted">暂时没有训练任务。启动训练后这里会实时显示进度。</p>';
    return;
  }
  container.innerHTML = runs.map((run) => {
    const latest = run.latest_metrics || {};
    const runName = escapeHtml(run.name);
    const backend = ['python', 'cpp', 'unknown'].includes(run.backend) ? run.backend : 'unknown';
    const status = run.status === 'running' ? 'running' : 'done';
    const loss = formatMetric(metricFromLatest(latest, ['loss', 'loss_total', 'total_loss', 'train_loss']));
    const top1 = formatMetric(metricFromLatest(latest, ['descriptor_accuracy', 'top1_accuracy', 'top1', 'mean_top1']));
    const rank = formatMetric(metricFromLatest(latest, ['descriptor_positive_rank', 'mean_positive_rank', 'mean_rank']));
    const rows = Object.keys(latest).length ? '已更新' : '无指标';
    const epoch = runEpochLabel(run);
    const batch = runBatchLabel(run);
    const percent = Math.max(0, Math.min(100, Number(run.progress_percent) || 0));
    return `
      <article class="live-run-card">
        <header>
          <div>
            <a class="live-run-name" href="/compare?runs=${encodeURIComponent(run.name)}">${runName}</a>
            <div class="live-run-meta">
              <span class="backend backend-${backend}">${backendLabel(run.backend)}</span>
              <span class="status-pill status-${status}">${statusLabel(run.status)}</span>
            </div>
          </div>
          <a href="/runs/${encodeURIComponent(run.name)}/log">日志</a>
        </header>
        <div class="progress-cell">
          <div class="progress-track"><span style="width:${percent.toFixed(1)}%"></span></div>
          <small>${run.progress_label || '未开始'} · ${percent.toFixed(1)}%</small>
        </div>
        <div class="live-run-footer">
          <div><span>Epoch</span><strong>${epoch}</strong></div>
          <div><span>Batch</span><strong>${batch}</strong></div>
          <div><span>损失</span><strong>${loss}</strong></div>
          <div><span>Top1</span><strong>${top1}</strong></div>
          <div><span>排名</span><strong>${rank}</strong></div>
          <div><span>状态</span><strong>${rows}</strong></div>
        </div>
      </article>
    `;
  }).join('');
}

async function refreshLiveTraining() {
  const panel = document.querySelector('[data-live-training]');
  if (!panel) return;
  const runsResponse = await fetch('/api/runs', {cache: 'no-store'});
  const runsPayload = await runsResponse.json();
  const allRuns = runsPayload.runs || [];
  const selectedRuns = allRuns
    .filter((run) => run.status === 'running')
    .concat(allRuns.filter((run) => run.status !== 'running'))
    .slice(0, 4);
  const runningCount = allRuns.filter((run) => run.status === 'running').length;
  const query = selectedRuns.map((run) => 'runs=' + encodeURIComponent(run.name)).join('&');
  const metricsPayload = query ? await fetch('/api/metrics?' + query, {cache: 'no-store'}).then((response) => response.json()) : {metrics: {}};
  renderLiveRuns(document.querySelector('[data-live-runs]'), selectedRuns);
  const chartRun = liveChartRun(selectedRuns);
  const chartRuns = chartRun ? [chartRun] : [];
  const newest = selectedRuns[0] || {};
  const latest = newest.latest_metrics || {};
  const newestRows = selectedRuns.reduce((count, run) => count + runMetricRows(metricsPayload, run.name).length, 0);
  const setText = (selector, value) => {
    const element = document.querySelector(selector);
    if (element) element.textContent = value;
  };
  setText('[data-live-running]', String(runningCount));
  setText('[data-live-epoch]', chartRun ? runEpochLabel(chartRun) : '-');
  setText('[data-live-batch]', chartRun ? runBatchLabel(chartRun) : '-');
  setText('[data-live-loss]', formatMetric(metricFromLatest(latest, ['loss', 'loss_total', 'total_loss', 'train_loss'])));
  setText('[data-live-top1]', formatMetric(metricFromLatest(latest, ['descriptor_accuracy', 'top1_accuracy', 'top1', 'mean_top1'])));
  setText('[data-live-rows]', String(newestRows));
  setText('[data-live-updated]', '更新 ' + new Date().toLocaleTimeString('zh-CN', {hour12: false}));
  Object.entries(LIVE_CHART_METRICS).forEach(([chartKey, names]) => updateChartMeta(chartKey, chartRun, metricsPayload, names));
  renderLiveChart(document.querySelector('[data-live-chart="loss"]'), chartPoints(metricsPayload, chartRuns, LIVE_CHART_METRICS.loss));
  renderLiveChart(document.querySelector('[data-live-chart="top1"]'), chartPoints(metricsPayload, chartRuns, LIVE_CHART_METRICS.top1));
  renderLiveChart(document.querySelector('[data-live-chart="stop_confidence"]'), chartPoints(metricsPayload, chartRuns, LIVE_CHART_METRICS.stop_confidence));
  renderLiveChart(document.querySelector('[data-live-chart="graph_prune"]'), chartPoints(metricsPayload, chartRuns, LIVE_CHART_METRICS.graph_prune));
}

function installLiveTraining() {
  const panel = document.querySelector('[data-live-training]');
  if (!panel) return;
  const refresh = () => {
    refreshLiveTraining().catch((error) => {
      const updated = document.querySelector('[data-live-updated]');
      if (updated) updated.textContent = '刷新失败：' + error.message;
    });
  };
  const button = document.querySelector('[data-live-refresh]');
  if (button) button.addEventListener('click', refresh);
  refresh();
  window.setInterval(() => {
    if (document.visibilityState === 'visible') refresh();
  }, 2000);
}

async function loadCompareChart() {
  const canvas = document.getElementById('metricChart');
  if (!canvas) return;
  const runs = (canvas.dataset.runs || '').split(',').filter(Boolean);
  const context = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  context.scale(scale, scale);
  context.clearRect(0, 0, rect.width, rect.height);
  context.fillStyle = '#91a1b4';
  context.font = '13px system-ui, sans-serif';
  if (!runs.length) {
    context.fillText('请选择一个或多个任务绘制指标曲线。', 24, 36);
    return;
  }
  const response = await fetch('/api/metrics?' + runs.map(run => 'runs=' + encodeURIComponent(run)).join('&'));
  const payload = await response.json();
  const colors = ['#48bfc1', '#83a8cf', '#d37b49', '#79bf70', '#a58bbd'];
  const datasets = [];
  runs.forEach((run, index) => {
    const rows = (payload.metrics[run] || {}).rows || [];
    const points = rows.map((row, step) => ({
      x: numericValue(row, ['step', 'global_step', 'iteration', 'epoch']) || step + 1,
      y: numericValue(row, ['loss', 'loss_total', 'total_loss', 'train_loss'])
    })).filter(point => point.y !== null);
    datasets.push({label: run + ' 损失', data: points, color: colors[index % colors.length]});
  });
  const allPoints = datasets.flatMap(dataset => dataset.data);
  if (!allPoints.length) {
    context.fillText('所选任务暂时没有可绘制的损失指标。', 24, 36);
    return;
  }
  const padding = {left: 58, right: 28, top: 28, bottom: 54};
  const plotWidth = rect.width - padding.left - padding.right;
  const plotHeight = rect.height - padding.top - padding.bottom;
  const xMin = Math.min(...allPoints.map(point => point.x));
  const xMax = Math.max(...allPoints.map(point => point.x));
  const yMin = Math.min(...allPoints.map(point => point.y));
  const yMax = Math.max(...allPoints.map(point => point.y));
  const xScale = (value) => padding.left + ((value - xMin) / Math.max(1, xMax - xMin)) * plotWidth;
  const yScale = (value) => padding.top + (1 - ((value - yMin) / Math.max(1e-9, yMax - yMin))) * plotHeight;
  context.strokeStyle = 'rgba(168, 181, 194, 0.22)';
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + plotHeight);
  context.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  context.stroke();
  context.fillStyle = '#91a1b4';
  context.font = '12px system-ui, sans-serif';
  context.fillText(`损失 ${yMax.toFixed(4)}`, 10, padding.top + 5);
  context.fillText(`损失 ${yMin.toFixed(4)}`, 10, padding.top + plotHeight);
  context.fillText(`步数 ${xMin}`, padding.left, rect.height - 18);
  context.fillText(`步数 ${xMax}`, padding.left + plotWidth - 64, rect.height - 18);
  datasets.forEach((dataset, index) => {
    if (!dataset.data.length) return;
    context.strokeStyle = dataset.color;
    context.lineWidth = 2;
    context.beginPath();
    dataset.data.forEach((point, pointIndex) => {
      const x = xScale(point.x);
      const y = yScale(point.y);
      if (pointIndex === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    const legendY = 18 + index * 18;
    context.fillStyle = dataset.color;
    context.fillRect(padding.left + index * 210, legendY - 9, 10, 10);
    context.fillStyle = '#edf4fb';
    context.fillText(dataset.label.slice(0, 24), padding.left + 15 + index * 210, legendY);
  });
}
installPresets();
installAutoRefresh();
installConfirmButtons();
installLiveTraining();
loadCompareChart();
"""


def make_server(host: str, port: int, project_root: Path | None = None) -> ThreadingHTTPServer:
    handler = type("ConfiguredDashboardHandler", (DashboardHandler,), {"project_root": project_root or Path.cwd()})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="PFM Lab Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    server = make_server(args.host, args.port, project_root=args.project_root.resolve())
    print(f"PFM Lab Dashboard: http://{args.host}:{server.server_address[1]}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
