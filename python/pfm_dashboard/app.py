from __future__ import annotations

import argparse
import html
import json
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
        top1 = _metric_number(run, "descriptor_accuracy", "top1", "mean_top1")
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
            f"<td>{_metric(run, 'descriptor_accuracy', 'top1', 'mean_top1')}</td>"
            f"<td>{_metric(run, 'descriptor_positive_rank', 'mean_rank')}</td>"
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
        <article><span>最新损失</span><strong data-live-loss>-</strong></article>
        <article><span>最新 Top1</span><strong data-live-top1>-</strong></article>
        <article><span>指标行数</span><strong data-live-rows>0</strong></article>
      </div>
      <div class="live-chart-grid">
        <div class="live-chart-card"><div><strong>损失</strong><span data-live-chart-meta="loss">最近 300 batch</span></div><svg data-live-chart="loss" viewBox="0 0 520 220" preserveAspectRatio="none"></svg></div>
        <div class="live-chart-card"><div><strong>Top1</strong><span data-live-chart-meta="top1">最近 300 batch</span></div><svg data-live-chart="top1" viewBox="0 0 520 220" preserveAspectRatio="none"></svg></div>
        <div class="live-chart-card"><div><strong>图匹配</strong><span data-live-chart-meta="graph">最近 300 batch</span></div><svg data-live-chart="graph" viewBox="0 0 520 220" preserveAspectRatio="none"></svg></div>
        <div class="live-chart-card"><div><strong>排名</strong><span data-live-chart-meta="rank">最近 300 batch</span></div><svg data-live-chart="rank" viewBox="0 0 520 220" preserveAspectRatio="none"></svg></div>
      </div>
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
    <div class="panel-head"><div><h2>指标曲线</h2><p>损失曲线来自各任务的 metrics.csv。</p></div></div>
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
                name: read_metrics_csv(root / "runs" / name / "metrics.csv").__dict__
                for name in names
                if (root / "runs" / name).exists()
            }
            self._send_json({"metrics": metrics})
        elif parsed.path.startswith("/runs/") and parsed.path.endswith("/log"):
            name = parsed.path.split("/")[2]
            self._send_text(tail_text(root / "runs" / name / "train.log", lines=200))
        elif parsed.path.startswith("/runs/") and parsed.path.endswith("/report"):
            name = parsed.path.split("/")[2]
            report = root / "runs" / name / "run.html"
            self._send_html(report.read_text(encoding="utf-8") if report.exists() else "报告缺失")
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
  grid-template-columns: minmax(320px, 0.86fr) minmax(0, 1.44fr);
  gap: 14px;
}
.live-run-list {
  display: grid;
  gap: 10px;
  align-content: start;
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
  grid-template-columns: repeat(4, minmax(0, 1fr));
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
}
.live-metrics {
  min-width: 0;
  display: grid;
  gap: 12px;
}
.live-stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
.live-stat-grid article {
  padding: 12px;
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
  margin-top: 6px;
  color: #ffffff;
  font-size: 21px;
  line-height: 1;
}
.live-chart-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
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
  gap: 10px;
  margin-bottom: 8px;
}
.live-chart-card strong {
  color: #ffffff;
  font-size: 13px;
}
.live-chart-card span {
  color: var(--muted);
  font-size: 11px;
}
.live-chart-card svg {
  display: block;
  width: 100%;
  height: 220px;
  border-radius: 6px;
  background:
    linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    rgba(4, 8, 13, 0.48);
  background-size: 100% 44px, 86px 100%, auto;
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
.live-chart-dot {
  opacity: 0.74;
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
  .content-grid, .train-workbench, .compare-layout, .live-grid { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; }
  nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .topbar, .hero-panel, .sticky-submit { align-items: stretch; flex-direction: column; }
  main { padding: 16px; }
  .topbar { position: static; padding: 18px 16px; }
  .metric-grid, .form-grid.two, .form-grid.three, .live-stat-grid, .live-chart-grid, .live-run-footer { grid-template-columns: 1fr; }
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

function rowStep(row, index) {
  return numericValue(row, ['step', 'global_step', 'batch', 'epoch']) || index + 1;
}

function runMetricRows(metricsPayload, runName) {
  return (((metricsPayload.metrics || {})[runName] || {}).rows || []);
}

const LIVE_CHART_WINDOW_BATCHES = 300;
const LIVE_CHART_MAX_DOTS = 80;

function visibleMetricRows(rows) {
  return rows.slice(Math.max(0, rows.length - LIVE_CHART_WINDOW_BATCHES));
}

function chartPoints(metricsPayload, selectedRuns, names) {
  return selectedRuns.map((run) => {
    const rows = runMetricRows(metricsPayload, run.name);
    const visibleRows = visibleMetricRows(rows);
    return {
      name: run.name,
      color: run.backend === 'cpp' ? '#48bfc1' : '#83a8cf',
      points: visibleRows.map((row, index) => ({
        x: rowStep(row, index),
        y: numericValue(row, names)
      })).filter((point) => point.y !== null)
    };
  }).filter((series) => series.points.length);
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
  const yMin = Math.min(...points.map((point) => point.y));
  const yMax = Math.max(...points.map((point) => point.y));
  const xScale = (value) => pad.left + ((value - xMin) / Math.max(1, xMax - xMin)) * plotW;
  const yScale = (value) => pad.top + (1 - ((value - yMin) / Math.max(1e-9, yMax - yMin))) * plotH;
  const guides = [0.25, 0.5, 0.75].map((ratio) => {
    const y = pad.top + ratio * plotH;
    return `<line class="live-chart-guide" x1="${pad.left}" y1="${y.toFixed(1)}" x2="${pad.left + plotW}" y2="${y.toFixed(1)}"></line>`;
  }).join('');
  const lines = seriesList.map((series) => {
    const path = series.points.map((point, index) => {
      const command = index === 0 ? 'M' : 'L';
      return `${command}${xScale(point.x).toFixed(1)},${yScale(point.y).toFixed(1)}`;
    }).join(' ');
    return `<path class="live-chart-line" d="${path}" stroke="${series.color}"></path>`;
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

function updateChartMeta(chartKey, run, metricsPayload) {
  const element = document.querySelector(`[data-live-chart-meta="${chartKey}"]`);
  if (!element) return;
  if (!run) {
    element.textContent = '等待任务';
    return;
  }
  const rows = runMetricRows(metricsPayload, run.name);
  const visibleCount = visibleMetricRows(rows).length;
  element.textContent = `${run.name.slice(0, 28)} · 最近 ${visibleCount} batch`;
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
  setText('[data-live-loss]', formatMetric(metricFromLatest(latest, ['loss', 'loss_total', 'total_loss', 'train_loss'])));
  setText('[data-live-top1]', formatMetric(metricFromLatest(latest, ['descriptor_accuracy', 'top1_accuracy', 'top1', 'mean_top1'])));
  setText('[data-live-rows]', String(newestRows));
  setText('[data-live-updated]', '更新 ' + new Date().toLocaleTimeString('zh-CN', {hour12: false}));
  ['loss', 'top1', 'graph', 'rank'].forEach((chartKey) => updateChartMeta(chartKey, chartRun, metricsPayload));
  renderLiveChart(document.querySelector('[data-live-chart="loss"]'), chartPoints(metricsPayload, chartRuns, ['loss', 'loss_total', 'total_loss', 'train_loss']));
  renderLiveChart(document.querySelector('[data-live-chart="top1"]'), chartPoints(metricsPayload, chartRuns, ['descriptor_accuracy', 'top1_accuracy', 'top1', 'mean_top1']));
  renderLiveChart(document.querySelector('[data-live-chart="graph"]'), chartPoints(metricsPayload, chartRuns, ['graph_matching_accuracy', 'graph_accuracy', 'mean_graph_accuracy']));
  renderLiveChart(document.querySelector('[data-live-chart="rank"]'), chartPoints(metricsPayload, chartRuns, ['descriptor_positive_rank', 'mean_positive_rank', 'mean_rank']));
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
