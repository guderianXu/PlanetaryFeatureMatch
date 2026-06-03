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
from urllib.parse import parse_qs, quote, urlparse

from .commands import TrainingRequest, create_training_runs, start_generated_run
from .models import RunSummary
from .services import active_training_processes, discover_runs, read_metrics_csv, summarize_dataset, tail_text


DEFAULT_DATASETS = [
    Path("/media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据_regen/samepos_hx1_10view_2048_721/cache"),
    Path("/media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据_regen/crossres_lowdom_hx1_10view_2048_1200/cache"),
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


def _status_class(status: str) -> str:
    if status == "running":
        return "status-running"
    if status in {"logged", "stopped"}:
        return "status-done"
    if status in {"invalid", "unknown"}:
        return "status-warn"
    return "status-muted"


def _nav_item(path: str, label: str, active: str) -> str:
    selected = " active" if active == path else ""
    return f'<a class="nav-item{selected}" href="{path}">{label}</a>'


def _layout(title: str, body: str, active: str = "/") -> str:
    nav = "".join(
        [
            _nav_item("/", "Overview", active),
            _nav_item("/train", "Train", active),
            _nav_item("/runs", "Runs", active),
            _nav_item("/compare", "Compare", active),
            _nav_item("/datasets", "Datasets", active),
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
        <div class="brand-mark">PFM</div>
        <div>
          <div class="brand-title">PFM Lab</div>
          <div class="brand-subtitle">Training Console</div>
        </div>
      </div>
      <nav>{nav}</nav>
      <div class="sidebar-note">
        <strong>Local only</strong>
        <span>127.0.0.1 training control for Python and C++ experiments.</span>
      </div>
    </aside>
    <div class="page">
      <header class="topbar">
        <div>
          <h1>{title}</h1>
          <p>Planetary feature matching training, simulation checks, and run comparison.</p>
        </div>
        <div class="topbar-actions">
          <a class="button secondary" href="/runs">View runs</a>
          <a class="button primary" href="/train">New train</a>
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
        report = f'<a href="/runs/{run.name}/report">report</a>' if run.has_report else "-"
        log = f'<a href="/runs/{run.name}/log">log</a>' if run.has_log else "-"
        loss = _metric_number(run, "loss", "total_loss")
        top1 = _metric_number(run, "descriptor_accuracy", "top1", "mean_top1")
        quality = top1 if top1 is not None else loss
        quality_width = 0
        if quality is not None:
            quality_width = max(4, min(100, int((1.0 - quality) * 100 if quality == loss else quality * 100)))
        rows.append(
            "<tr>"
            f"<td><a class=\"run-link\" href=\"/compare?runs={encoded_name}\">{escaped_name}</a>"
            f"<span class=\"run-time\">{_format_time(run.updated_at)}</span></td>"
            f"<td><span class=\"backend backend-{html.escape(run.backend)}\">{html.escape(run.backend)}</span></td>"
            f"<td><span class=\"status-pill {_status_class(run.status)}\">{html.escape(run.status)}</span></td>"
            f"<td>{_metric(run, 'loss', 'total_loss')}</td>"
            f"<td>{_metric(run, 'descriptor_accuracy', 'top1', 'mean_top1')}</td>"
            f"<td>{_metric(run, 'descriptor_positive_rank', 'mean_rank')}</td>"
            f"<td><div class=\"quality-bar\"><span style=\"width:{quality_width}%\"></span></div></td>"
            f"<td>{run.checkpoint_count}</td>"
            f"<td class=\"row-actions\">{log} {report}</td>"
            "</tr>"
        )
    return (
        "<div class=\"table-wrap\"><table><thead><tr><th>Run</th><th>Backend</th><th>Status</th><th>Loss</th>"
        "<th>Top1</th><th>Rank</th><th>Signal</th><th>CKPT</th><th>Artifacts</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</div>"
    )


def render_index(project_root: Path) -> str:
    runs = discover_runs(project_root / "runs")
    active = active_training_processes()
    disk = shutil.disk_usage(project_root)
    active_rows = "".join(f"<li><code>{html.escape(line)}</code></li>" for line in active) or "<li>No active training process</li>"
    running_count = sum(1 for run in runs if run.status == "running")
    checkpoint_count = sum(run.checkpoint_count for run in runs)
    latest_loss = _metric(runs[0], "loss", "total_loss") if runs else "-"
    disk_used_percent = int((disk.used / disk.total) * 100) if disk.total else 0
    body = f"""
<section class="hero-panel">
  <div>
    <h2>Training operations cockpit</h2>
    <p>Start matched Python/C++ experiments, watch live run state, inspect metrics, and keep HTML records in one place.</p>
  </div>
  <div class="hero-actions">
    <a class="button primary" href="/train">Launch compare run</a>
    <a class="button secondary" href="/compare">Open comparison</a>
  </div>
</section>
<section class="metric-grid">
  <article class="metric-card"><span>Runs</span><strong>{len(runs)}</strong><small>{running_count} running</small></article>
  <article class="metric-card"><span>Latest loss</span><strong>{latest_loss}</strong><small>from newest run</small></article>
  <article class="metric-card"><span>Checkpoints</span><strong>{checkpoint_count}</strong><small>discovered artifacts</small></article>
  <article class="metric-card"><span>Disk free</span><strong>{_format_bytes(disk.free)}</strong><small>{disk_used_percent}% used</small></article>
</section>
<section class="content-grid">
  <article class="panel wide">
    <div class="panel-head"><div><h2>Recent Runs</h2><p>Newest experiments and training artifacts.</p></div><a href="/runs">All runs</a></div>
    {_runs_table(runs)}
  </article>
  <article class="panel">
    <div class="panel-head"><div><h2>Active Processes</h2><p>Simulation and training commands currently visible to pgrep.</p></div></div>
    <ul class="processes">{active_rows}</ul>
  </article>
</section>
"""
    return _layout("Overview", body, active="/")


def render_train(project_root: Path, message: str = "") -> str:
    default_cache = "\n".join(str(path / "train") for path in DEFAULT_DATASETS if (path / "train").exists())
    notice = f'<p class="notice">{html.escape(message)}</p>' if message else ""
    body = f"""
{notice}
<form method="post" action="/train" class="train-workbench">
  <section class="panel launch-panel">
    <div class="panel-head"><div><h2>Experiment</h2><p>One submit can launch Python, C++, or both for direct comparison.</p></div></div>
    <div class="form-grid two">
      <label>Experiment name <input name="experiment_name" value="dashboard_compare"></label>
      <label>Backend
        <select name="backend">
          <option value="compare">Python + C++ compare</option>
          <option value="python">Python only</option>
          <option value="cpp">C++ only</option>
        </select>
      </label>
      <label>Device <input name="device" value="cuda"></label>
      <label>Init checkpoint <input name="init_checkpoint" placeholder="optional path"></label>
    </div>
    <div class="quick-presets">
      <button type="button" data-preset="smoke">Smoke</button>
      <button type="button" data-preset="balanced">Balanced</button>
      <button type="button" data-preset="long">Long run</button>
    </div>
  </section>
  <section class="panel data-panel">
    <div class="panel-head"><div><h2>Data</h2><p>Cache directories are passed directly to Python and C++ launch scripts.</p></div><a href="/datasets">Inspect datasets</a></div>
    <label>Training cache dirs<textarea name="cache_dirs" rows="7">{html.escape(default_cache)}</textarea></label>
    <label>Validation cache dirs<textarea name="validation_cache_dirs" rows="3" placeholder="optional, one path per line"></textarea></label>
  </section>
  <section class="panel">
    <div class="panel-head"><div><h2>Training Parameters</h2><p>Core optimizer and crop settings.</p></div></div>
    <div class="form-grid three">
      <label>Epochs <input type="number" name="epochs" value="1" min="1"></label>
      <label>Batch <input type="number" name="batch_size" value="1" min="1"></label>
      <label>Crop <input type="number" name="training_crop_size" value="512" min="0"></label>
      <label>Resize <input type="number" name="resize" value="512" min="0"></label>
      <label>Samples <input type="number" name="samples_per_pair" value="512" min="1"></label>
      <label>Learning rate <input name="learning_rate" value="3e-5"></label>
      <label>Profile <input name="profile" value="python-compare"></label>
      <label>Max batches <input type="number" name="max_train_batches" value="0" min="0"></label>
    </div>
  </section>
  <section class="panel">
    <div class="panel-head"><div><h2>Cache & Loader</h2><p>Keep GPU fed by preloading pair tensors into memory.</p></div></div>
    <div class="form-grid three">
      <label>Memory cache items <input type="number" name="memory_cache_items" value="64" min="0"></label>
      <label>Prefetch batches <input type="number" name="prefetch_batches" value="4" min="1"></label>
      <label>Python workers <input type="number" name="prefetch_workers" value="2" min="0"></label>
      <label>C++ loader workers <input type="number" name="dataloader_workers" value="2" min="0"></label>
    </div>
  </section>
  <div class="sticky-submit">
    <span>Generated scripts and HTML records will be written under <code>runs/</code>.</span>
    <button class="button primary" type="submit">Launch training</button>
  </div>
</form>
"""
    return _layout("Train", body, active="/train")


def render_runs(project_root: Path) -> str:
    runs = discover_runs(project_root / "runs")
    body = f"""
<section class="panel">
  <div class="panel-head"><div><h2>Runs</h2><p>All discovered training folders under <code>runs/</code>.</p></div><a href="/train">New run</a></div>
  {_runs_table(runs)}
</section>
"""
    return _layout("Runs", body, active="/runs")


def render_compare(project_root: Path, query: dict[str, list[str]]) -> str:
    selected = query.get("runs", [])
    runs = discover_runs(project_root / "runs")
    options = "".join(
        f'<option value="{html.escape(run.name)}" {"selected" if run.name in selected else ""}>{html.escape(run.name)}</option>' for run in runs
    )
    body = f"""
<section class="compare-layout">
  <form method="get" action="/compare" class="panel compare-picker">
    <div class="panel-head"><div><h2>Compare Runs</h2><p>Select Python and C++ runs to overlay their metric curves.</p></div></div>
    <select name="runs" multiple size="16">{options}</select>
    <button class="button primary" type="submit">Load chart</button>
  </form>
  <div class="panel chart-panel">
    <div class="panel-head"><div><h2>Metric Timeline</h2><p>Loss curves update from metrics.csv.</p></div></div>
    <canvas id="metricChart" data-runs="{html.escape(','.join(selected))}"></canvas>
  </div>
</section>
"""
    return _layout("Compare", body, active="/compare")


def render_datasets() -> str:
    rows = []
    for path in DEFAULT_DATASETS:
        summary = summarize_dataset(path)
        exists = path.exists()
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(path))}</code><span class=\"run-time\">{'available' if exists else 'missing'}</span></td>"
            f"<td>{summary.counts['train']}</td>"
            f"<td>{summary.counts['val']}</td>"
            f"<td>{summary.counts['test']}</td>"
            f"<td>{summary.counts['total']}</td>"
            f"<td>{_format_bytes(summary.bytes_used)}</td>"
            "</tr>"
        )
    body = (
        "<section class=\"panel\"><div class=\"panel-head\"><div><h2>Datasets</h2>"
        "<p>Known pair-cache roots used by the dashboard launch form.</p></div></div>"
        "<div class=\"table-wrap\"><table><thead><tr><th>Path</th><th>Train</th><th>Val</th>"
        "<th>Test</th><th>Total</th><th>Size</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )
    return _layout("Datasets", body, active="/datasets")


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
        payload = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
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
            self._send_html(report.read_text(encoding="utf-8") if report.exists() else "missing report")
        elif parsed.path == "/static/dashboard.css":
            self._send_text(STYLE)
        elif parsed.path == "/static/dashboard.js":
            self._send_text(SCRIPT)
        else:
            self._send_html(_layout("Not Found", "<p>Not found</p>"), HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/train":
            self._send_html(_layout("Not Found", "<p>Not found</p>"), HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        fields = parse_qs(self.rfile.read(length).decode("utf-8"))
        value = lambda name, default="": fields.get(name, [default])[0]
        lines = lambda name: [line.strip() for line in value(name).splitlines() if line.strip()]
        request = TrainingRequest(
            experiment_name=value("experiment_name", "dashboard_run"),
            backend=value("backend", "compare"),
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
            profile=value("profile", "python-compare"),
            memory_cache_items=int(value("memory_cache_items", "64")),
            prefetch_batches=int(value("prefetch_batches", "4")),
            prefetch_workers=int(value("prefetch_workers", "2")),
            dataloader_workers=int(value("dataloader_workers", "2")),
            max_train_batches=int(value("max_train_batches", "0")),
        )
        try:
            generated = create_training_runs(request)
            pids = [start_generated_run(run) for run in generated]
            message = "Launched: " + ", ".join(f"{run.run_dir.name} pid={pid}" for run, pid in zip(generated, pids))
        except Exception as exc:
            message = f"Launch failed: {exc}"
        self._send_html(render_train(self.project_root, message=message))


STYLE = """
:root {
  --bg: #eef2f6;
  --sidebar: #17202b;
  --sidebar-muted: #95a3b3;
  --surface: #ffffff;
  --surface-soft: #f7f9fc;
  --line: #d9e1ea;
  --line-strong: #c3ccd8;
  --text: #18212d;
  --muted: #687789;
  --accent: #0f8b8d;
  --accent-strong: #0b6f71;
  --blue: #2855a8;
  --green: #188464;
  --amber: #b06b00;
  --red: #b42318;
  --shadow: 0 16px 45px rgba(15, 23, 42, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--accent-strong); text-decoration: none; }
a:hover { text-decoration: underline; }
.app-shell { display: grid; grid-template-columns: 248px minmax(0, 1fr); min-height: 100vh; }
.sidebar {
  background: var(--sidebar);
  color: #f8fafc;
  padding: 22px 18px;
  display: flex;
  flex-direction: column;
  gap: 22px;
  position: sticky;
  top: 0;
  height: 100vh;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
  width: 44px;
  height: 44px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: #e8fbf8;
  color: #0a5f61;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0;
}
.brand-title { font-size: 17px; font-weight: 750; }
.brand-subtitle { color: var(--sidebar-muted); font-size: 12px; margin-top: 2px; }
nav { display: grid; gap: 7px; }
.nav-item {
  color: #d5dde7;
  padding: 10px 11px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 650;
}
.nav-item:hover { background: rgba(255, 255, 255, 0.08); text-decoration: none; }
.nav-item.active { background: #effbf9; color: #0b5f61; }
.sidebar-note {
  margin-top: auto;
  padding: 12px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
}
.sidebar-note strong { display: block; font-size: 12px; margin-bottom: 5px; }
.sidebar-note span { display: block; color: var(--sidebar-muted); font-size: 12px; line-height: 1.45; }
.page { min-width: 0; }
.topbar {
  min-height: 104px;
  padding: 22px 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  background: rgba(255, 255, 255, 0.82);
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
.button.primary, button[type="submit"] { background: var(--accent); color: white; }
.button.primary:hover, button[type="submit"]:hover { background: var(--accent-strong); text-decoration: none; }
.button.secondary { background: white; color: var(--text); border-color: var(--line-strong); }
.hero-panel {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 22px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(135deg, #ffffff 0%, #f5fbfa 100%);
  box-shadow: var(--shadow);
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
.metric-card { padding: 16px; }
.metric-card span { display: block; color: var(--muted); font-size: 12px; font-weight: 750; text-transform: uppercase; }
.metric-card strong { display: block; margin-top: 8px; font-size: 27px; line-height: 1; }
.metric-card small { display: block; margin-top: 8px; color: var(--muted); font-size: 12px; }
.content-grid { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.8fr); gap: 16px; }
.panel { padding: 16px; min-width: 0; }
.panel + .panel { margin-top: 16px; }
.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 13px;
}
.panel-head h2 { margin: 0; font-size: 17px; line-height: 1.2; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; background: white; }
th, td { border-bottom: 1px solid #edf1f5; padding: 10px 11px; text-align: left; vertical-align: middle; white-space: nowrap; }
th { background: var(--surface-soft); color: #435061; font-size: 11px; text-transform: uppercase; font-weight: 800; }
tbody tr:hover { background: #f8fbfd; }
tbody tr:last-child td { border-bottom: 0; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
.run-link { display: block; color: var(--text); font-weight: 750; }
.run-time { display: block; margin-top: 3px; color: var(--muted); font-size: 11px; }
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
.backend-python { background: #e8f0ff; color: var(--blue); }
.backend-cpp { background: #e8f7f0; color: var(--green); }
.backend-unknown { background: #eef2f6; color: #5e6b7c; }
.status-running { background: #e9fbf4; color: var(--green); }
.status-done { background: #edf2f7; color: #536172; }
.status-warn { background: #fff4df; color: var(--amber); }
.status-muted { background: #f2f4f7; color: #778397; }
.quality-bar { width: 96px; height: 7px; background: #edf1f5; border-radius: 999px; overflow: hidden; }
.quality-bar span { display: block; height: 100%; background: var(--accent); border-radius: inherit; }
.row-actions { color: var(--muted); }
.row-actions a { margin-right: 8px; font-weight: 700; }
.processes { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.processes li { padding: 10px; background: var(--surface-soft); border: 1px solid var(--line); border-radius: 6px; line-height: 1.45; overflow-wrap: anywhere; }
.train-workbench { display: grid; grid-template-columns: minmax(340px, 0.9fr) minmax(0, 1.1fr); gap: 16px; align-items: start; }
.launch-panel, .data-panel { grid-column: auto; }
.form-grid { display: grid; gap: 12px; }
.form-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.form-grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
label { display: block; color: #4b5868; font-size: 12px; font-weight: 800; }
input, select, textarea {
  width: 100%;
  margin-top: 6px;
  padding: 9px 10px;
  border: 1px solid var(--line-strong);
  border-radius: 6px;
  background: white;
  color: var(--text);
  font: inherit;
  font-size: 13px;
}
textarea { resize: vertical; min-height: 88px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
input:focus, select:focus, textarea:focus { outline: 2px solid rgba(15, 139, 141, 0.22); border-color: var(--accent); }
.quick-presets { display: flex; gap: 8px; margin-top: 14px; }
.quick-presets button { background: var(--surface-soft); color: var(--text); border-color: var(--line); }
.sticky-submit {
  grid-column: 1 / -1;
  position: sticky;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 -8px 28px rgba(15, 23, 42, 0.08);
}
.sticky-submit span { color: var(--muted); font-size: 13px; }
.notice { padding: 12px 14px; margin: 0 0 16px; background: #e9fbf4; border: 1px solid #b7ead7; color: #0f5f49; border-radius: 8px; font-weight: 650; }
.compare-layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 16px; }
.compare-picker select { min-height: 420px; }
.chart-panel { height: 580px; }
.chart-panel canvas { min-height: 500px; }
@media (max-width: 1180px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .content-grid, .train-workbench, .compare-layout { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; }
  nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .topbar, .hero-panel, .sticky-submit { align-items: stretch; flex-direction: column; }
  main { padding: 16px; }
  .topbar { position: static; padding: 18px 16px; }
  .metric-grid, .form-grid.two, .form-grid.three { grid-template-columns: 1fr; }
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
    balanced: {epochs: 2, batch_size: 2, training_crop_size: 768, resize: 768, samples_per_pair: 512, max_train_batches: 0, memory_cache_items: 128, prefetch_batches: 4},
    long: {epochs: 6, batch_size: 2, training_crop_size: 1024, resize: 1024, samples_per_pair: 1024, max_train_batches: 0, memory_cache_items: 256, prefetch_batches: 6}
  };
  document.querySelectorAll('[data-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const preset = presets[button.dataset.preset] || {};
      Object.entries(preset).forEach(([name, value]) => setField(name, value));
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
  context.fillStyle = '#687789';
  context.font = '13px system-ui, sans-serif';
  if (!runs.length) {
    context.fillText('Select one or more runs to draw metrics.', 24, 36);
    return;
  }
  const response = await fetch('/api/metrics?' + runs.map(run => 'runs=' + encodeURIComponent(run)).join('&'));
  const payload = await response.json();
  const colors = ['#0f8b8d', '#2855a8', '#b06b00', '#b42318', '#7251b5'];
  const datasets = [];
  runs.forEach((run, index) => {
    const rows = (payload.metrics[run] || {}).rows || [];
    const points = rows.map((row, step) => ({
      x: numericValue(row, ['step', 'global_step', 'epoch']) || step + 1,
      y: numericValue(row, ['loss', 'total_loss', 'train_loss'])
    })).filter(point => point.y !== null);
    datasets.push({label: run + ' loss', data: points, color: colors[index % colors.length]});
  });
  const allPoints = datasets.flatMap(dataset => dataset.data);
  if (!allPoints.length) {
    context.fillText('Selected runs do not expose loss metrics yet.', 24, 36);
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
  context.strokeStyle = '#d9e1ea';
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + plotHeight);
  context.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  context.stroke();
  context.fillStyle = '#687789';
  context.font = '12px system-ui, sans-serif';
  context.fillText(`loss ${yMax.toFixed(4)}`, 10, padding.top + 5);
  context.fillText(`loss ${yMin.toFixed(4)}`, 10, padding.top + plotHeight);
  context.fillText(`step ${xMin}`, padding.left, rect.height - 18);
  context.fillText(`step ${xMax}`, padding.left + plotWidth - 64, rect.height - 18);
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
    context.fillStyle = '#18212d';
    context.fillText(dataset.label.slice(0, 24), padding.left + 15 + index * 210, legendY);
  });
}
installPresets();
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
