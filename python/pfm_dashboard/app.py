from __future__ import annotations

import argparse
import json
import os
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def _layout(title: str, body: str) -> str:
    nav = """
<nav>
  <a href="/">Overview</a>
  <a href="/train">Train</a>
  <a href="/runs">Runs</a>
  <a href="/compare">Compare</a>
  <a href="/datasets">Datasets</a>
</nav>
"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - PFM Lab</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
  <header><h1>PFM Lab Dashboard</h1>{nav}</header>
  <main>{body}</main>
  <script src="/static/dashboard.js"></script>
</body>
</html>"""


def _runs_table(runs: list[RunSummary]) -> str:
    rows = []
    for run in runs[:80]:
        report = f'<a href="/runs/{run.name}/report">report</a>' if run.has_report else "-"
        log = f'<a href="/runs/{run.name}/log">log</a>' if run.has_log else "-"
        rows.append(
            "<tr>"
            f"<td><a href=\"/compare?runs={run.name}\">{run.name}</a></td>"
            f"<td>{run.backend}</td>"
            f"<td>{run.status}</td>"
            f"<td>{_metric(run, 'loss', 'total_loss')}</td>"
            f"<td>{_metric(run, 'descriptor_accuracy', 'top1', 'mean_top1')}</td>"
            f"<td>{_metric(run, 'descriptor_positive_rank', 'mean_rank')}</td>"
            f"<td>{run.checkpoint_count}</td>"
            f"<td>{log}</td>"
            f"<td>{report}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Run</th><th>Backend</th><th>Status</th><th>Loss</th>"
        "<th>Top1</th><th>Rank</th><th>CKPT</th><th>Log</th><th>Report</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_index(project_root: Path) -> str:
    runs = discover_runs(project_root / "runs")
    active = active_training_processes()
    disk = shutil.disk_usage(project_root)
    active_rows = "".join(f"<li><code>{line}</code></li>" for line in active) or "<li>No active training process</li>"
    body = f"""
<section class="grid">
  <article><h2>Active Processes</h2><ul class="processes">{active_rows}</ul></article>
  <article><h2>Disk</h2><p>Free: {_format_bytes(disk.free)} / Total: {_format_bytes(disk.total)}</p></article>
  <article><h2>Recent Runs</h2><p>{len(runs)} run directories discovered.</p></article>
</section>
<section><h2>Recent Runs</h2>{_runs_table(runs)}</section>
"""
    return _layout("Overview", body)


def render_train(project_root: Path, message: str = "") -> str:
    default_cache = "\n".join(str(path / "train") for path in DEFAULT_DATASETS if (path / "train").exists())
    notice = f'<p class="notice">{message}</p>' if message else ""
    body = f"""
{notice}
<form method="post" action="/train" class="train-form">
  <fieldset><legend>Run</legend>
    <label>Experiment <input name="experiment_name" value="dashboard_compare"></label>
    <label>Backend
      <select name="backend">
        <option value="compare">Python + C++</option>
        <option value="python">Python</option>
        <option value="cpp">C++</option>
      </select>
    </label>
    <label>Device <input name="device" value="cuda"></label>
    <label>Init checkpoint <input name="init_checkpoint" placeholder="optional"></label>
  </fieldset>
  <fieldset><legend>Data</legend>
    <label>Cache dirs<textarea name="cache_dirs" rows="6">{default_cache}</textarea></label>
    <label>Validation cache dirs<textarea name="validation_cache_dirs" rows="3"></textarea></label>
  </fieldset>
  <fieldset><legend>Training</legend>
    <label>Epochs <input type="number" name="epochs" value="1" min="1"></label>
    <label>Batch <input type="number" name="batch_size" value="1" min="1"></label>
    <label>Crop <input type="number" name="training_crop_size" value="512" min="0"></label>
    <label>Resize <input type="number" name="resize" value="512" min="0"></label>
    <label>Samples <input type="number" name="samples_per_pair" value="512" min="1"></label>
    <label>Learning rate <input name="learning_rate" value="3e-5"></label>
    <label>Profile <input name="profile" value="python-compare"></label>
  </fieldset>
  <fieldset><legend>Loader</legend>
    <label>Memory cache <input type="number" name="memory_cache_items" value="64" min="0"></label>
    <label>Prefetch <input type="number" name="prefetch_batches" value="4" min="1"></label>
    <label>Workers <input type="number" name="prefetch_workers" value="2" min="0"></label>
    <label>C++ loader workers <input type="number" name="dataloader_workers" value="2" min="0"></label>
    <label>Max batches <input type="number" name="max_train_batches" value="0" min="0"></label>
  </fieldset>
  <button type="submit">Launch</button>
</form>
"""
    return _layout("Train", body)


def render_runs(project_root: Path) -> str:
    body = f"<section><h2>Runs</h2>{_runs_table(discover_runs(project_root / 'runs'))}</section>"
    return _layout("Runs", body)


def render_compare(project_root: Path, query: dict[str, list[str]]) -> str:
    selected = query.get("runs", [])
    runs = discover_runs(project_root / "runs")
    options = "".join(
        f'<option value="{run.name}" {"selected" if run.name in selected else ""}>{run.name}</option>' for run in runs
    )
    body = f"""
<section>
  <h2>Compare Runs</h2>
  <form method="get" action="/compare">
    <select name="runs" multiple size="12">{options}</select>
    <button type="submit">Load</button>
  </form>
</section>
<section class="chart-panel">
  <canvas id="metricChart" data-runs="{','.join(selected)}"></canvas>
</section>
"""
    return _layout("Compare", body)


def render_datasets() -> str:
    rows = []
    for path in DEFAULT_DATASETS:
        summary = summarize_dataset(path)
        rows.append(
            "<tr>"
            f"<td><code>{path}</code></td>"
            f"<td>{summary.counts['train']}</td>"
            f"<td>{summary.counts['val']}</td>"
            f"<td>{summary.counts['test']}</td>"
            f"<td>{summary.counts['total']}</td>"
            f"<td>{_format_bytes(summary.bytes_used)}</td>"
            "</tr>"
        )
    body = (
        "<section><h2>Datasets</h2><table><thead><tr><th>Path</th><th>Train</th><th>Val</th>"
        "<th>Test</th><th>Total</th><th>Size</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )
    return _layout("Datasets", body)


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
body { margin: 0; background: #f4f6f8; color: #1f2933; font-family: system-ui, sans-serif; }
header { background: #111827; color: white; padding: 14px 22px; }
h1 { margin: 0 0 10px; font-size: 22px; }
nav { display: flex; gap: 8px; flex-wrap: wrap; }
nav a { color: #dbeafe; text-decoration: none; padding: 5px 9px; border: 1px solid #374151; border-radius: 4px; }
main { padding: 18px 22px; }
section, article { background: white; border: 1px solid #d8dee6; border-radius: 6px; padding: 14px; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: left; vertical-align: top; }
th { background: #f8fafc; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
.train-form { display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 12px; }
fieldset { border: 1px solid #d8dee6; border-radius: 6px; }
label { display: block; font-size: 12px; margin: 8px 0; }
input, select, textarea { box-sizing: border-box; width: 100%; padding: 6px; border: 1px solid #bcc5d0; border-radius: 4px; }
button { padding: 8px 14px; border: 0; background: #0f766e; color: white; border-radius: 4px; cursor: pointer; }
.notice { padding: 10px; background: #ecfdf5; border: 1px solid #99f6e4; border-radius: 4px; }
.chart-panel { height: 520px; }
@media (max-width: 1100px) { .grid, .train-form { grid-template-columns: 1fr; } }
"""


SCRIPT = """
async function loadCompareChart() {
  const canvas = document.getElementById('metricChart');
  if (!canvas || !window.Chart) return;
  const runs = (canvas.dataset.runs || '').split(',').filter(Boolean);
  if (!runs.length) return;
  const response = await fetch('/api/metrics?' + runs.map(run => 'runs=' + encodeURIComponent(run)).join('&'));
  const payload = await response.json();
  const colors = ['#0f766e', '#b91c1c', '#1d4ed8', '#9333ea'];
  const datasets = [];
  runs.forEach((run, index) => {
    const rows = (payload.metrics[run] || {}).rows || [];
    const points = rows.map((row, step) => ({x: row.step || row.global_step || step + 1, y: row.loss || row.total_loss}));
    datasets.push({label: run + ' loss', data: points, borderColor: colors[index % colors.length], tension: 0.2});
  });
  new Chart(canvas, {type: 'line', data: {datasets}, options: {responsive: true, maintainAspectRatio: false, parsing: false}});
}
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
