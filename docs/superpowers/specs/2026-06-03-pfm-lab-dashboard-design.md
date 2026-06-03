# PFM Lab Dashboard Design

## Goal

Build a local engineering dashboard for PlanetaryFeatureMatch that can launch Python and C++ training jobs, monitor long-running runs, and compare training metrics without replacing the existing command-line workflow.

## Scope

The first version is a local web app. It runs on `127.0.0.1`, reads and writes project-local `runs/` artifacts, and starts training through generated shell scripts. It does not add authentication, a database, remote multi-user access, or a new training backend.

## Architecture

The dashboard lives under `python/pfm_dashboard/`. The MVP uses Python's standard-library HTTP server so it can run in the existing `plascan` environment without installing new packages. The backend owns four responsibilities:

- Discover run directories and summarize `metrics.csv`, `train.log`, `run.html`, checkpoints, and PID files.
- Generate reproducible Python and C++ training scripts under `runs/<experiment>/`.
- Start scripts with `setsid`, record PID files, and expose process status.
- Serve metric series as JSON for chart rendering.

The frontend is an information-dense engineering console using server-rendered HTML, small vanilla JavaScript, and Chart.js from a CDN. It keeps controls visible and compact, favors tables and charts over large decorative layouts, and links directly to generated logs and HTML reports.

## Pages

### Overview

Shows active training/simulation processes, recent runs, GPU status, disk status, and quick links to train and compare views.

### Train

Provides a compact form for Python, C++, or paired Python+C++ launches. The form exposes:

- cache directories
- validation cache directories for Python
- checkpoint/init checkpoint
- epochs, batch size, crop size, resize, samples per pair, learning rate
- workers, prefetch, memory cache
- training profile and full-v21 switch
- device and experiment name

Submitting the form creates one or two run directories and generated scripts. A paired launch creates sibling run directories sharing the same experiment prefix.

### Runs

Lists discovered runs with backend type, status, last metrics, checkpoint files, report links, and log links.

### Compare

Lets the user choose two or more run directories and overlays loss, descriptor accuracy/top1, graph loss, positive/negative score, margin, and rank when those columns exist.

### Dataset

Summarizes configured pair-cache roots by counting `pair_*.pt` files in `train`, `val`, and `test`, and reports disk usage/free space for relevant mount points.

## Data Model

No persistent database is used. The filesystem is the source of truth:

- `runs/<name>/train.sh`
- `runs/<name>/train.log`
- `runs/<name>/train.pid`
- `runs/<name>/run.html`
- `runs/<name>/metrics.csv`
- checkpoints and reports already produced by training scripts

The backend parses CSV columns defensively. Unknown columns are preserved in JSON responses but not required for rendering.

## Error Handling

The app validates required paths before launching. If a cache/checkpoint path is missing, it returns an error message without starting a process. If a run directory already exists, it appends a timestamp suffix. If process status cannot be determined, the run is shown as `unknown` instead of failing the page.

## Testing

Unit tests cover:

- metrics CSV parsing for C++ and Python formats
- run directory discovery
- train command generation
- process-status parsing for PID files
- dataset split counting

Smoke verification starts the server, opens the index endpoint, and checks that HTML is returned.

## Implementation Notes

The first version should avoid new heavyweight frontend tooling and external Python dependencies. The app should run with:

```bash
PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m pfm_dashboard.app --host 127.0.0.1 --port 7860
```

If a later version needs request validation, async endpoints, or websocket streaming, migrate the same service layer to FastAPI.
