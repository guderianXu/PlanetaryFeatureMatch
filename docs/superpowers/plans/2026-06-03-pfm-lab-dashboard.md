# PFM Lab Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local engineering dashboard that launches Python/C++ training jobs, monitors logs and metrics, and compares results from `runs/`.

**Architecture:** A standard-library HTTP app under `python/pfm_dashboard/` serves HTML and JSON endpoints. The filesystem remains the source of truth; generated scripts, logs, PID files, CSV metrics, and HTML reports stay under `runs/`.

**Tech Stack:** Python 3 standard library HTTP server, vanilla JavaScript, Chart.js CDN, existing C++ CLI and Python training scripts.

---

### Task 1: Core Run and Metrics Services

**Files:**
- Create: `python/pfm_dashboard/__init__.py`
- Create: `python/pfm_dashboard/models.py`
- Create: `python/pfm_dashboard/services.py`
- Test: `python/test_pfm_dashboard_services.py`

- [ ] **Step 1: Write tests for CSV parsing and run discovery**

Run: `PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m unittest python/test_pfm_dashboard_services.py`
Expected before implementation: import failure for `pfm_dashboard.services`.

- [ ] **Step 2: Implement dataclasses and service functions**

Implement `MetricSeries`, `RunSummary`, `read_metrics_csv()`, `discover_runs()`, `tail_text()`, `pid_status()`, and `dataset_split_counts()`.

- [ ] **Step 3: Run service tests**

Run: `PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m unittest python/test_pfm_dashboard_services.py`
Expected: all service tests pass.

### Task 2: Training Command Generation

**Files:**
- Create: `python/pfm_dashboard/commands.py`
- Test: `python/test_pfm_dashboard_commands.py`

- [ ] **Step 1: Write command-generation tests**

Cover Python-only, C++-only, and paired launch request generation. Confirm generated scripts include explicit paths, cache options, crop/resize, samples, workers, metrics path, and HTML summary path.

- [ ] **Step 2: Implement command generation**

Implement `TrainingRequest`, `GeneratedRun`, `build_python_training_script()`, `build_cpp_training_script()`, and `create_training_runs()`.

- [ ] **Step 3: Run command tests**

Run: `PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m unittest python/test_pfm_dashboard_commands.py`
Expected: all command tests pass.

### Task 3: HTTP App and Templates

**Files:**
- Create: `python/pfm_dashboard/app.py`
- Create: `python/pfm_dashboard/templates/base.html`
- Create: `python/pfm_dashboard/templates/index.html`
- Create: `python/pfm_dashboard/templates/train.html`
- Create: `python/pfm_dashboard/templates/runs.html`
- Create: `python/pfm_dashboard/templates/compare.html`
- Create: `python/pfm_dashboard/templates/datasets.html`
- Create: `python/pfm_dashboard/static/dashboard.css`
- Create: `python/pfm_dashboard/static/dashboard.js`
- Test: `python/test_pfm_dashboard_app.py`

- [ ] **Step 1: Write app smoke tests**

Use a local HTTP server thread to assert `/`, `/train`, `/runs`, `/compare`, `/datasets`, and `/api/runs` respond.

- [ ] **Step 2: Implement app routes**

Add HTML routes and JSON routes for runs, metrics, log tail, datasets, and training launch.

- [ ] **Step 3: Implement templates**

Use compact tables, dense forms, and Chart.js canvas elements. Keep text functional and avoid marketing-style layout.

- [ ] **Step 4: Run app tests**

Run: `PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m unittest python/test_pfm_dashboard_app.py`
Expected: all app tests pass.

### Task 4: Project Integration and Verification

**Files:**
- Create: `runs/dashboard_launch_20260603.sh`
- Create: `runs/dashboard_launch_20260603.html`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add launch script and HTML record**

The launch script starts the dashboard on `127.0.0.1:7860` using the `plascan` Python environment.

- [ ] **Step 2: Update project guidance**

Add a short Dashboard section to `AGENTS.md` documenting launch command, scope, and HTML logging expectations.

- [ ] **Step 3: Run full verification**

Run:

```bash
PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m unittest python/test_pfm_dashboard_services.py python/test_pfm_dashboard_commands.py python/test_pfm_dashboard_app.py
cmake --build build -j$(nproc)
./build/pfm_tests
```

Expected: Python dashboard tests pass, C++ build passes, C++ tests pass.

- [ ] **Step 4: Start dashboard**

Run:

```bash
setsid runs/dashboard_launch_20260603.sh > runs/dashboard_launch_20260603.log 2>&1 &
```

Expected: local URL `http://127.0.0.1:7860` is available.
