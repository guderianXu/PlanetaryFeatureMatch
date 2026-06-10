# Graph Filter Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable lazy-pair GraphMatcher filter sweep so we can diagnose whether low match counts and false matches come from model quality or inference thresholds.

**Architecture:** The existing `pytorch_cache_match_eval.graph_matcher_matches()` already implements dustbin, acceptance, raw-score, raw-margin, and accept-probability filters. This plan exposes those knobs through `scripts/visualize_lazy_pose_matches.py`, then adds a small wrapper script that runs multiple lazy visual reports and aggregates their CSV summaries into one CSV/HTML report.

**Tech Stack:** Python `argparse`, `subprocess`, `csv`, existing PyTorch lazy visual evaluation scripts, `unittest`.

---

### Task 1: Expose GraphMatcher Filter Controls In Lazy Visual Reports

**Files:**
- Modify: `scripts/visualize_lazy_pose_matches.py`
- Test: `python/tests/test_stress_eval_scripts.py`

- [ ] **Step 1: Write failing parser and command tests**

Add tests that parse `--graph-dustbin-delta`, `--graph-acceptance-margin`, `--graph-min-raw-score`, `--graph-min-raw-margin`, and `--graph-min-accept-probability`, and verify the lazy visual command builder can pass these values when called by wrapper scripts.

- [ ] **Step 2: Run the targeted test**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_stress_eval_scripts
```

Expected before implementation: fail because lazy visual parse args and graph depth command do not expose every requested filter argument.

- [ ] **Step 3: Implement minimal argument plumbing**

Add the five arguments to `parse_args()`, validate `graph_min_accept_probability` is in `[-1, 1]`, and pass all five values into `match_eval.graph_matcher_matches()` inside `compute_visual()`.

- [ ] **Step 4: Re-run the targeted test**

Expected after implementation: parser/command tests pass.

### Task 2: Add Lazy Graph Filter Sweep Script

**Files:**
- Create: `scripts/run_graph_filter_sweep.py`
- Modify: `python/tests/test_stress_eval_scripts.py`

- [ ] **Step 1: Write failing tests for sweep config parsing and command construction**

Add tests for:
- comma-separated float list parsing,
- deterministic config slugs,
- visual command construction with a representative filter config,
- summary aggregation from `summary.csv` and `filtered_summary.csv`.

- [ ] **Step 2: Run the targeted test**

Run the same unittest command. Expected before implementation: import or attribute failures for `run_graph_filter_sweep`.

- [ ] **Step 3: Implement the sweep script**

Create a wrapper that:
- takes the same dataset/checkpoint arguments as `run_graph_depth_ablation.py`,
- accepts comma-separated lists for min score, dustbin delta, acceptance margin, raw score, raw margin, and accept probability,
- runs `visualize_lazy_pose_matches.py` once per config,
- writes `graph_filter_sweep_summary.csv`,
- writes `index.html` with raw and filtered metrics plus links to child visual reports.

- [ ] **Step 4: Re-run the targeted test**

Expected after implementation: all new sweep tests pass.

### Task 3: Documentation, Verification, And Real Sweep

**Files:**
- Modify: `scripts/README.md`

- [ ] **Step 1: Update script README**

Document `visualize_lazy_pose_matches.py` filter knobs and `run_graph_filter_sweep.py` inputs/outputs.

- [ ] **Step 2: Run focused tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_stress_eval_scripts python.tests.test_pytorch_cache_match_eval
```

Expected: all tests pass.

- [ ] **Step 3: Run the real sweep on the latest best checkpoint**

Use the latest stable graph2 run best checkpoint and the 139k lazy pair snapshot. Keep the first sweep small, around 6-8 configurations, and write output under that training run's `train_output/`.

- [ ] **Step 4: Summarize results and push**

Report the best raw/filtered configuration, the precision/match-count tradeoff, commit the code/docs, and push to `origin/main`.
