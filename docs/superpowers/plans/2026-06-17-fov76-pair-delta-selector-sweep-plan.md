# fov76 Pair Delta Selector Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare phase6c, phase8, and phase9A at pair level, then run a cheap selector-threshold sweep from existing fov76 promotion CSVs before starting any new GPU training.

**Architecture:** Reuse existing promotion artifacts and `run_dual_checkpoint_rescue_eval.py` selector logic. Generate human-readable HTML and machine-readable CSV/JSON under `runs/`; do not change active config unless a selector candidate beats phase6c and passes validation. Keep fov76 isolated and leave graph8/extractor work out of this step.

**Tech Stack:** Python 3, CSV/JSON/HTML, `scripts/analyze_fov76_checkpoint_delta.py`, `scripts/run_dual_checkpoint_rescue_eval.py`, `scripts/evaluate_checkpoint_promotion.py`, `scripts/validate_fov76_active_selector.py`, Git.

---

## Current Inputs

```text
project root:
/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

python:
/home/w24/anaconda3/envs/cppTorch/bin/python

active phase6c selector metadata:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_p90delta0_expanded200_20260616_231706/dual_checkpoint_rescue_selector/metadata.json

phase8 selector metadata:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902/promotion_phase5g_phase8_profile_p90delta0_expanded200_20260617_031859/dual_checkpoint_rescue_selector/metadata.json

phase9A selector metadata:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase9a_fov76_false_edge_20260617_093722/promotion_phase5g_phase9a_profile_p90delta0_expanded200_20260617_101822/dual_checkpoint_rescue_selector/metadata.json

active config:
runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json
```

## Task 1: Check State

**Files:**
- Read: process table
- Read: `git status --short`

- [ ] **Step 1: Confirm no long run is active**

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Expected: no training, promotion, sweep, or visual process other than the `pgrep` command itself.

- [ ] **Step 2: Check git state**

```bash
git status --short
```

Expected: existing untracked `0` may remain; ignore it.

## Task 2: Run Existing Delta Analyzer

**Files:**
- Read: each candidate `combined_filtered_summary.csv`
- Create: `runs/fov76_pair_delta_selector_sweep_20260617/phase6c_delta/`
- Create: `runs/fov76_pair_delta_selector_sweep_20260617/phase8_delta/`
- Create: `runs/fov76_pair_delta_selector_sweep_20260617/phase9a_delta/`

- [ ] **Step 1: Run per-candidate delta summaries**

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
OUT=runs/fov76_pair_delta_selector_sweep_20260617
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/analyze_fov76_checkpoint_delta.py \
  --combined-csv "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_p90delta0_expanded200_20260616_231706/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --output-dir "$OUT/phase6c_delta"
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/analyze_fov76_checkpoint_delta.py \
  --combined-csv "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902/promotion_phase5g_phase8_profile_p90delta0_expanded200_20260617_031859/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --output-dir "$OUT/phase8_delta"
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/analyze_fov76_checkpoint_delta.py \
  --combined-csv "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase9a_fov76_false_edge_20260617_093722/promotion_phase5g_phase9a_profile_p90delta0_expanded200_20260617_101822/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --output-dir "$OUT/phase9a_delta"
```

Expected: each output directory contains `delta_summary.json`, `delta_by_variant.csv`, `delta_top_gains.csv`, `delta_top_losses.csv`, and `index.html`.

## Task 3: Build Three-Way Pair Delta

**Files:**
- Create: `runs/fov76_pair_delta_selector_sweep_20260617/three_way_pair_delta.csv`
- Create: `runs/fov76_pair_delta_selector_sweep_20260617/index.html`

- [ ] **Step 1: Align by source, split, row_index, base_id, and target_variant**

Use `combined_filtered_summary.csv` for phase6c, phase8, and phase9A. For each aligned row, write selected `matches/correct/wrong/precision`, selected model, selector reason, and deltas versus phase6c.

Expected: all three inputs align row-for-row; if not, stop and report the mismatch.

- [ ] **Step 2: Summarize high-value buckets**

Report these buckets in HTML:

```text
phase6c unique wins over phase8/phase9A
phase8 unique wins over phase6c
phase9A added wrong versus phase6c
blocked rescue rows with positive rescue_correct_delta and nonpositive rescue_wrong_delta
blocked rescue rows where homography or score gate prevented bad rescue
```

Expected: the report identifies whether filter/selector thresholds, model logits, or guard constraints are the next bottleneck.

## Task 4: Offline Selector Sweep

**Files:**
- Read: phase6c selector `metadata.json`
- Create: `runs/fov76_pair_delta_selector_sweep_20260617/phase6c_selector_offline_sweep.csv`
- Create: `runs/fov76_pair_delta_selector_sweep_20260617/phase6c_selector_offline_sweep.html`

- [ ] **Step 1: Sweep conservative selector configs**

Evaluate all combinations:

```text
min_match_gain: 2,3,4,5
min_rescue_matches: 12,14,16,18
max_rescue_homography_p90_px: 2.8,3.0,3.2,3.4
max_rescue_homography_median_px: 1.5,1.8,2.0
max_rescue_homography_p90_delta_px: 0.0,0.15
min_rescue_score_mean: 16.0,18.0,19.0
require_rescue_score_mean_not_lower: true,false
```

Expected: this is CPU-only and uses existing `all_filtered_summary.csv` files from metadata.

- [ ] **Step 2: Apply promotion-style gates**

For each selector config compute formal target, protected variants, regression guard, and extreme-gain summaries. Mark a candidate as eligible only if:

```text
formal target total correct_delta >= 22
formal target total wrong_delta <= -2
formal target total precision_delta >= 0.0005286628527899628
protected variants have no correct drop, wrong increase, or precision drop
regression_guard val/test have no correct drop, wrong increase, or precision drop
```

Expected: if no config is eligible, keep phase6c active and write why.

## Task 5: Validate Active And Decide Next Experiment

**Files:**
- Create: `runs/fov76_active_mainline_validation_after_pair_delta_selector_sweep_20260617.json`
- Create: `runs/fov76_pair_delta_selector_sweep_decision_20260617.html`

- [ ] **Step 1: Validate current active config**

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/validate_fov76_active_selector.py \
  --config runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json \
  --output-json runs/fov76_active_mainline_validation_after_pair_delta_selector_sweep_20260617.json
```

Expected: JSON contains `"valid": true`.

- [ ] **Step 2: Write final decision HTML**

If offline sweep finds no eligible config that beats phase6c, write:

```text
keep phase6c active
do not launch graph8 from this evidence
next GPU run should be phase9A2 with milder false-edge pressure only if pair-delta shows remaining clean false-edge opportunities
```

If offline sweep finds an eligible config, write:

```text
do not change active yet
run a formal selector regeneration/evaluation for the winning config before active replacement
```

Expected: no active config changes without validation evidence.

## Task 6: Final Verification

**Files:**
- Read: generated HTML/CSV/JSON
- Read: `git status --short`

- [ ] **Step 1: Verify outputs exist**

```bash
test -f runs/fov76_pair_delta_selector_sweep_20260617/index.html
test -f runs/fov76_pair_delta_selector_sweep_20260617/phase6c_selector_offline_sweep.csv
test -f runs/fov76_pair_delta_selector_sweep_decision_20260617.html
```

Expected: command exits 0.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_analyze_fov76_checkpoint_delta \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_dual_checkpoint_rescue_selector_only_switches_safe_extreme_rows \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_dual_checkpoint_rescue_selector_can_block_p90_regressions
```

Expected: `OK`.

- [ ] **Step 3: Check git status**

```bash
git status --short
```

Expected: the new plan file may be tracked for commit; ignored `runs/` artifacts do not need commit; untracked `0` remains untouched.
