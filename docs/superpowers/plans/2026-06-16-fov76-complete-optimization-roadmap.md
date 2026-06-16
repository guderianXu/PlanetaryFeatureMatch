# fov76 Complete Optimization Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the current fov76 optimization loop by selecting a safe active checkpoint/selector, then run controlled larger GraphMatcher experiments and hard-failure training without losing formal val/test correctness.

**Architecture:** Keep the current fov76 data scope fixed to the 76m internal pair graph. Treat checkpoint promotion as the source of truth: every candidate must pass formal val/test, regression guard, extreme gain guard, and dual-checkpoint selector checks before it can replace the active path. Optimize the matcher first, then only touch the feature extractor after matcher/filter/geometry evidence shows it is the bottleneck.

**Tech Stack:** Python 3, PyTorch, existing fov76 lazy-pair manifests, `benchmark_lazy_pose_pairs.py`, `run_fov76_checkpoint_promotion_pipeline.py`, `run_graph_filter_sweep.py`, `analyze_fov76_checkpoint_delta.py`, HTML/CSV run records, Git.

---

## Current State

Use these paths as the starting point:

```text
Project:
/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

Python:
/home/w24/anaconda3/envs/cppTorch/bin/python

Data root:
/media/w24/D/xjw深度学习训练数据

fov76 pair root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal

Current accepted base checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Current accepted rescue/checkpoint candidate:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6a_fov76_phase5g_residual_pattern_replay_4l384_20260616_105709/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Current active profile:
fov76_geo5_geo10_extreme_rescue_lowmatch_guard

Latest completed short run:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134
```

Latest phase6c quick promotion result:

```text
Decision:
REJECT

Positive evidence:
formal target extreme_02/extreme_03 val correct_delta = +23, wrong_delta = -1
formal target total all correct_delta = +29, wrong_delta = 0
regression_guard val/test passed
extreme_gain val/test passed
protected variants mid_01/mid_02/extreme_01 passed

Blocking failure:
formal target extreme_02/extreme_03 test correct_delta = +6, wrong_delta = +1, precision_delta = -0.000900

Interpretation:
phase6c contains useful extreme recall gain, but strict promotion rejects it because one added wrong match slightly lowers target test precision.
The next action is false-match and filter diagnostics on the added wrong match, not a longer blind continuation.
```

Current model direction:

```text
Feature extractor:
backbone + DualFPNLite + SparseHead + geometry-aware descriptor pooling + texture/descriptor fusion + quality/reliability outputs

Matcher:
LightGlue-like GraphMatcher, currently mainline 4 layers, hidden dim 384, candidate_topk 256, calibrated metadata, GraphMatcher margin scoring by pair_logit - row_dustbin - col_dustbin, MAGSAC final filtered output

Current optimization target:
increase valid extreme_02/extreme_03 matches and reduce zero/low-match failures without increasing wrong matches or reducing mid/nadir correctness
```

Hard constraints:

```text
Do not resume using fov90 for this branch.
Do not promote phase6b.
Do not increase dustbin/no-match/rejection weights as the first response to low matches.
Do not run 8-layer inference unless the 8-layer model was trained with 8-layer supervision.
Do not replace active model/selector unless promotion gates pass.
```

## Files And Responsibilities

Existing files to use:

```text
scripts/benchmark_lazy_pose_pairs.py
```

Main training and lazy-pair evaluation entry. Use it for matcher-only training, GraphMatcher calibration-only training, visual eval, and train metrics.

```text
scripts/run_fov76_checkpoint_promotion_pipeline.py
```

Formal checkpoint gate. Use it for quick 60/100 promotion and expanded 200/200 promotion.

```text
scripts/run_graph_filter_sweep.py
scripts/visualize_lazy_pose_matches.py
```

Post-filter and match-detail diagnostics. Use them to distinguish model failures from filter threshold failures.

```text
scripts/analyze_fov76_checkpoint_delta.py
scripts/mine_selector_disagreement_pairs.py
scripts/build_train_replay_from_pair_deltas.py
```

Delta analysis, hard pattern mining, and train-only replay generation.

```text
python/pfm_pytorch_training.py
```

Training losses and GraphMatcher loss wiring. Only modify this after a written diagnosis shows the current losses are insufficient.

New run artifacts to create during the matching task:

```text
runs/eval_h100_fov076_phase6c_expanded_20260616.sh
runs/train_h100_fov076_phase7a_graph8_20260616.sh
runs/train_h100_fov076_phase7b_long_best_model_20260616.sh
runs/phase7_fov76_optimization_decision_20260616.html
```

These run scripts are local artifacts under ignored `runs/`. The decision HTML is the human-readable record for this optimization round.

## Task 1: Finish And Record Current Phase6c Promotion

**Files:**
- Read: `runs/train_h100_fov076_phase6c_match_count_floor_20260616_225134.log`
- Read: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile/promotion_decision.json`
- Create: `runs/phase6c_fov76_promotion_summary_20260616.html`

- [ ] **Step 1: Check running processes**

Run:

```bash
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Expected:

```text
Only the current phase6c promotion/eval processes may be running.
No unrelated training process should be started until this finishes.
```

- [ ] **Step 2: Wait for promotion output**

Run:

```bash
tail -160 runs/train_h100_fov076_phase6c_match_count_floor_20260616_225134.log
```

Expected before proceeding:

```text
promotion_decision.json exists
promotion_decision.html exists
formal val/test reports exist
guard reports exist
dual_checkpoint_rescue_selector/combined_filtered_summary.csv exists
```

- [ ] **Step 3: Parse the decision**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile")
decision = root / "promotion_decision.json"
print(decision)
data = json.loads(decision.read_text(encoding="utf-8"))
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
```

Expected:

```text
The printed JSON clearly says PROMOTE or REJECT and lists every formal/guard gate.
```

- [ ] **Step 4: Write the phase6c decision HTML**

Create `runs/phase6c_fov76_promotion_summary_20260616.html` containing:

```text
candidate checkpoint path
baseline checkpoint path
promotion decision
formal val/test totals
formal variant totals
guard totals
selector totals
links to generated reports
one final action: expand, reject, or mine failures
```

- [ ] **Step 5: Branch based on result**

Use this decision table:

```text
If phase6c PROMOTE:
    Run Task 2 expanded 200/200 evaluation.

If phase6c REJECT because correct/match count drops:
    Run Task 4 delta mining and hard replay.

If phase6c REJECT because wrong matches increase:
    Run Task 5 false-match mining and filter diagnostics before any more training.

If phase6c REJECT only because target gain is zero while no metric regresses:
    Keep current active path and run Task 3 selector/filter sweep before another training run.
```

## Task 2: Expanded Promotion For Any Passing Candidate

**Files:**
- Create: `runs/eval_h100_fov076_phase6c_expanded_20260616.sh`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_expanded200_YYYYMMDD_HHMMSS`

- [ ] **Step 1: Create the expanded eval script**

Create `runs/eval_h100_fov076_phase6c_expanded_20260616.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
export PYTHONPATH=python:scripts

PYTHON=/home/w24/anaconda3/envs/cppTorch/bin/python
DATA_ROOT="/media/w24/D/xjw深度学习训练数据"
PAIR_ROOT="${DATA_ROOT}/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
GUARD_ROOT="${PAIR_ROOT}/hard_mining/phase3d_diff_guard_20260614"
BASE_RUN="${DATA_ROOT}/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433"
CAND_RUN="${DATA_ROOT}/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${CAND_RUN}/promotion_phase5g_profile_expanded200_${STAMP}"

"${PYTHON}" scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root "${PAIR_ROOT}" \
  --guard-root "${GUARD_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --baseline-state "${BASE_RUN}/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt" \
  --baseline-run-dir "${BASE_RUN}/train_output" \
  --candidate-state "${CAND_RUN}/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt" \
  --candidate-run-dir "${CAND_RUN}/train_output" \
  --baseline-label phase5g_active \
  --candidate-label phase6c_match_count_floor \
  --guard-baseline-label phase5g_active \
  --guard-candidate-label phase6c_match_count_floor \
  --dual-checkpoint-rescue-selector \
  --dual-checkpoint-rescue-label phase5g_phase6c_selector \
  --python-executable "${PYTHON}" \
  --post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \
  --formal-candidate-pairs 200 \
  --guard-candidate-pairs 200
```

- [ ] **Step 2: Validate script syntax**

Run:

```bash
chmod +x runs/eval_h100_fov076_phase6c_expanded_20260616.sh
bash -n runs/eval_h100_fov076_phase6c_expanded_20260616.sh
```

Expected:

```text
No syntax errors.
```

- [ ] **Step 3: Launch expanded eval**

Run:

```bash
setsid bash runs/eval_h100_fov076_phase6c_expanded_20260616.sh > runs/eval_h100_fov076_phase6c_expanded_20260616.log 2>&1 &
```

- [ ] **Step 4: Promotion rule**

Accept phase6c only if expanded eval satisfies all:

```text
formal val correct delta >= 0
formal test correct delta >= 0
formal target extreme_02/extreme_03 total correct delta >= +1
protected variants mid_01/mid_02/extreme_01/nadir have no correct drop
formal wrong increase <= configured promotion limit
guard precision drop = 0
dual selector total correct delta >= 0 and wrong delta <= 0
```

If any condition fails, do not promote. Continue with Task 4 or Task 5 based on failure type.

## Task 3: Filter And Selector Sweep Before More Training

**Files:**
- Read: `scripts/run_graph_filter_sweep.py`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/filter_sweep_phase6c_YYYYMMDD_HHMMSS`

- [ ] **Step 1: Run a small filter sweep**

Run:

```bash
export PYTHONPATH=python:scripts
/home/w24/anaconda3/envs/cppTorch/bin/python scripts/run_graph_filter_sweep.py \
  --render-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/manifests/h100km_fov076_render_manifest.csv" \
  --uint8-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/manifests/h100km_fov076_uint8_manifest.csv" \
  --pytorch-state "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt" \
  --output-dir "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/filter_sweep_phase6c_$(date +%Y%m%d_%H%M%S)" \
  --run-dir "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output" \
  --metrics-csv "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/train_metrics.csv" \
  --split val \
  --reference-variant nadir \
  --pair-spec-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/overlap_edges_val.csv" \
  --pair-mode same-position \
  --image-source uint8 \
  --candidate-pairs 120 \
  --select-count 0 \
  --seed 20260616 \
  --crop-size 2048 \
  --max-image-size 768 \
  --device cuda \
  --descriptor-mode learned \
  --keypoint-score-mode learned \
  --max-keypoints 512 \
  --matcher-candidate-topk 256 \
  --threshold-px 5.0 \
  --geometry-filter local \
  --geometry-threshold-px-values 5.0,8.0,10.0 \
  --filtered-geometry-filter magsac \
  --filtered-min-matches-values 12,16,20 \
  --low-match-geometry-guard-variants extreme_02,extreme_03 \
  --low-match-geometry-guard-min-matches 12 \
  --low-match-geometry-guard-max-matches 15 \
  --low-match-geometry-guard-max-homography-p90-px 2.8 \
  --low-match-geometry-guard-max-homography-median-px 1.5 \
  --low-match-geometry-guard-min-score-mean 19.0 \
  --adaptive-geometry-rescue-variants extreme_02,extreme_03 \
  --adaptive-geometry-rescue-threshold-px 10.0 \
  --adaptive-geometry-rescue-min-match-gain 5 \
  --adaptive-geometry-rescue-max-base-matches 16 \
  --adaptive-geometry-rescue-max-homography-p90-px 4.2 \
  --adaptive-geometry-rescue-max-homography-median-px 2.3 \
  --max-configs 9 \
  --write-all-summary \
  --no-shuffle \
  --no-illumination-stress \
  --input-local-contrast \
  --input-local-contrast-strength 0.35 \
  --input-local-contrast-kernel 31
```

- [ ] **Step 2: Select a filter only if it is label-safe**

Use this rule:

```text
Prefer the config with the highest filtered_correct.
Reject any config that increases filtered_wrong more than +1 on the 120-pair sweep.
Reject any config that reduces mid_01/mid_02/extreme_01/nadir precision.
Do not lower global minmatch below 16; only use variant-specific low-match guard for extreme_02/extreme_03.
```

- [ ] **Step 3: Re-run promotion with the selected filter**

If the selected filter differs from `fov76_geo5_geo10_extreme_rescue_lowmatch_guard`, add it as a named profile in `scripts/run_fov76_checkpoint_promotion_pipeline.py` and `scripts/README.md` only after adding tests for the new profile.

## Task 4: Hard Regression Mining And Replay

**Files:**
- Use: `scripts/mine_selector_disagreement_pairs.py`
- Use: `scripts/build_train_replay_from_pair_deltas.py`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase7_regression_replay_YYYYMMDD_HHMMSS`

- [ ] **Step 1: Mine candidate regressions**

Run:

```bash
export PYTHONPATH=python:scripts
export DATA_ROOT="/media/w24/D/xjw深度学习训练数据"
export PAIR_ROOT="${DATA_ROOT}/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
export ACTIVE_RUN="${DATA_ROOT}/pfm_runs/phase6a_fov76_phase5g_residual_pattern_replay_4l384_20260616_105709"
export CAND_RUN="${DATA_ROOT}/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134"
export OUT_ROOT="${PAIR_ROOT}/hard_mining/phase7_regression_replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"

/home/w24/anaconda3/envs/cppTorch/bin/python scripts/mine_selector_disagreement_pairs.py \
  --active-combined-csv "${ACTIVE_RUN}/promotion_phase5g_profile_gain3_expanded200_20260616_192526/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --candidate-combined-csv "${CAND_RUN}/promotion_phase5g_profile/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --pair-root "${PAIR_ROOT}" \
  --mine-mode active_regressions \
  --include-non-target-regressions \
  --output-manifest "${OUT_ROOT}/active_vs_phase6c_regressions.csv" \
  --output-summary-json "${OUT_ROOT}/active_vs_phase6c_regressions_summary.json" \
  --output-html "${OUT_ROOT}/active_vs_phase6c_regressions.html"
```

- [ ] **Step 2: Mine clean extreme gains**

Run:

```bash
/home/w24/anaconda3/envs/cppTorch/bin/python scripts/mine_selector_disagreement_pairs.py \
  --active-combined-csv "${ACTIVE_RUN}/promotion_phase5g_profile_gain3_expanded200_20260616_192526/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --candidate-combined-csv "${CAND_RUN}/promotion_phase5g_profile/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --pair-root "${PAIR_ROOT}" \
  --mine-mode candidate_gains \
  --max-candidate-wrong-increase 0 \
  --output-manifest "${OUT_ROOT}/phase6c_clean_extreme_gains.csv" \
  --output-summary-json "${OUT_ROOT}/phase6c_clean_extreme_gains_summary.json" \
  --output-html "${OUT_ROOT}/phase6c_clean_extreme_gains.html"
```

- [ ] **Step 3: Build train-only replay**

Run:

```bash
/home/w24/anaconda3/envs/cppTorch/bin/python scripts/build_train_replay_from_pair_deltas.py \
  --train-manifest "${PAIR_ROOT}/overlap_edges_train.csv" \
  --regression-delta-csv "${OUT_ROOT}/active_vs_phase6c_regressions.csv" \
  --gain-delta-csv "${OUT_ROOT}/phase6c_clean_extreme_gains.csv" \
  --output-manifest "${OUT_ROOT}/train_phase7_regression_replay.csv" \
  --mixed-output-manifest "${OUT_ROOT}/train_phase7_regression_replay_mix10.csv" \
  --mixed-base-manifest "${PAIR_ROOT}/overlap_edges_train.csv" \
  --mixed-replay-fraction 0.10 \
  --max-per-pattern 128 \
  --seed 20260616 \
  --report-html "${OUT_ROOT}/train_phase7_regression_replay.html"
```

- [ ] **Step 4: Replay acceptance rule**

Proceed to training only if:

```text
replay rows >= 128
mixed manifest rows >= 1000
all replay rows come from overlap_edges_train.csv
the mined reasons include at least one of: correct_regression, match_drop, low_match_count, wrong_increase
```

## Task 5: False-Match Mining For Repeated Texture Failures

**Files:**
- Use: `scripts/benchmark_lazy_pose_pairs.py`
- Use: `scripts/visualize_lazy_pose_matches.py`
- Output: current candidate run `false_matches.csv` and visual report match details

- [ ] **Step 1: Generate match details for failing extreme rows**

Run a visual report on the failing manifest with:

```text
--write-all-summary
--write-match-details
--filtered-report
--filtered-mutual
--geometry-filter local
--geometry-threshold-px 5.0
--filtered-geometry-filter magsac
--filtered-min-matches 16
```

Expected output:

```text
all_match_details.csv
all_filtered_match_details.csv
summary.csv
all_filtered_summary.csv
index.html
```

- [ ] **Step 2: Classify failure**

Use this classification:

```text
matches many, precision low:
    false-match / repeated-texture failure, train mined false-edge loss or tighten geometry guard

matches 0 to 2:
    recall / match-count failure, train teacher score floor and match-count floor, do not increase false loss

matches 8 to 15 with clean homography residual:
    post-filter low-match guard problem, tune low-match geometry guard

matches high before MAGSAC but low after MAGSAC:
    geometry filter problem, inspect homography residual and spatial spread
```

- [ ] **Step 3: Training decision**

Use this rule:

```text
If most failures are false-match failures:
    enable graph_matcher_mined_false_match_weight in the next run.

If most failures are match-count failures:
    keep teacher_score_floor and teacher_match_count_floor, reduce mined false loss to zero.

If failures are mixed:
    split hard set into recall replay and false replay, then run two short ablations instead of one combined run.
```

## Task 6: Controlled 8-Layer GraphMatcher Experiment

**Files:**
- Create: `runs/train_h100_fov076_phase7a_graph8_20260616.sh`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_graph8_teacher_floor_8l384_YYYYMMDD_HHMMSS`

- [ ] **Step 1: Create graph8 training script**

Create the script by copying the phase6c script and changing exactly these values:

```text
RUN_ROOT suffix:
phase7a_fov76_graph8_teacher_floor_8l384

checkpoint init:
phase6a accepted checkpoint, not rejected phase6b

teacher checkpoint:
phase5g active checkpoint

graph_attention_layers:
8

graph_max_attention_layers during train visual eval:
8

graph_hidden_dim:
384

steps:
160

learning rate:
5e-12

matcher_candidate_topk:
256

freeze extractor:
true for the whole run

train mode:
GraphMatcher calibration-only for first run

losses:
teacher_score_floor_weight = 0.08
teacher_match_count_floor_weight = 0.02
teacher_match_count_floor_threshold = 18.0
teacher_match_count_floor_margin = 0.5
no_match_prior_weight = 0
graph_matcher_no_match_weight = 0
graph_matcher_hard_negative_dustbin_weight = 0
graph_matcher_stop_confidence_weight = 0
graph_matcher_final_false_match_weight = 0
graph_matcher_mined_false_match_weight = 0
```

- [ ] **Step 2: Validate graph8 is truly trained and evaluated as 8 layers**

Before launching, inspect the script and ensure these both appear:

```text
--graph-attention-layers 8
--graph-max-attention-layers 8
```

Expected:

```text
No 2-layer or 4-layer inference shortcut is used in the graph8 run.
```

- [ ] **Step 3: Launch graph8 short run**

Run:

```bash
bash -n runs/train_h100_fov076_phase7a_graph8_20260616.sh
setsid bash runs/train_h100_fov076_phase7a_graph8_20260616.sh > runs/train_h100_fov076_phase7a_graph8_20260616.log 2>&1 &
```

- [ ] **Step 4: Stop condition**

Stop extending graph8 if any condition holds:

```text
visual_filtered_mean_matches drops below phase6a by more than 10 percent for two eval intervals
true_match_rejected_by_dustbin_ratio rises above 0.30
filtered wrong increases while correct does not increase
loss becomes non-finite for 3 consecutive recovery attempts
GPU OOM occurs twice with the same batch settings
```

## Task 7: Graph8 Promotion And Model-Size Decision

**Files:**
- Use: `scripts/run_fov76_checkpoint_promotion_pipeline.py`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_graph8_teacher_floor_8l384_<STAMP>/promotion_phase5g_profile`

- [ ] **Step 1: Quick promotion**

Run promotion with:

```text
formal-candidate-pairs = 60
guard-candidate-pairs = 100
post-filter-profile = fov76_geo5_geo10_extreme_rescue_lowmatch_guard
dual-checkpoint-rescue-selector = enabled
```

- [ ] **Step 2: Compare against 4-layer**

Decision table:

```text
If graph8 direct candidate improves target extreme correct and does not regress protected variants:
    run expanded 200/200 promotion.

If graph8 direct candidate is worse but selector can safely use graph8 only for extreme_02/extreme_03:
    test dual-checkpoint selector and expanded 200/200 selector promotion.

If graph8 does not improve either direct or selector result:
    keep 4-layer active and do not spend more time on graph8 until the extractor is improved.
```

- [ ] **Step 3: Hidden dim decision**

Only test hidden dim 512 if graph8 hidden 384 passes quick promotion but leaves clear target gains on the table:

```text
graph8 hidden384 PROMOTE or selector PROMOTE
GPU memory headroom >= 3 GB during training
no OOM in phase7a
low-match failures remain mostly recall failures, not false-match failures
```

## Task 8: Longer Training After A Passing Short Run

**Files:**
- Create: `runs/train_h100_fov076_phase7b_long_best_model_20260616.sh`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_long_best_model_YYYYMMDD_HHMMSS`

- [ ] **Step 1: Choose init**

Use:

```text
If phase6c expanded PROMOTE:
    init from phase6c best_by_match_score

If graph8 quick PROMOTE:
    init from graph8 best_by_match_score

If neither promotes:
    init from phase6a accepted checkpoint
```

- [ ] **Step 2: Choose training length**

Use:

```text
short verification:
160 steps

longer stable run:
400 steps

Do not run beyond 400 steps until a 200/200 promotion gate shows positive target gain.
```

- [ ] **Step 3: Keep the loss conservative**

Use:

```text
teacher_score_floor_weight = 0.08
teacher_match_count_floor_weight = 0.02
no_match_prior_weight = 0
hard_negative_dustbin_weight = 0
stop_confidence_weight = 0
mined_false_match_weight = 0 unless Task 5 proves repeated-texture false matches dominate
```

- [ ] **Step 4: Automatic eval after training**

Every run must produce:

```text
train_metrics.csv
visual_report_step_*
best_by_match_score_pytorch_pfm_state.pt
latest_pytorch_pfm_state.pt
promotion_decision.json
promotion_decision.html
```

## Task 9: Feature Extractor Optimization Only After Matcher Evidence

**Files:**
- Modify only when Task 9 Step 1 proves extractor bottleneck: `python/pfm_pytorch_training.py`
- Test after extractor edits: `python/tests/test_pfm_model.py`, `python/tests/test_pfm_pytorch_training.py`

- [ ] **Step 1: Run extractor diagnosis before editing**

Use existing visual and train metrics to check:

```text
selected keypoint count by variant
true_match_in_topk@64/@256
positive_vs_dustbin_margin
raw_cos_pos_mean vs raw_cos_neg_mean
canonical descriptor recall@1/@5 if available
failure concentration in extreme_02/extreme_03
```

- [ ] **Step 2: Extractor change A**

Only if keypoint coverage is poor:

```text
Add stage1 skip to keypoint branch.
Keep descriptor branch unchanged.
Expected benefit: more small crater/ridge keypoints.
Risk: more weak/unstable points, more false matches.
Gate: selected keypoint coverage improves and formal precision does not drop.
```

- [ ] **Step 3: Extractor change B**

Only if quality is suppressing true keypoints:

```text
Change score = heatmap * quality to score = heatmap * (0.5 + 0.5 * quality).
Expected benefit: quality becomes a ranking modulation, not a hard suppressor.
Risk: more low-quality points enter matcher.
Gate: low-match extreme pairs gain correct matches without protected precision drop.
```

- [ ] **Step 4: Extractor change C**

Only if canonical pooling hurts descriptor recall:

```text
Use descriptor_geometry_safety_schedule phase4.
Compare blend=0, blend=0.3, blend=0.5 on the same 120-pair filter sweep.
Gate: positive-negative descriptor margin improves or stays flat while filtered correct improves.
```

## Task 10: GPU Utilization And Data Throughput

**Files:**
- Use: training run logs
- Modify only if needed: `scripts/benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Separate eval gaps from data stalls**

Inspect train log:

```text
data_wait_ms
train_ms
rate step/s
visual eval intervals
GPU utilization during train step vs visual eval
```

Expected interpretation:

```text
Low utilization during visual eval is acceptable.
Low utilization during normal train steps means DataLoader/cache bottleneck.
```

- [ ] **Step 2: DataLoader action**

If `data_wait_ms` is high outside eval:

```text
increase workers
increase prefetch factor
enable persistent workers
keep pair manifests precomputed
avoid on-the-fly overlap calculation
```

- [ ] **Step 3: Cache action**

If disk IO dominates:

```text
use existing uint8 manifest
reuse pair CSV under pfm_overlap_graphs
avoid generating new overlap list during training
do not preload the whole dataset unless memory headroom is proven
```

## Task 11: Validation, GitHub, And Handoff

**Files:**
- Modify after any new public flag/profile/script: `scripts/README.md`
- Create for this optimization round: `runs/phase7_fov76_optimization_decision_20260616.html`

- [ ] **Step 1: Run focused tests after any code change**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_pfm_pytorch_training \
  python.tests.test_benchmark_lazy_pose_pairs \
  python.tests.test_stress_eval_scripts \
  python.tests.test_analyze_fov76_checkpoint_delta
```

Expected:

```text
OK
```

- [ ] **Step 2: Compile changed Python scripts**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m compileall -q \
  python/pfm_pytorch_training.py \
  scripts/benchmark_lazy_pose_pairs.py \
  scripts/run_fov76_checkpoint_promotion_pipeline.py \
  scripts/run_graph_filter_sweep.py \
  scripts/visualize_lazy_pose_matches.py \
  scripts/analyze_fov76_checkpoint_delta.py \
  scripts/mine_selector_disagreement_pairs.py \
  scripts/build_train_replay_from_pair_deltas.py
```

Expected:

```text
No output.
```

- [ ] **Step 3: Commit tracked code and docs**

Run:

```bash
git status --short
git add docs/superpowers/plans/2026-06-16-fov76-complete-optimization-roadmap.md scripts/README.md python/pfm_pytorch_training.py scripts/benchmark_lazy_pose_pairs.py scripts/run_fov76_checkpoint_promotion_pipeline.py scripts/run_graph_filter_sweep.py scripts/visualize_lazy_pose_matches.py scripts/analyze_fov76_checkpoint_delta.py scripts/mine_selector_disagreement_pairs.py scripts/build_train_replay_from_pair_deltas.py python/tests/test_pfm_pytorch_training.py python/tests/test_benchmark_lazy_pose_pairs.py python/tests/test_stress_eval_scripts.py python/tests/test_analyze_fov76_checkpoint_delta.py
git commit -m "Document fov76 complete optimization roadmap"
git push
```

Expected:

```text
Only tracked files relevant to the optimization roadmap are committed.
Untracked file "0" remains untouched.
```

- [ ] **Step 4: Final handoff summary**

The handoff must include:

```text
active checkpoint path
active selector/profile
latest promoted or rejected candidate
formal val/test metrics
guard metrics
known failure types
next exact run script
GitHub commit hash
```

## Promotion Policy

Use this final policy for the next stage:

```text
1. Keep active path unchanged until a candidate passes quick and expanded promotion.
2. Prefer selector-based promotion when a checkpoint is an extreme-only specialist.
3. Prefer direct checkpoint promotion only when all protected variants are neutral or better.
4. Do not solve low match count by increasing dustbin/rejection loss.
5. Do not keep training a rejected run unless delta mining identifies reusable clean gain patterns.
6. Train 8-layer models only with 8-layer training/eval consistency.
7. Touch the extractor only after matcher/filter/geometry diagnostics are exhausted.
```

## Immediate Next Action

The next command after this plan is:

```bash
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Then:

```text
If phase6c promotion is still running:
    monitor it to completion.

If phase6c promotion has finished with the current REJECT reason:
    run Task 5 first to locate the added target-test wrong match.
    then run Task 3 only if the wrong match can be removed by a label-safe filter/selector change.
    then run Task 6 graph8 only after the false-match source is understood.
```
