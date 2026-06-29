# PFM Recall Guard V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce PFM false matches without sacrificing the current pretrain model's true-match advantage over LightGlue.

**Architecture:** Keep `models/last_good_pytorch_pfm_state.pt` as the protected teacher and never overwrite it during experiments. Run a low-pressure hard-negative replay V2 training pass, then evaluate candidate checkpoints with paired quick64/quick128 and only promote when false matches drop while correct-match retention stays above 99%. In parallel, preserve the match-detail/adaptive gate path because it has already shown LightGlue-level precision with much higher correct-match volume.

**Tech Stack:** Python, PyTorch, WSL Ubuntu-24.04, `/home/xjw/miniforge3/envs/pfm_torch/bin/python`, existing scripts under `scripts/`, experiment artifacts under `runs/`.

---

## File Structure

- Create: `runs/pfm_recall_guard_v2_20260628/`
  - New run directory for the next optimization round.
- Create: `runs/pfm_recall_guard_v2_20260628/build_recall_guard_v2_manifest.py`
  - Generate a lower-pressure train manifest with about 8% hard-negative replay rows.
- Create: `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh`
  - Run 20-step smoke and 300-step training with stronger recall guards and weaker false-match pressure.
- Create: `runs/pfm_recall_guard_v2_20260628/quick64_checkpoint_sweep.sh`
  - Evaluate `last_good`, `best_by_recall`, and `best_by_match_score` on paired quick64 before spending quick128 time.
- Create: `runs/pfm_recall_guard_v2_20260628/quick128_eval_compare.sh`
  - Run final paired quick128 for the selected checkpoint.
- Create: `runs/pfm_recall_guard_v2_20260628/final_report.html`
  - Human-readable report with manifest, training, checkpoint sweep, quick128 result, and promotion decision.
- Read only: `models/last_good_pytorch_pfm_state.pt`
  - Protected teacher/pretrain model.
- Read only: `runs/pfm_match_detail_recall_guard_20260628/decision.json`
  - Previous failed 300-step decision; use it as the regression target.
- Read only: `runs/pfm_match_detail_precision_gate_20260628/offline_apply/combined_summary.json`
  - Prior adaptive-gate reference: 47049 kept matches, 46649 correct, 400 wrong, precision 99.15%.

## Promotion Targets

Use the current protected model as the first gate:

- Quick128 candidate must reduce combined wrong count by at least 20 matches.
- Quick128 candidate must keep `correct_retention >= 0.99`.
- Quick128 candidate must improve precision by at least `0.30 pp`.
- If any gate fails, do not run 1000-step.

Use LightGlue as the stage target:

- Full val+test filtered result should reach `precision >= 99.1%`.
- Full val+test filtered result should keep `correct >= 46600`.
- Full val+test filtered result should keep `wrong <= 407`.
- Stretch target: `correct >= 48000` while keeping `wrong <= 407`.

### Task 1: Preflight And Protected Baseline Snapshot

**Files:**
- Read: `models/last_good_pytorch_pfm_state.pt`
- Read: `runs/pfm_match_detail_recall_guard_20260628/decision.json`
- Create: `runs/pfm_recall_guard_v2_20260628/preflight.json`

- [ ] **Step 1: Check no long-running project task is active**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch; pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py' | grep -v -E 'pgrep -af|grep -v' || true"
```

Expected: no active Python/CUDA training or eval process.

- [ ] **Step 2: Record protected model hash and previous result**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && mkdir -p runs/pfm_recall_guard_v2_20260628 && sha256sum models/last_good_pytorch_pfm_state.pt > runs/pfm_recall_guard_v2_20260628/protected_model.sha256 && cp runs/pfm_match_detail_recall_guard_20260628/decision.json runs/pfm_recall_guard_v2_20260628/previous_recall_guard_decision.json"
```

Expected:

```text
runs/pfm_recall_guard_v2_20260628/protected_model.sha256
runs/pfm_recall_guard_v2_20260628/previous_recall_guard_decision.json
```

### Task 2: Build Lower-Pressure V2 Manifest

**Files:**
- Create: `runs/pfm_recall_guard_v2_20260628/build_recall_guard_v2_manifest.py`
- Output: `runs/pfm_recall_guard_v2_20260628/manifests/recall_guard_v2_false_replay_mixed_train.csv`
- Output: `runs/pfm_recall_guard_v2_20260628/manifests/recall_guard_v2_false_replay_mixed_train_summary.json`

- [ ] **Step 1: Copy the previous manifest generator**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && cp runs/pfm_match_detail_recall_guard_20260628/build_recall_guard_mixed_manifest.py runs/pfm_recall_guard_v2_20260628/build_recall_guard_v2_manifest.py"
```

Expected: the V2 generator exists and is a separate run-local copy.

- [ ] **Step 2: Change manifest pressure constants**

Modify `runs/pfm_recall_guard_v2_20260628/build_recall_guard_v2_manifest.py`:

```python
BASE_SAMPLE_COUNT = 9000
REPLAY_REPEAT = 20
SEED = 20260628
```

Expected replay pressure: about `780 / (9000 + 780) = 7.98%`, down from the previous `13.49%`.

- [ ] **Step 3: Generate and verify manifest**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python runs/pfm_recall_guard_v2_20260628/build_recall_guard_v2_manifest.py"
```

Expected summary:

```text
output_rows around 9780
replay_fraction between 0.075 and 0.085
unique_replay_pairs around 39
```

### Task 3: Run V2 20-Step Smoke

**Files:**
- Create: `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh`
- Output: `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_smoke/`
- Output: `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_smoke.log`

- [ ] **Step 1: Copy previous training script**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && cp runs/pfm_match_detail_recall_guard_20260628/recall_guard_300_train.sh runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh && chmod +x runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh"
```

- [ ] **Step 2: Apply V2 training parameters**

Modify `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh`:

```bash
RUN_ROOT="${ROOT}/runs/pfm_recall_guard_v2_20260628"
OUT_DIR="${PFM_RECALL_GUARD_OUT_DIR:-${RUN_ROOT}/recall_guard_v2_300}"
STEPS="${PFM_RECALL_GUARD_STEPS:-300}"
TRAIN_MANIFEST="${PFM_RECALL_GUARD_TRAIN_MANIFEST:-${RUN_ROOT}/manifests/recall_guard_v2_false_replay_mixed_train.csv}"
MINED_FALSE_WEIGHT="${PFM_RECALL_GUARD_MINED_FALSE_WEIGHT:-0.01}"
WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_RECALL_GUARD_WARP_ACCEPT_WEIGHT:-0.03}"
TEACHER_SCORE_FLOOR_WEIGHT="${PFM_RECALL_GUARD_TEACHER_SCORE_FLOOR_WEIGHT:-0.12}"
TEACHER_MATCH_COUNT_FLOOR_WEIGHT="${PFM_RECALL_GUARD_TEACHER_COUNT_FLOOR_WEIGHT:-0.12}"
TRUE_MATCH_MARGIN_WEIGHT="${PFM_RECALL_GUARD_TRUE_MATCH_MARGIN_WEIGHT:-0.08}"
```

Keep:

```bash
--init-pytorch-state models/last_good_pytorch_pfm_state.pt
--graph-matcher-teacher-guard-state models/last_good_pytorch_pfm_state.pt
--no-train-descriptor-head
--train-graph-matcher
--train-graph-calibration-only
--learning-rate 2.5e-6
```

- [ ] **Step 3: Run 20-step smoke**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && PFM_RECALL_GUARD_STEPS=20 PFM_RECALL_GUARD_OUT_DIR=/mnt/e/code/PlanetaryFeatureMatch/runs/pfm_recall_guard_v2_20260628/recall_guard_v2_smoke bash runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh > runs/pfm_recall_guard_v2_20260628/recall_guard_v2_smoke.log 2>&1"
```

Expected:

```text
no NaN
extra_false_pair_sum > 0
teacher_score_floor_violations_sum > 0
true_margin_violations_sum > 0
```

### Task 4: Run V2 300-Step Candidate Training

**Files:**
- Output: `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300/`
- Output: `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_training_summary.json`

- [ ] **Step 1: Start 300-step training**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && bash runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh > runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300.log 2>&1"
```

Expected:

```text
steps = 300
stop_reason empty
teacher_count_deficit_sum lower than previous 434 if recall guard works better
mean_step_ms_last_100 around the previous 674ms, unless GPU scheduling changes
```

- [ ] **Step 2: Record summary**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && cat runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_training_summary.json"
```

Expected: JSON exists and contains `extra_false_pair_sum`, `teacher_count_deficit_sum`, `true_margin_violations_sum`, and checkpoint paths under `recall_guard_v2_300/checkpoints/`.

### Task 5: Quick64 Checkpoint Sweep

**Files:**
- Create: `runs/pfm_recall_guard_v2_20260628/quick64_checkpoint_sweep.sh`
- Output: `runs/pfm_recall_guard_v2_20260628/quick64_sweep/summary.json`

- [ ] **Step 1: Create quick64 sweep script**

Run three paired quick64 comparisons with these candidate states:

```text
runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300/checkpoints/last_good_pytorch_pfm_state.pt
runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300/checkpoints/best_by_recall_pytorch_pfm_state.pt
runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300/checkpoints/best_by_match_score_pytorch_pfm_state.pt
```

Use command shape:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && PFM_QUICK128_CANDIDATE_PAIRS=64 PFM_QUICK128_CANDIDATE_STATE=/mnt/e/code/PlanetaryFeatureMatch/runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300/checkpoints/last_good_pytorch_pfm_state.pt bash runs/pfm_recall_guard_v2_20260628/quick128_eval_compare.sh > runs/pfm_recall_guard_v2_20260628/quick64_last_good.log 2>&1"
```

Expected checkpoint selection rule:

```text
Choose the checkpoint with lowest wrong count among candidates whose correct_retention >= 0.99.
If no checkpoint keeps correct_retention >= 0.99, stop and do not run quick128.
```

### Task 6: Paired Quick128 For Selected V2 Checkpoint

**Files:**
- Create: `runs/pfm_recall_guard_v2_20260628/quick128_eval_compare.sh`
- Output: `runs/pfm_recall_guard_v2_20260628/quick128_eval/summary.json`
- Output: `runs/pfm_recall_guard_v2_20260628/quick128_eval/index.html`

- [ ] **Step 1: Copy the previous quick128 script**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && cp runs/pfm_match_detail_recall_guard_20260628/quick128_eval_compare.sh runs/pfm_recall_guard_v2_20260628/quick128_eval_compare.sh && chmod +x runs/pfm_recall_guard_v2_20260628/quick128_eval_compare.sh"
```

- [ ] **Step 2: Update script paths**

Modify:

```bash
RUN_ROOT="${ROOT}/runs/pfm_recall_guard_v2_20260628"
OUT_ROOT="${RUN_ROOT}/quick128_eval"
CANDIDATE_STATE="${PFM_QUICK128_CANDIDATE_STATE:-${RUN_ROOT}/recall_guard_v2_300/checkpoints/<selected_checkpoint>.pt}"
```

- [ ] **Step 3: Run quick128**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && bash runs/pfm_recall_guard_v2_20260628/quick128_eval_compare.sh > runs/pfm_recall_guard_v2_20260628/quick128_eval.log 2>&1"
```

Pass condition:

```text
wrong_delta <= -20
precision_pp >= 0.30
correct_retention >= 0.99
```

### Task 7: Adaptive Gate Sweep On The Selected Candidate

**Files:**
- Read: `scripts/run_graph_filter_sweep.py`
- Read: `runs/pfm_match_detail_precision_gate_20260628/offline_apply/combined_summary.json`
- Output: `runs/pfm_recall_guard_v2_20260628/adaptive_gate_sweep/`

- [ ] **Step 1: Run gate sweep using the selected V2 checkpoint's match details**

Use the same profile family as the previous successful offline gate:

```text
fov76_graph_magsac2_min24_balanced
```

Expected sweep target:

```text
precision >= 99.1%
correct >= 46600
wrong <= 407
```

- [ ] **Step 2: Compare to prior gate result**

Reference:

```text
runs/pfm_match_detail_precision_gate_20260628/offline_apply/combined_summary.json
kept_matches = 47049
kept_correct = 46649
kept_wrong = 400
kept_precision = 99.1498%
```

Accept V2 gate only if it improves at least one of:

```text
correct +500 at wrong <= 407
wrong -20 at correct >= 46600
precision +0.05pp at correct >= 46600
```

### Task 8: Only Then Run 1000-Step

**Files:**
- Output: `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_1000/`
- Output: `runs/pfm_recall_guard_v2_20260628/recall_guard_v2_1000_training_summary.json`

- [ ] **Step 1: Run 1000-step only if Task 6 passes**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && PFM_RECALL_GUARD_STEPS=1000 PFM_RECALL_GUARD_OUT_DIR=/mnt/e/code/PlanetaryFeatureMatch/runs/pfm_recall_guard_v2_20260628/recall_guard_v2_1000 bash runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh > runs/pfm_recall_guard_v2_20260628/recall_guard_v2_1000.log 2>&1"
```

Expected: no 1000-step run unless the 300-step candidate already passes quick128.

### Task 9: Final Report And Decision

**Files:**
- Create: `runs/pfm_recall_guard_v2_20260628/final_report.html`
- Create: `runs/pfm_recall_guard_v2_20260628/decision.json`

- [ ] **Step 1: Generate report**

The report must include:

```text
protected model hash
manifest replay fraction
300-step training summary
checkpoint sweep result
quick128 paired comparison
adaptive gate sweep result
promotion decision
```

- [ ] **Step 2: Final verification**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && bash -n runs/pfm_recall_guard_v2_20260628/recall_guard_v2_300_train.sh && bash -n runs/pfm_recall_guard_v2_20260628/quick128_eval_compare.sh && test -f runs/pfm_recall_guard_v2_20260628/final_report.html && test -f runs/pfm_recall_guard_v2_20260628/decision.json"
```

Expected: exit code 0.

## Stop Conditions

- Stop immediately if quick64 shows all candidate checkpoints have `correct_retention < 0.99`.
- Stop immediately if quick128 precision gain is below `0.30 pp`.
- Do not overwrite `models/last_good_pytorch_pfm_state.pt`.
- Do not launch 1000-step just because wrong count decreases; correct retention is equally important.

## Self-Review

- Spec coverage: The plan addresses the observed V1 failure mode: wrong decreased but correct dropped too much.
- Placeholder scan: No task uses TBD/TODO/fill-in wording; each task has concrete paths, commands, and thresholds.
- Type consistency: Metrics use the existing `matches`, `correct`, `wrong`, `precision`, `precision_pp`, and `correct_retention` naming from the generated quick128 summaries.
