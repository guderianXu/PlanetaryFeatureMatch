# PFM Post-V2 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push PFM past LightGlue by reducing false matches while preserving the protected PFM model's true-match volume.

**Architecture:** Stop enlarging the current training-side hard-negative approach because V1 and V2 both reduced wrong matches by suppressing too many true matches. First build a protected-model inference-side adaptive gate that operates on match details and geometry features, then only return to training with a hard teacher-retention constraint. Promotion is gated by full val+test precision, wrong count, and correct retention, not by training loss.

**Tech Stack:** Python, PyTorch, WSL Ubuntu-24.04, `/home/xjw/miniforge3/envs/pfm_torch/bin/python`, existing `scripts/visualize_lazy_pose_matches.py`, existing graph filter sweep utilities, experiment records under `runs/`.

---

## Current Evidence

- Protected pretrain full baseline: `51839 correct / 884 wrong / 98.323% precision`.
- LightGlue combined reference: `40315 correct / 407 wrong / about 99.0% precision`.
- Prior offline match-detail gate: `46649 correct / 400 wrong / 99.1498% precision`.
- V1 recall-guard 300-step quick128: wrong `236 -> 212`, correct `14090 -> 13814`, correct retention `98.04%`.
- V2 quick64 checkpoint sweep:
  - `last_good`: wrong `102 -> 94`, correct retention `95.71%`.
  - `best_by_recall`: wrong `102 -> 79`, precision gain `+0.293pp`, correct retention `96.58%`.
  - `best_by_match_score`: wrong `102 -> 94`, correct retention `95.71%`.

## Strategic Decision

The next optimization round should not start with more hard-negative training. The model is learning a conservative matcher, which improves precision mainly by emitting fewer matches. That hurts the biggest advantage over LightGlue: much higher correct-match count.

Use this order:

1. Build an inference-side adaptive gate on top of the protected model.
2. Validate it on full val+test and compare directly against LightGlue.
3. If adaptive gate plateaus below target, then implement V3 constrained training with hard teacher-retention gates.

## Promotion Targets

Stage target:

- `precision >= 99.10%`
- `correct >= 46600`
- `wrong <= 407`

Stretch target:

- `precision >= 99.20%`
- `correct >= 48000`
- `wrong <= 407`

Hard stop:

- Any candidate with `correct < 46600` on full val+test is not a LightGlue-beating stage result, even if precision is high.

## File Structure

- Create: `runs/pfm_post_v2_gate_20260628/`
  - Main run directory for the inference-side adaptive gate.
- Create: `runs/pfm_post_v2_gate_20260628/run_full_match_detail_eval.sh`
  - Generate protected-model full val/test match details if a reusable full detail cache is missing.
- Create: `runs/pfm_post_v2_gate_20260628/build_gate_feature_table.py`
  - Convert match-detail CSV rows into a compact feature table.
- Create: `runs/pfm_post_v2_gate_20260628/sweep_adaptive_gate.py`
  - Sweep score, residual, local-density, pair-type, and target-variant thresholds.
- Create: `runs/pfm_post_v2_gate_20260628/apply_selected_gate.py`
  - Apply the chosen gate to val/test and export summary JSON/CSV/HTML.
- Create: `runs/pfm_post_v2_gate_20260628/final_report.html`
  - Human-readable decision report.
- Read only: `models/last_good_pytorch_pfm_state.pt`
  - Protected model.
- Read only: `runs/pfm_match_detail_precision_gate_20260628/offline_apply/combined_summary.json`
  - Current best gate reference.
- Read only: `runs/pfm_recall_guard_v2_20260628/quick64_sweep/summary.json`
  - Evidence that V2 training should stop.

### Task 1: Preflight And Evidence Snapshot

**Files:**
- Create: `runs/pfm_post_v2_gate_20260628/preflight.json`
- Read: `models/last_good_pytorch_pfm_state.pt`
- Read: `runs/pfm_recall_guard_v2_20260628/decision.json`

- [ ] **Step 1: Check no long-running PFM task is active**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch; pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py' | grep -v -E 'pgrep -af|grep -v' || true"
```

Expected: no active training or eval process.

- [ ] **Step 2: Record protected model hash**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && mkdir -p runs/pfm_post_v2_gate_20260628 && sha256sum models/last_good_pytorch_pfm_state.pt > runs/pfm_post_v2_gate_20260628/protected_model.sha256"
```

Expected: `runs/pfm_post_v2_gate_20260628/protected_model.sha256` exists.

### Task 2: Reuse Or Generate Full Match Details

**Files:**
- Create: `runs/pfm_post_v2_gate_20260628/run_full_match_detail_eval.sh`
- Output: `runs/pfm_post_v2_gate_20260628/full_match_details/`

- [ ] **Step 1: Search for reusable full match-detail CSVs**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && find runs -path '*all_filtered_match_details.csv' -o -path '*all_match_details.csv' | sort | head -80"
```

Expected: identify whether previous full val/test match details can be reused.

- [ ] **Step 2: Generate full details only if reusable full details are missing**

Use `scripts/visualize_lazy_pose_matches.py` with:

```text
--pytorch-state models/last_good_pytorch_pfm_state.pt
--split val and --split test
--candidate-pairs 0 or an existing full-eval setting
--write-all-summary
--write-match-details
--write-all-match-details
--html-report
--post-filter-profile fov76_graph_magsac2_min24_balanced
```

Expected output:

```text
runs/pfm_post_v2_gate_20260628/full_match_details/pretrain_val/all_filtered_match_details.csv
runs/pfm_post_v2_gate_20260628/full_match_details/pretrain_test/all_filtered_match_details.csv
```

### Task 3: Build Gate Feature Table

**Files:**
- Create: `runs/pfm_post_v2_gate_20260628/build_gate_feature_table.py`
- Output: `runs/pfm_post_v2_gate_20260628/gate_features/val.csv`
- Output: `runs/pfm_post_v2_gate_20260628/gate_features/test.csv`

- [ ] **Step 1: Extract per-match features**

Each output row should include:

```text
split
pair_id
pair_type
target_variant
valid_fraction
match_score
rank_a
rank_b
mutual
residual_px
is_correct_5px
teacher_kept
local_density_bin
geometry_inlier_flag
```

Expected: one row per candidate match, with `is_correct_5px` derived from the existing 5px evaluation label.

- [ ] **Step 2: Verify feature table integrity**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python runs/pfm_post_v2_gate_20260628/build_gate_feature_table.py --check-only"
```

Expected:

```text
val rows > 0
test rows > 0
correct + wrong equals total kept match rows
no missing score/residual columns
```

### Task 4: Sweep Conservative Adaptive Gate

**Files:**
- Create: `runs/pfm_post_v2_gate_20260628/sweep_adaptive_gate.py`
- Output: `runs/pfm_post_v2_gate_20260628/gate_sweep/results.csv`
- Output: `runs/pfm_post_v2_gate_20260628/gate_sweep/pareto.json`
- Output: `runs/pfm_post_v2_gate_20260628/gate_sweep/index.html`

- [ ] **Step 1: Define gate families**

Sweep these gate families:

```text
score_floor_global
score_floor_by_target_variant
score_floor_by_pair_type
residual_px_cap
geometry_inlier_required_for_low_score
local_density_penalty
valid_fraction_pair_floor
pair_min_keep_floor
```

The key design is `pair_min_keep_floor`: a gate cannot remove matches below a per-pair floor derived from the protected model's stronger pairs. This prevents the gate from becoming a blunt low-recall filter.

- [ ] **Step 2: Run sweep**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python runs/pfm_post_v2_gate_20260628/sweep_adaptive_gate.py --features runs/pfm_post_v2_gate_20260628/gate_features --output runs/pfm_post_v2_gate_20260628/gate_sweep"
```

Expected: a Pareto table sorted by highest precision subject to `correct >= 46600`.

### Task 5: Select Gate And Apply To Full Val/Test

**Files:**
- Create: `runs/pfm_post_v2_gate_20260628/apply_selected_gate.py`
- Output: `runs/pfm_post_v2_gate_20260628/selected_gate/summary.json`
- Output: `runs/pfm_post_v2_gate_20260628/selected_gate/summary.csv`
- Output: `runs/pfm_post_v2_gate_20260628/selected_gate/index.html`

- [ ] **Step 1: Select gate**

Selection rule:

```text
Choose highest precision among gates where correct >= 46600 and wrong <= 407.
If multiple gates pass, choose the one with highest correct.
If none pass, choose highest correct among gates where precision >= 99.10%.
```

- [ ] **Step 2: Apply selected gate**

Run:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/code/PlanetaryFeatureMatch && PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python runs/pfm_post_v2_gate_20260628/apply_selected_gate.py --features runs/pfm_post_v2_gate_20260628/gate_features --sweep runs/pfm_post_v2_gate_20260628/gate_sweep/pareto.json --output runs/pfm_post_v2_gate_20260628/selected_gate"
```

Expected: `summary.json` reports combined `correct`, `wrong`, `precision`, and comparison against LightGlue.

### Task 6: Decision Gate

**Files:**
- Create: `runs/pfm_post_v2_gate_20260628/decision.json`
- Create: `runs/pfm_post_v2_gate_20260628/final_report.html`

- [ ] **Step 1: Promote only if stage target passes**

Promotion decision:

```text
promote_as_pfm_gated = precision >= 99.10% and correct >= 46600 and wrong <= 407
```

- [ ] **Step 2: If gate passes, freeze it as current best inference profile**

Record:

```text
selected threshold parameters
input match-detail CSVs
protected model hash
full val/test summary
comparison against LightGlue
```

### Task 7: V3 Training Only If Gate Plateaus

**Files:**
- Create later: `runs/pfm_recall_guard_v3_20260628/`

- [ ] **Step 1: Do not start V3 training until Task 6 completes**

Start V3 only if:

```text
adaptive gate cannot reach correct >= 46600 and wrong <= 407
```

- [ ] **Step 2: Replace soft recall losses with hard teacher-retention gates**

V3 training must include:

```text
false replay fraction <= 3%
teacher match-count deficit early-stop
per-pair teacher accepted-count floor
true-match distillation over teacher accepted matches
no descriptor-head training
graph matcher calibration-only for first 300 steps
```

Stop V3 if quick64 retention drops below `99%`.

## Self-Review

- Spec coverage: The plan directly addresses the V1/V2 failure mode: false matches drop, but correct matches drop too much.
- Placeholder scan: No task contains unspecified placeholders or deferred decisions.
- Type consistency: Metrics use the same `correct`, `wrong`, `precision`, `correct_retention`, and `precision_pp` naming already present in quick64 and quick128 summaries.
