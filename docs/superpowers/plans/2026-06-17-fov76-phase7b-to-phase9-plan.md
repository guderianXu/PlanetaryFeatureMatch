# fov76 Phase7b To Phase9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the active fov76 graph8 evaluation chain, select the safest checkpoint/filter combination, then run hard-failure-driven matcher optimization before touching the extractor.

**Architecture:** Keep the current h100/fov76/dom76 branch isolated from fov90 and other datasets. Treat formal promotion as the only activation gate: training output is only a candidate until it passes expanded val/test, regression guard, and target extreme checks. Optimize in this order: post-filter evidence, hard failure replay, matcher loss/calibration, then extractor ablation.

**Tech Stack:** Python 3, PyTorch, CUDA AMP, existing fov76 lazy pair manifests, `benchmark_lazy_pose_pairs.py`, `run_fov76_checkpoint_promotion_pipeline.py`, `visualize_lazy_pose_matches.py`, `run_graph_filter_sweep.py`, `mine_hard_failure_pairs.py`, `build_train_replay_from_pair_deltas.py`, HTML/CSV run records, Git.

---

## Current Baseline

Authoritative project paths:

```text
project:
/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

python:
/home/w24/anaconda3/envs/cppTorch/bin/python

data root:
/media/w24/D/xjw深度学习训练数据

fov76 pair root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal

active selector config:
runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json

current active accepted run:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134

stable teacher checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

post-filter profile:
fov76_geo5_geo10_extreme_rescue_lowmatch_guard
```

Current candidate runs:

```text
phase7a 4-layer candidate:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507

phase7a status:
training clean, promotion rejected because it added small val-side wrong/regression deltas.

phase7b 8-layer candidate:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718

phase7b status:
training/evaluation chain is active; wait for final promotion decision before using it.
```

Hard constraints:

```text
Do not use fov90 in this branch.
Do not promote a candidate without expanded formal promotion.
Do not increase dustbin/no-match weights as the first response to low match count.
Do not run an 8-layer inference config on a checkpoint that was not trained/evaluated for 8 layers.
Do not unfreeze the full extractor until matcher/filter/geometry diagnostics are exhausted.
Do not touch the untracked file named "0".
Keep long-task logs, pid files, and HTML summaries under runs/.
```

## Success Criteria

Primary pass/fail gate:

```text
PROMOTE only if:
- target variants extreme_02/extreme_03 improve or stay useful on val/test;
- wrong_delta does not increase on target variants;
- protected variants mid_01/mid_02/extreme_01 do not regress;
- regression guard does not regress;
- homography p90 selector guard is respected;
- low-match rescue does not trade recall for unstable false matches.

KEEP ACTIVE BASELINE if:
- the candidate is neutral;
- the candidate wins only on train or only on a small visual slice;
- the candidate increases wrong matches or guard failures.

MINE if:
- the candidate shows target gains but has isolated failures.
```

Metrics to record in every summary HTML:

```text
filtered_correct
filtered_wrong
filtered_precision
filtered_matches
zero_match_rows
low_match_rows
homography_residual_p90_px
target correct_delta
target wrong_delta
protected wrong_delta
regression_guard precision_delta
best checkpoint path
promotion decision path
```

## Task 1: Finish Phase7b Graph8 Chain

**Files:**
- Read: `runs/train_h100_fov076_phase7b_graph8_20260617_001718.log`
- Read: `runs/watch_phase7b_then_promotion_20260616.log`
- Read: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/train_metrics.csv`
- Create: `runs/phase7b_promotion_decision_20260617.html`

- [ ] **Step 1: Check active processes**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py|watch_phase7b' || true
```

Expected:

```text
Only the phase7b train/final visual/promotion chain is active.
No fov90 training process is active.
```

- [ ] **Step 2: Verify graph8 training completed cleanly**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 runs/train_h100_fov076_phase7b_graph8_20260617_001718.log
```

Expected:

```text
The log reaches step 800/800.
The final visual report writes an index.html path.
There is no traceback, OOM, NaN, inf, or nonfinite-loss abort.
```

- [ ] **Step 3: Verify graph8 artifacts**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
test -f "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
test -f "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/train_metrics.csv"
test -f "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/visual_report_step_000800/index.html"
echo phase7b_artifacts_ok
```

Expected:

```text
phase7b_artifacts_ok
```

- [ ] **Step 4: Ensure promotion runs after training**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 runs/watch_phase7b_then_promotion_20260616.log
```

Expected:

```text
The watcher detects training exit and launches runs/eval_h100_fov076_phase7b_promotion_20260616.sh.
```

If promotion was not launched, run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
setsid runs/eval_h100_fov076_phase7b_promotion_20260616.sh "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718" > runs/eval_h100_fov076_phase7b_promotion_manual_20260617.launch.log 2>&1 &
```

Expected:

```text
A promotion process appears in pgrep.
```

- [ ] **Step 5: Parse promotion decision**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json
root = Path('/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718')
paths = sorted(root.glob('promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json'))
if not paths:
    raise SystemExit('phase7b promotion_decision.json not found')
data = json.loads(paths[-1].read_text(encoding='utf-8'))
print(paths[-1])
print('promote=', data.get('promote'))
for reason in data.get('reasons', []):
    print(reason)
PY
```

Expected:

```text
A concrete promotion_decision.json path is printed.
promote=True or promote=False is explicit.
Every failed gate is listed.
```

## Task 2: Choose Active Candidate

**Files:**
- Read: `runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json`
- Read: phase7a `promotion_decision.json`
- Read: phase7b `promotion_decision.json`
- Create: `runs/phase7_candidate_comparison_20260617.html`
- Modify only if promoted: `runs/fov76_active_mainline_config_*.json`

- [ ] **Step 1: Compare phase6c, phase7a, phase7b**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

items = {
    'phase6c_active': Path('/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134'),
    'phase7a_4l384': Path('/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507'),
    'phase7b_8l384': Path('/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718'),
}
for name, root in items.items():
    decisions = sorted(root.glob('promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json'))
    print(f'## {name}')
    if not decisions:
        print('decision=active baseline or not evaluated')
        continue
    data = json.loads(decisions[-1].read_text(encoding='utf-8'))
    print('decision_path=', decisions[-1])
    print('promote=', data.get('promote'))
    for reason in data.get('reasons', [])[:12]:
        print(reason)
PY
```

Expected:

```text
phase6c remains active unless phase7b promotes.
phase7a is rejected unless a later decision says otherwise.
phase7b is selected only if promote=True.
```

- [ ] **Step 2: Write comparison HTML**

Create `runs/phase7_candidate_comparison_20260617.html` with:

```html
<!doctype html>
<meta charset="utf-8">
<title>fov76 Phase7 Candidate Comparison</title>
<h1>fov76 Phase7 Candidate Comparison</h1>
<p>Decision rule: keep phase6c unless phase7b passes expanded formal promotion.</p>
<table border="1" cellspacing="0" cellpadding="4">
  <tr><th>Candidate</th><th>Status</th><th>Decision JSON</th><th>Action</th></tr>
  <tr><td>phase6c active</td><td>accepted baseline</td><td>runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json</td><td>keep unless phase7b promotes</td></tr>
  <tr><td>phase7a 4l384</td><td>promotion rejected</td><td>phase7a promotion_decision.json</td><td>use only for mining evidence</td></tr>
  <tr><td>phase7b 8l384</td><td>fill from promotion_decision.json</td><td>phase7b promotion_decision.json</td><td>promote only if promote=true</td></tr>
</table>
```

Expected:

```text
The HTML records the selected candidate, rejected candidates, and why.
```

- [ ] **Step 3: Update active selector only on promotion**

If phase7b `promote=True`, create a new active config:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json
src = Path('runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json')
data = json.loads(src.read_text(encoding='utf-8'))
data['label'] = 'phase7b_graph8_h384_selector_p90delta0'
data['candidate_run_root'] = '/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718'
data['checkpoint'] = '/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt'
data['graph_attention_layers'] = 8
out = Path('runs/fov76_active_mainline_config_phase7b_graph8_p90delta0_20260617.json')
out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(out)
PY
```

Expected:

```text
No active config changes if phase7b fails promotion.
```

## Task 3: Run Filter Sweep On The Selected Checkpoint

**Files:**
- Read: selected checkpoint from Task 2
- Read: fov76 pair manifests under pair root
- Create: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_filter_sweep_<timestamp>/`
- Create: `runs/phase8_filter_sweep_summary_20260617.html`

- [ ] **Step 1: Set checkpoint variables**

Use phase7b only if promoted; otherwise keep phase6c:

```bash
export PFM_SELECTED_STATE="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
export PFM_SELECTED_LAYERS="4"
export PFM_SELECTED_LABEL="phase6c_active"
```

If phase7b promoted:

```bash
export PFM_SELECTED_STATE="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
export PFM_SELECTED_LAYERS="8"
export PFM_SELECTED_LABEL="phase7b_graph8"
```

Expected:

```text
The selected checkpoint matches the selected graph depth.
```

- [ ] **Step 2: Run compact filter sweep**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
OUT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_filter_sweep_${PFM_SELECTED_LABEL}_$(date +%Y%m%d_%H%M%S)"
setsid env PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/run_graph_filter_sweep.py \
  --render-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/manifests/h100km_fov076_render_manifest.csv" \
  --uint8-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/manifests/h100km_fov076_uint8_manifest.csv" \
  --pytorch-state "$PFM_SELECTED_STATE" \
  --output-dir "$OUT" \
  --split val \
  --image-source uint8 \
  --pair-mode same-position \
  --pair-spec-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/overlap_edges_val.csv" \
  --candidate-pairs 200 \
  --select-count 24 \
  --crop-size 2048 \
  --device cuda \
  --descriptor-mode learned \
  --keypoint-score-mode learned \
  --max-keypoints 512 \
  --matcher-candidate-topk 256 \
  --geometry-filter local \
  --geometry-threshold-px-values 4,5,6,8,10 \
  --graph-max-attention-layers "$PFM_SELECTED_LAYERS" \
  --graph-width-prune-keep-ratio 1.0 \
  --min-score-values -1.0,0.0,0.02 \
  --filtered-report \
  --filtered-mutual \
  --filtered-geometry-filter magsac \
  --filtered-min-matches-values 8,12,16 \
  --filtered-min-margin 0.02 \
  --post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \
  --input-local-contrast \
  --input-local-contrast-strength 0.35 \
  > "runs/phase8_filter_sweep_${PFM_SELECTED_LABEL}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Expected:

```text
A sweep run directory is created with CSV/HTML outputs.
The best filter must reduce wrong matches without increasing zero-match rows.
```

## Task 4: Mine Hard Failure Pairs

**Files:**
- Read: selected visual summary CSV
- Read: `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/overlap_edges_train.csv`
- Create: `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures/`
- Create: `runs/phase8_hard_failure_manifest_summary_20260617.html`

- [ ] **Step 1: Mine train hard failures from selected visual report**

Run after choosing the report directory that contains `filtered_summary.csv`:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
export PFM_REPORT_DIR="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/visual_report_step_000800"
export PFM_HARD_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures"
mkdir -p "$PFM_HARD_ROOT"
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/mine_hard_failure_pairs.py \
  --pair-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/overlap_edges_train.csv" \
  --summary-csv "$PFM_REPORT_DIR/filtered_summary.csv" \
  --output-manifest "$PFM_HARD_ROOT/hard_failures_train.csv" \
  --mixed-base-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/overlap_edges_train.csv" \
  --mixed-output-manifest "$PFM_HARD_ROOT/hard_mixed_train.csv" \
  --mixed-hard-fraction 0.35 \
  --report-html "runs/phase8_hard_failure_manifest_summary_20260617.html" \
  --reference-variant nadir \
  --failure-preset residual_filtered \
  --extreme-variants extreme_02,extreme_03 \
  --include-extreme-without-failure
```

Expected:

```text
hard_failures_train.csv contains low-precision, wrong-heavy, low-match, and extreme failures.
hard_mixed_train.csv blends hard rows with normal train rows.
The report HTML lists counts by variant and reason.
```

- [ ] **Step 2: Verify hard manifests are train-only**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import csv
paths = [
    Path('/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures/hard_failures_train.csv'),
    Path('/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures/hard_mixed_train.csv'),
]
for path in paths:
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if r.get('split') not in ('', 'train')]
    print(path, 'rows=', len(rows), 'non_train=', len(bad))
    if bad:
        raise SystemExit(f'non-train rows found in {path}')
PY
```

Expected:

```text
Both manifests report non_train=0.
```

## Task 5: Train Phase8 Hard-Replay Matcher

**Files:**
- Create: `runs/train_h100_fov076_phase8_hard_replay_20260617.sh`
- Create: `runs/train_h100_fov076_phase8_hard_replay_20260617.log`
- Create: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_<timestamp>/`

- [ ] **Step 1: Create phase8 training script**

Create `runs/train_h100_fov076_phase8_hard_replay_20260617.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

PY=/home/w24/anaconda3/envs/cppTorch/bin/python
PAIR_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
HARD_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures"
OUT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_$(date +%Y%m%d_%H%M%S)"
INIT_STATE="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
TEACHER_STATE="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"

mkdir -p "$OUT"

PYTHONPATH=python:scripts "$PY" scripts/benchmark_lazy_pose_pairs.py \
  --render-manifest "$PAIR_ROOT/manifests/h100km_fov076_render_manifest.csv" \
  --uint8-manifest "$PAIR_ROOT/manifests/h100km_fov076_uint8_manifest.csv" \
  --output-dir "$OUT/train_output" \
  --mode train \
  --split train \
  --image-source uint8 \
  --pair-mode same-position \
  --pair-type-weights same_position_view=1.0 \
  --pair-spec-manifest "$HARD_ROOT/hard_mixed_train.csv" \
  --steps 1600 \
  --workers 10 \
  --prefetch-batches 32 \
  --worker-cache-items 128 \
  --batch-pairs 1 \
  --samples-per-pair 128 \
  --crop-size 2048 \
  --max-attempts 4 \
  --min-valid-fraction 0.02 \
  --absolute-depth-tolerance-m 100.0 \
  --relative-depth-tolerance 0.005 \
  --device cuda \
  --amp \
  --amp-dtype float16 \
  --activation-checkpointing \
  --init-pytorch-state "$INIT_STATE" \
  --seed 20260617 \
  --progress-every 20 \
  --save-every-steps 200 \
  --save-best-checkpoints \
  --gpu-monitor \
  --gpu-sample-interval-s 2.0 \
  --skip-bad-pairs \
  --max-bad-pairs 0 \
  --skip-nonfinite-steps \
  --learning-rate 1e-11 \
  --train-graph-matcher \
  --train-graph-calibration-only \
  --no-train-descriptor-head \
  --descriptor-geometry-mode full \
  --quality-score-mode soft \
  --graph-hidden-dim 384 \
  --graph-attention-layers 4 \
  --graph-matcher-loss-weight 0.006 \
  --graph-matcher-metadata-mode calibrated \
  --graph-matcher-no-match-points 0 \
  --graph-matcher-no-match-weight 0.0 \
  --graph-matcher-assignment-weight 0.003 \
  --graph-matcher-accept-weight 0.00001 \
  --graph-matcher-stop-confidence-weight 0.0 \
  --graph-matcher-hard-negative-dustbin-weight 0.0 \
  --graph-matcher-train-max-attention-layers 4 \
  --graph-matcher-train-width-keep-ratio 1.0 \
  --matcher-reliability-pair-bias off \
  --matcher-reliability-dustbin-bias off \
  --matcher-final-accept-score-mode none \
  --matcher-accept-assignment-mode add \
  --matcher-candidate-topk 256 \
  --graph-matcher-positive-dustbin-margin-weight 0.00008 \
  --graph-matcher-positive-dustbin-margin 0.04 \
  --graph-matcher-true-match-margin-weight 0.0003 \
  --graph-matcher-true-match-margin 0.08 \
  --graph-matcher-ransac-consistency-weight 0.003 \
  --graph-matcher-ransac-consistency-topk 8 \
  --graph-matcher-ransac-consistency-residual-threshold-px 3.0 \
  --graph-matcher-ransac-consistency-min-score 0.02 \
  --graph-matcher-ransac-consistency-margin 0.20 \
  --graph-matcher-teacher-guard-state "$TEACHER_STATE" \
  --graph-matcher-teacher-guard-weight 0.80 \
  --graph-matcher-teacher-guard-positive-margin-tolerance 0.0 \
  --graph-matcher-teacher-guard-false-margin-tolerance 0.010 \
  --graph-matcher-teacher-score-floor-weight 0.08 \
  --graph-matcher-teacher-score-floor-tolerance 0.015 \
  --graph-matcher-teacher-score-floor-min-score 0.0 \
  --graph-matcher-teacher-match-count-floor-weight 0.02 \
  --graph-matcher-teacher-match-count-floor-threshold 18.0 \
  --graph-matcher-teacher-match-count-floor-margin 0.5 \
  --freeze-extractor-warmup-steps 999999 \
  --synthetic-loss-weight 0.0 \
  --teacher-weight 0.0 \
  --hard-negative-weight 0.0 \
  --keypoint-weight 0.0 \
  --matchability-weight 0.0 \
  --descriptor-uncertainty-weight 0.0 \
  --no-match-prior-weight 0.0 \
  --reliability-negative-points 0 \
  --input-local-contrast \
  --input-local-contrast-strength 0.35 \
  --auto-visual-report \
  --visual-eval-every-steps 200 \
  --visual-post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \
  --visual-matcher-mode graph_matcher \
  --visual-keypoint-score-mode learned \
  --visual-geometry-filter local \
  --visual-geometry-threshold-px 5.0 \
  --visual-filtered-geometry-filter magsac \
  --visual-filtered-min-matches 16 \
  --visual-graph-max-attention-layers 4 \
  --visual-graph-width-prune-keep-ratio 1.0 \
  --visual-max-keypoints 512 \
  --visual-candidate-pairs 80 \
  --visual-select-count 12
```

Expected:

```text
The script trains a 4-layer/384 matcher-only hard replay pass from the active safe checkpoint.
Extractor remains frozen.
Dustbin/no-match pressure remains off.
```

- [ ] **Step 2: Syntax-check and launch**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
chmod +x runs/train_h100_fov076_phase8_hard_replay_20260617.sh
bash -n runs/train_h100_fov076_phase8_hard_replay_20260617.sh
setsid runs/train_h100_fov076_phase8_hard_replay_20260617.sh > runs/train_h100_fov076_phase8_hard_replay_20260617.log 2>&1 &
echo $! > runs/train_h100_fov076_phase8_hard_replay_20260617.pid
```

Expected:

```text
The training process starts and writes step logs.
GPU memory remains within the current 24 GB machine budget.
```

- [ ] **Step 3: Monitor**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 runs/train_h100_fov076_phase8_hard_replay_20260617.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

Expected:

```text
data_wait median stays near zero after warmup.
top1 does not collapse for several intervals.
No NaN/OOM appears.
Visual reports are written every 200 steps.
```

## Task 6: Promote Or Reject Phase8

**Files:**
- Read: latest `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_*/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt`
- Create: `runs/eval_h100_fov076_phase8_promotion_20260617.sh`
- Create: `runs/phase8_promotion_decision_20260617.html`

- [ ] **Step 1: Create promotion script**

Use the same gate as phase7b, but point to the phase8 run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
cp runs/eval_h100_fov076_phase7b_promotion_20260616.sh runs/eval_h100_fov076_phase8_promotion_20260617.sh
sed -i 's/phase7b_fov76_graph8_h384_/phase8_fov76_hard_replay_/g' runs/eval_h100_fov076_phase8_promotion_20260617.sh
chmod +x runs/eval_h100_fov076_phase8_promotion_20260617.sh
bash -n runs/eval_h100_fov076_phase8_promotion_20260617.sh
```

Expected:

```text
The script still uses phase5g/phase6c as guard baselines and the same p90delta0 promotion profile.
```

- [ ] **Step 2: Run formal promotion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PHASE8_ROOT="$(find "/media/w24/D/xjw深度学习训练数据/pfm_runs" -maxdepth 1 -type d -name 'phase8_fov76_hard_replay_*' | sort | tail -1)"
setsid runs/eval_h100_fov076_phase8_promotion_20260617.sh "$PHASE8_ROOT" > runs/eval_h100_fov076_phase8_promotion_20260617.log 2>&1 &
```

Expected:

```text
Promotion produces promotion_decision.json under the phase8 run root.
```

- [ ] **Step 3: Decide**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json
root = sorted(Path('/media/w24/D/xjw深度学习训练数据/pfm_runs').glob('phase8_fov76_hard_replay_*'))[-1]
paths = sorted(root.glob('promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json'))
if not paths:
    raise SystemExit('phase8 promotion_decision.json not found')
data = json.loads(paths[-1].read_text(encoding='utf-8'))
print(paths[-1])
print('promote=', data.get('promote'))
for reason in data.get('reasons', []):
    print(reason)
PY
```

Expected:

```text
If promote=True, write a new active config for phase8.
If promote=False but failures are localized, mine the residual failures for the next pass.
If promote=False with broad regressions, revert to phase6c active and do not train deeper from phase8.
```

## Task 7: Matcher Optimization Matrix After Phase8

**Files:**
- Create one run directory per experiment under `/media/w24/D/xjw深度学习训练数据/pfm_runs/`
- Create one launch script per experiment under `runs/`
- Create: `runs/phase9_matcher_optimization_matrix_20260617.html`

- [ ] **Step 1: Run only one change per experiment**

Use these experiments in order:

```text
M1: active 4-layer/384, candidate_topk 256, hard replay, current safe losses.
M2: active 4-layer/384, candidate_topk 384, same losses, same hard replay.
M3: active 8-layer/384 only if phase7b promoted or phase8 proves 8-layer safe.
M4: add final false-match loss with very small weight 0.00005, no dustbin increase.
M5: add mined false-match loss with very small weight 0.00005, no dustbin increase.
```

Expected:

```text
No experiment changes extractor, graph depth, candidate_topk, and false-match losses at the same time.
```

- [ ] **Step 2: Stop criteria**

Stop a run early if:

```text
true matches are mostly rejected by dustbin;
filtered matches collapse to near zero for repeated visual intervals;
wrong matches increase on protected variants;
top1 collapses for multiple intervals after warmup;
NaN/OOM appears.
```

Expected:

```text
Bad runs are used for diagnosis, not as parents for the next training pass.
```

## Task 8: Extractor Changes Only After Matcher Evidence

**Files:**
- Inspect before editing: `python/pfm_pytorch_training.py`
- Inspect before editing: extractor/model files under `python/`
- Test: focused Python unit tests or a 50-step smoke train
- Create: `runs/phase9_extractor_ablation_20260617.html`

- [ ] **Step 1: Stage1 skip ablation**

Only start this if Task 6/7 shows the matcher/filter path is stable and still misses fine small-scale structure.

Required ablation:

```text
E1 baseline: current extractor.
E2 detector stage1 skip enabled, descriptor unchanged.
E3 detector stage1 skip enabled and quality score soft modulation kept.
```

Expected:

```text
The extractor is not fully unfrozen in the first ablation.
The comparison uses the same selected checkpoint/filter gate.
```

- [ ] **Step 2: Quality score modulation ablation**

Compare:

```text
Q1 current score path.
Q2 score = heatmap * (0.5 + 0.5 * quality).
```

Expected:

```text
Q2 is accepted only if it improves keypoint recall without increasing wrong filtered matches.
```

## Task 9: Documentation, Git, And Handoff

**Files:**
- Modify if durable scripts/flags changed: `scripts/README.md`
- Read: `git status --short --branch`
- Create: final HTML summaries under `runs/`

- [ ] **Step 1: Verify repository status**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git status --short --branch
```

Expected:

```text
Only intentional tracked docs/code changes are staged for commit.
The untracked file "0" remains untouched and untracked.
```

- [ ] **Step 2: Run syntax checks for new scripts**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
bash -n runs/train_h100_fov076_phase8_hard_replay_20260617.sh
bash -n runs/eval_h100_fov076_phase8_promotion_20260617.sh
```

Expected:

```text
Both commands exit 0.
```

- [ ] **Step 3: Commit tracked plan/doc/code changes**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git add docs/superpowers/plans/2026-06-17-fov76-phase7b-to-phase9-plan.md
git commit -m "docs: plan fov76 phase7b to phase9 optimization"
git push
```

Expected:

```text
The plan is pushed.
No run output or large dataset artifact is committed.
```

## Final Decision Tree

```text
phase7b promotes:
    use graph8 as selected candidate;
    run filter sweep;
    mine residual failures;
    train phase8 from graph8 only if promotion is stable.

phase7b rejects but has useful target gains:
    keep phase6c active;
    mine phase7b residual failures and selector disagreements;
    train phase8 from phase6c with hard replay.

phase7b rejects with broad regressions:
    keep phase6c active;
    do not train deeper from phase7b;
    focus on filter sweep and hard failures from phase6c.

phase8 promotes:
    write new active config;
    run one controlled phase9 experiment at a time.

phase8 rejects:
    preserve best reports;
    mine the failure rows;
    adjust matcher loss or candidate_topk before any extractor change.
```

## Self-Review

Spec coverage:

```text
The plan covers current phase7b closure, candidate selection, filter sweep, hard failure mining, hard-replay training, formal promotion, matcher optimization, extractor ablation, and Git/documentation.
```

Placeholder scan:

```text
No task depends on TBD paths. Branching decisions use explicit promotion outputs.
```

Type/path consistency:

```text
The fov76 pair root, selected run roots, checkpoint paths, and run artifact names are consistent across tasks.
```
