# PFM LightGlue Overmatch Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the current PFM GraphMatcher from "more correct matches than LightGlue" to a cleaner operating point that keeps PFM's correct-match advantage while matching or beating LightGlue's precision on the fov76 val/test benchmark.

**Architecture:** Keep `last_good_pytorch_pfm_state.pt` as the protected teacher and baseline. Use the fixed post-filter profile `fov76_graph_magsac2_min24_balanced` as the main evaluation口径, mine the remaining false matches from its output, then run small matcher-only / accept-head training jobs with recall guards and promotion gates after each run.

**Tech Stack:** Python, PyTorch, OpenCV MAGSAC, existing `scripts/benchmark_lazy_pose_pairs.py`, `scripts/visualize_lazy_pose_matches.py`, `scripts/run_graph_filter_sweep.py`, `scripts/run_fov76_checkpoint_promotion_pipeline.py`, WSL `pfm_torch` environment.

---

## Current Baseline

- PFM profile: `fov76_graph_magsac2_min24_balanced`
- PFM combined val+test: `51839` correct, `884` wrong, precision `98.32%`
- LightGlue combined val+test: `40315` correct, `407` wrong, precision `99.00%`
- Optimization target: keep PFM correct matches above `48000`, reduce wrong matches below `500`, and reach precision at or above `99.0%`.

## Files And Responsibilities

- `runs/pfm_graph_filter_pareto_20260627/index.html`: Existing evidence for the selected operating point.
- `scripts/visualize_lazy_pose_matches.py`: Evaluation entry point; already supports `fov76_graph_magsac2_min24_balanced`.
- `scripts/benchmark_lazy_pose_pairs.py`: Training entry point; use matcher-only losses and periodic visual eval.
- `scripts/analyze_pfm_failure_buckets.py`: Bucket the remaining wrong matches by score, raw margin, accept probability, variant, and geometry residual.
- `scripts/build_lazy_false_match_csv.py`: Convert mined false details into replay data when applicable.
- `scripts/run_fov76_checkpoint_promotion_pipeline.py`: Formal candidate-vs-baseline gate using the same post-filter profile.
- `runs/<experiment_name>_<date>/`: Every training/eval run writes command, log, metrics CSV, checkpoints, and HTML summary here.

## Task 1: Freeze The Evaluation Contract

**Files:**
- Read: `runs/pfm_graph_filter_pareto_20260627/recommendations.json`
- Read: `runs/pfm_graph_filter_pareto_20260627/combined_sweep.csv`
- Read: `scripts/README.md`

- [ ] **Step 1: Verify the selected profile is still the current operating point**

Run:

```bash
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("runs/pfm_graph_filter_pareto_20260627/recommendations.json")
data = json.loads(path.read_text(encoding="utf-8"))
row = data["max_correct_wrong_le_1500"]
print(row["config_key"], row["correct"], row["wrong"], row["precision"])
PY
```

Expected output includes:

```text
score0_accept0.7_rawmargin0.02_magsac3_min24 55936 1352 0.976399944
```

Also manually confirm `combined_sweep.csv` contains the stricter mainline row:

```text
score18_accept0_rawmarginoff_magsac2_min24
```

- [ ] **Step 2: Record the target gate for the next experiments**

Use these thresholds for all next candidate comparisons:

```text
min_correct_combined = 48000
max_wrong_combined = 500
min_precision_combined = 0.9900
profile = fov76_graph_magsac2_min24_balanced
baseline_state = models/last_good_pytorch_pfm_state.pt
```

## Task 2: Mine The Remaining 884 Wrong Matches

**Files:**
- Read: `runs/corrected_lastgood_graph_vs_lightglue_20260627/val_lastgood_graph_magsac/all_match_details.csv`
- Read: `runs/corrected_lastgood_graph_vs_lightglue_20260627/test_lastgood_graph_magsac/all_match_details.csv`
- Create: `runs/pfm_wrong884_mining_20260627/`

- [ ] **Step 1: Bucket false matches by observable features**

Run:

```bash
mkdir -p runs/pfm_wrong884_mining_20260627
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/analyze_pfm_failure_buckets.py \
  --match-details runs/corrected_lastgood_graph_vs_lightglue_20260627/val_lastgood_graph_magsac/all_match_details.csv \
  --output-dir runs/pfm_wrong884_mining_20260627/val \
  --high-score-min 18 \
  --high-accept-min 0.0 \
  --high-raw-margin-min 0.0
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/analyze_pfm_failure_buckets.py \
  --match-details runs/corrected_lastgood_graph_vs_lightglue_20260627/test_lastgood_graph_magsac/all_match_details.csv \
  --output-dir runs/pfm_wrong884_mining_20260627/test \
  --high-score-min 18 \
  --high-accept-min 0.0 \
  --high-raw-margin-min 0.0
```

Expected artifacts:

```text
runs/pfm_wrong884_mining_20260627/val/index.html
runs/pfm_wrong884_mining_20260627/test/index.html
```

- [ ] **Step 2: Decide the first training target**

Use the bucket report to classify the wrong matches into exactly one of these first-run targets:

```text
target_a = high_score_high_geometry_outlier
target_b = repeated_texture_false_cluster
target_c = low_overlap_pair_accept_failure
```

Pick `target_a` if most wrong matches have high graph scores but large true warp residuals. Pick `target_b` if errors cluster around repeated terrain texture. Pick `target_c` if wrong matches concentrate on low-valid-fraction or weak-overlap pairs. Pick `target_d = near_miss_localization` if most wrong matches are in the `5-8px` error bins; in that case run an offset/localization evaluation before hard-negative training, because rejecting those matches can damage PFM's recall advantage.

## Task 2.5: Validate Keypoint Offsets Before Training

**Files:**
- Read: `models/last_good_pytorch_pfm_state.pt`
- Read: `runs/pfm_wrong884_mining_20260627/*/buckets/error_bin_summary.csv`
- Create: `runs/eval_lastgood_offsets_balanced_profile_20260627/`

- [ ] **Step 1: Re-evaluate last_good with keypoint offsets enabled**

Run:

```bash
bash runs/eval_lastgood_offsets_balanced_profile_20260627.sh
```

Expected artifacts:

```text
runs/eval_lastgood_offsets_balanced_profile_20260627/index.html
runs/eval_lastgood_offsets_balanced_profile_20260627/summary.csv
runs/eval_lastgood_offsets_balanced_profile_20260627/summary.json
```

- [ ] **Step 2: Decide whether offsets are already useful**

Use this rule:

```text
If wrong decreases by at least 150 and correct stays >= 48000:
    enable --use-keypoint-offsets in the balanced profile evaluation path.
If wrong does not decrease:
    train selected keypoint offset / soft-boundary losses before hard-negative losses.
If correct drops below 48000:
    keep offsets disabled and use soft-boundary training only as a guarded experiment.
```

## Task 3: Run A Small Matcher-Only Precision Training

**Files:**
- Read: `models/last_good_pytorch_pfm_state.pt`
- Read: `E:/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/spatial_pair_specs_train.csv`
- Create: `runs/phase_pfm_precision_matcher_only_20260627/`

- [ ] **Step 1: Launch a 10k-step matcher-only precision run**

Only run this exact hard-negative command if Task 2 does not classify the dominant failures as `target_d = near_miss_localization`. If near-miss dominates, replace the hard-negative-first run with a localization-first run using `--selected-keypoint-offset-weight` and `--graph-matcher-warp-soft-boundary-weight`, then evaluate with Task 4 before adding stronger false-match rejection.

Run from WSL:

```bash
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/benchmark_lazy_pose_pairs.py \
  --render-manifest /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/manifests/h100km_fov076_render_manifest.csv \
  --uint8-manifest runs/lastgood_vs_verified_vf005_noaug_noheads_lr5e6_10k_best_eval_20260627/empty_uint8_manifest.csv \
  --output-dir runs/phase_pfm_precision_matcher_only_20260627 \
  --mode train \
  --pair-spec-manifest /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/spatial_pair_specs_train.csv \
  --image-source render \
  --init-pytorch-state models/last_good_pytorch_pfm_state.pt \
  --device cuda \
  --steps 10000 \
  --batch-pairs 2 \
  --crop-size 1536 \
  --training-max-image-size 768 \
  --train-graph-matcher \
  --train-graph-calibration-only \
  --graph-matcher-warp-outlier-weight 0.10 \
  --graph-matcher-warp-outlier-topk 64 \
  --graph-matcher-warp-outlier-residual-threshold-px 5.0 \
  --graph-matcher-warp-outlier-min-score 18.0 \
  --graph-matcher-warp-outlier-margin 0.20 \
  --graph-matcher-warp-outlier-accept-weight 0.10 \
  --graph-matcher-warp-outlier-accept-topk 64 \
  --graph-matcher-warp-outlier-accept-residual-threshold-px 5.0 \
  --graph-matcher-warp-outlier-accept-min-score 18.0 \
  --graph-matcher-teacher-score-floor-weight 0.05 \
  --graph-matcher-teacher-score-floor-tolerance 0.05 \
  --graph-matcher-teacher-score-floor-min-score 18.0 \
  --visual-eval-every-steps 2000 \
  --visual-post-filter-profile fov76_graph_magsac2_min24_balanced \
  --batched-descriptor-forward \
  --gpu-monitor
```

Expected training signs:

```text
graph_matcher_warp_outlier_edges > 0
graph_matcher_warp_outlier_accept_edges > 0
graph_matcher_teacher_score_floor_violations remains bounded
visual_filtered_wrong decreases without visual_filtered_correct collapse
```

- [ ] **Step 2: Stop early if recall collapses**

Abort the run if a visual report row shows:

```text
visual_filtered_correct drops by more than 10% from the last_good baseline
visual_filtered_precision does not improve after 4000 steps
```

## Task 4: Evaluate Candidate Against LightGlue And Last Good

**Files:**
- Read: `runs/phase_pfm_precision_matcher_only_20260627/pytorch_pfm_state.pt`
- Create: `runs/phase_pfm_precision_matcher_only_eval_20260627/`

- [ ] **Step 1: Run formal val/test evaluation with the fixed profile**

Run:

```bash
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap \
  --guard-root /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/hard_mining/guard \
  --output-dir runs/phase_pfm_precision_matcher_only_eval_20260627 \
  --baseline-state models/last_good_pytorch_pfm_state.pt \
  --baseline-run-dir runs/corrected_lastgood_graph_vs_lightglue_20260627 \
  --candidate-state runs/phase_pfm_precision_matcher_only_20260627/pytorch_pfm_state.pt \
  --candidate-run-dir runs/phase_pfm_precision_matcher_only_20260627 \
  --candidate-label precision_matcher_only \
  --guard-candidate-label precision_matcher_only \
  --post-filter-profile fov76_graph_magsac2_min24_balanced \
  --formal-candidate-pairs 512 \
  --guard-candidate-pairs 512 \
  --write-match-details
```

Expected artifacts:

```text
runs/phase_pfm_precision_matcher_only_eval_20260627/index.html
runs/phase_pfm_precision_matcher_only_eval_20260627/formal_summary.csv
runs/phase_pfm_precision_matcher_only_eval_20260627/promotion_decision.json
```

- [ ] **Step 2: Apply promotion rule**

Promote only if combined val+test satisfies:

```text
candidate_correct >= 48000
candidate_wrong <= 500
candidate_precision >= 0.9900
candidate_correct > 40315
```

If `candidate_wrong <= 650` and `candidate_correct >= 50000`, keep it as a near-miss checkpoint and run Task 5.

## Task 5: If Needed, Add A Pair-Level Rejection Pass

**Files:**
- Read: candidate `all_filtered_summary.csv`
- Create: `runs/phase_pfm_pair_rejection_20260627/`

- [ ] **Step 1: Train only the pair accept head if false matches are pair-clustered**

Run this only when Task 2 shows wrong matches concentrated in a small set of low-overlap or unstable pairs:

```bash
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/benchmark_lazy_pose_pairs.py \
  --render-manifest /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/manifests/h100km_fov076_render_manifest.csv \
  --uint8-manifest runs/lastgood_vs_verified_vf005_noaug_noheads_lr5e6_10k_best_eval_20260627/empty_uint8_manifest.csv \
  --output-dir runs/phase_pfm_pair_rejection_20260627 \
  --mode train \
  --pair-spec-manifest /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/spatial_pair_specs_train.csv \
  --image-source render \
  --init-pytorch-state runs/phase_pfm_precision_matcher_only_20260627/pytorch_pfm_state.pt \
  --device cuda \
  --steps 5000 \
  --batch-pairs 2 \
  --crop-size 1536 \
  --training-max-image-size 768 \
  --train-graph-matcher \
  --train-pair-accept-head-only \
  --graph-matcher-accept-weight 0.20 \
  --graph-matcher-teacher-score-floor-weight 0.08 \
  --graph-matcher-teacher-score-floor-min-score 18.0 \
  --visual-eval-every-steps 1000 \
  --visual-post-filter-profile fov76_graph_magsac2_min24_balanced \
  --batched-descriptor-forward \
  --gpu-monitor
```

Expected behavior:

```text
pair_accept reject count increases on weak pairs
visual_filtered_wrong decreases
visual_filtered_correct does not drop below 48000 in formal evaluation
```

## Task 6: Final Comparison Report

**Files:**
- Read: best candidate evaluation directory
- Read: `runs/lightglue_baseline_fov76_20260627/`
- Create: `runs/pfm_overmatch_final_report_20260627/`

- [ ] **Step 1: Produce final side-by-side report**

Use existing comparison outputs and write an HTML summary with:

```text
PFM last_good raw graph + MAGSAC-min16
PFM last_good fov76_graph_magsac2_min24_balanced
PFM candidate fov76_graph_magsac2_min24_balanced
LightGlue-SIFT-MAGSAC-min16
```

The report must show:

```text
matches
correct
wrong
precision
correct_delta_vs_lightglue
wrong_delta_vs_lightglue
per-split val/test
per-variant nadir/mid/extreme
```

- [ ] **Step 2: Decide the next branch**

Use this rule:

```text
If precision >= LightGlue and correct > LightGlue by at least 5000:
    freeze candidate as new best model
If correct > LightGlue by at least 10000 but wrong remains 500-800:
    run one more accept-head-only pass
If correct falls below 48000:
    revert to last_good and mine recall-preservation failures
```

## Risk Controls

- Do not train on val/test pairs directly.
- Do not use LightGlue predictions as training labels.
- Keep `models/last_good_pytorch_pfm_state.pt` unchanged as the protected teacher.
- Every candidate must be evaluated with `fov76_graph_magsac2_min24_balanced`.
- A lower wrong count is not enough if correct matches collapse below `48000`.

## Self-Review

- Spec coverage: covers false-match analysis, precision training, recall guard, formal evaluation, and final comparison.
- Placeholder scan: no placeholder steps remain; every command has concrete paths and expected outputs.
- Type consistency: profile name and metric names match current scripts and reports.
