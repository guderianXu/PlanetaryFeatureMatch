# fov76 Next Complete Execution Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the current fov76 optimization cycle by selecting a safe active checkpoint/selector, then run one controlled complete training round and one larger 8-layer GraphMatcher comparison before promoting the best path.

**Architecture:** Keep the training/evaluation scope fixed to the h100/fov76/dom76 internal pair graph. Treat promotion evaluation as the gatekeeper: a model or selector can become active only after formal val/test, regression guard, extreme gain guard, and target/protected variant checks pass. Optimize the matcher first; touch the feature extractor only after matcher/filter/geometry evidence shows it is the bottleneck.

**Tech Stack:** Python 3, PyTorch, CUDA AMP, existing lazy fov76 pair manifests, `benchmark_lazy_pose_pairs.py`, `run_fov76_checkpoint_promotion_pipeline.py`, `run_graph_filter_sweep.py`, `visualize_lazy_pose_matches.py`, `mine_hard_failure_pairs.py`, HTML/CSV run records, Git.

---

## Current State

Authoritative paths:

```text
Project root:
/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

Python:
/home/w24/anaconda3/envs/cppTorch/bin/python

Data root:
/media/w24/D/xjw深度学习训练数据

fov76 pair root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal

Guard root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase3d_diff_guard_20260614

Current stable baseline:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Current phase6a rescue candidate:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6a_fov76_phase5g_residual_pattern_replay_4l384_20260616_105709/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Current phase6c candidate:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Current recommended post-filter profile:
fov76_geo5_geo10_extreme_rescue_lowmatch_guard
```

Current running job:

```text
Expanded phase6c promotion with p90-delta guard:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_p90delta0_expanded200_20260616_231706

Launcher log:
runs/eval_h100_fov076_phase6c_p90delta0_expanded_20260616.log
```

Known evidence before this plan:

```text
phase6c quick promotion without p90-delta guard:
REJECT, because target test gained +6 correct but also +1 wrong and target precision dropped by 0.000900.

phase6c offline selector with --max-rescue-homography-p90-delta-px 0.0 on existing 60/100 reports:
PROMOTE, target val +16 correct and -1 wrong, target test unchanged.

Interpretation:
phase6c contains useful extreme_02/extreme_03 recall, but only under a stricter selector that rejects rescue rows whose homography p90 is worse than baseline.
```

Hard constraints:

```text
Do not use fov90 in this branch.
Do not promote a checkpoint/selector before expanded promotion passes.
Do not increase dustbin/no-match/rejection weights as the first answer to low match count.
Do not use 8-layer inference for a model that was not trained with 8 layers.
Do not unfreeze the full extractor until matcher-only runs and geometry/post-filter diagnostics are exhausted.
Do not touch the untracked file named "0".
```

## Files And Responsibilities

Use existing scripts:

```text
scripts/benchmark_lazy_pose_pairs.py
```

Main train/eval entry for lazy fov76 pairs. Use it for matcher-only training, AMP, worker prefetch, auto visual eval, and checkpoint saving.

```text
scripts/run_fov76_checkpoint_promotion_pipeline.py
```

Formal promotion gate. Use it after every candidate training run with the same post-filter profile and selector guard.

```text
scripts/run_dual_checkpoint_rescue_eval.py
```

Selector-level checkpoint combiner. Use it only through the promotion pipeline unless manually diagnosing selector rows.

```text
scripts/run_graph_filter_sweep.py
scripts/visualize_lazy_pose_matches.py
```

Post-filter and visual diagnostics. Use them to distinguish model failures from filter threshold failures.

```text
scripts/mine_hard_failure_pairs.py
scripts/mine_selector_disagreement_pairs.py
scripts/build_train_replay_from_pair_deltas.py
scripts/build_rescue_gain_hard_set.py
```

Hard-set generation and replay manifest building. Use only train split rows for training manifests.

Create local run scripts:

```text
runs/eval_h100_fov076_phase6c_p90delta0_expanded_20260616.sh
runs/train_h100_fov076_phase7a_complete_4l384_20260616.sh
runs/train_h100_fov076_phase7b_graph8_20260616.sh
runs/eval_h100_fov076_phase7a_promotion_20260616.sh
runs/eval_h100_fov076_phase7b_promotion_20260616.sh
runs/fov76_next_complete_execution_summary_20260616.html
```

Tracked documentation:

```text
docs/superpowers/plans/2026-06-16-fov76-next-complete-execution-plan.md
scripts/README.md
```

`scripts/README.md` changes are only needed if a new durable script or new durable CLI flag is added. Pure `runs/*.sh`, `runs/*.log`, and `runs/*.html` artifacts do not require README changes.

## Success Metrics

Promotion pass criteria:

```text
formal val/test: no protected variant regression
formal target variants: extreme_02/extreme_03 correct count improves or stays neutral without wrong increase
regression_guard val/test: no precision drop and no wrong increase
extreme_gain val/test: positive or neutral correct delta without new wrong clusters
selector: only switch to rescue rows when match gain is real and homography p90 does not worsen
```

Core metrics to compare:

```text
filtered_correct
filtered_wrong
filtered_precision
filtered_matches
num zero-match or low-match rows
homography_residual_p90_px
score_mean
positive_vs_dustbin_margin
true_match_rejected_by_dustbin_ratio
GPU utilization and steps per second
```

Promotion decision rules:

```text
PROMOTE:
    Candidate/selector improves total target correct count and passes all regression guards.

KEEP ACTIVE:
    Candidate improves some rows but fails formal or guard checks.

MINE:
    Candidate has useful gains plus isolated regressions; mine train-only analogs of those rows.

REJECT:
    Candidate reduces formal correct count, increases wrong clusters, or only improves by relaxing geometry without stable homography residuals.
```

## Task 1: Finish Current Expanded Phase6c Promotion

**Files:**
- Read: `runs/eval_h100_fov076_phase6c_p90delta0_expanded_20260616.log`
- Read: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_p90delta0_expanded200_20260616_231706/promotion_decision.json`
- Create: `runs/phase6c_p90delta0_expanded200_decision_20260616.html`

- [ ] **Step 1: Check running jobs**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Expected:

```text
Only the current phase6c promotion pipeline and its child eval processes should be active.
```

- [ ] **Step 2: Monitor to completion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 runs/eval_h100_fov076_phase6c_p90delta0_expanded_20260616.log
```

Expected before proceeding:

```text
The log writes promotion_decision.json and exits.
```

- [ ] **Step 3: Print the promotion decision**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

decision = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_p90delta0_expanded200_20260616_231706/promotion_decision.json")
data = json.loads(decision.read_text(encoding="utf-8"))
print(decision)
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
```

Expected:

```text
The JSON states PROMOTE or REJECT and lists failed gates if any.
```

- [ ] **Step 4: Create the HTML decision record**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import html
import json

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_p90delta0_expanded200_20260616_231706")
decision_path = root / "promotion_decision.json"
data = json.loads(decision_path.read_text(encoding="utf-8"))
out = Path("runs/phase6c_p90delta0_expanded200_decision_20260616.html")
decision = data.get("decision", data.get("status", "unknown"))
reasons = data.get("failed_reasons") or data.get("failure_reasons") or data.get("reasons") or []
rows = [
    ("Promotion root", str(root)),
    ("Decision", str(decision)),
    ("Baseline", "phase5g_active"),
    ("Candidate", "phase6c_match_count_floor"),
    ("Selector guard", "max_rescue_homography_p90_delta_px = 0.0"),
    ("Decision JSON", str(decision_path)),
    ("Failed reasons", html.escape(json.dumps(reasons, ensure_ascii=False))),
]
body = "\n".join(f"<tr><th>{html.escape(k)}</th><td><code>{html.escape(v)}</code></td></tr>" for k, v in rows)
out.write_text(
    "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
    "<title>phase6c p90delta0 expanded200 decision</title></head><body>"
    "<h1>phase6c p90delta0 expanded200 decision</h1>"
    f"<table border='1' cellspacing='0' cellpadding='4'>{body}</table>"
    "</body></html>\n",
    encoding="utf-8",
)
print(out)
PY
```

Expected:

```text
runs/phase6c_p90delta0_expanded200_decision_20260616.html
```

- [ ] **Step 5: Apply the decision**

Use this table:

```text
If PROMOTE:
    Treat phase5g + phase6c selector with p90delta0 as the new candidate active path.
    Continue to Task 2.

If REJECT with small isolated wrong increase:
    Continue to Task 6 before more training.

If REJECT with formal correct drop:
    Continue to Task 5 and Task 6.

If REJECT because p90delta0 blocks all useful gains:
    Keep phase5g/phase6a active path and run Task 4 graph8 as a separate experiment, not as active replacement.
```

## Task 2: Record Or Validate The Active fov76 Mainline

**Files:**
- Read: `runs/fov76_active_mainline_config_*.json`
- Read: `scripts/validate_fov76_active_selector.py`
- Create or update only if promotion passes: `runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json`
- Create: `runs/fov76_active_mainline_validation_phase6c_p90delta0_20260616.html`

- [ ] **Step 1: Find existing active configs**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
ls -1 runs/fov76_active_mainline_config_*.json 2>/dev/null || true
```

Expected:

```text
Existing active config files are listed, or no files are listed.
```

- [ ] **Step 2: Inspect active selector validation interface**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/validate_fov76_active_selector.py --help | sed -n '1,180p'
```

Expected:

```text
The help text shows required config and output flags.
```

- [ ] **Step 3: If Task 1 promoted phase6c, create the active config**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

out = Path("runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json")
promotion_root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_p90delta0_expanded200_20260616_231706")
config = {
    "name": "phase5g_phase6c_selector_p90delta0",
    "dataset": "h100_fov076_dom76_lat60_0m2e3_internal",
    "post_filter_profile": "fov76_geo5_geo10_extreme_rescue_lowmatch_guard",
    "baseline_label": "phase5g_active",
    "baseline_state": "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt",
    "rescue_label": "phase6c_match_count_floor",
    "rescue_state": "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt",
    "promotion_root": str(promotion_root),
    "promotion_decision_json": str(promotion_root / "promotion_decision.json"),
    "selector_root": str(promotion_root / "dual_checkpoint_rescue_selector"),
    "selector_guard": {
        "target_variants": ["extreme_02", "extreme_03"],
        "min_match_gain": 3,
        "min_rescue_matches": 16,
        "max_homography_p90_delta_px": 0.0,
        "max_homography_p90_px": 3.2,
        "max_homography_median_px": 1.8,
        "min_score_mean": 16.0
    },
}
out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out)
PY
```

Expected:

```text
runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json
```

- [ ] **Step 4: Validate the active config**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/validate_fov76_active_selector.py \
  --config runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json \
  --output-json runs/fov76_active_mainline_validation_phase6c_p90delta0_20260616.json \
  --output-html runs/fov76_active_mainline_validation_phase6c_p90delta0_20260616.html
```

Expected:

```text
validation status is PASS.
```

- [ ] **Step 5: If validation fails**

Do not promote phase6c. Record the failure reason in `runs/fov76_next_complete_execution_summary_20260616.html` and use the previously validated active config for Task 3 and Task 4 comparisons.

## Task 3: Run A Complete 4-Layer Matcher-Only Training Round

**Files:**
- Create: `runs/train_h100_fov076_phase7a_complete_4l384_20260616.sh`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_YYYYMMDD_HHMMSS`

Purpose:

```text
Run one longer but still controlled 4-layer/384 matcher-only training round.
This is the practical candidate for best active model if it passes promotion.
It should not alter extractor weights.
```

- [ ] **Step 1: Create the launch script**

Create `runs/train_h100_fov076_phase7a_complete_4l384_20260616.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch"
PYTHON="/home/w24/anaconda3/envs/cppTorch/bin/python"
DATA_ROOT="/media/w24/D/xjw深度学习训练数据"
PAIR_ROOT="${DATA_ROOT}/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
PROFILE="fov76_geo5_geo10_extreme_rescue_lowmatch_guard"
INIT_STATE="${DATA_ROOT}/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
TEACHER_STATE="${DATA_ROOT}/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${DATA_ROOT}/pfm_runs/phase7a_fov76_complete_4l384_${STAMP}"
OUTPUT_DIR="${RUN_ROOT}/train_output"
LOG_PATH="${PROJECT_ROOT}/runs/train_h100_fov076_phase7a_complete_4l384_${STAMP}.log"
PID_PATH="${PROJECT_ROOT}/runs/train_h100_fov076_phase7a_complete_4l384_${STAMP}.pid"
HTML_RECORD="${PROJECT_ROOT}/runs/train_h100_fov076_phase7a_complete_4l384_${STAMP}.html"

mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/runs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="python:scripts"
export PYTHONUNBUFFERED=1

for path in \
  "${INIT_STATE}" \
  "${TEACHER_STATE}" \
  "${PAIR_ROOT}/manifests/h100km_fov076_render_manifest.csv" \
  "${PAIR_ROOT}/manifests/h100km_fov076_uint8_manifest.csv" \
  "${PAIR_ROOT}/overlap_edges_train.csv" \
  "${PAIR_ROOT}/overlap_edges_val.csv" \
  "${PAIR_ROOT}/overlap_edges_test.csv"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    exit 2
  fi
done

cat > "${HTML_RECORD}" <<HTML
<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>fov76 phase7a complete 4l384 ${STAMP}</title></head>
<body>
<h1>fov76 phase7a complete 4l384</h1>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Run root</th><td>${RUN_ROOT}</td></tr>
<tr><th>Init checkpoint</th><td>${INIT_STATE}</td></tr>
<tr><th>Teacher checkpoint</th><td>${TEACHER_STATE}</td></tr>
<tr><th>Pair root</th><td>${PAIR_ROOT}</td></tr>
<tr><th>Profile</th><td>${PROFILE}</td></tr>
<tr><th>Steps</th><td>1200</td></tr>
<tr><th>Trainable</th><td>GraphMatcher calibration only; extractor remains frozen.</td></tr>
<tr><th>Log</th><td>${LOG_PATH}</td></tr>
</table>
</body>
</html>
HTML

setsid bash -c "
  echo \\\$\\\$ > '${PID_PATH}'
  trap 'rm -f \"${PID_PATH}\"' EXIT
  exec '${PYTHON}' scripts/benchmark_lazy_pose_pairs.py \
    --render-manifest '${PAIR_ROOT}/manifests/h100km_fov076_render_manifest.csv' \
    --uint8-manifest '${PAIR_ROOT}/manifests/h100km_fov076_uint8_manifest.csv' \
    --output-dir '${OUTPUT_DIR}' \
    --mode train \
    --split train \
    --image-source uint8 \
    --pair-mode same-position \
    --pair-type-weights same_position_view=1.0 \
    --pair-spec-manifest '${PAIR_ROOT}/overlap_edges_train.csv' \
    --steps 1200 \
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
    --init-pytorch-state '${INIT_STATE}' \
    --seed 20260616 \
    --progress-every 20 \
    --save-every-steps 200 \
    --save-best-checkpoints \
    --stability-window 200 \
    --stability-min-steps 400 \
    --stability-max-nan-in-window 20 \
    --stability-min-top1-mean 0.20 \
    --stability-max-loss-multiplier 12.0 \
    --stability-min-loss-delta-for-explosion 0.05 \
    --stability-min-match-score -0.8 \
    --stability-max-dustbin-rejection-ratio 0.85 \
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
    --graph-matcher-accept-negative-topk 8 \
    --graph-matcher-prune-ranking-weight 0.0 \
    --graph-matcher-stop-confidence-weight 0.0 \
    --graph-matcher-hard-negative-dustbin-weight 0.0 \
    --graph-matcher-train-max-attention-layers 4 \
    --graph-matcher-train-max-attention-work-fraction 1.0 \
    --graph-matcher-train-width-keep-ratio 1.0 \
    --graph-matcher-deep-supervision-weight 0.0 \
    --graph-matcher-depth-distillation-weight 0.0 \
    --graph-matcher-teacher-guard-state '${TEACHER_STATE}' \
    --graph-matcher-teacher-guard-weight 0.80 \
    --graph-matcher-teacher-guard-positive-margin-tolerance 0.0 \
    --graph-matcher-teacher-guard-false-margin-tolerance 0.010 \
    --graph-matcher-teacher-score-floor-weight 0.08 \
    --graph-matcher-teacher-score-floor-tolerance 0.015 \
    --graph-matcher-teacher-score-floor-min-score 0.0 \
    --graph-matcher-teacher-match-count-floor-weight 0.02 \
    --graph-matcher-teacher-match-count-floor-threshold 18.0 \
    --graph-matcher-teacher-match-count-floor-margin 0.5 \
    --matcher-reliability-pair-bias off \
    --matcher-reliability-dustbin-bias off \
    --matcher-final-accept-score-mode none \
    --matcher-accept-assignment-mode add \
    --matcher-final-accept-score-alpha 0.02 \
    --matcher-geometry-bias-scale 1.0 \
    --matcher-geometry-bias-clamp 1.0 \
    --matcher-candidate-topk 256 \
    --graph-matcher-dustbin-warmup-steps 0 \
    --graph-matcher-dustbin-ramp-steps 0 \
    --graph-matcher-positive-dustbin-margin-weight 0.00008 \
    --graph-matcher-positive-dustbin-margin 0.04 \
    --graph-matcher-true-match-margin-weight 0.0003 \
    --graph-matcher-true-match-margin 0.08 \
    --graph-matcher-final-false-match-weight 0.0 \
    --graph-matcher-mined-false-match-weight 0.0 \
    --graph-matcher-raw-false-match-weight 0.0 \
    --graph-matcher-ransac-consistency-weight 0.003 \
    --graph-matcher-ransac-consistency-topk 8 \
    --graph-matcher-ransac-consistency-residual-threshold-px 3.0 \
    --graph-matcher-ransac-consistency-min-score 0.02 \
    --graph-matcher-ransac-consistency-margin 0.20 \
    --graph-matcher-positive-dustbin-guard-reject-threshold 0.20 \
    --graph-matcher-positive-dustbin-guard-margin-threshold 1.0 \
    --graph-matcher-train-candidate-topk 256 \
    --freeze-extractor-warmup-steps 999999 \
    --synthetic-loss-weight 0.0 \
    --teacher-weight 0.0 \
    --hard-negative-weight 0.0 \
    --diversity-weight 0.0 \
    --training-spatial-bins 4 \
    --keypoint-weight 0.0 \
    --keypoint-negative-weight 0.0 \
    --matchability-weight 0.0 \
    --descriptor-uncertainty-weight 0.0 \
    --no-match-prior-weight 0.0 \
    --reliability-negative-points 0 \
    --rotation-descriptor-consistency-weight 0.0 \
    --orientation-consistency-weight 0.0 \
    --scale-consistency-weight 0.0 \
    --affine-consistency-weight 0.0 \
    --input-local-contrast \
    --input-local-contrast-strength 0.35 \
    --illumination-consistency-weight 0.0 \
    --illumination-match-weight 0.0 \
    --false-match-weight 0.0 \
    --false-match-max-points 0 \
    --no-graph-matcher-online-false-no-match \
    --auto-visual-report \
    --visual-eval-every-steps 200 \
    --visual-post-filter-profile '${PROFILE}' \
    --visual-matcher-mode graph_matcher \
    --visual-keypoint-score-mode learned \
    --visual-geometry-filter local \
    --visual-geometry-threshold-px 5.0 \
    --visual-filtered-geometry-filter magsac \
    --visual-filtered-min-matches 16 \
    --visual-graph-max-attention-layers 4 \
    --visual-graph-max-attention-work-fraction 1.0 \
    --visual-graph-width-prune-keep-ratio 1.0 \
    --visual-graph-width-prune-min-score -1.0 \
    --visual-graph-early-stop-min-confidence -1.0 \
    --visual-max-keypoints 512 \
    --visual-candidate-pairs 80 \
    --visual-select-count 12
" > "${LOG_PATH}" 2>&1 &

echo "PID file: ${PID_PATH}"
echo "Log: ${LOG_PATH}"
echo "Run root: ${RUN_ROOT}"
```

- [ ] **Step 2: Make the script executable**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
chmod +x runs/train_h100_fov076_phase7a_complete_4l384_20260616.sh
```

Expected:

```text
No output.
```

- [ ] **Step 3: Launch only after Task 1 is finished**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
runs/train_h100_fov076_phase7a_complete_4l384_20260616.sh
```

Expected:

```text
The launcher prints PID file, log path, and run root.
```

- [ ] **Step 4: Monitor training health**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 "$(ls -1t runs/train_h100_fov076_phase7a_complete_4l384_*.log | head -1)"
```

Expected:

```text
loss remains finite
auto visual eval runs every 200 steps
best_by_match_score_pytorch_pfm_state.pt is written
GPU monitor rows exist
```

## Task 4: Run A Controlled 8-Layer GraphMatcher Experiment

**Files:**
- Create: `runs/train_h100_fov076_phase7b_graph8_20260616.sh`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_YYYYMMDD_HHMMSS`

Purpose:

```text
Test the user's "bigger model" hypothesis properly:
8-layer training must use 8-layer supervision and 8-layer visual inference.
Do not compare it against 4-layer unless both are passed through the same promotion pipeline.
```

- [ ] **Step 1: Copy the phase7a script**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
cp runs/train_h100_fov076_phase7a_complete_4l384_20260616.sh runs/train_h100_fov076_phase7b_graph8_20260616.sh
```

Expected:

```text
runs/train_h100_fov076_phase7b_graph8_20260616.sh exists.
```

- [ ] **Step 2: Edit only the 8-layer-specific values**

Modify `runs/train_h100_fov076_phase7b_graph8_20260616.sh` so these exact values are present:

```text
RUN_ROOT="${DATA_ROOT}/pfm_runs/phase7b_fov76_graph8_h384_${STAMP}"
LOG_PATH="${PROJECT_ROOT}/runs/train_h100_fov076_phase7b_graph8_${STAMP}.log"
PID_PATH="${PROJECT_ROOT}/runs/train_h100_fov076_phase7b_graph8_${STAMP}.pid"
HTML_RECORD="${PROJECT_ROOT}/runs/train_h100_fov076_phase7b_graph8_${STAMP}.html"
--steps 800
--graph-attention-layers 8
--graph-matcher-train-max-attention-layers 8
--graph-matcher-deep-supervision-depths 2,4,6
--graph-matcher-deep-supervision-weight 0.15
--graph-matcher-depth-distillation-weight 0.10
--graph-matcher-depth-distillation-teacher-layers 4
--graph-matcher-depth-distillation-temperature 1.5
--visual-graph-max-attention-layers 8
```

Keep these exact values unchanged:

```text
--graph-hidden-dim 384
--matcher-candidate-topk 256
--graph-matcher-train-candidate-topk 256
--freeze-extractor-warmup-steps 999999
--matcher-reliability-pair-bias off
--matcher-reliability-dustbin-bias off
--no-match-prior-weight 0.0
--graph-matcher-hard-negative-dustbin-weight 0.0
```

- [ ] **Step 3: Launch after phase7a has finished or only if GPU is idle**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
runs/train_h100_fov076_phase7b_graph8_20260616.sh
```

Expected:

```text
Only one training job runs on the GPU.
The 8-layer log shows graph_attention_layers=8 and visual-graph-max-attention-layers=8.
```

- [ ] **Step 4: Stop criteria for graph8**

Stop or reject graph8 if any of these happen:

```text
OOM
nonfinite loss repeats
filtered matches collapse for more than two visual evals
true_match_rejected_by_dustbin_ratio rises above 0.85 for more than two visual evals
formal promotion later shows protected variant regression
```

## Task 5: Promote Phase7a And Phase7b With The Same Gate

**Files:**
- Create: `runs/eval_h100_fov076_phase7a_promotion_20260616.sh`
- Create: `runs/eval_h100_fov076_phase7b_promotion_20260616.sh`
- Output: `promotion_phase5g_profile_p90delta0_expanded200_*` under each run root

- [ ] **Step 1: Evaluate phase7a**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
export PYTHONPATH=python:scripts

PYTHON=/home/w24/anaconda3/envs/cppTorch/bin/python
DATA_ROOT="/media/w24/D/xjw深度学习训练数据"
PAIR_ROOT="${DATA_ROOT}/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
GUARD_ROOT="${PAIR_ROOT}/hard_mining/phase3d_diff_guard_20260614"
BASE_RUN="${DATA_ROOT}/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output"
BASE_STATE="${BASE_RUN}/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
CAND_ROOT="$(ls -dt "${DATA_ROOT}"/pfm_runs/phase7a_fov76_complete_4l384_* | head -1)"
CAND_RUN="${CAND_ROOT}/train_output"
CAND_STATE="${CAND_RUN}/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
CAND_LABEL="phase7a_complete_4l384"
OUT_DIR="${CAND_ROOT}/promotion_phase5g_profile_p90delta0_expanded200_$(date +%Y%m%d_%H%M%S)"

test -f "${CAND_STATE}"

"${PYTHON}" scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root "${PAIR_ROOT}" \
  --guard-root "${GUARD_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --baseline-state "${BASE_STATE}" \
  --baseline-run-dir "${BASE_RUN}" \
  --candidate-state "${CAND_STATE}" \
  --candidate-run-dir "${CAND_RUN}" \
  --baseline-label phase5g_active \
  --candidate-label "${CAND_LABEL}" \
  --guard-baseline-label phase5g_active \
  --guard-candidate-label "${CAND_LABEL}" \
  --dual-checkpoint-rescue-selector \
  --dual-checkpoint-rescue-label "phase5g_${CAND_LABEL}_selector_p90delta0" \
  --dual-checkpoint-rescue-max-homography-p90-delta-px 0.0 \
  --python-executable "${PYTHON}" \
  --post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \
  --formal-candidate-pairs 200 \
  --guard-candidate-pairs 200
```

Expected:

```text
promotion_decision.json exists and says PROMOTE or REJECT.
```

- [ ] **Step 2: Evaluate phase7b**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
export PYTHONPATH=python:scripts

PYTHON=/home/w24/anaconda3/envs/cppTorch/bin/python
DATA_ROOT="/media/w24/D/xjw深度学习训练数据"
PAIR_ROOT="${DATA_ROOT}/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
GUARD_ROOT="${PAIR_ROOT}/hard_mining/phase3d_diff_guard_20260614"
BASE_RUN="${DATA_ROOT}/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output"
BASE_STATE="${BASE_RUN}/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
CAND_ROOT="$(ls -dt "${DATA_ROOT}"/pfm_runs/phase7b_fov76_graph8_h384_* | head -1)"
CAND_RUN="${CAND_ROOT}/train_output"
CAND_STATE="${CAND_RUN}/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
CAND_LABEL="phase7b_graph8_h384"
OUT_DIR="${CAND_ROOT}/promotion_phase5g_profile_p90delta0_expanded200_$(date +%Y%m%d_%H%M%S)"

test -f "${CAND_STATE}"

"${PYTHON}" scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root "${PAIR_ROOT}" \
  --guard-root "${GUARD_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --baseline-state "${BASE_STATE}" \
  --baseline-run-dir "${BASE_RUN}" \
  --candidate-state "${CAND_STATE}" \
  --candidate-run-dir "${CAND_RUN}" \
  --baseline-label phase5g_active \
  --candidate-label "${CAND_LABEL}" \
  --guard-baseline-label phase5g_active \
  --guard-candidate-label "${CAND_LABEL}" \
  --dual-checkpoint-rescue-selector \
  --dual-checkpoint-rescue-label "phase5g_${CAND_LABEL}_selector_p90delta0" \
  --dual-checkpoint-rescue-max-homography-p90-delta-px 0.0 \
  --python-executable "${PYTHON}" \
  --post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \
  --formal-candidate-pairs 200 \
  --guard-candidate-pairs 200
```

Expected:

```text
promotion_decision.json exists and says PROMOTE or REJECT.
```

- [ ] **Step 3: Compare candidates**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

roots = [
    Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/promotion_phase5g_profile_p90delta0_expanded200_20260616_231706"),
]
roots.extend(Path("/media/w24/D/xjw深度学习训练数据/pfm_runs").glob("phase7a_fov76_complete_4l384_*/promotion_phase5g_profile_p90delta0_expanded200_*"))
roots.extend(Path("/media/w24/D/xjw深度学习训练数据/pfm_runs").glob("phase7b_fov76_graph8_h384_*/promotion_phase5g_profile_p90delta0_expanded200_*"))
for root in sorted(roots):
    decision = root / "promotion_decision.json"
    if not decision.exists():
        continue
    data = json.loads(decision.read_text(encoding="utf-8"))
    print(root)
    print("  decision:", data.get("decision", data.get("status")))
    print("  failed:", data.get("failed_reasons") or data.get("failure_reasons") or [])
PY
```

Expected:

```text
The best candidate is clear from promotion status and failed reasons.
```

## Task 6: Mine Hard Failures From The Best Rejected Or Borderline Candidate

**Files:**
- Read: candidate promotion `dual_checkpoint_rescue_selector/combined_filtered_summary.csv`
- Read: candidate visual `all_filtered_summary.csv`
- Create: train-only hard manifest under `PAIR_ROOT/hard_mining/phase7_failure_mining_YYYYMMDD_HHMMSS/`

Purpose:

```text
Only mine failures after a formal evaluation reveals which failure mode remains.
The hard set must be train-only and must not directly replay val/test rows.
```

- [ ] **Step 1: Locate all filtered summaries**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
find "/media/w24/D/xjw深度学习训练数据/pfm_runs" -path '*phase7*promotion*all_filtered_summary.csv' -print | sort
```

Expected:

```text
One or more candidate all_filtered_summary.csv files are printed.
```

- [ ] **Step 2: Select the newest train-diagnostic filtered summary**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
export PYTHONPATH=python:scripts
PYTHON=/home/w24/anaconda3/envs/cppTorch/bin/python
PAIR_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
OUT_ROOT="${PAIR_ROOT}/hard_mining/phase7_failure_mining_$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$(find "/media/w24/D/xjw深度学习训练数据/pfm_runs" \
  -path '*phase7*' \
  -name 'all_filtered_summary.csv' \
  -print | sort | tail -1)"
test -f "${SUMMARY_CSV}"
echo "${SUMMARY_CSV}"
mkdir -p "${OUT_ROOT}"
```

Expected:

```text
The command prints one all_filtered_summary.csv path.
```

- [ ] **Step 3: Mine low precision and low match failures**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
export PYTHONPATH=python:scripts
PYTHON=/home/w24/anaconda3/envs/cppTorch/bin/python
PAIR_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
OUT_ROOT="$(ls -dt "${PAIR_ROOT}"/hard_mining/phase7_failure_mining_* | head -1)"
SUMMARY_CSV="$(find "/media/w24/D/xjw深度学习训练数据/pfm_runs" \
  -path '*phase7*' \
  -name 'all_filtered_summary.csv' \
  -print | sort | tail -1)"
mkdir -p "${OUT_ROOT}"

"${PYTHON}" scripts/mine_hard_failure_pairs.py \
  --pair-manifest "${PAIR_ROOT}/overlap_edges_train.csv" \
  --summary-csv "${SUMMARY_CSV}" \
  --output-manifest "${OUT_ROOT}/phase7_hard_failures_train.csv" \
  --output-html "${OUT_ROOT}/phase7_hard_failures_train.html" \
  --mixed-output-manifest "${OUT_ROOT}/phase7_hardmix_train.csv" \
  --mixed-base-manifest "${PAIR_ROOT}/overlap_edges_train.csv" \
  --mixed-hard-fraction 0.25 \
  --residual-filtered \
  --only-extreme-variants \
  --extreme-variants extreme_02,extreme_03
```

Expected:

```text
phase7_hard_failures_train.csv exists
phase7_hardmix_train.csv exists
phase7_hard_failures_train.html exists
```

- [ ] **Step 4: Use mined hardmix only for the next run**

For the next matcher-only run, replace:

```text
--pair-spec-manifest "${PAIR_ROOT}/overlap_edges_train.csv"
```

with:

```text
--pair-spec-manifest "${OUT_ROOT}/phase7_hardmix_train.csv"
```

Do not change dustbin/no-match/rejection weights at the same time.

## Task 7: Decide Whether The Bottleneck Is Model Or Post-Filter

**Files:**
- Read: promotion `all_filtered_summary.csv`
- Read: promotion `match_details.csv` if generated
- Create: `runs/phase7_filter_and_geometry_diagnosis_20260616.html`

- [ ] **Step 1: Run a small filter sweep on the best checkpoint**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
export PYTHONPATH=python:scripts
PYTHON=/home/w24/anaconda3/envs/cppTorch/bin/python
DATA_ROOT="/media/w24/D/xjw深度学习训练数据"
PAIR_ROOT="${DATA_ROOT}/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
readarray -t BEST_INFO < <("${PYTHON}" - <<'PY'
from pathlib import Path
import json

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs")
patterns = [
    "phase7b_fov76_graph8_h384_*/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json",
    "phase7a_fov76_complete_4l384_*/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json",
    "phase6c_fov76_phase6a_match_count_floor_4l384_*/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json",
]
candidates = []
for pattern in patterns:
    for decision_path in root.glob(pattern):
        try:
            data = json.loads(decision_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        decision = str(data.get("decision", data.get("status", ""))).upper()
        run_root = decision_path.parents[1]
        train_output = run_root / "train_output"
        state = train_output / "checkpoints" / "best_by_match_score_pytorch_pfm_state.pt"
        if not state.exists():
            state = train_output / "checkpoints" / "latest_pytorch_pfm_state.pt"
        if state.exists():
            priority = 1 if decision == "PROMOTE" else 0
            candidates.append((priority, decision_path.stat().st_mtime, state, train_output, decision_path))
if not candidates:
    raise SystemExit("no phase6c/phase7 promotion candidate with a checkpoint was found")
candidates.sort(reverse=True)
_, _, state, train_output, decision_path = candidates[0]
print(state)
print(train_output)
print(decision_path)
PY
)
BEST_STATE="${BEST_INFO[0]}"
BEST_RUN="${BEST_INFO[1]}"
BEST_DECISION_JSON="${BEST_INFO[2]}"
OUT_DIR="${DATA_ROOT}/pfm_runs/phase7_filter_sweep_$(date +%Y%m%d_%H%M%S)"
echo "Best candidate decision: ${BEST_DECISION_JSON}"

"${PYTHON}" scripts/run_graph_filter_sweep.py \
  --render-manifest "${PAIR_ROOT}/manifests/h100km_fov076_render_manifest.csv" \
  --uint8-manifest "${PAIR_ROOT}/manifests/h100km_fov076_uint8_manifest.csv" \
  --pytorch-state "${BEST_STATE}" \
  --output-dir "${OUT_DIR}" \
  --run-dir "${BEST_RUN}" \
  --split val \
  --reference-variant nadir \
  --pair-spec-manifest "${PAIR_ROOT}/overlap_edges_val.csv" \
  --pair-mode same-position \
  --image-source uint8 \
  --candidate-pairs 200 \
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
  --geometry-threshold-px 5.0 \
  --geometry-threshold-px-values 5.0 \
  --filtered-geometry-filter magsac \
  --filtered-min-matches-values 12,16,20 \
  --graph-max-attention-layers 4 \
  --graph-max-attention-work-fraction 1.0 \
  --graph-width-prune-keep-ratio 1.0 \
  --max-configs 3 \
  --write-all-summary \
  --no-shuffle \
  --no-illumination-stress \
  --input-local-contrast \
  --input-local-contrast-strength 0.35 \
  --input-local-contrast-kernel 31
```

Expected:

```text
The sweep shows whether minmatch12, minmatch16, or minmatch20 is best without a wrong-match increase.
```

- [ ] **Step 2: Interpret the sweep**

Use this table:

```text
If lower minmatch improves correct count without wrong increase:
    The bottleneck is filter strictness; update profile only after promotion gate passes.

If lower minmatch increases wrong count:
    Keep minmatch16 and mine low-match hard failures.

If all filters have low matches:
    The bottleneck is model recall; continue matcher training or graph8.

If all filters have many wrong matches:
    The bottleneck is geometric consistency; increase RANSAC consistency or add false-match hard loss.
```

## Task 8: Only Then Consider Feature Extractor Changes

Do not start this task until:

```text
Task 3 complete 4-layer training has been promoted or rejected with evidence.
Task 4 graph8 has been promoted or rejected with evidence.
Task 7 filter sweep says model-side feature recall is still the bottleneck.
```

Allowed first extractor changes:

```text
1. keypoint score fusion:
   score = heatmap * (0.5 + 0.5 * quality)

2. stage1 skip for keypoint branch:
   heatmap branch sees shallow stage1 features in addition to stage2/stage3.
```

Disallowed in this round:

```text
full backbone unfreeze
new dense descriptor dimension increase
new dustbin/reliability shortcuts
new C4-style artificial view buckets
```

Before editing extractor code, write a separate plan because this touches model architecture and tests.

## Task 9: Verification And GitHub Update

**Files:**
- Read: `git status`
- Modify only if docs/scripts changed: tracked files under `docs/`, `scripts/`, `python/tests/`, or `python/`

- [ ] **Step 1: Run relevant tests after code/doc changes**

If only this plan file changed, run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git diff --check
```

If promotion/selector scripts changed, run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_stress_eval_scripts
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m compileall -q scripts/run_dual_checkpoint_rescue_eval.py scripts/run_fov76_checkpoint_promotion_pipeline.py
git diff --check
```

If training code changed, run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_pytorch_training python.tests.test_benchmark_lazy_pose_pairs
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m compileall -q python/pfm_pytorch_training.py scripts/benchmark_lazy_pose_pairs.py
git diff --check
```

- [ ] **Step 2: Check git status**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git status --short --branch
```

Expected:

```text
Only intended tracked files are modified.
The untracked file "0" may still appear and must not be added.
```

- [ ] **Step 3: Commit tracked plan/code changes**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git add docs/superpowers/plans/2026-06-16-fov76-next-complete-execution-plan.md
git commit -m "Document fov76 next complete execution plan"
```

Expected:

```text
Commit succeeds.
```

- [ ] **Step 4: Push to GitHub**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git push
```

Expected:

```text
main is pushed to origin/main.
```

## Execution Order

Use this exact order:

```text
1. Finish current expanded phase6c p90delta0 promotion.
2. If it promotes, validate and record the phase5g + phase6c selector as candidate active.
3. Run phase7a complete 4-layer/384 matcher-only training.
4. Promote phase7a with the same 200/200 p90delta0 gate.
5. Run phase7b 8-layer/384 only after phase7a has either finished or the GPU is idle.
6. Promote phase7b with the same 200/200 p90delta0 gate.
7. Pick the best passing candidate by promotion decision first, then target correct gain, then wrong-match stability, then match count.
8. Mine failures from the best rejected/borderline candidate only after the promotion result is known.
9. Run a filter/geometry diagnosis before changing extractor architecture.
10. Commit and push every durable code/doc change.
```

## Final Decision Table

```text
Best outcome:
    phase7a or phase7b passes expanded promotion with target gain and no guard regression.
    Action: update active config, validate active selector, keep extractor frozen.

Good outcome:
    phase6c p90delta0 passes expanded promotion but phase7a/phase7b do not.
    Action: keep phase6c selector as active candidate, mine phase7 failures for the next round.

Neutral outcome:
    no candidate passes expanded promotion, but one has clean target gains with isolated regressions.
    Action: do not promote; mine train-only analogs and run a hardmix matcher-only round.

Bad outcome:
    graph8 or longer training increases wrong matches or collapses match count.
    Action: reject larger model for now; keep 4-layer path and focus on hard failure mining plus geometry consistency.

Extractor-change outcome:
    all matcher/filter experiments show persistent low recall with stable geometry.
    Action: write a separate extractor plan for heatmap-quality soft fusion and stage1 keypoint skip.
```
