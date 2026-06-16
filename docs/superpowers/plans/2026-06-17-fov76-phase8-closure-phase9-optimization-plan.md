# fov76 Phase8 Closure And Phase9 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the running fov76 phase8 hard-replay training, evaluate it against the current active branch with strict gates, promote only if it beats the active branch, then start the next controlled matcher/model optimization phase.

**Architecture:** Keep the fov76 branch isolated from fov90 and treat every checkpoint as a candidate until promotion evidence proves it is better than the current active selector. Continue optimizing the matcher, hard-failure sampling, and geometric post-filter first; only partially unfreeze extractor-side modules after matcher/filter experiments stop improving. Every training run must produce checkpoints, visual reports, promotion decisions, and a human-readable HTML summary.

**Tech Stack:** Python 3, PyTorch, CUDA AMP, lazy fov76 pair manifests, `benchmark_lazy_pose_pairs.py`, `visualize_lazy_pose_matches.py`, `run_graph_filter_sweep.py`, `mine_hard_failure_pairs.py`, `run_fov76_checkpoint_promotion_pipeline.py`, `validate_fov76_active_selector.py`, HTML/CSV/JSON run records, Git/GitHub.

---

## Current Ground Truth

Use these paths unless a later command prints a newer accepted active config.

```text
project root:
/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

python:
/home/w24/anaconda3/envs/cppTorch/bin/python

dataset root:
/media/w24/D/xjw深度学习训练数据

fov76 pair root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal

phase8 hard manifest root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures

current active config:
runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json

current active validation:
runs/fov76_active_mainline_validation_phase6c_p90delta0_20260617_after_strict_graph8.json

current active checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

stable phase5g baseline checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

phase8 running root:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902

phase8 training log:
runs/train_h100_fov076_phase8_hard_replay_20260617.log

phase8 promotion script:
runs/eval_h100_fov076_phase8_promotion_20260617.sh
```

Facts already established:

```text
phase7b strict graph8 promotion passed versus phase5g but did not beat phase6c.
phase6c remains the active mainline.
The selected phase8 hard-mining filter is score=-1, local geometry threshold=5, MAGSAC, filtered min matches=16.
The phase8 hard set is train-only and fov76-only.
The current phase8 run is matcher/calibration-focused with the extractor frozen.
```

## Hard Rules

```text
Do not stop the running phase8 training unless it hits non-recoverable NaN/inf failure.
Do not switch back to fov90 for this optimization branch.
Do not promote a checkpoint just because it beats phase5g; it must beat or at least justify replacing phase6c.
Do not increase no-match/dustbin/rejection pressure to fix low match count.
Do not fully unfreeze the backbone in the next phase.
Do not touch the untracked file named 0.
Keep raw stdout/stderr logs in runs/.
Write human-facing experiment decisions as HTML.
Push tracked code and docs changes to GitHub after verification.
```

## File Map

Monitoring and phase8 closure:

```text
runs/train_h100_fov076_phase8_hard_replay_20260617.log
    Current phase8 training log.

runs/train_h100_fov076_phase8_hard_replay_20260617.pid
    PID record for the current phase8 launcher.

/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902/train_output/train_metrics.csv
    Training and automatic visual metrics.

/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902/train_output/checkpoints/
    Phase8 checkpoint outputs.
```

Promotion and active selection:

```text
runs/eval_h100_fov076_phase8_promotion_20260617.sh
    Runs formal val/test, guard, and promotion decision for phase8.

runs/eval_h100_fov076_phase8_promotion_20260617.log
    Raw promotion stdout/stderr.

runs/phase8_promotion_decision_20260617.html
    Human-readable phase8 promotion summary to create after promotion finishes.

runs/fov76_active_mainline_config_phase8_hard_replay_p90delta0_20260617.json
    Create only if phase8 beats the current active branch and passes validation.

runs/fov76_active_mainline_validation_phase8_hard_replay_p90delta0_20260617.json
    Create only after phase8 active config validation passes.
```

Potential phase9 code files:

```text
python/pfm_model.py
    PlanetaryFeatureMatcher, SparseHead, TextureDescriptorAdapter, PlanetaryGraphMatcher, graph metadata, quality scoring.

python/pfm_model_descriptors.py
    Geometry-aware descriptor pooling and descriptor normalization.

python/pfm_pytorch_training.py
    GraphMatcher losses, teacher guard, false-match losses, RANSAC consistency, training diagnostics.

scripts/benchmark_lazy_pose_pairs.py
    Main training CLI and automatic visual-eval passthrough.

scripts/visualize_lazy_pose_matches.py
    Main visual evaluation and filtered-output metrics.

scripts/run_graph_filter_sweep.py
    Post-filter sweep runner.

scripts/run_fov76_checkpoint_promotion_pipeline.py
    Promotion gate orchestration.

python/tests/test_pfm_model.py
python/tests/test_pfm_pytorch_training.py
python/tests/test_benchmark_lazy_pose_pairs.py
python/tests/test_stress_eval_scripts.py
    Required focused tests for any phase9 code change.

scripts/README.md
    Update whenever a persistent script option or recommended mainline behavior changes.
```

## Acceptance Criteria

Phase8 closure is accepted only when all items below are true:

```text
The phase8 training process exits normally.
best_by_match_score_pytorch_pfm_state.pt exists for phase8.
train_metrics.csv has visual rows through the final planned visual interval.
phase8 promotion_decision.json exists.
phase8 promotion_decision.html exists.
The final decision explicitly says either:
    keep phase6c active, with reason;
    or promote phase8, with validation JSON proving active selector validity.
No active config is changed without validate_fov76_active_selector.py passing.
Git status is reviewed before final response.
```

Phase9 starts only after phase8 has a written decision.

## Task 1: Monitor The Running Phase8 Training

**Files:**
- Read: `runs/train_h100_fov076_phase8_hard_replay_20260617.log`
- Read: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902/train_output/train_metrics.csv`

- [ ] **Step 1: Confirm active long processes**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Expected:

```text
One phase8 benchmark_lazy_pose_pairs.py train process is present, plus worker or visual-eval children while visual reports are running.
No unrelated fov90 training or old promotion process is running.
```

- [ ] **Step 2: Read the latest training and visual progress**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
python - <<'PY'
from pathlib import Path

log = Path("runs/train_h100_fov076_phase8_hard_replay_20260617.log")
text = log.read_text(errors="ignore") if log.exists() else ""
last_train = "no train step found"
reports = []
for line in text.splitlines():
    if line.startswith("train step="):
        last_train = line
    if line.startswith("report="):
        reports.append(line)
print(last_train)
print("recent reports:")
for line in reports[-5:]:
    print(line)
PY
```

Expected:

```text
The latest train step increases over time.
report= lines appear every 200 steps until the planned 1600-step end.
```

- [ ] **Step 3: Check GPU and dataloader symptoms without changing the run**

Run:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

Expected:

```text
Memory stays below the device limit.
GPU utilization can dip during automatic visual eval; do not restart for visual-eval dips.
If data_wait in the train log stays near 0 ms after warmup, the dataloader is not the current bottleneck.
```

## Task 2: Close Phase8 Training

**Files:**
- Read: phase8 `train_output/checkpoints/`
- Read: phase8 `train_output/train_metrics.csv`
- Create: `runs/phase8_training_closeout_20260617.html`

- [ ] **Step 1: Verify training completion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py' || true
tail -80 runs/train_h100_fov076_phase8_hard_replay_20260617.log
```

Expected:

```text
No phase8 benchmark_lazy_pose_pairs.py process remains.
The log contains train step=1600/1600 or a normal final checkpoint line.
If a visual_eval child is still running after step 1600, wait for it to finish before promotion.
```

- [ ] **Step 2: Verify required phase8 checkpoint files**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PHASE8_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902"
ls -lh "${PHASE8_ROOT}/train_output/checkpoints/" | tail -20
test -f "${PHASE8_ROOT}/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
test -f "${PHASE8_ROOT}/train_output/train_metrics.csv"
```

Expected:

```text
best_by_match_score_pytorch_pfm_state.pt exists.
train_metrics.csv exists.
The command exits with status 0.
```

- [ ] **Step 3: Summarize final training metrics**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PHASE8_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902"
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import csv

metrics = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902/train_output/train_metrics.csv")
rows = list(csv.DictReader(metrics.open()))
if not rows:
    raise SystemExit("train_metrics.csv is empty")
last = rows[-1]
visual_rows = [r for r in rows if r.get("visual_filtered_precision")]
print("last_step", last.get("step"))
print("last_loss", last.get("loss"))
print("last_top1", last.get("top1"))
print("visual_rows", len(visual_rows))
if visual_rows:
    best = max(visual_rows, key=lambda r: float(r.get("visual_filtered_correct", "0") or 0.0))
    print("best_visual_step", best.get("step"))
    print("best_visual_correct", best.get("visual_filtered_correct"))
    print("best_visual_wrong", best.get("visual_filtered_wrong"))
    print("best_visual_precision", best.get("visual_filtered_precision"))
PY
```

Expected:

```text
The final step is 1600.
Visual rows are present.
The printed best visual row is used only for diagnosis, not active promotion.
```

- [ ] **Step 4: Write phase8 training closeout HTML**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PHASE8_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902"
python - <<'PY'
from pathlib import Path
import csv, html

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902")
metrics = root / "train_output" / "train_metrics.csv"
rows = list(csv.DictReader(metrics.open())) if metrics.exists() else []
last = rows[-1] if rows else {}
visual_rows = [r for r in rows if r.get("visual_filtered_precision")]
best = max(visual_rows, key=lambda r: float(r.get("visual_filtered_correct", "0") or 0.0)) if visual_rows else {}
out = Path("runs/phase8_training_closeout_20260617.html")
out.write_text(f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Phase8 Training Closeout</title></head>
<body>
<h1>Phase8 Training Closeout</h1>
<p><b>Run root:</b> {html.escape(str(root))}</p>
<p><b>Last step:</b> {html.escape(str(last.get("step", "")))}</p>
<p><b>Last loss:</b> {html.escape(str(last.get("loss", "")))}</p>
<p><b>Last top1:</b> {html.escape(str(last.get("top1", "")))}</p>
<p><b>Visual rows:</b> {len(visual_rows)}</p>
<p><b>Best visual step:</b> {html.escape(str(best.get("step", "")))}</p>
<p><b>Best visual correct/wrong/precision:</b>
{html.escape(str(best.get("visual_filtered_correct", "")))} /
{html.escape(str(best.get("visual_filtered_wrong", "")))} /
{html.escape(str(best.get("visual_filtered_precision", "")))}</p>
<p><b>Decision:</b> promotion not run yet.</p>
</body></html>
""")
print(out)
PY
```

Expected:

```text
runs/phase8_training_closeout_20260617.html is created.
```

## Task 3: Run Phase8 Promotion

**Files:**
- Use: `runs/eval_h100_fov076_phase8_promotion_20260617.sh`
- Create: `runs/eval_h100_fov076_phase8_promotion_20260617.log`
- Read: phase8 promotion `promotion_decision.json`

- [ ] **Step 1: Launch promotion only after training and visual eval have stopped**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py|run_fov76_checkpoint_promotion_pipeline.py' || true
PHASE8_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902"
test -f "${PHASE8_ROOT}/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
setsid runs/eval_h100_fov076_phase8_promotion_20260617.sh "$PHASE8_ROOT" \
  > runs/eval_h100_fov076_phase8_promotion_20260617.log 2>&1 &
echo $! > runs/eval_h100_fov076_phase8_promotion_20260617.pid
cat runs/eval_h100_fov076_phase8_promotion_20260617.pid
```

Expected:

```text
A new PID is printed.
run_fov76_checkpoint_promotion_pipeline.py starts under that launcher.
```

- [ ] **Step 2: Monitor promotion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -80 runs/eval_h100_fov076_phase8_promotion_20260617.log
pgrep -af 'run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Expected:

```text
The log advances through formal val/test, regression_guard, and extreme_gain sections.
No traceback appears.
```

- [ ] **Step 3: Locate promotion decision**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
python - <<'PY'
from pathlib import Path

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902")
paths = sorted(root.glob("promotion_phase5g_phase8_profile_p90delta0_expanded200_*/promotion_decision.json"))
if not paths:
    raise SystemExit("phase8 promotion_decision.json not found")
print(paths[-1])
PY
```

Expected:

```text
A concrete promotion_decision.json path is printed.
```

## Task 4: Decide Whether Phase8 Replaces Phase6c

**Files:**
- Read: latest phase8 `promotion_decision.json`
- Read: `runs/fov76_active_mainline_validation_phase6c_p90delta0_20260617_after_strict_graph8.json`
- Create: `runs/phase8_promotion_decision_20260617.html`

- [ ] **Step 1: Parse promotion and active evidence**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
python - <<'PY'
from pathlib import Path
import json

phase8_root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902")
decision_path = sorted(phase8_root.glob("promotion_phase5g_phase8_profile_p90delta0_expanded200_*/promotion_decision.json"))[-1]
decision = json.loads(decision_path.read_text())
active_path = Path("runs/fov76_active_mainline_validation_phase6c_p90delta0_20260617_after_strict_graph8.json")
active = json.loads(active_path.read_text())
print("phase8_decision", decision_path)
print("phase8_promote", decision.get("promote"))
print("phase8_failed_reasons", decision.get("failed_reasons"))
print("phase8_formal_target_total", decision.get("formal_target_total"))
print("active_valid", active.get("valid"))
print("active_score", active.get("active_score"))
PY
```

Expected:

```text
The output shows phase8 promote status and the current active phase6c score.
```

- [ ] **Step 2: Apply replacement rule**

Use this rule exactly:

```text
If phase8 promote=false:
    keep phase6c active.

If phase8 promote=true but formal target total is weaker than phase6c active_score:
    keep phase6c active and record phase8 as useful diagnostic/replay checkpoint.

If phase8 promote=true and phase8 is at least as strong as phase6c on correct_delta, wrong_delta, precision_delta, and guard cleanliness:
    create phase8 active config and validate it.

If metrics are mixed:
    keep phase6c active unless phase8 gives a clear target extreme gain with no protected or guard regression.
```

- [ ] **Step 3: Write phase8 promotion summary HTML**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
python - <<'PY'
from pathlib import Path
import json, html

phase8_root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902")
decision_path = sorted(phase8_root.glob("promotion_phase5g_phase8_profile_p90delta0_expanded200_*/promotion_decision.json"))[-1]
decision = json.loads(decision_path.read_text())
active_path = Path("runs/fov76_active_mainline_validation_phase6c_p90delta0_20260617_after_strict_graph8.json")
active = json.loads(active_path.read_text())
phase8_score = decision.get("formal_target_total", {})
active_score = active.get("active_score", {})
promote = bool(decision.get("promote"))
phase8_correct = float(phase8_score.get("correct_delta", 0) or 0)
phase8_wrong = float(phase8_score.get("wrong_delta", 0) or 0)
active_correct = float(active_score.get("correct_delta", 0) or 0)
active_wrong = float(active_score.get("wrong_delta", 0) or 0)
replace = promote and phase8_correct >= active_correct and phase8_wrong <= active_wrong
out = Path("runs/phase8_promotion_decision_20260617.html")
out.write_text(f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Phase8 Promotion Decision</title></head>
<body>
<h1>Phase8 Promotion Decision</h1>
<p><b>Decision JSON:</b> {html.escape(str(decision_path))}</p>
<p><b>Phase8 promote versus phase5g:</b> {promote}</p>
<p><b>Phase8 failed reasons:</b> {html.escape(str(decision.get("failed_reasons")))}</p>
<p><b>Phase8 formal target total:</b> {html.escape(str(phase8_score))}</p>
<p><b>Current active phase6c score:</b> {html.escape(str(active_score))}</p>
<p><b>Replace active:</b> {replace}</p>
<p><b>Operational decision:</b> {'promote phase8 after validation' if replace else 'keep phase6c active'}</p>
</body></html>
""")
print(out)
PY
```

Expected:

```text
runs/phase8_promotion_decision_20260617.html is created and states whether phase8 replaces phase6c.
```

## Task 5: Create Active Config Only If Phase8 Wins

**Files:**
- Read: `runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json`
- Create only on promotion: `runs/fov76_active_mainline_config_phase8_hard_replay_p90delta0_20260617.json`
- Create only on promotion: `runs/fov76_active_mainline_validation_phase8_hard_replay_p90delta0_20260617.json`

- [ ] **Step 1: Create phase8 active config when Task 4 says replace active**

Run only when `runs/phase8_promotion_decision_20260617.html` says `Replace active: True`.

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
python - <<'PY'
from pathlib import Path
import json

phase8_root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_20260617_023902")
decision_path = sorted(phase8_root.glob("promotion_phase5g_phase8_profile_p90delta0_expanded200_*/promotion_decision.json"))[-1]
src = Path("runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json")
data = json.loads(src.read_text())
data["active_label"] = "phase8_hard_replay_p90delta0"
data["active_selector"] = "phase8_hard_replay_selector_p90delta0"
data["active_checkpoint"] = str(phase8_root / "train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt")
data["promotion_decision_json"] = str(decision_path)
data["notes"] = "Phase8 hard-replay checkpoint promoted after passing formal, guard, and active-comparison gates."
out = Path("runs/fov76_active_mainline_config_phase8_hard_replay_p90delta0_20260617.json")
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(out)
PY
```

Expected:

```text
The phase8 active config JSON is created.
```

- [ ] **Step 2: Validate phase8 active selector**

Run only if the phase8 active config exists.

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/validate_fov76_active_selector.py \
  --config runs/fov76_active_mainline_config_phase8_hard_replay_p90delta0_20260617.json \
  --output-json runs/fov76_active_mainline_validation_phase8_hard_replay_p90delta0_20260617.json
```

Expected:

```text
The validation JSON contains "valid": true.
If validation is false, delete no files, keep phase6c active, and record the failure in the final summary.
```

- [ ] **Step 3: Keep phase6c active when phase8 does not win**

Run when Task 4 says `Replace active: False`.

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/validate_fov76_active_selector.py \
  --config runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json \
  --output-json runs/fov76_active_mainline_validation_phase6c_p90delta0_20260617_after_phase8.json
```

Expected:

```text
The validation JSON contains "valid": true.
```

## Task 6: Phase9 Experiment Selection

**Files:**
- Create: `runs/phase9_experiment_selection_20260617.html`
- No code changes in this task.

- [ ] **Step 1: Classify phase8 result**

Use the decision from Task 4 and the training closeout from Task 2.

```text
Case A: phase8 beats phase6c
    Start phase9 from phase8.

Case B: phase8 promotes versus phase5g but does not beat phase6c
    Keep phase6c active.
    Use phase8 hard set and metrics to design phase9.

Case C: phase8 fails promotion
    Keep phase6c active.
    Reduce the phase8 training idea to an ablation result and mine the remaining failed pairs.
```

- [ ] **Step 2: Choose exactly one phase9 training direction**

Pick one direction for the next run:

```text
Direction 1: stronger false-edge suppression
    Use current active checkpoint.
    Keep graph layers=4, hidden dim=384, candidate_topk=256.
    Increase final false-match or RANSAC consistency loss slightly.
    Keep dustbin/no-match weights at 0 or near 0.

Direction 2: deeper graph with depth consistency
    Use current active checkpoint or the best graph8 checkpoint.
    Train graph layers=8 with train max attention layers=8.
    Add random depth or per-depth supervision if already available.
    Promotion must compare graph4 active against graph8 candidate with per-side graph depth.

Direction 3: partial extractor-side unfreeze
    Use current active checkpoint.
    Unfreeze descriptor fusion, quality head, and geometry head only.
    Keep backbone frozen.
    Use lower training resolution if memory exceeds budget.
```

Default choice:

```text
Choose Direction 1 unless phase8 clearly proves the current hard set is no longer the bottleneck.
```

- [ ] **Step 3: Write the selection HTML**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
python - <<'PY'
from pathlib import Path
import html

phase8_decision = Path("runs/phase8_promotion_decision_20260617.html")
phase8_text = phase8_decision.read_text(errors="ignore") if phase8_decision.exists() else "phase8 decision missing"
direction = "Direction 1: stronger false-edge suppression, keep dustbin/no-match disabled"
out = Path("runs/phase9_experiment_selection_20260617.html")
out.write_text(f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Phase9 Experiment Selection</title></head>
<body>
<h1>Phase9 Experiment Selection</h1>
<p><b>Selected direction:</b> {html.escape(direction)}</p>
<p><b>Reason:</b> Continue reducing wrong matches on hard/extreme/repeated-texture pairs without making the matcher conservative.</p>
<h2>Phase8 Decision Snapshot</h2>
<pre>{html.escape(phase8_text[:8000])}</pre>
</body></html>
""")
print(out)
PY
```

Expected:

```text
runs/phase9_experiment_selection_20260617.html is created.
```

## Task 7: Phase9A Hard False-Edge Suppression Run

**Files:**
- Create: `runs/train_h100_fov076_phase9a_false_edge_20260617.sh`
- Create: `runs/train_h100_fov076_phase9a_false_edge_20260617.log`
- Read: phase9a `train_output/train_metrics.csv`

- [ ] **Step 1: Write the phase9a run script**

Use the same inputs as phase8 but start from the selected active checkpoint. The only intended training change is stronger false-edge suppression without stronger dustbin/no-match.

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
cat > /tmp/phase9a_expected_command.txt <<'TXT'
The run script must keep:
  --graph-hidden-dim 384
  --graph-attention-layers 4
  --graph-matcher-train-max-attention-layers 4
  --matcher-candidate-topk 256
  --graph-matcher-no-match-weight 0.0
  --graph-matcher-hard-negative-dustbin-weight 0.0
  --matcher-reliability-pair-bias off
  --matcher-reliability-dustbin-bias off
  --train-graph-calibration-only unless Task 6 explicitly selected partial unfreeze

The run script must increase one false-edge term:
  --graph-matcher-final-false-match-weight
or:
  --graph-matcher-ransac-consistency-weight

The run script must use:
  /media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures/hard_mixed_train.csv
TXT
cat /tmp/phase9a_expected_command.txt
```

Expected:

```text
The command spec is printed and used to write the run script.
```

- [ ] **Step 2: Syntax-check the phase9a script before launch**

Run after writing the script:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
bash -n runs/train_h100_fov076_phase9a_false_edge_20260617.sh
```

Expected:

```text
No output and exit status 0.
```

- [ ] **Step 3: Launch phase9a only if no other training is active**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|visualize_lazy_pose_matches.py' || true
setsid runs/train_h100_fov076_phase9a_false_edge_20260617.sh \
  > runs/train_h100_fov076_phase9a_false_edge_20260617.log 2>&1 &
echo $! > runs/train_h100_fov076_phase9a_false_edge_20260617.pid
cat runs/train_h100_fov076_phase9a_false_edge_20260617.pid
```

Expected:

```text
A new PID is printed.
The log begins with the planned fov76 pair manifest and active checkpoint.
```

## Task 8: Phase9B Graph8 Only If Phase9A Does Not Improve

**Files:**
- Create: `runs/train_h100_fov076_phase9b_graph8_depth_consistent_20260617.sh`
- Use: `scripts/run_fov76_checkpoint_promotion_pipeline.py`

- [ ] **Step 1: Start graph8 only after phase9a promotion decision**

Condition:

```text
Run phase9b only when phase9a fails to beat phase6c/phase8 and the error analysis still shows low recall or unresolved hard extreme pairs.
```

- [ ] **Step 2: Graph8 constraints**

Use these constraints:

```text
graph_attention_layers = 8
graph_hidden_dim = 384
matcher_candidate_topk = 256
train max attention layers = 8
inference candidate graph layers = 8
baseline graph layers = 4
no full extractor unfreeze
no dustbin/no-match weight increase
```

- [ ] **Step 3: Strict graph8 promotion rule**

Run graph8 promotion through `run_fov76_checkpoint_promotion_pipeline.py` with:

```text
--baseline-graph-layers 4
--candidate-graph-layers 8
--formal-candidate-pairs 200
--guard-candidate-pairs 200
--post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard
```

Expected:

```text
Graph8 is accepted only if it beats the active branch under strict graph8 inference.
```

## Task 9: Phase9C Partial Extractor Unfreeze Only After Matcher Plateau

**Files:**
- Modify if selected: `python/pfm_model.py`
- Modify if selected: `python/pfm_pytorch_training.py`
- Modify if selected: `scripts/benchmark_lazy_pose_pairs.py`
- Test if selected: `python/tests/test_pfm_model.py`
- Test if selected: `python/tests/test_pfm_pytorch_training.py`
- Test if selected: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Prove matcher plateau before extractor work**

Extractor work is allowed only when:

```text
phase8 and phase9a fail to improve active.
remaining failures show descriptor/geometry weakness rather than post-filter threshold issues.
GPU memory estimate leaves room for descriptor fusion, quality head, or geometry head training.
```

- [ ] **Step 2: First extractor-side target**

Use this order:

```text
1. descriptor fusion and texture adapter
2. quality head
3. orientation/scale/affine geometry head
4. stage1 detector skip or backbone changes
```

Do not start with full backbone unfreeze.

- [ ] **Step 3: Required tests for extractor code changes**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_pfm_model \
  python.tests.test_pfm_pytorch_training \
  python.tests.test_benchmark_lazy_pose_pairs
```

Expected:

```text
All tests pass before any extractor-side run is launched.
```

## Task 10: Final Verification And GitHub Update

**Files:**
- Read: `git status --short`
- Modify if needed: `scripts/README.md`
- Commit tracked plan/code/doc changes only.

- [ ] **Step 1: Run focused verification**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_per_side_graph_layers \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_plans_selector_as_promotion_candidate
```

Expected:

```text
OK
```

- [ ] **Step 2: Verify active selector**

Run the validation command for the active config selected by Task 5.

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/validate_fov76_active_selector.py \
  --config runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json \
  --output-json runs/fov76_active_mainline_validation_final_20260617.json
```

Expected:

```text
The validation JSON contains "valid": true for the selected active branch.
If phase8 becomes active, replace the config path with runs/fov76_active_mainline_config_phase8_hard_replay_p90delta0_20260617.json.
```

- [ ] **Step 3: Check git status**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git status --short
```

Expected:

```text
Only intended tracked changes are present.
The untracked file 0 remains untouched.
Ignored runs/ logs do not need to be committed.
```

- [ ] **Step 4: Commit and push tracked changes**

Run when there are tracked docs/code changes:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git add docs/superpowers/plans/2026-06-17-fov76-phase8-closure-phase9-optimization-plan.md
git commit -m "docs: plan fov76 phase8 closure and phase9 optimization"
git push
```

Expected:

```text
The commit is pushed to GitHub.
```

## Stop Conditions

Stop and report instead of continuing when any condition below happens:

```text
phase8 training exits before producing best_by_match_score_pytorch_pfm_state.pt.
promotion pipeline raises a traceback.
validate_fov76_active_selector.py reports valid=false for a config intended to become active.
GPU OOM repeats after lowering no code path.
train_metrics.csv is missing visual metrics after automatic visual eval was requested.
new untracked or modified files appear in tracked source paths and cannot be attributed to this plan.
```

## Expected Outcome

At the end of this plan:

```text
There is one explicit active fov76 mainline: phase6c or validated phase8.
The phase8 hard-replay run is either promoted or rejected with evidence.
The next phase9 direction is chosen from actual phase8 evidence.
No fov90 data enters the branch.
No full extractor unfreeze happens prematurely.
All durable code/docs changes are pushed to GitHub.
```
