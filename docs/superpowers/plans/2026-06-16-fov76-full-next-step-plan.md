# fov76 Full Next Step Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the current fov76 training/evaluation chain, select the best safe matcher path, then run the next optimization round without mixing data, filter, matcher, and extractor changes.

**Architecture:** Keep this branch scoped to the h100/fov76/dom76 internal data and treat formal promotion as the only authority for model activation. The execution order is: finish current 4-layer run, promote or reject it, train/evaluate 8-layer only with matched 8-layer supervision, compare candidates under the same gates, then mine hard failures and run the next controlled training pass. Feature extractor changes stay behind the matcher/post-filter evidence gate.

**Tech Stack:** Python 3, PyTorch, CUDA AMP, existing lazy fov76 pair manifests, `benchmark_lazy_pose_pairs.py`, `run_fov76_checkpoint_promotion_pipeline.py`, `visualize_lazy_pose_matches.py`, `run_graph_filter_sweep.py`, `mine_hard_failure_pairs.py`, HTML/CSV run records, Git.

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

Current accepted baseline checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Current accepted selector/checkpoint path:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Current active selector config:
runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json

Current post-filter profile:
fov76_geo5_geo10_extreme_rescue_lowmatch_guard
```

Current running chain:

```text
phase7a 4-layer/384 matcher-only complete run:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507

phase7a training log:
runs/train_h100_fov076_phase7a_complete_4l384_20260616_233507.log

phase7a promotion launcher:
runs/eval_h100_fov076_phase7a_promotion_20260616.sh

phase7a -> promotion watcher:
runs/watch_phase7a_then_promotion_20260616.sh

phase7a promotion -> graph8 watcher:
runs/watch_phase7a_promotion_then_graph8_20260616.sh

graph8 launcher:
runs/train_h100_fov076_phase7b_graph8_20260616.sh
```

Important constraints:

```text
Do not interrupt phase7a unless it emits NaN/OOM/error markers or exits non-zero.
Do not use fov90 in this branch.
Do not promote a model or selector before expanded formal promotion passes.
Do not increase dustbin/no-match/rejection weights as the first reaction to low match count.
Do not run 8-layer inference for a 4-layer-trained checkpoint.
Do not unfreeze the full extractor until matcher/filter/geometry diagnostics are exhausted.
Do not touch the untracked file named "0".
```

## Files And Responsibilities

Use existing implementation files:

```text
scripts/benchmark_lazy_pose_pairs.py
```

Main train/eval entry for lazy fov76 pairs. It owns matcher-only training, AMP, worker prefetch, checkpoint writing, visual eval, and training metrics.

```text
scripts/run_fov76_checkpoint_promotion_pipeline.py
```

Formal promotion gate. Every candidate checkpoint or selector must pass this before activation.

```text
scripts/visualize_lazy_pose_matches.py
scripts/run_graph_filter_sweep.py
```

Visual and post-filter diagnostics. These distinguish model failures from threshold/filter failures.

```text
scripts/mine_hard_failure_pairs.py
scripts/mine_selector_disagreement_pairs.py
scripts/build_train_replay_from_pair_deltas.py
```

Hard failure mining and replay set construction. Only train split rows may feed training.

```text
python/pfm_pytorch_training.py
python/pfm_training_stability.py
```

Training loss wiring and stability protection. Modify these only after a metric-backed diagnosis shows the current losses or guards are insufficient.

Create or update local run artifacts:

```text
runs/phase7_current_status_20260616.html
runs/eval_h100_fov076_phase7b_promotion_20260616.sh
runs/phase7_candidate_comparison_20260616.html
runs/phase8_hard_failure_manifest_summary_20260616.html
runs/train_h100_fov076_phase8_hard_replay_20260616.sh
runs/phase8_optimization_decision_20260616.html
```

Tracked documentation:

```text
docs/superpowers/plans/2026-06-16-fov76-full-next-step-plan.md
scripts/README.md
```

Update `scripts/README.md` only if a durable script or durable CLI flag is added. Pure `runs/*.sh`, `runs/*.log`, and `runs/*.html` artifacts do not require README changes.

## Success Metrics

Promotion must evaluate the same candidate families:

```text
formal target variants:
extreme_02/extreme_03 val/test

protected variants:
mid_01/mid_02/extreme_01 val/test

regression guard:
phase3d_diff_guard val/test

extreme gain guard:
extreme-focused train/test guard reports
```

Primary metrics:

```text
filtered_correct
filtered_wrong
filtered_precision
filtered_matches
zero-match row count
low-match row count
homography_residual_p90_px
target extreme_02/extreme_03 correct_delta
target extreme_02/extreme_03 wrong_delta
target extreme_02/extreme_03 precision_delta
protected variant wrong_delta
regression_guard precision_delta
```

Promotion rule:

```text
PROMOTE:
    target correct_delta improves or stays useful,
    target wrong_delta does not increase,
    protected variants do not regress,
    regression guard does not regress,
    homography p90 selector guard is respected.

KEEP ACTIVE:
    candidate is neutral or useful on a subset but fails one formal gate.

MINE:
    candidate has real gains but isolated wrong/low-match rows.

REJECT:
    candidate increases wrong clusters, reduces protected correctness, or wins only by weakening geometry.
```

## Task 1: Monitor Current Phase7a To A Clean Exit

**Files:**
- Read: `runs/train_h100_fov076_phase7a_complete_4l384_20260616_233507.log`
- Read: `runs/watch_phase7a_then_promotion_20260616.log`
- Create: `runs/phase7_current_status_20260616.html`

- [ ] **Step 1: Check active processes**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py|train_h100_fov076_phase7' || true
```

Expected:

```text
phase7a training or its final visual eval may be active.
No fov90 process should be active.
No unrelated training should be active.
```

- [ ] **Step 2: Tail phase7a training log**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 runs/train_h100_fov076_phase7a_complete_4l384_20260616_233507.log
```

Expected:

```text
The run reaches step 1200/1200.
The final visual report writes without traceback.
There are no NaN, OOM, nonfinite loss, or bad pair abort markers.
```

- [ ] **Step 3: Verify phase7a checkpoint artifacts**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
test -f "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
test -f "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507/train_output/train_metrics.csv"
test -f "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507/train_output/visual_report/index.html"
```

Expected:

```text
All three commands exit 0.
```

- [ ] **Step 4: Write a phase7a status HTML**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
from html import escape

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507")
log = Path("runs/train_h100_fov076_phase7a_complete_4l384_20260616_233507.log")
tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
out = Path("runs/phase7_current_status_20260616.html")
out.write_text(
    "<!doctype html><meta charset='utf-8'><title>phase7 current status</title>"
    "<h1>phase7a current status</h1>"
    f"<p><b>run root:</b> {escape(str(root))}</p>"
    f"<p><b>best checkpoint:</b> {escape(str(root / 'train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt'))}</p>"
    f"<p><b>visual report:</b> {escape(str(root / 'train_output/visual_report/index.html'))}</p>"
    "<h2>last log lines</h2><pre>"
    + escape(tail)
    + "</pre>",
    encoding="utf-8",
)
print(out)
PY
```

Expected:

```text
runs/phase7_current_status_20260616.html is created and links the current run root, checkpoint, and visual report.
```

## Task 2: Run And Parse Phase7a Promotion

**Files:**
- Run: `runs/eval_h100_fov076_phase7a_promotion_20260616.sh`
- Read: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json`
- Create: `runs/phase7a_promotion_decision_20260616.html`

- [ ] **Step 1: Confirm watcher or launch promotion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -100 runs/watch_phase7a_then_promotion_20260616.log
pgrep -af 'run_fov76_checkpoint_promotion_pipeline.py|eval_h100_fov076_phase7a_promotion' || true
```

Expected:

```text
Either the watcher has started phase7a promotion, or no promotion process exists after phase7a exited cleanly.
```

If no promotion process exists and no promotion decision exists, run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
setsid runs/eval_h100_fov076_phase7a_promotion_20260616.sh > runs/eval_h100_fov076_phase7a_promotion_manual_20260616.launch.log 2>&1 &
```

Expected:

```text
A run_fov76_checkpoint_promotion_pipeline.py process starts.
```

- [ ] **Step 2: Wait for promotion decision**

Run until a decision path prints:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
find "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507" \
  -path '*promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json' \
  -print | sort | tail -1
```

Expected:

```text
One promotion_decision.json path is printed.
```

- [ ] **Step 3: Parse decision**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507")
paths = sorted(root.glob("promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json"))
if not paths:
    raise SystemExit("phase7a promotion_decision.json not found")
data = json.loads(paths[-1].read_text(encoding="utf-8"))
print(paths[-1])
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
```

Expected:

```text
The JSON clearly states promote true/false and lists passed_reasons and failed_reasons.
```

- [ ] **Step 4: Write phase7a decision HTML**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
from html import escape
import json

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507")
decision_path = sorted(root.glob("promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json"))[-1]
data = json.loads(decision_path.read_text(encoding="utf-8"))
passed = data.get("passed_reasons", [])
failed = data.get("failed_reasons", [])
out = Path("runs/phase7a_promotion_decision_20260616.html")
out.write_text(
    "<!doctype html><meta charset='utf-8'><title>phase7a promotion</title>"
    "<h1>phase7a promotion decision</h1>"
    f"<p><b>decision path:</b> {escape(str(decision_path))}</p>"
    f"<p><b>promote:</b> {escape(str(data.get('promote')))}</p>"
    "<h2>passed reasons</h2><ul>"
    + "".join(f"<li>{escape(str(item))}</li>" for item in passed)
    + "</ul><h2>failed reasons</h2><ul>"
    + "".join(f"<li>{escape(str(item))}</li>" for item in failed)
    + "</ul><h2>raw decision</h2><pre>"
    + escape(json.dumps(data, ensure_ascii=False, indent=2))
    + "</pre>",
    encoding="utf-8",
)
print(out)
PY
```

Expected:

```text
runs/phase7a_promotion_decision_20260616.html is created.
```

- [ ] **Step 5: Branch on phase7a**

Use this rule:

```text
If phase7a PROMOTE:
    It becomes the candidate initialization for graph8 and later comparison.

If phase7a REJECT but has target gains with isolated regressions:
    Keep phase6c active, but still run graph8 only if graph8 init checkpoint exists and phase7a did not OOM or corrupt training.

If phase7a REJECT because it broadly regresses protected variants:
    Do not use phase7a as active. Use phase6c for the next hard-mining run.
```

## Task 3: Run The 8-Layer GraphMatcher Comparison

**Files:**
- Run: `runs/train_h100_fov076_phase7b_graph8_20260616.sh`
- Create after run starts: `runs/eval_h100_fov076_phase7b_promotion_20260616.sh`
- Read: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_*/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt`

- [ ] **Step 1: Verify graph8 watcher or launch manually**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 runs/watch_phase7a_promotion_then_graph8_20260616.log
pgrep -af 'train_h100_fov076_phase7b_graph8|benchmark_lazy_pose_pairs.py.*phase7b_fov76_graph8' || true
```

Expected:

```text
If phase7a promotion decision exists, graph8 should start after all promotion/eval processes exit.
```

If no graph8 process exists and phase7a promotion decision exists, run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
setsid runs/train_h100_fov076_phase7b_graph8_20260616.sh > runs/train_h100_fov076_phase7b_graph8_manual_20260616.launch.log 2>&1 &
```

Expected:

```text
A benchmark_lazy_pose_pairs.py graph8 process starts.
```

- [ ] **Step 2: Monitor initial graph8 stability**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
GRAPH8_LOG="$(ls -1t runs/train_h100_fov076_phase7b_graph8_*.log | head -1)"
echo "$GRAPH8_LOG"
tail -120 "$GRAPH8_LOG"
```

Expected:

```text
The run reaches at least step 20.
No CUDA OOM, NaN, nonfinite loss, or shape mismatch appears.
GPU memory stays below device capacity.
```

- [ ] **Step 3: Monitor graph8 completion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
GRAPH8_LOG="$(ls -1t runs/train_h100_fov076_phase7b_graph8_*.log | head -1)"
tail -160 "$GRAPH8_LOG"
```

Expected:

```text
The run reaches its configured final step.
Final visual report writes.
best_by_match_score_pytorch_pfm_state.pt exists.
```

- [ ] **Step 4: Create phase7b promotion launcher**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path

data_root = Path("/media/w24/D/xjw深度学习训练数据")
run_roots = sorted((data_root / "pfm_runs").glob("phase7b_fov76_graph8_h384_*"))
if not run_roots:
    raise SystemExit("phase7b graph8 run root not found")
run_root = run_roots[-1]
checkpoint = run_root / "train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
if not checkpoint.exists():
    raise SystemExit(f"checkpoint not found: {checkpoint}")
script = Path("runs/eval_h100_fov076_phase7b_promotion_20260616.sh")
script.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
STAMP="$(date +%Y%m%d_%H%M%S)"
export PYTHONPATH=python:scripts
/home/w24/anaconda3/envs/cppTorch/bin/python scripts/run_fov76_checkpoint_promotion_pipeline.py \\
  --baseline-run-root "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433" \\
  --candidate-run-root "{run_root}" \\
  --candidate-state "{checkpoint}" \\
  --output-dir "{run_root}/promotion_phase5g_profile_p90delta0_expanded200_${{STAMP}}" \\
  --post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \\
  --dual-checkpoint-rescue-selector \\
  --dual-checkpoint-rescue-label phase5g_phase7b_graph8_selector_p90delta0 \\
  --dual-checkpoint-rescue-max-homography-p90-delta-px 0.0 \\
  --formal-candidate-pairs 200 \\
  --guard-candidate-pairs 200 \\
  2>&1 | tee "runs/eval_h100_fov076_phase7b_promotion_${{STAMP}}.log"
""", encoding="utf-8")
script.chmod(0o755)
print(script)
print(run_root)
print(checkpoint)
PY
bash -n runs/eval_h100_fov076_phase7b_promotion_20260616.sh
```

Expected:

```text
runs/eval_h100_fov076_phase7b_promotion_20260616.sh exists and passes bash -n.
```

- [ ] **Step 5: Run phase7b promotion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
setsid runs/eval_h100_fov076_phase7b_promotion_20260616.sh > runs/eval_h100_fov076_phase7b_promotion_20260616.launch.log 2>&1 &
```

Expected:

```text
A run_fov76_checkpoint_promotion_pipeline.py process starts for phase7b.
```

- [ ] **Step 6: Parse phase7b decision**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs")
paths = sorted(root.glob("phase7b_fov76_graph8_h384_*/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json"))
if not paths:
    raise SystemExit("phase7b promotion_decision.json not found")
data = json.loads(paths[-1].read_text(encoding="utf-8"))
print(paths[-1])
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
```

Expected:

```text
The JSON clearly states promote true/false.
```

## Task 4: Compare Phase6c, Phase7a, And Phase7b

**Files:**
- Read: `runs/fov76_active_mainline_validation_phase6c_p90delta0_20260616.json`
- Read: `runs/phase7a_promotion_decision_20260616.html`
- Read: latest phase7b `promotion_decision.json`
- Create: `runs/phase7_candidate_comparison_20260616.html`

- [ ] **Step 1: Collect all decisions**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

items = []

phase6c = Path("runs/fov76_active_mainline_validation_phase6c_p90delta0_20260616.json")
if phase6c.exists():
    items.append(("phase6c_active_selector", phase6c, json.loads(phase6c.read_text(encoding="utf-8"))))

for label, pattern in [
    ("phase7a_4l384", "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json"),
    ("phase7b_graph8", "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_*/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json"),
]:
    paths = sorted(Path("/").glob(pattern.lstrip("/")))
    if paths:
        items.append((label, paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))))

for label, path, data in items:
    print("==", label)
    print(path)
    print("promote:", data.get("promote"), "valid:", data.get("valid"))
    print("passed:", len(data.get("passed_reasons", [])))
    print("failed:", data.get("failed_reasons", []))
PY
```

Expected:

```text
The command prints phase6c, phase7a, and phase7b availability with promote/valid flags.
```

- [ ] **Step 2: Select active path**

Use this selection table:

```text
If phase7b PROMOTE:
    Select phase7b graph8 as candidate active model.
    Keep phase6c selector as rollback reference.

If phase7a PROMOTE and phase7b REJECT:
    Select phase7a 4-layer/384 as active model.
    Do not use 8-layer inference in production.

If both phase7a and phase7b REJECT:
    Keep phase6c active selector and mine phase7 failures.

If graph8 improves recall but adds wrong matches:
    Keep phase6c active, mine graph8 false-match rows, and do not promote graph8.
```

- [ ] **Step 3: Write comparison HTML**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
from html import escape
import json

rows = []
phase6c = Path("runs/fov76_active_mainline_validation_phase6c_p90delta0_20260616.json")
if phase6c.exists():
    rows.append(("phase6c_active_selector", phase6c, json.loads(phase6c.read_text(encoding="utf-8"))))

for label, pattern in [
    ("phase7a_4l384", "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json"),
    ("phase7b_graph8", "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_*/promotion_phase5g_profile_p90delta0_expanded200_*/promotion_decision.json"),
]:
    paths = sorted(Path("/").glob(pattern.lstrip("/")))
    if paths:
        rows.append((label, paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))))

body = [
    "<!doctype html><meta charset='utf-8'><title>phase7 candidate comparison</title>",
    "<h1>phase7 candidate comparison</h1>",
    "<table border='1' cellspacing='0' cellpadding='6'>",
    "<tr><th>candidate</th><th>path</th><th>promote</th><th>valid</th><th>failed reasons</th></tr>",
]
for label, path, data in rows:
    failed = data.get("failed_reasons", [])
    body.append(
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{escape(str(path))}</td>"
        f"<td>{escape(str(data.get('promote')))}</td>"
        f"<td>{escape(str(data.get('valid')))}</td>"
        f"<td><pre>{escape(json.dumps(failed, ensure_ascii=False, indent=2))}</pre></td>"
        "</tr>"
    )
body.append("</table>")
body.append("<h2>Raw decisions</h2>")
for label, path, data in rows:
    body.append(f"<h3>{escape(label)}</h3><pre>{escape(json.dumps(data, ensure_ascii=False, indent=2))}</pre>")

out = Path("runs/phase7_candidate_comparison_20260616.html")
out.write_text("\n".join(body), encoding="utf-8")
print(out)
PY
```

Expected:

```text
runs/phase7_candidate_comparison_20260616.html is created.
```

## Task 5: Mine Hard Failures From The Best Non-Regressing Candidate

**Files:**
- Read: latest promotion formal reports under the selected candidate run root
- Create: hard mining output under selected run root
- Create: `runs/phase8_hard_failure_manifest_summary_20260616.html`

- [ ] **Step 1: Choose mining source**

Use this rule:

```text
If phase7b PROMOTE:
    Mine phase7b residual low-precision and low-match rows.

If phase7a PROMOTE:
    Mine phase7a residual low-precision and low-match rows.

If both reject:
    Mine the rejected candidate with the best target recall gain and phase6c active baseline disagreements.
```

- [ ] **Step 2: Run hard failure mining**

Run with the selected `RUN_ROOT`:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
RUN_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507"
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/mine_hard_failure_pairs.py \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/hard_mining/phase8_hard_failures_20260616" \
  --min-wrong 1 \
  --max-correct 2 \
  --max-filtered-matches 8 \
  --include-variants extreme_02,extreme_03 \
  --write-html
```

Expected:

```text
The output directory contains CSV/JSON/HTML hard failure files.
The mined rows are evaluation evidence only and are not directly used for training until mapped to train split analogs.
```

- [ ] **Step 3: Build train-only replay manifest**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
RUN_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507"
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/build_train_replay_from_pair_deltas.py \
  --pair-root "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal" \
  --delta-dir "$RUN_ROOT/hard_mining/phase8_hard_failures_20260616" \
  --output-dir "$RUN_ROOT/hard_mining/phase8_train_replay_20260616" \
  --split train \
  --include-variants extreme_02,extreme_03 \
  --max-pairs 2000 \
  --write-html
```

Expected:

```text
The output replay manifest contains train split pairs only.
No val/test pair is written to the replay manifest.
```

- [ ] **Step 4: Write hard mining summary**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
from html import escape

run_root = Path("/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7a_fov76_complete_4l384_20260616_233507")
mining = run_root / "hard_mining/phase8_hard_failures_20260616"
replay = run_root / "hard_mining/phase8_train_replay_20260616"
out = Path("runs/phase8_hard_failure_manifest_summary_20260616.html")
out.write_text(
    "<!doctype html><meta charset='utf-8'><title>phase8 hard mining</title>"
    "<h1>phase8 hard failure mining</h1>"
    f"<p><b>source run:</b> {escape(str(run_root))}</p>"
    f"<p><b>hard mining:</b> {escape(str(mining))}</p>"
    f"<p><b>train replay:</b> {escape(str(replay))}</p>",
    encoding="utf-8",
)
print(out)
PY
```

Expected:

```text
runs/phase8_hard_failure_manifest_summary_20260616.html is created.
```

## Task 6: Run The Next Controlled Matcher Training Round

**Files:**
- Create: `runs/train_h100_fov076_phase8_hard_replay_20260616.sh`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_*`

- [ ] **Step 1: Choose phase8 initialization**

Use this rule:

```text
If phase7b PROMOTE:
    INIT_STATE = phase7b best checkpoint
    graph_attention_layers = 8
    graph_train_max_attention_layers = 8

If phase7a PROMOTE:
    INIT_STATE = phase7a best checkpoint
    graph_attention_layers = 4
    graph_train_max_attention_layers = 4

If neither promotes:
    INIT_STATE = phase6c best checkpoint
    graph_attention_layers = 4
    graph_train_max_attention_layers = 4
```

- [ ] **Step 2: Create phase8 train script**

Use the same stable defaults:

```text
AMP on
activation checkpointing on
workers 10
prefetch batches 32
worker cache items 128
crop 2048
candidate_topk 256
hidden dim 384
extractor frozen
no no_match_prior
no hard_negative_dustbin increase
positive dustbin margin kept low
RANSAC consistency kept low/medium
teacher guard enabled
match count floor enabled
auto visual eval enabled
```

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
cp runs/train_h100_fov076_phase7a_complete_4l384_20260616.sh runs/train_h100_fov076_phase8_hard_replay_20260616.sh
chmod +x runs/train_h100_fov076_phase8_hard_replay_20260616.sh
bash -n runs/train_h100_fov076_phase8_hard_replay_20260616.sh
```

Expected:

```text
The copied script exists and passes bash -n before parameter edits.
```

Then edit these values in `runs/train_h100_fov076_phase8_hard_replay_20260616.sh`:

```text
RUN_ROOT:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_${STAMP}

steps:
1600 for a fast validation pass
4000 only after the 1600-step pass improves or stays neutral

pair sampling:
main fov76 train manifest remains active
hard replay manifest gets elevated sampling only if the script supports explicit replay input

loss:
do not increase dustbin/no-match weights
increase hard positive/false-match pressure only through pair-level margins and RANSAC consistency
```

- [ ] **Step 3: Launch phase8**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
setsid runs/train_h100_fov076_phase8_hard_replay_20260616.sh > runs/train_h100_fov076_phase8_hard_replay_20260616.launch.log 2>&1 &
```

Expected:

```text
phase8 benchmark_lazy_pose_pairs.py starts.
```

- [ ] **Step 4: Monitor phase8 health**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PHASE8_LOG="$(ls -1t runs/train_h100_fov076_phase8_hard_replay_*.log | head -1)"
tail -160 "$PHASE8_LOG"
```

Expected:

```text
No NaN/OOM/nonfinite markers.
data_wait remains mostly near zero after warmup.
visual eval writes at configured intervals.
```

## Task 7: Promote Phase8 And Decide Whether To Touch The Extractor

**Files:**
- Create: `runs/eval_h100_fov076_phase8_promotion_20260616.sh`
- Create: `runs/phase8_optimization_decision_20260616.html`

- [ ] **Step 1: Run phase8 promotion**

Create the promotion script using the same formal gate as phase7:

```text
post-filter profile:
fov76_geo5_geo10_extreme_rescue_lowmatch_guard

formal candidate pairs:
200

guard candidate pairs:
200

homography p90 delta:
0.0
```

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
bash -n runs/eval_h100_fov076_phase8_promotion_20260616.sh
setsid runs/eval_h100_fov076_phase8_promotion_20260616.sh > runs/eval_h100_fov076_phase8_promotion_20260616.launch.log 2>&1 &
```

Expected:

```text
phase8 promotion runs and writes promotion_decision.json.
```

- [ ] **Step 2: Decide extractor work**

Use this rule:

```text
If phase8 improves target recall but failures are mostly wrong matches in repeated texture:
    Do not unfreeze extractor.
    Add matcher-level false-match loss and geometry consistency loss first.

If phase8 still has many zero/low-match extreme_03 rows but precision is acceptable:
    Add hard replay sampling and candidate selection improvements.
    Keep extractor frozen.

If matcher/filter changes plateau and descriptor-level recall is poor:
    Start extractor ablation.

If geometry-aware pooling hurts descriptor recall:
    Run geometry pooling ablation before changing backbone.
```

- [ ] **Step 3: Extractor ablation only after evidence**

If extractor work is justified, run in this order:

```text
1. heatmap * quality -> heatmap * (0.5 + 0.5 * quality)
2. keypoint branch stage1 skip for high-frequency crater/shadow details
3. geometry pooling ablation:
   A full orientation + scale + affine
   B orientation + scale only
   C plain bilinear pooling
4. descriptor fusion ablation:
   learned only
   texture only
   learned + texture gated fusion
```

Expected:

```text
Only one extractor change is tested per training round.
Every extractor change gets a dedicated promotion report.
```

## Task 8: Commit And Push Tracked Changes

**Files:**
- Track: `docs/superpowers/plans/2026-06-16-fov76-full-next-step-plan.md`
- Track only if changed: `scripts/README.md`
- Do not track: ignored `runs/*.log`, `runs/*.html`, `runs/*.sh`
- Do not touch: `0`

- [ ] **Step 1: Check status**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git status --short --branch
```

Expected:

```text
The new plan doc appears as an untracked or modified tracked file.
The untracked file "0" remains untouched.
```

- [ ] **Step 2: Commit only tracked plan/code documentation**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git add docs/superpowers/plans/2026-06-16-fov76-full-next-step-plan.md
git commit -m "Document fov76 full next step plan"
```

Expected:

```text
One commit is created.
```

- [ ] **Step 3: Push**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git push
```

Expected:

```text
origin/main receives the new plan commit.
```

## Execution Order

Use this exact order:

```text
1. Do not interrupt phase7a; monitor it to clean exit.
2. Run/finish phase7a promotion.
3. Launch graph8 only after phase7a promotion decision exists.
4. Promote graph8 with the same gate.
5. Compare phase6c, phase7a, phase7b.
6. Activate only a candidate that passes promotion.
7. Mine failures from the best evidence source.
8. Run phase8 hard replay, still matcher-first.
9. Promote phase8.
10. Touch extractor only if matcher/filter/geometry evidence shows extractor is the bottleneck.
```

## Next Decision Summary

The immediate next action is not a new architecture change. The immediate next action is:

```text
finish phase7a -> formal promotion -> graph8 controlled comparison -> formal promotion -> choose active path
```

After that, the next optimization should be:

```text
hard failure mining + train-only replay + matcher-level geometry/false-match pressure
```

Extractor optimization is the third step, not the next step.
