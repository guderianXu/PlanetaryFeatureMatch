# fov76 Strict Graph8 To Phase8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strictly validate the phase7b 8-layer matcher, promote only if it beats the active fov76 branch under formal gates, then run filter sweep, hard-failure mining, and phase8 hard-replay training from the safest checkpoint.

**Architecture:** Keep h100/fov76/dom76 as an isolated branch and use `phase5g_active` plus the current fov76 post-filter profile as the promotion reference. Treat every trained checkpoint as a candidate until expanded formal val/test, regression guard, target extreme checks, and selector metadata all pass. Optimize matcher and filtering first; do not unfreeze the extractor until graph/filter/geometry evidence says the matcher is no longer the bottleneck.

**Tech Stack:** Python 3, PyTorch, CUDA AMP, lazy fov76 overlap manifests, `benchmark_lazy_pose_pairs.py`, `run_fov76_checkpoint_promotion_pipeline.py`, `run_graph_filter_sweep.py`, `mine_hard_failure_pairs.py`, `validate_fov76_active_selector.py`, HTML/CSV run records, Git/GitHub.

---

## Current State

Authoritative paths:

```text
project:
/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

python:
/home/w24/anaconda3/envs/cppTorch/bin/python

data root:
/media/w24/D/xjw深度学习训练数据

fov76 pair root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal

guard root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase3d_diff_guard_20260614

current active config:
runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json

current active checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

stable teacher/baseline checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

phase7b graph8 checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

active post-filter profile:
fov76_geo5_geo10_extreme_rescue_lowmatch_guard
```

Known issue to handle first:

```text
The first phase7b promotion run used the default graph depth of 4 for both baseline and candidate.
That result proves the phase7b checkpoint is useful at 4-layer inference, but it is not strict evidence that 8-layer inference is safe.
The promotion pipeline must support baseline graph depth 4 and candidate graph depth 8 in the same run.
```

Hard constraints:

```text
Do not use fov90 in this branch.
Do not touch the untracked file named 0.
Do not promote phase7b from a default-4 promotion result.
Do not increase dustbin/no-match/rejection weights to solve low match count.
Do not unfreeze the full extractor in phase8.
Keep run scripts, logs, pid files, and human-readable summaries under runs/.
Push durable code changes to GitHub after tests pass.
```

## File Map

Pipeline/tooling files:

```text
scripts/run_fov76_checkpoint_promotion_pipeline.py
    Adds and uses per-side graph layer arguments:
    --baseline-graph-layers
    --candidate-graph-layers

python/tests/test_stress_eval_scripts.py
    Covers the planned command list and proves baseline uses 4 layers while candidate uses 8 layers.

scripts/README.md
    Documents per-side graph-layer promotion behavior.
```

Run/control files:

```text
runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.sh
    Launches strict phase5g graph4 vs phase7b graph8 promotion.

runs/phase7_strict_graph8_decision_summary_20260617.html
    Human-readable decision summary.

runs/fov76_active_mainline_config_phase7b_graph8_strict_p90delta0_20260617.json
    Created only if strict phase7b promotion passes.

runs/fov76_active_mainline_validation_phase7b_graph8_strict_p90delta0_20260617.json
    Created only after active selector validation passes.
```

Experiment output files:

```text
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_filter_sweep_${PFM_SELECTED_LABEL}_$(date +%Y%m%d_%H%M%S)/
    Filter sweep outputs for the selected active candidate.

/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures/
    Reusable hard-failure manifests.

/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_$(date +%Y%m%d_%H%M%S)/
    Phase8 matcher-only hard replay training output.
```

## Promotion Rules

Promote a checkpoint only when all gates pass:

```text
formal val/test:
    target variants extreme_02/extreme_03 must improve or stay useful.
    target wrong_delta must not increase beyond the configured gate.
    protected variants mid_01/mid_02/extreme_01/nadir must not regress.

regression guard:
    val/test must remain clean.

dual checkpoint selector:
    selector metadata must exist.
    selector must respect max_rescue_homography_p90_delta_px <= 0.0.
    selector must not use labels such as correct/wrong/precision as runtime decision inputs.

strict graph depth:
    phase5g baseline must be evaluated with graph layers = 4.
    phase7b graph8 candidate must be evaluated with graph layers = 8.
```

Keep phase6c active when any of these happen:

```text
phase7b strict graph8 fails.
phase7b only wins under 4-layer inference.
phase7b increases protected wrong matches.
phase7b improves train/visual samples but fails formal or guard sets.
phase7b gives more matches but worse homography p90/median residuals.
```

## Task 1: Land Per-Side Graph-Depth Promotion Tooling

**Files:**
- Modify: `scripts/run_fov76_checkpoint_promotion_pipeline.py`
- Modify: `python/tests/test_stress_eval_scripts.py`
- Modify: `scripts/README.md`

- [ ] **Step 1: Verify no conflicting long job is running**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py|train_h100_fov076_phase7|eval_h100_fov076_phase7|watch_phase7b' || true
```

Expected:

```text
No unexpected training/evaluation process is running.
If a legitimate eval is running, wait for it before launching another promotion.
```

- [ ] **Step 2: Add the failing command-planning test**

Ensure `python/tests/test_stress_eval_scripts.py` contains a test named:

```python
def test_fov76_promotion_pipeline_allows_per_side_graph_layers(self) -> None:
    args = SimpleNamespace(
        pair_root=Path("/data/pairs"),
        guard_root=Path("/data/pairs/hard_mining/guard"),
        output_dir=Path("/out"),
        baseline_state=Path("/ckpt/phase5g.pt"),
        baseline_run_dir=Path("/runs/phase5g/train_output"),
        candidate_state=Path("/ckpt/phase7b.pt"),
        candidate_run_dir=Path("/runs/phase7b/train_output"),
        baseline_label="phase5g_active",
        candidate_label="phase7b_graph8_h384",
        guard_baseline_label="phase5g_active",
        guard_candidate_label="phase7b_graph8_h384",
        splits=["val", "test"],
        device="cuda",
        python_executable="/env/bin/python",
        seed=20260617,
        crop_size=2048,
        max_image_size=768,
        max_keypoints=512,
        matcher_candidate_topk=256,
        graph_layers=4,
        baseline_graph_layers=4,
        candidate_graph_layers=8,
        geometry_threshold_px=10.0,
        filtered_min_matches=16,
        filtered_min_matches_by_variant=[],
        baseline_filtered_min_matches_by_variant=[],
        candidate_filtered_min_matches_by_variant=[],
        post_filter_profile="",
        geometry_threshold_px_values="",
        min_score_values="",
        filtered_min_matches_values="",
        adaptive_geometry_rescue_variants="",
        baseline_adaptive_geometry_rescue_variants="",
        candidate_adaptive_geometry_rescue_variants="",
        low_match_geometry_guard_variants="",
        baseline_low_match_geometry_guard_variants="",
        candidate_low_match_geometry_guard_variants="",
        dual_checkpoint_rescue_selector=False,
        extra_regression_guard_set=[],
    )

    commands = fov76_gate_mod.planned_commands(args)
    baseline_formal = commands[0]
    candidate_formal = commands[1]
    baseline_guard = commands[2]
    candidate_guard = next(
        command
        for command in commands
        if "/out/eval/guard/phase7b_graph8_h384_" in command[command.index("--output-dir") + 1]
    )

    self.assertEqual(baseline_formal[baseline_formal.index("--graph-max-attention-layers") + 1], "4")
    self.assertEqual(candidate_formal[candidate_formal.index("--graph-max-attention-layers") + 1], "8")
    self.assertEqual(baseline_guard[baseline_guard.index("--graph-max-attention-layers") + 1], "4")
    self.assertEqual(candidate_guard[candidate_guard.index("--graph-max-attention-layers") + 1], "8")
```

- [ ] **Step 3: Run the focused test and confirm failure before implementation if the patch is not present**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_per_side_graph_layers
```

Expected before implementation:

```text
FAIL because candidate_formal uses graph-max-attention-layers 4 instead of 8.
```

- [ ] **Step 4: Implement per-side graph layer selection**

Ensure `scripts/run_fov76_checkpoint_promotion_pipeline.py` defines:

```python
def _graph_layers(args: argparse.Namespace, *, model: EvalModel) -> int:
    value = int(getattr(args, "graph_layers", 4))
    if _matches_baseline_model(args, model=model):
        baseline_value = getattr(args, "baseline_graph_layers", None)
        if baseline_value is not None:
            value = int(baseline_value)
    if _matches_candidate_model(args, model=model):
        candidate_value = getattr(args, "candidate_graph_layers", None)
        if candidate_value is not None:
            value = int(candidate_value)
    return value
```

Ensure the common sweep command uses:

```python
"--graph-max-attention-layers",
str(_graph_layers(args, model=model)),
```

Ensure argument parsing includes:

```python
parser.add_argument("--baseline-graph-layers", type=int, default=None)
parser.add_argument("--candidate-graph-layers", type=int, default=None)
```

- [ ] **Step 5: Document the new arguments**

Ensure `scripts/README.md` says:

```text
`--baseline-graph-layers` / `--candidate-graph-layers` can override `--graph-layers` for each side of one promotion run. Use this for phase5g graph4 baseline versus phase7b graph8 candidate.
```

- [ ] **Step 6: Run tests**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_per_side_graph_layers
```

Expected:

```text
OK
```

Run the related promotion-pipeline subset:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_plans_selector_as_promotion_candidate \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_larger_validation_pair_counts \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_candidate_only_variant_min_match_gate \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_candidate_only_adaptive_rescue \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_candidate_only_low_match_geometry_guard \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_per_side_adaptive_rescue_thresholds \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_per_side_graph_layers \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_validates_required_inputs_before_running
```

Expected:

```text
........
OK
```

- [ ] **Step 7: Commit and push**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git status --short
git add scripts/run_fov76_checkpoint_promotion_pipeline.py python/tests/test_stress_eval_scripts.py scripts/README.md
git commit -m "fix: allow per-side graph depth in fov76 promotion"
git push
```

Expected:

```text
The commit is on origin/main.
The untracked file 0 remains untracked and untouched.
```

## Task 2: Run Strict Phase7b Graph8 Promotion

**Files:**
- Create: `runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.sh`
- Create: `runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.log`
- Create: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/promotion_phase5g4_candidate8_profile_p90delta0_expanded200_$(date +%Y%m%d_%H%M%S)/`

- [ ] **Step 1: Create the strict promotion launcher**

Create `runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

PY=/home/w24/anaconda3/envs/cppTorch/bin/python
PAIR_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
GUARD_ROOT="$PAIR_ROOT/hard_mining/phase3d_diff_guard_20260614"
BASE_RUN="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output"
BASE_STATE="$BASE_RUN/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
CAND_RUN="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output"
CAND_STATE="$CAND_RUN/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
OUT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/promotion_phase5g4_candidate8_profile_p90delta0_expanded200_$(date +%Y%m%d_%H%M%S)"

PYTHONPATH=python:scripts "$PY" scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root "$PAIR_ROOT" \
  --guard-root "$GUARD_ROOT" \
  --output-dir "$OUT" \
  --baseline-state "$BASE_STATE" \
  --baseline-run-dir "$BASE_RUN" \
  --candidate-state "$CAND_STATE" \
  --candidate-run-dir "$CAND_RUN" \
  --baseline-label phase5g_active \
  --candidate-label phase7b_graph8_h384 \
  --guard-baseline-label phase5g_active \
  --guard-candidate-label phase7b_graph8_h384 \
  --baseline-graph-layers 4 \
  --candidate-graph-layers 8 \
  --dual-checkpoint-rescue-selector \
  --dual-checkpoint-rescue-label phase5g_phase7b_graph8_h384_layer8_selector_p90delta0 \
  --dual-checkpoint-rescue-max-homography-p90-delta-px 0.0 \
  --python-executable "$PY" \
  --post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \
  --formal-candidate-pairs 200 \
  --guard-candidate-pairs 200
```

- [ ] **Step 2: Syntax-check and launch**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
chmod +x runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.sh
bash -n runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.sh
setsid runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.sh \
  > runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.log 2>&1 &
echo $! > runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.pid
```

Expected:

```text
The process starts in the background.
The log shows baseline graph layers 4 and candidate graph layers 8 in planned graph sweep commands.
```

- [ ] **Step 3: Monitor strict promotion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.log
pgrep -af 'run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Expected:

```text
No traceback, OOM, or missing file error.
Eventually a promotion_decision.json path appears under promotion_phase5g4_candidate8_profile_p90delta0_expanded200_*.
```

- [ ] **Step 4: Parse strict promotion decision**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

root = Path('/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718')
paths = sorted(root.glob('promotion_phase5g4_candidate8_profile_p90delta0_expanded200_*/promotion_decision.json'))
if not paths:
    raise SystemExit('strict phase7b graph8 promotion_decision.json not found')
path = paths[-1]
data = json.loads(path.read_text(encoding='utf-8'))
print(path)
print('promote=', data.get('promote'))
for reason in data.get('reasons', []):
    print(reason)
PY
```

Expected:

```text
promote=True or promote=False is explicit.
Every failed gate is listed when promotion rejects.
```

## Task 3: Select Active Checkpoint

**Files:**
- Read: strict phase7b `promotion_decision.json`
- Read: `runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json`
- Create: `runs/phase7_strict_graph8_decision_summary_20260617.html`
- Create only if promoted: `runs/fov76_active_mainline_config_phase7b_graph8_strict_p90delta0_20260617.json`
- Create only if promoted: `runs/fov76_active_mainline_validation_phase7b_graph8_strict_p90delta0_20260617.json`

- [ ] **Step 1: Write a strict decision summary**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import html
import json

phase7b_root = Path('/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718')
decision_paths = sorted(phase7b_root.glob('promotion_phase5g4_candidate8_profile_p90delta0_expanded200_*/promotion_decision.json'))
if not decision_paths:
    raise SystemExit('strict graph8 decision not found')
decision_path = decision_paths[-1]
data = json.loads(decision_path.read_text(encoding='utf-8'))
rows = [
    ('decision_path', str(decision_path)),
    ('promote', str(data.get('promote'))),
    ('candidate', 'phase7b_graph8_h384'),
    ('baseline_graph_layers', '4'),
    ('candidate_graph_layers', '8'),
]
for i, reason in enumerate(data.get('reasons', []), start=1):
    rows.append((f'reason_{i}', str(reason)))
body = ''.join(f'<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>' for k, v in rows)
out = Path('runs/phase7_strict_graph8_decision_summary_20260617.html')
out.write_text(
    '<!doctype html><meta charset="utf-8"><title>phase7 strict graph8 decision</title>'
    '<h1>phase7 strict graph8 decision</h1><table border="1" cellspacing="0" cellpadding="4">'
    + body + '</table>',
    encoding='utf-8',
)
print(out)
PY
```

Expected:

```text
runs/phase7_strict_graph8_decision_summary_20260617.html is written.
```

- [ ] **Step 2: If strict promotion rejects, keep phase6c active**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
echo "active_config=runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json"
echo "active_label=phase5g_phase6c_selector_p90delta0"
echo "active_graph_layers=4"
```

Expected:

```text
The next filter sweep and hard mining use phase6c, not phase7b graph8.
The default-4 phase7b promotion may be used as diagnostic evidence only.
```

- [ ] **Step 3: If strict promotion passes, write the new active config**

Run only when strict `promote=True`:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

src = Path('runs/fov76_active_mainline_config_phase6c_p90delta0_20260616.json')
cfg = json.loads(src.read_text(encoding='utf-8'))
phase7b_root = Path('/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718')
decision = sorted(phase7b_root.glob('promotion_phase5g4_candidate8_profile_p90delta0_expanded200_*/promotion_decision.json'))[-1]
metadata = decision.parent / 'dual_checkpoint_rescue_selector' / 'metadata.json'
if not metadata.exists():
    raise SystemExit(f'selector metadata missing: {metadata}')
data = json.loads(decision.read_text(encoding='utf-8'))
if not data.get('promote'):
    raise SystemExit('strict phase7b graph8 did not promote')

candidate = {
    'name': 'phase7b_graph8_strict_selector_p90delta0',
    'label': 'phase5g_phase7b_graph8_h384_layer8_selector_p90delta0',
    'role': 'active_candidate',
    'baseline_label': 'phase5g_active',
    'rescue_label': 'phase7b_graph8_h384',
    'rescue_state': str(phase7b_root / 'train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt'),
    'rescue_run_dir': str(phase7b_root / 'train_output'),
    'selector_profile': 'lowmatch_guard_minmatch16_p90delta0_strict_graph8',
    'decision_path': str(decision),
    'metadata_path': str(metadata),
    'summary': {
        'strict_graph_depth': 'baseline=4 candidate=8',
        'promotion': 'strict phase7b graph8 promoted',
    },
}
cfg['date'] = '2026-06-17'
cfg['active_selector'] = candidate['name']
cfg['active_label'] = candidate['label']
cfg['candidates'] = [candidate]
out = Path('runs/fov76_active_mainline_config_phase7b_graph8_strict_p90delta0_20260617.json')
out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(out)
PY
```

Expected:

```text
The new config points to the strict graph8 decision path and selector metadata.
```

- [ ] **Step 4: Validate the active selector config**

Run only when a strict phase7b config was created:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/validate_fov76_active_selector.py \
  --config runs/fov76_active_mainline_config_phase7b_graph8_strict_p90delta0_20260617.json \
  --output-json runs/fov76_active_mainline_validation_phase7b_graph8_strict_p90delta0_20260617.json
```

Expected:

```text
Validation status is PASS.
```

## Task 4: Run Filter Sweep On The Selected Checkpoint

**Files:**
- Create: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_filter_sweep_${PFM_SELECTED_LABEL}_$(date +%Y%m%d_%H%M%S)/`
- Create: `runs/phase8_filter_sweep_${PFM_SELECTED_LABEL}_20260617.log`

- [ ] **Step 1: Export selected checkpoint variables**

Use phase6c if strict phase7b promotion failed:

```bash
export PFM_SELECTED_STATE="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
export PFM_SELECTED_LAYERS="4"
export PFM_SELECTED_LABEL="phase6c_active"
```

Use phase7b only if strict graph8 promotion passed:

```bash
export PFM_SELECTED_STATE="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
export PFM_SELECTED_LAYERS="8"
export PFM_SELECTED_LABEL="phase7b_graph8_strict"
```

- [ ] **Step 2: Launch the filter sweep**

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
  > "runs/phase8_filter_sweep_${PFM_SELECTED_LABEL}_20260617.log" 2>&1 &
echo $! > "runs/phase8_filter_sweep_${PFM_SELECTED_LABEL}_20260617.pid"
```

Expected:

```text
The sweep outputs graph_filter_sweep_summary.csv and index.html.
The selected filter must not trade large wrong increases for match count.
```

- [ ] **Step 3: Read the best filter rows**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import csv
root = sorted(Path('/media/w24/D/xjw深度学习训练数据/pfm_runs').glob('phase8_filter_sweep_*'))[-1]
summary = root / 'graph_filter_sweep_summary.csv'
if not summary.exists():
    raise SystemExit(f'missing {summary}')
with summary.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
def score(r):
    return (
        float(r.get('filtered_precision', 0.0)),
        int(float(r.get('filtered_correct', 0.0))),
        -int(float(r.get('filtered_wrong', 0.0))),
        -int(float(r.get('zero_match_rows', 0.0))),
    )
for row in sorted(rows, key=score, reverse=True)[:10]:
    print(row)
PY
```

Expected:

```text
Top rows show high precision, nonzero match count, and no new false cluster pattern.
```

## Task 5: Mine Hard Failures

**Files:**
- Read: selected visual or sweep report summary CSV
- Read: `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/overlap_edges_train.csv`
- Create: `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures/`
- Create: `runs/phase8_hard_failure_manifest_summary_20260617.html`

- [ ] **Step 1: Choose the source report directory**

Use the report directory from the selected checkpoint and selected filter. If strict phase7b passed and its final training visual is the source:

```bash
export PFM_REPORT_DIR="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/visual_report_step_000800"
```

If phase6c remains active, use the latest phase6c visual report with `all_filtered_summary.csv`:

```bash
export PFM_REPORT_DIR="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_20260616_225134/train_output/visual_report"
```

- [ ] **Step 2: Mine residual failures and extreme low-match rows**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
export PFM_HARD_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures"
mkdir -p "$PFM_HARD_ROOT"
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/mine_hard_failure_pairs.py \
  --pair-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/overlap_edges_train.csv" \
  --summary-csv "$PFM_REPORT_DIR/all_filtered_summary.csv" \
  --output-manifest "$PFM_HARD_ROOT/hard_failures_train.csv" \
  --mixed-base-manifest "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/overlap_edges_train.csv" \
  --mixed-output-manifest "$PFM_HARD_ROOT/hard_mixed_train.csv" \
  --mixed-hard-fraction 0.35 \
  --report-html "runs/phase8_hard_failure_manifest_summary_20260617.html" \
  --residual-filtered \
  --only-extreme-variants \
  --extreme-variants extreme_02,extreme_03 \
  --include-extreme-without-failure
```

Expected:

```text
hard_failures_train.csv exists.
hard_mixed_train.csv exists.
The HTML report lists low_precision, high_false, low_match_count, and extreme_view buckets.
```

- [ ] **Step 3: Verify manifests are train-only and nonempty**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import csv

for path in [
    Path('/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures/hard_failures_train.csv'),
    Path('/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260617_fov76_phase8_hard_failures/hard_mixed_train.csv'),
]:
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    non_train = [r for r in rows if r.get('split') not in ('', 'train')]
    print(path, 'rows=', len(rows), 'non_train=', len(non_train))
    if not rows:
        raise SystemExit(f'empty manifest: {path}')
    if non_train:
        raise SystemExit(f'non-train rows found in {path}')
PY
```

Expected:

```text
Both manifests report rows > 0 and non_train=0.
```

## Task 6: Train Phase8 Hard-Replay Matcher

**Files:**
- Create: `runs/train_h100_fov076_phase8_hard_replay_20260617.sh`
- Create: `runs/train_h100_fov076_phase8_hard_replay_20260617.log`
- Create: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_$(date +%Y%m%d_%H%M%S)/`

- [ ] **Step 1: Create the phase8 training launcher**

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
GRAPH_LAYERS=4

if [[ -f runs/fov76_active_mainline_config_phase7b_graph8_strict_p90delta0_20260617.json ]]; then
  INIT_STATE="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase7b_fov76_graph8_h384_20260617_001718/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
  GRAPH_LAYERS=8
fi

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
  --skip-nonfinite-steps \
  --learning-rate 1e-11 \
  --train-graph-matcher \
  --train-graph-calibration-only \
  --no-train-descriptor-head \
  --descriptor-geometry-mode full \
  --quality-score-mode soft \
  --graph-hidden-dim 384 \
  --graph-attention-layers "$GRAPH_LAYERS" \
  --graph-matcher-loss-weight 0.006 \
  --graph-matcher-metadata-mode calibrated \
  --graph-matcher-no-match-points 0 \
  --graph-matcher-no-match-weight 0.0 \
  --graph-matcher-assignment-weight 0.003 \
  --graph-matcher-accept-weight 0.00001 \
  --graph-matcher-stop-confidence-weight 0.0 \
  --graph-matcher-hard-negative-dustbin-weight 0.0 \
  --graph-matcher-train-max-attention-layers "$GRAPH_LAYERS" \
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
  --visual-graph-max-attention-layers "$GRAPH_LAYERS" \
  --visual-graph-width-prune-keep-ratio 1.0 \
  --visual-max-keypoints 512 \
  --visual-candidate-pairs 80 \
  --visual-select-count 12
```

- [ ] **Step 2: Launch phase8 training**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
chmod +x runs/train_h100_fov076_phase8_hard_replay_20260617.sh
bash -n runs/train_h100_fov076_phase8_hard_replay_20260617.sh
setsid runs/train_h100_fov076_phase8_hard_replay_20260617.sh \
  > runs/train_h100_fov076_phase8_hard_replay_20260617.log 2>&1 &
echo $! > runs/train_h100_fov076_phase8_hard_replay_20260617.pid
```

Expected:

```text
Training starts.
Extractor is frozen.
Graph layers match the selected active checkpoint: 4 for phase6c, 8 for strict phase7b.
```

- [ ] **Step 3: Monitor phase8 training**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
tail -120 runs/train_h100_fov076_phase8_hard_replay_20260617.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

Expected:

```text
No OOM.
No NaN/nonfinite-step storm.
Visual reports appear every 200 steps.
positive_vs_dustbin margin does not collapse.
Filtered match count does not trend to zero.
```

## Task 7: Promote Or Reject Phase8

**Files:**
- Read: latest `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase8_fov76_hard_replay_*/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt`
- Create: `runs/eval_h100_fov076_phase8_promotion_20260617.sh`
- Create: `runs/phase8_promotion_decision_20260617.html`

- [ ] **Step 1: Create the phase8 promotion launcher**

Use `apply_patch` to add `runs/eval_h100_fov076_phase8_promotion_20260617.sh` with this content:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

PHASE8_ROOT="${1:?usage: $0 /path/to/phase8_run_root}"
PY=/home/w24/anaconda3/envs/cppTorch/bin/python
PAIR_ROOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
GUARD_ROOT="$PAIR_ROOT/hard_mining/phase3d_diff_guard_20260614"
BASE_RUN="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output"
BASE_STATE="$BASE_RUN/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
CAND_RUN="$PHASE8_ROOT/train_output"
CAND_STATE="$CAND_RUN/checkpoints/best_by_match_score_pytorch_pfm_state.pt"
GRAPH_LAYERS=4
if grep -q 'graph8' <<< "$PHASE8_ROOT"; then
  GRAPH_LAYERS=8
fi
OUT="$PHASE8_ROOT/promotion_phase5g_phase8_profile_p90delta0_expanded200_$(date +%Y%m%d_%H%M%S)"

PYTHONPATH=python:scripts "$PY" scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root "$PAIR_ROOT" \
  --guard-root "$GUARD_ROOT" \
  --output-dir "$OUT" \
  --baseline-state "$BASE_STATE" \
  --baseline-run-dir "$BASE_RUN" \
  --candidate-state "$CAND_STATE" \
  --candidate-run-dir "$CAND_RUN" \
  --baseline-label phase5g_active \
  --candidate-label phase8_hard_replay \
  --guard-baseline-label phase5g_active \
  --guard-candidate-label phase8_hard_replay \
  --baseline-graph-layers 4 \
  --candidate-graph-layers "$GRAPH_LAYERS" \
  --dual-checkpoint-rescue-selector \
  --dual-checkpoint-rescue-label phase5g_phase8_selector_p90delta0 \
  --dual-checkpoint-rescue-max-homography-p90-delta-px 0.0 \
  --python-executable "$PY" \
  --post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \
  --formal-candidate-pairs 200 \
  --guard-candidate-pairs 200
```

Then run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
chmod +x runs/eval_h100_fov076_phase8_promotion_20260617.sh
bash -n runs/eval_h100_fov076_phase8_promotion_20260617.sh
```

Expected:

```text
Script syntax passes.
The script uses graph4 baseline and candidate graph depth matching phase8 lineage.
```

- [ ] **Step 2: Run phase8 promotion**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PHASE8_ROOT="$(find "/media/w24/D/xjw深度学习训练数据/pfm_runs" -maxdepth 1 -type d -name 'phase8_fov76_hard_replay_*' | sort | tail -1)"
setsid runs/eval_h100_fov076_phase8_promotion_20260617.sh "$PHASE8_ROOT" \
  > runs/eval_h100_fov076_phase8_promotion_20260617.log 2>&1 &
echo $! > runs/eval_h100_fov076_phase8_promotion_20260617.pid
```

Expected:

```text
Promotion writes a new promotion_decision.json under the phase8 run root.
```

- [ ] **Step 3: Parse phase8 decision**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
from pathlib import Path
import json

root = sorted(Path('/media/w24/D/xjw深度学习训练数据/pfm_runs').glob('phase8_fov76_hard_replay_*'))[-1]
paths = sorted(root.glob('promotion_phase5g_phase8_profile_p90delta0_expanded200_*/promotion_decision.json'))
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
If promote=True, phase8 becomes the new active candidate after validation.
If promote=False and failures are localized to extreme_02/extreme_03, mine those failures for phase9.
If promote=False with broad protected regressions, stop hard replay and keep phase6c or strict phase7b active.
```

## Task 8: Phase9 Model Optimization Matrix

**Files:**
- Create one `runs/train_h100_fov076_phase9_*.sh` per accepted experiment.
- Create one run directory under `/media/w24/D/xjw深度学习训练数据/pfm_runs/` per experiment.

Run phase9 only after Task 7 has a clear decision.

Experiment order:

```text
1. candidate_topk 256 -> 384 with the same graph depth and same checkpoint.
   Goal: determine whether missed extreme matches are candidate-limited.

2. graph8 h384 long run only if strict graph8 promotion or phase8 graph8 promotion passed.
   Goal: give the larger attention stack enough hard replay without using unvalidated depth.

3. graph8 h512 short smoke only after graph8 h384 passes.
   Goal: test capacity without committing to the memory cost.

4. mined false-edge loss on residual MAGSAC failures.
   Goal: reduce repeated-texture false positives without increasing dustbin/no-match pressure.

5. partial descriptor fusion / quality head unfreeze.
   Goal: improve feature quality only after matcher/filter bottlenecks are exhausted.
```

Do not run these phase9 experiments yet:

```text
full backbone unfreeze
strong dustbin/no-match weight increase
global fov90/fov110/fov76 mixed training
new pair generation outside the fov76 internal branch
```

## Task 9: Verification, Documentation, And GitHub

**Files:**
- Modify as needed: `runs/*.html`
- Modify as needed: `scripts/README.md`
- Modify as needed: `docs/superpowers/plans/*.md`

- [ ] **Step 1: Verify plan and tooling**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
bash -n runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.sh
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_stress_eval_scripts.StressEvalScriptsTest.test_fov76_promotion_pipeline_allows_per_side_graph_layers
```

Expected:

```text
Shell syntax passes.
Unit test passes.
```

- [ ] **Step 2: Commit documentation and run scripts**

Run:

```bash
cd /home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch
git status --short
git add docs/superpowers/plans/2026-06-17-fov76-strict-graph8-phase8-complete-plan.md
git add runs/eval_h100_fov076_phase7b_promotion_graph8strict_20260617.sh || true
git add runs/eval_h100_fov076_phase8_promotion_20260617.sh || true
git add runs/train_h100_fov076_phase8_hard_replay_20260617.sh || true
git commit -m "docs: plan strict fov76 graph8 phase8 workflow"
git push
```

Expected:

```text
Durable plan and reusable run scripts are pushed.
Untracked file 0 remains untouched.
Ignored run logs are not committed.
```

## Decision Table

```text
strict phase7b promotes:
    set phase7b graph8 strict selector as active.
    run filter sweep on graph8.
    mine hard failures from graph8 reports.
    train phase8 with graph layers 8.

strict phase7b rejects but default-4 phase7b was good:
    keep phase6c active.
    use phase7b only as diagnostic evidence.
    run filter sweep on phase6c.
    train phase8 with graph layers 4.

phase8 promotes:
    validate phase8 active config.
    proceed to phase9 capacity/topk experiments.

phase8 rejects with localized extreme failures:
    mine residual failures.
    run another short matcher-only replay with lower LR and same graph depth.

phase8 rejects with broad protected regressions:
    stop the branch.
    keep the previous active selector.
    do not unfreeze extractor or enlarge hidden dim.
```

## Self-Review

Spec coverage:

```text
Strict 8-layer validation is covered by Tasks 1-3.
Filter sweep is covered by Task 4.
Hard failure mining is covered by Task 5.
Phase8 matcher-only training is covered by Task 6.
Formal promotion and rollback are covered by Task 7.
Larger model exploration is gated in Task 8.
GitHub updates are covered by Task 9.
```

Placeholder scan:

```text
No task depends on an unspecified script.
Every dynamic run path uses $(date +%Y%m%d_%H%M%S), not a manual placeholder.
Every promote/reject branch has an explicit next action.
```

Type and argument consistency:

```text
Promotion graph depth uses --baseline-graph-layers and --candidate-graph-layers.
Training graph depth uses --graph-attention-layers, --graph-matcher-train-max-attention-layers, and --visual-graph-max-attention-layers consistently.
The active config is validated by validate_fov76_active_selector.py before being considered active.
```
