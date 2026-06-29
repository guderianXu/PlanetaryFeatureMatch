# PFM Match-Detail Precision Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the current fov76 PFM operating point from about 98.3-98.6% precision to at least 99.0% precision while keeping combined val+test correct matches above 48000 and above LightGlue by a large margin.

**Architecture:** Keep `models/last_good_pytorch_pfm_state.pt` protected. First add an inference-observable per-match precision gate on top of `fov76_graph_magsac2_min24_balanced`, trained only from train split true-geometry labels and selected on val. Only if the gate cannot hit the precision target, run a small false-cluster/geometry-bias GraphMatcher training pass using train-only hard negatives.

**Tech Stack:** Python, PyTorch, OpenCV MAGSAC, existing lazy visual reports, `build_geometry_edge_supervision_dataset.py`, `train_geometry_edge_filter_calibrator.py`, `apply_match_detail_filter_calibrator.py`, `visualize_lazy_pose_matches.py`, `run_fov76_checkpoint_promotion_pipeline.py`.

---

## Current Evidence

- Protected model: `models/last_good_pytorch_pfm_state.pt`
- Current main profile: `fov76_graph_magsac2_min24_balanced`
- Balanced profile combined val+test: `52723` matches, `51839` correct, `884` wrong, precision `98.323%`
- LightGlue combined val+test: `40722` matches, `40315` correct, `407` wrong, precision about `99.0%`
- Wrong-match bucket diagnosis: `651 / 884` wrong matches are near-miss 5-8px, `186 / 884` are false-cluster high-confidence wrongs.
- Offset and soft-boundary experiments fixed training mechanics but did not reach the precision gate:
  - offset-only 2k val: `30461 / 30021 / 440`, precision `98.556%`
  - soft-boundary graph-on 1k quick128 val: `8244 / 8127 / 117`, precision `98.581%`

## Target Gate

All promotion decisions must use these thresholds:

```text
profile = fov76_graph_magsac2_min24_balanced
baseline_state = models/last_good_pytorch_pfm_state.pt
min_correct_combined = 48000
max_wrong_combined = 500
min_precision_combined = 0.9900
must_exceed_lightglue_correct = true
training_must_not_use_val_or_test_pairs = true
```

## Files And Responsibilities

- `runs/pfm_match_detail_precision_gate_20260628/`: New experiment root for generated train details, geometry-edge labels, calibrator outputs, apply results, logs, and HTML summaries.
- `runs/pfm_wrong884_mining_20260627/val/filtered_match_details.csv`: Existing balanced-profile val filtered match details for diagnosis and threshold selection.
- `runs/pfm_wrong884_mining_20260627/test/filtered_match_details.csv`: Existing balanced-profile test filtered match details for final holdout check only.
- `scripts/build_geometry_edge_supervision_dataset.py`: Converts match details with true geometry fields into per-edge labels. Use only inference-observable `feature_*` fields for model input.
- `scripts/train_geometry_edge_filter_calibrator.py`: Trains standardized logistic per-match reject filter from train geometry-edge labels.
- `scripts/apply_match_detail_filter_calibrator.py`: Applies the learned reject filter offline to existing match details.
- `scripts/visualize_lazy_pose_matches.py`: Add optional model-backed match-detail gate only after offline gate proves useful.
- `scripts/run_graph_filter_sweep.py`: Add profile support only if the gate is promoted from offline diagnosis.
- `scripts/run_fov76_checkpoint_promotion_pipeline.py`: Add formal promotion wiring for the gated profile.
- `scripts/README.md`: Document new profile, commands, and leakage rules if code integration happens.
- `python/tests/test_*`: Extend existing tests for calibrator application, visual profile expansion, and promotion argument propagation.

## Task 1: Freeze Inputs And Create The Experiment Root

**Files:**
- Read: `runs/pfm_wrong884_mining_20260627/summary.csv`
- Read: `runs/pfm_wrong884_mining_20260627/val/filtered_match_details.csv`
- Read: `runs/pfm_wrong884_mining_20260627/test/filtered_match_details.csv`
- Create: `runs/pfm_match_detail_precision_gate_20260628/`

- [ ] **Step 1: Confirm no long-running PFM task is active**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc \
  "pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py' | grep -v -E 'pgrep -af|grep -v' || true"
```

Expected: no active process lines.

- [ ] **Step 2: Create a fixed run root and baseline audit**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
set -euo pipefail
RUN_ROOT="runs/pfm_match_detail_precision_gate_20260628"
mkdir -p "${RUN_ROOT}"
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python - <<'"'"'PY'"'"'
import csv
import json
from pathlib import Path

root = Path("runs/pfm_match_detail_precision_gate_20260628")
inputs = {
    "balanced_summary": "runs/pfm_wrong884_mining_20260627/summary.csv",
    "val_filtered_match_details": "runs/pfm_wrong884_mining_20260627/val/filtered_match_details.csv",
    "test_filtered_match_details": "runs/pfm_wrong884_mining_20260627/test/filtered_match_details.csv",
    "baseline_state": "models/last_good_pytorch_pfm_state.pt",
    "profile": "fov76_graph_magsac2_min24_balanced",
}
for name, path_text in inputs.items():
    if name == "profile":
        continue
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"{name} missing: {path}")
summary_rows = list(csv.DictReader(Path(inputs["balanced_summary"]).open(encoding="utf-8", newline="")))
payload = {
    "inputs": inputs,
    "target_gate": {
        "min_correct_combined": 48000,
        "max_wrong_combined": 500,
        "min_precision_combined": 0.9900,
    },
    "balanced_summary_rows": summary_rows,
}
(root / "input_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload["target_gate"], ensure_ascii=False))
PY
'
```

Expected:

```text
{"min_correct_combined": 48000, "max_wrong_combined": 500, "min_precision_combined": 0.99}
```

## Task 2: Generate Train-Split Balanced Match Details

**Files:**
- Read: `models/last_good_pytorch_pfm_state.pt`
- Read: `/mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/spatial_pair_specs_train.csv`
- Create: `runs/pfm_match_detail_precision_gate_20260628/train_balanced/`
- Create: `runs/pfm_match_detail_precision_gate_20260628/generate_train_balanced_details.sh`
- Create: `runs/pfm_match_detail_precision_gate_20260628/generate_train_balanced_details.log`

- [ ] **Step 1: Write the train detail generation script**

Create `runs/pfm_match_detail_precision_gate_20260628/generate_train_balanced_details.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/e/code/PlanetaryFeatureMatch"
cd "${ROOT}"

ACTIVE_PATTERN='batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py'
ACTIVE_TASKS="$(pgrep -af "${ACTIVE_PATTERN}" | grep -v -E 'pgrep -af|grep -v' || true)"
if [[ -n "${ACTIVE_TASKS}" ]]; then
  echo "[train_balanced_details] active long-running PFM task detected; refusing to start:" >&2
  echo "${ACTIVE_TASKS}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT}/python:${ROOT}/scripts"
export TMPDIR="${PFM_TMPDIR:-/tmp/pfm_pytorch_tmp}"
mkdir -p "${TMPDIR}"

PY="${PFM_PYTHON:-/home/xjw/miniforge3/envs/pfm_torch/bin/python}"
PAIR_ROOT="/mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap"
OUT_DIR="${ROOT}/runs/pfm_match_detail_precision_gate_20260628/train_balanced"

"${PY}" scripts/visualize_lazy_pose_matches.py \
  --render-manifest "${PAIR_ROOT}/manifests/h100km_fov076_render_manifest.csv" \
  --uint8-manifest "${ROOT}/runs/lastgood_vs_verified_vf005_noaug_noheads_lr5e6_10k_best_eval_20260627/empty_uint8_manifest.csv" \
  --pytorch-state "${ROOT}/models/last_good_pytorch_pfm_state.pt" \
  --output-dir "${OUT_DIR}" \
  --run-dir "${ROOT}/runs/pfm_match_detail_precision_gate_20260628" \
  --split train \
  --pair-spec-manifest "${PAIR_ROOT}/spatial_pair_specs_train.csv" \
  --pair-mode mixed \
  --pair-type-weights same_position_view=1,cross_camera=1,cross_fov=0 \
  --image-source render \
  --candidate-pairs 2048 \
  --select-count 0 \
  --seed 20260628 \
  --crop-size 1536 \
  --max-image-size 768 \
  --max-attempts 20 \
  --min-valid-fraction 0.02 \
  --device cuda \
  --descriptor-mode blend \
  --texture-blend-weight 0.35 \
  --keypoint-score-mode learned \
  --matcher-mode graph_matcher \
  --max-keypoints 2048 \
  --max-matches 512 \
  --texture-fraction 0.4 \
  --weak-texture-fraction 0.4 \
  --keypoint-spatial-bins 8 \
  --keypoint-cell-cap 48 \
  --draw-matches 0 \
  --threshold-px 5.0 \
  --post-filter-profile fov76_graph_magsac2_min24_balanced \
  --filtered-report \
  --filtered-mutual \
  --filtered-max-matches 0 \
  --write-all-summary \
  --write-match-details \
  --write-all-match-details \
  --all-match-details-max-results 2048 \
  --all-match-details-max-matches-per-result 512 \
  --html-report
```

- [ ] **Step 2: Run syntax check**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -n runs/pfm_match_detail_precision_gate_20260628/generate_train_balanced_details.sh
```

Expected: exit code `0`.

- [ ] **Step 3: Generate train details**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc \
  "bash runs/pfm_match_detail_precision_gate_20260628/generate_train_balanced_details.sh > runs/pfm_match_detail_precision_gate_20260628/generate_train_balanced_details.log 2>&1"
```

Expected artifacts:

```text
runs/pfm_match_detail_precision_gate_20260628/train_balanced/all_filtered_match_details.csv
runs/pfm_match_detail_precision_gate_20260628/train_balanced/all_filtered_summary.csv
runs/pfm_match_detail_precision_gate_20260628/train_balanced/index.html
```

- [ ] **Step 4: Verify train detail volume**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python - <<'"'"'PY'"'"'
import csv
from pathlib import Path

path = Path("runs/pfm_match_detail_precision_gate_20260628/train_balanced/all_filtered_match_details.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
correct = sum(1 for row in rows if row.get("correct") == "1")
wrong = len(rows) - correct
print(f"train_filtered_matches={len(rows)} correct={correct} wrong={wrong}")
if len(rows) < 10000:
    raise SystemExit("too few train filtered matches for calibrator")
if wrong < 100:
    raise SystemExit("too few train wrong matches for calibrator")
PY
'
```

Expected: at least `10000` filtered train matches and at least `100` wrong train matches.

## Task 3: Build True-Geometry Edge Supervision

**Files:**
- Read: `runs/pfm_match_detail_precision_gate_20260628/train_balanced/all_filtered_match_details.csv`
- Read: `runs/pfm_wrong884_mining_20260627/val/filtered_match_details.csv`
- Read: `runs/pfm_wrong884_mining_20260627/test/filtered_match_details.csv`
- Create: `runs/pfm_match_detail_precision_gate_20260628/geometry_edges/`

- [ ] **Step 1: Build train geometry-edge labels**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
set -euo pipefail
OUT="runs/pfm_match_detail_precision_gate_20260628/geometry_edges"
mkdir -p "${OUT}"
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/build_geometry_edge_supervision_dataset.py \
  --source train,runs/pfm_match_detail_precision_gate_20260628/train_balanced/all_filtered_match_details.csv \
  --output-csv "${OUT}/train_geometry_edges.csv" \
  --summary-json "${OUT}/train_geometry_edges_summary.json" \
  --output-html "${OUT}/train_geometry_edges.html" \
  --max-error-px 5.0 \
  --min-valid-fraction 0.10 \
  --hard-negative-error-px 8.0 \
  --positive-weight 1.0 \
  --invalid-weight 1.0 \
  --hard-negative-weight 4.0 \
  --low-visibility-weight 0.0 \
  --missing-geometry-weight 0.0
'
```

Expected: `train_geometry_edges.csv` exists and contains both `geometry_valid_label=1` and `geometry_invalid_label=1`.

- [ ] **Step 2: Build val geometry-edge labels**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
set -euo pipefail
OUT="runs/pfm_match_detail_precision_gate_20260628/geometry_edges"
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/build_geometry_edge_supervision_dataset.py \
  --source val,runs/pfm_wrong884_mining_20260627/val/filtered_match_details.csv \
  --output-csv "${OUT}/val_geometry_edges.csv" \
  --summary-json "${OUT}/val_geometry_edges_summary.json" \
  --output-html "${OUT}/val_geometry_edges.html" \
  --max-error-px 5.0 \
  --min-valid-fraction 0.10 \
  --hard-negative-error-px 8.0 \
  --positive-weight 1.0 \
  --invalid-weight 1.0 \
  --hard-negative-weight 4.0 \
  --low-visibility-weight 0.0 \
  --missing-geometry-weight 0.0
'
```

Expected: `val_geometry_edges.csv` exists and matches the val balanced-profile wrong population.

- [ ] **Step 3: Build test geometry-edge labels for final holdout only**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
set -euo pipefail
OUT="runs/pfm_match_detail_precision_gate_20260628/geometry_edges"
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/build_geometry_edge_supervision_dataset.py \
  --source test,runs/pfm_wrong884_mining_20260627/test/filtered_match_details.csv \
  --output-csv "${OUT}/test_geometry_edges.csv" \
  --summary-json "${OUT}/test_geometry_edges_summary.json" \
  --output-html "${OUT}/test_geometry_edges.html" \
  --max-error-px 5.0 \
  --min-valid-fraction 0.10 \
  --hard-negative-error-px 8.0 \
  --positive-weight 1.0 \
  --invalid-weight 1.0 \
  --hard-negative-weight 4.0 \
  --low-visibility-weight 0.0 \
  --missing-geometry-weight 0.0
'
```

Expected: `test_geometry_edges.csv` exists. Do not use this file for model selection.

## Task 4: Train A Conservative Per-Match Precision Gate

**Files:**
- Read: `runs/pfm_match_detail_precision_gate_20260628/geometry_edges/train_geometry_edges.csv`
- Read: `runs/pfm_match_detail_precision_gate_20260628/geometry_edges/val_geometry_edges.csv`
- Create: `runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/`

- [ ] **Step 1: Train the logistic geometry-edge filter**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/train_geometry_edge_filter_calibrator.py \
  --train-geometry-edges runs/pfm_match_detail_precision_gate_20260628/geometry_edges/train_geometry_edges.csv \
  --eval-geometry-edges runs/pfm_match_detail_precision_gate_20260628/geometry_edges/val_geometry_edges.csv \
  --output-dir runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1 \
  --epochs 160 \
  --learning-rate 0.04 \
  --l2 0.002 \
  --min-kept-valid-ratio 0.985 \
  --threshold-objective pfm_wrong_cap \
  --threshold-selection-source eval \
  --max-kept-wrong 250 \
  --max-train-rows 300000 \
  --balance-sampling-key target_variant \
  --hard-negative-repeat 4 \
  --max-thresholds 800
'
```

Expected artifacts:

```text
runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json
runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/threshold_sweep.csv
runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/index.html
```

- [ ] **Step 2: Check val gate metrics**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python - <<'"'"'PY'"'"'
import csv
import json
from pathlib import Path

root = Path("runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1")
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
print(json.dumps(summary.get("selected_threshold", summary), ensure_ascii=False, indent=2))
rows = list(csv.DictReader((root / "threshold_sweep.csv").open(encoding="utf-8", newline="")))
if not rows:
    raise SystemExit("missing threshold sweep rows")
best = max(rows, key=lambda row: float(row["eval_precision"]))
print("best_eval_precision", best["threshold"], best["eval_kept_correct"], best["eval_kept_wrong"], best["eval_precision"], best["eval_correct_retention"])
PY
'
```

Promotion-to-integration condition:

```text
eval_kept_wrong <= 250
eval_correct_retention >= 0.985
eval_precision >= 0.990
```

If any condition fails, skip Task 6 and go to Task 8.

## Task 5: Offline Apply On Val And Test

**Files:**
- Read: `runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json`
- Read: `runs/pfm_wrong884_mining_20260627/val/filtered_match_details.csv`
- Read: `runs/pfm_wrong884_mining_20260627/test/filtered_match_details.csv`
- Create: `runs/pfm_match_detail_precision_gate_20260628/offline_apply/`

- [ ] **Step 1: Apply the gate on val**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/apply_match_detail_filter_calibrator.py \
  --match-details runs/pfm_wrong884_mining_20260627/val/filtered_match_details.csv \
  --model-json runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json \
  --output-dir runs/pfm_match_detail_precision_gate_20260628/offline_apply/val
'
```

Expected: `offline_apply/val/summary.json` and `offline_apply/val/pair_summary.csv`.

- [ ] **Step 2: Apply the same frozen gate on test**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/apply_match_detail_filter_calibrator.py \
  --match-details runs/pfm_wrong884_mining_20260627/test/filtered_match_details.csv \
  --model-json runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json \
  --output-dir runs/pfm_match_detail_precision_gate_20260628/offline_apply/test
'
```

Expected: `offline_apply/test/summary.json` and `offline_apply/test/pair_summary.csv`.

- [ ] **Step 3: Compute combined offline gate result**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python - <<'"'"'PY'"'"'
import json
from pathlib import Path

root = Path("runs/pfm_match_detail_precision_gate_20260628/offline_apply")
rows = []
for split in ("val", "test"):
    summary = json.loads((root / split / "summary.json").read_text(encoding="utf-8"))
    rows.append({"split": split, **summary})
combined = {
    "split": "combined",
    "kept_matches": sum(int(row["kept_matches"]) for row in rows),
    "kept_correct": sum(int(row["kept_correct"]) for row in rows),
    "kept_wrong": sum(int(row["kept_wrong"]) for row in rows),
}
combined["kept_precision"] = combined["kept_correct"] / combined["kept_matches"] if combined["kept_matches"] else 0.0
payload = {"splits": rows, "combined": combined}
(root / "combined_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if combined["kept_correct"] < 48000:
    raise SystemExit("combined correct below promotion gate")
if combined["kept_wrong"] > 500:
    raise SystemExit("combined wrong above promotion gate")
if combined["kept_precision"] < 0.9900:
    raise SystemExit("combined precision below promotion gate")
PY
'
```

Expected if the offline gate is worth integrating:

```text
combined kept_correct >= 48000
combined kept_wrong <= 500
combined kept_precision >= 0.9900
```

If this fails, skip Task 6 and run Task 8.

## Task 6: Integrate The Gate Into The Visual/Pipeline Profile

Run this task only if Task 5 passes the combined offline gate.

**Files:**
- Modify: `scripts/visualize_lazy_pose_matches.py`
- Modify: `scripts/run_graph_filter_sweep.py`
- Modify: `scripts/run_fov76_checkpoint_promotion_pipeline.py`
- Modify: `scripts/README.md`
- Test: `python/tests/test_visualize_lazy_pose_matches.py`
- Test: `python/tests/test_run_fov76_checkpoint_promotion_pipeline.py`

- [ ] **Step 1: Write failing tests for the new visualizer CLI**

Add tests in `python/tests/test_visualize_lazy_pose_matches.py`:

```python
def test_parse_args_accepts_match_detail_filter_model(self):
    args = viz.parse_args([
        "--render-manifest", "render.csv",
        "--output-dir", "out",
        "--pytorch-state", "state.pt",
        "--match-detail-filter-model", "runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json",
        "--match-detail-filter-variant-threshold", "extreme_03=0.72",
    ])

    self.assertEqual(
        str(args.match_detail_filter_model),
        "runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json",
    )
    self.assertEqual(args.match_detail_filter_variant_threshold, ["extreme_03=0.72"])
```

Expected red result before implementation:

```bash
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python -m unittest \
  python.tests.test_visualize_lazy_pose_matches.VisualizeLazyPoseMatchesTest.test_parse_args_accepts_match_detail_filter_model
```

Expected: fails because the arguments do not exist.

- [ ] **Step 2: Implement visualizer CLI arguments**

In `scripts/visualize_lazy_pose_matches.py`, add:

```python
parser.add_argument("--match-detail-filter-model", type=Path, default=None)
parser.add_argument("--match-detail-filter-variant-threshold", action="append", default=[])
```

The filter must run after the existing filtered match set is produced and before summary CSV/HTML rows are written. Use `apply_match_detail_filter_calibrator._load_model`, `build_training_rows`, and `build_prediction_rows` so the inference features match the offline calibrator.

- [ ] **Step 3: Write failing tests for profile expansion**

Add a test in `python/tests/test_run_fov76_checkpoint_promotion_pipeline.py` that expands a new profile named:

```text
fov76_graph_magsac2_min24_balanced_detailgate_v1
```

The expected command must include:

```text
--post-filter-profile fov76_graph_magsac2_min24_balanced
--match-detail-filter-model runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json
```

Expected red result: the profile name is unknown.

- [ ] **Step 4: Implement profile expansion**

Add `fov76_graph_magsac2_min24_balanced_detailgate_v1` in the same profile mapping style currently used for `fov76_graph_magsac2_min24_balanced`. The model path must be:

```text
runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json
```

Do not make this profile the default. It is a candidate profile.

- [ ] **Step 5: Update README**

Add one row or note in `scripts/README.md`:

```text
fov76_graph_magsac2_min24_balanced_detailgate_v1 is a candidate post-filter profile that applies an inference-observable match-detail reject calibrator after the balanced MAGSAC-min24 filter. It must only be promoted after formal val/test and guard evaluation. The calibrator is trained from train split true-geometry edge labels and does not use LightGlue labels.
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python -m unittest \
  python.tests.test_visualize_lazy_pose_matches \
  python.tests.test_run_fov76_checkpoint_promotion_pipeline \
  python.tests.test_apply_match_detail_filter_calibrator \
  python.tests.test_train_geometry_edge_filter_calibrator
'
```

Expected: all selected tests pass.

## Task 7: Formal Promotion Evaluation

Run this task only if Task 6 is implemented.

**Files:**
- Read: `models/last_good_pytorch_pfm_state.pt`
- Read: `runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1/model.json`
- Create: `runs/pfm_match_detail_precision_gate_20260628/formal_promotion/`

- [ ] **Step 1: Run dry-run command generation**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap \
  --guard-root /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/hard_mining/guard \
  --output-dir runs/pfm_match_detail_precision_gate_20260628/formal_promotion \
  --baseline-state models/last_good_pytorch_pfm_state.pt \
  --baseline-run-dir runs/corrected_lastgood_graph_vs_lightglue_20260627 \
  --candidate-state models/last_good_pytorch_pfm_state.pt \
  --candidate-run-dir runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1 \
  --candidate-label balanced_detailgate_v1 \
  --guard-candidate-label balanced_detailgate_v1 \
  --post-filter-profile fov76_graph_magsac2_min24_balanced_detailgate_v1 \
  --formal-candidate-pairs 512 \
  --guard-candidate-pairs 512 \
  --write-match-details \
  --dry-run
'
```

Expected: planned commands include the detail-gate profile and write `promotion_pipeline_metadata.json`.

- [ ] **Step 2: Run formal val/test**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap \
  --guard-root /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/hard_mining/guard \
  --output-dir runs/pfm_match_detail_precision_gate_20260628/formal_promotion \
  --baseline-state models/last_good_pytorch_pfm_state.pt \
  --baseline-run-dir runs/corrected_lastgood_graph_vs_lightglue_20260627 \
  --candidate-state models/last_good_pytorch_pfm_state.pt \
  --candidate-run-dir runs/pfm_match_detail_precision_gate_20260628/geometry_edge_filter_v1 \
  --candidate-label balanced_detailgate_v1 \
  --guard-candidate-label balanced_detailgate_v1 \
  --post-filter-profile fov76_graph_magsac2_min24_balanced_detailgate_v1 \
  --formal-candidate-pairs 512 \
  --guard-candidate-pairs 512 \
  --write-match-details
'
```

Promotion condition:

```text
combined correct >= 48000
combined wrong <= 500
combined precision >= 0.9900
combined correct > 40315
regression guard wrong increase <= 0
```

## Task 8: Fallback If The Offline Gate Fails

Run this task only if Task 4 or Task 5 fails.

**Files:**
- Read: `runs/pfm_match_detail_precision_gate_20260628/geometry_edges/train_geometry_edges.csv`
- Create: `runs/pfm_match_detail_precision_gate_20260628/hard_negative_mixture/`
- Create: `runs/pfm_match_detail_precision_gate_20260628/false_cluster_accept_1k/`

- [ ] **Step 1: Build a train-only hard-negative geometry mixture**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
OUT="runs/pfm_match_detail_precision_gate_20260628/hard_negative_mixture"
mkdir -p "${OUT}"
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/build_geometry_edge_hard_negative_mixture.py \
  --base-geometry-edges runs/pfm_match_detail_precision_gate_20260628/geometry_edges/train_geometry_edges.csv \
  --hard-negative-geometry-edges runs/pfm_match_detail_precision_gate_20260628/geometry_edges/train_geometry_edges.csv \
  --output-csv "${OUT}/train_geometry_hard_negative_mixture.csv" \
  --summary-json "${OUT}/summary.json" \
  --output-html "${OUT}/index.html" \
  --target-variant extreme_01 \
  --target-variant extreme_02 \
  --target-variant extreme_03 \
  --min-score 18.0 \
  --min-accept-probability 0.0 \
  --min-raw-margin 0.0 \
  --max-hard-negatives 50000 \
  --max-hard-negatives-per-pair 64
'
```

Expected: mixture contains hard negatives from train only. Do not use val/test rows.

- [ ] **Step 2: Convert the mixture into a false-match replay CSV**

Use the existing false-match CSV format consumed by `benchmark_lazy_pose_pairs.py` only if the mixture rows include the fields required by `build_lazy_false_match_csv.py`: `lazy_pair_key`, `ax`, `ay`, `bx`, `by`, and crop fields. If those fields are absent, stop here and add a small converter script under `scripts/` with tests before training. The converter must read only `train_geometry_hard_negative_mixture.csv` and write `false_match_replay.csv`.

Required converter test name:

```text
python.tests.test_build_geometry_edge_false_match_replay
```

The test must assert that a row with `geometry_hard_negative_label=1` becomes one false-match replay row and a row with `geometry_valid_label=1` is ignored.

- [ ] **Step 3: Run a guarded 1000-step accept/calibration-only training**

Run only after `false_match_replay.csv` exists.

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc '
PYTHONPATH=python:scripts /home/xjw/miniforge3/envs/pfm_torch/bin/python scripts/benchmark_lazy_pose_pairs.py \
  --render-manifest /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/manifests/h100km_fov076_render_manifest.csv \
  --uint8-manifest runs/lastgood_vs_verified_vf005_noaug_noheads_lr5e6_10k_best_eval_20260627/empty_uint8_manifest.csv \
  --output-dir runs/pfm_match_detail_precision_gate_20260628/false_cluster_accept_1k \
  --mode train \
  --pair-spec-manifest /mnt/e/训练数据/火星仿真_h100km_fov076_lat60/h100km_fov076/overlap/spatial_pair_specs_train.csv \
  --image-source render \
  --init-pytorch-state models/last_good_pytorch_pfm_state.pt \
  --device cuda \
  --steps 1000 \
  --batch-pairs 2 \
  --samples-per-pair 512 \
  --crop-size 1536 \
  --training-max-image-size 768 \
  --workers 8 \
  --prefetch-batches 32 \
  --worker-cache-items 64 \
  --learning-rate 3e-6 \
  --weight-decay 1e-4 \
  --amp \
  --amp-dtype float16 \
  --no-activation-checkpointing \
  --batched-descriptor-forward \
  --shuffle \
  --seed 20260628 \
  --progress-every 20 \
  --save-every-steps 250 \
  --gpu-monitor \
  --no-auto-visual-report \
  --visual-eval-every-steps 0 \
  --no-train-descriptor-head \
  --train-graph-matcher \
  --train-graph-calibration-only \
  --graph-matcher-loss-weight 1.0 \
  --graph-matcher-accept-weight 0.0 \
  --graph-matcher-warp-outlier-weight 0.0 \
  --graph-matcher-warp-outlier-accept-weight 0.08 \
  --graph-matcher-warp-outlier-accept-topk 64 \
  --graph-matcher-warp-outlier-accept-residual-threshold-px 8.0 \
  --graph-matcher-warp-outlier-accept-min-score 18.0 \
  --graph-matcher-teacher-guard-state models/last_good_pytorch_pfm_state.pt \
  --graph-matcher-teacher-score-floor-weight 0.06 \
  --graph-matcher-teacher-score-floor-tolerance 0.05 \
  --graph-matcher-teacher-score-floor-min-score 18.0
'
```

Expected training signs:

```text
graph_matcher_warp_outlier_accept_edges > 0
graph_matcher_teacher_score_floor_violations trends down or stays bounded
selected checkpoint does not reduce quick128 correct by more than 2%
```

After this fallback run, evaluate the checkpoint with the existing quick128 flow before any formal 512-pair run.

## Task 9: Final Report

**Files:**
- Read: `runs/pfm_match_detail_precision_gate_20260628/**`
- Create: `runs/pfm_match_detail_precision_gate_20260628/final_report.html`

- [ ] **Step 1: Write the final HTML report**

The report must include:

```text
baseline balanced val/test/combined
LightGlue val/test/combined
offline gate val/test/combined if available
integrated formal promotion result if available
fallback training quick/formal result if run
whether the promotion gate passed
exact model/profile recommended for use
```

- [ ] **Step 2: Verify no long task remains**

Run:

```bash
wsl -d Ubuntu-24.04 --cd /mnt/e/code/PlanetaryFeatureMatch -- bash -lc \
  "pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py' | grep -v -E 'pgrep -af|grep -v' || true"
```

Expected: no active process lines unless a deliberate formal evaluation is still running and its PID/log have been reported to the user.

## Risk Controls

- Do not train on `spatial_pair_specs_val.csv` or `spatial_pair_specs_test.csv`.
- Do not use LightGlue predictions as training labels.
- True geometry may be used for train supervision and val/test evaluation because the simulation provides dense warp labels; the deployed gate must not consume true-geometry fields at inference time.
- The filter model must reject using only inference-observable `feature_*` fields, not `correct`, `error_px`, or `valid_fraction`.
- Do not overwrite `models/last_good_pytorch_pfm_state.pt`.
- Do not promote any checkpoint or profile unless combined correct remains above `48000`.

## Self-Review

- Spec coverage: The plan covers the next precision target, uses train-only labels, defines val selection and test holdout, and includes a fallback training route if the offline gate fails.
- 占位扫描: All commands, paths, thresholds, file names, and pass/fail gates are concrete. There are no unbounded vague tuning steps.
- Type consistency: The plan uses existing script arguments verified from the current code: `--train-geometry-edges`, `--eval-geometry-edges`, `--threshold-objective pfm_wrong_cap`, `--variant-threshold`, `--candidate-pairs`, and `--post-filter-profile`.
