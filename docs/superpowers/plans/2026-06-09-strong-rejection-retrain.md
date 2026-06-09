# Strong Rejection Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen GraphMatcher rejection/keypoint training and start a new lazy-pair training run from the latest h100/fov090 overlap pair list.

**Architecture:** Keep the backbone and GraphMatcher structure unchanged for this round and make the training objective match inference better. The shared training step adds direct synthetic heatmap supervision for learned keypoints, while the lazy training preset emphasizes semi-dense no-match samples, hard-negative dustbin supervision, and rotation consistency. The launcher then trains from the latest reusable overlap CSV snapshot and auto-runs learned-keypoint visual evaluation.

**Tech Stack:** Python 3.12, PyTorch, existing `scripts/benchmark_lazy_pose_pairs.py`, `python/pfm_pytorch_training.py`, and local `runs/` shell launchers.

---

### Task 1: Lock Strong Rejection Defaults With Tests

**Files:**
- Modify: `python/tests/test_benchmark_lazy_pose_pairs.py`
- Modify: `python/tests/test_stress_eval_scripts.py`

- [ ] **Step 1: Add RED tests**

Add assertions that `--enable-rejection-training` sets nonzero semi-dense no-match, higher no-match/dustbin/accept weights, learned visual keypoints, and rotation consistency.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs.BenchmarkLazyPosePairsTest.test_parse_args_accepts_rejection_and_hard_negative_options
```

Expected: fail because current defaults are weaker and visual keypoint mode remains texture.

### Task 2: Implement Strong Rejection Preset

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Modify: `python/pfm_pytorch_training.py`

- [ ] **Step 1: Update defaults**

Set stronger no-match, accept, hard-negative dustbin, false-match, semi-dense no-match, reliability, and rotation consistency defaults when rejection training is enabled.

- [ ] **Step 2: Verify GREEN**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs python.tests.test_stress_eval_scripts
```

Expected: all tests pass.

### Task 3: Add Synthetic Keypoint Supervision

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_pfm_pytorch_training.py`

- [ ] **Step 1: Add RED test**

Add `test_train_step_allows_synthetic_keypoint_only_updates` to verify that `keypoint_weight` can update heatmap parameters without descriptor loss.

- [ ] **Step 2: Implement shared train-step support**

Add `keypoint_weight` and `keypoint_negative_weight`, apply `heatmap_point_loss` on sampled positive correspondences in both views, and report `keypoint_loss` / `keypoint_points`.

- [ ] **Step 3: Verify GREEN**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_pytorch_training.PFMPyTorchTrainingTest.test_train_step_allows_synthetic_keypoint_only_updates
```

Expected: test passes and the mocked parameter is updated by the weighted keypoint loss.

### Task 4: Update Local Launcher

**Files:**
- Modify local ignored file: `runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh`

- [ ] **Step 1: Set explicit training knobs**

Add explicit strong rejection values, learned keypoint visual evaluation, and a run stamp describing learned-keypoint/strong-rejection training.

- [ ] **Step 2: Syntax check**

Run:

```bash
bash -n runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh
```

Expected: no output and exit code 0.

### Task 5: Snapshot Latest Pairs And Start Training

**Files:**
- Create run artifacts under `/media/w24/D/xjw深度学习训练数据/pfm_runs/`
- Create snapshot CSV under `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260608_223338_h100_fov090_mid2_extreme3_overlap_train/`

- [ ] **Step 1: Copy current `overlap_edges.csv` to a timestamped snapshot**

Use `cp` to create a stable CSV so the long-running overlap generator can continue appending without changing this training run.

- [ ] **Step 2: Launch training in background**

Use `setsid` with `PFM_PAIR_SPEC_MANIFEST=<snapshot>` and log stdout/stderr to `runs/`.

- [ ] **Step 3: Verify process startup**

Run:

```bash
pgrep -af 'benchmark_lazy_pose_pairs.py.*mode train' || true
```

Expected: one training process with the new run stamp.

### Task 6: Commit And Push

**Files:**
- Commit tracked code and test files only.

- [ ] **Step 1: Run verification**

Run unit tests, `compileall`, `git diff --check`, and `git status --short`.

- [ ] **Step 2: Commit and push**

Commit the tracked training preset changes and push `main`.
