# Graph8 Teacher Guard Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the 8-layer GraphMatcher path, but stabilize it by preserving the 4-layer checkpoint behavior, distilling shallow matcher outputs into full-depth outputs, guarding dustbin losses on bad positive batches, and freezing the extractor during the initial matcher warmup.

**Architecture:** Extend existing matcher calibration and graph matcher loss controls without changing the extractor/matcher boundary. The training loop will expose the new controls through both Python entrypoints and the reusable shell launcher, then start a new graph8 run with full-depth attention enabled.

**Tech Stack:** Python, PyTorch, unittest, existing `scripts/benchmark_lazy_pose_pairs.py` training loop.

---

### Task 1: Layer-Selective Residual Gate Initialization

**Files:**
- Modify: `python/pfm_model.py`
- Test: `python/tests/test_pfm_model.py`

- [ ] Add `matcher_attention_residual_gate_start_layer` so `set_matcher_calibration()` can initialize only layers at or after a 1-based start layer.
- [ ] Verify layers before the start layer keep their existing gate values while later layers receive the configured small gate.

### Task 2: Shallow-To-Deep Matcher Distillation

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_pfm_pytorch_training.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] Add a `graph_matcher_depth_distillation_loss()` helper comparing student full-depth logits to a detached shallow teacher.
- [ ] Add `graph_matcher_depth_distillation_weight`, `graph_matcher_depth_distillation_teacher_layers`, and temperature controls to the graph matcher loss path.
- [ ] Log distillation loss and teacher depth in CSV metrics.

### Task 3: Positive Dustbin Guard

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_pfm_pytorch_training.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] Add a guard that disables no-match and hard-negative dustbin losses for a batch when positive matches are already losing to dustbin.
- [ ] Keep positive CE and positive-vs-dustbin margin active when the guard fires.
- [ ] Log guard activity and guarded effective weights.

### Task 4: Extractor Freeze Warmup

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_pfm_pytorch_training.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] Add a helper that freezes non-GraphMatcher parameters for the first `N` steps while preserving the original trainable mask.
- [ ] Wire `--freeze-extractor-warmup-steps` through both training entrypoints.
- [ ] Log whether extractor warmup freeze is active.

### Task 5: Verify And Launch Graph8

**Files:**
- Modify: `runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh`

- [ ] Run focused tests for model, graph matcher training, script args, syntax, and diff checks.
- [ ] Launch a new 8-layer run from the best graph4 checkpoint with full-depth training, shallow teacher distillation, positive dustbin guard, new-layer gate init, and extractor freeze warmup.
- [ ] Confirm the run reaches training steps and the logged guard/dustbin metrics are present.
