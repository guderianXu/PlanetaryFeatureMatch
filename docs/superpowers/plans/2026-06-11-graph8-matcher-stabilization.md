# Graph8 Matcher Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the larger 6/8-layer GraphMatcher train stably, avoid dustbin swallowing true matches, and start a new graph8 training run.

**Architecture:** Keep the existing extractor and GraphMatcher boundary. Add stabilizers inside `PlanetaryGraphMatcher`, expose training loss controls in `graph_matcher_correspondence_loss`, and wire the options through both Python training CLIs and the reusable run launcher.

**Tech Stack:** Python, PyTorch, unittest, existing `scripts/benchmark_lazy_pose_pairs.py` training loop.

---

### Task 1: Residual Gate And Geometry Clamp

**Files:**
- Modify: `python/pfm_model.py`
- Test: `python/tests/test_pfm_model.py`

- [ ] Add residual gates to `PlanetaryGraphAttentionLayer` for self, cross, and feed-forward residuals.
- [ ] Initialize gates from a configurable `matcher_attention_residual_gate_init` value, with training runs using a small value such as `0.05`.
- [ ] Clamp geometry compatibility bias with configurable `matcher_geometry_bias_clamp`, defaulting to `2.0`.
- [ ] Verify gates start near the configured value and geometry bias is bounded.

### Task 2: Candidate Top-K Assignment

**Files:**
- Modify: `python/pfm_model.py`
- Modify: `python/pfm_pytorch_training.py`
- Test: `python/tests/test_pfm_model.py`
- Test: `python/tests/test_pfm_pytorch_training.py`

- [ ] Extend the matcher candidate mask to use raw similarity plus weak geometry bias.
- [ ] Allow training to enable candidate top-k assignment while forcibly preserving the positive diagonal.
- [ ] Add metrics showing the active train candidate top-k.

### Task 3: Dustbin Schedule And Positive Margin

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_pfm_pytorch_training.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] Add a warmup/ramp schedule for no-match and hard-negative dustbin weights.
- [ ] Add `positive_vs_dustbin_margin_loss` so true matches are explicitly trained to beat dustbin.
- [ ] Log effective dustbin weights and the new margin loss.

### Task 4: Accept Head Decoupling

**Files:**
- Modify: `python/pfm_model.py`
- Modify: CLI parsers in both training entrypoints
- Test: `python/tests/test_pfm_model.py`

- [ ] Add a mode to prevent accept logits from contributing to assignment logits.
- [ ] Add an additive final score mode so accept can calibrate output scores without rewriting assignment.

### Task 5: Verify And Train

**Files:**
- Modify: `runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh`

- [ ] Run focused model/training/script tests and syntax checks.
- [ ] Launch a new 2048 graph8 run from the best graph4 state with residual gates, dustbin schedule, candidate top-k, accept decoupling, and geometry clamp enabled.
- [ ] Confirm the process reaches training steps and reports stable dustbin diagnostics.
