# Matcher Calibration And Deep Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the graph matcher less over-conservative and train deeper attention stacks with direct supervision before running a short comparison training round.

**Architecture:** Keep the current LightGlue-like matcher shape, but add explicit gates for reliability pair bias, reliability dustbin bias, final accept score gating, and geometry bias scale. Add optional multi-depth graph supervision in the training loss by evaluating the same matcher at configured intermediate attention depths and averaging assignment-style losses, so 4/8 layer models receive useful gradients before final-layer inference.

**Tech Stack:** Python 3.12, PyTorch, argparse, unittest, existing `python/pfm_model.py`, `python/pfm_pytorch_training.py`, and `scripts/benchmark_lazy_pose_pairs.py`.

---

## File Structure

- Modify `python/pfm_model.py`
  - Add matcher runtime knobs: reliability pair bias, reliability dustbin bias, final accept score mode, geometry bias scale.
  - Persist matcher knobs in `CheckpointConfig` and restore them when loading PyTorch state.
  - Add setter methods on `PlanetaryFeatureMatcher` so training scripts can override checkpoint defaults.
- Modify `python/pfm_pytorch_training.py`
  - Add CLI args for matcher knobs.
  - Add graph matcher deep supervision depth parsing.
  - Add a helper that computes loss for the final matcher output and optional intermediate depth outputs.
  - Log `graph_matcher_deep_supervision_loss` and configured matcher calibration values.
- Modify `scripts/benchmark_lazy_pose_pairs.py`
  - Expose the same matcher knobs and deep supervision options in the current lazy training entrypoint.
  - Save the knobs in run metadata and checkpoint training metadata.
  - Log the new metrics in `train_metrics.csv`.
- Modify tests:
  - `python/tests/test_pfm_model.py`
  - `python/tests/test_pfm_pytorch_training.py`
  - `python/tests/test_benchmark_lazy_pose_pairs.py`

---

### Task 1: Matcher Runtime Knob Tests

**Files:**
- Modify: `python/tests/test_pfm_model.py`
- Modify: `python/pfm_model.py`

- [ ] **Step 1: Write failing tests**

Add tests that construct `PlanetaryGraphMatcher` with identical descriptors and metadata where reliability columns are intentionally poor. Verify that disabling reliability pair bias and dustbin bias changes logits exactly in the intended places, and verify `final_accept_score_mode="none"` leaves returned scores equal to dual softmax scores instead of multiplying by accept probability.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_model
```

Expected: failures for missing matcher constructor args or setter behavior.

- [ ] **Step 3: Implement matcher knobs**

Add constructor args and validation in `PlanetaryGraphMatcher`; add config fields and setters in `PlanetaryFeatureMatcher`; wire the knobs into `_geometry_compatibility_bias()`, `_pair_reliability_bias()` use, dustbin row/column assignment, and final accept score multiplication.

- [ ] **Step 4: Verify GREEN**

Run the same unittest command. Expected: all `python.tests.test_pfm_model` tests pass.

---

### Task 2: Deep Supervision Loss Tests

**Files:**
- Modify: `python/tests/test_pfm_pytorch_training.py`
- Modify: `python/pfm_pytorch_training.py`

- [ ] **Step 1: Write failing tests**

Add tests for `parse_graph_supervision_depths("1,2,4")`, invalid depth strings, and a mocked graph matcher that records `max_attention_layers` calls. Verify that the helper trains intermediate depths before the final full-depth output and returns `graph_matcher_deep_supervision_loss`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_pytorch_training
```

Expected: missing parser/helper failures.

- [ ] **Step 3: Implement parser and helper**

Add `parse_graph_supervision_depths()`. Add `compute_graph_matcher_training_losses()` that accepts descriptors, metadata, positive count, and existing graph loss weights. It should call the matcher for each supervision depth with no adaptive pruning, then call the final configured matcher output once, average intermediate assignment losses under `graph_matcher_deep_supervision_weight`, and return final output plus metrics.

- [ ] **Step 4: Wire into `train_step()`**

Replace the direct graph matcher call inside `train_step()` with the helper. Keep existing final losses unchanged when deep supervision is disabled.

- [ ] **Step 5: Verify GREEN**

Run the same unittest command. Expected: all training tests pass.

---

### Task 3: Lazy Training CLI And Metadata

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Modify: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Write failing CLI tests**

Extend lazy parse tests with:

```text
--matcher-reliability-pair-bias off
--matcher-reliability-dustbin-bias matchability
--matcher-final-accept-score-mode none
--matcher-geometry-bias-scale 0.25
--graph-matcher-deep-supervision-depths 1,2,4
--graph-matcher-deep-supervision-weight 0.4
```

- [ ] **Step 2: Run lazy tests and verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs
```

Expected: unrecognized-argument failures.

- [ ] **Step 3: Implement CLI and metadata**

Add args, validation, `_load_model()` setter calls, `run_train()` `train_step()` wiring, metric fields, checkpoint metadata, and input summary metadata.

- [ ] **Step 4: Verify GREEN**

Run the same unittest command. Expected: all lazy tests pass.

---

### Task 4: Verification And Short Training

**Files:**
- Modify: `runs/*.log` and `runs/<new-run>/` generated outputs only.

- [ ] **Step 1: Run relevant tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs python.tests.test_pfm_model python.tests.test_pfm_pytorch_training
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m py_compile python/pfm_model.py python/pfm_pytorch_training.py scripts/benchmark_lazy_pose_pairs.py
git diff --check
```

- [ ] **Step 2: Start short training**

Use the existing `runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh` with a new timestamped output directory, reduced step count, matcher calibration enabled, and deep supervision depths appropriate for the selected attention layer count.

- [ ] **Step 3: Monitor initial metrics**

Read the latest `train_metrics.csv`, visual report summary if produced, and log tail. Report loss, top1, filtered matches, true-match dustbin rejection, and whether the run is still active or finished.
