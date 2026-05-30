# Abstention-Aware Descriptor Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable descriptor false-match suppression loss and run a short conservative training probe.

**Architecture:** Keep the model unchanged and add a loss-only extension in `python/pfm_pytorch_training.py`. `python/cross_view_experiment.py` only passes CLI options through to the training subprocess.

**Tech Stack:** Python, PyTorch, unittest, existing PFM training/evaluation scripts.

---

### Task 1: Add Loss Unit Tests

**Files:**
- Modify: `python/test_pfm_pytorch_training.py`

- [x] Add tests that show a high-similarity false B candidate is penalized.
- [x] Add tests that show candidates near the true target are masked.
- [x] Add tests that show `descriptor_map_pair_loss()` increases when `abstention_weight` is enabled.
- [x] Run `PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_pfm_pytorch_training.py -k abstention` and confirm RED before implementation.

### Task 2: Implement Loss and Training CLI

**Files:**
- Modify: `python/pfm_pytorch_training.py`

- [x] Implement `descriptor_false_match_suppression_loss()`.
- [x] Add abstention parameters to `descriptor_map_pair_loss()`.
- [x] Thread parameters through `train_step()`.
- [x] Add parser validation and metrics CSV columns.
- [x] Run focused tests and confirm GREEN.

### Task 3: Add Orchestrator Pass-Through

**Files:**
- Modify: `python/test_cross_view_experiment.py`
- Modify: `python/cross_view_experiment.py`

- [x] Add a failing test that `build_training_command()` passes abstention flags.
- [x] Add function arguments, CLI args, and subprocess pass-throughs.
- [x] Run focused tests and compile checks.

### Task 4: Short Probe

**Files:**
- Output under `runs/`

- [x] Run a short P1 viewpoint probe with `synthetic_loss_weight=0`, strict pseudo labels, conservative abstention weight, and small batch size.
- [x] Run a narrow sparse evaluation using the existing selected parameters, not broad calibration.
- [x] Record whether false-match activation improves before deciding on another longer run.

**Probe result:** `runs/cross_view_1024_abstention_p1_viewpoint_desc_w025_m035_lr3e7_b4_80_seed1234` completed, but same-split raw evaluation versus the base checkpoint showed no useful improvement. Descriptor retrieval moved slightly positive, but sparse raw matching was effectively unchanged or slightly worse on the weak groups.
