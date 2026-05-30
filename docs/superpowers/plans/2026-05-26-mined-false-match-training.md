# Mined False-Match Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine current PFM false matches and train a cyclic negative loss on those exact wrong pairs.

**Architecture:** Add one focused mining script and extend the existing PyTorch training loop with a CSV-backed false-match loss. Keep the model architecture and inference code unchanged.

**Tech Stack:** Python, PyTorch, unittest, existing PFM cache/evaluation utilities.

---

### Task 1: Add False-Match Loss Tests

**Files:**
- Modify: `python/test_pfm_pytorch_training.py`

- [x] Add a test that cyclic paired similarity recognizes quarter-channel shifts.
- [x] Add a test that false-match negative loss penalizes a cyclic high-similarity wrong pair.
- [x] Add tests for reading/scaling false-match CSV rows.
- [x] Add a train-step test showing false-match loss is weighted and counted.

### Task 2: Implement Training Loss and CLI

**Files:**
- Modify: `python/pfm_pytorch_training.py`

- [x] Add false-match label dataclass and CSV reader.
- [x] Add feature-grid scaling helper for false-match labels.
- [x] Add `paired_cyclic_similarity()` and `false_match_negative_loss()`.
- [x] Thread false-match labels through `train_step()`.
- [x] Add CLI validation, metrics fields, and supervised pair union handling.

### Task 3: Add Cross-View Pass-Through

**Files:**
- Modify: `python/test_cross_view_experiment.py`
- Modify: `python/cross_view_experiment.py`

- [x] Add a failing command-construction test for false-match flags.
- [x] Add `build_training_command()` parameters and parse/pass-through logic.

### Task 4: Add Miner

**Files:**
- Create: `scripts/mine_pfm_false_matches.py`
- Create: `python/test_pfm_false_match_mining.py`

- [x] Add tests for turning match tensors plus warp truth into false-match CSV rows.
- [x] Implement a raw evaluator-aligned miner that writes wrong accepted matches.
- [x] Compile and run a smoke mine on a small train-split sample.

### Task 5: Probe

**Files:**
- Output under `runs/`

- [x] Mine false matches from weak train groups.
- [x] Run an 80-step P1-positive plus false-negative probe.
- [x] Run same-split raw base/trained evaluation and record the decision.
- [x] Run a second style/gate-specific positive-label plus mined-false probe and record the decision.
