# Pose Metadata Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach simulation camera metadata to training pairs and use it for training-only sampling and loss weighting.

**Architecture:** Add a focused Python module for manifest/TSAI parsing and pair metadata indexing. Extend `pfm_pytorch_training.py` with optional CLI flags and training-loop hooks while preserving existing defaults.

**Tech Stack:** Python 3, PyTorch, `unittest`, CSV/TSAI text parsing.

---

### Task 1: Metadata Loader

**Files:**
- Create: `python/pose_pair_metadata.py`
- Test: `python/test_pose_pair_metadata.py`

- [ ] Write tests for TSAI parsing, manifest indexing, and difficulty scoring.
- [ ] Implement dataclasses for camera and pair metadata.
- [ ] Implement `load_pose_metadata_index()`, `lookup_pose_metadata()`, and root inference helpers.
- [ ] Run focused metadata tests.

### Task 2: Training Integration

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Test: `python/test_pfm_pytorch_training.py`

- [ ] Add CLI options for pose metadata root, pose-balanced sampling, minimum overlap, and difficulty loss weight.
- [ ] Add pose-balanced sampling helper and optional loss weighting.
- [ ] Extend metrics CSV with pose bucket counts and average pose loss weight.
- [ ] Run focused training tests.

### Task 3: Smoke Verification

**Files:**
- No production files.

- [ ] Run full Python test discovery.
- [ ] Run one small training smoke test on currently generated simulation cache.
- [ ] Record output path and metrics.
