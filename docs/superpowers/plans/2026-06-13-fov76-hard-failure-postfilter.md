# fov76 Hard Failure And Postfilter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn fov76 failure analysis into reusable hard-pair manifests, add robust geometric postfiltering, and prepare conservative matcher-only follow-up training.

**Architecture:** Keep the current lazy-pair training and visual evaluation flow. Add a small hard-failure mining script that reads visual summaries plus an existing pair manifest and writes a compatible pair manifest with extra diagnostic columns. Extend the visual matcher postfilter with homography RANSAC/MAGSAC while keeping existing local/affine filters unchanged.

**Tech Stack:** Python 3, PyTorch, OpenCV if available, existing `benchmark_lazy_pose_pairs.py` pair manifest schema, `unittest`.

---

### Task 1: Hard Failure Mining

**Files:**
- Create: `scripts/mine_hard_failure_pairs.py`
- Modify: `python/tests/test_stress_eval_scripts.py`
- Modify: `scripts/README.md`

- [x] Write a failing test that calls `mine_hard_failure_rows()` with visual summary rows and pair manifest rows.
- [ ] Implement `HardFailureConfig`, CSV readers, row matching by split/base/variant, reason classification, deduplication, and CSV output.
- [ ] Verify the generated hard manifest keeps all pair manifest columns required by `read_pair_spec_manifest()`.

### Task 2: Robust Geometry Postfilter

**Files:**
- Modify: `scripts/visualize_lazy_pose_matches.py`
- Modify: `python/tests/test_stress_eval_scripts.py`

- [x] Write failing tests for `--geometry-filter magsac`, `--filtered-geometry-filter ransac`, and homography outlier rejection.
- [ ] Add `ransac` and `magsac` filter modes to `filter_visual_matches()`.
- [ ] Use `cv2.findHomography()` with `cv2.USAC_MAGSAC` when available, otherwise fall back to `cv2.RANSAC`.

### Task 3: fov76 Run Scripts

**Files:**
- Create: `runs/mine_h100_fov076_hard_failures_20260613.sh`
- Create: `runs/train_h100_fov076_matcher_only_hard_20260613.sh`
- Create: `runs/sweep_h100_fov076_candidate_topk_20260613.sh`

- [ ] Generate hard failure manifests from current fov76 reports.
- [ ] Prepare conservative matcher-only training from the stable init checkpoint, not from the failed falsemine checkpoint.
- [ ] Prepare candidate_topk 128/256 ablation using the stable checkpoint and local10/MAGSAC reports.

### Task 4: Verification

**Files:**
- Modify: `scripts/README.md`

- [ ] Run targeted unit tests.
- [ ] Run `py_compile` on changed scripts.
- [ ] Run the hard mining script once on existing fov76 reports and report output counts.
