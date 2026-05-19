# Training Diagnostics Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add async training diagnostics PNG output for selected synthetic training pairs.

**Architecture:** CLI parses `--visualization-dir` and `--visualization-samples`; `TrainConfig` carries a sample selector. Training enqueues CPU diagnostic jobs into a bounded `AsyncVisualizationWriter`, which writes synthetic views, masks, warp correspondences, feature overlays, and model match overlays.

**Tech Stack:** C++17, CLI11, LibTorch, OpenCV, CMake, custom `pfm_tests` harness.

---

### Task 1: CLI options

**Files:**
- Modify: `modules/cli/commands.h`
- Modify: `modules/cli/commands.cpp`
- Modify: `modules/cli/commands_test.cpp`

- [ ] Add failing parser tests for default sample count, explicit numeric sample count, `all`, and invalid values.
- [ ] Add `visualization_samples` and `visualization_samples_all` fields to `CliOptions`.
- [ ] Bind train options `--visualization-dir` and `--visualization-samples` with validation.
- [ ] Run `./build/pfm_tests --filter commands` or full `./build/pfm_tests` if filtering is unsupported.

### Task 2: Training config plumbing

**Files:**
- Modify: `modules/train/trainer.h`
- Modify: `modules/infer/pipeline.cpp`
- Modify: `modules/infer/pipeline_test.cpp`

- [ ] Add failing test proving train command forwards visualization options into `TrainConfig` or training output behavior.
- [ ] Add `visualization_dir`, `visualization_sample_count`, and `visualization_samples_all` to `TrainConfig`.
- [ ] Copy CLI fields in `run_train_command()`.
- [ ] Print `training visualization: dir=<dir> samples=<N|all> async_queue=256` when enabled.

### Task 3: Async writer and image diagnostics

**Files:**
- Create: `modules/train/training_visualization.h`
- Create: `modules/train/training_visualization.cpp`
- Create: `modules/train/training_visualization_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] Add failing async writer flush test that enqueues several PNG jobs and verifies all files exist after join.
- [ ] Implement `AsyncVisualizationWriter` with one worker thread, bounded queue capacity 256, flush on join, and rethrow writer exceptions.
- [ ] Add failing overlay test that verifies non-background text pixels appear in the upper-left image region.
- [ ] Implement helpers for grayscale view, valid mask, feature overlay, warp match overlay, and model match overlay PNGs with text counts.

### Task 4: Training loop integration

**Files:**
- Modify: `modules/train/trainer.cpp`
- Modify: `modules/train/trainer_test.cpp`

- [ ] Add failing training test with visualization enabled that expects the seven PNGs for selected pairs.
- [ ] Add failing `all` test on a tiny dataset that expects every pair to be written.
- [ ] During training, select first 4 pairs by default or all pairs when requested.
- [ ] Run the current model on selected pairs, match features with `matchFeatureSets`, and enqueue diagnostic outputs.
- [ ] Ensure writer joins on success and exception before `train_model` returns.

### Task 5: Docs and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/training.md` if present
- Modify: `docs/usage.md` if present

- [ ] Document sampled and full training diagnostics examples.
- [ ] Run `cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)`.
- [ ] Run `./build/pfm_tests`.
