# Keypoint Distribution and CLI Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first optimization round from `docs/superpowers/specs/2026-05-16-model-distribution-evaluation-design.md`: spatially balanced sparse keypoint decoding and CLI timing output.

**Architecture:** Keep model and saved `.pt` feature formats unchanged. Add a decode configuration that is parsed by CLI and passed through extract/match/eval so all inference paths share the same intensity mask, NMS, grid top-k, and global backfill behavior. Add a small `core/timer` utility and use it at command boundaries plus training epoch/batch boundaries without adding extra GPU synchronization beyond the existing loss `.item()` calls.

**Tech Stack:** C++17, LibTorch C++ API, OpenCV, CLI11, CMake, existing `pfm_tests` harness.

---

## File Structure

- `modules/infer/feature_extractor.h/.cpp`: define `FeatureDecodeConfig`; validate decode parameters; implement local NMS, grid top-k, and global backfill in sparse decode.
- `modules/infer/feature_extractor_test.cpp`: focused tests for NMS suppression, per-cell limits, backfill, and invalid config validation.
- `modules/cli/commands.h/.cpp`: add CLI options `--keypoint-grid-rows`, `--keypoint-grid-cols`, `--keypoints-per-cell`, `--nms-radius`; keep defaults aligned with the spec.
- `modules/cli/commands_test.cpp`: parse and help tests for new decode options.
- `modules/infer/pipeline.cpp`: pass `FeatureDecodeConfig` into extraction/match/eval; add extraction/match/eval/export timing output.
- `modules/infer/pipeline_test.cpp`: capture command stdout and verify timing fields and new decode options are honored.
- `modules/core/timer.h/.cpp/.test.cpp`: small timer and formatter utility.
- `modules/train/trainer.h/.cpp`: add total and average batch timing to `TrainResult`; print epoch timing during training.
- `modules/train/trainer_test.cpp`: verify training returns timing fields and prints epoch timing.
- `CMakeLists.txt`, `tests/test_main.cpp`: register new timer source/test.
- `README.md`, `docs/usage.md`, `docs/training.md`: document new decode options and timing output.

Before implementation, note the working tree already contains an independent, verified visualization coordinate-scaling fix in `modules/infer/pipeline.cpp`, `modules/infer/visualization.cpp`, `modules/infer/visualization.h`, and `modules/infer/visualization_test.cpp`. Commit or preserve that fix before starting this plan so later pipeline edits do not accidentally discard it.

---

### Task 1: Add decode configuration API and CLI parsing

**Files:**
- Modify: `modules/infer/feature_extractor.h`
- Modify: `modules/cli/commands.h`
- Modify: `modules/cli/commands.cpp`
- Modify: `modules/cli/commands_test.cpp`

- [ ] **Step 1: Write failing CLI parse tests**

Add these tests near the existing parse tests in `modules/cli/commands_test.cpp`:

```cpp
static void parse_extract_keypoint_distribution_options() {
    const auto options = pfm::parse_cli({
        "pfm",
        "extract",
        "--image",
        "a.png",
        "--checkpoint",
        "model.pt",
        "--output",
        "features.pt",
        "--keypoint-grid-rows",
        "4",
        "--keypoint-grid-cols",
        "6",
        "--keypoints-per-cell",
        "3",
        "--nms-radius",
        "2"});

    PFM_REQUIRE(options.keypoint_grid_rows == 4);
    PFM_REQUIRE(options.keypoint_grid_cols == 6);
    PFM_REQUIRE(options.keypoints_per_cell == 3);
    PFM_REQUIRE(options.nms_radius == 2);
}

static void parse_invalid_keypoint_distribution_options_throw() {
    PFM_REQUIRE_THROWS_AS(
        pfm::parse_cli({
            "pfm",
            "extract",
            "--image",
            "a.png",
            "--checkpoint",
            "model.pt",
            "--output",
            "features.pt",
            "--keypoint-grid-rows",
            "0"}),
        CLI::ParseError);
    PFM_REQUIRE_THROWS_AS(
        pfm::parse_cli({
            "pfm",
            "match",
            "--image-a",
            "a.png",
            "--image-b",
            "b.png",
            "--checkpoint",
            "model.pt",
            "--output",
            "matches.pt",
            "--nms-radius",
            "-1"}),
        CLI::ParseError);
}
```

Register both in `register_cli_tests()`:

```cpp
register_test("parse_extract_keypoint_distribution_options", parse_extract_keypoint_distribution_options);
register_test("parse_invalid_keypoint_distribution_options_throw", parse_invalid_keypoint_distribution_options_throw);
```

Also extend `top_level_help_lists_subcommand_options()`:

```cpp
PFM_REQUIRE(help.find("--keypoint-grid-rows") != std::string::npos);
PFM_REQUIRE(help.find("--keypoint-grid-cols") != std::string::npos);
PFM_REQUIRE(help.find("--keypoints-per-cell") != std::string::npos);
PFM_REQUIRE(help.find("--nms-radius") != std::string::npos);
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: compile failure or test failure because `CliOptions` does not yet contain `keypoint_grid_rows`, `keypoint_grid_cols`, `keypoints_per_cell`, or `nms_radius`.

- [ ] **Step 3: Add config fields and parser options**

In `modules/infer/feature_extractor.h`, insert after `RawFeatureMaps`:

```cpp
struct FeatureDecodeConfig {
    int max_keypoints = 1024;
    double semi_dense_threshold = 0.5;
    int keypoint_grid_rows = 8;
    int keypoint_grid_cols = 8;
    int keypoints_per_cell = 0;
    int nms_radius = 4;
};
```

Add header documentation for each field using short `///` comments to satisfy project header rules.

In `modules/cli/commands.h`, add fields to `CliOptions` after `min_keypoint_intensity`:

```cpp
int keypoint_grid_rows = 8;
int keypoint_grid_cols = 8;
int keypoints_per_cell = 0;
int nms_radius = 4;
```

In `modules/cli/commands.cpp`, extend the footer for extract/match/eval:

```cpp
"[--min-keypoint-intensity 0.0] [--keypoint-grid-rows 8] [--keypoint-grid-cols 8] "
"[--keypoints-per-cell 0] [--nms-radius 4]\n"
```

Add the four options to extract, match, and eval subcommands:

```cpp
extract->add_option("--keypoint-grid-rows", options.keypoint_grid_rows, "Sparse keypoint grid rows")
    ->check(CLI::PositiveNumber);
extract->add_option("--keypoint-grid-cols", options.keypoint_grid_cols, "Sparse keypoint grid columns")
    ->check(CLI::PositiveNumber);
extract->add_option("--keypoints-per-cell",
                    options.keypoints_per_cell,
                    "Sparse keypoints per grid cell; 0 derives from max-keypoints")
    ->check(CLI::NonNegativeNumber);
extract->add_option("--nms-radius", options.nms_radius, "Sparse keypoint NMS radius in feature-map pixels")
    ->check(CLI::NonNegativeNumber);
```

Repeat the same four `add_option` blocks for `match` and `eval`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all current tests pass, including the new CLI parse tests.

- [ ] **Step 5: Commit**

```bash
git add modules/infer/feature_extractor.h modules/cli/commands.h modules/cli/commands.cpp modules/cli/commands_test.cpp
git commit -m "$(cat <<'EOF'
Add keypoint distribution CLI options.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Implement local NMS for sparse keypoint decode

**Files:**
- Modify: `modules/infer/feature_extractor.h`
- Modify: `modules/infer/feature_extractor.cpp`
- Modify: `modules/infer/feature_extractor_test.cpp`

- [ ] **Step 1: Write failing NMS test**

Add to `modules/infer/feature_extractor_test.cpp`:

```cpp
static void decode_feature_maps_suppresses_neighbors_with_nms_radius() {
    auto heatmap = torch::zeros({1, 1, 4, 4}, torch::kFloat32);
    heatmap.index_put_({0, 0, 1, 1}, 10.0F);
    heatmap.index_put_({0, 0, 1, 2}, 9.0F);
    heatmap.index_put_({0, 0, 3, 3}, 8.0F);
    auto maps = makeMaps(heatmap, torch::ones({1, 1, 4, 4}, torch::kFloat32));
    maps.descriptors = torch::ones({1, 4, 4, 4}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 4, 4}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 2, 4, 4}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 4, 4}, torch::kFloat32);
    pfm::FeatureDecodeConfig config;
    config.max_keypoints = 3;
    config.semi_dense_threshold = 0.5;
    config.keypoint_grid_rows = 1;
    config.keypoint_grid_cols = 1;
    config.keypoints_per_cell = 3;
    config.nms_radius = 1;

    const auto features = pfm::decode_feature_maps(maps, config, torch::Tensor());

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 0}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 1}).item<float>(), 3.0F, 1.0e-6F);
}
```

Register:

```cpp
register_test("decode_feature_maps_suppresses_neighbors_with_nms_radius",
              decode_feature_maps_suppresses_neighbors_with_nms_radius);
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: compile failure because `decode_feature_maps(maps, config, mask)` overload does not exist, or assertion failure if overload exists without NMS.

- [ ] **Step 3: Add decode overload and NMS candidate selection**

In `modules/infer/feature_extractor.h`, add overload:

```cpp
FeatureSet decode_feature_maps(
    const RawFeatureMaps& maps,
    const FeatureDecodeConfig& config,
    const torch::Tensor& intensity_mask
);
```

In `modules/infer/feature_extractor.cpp`, add helpers in the anonymous namespace:

```cpp
void validate_decode_config(const FeatureDecodeConfig& config) {
    if (config.max_keypoints <= 0) {
        throw std::invalid_argument("max_keypoints must be positive");
    }
    if (config.keypoint_grid_rows <= 0) {
        throw std::invalid_argument("keypoint_grid_rows must be positive");
    }
    if (config.keypoint_grid_cols <= 0) {
        throw std::invalid_argument("keypoint_grid_cols must be positive");
    }
    if (config.keypoints_per_cell < 0) {
        throw std::invalid_argument("keypoints_per_cell must be non-negative");
    }
    if (config.nms_radius < 0) {
        throw std::invalid_argument("nms_radius must be non-negative");
    }
}

struct SparseCandidate {
    int64_t y = 0;
    int64_t x = 0;
    float score = 0.0F;
};

bool is_suppressed_by_selected(const std::vector<SparseCandidate>& selected, const SparseCandidate& candidate, int radius) {
    for (const auto& point : selected) {
        if (std::abs(point.y - candidate.y) <= radius && std::abs(point.x - candidate.x) <= radius) {
            return true;
        }
    }
    return false;
}

std::vector<SparseCandidate> make_nms_candidates(
    const torch::Tensor& heatmap,
    const torch::Tensor& valid_mask,
    int nms_radius
) {
    std::vector<SparseCandidate> candidates;
    for (int64_t y = 0; y < heatmap.size(2); ++y) {
        for (int64_t x = 0; x < heatmap.size(3); ++x) {
            if (valid_mask.index({y, x}).item<bool>()) {
                candidates.push_back(SparseCandidate{y, x, heatmap.index({0, 0, y, x}).item<float>()});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const SparseCandidate& lhs, const SparseCandidate& rhs) {
        if (lhs.score == rhs.score) {
            if (lhs.y == rhs.y) {
                return lhs.x < rhs.x;
            }
            return lhs.y < rhs.y;
        }
        return lhs.score > rhs.score;
    });

    std::vector<SparseCandidate> selected;
    selected.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (!is_suppressed_by_selected(selected, candidate, nms_radius)) {
            selected.push_back(candidate);
        }
    }
    return selected;
}
```

Refactor existing overloads so they create `FeatureDecodeConfig` and call the new overload:

```cpp
FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold) {
    FeatureDecodeConfig config;
    config.max_keypoints = max_keypoints;
    config.semi_dense_threshold = semi_dense_threshold;
    return decode_feature_maps(maps, config, torch::Tensor());
}

FeatureSet decode_feature_maps(
    const RawFeatureMaps& maps,
    int max_keypoints,
    double semi_dense_threshold,
    const torch::Tensor& intensity_mask
) {
    FeatureDecodeConfig config;
    config.max_keypoints = max_keypoints;
    config.semi_dense_threshold = semi_dense_threshold;
    return decode_feature_maps(maps, config, intensity_mask);
}
```

In the new config overload, replace the `topk` sparse candidate block with a `make_nms_candidates()` call and use the returned candidate list for sparse tensors. Keep dense decode unchanged. Use `selected_count = std::min<int64_t>(config.max_keypoints, candidates.size())` and set scores from `candidate.score`.

- [ ] **Step 4: Run tests to verify NMS passes**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all feature extractor tests pass, including neighbor suppression.

- [ ] **Step 5: Commit**

```bash
git add modules/infer/feature_extractor.h modules/infer/feature_extractor.cpp modules/infer/feature_extractor_test.cpp
git commit -m "$(cat <<'EOF'
Add NMS for sparse keypoint decoding.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add grid top-k and global backfill to sparse decode

**Files:**
- Modify: `modules/infer/feature_extractor.cpp`
- Modify: `modules/infer/feature_extractor_test.cpp`

- [ ] **Step 1: Write failing grid distribution test**

Add to `modules/infer/feature_extractor_test.cpp`:

```cpp
static void decode_feature_maps_limits_sparse_keypoints_per_grid_cell_then_backfills() {
    auto heatmap = torch::zeros({1, 1, 4, 4}, torch::kFloat32);
    heatmap.index_put_({0, 0, 0, 0}, 10.0F);
    heatmap.index_put_({0, 0, 0, 1}, 9.0F);
    heatmap.index_put_({0, 0, 1, 0}, 8.0F);
    heatmap.index_put_({0, 0, 3, 3}, 7.0F);
    auto maps = makeMaps(heatmap, torch::ones({1, 1, 4, 4}, torch::kFloat32));
    maps.descriptors = torch::ones({1, 4, 4, 4}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 4, 4}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 2, 4, 4}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 4, 4}, torch::kFloat32);
    pfm::FeatureDecodeConfig config;
    config.max_keypoints = 3;
    config.semi_dense_threshold = 0.5;
    config.keypoint_grid_rows = 2;
    config.keypoint_grid_cols = 2;
    config.keypoints_per_cell = 1;
    config.nms_radius = 0;

    const auto features = pfm::decode_feature_maps(maps, config, torch::Tensor());

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({3, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 0}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 1}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({2, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({2, 1}).item<float>(), 0.0F, 1.0e-6F);
}
```

This expects one keypoint from the high-response top-left cell, one from the bottom-right cell, then global backfill from the remaining top-left candidates.

Register:

```cpp
register_test("decode_feature_maps_limits_sparse_keypoints_per_grid_cell_then_backfills",
              decode_feature_maps_limits_sparse_keypoints_per_grid_cell_then_backfills);
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: the new test fails because current NMS-only selection returns the top three global points from the top-left cluster before the bottom-right point.

- [ ] **Step 3: Implement grid selection and backfill**

In `modules/infer/feature_extractor.cpp`, add helpers:

```cpp
int resolved_keypoints_per_cell(const FeatureDecodeConfig& config) {
    if (config.keypoints_per_cell > 0) {
        return config.keypoints_per_cell;
    }
    const int cell_count = config.keypoint_grid_rows * config.keypoint_grid_cols;
    return std::max(1, (config.max_keypoints + cell_count - 1) / cell_count);
}

int64_t cell_start(int64_t extent, int cell_index, int cell_count) {
    return extent * cell_index / cell_count;
}

bool candidate_in_cell(
    const SparseCandidate& candidate,
    int row,
    int col,
    const FeatureDecodeConfig& config,
    int64_t height,
    int64_t width
) {
    const auto y0 = cell_start(height, row, config.keypoint_grid_rows);
    const auto y1 = cell_start(height, row + 1, config.keypoint_grid_rows);
    const auto x0 = cell_start(width, col, config.keypoint_grid_cols);
    const auto x1 = cell_start(width, col + 1, config.keypoint_grid_cols);
    return candidate.y >= y0 && candidate.y < y1 && candidate.x >= x0 && candidate.x < x1;
}

bool same_candidate(const SparseCandidate& lhs, const SparseCandidate& rhs) {
    return lhs.y == rhs.y && lhs.x == rhs.x;
}

bool contains_candidate(const std::vector<SparseCandidate>& candidates, const SparseCandidate& candidate) {
    return std::any_of(candidates.begin(), candidates.end(), [&](const SparseCandidate& selected) {
        return same_candidate(selected, candidate);
    });
}

std::vector<SparseCandidate> select_grid_balanced_candidates(
    const std::vector<SparseCandidate>& candidates,
    const FeatureDecodeConfig& config,
    int64_t height,
    int64_t width
) {
    std::vector<SparseCandidate> selected;
    selected.reserve(static_cast<size_t>(std::min<int64_t>(config.max_keypoints, candidates.size())));
    const int per_cell = resolved_keypoints_per_cell(config);
    for (int row = 0; row < config.keypoint_grid_rows; ++row) {
        for (int col = 0; col < config.keypoint_grid_cols; ++col) {
            int taken = 0;
            for (const auto& candidate : candidates) {
                if (taken >= per_cell || static_cast<int>(selected.size()) >= config.max_keypoints) {
                    break;
                }
                if (candidate_in_cell(candidate, row, col, config, height, width)) {
                    selected.push_back(candidate);
                    ++taken;
                }
            }
        }
    }
    for (const auto& candidate : candidates) {
        if (static_cast<int>(selected.size()) >= config.max_keypoints) {
            break;
        }
        if (!contains_candidate(selected, candidate)) {
            selected.push_back(candidate);
        }
    }
    return selected;
}
```

In the config overload, change sparse selection to:

```cpp
const auto nms_candidates = make_nms_candidates(heatmap, valid_mask, config.nms_radius);
const auto selected_candidates = select_grid_balanced_candidates(nms_candidates, config, height, width);
const int64_t sparse_count = static_cast<int64_t>(selected_candidates.size());
```

Iterate over `selected_candidates` instead of top-k indices when filling `sparse_points`, descriptors, scale, orientation, affine, and scores.

- [ ] **Step 4: Run tests to verify distribution behavior**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all tests pass, including NMS and grid/backfill tests.

- [ ] **Step 5: Commit**

```bash
git add modules/infer/feature_extractor.cpp modules/infer/feature_extractor_test.cpp
git commit -m "$(cat <<'EOF'
Balance sparse keypoint decoding across grid cells.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Pass decode configuration through extract, match, and eval

**Files:**
- Modify: `modules/infer/pipeline.cpp`
- Modify: `modules/infer/pipeline_test.cpp`

- [ ] **Step 1: Write failing pipeline test for decode config propagation**

Add to `modules/infer/pipeline_test.cpp` near extract tests:

```cpp
static void pipeline_extract_uses_keypoint_distribution_options() {
    TempPipelineDirectory temp_dir("pfm_pipeline_decode_distribution");
    write_training_images(temp_dir.image_dir());
    const auto checkpoint = temp_dir.file("model.pt");
    auto train_options = make_train_options(temp_dir);
    train_options.checkpoint = checkpoint.string();
    PFM_REQUIRE(pfm::run_train_command(train_options) == 0);

    pfm::CliOptions options;
    options.command = pfm::Command::Extract;
    options.image = (temp_dir.image_dir() / "image_0.png").string();
    options.checkpoint = checkpoint.string();
    options.output = temp_dir.file("features.pt").string();
    options.max_keypoints = 4;
    options.keypoint_grid_rows = 2;
    options.keypoint_grid_cols = 2;
    options.keypoints_per_cell = 1;
    options.nms_radius = 0;

    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto features = pfm::load_feature_set(options.output);

    PFM_REQUIRE(features.keypoints.size(0) <= 4);
}
```

This test compiles only after pipeline knows the new `CliOptions` fields and constructs `FeatureDecodeConfig`.

Register:

```cpp
register_test("pipeline_extract_uses_keypoint_distribution_options", pipeline_extract_uses_keypoint_distribution_options);
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: compile failure because pipeline does not yet include or pass `FeatureDecodeConfig`, or test failure if options are ignored.

- [ ] **Step 3: Update pipeline extraction API**

In `modules/infer/pipeline.cpp`, add helper in anonymous namespace:

```cpp
FeatureDecodeConfig make_feature_decode_config(const CliOptions& options) {
    FeatureDecodeConfig config;
    config.max_keypoints = options.max_keypoints;
    config.semi_dense_threshold = options.semi_dense_threshold;
    config.keypoint_grid_rows = options.keypoint_grid_rows;
    config.keypoint_grid_cols = options.keypoint_grid_cols;
    config.keypoints_per_cell = options.keypoints_per_cell;
    config.nms_radius = options.nms_radius;
    return config;
}
```

Change `extract_feature_set` signature to accept `const FeatureDecodeConfig& decode_config` instead of `int max_keypoints, double semi_dense_threshold`:

```cpp
ExtractedFeatureSet extract_feature_set(
    const std::string& image_path,
    InferenceModules& modules,
    const CheckpointConfig& checkpoint_config,
    torch::Device device,
    const FeatureDecodeConfig& decode_config,
    double min_keypoint_intensity
)
```

Inside it, call:

```cpp
decode_feature_maps(maps, decode_config, intensity_mask)
```

Update `run_extract_command`, `run_match_command`, and `run_eval_command` to create:

```cpp
const auto decode_config = make_feature_decode_config(options);
```

Pass `decode_config` to every `extract_feature_set()` call.

- [ ] **Step 4: Run tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add modules/infer/pipeline.cpp modules/infer/pipeline_test.cpp
git commit -m "$(cat <<'EOF'
Use keypoint distribution config in inference commands.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add timer utility module

**Files:**
- Create: `modules/core/timer.h`
- Create: `modules/core/timer.cpp`
- Create: `modules/core/timer_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Write failing timer test**

Create `modules/core/timer_test.cpp`:

```cpp
#include <string>

#include "core/timer.h"
#include "tests/test_harness.h"

static void timer_formats_seconds_with_three_decimals() {
    PFM_REQUIRE(pfm::format_seconds(1.23456) == "1.235");
    PFM_REQUIRE(pfm::format_seconds(0.0) == "0.000");
}

static void timer_elapsed_seconds_is_non_negative() {
    pfm::Timer timer;

    PFM_REQUIRE(timer.elapsed_seconds() >= 0.0);
}

void register_timer_tests() {
    register_test("timer_formats_seconds_with_three_decimals", timer_formats_seconds_with_three_decimals);
    register_test("timer_elapsed_seconds_is_non_negative", timer_elapsed_seconds_is_non_negative);
}
```

In `tests/test_main.cpp`, add declaration and registration:

```cpp
void register_timer_tests();
```

```cpp
register_timer_tests();
```

In `CMakeLists.txt`, add `modules/core/timer.cpp` to `pfm` sources and `modules/core/timer_test.cpp` to `pfm_tests` sources.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: compile failure because `core/timer.h` and `core/timer.cpp` do not exist.

- [ ] **Step 3: Implement timer utility**

Create `modules/core/timer.h`:

```cpp
#pragma once

#include <chrono>
#include <string>

namespace pfm {

class Timer {
public:
    /// Creates a timer starting at construction time.
    Timer();

    /// Resets the timer start time to now.
    void reset();

    /// Returns elapsed wall-clock seconds since construction or reset.
    /// @return Non-negative elapsed seconds.
    double elapsed_seconds() const;

private:
    std::chrono::steady_clock::time_point _start;
};

/// Formats seconds with three decimal places for CLI output.
/// @param seconds Duration in seconds.
/// @return Fixed-point string without unit suffix.
std::string format_seconds(double seconds);

}  // namespace pfm
```

Create `modules/core/timer.cpp`:

```cpp
#include "core/timer.h"

#include <iomanip>
#include <sstream>

namespace pfm {

Timer::Timer() : _start(std::chrono::steady_clock::now()) {}

void Timer::reset() {
    _start = std::chrono::steady_clock::now();
}

double Timer::elapsed_seconds() const {
    const auto elapsed = std::chrono::steady_clock::now() - _start;
    return std::chrono::duration<double>(elapsed).count();
}

std::string format_seconds(double seconds) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << seconds;
    return stream.str();
}

}  // namespace pfm
```

- [ ] **Step 4: Run tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt tests/test_main.cpp modules/core/timer.h modules/core/timer.cpp modules/core/timer_test.cpp
git commit -m "$(cat <<'EOF'
Add CLI timing utility.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Add training timing output

**Files:**
- Modify: `modules/train/trainer.h`
- Modify: `modules/train/trainer.cpp`
- Modify: `modules/train/trainer_test.cpp`
- Modify: `modules/infer/pipeline.cpp`
- Modify: `modules/infer/pipeline_test.cpp`

- [ ] **Step 1: Write failing training timing tests**

In `modules/train/trainer_test.cpp`, add a stdout capture helper if one does not exist:

```cpp
struct CoutCapture {
    std::ostringstream stream;
    std::streambuf* old = nullptr;

    CoutCapture() : old(std::cout.rdbuf(stream.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(old); }
    std::string str() const { return stream.str(); }
};
```

Add includes:

```cpp
#include <sstream>
```

Add test:

```cpp
static void trainer_reports_epoch_and_batch_timing() {
    TempTrainerDirectory temp_dir("pfm_trainer_timing");
    write_training_images(temp_dir.image_dir());
    pfm::TrainConfig config;
    config.image_dir = temp_dir.image_dir().string();
    config.checkpoint = temp_dir.file("model.pt").string();
    config.epochs = 1;
    config.batch_size = 1;
    config.resize = 16;

    CoutCapture capture;
    const auto result = pfm::train_model(config);
    const auto output = capture.str();

    PFM_REQUIRE(result.total_time_seconds >= 0.0);
    PFM_REQUIRE(result.avg_batch_time_seconds >= 0.0);
    PFM_REQUIRE(output.find("epoch_time=") != std::string::npos);
}
```

Register:

```cpp
register_test("trainer_reports_epoch_and_batch_timing", trainer_reports_epoch_and_batch_timing);
```

In `modules/infer/pipeline_test.cpp`, add test for final train command line output:

```cpp
static void pipeline_train_prints_total_and_average_batch_time() {
    TempPipelineDirectory temp_dir("pfm_pipeline_train_timing");
    write_training_images(temp_dir.image_dir());
    auto options = make_train_options(temp_dir);
    options.checkpoint = temp_dir.file("timed_model.pt").string();

    CoutCapture capture;
    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    const auto output = capture.str();

    PFM_REQUIRE(output.find("total_time=") != std::string::npos);
    PFM_REQUIRE(output.find("avg_batch_time=") != std::string::npos);
}
```

Register it in `register_pipeline_tests()`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: compile failure because `TrainResult` does not have timing fields, or assertion failure because output lacks timing fields.

- [ ] **Step 3: Implement training timers**

In `modules/train/trainer.h`, add fields:

```cpp
double total_time_seconds = 0.0;
double avg_batch_time_seconds = 0.0;
```

In `modules/train/trainer.cpp`, include:

```cpp
#include "core/timer.h"
```

In `train_model()`, create a total timer before loading/training work that should be counted:

```cpp
Timer total_timer;
int64_t completed_batches = 0;
double accumulated_batch_seconds = 0.0;
```

Inside each epoch before the batch loop:

```cpp
Timer epoch_timer;
```

Inside each batch loop at the top:

```cpp
Timer batch_timer;
```

After `optimizer.step()` and existing loss `.item()` reads:

```cpp
const double batch_seconds = batch_timer.elapsed_seconds();
accumulated_batch_seconds += batch_seconds;
++completed_batches;
```

After the epoch batch loop, print:

```cpp
std::cout << "train epoch summary: epoch=" << epoch + 1 << '/' << config.epochs
          << " epoch_time=" << format_seconds(epoch_timer.elapsed_seconds()) << "s\n";
```

Before `save_checkpoint(config, modules);`, set result fields:

```cpp
result.total_time_seconds = total_timer.elapsed_seconds();
result.avg_batch_time_seconds = completed_batches == 0
    ? 0.0
    : accumulated_batch_seconds / static_cast<double>(completed_batches);
```

In `modules/infer/pipeline.cpp`, update `run_train_command()` final output:

```cpp
std::cout << "training complete: epochs=" << result.epochs_completed
          << " final_loss=" << result.final_loss
          << " total_time=" << format_seconds(result.total_time_seconds) << "s"
          << " avg_batch_time=" << format_seconds(result.avg_batch_time_seconds) << "s\n";
```

Include `core/timer.h` in `pipeline.cpp`.

- [ ] **Step 4: Run tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all tests pass and train output contains `epoch_time=`, `total_time=`, and `avg_batch_time=`.

- [ ] **Step 5: Commit**

```bash
git add modules/train/trainer.h modules/train/trainer.cpp modules/train/trainer_test.cpp modules/infer/pipeline.cpp modules/infer/pipeline_test.cpp
git commit -m "$(cat <<'EOF'
Report training timing in CLI output.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add extract, match, eval, and export timing output

**Files:**
- Modify: `modules/infer/pipeline.cpp`
- Modify: `modules/infer/pipeline_test.cpp`

- [ ] **Step 1: Write failing command timing tests**

In `modules/infer/pipeline_test.cpp`, add tests using the existing temp helpers:

```cpp
static void pipeline_extract_prints_stage_timing() {
    TempPipelineDirectory temp_dir("pfm_pipeline_extract_timing");
    write_training_images(temp_dir.image_dir());
    const auto checkpoint = temp_dir.file("model.pt");
    auto train_options = make_train_options(temp_dir);
    train_options.checkpoint = checkpoint.string();
    PFM_REQUIRE(pfm::run_train_command(train_options) == 0);

    pfm::CliOptions options;
    options.command = pfm::Command::Extract;
    options.image = (temp_dir.image_dir() / "image_0.png").string();
    options.checkpoint = checkpoint.string();
    options.output = temp_dir.file("features.pt").string();
    options.visualization_dir = temp_dir.file("vis").string();

    CoutCapture capture;
    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto output = capture.str();

    PFM_REQUIRE(output.find("elapsed=") != std::string::npos);
    PFM_REQUIRE(output.find("image_load=") != std::string::npos);
    PFM_REQUIRE(output.find("model_forward=") != std::string::npos);
    PFM_REQUIRE(output.find("decode=") != std::string::npos);
    PFM_REQUIRE(output.find("save=") != std::string::npos);
    PFM_REQUIRE(output.find("visualization=") != std::string::npos);
}

static void pipeline_match_eval_and_export_print_timing() {
    TempPipelineDirectory temp_dir("pfm_pipeline_command_timing");
    write_training_images(temp_dir.image_dir());
    const auto checkpoint = temp_dir.file("model.pt");
    auto train_options = make_train_options(temp_dir);
    train_options.checkpoint = checkpoint.string();
    PFM_REQUIRE(pfm::run_train_command(train_options) == 0);

    pfm::CliOptions match_options;
    match_options.command = pfm::Command::Match;
    match_options.image_a = (temp_dir.image_dir() / "image_0.png").string();
    match_options.image_b = (temp_dir.image_dir() / "image_1.png").string();
    match_options.checkpoint = checkpoint.string();
    match_options.output = temp_dir.file("matches.pt").string();
    match_options.visualization_dir = temp_dir.file("match_vis").string();

    CoutCapture match_capture;
    PFM_REQUIRE(pfm::run_match_command(match_options) == 0);
    const auto match_output = match_capture.str();
    PFM_REQUIRE(match_output.find("elapsed=") != std::string::npos);
    PFM_REQUIRE(match_output.find("extract_a=") != std::string::npos);
    PFM_REQUIRE(match_output.find("extract_b=") != std::string::npos);
    PFM_REQUIRE(match_output.find("match_time=") != std::string::npos);
    PFM_REQUIRE(match_output.find("save=") != std::string::npos);
    PFM_REQUIRE(match_output.find("visualization=") != std::string::npos);

    const auto pairs_path = temp_dir.file("pairs.txt");
    write_text_file(pairs_path, match_options.image_a + " " + match_options.image_b + "\n");
    pfm::CliOptions eval_options;
    eval_options.command = pfm::Command::Eval;
    eval_options.pairs = pairs_path.string();
    eval_options.checkpoint = checkpoint.string();
    eval_options.output = temp_dir.file("report.pt").string();

    CoutCapture eval_capture;
    PFM_REQUIRE(pfm::run_eval_command(eval_options) == 0);
    const auto eval_output = eval_capture.str();
    PFM_REQUIRE(eval_output.find("pairs=1") != std::string::npos);
    PFM_REQUIRE(eval_output.find("elapsed=") != std::string::npos);
    PFM_REQUIRE(eval_output.find("avg_pair_time=") != std::string::npos);

    pfm::CliOptions export_options;
    export_options.command = pfm::Command::Export;
    export_options.checkpoint = checkpoint.string();
    export_options.output = temp_dir.file("exported.pt").string();

    CoutCapture export_capture;
    PFM_REQUIRE(pfm::run_export_command(export_options) == 0);
    PFM_REQUIRE(export_capture.str().find("elapsed=") != std::string::npos);
}
```

If `write_text_file` does not exist in `pipeline_test.cpp`, add:

```cpp
void write_text_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    PFM_REQUIRE(static_cast<bool>(output));
    output << text;
}
```

Register both tests in `register_pipeline_tests()`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: assertions fail because command outputs do not yet include timing fields.

- [ ] **Step 3: Add extraction timing fields**

In `modules/infer/pipeline.cpp`, extend `ExtractedFeatureSet`:

```cpp
struct ExtractionTiming {
    double image_load_seconds = 0.0;
    double model_forward_seconds = 0.0;
    double decode_seconds = 0.0;
};

struct ExtractedFeatureSet {
    FeatureSet features;
    int64_t feature_map_width = 0;
    int64_t feature_map_height = 0;
    ExtractionTiming timing;
};
```

In `extract_feature_set()`, time stages:

```cpp
ExtractionTiming timing;
Timer image_timer;
const auto image = load_image_tensor(image_path);
timing.image_load_seconds = image_timer.elapsed_seconds();

Timer forward_timer;
const auto maps = run_mvp_model(image, modules, checkpoint_config, device);
timing.model_forward_seconds = forward_timer.elapsed_seconds();

Timer decode_timer;
const auto intensity_mask = make_intensity_mask(image, min_keypoint_intensity).to(torch::kCPU);
auto features = decode_feature_maps(maps, decode_config, intensity_mask);
timing.decode_seconds = decode_timer.elapsed_seconds();

return ExtractedFeatureSet{std::move(features), maps.heatmap.size(3), maps.heatmap.size(2), timing};
```

- [ ] **Step 4: Add output timing for commands**

In `run_extract_command()`, create `Timer total_timer;`, time save and visualization, then print:

```cpp
std::cout << "extraction complete: features=" << options.output
          << " elapsed=" << format_seconds(total_timer.elapsed_seconds()) << "s"
          << " image_load=" << format_seconds(extracted.timing.image_load_seconds) << "s"
          << " model_forward=" << format_seconds(extracted.timing.model_forward_seconds) << "s"
          << " decode=" << format_seconds(extracted.timing.decode_seconds) << "s"
          << " save=" << format_seconds(save_seconds) << "s"
          << " visualization=" << format_seconds(visualization_seconds) << "s\n";
```

In `run_match_command()`, time total, matching, saving, visualization and print:

```cpp
std::cout << "matching complete: matches=" << options.output
          << " elapsed=" << format_seconds(total_timer.elapsed_seconds()) << "s"
          << " extract_a=" << format_seconds(extract_a_seconds) << "s"
          << " extract_b=" << format_seconds(extract_b_seconds) << "s"
          << " match_time=" << format_seconds(match_seconds) << "s"
          << " save=" << format_seconds(save_seconds) << "s"
          << " visualization=" << format_seconds(visualization_seconds) << "s\n";
```

Set `extract_a_seconds` and `extract_b_seconds` to each extraction stage sum:

```cpp
const auto extract_a_seconds = extracted_a.timing.image_load_seconds + extracted_a.timing.model_forward_seconds + extracted_a.timing.decode_seconds;
```

In `run_eval_command()`, create total timer before loading pairs and print:

```cpp
const auto elapsed = total_timer.elapsed_seconds();
const auto avg_pair_time = pairs.empty() ? 0.0 : elapsed / static_cast<double>(pairs.size());
std::cout << "evaluation complete: report=" << options.output
          << " pairs=" << pairs.size()
          << " elapsed=" << format_seconds(elapsed) << "s"
          << " avg_pair_time=" << format_seconds(avg_pair_time) << "s\n";
```

In `run_export_command()`, create total timer and print:

```cpp
std::cout << "export complete: checkpoint=" << options.output
          << " elapsed=" << format_seconds(total_timer.elapsed_seconds()) << "s\n";
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all tests pass and command outputs include timing fields.

- [ ] **Step 6: Commit**

```bash
git add modules/infer/pipeline.cpp modules/infer/pipeline_test.cpp
git commit -m "$(cat <<'EOF'
Report inference command timing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Update docs and run final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/usage.md`
- Modify: `docs/training.md`

- [ ] **Step 1: Update CLI option documentation**

In `README.md` and `docs/usage.md`, add extract example options:

```bash
--keypoint-grid-rows 8 \
--keypoint-grid-cols 8 \
--keypoints-per-cell 0 \
--nms-radius 4
```

Add this Chinese explanation:

```markdown
推理阶段的稀疏特征点默认会先应用低灰度过滤，再做局部 NMS，随后按网格分块选点，最后用全局高分候选补足 `--max-keypoints`。`--keypoints-per-cell 0` 表示按 `max_keypoints / (rows * cols)` 自动推导，每个 cell 至少 1 个候选。
```

- [ ] **Step 2: Update timing documentation**

In `docs/training.md`, add:

```markdown
训练命令每个 epoch 会输出 `epoch_time=<seconds>s`，训练结束输出 `total_time=<seconds>s` 和 `avg_batch_time=<seconds>s`，用于判断整体耗时和 batch 级吞吐。
```

In `docs/usage.md`, add:

```markdown
`extract` 输出 `elapsed`、`image_load`、`model_forward`、`decode`、`save`、`visualization`。`match` 输出两张图的 `extract_a`、`extract_b`、`match_time`、`save`、`visualization`。`eval` 输出 `pairs`、`elapsed`、`avg_pair_time`。`export` 输出 `elapsed`。
```

- [ ] **Step 3: Run final verification**

Run:

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
./build/pfm_cli extract --help
./build/pfm_cli match --help
./build/pfm_cli eval --help
```

Expected:

- `./build/pfm_tests` reports all tests passed.
- `ctest` reports `100% tests passed`.
- CLI help includes `--keypoint-grid-rows`, `--keypoint-grid-cols`, `--keypoints-per-cell`, and `--nms-radius` for extract/match/eval.

- [ ] **Step 4: Commit docs**

```bash
git add README.md docs/usage.md docs/training.md
git commit -m "$(cat <<'EOF'
Document keypoint distribution and timing controls.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

- Spec coverage: Stage 1 is covered by Tasks 1-4: low-gray mask remains first, then local NMS, grid top-k, and global backfill, with CLI options shared by extract/match/eval. Stage 2 is covered by Tasks 5-7: timer utility, train epoch/total/average batch timing, extract/match/eval/export command timing. Docs and final verification are covered by Task 8.
- Scope intentionally excludes Stage 3 train/val/test split and Stage 4 residual/FPN model architecture because the spec recommends implementing Stage 1 and Stage 2 first.
- Placeholder scan: no TBD/TODO/fill-in steps remain; each task has explicit file paths, test code, implementation shape, commands, expected outcomes, and commit commands.
- Type consistency: `FeatureDecodeConfig`, `CliOptions` decode fields, `Timer`, `format_seconds`, `total_time_seconds`, and `avg_batch_time_seconds` are consistently named across tasks.
