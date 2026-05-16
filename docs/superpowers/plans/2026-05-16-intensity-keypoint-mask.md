# Intensity Keypoint Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed normalized intensity threshold that prevents low-gray planetary background/edge artifacts from producing inference keypoints or contributing training supervision.

**Architecture:** Add a shared intensity-mask helper under `modules/data`, thread `min_keypoint_intensity` through CLI, trainer, and inference pipeline, and extend feature decoding to accept an optional image-derived mask. Training combines intensity masks with the existing geometric valid mask so online and cached synthetic pairs behave consistently without changing cache or checkpoint formats.

**Tech Stack:** C++17, LibTorch tensors, OpenCV image loading through existing image IO, CLI11, CMake, custom `pfm_tests` runner.

---

## File Structure

- Create `modules/data/intensity_mask.h`: shared API for validating threshold and building `H x W` masks from `C x H x W` tensors.
- Create `modules/data/intensity_mask.cpp`: threshold validation and channel-mean mask implementation.
- Create `modules/data/intensity_mask_test.cpp`: unit tests for grayscale/RGB masks and invalid thresholds.
- Modify `CMakeLists.txt`: add the new source and test file.
- Modify `tests/test_main.cpp`: register intensity mask tests.
- Modify `modules/cli/commands.h`: add `CliOptions::min_keypoint_intensity`.
- Modify `modules/cli/commands.cpp`: add `--min-keypoint-intensity` to train/extract/match/eval with CLI11 range check and footer text.
- Modify `modules/cli/commands_test.cpp`: parse and invalid-value coverage for all affected commands.
- Modify `modules/infer/feature_extractor.h/.cpp`: add an overload accepting an optional intensity mask and apply it to sparse top-k and dense points.
- Modify `modules/infer/feature_extractor_test.cpp`: prove masked locations are excluded.
- Modify `modules/infer/pipeline.cpp`: generate masks from loaded images and pass them to decoding for extract/match/eval.
- Modify `modules/infer/pipeline_test.cpp`: verify extract visualization/output excludes dark-region keypoints and match still writes outputs.
- Modify `modules/train/trainer.h/.cpp`: add config field, validation, and mask-combined training loss.
- Modify `modules/train/trainer_test.cpp`: verify invalid threshold fails and thresholded training succeeds.
- Modify `README.md`, `docs/training.md`, `docs/usage.md`: document the new parameter and recommended values.

---

### Task 1: Shared Intensity Mask Module

**Files:**
- Create: `modules/data/intensity_mask.h`
- Create: `modules/data/intensity_mask.cpp`
- Create: `modules/data/intensity_mask_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Write failing tests**

Add `modules/data/intensity_mask_test.cpp` with tests:

```cpp
#include <torch/torch.h>

#include "data/intensity_mask.h"
#include "tests/test_harness.h"

static void intensity_mask_thresholds_single_channel_image() {
    const auto image = torch::tensor({{{0.0F, 0.05F}, {0.1F, 0.2F}}}, torch::kFloat32);
    const auto mask = pfm::make_intensity_mask(image, 0.1);

    PFM_REQUIRE(mask.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(mask.dtype() == torch::kFloat32);
    PFM_REQUIRE_CLOSE(mask.index({0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({1, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({1, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void intensity_mask_uses_mean_for_multi_channel_image() {
    const auto image = torch::tensor(
        {{{0.0F, 0.4F}}, {{0.0F, 0.4F}}, {{0.3F, 0.4F}}}, torch::kFloat32);
    const auto mask = pfm::make_intensity_mask(image, 0.2);

    PFM_REQUIRE(mask.sizes() == torch::IntArrayRef({1, 2}));
    PFM_REQUIRE_CLOSE(mask.index({0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({0, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void intensity_mask_rejects_invalid_thresholds() {
    const auto image = torch::ones({1, 2, 2}, torch::kFloat32);

    PFM_REQUIRE_INVALID_ARG(pfm::validate_min_keypoint_intensity(-0.1));
    PFM_REQUIRE_INVALID_ARG(pfm::validate_min_keypoint_intensity(1.1));
    PFM_REQUIRE_INVALID_ARG(pfm::make_intensity_mask(image, -0.1));
}

void register_intensity_mask_tests() {
    register_test("intensity_mask_thresholds_single_channel_image", intensity_mask_thresholds_single_channel_image);
    register_test("intensity_mask_uses_mean_for_multi_channel_image", intensity_mask_uses_mean_for_multi_channel_image);
    register_test("intensity_mask_rejects_invalid_thresholds", intensity_mask_rejects_invalid_thresholds);
}
```

Register it in `tests/test_main.cpp`:

```cpp
void register_intensity_mask_tests();
```

and call:

```cpp
register_intensity_mask_tests();
```

Add the test file to `pfm_tests` in `CMakeLists.txt`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cmake -S . -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
```

Expected: build fails because `data/intensity_mask.h` does not exist.

- [ ] **Step 3: Implement minimal module**

Create `modules/data/intensity_mask.h`:

```cpp
#pragma once

#include <torch/torch.h>

namespace pfm {

/// Validates a normalized minimum keypoint intensity threshold.
/// @param min_keypoint_intensity Threshold in normalized image intensity units.
/// @throws std::invalid_argument if the threshold is non-finite or outside [0, 1].
void validate_min_keypoint_intensity(double min_keypoint_intensity);

/// Builds an H x W float mask from a C x H x W image tensor using channel-mean intensity.
/// @param image Float image tensor in C x H x W layout.
/// @param min_keypoint_intensity Pixels below this normalized threshold become 0.
/// @return H x W CPU/GPU float mask on the same device as image, with values 0 or 1.
/// @throws std::invalid_argument if the image layout or threshold is invalid.
torch::Tensor make_intensity_mask(const torch::Tensor& image, double min_keypoint_intensity);

}  // namespace pfm
```

Create `modules/data/intensity_mask.cpp`:

```cpp
#include <cmath>
#include <stdexcept>

#include <torch/torch.h>

#include "core/tensor_utils.h"
#include "data/intensity_mask.h"

namespace pfm {

void validate_min_keypoint_intensity(double min_keypoint_intensity) {
    if (!std::isfinite(min_keypoint_intensity) || min_keypoint_intensity < 0.0 || min_keypoint_intensity > 1.0) {
        throw std::invalid_argument("min_keypoint_intensity must be between 0 and 1");
    }
}

torch::Tensor make_intensity_mask(const torch::Tensor& image, double min_keypoint_intensity) {
    validate_min_keypoint_intensity(min_keypoint_intensity);
    require_chw_image(image);
    const auto intensity = image.to(torch::kFloat32).mean(0).contiguous();
    return intensity.ge(min_keypoint_intensity).to(torch::kFloat32).contiguous();
}

}  // namespace pfm
```

Add `modules/data/intensity_mask.cpp` to the `pfm` library in `CMakeLists.txt`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
cmake -S . -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: intensity mask tests pass.

---

### Task 2: CLI Option Plumbing

**Files:**
- Modify: `modules/cli/commands.h`
- Modify: `modules/cli/commands.cpp`
- Modify: `modules/cli/commands_test.cpp`

- [ ] **Step 1: Write failing CLI tests**

Update existing parse tests to include `--min-keypoint-intensity`:

```cpp
PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
```

Add this argument pair to train/extract/match/eval parse vectors:

```cpp
"--min-keypoint-intensity",
"0.08",
```

Add invalid parse test:

```cpp
static void parse_min_keypoint_intensity_out_of_range_throws() {
    PFM_REQUIRE_PARSE_ERROR(pfm::parse_cli({"pfm", "extract", "--image", "a.tif", "--checkpoint", "model.pt", "--output", "features.pt", "--min-keypoint-intensity", "1.5"}));
    PFM_REQUIRE_PARSE_ERROR(pfm::parse_cli({"pfm", "train", "--image-dir", "images", "--checkpoint", "model.pt", "--min-keypoint-intensity", "-0.1"}));
}
```

Register it in `register_cli_tests()`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cmake --build build -j$(nproc)
```

Expected: build fails because `CliOptions` has no `min_keypoint_intensity`.

- [ ] **Step 3: Implement CLI plumbing**

In `modules/cli/commands.h`, add:

```cpp
double min_keypoint_intensity = 0.0;
```

In `modules/cli/commands.cpp`, include CLI validator for train/extract/match/eval:

```cpp
train->add_option("--min-keypoint-intensity", options.min_keypoint_intensity, "Minimum normalized image intensity for keypoint supervision")
    ->check(CLI::Range(0.0, 1.0));
```

For extract/match/eval use descriptions matching inference:

```cpp
extract->add_option("--min-keypoint-intensity", options.min_keypoint_intensity, "Minimum normalized image intensity for output keypoints")
    ->check(CLI::Range(0.0, 1.0));
```

Also add `[--min-keypoint-intensity 0.0]` to the footer lines for train/extract/match/eval.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: CLI tests pass and help includes `--min-keypoint-intensity`.

---

### Task 3: Masked Feature Decoding

**Files:**
- Modify: `modules/infer/feature_extractor.h`
- Modify: `modules/infer/feature_extractor.cpp`
- Modify: `modules/infer/feature_extractor_test.cpp`

- [ ] **Step 1: Write failing feature extractor tests**

Add tests:

```cpp
static void decode_feature_maps_excludes_masked_sparse_locations() {
    auto heatmap = torch::zeros({1, 1, 2, 3}, torch::kFloat32);
    heatmap.index_put_({0, 0, 0, 0}, 10.0F);
    heatmap.index_put_({0, 0, 1, 2}, 9.0F);
    auto maps = makeMaps(heatmap, torch::ones({1, 1, 2, 3}, torch::kFloat32));
    const auto mask = torch::tensor({{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F}}, torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 2, 0.5, mask);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({1, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void decode_feature_maps_returns_empty_sparse_features_when_mask_is_empty() {
    auto maps = makeMaps(torch::ones({1, 1, 2, 2}, torch::kFloat32), torch::ones({1, 1, 2, 2}, torch::kFloat32));
    const auto mask = torch::zeros({2, 2}, torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 4, 0.5, mask);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({0, 2}));
    PFM_REQUIRE(features.scores.sizes() == torch::IntArrayRef({0}));
    PFM_REQUIRE(features.descriptors.size(0) == 0);
}

static void decode_feature_maps_filters_dense_points_with_mask() {
    const auto heatmap = torch::zeros({1, 1, 2, 2}, torch::kFloat32);
    const auto dense_confidence = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    auto maps = makeMaps(heatmap, dense_confidence);
    const auto mask = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 1, 0.5, mask);

    PFM_REQUIRE(features.dense_points.sizes() == torch::IntArrayRef({2, 2}));
}
```

Register these tests.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cmake --build build -j$(nproc)
```

Expected: build fails because the four-argument `decode_feature_maps` overload does not exist.

- [ ] **Step 3: Implement masked decoding**

In `modules/infer/feature_extractor.h`, add overload:

```cpp
/// Decodes sparse and dense feature tensors while suppressing invalid image locations.
/// @param intensity_mask Optional H x W float/bool mask in image coordinates; nonzero values are valid.
FeatureSet decode_feature_maps(
    const RawFeatureMaps& maps,
    int max_keypoints,
    double semi_dense_threshold,
    const torch::Tensor& intensity_mask
);
```

Keep the existing three-argument function and implement it as:

```cpp
FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold) {
    return decode_feature_maps(maps, max_keypoints, semi_dense_threshold, torch::Tensor());
}
```

In `.cpp`, add helper to resize optional mask:

```cpp
torch::Tensor prepare_decode_mask(const torch::Tensor& mask, int64_t height, int64_t width) {
    if (!mask.defined()) {
        return torch::ones({height, width}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    }
    if (!mask.device().is_cpu()) {
        throw std::invalid_argument("intensity_mask must be a CPU tensor");
    }
    if (mask.dim() != 2) {
        throw std::invalid_argument("intensity_mask must be 2D");
    }
    auto float_mask = mask.to(torch::kFloat32).unsqueeze(0).unsqueeze(0);
    if (mask.size(0) != height || mask.size(1) != width) {
        float_mask = torch::nn::functional::interpolate(
            float_mask,
            torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{height, width}).mode(torch::kNearest));
    }
    return float_mask.squeeze().gt(0.0).contiguous();
}
```

Before sparse top-k:

```cpp
const auto valid_mask = prepare_decode_mask(intensity_mask, height, width);
const auto valid_flat = valid_mask.flatten();
const int64_t valid_count = valid_flat.to(torch::kLong).sum().item<int64_t>();
const int64_t sparse_count = std::min<int64_t>(max_keypoints, valid_count);
```

If `sparse_count == 0`, construct empty sparse tensors with correct shapes.

Otherwise use:

```cpp
const auto masked_heatmap = heatmap.flatten().masked_fill(valid_flat.logical_not(), -std::numeric_limits<float>::infinity());
const auto topk = torch::topk(masked_heatmap, sparse_count);
```

For dense loop, skip points when `!dense_mask.index({y, x}).item<bool>()`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: feature extractor tests pass.

---

### Task 4: Inference Pipeline Integration

**Files:**
- Modify: `modules/infer/pipeline.cpp`
- Modify: `modules/infer/pipeline_test.cpp`

- [ ] **Step 1: Write failing pipeline tests**

In `pipeline_extract_writes_loadable_feature_file`, set `options.min_keypoint_intensity = 0.1` and keep existing assertions.

Add a focused test using a dark image with only one bright island:

```cpp
static void pipeline_extract_filters_keypoints_below_min_intensity() {
    TempPipelineDirectory temp_dir("pfm_pipeline_extract_intensity_mask");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("masked_extract.png");
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(0));
    mat(cv::Rect(20, 20, 8, 8)).setTo(cv::Scalar(255));
    PFM_REQUIRE(cv::imwrite(image.string(), mat));
    const auto output = temp_dir.file("features.pt");

    pfm::CliOptions options;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 16;
    options.semi_dense_threshold = 0.0;
    options.device = "cpu";
    options.min_keypoint_intensity = 0.5;

    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto features = pfm::load_feature_set(options.output);
    for (int64_t index = 0; index < features.keypoints.size(0); ++index) {
        PFM_REQUIRE(features.keypoints.index({index, 0}).item<float>() >= 20.0F);
        PFM_REQUIRE(features.keypoints.index({index, 1}).item<float>() >= 20.0F);
    }
}
```

Register the test.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: the new pipeline test fails because extraction ignores `min_keypoint_intensity`.

- [ ] **Step 3: Implement pipeline integration**

In `modules/infer/pipeline.cpp`, include:

```cpp
#include "data/intensity_mask.h"
```

Change `extract_feature_set` to call:

```cpp
const auto image = load_image_tensor(image_path);
const auto maps = run_mvp_model(image, modules, checkpoint_config, device);
const auto mask = make_intensity_mask(image, min_keypoint_intensity).to(torch::kCPU);
return decode_feature_maps(maps, max_keypoints, semi_dense_threshold, mask);
```

Add `double min_keypoint_intensity` to `extract_feature_set` parameters and pass `options.min_keypoint_intensity` from `run_extract_command`, `run_match_command`, and `run_eval_command`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: pipeline tests pass.

---

### Task 5: Training Mask Integration

**Files:**
- Modify: `modules/train/trainer.h`
- Modify: `modules/train/trainer.cpp`
- Modify: `modules/train/trainer_test.cpp`
- Modify: `modules/infer/pipeline.cpp`

- [ ] **Step 1: Write failing trainer tests**

Add to `trainer_invalid_numeric_parameters_throw_invalid_argument`:

```cpp
auto invalid_intensity = config;
invalid_intensity.min_keypoint_intensity = 1.5;
PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_intensity));
```

Add a thresholded training smoke:

```cpp
static void trainer_with_min_keypoint_intensity_saves_checkpoint() {
    TempTrainingDirectory temp_dir("pfm_trainer_intensity_mask");
    write_training_image(temp_dir.file("image.png"), 33);

    pfm::TrainConfig config;
    config.image_dir = temp_dir.path().string();
    config.checkpoint = temp_dir.file("checkpoint.pt").string();
    config.epochs = 1;
    config.batch_size = 1;
    config.base_channels = 2;
    config.descriptor_dim = 4;
    config.min_keypoint_intensity = 0.05;

    const auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
}
```

Register it.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cmake --build build -j$(nproc)
```

Expected: build fails because `TrainConfig` has no `min_keypoint_intensity`.

- [ ] **Step 3: Implement training config and mask combination**

In `modules/train/trainer.h`, add:

```cpp
double min_keypoint_intensity = 0.0;
```

In `validate_config`, call:

```cpp
validate_min_keypoint_intensity(config.min_keypoint_intensity);
```

Include `data/intensity_mask.h` in `trainer.cpp`.

Change `training_loss_from_pairs` signature:

```cpp
TrainingLossComponents training_loss_from_pairs(
    TrainModules& modules,
    const std::vector<SyntheticPair>& pairs,
    double min_keypoint_intensity
)
```

After `valid_mask` is stacked:

```cpp
const auto mask_a = stack_batch(make masks from views_a with make_intensity_mask(view, min_keypoint_intensity));
const auto mask_b = stack_batch(make masks from views_b with make_intensity_mask(view, min_keypoint_intensity));
const auto combined_valid_mask = valid_mask * mask_a * mask_b;
```

Use `combined_valid_mask` for `sparse_mask`, `dense_mask`, and `make_sparse_descriptor_metrics`.

Pass `config.min_keypoint_intensity` from training loop calls.

In `modules/infer/pipeline.cpp`, set:

```cpp
config.min_keypoint_intensity = options.min_keypoint_intensity;
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: trainer tests pass.

---

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/training.md`
- Modify: `docs/usage.md`

- [ ] **Step 1: Update docs**

Add Chinese documentation explaining:

```markdown
`--min-keypoint-intensity` 用于排除低灰度背景和行星边缘拍摄伪影。图像读取后会归一化到 `[0, 1]`，低于该阈值的位置不会输出为特征点；训练时这些位置也不会参与主要监督损失。默认值 `0.0` 保持旧行为。建议先从 `0.05` 到 `0.1` 试起，并结合 `--visualization-dir` 检查过滤效果。
```

Update train/extract/match/eval command examples where appropriate with:

```bash
--min-keypoint-intensity 0.08
```

- [ ] **Step 2: Run help checks**

Run:

```bash
./build/pfm_cli train --help
./build/pfm_cli extract --help
./build/pfm_cli match --help
./build/pfm_cli eval --help
```

Expected: all affected commands list `--min-keypoint-intensity`.

- [ ] **Step 3: Run full verification**

Run:

```bash
cmake -S . -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc) && ./build/pfm_tests && ctest --test-dir build --output-on-failure
```

Expected: build succeeds, `pfm_tests` reports all tests passed, and CTest reports `100% tests passed`.

- [ ] **Step 4: Commit implementation locally**

Run:

```bash
git status --short
git diff -- README.md docs/training.md docs/usage.md modules/cli/commands.h modules/cli/commands.cpp modules/cli/commands_test.cpp modules/data/intensity_mask.h modules/data/intensity_mask.cpp modules/data/intensity_mask_test.cpp modules/infer/feature_extractor.h modules/infer/feature_extractor.cpp modules/infer/feature_extractor_test.cpp modules/infer/pipeline.cpp modules/infer/pipeline_test.cpp modules/train/trainer.h modules/train/trainer.cpp modules/train/trainer_test.cpp CMakeLists.txt tests/test_main.cpp
git add README.md docs/training.md docs/usage.md docs/superpowers/specs/2026-05-16-intensity-keypoint-mask-design.md docs/superpowers/plans/2026-05-16-intensity-keypoint-mask.md modules/cli/commands.h modules/cli/commands.cpp modules/cli/commands_test.cpp modules/data/intensity_mask.h modules/data/intensity_mask.cpp modules/data/intensity_mask_test.cpp modules/infer/feature_extractor.h modules/infer/feature_extractor.cpp modules/infer/feature_extractor_test.cpp modules/infer/pipeline.cpp modules/infer/pipeline_test.cpp modules/train/trainer.h modules/train/trainer.cpp modules/train/trainer_test.cpp CMakeLists.txt tests/test_main.cpp
git commit -m "$(cat <<'EOF'
Add intensity mask filtering for feature points.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds after hooks.

- [ ] **Step 5: Merge back to main branch locally**

If this worktree is on a feature branch, merge it into `main` after verification:

```bash
git checkout main
git merge --no-ff <feature-branch>
```

If this worktree is already on `main`, report that the work is already on `main` locally after the commit. Do not push unless the user explicitly asks for a push.

---

## Self-Review Checklist

- Spec coverage: CLI, inference sparse/dense filtering, training filtering, docs, tests, and verification are covered.
- Placeholder scan: no TODO/TBD placeholders are present.
- Type consistency: the plan consistently uses `min_keypoint_intensity`, `make_intensity_mask`, and the four-argument `decode_feature_maps` overload.
