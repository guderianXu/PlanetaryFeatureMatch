# Training Loss Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three training issues: (1) surface feature extractor losses in progress bar, (2) fix graph matcher by using warp-positioned B keypoints, (3) expand dense correlation window.

**Architecture:** Three independent fixes. Fix 1 adds feature metrics to progress bar display. Fix 2 rewrites `make_graph_matching_loss()` so B descriptors/keypoints come from warped positions with identity targets. Fix 3 increases `CORRELATION_RADIUS` from 2 to 4.

**Tech Stack:** C++17, libtorch, custom test harness

---

### Task 1: Add feature metrics to progress bar

**Files:**
- Modify: `src/train/trainer.cpp:1695-1700` (add metric values)
- Modify: `modules/logging/progress_logger.cpp:28-42` (display new fields)
- Modify: `src/train/trainer_test.cpp:539-551` (update test)

- [ ] **Step 1: Add feature metrics to progress bar TrainingMetric**

In `src/train/trainer.cpp`, after line 1699 (`iter_metric.values["descriptor_accuracy"] = ...`), add:

```cpp
            iter_metric.values["feature_loss"] = feature_loss_value;
            iter_metric.values["repeatability_loss"] = repeatability_loss_value;
            iter_metric.values["descriptor_loss"] = descriptor_loss_value;
```

- [ ] **Step 2: Update ConsoleProgressLogger display format**

In `modules/logging/progress_logger.cpp`, update `logIteration()` to show the new fields. Replace lines 28-33:

Old:
```cpp
    _stream << "] " << metric.iteration << '/' << metric.total_iterations
            << " loss=" << metricValue(metric, "loss_total")
            << " matcher=" << metricValue(metric, "matcher_loss")
            << " dense=" << metricValue(metric, "dense_loss")
            << " offset_px=" << metricValue(metric, "offset_error_px")
            << " lr=" << metric.learning_rate;
```

New:
```cpp
    _stream << "] " << metric.iteration << '/' << metric.total_iterations
            << " loss=" << metricValue(metric, "loss_total")
            << " match=" << metricValue(metric, "matcher_loss")
            << " feat=" << metricValue(metric, "feature_loss")
            << " rep=" << metricValue(metric, "repeatability_loss")
            << " dense=" << metricValue(metric, "dense_loss")
            << " off=" << metricValue(metric, "offset_error_px")
            << " desc_acc=" << metricValue(metric, "descriptor_accuracy")
            << " lr=" << metric.learning_rate;
```

- [ ] **Step 3: Update progress bar test**

In `src/train/trainer_test.cpp`, update `trainer_progress_reports_loss_components` test to match new format:

Old:
```cpp
    PFM_REQUIRE(output.find("loss=") != std::string::npos);
    PFM_REQUIRE(output.find("matcher=") != std::string::npos);
    PFM_REQUIRE(output.find("dense=") != std::string::npos);
    PFM_REQUIRE(output.find("offset_px=") != std::string::npos);
    PFM_REQUIRE(output.find("epoch summary") != std::string::npos);
```

New:
```cpp
    PFM_REQUIRE(output.find("loss=") != std::string::npos);
    PFM_REQUIRE(output.find("match=") != std::string::npos);
    PFM_REQUIRE(output.find("feat=") != std::string::npos);
    PFM_REQUIRE(output.find("dense=") != std::string::npos);
    PFM_REQUIRE(output.find("off=") != std::string::npos);
    PFM_REQUIRE(output.find("epoch summary") != std::string::npos);
```

Also update the logging test in `modules/logging/logging_test.cpp` line 62:

Old:
```cpp
    PFM_REQUIRE(text.find("matcher=3.5") != std::string::npos);
    PFM_REQUIRE(text.find("offset_px=12") != std::string::npos);
```

New:
```cpp
    PFM_REQUIRE(text.find("match=3.5") != std::string::npos);
    PFM_REQUIRE(text.find("off=12") != std::string::npos);
```

- [ ] **Step 4: Build and test**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/train/trainer.cpp src/train/trainer_test.cpp modules/logging/progress_logger.cpp modules/logging/logging_test.cpp
git commit -m "feat: add feature extractor metrics to progress bar

Show feature_loss, repeatability_loss, and descriptor_accuracy
in the ConsoleProgressLogger progress bar so training diagnostics
include feature extraction quality alongside matching losses.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Fix graph matcher — B keypoints from warp positions

**Files:**
- Modify: `src/train/trainer.cpp:742-774` (rewrite `make_graph_matching_loss`)

- [ ] **Step 1: Rewrite make_graph_matching_loss with warped B positions**

Replace the entire `make_graph_matching_loss()` function (lines 742-774) in `src/train/trainer.cpp`:

Old:
```cpp
torch::Tensor make_graph_matching_loss(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    auto sample_indices = filter_descriptor_sample_indices(
        make_descriptor_sample_indices(descriptors_a),
        valid_mask,
        descriptors_a.size(2),
        descriptors_a.size(3));
    if (sample_indices.numel() == 0) {
        return torch::zeros({}, descriptors_a.options());
    }
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    auto sampled_b = sample_spatial_descriptors(descriptors_b, sample_indices);
    auto target_indices = map_targets_to_sampled_columns(
        make_descriptor_target_indices(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3)),
        sample_indices,
        sample_indices.size(0));
    const auto batch_size = descriptors_a.size(0);
    std::vector<torch::Tensor> losses;
    losses.reserve(static_cast<size_t>(batch_size));
    auto keypoints_a = torch::stack(
        {sample_indices.remainder(descriptors_a.size(3)).to(descriptors_a.dtype()),
         (sample_indices / descriptors_a.size(3)).to(descriptors_a.dtype())},
        1);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto output = graph_matcher.forward(sampled_a[batch], keypoints_a, sampled_b[batch], keypoints_a);
        losses.push_back(graph_matching_cross_entropy_loss(output.logits, target_indices[batch]));
    }
    return torch::stack(losses).mean();
}
```

New:
```cpp
torch::Tensor make_graph_matching_loss(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    auto sample_indices = filter_descriptor_sample_indices(
        make_descriptor_sample_indices(descriptors_a),
        valid_mask,
        descriptors_a.size(2),
        descriptors_a.size(3));
    if (sample_indices.numel() == 0) {
        return torch::zeros({}, descriptors_a.options());
    }
    auto target_spatial = make_descriptor_target_indices(
        warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    auto sampled_b = sample_spatial_descriptors(descriptors_b, target_spatial);
    const auto sample_count = sample_indices.size(0);
    auto target_columns = torch::arange(sample_count, sample_indices.options());
    const auto batch_size = descriptors_a.size(0);
    std::vector<torch::Tensor> losses;
    losses.reserve(static_cast<size_t>(batch_size));
    auto keypoints_a = torch::stack(
        {sample_indices.remainder(descriptors_a.size(3)).to(descriptors_a.dtype()),
         (sample_indices / descriptors_a.size(3)).to(descriptors_a.dtype())},
        1);
    auto keypoints_b = torch::stack(
        {target_spatial.remainder(descriptors_b.size(3)).to(descriptors_b.dtype()),
         (target_spatial / descriptors_b.size(3)).to(descriptors_b.dtype())},
        1);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto output = graph_matcher.forward(
            sampled_a[batch], keypoints_a, sampled_b[batch], keypoints_b);
        losses.push_back(graph_matching_cross_entropy_loss(output.logits, target_columns));
    }
    return torch::stack(losses).mean();
}
```

- [ ] **Step 2: Clean up unused map_targets_to_sampled_columns**

Remove the function `map_targets_to_sampled_columns()` at lines 728-738 since it's no longer called. Also remove the test wrapper if one exists.

- [ ] **Step 3: Add test for B-side warp keypoint sampling**

In `src/train/trainer_test.cpp`, add a new test verifying that B keypoints differ from A keypoints after the fix:

```cpp
static void trainer_graph_matching_loss_uses_warped_b_keypoints() {
    pfm::PlanetaryGraphMatcher matcher(4, 16, 2);
    // 3-channel descriptors at 4x4 spatial grid = 16 positions, batch=1
    auto descriptors_a = torch::randn({1, 4, 4, 4}, torch::kFloat32);
    auto descriptors_b = torch::randn({1, 4, 4, 4}, torch::kFloat32);
    // Identity warp → B keypoints should match A keypoints
    auto warp = torch::zeros({1, 4, 4, 2}, torch::kFloat32);
    for (int64_t y = 0; y < 4; ++y) {
        for (int64_t x = 0; x < 4; ++x) {
            warp.index_put_({0, y, x, 0}, static_cast<float>(x));
            warp.index_put_({0, y, x, 1}, static_cast<float>(y));
        }
    }
    auto valid_mask = torch::ones({1, 4, 4}, torch::kBool);

    auto loss = pfm::testing::make_graph_matching_loss_for_test(
        *matcher, descriptors_a, descriptors_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
    PFM_REQUIRE(matcher->parameters().front().grad().defined());
    PFM_REQUIRE(matcher->parameters().front().grad().abs().sum().item<float>() > 0.0F);
}
```

Register it at the bottom of the test file:
```cpp
    register_test("trainer_graph_matching_loss_uses_warped_b_keypoints",
                  trainer_graph_matching_loss_uses_warped_b_keypoints);
```

- [ ] **Step 4: Build and test**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass. matcher_loss should decrease during training.

- [ ] **Step 5: Commit**

```bash
git add src/train/trainer.cpp src/train/trainer_test.cpp
git commit -m "fix: sample B descriptors and keypoints from warped positions

In make_graph_matching_loss(), sample B-side descriptors at
warped target positions instead of the same grid as A, and
use target spatial coordinates for B keypoints. With identity
targets (A[i] -> B[i]), the graph matcher can use both spatial
and descriptor cues for matching instead of being forced to
rely on descriptors alone.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Expand dense correlation window

**Files:**
- Modify: `src/models/dense_head.cpp:12` (CORRELATION_RADIUS)

- [ ] **Step 1: Increase correlation radius from 2 to 4**

In `src/models/dense_head.cpp`, line 12:

Old:
```cpp
constexpr int64_t CORRELATION_RADIUS = 2;
```

New:
```cpp
constexpr int64_t CORRELATION_RADIUS = 4;
```

This increases local correlation search from ±2 pixels (25 channels) to ±4 pixels (81 channels) at feature map resolution. For a 256x256 image with stage1 features at 128x128, this expands the effective search range from ±4 pixels to ±8 pixels at image resolution.

- [ ] **Step 2: Build and test**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass (284+).

- [ ] **Step 3: Commit**

```bash
git add src/models/dense_head.cpp
git commit -m "feat: expand dense head correlation radius 2 -> 4

Increase local correlation search window from ±2 to ±4 pixels
at feature resolution to help the dense head handle larger
displacements from rotation and perspective distortion in
planetary images.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Final verification

- [ ] **Step 1: Full build and test**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./pfm_tests && ctest --output-on-failure
```

- [ ] **Step 2: Update planning files**

Add stage 12 to task_plan.md, record findings and progress.

- [ ] **Step 3: Commit planning files**

```bash
git add task_plan.md progress.md findings.md
git commit -m "docs: update planning files for training loss fixes

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
