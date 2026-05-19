# Graph Matching Loss Fix + Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix graph_matching_loss stuck at ~18 by replacing exact column matching with nearest-neighbor, and wire ConsoleProgressLogger into the training loop to show a progress bar.

**Architecture:** Two independent fixes: (1) `map_targets_to_sampled_columns()` in trainer.cpp switches from exact equality to nearest-neighbor, so warped targets map to the closest sampled column instead of dustbin. (2) trainer creates a `ConsoleProgressLogger` instance and routes per-batch progress through it instead of raw `std::cout`.

**Tech Stack:** C++17, libtorch, custom test harness (no Google Test)

---

### Task 1: Fix map_targets_to_sampled_columns with nearest-neighbor

**Files:**
- Modify: `src/train/trainer.cpp:728-738`
- Test: `src/train/trainer_test.cpp:391-403` (existing test, may need update)

- [ ] **Step 1: Replace exact equality mapping with nearest-neighbor**

Replace the body of `map_targets_to_sampled_columns()` (lines 728-738) in `src/train/trainer.cpp`:

```cpp
torch::Tensor map_targets_to_sampled_columns(
    const torch::Tensor& target_indices,
    const torch::Tensor& sample_indices,
    int64_t dustbin_index
) {
    auto mapped = torch::full_like(target_indices, dustbin_index);
    for (int64_t column = 0; column < sample_indices.size(0); ++column) {
        mapped.index_put_({target_indices.eq(sample_indices[column])}, column);
    }
    return mapped;
}
```

Replace with:

```cpp
torch::Tensor map_targets_to_sampled_columns(
    const torch::Tensor& target_indices,
    const torch::Tensor& sample_indices,
    int64_t dustbin_index
) {
    const auto sample_count = sample_indices.size(0);
    auto target_exp = target_indices.unsqueeze(1).expand({-1, sample_count});
    auto sample_exp = sample_indices.unsqueeze(0).expand({target_indices.size(0), -1});
    auto distances = (target_exp - sample_exp).abs();
    return distances.argmin(1);
}
```

- [ ] **Step 2: Update test for new behavior**

The test `trainer_graph_matching_loss_uses_dustbin_for_unsampled_targets` at line 391 expects dustbin behavior. With nearest-neighbor, every target maps to some column. Update the test to verify the loss is finite and trains the matcher (combining it with the training test at line 377).

In `src/train/trainer_test.cpp`, replace lines 377-403:

Old:
```cpp
static void trainer_graph_matching_loss_trains_graph_matcher_parameters() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    auto descriptors_a = torch::tensor({{{{1.0F, 0.0F}}, {{0.0F, 1.0F}}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{{0.0F, 1.0F}}, {{1.0F, 0.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{1.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::ones({1, 1, 2}, torch::kBool);

    auto loss = pfm::testing::make_graph_matching_loss_for_test(*matcher, descriptors_a, descriptors_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(matcher->parameters().front().grad().defined());
    PFM_REQUIRE(matcher->parameters().front().grad().abs().sum().item<float>() > 0.0F);
}

static void trainer_graph_matching_loss_uses_dustbin_for_unsampled_targets() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    auto descriptors_a = torch::ones({1, 2, 1, 1026}, torch::kFloat32);
    auto descriptors_b = torch::ones({1, 2, 1, 1026}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 1026, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, 1025.0F);
    auto valid_mask = torch::ones({1, 1, 1026}, torch::kBool);

    auto loss = pfm::testing::make_graph_matching_loss_for_test(*matcher, descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
}
```

Replace with:

```cpp
static void trainer_graph_matching_loss_trains_graph_matcher_parameters() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    auto descriptors_a = torch::tensor({{{{1.0F, 0.0F}}, {{0.0F, 1.0F}}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{{0.0F, 1.0F}}, {{1.0F, 0.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{1.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::ones({1, 1, 2}, torch::kBool);

    auto loss = pfm::testing::make_graph_matching_loss_for_test(*matcher, descriptors_a, descriptors_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(matcher->parameters().front().grad().defined());
    PFM_REQUIRE(matcher->parameters().front().grad().abs().sum().item<float>() > 0.0F);
}

static void trainer_graph_matching_loss_is_finite_with_many_descriptors() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    auto descriptors_a = torch::ones({1, 2, 1, 1026}, torch::kFloat32);
    auto descriptors_b = torch::ones({1, 2, 1, 1026}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 1026, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, 1025.0F);
    auto valid_mask = torch::ones({1, 1, 1026}, torch::kBool);

    auto loss = pfm::testing::make_graph_matching_loss_for_test(*matcher, descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
}
```

Update the registration at line 969:

Old:
```cpp
    register_test("trainer_graph_matching_loss_uses_dustbin_for_unsampled_targets",
                  trainer_graph_matching_loss_uses_dustbin_for_unsampled_targets);
```

Replace with:
```cpp
    register_test("trainer_graph_matching_loss_is_finite_with_many_descriptors",
                  trainer_graph_matching_loss_is_finite_with_many_descriptors);
```

- [ ] **Step 3: Build and run tests**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass (284 tests).

- [ ] **Step 4: Commit**

```bash
git add src/train/trainer.cpp src/train/trainer_test.cpp
git commit -m "fix: use nearest-neighbor column mapping in graph matching loss

Replace exact equality with argmin distance search in
map_targets_to_sampled_columns() so that warped target positions
always map to the closest sampled descriptor column instead of
being routed to the dustbin. This fixes the graph_matching_loss
being stuck at ~18 where most targets were incorrectly assigned
to the dustbin column.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Wire ConsoleProgressLogger into training loop

**Files:**
- Modify: `src/train/trainer.cpp:1-40` (add include), `src/train/trainer.cpp:1523-1528` (create logger), `src/train/trainer.cpp:1680-1710` (replace cout with logger)

- [ ] **Step 1: Add include and create ConsoleProgressLogger**

In `src/train/trainer.cpp`, add the include after the existing logging includes (after line 30):

```cpp
#include "logging/progress_logger.h"
```

After the CSV logger creation block (after line 1528), add the progress logger:

```cpp
    ConsoleProgressLogger progress_logger(std::cout, 30);
```

- [ ] **Step 2: Replace per-batch std::cout with progress logger**

Replace the `std::cout << "train progress: ..."` block at lines 1680-1696:

Old:
```cpp
            std::cout << "train progress: epoch=" << epoch + 1 << '/' << config.epochs
                      << " batch=" << (offset / static_cast<std::size_t>(config.batch_size)) + 1 << '/'
                      << (epoch_size + static_cast<std::size_t>(config.batch_size) - 1) /
                             static_cast<std::size_t>(config.batch_size)
                      << " images=" << end << '/' << epoch_size
                      << " loss_total=" << last_loss
                      << " feature_loss=" << feature_loss_value
                      << " repeatability_loss=" << repeatability_loss_value
                      << " descriptor_loss=" << descriptor_loss_value
                      << " matcher_loss=" << graph_matching_loss_value
                      << " graph_matching_loss=" << graph_matching_loss_value
                      << " dense_loss=" << dense_loss_value
                      << " offset_loss=" << offset_loss_value
                      << " confidence_loss=" << confidence_loss_value
                      << " descriptor_accuracy=" << descriptor_accuracy_value
                      << " descriptor_diversity=" << descriptor_diversity_value
                      << " offset_error_px=" << offset_error_value << '\n';
```

Replace with:

```cpp
            TrainingMetric iter_metric;
            iter_metric.epoch = epoch + 1;
            iter_metric.total_epochs = config.epochs;
            iter_metric.iteration = static_cast<int>((offset / static_cast<std::size_t>(config.batch_size)) + 1);
            iter_metric.total_iterations = static_cast<int>(
                (epoch_size + static_cast<std::size_t>(config.batch_size) - 1) /
                static_cast<std::size_t>(config.batch_size));
            iter_metric.images_seen = static_cast<int>(end);
            iter_metric.total_images = static_cast<int>(epoch_size);
            iter_metric.elapsed_seconds = total_timer.elapsedSeconds();
            iter_metric.values["loss_total"] = last_loss;
            iter_metric.values["matcher_loss"] = graph_matching_loss_value;
            iter_metric.values["dense_loss"] = dense_loss_value;
            iter_metric.values["offset_error_px"] = offset_error_value;
            iter_metric.values["descriptor_accuracy"] = descriptor_accuracy_value;
            iter_metric.values["descriptor_diversity"] = descriptor_diversity_value;
            iter_metric.values["offset_loss"] = offset_loss_value;
            iter_metric.values["confidence_loss"] = confidence_loss_value;
            progress_logger.logIteration(iter_metric);
```

- [ ] **Step 3: Replace epoch summary std::cout**

Replace the epoch summary at line 1709:

Old:
```cpp
        std::cout << "train epoch summary: epoch=" << epoch + 1 << '/' << config.epochs
                  << " epoch_time=" << formatSeconds(epoch_timer.elapsedSeconds()) << "s\n";
```

Replace with:

```cpp
        {
            TrainingMetric epoch_metric;
            epoch_metric.epoch = epoch + 1;
            epoch_metric.total_epochs = config.epochs;
            epoch_metric.elapsed_seconds = epoch_timer.elapsedSeconds();
            progress_logger.logEpochSummary(epoch_metric);
        }
```

- [ ] **Step 4: Update trainer_progress_reports_loss_components test**

Since per-batch output now goes through ConsoleProgressLogger (single-line `\r` format) instead of multiline `std::cout`, the test needs to match the new format. Update the test at line 527 in `src/train/trainer_test.cpp`:

Old:
```cpp
static void trainer_progress_reports_loss_components() {
    TempTrainingDirectory temp_dir("pfm_trainer_loss_components");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    CoutCapture capture;
    auto result = pfm::train_model(config);
    const auto output = capture.str();

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(output.find("loss_total=") != std::string::npos);
    PFM_REQUIRE(output.find("feature_loss=") != std::string::npos);
    PFM_REQUIRE(output.find("repeatability_loss=") != std::string::npos);
    PFM_REQUIRE(output.find("descriptor_loss=") != std::string::npos);
    PFM_REQUIRE(output.find("matcher_loss=") != std::string::npos);
    PFM_REQUIRE(output.find("graph_matching_loss=") != std::string::npos);
    PFM_REQUIRE(output.find("dense_loss=") != std::string::npos);
    PFM_REQUIRE(output.find("offset_loss=") != std::string::npos);
    PFM_REQUIRE(output.find("confidence_loss=") != std::string::npos);
    PFM_REQUIRE(output.find("descriptor_accuracy=") != std::string::npos);
    PFM_REQUIRE(output.find("descriptor_diversity=") != std::string::npos);
    PFM_REQUIRE(output.find("offset_error_px=") != std::string::npos);
}
```

Replace with:

```cpp
static void trainer_progress_reports_loss_components() {
    TempTrainingDirectory temp_dir("pfm_trainer_loss_components");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    CoutCapture capture;
    auto result = pfm::train_model(config);
    const auto output = capture.str();

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(output.find("loss=") != std::string::npos);
    PFM_REQUIRE(output.find("matcher=") != std::string::npos);
    PFM_REQUIRE(output.find("dense=") != std::string::npos);
    PFM_REQUIRE(output.find("offset_px=") != std::string::npos);
    PFM_REQUIRE(output.find("epoch summary") != std::string::npos);
}
```

Also update `trainer_reports_epoch_and_batch_timing` at line 554. The old test checks for `"epoch_time="` but the new format uses `"elapsed="`:

Old:
```cpp
    PFM_REQUIRE(output.find("epoch_time=") != std::string::npos);
```

Replace with:
```cpp
    PFM_REQUIRE(output.find("elapsed=") != std::string::npos);
```

- [ ] **Step 5: Build and run tests**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/train/trainer.cpp src/train/trainer_test.cpp
git commit -m "feat: wire ConsoleProgressLogger into training loop

Replace per-batch multiline std::cout output with single-line
ConsoleProgressLogger progress bar showing epoch progress,
loss, matcher, dense, offset_px, and timing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Replace remaining std::cout in pipeline with appropriate logging

**Files:**
- Modify: `src/infer/pipeline.cpp:314,363,466,530,557`

- [ ] **Step 1: Assess pipeline.cpp std::cout usage**

The pipeline.cpp uses `std::cout` for one-time completion messages (not per-batch progress). These are:
- Line 314: `"training complete:"` — after `train_model()` completes
- Line 363: `"extraction complete:"` — after feature extraction
- Line 466: `"matching complete:"` — after matching
- Line 530: `"evaluation complete:"` — after evaluation
- Line 557: `"export complete:"` — after checkpoint export

These are CLI-level status messages, not training progress. They are appropriate as `std::cout` since they're user-facing completion notifications. No change needed.

- [ ] **Step 2: Mark complete**

No changes needed for pipeline.cpp — the one-time status messages are correctly using `std::cout`.

---

### Task 4: Final verification and plan update

- [ ] **Step 1: Full build and test**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./pfm_tests && ctest --output-on-failure
```

Expected: 284+ tests passed, 100% CTest pass.

- [ ] **Step 2: Check no std::cout remains in training hot path**

```bash
grep -n "std::cout" src/train/trainer.cpp
```

Expected: only the visualization info line (one-time message) and no per-batch output.

- [ ] **Step 3: Update task_plan.md and progress.md**

Update the project planning files to record the completion of this phase.

- [ ] **Step 4: Commit**

```bash
git add task_plan.md progress.md findings.md
git commit -m "docs: update planning files for loss fix and progress bar

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
