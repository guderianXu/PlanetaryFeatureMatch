# Keypoint Graph Matching Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `graph_matching_loss` train decoded sparse keypoint matching with real negatives and dustbin targets so the loss can decrease under the user's training command.

**Architecture:** Keep the graph matcher API intact, but make it actually consume normalized keypoint coordinates via the existing `_keypoint_projection`. Replace the trainer's fixed descriptor-grid graph loss with a keypoint-based training helper that decodes sparse features, assigns positives using `warp_a_to_b`, builds a compact deterministic candidate set, and computes row-wise CE over positive/negative/dustbin candidates.

**Tech Stack:** C++17, LibTorch C++ API, OpenCV, existing custom C++ test harness, CMake target `pfm_tests`.

---

## File map

- Modify `src/models/planetary_graph_matcher.cpp`: add keypoint normalization/projection into `PlanetaryGraphMatcherImpl::forward()`.
- Modify `src/models/planetary_graph_matcher_test.cpp`: add regression test proving keypoints affect logits.
- Modify `src/train/trainer.cpp`: add keypoint graph training helpers and replace the current grid-based graph loss call in `training_loss_from_pairs()`.
- Modify `src/train/trainer_test.cpp`: expose helper wrappers for tests and add target assignment/candidate/loss tests.
- No CLI changes: keep low-level graph matching knobs internal for this fix.
- No docs changes beyond the already committed spec unless behavior visible to users changes during implementation.

## Constants and conventions

Use these internal constants in `src/train/trainer.cpp` near the existing training constants:

```cpp
constexpr int64_t GRAPH_MATCHING_MAX_QUERIES = 256;
constexpr int64_t GRAPH_MATCHING_MAX_CANDIDATES = 512;
constexpr int64_t GRAPH_MATCHING_LOCAL_NEGATIVES = 8;
constexpr int64_t GRAPH_MATCHING_RANDOM_NEGATIVES = 32;
constexpr double GRAPH_MATCHING_POSITIVE_RADIUS_PIXELS = 4.0;
```

Keypoints passed to `PlanetaryGraphMatcherImpl::forward()` should be normalized to roughly `[-1, 1]` by the caller when possible. For tests and safety, the matcher should normalize any coordinate set by dividing by its maximum absolute coordinate value, clamped to at least `1.0`, before applying `_keypoint_projection`.

---

### Task 1: Make graph matcher keypoints affect logits

**Files:**
- Modify: `src/models/planetary_graph_matcher_test.cpp`
- Modify: `src/models/planetary_graph_matcher.cpp`

- [ ] **Step 1: Write the failing test**

Add this test in `src/models/planetary_graph_matcher_test.cpp` after `graph_matcher_outputs_logits_matches_and_scores()`:

```cpp
static void graph_matcher_keypoints_affect_logits() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    matcher->eval();
    auto descriptors_a = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_a = torch::tensor({{-1.0F, -1.0F}, {1.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{-1.0F, -1.0F}, {1.0F, 1.0F}}, torch::kFloat32);
    auto shifted_keypoints_b = torch::tensor({{1.0F, 1.0F}, {-1.0F, -1.0F}}, torch::kFloat32);

    const auto original = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b).logits;
    const auto shifted = matcher->forward(descriptors_a, keypoints_a, descriptors_b, shifted_keypoints_b).logits;

    PFM_REQUIRE(!torch::allclose(original, shifted));
}
```

Register it in `register_planetary_graph_matcher_tests()`:

```cpp
register_test("graph matcher keypoints affect logits", graph_matcher_keypoints_affect_logits);
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake --build build-pfm-cf -j$(nproc) && env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" ./build-pfm-cf/pfm_tests
```

Expected: build succeeds, test suite fails with `FAIL graph matcher keypoints affect logits` because keypoints are ignored.

- [ ] **Step 3: Implement minimal keypoint embedding**

In `src/models/planetary_graph_matcher.cpp`, add this helper in the anonymous namespace after `validate_matcher_inputs()`:

```cpp
torch::Tensor normalize_keypoints_for_embedding(const torch::Tensor& keypoints) {
    auto points = keypoints.to(torch::TensorOptions().dtype(torch::kFloat32).device(keypoints.device()));
    if (points.numel() == 0) {
        return points;
    }
    auto scale = points.abs().amax().clamp_min(1.0);
    return points / scale;
}
```

Then replace these lines in `PlanetaryGraphMatcherImpl::forward()`:

```cpp
(void)keypoints_a;
(void)keypoints_b;
auto embed_a = torch::relu(_descriptor_projection(desc_a));
auto embed_b = torch::relu(_descriptor_projection(desc_b));
```

with:

```cpp
auto kp_a = normalize_keypoints_for_embedding(keypoints_a).to(desc_a.device());
auto kp_b = normalize_keypoints_for_embedding(keypoints_b).to(desc_b.device());
auto embed_a = torch::relu(_descriptor_projection(desc_a) + _keypoint_projection(kp_a));
auto embed_b = torch::relu(_descriptor_projection(desc_b) + _keypoint_projection(kp_b));
```

- [ ] **Step 4: Run test to verify GREEN**

Run the same command from Step 2.

Expected: all existing tests plus the new keypoint test pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/planetary_graph_matcher.cpp src/models/planetary_graph_matcher_test.cpp
git commit -m "Use keypoints in graph matcher"
```

---

### Task 2: Add keypoint assignment helper and tests

**Files:**
- Modify: `src/train/trainer.cpp`
- Modify: `src/train/trainer_test.cpp`

- [ ] **Step 1: Write failing helper declarations and tests**

In `src/train/trainer_test.cpp`, add declarations in `namespace pfm::testing` after `make_graph_matching_loss_for_test(...)`:

```cpp
torch::Tensor assign_graph_matching_targets_for_test(
    const torch::Tensor& keypoints_a,
    const torch::Tensor& keypoints_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double positive_radius_pixels);
```

Add these tests after `trainer_graph_matching_loss_is_finite_with_many_descriptors()`:

```cpp
static void trainer_keypoint_graph_targets_use_warped_nearest_b_keypoint() {
    auto keypoints_a = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{5.0F, 1.0F}, {7.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto targets = pfm::testing::assign_graph_matching_targets_for_test(
        keypoints_a, keypoints_b, warp, valid_mask, 2.0);

    PFM_REQUIRE(targets.sizes() == std::vector<int64_t>({2}));
    PFM_REQUIRE(targets[0].item<int64_t>() == 0);
    PFM_REQUIRE(targets[1].item<int64_t>() == 1);
}

static void trainer_keypoint_graph_targets_use_dustbin_for_unmatched_keypoints() {
    auto keypoints_a = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{6.0F, 6.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto targets = pfm::testing::assign_graph_matching_targets_for_test(
        keypoints_a, keypoints_b, warp, valid_mask, 1.0);

    PFM_REQUIRE(targets[0].item<int64_t>() == 1);
    PFM_REQUIRE(targets[1].item<int64_t>() == 1);
}
```

Register them:

```cpp
register_test(
    "trainer keypoint graph targets use warped nearest b keypoint",
    trainer_keypoint_graph_targets_use_warped_nearest_b_keypoint);
register_test(
    "trainer keypoint graph targets use dustbin for unmatched keypoints",
    trainer_keypoint_graph_targets_use_dustbin_for_unmatched_keypoints);
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake --build build-pfm-cf -j$(nproc)
```

Expected: link fails because `assign_graph_matching_targets_for_test` is declared but not defined.

- [ ] **Step 3: Implement assignment helper**

In `src/train/trainer.cpp`, add this helper near the existing `make_graph_matching_loss()` helpers:

```cpp
torch::Tensor assign_graph_matching_targets(
    const torch::Tensor& keypoints_a,
    const torch::Tensor& keypoints_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double positive_radius_pixels
) {
    using torch::indexing::Slice;

    const auto dustbin = keypoints_b.size(0);
    auto targets = torch::full(
        {keypoints_a.size(0)},
        dustbin,
        torch::TensorOptions().dtype(torch::kLong).device(keypoints_a.device()));
    if (keypoints_a.size(0) == 0 || keypoints_b.size(0) == 0) {
        return targets;
    }

    auto points_a_cpu = keypoints_a.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b_cpu = keypoints_b.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    auto warp_cpu = warp.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    auto mask_cpu = valid_mask.detach().to(torch::kCPU, torch::kBool).contiguous();
    std::vector<int64_t> labels(static_cast<size_t>(keypoints_a.size(0)), dustbin);
    const auto radius_sq = positive_radius_pixels * positive_radius_pixels;

    for (int64_t index = 0; index < points_a_cpu.size(0); ++index) {
        const auto x = static_cast<int64_t>(std::llround(points_a_cpu.index({index, 0}).item<float>()));
        const auto y = static_cast<int64_t>(std::llround(points_a_cpu.index({index, 1}).item<float>()));
        if (y < 0 || y >= warp_cpu.size(1) || x < 0 || x >= warp_cpu.size(2)) {
            continue;
        }
        if (!mask_cpu.index({0, y, x}).item<bool>()) {
            continue;
        }
        const auto expected_x = warp_cpu.index({0, y, x, 0}).item<float>();
        const auto expected_y = warp_cpu.index({0, y, x, 1}).item<float>();
        int64_t best = dustbin;
        double best_distance = radius_sq;
        for (int64_t candidate = 0; candidate < points_b_cpu.size(0); ++candidate) {
            const auto dx = static_cast<double>(points_b_cpu.index({candidate, 0}).item<float>()) - expected_x;
            const auto dy = static_cast<double>(points_b_cpu.index({candidate, 1}).item<float>()) - expected_y;
            const auto distance = dx * dx + dy * dy;
            if (distance <= best_distance) {
                best_distance = distance;
                best = candidate;
            }
        }
        labels[static_cast<size_t>(index)] = best;
    }

    return torch::tensor(labels, targets.options());
}
```

Add the test wrapper in `namespace testing`:

```cpp
torch::Tensor assign_graph_matching_targets_for_test(
    const torch::Tensor& keypoints_a,
    const torch::Tensor& keypoints_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double positive_radius_pixels
) {
    return assign_graph_matching_targets(keypoints_a, keypoints_b, warp, valid_mask, positive_radius_pixels);
}
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake --build build-pfm-cf -j$(nproc) && env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" ./build-pfm-cf/pfm_tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/train/trainer.cpp src/train/trainer_test.cpp
git commit -m "Add graph matching target assignment"
```

---

### Task 3: Add deterministic candidate construction

**Files:**
- Modify: `src/train/trainer.cpp`
- Modify: `src/train/trainer_test.cpp`

- [ ] **Step 1: Write failing tests for candidate construction**

In `src/train/trainer_test.cpp`, add this declaration after `assign_graph_matching_targets_for_test(...)`:

```cpp
torch::Tensor make_graph_candidate_indices_for_test(
    const torch::Tensor& target_indices,
    int64_t keypoint_count_b,
    int64_t max_candidates);
```

Add this test after the graph target tests:

```cpp
static void trainer_graph_candidates_include_positives_once_and_dustbin_last() {
    auto target_indices = torch::tensor({0, 2, 2, 5}, torch::kLong);

    auto candidates = pfm::testing::make_graph_candidate_indices_for_test(target_indices, 5, 6);

    PFM_REQUIRE(candidates.size(0) == 6);
    PFM_REQUIRE(candidates[-1].item<int64_t>() == 5);
    PFM_REQUIRE((candidates == 0).sum().item<int64_t>() == 1);
    PFM_REQUIRE((candidates == 2).sum().item<int64_t>() == 1);
    PFM_REQUIRE((candidates == 5).sum().item<int64_t>() == 1);
}
```

Register it:

```cpp
register_test(
    "trainer graph candidates include positives once and dustbin last",
    trainer_graph_candidates_include_positives_once_and_dustbin_last);
```

- [ ] **Step 2: Run build to verify RED**

Run:

```bash
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake --build build-pfm-cf -j$(nproc)
```

Expected: link fails because `make_graph_candidate_indices_for_test` is missing.

- [ ] **Step 3: Implement candidate helper**

Add this helper in `src/train/trainer.cpp` after `assign_graph_matching_targets()`:

```cpp
torch::Tensor make_graph_candidate_indices(
    const torch::Tensor& target_indices,
    int64_t keypoint_count_b,
    int64_t max_candidates
) {
    const auto dustbin = keypoint_count_b;
    std::vector<int64_t> candidates;
    candidates.reserve(static_cast<size_t>(std::max<int64_t>(1, max_candidates)));

    auto targets_cpu = target_indices.detach().to(torch::kCPU, torch::kLong).contiguous();
    for (int64_t index = 0; index < targets_cpu.numel(); ++index) {
        const auto label = targets_cpu[index].item<int64_t>();
        if (label >= 0 && label < keypoint_count_b &&
            std::find(candidates.begin(), candidates.end(), label) == candidates.end()) {
            candidates.push_back(label);
        }
    }

    for (int64_t candidate = 0; candidate < keypoint_count_b &&
         static_cast<int64_t>(candidates.size()) < max_candidates - 1; ++candidate) {
        if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
            candidates.push_back(candidate);
        }
    }

    candidates.push_back(dustbin);
    return torch::tensor(candidates, torch::TensorOptions().dtype(torch::kLong).device(target_indices.device()));
}
```

Add `#include <algorithm>` at the top of `src/train/trainer.cpp` if it is not already present.

Add wrapper in `namespace testing`:

```cpp
torch::Tensor make_graph_candidate_indices_for_test(
    const torch::Tensor& target_indices,
    int64_t keypoint_count_b,
    int64_t max_candidates
) {
    return make_graph_candidate_indices(target_indices, keypoint_count_b, max_candidates);
}
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake --build build-pfm-cf -j$(nproc) && env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" ./build-pfm-cf/pfm_tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/train/trainer.cpp src/train/trainer_test.cpp
git commit -m "Build graph matching candidate sets"
```

---

### Task 4: Replace grid graph loss with decoded keypoint graph loss

**Files:**
- Modify: `src/train/trainer.cpp`
- Modify: `src/train/trainer_test.cpp`

- [ ] **Step 1: Write failing loss behavior test**

In `src/train/trainer_test.cpp`, add this declaration after `make_graph_candidate_indices_for_test(...)`:

```cpp
torch::Tensor make_keypoint_graph_matching_loss_for_test(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
```

Add this test after the candidate construction test:

```cpp
static void trainer_keypoint_graph_matching_loss_trains_graph_matcher_parameters() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    pfm::FeatureSet features_a;
    features_a.keypoints = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    features_a.scores = torch::tensor({1.0F, 0.9F}, torch::kFloat32);
    features_a.descriptors = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    pfm::FeatureSet features_b;
    features_b.keypoints = torch::tensor({{5.0F, 1.0F}, {7.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32);
    features_b.scores = torch::tensor({1.0F, 0.9F, 0.1F}, torch::kFloat32);
    features_b.descriptors = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {0.5F, 0.5F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto loss = pfm::testing::make_keypoint_graph_matching_loss_for_test(
        *matcher, features_a, features_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
    PFM_REQUIRE(matcher->parameters().front().grad().defined());
    PFM_REQUIRE(matcher->parameters().front().grad().abs().sum().item<float>() > 0.0F);
}
```

Register it:

```cpp
register_test(
    "trainer keypoint graph matching loss trains graph matcher parameters",
    trainer_keypoint_graph_matching_loss_trains_graph_matcher_parameters);
```

- [ ] **Step 2: Run build to verify RED**

Run:

```bash
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake --build build-pfm-cf -j$(nproc)
```

Expected: link fails because `make_keypoint_graph_matching_loss_for_test` is missing.

- [ ] **Step 3: Implement keypoint graph loss helper**

Add this struct near `DescriptorTrainingMetrics` in `src/train/trainer.cpp`:

```cpp
struct GraphMatchingTrainingMetrics {
    torch::Tensor loss;
    torch::Tensor accuracy;
    int64_t query_count = 0;
    int64_t positive_count = 0;
    int64_t dustbin_count = 0;
};
```

Add this helper after `make_graph_candidate_indices()`:

```cpp
GraphMatchingTrainingMetrics make_keypoint_graph_matching_metrics(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    auto zero = torch::zeros({}, warp.options());
    if (!features_a.keypoints.defined() || !features_b.keypoints.defined() ||
        features_a.keypoints.size(0) == 0 || features_b.keypoints.size(0) == 0) {
        return GraphMatchingTrainingMetrics{zero, zero, 0, 0, 0};
    }

    const auto query_count = std::min<int64_t>(features_a.keypoints.size(0), GRAPH_MATCHING_MAX_QUERIES);
    auto query_indices = torch::arange(query_count, torch::TensorOptions().dtype(torch::kLong).device(features_a.keypoints.device()));
    auto keypoints_a = features_a.keypoints.index_select(0, query_indices).to(warp.device());
    auto descriptors_a = features_a.descriptors.index_select(0, query_indices).to(warp.device());
    auto keypoints_b = features_b.keypoints.to(warp.device());
    auto descriptors_b = features_b.descriptors.to(warp.device());

    auto target_full = assign_graph_matching_targets(
        keypoints_a,
        keypoints_b,
        warp,
        valid_mask,
        GRAPH_MATCHING_POSITIVE_RADIUS_PIXELS);
    auto candidate_indices = make_graph_candidate_indices(
        target_full,
        keypoints_b.size(0),
        std::min<int64_t>(GRAPH_MATCHING_MAX_CANDIDATES, keypoints_b.size(0) + 1));

    auto candidate_keypoints = keypoints_b.index_select(0, candidate_indices.narrow(0, 0, candidate_indices.size(0) - 1));
    auto candidate_descriptors = descriptors_b.index_select(0, candidate_indices.narrow(0, 0, candidate_indices.size(0) - 1));
    auto remapped_targets = torch::full(
        {target_full.size(0)},
        candidate_indices.size(0) - 1,
        torch::TensorOptions().dtype(torch::kLong).device(target_full.device()));
    for (int64_t col = 0; col < candidate_indices.size(0) - 1; ++col) {
        remapped_targets.index_put_({target_full == candidate_indices[col]}, col);
    }

    auto output = graph_matcher.forward(descriptors_a, keypoints_a, candidate_descriptors, candidate_keypoints);
    auto loss = graph_matching_cross_entropy_loss(output.logits, remapped_targets);
    auto predictions = output.logits.narrow(0, 0, remapped_targets.size(0)).argmax(1);
    auto accuracy = predictions.eq(remapped_targets).to(torch::kFloat32).mean();
    const auto dustbin_label = candidate_indices.size(0) - 1;
    const auto dustbin_count = remapped_targets.eq(dustbin_label).sum().item<int64_t>();
    return GraphMatchingTrainingMetrics{
        loss,
        accuracy,
        remapped_targets.size(0),
        remapped_targets.size(0) - dustbin_count,
        dustbin_count};
}
```

Add a convenience loss wrapper:

```cpp
torch::Tensor make_keypoint_graph_matching_loss(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    return make_keypoint_graph_matching_metrics(graph_matcher, features_a, features_b, warp, valid_mask).loss;
}
```

Add the test wrapper in `namespace testing`:

```cpp
torch::Tensor make_keypoint_graph_matching_loss_for_test(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    return make_keypoint_graph_matching_loss(graph_matcher, features_a, features_b, warp, valid_mask);
}
```

- [ ] **Step 4: Wire training_loss_from_pairs to decoded features**

In `training_loss_from_pairs()`, after `const auto dense = modules.dense_head->forward(...)`, create decode config and decoded feature vectors:

```cpp
auto decode_config = FeatureDecodeConfig{};
decode_config.max_keypoints = 1024;
decode_config.min_keypoints = 0;
decode_config.nms_radius = 4;
```

Then, before computing `graph_matching`, replace:

```cpp
auto graph_matching = make_graph_matching_loss(*modules.graph_matcher, sparse_a.descriptors, sparse_b.descriptors, warp, valid_mask);
```

with batch-wise decoded feature loss:

```cpp
std::vector<torch::Tensor> graph_losses;
graph_losses.reserve(pairs.size());
for (int64_t batch = 0; batch < view_a.size(0); ++batch) {
    auto features_a = decode_features(
        sparse_a.heatmap.index({batch}).detach(),
        sparse_a.descriptors.index({batch}),
        sparse_a.scale.index({batch}).detach(),
        sparse_a.orientation.index({batch}).detach(),
        sparse_a.affine.index({batch}).detach(),
        dense.confidence.index({batch}).detach(),
        dense.offsets.index({batch}).detach(),
        decode_config);
    auto features_b = decode_features(
        sparse_b.heatmap.index({batch}).detach(),
        sparse_b.descriptors.index({batch}),
        sparse_b.scale.index({batch}).detach(),
        sparse_b.orientation.index({batch}).detach(),
        sparse_b.affine.index({batch}).detach(),
        dense.confidence.index({batch}).detach(),
        dense.offsets.index({batch}).detach(),
        decode_config);
    graph_losses.push_back(make_keypoint_graph_matching_loss(
        *modules.graph_matcher,
        features_a,
        features_b,
        warp.index({batch}).unsqueeze(0),
        valid_mask.index({batch}).unsqueeze(0)));
}
auto graph_matching = graph_losses.empty()
    ? torch::zeros({}, sparse_a.descriptors.options())
    : torch::stack(graph_losses).mean();
```

Add `#include "infer/feature_extractor.h"` if `decode_features` and `FeatureDecodeConfig` are not already visible in this file.

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake --build build-pfm-cf -j$(nproc) && env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" ./build-pfm-cf/pfm_tests
```

Expected: all tests pass. If compilation fails because `decode_features` expects tensors on CPU, change the helper to move decoded `FeatureSet` tensors back to `warp.device()` inside `make_keypoint_graph_matching_metrics()`.

- [ ] **Step 6: Commit**

```bash
git add src/train/trainer.cpp src/train/trainer_test.cpp
git commit -m "Train graph matcher on decoded keypoints"
```

---

### Task 5: Verify training behavior and clean diagnostics

**Files:**
- No source changes expected.
- Generated files to keep untracked: `metrics_debug.csv`, `pair_cache_debug/`, `vis_debug/`, `train_debug.pt`.

- [ ] **Step 1: Run full build and tests**

Run:

```bash
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake -S . -B build-pfm-cf -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH="/home/guderian/anaconda3/envs/pfm-cf" -DCMAKE_CUDA_COMPILER=/home/guderian/anaconda3/envs/pfm-cf/bin/nvcc -DCMAKE_CXX_COMPILER=/usr/bin/c++ && env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" cmake --build build-pfm-cf -j$(nproc) && env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" ./build-pfm-cf/pfm_tests && env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" ctest --test-dir build-pfm-cf --output-on-failure
```

Expected: `pfm_tests` reports all tests passed and CTest reports `100% tests passed`.

- [ ] **Step 2: Run short reproduction training**

Run:

```bash
rm -rf pair_cache_debug vis_debug train_debug.pt metrics_debug.csv
env -i HOME="$HOME" PATH="/home/guderian/anaconda3/envs/pfm-cf/bin:/home/guderian/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" LD_LIBRARY_PATH="/home/guderian/anaconda3/envs/pfm-cf/lib" ./build-pfm-cf/pfm_cli train --image-dir ./img/ --checkpoint train_debug.pt --epochs 3 --batch-size 2 --resize 512 --device cuda --synthetic-pair-cache-dir ./pair_cache_debug --pairs-per-image 15 --augmentation-profile mixed --min-keypoint-intensity 0.05 --visualization-dir vis_debug --visualization-samples 2 --min-keypoints 1024 --log-csv metrics_debug.csv
```

Expected: training completes without runtime errors and writes `metrics_debug.csv`.

- [ ] **Step 3: Summarize graph loss trend**

Run:

```bash
python3 - <<'PY'
import csv, statistics
rows=[]
with open('metrics_debug.csv') as f:
    for row in csv.DictReader(f):
        if row.get('type') == 'epoch':
            continue
        rows.append(row)
values=[float(r['graph_matching_loss']) for r in rows]
window=min(100, max(1, len(values)//5))
print('rows', len(values))
print('first_mean', statistics.mean(values[:window]))
print('last_mean', statistics.mean(values[-window:]))
print('min', min(values))
print('max', max(values))
PY
```

Expected: `last_mean` is lower than `first_mean`. If the loss still does not trend down, do not add more fixes; report that this design did not solve the training signal and propose the next architecture step from the spec: bidirectional/Sinkhorn-style objective.

- [ ] **Step 4: Check Git status**

Run:

```bash
git status --short --branch
```

Expected: only generated debug artifacts are untracked. Source files should be clean after the previous commits.

---

## Self-review notes

- Spec coverage: Task 1 covers keypoint embedding. Tasks 2-4 cover decoded keypoint supervision, positive assignment, candidate set construction, dustbin semantics, and graph CE. Task 5 covers build/test and short-run metric verification.
- Scope: This plan does not redesign backbone, dense head, augmentation, or CLI knobs.
- Known implementation risk: `decode_features()` is currently used mostly for inference/visualization. If it detaches too much for graph training, keep descriptors with gradients by gathering descriptors at decoded keypoint feature-map positions instead of relying on detached descriptor tensors. That adjustment belongs inside Task 4 only.
