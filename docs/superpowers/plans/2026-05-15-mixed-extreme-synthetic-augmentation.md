# 混合强度合成增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 synthetic pair 缓存和在线训练按 `mixed/mild/medium/hard/extreme` profile 生成明显不同、可复现且监督 warp 正确的增强匹配对。

**Architecture:** 在 `modules/data/synthetic_pair.*` 中加入 profile 与 deterministic variant 解析，保留 `make_synthetic_pair()` 作为生成入口。CLI/pipeline/trainer 只负责传递 `augmentation_profile` 和 `extreme_pair_ratio`，cache manifest 负责记录这些参数并触发重建。

**Tech Stack:** C++17, LibTorch tensors/serialization, OpenCV PNG IO, CLI11, project `pfm_tests` harness.

---

## File Structure

- Modify `modules/data/synthetic_pair.h`: add profile enum/config fields and public parse/name helpers.
- Modify `modules/data/synthetic_pair.cpp`: implement deterministic profile selection, stronger affine/photometric variants, gamma and gradient shadow.
- Modify `modules/data/synthetic_pair_test.cpp`: profile strength and valid mask tests.
- Modify `modules/data/synthetic_pair_cache.h/.cpp`: persist profile/ratio in manifest and pass `source_index` into variant generation.
- Modify `modules/data/synthetic_pair_cache_test.cpp`: cache rebuild and varied pair assertions.
- Modify `modules/train/trainer.h/.cpp`: add config fields, validate ratio/profile, pass source/variant into pair generation.
- Modify `modules/train/trainer_test.cpp`: invalid ratio/profile and cache expansion tests.
- Modify `modules/cli/commands.h/.cpp` and `modules/cli/commands_test.cpp`: parse new CLI parameters and help text.
- Modify `modules/infer/pipeline.cpp/.test.cpp`: forward CLI options into `TrainConfig`.
- Modify `README.md`, `docs/training.md`, `docs/usage.md`: Chinese usage and risk notes.

---

### Task 1: CLI and train config parameters

**Files:**
- Modify: `modules/data/synthetic_pair.h`
- Modify: `modules/cli/commands.h`
- Modify: `modules/cli/commands.cpp`
- Modify: `modules/cli/commands_test.cpp`
- Modify: `modules/train/trainer.h`
- Modify: `modules/train/trainer.cpp`
- Modify: `modules/infer/pipeline.cpp`
- Modify: `modules/infer/pipeline_test.cpp`

- [ ] **Step 1: Write failing CLI parse test**

In `modules/cli/commands_test.cpp`, extend `parse_train_command()` arguments:

```cpp
"--pairs-per-image",
"3",
"--augmentation-profile",
"hard",
"--extreme-pair-ratio",
"0.35",
"--synthetic-pair-cache-dir",
```

Add assertions:

```cpp
PFM_REQUIRE(parsed.pairs_per_image == 3);
PFM_REQUIRE(parsed.augmentation_profile == "hard");
PFM_REQUIRE_CLOSE(parsed.extreme_pair_ratio, 0.35, 1.0e-6);
```

Also extend `top_level_help_lists_subcommand_options()`:

```cpp
PFM_REQUIRE(help.find("--augmentation-profile") != std::string::npos);
PFM_REQUIRE(help.find("--extreme-pair-ratio") != std::string::npos);
```

- [ ] **Step 2: Run red build**

Run:

```bash
cmake --build build -j$(nproc)
```

Expected: FAIL because `CliOptions` has no `augmentation_profile` or `extreme_pair_ratio`.

- [ ] **Step 3: Add CLI fields and options**

In `modules/cli/commands.h`, add fields:

```cpp
std::string augmentation_profile = "mixed";
double extreme_pair_ratio = 0.2;
```

In `modules/cli/commands.cpp`, update footer train line:

```cpp
"[--resize 512] [--pairs-per-image 1] [--augmentation-profile mixed] "
"[--extreme-pair-ratio 0.2] [--synthetic-pair-cache-dir build/pair_cache] "
```

Add options after `--pairs-per-image`:

```cpp
train->add_option("--augmentation-profile",
                  options.augmentation_profile,
                  "Synthetic augmentation profile: mixed, mild, medium, hard, or extreme");
train->add_option("--extreme-pair-ratio",
                  options.extreme_pair_ratio,
                  "Extreme pair ratio used by mixed augmentation profile");
```

- [ ] **Step 4: Add TrainConfig fields and pipeline forwarding**

In `modules/train/trainer.h` add:

```cpp
std::string augmentation_profile = "mixed";
double extreme_pair_ratio = 0.2;
```

In `modules/infer/pipeline.cpp` inside `run_train_command()` add:

```cpp
config.augmentation_profile = options.augmentation_profile;
config.extreme_pair_ratio = options.extreme_pair_ratio;
```

- [ ] **Step 5: Write pipeline forwarding test**

In `modules/infer/pipeline_test.cpp`, in `pipeline_train_forwards_pairs_per_image_to_cache_generation()`, set:

```cpp
options.augmentation_profile = "extreme";
options.extreme_pair_ratio = 0.4;
```

The existing check for `pair_000003.pt` remains; later cache manifest tests prove these values are persisted.

- [ ] **Step 6: Run green build/test subset**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: build succeeds; tests may still fail only where profile parsing is not validated or cache manifest is not updated in later tasks.

---

### Task 2: Deterministic strong augmentation profiles

**Files:**
- Modify: `modules/data/synthetic_pair.h`
- Modify: `modules/data/synthetic_pair.cpp`
- Modify: `modules/data/synthetic_pair_test.cpp`
- Modify: `modules/train/trainer.cpp`

- [ ] **Step 1: Write failing profile strength tests**

In `modules/data/synthetic_pair_test.cpp`, add helper:

```cpp
torch::Tensor mean_warp_displacement(const pfm::SyntheticPair& pair) {
    auto source_grid = pfm::make_xy_grid(pair.warp_a_to_b.size(0), pair.warp_a_to_b.size(1), pair.warp_a_to_b.device());
    auto delta = pair.warp_a_to_b - source_grid;
    return delta.pow(2).sum(2).sqrt().masked_select(pair.valid_mask).mean();
}
```

Add test:

```cpp
static void synthetic_pair_extreme_profile_is_stronger_than_mild() {
    auto image = torch::linspace(0.0F, 1.0F, 64 * 64).reshape({1, 64, 64});
    pfm::SyntheticPairConfig mild;
    mild.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Mild;
    mild.variant_index = 2;
    pfm::SyntheticPairConfig extreme;
    extreme.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Extreme;
    extreme.variant_index = 2;

    auto mild_pair = pfm::make_synthetic_pair(image, mild);
    auto extreme_pair = pfm::make_synthetic_pair(image, extreme);

    PFM_REQUIRE(mean_warp_displacement(extreme_pair).item<float>() > mean_warp_displacement(mild_pair).item<float>() + 3.0F);
    PFM_REQUIRE(extreme_pair.valid_mask.to(torch::kFloat32).mean().item<float>() > 0.15F);
}
```

Add test:

```cpp
static void synthetic_pair_mixed_profile_varies_variant_strength() {
    auto image = torch::linspace(0.0F, 1.0F, 64 * 64).reshape({1, 64, 64});
    pfm::SyntheticPairConfig config;
    config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Mixed;
    config.extreme_pair_ratio = 0.25;
    config.variant_index = 1;
    auto first = pfm::make_synthetic_pair(image, config);
    config.variant_index = 5;
    auto later = pfm::make_synthetic_pair(image, config);

    PFM_REQUIRE(!torch::allclose(first.view_b, later.view_b));
    PFM_REQUIRE(!torch::allclose(first.warp_a_to_b, later.warp_a_to_b));
}
```

Register both tests.

- [ ] **Step 2: Run red tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: FAIL because `SyntheticPairAugmentationProfile` and config fields do not exist or do not affect strength enough.

- [ ] **Step 3: Add profile enum and config fields**

In `modules/data/synthetic_pair.h`, add:

```cpp
enum class SyntheticPairAugmentationProfile {
    Mixed,
    Mild,
    Medium,
    Hard,
    Extreme,
};

SyntheticPairAugmentationProfile parse_synthetic_pair_augmentation_profile(const std::string& value);
std::string synthetic_pair_augmentation_profile_name(SyntheticPairAugmentationProfile profile);
```

Add to `SyntheticPairConfig`:

```cpp
SyntheticPairAugmentationProfile augmentation_profile = SyntheticPairAugmentationProfile::Mixed;
double extreme_pair_ratio = 0.2;
int64_t source_index = 0;
```

- [ ] **Step 4: Implement deterministic profile parameter resolver**

In `modules/data/synthetic_pair.cpp`, replace the weak `resolve_variant_config()` with a resolver using fixed profile ranges:

```cpp
float deterministic_unit(int64_t source_index, int64_t variant_index, float salt) {
    const float value = std::sin(static_cast<float>(source_index + 1) * 12.9898F +
                                 static_cast<float>(variant_index + 1) * 78.233F + salt) * 43758.5453F;
    return value - std::floor(value);
}

float signed_unit(int64_t source_index, int64_t variant_index, float salt) {
    return deterministic_unit(source_index, variant_index, salt) * 2.0F - 1.0F;
}
```

Use ranges:

```cpp
// Mild: rotation ±8, scale 0.95..1.05, shear ±0.04, translation up to 6% edge.
// Medium: rotation ±25, scale 0.80..1.20, shear ±0.12, translation up to 12% edge.
// Hard: rotation ±55, scale 0.65..1.45, shear ±0.25, translation up to 20% edge.
// Extreme: rotation ±85, scale 0.50..1.70, shear ±0.38, translation up to 25% edge.
```

Clamp generated translation to integer values because existing validation requires integer translations.

- [ ] **Step 5: Add affine shear / anisotropic scale consistently**

Keep `AffineTransform` as the single source of geometry. Build matrix components in `make_pair_transform()` so `affine_warp_chw()` and `dense_warp_field()` use identical transform. The transform should include rotation, uniform scale, anisotropic scale, shear, and translation.

- [ ] **Step 6: Add photometric profile effects**

After warping, apply:

```cpp
view_b = clamp_unit(view_b.pow(gamma) * contrast_scale + brightness_delta);
```

Add deterministic gradient shadow:

```cpp
shadow = 1.0F + shadow_strength * signed_gradient;
view_b = clamp_unit(view_b * shadow);
```

Use stronger `gamma`, contrast, brightness, and shadow ranges for hard/extreme.

- [ ] **Step 7: Pass source index in trainer/cache**

In cache generation:

```cpp
pair_config.source_index = static_cast<int64_t>(source_index);
pair_config.variant_index = static_cast<int64_t>(variant_index);
```

In online trainer generation, pass both `source_index` and `variant_index` into `make_synthetic_pair()`.

- [ ] **Step 8: Run green tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: profile strength tests pass and previous synthetic pair tests still pass for baseline variant 0.

---

### Task 3: Cache manifest and rebuild behavior

**Files:**
- Modify: `modules/data/synthetic_pair_cache.h`
- Modify: `modules/data/synthetic_pair_cache.cpp`
- Modify: `modules/data/synthetic_pair_cache_test.cpp`
- Modify: `modules/train/trainer.cpp`

- [ ] **Step 1: Write failing cache rebuild tests**

In `modules/data/synthetic_pair_cache_test.cpp`, add:

```cpp
static void synthetic_pair_cache_rebuilds_when_augmentation_profile_changes() {
    TempCacheDirectory temp_dir("pfm_pair_cache_profile_rebuild");
    auto dataset = make_dataset(temp_dir);
    auto config = make_cache_config(temp_dir, 2);
    config.pair_config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Mild;
    pfm::prepare_synthetic_pair_cache(dataset, config);
    const auto pair_path = std::filesystem::path(config.cache_dir) / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.pair_config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Hard;
    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) != first_write_time);
}
```

Add:

```cpp
static void synthetic_pair_cache_rebuilds_when_extreme_ratio_changes() {
    TempCacheDirectory temp_dir("pfm_pair_cache_ratio_rebuild");
    auto dataset = make_dataset(temp_dir);
    auto config = make_cache_config(temp_dir, 2);
    config.pair_config.extreme_pair_ratio = 0.1;
    pfm::prepare_synthetic_pair_cache(dataset, config);
    const auto pair_path = std::filesystem::path(config.cache_dir) / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.pair_config.extreme_pair_ratio = 0.4;
    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) != first_write_time);
}
```

Register both tests.

- [ ] **Step 2: Run red tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: FAIL because manifest does not record profile/ratio yet.

- [ ] **Step 3: Write manifest fields**

In `write_manifest()` add:

```cpp
write_int64(archive,
            "augmentation_profile",
            static_cast<int64_t>(config.pair_config.augmentation_profile));
write_float(archive,
            "extreme_pair_ratio",
            static_cast<float>(config.pair_config.extreme_pair_ratio));
```

- [ ] **Step 4: Compare manifest fields**

In `manifest_matches()` add comparisons for profile and ratio:

```cpp
read_int64(archive, "augmentation_profile") ==
    static_cast<int64_t>(config.pair_config.augmentation_profile)
```

and:

```cpp
float_matches(read_float(archive, "extreme_pair_ratio"),
              static_cast<float>(config.pair_config.extreme_pair_ratio))
```

- [ ] **Step 5: Run green tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: cache rebuild tests pass.

---

### Task 4: Trainer validation, docs, and final verification

**Files:**
- Modify: `modules/train/trainer.cpp`
- Modify: `modules/train/trainer_test.cpp`
- Modify: `README.md`
- Modify: `docs/training.md`
- Modify: `docs/usage.md`

- [ ] **Step 1: Write failing trainer validation test**

In `modules/train/trainer_test.cpp`, extend `trainer_invalid_numeric_parameters_throw_invalid_argument()`:

```cpp
auto invalid_extreme_ratio_low = config;
invalid_extreme_ratio_low.extreme_pair_ratio = -0.1;
PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_extreme_ratio_low));

auto invalid_extreme_ratio_high = config;
invalid_extreme_ratio_high.extreme_pair_ratio = 1.1;
PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_extreme_ratio_high));

auto invalid_profile = config;
invalid_profile.augmentation_profile = "unknown";
PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_profile));
```

- [ ] **Step 2: Run red tests**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: FAIL until trainer validates and parses profile.

- [ ] **Step 3: Implement trainer validation and default pair config**

In `validate_config()` add:

```cpp
if (config.extreme_pair_ratio < 0.0 || config.extreme_pair_ratio > 1.0) {
    throw std::invalid_argument("extreme_pair_ratio must be between 0 and 1");
}
(void)parse_synthetic_pair_augmentation_profile(config.augmentation_profile);
```

In `make_default_pair_config(const TrainConfig& config)` set:

```cpp
pair_config.augmentation_profile = parse_synthetic_pair_augmentation_profile(config.augmentation_profile);
pair_config.extreme_pair_ratio = config.extreme_pair_ratio;
```

Update callers from `make_default_pair_config()` to `make_default_pair_config(config)`.

- [ ] **Step 4: Update Chinese docs**

In `README.md`, `docs/training.md`, and `docs/usage.md`, document:

```bash
--augmentation-profile mixed
--extreme-pair-ratio 0.2
```

Mention that `mixed` is recommended, `hard/extreme` may raise early loss, and cached PNG should be inspected for transform strength.

- [ ] **Step 5: Run final verification**

Run:

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
./build/pfm_cli train --help
```

Expected:

- Build exits 0.
- `pfm_tests` reports all tests passed.
- `ctest` reports 100% tests passed.
- train help includes `--augmentation-profile` and `--extreme-pair-ratio`.

---

## Self-Review

Spec coverage:

- Multiple pair diversity: Task 2 tests and implementation.
- Mixed default profile: Task 2 resolver and Task 4 trainer default config.
- CLI knobs: Task 1 and Task 4 docs.
- Cache manifest: Task 3.
- Warp/mask correctness: Task 2 keeps geometry in `AffineTransform`, tested by valid mask and existing warp tests.
- TDD: Each task starts with failing tests and red run.

Placeholder scan: no TBD/TODO placeholders.

Type consistency: plan uses `SyntheticPairAugmentationProfile`, `augmentation_profile`, `extreme_pair_ratio`, `source_index`, and `variant_index` consistently across CLI, TrainConfig, SyntheticPairConfig, and cache.