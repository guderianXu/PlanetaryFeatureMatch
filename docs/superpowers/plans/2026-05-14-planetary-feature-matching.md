# Planetary Feature Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++/LibTorch test-driven foundation for planetary sparse and semi-dense feature extraction and matching.

**Architecture:** Implement the project as a small C++ library plus CLI executable. Start with deterministic data/geometry utilities, then add model-output data structures, lightweight LibTorch modules, losses, evaluator scaffolding, and CLI commands. Each task adds tests first, then minimal code to pass.

**Tech Stack:** C++17, CMake, LibTorch, CLI11 header at `CLI11.hpp`, standard library test harness, optional OpenCV only for image I/O after the core tensor path is tested.

---

## Scope

This plan builds the first working vertical slice, not a fully trained production model. It creates the project structure, test harness, deterministic augmentation/warp utilities, model modules with correct tensor contracts, loss functions, CLI parsing, and tiny integration commands. Later research iterations can replace the lightweight model internals with larger architectures without changing public interfaces.

The repository is currently not a git repository. Commit steps are written as optional checkpoints. If the user initializes git, run them; otherwise record the checkpoint in the terminal and continue.

## File Structure

Create this structure under `/home/xjw/code/deeplearning/Feature Extraction`:

```text
CMakeLists.txt
CLI11.hpp
include/pfm/core/types.hpp
include/pfm/core/tensor_utils.hpp
include/pfm/data/normalization.hpp
include/pfm/data/synthetic_pair.hpp
include/pfm/geometry/warp.hpp
include/pfm/models/backbone.hpp
include/pfm/models/sparse_head.hpp
include/pfm/models/dense_head.hpp
include/pfm/models/matcher.hpp
include/pfm/losses/losses.hpp
include/pfm/eval/metrics.hpp
include/pfm/cli/commands.hpp
src/core/tensor_utils.cpp
src/data/normalization.cpp
src/data/synthetic_pair.cpp
src/geometry/warp.cpp
src/models/backbone.cpp
src/models/sparse_head.cpp
src/models/dense_head.cpp
src/models/matcher.cpp
src/losses/losses.cpp
src/eval/metrics.cpp
src/cli/commands.cpp
src/main.cpp
tests/test_main.cpp
tests/test_normalization.cpp
tests/test_warp.cpp
tests/test_synthetic_pair.cpp
tests/test_model_shapes.cpp
tests/test_losses.cpp
tests/test_metrics.cpp
tests/test_cli.cpp
```

Responsibilities:

- `core/types.hpp`: shared result structs and constants.
- `core/tensor_utils.*`: shape checks and tensor helpers.
- `data/normalization.*`: 8/16-bit image tensor normalization.
- `geometry/warp.*`: affine/perspective-style coordinate transforms and valid masks.
- `data/synthetic_pair.*`: deterministic synthetic pair generation from an input tensor.
- `models/*`: LibTorch module definitions with stable tensor contracts.
- `losses/*`: differentiable losses used by training.
- `eval/*`: metrics for offline evaluation.
- `cli/*`: CLI11 command definitions and argument structs.
- `src/main.cpp`: CLI entry point.
- `tests/*`: standard-library test harness and test cases.

## Test Harness Convention

Use a tiny C++ test harness to avoid introducing a test dependency before the project has a dependency policy.

Each test file registers functions in `tests/test_main.cpp` and uses simple assertions:

```cpp
#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)
#define PFM_REQUIRE_CLOSE(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("close check failed"); } while (0)
```

Run all tests with:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
./build/pfm_tests
```

If LibTorch is installed in a standard CMake path, omit `-DCMAKE_PREFIX_PATH`.

---

### Task 1: Create Build System and Test Harness

**Files:**
- Create: `CMakeLists.txt`
- Create: `tests/test_main.cpp`

- [ ] **Step 1: Write the failing test harness**

Create `tests/test_main.cpp`:

```cpp
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)
#define PFM_REQUIRE_CLOSE(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("close check failed"); } while (0)

using TestFn = void (*)();

struct TestCase {
    std::string name;
    TestFn fn;
};

std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

void register_test(const std::string& name, TestFn fn) {
    registry().push_back({name, fn});
}

void register_normalization_tests();
void register_warp_tests();
void register_synthetic_pair_tests();
void register_model_shape_tests();
void register_loss_tests();
void register_metric_tests();
void register_cli_tests();

int main() {
    register_normalization_tests();
    register_warp_tests();
    register_synthetic_pair_tests();
    register_model_shape_tests();
    register_loss_tests();
    register_metric_tests();
    register_cli_tests();

    int failures = 0;
    for (const auto& test : registry()) {
        try {
            test.fn();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& e) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << e.what() << '\n';
        }
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << registry().size() << " test(s) passed\n";
    return 0;
}
```

- [ ] **Step 2: Add initial CMake build**

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(planetary_feature_matching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_library(pfm STATIC
)

target_include_directories(pfm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pfm PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(pfm_cli src/main.cpp)
target_link_libraries(pfm_cli PRIVATE pfm)

add_executable(pfm_tests
    tests/test_main.cpp
)
target_link_libraries(pfm_tests PRIVATE pfm)
```

- [ ] **Step 3: Run build and verify expected failure**

Run:

```bash
cmake -S . -B build
cmake --build build -j
```

Expected: build fails because `src/main.cpp` does not exist and test registration functions are declared but not defined.

- [ ] **Step 4: Add temporary empty files to make harness compile**

Create `src/main.cpp`:

```cpp
int main(int argc, char** argv) {
    return argc > 0 && argv != nullptr ? 0 : 1;
}
```

Append to `tests/test_main.cpp` after the declarations if separate test files are not yet present:

```cpp
void register_normalization_tests() {}
void register_warp_tests() {}
void register_synthetic_pair_tests() {}
void register_model_shape_tests() {}
void register_loss_tests() {}
void register_metric_tests() {}
void register_cli_tests() {}
```

- [ ] **Step 5: Run build and tests**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
```

Expected: build succeeds and output ends with `0 test(s) passed`.

- [ ] **Step 6: Optional checkpoint**

If this directory has been initialized as git:

```bash
git add CMakeLists.txt src/main.cpp tests/test_main.cpp
git commit -m "test: add C++ test harness"
```

If not a git repository, skip this command.

---

### Task 2: Add Shared Types and Tensor Utilities

**Files:**
- Create: `include/pfm/core/types.hpp`
- Create: `include/pfm/core/tensor_utils.hpp`
- Create: `src/core/tensor_utils.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`
- Create: `tests/test_normalization.cpp`

- [ ] **Step 1: Write failing tensor utility test**

Create `tests/test_normalization.cpp`:

```cpp
#include <stdexcept>
#include <torch/torch.h>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)
#define PFM_REQUIRE_CLOSE(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("close check failed"); } while (0)

void register_test(const std::string& name, void (*fn)());

#include "pfm/core/tensor_utils.hpp"

static void chw_float_tensor_is_accepted() {
    auto tensor = torch::zeros({1, 8, 9}, torch::kFloat32);
    pfm::require_chw_image(tensor);
    PFM_REQUIRE(pfm::height(tensor) == 8);
    PFM_REQUIRE(pfm::width(tensor) == 9);
}

static void hw_tensor_is_rejected() {
    auto tensor = torch::zeros({8, 9}, torch::kFloat32);
    bool thrown = false;
    try {
        pfm::require_chw_image(tensor);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    PFM_REQUIRE(thrown);
}

void register_normalization_tests() {
    register_test("tensor utils accept CHW float image", chw_float_tensor_is_accepted);
    register_test("tensor utils reject HW tensor", hw_tensor_is_rejected);
}
```

- [ ] **Step 2: Remove temporary empty registration**

In `tests/test_main.cpp`, remove the line:

```cpp
void register_normalization_tests() {}
```

Keep the other empty registration functions for now.

- [ ] **Step 3: Add test file to CMake**

Modify the `pfm_tests` target in `CMakeLists.txt`:

```cmake
add_executable(pfm_tests
    tests/test_main.cpp
    tests/test_normalization.cpp
)
```

- [ ] **Step 4: Run test to verify it fails**

Run:

```bash
cmake --build build -j
```

Expected: compile fails because `pfm/core/tensor_utils.hpp` does not exist.

- [ ] **Step 5: Implement shared types and tensor utilities**

Create `include/pfm/core/types.hpp`:

```cpp
#pragma once

#include <torch/torch.h>

namespace pfm {

struct SparseFeatures {
    torch::Tensor keypoints;
    torch::Tensor scores;
    torch::Tensor descriptors;
    torch::Tensor scales;
    torch::Tensor orientations;
    torch::Tensor affine;
};

struct SparseMatches {
    torch::Tensor indices;
    torch::Tensor scores;
};

struct SemiDenseMatches {
    torch::Tensor points_a;
    torch::Tensor points_b;
    torch::Tensor confidence;
    torch::Tensor offsets;
    torch::Tensor valid_mask;
};

struct MatchResult {
    SparseMatches sparse;
    SemiDenseMatches semi_dense;
};

}  // namespace pfm
```

Create `include/pfm/core/tensor_utils.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace pfm {

void require_chw_image(const torch::Tensor& image);
int64_t channels(const torch::Tensor& image);
int64_t height(const torch::Tensor& image);
int64_t width(const torch::Tensor& image);
torch::Tensor make_xy_grid(int64_t height, int64_t width, torch::Device device);

}  // namespace pfm
```

Create `src/core/tensor_utils.cpp`:

```cpp
#include "pfm/core/tensor_utils.hpp"

#include <stdexcept>

namespace pfm {

void require_chw_image(const torch::Tensor& image) {
    if (!image.defined()) {
        throw std::invalid_argument("image tensor is undefined");
    }
    if (image.dim() != 3) {
        throw std::invalid_argument("image tensor must have shape CxHxW");
    }
    if (image.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("image tensor must be float32");
    }
    const auto c = image.size(0);
    if (c != 1 && c != 3) {
        throw std::invalid_argument("image tensor must have 1 or 3 channels");
    }
    if (image.size(1) <= 0 || image.size(2) <= 0) {
        throw std::invalid_argument("image tensor height and width must be positive");
    }
}

int64_t channels(const torch::Tensor& image) {
    require_chw_image(image);
    return image.size(0);
}

int64_t height(const torch::Tensor& image) {
    require_chw_image(image);
    return image.size(1);
}

int64_t width(const torch::Tensor& image) {
    require_chw_image(image);
    return image.size(2);
}

torch::Tensor make_xy_grid(int64_t h, int64_t w, torch::Device device) {
    if (h <= 0 || w <= 0) {
        throw std::invalid_argument("grid dimensions must be positive");
    }
    auto ys = torch::arange(h, torch::TensorOptions().dtype(torch::kFloat32).device(device)).view({h, 1}).repeat({1, w});
    auto xs = torch::arange(w, torch::TensorOptions().dtype(torch::kFloat32).device(device)).view({1, w}).repeat({h, 1});
    return torch::stack({xs, ys}, -1);
}

}  // namespace pfm
```

- [ ] **Step 6: Add source file to CMake**

Modify `add_library(pfm STATIC ...)`:

```cmake
add_library(pfm STATIC
    src/core/tensor_utils.cpp
)
```

- [ ] **Step 7: Run tests**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
```

Expected: both tensor utility tests pass.

- [ ] **Step 8: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/core src/core tests/test_main.cpp tests/test_normalization.cpp
git commit -m "feat: add tensor utility foundation"
```

---

### Task 3: Implement Image Normalization

**Files:**
- Create: `include/pfm/data/normalization.hpp`
- Create: `src/data/normalization.cpp`
- Modify: `tests/test_normalization.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add failing normalization tests**

Append to `tests/test_normalization.cpp` before `register_normalization_tests()`:

```cpp
#include "pfm/data/normalization.hpp"

static void uint8_tensor_normalizes_to_unit_range() {
    auto input = torch::tensor({{{0, 255}}}, torch::kUInt8);
    auto output = pfm::normalize_u8(input);
    PFM_REQUIRE(output.scalar_type() == torch::kFloat32);
    PFM_REQUIRE_CLOSE(output.index({0, 0, 0}).item<float>(), 0.0f, 1e-6f);
    PFM_REQUIRE_CLOSE(output.index({0, 0, 1}).item<float>(), 1.0f, 1e-6f);
}

static void uint16_tensor_normalizes_to_unit_range() {
    auto input = torch::tensor({{{0, 65535}}}, torch::kInt32).to(torch::kUInt16);
    auto output = pfm::normalize_u16(input);
    PFM_REQUIRE(output.scalar_type() == torch::kFloat32);
    PFM_REQUIRE_CLOSE(output.index({0, 0, 0}).item<float>(), 0.0f, 1e-6f);
    PFM_REQUIRE_CLOSE(output.index({0, 0, 1}).item<float>(), 1.0f, 1e-6f);
}

static void local_contrast_preserves_shape() {
    auto input = torch::ones({1, 8, 8}, torch::kFloat32) * 0.5f;
    auto output = pfm::local_contrast_normalize(input, 3);
    PFM_REQUIRE(output.sizes() == input.sizes());
    PFM_REQUIRE(torch::isfinite(output).all().item<bool>());
}
```

Update `register_normalization_tests()`:

```cpp
void register_normalization_tests() {
    register_test("tensor utils accept CHW float image", chw_float_tensor_is_accepted);
    register_test("tensor utils reject HW tensor", hw_tensor_is_rejected);
    register_test("uint8 tensor normalizes to unit range", uint8_tensor_normalizes_to_unit_range);
    register_test("uint16 tensor normalizes to unit range", uint16_tensor_normalizes_to_unit_range);
    register_test("local contrast preserves shape", local_contrast_preserves_shape);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build -j
```

Expected: compile fails because `pfm/data/normalization.hpp` does not exist.

- [ ] **Step 3: Implement normalization API**

Create `include/pfm/data/normalization.hpp`:

```cpp
#pragma once

#include <torch/torch.h>

namespace pfm {

torch::Tensor normalize_u8(const torch::Tensor& image);
torch::Tensor normalize_u16(const torch::Tensor& image);
torch::Tensor clamp_unit(const torch::Tensor& image);
torch::Tensor local_contrast_normalize(const torch::Tensor& image, int64_t kernel_size);

}  // namespace pfm
```

Create `src/data/normalization.cpp`:

```cpp
#include "pfm/data/normalization.hpp"

#include "pfm/core/tensor_utils.hpp"

#include <stdexcept>

namespace pfm {

torch::Tensor normalize_u8(const torch::Tensor& image) {
    if (image.scalar_type() != torch::kUInt8) {
        throw std::invalid_argument("normalize_u8 expects uint8 tensor");
    }
    auto output = image.to(torch::kFloat32) / 255.0f;
    require_chw_image(output);
    return output;
}

torch::Tensor normalize_u16(const torch::Tensor& image) {
    if (image.scalar_type() != torch::kUInt16) {
        throw std::invalid_argument("normalize_u16 expects uint16 tensor");
    }
    auto output = image.to(torch::kFloat32) / 65535.0f;
    require_chw_image(output);
    return output;
}

torch::Tensor clamp_unit(const torch::Tensor& image) {
    require_chw_image(image);
    return torch::clamp(image, 0.0, 1.0);
}

torch::Tensor local_contrast_normalize(const torch::Tensor& image, int64_t kernel_size) {
    require_chw_image(image);
    if (kernel_size < 1 || kernel_size % 2 == 0) {
        throw std::invalid_argument("local contrast kernel size must be positive and odd");
    }
    auto batch = image.unsqueeze(0);
    auto mean = torch::avg_pool2d(batch, {kernel_size, kernel_size}, {1, 1}, {kernel_size / 2, kernel_size / 2});
    auto centered = batch - mean;
    auto variance = torch::avg_pool2d(centered * centered, {kernel_size, kernel_size}, {1, 1}, {kernel_size / 2, kernel_size / 2});
    auto normalized = centered / torch::sqrt(variance + 1e-6f);
    return normalized.squeeze(0);
}

}  // namespace pfm
```

- [ ] **Step 4: Add source file to CMake**

Update `add_library(pfm STATIC ...)`:

```cmake
add_library(pfm STATIC
    src/core/tensor_utils.cpp
    src/data/normalization.cpp
)
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
```

Expected: normalization tests pass.

- [ ] **Step 6: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/data src/data tests/test_normalization.cpp
git commit -m "feat: add image normalization utilities"
```

---

### Task 4: Implement Warp Geometry and Valid Masks

**Files:**
- Create: `include/pfm/geometry/warp.hpp`
- Create: `src/geometry/warp.cpp`
- Create: `tests/test_warp.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing warp tests**

Create `tests/test_warp.cpp`:

```cpp
#include <cmath>
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)
#define PFM_REQUIRE_CLOSE(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("close check failed"); } while (0)

void register_test(const std::string& name, void (*fn)());

#include "pfm/geometry/warp.hpp"

static void identity_warp_maps_pixels_to_themselves() {
    auto transform = pfm::AffineTransform::identity();
    auto field = pfm::dense_warp_field(3, 4, transform, torch::kCPU);
    PFM_REQUIRE(field.sizes() == torch::IntArrayRef({3, 4, 2}));
    PFM_REQUIRE_CLOSE(field.index({2, 3, 0}).item<float>(), 3.0f, 1e-6f);
    PFM_REQUIRE_CLOSE(field.index({2, 3, 1}).item<float>(), 2.0f, 1e-6f);
}

static void translation_warp_offsets_coordinates() {
    auto transform = pfm::AffineTransform::translation(2.0f, -1.0f);
    auto field = pfm::dense_warp_field(3, 4, transform, torch::kCPU);
    PFM_REQUIRE_CLOSE(field.index({1, 1, 0}).item<float>(), 3.0f, 1e-6f);
    PFM_REQUIRE_CLOSE(field.index({1, 1, 1}).item<float>(), 0.0f, 1e-6f);
}

static void valid_mask_rejects_out_of_bounds_coordinates() {
    auto transform = pfm::AffineTransform::translation(10.0f, 0.0f);
    auto field = pfm::dense_warp_field(3, 4, transform, torch::kCPU);
    auto mask = pfm::valid_warp_mask(field, 3, 4);
    PFM_REQUIRE(mask.scalar_type() == torch::kBool);
    PFM_REQUIRE(mask.sum().item<int64_t>() == 0);
}

void register_warp_tests() {
    register_test("identity warp maps pixels to themselves", identity_warp_maps_pixels_to_themselves);
    register_test("translation warp offsets coordinates", translation_warp_offsets_coordinates);
    register_test("valid mask rejects out of bounds coordinates", valid_mask_rejects_out_of_bounds_coordinates);
}
```

- [ ] **Step 2: Remove temporary warp registration**

In `tests/test_main.cpp`, remove:

```cpp
void register_warp_tests() {}
```

- [ ] **Step 3: Add test to CMake**

Update `pfm_tests`:

```cmake
add_executable(pfm_tests
    tests/test_main.cpp
    tests/test_normalization.cpp
    tests/test_warp.cpp
)
```

- [ ] **Step 4: Run test to verify it fails**

Run:

```bash
cmake --build build -j
```

Expected: compile fails because `pfm/geometry/warp.hpp` does not exist.

- [ ] **Step 5: Implement warp geometry**

Create `include/pfm/geometry/warp.hpp`:

```cpp
#pragma once

#include <array>
#include <torch/torch.h>

namespace pfm {

struct AffineTransform {
    std::array<float, 6> matrix;

    static AffineTransform identity();
    static AffineTransform translation(float tx, float ty);
    static AffineTransform scale_rotate(float scale, float radians, float cx, float cy);
};

torch::Tensor dense_warp_field(int64_t height, int64_t width, const AffineTransform& transform, torch::Device device);
torch::Tensor valid_warp_mask(const torch::Tensor& field, int64_t target_height, int64_t target_width);
torch::Tensor warp_points(const torch::Tensor& points, const AffineTransform& transform);

}  // namespace pfm
```

Create `src/geometry/warp.cpp`:

```cpp
#include "pfm/geometry/warp.hpp"

#include "pfm/core/tensor_utils.hpp"

#include <cmath>
#include <stdexcept>

namespace pfm {

AffineTransform AffineTransform::identity() {
    return AffineTransform{{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
}

AffineTransform AffineTransform::translation(float tx, float ty) {
    return AffineTransform{{1.0f, 0.0f, tx, 0.0f, 1.0f, ty}};
}

AffineTransform AffineTransform::scale_rotate(float scale, float radians, float cx, float cy) {
    const float c = std::cos(radians) * scale;
    const float s = std::sin(radians) * scale;
    const float tx = cx - c * cx + s * cy;
    const float ty = cy - s * cx - c * cy;
    return AffineTransform{{c, -s, tx, s, c, ty}};
}

torch::Tensor dense_warp_field(int64_t h, int64_t w, const AffineTransform& transform, torch::Device device) {
    auto grid = make_xy_grid(h, w, device);
    auto x = grid.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto y = grid.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    const auto& m = transform.matrix;
    auto xp = x * m[0] + y * m[1] + m[2];
    auto yp = x * m[3] + y * m[4] + m[5];
    return torch::stack({xp, yp}, -1);
}

torch::Tensor valid_warp_mask(const torch::Tensor& field, int64_t target_height, int64_t target_width) {
    if (field.dim() != 3 || field.size(2) != 2) {
        throw std::invalid_argument("warp field must have shape HxWx2");
    }
    auto x = field.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto y = field.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    return (x >= 0.0f) & (x <= static_cast<float>(target_width - 1)) &
           (y >= 0.0f) & (y <= static_cast<float>(target_height - 1));
}

torch::Tensor warp_points(const torch::Tensor& points, const AffineTransform& transform) {
    if (points.dim() != 2 || points.size(1) != 2) {
        throw std::invalid_argument("points must have shape Nx2");
    }
    auto x = points.index({torch::indexing::Slice(), 0});
    auto y = points.index({torch::indexing::Slice(), 1});
    const auto& m = transform.matrix;
    auto xp = x * m[0] + y * m[1] + m[2];
    auto yp = x * m[3] + y * m[4] + m[5];
    return torch::stack({xp, yp}, 1);
}

}  // namespace pfm
```

- [ ] **Step 6: Add source file to CMake**

Update `add_library(pfm STATIC ...)`:

```cmake
add_library(pfm STATIC
    src/core/tensor_utils.cpp
    src/data/normalization.cpp
    src/geometry/warp.cpp
)
```

- [ ] **Step 7: Run tests**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
```

Expected: warp tests pass.

- [ ] **Step 8: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/geometry src/geometry tests/test_main.cpp tests/test_warp.cpp
git commit -m "feat: add affine warp utilities"
```

---

### Task 5: Implement Deterministic Synthetic Pair Generation

**Files:**
- Create: `include/pfm/data/synthetic_pair.hpp`
- Create: `src/data/synthetic_pair.cpp`
- Create: `tests/test_synthetic_pair.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing synthetic pair tests**

Create `tests/test_synthetic_pair.cpp`:

```cpp
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)

void register_test(const std::string& name, void (*fn)());

#include "pfm/data/synthetic_pair.hpp"

static void synthetic_pair_preserves_expected_shapes() {
    auto image = torch::rand({1, 16, 20}, torch::kFloat32);
    pfm::SyntheticPairConfig config;
    config.translation_x = 2.0f;
    config.translation_y = 1.0f;
    config.brightness_delta = 0.1f;
    auto pair = pfm::make_synthetic_pair(image, config);
    PFM_REQUIRE(pair.view_a.sizes() == image.sizes());
    PFM_REQUIRE(pair.view_b.sizes() == image.sizes());
    PFM_REQUIRE(pair.warp_a_to_b.sizes() == torch::IntArrayRef({16, 20, 2}));
    PFM_REQUIRE(pair.valid_mask.sizes() == torch::IntArrayRef({16, 20}));
}

static void synthetic_pair_marks_translation_invalid_border() {
    auto image = torch::rand({1, 8, 8}, torch::kFloat32);
    pfm::SyntheticPairConfig config;
    config.translation_x = 10.0f;
    config.translation_y = 0.0f;
    auto pair = pfm::make_synthetic_pair(image, config);
    PFM_REQUIRE(pair.valid_mask.sum().item<int64_t>() == 0);
}

void register_synthetic_pair_tests() {
    register_test("synthetic pair preserves expected shapes", synthetic_pair_preserves_expected_shapes);
    register_test("synthetic pair marks translation invalid border", synthetic_pair_marks_translation_invalid_border);
}
```

- [ ] **Step 2: Remove temporary registration and add CMake entry**

Remove this line from `tests/test_main.cpp`:

```cpp
void register_synthetic_pair_tests() {}
```

Add `tests/test_synthetic_pair.cpp` to `pfm_tests`.

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cmake --build build -j
```

Expected: compile fails because `pfm/data/synthetic_pair.hpp` does not exist.

- [ ] **Step 4: Implement synthetic pair API**

Create `include/pfm/data/synthetic_pair.hpp`:

```cpp
#pragma once

#include <torch/torch.h>

#include "pfm/geometry/warp.hpp"

namespace pfm {

struct SyntheticPairConfig {
    float translation_x = 0.0f;
    float translation_y = 0.0f;
    float brightness_delta = 0.0f;
    float contrast_scale = 1.0f;
    float noise_sigma = 0.0f;
};

struct SyntheticPair {
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp_a_to_b;
    torch::Tensor valid_mask;
};

SyntheticPair make_synthetic_pair(const torch::Tensor& image, const SyntheticPairConfig& config);

}  // namespace pfm
```

Create `src/data/synthetic_pair.cpp`:

```cpp
#include "pfm/data/synthetic_pair.hpp"

#include "pfm/core/tensor_utils.hpp"
#include "pfm/data/normalization.hpp"

namespace pfm {

SyntheticPair make_synthetic_pair(const torch::Tensor& image, const SyntheticPairConfig& config) {
    require_chw_image(image);
    auto view_a = clamp_unit(image.clone());
    auto view_b = clamp_unit(image * config.contrast_scale + config.brightness_delta);
    if (config.noise_sigma > 0.0f) {
        view_b = clamp_unit(view_b + torch::randn_like(view_b) * config.noise_sigma);
    }
    auto transform = AffineTransform::translation(config.translation_x, config.translation_y);
    auto field = dense_warp_field(height(image), width(image), transform, image.device());
    auto mask = valid_warp_mask(field, height(image), width(image));
    return SyntheticPair{view_a, view_b, field, mask};
}

}  // namespace pfm
```

- [ ] **Step 5: Add source to CMake**

Add `src/data/synthetic_pair.cpp` to `pfm` library sources.

- [ ] **Step 6: Run tests**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
```

Expected: synthetic pair tests pass.

- [ ] **Step 7: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/data/synthetic_pair.hpp src/data/synthetic_pair.cpp tests/test_main.cpp tests/test_synthetic_pair.cpp
git commit -m "feat: add synthetic pair generation"
```

---

### Task 6: Add Model Modules with Stable Tensor Contracts

**Files:**
- Create: `include/pfm/models/backbone.hpp`
- Create: `include/pfm/models/sparse_head.hpp`
- Create: `include/pfm/models/dense_head.hpp`
- Create: `include/pfm/models/matcher.hpp`
- Create: `src/models/backbone.cpp`
- Create: `src/models/sparse_head.cpp`
- Create: `src/models/dense_head.cpp`
- Create: `src/models/matcher.cpp`
- Create: `tests/test_model_shapes.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing model shape tests**

Create `tests/test_model_shapes.cpp`:

```cpp
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)

void register_test(const std::string& name, void (*fn)());

#include "pfm/models/backbone.hpp"
#include "pfm/models/dense_head.hpp"
#include "pfm/models/matcher.hpp"
#include "pfm/models/sparse_head.hpp"

static void backbone_returns_four_scales() {
    pfm::Backbone model(1, 16);
    auto x = torch::rand({2, 1, 64, 64});
    auto features = model->forward(x);
    PFM_REQUIRE(features.size() == 4);
    PFM_REQUIRE(features[0].sizes() == torch::IntArrayRef({2, 16, 32, 32}));
    PFM_REQUIRE(features[1].sizes() == torch::IntArrayRef({2, 32, 16, 16}));
    PFM_REQUIRE(features[2].sizes() == torch::IntArrayRef({2, 64, 8, 8}));
    PFM_REQUIRE(features[3].sizes() == torch::IntArrayRef({2, 128, 4, 4}));
}

static void sparse_head_outputs_expected_maps() {
    pfm::SparseHead head(16, 64);
    auto feature = torch::rand({2, 16, 32, 32});
    auto output = head->forward(feature);
    PFM_REQUIRE(output.heatmap.sizes() == torch::IntArrayRef({2, 1, 32, 32}));
    PFM_REQUIRE(output.descriptors.sizes() == torch::IntArrayRef({2, 64, 32, 32}));
    PFM_REQUIRE(output.scale.sizes() == torch::IntArrayRef({2, 1, 32, 32}));
    PFM_REQUIRE(output.orientation.sizes() == torch::IntArrayRef({2, 2, 32, 32}));
    PFM_REQUIRE(output.affine.sizes() == torch::IntArrayRef({2, 4, 32, 32}));
}

static void dense_head_outputs_coarse_confidence_and_offsets() {
    pfm::DenseHead head(64);
    auto a = torch::rand({2, 64, 8, 8});
    auto b = torch::rand({2, 64, 8, 8});
    auto output = head->forward(a, b);
    PFM_REQUIRE(output.confidence.sizes() == torch::IntArrayRef({2, 1, 8, 8}));
    PFM_REQUIRE(output.offsets.sizes() == torch::IntArrayRef({2, 2, 8, 8}));
}

static void matcher_outputs_sparse_score_matrix() {
    pfm::Matcher matcher(32);
    auto desc_a = torch::rand({2, 5, 32});
    auto desc_b = torch::rand({2, 7, 32});
    auto scores = matcher->forward(desc_a, desc_b);
    PFM_REQUIRE(scores.sizes() == torch::IntArrayRef({2, 5, 7}));
}

void register_model_shape_tests() {
    register_test("backbone returns four scales", backbone_returns_four_scales);
    register_test("sparse head outputs expected maps", sparse_head_outputs_expected_maps);
    register_test("dense head outputs coarse confidence and offsets", dense_head_outputs_coarse_confidence_and_offsets);
    register_test("matcher outputs sparse score matrix", matcher_outputs_sparse_score_matrix);
}
```

- [ ] **Step 2: Remove temporary registration and add CMake entry**

Remove this line from `tests/test_main.cpp`:

```cpp
void register_model_shape_tests() {}
```

Add `tests/test_model_shapes.cpp` and model source files to CMake after they are created.

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cmake --build build -j
```

Expected: compile fails because model headers do not exist.

- [ ] **Step 4: Implement backbone**

Create `include/pfm/models/backbone.hpp`:

```cpp
#pragma once

#include <torch/torch.h>
#include <vector>

namespace pfm {

struct BackboneImpl : torch::nn::Module {
    torch::nn::Sequential stage1{nullptr};
    torch::nn::Sequential stage2{nullptr};
    torch::nn::Sequential stage3{nullptr};
    torch::nn::Sequential stage4{nullptr};

    BackboneImpl(int64_t in_channels, int64_t base_channels);
    std::vector<torch::Tensor> forward(const torch::Tensor& x);
};

TORCH_MODULE(Backbone);

}  // namespace pfm
```

Create `src/models/backbone.cpp`:

```cpp
#include "pfm/models/backbone.hpp"

namespace pfm {

static torch::nn::Sequential conv_stage(int64_t in_channels, int64_t out_channels) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(2).padding(1)),
        torch::nn::BatchNorm2d(out_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)),
        torch::nn::BatchNorm2d(out_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions(true))
    );
}

BackboneImpl::BackboneImpl(int64_t in_channels, int64_t base_channels) {
    stage1 = register_module("stage1", conv_stage(in_channels, base_channels));
    stage2 = register_module("stage2", conv_stage(base_channels, base_channels * 2));
    stage3 = register_module("stage3", conv_stage(base_channels * 2, base_channels * 4));
    stage4 = register_module("stage4", conv_stage(base_channels * 4, base_channels * 8));
}

std::vector<torch::Tensor> BackboneImpl::forward(const torch::Tensor& x) {
    auto f1 = stage1->forward(x);
    auto f2 = stage2->forward(f1);
    auto f3 = stage3->forward(f2);
    auto f4 = stage4->forward(f3);
    return {f1, f2, f3, f4};
}

}  // namespace pfm
```

- [ ] **Step 5: Implement sparse head**

Create `include/pfm/models/sparse_head.hpp`:

```cpp
#pragma once

#include <torch/torch.h>

namespace pfm {

struct SparseHeadOutput {
    torch::Tensor heatmap;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
};

struct SparseHeadImpl : torch::nn::Module {
    torch::nn::Conv2d heatmap{nullptr};
    torch::nn::Conv2d descriptors{nullptr};
    torch::nn::Conv2d scale{nullptr};
    torch::nn::Conv2d orientation{nullptr};
    torch::nn::Conv2d affine{nullptr};

    SparseHeadImpl(int64_t in_channels, int64_t descriptor_dim);
    SparseHeadOutput forward(const torch::Tensor& feature);
};

TORCH_MODULE(SparseHead);

}  // namespace pfm
```

Create `src/models/sparse_head.cpp`:

```cpp
#include "pfm/models/sparse_head.hpp"

namespace pfm {

SparseHeadImpl::SparseHeadImpl(int64_t in_channels, int64_t descriptor_dim) {
    heatmap = register_module("heatmap", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 1, 1)));
    descriptors = register_module("descriptors", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, descriptor_dim, 1)));
    scale = register_module("scale", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 1, 1)));
    orientation = register_module("orientation", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 2, 1)));
    affine = register_module("affine", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 4, 1)));
}

SparseHeadOutput SparseHeadImpl::forward(const torch::Tensor& feature) {
    auto desc = torch::nn::functional::normalize(descriptors->forward(feature), torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    return SparseHeadOutput{
        torch::sigmoid(heatmap->forward(feature)),
        desc,
        torch::softplus(scale->forward(feature)) + 1e-3f,
        torch::nn::functional::normalize(orientation->forward(feature), torch::nn::functional::NormalizeFuncOptions().p(2).dim(1)),
        affine->forward(feature)
    };
}

}  // namespace pfm
```

- [ ] **Step 6: Implement dense head**

Create `include/pfm/models/dense_head.hpp`:

```cpp
#pragma once

#include <torch/torch.h>

namespace pfm {

struct DenseHeadOutput {
    torch::Tensor confidence;
    torch::Tensor offsets;
};

struct DenseHeadImpl : torch::nn::Module {
    torch::nn::Sequential predictor{nullptr};

    explicit DenseHeadImpl(int64_t channels);
    DenseHeadOutput forward(const torch::Tensor& feature_a, const torch::Tensor& feature_b);
};

TORCH_MODULE(DenseHead);

}  // namespace pfm
```

Create `src/models/dense_head.cpp`:

```cpp
#include "pfm/models/dense_head.hpp"

namespace pfm {

DenseHeadImpl::DenseHeadImpl(int64_t channels) {
    predictor = register_module("predictor", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels * 3, channels, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 3, 1))
    ));
}

DenseHeadOutput DenseHeadImpl::forward(const torch::Tensor& feature_a, const torch::Tensor& feature_b) {
    auto diff = torch::abs(feature_a - feature_b);
    auto x = torch::cat({feature_a, feature_b, diff}, 1);
    auto y = predictor->forward(x);
    return DenseHeadOutput{torch::sigmoid(y.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})), y.index({torch::indexing::Slice(), torch::indexing::Slice(1, 3)})};
}

}  // namespace pfm
```

- [ ] **Step 7: Implement matcher**

Create `include/pfm/models/matcher.hpp`:

```cpp
#pragma once

#include <torch/torch.h>

namespace pfm {

struct MatcherImpl : torch::nn::Module {
    int64_t descriptor_dim;

    explicit MatcherImpl(int64_t descriptor_dim);
    torch::Tensor forward(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b);
};

TORCH_MODULE(Matcher);

}  // namespace pfm
```

Create `src/models/matcher.cpp`:

```cpp
#include "pfm/models/matcher.hpp"

#include <cmath>
#include <stdexcept>

namespace pfm {

MatcherImpl::MatcherImpl(int64_t descriptor_dim_) : descriptor_dim(descriptor_dim_) {}

torch::Tensor MatcherImpl::forward(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b) {
    if (descriptors_a.dim() != 3 || descriptors_b.dim() != 3) {
        throw std::invalid_argument("descriptors must have shape BxNxD and BxMxD");
    }
    if (descriptors_a.size(2) != descriptor_dim || descriptors_b.size(2) != descriptor_dim) {
        throw std::invalid_argument("descriptor dimension mismatch");
    }
    auto a = torch::nn::functional::normalize(descriptors_a, torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));
    auto b = torch::nn::functional::normalize(descriptors_b, torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));
    return torch::bmm(a, b.transpose(1, 2)) / std::sqrt(static_cast<float>(descriptor_dim));
}

}  // namespace pfm
```

- [ ] **Step 8: Add model sources and test to CMake**

Add to `pfm` sources:

```cmake
    src/models/backbone.cpp
    src/models/sparse_head.cpp
    src/models/dense_head.cpp
    src/models/matcher.cpp
```

Add to `pfm_tests` sources:

```cmake
    tests/test_model_shapes.cpp
```

- [ ] **Step 9: Run tests**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
```

Expected: model shape tests pass.

- [ ] **Step 10: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/models src/models tests/test_main.cpp tests/test_model_shapes.cpp
git commit -m "feat: add model tensor contracts"
```

---

### Task 7: Implement Loss Functions

**Files:**
- Create: `include/pfm/losses/losses.hpp`
- Create: `src/losses/losses.cpp`
- Create: `tests/test_losses.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing loss tests**

Create `tests/test_losses.cpp`:

```cpp
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)
#define PFM_REQUIRE_CLOSE(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("close check failed"); } while (0)

void register_test(const std::string& name, void (*fn)());

#include "pfm/losses/losses.hpp"

static void repeatability_loss_zero_for_identical_heatmaps() {
    auto heatmap = torch::ones({1, 1, 4, 4}) * 0.5f;
    auto mask = torch::ones({1, 1, 4, 4}, torch::kBool);
    auto loss = pfm::repeatability_loss(heatmap, heatmap, mask);
    PFM_REQUIRE_CLOSE(loss.item<float>(), 0.0f, 1e-6f);
}

static void descriptor_loss_lower_for_matching_pairs() {
    auto a = torch::tensor({{{1.0f, 0.0f}, {0.0f, 1.0f}}});
    auto b = torch::tensor({{{1.0f, 0.0f}, {0.0f, 1.0f}}});
    auto labels = torch::tensor({{0, 1}}, torch::kLong);
    auto loss = pfm::descriptor_cross_entropy_loss(a, b, labels);
    PFM_REQUIRE(loss.item<float>() < 1.0f);
}

static void offset_loss_zero_when_mask_empty() {
    auto pred = torch::ones({1, 2, 4, 4});
    auto target = torch::zeros({1, 2, 4, 4});
    auto mask = torch::zeros({1, 1, 4, 4}, torch::kBool);
    auto loss = pfm::masked_l1_loss(pred, target, mask);
    PFM_REQUIRE_CLOSE(loss.item<float>(), 0.0f, 1e-6f);
}

void register_loss_tests() {
    register_test("repeatability loss zero for identical heatmaps", repeatability_loss_zero_for_identical_heatmaps);
    register_test("descriptor loss lower for matching pairs", descriptor_loss_lower_for_matching_pairs);
    register_test("offset loss zero when mask empty", offset_loss_zero_when_mask_empty);
}
```

- [ ] **Step 2: Remove temporary registration and add CMake entry**

Remove this line from `tests/test_main.cpp`:

```cpp
void register_loss_tests() {}
```

Add `tests/test_losses.cpp` to `pfm_tests`.

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cmake --build build -j
```

Expected: compile fails because loss header does not exist.

- [ ] **Step 4: Implement losses**

Create `include/pfm/losses/losses.hpp`:

```cpp
#pragma once

#include <torch/torch.h>

namespace pfm {

torch::Tensor repeatability_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b, const torch::Tensor& valid_mask);
torch::Tensor descriptor_cross_entropy_loss(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b, const torch::Tensor& target_indices);
torch::Tensor masked_l1_loss(const torch::Tensor& prediction, const torch::Tensor& target, const torch::Tensor& valid_mask);
torch::Tensor confidence_bce_loss(const torch::Tensor& confidence, const torch::Tensor& target);

}  // namespace pfm
```

Create `src/losses/losses.cpp`:

```cpp
#include "pfm/losses/losses.hpp"

#include <stdexcept>

namespace pfm {

torch::Tensor repeatability_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b, const torch::Tensor& valid_mask) {
    auto mask = valid_mask.to(torch::kFloat32);
    auto denom = mask.sum().clamp_min(1.0f);
    return (((heatmap_a - heatmap_b).pow(2) * mask).sum() / denom);
}

torch::Tensor descriptor_cross_entropy_loss(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b, const torch::Tensor& target_indices) {
    if (descriptors_a.dim() != 3 || descriptors_b.dim() != 3) {
        throw std::invalid_argument("descriptors must have shape BxNxD and BxMxD");
    }
    auto a = torch::nn::functional::normalize(descriptors_a, torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));
    auto b = torch::nn::functional::normalize(descriptors_b, torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));
    auto logits = torch::bmm(a, b.transpose(1, 2));
    return torch::nn::functional::cross_entropy(logits.view({-1, logits.size(2)}), target_indices.reshape({-1}));
}

torch::Tensor masked_l1_loss(const torch::Tensor& prediction, const torch::Tensor& target, const torch::Tensor& valid_mask) {
    auto mask = valid_mask.to(torch::kFloat32);
    while (mask.dim() < prediction.dim()) {
        mask = mask.expand_as(prediction);
    }
    auto denom = mask.sum();
    if (denom.item<float>() <= 0.0f) {
        return torch::zeros({}, prediction.options());
    }
    return (torch::abs(prediction - target) * mask).sum() / denom;
}

torch::Tensor confidence_bce_loss(const torch::Tensor& confidence, const torch::Tensor& target) {
    return torch::binary_cross_entropy(confidence, target.to(confidence.dtype()));
}

}  // namespace pfm
```

- [ ] **Step 5: Add source to CMake**

Add `src/losses/losses.cpp` to `pfm` sources and `tests/test_losses.cpp` to `pfm_tests`.

- [ ] **Step 6: Run tests**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
```

Expected: loss tests pass.

- [ ] **Step 7: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/losses src/losses tests/test_main.cpp tests/test_losses.cpp
git commit -m "feat: add core training losses"
```

---

### Task 8: Implement Evaluation Metrics

**Files:**
- Create: `include/pfm/eval/metrics.hpp`
- Create: `src/eval/metrics.cpp`
- Create: `tests/test_metrics.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metrics.cpp`:

```cpp
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)
#define PFM_REQUIRE_CLOSE(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("close check failed"); } while (0)

void register_test(const std::string& name, void (*fn)());

#include "pfm/eval/metrics.hpp"

static void matching_precision_counts_matches_within_threshold() {
    auto predicted_a = torch::tensor({{0.0f, 0.0f}, {10.0f, 10.0f}});
    auto predicted_b = torch::tensor({{1.0f, 0.0f}, {30.0f, 30.0f}});
    auto expected_b = torch::tensor({{1.0f, 0.0f}, {11.0f, 10.0f}});
    auto precision = pfm::matching_precision(predicted_a, predicted_b, expected_b, 1.5f);
    PFM_REQUIRE_CLOSE(precision, 0.5f, 1e-6f);
}

static void semi_dense_coverage_uses_valid_mask_area() {
    auto confidence = torch::tensor({{0.9f, 0.1f}, {0.8f, 0.7f}});
    auto valid = torch::tensor({{true, true}, {false, true}}, torch::kBool);
    auto coverage = pfm::semi_dense_coverage(confidence, valid, 0.75f);
    PFM_REQUIRE_CLOSE(coverage, 1.0f / 3.0f, 1e-6f);
}

void register_metric_tests() {
    register_test("matching precision counts matches within threshold", matching_precision_counts_matches_within_threshold);
    register_test("semi dense coverage uses valid mask area", semi_dense_coverage_uses_valid_mask_area);
}
```

- [ ] **Step 2: Remove temporary registration and add CMake entry**

Remove this line from `tests/test_main.cpp`:

```cpp
void register_metric_tests() {}
```

Add `tests/test_metrics.cpp` to `pfm_tests`.

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cmake --build build -j
```

Expected: compile fails because metrics header does not exist.

- [ ] **Step 4: Implement metrics**

Create `include/pfm/eval/metrics.hpp`:

```cpp
#pragma once

#include <torch/torch.h>

namespace pfm {

float matching_precision(const torch::Tensor& points_a, const torch::Tensor& predicted_b, const torch::Tensor& expected_b, float threshold_pixels);
float semi_dense_coverage(const torch::Tensor& confidence, const torch::Tensor& valid_mask, float threshold);

}  // namespace pfm
```

Create `src/eval/metrics.cpp`:

```cpp
#include "pfm/eval/metrics.hpp"

#include <stdexcept>

namespace pfm {

float matching_precision(const torch::Tensor& points_a, const torch::Tensor& predicted_b, const torch::Tensor& expected_b, float threshold_pixels) {
    (void)points_a;
    if (predicted_b.sizes() != expected_b.sizes()) {
        throw std::invalid_argument("predicted and expected points must have matching shapes");
    }
    if (predicted_b.numel() == 0) {
        return 0.0f;
    }
    auto dist = torch::norm(predicted_b - expected_b, 2, 1);
    return (dist <= threshold_pixels).to(torch::kFloat32).mean().item<float>();
}

float semi_dense_coverage(const torch::Tensor& confidence, const torch::Tensor& valid_mask, float threshold) {
    auto valid = valid_mask.to(torch::kBool);
    auto denom = valid.sum().item<float>();
    if (denom <= 0.0f) {
        return 0.0f;
    }
    auto selected = (confidence >= threshold) & valid;
    return selected.sum().item<float>() / denom;
}

}  // namespace pfm
```

- [ ] **Step 5: Add source to CMake**

Add `src/eval/metrics.cpp` to `pfm` sources and `tests/test_metrics.cpp` to `pfm_tests`.

- [ ] **Step 6: Run tests**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
```

Expected: metric tests pass.

- [ ] **Step 7: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/eval src/eval tests/test_main.cpp tests/test_metrics.cpp
git commit -m "feat: add evaluation metrics"
```

---

### Task 9: Implement CLI11 Command Parsing

**Files:**
- Create: `include/pfm/cli/commands.hpp`
- Create: `src/cli/commands.cpp`
- Create: `tests/test_cli.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `src/main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli.cpp`:

```cpp
#include <stdexcept>
#include <string>
#include <vector>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)

void register_test(const std::string& name, void (*fn)());

#include "pfm/cli/commands.hpp"

static void parse_extract_command() {
    const std::vector<std::string> args = {"pfm", "extract", "--image", "a.png", "--checkpoint", "model.pt", "--output", "a.pfm"};
    auto parsed = pfm::parse_cli(args);
    PFM_REQUIRE(parsed.command == pfm::Command::Extract);
    PFM_REQUIRE(parsed.image == "a.png");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.output == "a.pfm");
}

static void parse_match_command() {
    const std::vector<std::string> args = {"pfm", "match", "--image-a", "a.png", "--image-b", "b.png", "--checkpoint", "model.pt", "--output", "matches.json", "--max-keypoints", "2048"};
    auto parsed = pfm::parse_cli(args);
    PFM_REQUIRE(parsed.command == pfm::Command::Match);
    PFM_REQUIRE(parsed.image_a == "a.png");
    PFM_REQUIRE(parsed.image_b == "b.png");
    PFM_REQUIRE(parsed.max_keypoints == 2048);
}

void register_cli_tests() {
    register_test("parse extract command", parse_extract_command);
    register_test("parse match command", parse_match_command);
}
```

- [ ] **Step 2: Remove temporary registration and add CMake entry**

Remove this line from `tests/test_main.cpp`:

```cpp
void register_cli_tests() {}
```

Add `tests/test_cli.cpp` to `pfm_tests`.

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cmake --build build -j
```

Expected: compile fails because CLI header does not exist.

- [ ] **Step 4: Implement CLI parser**

Create `include/pfm/cli/commands.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pfm {

enum class Command {
    None,
    Train,
    Extract,
    Match,
    Eval,
    Export
};

struct CliOptions {
    Command command = Command::None;
    std::string image_dir;
    std::string image;
    std::string image_a;
    std::string image_b;
    std::string pairs;
    std::string checkpoint;
    std::string config;
    std::string output;
    std::string device = "cpu";
    int64_t max_keypoints = 1024;
    double semi_dense_threshold = 0.5;
    int64_t epochs = 1;
    int64_t batch_size = 1;
};

CliOptions parse_cli(const std::vector<std::string>& args);
int run_cli(int argc, char** argv);

}  // namespace pfm
```

Create `src/cli/commands.cpp`:

```cpp
#include "pfm/cli/commands.hpp"

#include "CLI11.hpp"

#include <iostream>
#include <stdexcept>

namespace pfm {

CliOptions parse_cli(const std::vector<std::string>& args) {
    CliOptions options;
    CLI::App app{"Planetary feature extraction and matching"};

    auto* train = app.add_subcommand("train", "Train from an image directory");
    train->add_option("--image-dir", options.image_dir)->required();
    train->add_option("--checkpoint", options.checkpoint)->required();
    train->add_option("--config", options.config);
    train->add_option("--device", options.device);
    train->add_option("--epochs", options.epochs);
    train->add_option("--batch-size", options.batch_size);
    train->callback([&]() { options.command = Command::Train; });

    auto* extract = app.add_subcommand("extract", "Extract sparse features from one image");
    extract->add_option("--image", options.image)->required();
    extract->add_option("--checkpoint", options.checkpoint)->required();
    extract->add_option("--output", options.output)->required();
    extract->add_option("--max-keypoints", options.max_keypoints);
    extract->add_option("--device", options.device);
    extract->callback([&]() { options.command = Command::Extract; });

    auto* match = app.add_subcommand("match", "Match two images");
    match->add_option("--image-a", options.image_a)->required();
    match->add_option("--image-b", options.image_b)->required();
    match->add_option("--checkpoint", options.checkpoint)->required();
    match->add_option("--output", options.output)->required();
    match->add_option("--max-keypoints", options.max_keypoints);
    match->add_option("--semi-dense-threshold", options.semi_dense_threshold);
    match->add_option("--device", options.device);
    match->callback([&]() { options.command = Command::Match; });

    auto* eval = app.add_subcommand("eval", "Evaluate matching quality");
    eval->add_option("--pairs", options.pairs)->required();
    eval->add_option("--checkpoint", options.checkpoint)->required();
    eval->add_option("--output", options.output)->required();
    eval->add_option("--device", options.device);
    eval->callback([&]() { options.command = Command::Eval; });

    auto* export_cmd = app.add_subcommand("export", "Export model artifacts");
    export_cmd->add_option("--checkpoint", options.checkpoint)->required();
    export_cmd->add_option("--output", options.output)->required();
    export_cmd->callback([&]() { options.command = Command::Export; });

    app.require_subcommand(1);
    std::vector<std::string> mutable_args(args.rbegin(), args.rend());
    app.parse(mutable_args);
    return options;
}

int run_cli(int argc, char** argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    try {
        auto options = parse_cli(args);
        std::cout << "parsed command " << static_cast<int>(options.command) << '\n';
        return 0;
    } catch (const CLI::ParseError& e) {
        CLI::App app{"Planetary feature extraction and matching"};
        return app.exit(e);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

}  // namespace pfm
```

Modify `src/main.cpp`:

```cpp
#include "pfm/cli/commands.hpp"

int main(int argc, char** argv) {
    return pfm::run_cli(argc, argv);
}
```

- [ ] **Step 5: Add CLI source to CMake**

Add `src/cli/commands.cpp` to `pfm` sources and `tests/test_cli.cpp` to `pfm_tests`.

- [ ] **Step 6: Run tests and CLI smoke commands**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
./build/pfm_cli extract --image a.png --checkpoint model.pt --output a.pfm
./build/pfm_cli match --image-a a.png --image-b b.png --checkpoint model.pt --output matches.json
```

Expected: tests pass and both CLI commands print `parsed command` with exit code 0.

- [ ] **Step 7: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/cli src/cli src/main.cpp tests/test_main.cpp tests/test_cli.cpp
git commit -m "feat: add CLI11 command parsing"
```

---

### Task 10: Add Integration Stubs for Extract, Match, and Eval

**Files:**
- Modify: `include/pfm/cli/commands.hpp`
- Modify: `src/cli/commands.cpp`
- Create: `include/pfm/infer/pipeline.hpp`
- Create: `src/infer/pipeline.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_cli.cpp`

- [ ] **Step 1: Write failing command execution tests**

Append to `tests/test_cli.cpp` before `register_cli_tests()`:

```cpp
static void run_extract_without_checkpoint_path_fails_cleanly() {
    const char* argv[] = {"pfm", "extract", "--image", "a.png", "--checkpoint", "", "--output", "a.pfm"};
    int code = pfm::run_cli(7, const_cast<char**>(argv));
    PFM_REQUIRE(code != 0);
}
```

Update `register_cli_tests()`:

```cpp
void register_cli_tests() {
    register_test("parse extract command", parse_extract_command);
    register_test("parse match command", parse_match_command);
    register_test("run extract without checkpoint path fails cleanly", run_extract_without_checkpoint_path_fails_cleanly);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build -j
./build/pfm_tests
```

Expected: new test fails because `run_cli` currently returns success after parsing.

- [ ] **Step 3: Add inference pipeline API**

Create `include/pfm/infer/pipeline.hpp`:

```cpp
#pragma once

#include <string>

#include "pfm/cli/commands.hpp"

namespace pfm {

int run_train_command(const CliOptions& options);
int run_extract_command(const CliOptions& options);
int run_match_command(const CliOptions& options);
int run_eval_command(const CliOptions& options);
int run_export_command(const CliOptions& options);

}  // namespace pfm
```

Create `src/infer/pipeline.cpp`:

```cpp
#include "pfm/infer/pipeline.hpp"

#include <iostream>

namespace pfm {

static bool missing(const std::string& value) {
    return value.empty();
}

int run_train_command(const CliOptions& options) {
    if (missing(options.image_dir) || missing(options.checkpoint)) {
        std::cerr << "train requires --image-dir and --checkpoint\n";
        return 1;
    }
    std::cout << "train command accepted\n";
    return 0;
}

int run_extract_command(const CliOptions& options) {
    if (missing(options.image) || missing(options.checkpoint) || missing(options.output)) {
        std::cerr << "extract requires --image, --checkpoint, and --output\n";
        return 1;
    }
    std::cout << "extract command accepted\n";
    return 0;
}

int run_match_command(const CliOptions& options) {
    if (missing(options.image_a) || missing(options.image_b) || missing(options.checkpoint) || missing(options.output)) {
        std::cerr << "match requires --image-a, --image-b, --checkpoint, and --output\n";
        return 1;
    }
    std::cout << "match command accepted\n";
    return 0;
}

int run_eval_command(const CliOptions& options) {
    if (missing(options.pairs) || missing(options.checkpoint) || missing(options.output)) {
        std::cerr << "eval requires --pairs, --checkpoint, and --output\n";
        return 1;
    }
    std::cout << "eval command accepted\n";
    return 0;
}

int run_export_command(const CliOptions& options) {
    if (missing(options.checkpoint) || missing(options.output)) {
        std::cerr << "export requires --checkpoint and --output\n";
        return 1;
    }
    std::cout << "export command accepted\n";
    return 0;
}

}  // namespace pfm
```

- [ ] **Step 4: Route parsed commands to pipeline**

Modify `src/cli/commands.cpp` to include pipeline and replace the success print inside `run_cli`:

```cpp
#include "pfm/infer/pipeline.hpp"
```

Replace:

```cpp
std::cout << "parsed command " << static_cast<int>(options.command) << '\n';
return 0;
```

With:

```cpp
switch (options.command) {
    case Command::Train:
        return run_train_command(options);
    case Command::Extract:
        return run_extract_command(options);
    case Command::Match:
        return run_match_command(options);
    case Command::Eval:
        return run_eval_command(options);
    case Command::Export:
        return run_export_command(options);
    case Command::None:
        std::cerr << "no command selected\n";
        return 1;
}
return 1;
```

- [ ] **Step 5: Add pipeline source to CMake**

Add `src/infer/pipeline.cpp` to `pfm` sources.

- [ ] **Step 6: Run tests and smoke commands**

Run:

```bash
cmake -S . -B build
cmake --build build -j
./build/pfm_tests
./build/pfm_cli extract --image a.png --checkpoint model.pt --output a.pfm
./build/pfm_cli match --image-a a.png --image-b b.png --checkpoint model.pt --output matches.json
./build/pfm_cli eval --pairs pairs.txt --checkpoint model.pt --output report.json
```

Expected: tests pass and each smoke command prints `<command> command accepted`.

- [ ] **Step 7: Optional checkpoint**

```bash
git add CMakeLists.txt include/pfm/infer src/infer src/cli/commands.cpp tests/test_cli.cpp
git commit -m "feat: add CLI execution stubs"
```

---

### Task 11: Final Verification

**Files:**
- Read: all files created in previous tasks.

- [ ] **Step 1: Run full build**

Run:

```bash
cmake -S . -B build
cmake --build build -j
```

Expected: build succeeds without compiler errors.

- [ ] **Step 2: Run full test suite**

Run:

```bash
./build/pfm_tests
```

Expected: all registered tests pass.

- [ ] **Step 3: Run CLI smoke checks**

Run:

```bash
./build/pfm_cli train --image-dir images --checkpoint model.pt --epochs 1 --batch-size 1
./build/pfm_cli extract --image a.png --checkpoint model.pt --output a.pfm
./build/pfm_cli match --image-a a.png --image-b b.png --checkpoint model.pt --output matches.json --max-keypoints 1024 --semi-dense-threshold 0.5
./build/pfm_cli eval --pairs pairs.txt --checkpoint model.pt --output report.json
./build/pfm_cli export --checkpoint model.pt --output exported.pt
```

Expected: each command exits 0 and prints `<command> command accepted`.

- [ ] **Step 4: Review spec coverage**

Confirm these implementation foundations exist:

- C++/LibTorch build.
- CLI11 command parsing.
- 8/16-bit tensor normalization utilities.
- Synthetic pair generation with warp field and valid mask.
- Sparse outputs for heatmap, descriptor, scale, orientation, affine shape.
- Semi-dense confidence and offset output contract.
- Learned matcher score matrix contract.
- Core losses for repeatability, descriptor matching, offsets, and confidence.
- Evaluation metrics for matching precision and semi-dense coverage.

- [ ] **Step 5: Optional final checkpoint**

```bash
git status --short
git add CMakeLists.txt include src tests docs/superpowers
git commit -m "feat: add planetary feature matching foundation"
```

Skip if the directory is not a git repository.

## Self-Review Result

Spec coverage:

- Inputs and normalization are covered by Tasks 2 and 3.
- Synthetic single-image pair generation and valid masks are covered by Tasks 4 and 5.
- Sparse keypoint/descriptor/scale/orientation/affine tensor contracts are covered by Task 6.
- Semi-dense coarse output contracts are covered by Task 6.
- Learned matcher contract is covered by Task 6.
- Loss foundations are covered by Task 7.
- Evaluation metrics are covered by Task 8.
- CLI11 commands are covered by Tasks 9 and 10.
- TDD is enforced by every task writing tests first.

Known limits of this first implementation plan:

- It does not train a high-quality final planetary model.
- It creates extensible contracts and tested foundations for the research model.
- It leaves image file decoding and full training loop for the next implementation plan after this foundation is verified.

Placeholder scan: no `TBD`, `TODO`, or undefined task references remain.

Type consistency: public names use the same `pfm::` namespace, tensor shape conventions, and command option names throughout.
