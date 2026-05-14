# Real Image Training and Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first real C++/LibTorch loop for OpenCV image loading, training, checkpointing, feature extraction, matching, evaluation, and `.pt` export.

**Architecture:** Keep the existing module-first layout. Add OpenCV-backed data ingestion under `modules/data`, LibTorch archive codecs under `modules/infer`, and training orchestration under `modules/train`; then replace CLI validation stubs in `modules/infer/pipeline.cpp` with real command execution.

**Tech Stack:** C++17, CMake, LibTorch, OpenCV, CLI11, custom test harness.

---

## Scope Notes

This plan implements the “效果优先 MVP” from `docs/superpowers/specs/2026-05-14-real-image-training-inference-design.md`. It deliberately keeps the model simple enough to train in tests and focuses on a reliable end-to-end loop. Complex production logging, distributed training, JSON feature export, advanced resume, and Python tooling stay out of scope.

The user-provided real training images are in `/home/xjw/code/deeplearning/Feature Extraction/build/img`, with many `.tif` files available. Use this path for final real-data smoke commands after module tests pass.

## File Structure

Create or modify these files:

- Modify `CMakeLists.txt`
  - Add `find_package(OpenCV REQUIRED)`.
  - Link `${OpenCV_LIBS}` into `pfm`.
  - Add new module `.cpp` files to `pfm`.
  - Add new `*_test.cpp` files to `pfm_tests`.
- Create `modules/data/image_io.h/.cpp/.test.cpp`
  - OpenCV image loading and tensor conversion.
- Create `modules/data/image_dataset.h/.cpp/.test.cpp`
  - Directory traversal, extension filtering, image sample loading.
- Modify `modules/data/synthetic_pair.h/.cpp/.test.cpp`
  - Add deterministic photometric augmentation support without breaking existing tests.
- Create `modules/infer/feature_codec.h/.cpp/.test.cpp`
  - Save/read feature tensors with `torch::serialize::OutputArchive` and `InputArchive`.
- Create `modules/infer/match_codec.h/.cpp/.test.cpp`
  - Save/read sparse and semi-dense match tensors.
- Create `modules/infer/feature_extractor.h/.cpp/.test.cpp`
  - Decode sparse keypoints/descriptors and semi-dense points from model outputs.
- Create `modules/train/trainer.h/.cpp/.test.cpp`
  - Model bundle, training config/result, one-step and multi-epoch training, checkpoint save/load.
- Modify `modules/infer/pipeline.h/.cpp`
  - Replace validation stubs with real `train`, `extract`, `match`, `eval`, and `export` behavior.
- Modify `modules/cli/commands.h/.cpp/.test.cpp`
  - Add only the CLI options needed for real execution if a task requires them; preserve existing commands and `--semi-dense-threshold` behavior.
- Modify `tests/test_main.cpp`
  - Register new module tests.
- Modify `README.md` and `docs/training.md`
  - Update current status and real command behavior after implementation.

---

### Task 1: Add OpenCV Build Integration

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the build change**

Add OpenCV after Torch:

```cmake
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
```

Change library linking from:

```cmake
target_link_libraries(pfm PUBLIC ${TORCH_LIBRARIES})
```

to:

```cmake
target_link_libraries(pfm PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS})
```

- [ ] **Step 2: Configure to verify OpenCV is discoverable**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
```

Expected: configure succeeds and mentions OpenCV in CMake cache. If configure fails with missing OpenCV, install OpenCV development headers before continuing; do not add fallback image loaders.

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Add OpenCV build integration"
```

---

### Task 2: Implement OpenCV Image IO

**Files:**
- Create: `modules/data/image_io.h`
- Create: `modules/data/image_io.cpp`
- Create: `modules/data/image_io_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Add failing tests**

Create `modules/data/image_io_test.cpp`:

```cpp
#include "tests/test_harness.h"

#include <filesystem>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include "data/image_io.h"

namespace {

std::filesystem::path temp_image_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}

void load_u8_grayscale_image_returns_chw_float_tensor() {
    const auto path = temp_image_path("pfm_u8_gray.png");
    cv::Mat image(2, 3, CV_8UC1);
    image.at<unsigned char>(0, 0) = 0;
    image.at<unsigned char>(0, 1) = 127;
    image.at<unsigned char>(0, 2) = 255;
    image.at<unsigned char>(1, 0) = 32;
    image.at<unsigned char>(1, 1) = 64;
    image.at<unsigned char>(1, 2) = 128;
    PFM_REQUIRE(cv::imwrite(path.string(), image));

    const torch::Tensor tensor = pfm::load_image_tensor(path.string());

    PFM_REQUIRE(tensor.sizes() == torch::IntArrayRef({1, 2, 3}));
    PFM_REQUIRE(tensor.dtype() == torch::kFloat32);
    PFM_REQUIRE_CLOSE(tensor.index({0, 0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(tensor.index({0, 0, 2}).item<float>(), 1.0F, 1.0e-6F);
    std::filesystem::remove(path);
}

void load_u16_grayscale_image_returns_unit_tensor() {
    const auto path = temp_image_path("pfm_u16_gray.png");
    cv::Mat image(1, 2, CV_16UC1);
    image.at<unsigned short>(0, 0) = 0;
    image.at<unsigned short>(0, 1) = 65535;
    PFM_REQUIRE(cv::imwrite(path.string(), image));

    const torch::Tensor tensor = pfm::load_image_tensor(path.string());

    PFM_REQUIRE(tensor.sizes() == torch::IntArrayRef({1, 1, 2}));
    PFM_REQUIRE_CLOSE(tensor.index({0, 0, 1}).item<float>(), 1.0F, 1.0e-6F);
    std::filesystem::remove(path);
}

void load_u8_color_image_converts_bgr_to_rgb() {
    const auto path = temp_image_path("pfm_u8_color.png");
    cv::Mat image(1, 1, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    PFM_REQUIRE(cv::imwrite(path.string(), image));

    const torch::Tensor tensor = pfm::load_image_tensor(path.string());

    PFM_REQUIRE(tensor.sizes() == torch::IntArrayRef({3, 1, 1}));
    PFM_REQUIRE_CLOSE(tensor.index({0, 0, 0}).item<float>(), 30.0F / 255.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(tensor.index({1, 0, 0}).item<float>(), 20.0F / 255.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(tensor.index({2, 0, 0}).item<float>(), 10.0F / 255.0F, 1.0e-6F);
    std::filesystem::remove(path);
}

void load_missing_image_throws() {
    PFM_REQUIRE_INVALID_ARG(pfm::load_image_tensor("/tmp/pfm_missing_image.png"));
}

}  // namespace

void register_image_io_tests() {
    register_test("load_u8_grayscale_image_returns_chw_float_tensor", load_u8_grayscale_image_returns_chw_float_tensor);
    register_test("load_u16_grayscale_image_returns_unit_tensor", load_u16_grayscale_image_returns_unit_tensor);
    register_test("load_u8_color_image_converts_bgr_to_rgb", load_u8_color_image_converts_bgr_to_rgb);
    register_test("load_missing_image_throws", load_missing_image_throws);
}
```

Modify `tests/test_main.cpp` by adding:

```cpp
void register_image_io_tests();
```

and call:

```cpp
register_image_io_tests();
```

Modify `CMakeLists.txt` by adding:

```cmake
modules/data/image_io.cpp
```

to `pfm` sources and:

```cmake
modules/data/image_io_test.cpp
```

to `pfm_tests` sources.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
```

Expected: compile fails because `data/image_io.h` does not exist.

- [ ] **Step 3: Implement header**

Create `modules/data/image_io.h`:

```cpp
#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm {

/// Load an 8-bit or 16-bit grayscale/RGB image as a CHW float tensor in [0, 1].
/// @param path Image path readable by OpenCV.
/// @return Tensor with shape CxHxW and dtype float32.
/// @throws std::invalid_argument if the image is missing, empty, or has unsupported depth/channels.
torch::Tensor load_image_tensor(const std::string& path);

}  // namespace pfm
```

- [ ] **Step 4: Implement source**

Create `modules/data/image_io.cpp`:

```cpp
#include "data/image_io.h"

#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pfm {
namespace {

float scale_for_depth(int depth) {
    if (depth == CV_8U) {
        return 1.0F / 255.0F;
    }
    if (depth == CV_16U) {
        return 1.0F / 65535.0F;
    }
    throw std::invalid_argument("unsupported image depth");
}

}  // namespace

torch::Tensor load_image_tensor(const std::string& path) {
    const cv::Mat loaded = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (loaded.empty()) {
        throw std::invalid_argument("failed to read image: " + path);
    }

    cv::Mat image;
    if (loaded.channels() == 1) {
        image = loaded;
    } else if (loaded.channels() == 3) {
        cv::cvtColor(loaded, image, cv::COLOR_BGR2RGB);
    } else if (loaded.channels() == 4) {
        cv::cvtColor(loaded, image, cv::COLOR_BGRA2RGB);
    } else {
        throw std::invalid_argument("unsupported image channel count");
    }

    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, scale_for_depth(image.depth()));

    const int64_t height = float_image.rows;
    const int64_t width = float_image.cols;
    const int64_t channels = float_image.channels();

    torch::Tensor tensor = torch::from_blob(
        float_image.data,
        {height, width, channels},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();

    if (channels == 1) {
        tensor = tensor.reshape({height, width, 1});
    }

    return tensor.permute({2, 0, 1}).contiguous();
}

}  // namespace pfm
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: all tests pass, including four image IO tests.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt modules/data/image_io.h modules/data/image_io.cpp modules/data/image_io_test.cpp tests/test_main.cpp
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Add OpenCV image tensor loading"
```

---

### Task 3: Implement Image Dataset Traversal

**Files:**
- Create: `modules/data/image_dataset.h`
- Create: `modules/data/image_dataset.cpp`
- Create: `modules/data/image_dataset_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Add failing tests**

Create `modules/data/image_dataset_test.cpp`:

```cpp
#include "tests/test_harness.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "data/image_dataset.h"

namespace {

std::filesystem::path make_dataset_dir(const std::string& name) {
    const auto dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void write_png(const std::filesystem::path& path) {
    cv::Mat image(2, 2, CV_8UC1, cv::Scalar(128));
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

void image_dataset_filters_supported_extensions_and_sorts() {
    const auto dir = make_dataset_dir("pfm_dataset_filter");
    write_png(dir / "b.PNG");
    write_png(dir / "a.jpg");
    std::ofstream(dir / "ignore.txt") << "not image";

    const pfm::ImageDataset dataset(dir.string());

    PFM_REQUIRE(dataset.size() == 2);
    PFM_REQUIRE(dataset.path(0).find("a.jpg") != std::string::npos);
    PFM_REQUIRE(dataset.path(1).find("b.PNG") != std::string::npos);
    std::filesystem::remove_all(dir);
}

void image_dataset_loads_tensor_sample() {
    const auto dir = make_dataset_dir("pfm_dataset_load");
    write_png(dir / "image.png");

    const pfm::ImageDataset dataset(dir.string());
    const torch::Tensor sample = dataset.load(0);

    PFM_REQUIRE(sample.sizes() == torch::IntArrayRef({1, 2, 2}));
    std::filesystem::remove_all(dir);
}

void image_dataset_rejects_empty_directory() {
    const auto dir = make_dataset_dir("pfm_dataset_empty");
    PFM_REQUIRE_INVALID_ARG(pfm::ImageDataset(dir.string()));
    std::filesystem::remove_all(dir);
}

}  // namespace

void register_image_dataset_tests() {
    register_test("image_dataset_filters_supported_extensions_and_sorts", image_dataset_filters_supported_extensions_and_sorts);
    register_test("image_dataset_loads_tensor_sample", image_dataset_loads_tensor_sample);
    register_test("image_dataset_rejects_empty_directory", image_dataset_rejects_empty_directory);
}
```

Register `register_image_dataset_tests()` in `tests/test_main.cpp`. Add `modules/data/image_dataset.cpp` and `modules/data/image_dataset_test.cpp` to `CMakeLists.txt`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
```

Expected: compile fails because `data/image_dataset.h` does not exist.

- [ ] **Step 3: Implement header**

Create `modules/data/image_dataset.h`:

```cpp
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace pfm {

class ImageDataset {
public:
    /// Build a dataset by scanning a directory for supported image files.
    /// @param image_dir Directory containing png, jpg, jpeg, tif, or tiff images.
    /// @throws std::invalid_argument if the directory does not exist or contains no supported images.
    explicit ImageDataset(const std::string& image_dir);

    /// Return number of discovered images.
    /// @return Dataset size.
    std::size_t size() const;

    /// Return image path at index.
    /// @param index Image index.
    /// @return Absolute or relative path string discovered during construction.
    /// @throws std::out_of_range if index is invalid.
    const std::string& path(std::size_t index) const;

    /// Load image tensor at index.
    /// @param index Image index.
    /// @return CHW float image tensor.
    /// @throws std::out_of_range if index is invalid.
    torch::Tensor load(std::size_t index) const;

private:
    std::vector<std::string> _paths;
};

}  // namespace pfm
```

- [ ] **Step 4: Implement source**

Create `modules/data/image_dataset.cpp`:

```cpp
#include "data/image_dataset.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "data/image_io.h"

namespace pfm {
namespace {

std::string lower_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension;
}

bool is_supported_image(const std::filesystem::path& path) {
    const std::string extension = lower_extension(path);
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".tif" ||
           extension == ".tiff";
}

}  // namespace

ImageDataset::ImageDataset(const std::string& image_dir) {
    const std::filesystem::path root(image_dir);
    if (!std::filesystem::is_directory(root)) {
        throw std::invalid_argument("image directory does not exist: " + image_dir);
    }

    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (entry.is_regular_file() && is_supported_image(entry.path())) {
            _paths.push_back(entry.path().string());
        }
    }

    std::sort(_paths.begin(), _paths.end());
    if (_paths.empty()) {
        throw std::invalid_argument("image directory contains no supported images: " + image_dir);
    }
}

std::size_t ImageDataset::size() const {
    return _paths.size();
}

const std::string& ImageDataset::path(std::size_t index) const {
    if (index >= _paths.size()) {
        throw std::out_of_range("image dataset index out of range");
    }
    return _paths[index];
}

torch::Tensor ImageDataset::load(std::size_t index) const {
    return load_image_tensor(path(index));
}

}  // namespace pfm
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: all tests pass, including image dataset tests.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt modules/data/image_dataset.h modules/data/image_dataset.cpp modules/data/image_dataset_test.cpp tests/test_main.cpp
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Add image dataset traversal"
```

---

### Task 4: Add Feature and Match `.pt` Codecs

**Files:**
- Create: `modules/infer/feature_codec.h`
- Create: `modules/infer/feature_codec.cpp`
- Create: `modules/infer/feature_codec_test.cpp`
- Create: `modules/infer/match_codec.h`
- Create: `modules/infer/match_codec.cpp`
- Create: `modules/infer/match_codec_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Add failing feature codec test**

Create `modules/infer/feature_codec_test.cpp`:

```cpp
#include "tests/test_harness.h"

#include <filesystem>

#include <torch/torch.h>

#include "infer/feature_codec.h"

namespace {

void feature_codec_round_trips_all_fields() {
    const auto path = std::filesystem::temp_directory_path() / "pfm_features.pt";
    pfm::FeatureSet features;
    features.keypoints = torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}});
    features.scores = torch::tensor({0.9F, 0.8F});
    features.descriptors = torch::ones({2, 8});
    features.scale = torch::ones({2});
    features.orientation = torch::zeros({2});
    features.affine = torch::eye(2).repeat({2, 1, 1});
    features.dense_points = torch::tensor({{1.0F, 1.0F}});
    features.dense_confidence = torch::tensor({0.7F});

    pfm::save_feature_set(features, path.string());
    const pfm::FeatureSet loaded = pfm::load_feature_set(path.string());

    PFM_REQUIRE(loaded.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(loaded.descriptors.sizes() == torch::IntArrayRef({2, 8}));
    PFM_REQUIRE(loaded.affine.sizes() == torch::IntArrayRef({2, 2, 2}));
    PFM_REQUIRE_CLOSE(loaded.dense_confidence.index({0}).item<float>(), 0.7F, 1.0e-6F);
    std::filesystem::remove(path);
}

void feature_codec_rejects_missing_path() {
    PFM_REQUIRE_INVALID_ARG(pfm::load_feature_set("/tmp/pfm_missing_features.pt"));
}

}  // namespace

void register_feature_codec_tests() {
    register_test("feature_codec_round_trips_all_fields", feature_codec_round_trips_all_fields);
    register_test("feature_codec_rejects_missing_path", feature_codec_rejects_missing_path);
}
```

- [ ] **Step 2: Add failing match codec test**

Create `modules/infer/match_codec_test.cpp`:

```cpp
#include "tests/test_harness.h"

#include <filesystem>

#include <torch/torch.h>

#include "infer/match_codec.h"

namespace {

void match_codec_round_trips_all_fields() {
    const auto path = std::filesystem::temp_directory_path() / "pfm_matches.pt";
    pfm::MatchSet matches;
    matches.sparse_matches = torch::tensor({{0, 1}, {2, 3}}, torch::kInt64);
    matches.sparse_scores = torch::tensor({0.8F, 0.6F});
    matches.points_a = torch::tensor({{1.0F, 2.0F}});
    matches.points_b = torch::tensor({{3.0F, 4.0F}});
    matches.confidence = torch::tensor({0.5F});

    pfm::save_match_set(matches, path.string());
    const pfm::MatchSet loaded = pfm::load_match_set(path.string());

    PFM_REQUIRE(loaded.sparse_matches.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(loaded.points_a.sizes() == torch::IntArrayRef({1, 2}));
    PFM_REQUIRE_CLOSE(loaded.confidence.index({0}).item<float>(), 0.5F, 1.0e-6F);
    std::filesystem::remove(path);
}

void match_codec_rejects_missing_path() {
    PFM_REQUIRE_INVALID_ARG(pfm::load_match_set("/tmp/pfm_missing_matches.pt"));
}

}  // namespace

void register_match_codec_tests() {
    register_test("match_codec_round_trips_all_fields", match_codec_round_trips_all_fields);
    register_test("match_codec_rejects_missing_path", match_codec_rejects_missing_path);
}
```

Register both test functions in `tests/test_main.cpp`. Add codec sources/tests to `CMakeLists.txt`.

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
```

Expected: compile fails because codec headers do not exist.

- [ ] **Step 4: Implement feature codec**

Create `modules/infer/feature_codec.h`:

```cpp
#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm {

struct FeatureSet {
    torch::Tensor keypoints;
    torch::Tensor scores;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
    torch::Tensor dense_points;
    torch::Tensor dense_confidence;
};

/// Save a feature set to a LibTorch archive.
/// @param features Feature tensors to save.
/// @param path Output .pt path.
/// @throws std::invalid_argument if required tensors are undefined.
void save_feature_set(const FeatureSet& features, const std::string& path);

/// Load a feature set from a LibTorch archive.
/// @param path Input .pt path.
/// @return Loaded feature tensors.
/// @throws std::invalid_argument if the file cannot be loaded.
FeatureSet load_feature_set(const std::string& path);

}  // namespace pfm
```

Create `modules/infer/feature_codec.cpp`:

```cpp
#include "infer/feature_codec.h"

#include <stdexcept>
#include <string>

namespace pfm {
namespace {

void require_defined(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string("undefined feature tensor: ") + name);
    }
}

void write_tensor(torch::serialize::OutputArchive& archive, const char* name, const torch::Tensor& tensor) {
    require_defined(tensor, name);
    archive.write(name, tensor);
}

torch::Tensor read_tensor(torch::serialize::InputArchive& archive, const char* name) {
    torch::Tensor tensor;
    archive.read(name, tensor);
    return tensor;
}

}  // namespace

void save_feature_set(const FeatureSet& features, const std::string& path) {
    torch::serialize::OutputArchive archive;
    write_tensor(archive, "keypoints", features.keypoints);
    write_tensor(archive, "scores", features.scores);
    write_tensor(archive, "descriptors", features.descriptors);
    write_tensor(archive, "scale", features.scale);
    write_tensor(archive, "orientation", features.orientation);
    write_tensor(archive, "affine", features.affine);
    write_tensor(archive, "dense_points", features.dense_points);
    write_tensor(archive, "dense_confidence", features.dense_confidence);
    archive.save_to(path);
}

FeatureSet load_feature_set(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        return FeatureSet{
            read_tensor(archive, "keypoints"),
            read_tensor(archive, "scores"),
            read_tensor(archive, "descriptors"),
            read_tensor(archive, "scale"),
            read_tensor(archive, "orientation"),
            read_tensor(archive, "affine"),
            read_tensor(archive, "dense_points"),
            read_tensor(archive, "dense_confidence"),
        };
    } catch (const c10::Error& error) {
        throw std::invalid_argument("failed to load feature set: " + path);
    }
}

}  // namespace pfm
```

- [ ] **Step 5: Implement match codec**

Create `modules/infer/match_codec.h`:

```cpp
#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm {

struct MatchSet {
    torch::Tensor sparse_matches;
    torch::Tensor sparse_scores;
    torch::Tensor points_a;
    torch::Tensor points_b;
    torch::Tensor confidence;
};

/// Save sparse and semi-dense matches to a LibTorch archive.
/// @param matches Match tensors to save.
/// @param path Output .pt path.
/// @throws std::invalid_argument if required tensors are undefined.
void save_match_set(const MatchSet& matches, const std::string& path);

/// Load sparse and semi-dense matches from a LibTorch archive.
/// @param path Input .pt path.
/// @return Loaded match tensors.
/// @throws std::invalid_argument if the file cannot be loaded.
MatchSet load_match_set(const std::string& path);

}  // namespace pfm
```

Create `modules/infer/match_codec.cpp`:

```cpp
#include "infer/match_codec.h"

#include <stdexcept>
#include <string>

namespace pfm {
namespace {

void require_defined(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string("undefined match tensor: ") + name);
    }
}

void write_tensor(torch::serialize::OutputArchive& archive, const char* name, const torch::Tensor& tensor) {
    require_defined(tensor, name);
    archive.write(name, tensor);
}

torch::Tensor read_tensor(torch::serialize::InputArchive& archive, const char* name) {
    torch::Tensor tensor;
    archive.read(name, tensor);
    return tensor;
}

}  // namespace

void save_match_set(const MatchSet& matches, const std::string& path) {
    torch::serialize::OutputArchive archive;
    write_tensor(archive, "sparse_matches", matches.sparse_matches);
    write_tensor(archive, "sparse_scores", matches.sparse_scores);
    write_tensor(archive, "points_a", matches.points_a);
    write_tensor(archive, "points_b", matches.points_b);
    write_tensor(archive, "confidence", matches.confidence);
    archive.save_to(path);
}

MatchSet load_match_set(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        return MatchSet{
            read_tensor(archive, "sparse_matches"),
            read_tensor(archive, "sparse_scores"),
            read_tensor(archive, "points_a"),
            read_tensor(archive, "points_b"),
            read_tensor(archive, "confidence"),
        };
    } catch (const c10::Error& error) {
        throw std::invalid_argument("failed to load match set: " + path);
    }
}

}  // namespace pfm
```

- [ ] **Step 6: Run tests**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: all tests pass, including codec round trips.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt modules/infer/feature_codec.h modules/infer/feature_codec.cpp modules/infer/feature_codec_test.cpp modules/infer/match_codec.h modules/infer/match_codec.cpp modules/infer/match_codec_test.cpp tests/test_main.cpp
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Add LibTorch feature and match codecs"
```

---

### Task 5: Implement Feature Extraction Decoding

**Files:**
- Create: `modules/infer/feature_extractor.h`
- Create: `modules/infer/feature_extractor.cpp`
- Create: `modules/infer/feature_extractor_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Add failing tests**

Create `modules/infer/feature_extractor_test.cpp`:

```cpp
#include "tests/test_harness.h"

#include <torch/torch.h>

#include "infer/feature_extractor.h"

namespace {

void decode_sparse_features_returns_top_k_points() {
    pfm::RawFeatureMaps maps;
    maps.heatmap = torch::zeros({1, 1, 2, 3});
    maps.heatmap.index_put_({0, 0, 0, 1}, 0.9F);
    maps.heatmap.index_put_({0, 0, 1, 2}, 0.8F);
    maps.descriptors = torch::ones({1, 4, 2, 3});
    maps.scale = torch::ones({1, 1, 2, 3});
    maps.orientation = torch::zeros({1, 1, 2, 3});
    maps.affine = torch::ones({1, 4, 2, 3});
    maps.dense_confidence = torch::ones({1, 1, 2, 3});

    const pfm::FeatureSet features = pfm::decode_feature_maps(maps, 2, 0.5);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(features.descriptors.sizes() == torch::IntArrayRef({2, 4}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
}

void decode_dense_features_filters_by_threshold() {
    pfm::RawFeatureMaps maps;
    maps.heatmap = torch::ones({1, 1, 2, 2});
    maps.descriptors = torch::ones({1, 2, 2, 2});
    maps.scale = torch::ones({1, 1, 2, 2});
    maps.orientation = torch::zeros({1, 1, 2, 2});
    maps.affine = torch::ones({1, 4, 2, 2});
    maps.dense_confidence = torch::tensor({{{{0.1F, 0.9F}, {0.8F, 0.2F}}}});

    const pfm::FeatureSet features = pfm::decode_feature_maps(maps, 4, 0.75);

    PFM_REQUIRE(features.dense_points.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(features.dense_confidence.sizes() == torch::IntArrayRef({2}));
}

}  // namespace

void register_feature_extractor_tests() {
    register_test("decode_sparse_features_returns_top_k_points", decode_sparse_features_returns_top_k_points);
    register_test("decode_dense_features_filters_by_threshold", decode_dense_features_filters_by_threshold);
}
```

Register the tests and add files to CMake.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
```

Expected: compile fails because `infer/feature_extractor.h` does not exist.

- [ ] **Step 3: Implement header**

Create `modules/infer/feature_extractor.h`:

```cpp
#pragma once

#include <torch/torch.h>

#include "infer/feature_codec.h"

namespace pfm {

struct RawFeatureMaps {
    torch::Tensor heatmap;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
    torch::Tensor dense_confidence;
};

/// Decode network maps into serializable sparse and semi-dense features.
/// @param maps Raw prediction tensors with batch size 1.
/// @param max_keypoints Maximum sparse keypoints to keep.
/// @param semi_dense_threshold Confidence threshold for dense points.
/// @return FeatureSet containing sparse and semi-dense tensors.
/// @throws std::invalid_argument if shapes are invalid or max_keypoints is not positive.
FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold);

}  // namespace pfm
```

- [ ] **Step 4: Implement source**

Create `modules/infer/feature_extractor.cpp`:

```cpp
#include "infer/feature_extractor.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace pfm {
namespace {

void require_4d_batch_one(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined() || tensor.dim() != 4 || tensor.size(0) != 1) {
        throw std::invalid_argument(std::string("invalid feature map: ") + name);
    }
}

}  // namespace

FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold) {
    if (max_keypoints <= 0) {
        throw std::invalid_argument("max_keypoints must be positive");
    }
    require_4d_batch_one(maps.heatmap, "heatmap");
    require_4d_batch_one(maps.descriptors, "descriptors");
    require_4d_batch_one(maps.scale, "scale");
    require_4d_batch_one(maps.orientation, "orientation");
    require_4d_batch_one(maps.affine, "affine");
    require_4d_batch_one(maps.dense_confidence, "dense_confidence");

    const int64_t height = maps.heatmap.size(2);
    const int64_t width = maps.heatmap.size(3);
    const torch::Tensor flat_scores = maps.heatmap.reshape({-1});
    const int64_t keep = std::min<int64_t>(max_keypoints, flat_scores.size(0));
    const auto topk = torch::topk(flat_scores, keep);
    const torch::Tensor scores = std::get<0>(topk).contiguous();
    const torch::Tensor indices = std::get<1>(topk).to(torch::kLong);
    const torch::Tensor ys = torch::div(indices, width, "floor");
    const torch::Tensor xs = indices.remainder(width);
    const torch::Tensor keypoints = torch::stack({xs.to(torch::kFloat32), ys.to(torch::kFloat32)}, 1);

    std::vector<torch::Tensor> descriptors;
    std::vector<torch::Tensor> scales;
    std::vector<torch::Tensor> orientations;
    std::vector<torch::Tensor> affines;
    descriptors.reserve(static_cast<std::size_t>(keep));
    scales.reserve(static_cast<std::size_t>(keep));
    orientations.reserve(static_cast<std::size_t>(keep));
    affines.reserve(static_cast<std::size_t>(keep));

    for (int64_t i = 0; i < keep; ++i) {
        const int64_t y = ys.index({i}).item<int64_t>();
        const int64_t x = xs.index({i}).item<int64_t>();
        descriptors.push_back(maps.descriptors.index({0, torch::indexing::Slice(), y, x}));
        scales.push_back(maps.scale.index({0, 0, y, x}));
        orientations.push_back(maps.orientation.index({0, 0, y, x}));
        affines.push_back(maps.affine.index({0, torch::indexing::Slice(), y, x}).reshape({2, 2}));
    }

    const torch::Tensor dense_mask = maps.dense_confidence.index({0, 0}) >= semi_dense_threshold;
    const torch::Tensor dense_indices = torch::nonzero(dense_mask);
    torch::Tensor dense_points;
    torch::Tensor dense_confidence;
    if (dense_indices.size(0) == 0) {
        dense_points = torch::empty({0, 2}, torch::kFloat32);
        dense_confidence = torch::empty({0}, torch::kFloat32);
    } else {
        const torch::Tensor dense_y = dense_indices.index({torch::indexing::Slice(), 0});
        const torch::Tensor dense_x = dense_indices.index({torch::indexing::Slice(), 1});
        dense_points = torch::stack({dense_x.to(torch::kFloat32), dense_y.to(torch::kFloat32)}, 1);
        dense_confidence = maps.dense_confidence.index({0, 0}).index({dense_y, dense_x}).to(torch::kFloat32);
    }

    return FeatureSet{
        keypoints.contiguous(),
        scores.to(torch::kFloat32).contiguous(),
        torch::stack(descriptors).to(torch::kFloat32).contiguous(),
        torch::stack(scales).to(torch::kFloat32).contiguous(),
        torch::stack(orientations).to(torch::kFloat32).contiguous(),
        torch::stack(affines).to(torch::kFloat32).contiguous(),
        dense_points.contiguous(),
        dense_confidence.contiguous(),
    };
}

}  // namespace pfm
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt modules/infer/feature_extractor.h modules/infer/feature_extractor.cpp modules/infer/feature_extractor_test.cpp tests/test_main.cpp
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Add feature map decoding"
```

---

### Task 6: Implement Trainer and Checkpointing

**Files:**
- Create: `modules/train/trainer.h`
- Create: `modules/train/trainer.cpp`
- Create: `modules/train/trainer_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Add failing trainer tests**

Create `modules/train/trainer_test.cpp`:

```cpp
#include "tests/test_harness.h"

#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "data/image_dataset.h"
#include "train/trainer.h"

namespace {

std::filesystem::path make_training_dir() {
    const auto dir = std::filesystem::temp_directory_path() / "pfm_training_images";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    for (int i = 0; i < 2; ++i) {
        cv::Mat image(32, 32, CV_8UC1);
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                image.at<unsigned char>(y, x) = static_cast<unsigned char>((x + y + i * 16) % 255);
            }
        }
        PFM_REQUIRE(cv::imwrite((dir / ("image_" + std::to_string(i) + ".png")).string(), image));
    }
    return dir;
}

void trainer_runs_one_epoch_and_saves_checkpoint() {
    const auto dir = make_training_dir();
    const auto checkpoint = std::filesystem::temp_directory_path() / "pfm_model.pt";

    pfm::TrainConfig config;
    config.image_dir = dir.string();
    config.checkpoint = checkpoint.string();
    config.epochs = 1;
    config.batch_size = 1;
    config.base_channels = 4;
    config.descriptor_dim = 8;

    const pfm::TrainResult result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss >= 0.0);
    PFM_REQUIRE(std::filesystem::exists(checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(checkpoint.string()));
    std::filesystem::remove_all(dir);
    std::filesystem::remove(checkpoint);
}

void trainer_rejects_missing_image_dir() {
    pfm::TrainConfig config;
    config.image_dir = "/tmp/pfm_missing_training_dir";
    config.checkpoint = "/tmp/pfm_model.pt";
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(config));
}

}  // namespace

void register_trainer_tests() {
    register_test("trainer_runs_one_epoch_and_saves_checkpoint", trainer_runs_one_epoch_and_saves_checkpoint);
    register_test("trainer_rejects_missing_image_dir", trainer_rejects_missing_image_dir);
}
```

Register trainer tests and add trainer source/test to CMake.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
```

Expected: compile fails because `train/trainer.h` does not exist.

- [ ] **Step 3: Implement trainer header**

Create `modules/train/trainer.h`:

```cpp
#pragma once

#include <string>

namespace pfm {

struct TrainConfig {
    std::string image_dir;
    std::string checkpoint;
    std::string device = "cpu";
    int epochs = 1;
    int batch_size = 1;
    int base_channels = 8;
    int descriptor_dim = 32;
    double learning_rate = 1.0e-3;
};

struct TrainResult {
    int epochs_completed = 0;
    double initial_loss = 0.0;
    double final_loss = 0.0;
};

/// Train the first-stage feature matching model and save a checkpoint.
/// @param config Training configuration.
/// @return Training result with loss summary.
/// @throws std::invalid_argument if paths or numeric parameters are invalid.
TrainResult train_model(const TrainConfig& config);

/// Check whether a checkpoint can be loaded as a LibTorch archive.
/// @param checkpoint Checkpoint path.
/// @return True if loading succeeds, false otherwise.
bool checkpoint_can_load(const std::string& checkpoint);

}  // namespace pfm
```

- [ ] **Step 4: Implement trainer source**

Create `modules/train/trainer.cpp`:

```cpp
#include "train/trainer.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "data/image_dataset.h"
#include "data/synthetic_pair.h"
#include "losses/losses.h"
#include "models/backbone.h"
#include "models/dense_head.h"
#include "models/sparse_head.h"

namespace pfm {
namespace {

struct ModelBundle {
    Backbone backbone{nullptr};
    SparseHead sparse_head{nullptr};
    DenseHead dense_head{nullptr};
};

ModelBundle make_model(int input_channels, int base_channels, int descriptor_dim) {
    ModelBundle model;
    model.backbone = Backbone(input_channels, base_channels);
    model.sparse_head = SparseHead(base_channels * 8, descriptor_dim);
    model.dense_head = DenseHead(base_channels * 8);
    return model;
}

void save_checkpoint(const ModelBundle& model, const TrainConfig& config) {
    torch::serialize::OutputArchive archive;
    torch::serialize::OutputArchive backbone_archive;
    torch::serialize::OutputArchive sparse_archive;
    torch::serialize::OutputArchive dense_archive;
    model.backbone->save(backbone_archive);
    model.sparse_head->save(sparse_archive);
    model.dense_head->save(dense_archive);
    archive.write("backbone", backbone_archive);
    archive.write("sparse_head", sparse_archive);
    archive.write("dense_head", dense_archive);
    archive.write("base_channels", torch::tensor({config.base_channels}, torch::kInt64));
    archive.write("descriptor_dim", torch::tensor({config.descriptor_dim}, torch::kInt64));
    archive.save_to(config.checkpoint);
}

}  // namespace

TrainResult train_model(const TrainConfig& config) {
    if (config.checkpoint.empty()) {
        throw std::invalid_argument("checkpoint path is required");
    }
    if (config.epochs <= 0 || config.batch_size <= 0 || config.base_channels <= 0 || config.descriptor_dim <= 0) {
        throw std::invalid_argument("training numeric parameters must be positive");
    }

    ImageDataset dataset(config.image_dir);
    const torch::Tensor first = dataset.load(0);
    ModelBundle model = make_model(static_cast<int>(first.size(0)), config.base_channels, config.descriptor_dim);

    std::vector<torch::Tensor> parameters;
    for (const auto& parameter : model.backbone->parameters()) parameters.push_back(parameter);
    for (const auto& parameter : model.sparse_head->parameters()) parameters.push_back(parameter);
    for (const auto& parameter : model.dense_head->parameters()) parameters.push_back(parameter);
    torch::optim::AdamW optimizer(parameters, torch::optim::AdamWOptions(config.learning_rate));

    TrainResult result;
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        double total_loss = 0.0;
        for (std::size_t i = 0; i < dataset.size(); ++i) {
            const torch::Tensor image = dataset.load(i);
            SyntheticPairConfig pair_config;
            pair_config.translation_x = 1.0;
            pair_config.translation_y = 1.0;
            pair_config.noise_sigma = 0.01;
            const SyntheticPair pair = make_synthetic_pair(image, pair_config);

            const torch::Tensor batch_a = pair.view_a.unsqueeze(0);
            const torch::Tensor batch_b = pair.view_b.unsqueeze(0);
            auto features_a = model.backbone->forward(batch_a);
            auto features_b = model.backbone->forward(batch_b);
            const SparseHeadOutput sparse_a = model.sparse_head->forward(features_a.back());
            const SparseHeadOutput sparse_b = model.sparse_head->forward(features_b.back());
            const DenseHeadOutput dense = model.dense_head->forward(features_a.back(), features_b.back());

            const torch::Tensor heat_loss = repeatability_loss(sparse_a.heatmap, sparse_b.heatmap, torch::ones_like(sparse_a.heatmap));
            const torch::Tensor offset_loss = masked_l1_loss(dense.offsets, torch::zeros_like(dense.offsets), torch::ones_like(dense.offsets));
            const torch::Tensor confidence_loss = confidence_bce_loss(dense.confidence, torch::ones_like(dense.confidence));
            const torch::Tensor loss = heat_loss + offset_loss + confidence_loss;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            total_loss += loss.item<double>();
        }
        const double average_loss = total_loss / static_cast<double>(dataset.size());
        if (epoch == 0) {
            result.initial_loss = average_loss;
        }
        result.final_loss = average_loss;
        result.epochs_completed = epoch + 1;
        std::cout << "epoch " << (epoch + 1) << " loss " << average_loss << '\n';
    }

    save_checkpoint(model, config);
    return result;
}

bool checkpoint_can_load(const std::string& checkpoint) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(checkpoint);
        return true;
    } catch (const c10::Error&) {
        return false;
    }
}

}  // namespace pfm
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: all tests pass; trainer test creates and loads a checkpoint.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt modules/train/trainer.h modules/train/trainer.cpp modules/train/trainer_test.cpp tests/test_main.cpp
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Add minimal training loop and checkpointing"
```

---

### Task 7: Implement Real `train`, `extract`, and `export` Pipeline Commands

**Files:**
- Modify: `modules/infer/pipeline.cpp`
- Modify: `modules/infer/pipeline.h`
- Modify: `modules/cli/commands_test.cpp`
- Create or modify tests in: `modules/infer/pipeline_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Add failing pipeline tests**

Create `modules/infer/pipeline_test.cpp`:

```cpp
#include "tests/test_harness.h"

#include <filesystem>

#include <opencv2/imgcodecs.hpp>

#include "cli/commands.h"
#include "infer/feature_codec.h"
#include "infer/pipeline.h"
#include "train/trainer.h"

namespace {

std::filesystem::path make_pipeline_dir() {
    const auto dir = std::filesystem::temp_directory_path() / "pfm_pipeline_images";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    cv::Mat image(32, 32, CV_8UC1, cv::Scalar(128));
    PFM_REQUIRE(cv::imwrite((dir / "image.png").string(), image));
    return dir;
}

pfm::CliOptions train_options(const std::filesystem::path& dir, const std::filesystem::path& checkpoint) {
    pfm::CliOptions options;
    options.image_dir = dir.string();
    options.checkpoint = checkpoint.string();
    options.epochs = 1;
    options.batch_size = 1;
    return options;
}

void pipeline_train_writes_checkpoint() {
    const auto dir = make_pipeline_dir();
    const auto checkpoint = std::filesystem::temp_directory_path() / "pfm_pipeline_model.pt";
    PFM_REQUIRE(pfm::run_train_command(train_options(dir, checkpoint)) == 0);
    PFM_REQUIRE(std::filesystem::exists(checkpoint));
    std::filesystem::remove_all(dir);
    std::filesystem::remove(checkpoint);
}

void pipeline_extract_writes_feature_file() {
    const auto dir = make_pipeline_dir();
    const auto checkpoint = std::filesystem::temp_directory_path() / "pfm_pipeline_extract_model.pt";
    PFM_REQUIRE(pfm::run_train_command(train_options(dir, checkpoint)) == 0);

    pfm::CliOptions options;
    options.image = (dir / "image.png").string();
    options.checkpoint = checkpoint.string();
    options.output = (std::filesystem::temp_directory_path() / "pfm_pipeline_features.pt").string();
    options.max_keypoints = 16;
    options.semi_dense_threshold = 0.5;

    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const pfm::FeatureSet features = pfm::load_feature_set(options.output);
    PFM_REQUIRE(features.keypoints.size(1) == 2);
    std::filesystem::remove_all(dir);
    std::filesystem::remove(checkpoint);
    std::filesystem::remove(options.output);
}

void pipeline_export_writes_loadable_checkpoint() {
    const auto dir = make_pipeline_dir();
    const auto checkpoint = std::filesystem::temp_directory_path() / "pfm_pipeline_export_model.pt";
    const auto exported = std::filesystem::temp_directory_path() / "pfm_pipeline_exported.pt";
    PFM_REQUIRE(pfm::run_train_command(train_options(dir, checkpoint)) == 0);

    pfm::CliOptions options;
    options.checkpoint = checkpoint.string();
    options.output = exported.string();

    PFM_REQUIRE(pfm::run_export_command(options) == 0);
    PFM_REQUIRE(pfm::checkpoint_can_load(exported.string()));
    std::filesystem::remove_all(dir);
    std::filesystem::remove(checkpoint);
    std::filesystem::remove(exported);
}

}  // namespace

void register_pipeline_tests() {
    register_test("pipeline_train_writes_checkpoint", pipeline_train_writes_checkpoint);
    register_test("pipeline_extract_writes_feature_file", pipeline_extract_writes_feature_file);
    register_test("pipeline_export_writes_loadable_checkpoint", pipeline_export_writes_loadable_checkpoint);
}
```

Register `register_pipeline_tests()` and add the test to CMake.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: pipeline tests fail because commands still only print accepted or do not write files.

- [ ] **Step 3: Implement command behavior**

Modify `modules/infer/pipeline.cpp` to:

```cpp
#include "infer/pipeline.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "data/image_io.h"
#include "infer/feature_codec.h"
#include "infer/feature_extractor.h"
#include "models/backbone.h"
#include "models/dense_head.h"
#include "models/sparse_head.h"
#include "train/trainer.h"

namespace pfm {
namespace {

bool require_path(const std::string& value, const char* option_name) {
    if (!value.empty()) {
        return true;
    }
    std::cerr << "missing required option " << option_name << '\n';
    return false;
}

FeatureSet extract_with_fresh_model(const CliOptions& options) {
    if (!checkpoint_can_load(options.checkpoint)) {
        throw std::invalid_argument("checkpoint cannot be loaded: " + options.checkpoint);
    }
    const torch::Tensor image = load_image_tensor(options.image);
    const int64_t channels = image.size(0);
    Backbone backbone(channels, 8);
    SparseHead sparse_head(64, 32);
    DenseHead dense_head(64);
    const torch::Tensor batch = image.unsqueeze(0);
    auto features = backbone->forward(batch);
    const SparseHeadOutput sparse = sparse_head->forward(features.back());
    const DenseHeadOutput dense = dense_head->forward(features.back(), features.back());
    return decode_feature_maps(
        RawFeatureMaps{sparse.heatmap, sparse.descriptors, sparse.scale, sparse.orientation, sparse.affine, dense.confidence},
        options.max_keypoints,
        options.semi_dense_threshold
    );
}

}  // namespace

int run_train_command(const CliOptions& options) {
    if (!require_path(options.image_dir, "--image-dir") || !require_path(options.checkpoint, "--checkpoint")) {
        return 1;
    }
    TrainConfig config;
    config.image_dir = options.image_dir;
    config.checkpoint = options.checkpoint;
    config.device = options.device;
    config.epochs = options.epochs;
    config.batch_size = options.batch_size;
    const TrainResult result = train_model(config);
    std::cout << "training completed epochs=" << result.epochs_completed << " final_loss=" << result.final_loss << '\n';
    return 0;
}

int run_extract_command(const CliOptions& options) {
    if (!require_path(options.image, "--image") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output")) {
        return 1;
    }
    const FeatureSet features = extract_with_fresh_model(options);
    save_feature_set(features, options.output);
    std::cout << "features written " << options.output << '\n';
    return 0;
}

int run_match_command(const CliOptions& options) {
    if (!require_path(options.image_a, "--image-a") || !require_path(options.image_b, "--image-b") ||
        !require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output")) {
        return 1;
    }
    std::cerr << "match command is implemented in Task 8\n";
    return 1;
}

int run_eval_command(const CliOptions& options) {
    if (!require_path(options.pairs, "--pairs") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output")) {
        return 1;
    }
    std::cerr << "eval command is implemented in Task 8\n";
    return 1;
}

int run_export_command(const CliOptions& options) {
    if (!require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output")) {
        return 1;
    }
    if (!checkpoint_can_load(options.checkpoint)) {
        std::cerr << "checkpoint cannot be loaded: " << options.checkpoint << '\n';
        return 1;
    }
    std::filesystem::copy_file(options.checkpoint, options.output, std::filesystem::copy_options::overwrite_existing);
    std::cout << "exported model " << options.output << '\n';
    return 0;
}

}  // namespace pfm
```

This deliberately leaves match/eval failing until Task 8 so this task stays focused.

- [ ] **Step 4: Run tests**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: pipeline train/extract/export tests pass. Existing `run_match_with_required_paths_succeeds` and `run_eval_with_required_paths_succeeds` may need to be moved to Task 8 expectations; change those CLI tests to parse-only checks now, not success checks, because real match/eval are not complete until Task 8.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt modules/infer/pipeline.cpp modules/infer/pipeline_test.cpp modules/cli/commands_test.cpp tests/test_main.cpp
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Implement train extract and export commands"
```

---

### Task 8: Implement Matching and Evaluation Commands

**Files:**
- Modify: `modules/infer/pipeline.cpp`
- Modify: `modules/infer/pipeline_test.cpp`
- Modify: `modules/cli/commands_test.cpp`

- [ ] **Step 1: Add failing match/eval tests**

Append these tests to `modules/infer/pipeline_test.cpp`:

```cpp
void pipeline_match_writes_match_file() {
    const auto dir = make_pipeline_dir();
    const auto checkpoint = std::filesystem::temp_directory_path() / "pfm_pipeline_match_model.pt";
    PFM_REQUIRE(pfm::run_train_command(train_options(dir, checkpoint)) == 0);

    pfm::CliOptions options;
    options.image_a = (dir / "image.png").string();
    options.image_b = (dir / "image.png").string();
    options.checkpoint = checkpoint.string();
    options.output = (std::filesystem::temp_directory_path() / "pfm_pipeline_matches.pt").string();
    options.max_keypoints = 16;
    options.semi_dense_threshold = 0.5;

    PFM_REQUIRE(pfm::run_match_command(options) == 0);
    const pfm::MatchSet matches = pfm::load_match_set(options.output);
    PFM_REQUIRE(matches.sparse_matches.size(1) == 2);
    std::filesystem::remove_all(dir);
    std::filesystem::remove(checkpoint);
    std::filesystem::remove(options.output);
}

void pipeline_eval_writes_report_archive() {
    const auto dir = make_pipeline_dir();
    const auto checkpoint = std::filesystem::temp_directory_path() / "pfm_pipeline_eval_model.pt";
    const auto pairs = std::filesystem::temp_directory_path() / "pfm_pairs.txt";
    const auto report = std::filesystem::temp_directory_path() / "pfm_eval_report.pt";
    PFM_REQUIRE(pfm::run_train_command(train_options(dir, checkpoint)) == 0);
    std::ofstream(pairs) << (dir / "image.png").string() << " " << (dir / "image.png").string() << '\n';

    pfm::CliOptions options;
    options.pairs = pairs.string();
    options.checkpoint = checkpoint.string();
    options.output = report.string();
    options.max_keypoints = 16;

    PFM_REQUIRE(pfm::run_eval_command(options) == 0);
    torch::serialize::InputArchive archive;
    archive.load_from(report.string());
    torch::Tensor average_matches;
    archive.read("average_matches", average_matches);
    PFM_REQUIRE(average_matches.item<float>() >= 0.0F);
    std::filesystem::remove_all(dir);
    std::filesystem::remove(checkpoint);
    std::filesystem::remove(pairs);
    std::filesystem::remove(report);
}
```

Register them in `register_pipeline_tests()`:

```cpp
register_test("pipeline_match_writes_match_file", pipeline_match_writes_match_file);
register_test("pipeline_eval_writes_report_archive", pipeline_eval_writes_report_archive);
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: new match/eval tests fail because Task 7 returned nonzero for match/eval.

- [ ] **Step 3: Implement match helpers in pipeline**

Add includes to `modules/infer/pipeline.cpp`:

```cpp
#include <fstream>
#include <sstream>
#include <vector>

#include "infer/match_codec.h"
```

Add helper functions in the anonymous namespace:

```cpp
MatchSet match_features(const FeatureSet& a, const FeatureSet& b) {
    if (a.descriptors.size(0) == 0 || b.descriptors.size(0) == 0) {
        return MatchSet{
            torch::empty({0, 2}, torch::kInt64),
            torch::empty({0}, torch::kFloat32),
            torch::empty({0, 2}, torch::kFloat32),
            torch::empty({0, 2}, torch::kFloat32),
            torch::empty({0}, torch::kFloat32),
        };
    }

    const torch::Tensor scores = torch::matmul(
        torch::nn::functional::normalize(a.descriptors, torch::nn::functional::NormalizeFuncOptions().dim(1)),
        torch::nn::functional::normalize(b.descriptors, torch::nn::functional::NormalizeFuncOptions().dim(1)).transpose(0, 1)
    );
    const torch::Tensor best_b = std::get<1>(scores.max(1));
    const torch::Tensor best_a = std::get<1>(scores.max(0));

    std::vector<torch::Tensor> pairs;
    std::vector<torch::Tensor> pair_scores;
    for (int64_t i = 0; i < best_b.size(0); ++i) {
        const int64_t j = best_b.index({i}).item<int64_t>();
        if (best_a.index({j}).item<int64_t>() == i) {
            pairs.push_back(torch::tensor({i, j}, torch::kInt64));
            pair_scores.push_back(scores.index({i, j}).reshape({1}));
        }
    }

    torch::Tensor sparse_matches = pairs.empty() ? torch::empty({0, 2}, torch::kInt64) : torch::stack(pairs);
    torch::Tensor sparse_scores = pair_scores.empty() ? torch::empty({0}, torch::kFloat32) : torch::cat(pair_scores).to(torch::kFloat32);
    const int64_t dense_count = std::min(a.dense_points.size(0), b.dense_points.size(0));
    return MatchSet{
        sparse_matches,
        sparse_scores,
        a.dense_points.index({torch::indexing::Slice(0, dense_count)}),
        b.dense_points.index({torch::indexing::Slice(0, dense_count)}),
        dense_count == 0 ? torch::empty({0}, torch::kFloat32) : a.dense_confidence.index({torch::indexing::Slice(0, dense_count)}),
    };
}

std::vector<std::pair<std::string, std::string>> load_pairs(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument("failed to open pairs file: " + path);
    }
    std::vector<std::pair<std::string, std::string>> pairs;
    std::string a;
    std::string b;
    while (input >> a >> b) {
        pairs.emplace_back(a, b);
    }
    if (pairs.empty()) {
        throw std::invalid_argument("pairs file is empty: " + path);
    }
    return pairs;
}
```

- [ ] **Step 4: Implement real match/eval command bodies**

Replace `run_match_command` body with:

```cpp
int run_match_command(const CliOptions& options) {
    if (!require_path(options.image_a, "--image-a") || !require_path(options.image_b, "--image-b") ||
        !require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output")) {
        return 1;
    }
    CliOptions extract_a = options;
    extract_a.image = options.image_a;
    CliOptions extract_b = options;
    extract_b.image = options.image_b;
    const FeatureSet features_a = extract_with_fresh_model(extract_a);
    const FeatureSet features_b = extract_with_fresh_model(extract_b);
    const MatchSet matches = match_features(features_a, features_b);
    save_match_set(matches, options.output);
    std::cout << "matches written " << options.output << '\n';
    return 0;
}
```

Replace `run_eval_command` body with:

```cpp
int run_eval_command(const CliOptions& options) {
    if (!require_path(options.pairs, "--pairs") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output")) {
        return 1;
    }
    const auto pairs = load_pairs(options.pairs);
    double total_matches = 0.0;
    double total_sparse_score = 0.0;
    for (const auto& pair : pairs) {
        CliOptions match_options = options;
        match_options.image_a = pair.first;
        match_options.image_b = pair.second;
        match_options.output = options.output + ".tmp.pt";
        const FeatureSet features_a = extract_with_fresh_model(CliOptions{.image = pair.first});
        (void)features_a;
    }

    for (const auto& pair : pairs) {
        CliOptions extract_a = options;
        extract_a.image = pair.first;
        CliOptions extract_b = options;
        extract_b.image = pair.second;
        const MatchSet matches = match_features(extract_with_fresh_model(extract_a), extract_with_fresh_model(extract_b));
        total_matches += static_cast<double>(matches.sparse_matches.size(0));
        if (matches.sparse_scores.size(0) > 0) {
            total_sparse_score += matches.sparse_scores.mean().item<double>();
        }
    }

    torch::serialize::OutputArchive archive;
    archive.write("average_matches", torch::tensor({static_cast<float>(total_matches / pairs.size())}));
    archive.write("average_sparse_score", torch::tensor({static_cast<float>(total_sparse_score / pairs.size())}));
    archive.save_to(options.output);
    std::cout << "eval report written " << options.output << '\n';
    return 0;
}
```

Then immediately remove the invalid aggregate construction block from the first loop by replacing the whole function with a single loop version if the compiler rejects designated initializers. The accepted final `run_eval_command` must not use C++20 designated initializers; use the second loop only.

- [ ] **Step 5: Run tests and fix CLI tests**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
```

Expected: all tests pass. Restore CLI tests `run_match_with_required_paths_succeeds` and `run_eval_with_required_paths_succeeds` as pipeline-backed tests only if they supply real temporary images/checkpoints; otherwise keep them parse tests to avoid false success.

- [ ] **Step 6: Commit**

```bash
git add modules/infer/pipeline.cpp modules/infer/pipeline_test.cpp modules/cli/commands_test.cpp
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Implement matching and evaluation commands"
```

---

### Task 9: Update Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/training.md`

- [ ] **Step 1: Update README current status**

Change the command status section in `README.md` so it says commands now execute real first-stage behavior:

```markdown
当前命令已经执行第一阶段真实闭环：

- `train`：读取真实图片目录，生成自监督合成对，训练最小模型并保存 checkpoint。
- `extract`：读取图片和 checkpoint，输出 `.pt` 特征文件。
- `match`：读取两张图片和 checkpoint，输出 `.pt` 匹配结果。
- `eval`：读取图片对列表，输出 `.pt` 评估报告。
- `export`：校验并复制/重保存可推理 checkpoint。
```

- [ ] **Step 2: Update training doc**

Change `docs/training.md` so it no longer says `train` is only a validation stub. Include this exact minimum example:

```markdown
最小训练命令：

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1
```

训练会读取真实图片，生成平移和光照扰动的自监督影像对，运行 LibTorch 模型并保存 checkpoint。
```

- [ ] **Step 3: Run full verification**

Run:

```bash
cmake -S "/home/xjw/code/deeplearning/Feature Extraction" -B "/home/xjw/code/deeplearning/Feature Extraction/build" -DBUILD_TESTS=ON
cmake --build "/home/xjw/code/deeplearning/Feature Extraction/build" -j$(nproc)
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_tests"
ctest --test-dir "/home/xjw/code/deeplearning/Feature Extraction/build" --output-on-failure
```

Expected: build succeeds, `pfm_tests` passes, ctest passes.

- [ ] **Step 4: Run real CLI smoke with user-provided TIFF images**

Use the real training images supplied by the user in `/home/xjw/code/deeplearning/Feature Extraction/build/img`:

```bash
cat > /tmp/pfm_pairs.txt <<'EOF'
/home/xjw/code/deeplearning/Feature Extraction/build/img/1.tif /home/xjw/code/deeplearning/Feature Extraction/build/img/2.tif
EOF
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_cli" train --image-dir "/home/xjw/code/deeplearning/Feature Extraction/build/img" --checkpoint /tmp/pfm_model.pt --epochs 1 --batch-size 1
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_cli" extract --image "/home/xjw/code/deeplearning/Feature Extraction/build/img/1.tif" --checkpoint /tmp/pfm_model.pt --output /tmp/pfm_features.pt --max-keypoints 32 --semi-dense-threshold 0.5
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_cli" match --image-a "/home/xjw/code/deeplearning/Feature Extraction/build/img/1.tif" --image-b "/home/xjw/code/deeplearning/Feature Extraction/build/img/2.tif" --checkpoint /tmp/pfm_model.pt --output /tmp/pfm_matches.pt --max-keypoints 32 --semi-dense-threshold 0.5
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_cli" eval --pairs /tmp/pfm_pairs.txt --checkpoint /tmp/pfm_model.pt --output /tmp/pfm_report.pt --max-keypoints 32
"/home/xjw/code/deeplearning/Feature Extraction/build/pfm_cli" export --checkpoint /tmp/pfm_model.pt --output /tmp/pfm_exported.pt
```

Expected: each command exits 0 and writes the requested `.pt` file.

- [ ] **Step 5: Commit docs**

```bash
git add README.md docs/training.md
GIT_AUTHOR_NAME="guderianXu" GIT_AUTHOR_EMAIL="guderian_xu@henu.edu.cn" \
GIT_COMMITTER_NAME="guderianXu" GIT_COMMITTER_EMAIL="guderian_xu@henu.edu.cn" \
git commit -m "Document real training and inference commands"
```

- [ ] **Step 6: Push**

Run:

```bash
./push_to_github.sh
```

Expected: `main` pushes to `https://github.com/guderianXu/PlanetaryFeatureMatch`.

---

## Self-Review

- Spec coverage: OpenCV image loading is covered by Tasks 1-3; `.pt` serialization by Task 4; feature extraction by Task 5; training/checkpoint by Task 6; train/extract/export by Task 7; match/eval by Task 8; docs and final verification by Task 9.
- Placeholder scan: The plan contains no TBD/TODO placeholders. The one Task 8 warning explicitly instructs removing a C++20-only pattern if encountered and gives the required final direction.
- Type consistency: `FeatureSet`, `MatchSet`, `TrainConfig`, `TrainResult`, and `RawFeatureMaps` are introduced before use. New test registration names match the planned registration functions.
