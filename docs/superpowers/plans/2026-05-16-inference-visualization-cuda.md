# Inference Visualization CUDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--visualization-dir` to extract and match so users can save PNG overlays of feature points and matches while documenting the current CUDA inference scope.

**Architecture:** Keep inference outputs unchanged and add a small `modules/infer/visualization` module that consumes existing `FeatureSet` and `MatchSet` data. CLI parsing stores the visualization directory in `CliOptions`; pipeline writes normal `.pt` outputs first, then optionally creates PNG visualizations.

**Tech Stack:** C++17, CMake, LibTorch tensors, OpenCV image loading/drawing/writing, existing custom C++ test harness.

---

## File Structure

- Create `modules/infer/visualization.h`: public visualization API declarations.
- Create `modules/infer/visualization.cpp`: OpenCV drawing implementation, filename sanitization, directory creation.
- Create `modules/infer/visualization_test.cpp`: unit tests for feature and match PNG output.
- Modify `modules/cli/commands.h`: add `std::string visualization_dir` to `CliOptions`.
- Modify `modules/cli/commands.cpp`: add `--visualization-dir` for extract and match help/parser.
- Modify `modules/cli/commands_test.cpp`: parser and help assertions.
- Modify `modules/infer/pipeline.cpp`: call visualization after saving extract/match `.pt` files.
- Modify `modules/infer/pipeline_test.cpp`: pipeline tests for generated PNG files.
- Modify `CMakeLists.txt`: add visualization source and test file.
- Modify `tests/test_main.cpp`: register visualization tests.
- Modify `README.md`, `docs/usage.md`, `docs/training.md`: document CUDA scope and visualization usage.

---

### Task 1: CLI option plumbing

**Files:**
- Modify: `modules/cli/commands.h`
- Modify: `modules/cli/commands.cpp`
- Test: `modules/cli/commands_test.cpp`

- [ ] **Step 1: Write failing parser tests**

In `modules/cli/commands_test.cpp`, update `parse_extract_command()` to include the option and assertion:

```cpp
static void parse_extract_command() {
    const auto parsed = pfm::parse_cli({
        "pfm",
        "extract",
        "--image",
        "a.png",
        "--checkpoint",
        "model.pt",
        "--output",
        "a.pfm",
        "--visualization-dir",
        "vis",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Extract);
    PFM_REQUIRE(parsed.image == "a.png");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.output == "a.pfm");
    PFM_REQUIRE(parsed.visualization_dir == "vis");
}
```

Update `parse_match_command()` to include the option and assertion:

```cpp
static void parse_match_command() {
    const auto parsed = pfm::parse_cli({
        "pfm",
        "match",
        "--image-a",
        "a.png",
        "--image-b",
        "b.png",
        "--checkpoint",
        "model.pt",
        "--output",
        "matches.json",
        "--max-keypoints",
        "2048",
        "--semi-dense-threshold",
        "0.5",
        "--visualization-dir",
        "vis",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Match);
    PFM_REQUIRE(parsed.image_a == "a.png");
    PFM_REQUIRE(parsed.image_b == "b.png");
    PFM_REQUIRE(parsed.max_keypoints == 2048);
    PFM_REQUIRE_CLOSE(parsed.semi_dense_threshold, 0.5, 1.0e-6);
    PFM_REQUIRE(parsed.visualization_dir == "vis");
}
```

Update `top_level_help_lists_subcommand_options()` with:

```cpp
PFM_REQUIRE(help.find("--visualization-dir") != std::string::npos);
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: build fails with errors like `const struct pfm::CliOptions has no member named visualization_dir`.

- [ ] **Step 3: Add `visualization_dir` to `CliOptions`**

In `modules/cli/commands.h`, add the field after `output`:

```cpp
std::string visualization_dir;
```

- [ ] **Step 4: Bind `--visualization-dir` for extract and match**

In `modules/cli/commands.cpp`, update the common command options help text so extract and match mention the new option:

```text
extract --image a.tif --checkpoint model.pt --output features.pt [--device cpu] [--max-keypoints 1024] [--semi-dense-threshold 0.5] [--visualization-dir vis]
match --image-a a.tif --image-b b.tif --checkpoint model.pt --output matches.pt [--device cpu] [--max-keypoints 1024] [--semi-dense-threshold 0.5] [--visualization-dir vis]
```

In the extract subcommand setup, add:

```cpp
extract->add_option("--visualization-dir", options.visualization_dir, "Directory for feature visualization PNG output");
```

In the match subcommand setup, add:

```cpp
match->add_option("--visualization-dir", options.visualization_dir, "Directory for match visualization PNG output");
```

- [ ] **Step 5: Run tests to verify CLI plumbing passes**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all tests pass, or only later visualization tests fail if already added.

---

### Task 2: Feature visualization module

**Files:**
- Create: `modules/infer/visualization.h`
- Create: `modules/infer/visualization.cpp`
- Test: `modules/infer/visualization_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Write failing feature visualization tests**

Create `modules/infer/visualization_test.cpp` with:

```cpp
#include <filesystem>
#include <random>
#include <string>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "infer/feature_codec.h"
#include "infer/visualization.h"
#include "tests/test_harness.h"

namespace {

class TempVisualizationDirectory {
public:
    explicit TempVisualizationDirectory(const std::string& stem) {
        const auto suffix = std::to_string(static_cast<long long>(getpid())) + "_" +
                            std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempVisualizationDirectory() {
        std::error_code ignored;
        for (const auto& entry : std::filesystem::directory_iterator(_path)) {
            std::filesystem::remove_all(entry.path(), ignored);
        }
        std::filesystem::remove(_path, ignored);
    }

    const std::filesystem::path& path() const {
        return _path;
    }

    std::filesystem::path file(const std::string& name) const {
        return _path / name;
    }

private:
    std::filesystem::path _path;
};

void write_test_image(const std::filesystem::path& path) {
    cv::Mat image(24, 32, CV_8UC1, cv::Scalar(40));
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

pfm::FeatureSet make_feature_set(torch::Tensor keypoints) {
    const auto count = keypoints.size(0);
    return pfm::FeatureSet{
        keypoints.to(torch::kFloat32).contiguous(),
        torch::ones({count}, torch::kFloat32),
        torch::zeros({count, 8}, torch::kFloat32),
        torch::ones({count}, torch::kFloat32),
        torch::zeros({count}, torch::kFloat32),
        torch::zeros({count, 2, 2}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)};
}

}  // namespace

static void visualization_writes_feature_keypoint_png() {
    TempVisualizationDirectory temp_dir("pfm_visualize_features");
    const auto image_path = temp_dir.file("source image.png");
    write_test_image(image_path);
    const auto features = make_feature_set(torch::tensor({{4.0F, 5.0F}, {20.0F, 12.0F}}, torch::kFloat32));

    const auto output_path = pfm::save_feature_visualization(image_path.string(), features, temp_dir.file("vis").string());

    PFM_REQUIRE(output_path.filename().string() == "source_image_features.png");
    PFM_REQUIRE(std::filesystem::exists(output_path));
    PFM_REQUIRE(!cv::imread(output_path.string(), cv::IMREAD_COLOR).empty());
}

static void visualization_writes_feature_png_without_keypoints() {
    TempVisualizationDirectory temp_dir("pfm_visualize_empty_features");
    const auto image_path = temp_dir.file("empty.png");
    write_test_image(image_path);
    const auto features = make_feature_set(torch::empty({0, 2}, torch::kFloat32));

    const auto output_path = pfm::save_feature_visualization(image_path.string(), features, temp_dir.file("vis").string());

    PFM_REQUIRE(std::filesystem::exists(output_path));
}

void register_visualization_tests() {
    register_test("visualization writes feature keypoint png", visualization_writes_feature_keypoint_png);
    register_test("visualization writes feature png without keypoints", visualization_writes_feature_png_without_keypoints);
}
```

- [ ] **Step 2: Register test file in build and test main**

In `CMakeLists.txt`, add `modules/infer/visualization_test.cpp` to `pfm_tests` sources.

In `tests/test_main.cpp`, add a forward declaration:

```cpp
void register_visualization_tests();
```

Then call it near other infer registrations:

```cpp
register_visualization_tests();
```

- [ ] **Step 3: Run tests to verify feature visualization fails**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: build fails because `infer/visualization.h` and `save_feature_visualization` do not exist.

- [ ] **Step 4: Add visualization header**

Create `modules/infer/visualization.h`:

```cpp
#pragma once

#include <filesystem>
#include <string>

#include "infer/feature_codec.h"
#include "infer/match_codec.h"

namespace pfm {

/// Saves a PNG overlay showing extracted feature points on the input image.
/// @param image_path Source image path used as the visualization background.
/// @param feature_set Decoded feature tensors with keypoints in image coordinate order {x, y}.
/// @param visualization_dir Directory where the PNG will be written; created if missing.
/// @return Path to the written PNG file.
/// @throws std::invalid_argument if the image cannot be read or the PNG cannot be written.
std::filesystem::path save_feature_visualization(
    const std::string& image_path,
    const FeatureSet& feature_set,
    const std::string& visualization_dir
);

/// Saves a PNG side-by-side overlay showing matched points between two images.
/// @param image_a_path First source image path.
/// @param image_b_path Second source image path.
/// @param match_set Match tensors with points in image coordinate order {x, y} and confidence scores.
/// @param visualization_dir Directory where the PNG will be written; created if missing.
/// @return Path to the written PNG file.
/// @throws std::invalid_argument if either image cannot be read or the PNG cannot be written.
std::filesystem::path save_match_visualization(
    const std::string& image_a_path,
    const std::string& image_b_path,
    const MatchSet& match_set,
    const std::string& visualization_dir
);

}  // namespace pfm
```

- [ ] **Step 5: Add minimal feature visualization implementation**

Create `modules/infer/visualization.cpp` with feature visualization implemented and match visualization as a stub that throws until Task 3:

```cpp
#include "infer/visualization.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

namespace pfm {
namespace {

std::string sanitized_stem(const std::string& image_path) {
    auto stem = std::filesystem::path(image_path).stem().string();
    if (stem.empty()) {
        stem = "image";
    }
    for (char& value : stem) {
        const auto byte = static_cast<unsigned char>(value);
        if (!std::isalnum(byte) && value != '-' && value != '_') {
            value = '_';
        }
    }
    return stem;
}

cv::Mat read_color_image(const std::string& image_path) {
    auto image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::invalid_argument("failed to read visualization image: " + image_path);
    }
    return image;
}

void write_png(const std::filesystem::path& output_path, const cv::Mat& image) {
    std::filesystem::create_directories(output_path.parent_path());
    if (!cv::imwrite(output_path.string(), image)) {
        throw std::invalid_argument("failed to write visualization png: " + output_path.string());
    }
}

void draw_keypoints(cv::Mat& image, const torch::Tensor& keypoints) {
    if (!keypoints.defined() || keypoints.numel() == 0) {
        return;
    }
    const auto points = keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    if (points.dim() != 2 || points.size(1) != 2) {
        throw std::invalid_argument("feature visualization keypoints must have shape {N,2}");
    }
    for (int64_t index = 0; index < points.size(0); ++index) {
        const auto x = static_cast<int>(std::round(points.index({index, 0}).item<float>()));
        const auto y = static_cast<int>(std::round(points.index({index, 1}).item<float>()));
        if (x >= 0 && y >= 0 && x < image.cols && y < image.rows) {
            cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }
    }
}

}  // namespace

std::filesystem::path save_feature_visualization(
    const std::string& image_path,
    const FeatureSet& feature_set,
    const std::string& visualization_dir
) {
    auto image = read_color_image(image_path);
    draw_keypoints(image, feature_set.keypoints);
    const auto output_path = std::filesystem::path(visualization_dir) / (sanitized_stem(image_path) + "_features.png");
    write_png(output_path, image);
    return output_path;
}

std::filesystem::path save_match_visualization(
    const std::string&,
    const std::string&,
    const MatchSet&,
    const std::string&
) {
    throw std::invalid_argument("match visualization is not implemented");
}

}  // namespace pfm
```

- [ ] **Step 6: Add implementation source to CMake**

In `CMakeLists.txt`, add `modules/infer/visualization.cpp` to the `pfm` library sources.

- [ ] **Step 7: Run tests to verify feature visualization passes**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: feature visualization tests pass. No match visualization tests exist yet.

---

### Task 3: Match visualization module

**Files:**
- Modify: `modules/infer/visualization_test.cpp`
- Modify: `modules/infer/visualization.cpp`

- [ ] **Step 1: Write failing match visualization tests**

In `modules/infer/visualization_test.cpp`, include match codec support if not already available:

```cpp
#include "infer/match_codec.h"
```

Add this helper inside the anonymous namespace:

```cpp
pfm::MatchSet make_match_set(torch::Tensor points_a, torch::Tensor points_b, torch::Tensor confidence) {
    const auto count = points_a.size(0);
    return pfm::MatchSet{
        torch::stack({torch::arange(count, torch::kLong), torch::arange(count, torch::kLong)}, 1),
        confidence.to(torch::kFloat32).contiguous(),
        points_a.to(torch::kFloat32).contiguous(),
        points_b.to(torch::kFloat32).contiguous(),
        confidence.to(torch::kFloat32).contiguous()};
}
```

Add tests:

```cpp
static void visualization_writes_match_png() {
    TempVisualizationDirectory temp_dir("pfm_visualize_matches");
    const auto image_a_path = temp_dir.file("left image.png");
    const auto image_b_path = temp_dir.file("right image.png");
    write_test_image(image_a_path);
    write_test_image(image_b_path);
    const auto matches = make_match_set(
        torch::tensor({{4.0F, 5.0F}, {18.0F, 10.0F}}, torch::kFloat32),
        torch::tensor({{6.0F, 5.0F}, {20.0F, 12.0F}}, torch::kFloat32),
        torch::tensor({0.4F, 0.9F}, torch::kFloat32));

    const auto output_path = pfm::save_match_visualization(
        image_a_path.string(),
        image_b_path.string(),
        matches,
        temp_dir.file("vis").string());

    PFM_REQUIRE(output_path.filename().string() == "left_image__right_image_matches.png");
    PFM_REQUIRE(std::filesystem::exists(output_path));
    PFM_REQUIRE(!cv::imread(output_path.string(), cv::IMREAD_COLOR).empty());
}

static void visualization_writes_match_png_without_matches() {
    TempVisualizationDirectory temp_dir("pfm_visualize_empty_matches");
    const auto image_a_path = temp_dir.file("left.png");
    const auto image_b_path = temp_dir.file("right.png");
    write_test_image(image_a_path);
    write_test_image(image_b_path);
    const auto matches = make_match_set(
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));

    const auto output_path = pfm::save_match_visualization(
        image_a_path.string(),
        image_b_path.string(),
        matches,
        temp_dir.file("vis").string());

    PFM_REQUIRE(std::filesystem::exists(output_path));
}
```

Register them in `register_visualization_tests()`:

```cpp
register_test("visualization writes match png", visualization_writes_match_png);
register_test("visualization writes match png without matches", visualization_writes_match_png_without_matches);
```

- [ ] **Step 2: Run tests to verify match visualization fails**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: `visualization writes match png` fails with `match visualization is not implemented`.

- [ ] **Step 3: Implement match visualization**

In `modules/infer/visualization.cpp`, add helpers inside the anonymous namespace:

```cpp
constexpr int64_t MAX_DRAWN_MATCHES = 200;

void validate_match_points(const torch::Tensor& points_a, const torch::Tensor& points_b, const torch::Tensor& confidence) {
    if (!points_a.defined() || !points_b.defined() || !confidence.defined()) {
        throw std::invalid_argument("match visualization tensors must be defined");
    }
    if (points_a.dim() != 2 || points_b.dim() != 2 || points_a.size(1) != 2 || points_b.size(1) != 2) {
        throw std::invalid_argument("match visualization points must have shape {N,2}");
    }
    if (confidence.dim() != 1 || confidence.size(0) != points_a.size(0) || points_b.size(0) != points_a.size(0)) {
        throw std::invalid_argument("match visualization confidence must match point count");
    }
}

cv::Mat make_side_by_side(const cv::Mat& image_a, const cv::Mat& image_b) {
    const auto height = std::max(image_a.rows, image_b.rows);
    cv::Mat canvas(height, image_a.cols + image_b.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    image_a.copyTo(canvas(cv::Rect(0, 0, image_a.cols, image_a.rows)));
    image_b.copyTo(canvas(cv::Rect(image_a.cols, 0, image_b.cols, image_b.rows)));
    return canvas;
}

std::vector<int64_t> sorted_match_indices(const torch::Tensor& confidence) {
    std::vector<int64_t> indices(static_cast<std::size_t>(confidence.size(0)));
    for (int64_t index = 0; index < confidence.size(0); ++index) {
        indices[static_cast<std::size_t>(index)] = index;
    }
    std::sort(indices.begin(), indices.end(), [&](int64_t left, int64_t right) {
        return confidence.index({left}).item<float>() > confidence.index({right}).item<float>();
    });
    if (indices.size() > static_cast<std::size_t>(MAX_DRAWN_MATCHES)) {
        indices.resize(static_cast<std::size_t>(MAX_DRAWN_MATCHES));
    }
    return indices;
}

void draw_matches(cv::Mat& canvas, int image_b_offset, const MatchSet& match_set) {
    const auto points_a = match_set.points_a.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto points_b = match_set.points_b.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto confidence = match_set.confidence.to(torch::kCPU, torch::kFloat32).contiguous();
    validate_match_points(points_a, points_b, confidence);
    for (const auto index : sorted_match_indices(confidence)) {
        const auto score = std::max(0.0F, std::min(1.0F, confidence.index({index}).item<float>()));
        const cv::Scalar color(0, 80.0 + 175.0 * score, 255.0 * score);
        const cv::Point point_a(
            static_cast<int>(std::round(points_a.index({index, 0}).item<float>())),
            static_cast<int>(std::round(points_a.index({index, 1}).item<float>())));
        const cv::Point point_b(
            image_b_offset + static_cast<int>(std::round(points_b.index({index, 0}).item<float>())),
            static_cast<int>(std::round(points_b.index({index, 1}).item<float>())));
        cv::line(canvas, point_a, point_b, color, 1, cv::LINE_AA);
        cv::circle(canvas, point_a, 2, color, 1, cv::LINE_AA);
        cv::circle(canvas, point_b, 2, color, 1, cv::LINE_AA);
    }
}
```

Replace the `save_match_visualization()` stub with:

```cpp
std::filesystem::path save_match_visualization(
    const std::string& image_a_path,
    const std::string& image_b_path,
    const MatchSet& match_set,
    const std::string& visualization_dir
) {
    const auto image_a = read_color_image(image_a_path);
    const auto image_b = read_color_image(image_b_path);
    auto canvas = make_side_by_side(image_a, image_b);
    draw_matches(canvas, image_a.cols, match_set);
    const auto output_path = std::filesystem::path(visualization_dir) /
                             (sanitized_stem(image_a_path) + "__" + sanitized_stem(image_b_path) + "_matches.png");
    write_png(output_path, canvas);
    return output_path;
}
```

- [ ] **Step 4: Run tests to verify match visualization passes**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: visualization tests pass.

---

### Task 4: Pipeline integration

**Files:**
- Modify: `modules/infer/pipeline.cpp`
- Test: `modules/infer/pipeline_test.cpp`

- [ ] **Step 1: Write failing pipeline tests**

In `modules/infer/pipeline_test.cpp`, add assertions to the existing extract/match pipeline tests rather than creating slow duplicate training setup.

In `pipeline_extract_writes_loadable_feature_file()`, set:

```cpp
options.visualization_dir = temp_dir.file("vis").string();
```

After the existing feature file assertions, add:

```cpp
PFM_REQUIRE(std::filesystem::exists(temp_dir.file("vis") / "target_features.png"));
```

Use the actual image filename stem already created in that test. If the test image is not named `target.png`, use its current stem with `_features.png`.

In `pipeline_match_writes_match_file()`, set:

```cpp
options.visualization_dir = temp_dir.file("vis").string();
```

After the existing match file assertions, add:

```cpp
PFM_REQUIRE(std::filesystem::exists(temp_dir.file("vis") / "image_a__image_b_matches.png"));
```

Use the actual two image stems already created in that test.

- [ ] **Step 2: Run tests to verify pipeline integration fails**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: pipeline tests fail because no visualization PNG is written.

- [ ] **Step 3: Call visualization functions from pipeline**

In `modules/infer/pipeline.cpp`, include:

```cpp
#include "infer/visualization.h"
```

In `run_extract_command()`, after `save_feature_set(feature_set, options.output);`, add:

```cpp
if (!options.visualization_dir.empty()) {
    (void)save_feature_visualization(options.image, feature_set, options.visualization_dir);
}
```

In `run_match_command()`, replace the direct save call:

```cpp
save_match_set(matchFeatureSets(features_a, features_b), options.output);
```

with:

```cpp
const auto match_set = matchFeatureSets(features_a, features_b);
save_match_set(match_set, options.output);
if (!options.visualization_dir.empty()) {
    (void)save_match_visualization(options.image_a, options.image_b, match_set, options.visualization_dir);
}
```

- [ ] **Step 4: Run tests to verify pipeline integration passes**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_tests
```

Expected: all tests pass.

---

### Task 5: Documentation and help verification

**Files:**
- Modify: `README.md`
- Modify: `docs/usage.md`
- Modify: `docs/training.md`

- [ ] **Step 1: Update README usage examples**

In `README.md`, add `--visualization-dir vis` to the extract example and match example. Add a short Chinese paragraph:

```markdown
推理命令可以加 `--device cuda` 使用 GPU 跑模型 forward。当前 CUDA 覆盖模型前向；图像读取、特征解码、匹配后处理、PNG 可视化和 `.pt` 写出仍在 CPU。若要肉眼观察效果，可以给 `extract` 或 `match` 添加 `--visualization-dir vis`：`extract` 会生成 `<image_stem>_features.png`，`match` 会生成 `<image_a_stem>__<image_b_stem>_matches.png`。
```

- [ ] **Step 2: Update docs/usage.md examples**

In `docs/usage.md`, add `--visualization-dir vis` to the extract and match command examples. Add text under the matching or output section:

```markdown
`--visualization-dir` 会自动创建目录并保存 PNG：特征提取保存特征点覆盖图，图像匹配保存左右拼接的匹配连线图。该参数不改变 `.pt` 输出，主要用于人工检查模型效果。
```

- [ ] **Step 3: Update docs/training.md CUDA note**

In `docs/training.md`, update the CUDA paragraph to state:

```markdown
推理侧 `extract` 和 `match` 也支持 `--device cuda`，模型 forward 会在 GPU 上执行；特征解码、匹配后处理、PNG 可视化和 `.pt` 写出仍在 CPU。训练或推理功耗没有接近显卡 TDP 时，通常是 batch、输入分辨率、模型规模或 CPU 数据准备限制导致 GPU 等待，并不等同于没有使用 CUDA。
```

- [ ] **Step 4: Verify CLI help includes the new option**

Run:

```bash
cmake --build build -j$(nproc) && ./build/pfm_cli extract --help && ./build/pfm_cli match --help
```

Expected: both help outputs include `--visualization-dir`.

- [ ] **Step 5: Run full verification**

Run:

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

Expected: configure and build succeed, `./build/pfm_tests` reports all tests passed, and `ctest` reports `100% tests passed`.

---

## Self-Review

- Spec coverage: The plan covers `--visualization-dir` for extract/match, feature PNG output, match PNG output, unchanged `.pt` output, CUDA scope documentation, tests, and verification commands.
- Placeholder scan: No placeholders, no TBDs, and each implementation step includes exact code or exact commands.
- Type consistency: The plan consistently uses `CliOptions::visualization_dir`, `save_feature_visualization`, `save_match_visualization`, `FeatureSet`, and `MatchSet`.
