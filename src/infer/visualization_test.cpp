#include <filesystem>
#include <random>
#include <string>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "infer/feature_codec.h"
#include "infer/match_codec.h"
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
            if (entry.is_directory(ignored)) {
                for (const auto& nested_entry : std::filesystem::directory_iterator(entry.path())) {
                    std::filesystem::remove(nested_entry.path(), ignored);
                }
            }
            std::filesystem::remove(entry.path(), ignored);
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

void write_scaled_test_image(const std::filesystem::path& path) {
    cv::Mat image(32, 40, CV_8UC1, cv::Scalar(40));
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

bool has_yellow_pixel_near(const cv::Mat& image, int center_x, int center_y) {
    for (int y = std::max(0, center_y - 3); y <= std::min(image.rows - 1, center_y + 3); ++y) {
        for (int x = std::max(0, center_x - 3); x <= std::min(image.cols - 1, center_x + 3); ++x) {
            const auto pixel = image.at<cv::Vec3b>(y, x);
            if (pixel[0] < 80 && pixel[1] > 150 && pixel[2] > 150) {
                return true;
            }
        }
    }
    return false;
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

pfm::MatchSet make_match_set(torch::Tensor points_a, torch::Tensor points_b, torch::Tensor confidence) {
    const auto count = points_a.size(0);
    return pfm::MatchSet{
        torch::stack({torch::arange(count, torch::kLong), torch::arange(count, torch::kLong)}, 1),
        confidence.to(torch::kFloat32).contiguous(),
        points_a.to(torch::kFloat32).contiguous(),
        points_b.to(torch::kFloat32).contiguous(),
        confidence.to(torch::kFloat32).contiguous()};
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

static void visualization_scales_feature_map_keypoints_to_image_pixels() {
    TempVisualizationDirectory temp_dir("pfm_visualize_scaled_features");
    const auto image_path = temp_dir.file("scaled.png");
    write_scaled_test_image(image_path);
    const auto features = make_feature_set(torch::tensor({{5.0F, 4.0F}}, torch::kFloat32));

    const auto output_path = pfm::save_feature_visualization(
        image_path.string(),
        features,
        temp_dir.file("vis").string(),
        10,
        8);
    const auto output = cv::imread(output_path.string(), cv::IMREAD_COLOR);

    PFM_REQUIRE(has_yellow_pixel_near(output, 20, 16));
    PFM_REQUIRE(!has_yellow_pixel_near(output, 5, 4));
}

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

static void visualization_draws_sparse_match_indices() {
    TempVisualizationDirectory temp_dir("pfm_visualize_sparse_matches");
    const auto image_a_path = temp_dir.file("left.png");
    const auto image_b_path = temp_dir.file("right.png");
    write_scaled_test_image(image_a_path);
    write_scaled_test_image(image_b_path);
    const auto features_a = make_feature_set(torch::tensor({{2.0F, 2.0F}, {5.0F, 4.0F}}, torch::kFloat32));
    const auto features_b = make_feature_set(torch::tensor({{6.0F, 5.0F}, {2.0F, 1.0F}}, torch::kFloat32));
    const pfm::MatchSet matches{
        torch::tensor({{1, 0}}, torch::kInt64),
        torch::tensor({1.0F}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)};

    const auto output_path = pfm::save_match_visualization(
        image_a_path.string(),
        image_b_path.string(),
        features_a,
        features_b,
        matches,
        temp_dir.file("vis").string(),
        10,
        8,
        10,
        8);
    const auto output = cv::imread(output_path.string(), cv::IMREAD_COLOR);

    PFM_REQUIRE(has_yellow_pixel_near(output, 20, 16));
    PFM_REQUIRE(has_yellow_pixel_near(output, 40 + 24, 20));
    PFM_REQUIRE(!has_yellow_pixel_near(output, 5, 4));
}

static void visualization_scales_match_points_to_image_pixels() {
    TempVisualizationDirectory temp_dir("pfm_visualize_scaled_matches");
    const auto image_a_path = temp_dir.file("left.png");
    const auto image_b_path = temp_dir.file("right.png");
    write_scaled_test_image(image_a_path);
    write_scaled_test_image(image_b_path);
    const auto matches = make_match_set(
        torch::tensor({{5.0F, 4.0F}}, torch::kFloat32),
        torch::tensor({{6.0F, 5.0F}}, torch::kFloat32),
        torch::tensor({1.0F}, torch::kFloat32));

    const auto output_path = pfm::save_match_visualization(
        image_a_path.string(),
        image_b_path.string(),
        matches,
        temp_dir.file("vis").string(),
        10,
        8,
        10,
        8);
    const auto output = cv::imread(output_path.string(), cv::IMREAD_COLOR);

    PFM_REQUIRE(has_yellow_pixel_near(output, 20, 16));
    PFM_REQUIRE(has_yellow_pixel_near(output, 40 + 24, 20));
    PFM_REQUIRE(!has_yellow_pixel_near(output, 5, 4));
}

void register_visualization_tests() {
    register_test("visualization writes feature keypoint png", visualization_writes_feature_keypoint_png);
    register_test("visualization writes feature png without keypoints", visualization_writes_feature_png_without_keypoints);
    register_test("visualization scales feature map keypoints to image pixels",
                  visualization_scales_feature_map_keypoints_to_image_pixels);
    register_test("visualization writes match png", visualization_writes_match_png);
    register_test("visualization writes match png without matches", visualization_writes_match_png_without_matches);
    register_test("visualization draws sparse match indices", visualization_draws_sparse_match_indices);
    register_test("visualization scales match points to image pixels", visualization_scales_match_points_to_image_pixels);
}
