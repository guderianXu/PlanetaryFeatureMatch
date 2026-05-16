#include "infer/visualization.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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

torch::Tensor scaled_keypoints(const torch::Tensor& keypoints, int image_width, int image_height, int64_t map_width, int64_t map_height) {
    if (!keypoints.defined() || keypoints.numel() == 0) {
        return keypoints;
    }
    if (map_width <= 0 || map_height <= 0) {
        throw std::invalid_argument("feature visualization map dimensions must be positive");
    }
    auto points = keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    if (points.dim() != 2 || points.size(1) != 2) {
        throw std::invalid_argument("feature visualization keypoints must have shape {N,2}");
    }
    points.index_put_({torch::indexing::Slice(), 0},
                      points.index({torch::indexing::Slice(), 0}) * static_cast<double>(image_width) /
                          static_cast<double>(map_width));
    points.index_put_({torch::indexing::Slice(), 1},
                      points.index({torch::indexing::Slice(), 1}) * static_cast<double>(image_height) /
                          static_cast<double>(map_height));
    return points;
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
    constexpr int64_t max_drawn_matches = 200;
    std::vector<int64_t> indices(static_cast<std::size_t>(confidence.size(0)));
    for (int64_t index = 0; index < confidence.size(0); ++index) {
        indices[static_cast<std::size_t>(index)] = index;
    }
    std::sort(indices.begin(), indices.end(), [&](int64_t left, int64_t right) {
        return confidence.index({left}).item<float>() > confidence.index({right}).item<float>();
    });
    if (indices.size() > static_cast<std::size_t>(max_drawn_matches)) {
        indices.resize(static_cast<std::size_t>(max_drawn_matches));
    }
    return indices;
}

MatchSet scaled_match_set(
    const MatchSet& match_set,
    int image_a_width,
    int image_a_height,
    int image_b_width,
    int image_b_height,
    int64_t map_a_width,
    int64_t map_a_height,
    int64_t map_b_width,
    int64_t map_b_height
) {
    return MatchSet{
        match_set.sparse_matches,
        match_set.sparse_scores,
        scaled_keypoints(match_set.points_a, image_a_width, image_a_height, map_a_width, map_a_height),
        scaled_keypoints(match_set.points_b, image_b_width, image_b_height, map_b_width, map_b_height),
        match_set.confidence};
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

std::filesystem::path save_feature_visualization(
    const std::string& image_path,
    const FeatureSet& feature_set,
    const std::string& visualization_dir,
    int64_t feature_map_width,
    int64_t feature_map_height
) {
    auto image = read_color_image(image_path);
    draw_keypoints(
        image,
        scaled_keypoints(feature_set.keypoints, image.cols, image.rows, feature_map_width, feature_map_height));
    const auto output_path = std::filesystem::path(visualization_dir) / (sanitized_stem(image_path) + "_features.png");
    write_png(output_path, image);
    return output_path;
}

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

std::filesystem::path save_match_visualization(
    const std::string& image_a_path,
    const std::string& image_b_path,
    const MatchSet& match_set,
    const std::string& visualization_dir,
    int64_t feature_map_a_width,
    int64_t feature_map_a_height,
    int64_t feature_map_b_width,
    int64_t feature_map_b_height
) {
    const auto image_a = read_color_image(image_a_path);
    const auto image_b = read_color_image(image_b_path);
    auto canvas = make_side_by_side(image_a, image_b);
    draw_matches(
        canvas,
        image_a.cols,
        scaled_match_set(
            match_set,
            image_a.cols,
            image_a.rows,
            image_b.cols,
            image_b.rows,
            feature_map_a_width,
            feature_map_a_height,
            feature_map_b_width,
            feature_map_b_height));
    const auto output_path = std::filesystem::path(visualization_dir) /
                             (sanitized_stem(image_a_path) + "__" + sanitized_stem(image_b_path) + "_matches.png");
    write_png(output_path, canvas);
    return output_path;
}

}  // namespace pfm
