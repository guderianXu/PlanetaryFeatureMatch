#include "infer/match_metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/serialize.h>

namespace pfm
{
namespace
{

std::pair<int64_t, int64_t> map_feature_point_to_image(const torch::Tensor& point, int64_t map_width,
                                                       int64_t map_height, int64_t image_width, int64_t image_height)
{
    const auto scale_x = static_cast<float>(image_width) / static_cast<float>(std::max<int64_t>(1, map_width));
    const auto scale_y = static_cast<float>(image_height) / static_cast<float>(std::max<int64_t>(1, map_height));
    const int64_t x = std::min<int64_t>(
        image_width - 1, std::max<int64_t>(0, std::llround((point.index({0}).item<float>() + 0.5F) * scale_x - 0.5F)));
    const int64_t y = std::min<int64_t>(
        image_height - 1, std::max<int64_t>(0, std::llround((point.index({1}).item<float>() + 0.5F) * scale_y - 0.5F)));
    return {x, y};
}

std::pair<double, double> map_feature_point_to_image_float(const torch::Tensor& point, int64_t map_width,
                                                           int64_t map_height, int64_t image_width,
                                                           int64_t image_height)
{
    const auto scale_x = static_cast<double>(image_width) / static_cast<double>(std::max<int64_t>(1, map_width));
    const auto scale_y = static_cast<double>(image_height) / static_cast<double>(std::max<int64_t>(1, map_height));
    const auto x = (static_cast<double>(point.index({0}).item<float>()) + 0.5) * scale_x - 0.5;
    const auto y = (static_cast<double>(point.index({1}).item<float>()) + 0.5) * scale_y - 0.5;
    return {std::min<double>(image_width - 1, std::max<double>(0.0, x)),
            std::min<double>(image_height - 1, std::max<double>(0.0, y))};
}

bool is_correct_pixel_match(const torch::Tensor& warp, int64_t x_a, int64_t y_a, int64_t x_b, int64_t y_b,
                            double threshold)
{
    if (y_a < 0 || y_a >= warp.size(0) || x_a < 0 || x_a >= warp.size(1))
    {
        return false;
    }
    const auto expected_x = warp.index({y_a, x_a, 0}).item<float>();
    const auto expected_y = warp.index({y_a, x_a, 1}).item<float>();
    const auto dx = static_cast<double>(x_b) - static_cast<double>(expected_x);
    const auto dy = static_cast<double>(y_b) - static_cast<double>(expected_y);
    return std::sqrt(dx * dx + dy * dy) <= threshold;
}

double descriptor_cosine(const torch::Tensor& lhs, const torch::Tensor& rhs)
{
    const auto numerator = (lhs * rhs).sum().item<float>();
    const auto lhs_norm = std::sqrt(lhs.pow(2).sum().item<float>());
    const auto rhs_norm = std::sqrt(rhs.pow(2).sum().item<float>());
    const auto denom = std::max(1.0e-12, static_cast<double>(lhs_norm) * static_cast<double>(rhs_norm));
    return static_cast<double>(numerator) / denom;
}

void require_warp_shape(const torch::Tensor& warp_a_to_b)
{
    if (!warp_a_to_b.defined() || warp_a_to_b.dim() != 3 || warp_a_to_b.size(2) != 2)
    {
        throw std::invalid_argument("warp_a_to_b must have shape HxWx2");
    }
}

} // namespace

int64_t WarpMatchMetrics::total() const
{
    return sparse_total + dense_total;
}

int64_t WarpMatchMetrics::correct() const
{
    return sparse_correct + dense_correct;
}

double WarpMatchMetrics::precision() const
{
    const auto total_count = total();
    return total_count == 0 ? 0.0 : static_cast<double>(correct()) / static_cast<double>(total_count);
}

torch::Tensor load_warp_a_to_b_tensor(const std::string& path)
{
    try
    {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        torch::Tensor warp;
        archive.read("warp_a_to_b", warp);
        require_warp_shape(warp);
        return warp.to(torch::kCPU, torch::kFloat32).contiguous();
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

WarpMatchMetrics compute_warp_match_metrics(const FeatureSet& features_a, const FeatureSet& features_b,
                                            const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                            double correct_threshold_pixels)
{
    if (!std::isfinite(correct_threshold_pixels) || correct_threshold_pixels < 0.0)
    {
        throw std::invalid_argument("correct_threshold_pixels must be non-negative and finite");
    }
    require_warp_shape(warp_a_to_b);
    auto warp = warp_a_to_b.to(torch::kCPU, torch::kFloat32).contiguous();

    WarpMatchMetrics metrics;
    if (matches.sparse_matches.defined() && matches.sparse_matches.numel() > 0 && features_a.keypoints.defined() &&
        features_b.keypoints.defined())
    {
        auto sparse_matches = matches.sparse_matches.to(torch::kCPU, torch::kLong).contiguous();
        auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
        auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
        for (int64_t index = 0; index < sparse_matches.size(0); ++index)
        {
            const auto index_a = sparse_matches.index({index, 0}).item<int64_t>();
            const auto index_b = sparse_matches.index({index, 1}).item<int64_t>();
            if (index_a < 0 || index_a >= points_a.size(0) || index_b < 0 || index_b >= points_b.size(0))
            {
                continue;
            }
            const auto [x_a, y_a] =
                map_feature_point_to_image(points_a[index_a], features_a.feature_map_width,
                                           features_a.feature_map_height, warp.size(1), warp.size(0));
            const auto [x_b, y_b] =
                map_feature_point_to_image(points_b[index_b], features_b.feature_map_width,
                                           features_b.feature_map_height, warp.size(1), warp.size(0));
            ++metrics.sparse_total;
            if (is_correct_pixel_match(warp, x_a, y_a, x_b, y_b, correct_threshold_pixels))
            {
                ++metrics.sparse_correct;
            }
        }
    }

    if (matches.points_a.defined() && matches.points_b.defined() && matches.points_a.numel() > 0)
    {
        auto points_a = matches.points_a.to(torch::kCPU, torch::kFloat32).contiguous();
        auto points_b = matches.points_b.to(torch::kCPU, torch::kFloat32).contiguous();
        const int64_t count = std::min<int64_t>(points_a.size(0), points_b.size(0));
        for (int64_t index = 0; index < count; ++index)
        {
            const auto [x_a, y_a] =
                map_feature_point_to_image(points_a[index], features_a.feature_map_width, features_a.feature_map_height,
                                           warp.size(1), warp.size(0));
            const auto [x_b, y_b] =
                map_feature_point_to_image(points_b[index], features_b.feature_map_width, features_b.feature_map_height,
                                           warp.size(1), warp.size(0));
            ++metrics.dense_total;
            if (is_correct_pixel_match(warp, x_a, y_a, x_b, y_b, correct_threshold_pixels))
            {
                ++metrics.dense_correct;
            }
        }
    }

    return metrics;
}

WarpFeatureCoverageMetrics compute_warp_feature_coverage_metrics(const FeatureSet& features_a,
                                                                 const FeatureSet& features_b,
                                                                 const torch::Tensor& warp_a_to_b,
                                                                 double correct_threshold_pixels)
{
    if (!std::isfinite(correct_threshold_pixels) || correct_threshold_pixels < 0.0)
    {
        throw std::invalid_argument("correct_threshold_pixels must be non-negative and finite");
    }
    require_warp_shape(warp_a_to_b);
    if (!features_a.keypoints.defined() || !features_b.keypoints.defined() || !features_a.descriptors.defined() ||
        !features_b.descriptors.defined())
    {
        throw std::invalid_argument("feature keypoints and descriptors must be defined");
    }
    if (features_a.keypoints.dim() != 2 || features_b.keypoints.dim() != 2 || features_a.keypoints.size(1) != 2 ||
        features_b.keypoints.size(1) != 2 || features_a.descriptors.dim() != 2 || features_b.descriptors.dim() != 2 ||
        features_a.descriptors.size(0) != features_a.keypoints.size(0) ||
        features_b.descriptors.size(0) != features_b.keypoints.size(0) ||
        features_a.descriptors.size(1) != features_b.descriptors.size(1))
    {
        throw std::invalid_argument("feature tensors have incompatible shapes");
    }

    auto warp = warp_a_to_b.to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    auto descriptors_a = features_a.descriptors.to(torch::kCPU, torch::kFloat32).contiguous();
    auto descriptors_b = features_b.descriptors.to(torch::kCPU, torch::kFloat32).contiguous();

    std::vector<std::pair<double, double>> image_points_b;
    image_points_b.reserve(static_cast<std::size_t>(points_b.size(0)));
    for (int64_t index_b = 0; index_b < points_b.size(0); ++index_b)
    {
        image_points_b.push_back(map_feature_point_to_image_float(points_b[index_b], features_b.feature_map_width,
                                                                  features_b.feature_map_height, warp.size(1),
                                                                  warp.size(0)));
    }

    WarpFeatureCoverageMetrics metrics;
    metrics.source_total = points_a.size(0);
    double nearest_distance_sum = 0.0;
    for (int64_t index_a = 0; index_a < points_a.size(0); ++index_a)
    {
        const auto [x_a_float, y_a_float] = map_feature_point_to_image_float(
            points_a[index_a], features_a.feature_map_width, features_a.feature_map_height, warp.size(1), warp.size(0));
        const auto x_a = static_cast<int64_t>(std::llround(x_a_float));
        const auto y_a = static_cast<int64_t>(std::llround(y_a_float));
        if (y_a < 0 || y_a >= warp.size(0) || x_a < 0 || x_a >= warp.size(1))
        {
            continue;
        }
        const auto expected_x = warp.index({y_a, x_a, 0}).item<float>();
        const auto expected_y = warp.index({y_a, x_a, 1}).item<float>();
        if (!std::isfinite(expected_x) || !std::isfinite(expected_y) || expected_x < 0.0F ||
            expected_x > static_cast<float>(warp.size(1) - 1) || expected_y < 0.0F ||
            expected_y > static_cast<float>(warp.size(0) - 1))
        {
            continue;
        }
        ++metrics.valid_warp_total;

        std::vector<int64_t> positive_indices;
        double nearest_distance = std::numeric_limits<double>::infinity();
        for (int64_t index_b = 0; index_b < static_cast<int64_t>(image_points_b.size()); ++index_b)
        {
            const auto dx = image_points_b[static_cast<std::size_t>(index_b)].first - static_cast<double>(expected_x);
            const auto dy = image_points_b[static_cast<std::size_t>(index_b)].second - static_cast<double>(expected_y);
            const auto distance = std::sqrt(dx * dx + dy * dy);
            nearest_distance = std::min(nearest_distance, distance);
            if (distance <= correct_threshold_pixels)
            {
                positive_indices.push_back(index_b);
            }
        }
        if (std::isfinite(nearest_distance))
        {
            nearest_distance_sum += nearest_distance;
        }
        if (positive_indices.empty())
        {
            continue;
        }
        ++metrics.covered_by_target_keypoint;

        double best_positive_score = -std::numeric_limits<double>::infinity();
        for (const auto positive_index : positive_indices)
        {
            best_positive_score =
                std::max(best_positive_score, descriptor_cosine(descriptors_a[index_a], descriptors_b[positive_index]));
        }
        int64_t rank = 1;
        int64_t best_index = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int64_t index_b = 0; index_b < descriptors_b.size(0); ++index_b)
        {
            const auto score = descriptor_cosine(descriptors_a[index_a], descriptors_b[index_b]);
            if (score > best_score)
            {
                best_score = score;
                best_index = index_b;
            }
            if (score > best_positive_score)
            {
                ++rank;
            }
        }
        metrics.descriptor_rank_sum += rank;
        ++metrics.descriptor_rank_observed;
        if (std::find(positive_indices.begin(), positive_indices.end(), best_index) != positive_indices.end())
        {
            ++metrics.descriptor_top1_count;
        }
    }

    if (metrics.valid_warp_total > 0)
    {
        metrics.coverage_fraction =
            static_cast<double>(metrics.covered_by_target_keypoint) / static_cast<double>(metrics.valid_warp_total);
        metrics.mean_nearest_target_distance_pixels =
            nearest_distance_sum / static_cast<double>(metrics.valid_warp_total);
    }
    if (metrics.descriptor_rank_observed > 0)
    {
        metrics.mean_descriptor_positive_rank =
            static_cast<double>(metrics.descriptor_rank_sum) / static_cast<double>(metrics.descriptor_rank_observed);
        metrics.descriptor_top1_accuracy =
            static_cast<double>(metrics.descriptor_top1_count) / static_cast<double>(metrics.descriptor_rank_observed);
    }
    return metrics;
}

} // namespace pfm
