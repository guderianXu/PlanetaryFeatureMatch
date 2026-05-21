#include "infer/matching_pipeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace pfm {
namespace {

constexpr double PI = 3.14159265358979323846;
constexpr int64_t ROTATION_CONSISTENCY_MIN_MATCHES = 32;
constexpr int64_t ROTATION_CONSISTENCY_BINS = 72;
constexpr double ROTATION_CONSISTENCY_MAX_ANGLE_ERROR = PI / 36.0;
constexpr double ROTATION_CONSISTENCY_MAX_RADIUS_ERROR = 2.0;

torch::Tensor normalizeDescriptorRows(const torch::Tensor& descriptors) {
    return descriptors / descriptors.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
}

torch::Tensor descriptorSimilarityScores(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b) {
    const auto desc_a = normalizeDescriptorRows(descriptors_a.to(torch::kCPU, torch::kFloat32));
    const auto desc_b = normalizeDescriptorRows(descriptors_b.to(torch::kCPU, torch::kFloat32));
    return torch::matmul(desc_a, desc_b.transpose(0, 1));
}

double normalizeAngle(double angle) {
    while (angle <= -PI) {
        angle += 2.0 * PI;
    }
    while (angle > PI) {
        angle -= 2.0 * PI;
    }
    return angle;
}

double angleDistance(double lhs, double rhs) {
    return std::abs(normalizeAngle(lhs - rhs));
}

int64_t angleBin(double angle) {
    const auto normalized = normalizeAngle(angle) + PI;
    auto bin = static_cast<int64_t>(std::floor(normalized / (2.0 * PI) * ROTATION_CONSISTENCY_BINS));
    return std::min<int64_t>(ROTATION_CONSISTENCY_BINS - 1, std::max<int64_t>(0, bin));
}

std::pair<torch::Tensor, torch::Tensor> filterRotationConsistentMatches(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& matches,
    const torch::Tensor& scores
) {
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (!matches.defined() || matches.size(0) < ROTATION_CONSISTENCY_MIN_MATCHES) {
        return {matches, scores};
    }

    auto cpu_matches = matches.to(torch::kCPU, torch::kInt64).contiguous();
    auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const double center_ax = (static_cast<double>(features_a.feature_map_width) - 1.0) * 0.5;
    const double center_ay = (static_cast<double>(features_a.feature_map_height) - 1.0) * 0.5;
    const double center_bx = (static_cast<double>(features_b.feature_map_width) - 1.0) * 0.5;
    const double center_by = (static_cast<double>(features_b.feature_map_height) - 1.0) * 0.5;

    std::vector<int64_t> bins(static_cast<size_t>(ROTATION_CONSISTENCY_BINS), 0);
    std::vector<double> deltas(static_cast<size_t>(cpu_matches.size(0)), 0.0);
    std::vector<double> radius_errors(static_cast<size_t>(cpu_matches.size(0)), 0.0);
    for (int64_t index = 0; index < cpu_matches.size(0); ++index) {
        const auto ia = cpu_matches.index({index, 0}).item<int64_t>();
        const auto ib = cpu_matches.index({index, 1}).item<int64_t>();
        const double ax = points_a.index({ia, 0}).item<float>() - center_ax;
        const double ay = points_a.index({ia, 1}).item<float>() - center_ay;
        const double bx = points_b.index({ib, 0}).item<float>() - center_bx;
        const double by = points_b.index({ib, 1}).item<float>() - center_by;
        const auto delta = normalizeAngle(std::atan2(by, bx) - std::atan2(ay, ax));
        deltas[static_cast<size_t>(index)] = delta;
        radius_errors[static_cast<size_t>(index)] = std::abs(std::hypot(ax, ay) - std::hypot(bx, by));
        ++bins[static_cast<size_t>(angleBin(delta))];
    }

    const auto best_it = std::max_element(bins.begin(), bins.end());
    if (best_it == bins.end() || *best_it < ROTATION_CONSISTENCY_MIN_MATCHES / 4) {
        return {matches, scores};
    }
    const int64_t best_bin = static_cast<int64_t>(std::distance(bins.begin(), best_it));
    const double bin_center = (static_cast<double>(best_bin) + 0.5) / ROTATION_CONSISTENCY_BINS * 2.0 * PI - PI;
    const double dominant_angle = bin_center;

    std::vector<int64_t> keep_indices;
    keep_indices.reserve(static_cast<size_t>(*best_it));
    for (int64_t index = 0; index < cpu_matches.size(0); ++index) {
        if (angleDistance(deltas[static_cast<size_t>(index)], dominant_angle) <= ROTATION_CONSISTENCY_MAX_ANGLE_ERROR &&
            radius_errors[static_cast<size_t>(index)] <= ROTATION_CONSISTENCY_MAX_RADIUS_ERROR) {
            keep_indices.push_back(index);
        }
    }
    if (keep_indices.empty()) {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    auto keep = torch::from_blob(
                    keep_indices.data(),
                    {static_cast<int64_t>(keep_indices.size())},
                    torch::TensorOptions().dtype(torch::kInt64))
                    .clone();
    return {
        cpu_matches.index_select(0, keep).contiguous(),
        scores.to(torch::kCPU, torch::kFloat32).index_select(0, keep).contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> matchMutualDescriptorFeatures(
    const FeatureSet& features_a,
    const FeatureSet& features_b
) {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (features_a.descriptors.size(0) == 0 || features_b.descriptors.size(0) == 0) {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    const auto scores = descriptorSimilarityScores(features_a.descriptors, features_b.descriptors);
    const auto best_ab = scores.max(1);
    const auto target_indices = std::get<1>(best_ab);
    const auto best_ba = std::get<1>(scores.max(0));
    const auto source_indices = torch::arange(scores.size(0), long_options);
    const auto mutual_sources = best_ba.index_select(0, target_indices);
    const auto keep = mutual_sources.eq(source_indices);
    if (!keep.any().item<bool>()) {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    const auto kept_sources = source_indices.index({keep});
    const auto kept_targets = target_indices.index({keep});
    auto matches = torch::stack({kept_sources, kept_targets}, 1).to(torch::kCPU, torch::kInt64).contiguous();
    auto kept_scores = std::get<0>(best_ab).index({keep}).to(torch::kCPU, torch::kFloat32).contiguous();
    return filterRotationConsistentMatches(
        features_a,
        features_b,
        matches,
        kept_scores);
}

std::pair<torch::Tensor, torch::Tensor> matchSparseFeatures(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    PlanetaryGraphMatcherImpl& matcher
) {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (!features_a.descriptors.defined() || !features_b.descriptors.defined()) {
        throw std::invalid_argument("descriptors must be defined");
    }
    if (features_a.descriptors.dim() != 2 || features_b.descriptors.dim() != 2) {
        throw std::invalid_argument("descriptors must be 2D");
    }
    if (features_a.descriptors.size(1) != features_b.descriptors.size(1)) {
        throw std::invalid_argument("descriptor dimensions must match");
    }
    if (features_a.descriptors.size(0) == 0 || features_b.descriptors.size(0) == 0) {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    const auto descriptor_matches = matchMutualDescriptorFeatures(features_a, features_b);

    auto matcher_device = torch::Device(torch::kCPU);
    const auto parameters = matcher.parameters();
    if (!parameters.empty()) {
        matcher_device = parameters.front().device();
    }
    const auto output = matcher.forward(
        features_a.descriptors.to(matcher_device, torch::kFloat32),
        features_a.keypoints.to(matcher_device, torch::kFloat32),
        features_b.descriptors.to(matcher_device, torch::kFloat32),
        features_b.keypoints.to(matcher_device, torch::kFloat32));
    auto matches = output.matches.to(torch::kCPU, torch::kInt64).contiguous();
    auto scores = output.scores.to(torch::kCPU, torch::kFloat32).contiguous();
    if (features_a.descriptors.size(1) >= 4 && features_a.descriptors.size(1) % 4 == 0) {
        return descriptor_matches;
    }
    if (matches.size(0) == 0 || descriptor_matches.first.size(0) > matches.size(0)) {
        return descriptor_matches;
    }
    return {matches, scores};
}

}  // namespace

MatchSet matchFeatureSets(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    PlanetaryGraphMatcherImpl& matcher
) {
    if (!features_a.dense_points.defined() || !features_b.dense_points.defined() ||
        !features_a.dense_confidence.defined() || !features_b.dense_confidence.defined()) {
        throw std::invalid_argument("dense features must be defined");
    }
    const auto sparse = matchSparseFeatures(features_a, features_b, matcher);
    const int64_t dense_count = std::min(features_a.dense_points.size(0), features_b.dense_points.size(0));
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (dense_count == 0) {
        return MatchSet{
            sparse.first,
            sparse.second,
            torch::empty({0, 2}, float_options),
            torch::empty({0, 2}, float_options),
            torch::empty({0}, float_options)};
    }

    const auto confidence_a = features_a.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    const auto confidence_b = features_b.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    return MatchSet{
        sparse.first,
        sparse.second,
        features_a.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
        features_b.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
        torch::minimum(confidence_a, confidence_b).contiguous()};
}

MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b) {
    if (!features_a.descriptors.defined() || !features_b.descriptors.defined() ||
        features_a.descriptors.dim() != 2 || features_b.descriptors.dim() != 2) {
        throw std::invalid_argument("descriptors must be 2D");
    }
    if (!features_a.dense_points.defined() || !features_b.dense_points.defined() ||
        !features_a.dense_confidence.defined() || !features_b.dense_confidence.defined()) {
        throw std::invalid_argument("dense features must be defined");
    }
    const auto sparse = matchMutualDescriptorFeatures(features_a, features_b);
    const int64_t dense_count = std::min(features_a.dense_points.size(0), features_b.dense_points.size(0));
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (dense_count == 0) {
        return MatchSet{
            sparse.first,
            sparse.second,
            torch::empty({0, 2}, float_options),
            torch::empty({0, 2}, float_options),
            torch::empty({0}, float_options)};
    }

    const auto confidence_a = features_a.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    const auto confidence_b = features_b.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    return MatchSet{
        sparse.first,
        sparse.second,
        features_a.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
        features_b.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
        torch::minimum(confidence_a, confidence_b).contiguous()};
}

}  // namespace pfm
