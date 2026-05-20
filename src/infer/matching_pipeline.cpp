#include "infer/matching_pipeline.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace pfm {
namespace {

torch::Tensor normalizeDescriptorRows(const torch::Tensor& descriptors) {
    return descriptors / descriptors.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
}

torch::Tensor descriptorSimilarityScores(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b) {
    const auto desc_a = normalizeDescriptorRows(descriptors_a.to(torch::kCPU, torch::kFloat32));
    const auto desc_b = normalizeDescriptorRows(descriptors_b.to(torch::kCPU, torch::kFloat32));
    if (desc_a.size(1) < 4 || desc_a.size(1) % 4 != 0) {
        return torch::matmul(desc_a, desc_b.transpose(0, 1));
    }

    constexpr int64_t group_count = 4;
    const auto group_dim = desc_a.size(1) / group_count;
    auto grouped_a = desc_a.reshape({desc_a.size(0), group_count, group_dim});
    auto grouped_b = desc_b.reshape({desc_b.size(0), group_count, group_dim});
    std::vector<torch::Tensor> shifted_scores;
    shifted_scores.reserve(group_count);
    for (int64_t shift = 0; shift < group_count; ++shift) {
        auto shifted_b = torch::roll(grouped_b, {shift}, {1});
        shifted_scores.push_back((grouped_a.unsqueeze(1) * shifted_b.unsqueeze(0)).sum({2, 3}));
    }
    return std::get<0>(torch::stack(shifted_scores, -1).max(-1));
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
    return {
        torch::stack({kept_sources, kept_targets}, 1).to(torch::kCPU, torch::kInt64).contiguous(),
        std::get<0>(best_ab).index({keep}).to(torch::kCPU, torch::kFloat32).contiguous()};
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
    if (matches.size(0) == 0) {
        return matchMutualDescriptorFeatures(features_a, features_b);
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
    if (!features_a.descriptors.defined() || features_a.descriptors.dim() != 2) {
        throw std::invalid_argument("descriptors must be 2D");
    }
    PlanetaryGraphMatcher matcher(features_a.descriptors.size(1), std::max<int64_t>(32, features_a.descriptors.size(1)));
    return matchFeatureSets(features_a, features_b, *matcher);
}

}  // namespace pfm
