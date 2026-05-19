#include "infer/matching_pipeline.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <torch/torch.h>

namespace pfm {
namespace {

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
    return {output.matches.to(torch::kCPU, torch::kInt64).contiguous(), output.scores.to(torch::kCPU, torch::kFloat32).contiguous()};
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
