#include "infer/matching_pipeline.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace pfm {
namespace {

std::pair<torch::Tensor, torch::Tensor> matchSparseFeatures(
    const FeatureSet& features_a,
    const FeatureSet& features_b
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

    const auto descriptors_a = torch::nn::functional::normalize(
        features_a.descriptors.to(torch::kCPU, torch::kFloat32),
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12)
    );
    const auto descriptors_b = torch::nn::functional::normalize(
        features_b.descriptors.to(torch::kCPU, torch::kFloat32),
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12)
    );
    const auto similarity = torch::matmul(descriptors_a, descriptors_b.transpose(0, 1));
    const auto best_b = std::get<1>(torch::max(similarity, 1)).to(torch::kCPU, torch::kInt64).contiguous();
    const auto best_score = std::get<0>(torch::max(similarity, 1)).to(torch::kCPU, torch::kFloat32).contiguous();
    const auto best_a = std::get<1>(torch::max(similarity, 0)).to(torch::kCPU, torch::kInt64).contiguous();

    std::vector<int64_t> match_indices;
    std::vector<float> match_scores;
    auto best_b_accessor = best_b.accessor<int64_t, 1>();
    auto best_a_accessor = best_a.accessor<int64_t, 1>();
    auto score_accessor = best_score.accessor<float, 1>();
    for (int64_t index_a = 0; index_a < best_b.size(0); ++index_a) {
        const int64_t index_b = best_b_accessor[index_a];
        if (best_a_accessor[index_b] == index_a) {
            match_indices.push_back(index_a);
            match_indices.push_back(index_b);
            match_scores.push_back(score_accessor[index_a]);
        }
    }

    const int64_t match_count = static_cast<int64_t>(match_scores.size());
    if (match_count == 0) {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    return {
        torch::from_blob(match_indices.data(), {match_count, 2}, long_options).clone().contiguous(),
        torch::from_blob(match_scores.data(), {match_count}, float_options).clone().contiguous()};
}

}  // namespace

MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b) {
    if (!features_a.dense_points.defined() || !features_b.dense_points.defined() ||
        !features_a.dense_confidence.defined() || !features_b.dense_confidence.defined()) {
        throw std::invalid_argument("dense features must be defined");
    }
    const auto sparse = matchSparseFeatures(features_a, features_b);
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
