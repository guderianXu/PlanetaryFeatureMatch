#include <torch/torch.h>

#include "infer/feature_codec.h"
#include "infer/matching_pipeline.h"
#include "tests/test_harness.h"

namespace {

pfm::FeatureSet makeFeatureSet(
    const torch::Tensor& descriptors,
    const torch::Tensor& dense_points,
    const torch::Tensor& dense_confidence
) {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const int64_t sparse_count = descriptors.size(0);
    return pfm::FeatureSet{
        torch::zeros({sparse_count, 2}, float_options),
        torch::zeros({sparse_count}, float_options),
        descriptors.to(torch::kCPU, torch::kFloat32).contiguous(),
        torch::ones({sparse_count}, float_options),
        torch::zeros({sparse_count}, float_options),
        torch::eye(2, float_options).reshape({1, 2, 2}).repeat({sparse_count, 1, 1}),
        dense_points.to(torch::kCPU, torch::kFloat32).contiguous(),
        dense_confidence.to(torch::kCPU, torch::kFloat32).contiguous()};
}

static void matching_pipeline_returns_expected_mutual_nn_pairs_and_scores() {
    const auto features_a = makeFeatureSet(
        torch::tensor({{2.0F, 0.0F}, {0.0F, 3.0F}, {1.0F, 1.0F}}, torch::kFloat32),
        torch::tensor({{0.0F, 0.0F}, {1.0F, 1.0F}}, torch::kFloat32),
        torch::tensor({0.9F, 0.2F}, torch::kFloat32)
    );
    const auto features_b = makeFeatureSet(
        torch::tensor({{0.0F, 4.0F}, {5.0F, 0.0F}, {-1.0F, 0.0F}}, torch::kFloat32),
        torch::tensor({{2.0F, 2.0F}, {3.0F, 3.0F}, {4.0F, 4.0F}}, torch::kFloat32),
        torch::tensor({0.7F, 0.8F, 0.1F}, torch::kFloat32)
    );

    const auto matches = pfm::matchFeatureSets(features_a, features_b);

    PFM_REQUIRE(matches.sparse_matches.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(matches.sparse_matches.index({0, 0}).item<int64_t>() == 0);
    PFM_REQUIRE(matches.sparse_matches.index({0, 1}).item<int64_t>() == 1);
    PFM_REQUIRE(matches.sparse_matches.index({1, 0}).item<int64_t>() == 1);
    PFM_REQUIRE(matches.sparse_matches.index({1, 1}).item<int64_t>() == 0);
    PFM_REQUIRE_CLOSE(matches.sparse_scores.index({0}).item<float>(), 1.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(matches.sparse_scores.index({1}).item<float>(), 1.0F, 1.0e-5F);
    PFM_REQUIRE(matches.points_a.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(matches.points_b.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE_CLOSE(matches.confidence.index({0}).item<float>(), 0.7F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(matches.confidence.index({1}).item<float>(), 0.2F, 1.0e-5F);
}

static void matching_pipeline_handles_zero_sparse_descriptors_without_nan_scores() {
    const auto features_a = makeFeatureSet(
        torch::zeros({2, 3}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    const auto features_b = makeFeatureSet(
        torch::tensor({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );

    const auto matches = pfm::matchFeatureSets(features_a, features_b);

    PFM_REQUIRE(matches.sparse_matches.sizes() == torch::IntArrayRef({1, 2}));
    PFM_REQUIRE(matches.sparse_matches.index({0, 0}).item<int64_t>() == 0);
    PFM_REQUIRE(matches.sparse_matches.index({0, 1}).item<int64_t>() == 0);
    PFM_REQUIRE_CLOSE(matches.sparse_scores.index({0}).item<float>(), 0.0F, 1.0e-6F);
}

static void matching_pipeline_handles_empty_sparse_descriptors() {
    const auto features_a = makeFeatureSet(
        torch::empty({0, 3}, torch::kFloat32),
        torch::tensor({{0.0F, 0.0F}}, torch::kFloat32),
        torch::tensor({0.4F}, torch::kFloat32)
    );
    const auto features_b = makeFeatureSet(
        torch::empty({2, 3}, torch::kFloat32),
        torch::tensor({{1.0F, 1.0F}}, torch::kFloat32),
        torch::tensor({0.5F}, torch::kFloat32)
    );

    const auto matches = pfm::matchFeatureSets(features_a, features_b);

    PFM_REQUIRE(matches.sparse_matches.sizes() == torch::IntArrayRef({0, 2}));
    PFM_REQUIRE(matches.sparse_matches.scalar_type() == torch::kInt64);
    PFM_REQUIRE(matches.sparse_scores.sizes() == torch::IntArrayRef({0}));
    PFM_REQUIRE(matches.points_a.sizes() == torch::IntArrayRef({1, 2}));
    PFM_REQUIRE(matches.confidence.sizes() == torch::IntArrayRef({1}));
}

static void matching_pipeline_handles_empty_semi_dense_outputs() {
    const auto features_a = makeFeatureSet(
        torch::tensor({{1.0F, 0.0F}}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    const auto features_b = makeFeatureSet(
        torch::tensor({{1.0F, 0.0F}}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );

    const auto matches = pfm::matchFeatureSets(features_a, features_b);

    PFM_REQUIRE(matches.points_a.sizes() == torch::IntArrayRef({0, 2}));
    PFM_REQUIRE(matches.points_b.sizes() == torch::IntArrayRef({0, 2}));
    PFM_REQUIRE(matches.confidence.sizes() == torch::IntArrayRef({0}));
    PFM_REQUIRE(matches.points_a.scalar_type() == torch::kFloat32);
}

}  // namespace

void register_matching_pipeline_tests() {
    register_test(
        "matching_pipeline_returns_expected_mutual_nn_pairs_and_scores",
        matching_pipeline_returns_expected_mutual_nn_pairs_and_scores
    );
    register_test(
        "matching_pipeline_handles_zero_sparse_descriptors_without_nan_scores",
        matching_pipeline_handles_zero_sparse_descriptors_without_nan_scores
    );
    register_test("matching_pipeline_handles_empty_sparse_descriptors", matching_pipeline_handles_empty_sparse_descriptors);
    register_test("matching_pipeline_handles_empty_semi_dense_outputs", matching_pipeline_handles_empty_semi_dense_outputs);
}
