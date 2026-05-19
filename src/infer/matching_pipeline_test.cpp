#include <torch/torch.h>

#include "infer/feature_codec.h"
#include "infer/matching_pipeline.h"
#include "models/planetary_graph_matcher.h"
#include "tests/test_harness.h"

namespace {

pfm::FeatureSet makeFeatureSet(
    const torch::Tensor& keypoints,
    const torch::Tensor& descriptors,
    const torch::Tensor& dense_points,
    const torch::Tensor& dense_confidence
);

pfm::FeatureSet makeFeatureSet(
    const torch::Tensor& descriptors,
    const torch::Tensor& dense_points,
    const torch::Tensor& dense_confidence
) {
    const auto sparse_count = descriptors.size(0);
    return makeFeatureSet(
        torch::zeros({sparse_count, 2}, torch::kFloat32),
        descriptors,
        dense_points,
        dense_confidence);
}

pfm::FeatureSet makeFeatureSet(
    const torch::Tensor& keypoints,
    const torch::Tensor& descriptors,
    const torch::Tensor& dense_points,
    const torch::Tensor& dense_confidence
) {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const int64_t sparse_count = descriptors.size(0);
    return pfm::FeatureSet{
        keypoints.to(torch::kCPU, torch::kFloat32).contiguous(),
        torch::zeros({sparse_count}, float_options),
        descriptors.to(torch::kCPU, torch::kFloat32).contiguous(),
        torch::ones({sparse_count}, float_options),
        torch::zeros({sparse_count}, float_options),
        torch::eye(2, float_options).reshape({1, 2, 2}).repeat({sparse_count, 1, 1}),
        dense_points.to(torch::kCPU, torch::kFloat32).contiguous(),
        dense_confidence.to(torch::kCPU, torch::kFloat32).contiguous()};
}

static void matchingPipelineUsesPlanetaryGraphMatcherOutput() {
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
    pfm::PlanetaryGraphMatcher matcher(2, 8);

    const auto matches = pfm::matchFeatureSets(features_a, features_b, *matcher);

    PFM_REQUIRE(matches.sparse_matches.sizes() == torch::IntArrayRef({3, 2}));
    PFM_REQUIRE(matches.sparse_scores.sizes() == torch::IntArrayRef({3}));
    PFM_REQUIRE(matches.points_a.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(matches.points_b.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(matches.confidence.sizes() == torch::IntArrayRef({2}));
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

    PFM_REQUIRE(matches.sparse_matches.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(matches.sparse_matches.index({0, 0}).item<int64_t>() == 0);
    PFM_REQUIRE(matches.sparse_matches.index({1, 0}).item<int64_t>() == 1);
    PFM_REQUIRE(matches.sparse_scores.sizes() == torch::IntArrayRef({2}));
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

static void matchingPipelineKeepsOneWayGeometricallyConsistentSparseCandidates() {
    const auto features_a = makeFeatureSet(
        torch::tensor({{0.0F, 0.0F}, {10.0F, 0.0F}, {20.0F, 0.0F}}, torch::kFloat32),
        torch::tensor({{1.0F, 0.0F}, {0.985F, 0.174F}, {0.0F, 1.0F}}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    const auto features_b = makeFeatureSet(
        torch::tensor({{5.0F, 0.0F}, {15.0F, 0.0F}, {25.0F, 0.0F}}, torch::kFloat32),
        torch::tensor({{0.866F, 0.5F}, {0.996F, 0.087F}, {0.0F, 1.0F}}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );

    const auto matches = pfm::matchFeatureSets(features_a, features_b);

    PFM_REQUIRE(matches.sparse_matches.sizes() == torch::IntArrayRef({3, 2}));
}

static void matchingPipelineReturnsLearnedSparseCandidatesWithoutTranslationFilter() {
    const auto descriptors = torch::tensor(
        {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}},
        torch::kFloat32);
    const auto features_a = makeFeatureSet(
        torch::tensor({{10.0F, 10.0F}, {20.0F, 10.0F}, {30.0F, 10.0F}}, torch::kFloat32),
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    const auto features_b = makeFeatureSet(
        torch::tensor({{15.0F, 12.0F}, {25.0F, 12.0F}, {80.0F, 70.0F}}, torch::kFloat32),
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );

    const auto matches = pfm::matchFeatureSets(features_a, features_b);

    PFM_REQUIRE(matches.sparse_matches.sizes() == torch::IntArrayRef({3, 2}));
    PFM_REQUIRE(matches.sparse_scores.sizes() == torch::IntArrayRef({3}));
}

}  // namespace

void register_matching_pipeline_tests() {
    register_test("matching_pipeline_uses_planetary_graph_matcher_output",
                  matchingPipelineUsesPlanetaryGraphMatcherOutput);
    register_test(
        "matching_pipeline_handles_zero_sparse_descriptors_without_nan_scores",
        matching_pipeline_handles_zero_sparse_descriptors_without_nan_scores
    );
    register_test("matching_pipeline_handles_empty_sparse_descriptors", matching_pipeline_handles_empty_sparse_descriptors);
    register_test("matching_pipeline_handles_empty_semi_dense_outputs", matching_pipeline_handles_empty_semi_dense_outputs);
    register_test("matching_pipeline_keeps_one_way_geometrically_consistent_sparse_candidates",
                  matchingPipelineKeepsOneWayGeometricallyConsistentSparseCandidates);
    register_test("matching_pipeline_returns_learned_sparse_candidates_without_translation_filter",
                  matchingPipelineReturnsLearnedSparseCandidatesWithoutTranslationFilter);
}
