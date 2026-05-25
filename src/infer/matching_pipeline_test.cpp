#include <array>
#include <cstdlib>

#include <torch/torch.h>

#include "infer/feature_codec.h"
#include "infer/matching_pipeline.h"
#include "models/planetary_graph_matcher.h"
#include "tests/test_harness.h"

namespace pfm::testing {
int64_t geometric_consistency_max_output_matches_for_test();
std::vector<int64_t> geometric_consistency_prefix_sizes_for_test(int64_t candidate_count);
int64_t descriptor_topk_candidates_per_source_for_test();
int64_t descriptor_topk_candidates_per_source_for_test_env();
bool descriptor_topk_projective_before_rotation_for_test();
bool descriptor_reciprocal_topk_fallback_for_test();
bool sparse_geometry_filter_rotation_only_for_test();
bool should_return_rotation_only_matches_for_test(int64_t rotation_matches);
std::pair<torch::Tensor, torch::Tensor> merge_sparse_match_candidates_for_test(
    const torch::Tensor& primary_matches,
    const torch::Tensor& primary_scores,
    const torch::Tensor& fallback_matches,
    const torch::Tensor& fallback_scores);
double geometric_candidate_quality_for_test(double score_mean, int64_t inlier_count);
double geometric_candidate_quality_for_test(
    double score_mean,
    int64_t inlier_count,
    double source_spread,
    double target_spread);
bool should_use_graph_matcher_for_sparse_count_for_test(int64_t keypoint_count_a, int64_t keypoint_count_b);
bool should_use_wide_topk_fallback_for_test(int64_t base_matches, int64_t wide_matches, double wide_mean_score);
bool should_use_projective_topk_rescue_for_test(int64_t base_matches, int64_t projective_matches);
bool should_prefer_mutual_descriptor_geometry_for_test(int64_t mutual_matches, int64_t topk_matches);
bool should_use_conservative_topk_fallback_for_test(int64_t base_matches, int64_t conservative_matches);
std::pair<torch::Tensor, torch::Tensor> trim_low_confidence_topk_tail_for_test(
    const torch::Tensor& matches,
    const torch::Tensor& scores);
std::pair<torch::Tensor, torch::Tensor> descriptor_reciprocal_topk_matches_for_test(
    const pfm::FeatureSet& features_a,
    const pfm::FeatureSet& features_b,
    int64_t candidates_per_source);
torch::Device descriptor_similarity_compute_device_for_test(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Device& compute_device);
std::pair<torch::Tensor, torch::Tensor> affine_residual_cleanup_matches_for_test(
    const pfm::FeatureSet& features_a,
    const pfm::FeatureSet& features_b,
    const torch::Tensor& matches,
    const torch::Tensor& scores);
std::pair<torch::Tensor, torch::Tensor> projective_consistent_matches_for_test(
    const pfm::FeatureSet& features_a,
    const pfm::FeatureSet& features_b,
    const torch::Tensor& matches,
    const torch::Tensor& scores);
std::pair<torch::Tensor, torch::Tensor> rotation_consistent_matches_for_test(
    const pfm::FeatureSet& features_a,
    const pfm::FeatureSet& features_b,
    const torch::Tensor& matches,
    const torch::Tensor& scores);
std::pair<torch::Tensor, torch::Tensor> relaxed_graph_logit_matches_for_test(
    const torch::Tensor& logits,
    int64_t keypoint_count_a,
    int64_t keypoint_count_b);
}

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

    PFM_REQUIRE(matches.sparse_matches.dim() == 2);
    PFM_REQUIRE(matches.sparse_matches.size(1) == 2);
    PFM_REQUIRE(matches.sparse_scores.size(0) == matches.sparse_matches.size(0));
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

    PFM_REQUIRE(matches.sparse_matches.dim() == 2);
    PFM_REQUIRE(matches.sparse_matches.size(1) == 2);
    PFM_REQUIRE(matches.sparse_scores.size(0) == matches.sparse_matches.size(0));
    PFM_REQUIRE(!matches.sparse_scores.isnan().any().item<bool>());
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

    PFM_REQUIRE(matches.sparse_matches.dim() == 2);
    PFM_REQUIRE(matches.sparse_matches.size(1) == 2);
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

    PFM_REQUIRE(matches.sparse_matches.dim() == 2);
    PFM_REQUIRE(matches.sparse_matches.size(1) == 2);
    PFM_REQUIRE(matches.sparse_scores.size(0) == matches.sparse_matches.size(0));
}

static torch::Tensor makeC4DescriptorRows(const std::vector<int64_t>& slot_indices) {
    auto descriptors = torch::zeros({static_cast<int64_t>(slot_indices.size()), 16}, torch::kFloat32);
    for (int64_t row = 0; row < static_cast<int64_t>(slot_indices.size()); ++row) {
        descriptors.index_put_({row, row + slot_indices[static_cast<size_t>(row)] * 4}, 1.0F);
    }
    return descriptors;
}

static void matchingPipelineUsesOneGlobalCyclicDescriptorShift() {
    const auto keypoints_a = torch::tensor(
        {{0.0F, 0.0F}, {10.0F, 0.0F}, {20.0F, 0.0F}, {30.0F, 0.0F}},
        torch::kFloat32);
    const auto keypoints_b = torch::tensor(
        {{30.0F, 0.0F}, {20.0F, 0.0F}, {10.0F, 0.0F}, {0.0F, 0.0F}, {100.0F, 100.0F}},
        torch::kFloat32);
    const auto features_a = makeFeatureSet(
        keypoints_a,
        makeC4DescriptorRows({0, 0, 0, 0}),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    auto descriptors_b = torch::zeros({5, 16}, torch::kFloat32);
    descriptors_b.index_put_({0, 2 * 4 + 3}, 1.0F);
    descriptors_b.index_put_({1, 2 * 4 + 2}, 1.0F);
    descriptors_b.index_put_({2, 2 * 4 + 1}, 1.0F);
    descriptors_b.index_put_({3, 2 * 4 + 0}, 1.0F);
    descriptors_b.index_put_({4, 0}, 1.0F);
    const auto features_b = makeFeatureSet(
        keypoints_b,
        descriptors_b,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );

    const auto matches = pfm::matchFeatureSets(features_a, features_b);
    const auto expected = torch::tensor({{0, 3}, {1, 2}, {2, 1}, {3, 0}}, torch::kInt64);

    PFM_REQUIRE(torch::equal(matches.sparse_matches, expected));
}

static void matchingPipelineRecoversHalfTurnWithCyclicDescriptors() {
    const auto features_a = makeFeatureSet(
        torch::tensor({{2.0F, 2.0F}, {8.0F, 2.0F}, {2.0F, 8.0F}, {8.0F, 8.0F}}, torch::kFloat32),
        makeC4DescriptorRows({0, 0, 0, 0}),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    auto features_b = makeFeatureSet(
        torch::tensor({{7.0F, 7.0F}, {1.0F, 7.0F}, {7.0F, 1.0F}, {1.0F, 1.0F}, {2.0F, 2.0F}}, torch::kFloat32),
        torch::cat({makeC4DescriptorRows({2, 2, 2, 2}), torch::nn::functional::one_hot(torch::tensor({0}), 16).to(torch::kFloat32)}, 0),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    const auto matches = pfm::matchFeatureSets(features_a, features_b);
    const auto expected = torch::tensor({{0, 0}, {1, 1}, {2, 2}, {3, 3}}, torch::kInt64);

    PFM_REQUIRE(torch::equal(matches.sparse_matches, expected));
}

static void matchingPipelineAcceptsCudaGraphMatcherWithCpuFeatures() {
    if (!torch::cuda::is_available()) {
        return;
    }
    const auto features_a = makeFeatureSet(
        torch::tensor({{1.0F, 1.0F}, {2.0F, 2.0F}}, torch::kFloat32),
        torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    const auto features_b = makeFeatureSet(
        torch::tensor({{1.0F, 1.0F}, {2.0F, 2.0F}}, torch::kFloat32),
        torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32),
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32)
    );
    pfm::PlanetaryGraphMatcher matcher(2, 8);
    matcher->to(torch::kCUDA);

    const auto matches = pfm::matchFeatureSets(features_a, features_b, *matcher);

    PFM_REQUIRE(matches.sparse_matches.device().is_cpu());
    PFM_REQUIRE(matches.sparse_scores.device().is_cpu());
    PFM_REQUIRE(matches.sparse_matches.dim() == 2);
    PFM_REQUIRE(matches.sparse_matches.size(1) == 2);
}

static void matching_pipeline_keeps_more_than_legacy_256_geometric_matches() {
    PFM_REQUIRE(pfm::testing::geometric_consistency_max_output_matches_for_test() >= 512);
}

static void matching_pipeline_bounds_geometric_ransac_prefix_count_for_large_topk() {
    const auto prefixes = pfm::testing::geometric_consistency_prefix_sizes_for_test(65536);
    PFM_REQUIRE(!prefixes.empty());
    PFM_REQUIRE(prefixes.front() == 8);
    PFM_REQUIRE(prefixes.back() == 65536);
    PFM_REQUIRE(prefixes.size() <= 32);
    for (std::size_t index = 1; index < prefixes.size(); ++index) {
        PFM_REQUIRE(prefixes[index] > prefixes[index - 1]);
    }
}

static void matching_pipeline_geometric_quality_prefers_broader_inlier_support() {
    const auto small_high_score = pfm::testing::geometric_candidate_quality_for_test(0.99, 4);
    const auto broader_medium_score = pfm::testing::geometric_candidate_quality_for_test(0.65, 9);

    PFM_REQUIRE(broader_medium_score > small_high_score);
}

static void matching_pipeline_geometric_quality_penalizes_local_projective_clusters() {
    const auto local_large_cluster = pfm::testing::geometric_candidate_quality_for_test(0.99, 24, 0.03, 0.04);
    const auto spread_smaller_cluster = pfm::testing::geometric_candidate_quality_for_test(0.86, 12, 0.42, 0.38);

    PFM_REQUIRE(spread_smaller_cluster > local_large_cluster);
}

static void matching_pipeline_projective_candidate_selection_can_use_spatial_spread_weight() {
    std::vector<float> a_xy;
    std::vector<float> b_xy;
    std::vector<int64_t> match_indices;
    std::vector<float> scores;
    a_xy.reserve(36 * 2);
    b_xy.reserve(36 * 2);
    match_indices.reserve(36 * 2);
    scores.reserve(36);

    const std::array<std::pair<float, float>, 12> broad_points{{
        {0.0F, 0.0F}, {64.0F, 0.0F}, {128.0F, 0.0F}, {192.0F, 0.0F},
        {0.0F, 96.0F}, {64.0F, 96.0F}, {128.0F, 96.0F}, {192.0F, 96.0F},
        {0.0F, 192.0F}, {64.0F, 192.0F}, {128.0F, 192.0F}, {192.0F, 192.0F},
    }};

    auto append_match = [&](float ax, float ay, float bx, float by, float score) {
        const auto index = static_cast<int64_t>(scores.size());
        a_xy.push_back(ax);
        a_xy.push_back(ay);
        b_xy.push_back(bx);
        b_xy.push_back(by);
        match_indices.push_back(index);
        match_indices.push_back(index);
        scores.push_back(score);
    };

    for (const auto& point : broad_points) {
        append_match(point.first, point.second, point.first + 5.0F, point.second + 7.0F, 0.99F);
    }
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 6; ++col) {
            const auto ax = 80.0F + static_cast<float>(col) * 1.2F;
            const auto ay = 80.0F + static_cast<float>(row) * 1.2F;
            append_match(ax, ay, ax + 40.0F, ay - 25.0F, 0.95F);
        }
    }

    const auto keypoints_a = torch::from_blob(a_xy.data(), {36, 2}, torch::kFloat32).clone();
    const auto keypoints_b = torch::from_blob(b_xy.data(), {36, 2}, torch::kFloat32).clone();
    const auto descriptors = torch::zeros({36, 4}, torch::kFloat32);
    const auto confidence = torch::ones({36}, torch::kFloat32);
    const auto features_a = makeFeatureSet(keypoints_a, descriptors, keypoints_a, confidence);
    const auto features_b = makeFeatureSet(keypoints_b, descriptors, keypoints_b, confidence);
    const auto matches = torch::from_blob(match_indices.data(), {36, 2}, torch::kInt64).clone();
    const auto score_tensor = torch::from_blob(scores.data(), {36}, torch::kFloat32).clone();

    setenv("PFM_GEOMETRIC_SPREAD_QUALITY_WEIGHT", "40", 1);
    const auto filtered = pfm::testing::projective_consistent_matches_for_test(
        features_a,
        features_b,
        matches,
        score_tensor);
    unsetenv("PFM_GEOMETRIC_SPREAD_QUALITY_WEIGHT");

    PFM_REQUIRE(filtered.first.size(0) == 12);
}

static void matching_pipeline_uses_requested_device_for_descriptor_similarity_when_available() {
    if (!torch::cuda::is_available()) {
        return;
    }
    const auto descriptors_a = torch::randn({8, 16}, torch::kFloat32);
    const auto descriptors_b = torch::randn({9, 16}, torch::kFloat32);

    const auto device = pfm::testing::descriptor_similarity_compute_device_for_test(
        descriptors_a,
        descriptors_b,
        torch::Device(torch::kCUDA));

    PFM_REQUIRE(device.is_cuda());
}

static void matching_pipeline_uses_conservative_descriptor_topk_by_default() {
    PFM_REQUIRE(pfm::testing::descriptor_topk_candidates_per_source_for_test() == 4);
}

static void matching_pipeline_allows_descriptor_topk_override_for_hard_viewpoint_pairs() {
    setenv("PFM_DESCRIPTOR_TOPK_CANDIDATES", "64", 1);
    PFM_REQUIRE(pfm::testing::descriptor_topk_candidates_per_source_for_test_env() == 64);
    unsetenv("PFM_DESCRIPTOR_TOPK_CANDIDATES");
    PFM_REQUIRE(pfm::testing::descriptor_topk_candidates_per_source_for_test_env() ==
                pfm::testing::descriptor_topk_candidates_per_source_for_test());
}

static void matching_pipeline_wide_topk_fallback_requires_low_count_gain() {
    PFM_REQUIRE(pfm::testing::should_use_wide_topk_fallback_for_test(4, 6, 0.99));
    PFM_REQUIRE(pfm::testing::should_use_wide_topk_fallback_for_test(6, 8, 0.99));
    PFM_REQUIRE(!pfm::testing::should_use_wide_topk_fallback_for_test(8, 12, 0.99));
    PFM_REQUIRE(!pfm::testing::should_use_wide_topk_fallback_for_test(5, 6, 0.99));
    PFM_REQUIRE(!pfm::testing::should_use_wide_topk_fallback_for_test(5, 24, 0.99));
    PFM_REQUIRE(!pfm::testing::should_use_wide_topk_fallback_for_test(5, 7, 0.97));
    PFM_REQUIRE(!pfm::testing::should_use_wide_topk_fallback_for_test(5, 5, 0.995));
}

static void matching_pipeline_tries_projective_topk_before_rotation_filter_for_viewpoint() {
    PFM_REQUIRE(!pfm::testing::descriptor_topk_projective_before_rotation_for_test());
    setenv("PFM_DESCRIPTOR_TOPK_PROJECTIVE_BEFORE_ROTATION", "1", 1);
    PFM_REQUIRE(pfm::testing::descriptor_topk_projective_before_rotation_for_test());
    unsetenv("PFM_DESCRIPTOR_TOPK_PROJECTIVE_BEFORE_ROTATION");
    PFM_REQUIRE(!pfm::testing::descriptor_topk_projective_before_rotation_for_test());
}

static void matching_pipeline_projective_before_rotation_uses_unique_topk_pairs() {
    auto primary = torch::tensor({{0, 1}, {0, 2}, {2, 1}, {3, 4}}, torch::kInt64);
    auto scores = torch::tensor({0.99F, 0.98F, 0.97F, 0.96F}, torch::kFloat32);

    const auto merged = pfm::testing::merge_sparse_match_candidates_for_test(
        primary,
        scores,
        torch::empty({0, 2}, torch::kInt64),
        torch::empty({0}, torch::kFloat32));

    PFM_REQUIRE(merged.first.size(0) == 2);
    PFM_REQUIRE(torch::equal(merged.first, torch::tensor({{0, 1}, {3, 4}}, torch::kInt64)));
}

static void matching_pipeline_projective_rescue_requires_large_safe_gain() {
    PFM_REQUIRE(pfm::testing::should_use_projective_topk_rescue_for_test(44, 124));
    PFM_REQUIRE(!pfm::testing::should_use_projective_topk_rescue_for_test(220, 332));
    PFM_REQUIRE(!pfm::testing::should_use_projective_topk_rescue_for_test(44, 88));
    PFM_REQUIRE(!pfm::testing::should_use_projective_topk_rescue_for_test(44, 99));
}

static void matching_pipeline_conservative_topk_fallback_targets_low_count_results() {
    PFM_REQUIRE(pfm::testing::should_use_conservative_topk_fallback_for_test(16, 8));
    PFM_REQUIRE(pfm::testing::should_use_conservative_topk_fallback_for_test(64, 32));
    PFM_REQUIRE(!pfm::testing::should_use_conservative_topk_fallback_for_test(65, 64));
    PFM_REQUIRE(!pfm::testing::should_use_conservative_topk_fallback_for_test(16, 7));
    PFM_REQUIRE(!pfm::testing::should_use_conservative_topk_fallback_for_test(32, 15));
}

static void matching_pipeline_prefers_mutual_geometry_when_topk_cluster_expands_too_much() {
    PFM_REQUIRE(pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(39, 108));
    PFM_REQUIRE(pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(69, 108));
    PFM_REQUIRE(pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(39, 41));
    PFM_REQUIRE(!pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(44, 8));
    PFM_REQUIRE(!pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(35, 60));
    PFM_REQUIRE(!pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(35, 50));
    PFM_REQUIRE(!pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(18, 60));
    PFM_REQUIRE(!pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(7, 25));
    PFM_REQUIRE(!pfm::testing::should_prefer_mutual_descriptor_geometry_for_test(34, 220));
}

static void matching_pipeline_trims_low_confidence_tail_for_medium_topk_geometry() {
    auto matches = torch::arange(120, torch::kInt64).reshape({60, 2});
    auto scores = torch::cat({
        torch::full({54}, 0.981F, torch::kFloat32),
        torch::full({6}, 0.970F, torch::kFloat32)});

    const auto trimmed = pfm::testing::trim_low_confidence_topk_tail_for_test(matches, scores);

    PFM_REQUIRE(trimmed.first.size(0) == 54);
    PFM_REQUIRE(trimmed.second.min().item<float>() >= 0.98F);
}

static void matching_pipeline_keeps_large_rotation_topk_geometry_tail() {
    auto matches = torch::arange(440, torch::kInt64).reshape({220, 2});
    auto scores = torch::cat({
        torch::full({200}, 0.981F, torch::kFloat32),
        torch::full({20}, 0.970F, torch::kFloat32)});

    const auto trimmed = pfm::testing::trim_low_confidence_topk_tail_for_test(matches, scores);

    PFM_REQUIRE(trimmed.first.size(0) == 220);
}

static void matching_pipeline_reciprocal_topk_keeps_only_bidirectional_candidates() {
    const auto descriptors_a = torch::tensor(
        {{1.0F, 0.0F}, {0.98F, 0.2F}, {0.0F, 1.0F}},
        torch::kFloat32);
    const auto descriptors_b = torch::tensor(
        {{1.0F, 0.0F}, {0.0F, 1.0F}},
        torch::kFloat32);
    const auto features_a = makeFeatureSet(
        torch::zeros({3, 2}, torch::kFloat32),
        descriptors_a,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto features_b = makeFeatureSet(
        torch::zeros({2, 2}, torch::kFloat32),
        descriptors_b,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));

    const auto matches = pfm::testing::descriptor_reciprocal_topk_matches_for_test(features_a, features_b, 1);

    PFM_REQUIRE(torch::equal(matches.first, torch::tensor({{0, 0}, {2, 1}}, torch::kInt64)));
}

static void matching_pipeline_allows_reciprocal_topk_fallback_env() {
    PFM_REQUIRE(!pfm::testing::descriptor_reciprocal_topk_fallback_for_test());
    setenv("PFM_DESCRIPTOR_RECIPROCAL_TOPK_FALLBACK", "1", 1);
    PFM_REQUIRE(pfm::testing::descriptor_reciprocal_topk_fallback_for_test());
    unsetenv("PFM_DESCRIPTOR_RECIPROCAL_TOPK_FALLBACK");
    PFM_REQUIRE(!pfm::testing::descriptor_reciprocal_topk_fallback_for_test());
}

static void matching_pipeline_allows_rotation_only_geometry_filter_for_extreme_viewpoint() {
    unsetenv("PFM_SPARSE_GEOMETRY_FILTER");
    PFM_REQUIRE(!pfm::testing::sparse_geometry_filter_rotation_only_for_test());
    PFM_REQUIRE(!pfm::testing::should_return_rotation_only_matches_for_test(129));

    setenv("PFM_SPARSE_GEOMETRY_FILTER", "rotation-only", 1);
    PFM_REQUIRE(pfm::testing::sparse_geometry_filter_rotation_only_for_test());
    PFM_REQUIRE(!pfm::testing::should_return_rotation_only_matches_for_test(31));
    PFM_REQUIRE(pfm::testing::should_return_rotation_only_matches_for_test(32));

    setenv("PFM_SPARSE_GEOMETRY_FILTER", "projective", 1);
    PFM_REQUIRE(!pfm::testing::sparse_geometry_filter_rotation_only_for_test());
    PFM_REQUIRE(!pfm::testing::should_return_rotation_only_matches_for_test(129));
    unsetenv("PFM_SPARSE_GEOMETRY_FILTER");
}

static void matching_pipeline_rotation_filter_rejects_near_angle_position_outliers() {
    std::vector<float> keypoint_values_a;
    std::vector<float> keypoint_values_b;
    std::vector<int64_t> match_values;
    std::vector<float> score_values;
    constexpr int64_t good_count = 32;
    constexpr int64_t outlier_count = 8;
    keypoint_values_a.reserve((good_count + outlier_count) * 2);
    keypoint_values_b.reserve((good_count + outlier_count) * 2);
    match_values.reserve((good_count + outlier_count) * 2);
    score_values.reserve(good_count + outlier_count);
    const float center = 50.0F;
    const float radius = 24.0F;
    const float right_angle = 1.57079632679F;
    const float near_outlier_angle = right_angle + 0.075F;
    for (int64_t index = 0; index < good_count + outlier_count; ++index) {
        const float theta = static_cast<float>(index) * 0.16F;
        const float source_x = center + std::cos(theta) * radius;
        const float source_y = center + std::sin(theta) * radius;
        const float delta = index < good_count ? right_angle : near_outlier_angle;
        const float target_x = center + std::cos(theta + delta) * radius;
        const float target_y = center + std::sin(theta + delta) * radius;
        keypoint_values_a.push_back(source_x);
        keypoint_values_a.push_back(source_y);
        keypoint_values_b.push_back(target_x);
        keypoint_values_b.push_back(target_y);
        match_values.push_back(index);
        match_values.push_back(index);
        score_values.push_back(1.0F - static_cast<float>(index) * 0.001F);
    }
    const auto keypoints_a = torch::from_blob(
                                 keypoint_values_a.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto keypoints_b = torch::from_blob(
                                 keypoint_values_b.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto descriptors = torch::eye(good_count + outlier_count, torch::kFloat32);
    auto features_a = makeFeatureSet(
        keypoints_a,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    auto features_b = makeFeatureSet(
        keypoints_b,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    features_a.feature_map_width = 101;
    features_a.feature_map_height = 101;
    features_b.feature_map_width = 101;
    features_b.feature_map_height = 101;
    const auto matches = torch::from_blob(
                             match_values.data(),
                             {good_count + outlier_count, 2},
                             torch::kInt64)
                             .clone();
    const auto scores = torch::from_blob(
                            score_values.data(),
                            {good_count + outlier_count},
                            torch::kFloat32)
                            .clone();

    const auto filtered = pfm::testing::rotation_consistent_matches_for_test(
        features_a,
        features_b,
        matches,
        scores);

    PFM_REQUIRE(filtered.first.size(0) == good_count);
}

static void matching_pipeline_skips_graph_matcher_for_dense_sparse_feature_sets() {
    PFM_REQUIRE(!pfm::testing::should_use_graph_matcher_for_sparse_count_for_test(2048, 2048));
    PFM_REQUIRE(pfm::testing::should_use_graph_matcher_for_sparse_count_for_test(512, 512));
}

static void matching_pipeline_relaxed_graph_candidates_ignore_overconfident_dustbin() {
    auto logits = torch::full({5, 5}, -8.0F, torch::kFloat32);
    for (int64_t index = 0; index < 4; ++index) {
        logits.index_put_({index, index}, 3.0F - static_cast<float>(index) * 0.1F);
        logits.index_put_({index, 4}, 6.0F);
        logits.index_put_({4, index}, 6.0F);
    }

    const auto relaxed = pfm::testing::relaxed_graph_logit_matches_for_test(logits, 4, 4);

    PFM_REQUIRE(relaxed.first.size(0) == 4);
    PFM_REQUIRE(torch::equal(relaxed.first, torch::tensor({{0, 0}, {1, 1}, {2, 2}, {3, 3}}, torch::kInt64)));
}

static void matching_pipeline_filters_descriptor_matches_when_graph_is_skipped() {
    constexpr int64_t total_count = 1050;
    constexpr int64_t inlier_count = 700;
    std::vector<float> keypoint_values_a;
    std::vector<float> keypoint_values_b;
    keypoint_values_a.reserve(static_cast<std::size_t>(total_count * 2));
    keypoint_values_b.reserve(static_cast<std::size_t>(total_count * 2));
    for (int64_t index = 0; index < total_count; ++index) {
        const float x = static_cast<float>(index % 35) * 2.0F;
        const float y = static_cast<float>(index / 35) * 2.0F;
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        if (index < inlier_count) {
            keypoint_values_b.push_back(x + 4.0F);
            keypoint_values_b.push_back(y - 3.0F);
        } else {
            keypoint_values_b.push_back(200.0F + static_cast<float>(index % 17) * 3.0F);
            keypoint_values_b.push_back(150.0F + static_cast<float>(index % 23) * 5.0F);
        }
    }
    const auto keypoints_a = torch::from_blob(
                                 keypoint_values_a.data(),
                                 {total_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto keypoints_b = torch::from_blob(
                                 keypoint_values_b.data(),
                                 {total_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto descriptors = torch::eye(total_count, torch::kFloat32);
    const auto features_a = makeFeatureSet(
        keypoints_a,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto features_b = makeFeatureSet(
        keypoints_b,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    pfm::PlanetaryGraphMatcher matcher(total_count, 8);
    setenv("PFM_MATCH_DEBUG_USE_TOPK_GEOMETRY", "0", 1);

    const auto matches = pfm::matchFeatureSets(features_a, features_b, *matcher);

    unsetenv("PFM_MATCH_DEBUG_USE_TOPK_GEOMETRY");
    PFM_REQUIRE(matches.sparse_matches.size(0) < total_count);
    PFM_REQUIRE(matches.sparse_matches.size(0) >= pfm::testing::geometric_consistency_max_output_matches_for_test());
}

static void matching_pipeline_affine_cleanup_removes_high_residual_rotation_outliers() {
    std::vector<float> keypoint_values_a;
    std::vector<float> keypoint_values_b;
    std::vector<int64_t> match_values;
    std::vector<float> score_values;
    constexpr int64_t good_count = 72;
    constexpr int64_t outlier_count = 3;
    keypoint_values_a.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    keypoint_values_b.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    match_values.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    score_values.reserve(static_cast<std::size_t>(good_count + outlier_count));
    for (int64_t index = 0; index < good_count; ++index) {
        const float x = static_cast<float>(index % 12) * 2.0F;
        const float y = static_cast<float>(index / 12) * 2.0F;
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(10.0F - y);
        keypoint_values_b.push_back(4.0F + x);
        match_values.push_back(index);
        match_values.push_back(index);
        score_values.push_back(1.0F - static_cast<float>(index) * 0.001F);
    }
    for (int64_t index = 0; index < outlier_count; ++index) {
        const auto row = good_count + index;
        const float x = 4.0F + static_cast<float>(index) * 3.0F;
        const float y = 18.0F + static_cast<float>(index);
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(10.0F - y + 1.25F);
        keypoint_values_b.push_back(4.0F + x + 0.85F);
        match_values.push_back(row);
        match_values.push_back(row);
        score_values.push_back(0.97F - static_cast<float>(index) * 0.01F);
    }
    const auto keypoints_a = torch::from_blob(
                                 keypoint_values_a.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto keypoints_b = torch::from_blob(
                                 keypoint_values_b.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto descriptors = torch::eye(good_count + outlier_count, torch::kFloat32);
    const auto features_a = makeFeatureSet(
        keypoints_a,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto features_b = makeFeatureSet(
        keypoints_b,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto matches = torch::from_blob(
                             match_values.data(),
                             {good_count + outlier_count, 2},
                             torch::kInt64)
                             .clone();
    const auto scores = torch::from_blob(
                            score_values.data(),
                            {good_count + outlier_count},
                            torch::kFloat32)
                            .clone();

    const auto cleaned = pfm::testing::affine_residual_cleanup_matches_for_test(
        features_a,
        features_b,
        matches,
        scores);

    PFM_REQUIRE(cleaned.first.size(0) == good_count);
    PFM_REQUIRE(torch::equal(
        cleaned.first.index({torch::indexing::Slice(), 0}),
        torch::arange(good_count, torch::kInt64)));
}

static void matching_pipeline_affine_cleanup_preserves_high_count_rotation_floor() {
    std::vector<float> keypoint_values_a;
    std::vector<float> keypoint_values_b;
    std::vector<int64_t> match_values;
    std::vector<float> score_values;
    constexpr int64_t good_count = 148;
    constexpr int64_t outlier_count = 5;
    keypoint_values_a.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    keypoint_values_b.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    match_values.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    score_values.reserve(static_cast<std::size_t>(good_count + outlier_count));
    for (int64_t index = 0; index < good_count; ++index) {
        const float x = static_cast<float>(index % 20) * 1.5F;
        const float y = static_cast<float>(index / 20) * 1.5F;
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(3.0F + x);
        keypoint_values_b.push_back(7.0F + y);
        match_values.push_back(index);
        match_values.push_back(index);
        score_values.push_back(1.0F - static_cast<float>(index) * 0.0005F);
    }
    for (int64_t index = 0; index < outlier_count; ++index) {
        const auto row = good_count + index;
        const float x = 12.0F + static_cast<float>(index);
        const float y = 14.0F + static_cast<float>(index);
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(3.0F + x + 1.4F);
        keypoint_values_b.push_back(7.0F + y + 1.1F);
        match_values.push_back(row);
        match_values.push_back(row);
        score_values.push_back(0.9F - static_cast<float>(index) * 0.01F);
    }
    const auto keypoints_a = torch::from_blob(
                                 keypoint_values_a.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto keypoints_b = torch::from_blob(
                                 keypoint_values_b.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto descriptors = torch::eye(good_count + outlier_count, torch::kFloat32);
    const auto features_a = makeFeatureSet(
        keypoints_a,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto features_b = makeFeatureSet(
        keypoints_b,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto matches = torch::from_blob(
                             match_values.data(),
                             {good_count + outlier_count, 2},
                             torch::kInt64)
                             .clone();
    const auto scores = torch::from_blob(
                            score_values.data(),
                            {good_count + outlier_count},
                            torch::kFloat32)
                            .clone();

    const auto cleaned = pfm::testing::affine_residual_cleanup_matches_for_test(
        features_a,
        features_b,
        matches,
        scores);

    PFM_REQUIRE(cleaned.first.size(0) == good_count + outlier_count);
}

static void matching_pipeline_affine_cleanup_handles_medium_count_viewpoint_matches() {
    std::vector<float> keypoint_values_a;
    std::vector<float> keypoint_values_b;
    std::vector<int64_t> match_values;
    std::vector<float> score_values;
    constexpr int64_t good_count = 41;
    constexpr int64_t outlier_count = 4;
    keypoint_values_a.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    keypoint_values_b.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    match_values.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    score_values.reserve(static_cast<std::size_t>(good_count + outlier_count));
    for (int64_t index = 0; index < good_count; ++index) {
        const float x = static_cast<float>(index % 9) * 2.0F;
        const float y = static_cast<float>(index / 9) * 2.0F;
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(5.0F + 0.95F * x + 0.08F * y);
        keypoint_values_b.push_back(3.0F - 0.04F * x + 1.05F * y);
        match_values.push_back(index);
        match_values.push_back(index);
        score_values.push_back(1.0F - static_cast<float>(index) * 0.002F);
    }
    for (int64_t index = 0; index < outlier_count; ++index) {
        const auto row = good_count + index;
        const float x = 6.0F + static_cast<float>(index) * 2.0F;
        const float y = 11.0F + static_cast<float>(index);
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(5.0F + 0.95F * x + 0.08F * y + 2.0F);
        keypoint_values_b.push_back(3.0F - 0.04F * x + 1.05F * y - 1.7F);
        match_values.push_back(row);
        match_values.push_back(row);
        score_values.push_back(0.92F - static_cast<float>(index) * 0.01F);
    }
    const auto keypoints_a = torch::from_blob(
                                 keypoint_values_a.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto keypoints_b = torch::from_blob(
                                 keypoint_values_b.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto descriptors = torch::eye(good_count + outlier_count, torch::kFloat32);
    const auto features_a = makeFeatureSet(
        keypoints_a,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto features_b = makeFeatureSet(
        keypoints_b,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto matches = torch::from_blob(
                             match_values.data(),
                             {good_count + outlier_count, 2},
                             torch::kInt64)
                             .clone();
    const auto scores = torch::from_blob(
                            score_values.data(),
                            {good_count + outlier_count},
                            torch::kFloat32)
                            .clone();

    const auto cleaned = pfm::testing::affine_residual_cleanup_matches_for_test(
        features_a,
        features_b,
        matches,
        scores);

    PFM_REQUIRE(cleaned.first.size(0) == good_count);
}

static void matching_pipeline_affine_cleanup_handles_low_count_viewpoint_matches() {
    std::vector<float> keypoint_values_a;
    std::vector<float> keypoint_values_b;
    std::vector<int64_t> match_values;
    std::vector<float> score_values;
    constexpr int64_t good_count = 7;
    constexpr int64_t outlier_count = 2;
    keypoint_values_a.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    keypoint_values_b.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    match_values.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    score_values.reserve(static_cast<std::size_t>(good_count + outlier_count));
    for (int64_t index = 0; index < good_count; ++index) {
        const float x = static_cast<float>(index % 4) * 3.0F;
        const float y = static_cast<float>(index / 4) * 3.0F;
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(2.0F + 0.9F * x + 0.1F * y);
        keypoint_values_b.push_back(4.0F - 0.05F * x + 1.1F * y);
        match_values.push_back(index);
        match_values.push_back(index);
        score_values.push_back(0.95F - static_cast<float>(index) * 0.01F);
    }
    for (int64_t index = 0; index < outlier_count; ++index) {
        const auto row = good_count + index;
        const float x = 1.0F + static_cast<float>(index) * 3.0F;
        const float y = 8.0F + static_cast<float>(index);
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(2.0F + 0.9F * x + 0.1F * y + 2.4F);
        keypoint_values_b.push_back(4.0F - 0.05F * x + 1.1F * y - 2.1F);
        match_values.push_back(row);
        match_values.push_back(row);
        score_values.push_back(0.98F - static_cast<float>(index) * 0.01F);
    }
    const auto keypoints_a = torch::from_blob(
                                 keypoint_values_a.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto keypoints_b = torch::from_blob(
                                 keypoint_values_b.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto descriptors = torch::eye(good_count + outlier_count, torch::kFloat32);
    const auto features_a = makeFeatureSet(
        keypoints_a,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto features_b = makeFeatureSet(
        keypoints_b,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto matches = torch::from_blob(
                             match_values.data(),
                             {good_count + outlier_count, 2},
                             torch::kInt64)
                             .clone();
    const auto scores = torch::from_blob(
                            score_values.data(),
                            {good_count + outlier_count},
                            torch::kFloat32)
                            .clone();

    const auto cleaned = pfm::testing::affine_residual_cleanup_matches_for_test(
        features_a,
        features_b,
        matches,
        scores);

    PFM_REQUIRE(cleaned.first.size(0) == good_count);
}

static void matching_pipeline_affine_cleanup_removes_borderline_residual_outliers() {
    std::vector<float> keypoint_values_a;
    std::vector<float> keypoint_values_b;
    std::vector<int64_t> match_values;
    std::vector<float> score_values;
    constexpr int64_t good_count = 34;
    constexpr int64_t outlier_count = 2;
    keypoint_values_a.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    keypoint_values_b.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    match_values.reserve(static_cast<std::size_t>((good_count + outlier_count) * 2));
    score_values.reserve(static_cast<std::size_t>(good_count + outlier_count));
    for (int64_t index = 0; index < good_count; ++index) {
        const float x = static_cast<float>(index % 9) * 2.0F;
        const float y = static_cast<float>(index / 9) * 2.0F;
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(4.0F - y);
        keypoint_values_b.push_back(6.0F + x);
        match_values.push_back(index);
        match_values.push_back(index);
        score_values.push_back(1.0F - static_cast<float>(index) * 0.001F);
    }
    for (int64_t index = 0; index < outlier_count; ++index) {
        const auto row = good_count + index;
        const float x = 4.0F + static_cast<float>(index) * 2.0F;
        const float y = 9.0F + static_cast<float>(index);
        keypoint_values_a.push_back(x);
        keypoint_values_a.push_back(y);
        keypoint_values_b.push_back(4.0F - y + 0.58F);
        keypoint_values_b.push_back(6.0F + x);
        match_values.push_back(row);
        match_values.push_back(row);
        score_values.push_back(0.95F - static_cast<float>(index) * 0.01F);
    }
    const auto keypoints_a = torch::from_blob(
                                 keypoint_values_a.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto keypoints_b = torch::from_blob(
                                 keypoint_values_b.data(),
                                 {good_count + outlier_count, 2},
                                 torch::kFloat32)
                                 .clone();
    const auto descriptors = torch::eye(good_count + outlier_count, torch::kFloat32);
    const auto features_a = makeFeatureSet(
        keypoints_a,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto features_b = makeFeatureSet(
        keypoints_b,
        descriptors,
        torch::empty({0, 2}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32));
    const auto matches = torch::from_blob(
                             match_values.data(),
                             {good_count + outlier_count, 2},
                             torch::kInt64)
                             .clone();
    const auto scores = torch::from_blob(
                            score_values.data(),
                            {good_count + outlier_count},
                            torch::kFloat32)
                            .clone();

    const auto cleaned = pfm::testing::affine_residual_cleanup_matches_for_test(
        features_a,
        features_b,
        matches,
        scores);

    PFM_REQUIRE(cleaned.first.size(0) == good_count);
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
    register_test("matching_pipeline_accepts_cuda_graph_matcher_with_cpu_features",
                  matchingPipelineAcceptsCudaGraphMatcherWithCpuFeatures);
    register_test("matching_pipeline_keeps_more_than_legacy_256_geometric_matches",
                  matching_pipeline_keeps_more_than_legacy_256_geometric_matches);
    register_test("matching_pipeline_bounds_geometric_ransac_prefix_count_for_large_topk",
                  matching_pipeline_bounds_geometric_ransac_prefix_count_for_large_topk);
    register_test(
        "matching_pipeline_geometric_quality_prefers_broader_inlier_support",
        matching_pipeline_geometric_quality_prefers_broader_inlier_support);
    register_test(
        "matching_pipeline_geometric_quality_penalizes_local_projective_clusters",
        matching_pipeline_geometric_quality_penalizes_local_projective_clusters);
    register_test(
        "matching_pipeline_projective_candidate_selection_can_use_spatial_spread_weight",
        matching_pipeline_projective_candidate_selection_can_use_spatial_spread_weight);
    register_test("matching_pipeline_uses_requested_device_for_descriptor_similarity_when_available",
                  matching_pipeline_uses_requested_device_for_descriptor_similarity_when_available);
    register_test("matching_pipeline_uses_conservative_descriptor_topk_by_default",
                  matching_pipeline_uses_conservative_descriptor_topk_by_default);
    register_test("matching_pipeline_allows_descriptor_topk_override_for_hard_viewpoint_pairs",
                  matching_pipeline_allows_descriptor_topk_override_for_hard_viewpoint_pairs);
    register_test("matching_pipeline_wide_topk_fallback_requires_low_count_gain",
                  matching_pipeline_wide_topk_fallback_requires_low_count_gain);
    register_test("matching_pipeline_tries_projective_topk_before_rotation_filter_for_viewpoint",
                  matching_pipeline_tries_projective_topk_before_rotation_filter_for_viewpoint);
    register_test("matching_pipeline_projective_before_rotation_uses_unique_topk_pairs",
                  matching_pipeline_projective_before_rotation_uses_unique_topk_pairs);
    register_test("matching_pipeline_projective_rescue_requires_large_safe_gain",
                  matching_pipeline_projective_rescue_requires_large_safe_gain);
    register_test("matching_pipeline_conservative_topk_fallback_targets_low_count_results",
                  matching_pipeline_conservative_topk_fallback_targets_low_count_results);
    register_test("matching_pipeline_prefers_mutual_geometry_when_topk_cluster_expands_too_much",
                  matching_pipeline_prefers_mutual_geometry_when_topk_cluster_expands_too_much);
    register_test("matching_pipeline_trims_low_confidence_tail_for_medium_topk_geometry",
                  matching_pipeline_trims_low_confidence_tail_for_medium_topk_geometry);
    register_test("matching_pipeline_keeps_large_rotation_topk_geometry_tail",
                  matching_pipeline_keeps_large_rotation_topk_geometry_tail);
    register_test("matching_pipeline_reciprocal_topk_keeps_only_bidirectional_candidates",
                  matching_pipeline_reciprocal_topk_keeps_only_bidirectional_candidates);
    register_test("matching_pipeline_allows_reciprocal_topk_fallback_env",
                  matching_pipeline_allows_reciprocal_topk_fallback_env);
    register_test("matching_pipeline_allows_rotation_only_geometry_filter_for_extreme_viewpoint",
                  matching_pipeline_allows_rotation_only_geometry_filter_for_extreme_viewpoint);
    register_test("matching_pipeline_rotation_filter_rejects_near_angle_position_outliers",
                  matching_pipeline_rotation_filter_rejects_near_angle_position_outliers);
    register_test("matching_pipeline_skips_graph_matcher_for_dense_sparse_feature_sets",
                  matching_pipeline_skips_graph_matcher_for_dense_sparse_feature_sets);
    register_test("matching_pipeline_relaxed_graph_candidates_ignore_overconfident_dustbin",
                  matching_pipeline_relaxed_graph_candidates_ignore_overconfident_dustbin);
    register_test("matching_pipeline_filters_descriptor_matches_when_graph_is_skipped",
                  matching_pipeline_filters_descriptor_matches_when_graph_is_skipped);
    register_test("matching_pipeline_affine_cleanup_removes_high_residual_rotation_outliers",
                  matching_pipeline_affine_cleanup_removes_high_residual_rotation_outliers);
    register_test("matching_pipeline_affine_cleanup_preserves_high_count_rotation_floor",
                  matching_pipeline_affine_cleanup_preserves_high_count_rotation_floor);
    register_test("matching_pipeline_affine_cleanup_handles_medium_count_viewpoint_matches",
                  matching_pipeline_affine_cleanup_handles_medium_count_viewpoint_matches);
    register_test("matching_pipeline_affine_cleanup_handles_low_count_viewpoint_matches",
                  matching_pipeline_affine_cleanup_handles_low_count_viewpoint_matches);
    register_test("matching_pipeline_affine_cleanup_removes_borderline_residual_outliers",
                  matching_pipeline_affine_cleanup_removes_borderline_residual_outliers);
}
