#include <torch/torch.h>

#include "infer/feature_extractor.h"
#include "tests/test_harness.h"

namespace {

pfm::RawFeatureMaps makeMaps(const torch::Tensor& heatmap, const torch::Tensor& dense_confidence) {
    return pfm::RawFeatureMaps{
        heatmap,
        torch::arange(24, torch::kFloat32).reshape({1, 4, 2, 3}),
        torch::ones({1, 1, 2, 3}, torch::kFloat32),
        torch::zeros({1, 2, 2, 3}, torch::kFloat32),
        torch::ones({1, 4, 2, 3}, torch::kFloat32),
        dense_confidence};
}

pfm::RawFeatureMaps makeUniformMaps(const torch::Tensor& heatmap) {
    const auto height = heatmap.size(2);
    const auto width = heatmap.size(3);
    return pfm::RawFeatureMaps{
        heatmap,
        torch::ones({1, 4, height, width}, torch::kFloat32),
        torch::ones({1, 1, height, width}, torch::kFloat32),
        torch::zeros({1, 2, height, width}, torch::kFloat32),
        torch::ones({1, 4, height, width}, torch::kFloat32),
        torch::ones({1, 1, height, width}, torch::kFloat32)};
}

}  // namespace

static void decode_sparse_features_gathers_values_at_top_k_points() {
    auto heatmap = torch::zeros({1, 1, 2, 3}, torch::kFloat32);
    heatmap.index_put_({0, 0, 0, 1}, 0.9F);
    heatmap.index_put_({0, 0, 1, 2}, 0.8F);
    auto maps = makeMaps(heatmap, torch::ones({1, 1, 2, 3}, torch::kFloat32));
    maps.scale = torch::tensor({{{{1.0F, 1.5F, 2.0F}, {2.5F, 3.0F, 3.5F}}}}, torch::kFloat32);
    maps.orientation = torch::tensor(
        {{{{10.0F, 20.0F, 30.0F}, {40.0F, 50.0F, 60.0F}},
          {{11.0F, 21.0F, 31.0F}, {41.0F, 51.0F, 61.0F}}}},
        torch::kFloat32);
    maps.affine = torch::tensor(
        {{{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}},
          {{7.0F, 8.0F, 9.0F}, {10.0F, 11.0F, 12.0F}},
          {{13.0F, 14.0F, 15.0F}, {16.0F, 17.0F, 18.0F}},
          {{19.0F, 20.0F, 21.0F}, {22.0F, 23.0F, 24.0F}}}},
        torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 2, 0.5);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(features.descriptors.sizes() == torch::IntArrayRef({2, 4}));
    PFM_REQUIRE(features.orientation.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({0}).item<float>(), 0.9F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.descriptors.index({0, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.descriptors.index({0, 3}).item<float>(), 19.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scale.index({0}).item<float>(), 1.5F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.orientation.index({0, 0}).item<float>(), 20.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.orientation.index({0, 1}).item<float>(), 21.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.affine.index({0, 0, 0}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.affine.index({0, 0, 1}).item<float>(), 8.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.affine.index({0, 1, 0}).item<float>(), 14.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.affine.index({0, 1, 1}).item<float>(), 20.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 0}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.descriptors.index({1, 0}).item<float>(), 5.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.descriptors.index({1, 3}).item<float>(), 23.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scale.index({1}).item<float>(), 3.5F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.orientation.index({1, 0}).item<float>(), 60.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.orientation.index({1, 1}).item<float>(), 61.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.affine.index({1, 1, 1}).item<float>(), 24.0F, 1.0e-6F);
}

static void decode_feature_maps_excludes_masked_sparse_locations() {
    auto heatmap = torch::zeros({1, 1, 2, 3}, torch::kFloat32);
    heatmap.index_put_({0, 0, 0, 0}, 10.0F);
    heatmap.index_put_({0, 0, 1, 2}, 9.0F);
    auto maps = makeMaps(heatmap, torch::ones({1, 1, 2, 3}, torch::kFloat32));
    const auto mask = torch::tensor({{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F}}, torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 2, 0.5, mask);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({1, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({0}).item<float>(), 9.0F, 1.0e-6F);
}

static void decode_feature_maps_returns_zero_score_sparse_keypoints() {
    const auto heatmap = torch::zeros({1, 1, 2, 2}, torch::kFloat32);
    auto maps = makeMaps(heatmap, torch::ones({1, 1, 2, 2}, torch::kFloat32));
    maps.descriptors = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 2, 2}, torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 2, 0.5);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(features.scores.sizes() == torch::IntArrayRef({2}));
    PFM_REQUIRE_CLOSE(features.scores.index({0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({1}).item<float>(), 0.0F, 1.0e-6F);
}

static void decode_feature_maps_returns_empty_sparse_features_when_mask_is_empty() {
    auto maps = makeMaps(torch::ones({1, 1, 2, 2}, torch::kFloat32), torch::ones({1, 1, 2, 2}, torch::kFloat32));
    maps.descriptors = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    const auto mask = torch::zeros({2, 2}, torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 4, 0.5, mask);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({0, 2}));
    PFM_REQUIRE(features.scores.sizes() == torch::IntArrayRef({0}));
    PFM_REQUIRE(features.descriptors.sizes() == torch::IntArrayRef({0, 4}));
    PFM_REQUIRE(features.scale.sizes() == torch::IntArrayRef({0}));
    PFM_REQUIRE(features.orientation.sizes() == torch::IntArrayRef({0, 2}));
    PFM_REQUIRE(features.affine.sizes() == torch::IntArrayRef({0, 2, 2}));
}

static void decode_feature_maps_suppresses_neighbors_with_nms_radius() {
    auto heatmap = torch::zeros({1, 1, 4, 4}, torch::kFloat32);
    heatmap.index_put_({0, 0, 1, 1}, 10.0F);
    heatmap.index_put_({0, 0, 1, 2}, 9.0F);
    heatmap.index_put_({0, 0, 3, 3}, 8.0F);
    auto maps = makeMaps(heatmap, torch::ones({1, 1, 4, 4}, torch::kFloat32));
    maps.descriptors = torch::ones({1, 4, 4, 4}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 4, 4}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 2, 4, 4}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 4, 4}, torch::kFloat32);
    pfm::FeatureDecodeConfig config;
    config.max_keypoints = 2;
    config.semi_dense_threshold = 0.5;
    config.keypoint_grid_rows = 1;
    config.keypoint_grid_cols = 1;
    config.keypoints_per_cell = 2;
    config.nms_radius = 1;

    const auto features = pfm::decode_feature_maps(maps, config, torch::Tensor());

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 0}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 1}).item<float>(), 3.0F, 1.0e-6F);
}

static void decode_feature_maps_avoids_row_major_grid_truncation_bias() {
    auto heatmap = torch::zeros({1, 1, 4, 4}, torch::kFloat32);
    heatmap.index_put_({0, 0, 0, 0}, 10.0F);
    heatmap.index_put_({0, 0, 0, 1}, 9.0F);
    heatmap.index_put_({0, 0, 1, 0}, 8.0F);
    heatmap.index_put_({0, 0, 3, 3}, 7.0F);
    auto maps = makeMaps(heatmap, torch::ones({1, 1, 4, 4}, torch::kFloat32));
    maps.descriptors = torch::ones({1, 4, 4, 4}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 4, 4}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 2, 4, 4}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 4, 4}, torch::kFloat32);
    pfm::FeatureDecodeConfig config;
    config.max_keypoints = 3;
    config.semi_dense_threshold = 0.5;
    config.keypoint_grid_rows = 2;
    config.keypoint_grid_cols = 2;
    config.keypoints_per_cell = 1;
    config.nms_radius = 0;

    const auto features = pfm::decode_feature_maps(maps, config, torch::Tensor());

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({3, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({0}).item<float>(), 10.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 0}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 1}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({1}).item<float>(), 7.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({2, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({2, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({2}).item<float>(), 9.0F, 1.0e-6F);
}

static void decode_feature_maps_allows_zero_score_candidate_in_each_grid_cell() {
    const auto heatmap = torch::zeros({1, 1, 1, 4}, torch::kFloat32);
    const auto maps = makeUniformMaps(heatmap);
    pfm::FeatureDecodeConfig config;
    config.max_keypoints = 2;
    config.semi_dense_threshold = 0.5;
    config.keypoint_grid_rows = 1;
    config.keypoint_grid_cols = 2;
    config.keypoints_per_cell = 1;
    config.nms_radius = 0;

    const auto features = pfm::decode_feature_maps(maps, config, torch::Tensor());

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 0}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({1}).item<float>(), 0.0F, 1.0e-6F);
}

static void decode_feature_maps_filters_dense_points_with_mask() {
    const auto heatmap = torch::zeros({1, 1, 2, 2}, torch::kFloat32);
    const auto dense_confidence = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    auto maps = makeMaps(heatmap, dense_confidence);
    maps.descriptors = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    const auto mask = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 1, 0.5, mask);

    PFM_REQUIRE(features.dense_points.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE_CLOSE(features.dense_points.index({0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.dense_points.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.dense_points.index({1, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.dense_points.index({1, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void decode_dense_features_returns_exact_points_and_confidence() {
    const auto heatmap = torch::zeros({1, 1, 2, 2}, torch::kFloat32);
    const auto dense_confidence = torch::tensor({{{{0.1F, 0.9F}, {0.8F, 0.2F}}}}, torch::kFloat32);
    auto maps = makeMaps(heatmap, dense_confidence);
    maps.descriptors = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 2, 2}, torch::kFloat32);

    const auto features = pfm::decode_feature_maps(maps, 1, 0.75);

    PFM_REQUIRE(features.dense_points.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(features.dense_confidence.sizes() == torch::IntArrayRef({2}));
    PFM_REQUIRE_CLOSE(features.dense_points.index({0, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.dense_points.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.dense_confidence.index({0}).item<float>(), 0.9F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.dense_points.index({1, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.dense_points.index({1, 1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.dense_confidence.index({1}).item<float>(), 0.8F, 1.0e-6F);
}

static void decode_dense_features_returns_empty_tensors_when_no_confidence_matches() {
    auto maps = makeMaps(torch::zeros({1, 1, 2, 3}, torch::kFloat32), torch::zeros({1, 1, 2, 3}, torch::kFloat32));

    const auto features = pfm::decode_feature_maps(maps, 1, 0.5);

    PFM_REQUIRE(features.dense_points.sizes() == torch::IntArrayRef({0, 2}));
    PFM_REQUIRE(features.dense_confidence.sizes() == torch::IntArrayRef({0}));
}

static void decode_feature_maps_returns_cpu_float_contiguous_tensors() {
    auto maps = makeMaps(torch::zeros({1, 1, 2, 3}, torch::kFloat64), torch::ones({1, 1, 2, 3}, torch::kFloat64));
    maps.descriptors = torch::ones({1, 4, 2, 3}, torch::kFloat64);
    maps.scale = torch::ones({1, 1, 2, 3}, torch::kFloat64);
    maps.orientation = torch::zeros({1, 2, 2, 3}, torch::kFloat64);
    maps.affine = torch::ones({1, 4, 2, 3}, torch::kFloat64);

    const auto features = pfm::decode_feature_maps(maps, 1, 0.5);

    PFM_REQUIRE(features.keypoints.device().is_cpu());
    PFM_REQUIRE(features.scores.device().is_cpu());
    PFM_REQUIRE(features.descriptors.device().is_cpu());
    PFM_REQUIRE(features.scale.device().is_cpu());
    PFM_REQUIRE(features.orientation.device().is_cpu());
    PFM_REQUIRE(features.affine.device().is_cpu());
    PFM_REQUIRE(features.dense_points.device().is_cpu());
    PFM_REQUIRE(features.dense_confidence.device().is_cpu());
    PFM_REQUIRE(features.keypoints.dtype() == torch::kFloat32);
    PFM_REQUIRE(features.scores.dtype() == torch::kFloat32);
    PFM_REQUIRE(features.descriptors.dtype() == torch::kFloat32);
    PFM_REQUIRE(features.scale.dtype() == torch::kFloat32);
    PFM_REQUIRE(features.orientation.dtype() == torch::kFloat32);
    PFM_REQUIRE(features.affine.dtype() == torch::kFloat32);
    PFM_REQUIRE(features.dense_points.dtype() == torch::kFloat32);
    PFM_REQUIRE(features.dense_confidence.dtype() == torch::kFloat32);
    PFM_REQUIRE(features.keypoints.is_contiguous());
    PFM_REQUIRE(features.scores.is_contiguous());
    PFM_REQUIRE(features.descriptors.is_contiguous());
    PFM_REQUIRE(features.scale.is_contiguous());
    PFM_REQUIRE(features.orientation.is_contiguous());
    PFM_REQUIRE(features.affine.is_contiguous());
    PFM_REQUIRE(features.dense_points.is_contiguous());
    PFM_REQUIRE(features.dense_confidence.is_contiguous());
}

static void decode_feature_maps_rejects_invalid_arguments() {
    auto maps = makeMaps(torch::zeros({1, 1, 2, 3}, torch::kFloat32), torch::ones({1, 1, 2, 3}, torch::kFloat32));
    PFM_REQUIRE_INVALID_ARG(pfm::decode_feature_maps(maps, 0, 0.5));

    maps.heatmap = torch::zeros({2, 1, 2, 3}, torch::kFloat32);
    PFM_REQUIRE_INVALID_ARG(pfm::decode_feature_maps(maps, 1, 0.5));

    maps = makeMaps(torch::zeros({1, 1, 2, 3}, torch::kFloat32), torch::ones({1, 1, 2, 3}, torch::kFloat32));
    maps.affine = torch::ones({1, 3, 2, 3}, torch::kFloat32);
    PFM_REQUIRE_INVALID_ARG(pfm::decode_feature_maps(maps, 1, 0.5));

    maps = makeMaps(torch::zeros({1, 2, 2, 3}, torch::kFloat32), torch::ones({1, 1, 2, 3}, torch::kFloat32));
    PFM_REQUIRE_INVALID_ARG(pfm::decode_feature_maps(maps, 1, 0.5));

    maps = makeMaps(torch::zeros({1, 1, 2, 3}, torch::kFloat32), torch::ones({1, 1, 2, 3}, torch::kFloat32));
    maps.descriptors = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    PFM_REQUIRE_INVALID_ARG(pfm::decode_feature_maps(maps, 1, 0.5));

    maps = makeMaps(torch::zeros({1, 1, 2, 3}, torch::kFloat32), torch::ones({1, 1, 2, 3}, torch::kFloat32));
    maps.orientation = torch::zeros({1, 1, 2, 3}, torch::kFloat32);
    PFM_REQUIRE_INVALID_ARG(pfm::decode_feature_maps(maps, 1, 0.5));
}

void register_feature_extractor_tests() {
    register_test(
        "decode_sparse_features_gathers_values_at_top_k_points",
        decode_sparse_features_gathers_values_at_top_k_points);
    register_test(
        "decode_feature_maps_excludes_masked_sparse_locations",
        decode_feature_maps_excludes_masked_sparse_locations);
    register_test(
        "decode_feature_maps_returns_zero_score_sparse_keypoints",
        decode_feature_maps_returns_zero_score_sparse_keypoints);
    register_test(
        "decode_feature_maps_returns_empty_sparse_features_when_mask_is_empty",
        decode_feature_maps_returns_empty_sparse_features_when_mask_is_empty);
    register_test(
        "decode_feature_maps_suppresses_neighbors_with_nms_radius",
        decode_feature_maps_suppresses_neighbors_with_nms_radius);
    register_test(
        "decode_feature_maps_avoids_row_major_grid_truncation_bias",
        decode_feature_maps_avoids_row_major_grid_truncation_bias);
    register_test(
        "decode_feature_maps_allows_zero_score_candidate_in_each_grid_cell",
        decode_feature_maps_allows_zero_score_candidate_in_each_grid_cell);
    register_test(
        "decode_feature_maps_filters_dense_points_with_mask",
        decode_feature_maps_filters_dense_points_with_mask);
    register_test(
        "decode_dense_features_returns_exact_points_and_confidence",
        decode_dense_features_returns_exact_points_and_confidence);
    register_test(
        "decode_dense_features_returns_empty_tensors_when_no_confidence_matches",
        decode_dense_features_returns_empty_tensors_when_no_confidence_matches);
    register_test(
        "decode_feature_maps_returns_cpu_float_contiguous_tensors",
        decode_feature_maps_returns_cpu_float_contiguous_tensors);
    register_test("decode_feature_maps_rejects_invalid_arguments", decode_feature_maps_rejects_invalid_arguments);
}
