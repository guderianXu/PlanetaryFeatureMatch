#include <torch/torch.h>

#include "infer/feature_extractor.h"
#include "tests/test_harness.h"

namespace {

pfm::RawFeatureMaps makeMaps(const torch::Tensor& heatmap, const torch::Tensor& dense_confidence) {
    return pfm::RawFeatureMaps{
        heatmap,
        torch::arange(24, torch::kFloat32).reshape({1, 4, 2, 3}),
        torch::ones({1, 1, 2, 3}, torch::kFloat32),
        torch::zeros({1, 1, 2, 3}, torch::kFloat32),
        torch::ones({1, 4, 2, 3}, torch::kFloat32),
        dense_confidence};
}

}  // namespace

static void decode_sparse_features_returns_top_k_points() {
    auto heatmap = torch::zeros({1, 1, 2, 3}, torch::kFloat32);
    heatmap.index_put_({0, 0, 0, 1}, 0.9F);
    heatmap.index_put_({0, 0, 1, 2}, 0.8F);
    const auto dense_confidence = torch::ones({1, 1, 2, 3}, torch::kFloat32);
    const auto features = pfm::decode_feature_maps(makeMaps(heatmap, dense_confidence), 2, 0.5);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(features.descriptors.sizes() == torch::IntArrayRef({2, 4}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.scores.index({0}).item<float>(), 0.9F, 1.0e-6F);
}

static void decode_dense_features_filters_by_threshold() {
    const auto heatmap = torch::zeros({1, 1, 2, 2}, torch::kFloat32);
    const auto dense_confidence = torch::tensor({{{{0.1F, 0.9F}, {0.8F, 0.2F}}}}, torch::kFloat32);
    auto maps = makeMaps(torch::zeros({1, 1, 2, 3}, torch::kFloat32), torch::ones({1, 1, 2, 3}, torch::kFloat32));
    maps.heatmap = heatmap;
    maps.descriptors = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    maps.scale = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    maps.orientation = torch::zeros({1, 1, 2, 2}, torch::kFloat32);
    maps.affine = torch::ones({1, 4, 2, 2}, torch::kFloat32);
    maps.dense_confidence = dense_confidence;

    const auto features = pfm::decode_feature_maps(maps, 1, 0.75);

    PFM_REQUIRE(features.dense_points.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(features.dense_confidence.sizes() == torch::IntArrayRef({2}));
}

static void decode_feature_maps_rejects_invalid_arguments() {
    auto maps = makeMaps(torch::zeros({1, 1, 2, 3}, torch::kFloat32), torch::ones({1, 1, 2, 3}, torch::kFloat32));
    PFM_REQUIRE_INVALID_ARG(pfm::decode_feature_maps(maps, 0, 0.5));

    maps.heatmap = torch::zeros({2, 1, 2, 3}, torch::kFloat32);
    PFM_REQUIRE_INVALID_ARG(pfm::decode_feature_maps(maps, 1, 0.5));
}

void register_feature_extractor_tests() {
    register_test("decode_sparse_features_returns_top_k_points", decode_sparse_features_returns_top_k_points);
    register_test("decode_dense_features_filters_by_threshold", decode_dense_features_filters_by_threshold);
    register_test("decode_feature_maps_rejects_invalid_arguments", decode_feature_maps_rejects_invalid_arguments);
}
