#include <torch/torch.h>

#include "infer/cache_match_eval.h"
#include "tests/test_harness.h"

namespace
{

torch::Tensor makeIdentityWarp(int64_t height, int64_t width)
{
    auto warp = torch::zeros({height, width, 2}, torch::kFloat32);
    for (int64_t y = 0; y < height; ++y)
    {
        for (int64_t x = 0; x < width; ++x)
        {
            warp.index_put_({y, x, 0}, static_cast<float>(x));
            warp.index_put_({y, x, 1}, static_cast<float>(y));
        }
    }
    return warp;
}

} // namespace

static void cache_match_eval_descriptor_grid_uses_row_major_python_keypoints()
{
    const auto image = torch::ones({1, 5, 5}, torch::kFloat32);
    auto descriptors = torch::zeros({1, 2, 3, 3}, torch::kFloat32);
    descriptors.index_put_({0, 0}, torch::arange(9, torch::kFloat32).reshape({3, 3}));
    descriptors.index_put_({0, 1}, torch::ones({3, 3}, torch::kFloat32));
    pfm::PythonDescriptorGridConfig config;
    config.max_keypoints = 16;
    config.min_intensity = 0.0;

    const auto features = pfm::makePythonDescriptorGridFeatureSet(image, descriptors, config);

    PFM_REQUIRE(features.keypoints.size(0) == 9);
    PFM_REQUIRE(torch::equal(features.keypoints.index({0}), torch::tensor({0.0F, 0.0F})));
    PFM_REQUIRE(torch::equal(features.keypoints.index({8}), torch::tensor({2.0F, 2.0F})));
    PFM_REQUIRE_CLOSE(features.descriptors.index({8, 0}).item<float>(), 8.0F, 1.0e-6F);
    PFM_REQUIRE(features.feature_map_width == 3);
    PFM_REQUIRE(features.feature_map_height == 3);
}

static void cache_match_eval_raw_mutual_rejects_channel_shifted_identity_warp_match()
{
    pfm::PairArchiveSample pair;
    pair.path = "memory_pair.pt";
    pair.view_a = torch::ones({1, 3, 3}, torch::kFloat32);
    pair.view_b = torch::ones({1, 3, 3}, torch::kFloat32);
    pair.warp_a_to_b = makeIdentityWarp(3, 3);
    pair.valid_mask = torch::ones({3, 3}, torch::kBool);
    const auto descriptors_a = torch::tensor({{{{1.0F}}, {{0.0F}}, {{0.0F}}, {{0.0F}}}}, torch::kFloat32);
    const auto descriptors_b = torch::tensor({{{{0.0F}}, {{1.0F}}, {{0.0F}}, {{0.0F}}}}, torch::kFloat32);
    pfm::PythonDescriptorGridConfig config;
    config.max_keypoints = 1;
    config.min_intensity = 0.0;

    const auto result = pfm::evaluatePythonRawMutualDescriptorMaps(pair, descriptors_a, descriptors_b, config, 1, 0.1,
                                                                   0.1);

    PFM_REQUIRE(result.matches == 0);
    PFM_REQUIRE(result.correct == 0);
    PFM_REQUIRE(result.wrong == 0);
    PFM_REQUIRE_CLOSE(result.precision, 0.0, 1.0e-6);
}

void register_cache_match_eval_tests()
{
    register_test("cache_match_eval_descriptor_grid_uses_row_major_python_keypoints",
                  cache_match_eval_descriptor_grid_uses_row_major_python_keypoints);
    register_test("cache_match_eval_raw_mutual_rejects_channel_shifted_identity_warp_match",
                  cache_match_eval_raw_mutual_rejects_channel_shifted_identity_warp_match);
}
