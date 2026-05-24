#include <torch/torch.h>

#include "data/intensity_mask.h"
#include "tests/test_harness.h"

static void intensity_mask_thresholds_single_channel_image() {
    const auto image = torch::tensor({{{0.0F, 0.05F}, {0.1F, 0.2F}}}, torch::kFloat32);
    const auto mask = pfm::make_intensity_mask(image, 0.1);

    PFM_REQUIRE(mask.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE(mask.dtype() == torch::kFloat32);
    PFM_REQUIRE_CLOSE(mask.index({0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({0, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({1, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({1, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void intensity_mask_uses_mean_for_multi_channel_image() {
    const auto image = torch::tensor({{{0.0F, 0.4F}}, {{0.0F, 0.4F}}, {{0.3F, 0.4F}}}, torch::kFloat32);
    const auto mask = pfm::make_intensity_mask(image, 0.2);

    PFM_REQUIRE(mask.sizes() == torch::IntArrayRef({1, 2}));
    PFM_REQUIRE_CLOSE(mask.index({0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({0, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void intensity_mask_rejects_invalid_thresholds() {
    const auto image = torch::ones({1, 2, 2}, torch::kFloat32);

    PFM_REQUIRE_INVALID_ARG(pfm::validate_min_keypoint_intensity(-0.1));
    PFM_REQUIRE_INVALID_ARG(pfm::validate_min_keypoint_intensity(1.1));
    PFM_REQUIRE_INVALID_ARG(pfm::make_intensity_mask(image, -0.1));
}

static void intensity_mask_suppresses_isolated_bright_noise() {
    auto image = torch::zeros({1, 9, 9}, torch::kFloat32);
    image.index_put_({0, 1, 1}, 1.0F);
    image.index_put_({0, torch::indexing::Slice(3, 8), torch::indexing::Slice(3, 8)}, 1.0F);

    const auto mask = pfm::make_intensity_mask(image, 0.1);

    PFM_REQUIRE_CLOSE(mask.index({1, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(mask.index({5, 5}).item<float>(), 1.0F, 1.0e-6F);
}

void register_intensity_mask_tests() {
    register_test("intensity_mask_thresholds_single_channel_image", intensity_mask_thresholds_single_channel_image);
    register_test("intensity_mask_uses_mean_for_multi_channel_image", intensity_mask_uses_mean_for_multi_channel_image);
    register_test("intensity_mask_rejects_invalid_thresholds", intensity_mask_rejects_invalid_thresholds);
    register_test("intensity_mask_suppresses_isolated_bright_noise", intensity_mask_suppresses_isolated_bright_noise);
}
