#include <cmath>

#include <torch/torch.h>

#include "data/synthetic_pair.h"
#include "tests/test_harness.h"

static void synthetic_pair_preserves_expected_shapes()
{
    auto image = torch::rand({1, 16, 20}, torch::kFloat32);
    pfm::SyntheticPairConfig config;
    config.translation_x = 2.0F;
    config.translation_y = 1.0F;
    config.brightness_delta = 0.1F;

    auto pair = pfm::make_synthetic_pair(image, config);

    PFM_REQUIRE(pair.view_a.sizes() == image.sizes());
    PFM_REQUIRE(pair.view_b.sizes() == image.sizes());
    PFM_REQUIRE(pair.warp_a_to_b.sizes() == torch::IntArrayRef({16, 20, 2}));
    PFM_REQUIRE(pair.valid_mask.sizes() == torch::IntArrayRef({16, 20}));
}

static void synthetic_pair_marks_translation_invalid_border()
{
    auto image = torch::rand({1, 8, 8}, torch::kFloat32);
    pfm::SyntheticPairConfig config;
    config.translation_x = 10.0F;
    config.translation_y = 0.0F;

    auto pair = pfm::make_synthetic_pair(image, config);

    PFM_REQUIRE(pair.valid_mask.sum().item<int64_t>() == 0);
}

static void synthetic_pair_translation_shifts_view_b_content_without_wraparound()
{
    auto image = torch::zeros({1, 3, 4}, torch::kFloat32);
    image.index_put_({0, 0, 0}, 0.75F);
    image.index_put_({0, 0, 3}, 0.50F);
    pfm::SyntheticPairConfig config;
    config.translation_x = 1.0F;
    config.translation_y = 0.0F;

    auto pair = pfm::make_synthetic_pair(image, config);

    PFM_REQUIRE(std::abs(pair.view_b.index({0, 0, 1}).item<float>() - image.index({0, 0, 0}).item<float>()) < 1.0e-6F);
    PFM_REQUIRE(pair.view_b.index({0, 0, 0}).item<float>() == 0.0F);
}

static void synthetic_pair_rotation_scale_changes_warp_non_uniformly()
{
    auto image = torch::zeros({1, 9, 9}, torch::kFloat32);
    image.index_put_({0, 2, 4}, 1.0F);
    image.index_put_({0, 6, 4}, 0.5F);
    pfm::SyntheticPairConfig config;
    config.scale = 0.8F;
    config.rotation_degrees = 20.0F;

    auto pair = pfm::make_synthetic_pair(image, config);

    auto center = pair.warp_a_to_b.index({4, 4});
    auto above = pair.warp_a_to_b.index({2, 4});
    auto below = pair.warp_a_to_b.index({6, 4});

    PFM_REQUIRE(!torch::allclose(pair.view_a, pair.view_b));
    PFM_REQUIRE(std::abs(center.index({0}).item<float>() - 4.0F) < 1.0e-5F);
    PFM_REQUIRE(above.index({0}).item<float>() > 4.0F);
    PFM_REQUIRE(below.index({0}).item<float>() < 4.0F);
}

static void synthetic_pair_warp_matches_shifted_view_content()
{
    auto image = torch::zeros({1, 5, 5}, torch::kFloat32);
    image.index_put_({0, 2, 2}, 1.0F);
    pfm::SyntheticPairConfig config;
    config.translation_x = 1.0F;

    auto pair = pfm::make_synthetic_pair(image, config);

    auto target = pair.warp_a_to_b.index({2, 2}).round().to(torch::kLong);
    const auto target_x = target.index({0}).item<int64_t>();
    const auto target_y = target.index({1}).item<int64_t>();
    PFM_REQUIRE_CLOSE(pair.view_b.index({0, target_y, target_x}).item<float>(), 1.0F, 1.0e-6F);
}

static void synthetic_pair_warp_matches_scaled_view_content()
{
    auto image = torch::zeros({1, 5, 5}, torch::kFloat32);
    image.index_put_({0, 2, 1}, 1.0F);
    pfm::SyntheticPairConfig config;
    config.scale = 2.0F;

    auto pair = pfm::make_synthetic_pair(image, config);

    auto target = pair.warp_a_to_b.index({2, 1}).round().to(torch::kLong);
    const auto target_x = target.index({0}).item<int64_t>();
    const auto target_y = target.index({1}).item<int64_t>();
    PFM_REQUIRE_CLOSE(pair.view_b.index({0, target_y, target_x}).item<float>(), 1.0F, 1.0e-6F);
}

static void synthetic_pair_rejects_fractional_translations()
{
    auto image = torch::zeros({1, 3, 4}, torch::kFloat32);
    pfm::SyntheticPairConfig fractional_x;
    fractional_x.translation_x = 0.25F;
    pfm::SyntheticPairConfig fractional_y;
    fractional_y.translation_y = -1.25F;

    bool rejected_x = false;
    try
    {
        (void)pfm::make_synthetic_pair(image, fractional_x);
    }
    catch (const std::invalid_argument&)
    {
        rejected_x = true;
    }

    bool rejected_y = false;
    try
    {
        (void)pfm::make_synthetic_pair(image, fractional_y);
    }
    catch (const std::invalid_argument&)
    {
        rejected_y = true;
    }

    PFM_REQUIRE(rejected_x);
    PFM_REQUIRE(rejected_y);
}

static void synthetic_pair_noise_is_deterministic_and_changes_view_b()
{
    auto image = torch::full({1, 4, 5}, 0.5F, torch::kFloat32);
    pfm::SyntheticPairConfig noisy_config;
    noisy_config.noise_sigma = 0.1F;
    pfm::SyntheticPairConfig noiseless_config;
    noiseless_config.noise_sigma = 0.0F;

    auto first_pair = pfm::make_synthetic_pair(image, noisy_config);
    auto second_pair = pfm::make_synthetic_pair(image, noisy_config);
    auto noiseless_pair = pfm::make_synthetic_pair(image, noiseless_config);

    PFM_REQUIRE(torch::allclose(first_pair.view_b, second_pair.view_b));
    PFM_REQUIRE(!torch::allclose(first_pair.view_b, noiseless_pair.view_b));
}

static void synthetic_pair_rejects_negative_noise_sigma()
{
    auto image = torch::zeros({1, 3, 4}, torch::kFloat32);
    pfm::SyntheticPairConfig config;
    config.noise_sigma = -0.1F;

    bool rejected = false;
    try
    {
        (void)pfm::make_synthetic_pair(image, config);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    PFM_REQUIRE(rejected);
}

static void synthetic_pair_rejects_non_positive_contrast_scale()
{
    auto image = torch::zeros({1, 3, 4}, torch::kFloat32);
    pfm::SyntheticPairConfig zero_config;
    zero_config.contrast_scale = 0.0F;
    pfm::SyntheticPairConfig negative_config;
    negative_config.contrast_scale = -1.0F;

    bool rejected_zero = false;
    try
    {
        (void)pfm::make_synthetic_pair(image, zero_config);
    }
    catch (const std::invalid_argument&)
    {
        rejected_zero = true;
    }

    bool rejected_negative = false;
    try
    {
        (void)pfm::make_synthetic_pair(image, negative_config);
    }
    catch (const std::invalid_argument&)
    {
        rejected_negative = true;
    }

    PFM_REQUIRE(rejected_zero);
    PFM_REQUIRE(rejected_negative);
}

static float mean_warp_displacement(const pfm::SyntheticPair& pair)
{
    const auto grid = torch::stack(torch::meshgrid({torch::arange(pair.warp_a_to_b.size(0), torch::kFloat32),
                                                    torch::arange(pair.warp_a_to_b.size(1), torch::kFloat32)},
                                                   "ij"),
                                   2);
    const auto xy_grid = torch::stack({grid.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}),
                                       grid.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})},
                                      2);
    return (pair.warp_a_to_b - xy_grid).norm(2, 2).mean().item<float>();
}

static void synthetic_pair_extreme_profile_is_stronger_than_mild()
{
    auto image = torch::rand({1, 64, 64}, torch::kFloat32);
    pfm::SyntheticPairConfig mild_config;
    mild_config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Mild;
    mild_config.source_index = 2;
    mild_config.variant_index = 3;
    pfm::SyntheticPairConfig extreme_config = mild_config;
    extreme_config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Extreme;

    const auto mild_pair = pfm::make_synthetic_pair(image, mild_config);
    const auto extreme_pair = pfm::make_synthetic_pair(image, extreme_config);

    PFM_REQUIRE(mean_warp_displacement(extreme_pair) > mean_warp_displacement(mild_pair) * 2.0F);
    PFM_REQUIRE(extreme_pair.valid_mask.sum().item<int64_t>() > 0);
    PFM_REQUIRE(!torch::allclose(mild_pair.view_b, extreme_pair.view_b));
}

static void synthetic_pair_mixed_profile_varies_variant_strength()
{
    auto image = torch::rand({1, 64, 64}, torch::kFloat32);
    pfm::SyntheticPairConfig first_config;
    first_config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Mixed;
    first_config.extreme_pair_ratio = 0.5;
    first_config.source_index = 1;
    first_config.variant_index = 0;
    pfm::SyntheticPairConfig second_config = first_config;
    second_config.variant_index = 1;

    const auto first_pair = pfm::make_synthetic_pair(image, first_config);
    const auto second_pair = pfm::make_synthetic_pair(image, second_config);

    PFM_REQUIRE(std::abs(mean_warp_displacement(first_pair) - mean_warp_displacement(second_pair)) > 2.0F);
    PFM_REQUIRE(!torch::allclose(first_pair.view_b, second_pair.view_b));
}

void register_synthetic_pair_tests()
{
    register_test("synthetic pair preserves expected shapes", synthetic_pair_preserves_expected_shapes);
    register_test("synthetic pair marks translation invalid border", synthetic_pair_marks_translation_invalid_border);
    register_test("synthetic pair translation shifts view b content without wraparound",
                  synthetic_pair_translation_shifts_view_b_content_without_wraparound);
    register_test("synthetic pair rotation scale changes warp non uniformly",
                  synthetic_pair_rotation_scale_changes_warp_non_uniformly);
    register_test("synthetic pair warp matches shifted view content", synthetic_pair_warp_matches_shifted_view_content);
    register_test("synthetic pair warp matches scaled view content", synthetic_pair_warp_matches_scaled_view_content);
    register_test("synthetic pair rejects fractional translations", synthetic_pair_rejects_fractional_translations);
    register_test("synthetic pair noise is deterministic and changes view b",
                  synthetic_pair_noise_is_deterministic_and_changes_view_b);
    register_test("synthetic pair rejects negative noise sigma", synthetic_pair_rejects_negative_noise_sigma);
    register_test("synthetic pair rejects non positive contrast scale",
                  synthetic_pair_rejects_non_positive_contrast_scale);
    register_test("synthetic pair extreme profile is stronger than mild",
                  synthetic_pair_extreme_profile_is_stronger_than_mild);
    register_test("synthetic pair mixed profile varies variant strength",
                  synthetic_pair_mixed_profile_varies_variant_strength);
}
