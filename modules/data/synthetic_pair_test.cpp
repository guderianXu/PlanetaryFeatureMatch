#include "tests/test_harness.h"

#include <torch/torch.h>

#include "data/synthetic_pair.h"

static void synthetic_pair_preserves_expected_shapes() {
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

static void synthetic_pair_marks_translation_invalid_border() {
    auto image = torch::rand({1, 8, 8}, torch::kFloat32);
    pfm::SyntheticPairConfig config;
    config.translation_x = 10.0F;
    config.translation_y = 0.0F;

    auto pair = pfm::make_synthetic_pair(image, config);

    PFM_REQUIRE(pair.valid_mask.sum().item<int64_t>() == 0);
}

static void synthetic_pair_translation_shifts_view_b_content_without_wraparound() {
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

static void synthetic_pair_rejects_fractional_translations() {
    auto image = torch::zeros({1, 3, 4}, torch::kFloat32);
    pfm::SyntheticPairConfig fractional_x;
    fractional_x.translation_x = 0.25F;
    pfm::SyntheticPairConfig fractional_y;
    fractional_y.translation_y = -1.25F;

    bool rejected_x = false;
    try {
        (void)pfm::make_synthetic_pair(image, fractional_x);
    } catch (const std::invalid_argument&) {
        rejected_x = true;
    }

    bool rejected_y = false;
    try {
        (void)pfm::make_synthetic_pair(image, fractional_y);
    } catch (const std::invalid_argument&) {
        rejected_y = true;
    }

    PFM_REQUIRE(rejected_x);
    PFM_REQUIRE(rejected_y);
}

static void synthetic_pair_noise_is_deterministic_and_changes_view_b() {
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

static void synthetic_pair_rejects_negative_noise_sigma() {
    auto image = torch::zeros({1, 3, 4}, torch::kFloat32);
    pfm::SyntheticPairConfig config;
    config.noise_sigma = -0.1F;

    bool rejected = false;
    try {
        (void)pfm::make_synthetic_pair(image, config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    PFM_REQUIRE(rejected);
}

static void synthetic_pair_rejects_non_positive_contrast_scale() {
    auto image = torch::zeros({1, 3, 4}, torch::kFloat32);
    pfm::SyntheticPairConfig zero_config;
    zero_config.contrast_scale = 0.0F;
    pfm::SyntheticPairConfig negative_config;
    negative_config.contrast_scale = -1.0F;

    bool rejected_zero = false;
    try {
        (void)pfm::make_synthetic_pair(image, zero_config);
    } catch (const std::invalid_argument&) {
        rejected_zero = true;
    }

    bool rejected_negative = false;
    try {
        (void)pfm::make_synthetic_pair(image, negative_config);
    } catch (const std::invalid_argument&) {
        rejected_negative = true;
    }

    PFM_REQUIRE(rejected_zero);
    PFM_REQUIRE(rejected_negative);
}

void register_synthetic_pair_tests() {
    register_test("synthetic pair preserves expected shapes", synthetic_pair_preserves_expected_shapes);
    register_test("synthetic pair marks translation invalid border", synthetic_pair_marks_translation_invalid_border);
    register_test("synthetic pair translation shifts view b content without wraparound",
                  synthetic_pair_translation_shifts_view_b_content_without_wraparound);
    register_test("synthetic pair rejects fractional translations", synthetic_pair_rejects_fractional_translations);
    register_test("synthetic pair noise is deterministic and changes view b",
                  synthetic_pair_noise_is_deterministic_and_changes_view_b);
    register_test("synthetic pair rejects negative noise sigma", synthetic_pair_rejects_negative_noise_sigma);
    register_test("synthetic pair rejects non positive contrast scale", synthetic_pair_rejects_non_positive_contrast_scale);
}
