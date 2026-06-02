#include <torch/torch.h>

#include "augment/image_pair_augmentor.h"
#include "augment/transform_sampler.h"
#include "tests/test_harness.h"

namespace
{

static void transformSamplerIsDeterministic()
{
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Mixed;
    config.source_index = 3;
    config.variant_index = 4;
    config.seed = 11;

    const auto first = pfm::sampleImagePairTransform(config);
    const auto second = pfm::sampleImagePairTransform(config);

    PFM_REQUIRE_CLOSE(first.rotation_degrees, second.rotation_degrees, 1.0e-6F);
    PFM_REQUIRE_CLOSE(first.scale, second.scale, 1.0e-6F);
    PFM_REQUIRE_CLOSE(first.brightness_delta, second.brightness_delta, 1.0e-6F);
}

static void transformSamplerMixedIncludesHalfTurnVariants()
{
    bool found_half_turn = false;
    for (int64_t variant = 0; variant < 24; ++variant)
    {
        pfm::ImagePairAugmentationConfig config;
        config.profile = pfm::AugmentationProfile::Mixed;
        config.source_index = 3;
        config.variant_index = variant;
        const auto params = pfm::sampleImagePairTransform(config);
        if (std::abs(std::abs(params.rotation_degrees) - 180.0F) <= 5.0F)
        {
            found_half_turn = true;
            break;
        }
    }

    PFM_REQUIRE(found_half_turn);
}

static void transformSamplerMixedHalfTurnIsCleanRotationAnchor()
{
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Mixed;
    config.source_index = 0;
    config.variant_index = 7;

    const auto params = pfm::sampleImagePairTransform(config);

    PFM_REQUIRE_CLOSE(std::abs(params.rotation_degrees), 180.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.translation_x, 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.translation_y, 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.scale, 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.gamma, 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.shadow_strength, 0.0F, 1.0e-6F);
}

static void transformSamplerMixedIncludesQuarterTurnVariants()
{
    bool found_quarter_turn = false;
    for (int64_t variant = 0; variant < 24; ++variant)
    {
        pfm::ImagePairAugmentationConfig config;
        config.profile = pfm::AugmentationProfile::Mixed;
        config.source_index = 3;
        config.variant_index = variant;
        const auto params = pfm::sampleImagePairTransform(config);
        if (std::abs(std::abs(params.rotation_degrees) - 90.0F) <= 5.0F)
        {
            found_quarter_turn = true;
            break;
        }
    }

    PFM_REQUIRE(found_quarter_turn);
}

static void transformSamplerMixedQuarterTurnIsCleanRotationAnchor()
{
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Mixed;
    config.source_index = 0;
    config.variant_index = 3;

    const auto params = pfm::sampleImagePairTransform(config);

    PFM_REQUIRE_CLOSE(std::abs(params.rotation_degrees), 90.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.translation_x, 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.translation_y, 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.scale, 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.gamma, 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.shadow_strength, 0.0F, 1.0e-6F);
}

static void transformSamplerMixedIncludesCleanThirtyDegreeAnchors()
{
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Mixed;
    config.source_index = 0;
    config.variant_index = 2;

    const auto params = pfm::sampleImagePairTransform(config);

    PFM_REQUIRE_CLOSE(params.rotation_degrees, 30.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.translation_x, 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.translation_y, 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.scale, 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.gamma, 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(params.shadow_strength, 0.0F, 1.0e-6F);
}

static void transformSamplerViewpointAddsProjectiveTermsWithoutFullRotationAnchor()
{
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Viewpoint;
    config.source_index = 2;
    config.variant_index = 13;

    const auto params = pfm::sampleImagePairTransform(config);

    PFM_REQUIRE(std::abs(params.shear_x) > 1.0e-4F || std::abs(params.shear_y) > 1.0e-4F);
    PFM_REQUIRE(std::abs(params.perspective_x) > 1.0e-7F || std::abs(params.perspective_y) > 1.0e-7F);
    PFM_REQUIRE(std::abs(params.rotation_degrees) < 75.0F);
}

static void transformSamplerCompoundViewpointKeepsFullRotationBase()
{
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::CompoundViewpoint;
    config.source_index = 2;
    config.variant_index = 12;

    const auto params = pfm::sampleImagePairTransform(config);

    PFM_REQUIRE(std::abs(params.rotation_degrees) > 160.0F);
    PFM_REQUIRE(std::abs(params.perspective_x) > 1.0e-7F || std::abs(params.perspective_y) > 1.0e-7F);
}

static void imagePairAugmentorReturnsCurrentTrainingKeys()
{
    const auto image = torch::linspace(0.0, 1.0, 64, torch::kFloat32).reshape({1, 8, 8});
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Mild;
    config.source_index = 1;
    config.variant_index = 2;

    pfm::ImagePairAugmentor augmentor(config);
    const auto sample = augmentor.augment(image);

    PFM_REQUIRE(sample.view_a.sizes().equals(torch::IntArrayRef({1, 8, 8})));
    PFM_REQUIRE(sample.view_b.sizes().equals(torch::IntArrayRef({1, 8, 8})));
    PFM_REQUIRE(sample.warp_a_to_b.sizes().equals(torch::IntArrayRef({8, 8, 2})));
    PFM_REQUIRE(sample.valid_mask.sizes().equals(torch::IntArrayRef({8, 8})));
    PFM_REQUIRE(sample.view_a.dtype() == torch::kFloat32);
}

static void imagePairAugmentorMixedHalfTurnWarpCrossesImage()
{
    const auto image = torch::linspace(0.0, 1.0, 64, torch::kFloat32).reshape({1, 8, 8});
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Mixed;
    config.source_index = 0;
    config.variant_index = 7;

    pfm::ImagePairAugmentor augmentor(config);
    const auto sample = augmentor.augment(image);

    PFM_REQUIRE_CLOSE(sample.warp_a_to_b.index({0, 0, 0}).item<float>(), 7.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(sample.warp_a_to_b.index({0, 0, 1}).item<float>(), 7.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(sample.warp_a_to_b.index({7, 7, 0}).item<float>(), 0.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(sample.warp_a_to_b.index({7, 7, 1}).item<float>(), 0.0F, 1.0e-5F);
}

} // namespace

void register_augment_tests()
{
    register_test("transform sampler is deterministic", transformSamplerIsDeterministic);
    register_test("transform sampler mixed includes half turn variants", transformSamplerMixedIncludesHalfTurnVariants);
    register_test("transform sampler mixed half turn is clean rotation anchor",
                  transformSamplerMixedHalfTurnIsCleanRotationAnchor);
    register_test("transform sampler mixed includes quarter turn variants",
                  transformSamplerMixedIncludesQuarterTurnVariants);
    register_test("transform sampler mixed quarter turn is clean rotation anchor",
                  transformSamplerMixedQuarterTurnIsCleanRotationAnchor);
    register_test("transform sampler mixed includes clean thirty degree anchors",
                  transformSamplerMixedIncludesCleanThirtyDegreeAnchors);
    register_test("transform sampler viewpoint adds projective terms without full rotation anchor",
                  transformSamplerViewpointAddsProjectiveTermsWithoutFullRotationAnchor);
    register_test("transform sampler compound viewpoint keeps full rotation base",
                  transformSamplerCompoundViewpointKeepsFullRotationBase);
    register_test("image pair augmentor returns current training keys", imagePairAugmentorReturnsCurrentTrainingKeys);
    register_test("image pair augmentor mixed half turn warp crosses image",
                  imagePairAugmentorMixedHalfTurnWarpCrossesImage);
}
