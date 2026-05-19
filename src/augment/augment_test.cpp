#include <torch/torch.h>

#include "augment/image_pair_augmentor.h"
#include "augment/transform_sampler.h"
#include "tests/test_harness.h"

namespace {

static void transformSamplerIsDeterministic() {
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

static void imagePairAugmentorReturnsCurrentTrainingKeys() {
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

}  // namespace

void register_augment_tests() {
    register_test("transform sampler is deterministic", transformSamplerIsDeterministic);
    register_test("image pair augmentor returns current training keys", imagePairAugmentorReturnsCurrentTrainingKeys);
}
