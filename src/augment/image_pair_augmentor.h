#pragma once

#include <cstdint>

#include <torch/torch.h>

#include "augment/augmentation_profile.h"

namespace pfm {

struct ImagePairAugmentationConfig {
    float translation_x = 0.0F;
    float translation_y = 0.0F;
    float rotation_degrees = 0.0F;
    float scale = 1.0F;
    float brightness_delta = 0.0F;
    float contrast_scale = 1.0F;
    float noise_sigma = 0.0F;
    float rotation_step_degrees = 15.0F;
    int64_t variant_index = 0;
    int64_t source_index = 0;
    uint64_t seed = 0;
    AugmentationProfile profile = AugmentationProfile::Mixed;
    double extreme_pair_ratio = 0.2;
};

struct ImagePairSample {
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp_a_to_b;
    torch::Tensor valid_mask;
};

class ImagePairAugmentor {
public:
    /// Creates an image-pair augmentor.
    /// @param config Augmentation configuration.
    explicit ImagePairAugmentor(ImagePairAugmentationConfig config);

    /// Generates a synthetic image pair and dense correspondence field.
    /// @param image Source CHW float image.
    /// @return Augmented pair sample.
    /// @throws std::invalid_argument if the image or config is invalid.
    ImagePairSample augment(const torch::Tensor& image) const;

private:
    ImagePairAugmentationConfig _config;
};

}  // namespace pfm
