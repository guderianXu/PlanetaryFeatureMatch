#pragma once

#include "augment/image_pair_augmentor.h"

namespace pfm {

struct ImagePairTransformParameters {
    float translation_x = 0.0F;
    float translation_y = 0.0F;
    float rotation_degrees = 0.0F;
    float scale = 1.0F;
    float brightness_delta = 0.0F;
    float contrast_scale = 1.0F;
    float noise_sigma = 0.0F;
    float gamma = 1.0F;
    float shadow_strength = 0.0F;
};

/// Samples deterministic image-pair transform parameters.
/// @param config Base augmentation configuration.
/// @return Resolved transform and photometric parameters.
/// @throws std::invalid_argument if config values are invalid.
ImagePairTransformParameters sampleImagePairTransform(const ImagePairAugmentationConfig& config);

}  // namespace pfm
