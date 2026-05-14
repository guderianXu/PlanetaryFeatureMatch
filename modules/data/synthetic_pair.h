#pragma once

#include <torch/torch.h>

namespace pfm {

struct SyntheticPairConfig {
    float translation_x = 0.0F;
    float translation_y = 0.0F;
    float brightness_delta = 0.0F;
    float contrast_scale = 1.0F;
    float noise_sigma = 0.0F;
};

struct SyntheticPair {
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp_a_to_b;
    torch::Tensor valid_mask;
};

/// Creates a deterministic synthetic image pair and dense translation warp.
/// @param image Input CHW float32 image tensor with one or three channels.
/// @param config Photometric and translational augmentation settings.
/// @return Synthetic pair containing both views, A-to-B warp field, and valid mask.
/// @throws std::invalid_argument if image is not a valid CHW image tensor or config contains unsupported values.
SyntheticPair make_synthetic_pair(const torch::Tensor& image, const SyntheticPairConfig& config);

}  // namespace pfm
