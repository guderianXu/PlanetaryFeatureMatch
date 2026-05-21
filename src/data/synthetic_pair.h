#pragma once

#include <cstdint>
#include <string>

#include <torch/torch.h>

namespace pfm {

enum class SyntheticPairAugmentationProfile {
    Mixed,
    RotationOnly,
    Mild,
    Medium,
    Hard,
    Extreme,
};

/// Parses a synthetic pair augmentation profile name.
/// @param value Profile name: mixed, mild, medium, hard, or extreme.
/// @return Parsed profile enum.
/// @throws std::invalid_argument if the profile name is unsupported.
SyntheticPairAugmentationProfile parse_synthetic_pair_augmentation_profile(const std::string& value);

/// Converts a synthetic pair augmentation profile to its CLI name.
/// @param profile Profile enum value.
/// @return Stable lowercase profile name.
std::string synthetic_pair_augmentation_profile_name(SyntheticPairAugmentationProfile profile);

struct SyntheticPairConfig {
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
    SyntheticPairAugmentationProfile augmentation_profile = SyntheticPairAugmentationProfile::Mixed;
    double extreme_pair_ratio = 0.2;
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
