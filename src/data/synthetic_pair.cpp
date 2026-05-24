#include "data/synthetic_pair.h"

#include <stdexcept>

#include "augment/image_pair_augmentor.h"

namespace pfm {
namespace {

AugmentationProfile toAugmentationProfile(SyntheticPairAugmentationProfile profile) {
    switch (profile) {
        case SyntheticPairAugmentationProfile::Mixed:
            return AugmentationProfile::Mixed;
        case SyntheticPairAugmentationProfile::RotationOnly:
            return AugmentationProfile::RotationOnly;
        case SyntheticPairAugmentationProfile::Mild:
            return AugmentationProfile::Mild;
        case SyntheticPairAugmentationProfile::Medium:
            return AugmentationProfile::Medium;
        case SyntheticPairAugmentationProfile::Hard:
            return AugmentationProfile::Hard;
        case SyntheticPairAugmentationProfile::Extreme:
            return AugmentationProfile::Extreme;
        case SyntheticPairAugmentationProfile::Viewpoint:
            return AugmentationProfile::Viewpoint;
        case SyntheticPairAugmentationProfile::CompoundViewpoint:
            return AugmentationProfile::CompoundViewpoint;
    }
    return AugmentationProfile::Mixed;
}

ImagePairAugmentationConfig toAugmentationConfig(const SyntheticPairConfig& config) {
    ImagePairAugmentationConfig result;
    result.translation_x = config.translation_x;
    result.translation_y = config.translation_y;
    result.rotation_degrees = config.rotation_degrees;
    result.scale = config.scale;
    result.brightness_delta = config.brightness_delta;
    result.contrast_scale = config.contrast_scale;
    result.noise_sigma = config.noise_sigma;
    result.rotation_step_degrees = config.rotation_step_degrees;
    result.variant_index = config.variant_index;
    result.source_index = config.source_index;
    result.profile = toAugmentationProfile(config.augmentation_profile);
    result.extreme_pair_ratio = config.extreme_pair_ratio;
    return result;
}

}  // namespace

SyntheticPairAugmentationProfile parse_synthetic_pair_augmentation_profile(const std::string& value) {
    if (value == "mixed") {
        return SyntheticPairAugmentationProfile::Mixed;
    }
    if (value == "rotation-only" || value == "rotation_only") {
        return SyntheticPairAugmentationProfile::RotationOnly;
    }
    if (value == "mild") {
        return SyntheticPairAugmentationProfile::Mild;
    }
    if (value == "medium") {
        return SyntheticPairAugmentationProfile::Medium;
    }
    if (value == "hard") {
        return SyntheticPairAugmentationProfile::Hard;
    }
    if (value == "extreme") {
        return SyntheticPairAugmentationProfile::Extreme;
    }
    if (value == "viewpoint" || value == "cross-view" || value == "cross_view") {
        return SyntheticPairAugmentationProfile::Viewpoint;
    }
    if (value == "compound-viewpoint" || value == "compound_viewpoint" || value == "rotation-viewpoint" ||
        value == "rotation_viewpoint") {
        return SyntheticPairAugmentationProfile::CompoundViewpoint;
    }
    throw std::invalid_argument("unsupported synthetic pair augmentation profile: " + value);
}

std::string synthetic_pair_augmentation_profile_name(SyntheticPairAugmentationProfile profile) {
    switch (profile) {
        case SyntheticPairAugmentationProfile::Mixed:
            return "mixed";
        case SyntheticPairAugmentationProfile::RotationOnly:
            return "rotation-only";
        case SyntheticPairAugmentationProfile::Mild:
            return "mild";
        case SyntheticPairAugmentationProfile::Medium:
            return "medium";
        case SyntheticPairAugmentationProfile::Hard:
            return "hard";
        case SyntheticPairAugmentationProfile::Extreme:
            return "extreme";
        case SyntheticPairAugmentationProfile::Viewpoint:
            return "viewpoint";
        case SyntheticPairAugmentationProfile::CompoundViewpoint:
            return "compound-viewpoint";
    }
    return "mixed";
}

SyntheticPair make_synthetic_pair(const torch::Tensor& image, const SyntheticPairConfig& config) {
    const auto sample = ImagePairAugmentor(toAugmentationConfig(config)).augment(image);
    return SyntheticPair{sample.view_a, sample.view_b, sample.warp_a_to_b, sample.valid_mask};
}

}  // namespace pfm
