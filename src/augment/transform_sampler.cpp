#include "augment/transform_sampler.h"

#include <cmath>
#include <stdexcept>

namespace pfm {
namespace {

struct ProfileStrength {
    float translation = 0.0F;
    float rotation = 0.0F;
    float scale = 0.0F;
    float brightness = 0.0F;
    float contrast = 0.0F;
    float noise = 0.0F;
};

float deterministicWave(int64_t source_index, int64_t variant_index, float frequency, float phase) {
    return std::sin(static_cast<float>(source_index + 1) * phase + static_cast<float>(variant_index + 1) * frequency);
}

AugmentationProfile mixedProfileForVariant(const ImagePairAugmentationConfig& config) {
    if (config.extreme_pair_ratio > 0.0 && config.variant_index % 4 == 1) {
        return AugmentationProfile::Extreme;
    }
    switch (config.variant_index % 3) {
        case 0:
            return AugmentationProfile::Mild;
        case 1:
            return AugmentationProfile::Hard;
        default:
            return AugmentationProfile::Medium;
    }
}

ProfileStrength profileStrength(AugmentationProfile profile) {
    switch (profile) {
        case AugmentationProfile::Mild:
            return ProfileStrength{3.0F, 6.0F, 0.04F, 0.03F, 0.05F, 0.004F};
        case AugmentationProfile::Medium:
            return ProfileStrength{7.0F, 16.0F, 0.10F, 0.07F, 0.12F, 0.010F};
        case AugmentationProfile::Hard:
            return ProfileStrength{12.0F, 32.0F, 0.18F, 0.12F, 0.22F, 0.018F};
        case AugmentationProfile::Extreme:
            return ProfileStrength{18.0F, 55.0F, 0.30F, 0.18F, 0.35F, 0.030F};
        case AugmentationProfile::Mixed:
            return profileStrength(AugmentationProfile::Medium);
    }
    return profileStrength(AugmentationProfile::Medium);
}

bool usesProfileAugmentation(const ImagePairAugmentationConfig& config) {
    return config.source_index != 0 || config.variant_index != 0 || config.profile != AugmentationProfile::Mixed;
}

AugmentationProfile resolvedProfile(const ImagePairAugmentationConfig& config) {
    return config.profile == AugmentationProfile::Mixed ? mixedProfileForVariant(config) : config.profile;
}

void validateConfig(const ImagePairAugmentationConfig& config) {
    constexpr float integer_tolerance = 1.0e-6F;

    if (std::abs(config.translation_x - std::round(config.translation_x)) > integer_tolerance) {
        throw std::invalid_argument("translation_x must be an integer translation");
    }
    if (std::abs(config.translation_y - std::round(config.translation_y)) > integer_tolerance) {
        throw std::invalid_argument("translation_y must be an integer translation");
    }
    if (config.scale <= 0.0F) {
        throw std::invalid_argument("scale must be positive");
    }
    if (config.contrast_scale <= 0.0F) {
        throw std::invalid_argument("contrast_scale must be positive");
    }
    if (config.noise_sigma < 0.0F) {
        throw std::invalid_argument("noise_sigma must be non-negative");
    }
    if (config.extreme_pair_ratio < 0.0 || config.extreme_pair_ratio > 1.0) {
        throw std::invalid_argument("extreme_pair_ratio must be between 0 and 1");
    }
}

}  // namespace

ImagePairTransformParameters sampleImagePairTransform(const ImagePairAugmentationConfig& config) {
    validateConfig(config);

    ImagePairTransformParameters params;
    params.translation_x = config.translation_x;
    params.translation_y = config.translation_y;
    params.rotation_degrees = config.rotation_degrees;
    params.scale = config.scale;
    params.brightness_delta = config.brightness_delta;
    params.contrast_scale = config.contrast_scale;
    params.noise_sigma = config.noise_sigma;

    if (!usesProfileAugmentation(config)) {
        return params;
    }

    const auto profile = resolvedProfile(config);
    const auto strength = profileStrength(profile);
    const auto source = config.source_index;
    const auto index = config.variant_index;
    const bool mixed_quarter_turn = config.profile == AugmentationProfile::Mixed && config.variant_index % 8 == 3;
    const bool mixed_half_turn = config.profile == AugmentationProfile::Mixed && config.variant_index % 8 == 7;

    if (mixed_quarter_turn) {
        params.rotation_degrees += deterministicWave(source, index, 0.23F, 0.61F) >= 0.0F ? 90.0F : -90.0F;
        params.gamma = 1.0F;
        params.shadow_strength = 0.0F;
        return params;
    }
    if (mixed_half_turn) {
        params.rotation_degrees += deterministicWave(source, index, 0.19F, 0.43F) >= 0.0F ? 180.0F : -180.0F;
        params.gamma = 1.0F;
        params.shadow_strength = 0.0F;
        return params;
    }

    params.translation_x += std::round(deterministicWave(source, index, 1.37F, 0.71F) * strength.translation);
    params.translation_y += std::round(deterministicWave(source, index, 1.91F, 1.13F) * strength.translation);
    params.rotation_degrees += deterministicWave(source, index, 0.73F, 1.53F) * strength.rotation;
    params.scale *= 1.0F + deterministicWave(source, index, 0.41F, 0.37F) * strength.scale;
    params.brightness_delta += deterministicWave(source, index, 1.11F, 0.83F) * strength.brightness;
    params.contrast_scale *= 1.0F + deterministicWave(source, index, 0.97F, 1.31F) * strength.contrast;
    params.noise_sigma += std::abs(deterministicWave(source, index, 1.63F, 0.59F)) * strength.noise;
    params.gamma = 1.0F + deterministicWave(source, index, 0.67F, 1.79F) * 0.35F;
    params.shadow_strength = profile == AugmentationProfile::Extreme ? 0.30F : 0.12F;
    return params;
}

}  // namespace pfm
