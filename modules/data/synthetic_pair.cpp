#include "data/synthetic_pair.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <ATen/Functions.h>

#include "core/tensor_utils.h"
#include "data/normalization.h"
#include "geometry/warp.h"

namespace pfm {
namespace {

constexpr float PI = 3.14159265358979323846F;

struct ProfileStrength {
    float translation = 0.0F;
    float rotation = 0.0F;
    float scale = 0.0F;
    float brightness = 0.0F;
    float contrast = 0.0F;
    float noise = 0.0F;
};

float deterministic_wave(int64_t source_index, int64_t variant_index, float frequency, float phase) {
    return std::sin(static_cast<float>(source_index + 1) * phase + static_cast<float>(variant_index + 1) * frequency);
}

SyntheticPairAugmentationProfile mixed_profile_for_variant(const SyntheticPairConfig& config) {
    if (config.extreme_pair_ratio > 0.0 && config.variant_index % 4 == 1) {
        return SyntheticPairAugmentationProfile::Extreme;
    }
    switch (config.variant_index % 3) {
        case 0:
            return SyntheticPairAugmentationProfile::Mild;
        case 1:
            return SyntheticPairAugmentationProfile::Hard;
        default:
            return SyntheticPairAugmentationProfile::Medium;
    }
}

ProfileStrength profile_strength(SyntheticPairAugmentationProfile profile) {
    switch (profile) {
        case SyntheticPairAugmentationProfile::Mild:
            return ProfileStrength{3.0F, 6.0F, 0.04F, 0.03F, 0.05F, 0.004F};
        case SyntheticPairAugmentationProfile::Medium:
            return ProfileStrength{7.0F, 16.0F, 0.10F, 0.07F, 0.12F, 0.010F};
        case SyntheticPairAugmentationProfile::Hard:
            return ProfileStrength{12.0F, 32.0F, 0.18F, 0.12F, 0.22F, 0.018F};
        case SyntheticPairAugmentationProfile::Extreme:
            return ProfileStrength{18.0F, 55.0F, 0.30F, 0.18F, 0.35F, 0.030F};
        case SyntheticPairAugmentationProfile::Mixed:
            return profile_strength(SyntheticPairAugmentationProfile::Medium);
    }
    return profile_strength(SyntheticPairAugmentationProfile::Medium);
}

bool uses_profile_augmentation(const SyntheticPairConfig& config) {
    return config.source_index != 0 || config.variant_index != 0 ||
           config.augmentation_profile != SyntheticPairAugmentationProfile::Mixed;
}

SyntheticPairConfig resolve_variant_config(const SyntheticPairConfig& config) {
    auto variant = config;
    if (!uses_profile_augmentation(config)) {
        return variant;
    }

    const auto profile = config.augmentation_profile == SyntheticPairAugmentationProfile::Mixed
                             ? mixed_profile_for_variant(config)
                             : config.augmentation_profile;
    const auto strength = profile_strength(profile);
    const auto source = config.source_index;
    const auto index = config.variant_index;

    variant.translation_x += std::round(deterministic_wave(source, index, 1.37F, 0.71F) * strength.translation);
    variant.translation_y += std::round(deterministic_wave(source, index, 1.91F, 1.13F) * strength.translation);
    variant.rotation_degrees += deterministic_wave(source, index, 0.73F, 1.53F) * strength.rotation;
    variant.scale *= 1.0F + deterministic_wave(source, index, 0.41F, 0.37F) * strength.scale;
    variant.brightness_delta += deterministic_wave(source, index, 1.11F, 0.83F) * strength.brightness;
    variant.contrast_scale *= 1.0F + deterministic_wave(source, index, 0.97F, 1.31F) * strength.contrast;
    variant.noise_sigma += std::abs(deterministic_wave(source, index, 1.63F, 0.59F)) * strength.noise;
    return variant;
}

AffineTransform make_pair_transform(const torch::Tensor& image, const SyntheticPairConfig& config) {
    const auto cx = static_cast<float>(width(image) - 1) * 0.5F;
    const auto cy = static_cast<float>(height(image) - 1) * 0.5F;
    const auto variant = resolve_variant_config(config);
    auto transform = AffineTransform::scale_rotate(variant.scale, variant.rotation_degrees * PI / 180.0F, cx, cy);
    transform.matrix[2] += variant.translation_x;
    transform.matrix[5] += variant.translation_y;
    return transform;
}

torch::Tensor affine_warp_chw(const torch::Tensor& image, const AffineTransform& transform) {
    const auto h = height(image);
    const auto w = width(image);
    const auto options = image.options();
    auto theta = torch::empty({1, 2, 3}, options);
    theta.index_put_({0, 0, 0}, transform.matrix[0]);
    theta.index_put_({0, 0, 1}, transform.matrix[1]);
    theta.index_put_({0, 0, 2}, transform.matrix[2]);
    theta.index_put_({0, 1, 0}, transform.matrix[3]);
    theta.index_put_({0, 1, 1}, transform.matrix[4]);
    theta.index_put_({0, 1, 2}, transform.matrix[5]);

    const auto determinant = transform.matrix[0] * transform.matrix[4] - transform.matrix[1] * transform.matrix[3];
    auto output_to_input = theta.clone();
    output_to_input.index_put_({0, 0, 0}, transform.matrix[4] / determinant);
    output_to_input.index_put_({0, 0, 1}, -transform.matrix[1] / determinant);
    output_to_input.index_put_({0, 1, 0}, -transform.matrix[3] / determinant);
    output_to_input.index_put_({0, 1, 1}, transform.matrix[0] / determinant);
    output_to_input.index_put_(
        {0, 0, 2},
        (transform.matrix[1] * transform.matrix[5] - transform.matrix[4] * transform.matrix[2]) / determinant);
    output_to_input.index_put_(
        {0, 1, 2},
        (transform.matrix[3] * transform.matrix[2] - transform.matrix[0] * transform.matrix[5]) / determinant);

    auto normalized = output_to_input.clone();
    normalized.index_put_({0, 0, 0}, output_to_input.index({0, 0, 0}));
    normalized.index_put_({0, 0, 1}, output_to_input.index({0, 0, 1}) * static_cast<float>(h - 1) / static_cast<float>(w - 1));
    normalized.index_put_({0, 0, 2}, (output_to_input.index({0, 0, 2}) * 2.0F + output_to_input.index({0, 0, 0}) * static_cast<float>(w - 1) + output_to_input.index({0, 0, 1}) * static_cast<float>(h - 1) - static_cast<float>(w - 1)) / static_cast<float>(w - 1));
    normalized.index_put_({0, 1, 0}, output_to_input.index({0, 1, 0}) * static_cast<float>(w - 1) / static_cast<float>(h - 1));
    normalized.index_put_({0, 1, 1}, output_to_input.index({0, 1, 1}));
    normalized.index_put_({0, 1, 2}, (output_to_input.index({0, 1, 2}) * 2.0F + output_to_input.index({0, 1, 0}) * static_cast<float>(w - 1) + output_to_input.index({0, 1, 1}) * static_cast<float>(h - 1) - static_cast<float>(h - 1)) / static_cast<float>(h - 1));

    auto grid = at::affine_grid_generator(normalized, std::vector<int64_t>{1, image.size(0), h, w}, true);
    return at::grid_sampler(image.unsqueeze(0), grid, 0, 0, true).squeeze(0).contiguous();
}

torch::Tensor make_deterministic_noise_like(const torch::Tensor& image) {
    const auto grid = make_xy_grid(height(image), width(image), image.device());
    using torch::indexing::Slice;

    const auto xs = grid.index({Slice(), Slice(), 0});
    const auto ys = grid.index({Slice(), Slice(), 1});
    const auto hashed = torch::sin(xs * 12.9898F + ys * 78.233F) * 43758.5453F;
    const auto unit_noise = (hashed - torch::floor(hashed)) * 2.0F - 1.0F;
    return unit_noise.unsqueeze(0).expand_as(image);
}

float profile_gamma(const SyntheticPairConfig& config) {
    if (!uses_profile_augmentation(config)) {
        return 1.0F;
    }
    return 1.0F + deterministic_wave(config.source_index, config.variant_index, 0.67F, 1.79F) * 0.35F;
}

float shadow_strength(const SyntheticPairConfig& config) {
    if (!uses_profile_augmentation(config)) {
        return 0.0F;
    }
    const auto profile = config.augmentation_profile == SyntheticPairAugmentationProfile::Mixed
                             ? mixed_profile_for_variant(config)
                             : config.augmentation_profile;
    return profile == SyntheticPairAugmentationProfile::Extreme ? 0.30F : 0.12F;
}

void apply_profile_photometric(const SyntheticPairConfig& config, torch::Tensor& view_b) {
    const auto gamma = profile_gamma(config);
    if (std::abs(gamma - 1.0F) > 1.0e-6F) {
        view_b = torch::pow(torch::clamp(view_b, 1.0e-4F, 1.0F), gamma);
    }

    const auto strength = shadow_strength(config);
    if (strength <= 0.0F) {
        return;
    }

    const auto grid = make_xy_grid(height(view_b), width(view_b), view_b.device());
    using torch::indexing::Slice;
    const auto xs = grid.index({Slice(), Slice(), 0}) / static_cast<float>(std::max<int64_t>(1, width(view_b) - 1));
    const auto ys = grid.index({Slice(), Slice(), 1}) / static_cast<float>(std::max<int64_t>(1, height(view_b) - 1));
    const auto shadow = 1.0F - strength * torch::clamp(xs * 0.65F + ys * 0.35F, 0.0F, 1.0F);
    view_b = view_b * shadow.unsqueeze(0).expand_as(view_b);
}

void validate_config(const SyntheticPairConfig& config) {
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

SyntheticPairAugmentationProfile parse_synthetic_pair_augmentation_profile(const std::string& value) {
    if (value == "mixed") {
        return SyntheticPairAugmentationProfile::Mixed;
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
    throw std::invalid_argument("unsupported synthetic pair augmentation profile: " + value);
}

std::string synthetic_pair_augmentation_profile_name(SyntheticPairAugmentationProfile profile) {
    switch (profile) {
        case SyntheticPairAugmentationProfile::Mixed:
            return "mixed";
        case SyntheticPairAugmentationProfile::Mild:
            return "mild";
        case SyntheticPairAugmentationProfile::Medium:
            return "medium";
        case SyntheticPairAugmentationProfile::Hard:
            return "hard";
        case SyntheticPairAugmentationProfile::Extreme:
            return "extreme";
    }
    return "mixed";
}

SyntheticPair make_synthetic_pair(const torch::Tensor& image, const SyntheticPairConfig& config) {
    require_chw_image(image);
    validate_config(config);

    const auto variant = resolve_variant_config(config);
    auto view_a = clamp_unit(image.clone());
    auto transform = make_pair_transform(image, config);
    auto view_b = affine_warp_chw(image, transform);
    apply_profile_photometric(config, view_b);
    view_b = clamp_unit(view_b * variant.contrast_scale + variant.brightness_delta);
    if (variant.noise_sigma > 0.0F) {
        view_b = clamp_unit(view_b + make_deterministic_noise_like(view_b) * variant.noise_sigma);
    }

    auto field = dense_warp_field(height(image), width(image), transform, image.device());
    auto mask = valid_warp_mask(field, height(image), width(image));

    return SyntheticPair{view_a, view_b, field, mask};
}

}  // namespace pfm
