#include "data/synthetic_pair.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/tensor_utils.h"
#include "data/normalization.h"
#include "geometry/warp.h"

namespace pfm {
namespace {

torch::Tensor translate_chw_without_wrap(const torch::Tensor& image, float translation_x, float translation_y) {
    using torch::indexing::Slice;

    const auto shift_x = static_cast<int64_t>(std::lround(translation_x));
    const auto shift_y = static_cast<int64_t>(std::lround(translation_y));
    auto translated = torch::zeros_like(image);

    const auto h = height(image);
    const auto w = width(image);
    const auto src_x0 = std::max<int64_t>(0, -shift_x);
    const auto src_y0 = std::max<int64_t>(0, -shift_y);
    const auto src_x1 = std::min<int64_t>(w, w - shift_x);
    const auto src_y1 = std::min<int64_t>(h, h - shift_y);
    if (src_x0 >= src_x1 || src_y0 >= src_y1) {
        return translated;
    }

    const auto dst_x0 = src_x0 + shift_x;
    const auto dst_y0 = src_y0 + shift_y;
    const auto dst_x1 = src_x1 + shift_x;
    const auto dst_y1 = src_y1 + shift_y;
    translated.index_put_({Slice(), Slice(dst_y0, dst_y1), Slice(dst_x0, dst_x1)},
                          image.index({Slice(), Slice(src_y0, src_y1), Slice(src_x0, src_x1)}));
    return translated;
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

void validate_config(const SyntheticPairConfig& config) {
    constexpr float integer_tolerance = 1.0e-6F;

    if (std::abs(config.translation_x - std::round(config.translation_x)) > integer_tolerance) {
        throw std::invalid_argument("translation_x must be an integer translation");
    }
    if (std::abs(config.translation_y - std::round(config.translation_y)) > integer_tolerance) {
        throw std::invalid_argument("translation_y must be an integer translation");
    }
    if (config.contrast_scale <= 0.0F) {
        throw std::invalid_argument("contrast_scale must be positive");
    }
    if (config.noise_sigma < 0.0F) {
        throw std::invalid_argument("noise_sigma must be non-negative");
    }
}

}  // namespace

SyntheticPair make_synthetic_pair(const torch::Tensor& image, const SyntheticPairConfig& config) {
    require_chw_image(image);
    validate_config(config);

    auto view_a = clamp_unit(image.clone());
    auto view_b = translate_chw_without_wrap(image, config.translation_x, config.translation_y);
    view_b = clamp_unit(view_b * config.contrast_scale + config.brightness_delta);
    if (config.noise_sigma > 0.0F) {
        view_b = clamp_unit(view_b + make_deterministic_noise_like(view_b) * config.noise_sigma);
    }

    auto transform = AffineTransform::translation(config.translation_x, config.translation_y);
    auto field = dense_warp_field(height(image), width(image), transform, image.device());
    auto mask = valid_warp_mask(field, height(image), width(image));

    return SyntheticPair{view_a, view_b, field, mask};
}

}  // namespace pfm
