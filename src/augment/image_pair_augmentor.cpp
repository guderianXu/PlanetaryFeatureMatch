#include "augment/image_pair_augmentor.h"

#include <algorithm>
#include <cmath>

#include <ATen/Functions.h>

#include "augment/transform_sampler.h"
#include "core/tensor_utils.h"
#include "data/normalization.h"
#include "geometry/warp.h"

namespace pfm {
namespace {

constexpr float PI = 3.14159265358979323846F;

AffineTransform makePairTransform(const torch::Tensor& image, const ImagePairTransformParameters& params) {
    const auto cx = static_cast<float>(width(image) - 1) * 0.5F;
    const auto cy = static_cast<float>(height(image) - 1) * 0.5F;
    auto transform = AffineTransform::scale_rotate(params.scale, params.rotation_degrees * PI / 180.0F, cx, cy);
    transform.matrix[2] += params.translation_x;
    transform.matrix[5] += params.translation_y;
    return transform;
}

torch::Tensor affineWarpChw(const torch::Tensor& image, const AffineTransform& transform) {
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

torch::Tensor makeDeterministicNoiseLike(const torch::Tensor& image) {
    const auto grid = make_xy_grid(height(image), width(image), image.device());
    using torch::indexing::Slice;

    const auto xs = grid.index({Slice(), Slice(), 0});
    const auto ys = grid.index({Slice(), Slice(), 1});
    const auto hashed = torch::sin(xs * 12.9898F + ys * 78.233F) * 43758.5453F;
    const auto unit_noise = (hashed - torch::floor(hashed)) * 2.0F - 1.0F;
    return unit_noise.unsqueeze(0).expand_as(image);
}

void applyPhotometric(const ImagePairTransformParameters& params, torch::Tensor& view_b) {
    if (std::abs(params.gamma - 1.0F) > 1.0e-6F) {
        view_b = torch::pow(torch::clamp(view_b, 1.0e-4F, 1.0F), params.gamma);
    }

    if (params.shadow_strength <= 0.0F) {
        return;
    }

    const auto grid = make_xy_grid(height(view_b), width(view_b), view_b.device());
    using torch::indexing::Slice;
    const auto xs = grid.index({Slice(), Slice(), 0}) / static_cast<float>(std::max<int64_t>(1, width(view_b) - 1));
    const auto ys = grid.index({Slice(), Slice(), 1}) / static_cast<float>(std::max<int64_t>(1, height(view_b) - 1));
    const auto shadow = 1.0F - params.shadow_strength * torch::clamp(xs * 0.65F + ys * 0.35F, 0.0F, 1.0F);
    view_b = view_b * shadow.unsqueeze(0).expand_as(view_b);
}

}  // namespace

ImagePairAugmentor::ImagePairAugmentor(ImagePairAugmentationConfig config) : _config(config) {}

ImagePairSample ImagePairAugmentor::augment(const torch::Tensor& image) const {
    require_chw_image(image);
    const auto params = sampleImagePairTransform(_config);

    auto view_a = clamp_unit(image.clone());
    auto transform = makePairTransform(image, params);
    auto view_b = affineWarpChw(image, transform);
    applyPhotometric(params, view_b);
    view_b = clamp_unit(view_b * params.contrast_scale + params.brightness_delta);
    if (params.noise_sigma > 0.0F) {
        view_b = clamp_unit(view_b + makeDeterministicNoiseLike(view_b) * params.noise_sigma);
    }

    auto field = dense_warp_field(height(image), width(image), transform, image.device());
    auto mask = valid_warp_mask(field, height(image), width(image));

    return ImagePairSample{view_a, view_b, field, mask};
}

}  // namespace pfm
