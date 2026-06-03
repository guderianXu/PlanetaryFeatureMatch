#include "augment/image_pair_augmentor.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <ATen/Functions.h>

#include "augment/transform_sampler.h"
#include "core/tensor_utils.h"
#include "geometry/warp.h"
#include "image/normalization.h"

namespace pfm
{
namespace
{

constexpr float PI = 3.14159265358979323846F;

AffineTransform makePairTransform(const torch::Tensor& image, const ImagePairTransformParameters& params)
{
    // 几何增强以图像中心为旋转/缩放中心，再叠加平移，避免尺度变化把主体直接移出画面。
    const auto cx = static_cast<float>(width(image) - 1) * 0.5F;
    const auto cy = static_cast<float>(height(image) - 1) * 0.5F;
    auto transform = AffineTransform::scale_rotate(params.scale, params.rotation_degrees * PI / 180.0F, cx, cy);
    transform.matrix[2] += params.translation_x;
    transform.matrix[5] += params.translation_y;
    return transform;
}

ProjectiveTransform makeProjectivePairTransform(const torch::Tensor& image, const ImagePairTransformParameters& params)
{
    // viewpoint/compound-viewpoint 档位需要剪切和轻量透视项，因此在仿射矩阵外再组合 3x3 投影矩阵。
    const auto affine = makePairTransform(image, params);
    const auto cx = static_cast<float>(width(image) - 1) * 0.5F;
    const auto cy = static_cast<float>(height(image) - 1) * 0.5F;

    const std::array<float, 9> shear{{
        1.0F,
        params.shear_x,
        -params.shear_x * cy,
        params.shear_y,
        1.0F,
        -params.shear_y * cx,
        0.0F,
        0.0F,
        1.0F,
    }};
    const std::array<float, 9> a{{
        affine.matrix[0],
        affine.matrix[1],
        affine.matrix[2],
        affine.matrix[3],
        affine.matrix[4],
        affine.matrix[5],
        0.0F,
        0.0F,
        1.0F,
    }};
    std::array<float, 9> composed{};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            composed[static_cast<size_t>(row * 3 + col)] =
                shear[static_cast<size_t>(row * 3 + 0)] * a[static_cast<size_t>(0 * 3 + col)] +
                shear[static_cast<size_t>(row * 3 + 1)] * a[static_cast<size_t>(1 * 3 + col)] +
                shear[static_cast<size_t>(row * 3 + 2)] * a[static_cast<size_t>(2 * 3 + col)];
        }
    }
    composed[6] = params.perspective_x;
    composed[7] = params.perspective_y;
    composed[8] = 1.0F - params.perspective_x * cx - params.perspective_y * cy;
    return ProjectiveTransform{composed};
}

torch::Tensor projectiveWarpChw(const torch::Tensor& image, const ProjectiveTransform& transform)
{
    // grid_sampler 使用目标到源的反向采样坐标，因此先对投影矩阵求逆。
    const auto h = height(image);
    const auto w = width(image);
    const auto inverse = transform.inverse();
    const auto output_grid = make_xy_grid(h, w, image.device());
    const auto input_grid = warp_points(output_grid.reshape({h * w, 2}), inverse).reshape({h, w, 2});

    using torch::indexing::Slice;
    const auto input_x = input_grid.index({Slice(), Slice(), 0});
    const auto input_y = input_grid.index({Slice(), Slice(), 1});
    const auto norm_x = input_x / static_cast<float>(std::max<int64_t>(1, w - 1)) * 2.0F - 1.0F;
    const auto norm_y = input_y / static_cast<float>(std::max<int64_t>(1, h - 1)) * 2.0F - 1.0F;
    const auto sampler_grid = torch::stack({norm_x, norm_y}, 2).unsqueeze(0).contiguous();
    return at::grid_sampler(image.unsqueeze(0), sampler_grid, 0, 0, true).squeeze(0).contiguous();
}

torch::Tensor makeDeterministicNoiseLike(const torch::Tensor& image)
{
    // 使用坐标哈希噪声代替随机数，保证相同 source/variant 每次生成完全一致。
    const auto grid = make_xy_grid(height(image), width(image), image.device());
    using torch::indexing::Slice;

    const auto xs = grid.index({Slice(), Slice(), 0});
    const auto ys = grid.index({Slice(), Slice(), 1});
    const auto hashed = torch::sin(xs * 12.9898F + ys * 78.233F) * 43758.5453F;
    const auto unit_noise = (hashed - torch::floor(hashed)) * 2.0F - 1.0F;
    return unit_noise.unsqueeze(0).expand_as(image);
}

void applyPhotometric(const ImagePairTransformParameters& params, torch::Tensor& view_b)
{
    // 光度扰动只作用于 B 图，让训练目标同时学习几何对应和亮度差异鲁棒性。
    if (std::abs(params.gamma - 1.0F) > 1.0e-6F)
    {
        view_b = torch::pow(torch::clamp(view_b, 1.0e-4F, 1.0F), params.gamma);
    }

    if (params.shadow_strength <= 0.0F)
    {
        return;
    }

    const auto grid = make_xy_grid(height(view_b), width(view_b), view_b.device());
    using torch::indexing::Slice;
    const auto xs = grid.index({Slice(), Slice(), 0}) / static_cast<float>(std::max<int64_t>(1, width(view_b) - 1));
    const auto ys = grid.index({Slice(), Slice(), 1}) / static_cast<float>(std::max<int64_t>(1, height(view_b) - 1));
    const auto shadow = 1.0F - params.shadow_strength * torch::clamp(xs * 0.65F + ys * 0.35F, 0.0F, 1.0F);
    view_b = view_b * shadow.unsqueeze(0).expand_as(view_b);
}

} // namespace

ImagePairAugmentor::ImagePairAugmentor(ImagePairAugmentationConfig config) : _config(config)
{
}

ImagePairSample ImagePairAugmentor::augment(const torch::Tensor& image) const
{
    require_chw_image(image);
    const auto params = sampleImagePairTransform(_config);

    // view_a 保持原图，view_b 经过几何、光度和噪声扰动；warp 字段记录 A 像素在 B 中的位置。
    auto view_a = clamp_unit(image.clone());
    auto transform = makeProjectivePairTransform(image, params);
    auto view_b = projectiveWarpChw(image, transform);
    applyPhotometric(params, view_b);
    view_b = clamp_unit(view_b * params.contrast_scale + params.brightness_delta);
    if (params.noise_sigma > 0.0F)
    {
        view_b = clamp_unit(view_b + makeDeterministicNoiseLike(view_b) * params.noise_sigma);
    }

    auto field = dense_warp_field(height(image), width(image), transform, image.device());
    const auto finite_field = torch::isfinite(field);
    field = torch::where(finite_field, field, torch::full_like(field, -1.0e6F));
    auto mask = valid_warp_mask(field, height(image), width(image)).logical_and(finite_field.all(2));

    return ImagePairSample{view_a, view_b, field, mask};
}

} // namespace pfm
