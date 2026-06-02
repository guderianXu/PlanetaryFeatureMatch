#include "image/intensity_mask.h"

#include <cmath>
#include <stdexcept>

#include <torch/torch.h>

#include "core/tensor_utils.h"

namespace pfm
{

void validate_min_keypoint_intensity(double min_keypoint_intensity)
{
    if (!std::isfinite(min_keypoint_intensity) || min_keypoint_intensity < 0.0 || min_keypoint_intensity > 1.0)
    {
        throw std::invalid_argument("min_keypoint_intensity must be between 0 and 1");
    }
}

torch::Tensor make_intensity_mask(const torch::Tensor& image, double min_keypoint_intensity)
{
    validate_min_keypoint_intensity(min_keypoint_intensity);
    require_chw_image(image);
    const auto intensity = image.to(torch::kFloat32).mean(0).contiguous();
    auto bright = intensity.ge(min_keypoint_intensity);
    if (min_keypoint_intensity <= 0.0 || intensity.size(0) < 7 || intensity.size(1) < 7)
    {
        return bright.to(torch::kFloat32).contiguous();
    }

    // 行星影像里常见孤立亮点和噪声峰。局部支持比例与局部均值双重约束可以保留成片亮区，
    // 同时避免把单个噪声像素当作稳定关键点区域。
    const int64_t kernel = 7;
    auto local_support = torch::nn::functional::avg_pool2d(
        bright.to(torch::kFloat32).reshape({1, 1, intensity.size(0), intensity.size(1)}),
        torch::nn::functional::AvgPool2dFuncOptions({kernel, kernel})
            .stride(1)
            .padding(kernel / 2)
            .count_include_pad(false));
    auto local_mean = torch::nn::functional::avg_pool2d(intensity.reshape({1, 1, intensity.size(0), intensity.size(1)}),
                                                        torch::nn::functional::AvgPool2dFuncOptions({kernel, kernel})
                                                            .stride(1)
                                                            .padding(kernel / 2)
                                                            .count_include_pad(false));
    local_support = local_support.reshape({intensity.size(0), intensity.size(1)}).ge(0.25);
    local_mean = local_mean.reshape({intensity.size(0), intensity.size(1)}).ge(min_keypoint_intensity);
    return bright.logical_and(local_support).logical_and(local_mean).to(torch::kFloat32).contiguous();
}

} // namespace pfm
