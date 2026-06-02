#include "image/normalization.h"

#include <stdexcept>

#include "core/tensor_utils.h"

namespace pfm
{

namespace
{

void require_scalar_type(const torch::Tensor& image, c10::ScalarType scalar_type, const char* message)
{
    if (!image.defined())
    {
        throw std::invalid_argument("image tensor is undefined");
    }
    if (image.scalar_type() != scalar_type)
    {
        throw std::invalid_argument(message);
    }
}

} // namespace

torch::Tensor normalize_u8(const torch::Tensor& image)
{
    require_scalar_type(image, torch::kUInt8, "image tensor must be uint8");
    auto output = image.to(torch::kFloat32) / 255.0F;
    require_chw_image(output);
    return output;
}

torch::Tensor normalize_u16(const torch::Tensor& image)
{
    require_scalar_type(image, torch::kUInt16, "image tensor must be uint16");
    auto output = image.to(torch::kFloat32) / 65535.0F;
    require_chw_image(output);
    return output;
}

torch::Tensor clamp_unit(const torch::Tensor& image)
{
    require_chw_image(image);
    return image.clamp(0.0F, 1.0F);
}

torch::Tensor local_contrast_normalize(const torch::Tensor& image, int64_t kernel_size)
{
    require_chw_image(image);
    if (kernel_size <= 0 || kernel_size % 2 == 0)
    {
        throw std::invalid_argument("kernel size must be positive and odd");
    }

    const auto padding = kernel_size / 2;
    auto batched = image.unsqueeze(0);
    auto options =
        torch::nn::functional::AvgPool2dFuncOptions(kernel_size).stride(1).padding(padding).count_include_pad(false);
    auto mean = torch::nn::functional::avg_pool2d(batched, options);
    auto centered = batched - mean;
    auto variance = torch::nn::functional::avg_pool2d(centered * centered, options);
    auto normalized = centered / torch::sqrt(variance + 1.0e-6F);
    return normalized.squeeze(0);
}

} // namespace pfm
