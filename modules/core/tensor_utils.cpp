#include "core/tensor_utils.h"

#include <stdexcept>

namespace pfm
{

void require_chw_image(const torch::Tensor& image)
{
    if (!image.defined())
    {
        throw std::invalid_argument("image tensor is undefined");
    }
    if (image.dim() != 3)
    {
        throw std::invalid_argument("image tensor must have shape CxHxW");
    }
    if (image.scalar_type() != torch::kFloat32)
    {
        throw std::invalid_argument("image tensor must be float32");
    }
    const auto c = image.size(0);
    if (c != 1 && c != 3)
    {
        throw std::invalid_argument("image tensor must have 1 or 3 channels");
    }
    if (image.size(1) <= 0 || image.size(2) <= 0)
    {
        throw std::invalid_argument("image tensor height and width must be positive");
    }
}

int64_t channels(const torch::Tensor& image)
{
    require_chw_image(image);
    return image.size(0);
}

int64_t height(const torch::Tensor& image)
{
    require_chw_image(image);
    return image.size(1);
}

int64_t width(const torch::Tensor& image)
{
    require_chw_image(image);
    return image.size(2);
}

torch::Tensor make_xy_grid(int64_t h, int64_t w, torch::Device device)
{
    if (h <= 0 || w <= 0)
    {
        throw std::invalid_argument("grid dimensions must be positive");
    }
    auto ys =
        torch::arange(h, torch::TensorOptions().dtype(torch::kFloat32).device(device)).view({h, 1}).repeat({1, w});
    auto xs =
        torch::arange(w, torch::TensorOptions().dtype(torch::kFloat32).device(device)).view({1, w}).repeat({h, 1});
    return torch::stack({xs, ys}, -1);
}

} // namespace pfm
