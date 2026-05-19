#pragma once

#include <torch/torch.h>

namespace pfm {

torch::Tensor normalize_u8(const torch::Tensor& image);
torch::Tensor normalize_u16(const torch::Tensor& image);
torch::Tensor clamp_unit(const torch::Tensor& image);
torch::Tensor local_contrast_normalize(const torch::Tensor& image, int64_t kernel_size);

}  // namespace pfm
