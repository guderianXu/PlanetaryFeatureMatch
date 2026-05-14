#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace pfm {

void require_chw_image(const torch::Tensor& image);
int64_t channels(const torch::Tensor& image);
int64_t height(const torch::Tensor& image);
int64_t width(const torch::Tensor& image);
torch::Tensor make_xy_grid(int64_t height, int64_t width, torch::Device device);

}  // namespace pfm
