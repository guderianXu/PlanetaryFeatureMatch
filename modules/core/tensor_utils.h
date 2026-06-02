#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace pfm
{

/// 校验图像张量是否为 C x H x W 布局的浮点图像。
/// @param image 待校验的图像张量。
/// @throws std::invalid_argument 当张量未定义、维度不为 3、通道数非法或空间尺寸为空时抛出。
void require_chw_image(const torch::Tensor& image);

/// 返回 C x H x W 图像张量的通道数。
int64_t channels(const torch::Tensor& image);

/// 返回 C x H x W 图像张量的高度。
int64_t height(const torch::Tensor& image);

/// 返回 C x H x W 图像张量的宽度。
int64_t width(const torch::Tensor& image);

/// 生成 H x W x 2 的像素坐标网格，最后一维为 (x, y)。
/// @param height 网格高度，必须为正。
/// @param width 网格宽度，必须为正。
/// @param device 输出张量所在设备。
/// @return float32 坐标网格。
torch::Tensor make_xy_grid(int64_t height, int64_t width, torch::Device device);

} // namespace pfm
