#pragma once

#include <torch/torch.h>

namespace pfm
{

/// 将 uint8 的 C x H x W 图像转换为 float32，并按 255 归一化。
/// @throws std::invalid_argument 当输入未定义、dtype 非 uint8 或布局非法时抛出。
torch::Tensor normalize_u8(const torch::Tensor& image);

/// 将 uint16 的 C x H x W 图像转换为 float32，并按 65535 归一化。
/// @throws std::invalid_argument 当输入未定义、dtype 非 uint16 或布局非法时抛出。
torch::Tensor normalize_u16(const torch::Tensor& image);

/// 将 C x H x W 浮点图像裁剪到 [0, 1]，用于保护后续模型输入值域。
torch::Tensor clamp_unit(const torch::Tensor& image);

/// 对 C x H x W 图像执行局部对比归一化。
///
/// 该函数使用滑窗均值和方差抑制全局光照差异，适合火星/月球影像中阴影和低纹理区域的预处理。
/// @param image C x H x W 浮点图像。
/// @param kernel_size 奇数滑窗大小，必须大于 0。
/// @return 与输入同形状的归一化图像。
/// @throws std::invalid_argument 当输入布局非法或 kernel_size 非正/非奇数时抛出。
torch::Tensor local_contrast_normalize(const torch::Tensor& image, int64_t kernel_size);

} // namespace pfm
