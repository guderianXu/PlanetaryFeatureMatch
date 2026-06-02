#pragma once

#include <torch/torch.h>

namespace pfm
{

/// 校验归一化亮度阈值是否可用于关键点筛选。
/// @param min_keypoint_intensity 归一化图像亮度阈值，合法范围为 [0, 1]。
/// @throws std::invalid_argument 当阈值非有限数或越界时抛出。
void validate_min_keypoint_intensity(double min_keypoint_intensity);

/// 根据 C x H x W 图像的通道均值亮度生成 H x W float mask。
///
/// 该 mask 用于抑制过暗区域中的关键点候选。阈值大于 0 且图像足够大时，会额外检查局部
/// 7x7 邻域支持，避免孤立亮点噪声被误当作稳定纹理。
///
/// @param image C x H x W 布局的浮点图像张量。
/// @param min_keypoint_intensity 低于该归一化亮度阈值的像素置 0。
/// @return 与输入同设备的 H x W float mask，取值为 0 或 1。
/// @throws std::invalid_argument 当图像布局或阈值非法时抛出。
torch::Tensor make_intensity_mask(const torch::Tensor& image, double min_keypoint_intensity);

} // namespace pfm
