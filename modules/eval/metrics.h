#pragma once

#include <torch/torch.h>

namespace pfm
{

/// 计算预测匹配点在像素阈值内的精度。
///
/// \param points_a 可选 Nx2 源点张量；若已定义，必须与 predicted_b 形状、设备和浮点 dtype 一致。
/// \param predicted_b 已定义的 Nx2 浮点张量，表示预测目标点。
/// \param expected_b 已定义的 Nx2 浮点张量，表示真值目标点。
/// \param threshold_pixels 判定为正确匹配的最大欧氏像素距离。
/// \return 阈值内预测点比例；没有预测点时返回 0。
/// \throws std::invalid_argument 当必要张量未定义、形状不兼容、dtype 非法或设备不一致时抛出。
float matching_precision(const torch::Tensor& points_a, const torch::Tensor& predicted_b,
                         const torch::Tensor& expected_b, float threshold_pixels);

/// 计算有效区域内的半稠密置信覆盖率。
///
/// \param confidence 已定义的浮点置信度张量。
/// \param valid_mask 与 confidence 形状和设备一致的 bool 或数值 mask。
/// \param threshold 选中像素所需的置信度阈值。
/// \return 有效 mask 中置信度不低于阈值的像素比例；有效区域为空时返回 0。
/// \throws std::invalid_argument 当必要张量未定义、形状不兼容、confidence dtype 非法或设备不一致时抛出。
float semi_dense_coverage(const torch::Tensor& confidence, const torch::Tensor& valid_mask, float threshold);

/// 计算匹配点中符合绕图像中心旋转 180 度关系的比例。
///
/// 源点 (x, y) 的期望目标点为 (image_width - 1 - x, image_height - 1 - y)。
/// 没有预测点时返回 0。
/// \throws std::invalid_argument 当张量非法或图像尺寸非正时抛出。
float half_turn_consistency(const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t image_width,
                            int64_t image_height, float threshold_pixels);

/// 计算相对 180 度旋转真值的平均欧氏像素误差；没有预测点时返回 0。
///
/// \throws std::invalid_argument 当张量非法或图像尺寸非正时抛出。
float half_turn_mean_error(const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t image_width,
                           int64_t image_height);

} // namespace pfm
