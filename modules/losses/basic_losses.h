#pragma once

#include <torch/torch.h>

namespace pfm
{

/// 计算带 mask 的平均 L1 loss。
///
/// mask 可以与 prediction 同形状、为标量，或在 BxCxHxW prediction 上使用 Bx1xHxW 形状。
/// 空 mask 返回 0，便于训练中跳过没有有效监督的样本。
///
/// @param prediction 预测张量。
/// @param target 与 prediction 同形状的目标张量。
/// @param mask bool 或数值 mask。
/// @return 标量 loss 张量。
/// @throws std::invalid_argument 当 prediction/target 形状不同、设备不一致或 mask 形状不支持时抛出。
torch::Tensor masked_l1_loss(const torch::Tensor& prediction, const torch::Tensor& target, const torch::Tensor& mask);

/// 计算带 mask 的平均 Smooth L1 loss，beta 固定为 1。
///
/// 适合 dense offset、局部位移等像素回归任务。空 mask 返回 0。
///
/// @param prediction 预测张量。
/// @param target 与 prediction 同形状的目标张量。
/// @param mask bool 或数值 mask。
/// @return 标量 loss 张量。
/// @throws std::invalid_argument 当 prediction/target 形状不同、设备不一致或 mask 形状不支持时抛出。
torch::Tensor masked_smooth_l1_loss(const torch::Tensor& prediction, const torch::Tensor& target,
                                    const torch::Tensor& mask);

/// 计算置信度概率的 binary cross entropy。
///
/// confidence 会先替换非有限值并裁剪到合法概率范围，避免训练日志中出现 NaN loss。
/// target 可以是同形状张量，也可以是标量并自动扩展。
///
/// @param confidence 概率形式的置信度预测。
/// @param target 同形状目标张量或标量目标。
/// @return 标量 BCE loss。
/// @throws std::invalid_argument 当 target 形状不支持或设备不一致时抛出。
torch::Tensor confidence_bce_loss(const torch::Tensor& confidence, const torch::Tensor& target);

} // namespace pfm
