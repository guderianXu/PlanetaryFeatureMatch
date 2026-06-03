#pragma once

#include <torch/torch.h>

#include "losses/basic_losses.h"

namespace pfm
{

/// 计算两幅可重复性热力图在有效区域内的均方差。
///
/// @param heatmap_a 第一幅热力图张量。
/// @param heatmap_b 与 heatmap_a 同形状的第二幅热力图张量。
/// @param mask 可广播到 heatmap 形状的 bool 或数值 mask。
/// @return 标量损失；空 mask 返回 0。
/// @throws std::invalid_argument 当形状不兼容、mask 不能广播或 mask 权重为负时抛出。
torch::Tensor repeatability_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b,
                                 const torch::Tensor& mask);

/// 基于两组批量描述子计算匹配交叉熵。
///
/// @param descriptors_a BxNxD 查询描述子。
/// @param descriptors_b BxMxD 候选描述子。
/// @param target_indices BxN 的 long 标签，记录每个查询点匹配到的候选下标。
/// @return 描述子相似度上的标量交叉熵。
/// @throws std::invalid_argument 当描述子或标签的形状、dtype、设备或取值非法时抛出。
torch::Tensor descriptor_cross_entropy_loss(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                                            const torch::Tensor& target_indices);

/// 基于每个查询点独立候选集计算描述子匹配交叉熵。
///
/// @param descriptors_a BxNxD 查询描述子。
/// @param candidate_descriptors BxNxKxD 候选描述子，每个查询点有 K 个候选。
/// @param target_indices BxN 的 long 标签，记录每个查询点匹配到的候选下标。
/// @return 每个查询点候选相似度上的标量交叉熵。
/// @throws std::invalid_argument 当描述子或标签的形状、dtype、设备或取值非法时抛出。
torch::Tensor descriptor_candidate_cross_entropy_loss(const torch::Tensor& descriptors_a,
                                                      const torch::Tensor& candidate_descriptors,
                                                      const torch::Tensor& target_indices);

/// 惩罚同一图像内描述子之间的正余弦相似度，降低描述子坍缩风险。
///
/// @param descriptors BxNxD 的描述子样本。
/// @return 描述子数量少于 2 时返回 0，否则返回正成对相似度均值。
/// @throws std::invalid_argument 当 descriptors 不是 BxNxD 时抛出。
torch::Tensor descriptor_diversity_loss(const torch::Tensor& descriptors);

/// 计算图匹配交叉熵，最后一列作为未匹配 dustbin。
///
/// @param logits (Na+1)x(Nb+1) 匹配 logits，包含 dustbin 行和列。
/// @param target_indices Na 维 long 标签，取值范围 [0, Nb]，其中 Nb 表示 dustbin。
/// @return 所有源关键点上的标量交叉熵。
/// @throws std::invalid_argument 当 logits 或标签的形状、dtype、设备或标签范围非法时抛出。
torch::Tensor graph_matching_cross_entropy_loss(const torch::Tensor& logits, const torch::Tensor& target_indices);

} // namespace pfm
