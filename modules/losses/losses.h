#pragma once

#include <torch/torch.h>

namespace pfm {

/// Computes masked mean squared difference between two repeatability heatmaps.
///
/// \param heatmap_a First heatmap tensor.
/// \param heatmap_b Second heatmap tensor with the same shape as heatmap_a.
/// \param mask Boolean or numeric mask broadcastable to the heatmap shape.
/// \return Scalar tensor containing zero for an empty mask, otherwise the weighted repeatability loss.
/// \throws std::invalid_argument if shapes differ, the mask cannot broadcast, or mask weights are negative.
torch::Tensor repeatability_loss(
    const torch::Tensor& heatmap_a,
    const torch::Tensor& heatmap_b,
    const torch::Tensor& mask);

/// Computes descriptor matching cross entropy from batched descriptor sets.
///
/// \param descriptors_a Tensor shaped BxNxD containing query descriptors.
/// \param descriptors_b Tensor shaped BxMxD containing candidate descriptors.
/// \param target_indices Long tensor shaped BxN containing the matching candidate index for each query.
/// \return Scalar tensor containing cross entropy over descriptor similarities.
/// \throws std::invalid_argument if descriptors or labels have incompatible shape, dtype, device, or values.
torch::Tensor descriptor_cross_entropy_loss(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& target_indices);

/// Computes masked mean L1 loss.
///
/// \param prediction Predicted tensor.
/// \param target Target tensor with the same shape as prediction.
/// \param mask Boolean or numeric mask shaped as prediction, scalar, or Bx1xHxW for BxCxHxW prediction.
/// \return Scalar tensor containing zero for an empty mask, otherwise the masked L1 average.
/// \throws std::invalid_argument if prediction and target shapes differ or the mask shape is unsupported.
torch::Tensor masked_l1_loss(const torch::Tensor& prediction, const torch::Tensor& target, const torch::Tensor& mask);

/// Computes binary cross entropy for confidence predictions.
///
/// \param confidence Confidence prediction tensor containing probabilities.
/// \param target Target tensor with the same shape as confidence, or a scalar target expanded to confidence.
/// \return Scalar tensor containing binary cross entropy.
/// \throws std::invalid_argument if target shape is unsupported or the target is on a different device.
torch::Tensor confidence_bce_loss(const torch::Tensor& confidence, const torch::Tensor& target);

}  // namespace pfm
