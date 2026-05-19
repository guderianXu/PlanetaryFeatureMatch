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

/// Computes descriptor matching cross entropy from per-query candidate descriptors.
///
/// \param descriptors_a Tensor shaped BxNxD containing query descriptors.
/// \param candidate_descriptors Tensor shaped BxNxKxD containing candidates for each query.
/// \param target_indices Long tensor shaped BxN containing the matching candidate index for each query.
/// \return Scalar tensor containing cross entropy over per-query descriptor similarities.
/// \throws std::invalid_argument if descriptors or labels have incompatible shape, dtype, device, or values.
torch::Tensor descriptor_candidate_cross_entropy_loss(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& candidate_descriptors,
    const torch::Tensor& target_indices);

/// Penalizes descriptor collapse by discouraging positive cosine similarity among descriptors from the same image.
///
/// \param descriptors Tensor shaped BxNxD containing descriptor samples.
/// \return Scalar tensor containing zero for fewer than two descriptors, otherwise mean positive pairwise similarity.
/// \throws std::invalid_argument if descriptors is not BxNxD.
torch::Tensor descriptor_diversity_loss(const torch::Tensor& descriptors);

/// Computes graph matching cross entropy with the final column used as the unmatched dustbin.
///
/// \param logits Match logits shaped (Na+1)x(Nb+1), including dustbin row and column.
/// \param target_indices Long tensor shaped Na, with values in [0, Nb] where Nb is the dustbin label.
/// \return Scalar tensor containing cross entropy over all source keypoints.
/// \throws std::invalid_argument if logits or target_indices have invalid shape, dtype, device, or label range.
torch::Tensor graph_matching_cross_entropy_loss(
    const torch::Tensor& logits,
    const torch::Tensor& target_indices);

/// Computes masked mean L1 loss.
///
/// \param prediction Predicted tensor.
/// \param target Target tensor with the same shape as prediction.
/// \param mask Boolean or numeric mask shaped as prediction, scalar, or Bx1xHxW for BxCxHxW prediction.
/// \return Scalar tensor containing zero for an empty mask, otherwise the masked L1 average.
/// \throws std::invalid_argument if prediction and target shapes differ or the mask shape is unsupported.
torch::Tensor masked_l1_loss(const torch::Tensor& prediction, const torch::Tensor& target, const torch::Tensor& mask);

/// Computes masked mean Smooth L1 loss with beta equal to 1.
///
/// \param prediction Predicted tensor.
/// \param target Target tensor with the same shape as prediction.
/// \param mask Boolean or numeric mask shaped as prediction, scalar, or Bx1xHxW for BxCxHxW prediction.
/// \return Scalar tensor containing zero for an empty mask, otherwise the masked Smooth L1 average.
/// \throws std::invalid_argument if prediction and target shapes differ or the mask shape is unsupported.
torch::Tensor masked_smooth_l1_loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mask);

/// Computes binary cross entropy for confidence predictions.
///
/// \param confidence Confidence prediction tensor containing probabilities.
/// \param target Target tensor with the same shape as confidence, or a scalar target expanded to confidence.
/// \return Scalar tensor containing binary cross entropy.
/// \throws std::invalid_argument if target shape is unsupported or the target is on a different device.
torch::Tensor confidence_bce_loss(const torch::Tensor& confidence, const torch::Tensor& target);

}  // namespace pfm
