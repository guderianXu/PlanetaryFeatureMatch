#pragma once

#include <torch/torch.h>

namespace pfm {

/// Computes precision for predicted correspondences within a pixel threshold.
///
/// \param points_a Optional tensor shaped Nx2 for source points; if defined, it must match predicted_b shape,
///     device, and floating dtype.
/// \param predicted_b Defined floating tensor shaped Nx2 containing predicted target points.
/// \param expected_b Defined floating tensor shaped Nx2 containing expected target points.
/// \param threshold_pixels Maximum Euclidean pixel distance counted as a correct match.
/// \return Fraction of predicted target points within threshold, or zero for empty predictions.
/// \throws std::invalid_argument if required tensors are undefined, shapes are incompatible, dtypes are invalid,
///     or devices differ.
float matching_precision(
    const torch::Tensor& points_a,
    const torch::Tensor& predicted_b,
    const torch::Tensor& expected_b,
    float threshold_pixels);

/// Computes semi-dense coverage over the valid mask area.
///
/// \param confidence Defined floating confidence tensor.
/// \param valid_mask Defined boolean or numeric mask with the same shape and device as confidence.
/// \param threshold Confidence threshold for selecting valid pixels.
/// \return Fraction of valid pixels with confidence at or above threshold, or zero for an empty valid mask.
/// \throws std::invalid_argument if required tensors are undefined, shapes are incompatible, confidence dtype is
///     invalid, or devices differ.
float semi_dense_coverage(const torch::Tensor& confidence, const torch::Tensor& valid_mask, float threshold);

/// Computes the fraction of matches consistent with a 180-degree rotation around the image center.
///
/// The expected target for source point (x, y) is (image_width - 1 - x, image_height - 1 - y).
/// Returns zero for empty predictions.
/// \throws std::invalid_argument if tensors are invalid or image size is non-positive.
float half_turn_consistency(
    const torch::Tensor& points_a,
    const torch::Tensor& points_b,
    int64_t image_width,
    int64_t image_height,
    float threshold_pixels);

/// Computes mean Euclidean pixel error from the 180-degree rotated target, or zero for empty predictions.
///
/// \throws std::invalid_argument if tensors are invalid or image size is non-positive.
float half_turn_mean_error(
    const torch::Tensor& points_a,
    const torch::Tensor& points_b,
    int64_t image_width,
    int64_t image_height);

}  // namespace pfm
