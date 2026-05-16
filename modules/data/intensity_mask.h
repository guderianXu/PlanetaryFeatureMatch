#pragma once

#include <torch/torch.h>

namespace pfm {

/// Validates a normalized minimum keypoint intensity threshold.
/// @param min_keypoint_intensity Threshold in normalized image intensity units.
/// @throws std::invalid_argument if the threshold is non-finite or outside [0, 1].
void validate_min_keypoint_intensity(double min_keypoint_intensity);

/// Builds an H x W float mask from a C x H x W image tensor using channel-mean intensity.
/// @param image Float image tensor in C x H x W layout.
/// @param min_keypoint_intensity Pixels below this normalized threshold become 0.
/// @return H x W float mask on the same device as image, with values 0 or 1.
/// @throws std::invalid_argument if the image layout or threshold is invalid.
torch::Tensor make_intensity_mask(const torch::Tensor& image, double min_keypoint_intensity);

}  // namespace pfm
