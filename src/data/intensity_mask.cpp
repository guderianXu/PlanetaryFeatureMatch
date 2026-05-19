#include <cmath>
#include <stdexcept>

#include <torch/torch.h>

#include "core/tensor_utils.h"
#include "data/intensity_mask.h"

namespace pfm {

void validate_min_keypoint_intensity(double min_keypoint_intensity) {
    if (!std::isfinite(min_keypoint_intensity) || min_keypoint_intensity < 0.0 || min_keypoint_intensity > 1.0) {
        throw std::invalid_argument("min_keypoint_intensity must be between 0 and 1");
    }
}

torch::Tensor make_intensity_mask(const torch::Tensor& image, double min_keypoint_intensity) {
    validate_min_keypoint_intensity(min_keypoint_intensity);
    require_chw_image(image);
    const auto intensity = image.to(torch::kFloat32).mean(0).contiguous();
    return intensity.ge(min_keypoint_intensity).to(torch::kFloat32).contiguous();
}

}  // namespace pfm
