#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "eval/metrics.h"

namespace pfm {
namespace {

void requireDefined(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
}

void requireFloating(const torch::Tensor& tensor, const char* name) {
    const auto dtype = tensor.scalar_type();
    if (dtype != torch::kFloat32 && dtype != torch::kFloat64) {
        throw std::invalid_argument(std::string(name) + " must have floating dtype");
    }
}

void requireSameDevice(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name) {
    if (lhs.device() != rhs.device()) {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must be on the same device");
    }
}

void requireSameShape(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name) {
    if (!lhs.sizes().equals(rhs.sizes())) {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must have the same shape");
    }
}

void requirePointPairs(const torch::Tensor& points, const char* name) {
    if (points.dim() != 2 || points.size(1) != 2) {
        throw std::invalid_argument(std::string(name) + " must have shape Nx2");
    }
}

}  // namespace

float matching_precision(
    const torch::Tensor& points_a,
    const torch::Tensor& predicted_b,
    const torch::Tensor& expected_b,
    float threshold_pixels) {
    requireDefined(predicted_b, "predicted_b");
    requireDefined(expected_b, "expected_b");
    requirePointPairs(predicted_b, "predicted_b");
    requirePointPairs(expected_b, "expected_b");
    requireSameShape(predicted_b, expected_b, "predicted_b", "expected_b");
    requireSameDevice(predicted_b, expected_b, "predicted_b", "expected_b");
    requireFloating(predicted_b, "predicted_b");
    requireFloating(expected_b, "expected_b");
    if (points_a.defined()) {
        requirePointPairs(points_a, "points_a");
        requireSameShape(points_a, predicted_b, "points_a", "predicted_b");
        requireSameDevice(points_a, predicted_b, "points_a", "predicted_b");
        requireFloating(points_a, "points_a");
    }
    if (predicted_b.size(0) == 0) {
        return 0.0F;
    }

    auto distances = (predicted_b - expected_b).pow(2).sum(1).sqrt();
    return distances.le(threshold_pixels).to(torch::kFloat32).mean().item<float>();
}

float semi_dense_coverage(const torch::Tensor& confidence, const torch::Tensor& valid_mask, float threshold) {
    requireDefined(confidence, "confidence");
    requireDefined(valid_mask, "valid_mask");
    requireSameShape(confidence, valid_mask, "confidence", "valid_mask");
    requireSameDevice(confidence, valid_mask, "confidence", "valid_mask");
    requireFloating(confidence, "confidence");
    auto valid = valid_mask.to(torch::kBool);
    auto denominator = valid.sum().item<float>();
    if (denominator <= 0.0F) {
        return 0.0F;
    }

    auto selected = confidence.ge(threshold).logical_and(valid);
    return selected.sum().item<float>() / denominator;
}

}  // namespace pfm
