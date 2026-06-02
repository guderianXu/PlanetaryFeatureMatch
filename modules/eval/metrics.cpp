#include "eval/metrics.h"

#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace pfm
{
namespace
{

void requireDefined(const torch::Tensor& tensor, const char* name)
{
    if (!tensor.defined())
    {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
}

void requireFloating(const torch::Tensor& tensor, const char* name)
{
    const auto dtype = tensor.scalar_type();
    if (dtype != torch::kFloat32 && dtype != torch::kFloat64)
    {
        throw std::invalid_argument(std::string(name) + " must have floating dtype");
    }
}

void requireSameDevice(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name)
{
    if (lhs.device() != rhs.device())
    {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must be on the same device");
    }
}

void requireSameShape(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name)
{
    if (!lhs.sizes().equals(rhs.sizes()))
    {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must have the same shape");
    }
}

void requirePointPairs(const torch::Tensor& points, const char* name)
{
    if (points.dim() != 2 || points.size(1) != 2)
    {
        throw std::invalid_argument(std::string(name) + " must have shape Nx2");
    }
}

void requirePositiveImageSize(int64_t image_width, int64_t image_height)
{
    if (image_width <= 0 || image_height <= 0)
    {
        throw std::invalid_argument("image width and height must be positive");
    }
}

void requireMatchingPointPairs(const torch::Tensor& points_a, const torch::Tensor& points_b)
{
    requireDefined(points_a, "points_a");
    requireDefined(points_b, "points_b");
    requirePointPairs(points_a, "points_a");
    requirePointPairs(points_b, "points_b");
    requireSameShape(points_a, points_b, "points_a", "points_b");
    requireSameDevice(points_a, points_b, "points_a", "points_b");
    requireFloating(points_a, "points_a");
    requireFloating(points_b, "points_b");
}

torch::Tensor halfTurnExpectedB(const torch::Tensor& points_a, int64_t image_width, int64_t image_height)
{
    auto center = torch::tensor({static_cast<double>(image_width - 1), static_cast<double>(image_height - 1)},
                                points_a.options().dtype(torch::kFloat64));
    center = center.to(points_a.scalar_type());
    return center - points_a;
}

} // namespace

float matching_precision(const torch::Tensor& points_a, const torch::Tensor& predicted_b,
                         const torch::Tensor& expected_b, float threshold_pixels)
{
    requireDefined(predicted_b, "predicted_b");
    requireDefined(expected_b, "expected_b");
    requirePointPairs(predicted_b, "predicted_b");
    requirePointPairs(expected_b, "expected_b");
    requireSameShape(predicted_b, expected_b, "predicted_b", "expected_b");
    requireSameDevice(predicted_b, expected_b, "predicted_b", "expected_b");
    requireFloating(predicted_b, "predicted_b");
    requireFloating(expected_b, "expected_b");
    if (points_a.defined())
    {
        requirePointPairs(points_a, "points_a");
        requireSameShape(points_a, predicted_b, "points_a", "predicted_b");
        requireSameDevice(points_a, predicted_b, "points_a", "predicted_b");
        requireFloating(points_a, "points_a");
    }
    if (predicted_b.size(0) == 0)
    {
        return 0.0F;
    }

    auto distances = (predicted_b - expected_b).pow(2).sum(1).sqrt();
    return distances.le(threshold_pixels).to(torch::kFloat32).mean().item<float>();
}

float semi_dense_coverage(const torch::Tensor& confidence, const torch::Tensor& valid_mask, float threshold)
{
    requireDefined(confidence, "confidence");
    requireDefined(valid_mask, "valid_mask");
    requireSameShape(confidence, valid_mask, "confidence", "valid_mask");
    requireSameDevice(confidence, valid_mask, "confidence", "valid_mask");
    requireFloating(confidence, "confidence");
    auto valid = valid_mask.to(torch::kBool);
    auto denominator = valid.sum().item<float>();
    if (denominator <= 0.0F)
    {
        return 0.0F;
    }

    auto selected = confidence.ge(threshold).logical_and(valid);
    return selected.sum().item<float>() / denominator;
}

float half_turn_consistency(const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t image_width,
                            int64_t image_height, float threshold_pixels)
{
    requirePositiveImageSize(image_width, image_height);
    requireMatchingPointPairs(points_a, points_b);
    if (points_a.size(0) == 0)
    {
        return 0.0F;
    }

    auto expected_b = halfTurnExpectedB(points_a, image_width, image_height);
    auto distances = (points_b - expected_b).pow(2).sum(1).sqrt();
    return distances.le(threshold_pixels).to(torch::kFloat32).mean().item<float>();
}

float half_turn_mean_error(const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t image_width,
                           int64_t image_height)
{
    requirePositiveImageSize(image_width, image_height);
    requireMatchingPointPairs(points_a, points_b);
    if (points_a.size(0) == 0)
    {
        return 0.0F;
    }

    auto expected_b = halfTurnExpectedB(points_a, image_width, image_height);
    return (points_b - expected_b).pow(2).sum(1).sqrt().mean().item<float>();
}

} // namespace pfm
