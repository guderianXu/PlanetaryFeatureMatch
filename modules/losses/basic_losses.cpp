#include "losses/basic_losses.h"

#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace pfm
{
namespace
{

void requireSameShape(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name)
{
    if (!lhs.sizes().equals(rhs.sizes()))
    {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must have the same shape");
    }
}

void requireSameDevice(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name)
{
    if (lhs.device() != rhs.device())
    {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must be on the same device");
    }
}

torch::Tensor expandMaskedRegressionMask(const torch::Tensor& mask, const torch::Tensor& prediction)
{
    if (mask.dim() == 0)
    {
        return mask.expand_as(prediction);
    }
    if (mask.sizes().equals(prediction.sizes()))
    {
        return mask;
    }
    if (prediction.dim() == 4 && mask.dim() == 4 && mask.size(0) == prediction.size(0) && mask.size(1) == 1 &&
        mask.size(2) == prediction.size(2) && mask.size(3) == prediction.size(3))
    {
        return mask.expand_as(prediction);
    }
    if (prediction.dim() == 4 && mask.dim() == 3)
    {
        throw std::invalid_argument("mask shape is ambiguous for BxCxHxW prediction");
    }
    throw std::invalid_argument("mask shape must match prediction, be scalar, or be Bx1xHxW for BxCxHxW prediction");
}

torch::Tensor expandScalarOrSameShape(const torch::Tensor& target, const torch::Tensor& reference,
                                      const char* target_name)
{
    if (target.dim() == 0)
    {
        return target.expand_as(reference);
    }
    if (target.sizes().equals(reference.sizes()))
    {
        return target;
    }
    throw std::invalid_argument(std::string(target_name) + " must be scalar or have the same shape as reference");
}

} // namespace

torch::Tensor masked_l1_loss(const torch::Tensor& prediction, const torch::Tensor& target, const torch::Tensor& mask)
{
    requireSameShape(prediction, target, "prediction", "target");
    requireSameDevice(prediction, mask, "prediction", "mask");
    auto mask_float = expandMaskedRegressionMask(mask.to(prediction.dtype()), prediction);
    auto denom = mask_float.sum();
    if (denom.item<double>() <= 0.0)
    {
        return torch::zeros({}, prediction.options());
    }
    return ((prediction - target).abs() * mask_float).sum() / denom;
}

torch::Tensor masked_smooth_l1_loss(const torch::Tensor& prediction, const torch::Tensor& target,
                                    const torch::Tensor& mask)
{
    requireSameShape(prediction, target, "prediction", "target");
    requireSameDevice(prediction, mask, "prediction", "mask");
    auto mask_float = expandMaskedRegressionMask(mask.to(prediction.dtype()), prediction);
    auto denom = mask_float.sum();
    if (denom.item<double>() <= 0.0)
    {
        return torch::zeros({}, prediction.options());
    }

    // beta=1 的 Smooth L1：小误差二次惩罚，大误差线性惩罚，兼顾稳定性和鲁棒性。
    auto diff = (prediction - target).abs();
    auto loss = torch::where(diff < 1.0, 0.5 * diff.pow(2), diff - 0.5);
    return (loss * mask_float).sum() / denom;
}

torch::Tensor confidence_bce_loss(const torch::Tensor& confidence, const torch::Tensor& target)
{
    requireSameDevice(confidence, target, "confidence", "target");
    auto expanded_target = expandScalarOrSameShape(target, confidence, "target");

    // 训练中置信度头可能短暂产生 NaN/Inf；这里用 0.5 作为中性概率保护 BCE。
    auto finite_confidence = torch::where(torch::isfinite(confidence), confidence, torch::full_like(confidence, 0.5));
    auto probabilities = finite_confidence.clamp(1.0e-6, 1.0 - 1.0e-6);
    return torch::binary_cross_entropy(probabilities, expanded_target.to(confidence.dtype()));
}

} // namespace pfm
