#include "losses/losses.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace pfm
{
namespace
{

constexpr float DESCRIPTOR_LOGIT_SCALE = 20.0F;

// descriptor 相关交叉熵使用同一 logit scale，避免相似度过平导致梯度太弱。
void requireSameShape(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name)
{
    if (!lhs.sizes().equals(rhs.sizes()))
    {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must have the same shape");
    }
}

torch::Tensor expandMaskToShape(const torch::Tensor& mask, const torch::Tensor& reference)
{
    // 允许 HxW、1xHxW 等 mask 广播到 heatmap，便于训练端复用不同粒度的有效区域。
    if (mask.dim() > reference.dim())
    {
        throw std::invalid_argument("mask cannot have more dimensions than the reference tensor");
    }

    std::vector<int64_t> shape(reference.dim(), 1);
    const auto offset = reference.dim() - mask.dim();
    for (int64_t i = 0; i < mask.dim(); ++i)
    {
        shape[offset + i] = mask.size(i);
    }

    auto reshaped = mask.reshape(shape);
    try
    {
        return reshaped.expand_as(reference);
    }
    catch (const c10::Error&)
    {
        throw std::invalid_argument("mask shape is not broadcastable to the reference tensor");
    }
}

void requireSameDevice(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name)
{
    if (lhs.device() != rhs.device())
    {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must be on the same device");
    }
}

void requireSameDtype(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name)
{
    if (lhs.dtype() != rhs.dtype())
    {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must have the same dtype");
    }
}

void requireNoNegativeWeights(const torch::Tensor& weights, const char* name)
{
    if (weights.numel() > 0 && weights.lt(0).any().item<bool>())
    {
        throw std::invalid_argument(std::string(name) + " cannot contain negative weights");
    }
}

torch::Tensor normalizeDescriptors(const torch::Tensor& descriptors)
{
    auto norm = descriptors.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    return descriptors / norm;
}

torch::Tensor cyclicDescriptorLogits(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b)
{
    auto normalized_a = normalizeDescriptors(descriptors_a);
    auto normalized_b = normalizeDescriptors(descriptors_b);
    return torch::bmm(normalized_a, normalized_b.transpose(1, 2));
}

torch::Tensor strictCandidateDescriptorLogits(const torch::Tensor& descriptors_a,
                                              const torch::Tensor& candidate_descriptors)
{
    // 独立候选集已经由几何采样给出，不再做循环滚动，避免把硬负样本误当旋转等价正样本。
    auto normalized_a = normalizeDescriptors(descriptors_a);
    auto normalized_candidates =
        candidate_descriptors / candidate_descriptors.pow(2).sum(3, true).clamp_min(1.0e-12).sqrt();
    return (normalized_a.unsqueeze(2) * normalized_candidates).sum(3);
}

} // namespace

torch::Tensor repeatability_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b,
                                 const torch::Tensor& mask)
{
    requireSameShape(heatmap_a, heatmap_b, "heatmap_a", "heatmap_b");
    requireSameDevice(heatmap_a, mask, "heatmap_a", "mask");
    auto mask_float = expandMaskToShape(mask.to(heatmap_a.dtype()), heatmap_a);
    requireNoNegativeWeights(mask_float, "mask");
    auto denom = mask_float.sum();
    if (denom.item<double>() <= 0.0)
    {
        // 空有效区域不产生梯度，避免无效样本污染 detector loss。
        return torch::zeros({}, heatmap_a.options());
    }
    auto diff = heatmap_a - heatmap_b;
    return (diff.pow(2) * mask_float).sum() / denom;
}

torch::Tensor descriptor_cross_entropy_loss(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                                            const torch::Tensor& target_indices)
{
    if (descriptors_a.dim() != 3 || descriptors_b.dim() != 3)
    {
        throw std::invalid_argument("descriptor tensors must have shape BxNxD and BxMxD");
    }
    if (descriptors_a.size(0) != descriptors_b.size(0) || descriptors_a.size(2) != descriptors_b.size(2))
    {
        throw std::invalid_argument("descriptor tensors must share batch size and descriptor dimension");
    }
    requireSameDtype(descriptors_a, descriptors_b, "descriptors_a", "descriptors_b");
    requireSameDevice(descriptors_a, descriptors_b, "descriptors_a", "descriptors_b");
    requireSameDevice(descriptors_a, target_indices, "descriptors_a", "target_indices");
    if (target_indices.dtype() != torch::kLong)
    {
        throw std::invalid_argument("target_indices must have dtype torch::kLong");
    }
    if (target_indices.dim() != 2 || target_indices.size(0) != descriptors_a.size(0) ||
        target_indices.size(1) != descriptors_a.size(1))
    {
        throw std::invalid_argument("target_indices must have shape BxN");
    }

    const auto candidate_count = descriptors_b.size(1);
    if (target_indices.numel() > 0 && target_indices.lt(0).any().item<bool>())
    {
        throw std::invalid_argument("target_indices cannot contain negative labels");
    }
    if (target_indices.numel() > 0 && target_indices.ge(candidate_count).any().item<bool>())
    {
        throw std::invalid_argument("target_indices labels must be less than descriptor candidate count");
    }

    auto logits = cyclicDescriptorLogits(descriptors_a, descriptors_b) * DESCRIPTOR_LOGIT_SCALE;
    return torch::nn::functional::cross_entropy(logits.reshape({-1, candidate_count}), target_indices.reshape({-1}));
}

torch::Tensor descriptor_candidate_cross_entropy_loss(const torch::Tensor& descriptors_a,
                                                      const torch::Tensor& candidate_descriptors,
                                                      const torch::Tensor& target_indices)
{
    if (descriptors_a.dim() != 3 || candidate_descriptors.dim() != 4)
    {
        throw std::invalid_argument("descriptor tensors must have shape BxNxD and BxNxKxD");
    }
    if (descriptors_a.size(0) != candidate_descriptors.size(0) ||
        descriptors_a.size(1) != candidate_descriptors.size(1) ||
        descriptors_a.size(2) != candidate_descriptors.size(3))
    {
        throw std::invalid_argument("candidate descriptors must share batch, query count, and descriptor dimension");
    }
    requireSameDtype(descriptors_a, candidate_descriptors, "descriptors_a", "candidate_descriptors");
    requireSameDevice(descriptors_a, candidate_descriptors, "descriptors_a", "candidate_descriptors");
    requireSameDevice(descriptors_a, target_indices, "descriptors_a", "target_indices");
    if (target_indices.dtype() != torch::kLong)
    {
        throw std::invalid_argument("target_indices must have dtype torch::kLong");
    }
    if (target_indices.dim() != 2 || target_indices.size(0) != descriptors_a.size(0) ||
        target_indices.size(1) != descriptors_a.size(1))
    {
        throw std::invalid_argument("target_indices must have shape BxN");
    }

    const auto candidate_count = candidate_descriptors.size(2);
    if (target_indices.numel() > 0 && target_indices.lt(0).any().item<bool>())
    {
        throw std::invalid_argument("target_indices cannot contain negative labels");
    }
    if (target_indices.numel() > 0 && target_indices.ge(candidate_count).any().item<bool>())
    {
        throw std::invalid_argument("target_indices labels must be less than descriptor candidate count");
    }

    auto logits = strictCandidateDescriptorLogits(descriptors_a, candidate_descriptors) * DESCRIPTOR_LOGIT_SCALE;
    return torch::nn::functional::cross_entropy(logits.reshape({-1, candidate_count}), target_indices.reshape({-1}));
}

torch::Tensor descriptor_diversity_loss(const torch::Tensor& descriptors)
{
    if (descriptors.dim() != 3)
    {
        throw std::invalid_argument("descriptors must have shape BxNxD");
    }
    const auto descriptor_count = descriptors.size(1);
    if (descriptor_count < 2)
    {
        return torch::zeros({}, descriptors.options());
    }

    auto normalized = normalizeDescriptors(descriptors);
    auto similarity = torch::bmm(normalized, normalized.transpose(1, 2));
    // 去掉对角线，只惩罚不同位置之间的相似度，抑制整图描述子坍缩。
    auto eye = torch::eye(descriptor_count, descriptors.options()).unsqueeze(0).to(torch::kBool);
    auto off_diagonal = similarity.masked_select(eye.logical_not());
    return off_diagonal.pow(2).mean();
}

torch::Tensor graph_matching_cross_entropy_loss(const torch::Tensor& logits, const torch::Tensor& target_indices)
{
    if (!logits.defined() || logits.dim() != 2 || logits.size(0) < 1 || logits.size(1) < 1)
    {
        throw std::invalid_argument("graph matching logits must have shape (Na+1)x(Nb+1)");
    }
    if (!target_indices.defined() || target_indices.dtype() != torch::kLong || target_indices.dim() != 1)
    {
        throw std::invalid_argument("graph matching target_indices must be a Long tensor shaped Na");
    }
    requireSameDevice(logits, target_indices, "logits", "target_indices");
    const auto source_count = logits.size(0) - 1;
    const auto candidate_count = logits.size(1);
    if (target_indices.size(0) != source_count)
    {
        throw std::invalid_argument("graph matching target_indices must contain one label per source keypoint");
    }
    if (source_count == 0)
    {
        return torch::zeros({}, logits.options());
    }
    if (target_indices.numel() > 0 && target_indices.lt(0).any().item<bool>())
    {
        throw std::invalid_argument("graph matching target_indices cannot contain negative labels");
    }
    if (target_indices.numel() > 0 && target_indices.ge(candidate_count).any().item<bool>())
    {
        throw std::invalid_argument("graph matching target_indices labels must be less than Nb+1");
    }
    return torch::nn::functional::cross_entropy(logits.narrow(0, 0, source_count), target_indices);
}

} // namespace pfm
