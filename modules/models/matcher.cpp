#include "models/matcher.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace pfm {
namespace {

torch::Tensor normalize_descriptors(const torch::Tensor& tensor) {
    return tensor / tensor.norm(2, 2, true).clamp_min(1.0e-12);
}

}  // namespace

MatcherImpl::MatcherImpl(int64_t descriptor_dim) : _descriptor_dim(descriptor_dim) {
    if (_descriptor_dim <= 0) {
        throw std::invalid_argument("descriptor_dim must be positive");
    }
}

torch::Tensor MatcherImpl::forward(const torch::Tensor& desc_a, const torch::Tensor& desc_b) {
    if (!desc_a.defined() || !desc_b.defined()) {
        throw std::invalid_argument("descriptor tensors must be defined");
    }
    if (desc_a.dim() != 3 || desc_b.dim() != 3) {
        throw std::invalid_argument("descriptor tensors must have shapes BxNxD and BxMxD");
    }
    if (desc_a.size(0) != desc_b.size(0)) {
        throw std::invalid_argument("descriptor tensors must have matching batch sizes");
    }
    if (desc_a.size(2) != _descriptor_dim || desc_b.size(2) != _descriptor_dim) {
        throw std::invalid_argument("descriptor tensor dimensions do not match matcher");
    }

    auto normalized_a = normalize_descriptors(desc_a);
    auto normalized_b = normalize_descriptors(desc_b);
    return torch::bmm(normalized_a, normalized_b.transpose(1, 2)) / std::sqrt(static_cast<double>(_descriptor_dim));
}

}  // namespace pfm
