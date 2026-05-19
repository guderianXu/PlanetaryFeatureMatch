#pragma once

#include <torch/torch.h>

namespace pfm {

class MatcherImpl : public torch::nn::Module {
public:
    /// Creates a batched descriptor matcher.
    /// @param descriptor_dim Expected descriptor dimension D for BxNxD and BxMxD inputs.
    /// @throws std::invalid_argument if descriptor_dim is not positive.
    explicit MatcherImpl(int64_t descriptor_dim);

    /// Computes normalized descriptor similarity scores.
    /// @param desc_a Descriptor tensor with shape BxNxD.
    /// @param desc_b Descriptor tensor with shape BxMxD.
    /// @return Score matrix with shape BxNxM scaled by sqrt(D).
    /// @throws std::invalid_argument if descriptors are not 3D, batch sizes differ, or descriptor dimensions mismatch.
    torch::Tensor forward(const torch::Tensor& desc_a, const torch::Tensor& desc_b);

private:
    int64_t _descriptor_dim;
};

TORCH_MODULE(Matcher);

}  // namespace pfm
