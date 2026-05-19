#pragma once

#include <torch/torch.h>

namespace pfm {

struct DenseHeadOutput {
    torch::Tensor confidence;
    torch::Tensor offsets;
};

class DenseHeadImpl : public torch::nn::Module {
public:
    /// Creates a dense pairwise coarse matching head.
    /// @param feature_channels Number of channels in each input feature tensor.
    /// @throws std::invalid_argument if feature_channels is not positive.
    explicit DenseHeadImpl(int64_t feature_channels);

    /// Predicts coarse confidence and x/y offsets from two feature maps.
    /// @param feature_a First feature tensor with shape BxCxHxW.
    /// @param feature_b Second feature tensor with shape BxCxHxW.
    /// @return Confidence tensor Bx1xHxW and offsets tensor Bx2xHxW.
    /// @throws std::invalid_argument if inputs are not 4D, have mismatched shapes, or channel count does not match.
    DenseHeadOutput forward(const torch::Tensor& feature_a, const torch::Tensor& feature_b);

private:
    int64_t _feature_channels;
    torch::nn::Conv2d _correlation_projection{nullptr};
    torch::nn::Sequential _predictor{nullptr};
};

TORCH_MODULE(DenseHead);

}  // namespace pfm
