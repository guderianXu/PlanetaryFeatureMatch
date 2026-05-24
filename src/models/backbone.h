#pragma once

#include <torch/torch.h>

#include <vector>

namespace pfm {

class BackboneImpl : public torch::nn::Module {
public:
    /// Creates a four-stage convolutional backbone.
    /// @param input_channels Number of channels in the input NCHW image tensor.
    /// @param base_channels Number of output channels in the first stage.
    /// @throws std::invalid_argument if input_channels or base_channels is not positive.
    BackboneImpl(int64_t input_channels, int64_t base_channels);

    /// Computes four stride-2 feature maps.
    /// @param x Input tensor with shape BxCxHxW.
    /// @return Feature maps at strides 2, 4, 8, and 16 with channel multipliers 1, 2, 4, and 8.
    /// @throws std::invalid_argument if x is not a 4D tensor or its channel count does not match the constructor.
    std::vector<torch::Tensor> forward(const torch::Tensor& x);

    /// Replaces non-finite normalization buffers so legacy checkpoints remain usable in eval mode.
    void sanitize_nonfinite_state();

private:
    int64_t _input_channels;
    int64_t _base_channels;
    torch::nn::Sequential _stage1{nullptr};
    torch::nn::Sequential _stage2{nullptr};
    torch::nn::Sequential _stage3{nullptr};
    torch::nn::Sequential _stage4{nullptr};
};

TORCH_MODULE(Backbone);

}  // namespace pfm
