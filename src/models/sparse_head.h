#pragma once

#include <torch/torch.h>

namespace pfm {

struct SparseHeadOutput {
    torch::Tensor heatmap;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
};

class SparseHeadImpl : public torch::nn::Module {
public:
    /// Creates sparse prediction heads from a shared feature tensor.
    /// @param input_channels Number of channels in the input NCHW feature tensor.
    /// @param descriptor_dim Number of output descriptor channels.
    /// @throws std::invalid_argument if input_channels or descriptor_dim is not positive.
    SparseHeadImpl(int64_t input_channels, int64_t descriptor_dim);

    /// Predicts sparse keypoint maps and normalized descriptors.
    /// @param feature Input feature tensor with shape BxCxHxW.
    /// @return Heatmap, descriptors, scale, orientation, and affine tensors with stable channel counts.
    /// @throws std::invalid_argument if feature is not a 4D tensor or its channel count does not match the constructor.
    SparseHeadOutput forward(const torch::Tensor& feature);

private:
    int64_t _input_channels;
    int64_t _descriptor_dim;
    torch::nn::Sequential _context{nullptr};
    torch::nn::Conv2d _heatmap{nullptr};
    torch::nn::Sequential _descriptors{nullptr};
    torch::nn::Conv2d _scale{nullptr};
    torch::nn::Conv2d _orientation{nullptr};
    torch::nn::Conv2d _affine{nullptr};
};

TORCH_MODULE(SparseHead);

}  // namespace pfm
