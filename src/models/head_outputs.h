#pragma once

#include <torch/torch.h>

namespace pfm
{

struct SparseHeadOutput
{
    torch::Tensor heatmap;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
};

struct DenseHeadOutput
{
    torch::Tensor confidence;
    torch::Tensor offsets;
};

} // namespace pfm
