#pragma once

#include <torch/torch.h>

namespace pfm {

struct SparseFeatures {
    torch::Tensor keypoints;
    torch::Tensor scores;
    torch::Tensor descriptors;
    torch::Tensor scales;
    torch::Tensor orientations;
    torch::Tensor affine;
};

struct SparseMatches {
    torch::Tensor indices;
    torch::Tensor scores;
};

struct SemiDenseMatches {
    torch::Tensor points_a;
    torch::Tensor points_b;
    torch::Tensor confidence;
    torch::Tensor offsets;
    torch::Tensor valid_mask;
};

struct MatchResult {
    SparseMatches sparse;
    SemiDenseMatches semi_dense;
};

}  // namespace pfm
