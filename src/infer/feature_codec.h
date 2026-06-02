#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm
{

struct FeatureSet
{
    torch::Tensor keypoints;
    torch::Tensor scores;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
    torch::Tensor dense_points;
    torch::Tensor dense_confidence;
    int64_t feature_map_width = 0;
    int64_t feature_map_height = 0;
};

/// Saves a feature tensor set to a LibTorch archive.
/// @param feature_set Feature tensors to serialize; every tensor must be defined.
/// @param path Destination .pt file path.
/// @throws std::invalid_argument if any required tensor is undefined or serialization fails.
void save_feature_set(const FeatureSet& feature_set, const std::string& path);

/// Loads a feature tensor set from a LibTorch archive.
/// @param path Source .pt file path.
/// @return FeatureSet populated from the archive fields.
/// @throws std::invalid_argument if loading fails or any required tensor is missing.
FeatureSet load_feature_set(const std::string& path);

} // namespace pfm
