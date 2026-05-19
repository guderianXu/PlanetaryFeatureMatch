#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm {

struct MatchSet {
    torch::Tensor sparse_matches;
    torch::Tensor sparse_scores;
    torch::Tensor points_a;
    torch::Tensor points_b;
    torch::Tensor confidence;
};

/// Saves a match tensor set to a LibTorch archive.
/// @param match_set Match tensors to serialize; every tensor must be defined.
/// @param path Destination .pt file path.
/// @throws std::invalid_argument if any required tensor is undefined or serialization fails.
void save_match_set(const MatchSet& match_set, const std::string& path);

/// Loads a match tensor set from a LibTorch archive.
/// @param path Source .pt file path.
/// @return MatchSet populated from the archive fields.
/// @throws std::invalid_argument if loading fails or any required tensor is missing.
MatchSet load_match_set(const std::string& path);

}  // namespace pfm
