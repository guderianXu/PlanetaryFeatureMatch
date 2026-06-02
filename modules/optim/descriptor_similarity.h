#pragma once

#include <torch/torch.h>

namespace pfm
{

/// Computes normalized cosine descriptor similarities.
/// Shapes: descriptors_a BxQxD, descriptors_b BxCxD, result BxQxC.
torch::Tensor cyclicDescriptorSimilarityScores(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b);

/// Computes normalized cosine descriptor similarities in query chunks to reduce peak memory.
/// Shapes and output match cyclicDescriptorSimilarityScores.
torch::Tensor cyclicDescriptorSimilarityScoresChunked(const torch::Tensor& descriptors_a,
                                                      const torch::Tensor& descriptors_b, int64_t query_chunk_size);

} // namespace pfm
