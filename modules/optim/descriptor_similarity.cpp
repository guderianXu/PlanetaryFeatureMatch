#include "optim/descriptor_similarity.h"

#include <algorithm>
#include <stdexcept>

namespace pfm
{
namespace
{

torch::Tensor normalizeDescriptors(const torch::Tensor& descriptors)
{
    return descriptors / descriptors.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
}

void validateDescriptorInputs(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b)
{
    if (!descriptors_a.defined() || !descriptors_b.defined())
    {
        throw std::invalid_argument("descriptor similarity inputs must be defined");
    }
    if (descriptors_a.dim() != 3 || descriptors_b.dim() != 3)
    {
        throw std::invalid_argument("descriptor similarity inputs must have shape BxNxD");
    }
    if (descriptors_a.size(0) != descriptors_b.size(0) || descriptors_a.size(2) != descriptors_b.size(2))
    {
        throw std::invalid_argument("descriptor similarity inputs have incompatible shapes");
    }
}

} // namespace

torch::Tensor cyclicDescriptorSimilarityScores(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b)
{
    validateDescriptorInputs(descriptors_a, descriptors_b);
    auto normalized_a = normalizeDescriptors(descriptors_a);
    auto normalized_b = normalizeDescriptors(descriptors_b);
    return torch::bmm(normalized_a, normalized_b.transpose(1, 2));
}

torch::Tensor cyclicDescriptorSimilarityScoresChunked(const torch::Tensor& descriptors_a,
                                                      const torch::Tensor& descriptors_b, int64_t query_chunk_size)
{
    validateDescriptorInputs(descriptors_a, descriptors_b);
    if (query_chunk_size <= 0 || descriptors_a.size(1) <= query_chunk_size)
    {
        return cyclicDescriptorSimilarityScores(descriptors_a, descriptors_b);
    }

    std::vector<torch::Tensor> chunks;
    chunks.reserve(static_cast<std::size_t>((descriptors_a.size(1) + query_chunk_size - 1) / query_chunk_size));
    for (int64_t begin = 0; begin < descriptors_a.size(1); begin += query_chunk_size)
    {
        const auto count = std::min<int64_t>(query_chunk_size, descriptors_a.size(1) - begin);
        chunks.push_back(cyclicDescriptorSimilarityScores(descriptors_a.narrow(1, begin, count), descriptors_b));
    }
    return torch::cat(chunks, 1);
}

} // namespace pfm
