#include "dataloader/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace pfm
{
namespace
{

std::vector<size_t> ordered_indices(size_t count)
{
    std::vector<size_t> result(count);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

void require_valid_ratios(double train_ratio, double validation_ratio, double test_ratio)
{
    if (!std::isfinite(train_ratio) || !std::isfinite(validation_ratio) || !std::isfinite(test_ratio))
    {
        throw std::invalid_argument("dataset split ratios must be finite");
    }
    if (train_ratio < 0.0 || validation_ratio < 0.0 || test_ratio < 0.0)
    {
        throw std::invalid_argument("dataset split ratios must be non-negative");
    }
    const auto total = train_ratio + validation_ratio + test_ratio;
    if (std::abs(total - 1.0) > 1.0e-9)
    {
        throw std::invalid_argument("dataset split ratios must sum to one");
    }
}

} // namespace

SequentialSampler::SequentialSampler(size_t count) : _count(count)
{
}

std::vector<size_t> SequentialSampler::indices() const
{
    return ordered_indices(_count);
}

ShuffleSampler::ShuffleSampler(size_t count, uint64_t seed) : _count(count), _seed(seed)
{
}

std::vector<size_t> ShuffleSampler::indices() const
{
    auto result = ordered_indices(_count);
    std::mt19937_64 generator(_seed);
    std::shuffle(result.begin(), result.end(), generator);
    return result;
}

SubsetSampler::SubsetSampler(std::vector<size_t> indices) : _indices(std::move(indices))
{
}

std::vector<size_t> SubsetSampler::indices() const
{
    return _indices;
}

DatasetSplit make_train_validation_test_split(size_t count, double train_ratio, double validation_ratio,
                                              double test_ratio, uint64_t seed, bool shuffle)
{
    require_valid_ratios(train_ratio, validation_ratio, test_ratio);
    auto indices = ordered_indices(count);
    if (shuffle)
    {
        std::mt19937_64 generator(seed);
        std::shuffle(indices.begin(), indices.end(), generator);
    }

    const auto train_count = static_cast<size_t>(std::floor(static_cast<double>(count) * train_ratio));
    const auto validation_count = static_cast<size_t>(std::floor(static_cast<double>(count) * validation_ratio));
    const auto test_count = count - train_count - validation_count;

    DatasetSplit split;
    split.train.assign(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(train_count));
    split.validation.assign(indices.begin() + static_cast<std::ptrdiff_t>(train_count),
                            indices.begin() + static_cast<std::ptrdiff_t>(train_count + validation_count));
    split.test.assign(indices.end() - static_cast<std::ptrdiff_t>(test_count), indices.end());
    return split;
}

} // namespace pfm
