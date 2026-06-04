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

constexpr size_t PYTHON_MT_N = 624;
constexpr size_t PYTHON_MT_M = 397;
constexpr uint32_t PYTHON_MATRIX_A = 0x9908B0DFU;
constexpr uint32_t PYTHON_UPPER_MASK = 0x80000000U;
constexpr uint32_t PYTHON_LOWER_MASK = 0x7FFFFFFFU;

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

int bit_length(uint64_t value)
{
    int bits = 0;
    while (value != 0)
    {
        ++bits;
        value >>= 1;
    }
    return bits;
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
    seedPythonRandom(seed);
}

std::vector<size_t> ShuffleSampler::indices() const
{
    auto result = ordered_indices(_count);
    for (size_t index = result.size(); index > 1; --index)
    {
        const auto swap_index = static_cast<size_t>(pythonRandBelow(static_cast<uint64_t>(index)));
        std::swap(result[index - 1], result[swap_index]);
    }
    return result;
}

void ShuffleSampler::seedPythonRandom(uint64_t seed)
{
    std::array<uint32_t, 2> key{};
    auto key_length = size_t{1};
    key[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
    key[1] = static_cast<uint32_t>((seed >> 32U) & 0xFFFFFFFFULL);
    if (key[1] != 0U)
    {
        key_length = 2;
    }

    _python_mt[0] = 19650218U;
    for (_python_mti = 1; _python_mti < static_cast<int>(PYTHON_MT_N); ++_python_mti)
    {
        const auto previous = _python_mt[static_cast<size_t>(_python_mti - 1)];
        _python_mt[static_cast<size_t>(_python_mti)] =
            static_cast<uint32_t>(1812433253U * (previous ^ (previous >> 30U)) + static_cast<uint32_t>(_python_mti));
    }

    auto i = size_t{1};
    auto j = size_t{0};
    for (auto k = std::max(PYTHON_MT_N, key_length); k > 0; --k)
    {
        const auto previous = _python_mt[i - 1];
        _python_mt[i] = static_cast<uint32_t>((_python_mt[i] ^ ((previous ^ (previous >> 30U)) * 1664525U)) +
                                              key[j] + static_cast<uint32_t>(j));
        ++i;
        ++j;
        if (i >= PYTHON_MT_N)
        {
            _python_mt[0] = _python_mt[PYTHON_MT_N - 1];
            i = 1;
        }
        if (j >= key_length)
        {
            j = 0;
        }
    }
    for (auto k = PYTHON_MT_N - 1; k > 0; --k)
    {
        const auto previous = _python_mt[i - 1];
        _python_mt[i] = static_cast<uint32_t>((_python_mt[i] ^ ((previous ^ (previous >> 30U)) * 1566083941U)) -
                                              static_cast<uint32_t>(i));
        ++i;
        if (i >= PYTHON_MT_N)
        {
            _python_mt[0] = _python_mt[PYTHON_MT_N - 1];
            i = 1;
        }
    }
    _python_mt[0] = 0x80000000U;
    _python_mti = static_cast<int>(PYTHON_MT_N);
}

uint32_t ShuffleSampler::nextPythonRandomWord() const
{
    static constexpr std::array<uint32_t, 2> mag01{0x0U, PYTHON_MATRIX_A};
    uint32_t value = 0;

    if (_python_mti >= static_cast<int>(PYTHON_MT_N))
    {
        size_t kk = 0;
        for (; kk < PYTHON_MT_N - PYTHON_MT_M; ++kk)
        {
            value = (_python_mt[kk] & PYTHON_UPPER_MASK) | (_python_mt[kk + 1] & PYTHON_LOWER_MASK);
            _python_mt[kk] = _python_mt[kk + PYTHON_MT_M] ^ (value >> 1U) ^ mag01[value & 0x1U];
        }
        for (; kk < PYTHON_MT_N - 1; ++kk)
        {
            value = (_python_mt[kk] & PYTHON_UPPER_MASK) | (_python_mt[kk + 1] & PYTHON_LOWER_MASK);
            _python_mt[kk] =
                _python_mt[kk + PYTHON_MT_M - PYTHON_MT_N] ^ (value >> 1U) ^ mag01[value & 0x1U];
        }
        value = (_python_mt[PYTHON_MT_N - 1] & PYTHON_UPPER_MASK) | (_python_mt[0] & PYTHON_LOWER_MASK);
        _python_mt[PYTHON_MT_N - 1] = _python_mt[PYTHON_MT_M - 1] ^ (value >> 1U) ^ mag01[value & 0x1U];
        _python_mti = 0;
    }

    value = _python_mt[static_cast<size_t>(_python_mti)];
    ++_python_mti;

    value ^= value >> 11U;
    value ^= (value << 7U) & 0x9D2C5680U;
    value ^= (value << 15U) & 0xEFC60000U;
    value ^= value >> 18U;

    return value;
}

uint64_t ShuffleSampler::pythonGetRandBits(int bits) const
{
    if (bits <= 0)
    {
        return 0;
    }
    if (bits <= 32)
    {
        return static_cast<uint64_t>(nextPythonRandomWord() >> (32 - bits));
    }

    uint64_t result = 0;
    auto remaining = bits;
    auto shift = 0;
    while (remaining > 0)
    {
        const auto chunk_bits = std::min(remaining, 32);
        auto word = nextPythonRandomWord();
        if (chunk_bits < 32)
        {
            word >>= 32 - chunk_bits;
        }
        result |= static_cast<uint64_t>(word) << shift;
        remaining -= chunk_bits;
        shift += chunk_bits;
    }
    return result;
}

uint64_t ShuffleSampler::pythonRandBelow(uint64_t upper) const
{
    if (upper == 0)
    {
        throw std::invalid_argument("pythonRandBelow requires positive upper bound");
    }

    const auto bits = std::max(1, bit_length(upper));
    auto value = pythonGetRandBits(bits);
    while (value >= upper)
    {
        value = pythonGetRandBits(bits);
    }
    return value;
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
