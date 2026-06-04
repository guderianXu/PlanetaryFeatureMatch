#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

namespace pfm
{

class Sampler
{
  public:
    /// Destroys the sampler.
    virtual ~Sampler() = default;

    /// Returns the index order for one epoch.
    /// \return Sample indices.
    virtual std::vector<size_t> indices() const = 0;
};

class SequentialSampler : public Sampler
{
  public:
    /// Creates an ordered sampler for count samples.
    /// \param count Number of samples.
    explicit SequentialSampler(size_t count);

    /// Returns indices 0..count-1.
    /// \return Ordered indices.
    std::vector<size_t> indices() const override;

  private:
    size_t _count;
};

class ShuffleSampler : public Sampler
{
  public:
    /// Creates a deterministic shuffled sampler using Python random.Random compatible shuffle order.
    /// \param count Number of samples.
    /// \param seed Random seed.
    ShuffleSampler(size_t count, uint64_t seed);

    /// Returns shuffled indices for the next epoch.
    /// \return Deterministically shuffled indices for the sampler's current epoch.
    std::vector<size_t> indices() const override;

  private:
    void seedPythonRandom(uint64_t seed);
    uint32_t nextPythonRandomWord() const;
    uint64_t pythonGetRandBits(int bits) const;
    uint64_t pythonRandBelow(uint64_t upper) const;

    size_t _count;
    uint64_t _seed;
    mutable std::array<uint32_t, 624> _python_mt{};
    mutable int _python_mti = 625;
};

class SubsetSampler : public Sampler
{
  public:
    /// Creates a sampler over an explicit index list.
    /// \param indices Indices to return.
    explicit SubsetSampler(std::vector<size_t> indices);

    /// Returns the configured subset.
    /// \return Subset indices.
    std::vector<size_t> indices() const override;

  private:
    std::vector<size_t> _indices;
};

struct DatasetSplit
{
    std::vector<size_t> train;
    std::vector<size_t> validation;
    std::vector<size_t> test;
};

/// Splits dataset indices into train, validation, and test subsets.
/// \param count Number of samples.
/// \param train_ratio Train split ratio.
/// \param validation_ratio Validation split ratio.
/// \param test_ratio Test split ratio.
/// \param seed Shuffle seed used when shuffle is true.
/// \param shuffle Whether to shuffle before splitting.
/// \return Three disjoint index vectors covering every sample exactly once.
/// \throws std::invalid_argument if ratios are negative or do not sum to one.
DatasetSplit make_train_validation_test_split(size_t count, double train_ratio, double validation_ratio,
                                              double test_ratio, uint64_t seed, bool shuffle);

} // namespace pfm
