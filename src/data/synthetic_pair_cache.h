#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "data/image_dataset.h"
#include "data/synthetic_pair.h"
#include "dataloader/dataset.h"

namespace pfm
{

struct SyntheticPairCacheConfig
{
    std::string cache_dir;
    int64_t resize = 512;
    std::size_t pair_count = 0;
    std::size_t pairs_per_image = 1;
    std::size_t source_count = 0;
    SyntheticPairConfig pair_config;
    bool rebuild = false;
};

/// Prepares a synthetic pair cache for a deterministic training configuration.
/// @param dataset Source image dataset used to generate cached pairs.
/// @param config Cache directory, pair count, resize limit, synthetic pair configuration, and rebuild flag.
/// @throws std::invalid_argument if the cache configuration or source image data is invalid.
void prepare_synthetic_pair_cache(const ImageDataset& dataset, const SyntheticPairCacheConfig& config);

class SyntheticPairCacheDataset
{
  public:
    /// Opens an existing synthetic pair cache directory.
    /// @param cache_dir Directory containing manifest.pt and pair_*.pt files.
    /// @throws std::invalid_argument if the cache manifest cannot be loaded.
    explicit SyntheticPairCacheDataset(std::string cache_dir);

    /// Returns the number of cached synthetic pairs.
    /// @return Cached pair count recorded in the manifest.
    std::size_t size() const;

    /// Loads one cached synthetic pair.
    /// @param index Zero-based cached pair index.
    /// @return Synthetic pair tensors stored in the cache.
    /// @throws std::out_of_range if index is not less than size().
    /// @throws std::invalid_argument if the cached pair file is missing or invalid.
    SyntheticPair load(std::size_t index) const;

  private:
    std::string _cache_dir;
    std::size_t _pair_count = 0;
    std::size_t _pairs_per_image = 1;
    std::size_t _source_count = 1;
};

class SyntheticPairCacheTensorDataset : public TensorDataset
{
  public:
    /// Wraps an existing synthetic pair cache as TensorBatch samples.
    /// @param cache_dir Directory containing a prepared synthetic pair cache.
    explicit SyntheticPairCacheTensorDataset(std::string cache_dir);

    /// Returns cached pair count.
    /// @return Dataset size.
    size_t size() const override;

    /// Loads one cached synthetic pair as an unbatched TensorBatch sample.
    /// @param index Cached pair index.
    /// @return Tensor batch with view_a, view_b, warp_a_to_b, and valid_mask.
    TensorBatch get(size_t index) override;

  private:
    SyntheticPairCacheDataset _cache;
};

} // namespace pfm
