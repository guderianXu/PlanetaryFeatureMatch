#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace pfm
{

struct PairArchiveSample
{
    std::filesystem::path path;
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp_a_to_b;
    torch::Tensor valid_mask;
};

struct PairArchiveDatasetConfig
{
    std::filesystem::path cache_dir;
    int64_t limit_pairs = 0;
    bool require_nonempty_valid_mask = true;
};

/// Discovers sorted `pair_*.pt` archives under a cache split directory.
/// @param cache_dir Directory such as `dataset/cache/train`.
/// @param limit_pairs Optional positive limit; zero means no limit.
/// @return Stable sorted archive paths.
/// @throws std::invalid_argument if cache_dir is missing or limit_pairs is negative.
std::vector<std::filesystem::path> discoverPairArchivePaths(const std::filesystem::path& cache_dir,
                                                            int64_t limit_pairs = 0);

/// Loads and validates a TorchScript pair archive.
/// @param path Archive with view_a, view_b, warp_a_to_b, and valid_mask tensor attributes.
/// @param require_nonempty_valid_mask Whether valid_mask must contain at least one true pixel.
/// @return Loaded CPU contiguous tensors and source path.
/// @throws std::invalid_argument if the archive is missing, malformed, or has invalid tensor shapes.
PairArchiveSample loadPairArchiveSample(const std::filesystem::path& path, bool require_nonempty_valid_mask = true);

class PairArchiveDataset
{
  public:
    /// Creates a dataset from discovered `pair_*.pt` archives.
    /// @param config Cache directory, optional pair limit, and validation options.
    /// @throws std::invalid_argument if no archives are found.
    explicit PairArchiveDataset(const PairArchiveDatasetConfig& config);

    /// Returns the number of discovered pair archives.
    std::size_t size() const;

    /// Returns the archive path at index.
    /// @param index Zero-based sample index.
    /// @throws std::out_of_range if index is invalid.
    const std::filesystem::path& path(std::size_t index) const;

    /// Loads and validates one pair archive.
    /// @param index Zero-based sample index.
    /// @throws std::out_of_range if index is invalid.
    /// @throws std::invalid_argument if the archive is malformed.
    PairArchiveSample load(std::size_t index) const;

  private:
    std::vector<std::filesystem::path> _paths;
    bool _require_nonempty_valid_mask = true;
};

} // namespace pfm
