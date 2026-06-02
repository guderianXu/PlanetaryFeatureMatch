#include "data/pair_archive_dataset.h"

#include <algorithm>
#include <stdexcept>

#include <torch/script.h>

namespace pfm
{
namespace
{

bool isPairArchivePath(const std::filesystem::path& path)
{
    const auto name = path.filename().string();
    return path.extension() == ".pt" && name.rfind("pair_", 0) == 0;
}

torch::Tensor requireTensor(const torch::jit::Module& module, const char* name, const std::filesystem::path& path)
{
    try
    {
        return module.attr(name).toTensor().detach().cpu().contiguous();
    }
    catch (const std::exception& exc)
    {
        throw std::invalid_argument("failed to read tensor " + std::string(name) + " from " + path.string() + ": " +
                                    exc.what());
    }
}

void validatePairArchiveSample(const PairArchiveSample& sample, bool require_nonempty_valid_mask)
{
    if (sample.view_a.dim() != 3 || sample.view_b.dim() != 3)
    {
        throw std::invalid_argument(sample.path.string() + " view tensors must have shape CxHxW");
    }
    if (sample.view_a.sizes() != sample.view_b.sizes())
    {
        throw std::invalid_argument(sample.path.string() + " view_a and view_b shapes do not match");
    }
    if (sample.warp_a_to_b.dim() != 3 || sample.warp_a_to_b.size(2) != 2)
    {
        throw std::invalid_argument(sample.path.string() + " warp_a_to_b must have shape HxWx2");
    }
    if (sample.valid_mask.dim() != 2)
    {
        throw std::invalid_argument(sample.path.string() + " valid_mask must have shape HxW");
    }
    if (sample.view_a.size(1) != sample.valid_mask.size(0) || sample.view_a.size(2) != sample.valid_mask.size(1))
    {
        throw std::invalid_argument(sample.path.string() + " image and valid_mask shapes do not match");
    }
    if (sample.warp_a_to_b.size(0) != sample.valid_mask.size(0) ||
        sample.warp_a_to_b.size(1) != sample.valid_mask.size(1))
    {
        throw std::invalid_argument(sample.path.string() + " warp and valid_mask shapes do not match");
    }
    if (!torch::isfinite(sample.view_a).all().item<bool>() || !torch::isfinite(sample.view_b).all().item<bool>() ||
        !torch::isfinite(sample.warp_a_to_b).all().item<bool>())
    {
        throw std::invalid_argument(sample.path.string() + " contains non-finite tensor values");
    }
    if (require_nonempty_valid_mask && sample.valid_mask.sum().item<int64_t>() <= 0)
    {
        throw std::invalid_argument(sample.path.string() + " has no valid correspondence pixels");
    }
}

} // namespace

std::vector<std::filesystem::path> discoverPairArchivePaths(const std::filesystem::path& cache_dir, int64_t limit_pairs)
{
    if (limit_pairs < 0)
    {
        throw std::invalid_argument("limit_pairs must be nonnegative");
    }
    if (!std::filesystem::exists(cache_dir))
    {
        throw std::invalid_argument("pair archive cache directory does not exist: " + cache_dir.string());
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(cache_dir))
    {
        if (entry.is_regular_file() && isPairArchivePath(entry.path()))
        {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (limit_pairs > 0 && static_cast<int64_t>(paths.size()) > limit_pairs)
    {
        paths.resize(static_cast<std::size_t>(limit_pairs));
    }
    return paths;
}

PairArchiveSample loadPairArchiveSample(const std::filesystem::path& path, bool require_nonempty_valid_mask)
{
    if (!std::filesystem::exists(path))
    {
        throw std::invalid_argument("pair archive does not exist: " + path.string());
    }

    torch::jit::Module module;
    try
    {
        module = torch::jit::load(path.string(), torch::Device(torch::kCPU));
    }
    catch (const std::exception& exc)
    {
        throw std::invalid_argument("failed to load TorchScript pair archive " + path.string() + ": " + exc.what());
    }

    PairArchiveSample sample{path, requireTensor(module, "view_a", path).to(torch::kFloat32),
                             requireTensor(module, "view_b", path).to(torch::kFloat32),
                             requireTensor(module, "warp_a_to_b", path).to(torch::kFloat32),
                             requireTensor(module, "valid_mask", path).to(torch::kBool)};
    validatePairArchiveSample(sample, require_nonempty_valid_mask);
    return sample;
}

PairArchiveDataset::PairArchiveDataset(const PairArchiveDatasetConfig& config)
    : _paths(discoverPairArchivePaths(config.cache_dir, config.limit_pairs)),
      _require_nonempty_valid_mask(config.require_nonempty_valid_mask)
{
    if (_paths.empty())
    {
        throw std::invalid_argument("no pair_*.pt archives found under " + config.cache_dir.string());
    }
}

std::size_t PairArchiveDataset::size() const
{
    return _paths.size();
}

const std::filesystem::path& PairArchiveDataset::path(std::size_t index) const
{
    if (index >= _paths.size())
    {
        throw std::out_of_range("pair archive dataset index out of range");
    }
    return _paths[index];
}

PairArchiveSample PairArchiveDataset::load(std::size_t index) const
{
    return loadPairArchiveSample(path(index), _require_nonempty_valid_mask);
}

} // namespace pfm
