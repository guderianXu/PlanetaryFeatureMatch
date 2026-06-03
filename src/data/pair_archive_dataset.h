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
    /// 当前样本对应的 pair_*.pt archive 路径。
    std::filesystem::path path;
    /// 视图 A 的 CHW float 图像张量。
    torch::Tensor view_a;
    /// 视图 B 的 CHW float 图像张量。
    torch::Tensor view_b;
    /// A 到 B 的稠密变形场。
    torch::Tensor warp_a_to_b;
    /// A 图有效投影区域 mask。
    torch::Tensor valid_mask;
};

struct PairArchiveDatasetConfig
{
    /// cache split 目录，例如 dataset/cache/train。
    std::filesystem::path cache_dir;
    /// 最多加载的 pair 数量；0 表示不限制。
    int64_t limit_pairs = 0;
    /// 是否要求 valid_mask 至少包含一个有效像素。
    bool require_nonempty_valid_mask = true;
};

/// 在 cache split 目录下发现并排序 `pair_*.pt` archive。
/// @param cache_dir 例如 `dataset/cache/train` 的目录。
/// @param limit_pairs 可选正数上限；0 表示不限制。
/// @return 稳定排序后的 archive 路径。
/// @throws std::invalid_argument 当 cache_dir 缺失或 limit_pairs 为负数时抛出。
std::vector<std::filesystem::path> discoverPairArchivePaths(const std::filesystem::path& cache_dir,
                                                            int64_t limit_pairs = 0);

/// 加载并校验 TorchScript 影像对 archive。
/// @param path 包含 view_a、view_b、warp_a_to_b 和 valid_mask 张量字段的 archive。
/// @param require_nonempty_valid_mask 是否要求 valid_mask 至少包含一个 true 像素。
/// @return 已加载的 CPU 连续张量和源路径。
/// @throws std::invalid_argument 当 archive 缺失、格式错误或张量形状非法时抛出。
PairArchiveSample loadPairArchiveSample(const std::filesystem::path& path, bool require_nonempty_valid_mask = true);

class PairArchiveDataset
{
  public:
    /// 基于发现的 `pair_*.pt` archive 创建数据集。
    /// @param config cache 目录、可选 pair 上限和校验选项。
    /// @throws std::invalid_argument 当没有找到 archive 时抛出。
    explicit PairArchiveDataset(const PairArchiveDatasetConfig& config);

    /// 返回发现的影像对 archive 数量。
    std::size_t size() const;

    /// 返回指定索引处的 archive 路径。
    /// @param index 从 0 开始的样本索引。
    /// @throws std::out_of_range 当 index 非法时抛出。
    const std::filesystem::path& path(std::size_t index) const;

    /// 加载并校验一个影像对 archive。
    /// @param index 从 0 开始的样本索引。
    /// @throws std::out_of_range 当 index 非法时抛出。
    /// @throws std::invalid_argument 当 archive 格式错误时抛出。
    PairArchiveSample load(std::size_t index) const;

  private:
    std::vector<std::filesystem::path> _paths;
    bool _require_nonempty_valid_mask = true;
};

} // namespace pfm
