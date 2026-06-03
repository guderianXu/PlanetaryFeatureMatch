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
    /// cache 输出目录。
    std::string cache_dir;
    /// 生成 cache 前的图像缩放上限；0 或负值表示不缩放。
    int64_t resize = 512;
    /// 目标影像对数量；0 时根据 source_count 和 pairs_per_image 推导。
    std::size_t pair_count = 0;
    /// 每张源图像生成的影像对数量。
    std::size_t pairs_per_image = 1;
    /// 参与生成的源图像数量；0 表示使用全部图像。
    std::size_t source_count = 0;
    /// 合成影像对的基础增强配置。
    SyntheticPairConfig pair_config;
    /// 是否强制重建已有 cache。
    bool rebuild = false;
};

/// 为确定性训练配置准备合成影像对 cache。
/// @param dataset 用于生成 cache 的源图像数据集。
/// @param config cache 目录、影像对数量、缩放上限、合成配置和重建标志。
/// @throws std::invalid_argument 当 cache 配置或源图像数据非法时抛出。
void prepare_synthetic_pair_cache(const ImageDataset& dataset, const SyntheticPairCacheConfig& config);

class SyntheticPairCacheDataset
{
  public:
    /// 打开已有合成影像对 cache 目录。
    /// @param cache_dir 包含 manifest.pt 和 pair_*.pt 文件的目录。
    /// @throws std::invalid_argument 当 cache manifest 无法加载时抛出。
    explicit SyntheticPairCacheDataset(std::string cache_dir);

    /// 返回已缓存的合成影像对数量。
    /// @return manifest 中记录的 cached pair 数量。
    std::size_t size() const;

    /// 加载一个已缓存的合成影像对。
    /// @param index 从 0 开始的 cache 索引。
    /// @return cache 中保存的合成影像对张量。
    /// @throws std::out_of_range 当 index 不小于 size() 时抛出。
    /// @throws std::invalid_argument 当 cached pair 文件缺失或非法时抛出。
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
    /// 将已有合成影像对 cache 包装为 TensorBatch 样本。
    /// @param cache_dir 包含已准备 cache 的目录。
    explicit SyntheticPairCacheTensorDataset(std::string cache_dir);

    /// 返回 cached pair 数量。
    /// @return 数据集大小。
    size_t size() const override;

    /// 将一个 cached pair 加载为未组 batch 的 TensorBatch 样本。
    /// @param index cached pair 索引。
    /// @return 包含 view_a、view_b、warp_a_to_b 和 valid_mask 的 TensorBatch。
    TensorBatch get(size_t index) override;

  private:
    SyntheticPairCacheDataset _cache;
};

} // namespace pfm
