#pragma once

#include <vector>

#include "augment/image_pair_augmentor.h"
#include "dataloader/dataset.h"

namespace pfm
{

class SyntheticPairTensorDataset : public TensorDataset
{
  public:
    /// 创建在线合成影像对数据集。
    /// @param images 源 CHW float 图像张量。
    /// @param pairs_per_image 每张源图像生成的影像对数量。
    /// @param config 基础增强配置。
    /// @throws std::invalid_argument 当 images 为空或 pairs_per_image 为 0 时抛出。
    SyntheticPairTensorDataset(std::vector<torch::Tensor> images, size_t pairs_per_image,
                               ImagePairAugmentationConfig config);

    /// 返回生成的影像对总数。
    /// @return images.size() 与 pairs_per_image 的乘积。
    size_t size() const override;

    /// 生成一个合成影像对样本。
    /// @param index 数据集索引。
    /// @return 包含 view_a、view_b、warp_a_to_b 和 valid_mask 的 TensorBatch。
    /// @throws std::out_of_range 当 index 非法时抛出。
    TensorBatch get(size_t index) override;

  private:
    std::vector<torch::Tensor> _images;
    size_t _pairs_per_image;
    ImagePairAugmentationConfig _config;
};

} // namespace pfm
