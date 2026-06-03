#pragma once

#include <cstdint>

#include <torch/torch.h>

#include "augment/augmentation_profile.h"

namespace pfm
{

struct ImagePairAugmentationConfig
{
    /// B 图相对 A 图的 x 方向平移，单位为像素。
    float translation_x = 0.0F;
    /// B 图相对 A 图的 y 方向平移，单位为像素。
    float translation_y = 0.0F;
    /// B 图相对 A 图的旋转角度，单位为度。
    float rotation_degrees = 0.0F;
    /// B 图相对 A 图的缩放比例。
    float scale = 1.0F;
    /// B 图亮度增量。
    float brightness_delta = 0.0F;
    /// B 图对比度缩放。
    float contrast_scale = 1.0F;
    /// B 图高斯噪声标准差。
    float noise_sigma = 0.0F;
    /// 离散旋转增强的角度步长。
    float rotation_step_degrees = 15.0F;
    /// 同一源图像下的增强变体序号。
    int64_t variant_index = 0;
    /// 源图像序号，用于确定性采样。
    int64_t source_index = 0;
    /// 随机采样种子。
    uint64_t seed = 0;
    /// 增强强度和几何类型档位。
    AugmentationProfile profile = AugmentationProfile::Mixed;
    /// extreme 样本在 mixed 档位中的占比。
    double extreme_pair_ratio = 0.2;
};

struct ImagePairSample
{
    /// 原始视图 A。
    torch::Tensor view_a;
    /// 增强后的视图 B。
    torch::Tensor view_b;
    /// A 到 B 的稠密变形场。
    torch::Tensor warp_a_to_b;
    /// A 图中可投影到 B 图有效区域的 mask。
    torch::Tensor valid_mask;
};

class ImagePairAugmentor
{
  public:
    /// 创建影像对增强器。
    /// @param config 增强配置。
    explicit ImagePairAugmentor(ImagePairAugmentationConfig config);

    /// 生成合成影像对和稠密对应场。
    /// @param image 源 CHW float 图像。
    /// @return 增强后的影像对样本。
    /// @throws std::invalid_argument 当图像或配置非法时抛出。
    ImagePairSample augment(const torch::Tensor& image) const;

  private:
    ImagePairAugmentationConfig _config;
};

} // namespace pfm
