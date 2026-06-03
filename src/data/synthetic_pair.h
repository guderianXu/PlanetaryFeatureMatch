#pragma once

#include <cstdint>
#include <string>

#include <torch/torch.h>

namespace pfm
{

enum class SyntheticPairAugmentationProfile
{
    Mixed,
    RotationOnly,
    Mild,
    Medium,
    Hard,
    Extreme,
    Viewpoint,
    CompoundViewpoint,
};

/// 解析合成影像对增强档位名称。
/// @param value 档位名称：mixed、rotation-only、mild、medium、hard、extreme、viewpoint 或 compound-viewpoint。
/// @return 解析后的增强档位枚举。
/// @throws std::invalid_argument 当档位名称不受支持时抛出。
SyntheticPairAugmentationProfile parse_synthetic_pair_augmentation_profile(const std::string& value);

/// 将合成影像对增强档位转换为 CLI 名称。
/// @param profile 增强档位枚举值。
/// @return 稳定的小写档位名称。
std::string synthetic_pair_augmentation_profile_name(SyntheticPairAugmentationProfile profile);

struct SyntheticPairConfig
{
    /// B 图相对 A 图的 x 方向平移，单位为像素。
    float translation_x = 0.0F;
    /// B 图相对 A 图的 y 方向平移，单位为像素。
    float translation_y = 0.0F;
    /// B 图相对 A 图的旋转角度，单位为度。
    float rotation_degrees = 0.0F;
    /// B 图相对 A 图的缩放比例。
    float scale = 1.0F;
    /// 亮度增量，应用在 B 图光度扰动中。
    float brightness_delta = 0.0F;
    /// 对比度缩放，应用在 B 图光度扰动中。
    float contrast_scale = 1.0F;
    /// 高斯噪声标准差。
    float noise_sigma = 0.0F;
    /// 离散旋转增强的步长，单位为度。
    float rotation_step_degrees = 15.0F;
    /// 同一源图像下的增强变体序号，用于确定性采样。
    int64_t variant_index = 0;
    /// 源图像序号，用于跨图像确定性采样。
    int64_t source_index = 0;
    /// 增强强度和几何类型档位。
    SyntheticPairAugmentationProfile augmentation_profile = SyntheticPairAugmentationProfile::Mixed;
    /// extreme 档位样本在 mixed 配置中的占比。
    double extreme_pair_ratio = 0.2;
};

struct SyntheticPair
{
    /// 原始视图 A，CHW float 张量。
    torch::Tensor view_a;
    /// 增强后的视图 B，CHW float 张量。
    torch::Tensor view_b;
    /// A 到 B 的稠密变形场，通常为 2xHxW 或 HxWx2。
    torch::Tensor warp_a_to_b;
    /// A 图中可投影到 B 图有效区域的布尔 mask。
    torch::Tensor valid_mask;
};

/// 创建确定性的合成影像对和稠密对应变形场。
/// @param image 输入 CHW float32 图像张量，支持单通道或三通道。
/// @param config 光度和几何增强配置。
/// @return 包含两个视图、A-to-B 变形场和有效 mask 的合成影像对。
/// @throws std::invalid_argument 当 image 不是合法 CHW 图像张量或 config 包含非法值时抛出。
SyntheticPair make_synthetic_pair(const torch::Tensor& image, const SyntheticPairConfig& config);

} // namespace pfm
