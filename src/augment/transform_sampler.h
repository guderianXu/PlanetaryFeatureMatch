#pragma once

#include "augment/image_pair_augmentor.h"

namespace pfm
{

struct ImagePairTransformParameters
{
    /// x 方向平移，单位为像素。
    float translation_x = 0.0F;
    /// y 方向平移，单位为像素。
    float translation_y = 0.0F;
    /// 旋转角度，单位为度。
    float rotation_degrees = 0.0F;
    /// 缩放比例。
    float scale = 1.0F;
    /// 亮度增量。
    float brightness_delta = 0.0F;
    /// 对比度缩放。
    float contrast_scale = 1.0F;
    /// 高斯噪声标准差。
    float noise_sigma = 0.0F;
    /// gamma 光度扰动系数。
    float gamma = 1.0F;
    /// 阴影扰动强度。
    float shadow_strength = 0.0F;
    /// x 方向剪切系数。
    float shear_x = 0.0F;
    /// y 方向剪切系数。
    float shear_y = 0.0F;
    /// x 方向透视扰动系数。
    float perspective_x = 0.0F;
    /// y 方向透视扰动系数。
    float perspective_y = 0.0F;
};

/// 采样确定性的影像对几何和光度变换参数。
/// @param config 基础增强配置。
/// @return 已解析的几何和光度参数。
/// @throws std::invalid_argument 当配置值非法时抛出。
ImagePairTransformParameters sampleImagePairTransform(const ImagePairAugmentationConfig& config);

} // namespace pfm
