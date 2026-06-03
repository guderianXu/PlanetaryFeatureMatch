#pragma once

#include <torch/torch.h>

#include "feature_io/feature_codec.h"

namespace pfm
{

struct RawFeatureMaps
{
    /// 稀疏关键点热力图，形状通常为 Bx1xHxW。
    torch::Tensor heatmap;
    /// 描述子特征图，形状通常为 BxDxHxW。
    torch::Tensor descriptors;
    /// 每个候选关键点的尺度预测图。
    torch::Tensor scale;
    /// 每个候选关键点的方向预测图。
    torch::Tensor orientation;
    /// 每个候选关键点的局部 affine 参数预测图。
    torch::Tensor affine;
    /// 稠密/半稠密分支输出的置信度图。
    torch::Tensor dense_confidence;
};

/// 控制稀疏关键点和半稠密特征解码。
struct FeatureDecodeConfig
{
    /// 最多返回的稀疏关键点数量。
    int max_keypoints = 1024;
    /// 稀疏关键点软下限；为 0 时关闭补点。
    int min_keypoints = 0;
    /// 输出半稠密点所需的最小稠密置信度。
    double semi_dense_threshold = 0.5;
    /// 稀疏关键点网格行数。
    int keypoint_grid_rows = 8;
    /// 稀疏关键点网格列数。
    int keypoint_grid_cols = 8;
    /// 每个网格单元最多保留的稀疏关键点数；为 0 时由 max_keypoints 自动推导。
    int keypoints_per_cell = 0;
    /// 稀疏关键点 NMS 半径，单位为特征图像素。
    int nms_radius = 4;
    /// 感知方向的稀疏描述子池化半径；为 0 时只使用关键点所在单元。
    int descriptor_pool_radius = 0;
    /// 将四向描述子通道组滚动到预测的局部方向坐标系。
    bool descriptor_orientation_canonicalization = false;
};

/// 从网络原始输出图中解码稀疏和稠密特征张量。
/// @param maps 原始热力图、描述子、尺度、方向、仿射和稠密置信度图；每个张量都必须是已
/// 定义的 CPU 4D 张量，batch size 为 1，空间尺寸一致且为正数，通道数合法。
/// @param max_keypoints 最多返回的稀疏热力图位置数量，必须为正。
/// @param semi_dense_threshold 输出稠密点所需的最小稠密置信度。
/// @return FeatureSet，包含以特征图坐标表示的连续 CPU float 张量。
/// @throws std::invalid_argument 当 maps 未定义、不是 CPU 4D、batch size 不为 1、通道数非法、空间尺寸不一致或
/// max_keypoints 非正时抛出。
FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold);

/// 解码稀疏和稠密特征张量，同时屏蔽无效影像位置。
/// @param maps 原始热力图、描述子、尺度、方向、仿射和稠密置信度图。
/// @param max_keypoints 最多返回的稀疏热力图位置数量，必须为正。
/// @param semi_dense_threshold 输出稠密点所需的最小稠密置信度。
/// @param intensity_mask 可选的 HxW CPU mask，坐标位于原始影像空间；非零值表示有效。
/// @return 只包含有效稀疏和稠密位置的 FeatureSet。
/// @throws std::invalid_argument 当 maps、参数或 mask 形状/设备非法时抛出。
FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold,
                               const torch::Tensor& intensity_mask);

/// 使用显式解码配置解码稀疏和稠密特征张量。
/// @param maps 原始热力图、描述子、尺度、方向、仿射和稠密置信度图。
/// @param config 稀疏关键点与半稠密解码参数；计数类参数必须符合 FeatureDecodeConfig 文档要求。
/// @param intensity_mask 可选的 HxW CPU mask，坐标位于原始影像空间；非零值表示有效。
/// @return 经过局部 NMS 的稀疏位置，以及高于配置阈值的稠密位置。
/// @throws std::invalid_argument 当 maps、config 或 mask 形状/设备非法时抛出。
FeatureSet decode_feature_maps(const RawFeatureMaps& maps, const FeatureDecodeConfig& config,
                               const torch::Tensor& intensity_mask);

} // namespace pfm
