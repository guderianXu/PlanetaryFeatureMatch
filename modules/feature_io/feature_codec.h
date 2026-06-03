#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm
{

struct FeatureSet
{
    /// 稀疏关键点，通常为 Nx2，坐标顺序为 {x, y}。
    torch::Tensor keypoints;
    /// 每个稀疏关键点的分数。
    torch::Tensor scores;
    /// 稀疏描述子张量，通常为 NxD。
    torch::Tensor descriptors;
    /// 每个稀疏关键点的尺度。
    torch::Tensor scale;
    /// 每个稀疏关键点的方向。
    torch::Tensor orientation;
    /// 每个稀疏关键点的局部仿射参数。
    torch::Tensor affine;
    /// 稠密/半稠密点坐标。
    torch::Tensor dense_points;
    /// 稠密/半稠密点置信度。
    torch::Tensor dense_confidence;
    /// 产生稀疏关键点的特征图宽度；为 0 表示旧 archive 未记录。
    int64_t feature_map_width = 0;
    /// 产生稀疏关键点的特征图高度；为 0 表示旧 archive 未记录。
    int64_t feature_map_height = 0;
};

/// 将一组特征张量保存为 LibTorch archive。
/// @param feature_set 待序列化的特征张量集合；所有必需张量都必须已定义。
/// @param path 目标 .pt 文件路径。
/// @throws std::invalid_argument 当必需张量未定义或序列化失败时抛出。
void save_feature_set(const FeatureSet& feature_set, const std::string& path);

/// 从 LibTorch archive 读取特征张量集合。
/// @param path 源 .pt 文件路径。
/// @return 从 archive 字段填充的 FeatureSet。
/// @throws std::invalid_argument 当加载失败或必需字段缺失时抛出。
FeatureSet load_feature_set(const std::string& path);

} // namespace pfm
