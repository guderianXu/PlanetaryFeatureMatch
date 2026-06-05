#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm
{

struct MatchSet
{
    /// 稀疏匹配索引对，通常为 Kx2，列分别索引 A/B 的稀疏关键点。
    torch::Tensor sparse_matches;
    /// 稀疏匹配分数。
    torch::Tensor sparse_scores;
    /// A 图匹配点坐标。
    torch::Tensor points_a;
    /// B 图匹配点坐标。
    torch::Tensor points_b;
    /// 稠密/半稠密匹配置信度。
    torch::Tensor confidence;
    /// v2.1 图匹配器实际执行的图注意力层数；非图匹配路径为 0。
    int64_t graph_executed_layers = 0;
    /// v2.1 图匹配器实际执行的注意力点对计算量代理。
    int64_t graph_attention_work_units = 0;
    /// v2.1 图匹配器满层注意力点对计算量代理。
    int64_t graph_full_attention_work_units = 0;
    /// v2.1 图匹配器实际计算量占满计算量的比例。
    double graph_attention_work_fraction = 0.0;
};

/// 将匹配张量集合保存为 LibTorch archive。
/// @param match_set 待序列化的匹配张量集合；所有必需张量都必须已定义。
/// @param path 目标 .pt 文件路径。
/// @throws std::invalid_argument 当必需张量未定义或序列化失败时抛出。
void save_match_set(const MatchSet& match_set, const std::string& path);

/// 从 LibTorch archive 读取匹配张量集合。
/// @param path 源 .pt 文件路径。
/// @return 从 archive 字段填充的 MatchSet。
/// @throws std::invalid_argument 当加载失败或必需字段缺失时抛出。
MatchSet load_match_set(const std::string& path);

} // namespace pfm
