#pragma once

#include <string>

#include <torch/torch.h>

#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"

namespace pfm
{

struct WarpMatchMetrics
{
    /// 稀疏匹配总数。
    int64_t sparse_total = 0;
    /// 通过变形场几何阈值判定正确的稀疏匹配数。
    int64_t sparse_correct = 0;
    /// 稠密匹配总数。
    int64_t dense_total = 0;
    /// 通过变形场几何阈值判定正确的稠密匹配数。
    int64_t dense_correct = 0;

    /// @return 稀疏和稠密匹配总数。
    int64_t total() const;
    /// @return 稀疏和稠密正确匹配总数。
    int64_t correct() const;
    /// @return correct()/total()，无匹配时返回 0。
    double precision() const;
};

struct WarpFeatureCoverageMetrics
{
    /// A 图检测到的源关键点总数。
    int64_t source_total = 0;
    /// 可通过变形场投影到 B 图有效区域的源关键点数。
    int64_t valid_warp_total = 0;
    /// 被 B 图检测关键点覆盖的投影位置数量。
    int64_t covered_by_target_keypoint = 0;
    /// 能观察到描述子正样本排序名次的源关键点数量。
    int64_t descriptor_rank_observed = 0;
    /// 描述子正样本排序名次为 top-1 的数量。
    int64_t descriptor_top1_count = 0;
    /// 描述子正样本排序名次累加值。
    int64_t descriptor_rank_sum = 0;
    /// 几何覆盖率。
    double coverage_fraction = 0.0;
    /// 投影位置到最近 B 图关键点的平均像素距离。
    double mean_nearest_target_distance_pixels = 0.0;
    /// 描述子正样本平均排序名次。
    double mean_descriptor_positive_rank = 0.0;
    /// 描述子 top-1 准确率。
    double descriptor_top1_accuracy = 0.0;
};

/// 从合成影像对的 .pt archive 中读取稠密 A-to-B 变形场张量。
/// @param path 合成影像对 archive 路径。
/// @return A 到 B 的稠密变形场张量。
/// @throws std::invalid_argument 当 archive 无法加载或字段缺失时抛出。
torch::Tensor load_warp_a_to_b_tensor(const std::string& path);

/// 使用合成数据提供的稠密 A-to-B 变形场评估预测匹配。
/// @param features_a A 图解码特征。
/// @param features_b B 图解码特征。
/// @param matches 预测匹配集合。
/// @param warp_a_to_b A 到 B 的稠密变形场张量。
/// @param correct_threshold_pixels 判定匹配正确的像素阈值。
/// @return 稀疏/稠密匹配总数、正确数和精度。
/// @throws std::invalid_argument 当输入张量形状、dtype 或阈值非法时抛出。
WarpMatchMetrics compute_warp_match_metrics(const FeatureSet& features_a, const FeatureSet& features_b,
                                            const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                            double correct_threshold_pixels);

/// 衡量 A 图检测关键点在真实变形后的位置是否被 B 图关键点覆盖，并统计描述子对这些近邻正样本的排序质量。
/// @param features_a A 图解码特征。
/// @param features_b B 图解码特征。
/// @param warp_a_to_b A 到 B 的稠密变形场张量。
/// @param correct_threshold_pixels 判定几何覆盖的像素阈值。
/// @return 几何覆盖率、最近关键点距离、描述子排序名次和 top-1 准确率。
/// @throws std::invalid_argument 当输入张量形状、dtype 或阈值非法时抛出。
WarpFeatureCoverageMetrics compute_warp_feature_coverage_metrics(const FeatureSet& features_a,
                                                                 const FeatureSet& features_b,
                                                                 const torch::Tensor& warp_a_to_b,
                                                                 double correct_threshold_pixels);

} // namespace pfm
