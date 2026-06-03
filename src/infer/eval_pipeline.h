#pragma once

#include <string>
#include <utility>
#include <vector>

#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"

namespace pfm
{

struct EvalReport
{
    /// 平均匹配数量，统计每个影像对输出的稀疏/稠密匹配规模。
    double average_matches = 0.0;
    /// 稀疏匹配平均置信度或质量分数。
    double average_sparse_score = 0.0;
    /// 稠密匹配平均置信度。
    double average_dense_confidence = 0.0;
    /// 半稠密匹配覆盖率。
    double semi_dense_coverage = 0.0;
    /// 半周旋转一致性比例，用于评估旋转鲁棒性。
    double half_turn_consistency = 0.0;
    /// 半周旋转后的平均几何误差。
    double half_turn_mean_error = 0.0;
};

/// 从文本文件读取以空白字符分隔的影像对。
/// @param path 源影像对文件路径。
/// @return 影像路径对列表。
/// @throws std::invalid_argument 当文件无法打开或没有有效影像对时抛出。
std::vector<std::pair<std::string, std::string>> loadEvalPairs(const std::string& path);

/// 汇总多组已解码特征和匹配结果的评估指标。
/// @param feature_sets 用作评估分母的特征影像对列表。
/// @param match_sets 与 feature_sets 一一对应的匹配输出。
/// @return 平均匹配数、稀疏分数、稠密置信度和半稠密覆盖率等指标。
/// @throws std::invalid_argument 当输入为空或两组输入数量不一致时抛出。
EvalReport aggregateEvalReport(const std::vector<std::pair<FeatureSet, FeatureSet>>& feature_sets,
                               const std::vector<MatchSet>& match_sets);

/// 将评估指标保存为 LibTorch archive。
/// @param path 目标 .pt 报告路径。
/// @param report 待序列化的指标。
/// @throws std::invalid_argument 当序列化失败时抛出。
void saveEvalReport(const std::string& path, const EvalReport& report);

} // namespace pfm
