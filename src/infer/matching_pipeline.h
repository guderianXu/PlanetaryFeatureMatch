#pragma once

#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"
#include "models/pfm_model_v21.h"
#include "models/planetary_graph_matcher.h"

namespace pfm
{

enum class GraphMatcherFallbackMode
{
    Geometry,
    None,
};

struct GraphMatcherInferenceOptions
{
    /// LightGlue 风格宽度剪枝阈值；-1 表示关闭，0..1 表示保留 raw 相似度达到阈值的点。
    double width_prune_min_score = -1.0;
    /// LightGlue 风格深度提前停止阈值；-1 表示关闭，0..1 表示达到 assignment 置信度后提前结束。
    double early_stop_min_confidence = -1.0;
    /// LightGlue 风格 matchability/accept 概率阈值；-1 表示关闭，0..1 表示低于阈值的匹配被丢弃。
    double min_accept_probability = -1.0;
    /// LightGlue 风格深度预算硬上限；0 表示使用 checkpoint 中的完整图注意力层数。
    int64_t max_attention_layers = 0;
    /// 图匹配输出后的回退策略；Geometry 表示继续 descriptor/top-k/几何回退，None 表示严格返回 graph 输出。
    GraphMatcherFallbackMode fallback_mode = GraphMatcherFallbackMode::Geometry;
};

/// 使用已学习的行星图匹配器匹配两组已解码特征。
/// @param features_a 第一幅影像的特征集合。
/// @param features_b 第二幅影像的特征集合。
/// @param matcher 已学习的图匹配模块。
/// @return MatchSet，包含已学习稀疏索引对/分数以及半稠密点对应关系/置信度。
/// @throws std::invalid_argument 当必需稀疏或稠密张量未定义，或维度非法时抛出。
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b,
                          PlanetaryGraphMatcherImpl& matcher);

/// 使用已学习的行星图匹配器和显式推理选项匹配两组已解码特征。
/// @param features_a 第一幅影像的特征集合。
/// @param features_b 第二幅影像的特征集合。
/// @param matcher 已学习的图匹配模块。
/// @param graph_options 图匹配推理选项；旧版 matcher 不支持 LightGlue 风格剪枝/提前停止。
/// @return MatchSet，包含稀疏与半稠密匹配。
/// @throws std::invalid_argument 当选项非法，或旧版 matcher 收到 LightGlue 选项时抛出。
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b,
                          PlanetaryGraphMatcherImpl& matcher, const GraphMatcherInferenceOptions& graph_options);

/// 使用 v2.1 行星图匹配器匹配两组已解码特征。
/// @param features_a 第一幅影像的特征集合。
/// @param features_b 第二幅影像的特征集合。
/// @param matcher v2.1 已学习图匹配模块。
/// @return MatchSet，包含已学习稀疏索引对/分数以及半稠密点对应关系/置信度。
/// @throws std::invalid_argument 当必需稀疏或稠密张量未定义，或维度非法时抛出。
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b,
                          v21::PfmV21GraphMatcherImpl& matcher);

/// 使用 v2.1 行星图匹配器和显式推理选项匹配两组已解码特征。
/// @param features_a 第一幅影像的特征集合。
/// @param features_b 第二幅影像的特征集合。
/// @param matcher v2.1 已学习图匹配模块。
/// @param graph_options 图匹配推理选项。
/// @return MatchSet，包含稀疏与半稠密匹配。
/// @throws std::invalid_argument 当选项非法或特征维度非法时抛出。
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b,
                          v21::PfmV21GraphMatcherImpl& matcher,
                          const GraphMatcherInferenceOptions& graph_options);

/// 使用与 Python `pytorch_cache_match_eval.py` raw mutual 模式一致的描述子互最近邻匹配。
/// @param features_a 第一幅影像的特征集合。
/// @param features_b 第二幅影像的特征集合。
/// @param max_matches 最多输出的稀疏匹配数量，必须为正。
/// @return MatchSet，仅包含稀疏匹配索引和分数；半稠密点输出为空。
/// @throws std::invalid_argument 当描述子维度非法或 max_matches 非正时抛出。
MatchSet matchFeatureSetsPythonRawMutual(const FeatureSet& features_a, const FeatureSet& features_b,
                                         int64_t max_matches = 512);

/// 使用默认已学习行星图匹配器匹配两组已解码特征。
/// @param features_a 第一幅影像的特征集合。
/// @param features_b 第二幅影像的特征集合。
/// @return MatchSet，包含已学习稀疏索引对/分数以及半稠密点对应关系/置信度。
/// @throws std::invalid_argument 当必需稀疏或稠密张量未定义，或维度非法时抛出。
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b);

} // namespace pfm
