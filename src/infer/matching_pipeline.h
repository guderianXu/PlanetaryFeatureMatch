#pragma once

#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"
#include "models/pfm_model_v21.h"
#include "models/planetary_graph_matcher.h"

namespace pfm
{

/// 使用已学习的行星图匹配器匹配两组已解码特征。
/// @param features_a 第一幅影像的特征集合。
/// @param features_b 第二幅影像的特征集合。
/// @param matcher 已学习的图匹配模块。
/// @return MatchSet，包含已学习稀疏索引对/分数以及半稠密点对应关系/置信度。
/// @throws std::invalid_argument 当必需稀疏或稠密张量未定义，或维度非法时抛出。
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b,
                          PlanetaryGraphMatcherImpl& matcher);

/// 使用 v2.1 行星图匹配器匹配两组已解码特征。
/// @param features_a 第一幅影像的特征集合。
/// @param features_b 第二幅影像的特征集合。
/// @param matcher v2.1 已学习图匹配模块。
/// @return MatchSet，包含已学习稀疏索引对/分数以及半稠密点对应关系/置信度。
/// @throws std::invalid_argument 当必需稀疏或稠密张量未定义，或维度非法时抛出。
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b,
                          v21::PfmV21GraphMatcherImpl& matcher);

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
