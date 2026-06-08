#pragma once

#include <cstdint>

#include <torch/torch.h>

#include "data/pair_archive_dataset.h"
#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"

namespace pfm
{

struct PythonDescriptorGridConfig
{
    /// 最多从 descriptor map 中选出的关键点数。
    int64_t max_keypoints = 4096;
    /// 原图强度阈值；和 Python pytorch_cache_match_eval.py 的 min_intensity 保持一致。
    double min_intensity = 0.01;
    /// 强纹理关键点比例；当前 C++ 批量对齐工具主要使用 Python 默认值 1.0。
    double texture_fraction = 1.0;
    /// 弱纹理关键点比例；当前实现支持 0.0 默认路径。
    double weak_texture_fraction = 0.0;
};

struct CacheRawMutualEvalResult
{
    int64_t matches = 0;
    int64_t correct = 0;
    int64_t wrong = 0;
    double precision = 0.0;
};

/// 按 Python pytorch_cache_match_eval.py 的 descriptor-grid 规则从 descriptor map 构造 FeatureSet。
/// @param image CxHxW 输入影像。
/// @param descriptors 1xDxHfxWf descriptor map。
/// @param config Python-style 选点参数。
/// @return 稀疏关键点和描述子；坐标位于 descriptor map 空间。
/// @throws std::invalid_argument 当张量形状或参数非法时抛出。
FeatureSet makePythonDescriptorGridFeatureSet(const torch::Tensor& image, const torch::Tensor& descriptors,
                                              const PythonDescriptorGridConfig& config);

/// 使用 Python raw mutual 评估两个 descriptor map 在 pair archive warp 下的匹配数量和正确率。
/// @param pair pair cache 样本，包含 view_a/view_b/warp_a_to_b。
/// @param descriptors_a A 图 1xDxHfxWf descriptor map。
/// @param descriptors_b B 图 1xDxHfxWf descriptor map。
/// @param keypoint_config Python-style 选点参数。
/// @param max_matches 最多输出的 mutual 匹配数量。
/// @param threshold_px 正确匹配阈值，单位为原始 pair cache 像素。
/// @param min_descriptor_score 最低 descriptor cosine 分数，默认保留旧行为。
/// @return matches/correct/wrong/precision。
/// @throws std::invalid_argument 当输入非法时抛出。
CacheRawMutualEvalResult evaluatePythonRawMutualDescriptorMaps(const PairArchiveSample& pair,
                                                               const torch::Tensor& descriptors_a,
                                                               const torch::Tensor& descriptors_b,
                                                               const PythonDescriptorGridConfig& keypoint_config,
                                                               int64_t max_matches, double threshold_px,
                                                               double min_descriptor_score = -1.0);

} // namespace pfm
