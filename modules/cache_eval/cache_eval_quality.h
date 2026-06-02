#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cache_eval/cache_eval_metrics.h"

namespace pfm::cache_eval
{

struct QualityThresholds
{
    int64_t min_total_matches = 0;
    int64_t min_correct_matches = 0;
    double min_precision = 0.0;
    double min_feature_coverage = 0.0;
    double min_descriptor_top1_accuracy = 0.0;
    double max_mean_descriptor_rank = 0.0;
};

struct QualityDecision
{
    bool passed = true;
    std::vector<std::string> failed_fields;
    std::string reason;
    double hard_score = 0.0;
};

struct HardPair
{
    std::string pair_id;
    double hard_score = 0.0;
    std::string reason;
};

QualityDecision evaluatePairQuality(const PairMetrics& pair, const QualityThresholds& thresholds);

std::vector<HardPair> selectHardPairs(const std::vector<PairMetrics>& pairs, const QualityThresholds& thresholds,
                                      std::size_t limit);

std::string qualityDecisionsCsv(const std::vector<PairMetrics>& pairs, const QualityThresholds& thresholds);

} // namespace pfm::cache_eval
