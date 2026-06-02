#pragma once

#include <string>
#include <utility>
#include <vector>

#include "infer/feature_codec.h"
#include "infer/match_codec.h"

namespace pfm
{

struct EvalReport
{
    double average_matches = 0.0;
    double average_sparse_score = 0.0;
    double average_dense_confidence = 0.0;
    double semi_dense_coverage = 0.0;
    double half_turn_consistency = 0.0;
    double half_turn_mean_error = 0.0;
};

/// Loads whitespace-separated image pairs from a text file.
/// @param path Source pairs file path.
/// @return Vector of image path pairs.
/// @throws std::invalid_argument if the file cannot be opened or contains no pairs.
std::vector<std::pair<std::string, std::string>> loadEvalPairs(const std::string& path);

/// Aggregates matching metrics over decoded feature and match sets.
/// @param feature_sets Feature set pairs used as evaluation denominators.
/// @param match_sets Match outputs corresponding one-to-one with feature_sets.
/// @return Average match count, sparse score, dense confidence, and semi-dense coverage.
/// @throws std::invalid_argument if inputs are empty or sizes differ.
EvalReport aggregateEvalReport(const std::vector<std::pair<FeatureSet, FeatureSet>>& feature_sets,
                               const std::vector<MatchSet>& match_sets);

/// Saves evaluation metrics to a LibTorch archive.
/// @param path Destination .pt report path.
/// @param report Metrics to serialize.
/// @throws std::invalid_argument if serialization fails.
void saveEvalReport(const std::string& path, const EvalReport& report);

} // namespace pfm
