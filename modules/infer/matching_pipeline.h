#pragma once

#include "infer/feature_codec.h"
#include "infer/match_codec.h"

namespace pfm {

/// Matches two decoded feature sets with sparse mutual nearest-neighbor cross-check and semi-dense pairing.
/// @param features_a First image feature set.
/// @param features_b Second image feature set.
/// @return MatchSet containing sparse index pairs/scores and semi-dense point correspondences/confidence.
/// @throws std::invalid_argument if required sparse or dense tensors are undefined or have invalid dimensions.
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b);

}  // namespace pfm
