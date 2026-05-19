#pragma once

#include "infer/feature_codec.h"
#include "infer/match_codec.h"
#include "models/planetary_graph_matcher.h"

namespace pfm {

/// Matches two decoded feature sets with the learned planetary graph matcher.
/// @param features_a First image feature set.
/// @param features_b Second image feature set.
/// @param matcher Learned graph matcher module.
/// @return MatchSet containing learned sparse index pairs/scores and semi-dense point correspondences/confidence.
/// @throws std::invalid_argument if required sparse or dense tensors are undefined or have invalid dimensions.
MatchSet matchFeatureSets(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    PlanetaryGraphMatcherImpl& matcher
);

/// Matches two decoded feature sets with the default learned planetary graph matcher.
/// @param features_a First image feature set.
/// @param features_b Second image feature set.
/// @return MatchSet containing learned sparse index pairs/scores and semi-dense point correspondences/confidence.
/// @throws std::invalid_argument if required sparse or dense tensors are undefined or have invalid dimensions.
MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b);

}  // namespace pfm
