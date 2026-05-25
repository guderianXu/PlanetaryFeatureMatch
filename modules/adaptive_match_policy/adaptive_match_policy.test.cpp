#include "tests/test_harness.h"

#include <limits>
#include <string>

#include "adaptive_match_policy/adaptive_match_policy.h"

namespace {

using pfm::adaptive_match_policy::AdaptiveMatchPolicyConfig;
using pfm::adaptive_match_policy::DescriptorTopKBand;
using pfm::adaptive_match_policy::MatchObservation;
using pfm::adaptive_match_policy::selectDescriptorTopKPolicy;
using pfm::adaptive_match_policy::verifiedFraction;

void highConfidenceStatsChooseConservativeTopK() {
    MatchObservation observation;
    observation.tentative_match_count = 150;
    observation.verified_match_count = 115;
    observation.spatial_coverage = 0.62;
    observation.mean_descriptor_margin = 0.24;

    const auto decision = selectDescriptorTopKPolicy(observation);

    PFM_REQUIRE(decision.band == DescriptorTopKBand::Conservative);
    PFM_REQUIRE(decision.descriptor_top_k == 32);
    PFM_REQUIRE(!decision.prefer_reciprocal_check);
    PFM_REQUIRE(!decision.request_geometric_retry);
    PFM_REQUIRE(decision.reason == "stable_high_confidence");
    PFM_REQUIRE_CLOSE(verifiedFraction(observation), 115.0 / 150.0, 1.0e-12);
}

void sparseSupportChoosesLooseTopKForRecovery() {
    MatchObservation observation;
    observation.tentative_match_count = 28;
    observation.verified_match_count = 22;
    observation.spatial_coverage = 0.14;
    observation.mean_descriptor_margin = 0.21;

    const auto decision = selectDescriptorTopKPolicy(observation);

    PFM_REQUIRE(decision.band == DescriptorTopKBand::Loose);
    PFM_REQUIRE(decision.descriptor_top_k == 128);
    PFM_REQUIRE(decision.prefer_reciprocal_check);
    PFM_REQUIRE(decision.request_geometric_retry);
    PFM_REQUIRE(decision.reason == "insufficient_match_support");
}

void noisyCandidatePoolChoosesConservativeTopKWithSafeguards() {
    MatchObservation observation;
    observation.tentative_match_count = 220;
    observation.verified_match_count = 28;
    observation.spatial_coverage = 0.52;
    observation.mean_descriptor_margin = 0.05;

    const auto decision = selectDescriptorTopKPolicy(observation);

    PFM_REQUIRE(decision.band == DescriptorTopKBand::Conservative);
    PFM_REQUIRE(decision.descriptor_top_k == 32);
    PFM_REQUIRE(decision.prefer_reciprocal_check);
    PFM_REQUIRE(decision.request_geometric_retry);
    PFM_REQUIRE(decision.reason == "noisy_candidate_pool");
}

void middlingStatsUseBalancedTopK() {
    MatchObservation observation;
    observation.tentative_match_count = 100;
    observation.verified_match_count = 46;
    observation.spatial_coverage = 0.35;
    observation.mean_descriptor_margin = 0.13;

    const auto decision = selectDescriptorTopKPolicy(observation);

    PFM_REQUIRE(decision.band == DescriptorTopKBand::Balanced);
    PFM_REQUIRE(decision.descriptor_top_k == 64);
    PFM_REQUIRE(decision.prefer_reciprocal_check);
    PFM_REQUIRE(!decision.request_geometric_retry);
    PFM_REQUIRE(decision.reason == "balanced_default");
}

void exportedExperimentConfigOverridesDescriptorTopK() {
    AdaptiveMatchPolicyConfig config;
    config.conservative_top_k = 16;
    config.balanced_top_k = 96;
    config.loose_top_k = 192;
    config.minimum_viable_matches = 16;
    config.healthy_match_count = 64;

    MatchObservation sparse;
    sparse.tentative_match_count = 12;
    sparse.verified_match_count = 10;
    sparse.spatial_coverage = 0.11;
    sparse.mean_descriptor_margin = 0.18;

    MatchObservation balanced;
    balanced.tentative_match_count = 70;
    balanced.verified_match_count = 31;
    balanced.spatial_coverage = 0.34;
    balanced.mean_descriptor_margin = 0.12;

    PFM_REQUIRE(selectDescriptorTopKPolicy(sparse, config).descriptor_top_k == 192);
    PFM_REQUIRE(selectDescriptorTopKPolicy(balanced, config).descriptor_top_k == 96);
}

void invalidStatisticsAndConfigAreRejected() {
    MatchObservation impossible_counts;
    impossible_counts.tentative_match_count = 8;
    impossible_counts.verified_match_count = 9;
    impossible_counts.spatial_coverage = 0.20;
    impossible_counts.mean_descriptor_margin = 0.10;
    PFM_REQUIRE_INVALID_ARG(selectDescriptorTopKPolicy(impossible_counts));

    MatchObservation invalid_coverage;
    invalid_coverage.tentative_match_count = 8;
    invalid_coverage.verified_match_count = 7;
    invalid_coverage.spatial_coverage = std::numeric_limits<double>::quiet_NaN();
    invalid_coverage.mean_descriptor_margin = 0.10;
    PFM_REQUIRE_INVALID_ARG(selectDescriptorTopKPolicy(invalid_coverage));

    AdaptiveMatchPolicyConfig invalid_config;
    invalid_config.conservative_top_k = 64;
    invalid_config.balanced_top_k = 32;
    MatchObservation valid_observation;
    valid_observation.tentative_match_count = 16;
    valid_observation.verified_match_count = 12;
    valid_observation.spatial_coverage = 0.20;
    valid_observation.mean_descriptor_margin = 0.10;
    PFM_REQUIRE_INVALID_ARG(selectDescriptorTopKPolicy(valid_observation, invalid_config));
}

}  // namespace

void register_adaptive_match_policy_tests() {
    register_test("adaptive match policy high confidence stats choose conservative top-k",
                  highConfidenceStatsChooseConservativeTopK);
    register_test("adaptive match policy sparse support chooses loose top-k for recovery",
                  sparseSupportChoosesLooseTopKForRecovery);
    register_test("adaptive match policy noisy candidate pool chooses conservative top-k with safeguards",
                  noisyCandidatePoolChoosesConservativeTopKWithSafeguards);
    register_test("adaptive match policy middling stats use balanced top-k", middlingStatsUseBalancedTopK);
    register_test("adaptive match policy exported experiment config overrides descriptor top-k",
                  exportedExperimentConfigOverridesDescriptorTopK);
    register_test("adaptive match policy invalid statistics and config are rejected",
                  invalidStatisticsAndConfigAreRejected);
}
