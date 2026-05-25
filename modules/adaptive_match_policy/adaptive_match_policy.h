#pragma once

#include <cstddef>
#include <string>

namespace pfm::adaptive_match_policy {

enum class DescriptorTopKBand {
    Conservative,
    Balanced,
    Loose,
};

struct MatchObservation {
    std::size_t tentative_match_count = 0;
    std::size_t verified_match_count = 0;
    double spatial_coverage = 0.0;
    double mean_descriptor_margin = 0.0;
};

struct AdaptiveMatchPolicyConfig {
    int conservative_top_k = 32;
    int balanced_top_k = 64;
    int loose_top_k = 128;
    std::size_t minimum_viable_matches = 24;
    std::size_t healthy_match_count = 80;
    double weak_verified_fraction = 0.25;
    double strong_verified_fraction = 0.60;
    double weak_spatial_coverage = 0.18;
    double strong_spatial_coverage = 0.45;
    double confident_descriptor_margin = 0.18;
};

struct DescriptorTopKDecision {
    DescriptorTopKBand band = DescriptorTopKBand::Balanced;
    int descriptor_top_k = 64;
    bool prefer_reciprocal_check = true;
    bool request_geometric_retry = false;
    std::string reason;
};

std::string descriptorTopKBandName(DescriptorTopKBand band);
double verifiedFraction(const MatchObservation& observation);
DescriptorTopKDecision selectDescriptorTopKPolicy(
    const MatchObservation& observation,
    const AdaptiveMatchPolicyConfig& config = {}
);

}  // namespace pfm::adaptive_match_policy
