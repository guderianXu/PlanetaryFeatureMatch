#include "adaptive_match_policy/adaptive_match_policy.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace pfm::adaptive_match_policy
{
namespace
{

void requireFiniteUnit(const double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0 || value > 1.0)
    {
        throw std::invalid_argument(std::string(name) + " must be finite and in [0, 1]");
    }
}

void requireFiniteNonNegative(const double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
}

void validateObservation(const MatchObservation& observation)
{
    if (observation.verified_match_count > observation.tentative_match_count)
    {
        throw std::invalid_argument("verified_match_count cannot exceed tentative_match_count");
    }
    requireFiniteUnit(observation.spatial_coverage, "spatial_coverage");
    requireFiniteNonNegative(observation.mean_descriptor_margin, "mean_descriptor_margin");
}

void validateConfig(const AdaptiveMatchPolicyConfig& config)
{
    if (config.conservative_top_k <= 0 || config.balanced_top_k <= 0 || config.loose_top_k <= 0)
    {
        throw std::invalid_argument("descriptor top-k values must be positive");
    }
    if (config.conservative_top_k > config.balanced_top_k || config.balanced_top_k > config.loose_top_k)
    {
        throw std::invalid_argument("descriptor top-k values must be ordered conservative <= balanced <= loose");
    }
    if (config.minimum_viable_matches > config.healthy_match_count)
    {
        throw std::invalid_argument("minimum_viable_matches cannot exceed healthy_match_count");
    }
    requireFiniteUnit(config.weak_verified_fraction, "weak_verified_fraction");
    requireFiniteUnit(config.strong_verified_fraction, "strong_verified_fraction");
    if (config.weak_verified_fraction > config.strong_verified_fraction)
    {
        throw std::invalid_argument("weak_verified_fraction cannot exceed strong_verified_fraction");
    }
    requireFiniteUnit(config.weak_spatial_coverage, "weak_spatial_coverage");
    requireFiniteUnit(config.strong_spatial_coverage, "strong_spatial_coverage");
    if (config.weak_spatial_coverage > config.strong_spatial_coverage)
    {
        throw std::invalid_argument("weak_spatial_coverage cannot exceed strong_spatial_coverage");
    }
    requireFiniteNonNegative(config.confident_descriptor_margin, "confident_descriptor_margin");
}

double verifiedFractionUnchecked(const MatchObservation& observation)
{
    if (observation.tentative_match_count == 0)
    {
        return 0.0;
    }
    return static_cast<double>(observation.verified_match_count) /
           static_cast<double>(observation.tentative_match_count);
}

DescriptorTopKDecision makeDecision(const DescriptorTopKBand band, const int top_k, const bool prefer_reciprocal_check,
                                    const bool request_geometric_retry, std::string reason)
{
    DescriptorTopKDecision decision;
    decision.band = band;
    decision.descriptor_top_k = top_k;
    decision.prefer_reciprocal_check = prefer_reciprocal_check;
    decision.request_geometric_retry = request_geometric_retry;
    decision.reason = std::move(reason);
    return decision;
}

} // namespace

std::string descriptorTopKBandName(const DescriptorTopKBand band)
{
    switch (band)
    {
    case DescriptorTopKBand::Conservative:
        return "conservative";
    case DescriptorTopKBand::Balanced:
        return "balanced";
    case DescriptorTopKBand::Loose:
        return "loose";
    }
    throw std::invalid_argument("unknown descriptor top-k band");
}

double verifiedFraction(const MatchObservation& observation)
{
    validateObservation(observation);
    return verifiedFractionUnchecked(observation);
}

DescriptorTopKDecision selectDescriptorTopKPolicy(const MatchObservation& observation,
                                                  const AdaptiveMatchPolicyConfig& config)
{
    validateConfig(config);
    validateObservation(observation);

    const double verified_fraction = verifiedFractionUnchecked(observation);
    const bool noisy_candidate_pool = observation.tentative_match_count >= config.healthy_match_count &&
                                      verified_fraction < config.weak_verified_fraction;
    if (noisy_candidate_pool)
    {
        return makeDecision(DescriptorTopKBand::Conservative, config.conservative_top_k, true, true,
                            "noisy_candidate_pool");
    }

    const bool insufficient_match_support = observation.verified_match_count < config.minimum_viable_matches ||
                                            observation.spatial_coverage < config.weak_spatial_coverage;
    if (insufficient_match_support)
    {
        return makeDecision(DescriptorTopKBand::Loose, config.loose_top_k, true, true, "insufficient_match_support");
    }

    const bool stable_high_confidence = observation.verified_match_count >= config.healthy_match_count &&
                                        verified_fraction >= config.strong_verified_fraction &&
                                        observation.spatial_coverage >= config.strong_spatial_coverage &&
                                        observation.mean_descriptor_margin >= config.confident_descriptor_margin;
    if (stable_high_confidence)
    {
        return makeDecision(DescriptorTopKBand::Conservative, config.conservative_top_k, false, false,
                            "stable_high_confidence");
    }

    return makeDecision(DescriptorTopKBand::Balanced, config.balanced_top_k, true, false, "balanced_default");
}

} // namespace pfm::adaptive_match_policy
