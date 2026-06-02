#include "cache_eval/cache_eval_quality.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace pfm::cache_eval
{
namespace
{

void requireFiniteNonNegative(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
}

void requireNonNegative(int64_t value, const char* name)
{
    if (value < 0)
    {
        throw std::invalid_argument(std::string(name) + " must be non-negative");
    }
}

void validateThresholds(const QualityThresholds& thresholds)
{
    requireNonNegative(thresholds.min_total_matches, "min_total_matches");
    requireNonNegative(thresholds.min_correct_matches, "min_correct_matches");
    requireFiniteNonNegative(thresholds.min_precision, "min_precision");
    requireFiniteNonNegative(thresholds.min_feature_coverage, "min_feature_coverage");
    requireFiniteNonNegative(thresholds.min_descriptor_top1_accuracy, "min_descriptor_top1_accuracy");
    requireFiniteNonNegative(thresholds.max_mean_descriptor_rank, "max_mean_descriptor_rank");
    if (thresholds.min_precision > 1.0)
    {
        throw std::invalid_argument("min_precision must be <= 1");
    }
    if (thresholds.min_feature_coverage > 1.0)
    {
        throw std::invalid_argument("min_feature_coverage must be <= 1");
    }
    if (thresholds.min_descriptor_top1_accuracy > 1.0)
    {
        throw std::invalid_argument("min_descriptor_top1_accuracy must be <= 1");
    }
}

double countDeficit(double actual, double required)
{
    if (required <= 0.0 || actual >= required)
    {
        return 0.0;
    }
    return (required - actual) / std::max(1.0, required);
}

double ratioDeficit(double actual, double required)
{
    if (required <= 0.0 || actual >= required)
    {
        return 0.0;
    }
    return (required - actual) / required;
}

double maxRatioExcess(double actual, double maximum)
{
    if (maximum <= 0.0 || actual <= maximum)
    {
        return 0.0;
    }
    return (actual - maximum) / maximum;
}

void addFailure(QualityDecision& decision, std::string field, double contribution)
{
    decision.passed = false;
    decision.hard_score += contribution;
    decision.failed_fields.push_back(std::move(field));
}

std::string joinReasons(const std::vector<std::string>& fields)
{
    std::ostringstream out;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        if (index > 0)
        {
            out << ';';
        }
        out << fields[index];
    }
    return out.str();
}

std::string csvEscape(const std::string& value)
{
    const bool needs_quotes = value.find_first_of(",\"\n\r") != std::string::npos;
    if (!needs_quotes)
    {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value)
    {
        if (ch == '"')
        {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

} // namespace

QualityDecision evaluatePairQuality(const PairMetrics& pair, const QualityThresholds& thresholds)
{
    validateThresholds(thresholds);

    QualityDecision decision;
    if (const auto deficit = countDeficit(pair.totalMatches(), thresholds.min_total_matches); deficit > 0.0)
    {
        addFailure(decision, "total_matches", deficit);
    }
    if (const auto deficit = countDeficit(pair.correctMatches(), thresholds.min_correct_matches); deficit > 0.0)
    {
        addFailure(decision, "correct_matches", deficit);
    }
    if (const auto deficit = ratioDeficit(pair.precision(), thresholds.min_precision); deficit > 0.0)
    {
        addFailure(decision, "precision", deficit);
    }
    if (const auto deficit = ratioDeficit(pair.featureCoverage(), thresholds.min_feature_coverage); deficit > 0.0)
    {
        addFailure(decision, "feature_coverage", deficit);
    }
    if (const auto deficit = ratioDeficit(pair.descriptorTop1Accuracy(), thresholds.min_descriptor_top1_accuracy);
        deficit > 0.0)
    {
        addFailure(decision, "descriptor_top1_accuracy", deficit);
    }
    if (const auto excess = maxRatioExcess(pair.meanDescriptorRank(), thresholds.max_mean_descriptor_rank);
        excess > 0.0)
    {
        addFailure(decision, "mean_descriptor_rank", excess);
    }
    decision.reason = decision.passed ? "passed" : joinReasons(decision.failed_fields);
    return decision;
}

std::vector<HardPair> selectHardPairs(const std::vector<PairMetrics>& pairs, const QualityThresholds& thresholds,
                                      std::size_t limit)
{
    std::vector<HardPair> hard_pairs;
    for (const auto& pair : pairs)
    {
        const auto decision = evaluatePairQuality(pair, thresholds);
        if (!decision.passed)
        {
            hard_pairs.push_back(HardPair{pair.pairId(), decision.hard_score, decision.reason});
        }
    }
    std::sort(hard_pairs.begin(), hard_pairs.end(),
              [](const HardPair& lhs, const HardPair& rhs)
              {
                  if (lhs.hard_score != rhs.hard_score)
                  {
                      return lhs.hard_score > rhs.hard_score;
                  }
                  return lhs.pair_id < rhs.pair_id;
              });
    if (limit > 0 && hard_pairs.size() > limit)
    {
        hard_pairs.resize(limit);
    }
    return hard_pairs;
}

std::string qualityDecisionsCsv(const std::vector<PairMetrics>& pairs, const QualityThresholds& thresholds)
{
    std::ostringstream out;
    out << "pair_id,passed,hard_score,reason\n";
    out << std::setprecision(12);
    for (const auto& pair : pairs)
    {
        const auto decision = evaluatePairQuality(pair, thresholds);
        out << csvEscape(pair.pairId()) << ',' << (decision.passed ? 1 : 0) << ',' << decision.hard_score << ','
            << csvEscape(decision.reason) << '\n';
    }
    return out.str();
}

} // namespace pfm::cache_eval
