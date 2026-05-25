#include "cache_eval/cache_eval_metrics.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace pfm::cache_eval {
namespace {

struct HardPairCandidate {
    MatchSummaryRow row;
};

double safeRatio(int64_t numerator, int64_t denominator) {
    if (denominator <= 0) {
        return 0.0;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

std::string csvEscape(const std::string& value) {
    const bool needs_quotes = value.find_first_of(",\"\n\r") != std::string::npos;
    if (!needs_quotes) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

void requireNonNegative(int64_t value, const char* name) {
    if (value < 0) {
        throw std::invalid_argument(std::string(name) + " must be non-negative");
    }
}

void requireFiniteNonNegative(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
}

void validateSummaryRow(const MatchSummaryRow& row) {
    requireNonNegative(row.pair_index, "pair_index");
    requireFiniteNonNegative(row.precision, "precision");
    requireNonNegative(row.correct, "correct");
    requireNonNegative(row.matches, "matches");
    if (row.correct > row.matches) {
        throw std::invalid_argument("correct must be <= matches");
    }
}

}  // namespace

std::vector<int64_t> selectHardPairIndices(
    const std::vector<MatchSummaryRow>& rows,
    const HardPairMiningOptions& options
) {
    requireNonNegative(options.min_matches, "min_matches");
    requireFiniteNonNegative(options.max_precision, "max_precision");
    if (options.limit == 0) {
        return {};
    }

    std::vector<HardPairCandidate> candidates;
    candidates.reserve(rows.size());
    for (const auto& row : rows) {
        validateSummaryRow(row);
        if (row.matches < options.min_matches || row.precision > options.max_precision) {
            continue;
        }
        candidates.push_back(HardPairCandidate{row});
    }

    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.row.precision != rhs.row.precision) {
            return lhs.row.precision < rhs.row.precision;
        }
        if (lhs.row.matches != rhs.row.matches) {
            return lhs.row.matches > rhs.row.matches;
        }
        return false;
    });

    std::vector<int64_t> indices;
    indices.reserve(std::min(options.limit, candidates.size()));
    std::unordered_set<int64_t> seen;
    for (const auto& candidate : candidates) {
        if (!seen.insert(candidate.row.pair_index).second) {
            continue;
        }
        indices.push_back(candidate.row.pair_index);
        if (indices.size() == options.limit) {
            break;
        }
    }
    return indices;
}

PairMetrics::PairMetrics(std::string pair_id) : _pair_id(std::move(pair_id)) {
    if (_pair_id.empty()) {
        throw std::invalid_argument("pair_id must not be empty");
    }
}

const std::string& PairMetrics::pairId() const {
    return _pair_id;
}

void PairMetrics::addMatches(int64_t total, int64_t correct) {
    requireNonNegative(total, "total");
    requireNonNegative(correct, "correct");
    if (correct > total) {
        throw std::invalid_argument("correct must be <= total");
    }
    _total_matches += total;
    _correct_matches += correct;
}

void PairMetrics::setFeatureCounts(int64_t image_a_features, int64_t image_b_features) {
    requireNonNegative(image_a_features, "image_a_features");
    requireNonNegative(image_b_features, "image_b_features");
    if (_matched_features_a > image_a_features || _matched_features_b > image_b_features) {
        throw std::invalid_argument("matched feature count must be <= feature count");
    }
    _features_a = image_a_features;
    _features_b = image_b_features;
}

void PairMetrics::setMatchedFeatureCounts(int64_t image_a_matched, int64_t image_b_matched) {
    requireNonNegative(image_a_matched, "image_a_matched");
    requireNonNegative(image_b_matched, "image_b_matched");
    if (image_a_matched > _features_a || image_b_matched > _features_b) {
        throw std::invalid_argument("matched feature count must be <= feature count");
    }
    _matched_features_a = image_a_matched;
    _matched_features_b = image_b_matched;
}

void PairMetrics::setFeatureCoverage(int64_t source_features, int64_t valid_warp_features, int64_t covered_features) {
    requireNonNegative(source_features, "source_features");
    requireNonNegative(valid_warp_features, "valid_warp_features");
    requireNonNegative(covered_features, "covered_features");
    if (valid_warp_features > source_features) {
        throw std::invalid_argument("valid_warp_features must be <= source_features");
    }
    if (covered_features > valid_warp_features) {
        throw std::invalid_argument("covered_features must be <= valid_warp_features");
    }
    _source_features = source_features;
    _valid_warp_features = valid_warp_features;
    _covered_features = covered_features;
}

void PairMetrics::addDescriptorQuery(bool top1_correct, int64_t one_based_rank) {
    if (one_based_rank <= 0) {
        throw std::invalid_argument("one_based_rank must be positive");
    }
    if (top1_correct && one_based_rank != 1) {
        throw std::invalid_argument("top1_correct requires rank 1");
    }
    addDescriptorQueries(1, top1_correct ? 1 : 0, 1, one_based_rank);
}

void PairMetrics::addDescriptorQueries(int64_t queries, int64_t top1_correct, int64_t rank_sum) {
    addDescriptorQueries(queries, top1_correct, queries, rank_sum);
}

void PairMetrics::addDescriptorQueries(
    int64_t queries,
    int64_t top1_correct,
    int64_t rank_observed,
    int64_t rank_sum
) {
    requireNonNegative(queries, "queries");
    requireNonNegative(top1_correct, "top1_correct");
    requireNonNegative(rank_observed, "rank_observed");
    requireNonNegative(rank_sum, "rank_sum");
    if (top1_correct > queries) {
        throw std::invalid_argument("top1_correct must be <= queries");
    }
    if (rank_observed > queries) {
        throw std::invalid_argument("rank_observed must be <= queries");
    }
    if (rank_sum < rank_observed) {
        throw std::invalid_argument("rank_sum must be at least rank_observed for one-based ranks");
    }
    if (top1_correct > rank_observed) {
        throw std::invalid_argument("top1_correct must be <= rank_observed");
    }

    _descriptor_queries += queries;
    _descriptor_top1 += top1_correct;
    _descriptor_rank_observed += rank_observed;
    _descriptor_rank_sum += rank_sum;
}

int64_t PairMetrics::totalMatches() const {
    return _total_matches;
}

int64_t PairMetrics::correctMatches() const {
    return _correct_matches;
}

double PairMetrics::precision() const {
    return safeRatio(_correct_matches, _total_matches);
}

int64_t PairMetrics::featureCountA() const {
    return _features_a;
}

int64_t PairMetrics::featureCountB() const {
    return _features_b;
}

int64_t PairMetrics::matchedFeatureCountA() const {
    return _matched_features_a;
}

int64_t PairMetrics::matchedFeatureCountB() const {
    return _matched_features_b;
}

double PairMetrics::coverageA() const {
    return safeRatio(_matched_features_a, _features_a);
}

double PairMetrics::coverageB() const {
    return safeRatio(_matched_features_b, _features_b);
}

int64_t PairMetrics::sourceFeatureCount() const {
    return _source_features;
}

int64_t PairMetrics::validWarpFeatureCount() const {
    return _valid_warp_features;
}

int64_t PairMetrics::coveredFeatureCount() const {
    return _covered_features;
}

double PairMetrics::featureCoverage() const {
    return safeRatio(_covered_features, _valid_warp_features);
}

int64_t PairMetrics::descriptorQueries() const {
    return _descriptor_queries;
}

int64_t PairMetrics::descriptorTop1Count() const {
    return _descriptor_top1;
}

double PairMetrics::descriptorTop1Accuracy() const {
    return safeRatio(_descriptor_top1, _descriptor_queries);
}

int64_t PairMetrics::descriptorRankObserved() const {
    return _descriptor_rank_observed;
}

int64_t PairMetrics::descriptorRankSum() const {
    return _descriptor_rank_sum;
}

double PairMetrics::meanDescriptorRank() const {
    return safeRatio(_descriptor_rank_sum, _descriptor_rank_observed);
}

std::string PairMetrics::csvHeader() {
    return "pair_id,total_matches,correct_matches,precision,features_a,features_b,"
           "matched_features_a,matched_features_b,coverage_a,coverage_b,"
           "source_features,valid_warp_features,covered_features,feature_coverage,"
           "descriptor_queries,descriptor_top1,descriptor_top1_accuracy,"
           "descriptor_rank_observed,descriptor_rank_sum,mean_descriptor_rank";
}

std::string PairMetrics::csvRow() const {
    std::ostringstream out;
    out << std::setprecision(12)
        << csvEscape(_pair_id) << ','
        << _total_matches << ','
        << _correct_matches << ','
        << precision() << ','
        << _features_a << ','
        << _features_b << ','
        << _matched_features_a << ','
        << _matched_features_b << ','
        << coverageA() << ','
        << coverageB() << ','
        << _source_features << ','
        << _valid_warp_features << ','
        << _covered_features << ','
        << featureCoverage() << ','
        << _descriptor_queries << ','
        << _descriptor_top1 << ','
        << descriptorTop1Accuracy() << ','
        << _descriptor_rank_observed << ','
        << _descriptor_rank_sum << ','
        << meanDescriptorRank();
    return out.str();
}

void MetricsAccumulator::addPair(const PairMetrics& pair) {
    _pairs.push_back(pair);
}

const std::vector<PairMetrics>& MetricsAccumulator::pairs() const {
    return _pairs;
}

PairMetrics MetricsAccumulator::summary() const {
    PairMetrics aggregate("ALL");
    for (const auto& pair : _pairs) {
        aggregate.addMatches(pair.totalMatches(), pair.correctMatches());
        aggregate.setFeatureCounts(
            aggregate.featureCountA() + pair.featureCountA(),
            aggregate.featureCountB() + pair.featureCountB());
        aggregate.setMatchedFeatureCounts(
            aggregate.matchedFeatureCountA() + pair.matchedFeatureCountA(),
            aggregate.matchedFeatureCountB() + pair.matchedFeatureCountB());
        aggregate.setFeatureCoverage(
            aggregate.sourceFeatureCount() + pair.sourceFeatureCount(),
            aggregate.validWarpFeatureCount() + pair.validWarpFeatureCount(),
            aggregate.coveredFeatureCount() + pair.coveredFeatureCount());
        aggregate.addDescriptorQueries(
            pair.descriptorQueries(),
            pair.descriptorTop1Count(),
            pair.descriptorRankObserved(),
            pair.descriptorRankSum());
    }
    return aggregate;
}

std::string MetricsAccumulator::csvHeader() const {
    return PairMetrics::csvHeader();
}

std::string MetricsAccumulator::csvTable(bool include_summary) const {
    std::ostringstream out;
    out << csvHeader() << '\n';
    for (const auto& pair : _pairs) {
        out << pair.csvRow() << '\n';
    }
    if (include_summary) {
        out << summary().csvRow() << '\n';
    }
    return out.str();
}

}  // namespace pfm::cache_eval
