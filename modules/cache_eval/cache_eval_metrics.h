#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pfm::cache_eval
{

struct MatchSummaryRow
{
    int64_t pair_index = 0;
    double precision = 0.0;
    int64_t correct = 0;
    int64_t matches = 0;
};

struct HardPairMiningOptions
{
    std::size_t limit = 0;
    int64_t min_matches = 0;
    double max_precision = 1.0;
};

std::vector<int64_t> selectHardPairIndices(const std::vector<MatchSummaryRow>& rows,
                                           const HardPairMiningOptions& options);

class PairMetrics
{
  public:
    explicit PairMetrics(std::string pair_id);

    const std::string& pairId() const;

    void addMatches(int64_t total, int64_t correct);
    void setFeatureCounts(int64_t image_a_features, int64_t image_b_features);
    void setMatchedFeatureCounts(int64_t image_a_matched, int64_t image_b_matched);
    void setFeatureCoverage(int64_t source_features, int64_t valid_warp_features, int64_t covered_features);
    void addDescriptorQuery(bool top1_correct, int64_t one_based_rank);
    void addDescriptorQueries(int64_t queries, int64_t top1_correct, int64_t rank_sum);
    void addDescriptorQueries(int64_t queries, int64_t top1_correct, int64_t rank_observed, int64_t rank_sum);

    int64_t totalMatches() const;
    int64_t correctMatches() const;
    double precision() const;

    int64_t featureCountA() const;
    int64_t featureCountB() const;
    int64_t matchedFeatureCountA() const;
    int64_t matchedFeatureCountB() const;
    double coverageA() const;
    double coverageB() const;
    int64_t sourceFeatureCount() const;
    int64_t validWarpFeatureCount() const;
    int64_t coveredFeatureCount() const;
    double featureCoverage() const;

    int64_t descriptorQueries() const;
    int64_t descriptorTop1Count() const;
    double descriptorTop1Accuracy() const;
    int64_t descriptorRankObserved() const;
    int64_t descriptorRankSum() const;
    double meanDescriptorRank() const;

    static std::string csvHeader();
    std::string csvRow() const;

  private:
    std::string _pair_id;
    int64_t _total_matches = 0;
    int64_t _correct_matches = 0;
    int64_t _features_a = 0;
    int64_t _features_b = 0;
    int64_t _matched_features_a = 0;
    int64_t _matched_features_b = 0;
    int64_t _source_features = 0;
    int64_t _valid_warp_features = 0;
    int64_t _covered_features = 0;
    int64_t _descriptor_queries = 0;
    int64_t _descriptor_top1 = 0;
    int64_t _descriptor_rank_observed = 0;
    int64_t _descriptor_rank_sum = 0;
};

class MetricsAccumulator
{
  public:
    void addPair(const PairMetrics& pair);

    const std::vector<PairMetrics>& pairs() const;
    PairMetrics summary() const;

    std::string csvHeader() const;
    std::string csvTable(bool include_summary) const;

  private:
    std::vector<PairMetrics> _pairs;
};

} // namespace pfm::cache_eval
