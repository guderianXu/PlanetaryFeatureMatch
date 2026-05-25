#include "tests/test_harness.h"

#include <sstream>
#include <string>
#include <vector>

#include "cache_eval/cache_eval_manifest.h"
#include "cache_eval/cache_eval_metrics.h"
#include "cache_eval/cache_eval_quality.h"
#include "cache_eval/cache_eval_training.h"

namespace {

pfm::cache_eval::PairMetrics makePairA() {
    pfm::cache_eval::PairMetrics pair("pair,a");
    pair.addMatches(10, 7);
    pair.setFeatureCounts(100, 80);
    pair.setMatchedFeatureCounts(50, 32);
    pair.setFeatureCoverage(12, 10, 6);
    pair.addDescriptorQuery(true, 1);
    pair.addDescriptorQuery(false, 4);
    return pair;
}

pfm::cache_eval::PairMetrics makePairB() {
    pfm::cache_eval::PairMetrics pair("pair_b");
    pair.addMatches(5, 5);
    pair.setFeatureCounts(20, 10);
    pair.setMatchedFeatureCounts(10, 8);
    pair.setFeatureCoverage(8, 5, 5);
    pair.addDescriptorQueries(2, 2, 3);
    return pair;
}

void pairMetricsComputesRatiosAndCsvRows() {
    const auto pair = makePairA();

    PFM_REQUIRE(pair.totalMatches() == 10);
    PFM_REQUIRE(pair.correctMatches() == 7);
    PFM_REQUIRE_CLOSE(pair.precision(), 0.7, 1.0e-12);
    PFM_REQUIRE_CLOSE(pair.coverageA(), 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(pair.coverageB(), 0.4, 1.0e-12);
    PFM_REQUIRE(pair.sourceFeatureCount() == 12);
    PFM_REQUIRE(pair.validWarpFeatureCount() == 10);
    PFM_REQUIRE(pair.coveredFeatureCount() == 6);
    PFM_REQUIRE_CLOSE(pair.featureCoverage(), 0.6, 1.0e-12);
    PFM_REQUIRE_CLOSE(pair.descriptorTop1Accuracy(), 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(pair.meanDescriptorRank(), 2.5, 1.0e-12);

    PFM_REQUIRE(
        pfm::cache_eval::PairMetrics::csvHeader() ==
        "pair_id,total_matches,correct_matches,precision,features_a,features_b,matched_features_a,matched_features_b,coverage_a,coverage_b,source_features,valid_warp_features,covered_features,feature_coverage,descriptor_queries,descriptor_top1,descriptor_top1_accuracy,descriptor_rank_observed,descriptor_rank_sum,mean_descriptor_rank");
    PFM_REQUIRE(pair.csvRow().find("\"pair,a\",10,7,0.7,100,80,50,32,0.5,0.4,12,10,6,0.6,2,1,0.5,2,5,2.5") == 0);
}

void accumulatorSummarizesPairsWithWeightedCounts() {
    pfm::cache_eval::MetricsAccumulator accumulator;
    accumulator.addPair(makePairA());
    accumulator.addPair(makePairB());

    const auto summary = accumulator.summary();

    PFM_REQUIRE(summary.pairId() == "ALL");
    PFM_REQUIRE(summary.totalMatches() == 15);
    PFM_REQUIRE(summary.correctMatches() == 12);
    PFM_REQUIRE_CLOSE(summary.precision(), 0.8, 1.0e-12);
    PFM_REQUIRE(summary.featureCountA() == 120);
    PFM_REQUIRE(summary.featureCountB() == 90);
    PFM_REQUIRE(summary.matchedFeatureCountA() == 60);
    PFM_REQUIRE(summary.matchedFeatureCountB() == 40);
    PFM_REQUIRE_CLOSE(summary.coverageA(), 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(summary.coverageB(), 40.0 / 90.0, 1.0e-12);
    PFM_REQUIRE(summary.sourceFeatureCount() == 20);
    PFM_REQUIRE(summary.validWarpFeatureCount() == 15);
    PFM_REQUIRE(summary.coveredFeatureCount() == 11);
    PFM_REQUIRE_CLOSE(summary.featureCoverage(), 11.0 / 15.0, 1.0e-12);
    PFM_REQUIRE(summary.descriptorQueries() == 4);
    PFM_REQUIRE(summary.descriptorTop1Count() == 3);
    PFM_REQUIRE_CLOSE(summary.descriptorTop1Accuracy(), 0.75, 1.0e-12);
    PFM_REQUIRE(summary.descriptorRankObserved() == 4);
    PFM_REQUIRE(summary.descriptorRankSum() == 8);
    PFM_REQUIRE_CLOSE(summary.meanDescriptorRank(), 2.0, 1.0e-12);
}

void accumulatorWritesCsvTableWithOptionalSummary() {
    pfm::cache_eval::MetricsAccumulator accumulator;
    accumulator.addPair(makePairA());
    accumulator.addPair(makePairB());

    const auto table = accumulator.csvTable(true);

    PFM_REQUIRE(table.find(pfm::cache_eval::PairMetrics::csvHeader() + "\n") == 0);
    PFM_REQUIRE(table.find("\"pair,a\",10,7") != std::string::npos);
    PFM_REQUIRE(table.find("pair_b,5,5") != std::string::npos);
    PFM_REQUIRE(table.find("ALL,15,12,0.8") != std::string::npos);
}

void invalidCountersAreRejected() {
    pfm::cache_eval::PairMetrics pair("bad_pair");

    PFM_REQUIRE_INVALID_ARG(pair.addMatches(-1, 0));
    PFM_REQUIRE_INVALID_ARG(pair.addMatches(2, 3));
    PFM_REQUIRE_INVALID_ARG(pair.setFeatureCounts(-1, 10));
    PFM_REQUIRE_INVALID_ARG(pair.setMatchedFeatureCounts(11, 0));
    pair.setFeatureCounts(10, 5);
    PFM_REQUIRE_INVALID_ARG(pair.setMatchedFeatureCounts(11, 0));
    PFM_REQUIRE_INVALID_ARG(pair.setFeatureCoverage(-1, 0, 0));
    PFM_REQUIRE_INVALID_ARG(pair.setFeatureCoverage(1, 2, 0));
    PFM_REQUIRE_INVALID_ARG(pair.setFeatureCoverage(2, 1, 2));
    PFM_REQUIRE_INVALID_ARG(pair.addDescriptorQuery(true, 0));
    PFM_REQUIRE_INVALID_ARG(pair.addDescriptorQueries(1, 2, 1));
}

void hardPairMiningFiltersByMinMatchesAndMaxPrecision() {
    const std::vector<pfm::cache_eval::MatchSummaryRow> rows{
        {10, 0.10, 1, 4},
        {11, 0.80, 8, 10},
        {12, 0.30, 3, 10}};

    pfm::cache_eval::HardPairMiningOptions options;
    options.limit = 4;
    options.min_matches = 5;
    options.max_precision = 0.50;

    const auto indices = pfm::cache_eval::selectHardPairIndices(rows, options);

    PFM_REQUIRE(indices == std::vector<int64_t>({12}));
}

void hardPairMiningPrioritizesLowPrecision() {
    const std::vector<pfm::cache_eval::MatchSummaryRow> rows{
        {20, 0.80, 8, 10},
        {21, 0.20, 2, 10},
        {22, 0.50, 5, 10}};

    pfm::cache_eval::HardPairMiningOptions options;
    options.limit = 3;

    const auto indices = pfm::cache_eval::selectHardPairIndices(rows, options);

    PFM_REQUIRE(indices == std::vector<int64_t>({21, 22, 20}));
}

void hardPairMiningBreaksPrecisionTiesByMoreMatches() {
    const std::vector<pfm::cache_eval::MatchSummaryRow> rows{
        {30, 0.40, 4, 10},
        {31, 0.40, 8, 20},
        {32, 0.40, 2, 5}};

    pfm::cache_eval::HardPairMiningOptions options;
    options.limit = 3;

    const auto indices = pfm::cache_eval::selectHardPairIndices(rows, options);

    PFM_REQUIRE(indices == std::vector<int64_t>({31, 30, 32}));
}

void hardPairMiningDeduplicatesIndicesStably() {
    const std::vector<pfm::cache_eval::MatchSummaryRow> rows{
        {40, 0.20, 2, 10},
        {41, 0.20, 2, 10},
        {40, 0.20, 2, 10},
        {42, 0.20, 2, 10}};

    pfm::cache_eval::HardPairMiningOptions options;
    options.limit = 10;

    const auto indices = pfm::cache_eval::selectHardPairIndices(rows, options);

    PFM_REQUIRE(indices == std::vector<int64_t>({40, 41, 42}));
}

void manifestParserReadsHeaderCommentsAndQuotedPaths() {
    std::istringstream input(
        "# cache eval input\n"
        "pair_id,feature_a,feature_b,matches,warp_a_to_b\n"
        "pair_000,\"features/a,0.pt\",features/b0.pt,matches/0.pt,warp/0.pt\n"
        "pair_001,features/a1.pt,features/b1.pt,matches/1.pt,warp/1.pt\n");

    const auto pairs = pfm::cache_eval::parsePairManifest(input);

    PFM_REQUIRE(pairs.size() == 2);
    PFM_REQUIRE(pairs[0].pair_id == "pair_000");
    PFM_REQUIRE(pairs[0].feature_a == "features/a,0.pt");
    PFM_REQUIRE(pairs[0].feature_b == "features/b0.pt");
    PFM_REQUIRE(pairs[0].matches == "matches/0.pt");
    PFM_REQUIRE(pairs[0].warp_a_to_b == "warp/0.pt");
    PFM_REQUIRE(pairs[1].pair_id == "pair_001");
}

void manifestWriterEscapesAndRoundTripsRows() {
    const std::vector<pfm::cache_eval::PairManifestEntry> entries{
        {"pair,0", "features/a\"0.pt", "features/b0.pt", "matches/0.pt", "warp/0.pt"},
        {"pair_1", "features/a1.pt", "features/b1.pt", "matches/1.pt", "warp/1.pt"}};

    const auto csv = pfm::cache_eval::pairManifestCsv(entries);
    std::istringstream input(csv);
    const auto parsed = pfm::cache_eval::parsePairManifest(input);

    PFM_REQUIRE(csv.find("pair_id,feature_a,feature_b,matches,warp_a_to_b\n") == 0);
    PFM_REQUIRE(csv.find("\"pair,0\",\"features/a\"\"0.pt\"") != std::string::npos);
    PFM_REQUIRE(parsed.size() == 2);
    PFM_REQUIRE(parsed[0].pair_id == "pair,0");
    PFM_REQUIRE(parsed[0].feature_a == "features/a\"0.pt");
    PFM_REQUIRE(parsed[1].warp_a_to_b == "warp/1.pt");
}

void manifestParserRejectsMissingFields() {
    std::istringstream input("pair_000,a.pt,b.pt,matches.pt\n");

    PFM_REQUIRE_INVALID_ARG(pfm::cache_eval::parsePairManifest(input));
}

void qualityGateReportsFailingFields() {
    pfm::cache_eval::PairMetrics pair("bad_pair");
    pair.addMatches(20, 9);
    pair.setFeatureCounts(40, 30);
    pair.setMatchedFeatureCounts(10, 9);
    pair.setFeatureCoverage(40, 30, 12);
    pair.addDescriptorQueries(30, 8, 120);

    pfm::cache_eval::QualityThresholds thresholds;
    thresholds.min_total_matches = 16;
    thresholds.min_correct_matches = 12;
    thresholds.min_precision = 0.75;
    thresholds.min_feature_coverage = 0.6;
    thresholds.min_descriptor_top1_accuracy = 0.5;
    thresholds.max_mean_descriptor_rank = 3.0;

    const auto decision = pfm::cache_eval::evaluatePairQuality(pair, thresholds);

    PFM_REQUIRE(!decision.passed);
    PFM_REQUIRE(decision.failed_fields.size() == 5);
    PFM_REQUIRE(decision.reason.find("correct_matches") != std::string::npos);
    PFM_REQUIRE(decision.reason.find("precision") != std::string::npos);
    PFM_REQUIRE(decision.reason.find("feature_coverage") != std::string::npos);
    PFM_REQUIRE(decision.reason.find("descriptor_top1_accuracy") != std::string::npos);
    PFM_REQUIRE(decision.reason.find("mean_descriptor_rank") != std::string::npos);
    PFM_REQUIRE(decision.hard_score > 0.0);
}

void qualityGateAcceptsGoodPairsAndSelectsWorstHardPairs() {
    auto good = makePairA();
    good.addMatches(40, 33);
    good.setFeatureCoverage(60, 50, 45);
    good.addDescriptorQueries(50, 42, 65);

    auto low_precision = makePairB();
    low_precision.addMatches(30, 3);
    low_precision.setFeatureCoverage(40, 30, 24);
    low_precision.addDescriptorQueries(30, 20, 40);

    pfm::cache_eval::PairMetrics low_coverage("low_coverage");
    low_coverage.addMatches(30, 28);
    low_coverage.setFeatureCounts(30, 30);
    low_coverage.setMatchedFeatureCounts(20, 20);
    low_coverage.setFeatureCoverage(30, 30, 3);
    low_coverage.addDescriptorQueries(30, 25, 32);

    pfm::cache_eval::QualityThresholds thresholds;
    thresholds.min_correct_matches = 20;
    thresholds.min_precision = 0.7;
    thresholds.min_feature_coverage = 0.5;
    thresholds.min_descriptor_top1_accuracy = 0.5;
    thresholds.max_mean_descriptor_rank = 3.0;

    PFM_REQUIRE(pfm::cache_eval::evaluatePairQuality(good, thresholds).passed);

    const std::vector<pfm::cache_eval::PairMetrics> pairs{good, low_precision, low_coverage};
    const auto hard_pairs = pfm::cache_eval::selectHardPairs(pairs, thresholds, 2);

    PFM_REQUIRE(hard_pairs.size() == 2);
    PFM_REQUIRE(hard_pairs[0].pair_id == "pair_b");
    PFM_REQUIRE(hard_pairs[1].pair_id == "low_coverage");
    PFM_REQUIRE(hard_pairs[0].hard_score >= hard_pairs[1].hard_score);
}

void qualityDecisionCsvReportsEveryPair() {
    auto good = makePairA();
    good.addMatches(40, 35);
    good.setFeatureCoverage(40, 30, 28);
    good.addDescriptorQueries(30, 25, 35);

    pfm::cache_eval::PairMetrics bad("bad,pair");
    bad.addMatches(10, 2);
    bad.setFeatureCounts(20, 20);
    bad.setMatchedFeatureCounts(5, 4);
    bad.setFeatureCoverage(20, 20, 4);
    bad.addDescriptorQueries(20, 2, 90);

    pfm::cache_eval::QualityThresholds thresholds;
    thresholds.min_correct_matches = 10;
    thresholds.min_precision = 0.6;
    thresholds.min_feature_coverage = 0.5;
    thresholds.min_descriptor_top1_accuracy = 0.5;
    thresholds.max_mean_descriptor_rank = 3.0;

    const auto csv = pfm::cache_eval::qualityDecisionsCsv({good, bad}, thresholds);

    PFM_REQUIRE(csv.find("pair_id,passed,hard_score,reason\n") == 0);
    PFM_REQUIRE(csv.find("\"pair,a\",1,0,passed") != std::string::npos);
    PFM_REQUIRE(csv.find("\"bad,pair\",0,") != std::string::npos);
    PFM_REQUIRE(csv.find("correct_matches;precision;feature_coverage;descriptor_top1_accuracy;mean_descriptor_rank") !=
                std::string::npos);
}

void trainingIndexExportParsesStableUniquePairIndices() {
    const std::vector<pfm::cache_eval::PairManifestEntry> entries{
        {"pair_000077", "a.pt", "b.pt", "m.pt", "cache/source/pair_000077.pt"},
        {"custom_id", "a.pt", "b.pt", "m.pt", "/tmp/cache/source/pair_000003.pt"},
        {"pair_000077", "a.pt", "b.pt", "m.pt", "cache/source/pair_000077.pt"}};

    const auto indices = pfm::cache_eval::extractSyntheticPairCacheIndices(entries);
    const auto csv = pfm::cache_eval::hardCacheIndexCsv(indices);

    PFM_REQUIRE(indices == std::vector<int64_t>({77, 3}));
    PFM_REQUIRE(csv == "hard_cache_index\n77\n3\n");
}

void trainingIndexExportRejectsUnparseablePairIndex() {
    const pfm::cache_eval::PairManifestEntry entry{"bad", "a.pt", "b.pt", "m.pt", "cache/source/not_pair.pt"};

    PFM_REQUIRE_INVALID_ARG(pfm::cache_eval::extractSyntheticPairCacheIndex(entry));
}

}  // namespace

void register_cache_eval_tests() {
    register_test("cache eval pair metrics computes ratios and csv rows", pairMetricsComputesRatiosAndCsvRows);
    register_test("cache eval accumulator summarizes pairs with weighted counts", accumulatorSummarizesPairsWithWeightedCounts);
    register_test("cache eval accumulator writes csv table with optional summary", accumulatorWritesCsvTableWithOptionalSummary);
    register_test("cache eval invalid counters are rejected", invalidCountersAreRejected);
    register_test("cache eval hard pair mining filters by min matches and max precision",
                  hardPairMiningFiltersByMinMatchesAndMaxPrecision);
    register_test("cache eval hard pair mining prioritizes low precision", hardPairMiningPrioritizesLowPrecision);
    register_test("cache eval hard pair mining breaks precision ties by more matches",
                  hardPairMiningBreaksPrecisionTiesByMoreMatches);
    register_test("cache eval hard pair mining deduplicates indices stably", hardPairMiningDeduplicatesIndicesStably);
    register_test("cache eval manifest parser reads header comments and quoted paths",
                  manifestParserReadsHeaderCommentsAndQuotedPaths);
    register_test("cache eval manifest writer escapes and round trips rows", manifestWriterEscapesAndRoundTripsRows);
    register_test("cache eval manifest parser rejects missing fields", manifestParserRejectsMissingFields);
    register_test("cache eval quality gate reports failing fields", qualityGateReportsFailingFields);
    register_test("cache eval quality gate accepts good pairs and selects worst hard pairs",
                  qualityGateAcceptsGoodPairsAndSelectsWorstHardPairs);
    register_test("cache eval quality decision csv reports every pair", qualityDecisionCsvReportsEveryPair);
    register_test("cache eval training index export parses stable unique pair indices",
                  trainingIndexExportParsesStableUniquePairIndices);
    register_test("cache eval training index export rejects unparseable pair index",
                  trainingIndexExportRejectsUnparseablePairIndex);
}
