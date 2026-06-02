#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unordered_set>

#include "cache_eval/cache_eval_manifest.h"
#include "cache_eval/cache_eval_metrics.h"
#include "cache_eval/cache_eval_quality.h"
#include "cache_eval/cache_eval_training.h"
#include "infer/feature_codec.h"
#include "infer/match_codec.h"
#include "infer/match_metrics.h"

namespace
{

struct Options
{
    std::string feature_a;
    std::string feature_b;
    std::string matches;
    std::string warp;
    std::string manifest;
    std::string pair_id;
    std::string output;
    std::string quality_output;
    std::string hard_output;
    std::string hard_manifest_output;
    std::string hard_cache_index_output;
    pfm::cache_eval::QualityThresholds thresholds;
    std::size_t hard_limit = 0;
    double threshold = 5.0;
    bool include_summary = false;
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        auto require_value = [&](const char* name) -> std::string
        {
            if (index + 1 >= argc)
            {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (arg == "--feature-a")
        {
            options.feature_a = require_value("--feature-a");
        }
        else if (arg == "--feature-b")
        {
            options.feature_b = require_value("--feature-b");
        }
        else if (arg == "--matches")
        {
            options.matches = require_value("--matches");
        }
        else if (arg == "--warp-a-to-b")
        {
            options.warp = require_value("--warp-a-to-b");
        }
        else if (arg == "--manifest")
        {
            options.manifest = require_value("--manifest");
        }
        else if (arg == "--pair-id")
        {
            options.pair_id = require_value("--pair-id");
        }
        else if (arg == "--output")
        {
            options.output = require_value("--output");
        }
        else if (arg == "--quality-output")
        {
            options.quality_output = require_value("--quality-output");
        }
        else if (arg == "--hard-output")
        {
            options.hard_output = require_value("--hard-output");
        }
        else if (arg == "--hard-manifest-output")
        {
            options.hard_manifest_output = require_value("--hard-manifest-output");
        }
        else if (arg == "--hard-cache-index-output")
        {
            options.hard_cache_index_output = require_value("--hard-cache-index-output");
        }
        else if (arg == "--hard-limit")
        {
            options.hard_limit = static_cast<std::size_t>(std::stoull(require_value("--hard-limit")));
        }
        else if (arg == "--min-total-matches")
        {
            options.thresholds.min_total_matches = std::stoll(require_value("--min-total-matches"));
        }
        else if (arg == "--min-correct-matches")
        {
            options.thresholds.min_correct_matches = std::stoll(require_value("--min-correct-matches"));
        }
        else if (arg == "--min-precision")
        {
            options.thresholds.min_precision = std::stod(require_value("--min-precision"));
        }
        else if (arg == "--min-feature-coverage")
        {
            options.thresholds.min_feature_coverage = std::stod(require_value("--min-feature-coverage"));
        }
        else if (arg == "--min-descriptor-top1")
        {
            options.thresholds.min_descriptor_top1_accuracy = std::stod(require_value("--min-descriptor-top1"));
        }
        else if (arg == "--max-mean-descriptor-rank")
        {
            options.thresholds.max_mean_descriptor_rank = std::stod(require_value("--max-mean-descriptor-rank"));
        }
        else if (arg == "--threshold-px")
        {
            options.threshold = std::stod(require_value("--threshold-px"));
        }
        else if (arg == "--include-summary")
        {
            options.include_summary = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: pfm_cache_eval_report --feature-a a.pt --feature-b b.pt "
                         "--matches matches.pt --warp-a-to-b pair.pt [--pair-id id] "
                         "[--threshold-px 5] [--output report.csv] [--include-summary]\n"
                         "   or: pfm_cache_eval_report --manifest pairs.csv "
                         "[--threshold-px 5] [--output report.csv] [--include-summary]\n"
                         "Optional quality table: --quality-output quality.csv\n"
                         "Optional hard-pair selection: --hard-output hard.csv "
                         "[--hard-manifest-output hard_pairs_manifest.csv] "
                         "[--hard-cache-index-output hard_indices.csv] "
                         "[--hard-limit N] [--min-correct-matches N] [--min-precision X] "
                         "[--min-feature-coverage X] [--min-descriptor-top1 X] "
                         "[--max-mean-descriptor-rank X]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    const bool has_manifest = !options.manifest.empty();
    const bool has_single_pair =
        !options.feature_a.empty() || !options.feature_b.empty() || !options.matches.empty() || !options.warp.empty();
    if (has_manifest && has_single_pair)
    {
        throw std::invalid_argument("use either --manifest or single-pair feature/match/warp options, not both");
    }
    if (!has_manifest &&
        (options.feature_a.empty() || options.feature_b.empty() || options.matches.empty() || options.warp.empty()))
    {
        throw std::invalid_argument("feature-a, feature-b, matches, and warp-a-to-b are required");
    }
    if (options.pair_id.empty())
    {
        options.pair_id = std::filesystem::path(options.matches).stem().string();
    }
    return options;
}

std::vector<pfm::cache_eval::PairManifestEntry> make_entries(const Options& options)
{
    if (!options.manifest.empty())
    {
        return pfm::cache_eval::loadPairManifest(options.manifest);
    }
    return {pfm::cache_eval::PairManifestEntry{options.pair_id, options.feature_a, options.feature_b, options.matches,
                                               options.warp}};
}

int64_t keypoint_count(const pfm::FeatureSet& features)
{
    if (!features.keypoints.defined() || features.keypoints.dim() == 0)
    {
        return 0;
    }
    return features.keypoints.size(0);
}

std::pair<int64_t, int64_t> count_unique_sparse_features(const pfm::FeatureSet& features_a,
                                                         const pfm::FeatureSet& features_b,
                                                         const pfm::MatchSet& matches)
{
    if (!matches.sparse_matches.defined() || matches.sparse_matches.numel() == 0)
    {
        return {0, 0};
    }
    auto sparse = matches.sparse_matches.to(torch::kCPU, torch::kLong).contiguous();
    if (sparse.dim() != 2 || sparse.size(1) != 2)
    {
        throw std::invalid_argument("sparse_matches must have shape Nx2");
    }

    const auto count_a = keypoint_count(features_a);
    const auto count_b = keypoint_count(features_b);
    std::unordered_set<int64_t> used_a;
    std::unordered_set<int64_t> used_b;
    for (int64_t row = 0; row < sparse.size(0); ++row)
    {
        const auto index_a = sparse.index({row, 0}).item<int64_t>();
        const auto index_b = sparse.index({row, 1}).item<int64_t>();
        if (index_a >= 0 && index_a < count_a)
        {
            used_a.insert(index_a);
        }
        if (index_b >= 0 && index_b < count_b)
        {
            used_b.insert(index_b);
        }
    }
    return {static_cast<int64_t>(used_a.size()), static_cast<int64_t>(used_b.size())};
}

void write_report(const std::string& text, const std::string& output)
{
    if (output.empty())
    {
        std::cout << text;
        return;
    }

    const auto path = std::filesystem::path(output);
    if (path.has_parent_path())
    {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path);
    if (!stream)
    {
        throw std::invalid_argument("failed to open output csv: " + output);
    }
    stream << text;
}

std::string hard_pair_csv(const std::vector<pfm::cache_eval::HardPair>& hard_pairs)
{
    std::ostringstream out;
    out << "pair_id,hard_score,reason\n";
    for (const auto& pair : hard_pairs)
    {
        out << pair.pair_id << ',' << pair.hard_score << ',' << pair.reason << '\n';
    }
    return out.str();
}

std::vector<pfm::cache_eval::PairManifestEntry>
select_hard_manifest_entries(const std::vector<pfm::cache_eval::PairManifestEntry>& entries,
                             const std::vector<pfm::cache_eval::HardPair>& hard_pairs)
{
    std::vector<pfm::cache_eval::PairManifestEntry> selected;
    selected.reserve(hard_pairs.size());
    for (const auto& hard_pair : hard_pairs)
    {
        const auto iter = std::find_if(entries.begin(), entries.end(),
                                       [&](const auto& entry)
                                       {
                                           return entry.pair_id == hard_pair.pair_id;
                                       });
        if (iter != entries.end())
        {
            selected.push_back(*iter);
        }
    }
    return selected;
}

pfm::cache_eval::PairMetrics evaluate_pair(const pfm::cache_eval::PairManifestEntry& entry, double threshold)
{
    const auto features_a = pfm::load_feature_set(entry.feature_a);
    const auto features_b = pfm::load_feature_set(entry.feature_b);
    const auto matches = pfm::load_match_set(entry.matches);
    const auto warp = pfm::load_warp_a_to_b_tensor(entry.warp_a_to_b);

    const auto match_metrics = pfm::compute_warp_match_metrics(features_a, features_b, matches, warp, threshold);
    const auto coverage_metrics = pfm::compute_warp_feature_coverage_metrics(features_a, features_b, warp, threshold);
    const auto [matched_a, matched_b] = count_unique_sparse_features(features_a, features_b, matches);

    pfm::cache_eval::PairMetrics pair(entry.pair_id);
    pair.addMatches(match_metrics.total(), match_metrics.correct());
    pair.setFeatureCounts(keypoint_count(features_a), keypoint_count(features_b));
    pair.setMatchedFeatureCounts(matched_a, matched_b);
    pair.setFeatureCoverage(coverage_metrics.source_total, coverage_metrics.valid_warp_total,
                            coverage_metrics.covered_by_target_keypoint);
    pair.addDescriptorQueries(coverage_metrics.descriptor_rank_observed, coverage_metrics.descriptor_top1_count,
                              coverage_metrics.descriptor_rank_observed, coverage_metrics.descriptor_rank_sum);
    return pair;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        const auto entries = make_entries(options);
        pfm::cache_eval::MetricsAccumulator accumulator;
        for (const auto& entry : entries)
        {
            accumulator.addPair(evaluate_pair(entry, options.threshold));
        }
        write_report(accumulator.csvTable(options.include_summary), options.output);
        if (!options.quality_output.empty())
        {
            write_report(pfm::cache_eval::qualityDecisionsCsv(accumulator.pairs(), options.thresholds),
                         options.quality_output);
        }
        if (!options.hard_output.empty() || !options.hard_manifest_output.empty() ||
            !options.hard_cache_index_output.empty())
        {
            const auto hard_pairs =
                pfm::cache_eval::selectHardPairs(accumulator.pairs(), options.thresholds, options.hard_limit);
            const auto hard_manifest_entries = select_hard_manifest_entries(entries, hard_pairs);
            if (!options.hard_output.empty())
            {
                write_report(hard_pair_csv(hard_pairs), options.hard_output);
            }
            if (!options.hard_manifest_output.empty() && !hard_pairs.empty())
            {
                pfm::cache_eval::writePairManifest(hard_manifest_entries, options.hard_manifest_output);
            }
            else if (!options.hard_manifest_output.empty())
            {
                write_report("pair_id,feature_a,feature_b,matches,warp_a_to_b\n", options.hard_manifest_output);
            }
            if (!options.hard_cache_index_output.empty())
            {
                write_report(pfm::cache_eval::hardCacheIndexCsv(
                                 pfm::cache_eval::extractSyntheticPairCacheIndices(hard_manifest_entries)),
                             options.hard_cache_index_output);
            }
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "cache eval report failed: " << e.what() << '\n';
        return 1;
    }
}
