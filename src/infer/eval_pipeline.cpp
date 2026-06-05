#include "infer/eval_pipeline.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <torch/serialize.h>
#include <torch/torch.h>

#include "eval/metrics.h"

namespace pfm
{
namespace
{

float tensorAverageOrZero(const torch::Tensor& tensor)
{
    if (!tensor.defined() || tensor.numel() == 0)
    {
        return 0.0F;
    }
    return tensor.to(torch::kCPU, torch::kFloat32).mean().item<float>();
}

std::string parsePathToken(std::istringstream& stream)
{
    stream >> std::ws;
    if (!stream)
    {
        return {};
    }
    if (stream.peek() != '"')
    {
        std::string token;
        stream >> token;
        return token;
    }

    stream.get();
    std::string token;
    std::getline(stream, token, '"');
    return token;
}

} // namespace

std::vector<std::pair<std::string, std::string>> loadEvalPairs(const std::string& path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::invalid_argument("failed to open pairs file: " + path);
    }

    std::vector<std::pair<std::string, std::string>> pairs;
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream stream(line);
        auto image_a = parsePathToken(stream);
        auto image_b = parsePathToken(stream);
        if (!image_a.empty() && !image_b.empty())
        {
            pairs.push_back(std::make_pair(image_a, image_b));
        }
    }
    if (pairs.empty())
    {
        throw std::invalid_argument("pairs file is empty: " + path);
    }
    return pairs;
}

EvalReport aggregateEvalReport(const std::vector<std::pair<FeatureSet, FeatureSet>>& feature_sets,
                               const std::vector<MatchSet>& match_sets)
{
    if (feature_sets.empty())
    {
        throw std::invalid_argument("evaluation inputs must not be empty");
    }
    if (feature_sets.size() != match_sets.size())
    {
        throw std::invalid_argument("feature and match counts must match");
    }

    double total_matches = 0.0;
    double total_sparse_score = 0.0;
    double total_dense_confidence = 0.0;
    double total_coverage = 0.0;
    double total_half_turn_consistency = 0.0;
    double total_half_turn_mean_error = 0.0;
    double total_graph_executed_layers = 0.0;
    double total_graph_input_keypoints_a = 0.0;
    double total_graph_input_keypoints_b = 0.0;
    double total_graph_kept_keypoints_a = 0.0;
    double total_graph_kept_keypoints_b = 0.0;
    double total_graph_pruned_keypoints = 0.0;
    double total_graph_input_keypoints = 0.0;
    double total_graph_attention_work_units = 0.0;
    double total_graph_full_attention_work_units = 0.0;
    for (size_t index = 0; index < match_sets.size(); ++index)
    {
        const auto& features_a = feature_sets[index].first;
        const auto& features_b = feature_sets[index].second;
        const auto& matches = match_sets[index];
        total_matches += static_cast<double>(matches.sparse_matches.size(0));
        total_sparse_score += static_cast<double>(tensorAverageOrZero(matches.sparse_scores));
        total_dense_confidence += static_cast<double>(tensorAverageOrZero(matches.confidence));
        total_graph_executed_layers += static_cast<double>(matches.graph_executed_layers);
        total_graph_input_keypoints_a += static_cast<double>(matches.graph_input_keypoints_a);
        total_graph_input_keypoints_b += static_cast<double>(matches.graph_input_keypoints_b);
        total_graph_kept_keypoints_a += static_cast<double>(matches.graph_kept_keypoints_a);
        total_graph_kept_keypoints_b += static_cast<double>(matches.graph_kept_keypoints_b);
        total_graph_pruned_keypoints +=
            static_cast<double>(matches.graph_pruned_keypoints_a + matches.graph_pruned_keypoints_b);
        total_graph_input_keypoints +=
            static_cast<double>(matches.graph_input_keypoints_a + matches.graph_input_keypoints_b);
        total_graph_attention_work_units += static_cast<double>(matches.graph_attention_work_units);
        total_graph_full_attention_work_units += static_cast<double>(matches.graph_full_attention_work_units);
        const int64_t dense_base = std::max<int64_t>(features_a.dense_points.size(0), 1);
        total_coverage += static_cast<double>(matches.points_a.size(0)) / static_cast<double>(dense_base);
        if (features_b.feature_map_width > 0 && features_b.feature_map_height > 0)
        {
            total_half_turn_consistency += static_cast<double>(half_turn_consistency(
                matches.points_a, matches.points_b, features_b.feature_map_width, features_b.feature_map_height, 2.0F));
            total_half_turn_mean_error += static_cast<double>(half_turn_mean_error(
                matches.points_a, matches.points_b, features_b.feature_map_width, features_b.feature_map_height));
        }
    }

    const double pair_count = static_cast<double>(match_sets.size());
    EvalReport report;
    report.average_matches = total_matches / pair_count;
    report.average_sparse_score = total_sparse_score / pair_count;
    report.average_dense_confidence = total_dense_confidence / pair_count;
    report.semi_dense_coverage = total_coverage / pair_count;
    report.half_turn_consistency = total_half_turn_consistency / pair_count;
    report.half_turn_mean_error = total_half_turn_mean_error / pair_count;
    report.average_graph_executed_layers = total_graph_executed_layers / pair_count;
    report.average_graph_input_keypoints_a = total_graph_input_keypoints_a / pair_count;
    report.average_graph_input_keypoints_b = total_graph_input_keypoints_b / pair_count;
    report.average_graph_kept_keypoints_a = total_graph_kept_keypoints_a / pair_count;
    report.average_graph_kept_keypoints_b = total_graph_kept_keypoints_b / pair_count;
    report.graph_pruned_keypoint_fraction = total_graph_input_keypoints <= 0.0
                                                 ? 0.0
                                                 : total_graph_pruned_keypoints / total_graph_input_keypoints;
    report.graph_attention_work_fraction =
        total_graph_full_attention_work_units <= 0.0
            ? 0.0
            : total_graph_attention_work_units / total_graph_full_attention_work_units;
    return report;
}

void saveEvalReport(const std::string& path, const EvalReport& report)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::serialize::OutputArchive archive;
    auto write_scalar = [&](const char* name, double value)
    {
        archive.write(name, torch::tensor({static_cast<float>(value)}, options));
    };
    write_scalar("average_matches", report.average_matches);
    write_scalar("average_sparse_score", report.average_sparse_score);
    write_scalar("average_dense_confidence", report.average_dense_confidence);
    write_scalar("semi_dense_coverage", report.semi_dense_coverage);
    write_scalar("half_turn_consistency", report.half_turn_consistency);
    write_scalar("half_turn_mean_error", report.half_turn_mean_error);
    write_scalar("average_graph_executed_layers", report.average_graph_executed_layers);
    write_scalar("average_graph_input_keypoints_a", report.average_graph_input_keypoints_a);
    write_scalar("average_graph_input_keypoints_b", report.average_graph_input_keypoints_b);
    write_scalar("average_graph_kept_keypoints_a", report.average_graph_kept_keypoints_a);
    write_scalar("average_graph_kept_keypoints_b", report.average_graph_kept_keypoints_b);
    write_scalar("graph_pruned_keypoint_fraction", report.graph_pruned_keypoint_fraction);
    write_scalar("graph_attention_work_fraction", report.graph_attention_work_fraction);
    archive.save_to(path);
}

} // namespace pfm
