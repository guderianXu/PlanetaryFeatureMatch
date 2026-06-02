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
    for (size_t index = 0; index < match_sets.size(); ++index)
    {
        const auto& features_a = feature_sets[index].first;
        const auto& features_b = feature_sets[index].second;
        const auto& matches = match_sets[index];
        total_matches += static_cast<double>(matches.sparse_matches.size(0));
        total_sparse_score += static_cast<double>(tensorAverageOrZero(matches.sparse_scores));
        total_dense_confidence += static_cast<double>(tensorAverageOrZero(matches.confidence));
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
    return EvalReport{
        total_matches / pair_count,  total_sparse_score / pair_count,          total_dense_confidence / pair_count,
        total_coverage / pair_count, total_half_turn_consistency / pair_count, total_half_turn_mean_error / pair_count};
}

void saveEvalReport(const std::string& path, const EvalReport& report)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::serialize::OutputArchive archive;
    archive.write("average_matches", torch::tensor({static_cast<float>(report.average_matches)}, options));
    archive.write("average_sparse_score", torch::tensor({static_cast<float>(report.average_sparse_score)}, options));
    archive.write("average_dense_confidence",
                  torch::tensor({static_cast<float>(report.average_dense_confidence)}, options));
    archive.write("semi_dense_coverage", torch::tensor({static_cast<float>(report.semi_dense_coverage)}, options));
    archive.write("half_turn_consistency", torch::tensor({static_cast<float>(report.half_turn_consistency)}, options));
    archive.write("half_turn_mean_error", torch::tensor({static_cast<float>(report.half_turn_mean_error)}, options));
    archive.save_to(path);
}

} // namespace pfm
