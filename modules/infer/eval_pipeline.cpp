#include "infer/eval_pipeline.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include <torch/serialize.h>
#include <torch/torch.h>

namespace pfm {
namespace {

float tensorAverageOrZero(const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return 0.0F;
    }
    return tensor.to(torch::kCPU, torch::kFloat32).mean().item<float>();
}

}  // namespace

std::vector<std::pair<std::string, std::string>> loadEvalPairs(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument("failed to open pairs file: " + path);
    }

    std::vector<std::pair<std::string, std::string>> pairs;
    std::string image_a;
    std::string image_b;
    while (input >> image_a >> image_b) {
        pairs.push_back(std::make_pair(image_a, image_b));
    }
    if (pairs.empty()) {
        throw std::invalid_argument("pairs file is empty: " + path);
    }
    return pairs;
}

EvalReport aggregateEvalReport(
    const std::vector<std::pair<FeatureSet, FeatureSet>>& feature_sets,
    const std::vector<MatchSet>& match_sets
) {
    if (feature_sets.empty()) {
        throw std::invalid_argument("evaluation inputs must not be empty");
    }
    if (feature_sets.size() != match_sets.size()) {
        throw std::invalid_argument("feature and match counts must match");
    }

    double total_matches = 0.0;
    double total_sparse_score = 0.0;
    double total_dense_confidence = 0.0;
    double total_coverage = 0.0;
    for (size_t index = 0; index < match_sets.size(); ++index) {
        const auto& features_a = feature_sets[index].first;
        const auto& matches = match_sets[index];
        total_matches += static_cast<double>(matches.sparse_matches.size(0));
        total_sparse_score += static_cast<double>(tensorAverageOrZero(matches.sparse_scores));
        total_dense_confidence += static_cast<double>(tensorAverageOrZero(matches.confidence));
        const int64_t dense_base = std::max<int64_t>(features_a.dense_points.size(0), 1);
        total_coverage += static_cast<double>(matches.points_a.size(0)) / static_cast<double>(dense_base);
    }

    const double pair_count = static_cast<double>(match_sets.size());
    return EvalReport{
        total_matches / pair_count,
        total_sparse_score / pair_count,
        total_dense_confidence / pair_count,
        total_coverage / pair_count};
}

void saveEvalReport(const std::string& path, const EvalReport& report) {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::serialize::OutputArchive archive;
    archive.write("average_matches", torch::tensor({static_cast<float>(report.average_matches)}, options));
    archive.write("average_sparse_score", torch::tensor({static_cast<float>(report.average_sparse_score)}, options));
    archive.write("average_dense_confidence", torch::tensor({static_cast<float>(report.average_dense_confidence)}, options));
    archive.write("semi_dense_coverage", torch::tensor({static_cast<float>(report.semi_dense_coverage)}, options));
    archive.save_to(path);
}

}  // namespace pfm
