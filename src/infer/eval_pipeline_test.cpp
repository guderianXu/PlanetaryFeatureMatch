#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <torch/serialize.h>
#include <torch/torch.h>
#include <unistd.h>

#include "infer/eval_pipeline.h"
#include "tests/test_harness.h"

namespace
{

class TempEvalDirectory
{
  public:
    explicit TempEvalDirectory(const std::string& stem)
    {
        const auto suffix =
            std::to_string(static_cast<long long>(getpid())) + "_" + std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempEvalDirectory()
    {
        for (const auto& file_path : _files)
        {
            std::remove(file_path.string().c_str());
        }
        std::filesystem::remove(_path);
    }

    std::filesystem::path file(const std::string& name)
    {
        auto file_path = _path / name;
        _files.push_back(file_path);
        return file_path;
    }

  private:
    std::filesystem::path _path;
    std::vector<std::filesystem::path> _files;
};

pfm::FeatureSet makeEvalFeatureSet(int64_t dense_count)
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    return pfm::FeatureSet{torch::empty({0, 2}, float_options),
                           torch::empty({0}, float_options),
                           torch::empty({0, 3}, float_options),
                           torch::empty({0}, float_options),
                           torch::empty({0}, float_options),
                           torch::empty({0, 2, 2}, float_options),
                           torch::zeros({dense_count, 2}, float_options),
                           torch::ones({dense_count}, float_options)};
}

pfm::MatchSet makeEvalMatchSet(const torch::Tensor& sparse_scores, const torch::Tensor& confidence,
                               int64_t dense_match_count)
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    return pfm::MatchSet{torch::zeros({sparse_scores.size(0), 2}, long_options),
                         sparse_scores.to(torch::kCPU, torch::kFloat32).contiguous(),
                         torch::zeros({dense_match_count, 2}, float_options),
                         torch::zeros({dense_match_count, 2}, float_options),
                         confidence.to(torch::kCPU, torch::kFloat32).contiguous()};
}

pfm::MatchSet makeEvalMatchSetWithSparsePoints(const torch::Tensor& points_a, const torch::Tensor& points_b)
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    return pfm::MatchSet{
        torch::zeros({points_a.size(0), 2}, long_options), torch::ones({points_a.size(0)}, float_options),
        points_a.to(torch::kCPU, torch::kFloat32).contiguous(), points_b.to(torch::kCPU, torch::kFloat32).contiguous(),
        torch::ones({points_a.size(0)}, float_options)};
}

float readReportScalar(const std::string& path, const char* name)
{
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::Tensor tensor;
    archive.read(name, tensor);
    return tensor.to(torch::kCPU, torch::kFloat32).reshape({1}).item<float>();
}

static void eval_pipeline_rejects_empty_pairs_file()
{
    TempEvalDirectory temp_dir("pfm_eval_pairs_empty");
    const auto pairs = temp_dir.file("pairs.txt");
    {
        std::ofstream stream(pairs);
    }

    PFM_REQUIRE_INVALID_ARG(pfm::loadEvalPairs(pairs.string()));
}

static void eval_pipeline_loads_quoted_paths_with_spaces()
{
    TempEvalDirectory temp_dir("pfm_eval_pairs_quoted");
    const auto pairs = temp_dir.file("pairs.txt");
    {
        std::ofstream stream(pairs);
        stream << "\"/tmp/path with spaces/a.tif\" \"/tmp/path with spaces/b.tif\"\n";
    }

    const auto loaded = pfm::loadEvalPairs(pairs.string());

    PFM_REQUIRE(loaded.size() == 1);
    PFM_REQUIRE(loaded[0].first == "/tmp/path with spaces/a.tif");
    PFM_REQUIRE(loaded[0].second == "/tmp/path with spaces/b.tif");
}

static void eval_pipeline_aggregates_known_metrics()
{
    const std::vector<std::pair<pfm::FeatureSet, pfm::FeatureSet>> feature_sets = {
        {makeEvalFeatureSet(4), makeEvalFeatureSet(9)}, {makeEvalFeatureSet(5), makeEvalFeatureSet(2)}};
    const std::vector<pfm::MatchSet> match_sets = {
        makeEvalMatchSet(torch::tensor({0.5F, 1.0F}, torch::kFloat32), torch::tensor({0.25F, 0.75F}, torch::kFloat32),
                         2),
        makeEvalMatchSet(torch::tensor({0.25F}, torch::kFloat32), torch::tensor({1.0F}, torch::kFloat32), 1)};

    const auto report = pfm::aggregateEvalReport(feature_sets, match_sets);

    PFM_REQUIRE_CLOSE(report.average_matches, 1.5, 1.0e-6);
    PFM_REQUIRE_CLOSE(report.average_sparse_score, 0.5, 1.0e-6);
    PFM_REQUIRE_CLOSE(report.average_dense_confidence, 0.75, 1.0e-6);
    PFM_REQUIRE_CLOSE(report.semi_dense_coverage, 0.35, 1.0e-6);
}

static void eval_pipeline_aggregates_half_turn_metrics()
{
    auto features_a = makeEvalFeatureSet(4);
    auto features_b = makeEvalFeatureSet(4);
    features_a.feature_map_width = 100;
    features_a.feature_map_height = 100;
    features_b.feature_map_width = 100;
    features_b.feature_map_height = 100;
    const std::vector<std::pair<pfm::FeatureSet, pfm::FeatureSet>> feature_sets = {{features_a, features_b}};
    const std::vector<pfm::MatchSet> match_sets = {
        makeEvalMatchSetWithSparsePoints(torch::tensor({{10.0F, 20.0F}, {30.0F, 40.0F}}, torch::kFloat32),
                                         torch::tensor({{89.0F, 79.0F}, {30.0F, 40.0F}}, torch::kFloat32))};

    const auto report = pfm::aggregateEvalReport(feature_sets, match_sets);

    PFM_REQUIRE_CLOSE(report.half_turn_consistency, 0.5, 1.0e-6);
    PFM_REQUIRE_CLOSE(report.half_turn_mean_error, 21.691013, 1.0e-5);
}

static void eval_pipeline_no_sparse_matches_returns_zero_sparse_score()
{
    const std::vector<std::pair<pfm::FeatureSet, pfm::FeatureSet>> feature_sets = {
        {makeEvalFeatureSet(3), makeEvalFeatureSet(3)}};
    const std::vector<pfm::MatchSet> match_sets = {
        makeEvalMatchSet(torch::empty({0}, torch::kFloat32), torch::empty({0}, torch::kFloat32), 0)};

    const auto report = pfm::aggregateEvalReport(feature_sets, match_sets);

    PFM_REQUIRE_CLOSE(report.average_matches, 0.0, 1.0e-6);
    PFM_REQUIRE_CLOSE(report.average_sparse_score, 0.0, 1.0e-6);
    PFM_REQUIRE_CLOSE(report.average_dense_confidence, 0.0, 1.0e-6);
    PFM_REQUIRE_CLOSE(report.semi_dense_coverage, 0.0, 1.0e-6);
}

static void eval_pipeline_saves_report_archive_fields()
{
    TempEvalDirectory temp_dir("pfm_eval_report");
    const auto output = temp_dir.file("report.pt");
    pfm::EvalReport report;
    report.average_matches = 2.0;
    report.average_sparse_score = 0.5;
    report.average_dense_confidence = 0.75;
    report.semi_dense_coverage = 0.25;
    report.half_turn_consistency = 0.5;
    report.half_turn_mean_error = 12.0;

    pfm::saveEvalReport(output.string(), report);

    PFM_REQUIRE_CLOSE(readReportScalar(output.string(), "average_matches"), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(readReportScalar(output.string(), "average_sparse_score"), 0.5F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(readReportScalar(output.string(), "average_dense_confidence"), 0.75F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(readReportScalar(output.string(), "semi_dense_coverage"), 0.25F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(readReportScalar(output.string(), "half_turn_consistency"), 0.5F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(readReportScalar(output.string(), "half_turn_mean_error"), 12.0F, 1.0e-6F);
}

} // namespace

void register_eval_pipeline_tests()
{
    register_test("eval_pipeline_rejects_empty_pairs_file", eval_pipeline_rejects_empty_pairs_file);
    register_test("eval_pipeline_loads_quoted_paths_with_spaces", eval_pipeline_loads_quoted_paths_with_spaces);
    register_test("eval_pipeline_aggregates_known_metrics", eval_pipeline_aggregates_known_metrics);
    register_test("eval_pipeline_aggregates_half_turn_metrics", eval_pipeline_aggregates_half_turn_metrics);
    register_test("eval_pipeline_no_sparse_matches_returns_zero_sparse_score",
                  eval_pipeline_no_sparse_matches_returns_zero_sparse_score);
    register_test("eval_pipeline_saves_report_archive_fields", eval_pipeline_saves_report_archive_fields);
}
