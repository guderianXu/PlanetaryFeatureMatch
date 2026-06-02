#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include <torch/serialize.h>
#include <torch/torch.h>
#include <unistd.h>

#include "infer/match_metrics.h"
#include "tests/test_harness.h"

namespace
{

class TempMetricFile
{
  public:
    explicit TempMetricFile(const std::string& stem)
    {
        const auto suffix =
            std::to_string(static_cast<long long>(getpid())) + "_" + std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix + ".pt");
    }

    ~TempMetricFile()
    {
        std::remove(_path.string().c_str());
    }

    const std::filesystem::path& path() const
    {
        return _path;
    }

  private:
    std::filesystem::path _path;
};

pfm::FeatureSet makeMetricFeatureSet(const torch::Tensor& sparse_points, const torch::Tensor& dense_points)
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    pfm::FeatureSet features{sparse_points.to(torch::kFloat32),
                             torch::ones({sparse_points.size(0)}, float_options),
                             torch::zeros({sparse_points.size(0), 4}, float_options),
                             torch::ones({sparse_points.size(0)}, float_options),
                             torch::zeros({sparse_points.size(0)}, float_options),
                             torch::zeros({sparse_points.size(0), 2, 2}, float_options),
                             dense_points.to(torch::kFloat32),
                             torch::ones({dense_points.size(0)}, float_options)};
    features.feature_map_width = 4;
    features.feature_map_height = 4;
    return features;
}

torch::Tensor makeIdentityWarp()
{
    auto warp = torch::zeros({8, 8, 2}, torch::kFloat32);
    for (int64_t y = 0; y < 8; ++y)
    {
        for (int64_t x = 0; x < 8; ++x)
        {
            warp.index_put_({y, x, 0}, static_cast<float>(x));
            warp.index_put_({y, x, 1}, static_cast<float>(y));
        }
    }
    return warp;
}

} // namespace

static void match_metrics_scores_sparse_and_dense_matches_against_warp()
{
    const auto features_a =
        makeMetricFeatureSet(torch::tensor({{1.0F, 1.0F}, {2.0F, 2.0F}}), torch::tensor({{0.0F, 0.0F}, {3.0F, 3.0F}}));
    const auto features_b =
        makeMetricFeatureSet(torch::tensor({{1.0F, 1.0F}, {3.0F, 3.0F}}), torch::tensor({{0.0F, 0.0F}, {1.0F, 1.0F}}));
    const pfm::MatchSet matches{torch::tensor({{0, 0}, {1, 1}}, torch::kInt64), torch::tensor({1.0F, 0.5F}),
                                torch::tensor({{0.0F, 0.0F}, {3.0F, 3.0F}}),
                                torch::tensor({{0.0F, 0.0F}, {1.0F, 1.0F}}), torch::tensor({1.0F, 0.5F})};

    const auto metrics = pfm::compute_warp_match_metrics(features_a, features_b, matches, makeIdentityWarp(), 1.0);

    PFM_REQUIRE(metrics.sparse_total == 2);
    PFM_REQUIRE(metrics.sparse_correct == 1);
    PFM_REQUIRE(metrics.dense_total == 2);
    PFM_REQUIRE(metrics.dense_correct == 1);
    PFM_REQUIRE(metrics.total() == 4);
    PFM_REQUIRE(metrics.correct() == 2);
    PFM_REQUIRE_CLOSE(metrics.precision(), 0.5, 1.0e-6);
}

static void match_metrics_maps_feature_keypoints_by_pixel_centers()
{
    const auto features_a = makeMetricFeatureSet(torch::tensor({{0.0F, 0.0F}}), torch::empty({0, 2}, torch::kFloat32));
    const auto features_b = makeMetricFeatureSet(torch::tensor({{0.0F, 0.0F}}), torch::empty({0, 2}, torch::kFloat32));
    const pfm::MatchSet matches{torch::tensor({{0, 0}}, torch::kInt64), torch::tensor({1.0F}),
                                torch::empty({0, 2}, torch::kFloat32), torch::empty({0, 2}, torch::kFloat32),
                                torch::empty({0}, torch::kFloat32)};

    auto warp = makeIdentityWarp();
    warp.index_put_({0, 0, 0}, 7.0F);
    warp.index_put_({0, 0, 1}, 7.0F);

    const auto metrics = pfm::compute_warp_match_metrics(features_a, features_b, matches, warp, 0.1);

    PFM_REQUIRE(metrics.sparse_total == 1);
    PFM_REQUIRE(metrics.sparse_correct == 1);
}

static void match_metrics_loads_warp_from_synthetic_pair_archive()
{
    TempMetricFile temp_file("pfm_match_metrics_warp");
    torch::serialize::OutputArchive archive;
    archive.write("warp_a_to_b", makeIdentityWarp());
    archive.save_to(temp_file.path().string());

    const auto warp = pfm::load_warp_a_to_b_tensor(temp_file.path().string());

    PFM_REQUIRE(warp.sizes() == torch::IntArrayRef({8, 8, 2}));
    PFM_REQUIRE_CLOSE(warp.index({4, 5, 0}).item<float>(), 5.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(warp.index({4, 5, 1}).item<float>(), 4.0F, 1.0e-6F);
}

static void match_metrics_reports_keypoint_coverage_and_descriptor_rank()
{
    auto features_a =
        makeMetricFeatureSet(torch::tensor({{0.0F, 0.0F}, {2.0F, 2.0F}}), torch::empty({0, 2}, torch::kFloat32));
    auto features_b = makeMetricFeatureSet(torch::tensor({{0.0F, 0.0F}, {2.0F, 2.0F}, {3.0F, 3.0F}}),
                                           torch::empty({0, 2}, torch::kFloat32));
    features_a.descriptors = torch::tensor({{1.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F, 0.0F}}, torch::kFloat32);
    features_b.descriptors =
        torch::tensor({{1.0F, 0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F, 0.0F}}, torch::kFloat32);
    auto warp = makeIdentityWarp();
    warp.index_put_({5, 5, 0}, 7.0F);
    warp.index_put_({5, 5, 1}, 7.0F);

    const auto metrics = pfm::compute_warp_feature_coverage_metrics(features_a, features_b, warp, 1.0);

    PFM_REQUIRE(metrics.source_total == 2);
    PFM_REQUIRE(metrics.valid_warp_total == 2);
    PFM_REQUIRE(metrics.covered_by_target_keypoint == 2);
    PFM_REQUIRE_CLOSE(metrics.coverage_fraction, 1.0, 1.0e-6);
    PFM_REQUIRE(metrics.descriptor_rank_observed == 2);
    PFM_REQUIRE(metrics.descriptor_top1_count == 2);
    PFM_REQUIRE(metrics.descriptor_rank_sum == 2);
    PFM_REQUIRE_CLOSE(metrics.mean_descriptor_positive_rank, 1.0, 1.0e-6);
    PFM_REQUIRE_CLOSE(metrics.descriptor_top1_accuracy, 1.0, 1.0e-6);
}

void register_match_metrics_tests()
{
    register_test("match_metrics_scores_sparse_and_dense_matches_against_warp",
                  match_metrics_scores_sparse_and_dense_matches_against_warp);
    register_test("match_metrics_maps_feature_keypoints_by_pixel_centers",
                  match_metrics_maps_feature_keypoints_by_pixel_centers);
    register_test("match_metrics_loads_warp_from_synthetic_pair_archive",
                  match_metrics_loads_warp_from_synthetic_pair_archive);
    register_test("match_metrics_reports_keypoint_coverage_and_descriptor_rank",
                  match_metrics_reports_keypoint_coverage_and_descriptor_rank);
}
