#pragma once

#include <string>

#include <torch/torch.h>

#include "infer/feature_codec.h"
#include "infer/match_codec.h"

namespace pfm
{

struct WarpMatchMetrics
{
    int64_t sparse_total = 0;
    int64_t sparse_correct = 0;
    int64_t dense_total = 0;
    int64_t dense_correct = 0;

    int64_t total() const;
    int64_t correct() const;
    double precision() const;
};

struct WarpFeatureCoverageMetrics
{
    int64_t source_total = 0;
    int64_t valid_warp_total = 0;
    int64_t covered_by_target_keypoint = 0;
    int64_t descriptor_rank_observed = 0;
    int64_t descriptor_top1_count = 0;
    int64_t descriptor_rank_sum = 0;
    double coverage_fraction = 0.0;
    double mean_nearest_target_distance_pixels = 0.0;
    double mean_descriptor_positive_rank = 0.0;
    double descriptor_top1_accuracy = 0.0;
};

/// Loads the dense A-to-B warp tensor from a synthetic pair .pt archive.
torch::Tensor load_warp_a_to_b_tensor(const std::string& path);

/// Scores predicted matches against a dense synthetic A-to-B warp field.
WarpMatchMetrics compute_warp_match_metrics(const FeatureSet& features_a, const FeatureSet& features_b,
                                            const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                            double correct_threshold_pixels);

/// Measures whether detected A keypoints have any detected B keypoint near the true warped location,
/// and how well descriptors rank those true-near B candidates.
WarpFeatureCoverageMetrics compute_warp_feature_coverage_metrics(const FeatureSet& features_a,
                                                                 const FeatureSet& features_b,
                                                                 const torch::Tensor& warp_a_to_b,
                                                                 double correct_threshold_pixels);

} // namespace pfm
