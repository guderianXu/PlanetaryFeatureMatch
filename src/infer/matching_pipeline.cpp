#include "infer/matching_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <torch/torch.h>

#include "core/timer.h"
#include "optim/descriptor_similarity.h"

namespace pfm
{
namespace
{

constexpr double PI = 3.14159265358979323846;
constexpr int64_t ROTATION_CONSISTENCY_MIN_MATCHES = 32;
constexpr int64_t ROTATION_CONSISTENCY_BINS = 72;
constexpr double ROTATION_CONSISTENCY_MAX_ANGLE_ERROR = PI / 36.0;
constexpr double ROTATION_CONSISTENCY_MAX_RADIUS_ERROR = 2.0;
constexpr double ROTATION_CONSISTENCY_MAX_POSITION_ERROR = 0.5;
constexpr int64_t GRAPH_GREEDY_MAX_MATCHES = 256;
constexpr int64_t GRAPH_MATCHER_MAX_SPARSE_KEYPOINTS = 1024;
constexpr int64_t GEOMETRIC_CONSISTENCY_MIN_MATCHES = 8;
constexpr int64_t GEOMETRIC_CONSISTENCY_MIN_INLIERS = 4;
constexpr double GEOMETRIC_CONSISTENCY_RANSAC_THRESHOLD = 0.75;
constexpr int64_t GEOMETRIC_CONSISTENCY_MAX_OUTPUT_MATCHES = 512;
constexpr double GEOMETRIC_SPREAD_QUALITY_WEIGHT = 40.0;
constexpr int64_t GEOMETRIC_RESIDUAL_CLEANUP_MIN_MATCHES = 32;
constexpr int64_t GEOMETRIC_RESIDUAL_CLEANUP_LOW_COUNT_MIN_MATCHES = 8;
constexpr int64_t GEOMETRIC_RESIDUAL_CLEANUP_LOW_COUNT_MIN_KEEP = 4;
constexpr int64_t GEOMETRIC_RESIDUAL_CLEANUP_PROJECTIVE_MIN_GAIN = 2;
constexpr int64_t GEOMETRIC_RESIDUAL_CLEANUP_HIGH_COUNT_FLOOR = 150;
constexpr double GEOMETRIC_RESIDUAL_CLEANUP_THRESHOLD = 0.5;
constexpr double GEOMETRIC_RESIDUAL_CLEANUP_LOW_COUNT_THRESHOLD = 0.5;
constexpr double LOCAL_DISPLACEMENT_CONSISTENCY_THRESHOLD = 3.0;
constexpr int64_t LOCAL_DISPLACEMENT_CONSISTENCY_NEIGHBORS = 12;
constexpr int64_t LOCAL_DISPLACEMENT_CONSISTENCY_MIN_INLIERS = 4;
constexpr int64_t LOCAL_DISPLACEMENT_CONSISTENCY_MAX_ADAPTIVE_CANDIDATES = 8192;
constexpr int64_t LOCAL_DISPLACEMENT_ADAPTIVE_MIN_GAIN = 8;
constexpr int64_t DESCRIPTOR_TOPK_CANDIDATES_PER_SOURCE = 4;
constexpr int64_t DESCRIPTOR_CONSERVATIVE_TOPK_CANDIDATES_PER_SOURCE = 2;
constexpr int64_t DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MAX_BASE_MATCHES = 64;
constexpr int64_t DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MIN_MATCHES = 8;
constexpr int64_t DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MIN_RATIO_NUMERATOR = 1;
constexpr int64_t DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MIN_RATIO_DENOMINATOR = 2;
constexpr int64_t DESCRIPTOR_WIDE_TOPK_CANDIDATES_PER_SOURCE = 24;
constexpr int64_t DESCRIPTOR_WIDE_TOPK_FALLBACK_MAX_BASE_MATCHES = 6;
constexpr int64_t DESCRIPTOR_WIDE_TOPK_FALLBACK_MIN_GAIN = 2;
constexpr int64_t DESCRIPTOR_WIDE_TOPK_FALLBACK_MAX_MATCHES = 12;
constexpr double DESCRIPTOR_WIDE_TOPK_FALLBACK_MIN_MEAN_SCORE = 0.98;
constexpr int64_t DESCRIPTOR_TOPK_MAX_CANDIDATES = 65536;
constexpr int64_t DESCRIPTOR_PROJECTIVE_RESCUE_MAX_BASE_MATCHES = 128;
constexpr int64_t DESCRIPTOR_PROJECTIVE_RESCUE_MIN_MATCHES = 100;
constexpr int64_t DESCRIPTOR_PROJECTIVE_RESCUE_MIN_GAIN_MULTIPLIER = 2;
constexpr int64_t DESCRIPTOR_MUTUAL_GEOMETRY_MIN_SAFE_MATCHES = 24;
constexpr int64_t DESCRIPTOR_MUTUAL_GEOMETRY_MIN_TOPK_GUARD_MATCHES = 100;
constexpr int64_t DESCRIPTOR_MUTUAL_GEOMETRY_MAX_TOPK_GUARD_MATCHES = 128;
constexpr int64_t DESCRIPTOR_MUTUAL_GEOMETRY_CLOSE_TOPK_MAX_MATCHES = 64;
constexpr int64_t DESCRIPTOR_MUTUAL_GEOMETRY_CLOSE_TOPK_MAX_GAIN = 4;
constexpr int64_t DESCRIPTOR_TOPK_GEOMETRY_MAX_SAFE_GAIN_NUMERATOR = 3;
constexpr int64_t DESCRIPTOR_TOPK_GEOMETRY_MAX_SAFE_GAIN_DENOMINATOR = 2;
constexpr int64_t DESCRIPTOR_TOPK_TAIL_TRIM_MIN_MATCHES = 32;
constexpr int64_t DESCRIPTOR_TOPK_TAIL_TRIM_MAX_MATCHES = 64;
constexpr int64_t DESCRIPTOR_TOPK_TAIL_TRIM_MIN_KEEP = 32;
constexpr int64_t DESCRIPTOR_TOPK_TAIL_TRIM_MIN_DROP = 4;
constexpr float DESCRIPTOR_TOPK_TAIL_TRIM_MIN_SCORE = 0.98F;
constexpr bool DESCRIPTOR_RECIPROCAL_TOPK_FALLBACK = false;
constexpr bool DESCRIPTOR_TOPK_PROJECTIVE_BEFORE_ROTATION = false;

enum class SparseGeometryFilter
{
    Adaptive,
    Projective,
    RotationOnly,
    Local,
};

bool matchDebugEnabled()
{
    const char* value = std::getenv("PFM_MATCH_DEBUG");
    return value != nullptr && std::string(value) != "0";
}

bool returnRawGraphMatchesForDebug()
{
    const char* value = std::getenv("PFM_MATCH_DEBUG_RETURN_GRAPH");
    return value != nullptr && std::string(value) != "0";
}

bool useTopKGeometryForDebug()
{
    const char* value = std::getenv("PFM_MATCH_DEBUG_USE_TOPK_GEOMETRY");
    return value == nullptr || std::string(value) != "0";
}

bool shouldUseGraphMatcherForSparseCount(int64_t keypoint_count_a, int64_t keypoint_count_b)
{
    return keypoint_count_a <= GRAPH_MATCHER_MAX_SPARSE_KEYPOINTS &&
           keypoint_count_b <= GRAPH_MATCHER_MAX_SPARSE_KEYPOINTS;
}

bool shouldUseWideTopKFallback(int64_t base_matches, int64_t wide_matches, double wide_mean_score)
{
    return base_matches <= DESCRIPTOR_WIDE_TOPK_FALLBACK_MAX_BASE_MATCHES &&
           wide_matches >= base_matches + DESCRIPTOR_WIDE_TOPK_FALLBACK_MIN_GAIN &&
           wide_matches <= DESCRIPTOR_WIDE_TOPK_FALLBACK_MAX_MATCHES &&
           wide_mean_score >= DESCRIPTOR_WIDE_TOPK_FALLBACK_MIN_MEAN_SCORE;
}

bool shouldUseProjectiveTopKRescue(int64_t base_matches, int64_t projective_matches)
{
    return base_matches < DESCRIPTOR_PROJECTIVE_RESCUE_MAX_BASE_MATCHES &&
           projective_matches >= DESCRIPTOR_PROJECTIVE_RESCUE_MIN_MATCHES &&
           projective_matches >= base_matches * DESCRIPTOR_PROJECTIVE_RESCUE_MIN_GAIN_MULTIPLIER;
}

bool shouldPreferMutualDescriptorGeometry(int64_t mutual_matches, int64_t topk_matches)
{
    if (mutual_matches < DESCRIPTOR_MUTUAL_GEOMETRY_MIN_SAFE_MATCHES)
    {
        return false;
    }
    if (topk_matches <= DESCRIPTOR_MUTUAL_GEOMETRY_CLOSE_TOPK_MAX_MATCHES && topk_matches >= mutual_matches &&
        topk_matches <= mutual_matches + DESCRIPTOR_MUTUAL_GEOMETRY_CLOSE_TOPK_MAX_GAIN)
    {
        return true;
    }
    return topk_matches >= DESCRIPTOR_MUTUAL_GEOMETRY_MIN_TOPK_GUARD_MATCHES &&
           topk_matches < DESCRIPTOR_MUTUAL_GEOMETRY_MAX_TOPK_GUARD_MATCHES &&
           topk_matches * DESCRIPTOR_TOPK_GEOMETRY_MAX_SAFE_GAIN_DENOMINATOR >
               mutual_matches * DESCRIPTOR_TOPK_GEOMETRY_MAX_SAFE_GAIN_NUMERATOR;
}

bool shouldUseConservativeTopKFallback(int64_t base_matches, int64_t conservative_matches)
{
    return base_matches <= DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MAX_BASE_MATCHES &&
           conservative_matches >= DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MIN_MATCHES &&
           conservative_matches * DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MIN_RATIO_DENOMINATOR >=
               base_matches * DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MIN_RATIO_NUMERATOR;
}

double geometricSpreadQualityWeight()
{
    const char* value = std::getenv("PFM_GEOMETRIC_SPREAD_QUALITY_WEIGHT");
    if (value == nullptr)
    {
        return 0.0;
    }
    try
    {
        const auto parsed = std::stod(value);
        if (std::isfinite(parsed) && parsed >= 0.0)
        {
            return parsed;
        }
    }
    catch (const std::exception&)
    {
    }
    return 0.0;
}

int64_t descriptorTopKCandidatesPerSource()
{
    const char* value = std::getenv("PFM_DESCRIPTOR_TOPK_CANDIDATES");
    if (value == nullptr)
    {
        return DESCRIPTOR_TOPK_CANDIDATES_PER_SOURCE;
    }
    try
    {
        const auto parsed = std::stoll(value);
        if (parsed > 0)
        {
            return parsed;
        }
    }
    catch (const std::exception&)
    {
    }
    return DESCRIPTOR_TOPK_CANDIDATES_PER_SOURCE;
}

bool descriptorTopKProjectiveBeforeRotation()
{
    const char* value = std::getenv("PFM_DESCRIPTOR_TOPK_PROJECTIVE_BEFORE_ROTATION");
    if (value == nullptr)
    {
        return DESCRIPTOR_TOPK_PROJECTIVE_BEFORE_ROTATION;
    }
    return std::string(value) != "0";
}

bool descriptorReciprocalTopKFallback()
{
    const char* value = std::getenv("PFM_DESCRIPTOR_RECIPROCAL_TOPK_FALLBACK");
    if (value == nullptr)
    {
        return DESCRIPTOR_RECIPROCAL_TOPK_FALLBACK;
    }
    return std::string(value) != "0";
}

double rotationConsistencyMaxPositionError()
{
    const char* value = std::getenv("PFM_ROTATION_CONSISTENCY_MAX_POSITION_ERROR");
    if (value == nullptr)
    {
        return ROTATION_CONSISTENCY_MAX_POSITION_ERROR;
    }
    try
    {
        const auto parsed = std::stod(value);
        if (std::isfinite(parsed) && parsed > 0.0)
        {
            return parsed;
        }
    }
    catch (const std::exception&)
    {
    }
    return ROTATION_CONSISTENCY_MAX_POSITION_ERROR;
}

SparseGeometryFilter sparseGeometryFilter()
{
    const char* value = std::getenv("PFM_SPARSE_GEOMETRY_FILTER");
    if (value == nullptr)
    {
        return SparseGeometryFilter::Adaptive;
    }
    const std::string mode(value);
    if (mode == "adaptive" || mode == "auto")
    {
        return SparseGeometryFilter::Adaptive;
    }
    if (mode == "rotation" || mode == "rotation-only")
    {
        return SparseGeometryFilter::RotationOnly;
    }
    if (mode == "local" || mode == "local-displacement")
    {
        return SparseGeometryFilter::Local;
    }
    return SparseGeometryFilter::Projective;
}

bool shouldReturnRotationOnlyMatches(int64_t rotation_matches)
{
    return sparseGeometryFilter() == SparseGeometryFilter::RotationOnly &&
           rotation_matches >= ROTATION_CONSISTENCY_MIN_MATCHES;
}

bool shouldPreferLocalDisplacementGeometry(int64_t projective_matches, int64_t local_matches)
{
    if (local_matches < LOCAL_DISPLACEMENT_CONSISTENCY_MIN_INLIERS)
    {
        return false;
    }
    if (projective_matches < GEOMETRIC_CONSISTENCY_MIN_INLIERS)
    {
        return true;
    }
    const auto required_gain = std::max<int64_t>(LOCAL_DISPLACEMENT_ADAPTIVE_MIN_GAIN, projective_matches / 4);
    return local_matches >= projective_matches + required_gain;
}

torch::Tensor descriptorSimilarityScores(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                                         const torch::Device& compute_device = torch::Device(torch::kCPU))
{
    const auto desc_a = descriptors_a.to(compute_device, torch::kFloat32).unsqueeze(0);
    const auto desc_b = descriptors_b.to(compute_device, torch::kFloat32).unsqueeze(0);
    return cyclicDescriptorSimilarityScores(desc_a, desc_b).squeeze(0);
}

double normalizeAngle(double angle)
{
    while (angle <= -PI)
    {
        angle += 2.0 * PI;
    }
    while (angle > PI)
    {
        angle -= 2.0 * PI;
    }
    return angle;
}

double angleDistance(double lhs, double rhs)
{
    return std::abs(normalizeAngle(lhs - rhs));
}

int64_t angleBin(double angle)
{
    const auto normalized = normalizeAngle(angle) + PI;
    auto bin = static_cast<int64_t>(std::floor(normalized / (2.0 * PI) * ROTATION_CONSISTENCY_BINS));
    return std::min<int64_t>(ROTATION_CONSISTENCY_BINS - 1, std::max<int64_t>(0, bin));
}

std::pair<torch::Tensor, torch::Tensor> filterRotationConsistentMatches(const FeatureSet& features_a,
                                                                        const FeatureSet& features_b,
                                                                        const torch::Tensor& matches,
                                                                        const torch::Tensor& scores)
{
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (!matches.defined() || matches.size(0) < ROTATION_CONSISTENCY_MIN_MATCHES)
    {
        return {matches, scores};
    }

    auto cpu_matches = matches.to(torch::kCPU, torch::kInt64).contiguous();
    auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto* match_data = cpu_matches.data_ptr<int64_t>();
    const auto* points_a_data = points_a.data_ptr<float>();
    const auto* points_b_data = points_b.data_ptr<float>();
    const double center_ax = (static_cast<double>(features_a.feature_map_width) - 1.0) * 0.5;
    const double center_ay = (static_cast<double>(features_a.feature_map_height) - 1.0) * 0.5;
    const double center_bx = (static_cast<double>(features_b.feature_map_width) - 1.0) * 0.5;
    const double center_by = (static_cast<double>(features_b.feature_map_height) - 1.0) * 0.5;

    std::vector<int64_t> bins(static_cast<size_t>(ROTATION_CONSISTENCY_BINS), 0);
    std::vector<double> deltas(static_cast<size_t>(cpu_matches.size(0)), 0.0);
    std::vector<double> radius_errors(static_cast<size_t>(cpu_matches.size(0)), 0.0);
    for (int64_t index = 0; index < cpu_matches.size(0); ++index)
    {
        const auto ia = match_data[index * 2];
        const auto ib = match_data[index * 2 + 1];
        const double ax = static_cast<double>(points_a_data[ia * 2]) - center_ax;
        const double ay = static_cast<double>(points_a_data[ia * 2 + 1]) - center_ay;
        const double bx = static_cast<double>(points_b_data[ib * 2]) - center_bx;
        const double by = static_cast<double>(points_b_data[ib * 2 + 1]) - center_by;
        const auto delta = normalizeAngle(std::atan2(by, bx) - std::atan2(ay, ax));
        deltas[static_cast<size_t>(index)] = delta;
        radius_errors[static_cast<size_t>(index)] = std::abs(std::hypot(ax, ay) - std::hypot(bx, by));
        ++bins[static_cast<size_t>(angleBin(delta))];
    }

    const auto best_it = std::max_element(bins.begin(), bins.end());
    if (best_it == bins.end() || *best_it < ROTATION_CONSISTENCY_MIN_MATCHES / 4)
    {
        return {matches, scores};
    }
    const int64_t best_bin = static_cast<int64_t>(std::distance(bins.begin(), best_it));
    const double bin_center = (static_cast<double>(best_bin) + 0.5) / ROTATION_CONSISTENCY_BINS * 2.0 * PI - PI;
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (const auto delta : deltas)
    {
        if (angleBin(delta) == best_bin)
        {
            sin_sum += std::sin(delta);
            cos_sum += std::cos(delta);
        }
    }
    const double dominant_angle = std::hypot(sin_sum, cos_sum) > 1.0e-9 ? std::atan2(sin_sum, cos_sum) : bin_center;
    const double dominant_cos = std::cos(dominant_angle);
    const double dominant_sin = std::sin(dominant_angle);
    const double max_position_error = rotationConsistencyMaxPositionError();
    const bool can_check_position = features_a.feature_map_width > 0 && features_a.feature_map_height > 0 &&
                                    features_b.feature_map_width > 0 && features_b.feature_map_height > 0;

    std::vector<int64_t> keep_indices;
    keep_indices.reserve(static_cast<size_t>(*best_it));
    for (int64_t index = 0; index < cpu_matches.size(0); ++index)
    {
        const auto ia = match_data[index * 2];
        const auto ib = match_data[index * 2 + 1];
        const double ax = static_cast<double>(points_a_data[ia * 2]) - center_ax;
        const double ay = static_cast<double>(points_a_data[ia * 2 + 1]) - center_ay;
        const double bx = static_cast<double>(points_b_data[ib * 2]) - center_bx;
        const double by = static_cast<double>(points_b_data[ib * 2 + 1]) - center_by;
        const double predicted_bx = dominant_cos * ax - dominant_sin * ay;
        const double predicted_by = dominant_sin * ax + dominant_cos * ay;
        const double position_error = std::hypot(predicted_bx - bx, predicted_by - by);
        const bool position_ok = !can_check_position || position_error <= max_position_error;
        if (angleDistance(deltas[static_cast<size_t>(index)], dominant_angle) <= ROTATION_CONSISTENCY_MAX_ANGLE_ERROR &&
            radius_errors[static_cast<size_t>(index)] <= ROTATION_CONSISTENCY_MAX_RADIUS_ERROR && position_ok)
        {
            keep_indices.push_back(index);
        }
    }
    if (keep_indices.empty())
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    auto keep = torch::from_blob(keep_indices.data(), {static_cast<int64_t>(keep_indices.size())},
                                 torch::TensorOptions().dtype(torch::kInt64))
                    .clone();
    return {cpu_matches.index_select(0, keep).contiguous(),
            scores.to(torch::kCPU, torch::kFloat32).index_select(0, keep).contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> matchMutualDescriptorFeatures(const FeatureSet& features_a,
                                                                      const FeatureSet& features_b,
                                                                      bool apply_rotation_filter = true)
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (features_a.descriptors.size(0) == 0 || features_b.descriptors.size(0) == 0)
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    const auto scores = descriptorSimilarityScores(features_a.descriptors, features_b.descriptors);
    const auto best_ab = scores.max(1);
    const auto target_indices = std::get<1>(best_ab);
    const auto best_ba = std::get<1>(scores.max(0));
    const auto source_indices = torch::arange(scores.size(0), long_options);
    const auto mutual_sources = best_ba.index_select(0, target_indices);
    const auto keep = mutual_sources.eq(source_indices);
    if (!keep.any().item<bool>())
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    const auto kept_sources = source_indices.index({keep});
    const auto kept_targets = target_indices.index({keep});
    auto matches = torch::stack({kept_sources, kept_targets}, 1).to(torch::kCPU, torch::kInt64).contiguous();
    auto kept_scores = std::get<0>(best_ab).index({keep}).to(torch::kCPU, torch::kFloat32).contiguous();
    if (!apply_rotation_filter)
    {
        return {matches, kept_scores};
    }
    return filterRotationConsistentMatches(features_a, features_b, matches, kept_scores);
}

std::pair<torch::Tensor, torch::Tensor>
matchDescriptorTopKFeatures(const FeatureSet& features_a, const FeatureSet& features_b,
                            int64_t candidates_per_source = DESCRIPTOR_TOPK_CANDIDATES_PER_SOURCE,
                            const torch::Device& compute_device = torch::Device(torch::kCPU))
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (features_a.descriptors.size(0) == 0 || features_b.descriptors.size(0) == 0 || candidates_per_source <= 0)
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    const auto scores = descriptorSimilarityScores(features_a.descriptors, features_b.descriptors, compute_device);
    const auto k = std::min<int64_t>(candidates_per_source, scores.size(1));
    auto topk = scores.topk(k, 1, true, true);
    auto top_scores = std::get<0>(topk).reshape({-1}).to(torch::kCPU, torch::kFloat32).contiguous();
    auto top_targets = std::get<1>(topk).reshape({-1}).to(torch::kCPU, torch::kInt64).contiguous();
    auto source_indices =
        torch::arange(scores.size(0), long_options).unsqueeze(1).expand({scores.size(0), k}).reshape({-1}).contiguous();
    if (top_scores.size(0) > DESCRIPTOR_TOPK_MAX_CANDIDATES)
    {
        auto sorted = top_scores.sort(0, true);
        auto keep = std::get<1>(sorted).narrow(0, 0, DESCRIPTOR_TOPK_MAX_CANDIDATES).to(torch::kCPU, torch::kInt64);
        source_indices = source_indices.index_select(0, keep).contiguous();
        top_targets = top_targets.index_select(0, keep).contiguous();
        top_scores = top_scores.index_select(0, keep).contiguous();
    }
    return {torch::stack({source_indices, top_targets}, 1).to(torch::kCPU, torch::kInt64).contiguous(), top_scores};
}

std::pair<torch::Tensor, torch::Tensor>
matchDescriptorReciprocalTopKFeatures(const FeatureSet& features_a, const FeatureSet& features_b,
                                      int64_t candidates_per_source = DESCRIPTOR_TOPK_CANDIDATES_PER_SOURCE,
                                      const torch::Device& compute_device = torch::Device(torch::kCPU))
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (features_a.descriptors.size(0) == 0 || features_b.descriptors.size(0) == 0 || candidates_per_source <= 0)
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }

    const auto scores = descriptorSimilarityScores(features_a.descriptors, features_b.descriptors, compute_device);
    const auto k_ab = std::min<int64_t>(candidates_per_source, scores.size(1));
    const auto k_ba = std::min<int64_t>(candidates_per_source, scores.size(0));
    const auto top_ab = scores.topk(k_ab, 1, true, true);
    const auto top_ba = scores.transpose(0, 1).topk(k_ba, 1, true, true);
    const auto ab_scores = std::get<0>(top_ab).to(torch::kCPU, torch::kFloat32).contiguous();
    const auto ab_targets = std::get<1>(top_ab).to(torch::kCPU, torch::kInt64).contiguous();
    const auto ba_sources = std::get<1>(top_ba).to(torch::kCPU, torch::kInt64).contiguous();
    const auto* ab_score_data = ab_scores.data_ptr<float>();
    const auto* ab_target_data = ab_targets.data_ptr<int64_t>();
    const auto* ba_source_data = ba_sources.data_ptr<int64_t>();

    std::vector<int64_t> match_values;
    std::vector<float> score_values;
    match_values.reserve(static_cast<std::size_t>(scores.size(0) * std::min<int64_t>(k_ab, 4) * 2));
    score_values.reserve(static_cast<std::size_t>(scores.size(0) * std::min<int64_t>(k_ab, 4)));
    for (int64_t source = 0; source < scores.size(0); ++source)
    {
        for (int64_t rank = 0; rank < k_ab; ++rank)
        {
            const auto target = ab_target_data[source * k_ab + rank];
            bool reciprocal = false;
            for (int64_t reverse_rank = 0; reverse_rank < k_ba; ++reverse_rank)
            {
                if (ba_source_data[target * k_ba + reverse_rank] == source)
                {
                    reciprocal = true;
                    break;
                }
            }
            if (!reciprocal)
            {
                continue;
            }
            match_values.push_back(source);
            match_values.push_back(target);
            score_values.push_back(ab_score_data[source * k_ab + rank]);
        }
    }
    if (score_values.empty())
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    return {torch::from_blob(match_values.data(), {static_cast<int64_t>(score_values.size()), 2}, long_options)
                .clone()
                .contiguous(),
            torch::from_blob(score_values.data(), {static_cast<int64_t>(score_values.size())}, float_options)
                .clone()
                .contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> mergeSparseMatchCandidates(const torch::Tensor& primary_matches,
                                                                   const torch::Tensor& primary_scores,
                                                                   const torch::Tensor& fallback_matches,
                                                                   const torch::Tensor& fallback_scores)
{
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    std::vector<int64_t> merged_matches;
    std::vector<float> merged_scores;

    auto append_unique = [&](const torch::Tensor& matches, const torch::Tensor& scores)
    {
        auto cpu_matches = matches.to(torch::kCPU, torch::kInt64).contiguous();
        auto cpu_scores = scores.to(torch::kCPU, torch::kFloat32).contiguous();
        for (int64_t index = 0; index < cpu_matches.size(0); ++index)
        {
            const auto source = cpu_matches.index({index, 0}).item<int64_t>();
            const auto target = cpu_matches.index({index, 1}).item<int64_t>();
            bool exists = false;
            for (std::size_t offset = 0; offset + 1 < merged_matches.size(); offset += 2)
            {
                if (merged_matches[offset] == source || merged_matches[offset + 1] == target)
                {
                    exists = true;
                    break;
                }
            }
            if (!exists)
            {
                merged_matches.push_back(source);
                merged_matches.push_back(target);
                merged_scores.push_back(cpu_scores.index({index}).item<float>());
            }
        }
    };

    append_unique(primary_matches, primary_scores);
    append_unique(fallback_matches, fallback_scores);
    if (merged_scores.empty())
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    return {torch::from_blob(merged_matches.data(), {static_cast<int64_t>(merged_scores.size()), 2}, long_options)
                .clone()
                .contiguous(),
            torch::from_blob(merged_scores.data(), {static_cast<int64_t>(merged_scores.size())}, float_options)
                .clone()
                .contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> selectMatches(const torch::Tensor& matches, const torch::Tensor& scores,
                                                      const std::vector<int64_t>& keep_indices)
{
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (keep_indices.empty())
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    auto keep = torch::from_blob(const_cast<int64_t*>(keep_indices.data()), {static_cast<int64_t>(keep_indices.size())},
                                 long_options)
                    .clone();
    return {matches.to(torch::kCPU, torch::kInt64).index_select(0, keep).contiguous(),
            scores.to(torch::kCPU, torch::kFloat32).index_select(0, keep).contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> trimLowConfidenceTopKTail(const torch::Tensor& matches,
                                                                  const torch::Tensor& scores)
{
    if (!matches.defined() || !scores.defined() || matches.size(0) < DESCRIPTOR_TOPK_TAIL_TRIM_MIN_MATCHES ||
        matches.size(0) > DESCRIPTOR_TOPK_TAIL_TRIM_MAX_MATCHES)
    {
        return {matches, scores};
    }
    const auto cpu_scores = scores.to(torch::kCPU, torch::kFloat32).contiguous();
    auto keep_mask = cpu_scores.ge(DESCRIPTOR_TOPK_TAIL_TRIM_MIN_SCORE);
    const auto keep_count = keep_mask.sum().item<int64_t>();
    const auto drop_count = scores.size(0) - keep_count;
    if (keep_count < DESCRIPTOR_TOPK_TAIL_TRIM_MIN_KEEP || drop_count < DESCRIPTOR_TOPK_TAIL_TRIM_MIN_DROP)
    {
        return {matches, scores};
    }
    auto keep = torch::nonzero(keep_mask).reshape({-1}).to(torch::kCPU, torch::kInt64).contiguous();
    return {matches.to(torch::kCPU, torch::kInt64).index_select(0, keep).contiguous(),
            cpu_scores.index_select(0, keep).contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> cleanupAffineResidualMatches(const FeatureSet& features_a,
                                                                     const FeatureSet& features_b,
                                                                     const torch::Tensor& matches,
                                                                     const torch::Tensor& scores)
{
    if (!matches.defined() || matches.size(0) < GEOMETRIC_RESIDUAL_CLEANUP_LOW_COUNT_MIN_MATCHES)
    {
        return {matches, scores};
    }

    const auto cpu_matches = matches.to(torch::kCPU, torch::kInt64).contiguous();
    const auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<cv::Point2f> source_points;
    std::vector<cv::Point2f> target_points;
    source_points.reserve(static_cast<std::size_t>(cpu_matches.size(0)));
    target_points.reserve(static_cast<std::size_t>(cpu_matches.size(0)));
    const bool low_count = cpu_matches.size(0) < GEOMETRIC_RESIDUAL_CLEANUP_MIN_MATCHES;
    const auto cleanup_threshold =
        low_count ? GEOMETRIC_RESIDUAL_CLEANUP_LOW_COUNT_THRESHOLD : GEOMETRIC_RESIDUAL_CLEANUP_THRESHOLD;
    for (int64_t row = 0; row < cpu_matches.size(0); ++row)
    {
        const auto source = cpu_matches.index({row, 0}).item<int64_t>();
        const auto target = cpu_matches.index({row, 1}).item<int64_t>();
        source_points.emplace_back(points_a.index({source, 0}).item<float>(),
                                   points_a.index({source, 1}).item<float>());
        target_points.emplace_back(points_b.index({target, 0}).item<float>(),
                                   points_b.index({target, 1}).item<float>());
    }

    auto affine_keep = [&]()
    {
        std::vector<int64_t> keep;
        cv::Mat mask;
        const auto affine =
            cv::estimateAffine2D(source_points, target_points, mask, cv::RANSAC, GEOMETRIC_RESIDUAL_CLEANUP_THRESHOLD);
        if (affine.empty() || affine.rows != 2 || affine.cols != 3)
        {
            return keep;
        }
        keep.reserve(static_cast<std::size_t>(cpu_matches.size(0)));
        for (int64_t row = 0; row < cpu_matches.size(0); ++row)
        {
            const auto& source = source_points[static_cast<std::size_t>(row)];
            const auto& target = target_points[static_cast<std::size_t>(row)];
            const double predicted_x =
                affine.at<double>(0, 0) * source.x + affine.at<double>(0, 1) * source.y + affine.at<double>(0, 2);
            const double predicted_y =
                affine.at<double>(1, 0) * source.x + affine.at<double>(1, 1) * source.y + affine.at<double>(1, 2);
            const double residual =
                std::hypot(predicted_x - static_cast<double>(target.x), predicted_y - static_cast<double>(target.y));
            if (residual <= cleanup_threshold)
            {
                keep.push_back(row);
            }
        }
        return keep;
    };

    auto homography_keep = [&]()
    {
        std::vector<int64_t> keep;
        if (source_points.size() < 4)
        {
            return keep;
        }
        cv::Mat mask;
        const auto homography = cv::findHomography(source_points, target_points, cv::RANSAC, cleanup_threshold, mask);
        if (homography.empty() || homography.rows != 3 || homography.cols != 3)
        {
            return keep;
        }
        keep.reserve(static_cast<std::size_t>(cpu_matches.size(0)));
        for (int64_t row = 0; row < cpu_matches.size(0); ++row)
        {
            const auto& source = source_points[static_cast<std::size_t>(row)];
            const auto& target = target_points[static_cast<std::size_t>(row)];
            const double denom = homography.at<double>(2, 0) * source.x + homography.at<double>(2, 1) * source.y +
                                 homography.at<double>(2, 2);
            if (std::abs(denom) <= 1.0e-12)
            {
                continue;
            }
            const double predicted_x = (homography.at<double>(0, 0) * source.x +
                                        homography.at<double>(0, 1) * source.y + homography.at<double>(0, 2)) /
                                       denom;
            const double predicted_y = (homography.at<double>(1, 0) * source.x +
                                        homography.at<double>(1, 1) * source.y + homography.at<double>(1, 2)) /
                                       denom;
            const double residual =
                std::hypot(predicted_x - static_cast<double>(target.x), predicted_y - static_cast<double>(target.y));
            if (residual <= cleanup_threshold)
            {
                keep.push_back(row);
            }
        }
        return keep;
    };

    auto keep_indices = affine_keep();
    if (low_count)
    {
        auto projective_keep = homography_keep();
        if (static_cast<int64_t>(projective_keep.size()) >=
            static_cast<int64_t>(keep_indices.size()) + GEOMETRIC_RESIDUAL_CLEANUP_PROJECTIVE_MIN_GAIN)
        {
            keep_indices = std::move(projective_keep);
        }
    }

    const auto min_keep =
        low_count ? std::max<int64_t>(GEOMETRIC_RESIDUAL_CLEANUP_LOW_COUNT_MIN_KEEP, cpu_matches.size(0) / 2)
                  : std::max<int64_t>(GEOMETRIC_RESIDUAL_CLEANUP_MIN_MATCHES, cpu_matches.size(0) / 2);
    if (static_cast<int64_t>(keep_indices.size()) < min_keep)
    {
        return {matches, scores};
    }
    if (cpu_matches.size(0) >= GEOMETRIC_RESIDUAL_CLEANUP_HIGH_COUNT_FLOOR &&
        static_cast<int64_t>(keep_indices.size()) < GEOMETRIC_RESIDUAL_CLEANUP_HIGH_COUNT_FLOOR)
    {
        return {matches, scores};
    }
    return selectMatches(cpu_matches, scores, keep_indices);
}

double lowerMedian(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }
    const auto median_index = (values.size() - 1) / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(median_index), values.end());
    return values[median_index];
}

std::pair<torch::Tensor, torch::Tensor>
filterLocalDisplacementConsistentMatches(const FeatureSet& features_a, const FeatureSet& features_b,
                                         const torch::Tensor& matches, const torch::Tensor& scores,
                                         double threshold_px = LOCAL_DISPLACEMENT_CONSISTENCY_THRESHOLD,
                                         int64_t neighbors = LOCAL_DISPLACEMENT_CONSISTENCY_NEIGHBORS,
                                         int64_t min_inliers = LOCAL_DISPLACEMENT_CONSISTENCY_MIN_INLIERS)
{
    if (!matches.defined() || matches.size(0) < min_inliers)
    {
        return {matches, scores};
    }

    const auto cpu_matches = matches.to(torch::kCPU, torch::kInt64).contiguous();
    const auto cpu_scores = scores.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto count = cpu_matches.size(0);
    const auto neighbor_count = std::min<int64_t>(std::max<int64_t>(2, neighbors), count);
    const auto* match_data = cpu_matches.data_ptr<int64_t>();
    const auto* score_data = cpu_scores.data_ptr<float>();
    const auto* points_a_data = points_a.data_ptr<float>();
    const auto* points_b_data = points_b.data_ptr<float>();

    std::vector<cv::Point2d> source_points;
    std::vector<cv::Point2d> displacements;
    source_points.reserve(static_cast<std::size_t>(count));
    displacements.reserve(static_cast<std::size_t>(count));
    for (int64_t row = 0; row < count; ++row)
    {
        const auto source = match_data[row * 2];
        const auto target = match_data[row * 2 + 1];
        const double ax = static_cast<double>(points_a_data[source * 2]);
        const double ay = static_cast<double>(points_a_data[source * 2 + 1]);
        const double bx = static_cast<double>(points_b_data[target * 2]);
        const double by = static_cast<double>(points_b_data[target * 2 + 1]);
        source_points.emplace_back(ax, ay);
        displacements.emplace_back(bx - ax, by - ay);
    }

    std::vector<int64_t> keep_indices;
    keep_indices.reserve(static_cast<std::size_t>(count));
    for (int64_t row = 0; row < count; ++row)
    {
        std::vector<std::pair<double, int64_t>> distances;
        distances.reserve(static_cast<std::size_t>(count));
        for (int64_t candidate = 0; candidate < count; ++candidate)
        {
            const double dx =
                source_points[static_cast<std::size_t>(row)].x - source_points[static_cast<std::size_t>(candidate)].x;
            const double dy =
                source_points[static_cast<std::size_t>(row)].y - source_points[static_cast<std::size_t>(candidate)].y;
            distances.emplace_back(dx * dx + dy * dy, candidate);
        }
        std::partial_sort(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(neighbor_count),
                          distances.end(),
                          [](const auto& lhs, const auto& rhs)
                          {
                              if (lhs.first == rhs.first)
                              {
                                  return lhs.second < rhs.second;
                              }
                              return lhs.first < rhs.first;
                          });
        std::vector<double> local_dx;
        std::vector<double> local_dy;
        local_dx.reserve(static_cast<std::size_t>(neighbor_count));
        local_dy.reserve(static_cast<std::size_t>(neighbor_count));
        for (int64_t index = 0; index < neighbor_count; ++index)
        {
            const auto neighbor = distances[static_cast<std::size_t>(index)].second;
            local_dx.push_back(displacements[static_cast<std::size_t>(neighbor)].x);
            local_dy.push_back(displacements[static_cast<std::size_t>(neighbor)].y);
        }
        const double median_dx = lowerMedian(std::move(local_dx));
        const double median_dy = lowerMedian(std::move(local_dy));
        const double residual = std::hypot(displacements[static_cast<std::size_t>(row)].x - median_dx,
                                           displacements[static_cast<std::size_t>(row)].y - median_dy);
        if (residual <= threshold_px)
        {
            keep_indices.push_back(row);
        }
    }
    if (static_cast<int64_t>(keep_indices.size()) < min_inliers)
    {
        const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    std::sort(keep_indices.begin(), keep_indices.end(),
              [&](int64_t lhs, int64_t rhs)
              {
                  return score_data[lhs] > score_data[rhs];
              });
    return selectMatches(cpu_matches, cpu_scores, keep_indices);
}

std::vector<int64_t> maskToIndices(const cv::Mat& mask)
{
    std::vector<int64_t> indices;
    if (mask.empty())
    {
        return indices;
    }
    indices.reserve(static_cast<std::size_t>(mask.rows));
    for (int row = 0; row < mask.rows; ++row)
    {
        if (mask.at<unsigned char>(row, 0) != 0)
        {
            indices.push_back(row);
        }
    }
    return indices;
}

std::vector<float> tensorToScoreVector(const torch::Tensor& scores)
{
    const auto cpu_scores = scores.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<float> values(static_cast<std::size_t>(cpu_scores.size(0)), 0.0F);
    for (int64_t index = 0; index < cpu_scores.size(0); ++index)
    {
        values[static_cast<std::size_t>(index)] = cpu_scores.index({index}).item<float>();
    }
    return values;
}

std::vector<int64_t> sortedIndicesByScore(const std::vector<float>& scores)
{
    std::vector<int64_t> indices(scores.size());
    for (int64_t index = 0; index < static_cast<int64_t>(scores.size()); ++index)
    {
        indices[static_cast<std::size_t>(index)] = index;
    }
    std::sort(indices.begin(), indices.end(),
              [&](int64_t lhs, int64_t rhs)
              {
                  return scores[static_cast<std::size_t>(lhs)] > scores[static_cast<std::size_t>(rhs)];
              });
    return indices;
}

std::vector<int64_t> prefixIndices(const std::vector<int64_t>& sorted_indices, int64_t prefix_size)
{
    const auto clamped_size = std::min<int64_t>(prefix_size, static_cast<int64_t>(sorted_indices.size()));
    return std::vector<int64_t>(sorted_indices.begin(), sorted_indices.begin() + clamped_size);
}

std::vector<int64_t> geometricConsistencyPrefixSizes(int64_t candidate_count)
{
    std::vector<int64_t> sizes;
    if (candidate_count < GEOMETRIC_CONSISTENCY_MIN_MATCHES)
    {
        return sizes;
    }
    for (int64_t size = GEOMETRIC_CONSISTENCY_MIN_MATCHES; size < candidate_count; size *= 2)
    {
        sizes.push_back(size);
    }
    if (sizes.empty() || sizes.back() != candidate_count)
    {
        sizes.push_back(candidate_count);
    }
    return sizes;
}

double meanScoreForIndices(const std::vector<float>& scores, const std::vector<int64_t>& indices)
{
    if (indices.empty())
    {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto index : indices)
    {
        sum += static_cast<double>(scores[static_cast<std::size_t>(index)]);
    }
    return sum / static_cast<double>(indices.size());
}

std::vector<int64_t> remapMaskIndices(const std::vector<int64_t>& prefix_indices, const cv::Mat& mask)
{
    const auto local_indices = maskToIndices(mask);
    std::vector<int64_t> global_indices;
    global_indices.reserve(local_indices.size());
    for (const auto local_index : local_indices)
    {
        if (local_index >= 0 && local_index < static_cast<int64_t>(prefix_indices.size()))
        {
            global_indices.push_back(prefix_indices[static_cast<std::size_t>(local_index)]);
        }
    }
    return global_indices;
}

struct GeometricCandidate
{
    std::vector<int64_t> indices;
    double quality = -1.0;
};

double geometricCandidateQuality(double score_mean, int64_t inlier_count, double source_spread, double target_spread)
{
    const auto capped_inliers = std::min<int64_t>(inlier_count, GEOMETRIC_CONSISTENCY_MAX_OUTPUT_MATCHES);
    const auto spread = std::clamp(std::min(source_spread, target_spread), 0.0, 1.0);
    return static_cast<double>(capped_inliers) + score_mean * 0.25 + spread * GEOMETRIC_SPREAD_QUALITY_WEIGHT;
}

double geometricCandidateQuality(double score_mean, int64_t inlier_count)
{
    return geometricCandidateQuality(score_mean, inlier_count, 1.0, 1.0);
}

double pointBoundingArea(const std::vector<cv::Point2f>& points, const std::vector<int64_t>& indices)
{
    if (indices.size() < 2)
    {
        return 0.0;
    }
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto index : indices)
    {
        const auto& point = points[static_cast<std::size_t>(index)];
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }
    return static_cast<double>(std::max(0.0F, max_x - min_x)) * static_cast<double>(std::max(0.0F, max_y - min_y));
}

double pointSpreadRatio(const std::vector<cv::Point2f>& points, const std::vector<int64_t>& indices,
                        double reference_area)
{
    if (reference_area <= 1.0e-6)
    {
        return 0.0;
    }
    const auto area = pointBoundingArea(points, indices);
    return std::clamp(area / reference_area, 0.0, 1.0);
}

double pointReferenceArea(const std::vector<cv::Point2f>& points)
{
    std::vector<int64_t> indices;
    indices.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        indices.push_back(static_cast<int64_t>(index));
    }
    return pointBoundingArea(points, indices);
}

GeometricCandidate estimateBestGeometricCandidate(const std::vector<cv::Point2f>& source_points,
                                                  const std::vector<cv::Point2f>& target_points,
                                                  const std::vector<float>& scores,
                                                  const std::vector<int64_t>& prefix_indices)
{
    GeometricCandidate best;
    if (static_cast<int64_t>(prefix_indices.size()) < GEOMETRIC_CONSISTENCY_MIN_MATCHES)
    {
        return best;
    }

    std::vector<cv::Point2f> prefix_source;
    std::vector<cv::Point2f> prefix_target;
    prefix_source.reserve(prefix_indices.size());
    prefix_target.reserve(prefix_indices.size());
    for (const auto index : prefix_indices)
    {
        prefix_source.push_back(source_points[static_cast<std::size_t>(index)]);
        prefix_target.push_back(target_points[static_cast<std::size_t>(index)]);
    }
    const auto source_reference_area = pointReferenceArea(source_points);
    const auto target_reference_area = pointReferenceArea(target_points);

    auto consider = [&](const cv::Mat& mask)
    {
        auto indices = remapMaskIndices(prefix_indices, mask);
        if (static_cast<int64_t>(indices.size()) < GEOMETRIC_CONSISTENCY_MIN_INLIERS)
        {
            return;
        }
        const auto score_mean = meanScoreForIndices(scores, indices);
        const auto source_spread = pointSpreadRatio(source_points, indices, source_reference_area);
        const auto target_spread = pointSpreadRatio(target_points, indices, target_reference_area);
        const auto capped_inliers =
            std::min<int64_t>(static_cast<int64_t>(indices.size()), GEOMETRIC_CONSISTENCY_MAX_OUTPUT_MATCHES);
        const auto spread = std::clamp(std::min(source_spread, target_spread), 0.0, 1.0);
        const auto quality =
            static_cast<double>(capped_inliers) + score_mean * 0.25 + spread * geometricSpreadQualityWeight();
        if (quality > best.quality)
        {
            best.indices = std::move(indices);
            best.quality = quality;
        }
    };

    cv::Mat affine_mask;
    cv::estimateAffine2D(prefix_source, prefix_target, affine_mask, cv::RANSAC, GEOMETRIC_CONSISTENCY_RANSAC_THRESHOLD);
    consider(affine_mask);

    cv::Mat homography_mask;
    cv::findHomography(prefix_source, prefix_target, cv::RANSAC, GEOMETRIC_CONSISTENCY_RANSAC_THRESHOLD,
                       homography_mask);
    consider(homography_mask);
    return best;
}

std::pair<torch::Tensor, torch::Tensor> filterProjectiveConsistentMatches(const FeatureSet& features_a,
                                                                          const FeatureSet& features_b,
                                                                          const torch::Tensor& matches,
                                                                          const torch::Tensor& scores)
{
    if (!matches.defined() || matches.size(0) < GEOMETRIC_CONSISTENCY_MIN_MATCHES)
    {
        return {matches, scores};
    }

    const auto cpu_matches = matches.to(torch::kCPU, torch::kInt64).contiguous();
    const auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<cv::Point2f> source_points;
    std::vector<cv::Point2f> target_points;
    source_points.reserve(static_cast<std::size_t>(cpu_matches.size(0)));
    target_points.reserve(static_cast<std::size_t>(cpu_matches.size(0)));
    for (int64_t index = 0; index < cpu_matches.size(0); ++index)
    {
        const auto source = cpu_matches.index({index, 0}).item<int64_t>();
        const auto target = cpu_matches.index({index, 1}).item<int64_t>();
        source_points.emplace_back(points_a.index({source, 0}).item<float>(),
                                   points_a.index({source, 1}).item<float>());
        target_points.emplace_back(points_b.index({target, 0}).item<float>(),
                                   points_b.index({target, 1}).item<float>());
    }

    const auto score_values = tensorToScoreVector(scores);
    const auto sorted_indices = sortedIndicesByScore(score_values);
    GeometricCandidate best;
    for (const auto prefix_size : geometricConsistencyPrefixSizes(static_cast<int64_t>(sorted_indices.size())))
    {
        auto candidate = estimateBestGeometricCandidate(source_points, target_points, score_values,
                                                        prefixIndices(sorted_indices, prefix_size));
        if (candidate.quality > best.quality)
        {
            best = std::move(candidate);
        }
    }
    if (static_cast<int64_t>(best.indices.size()) < GEOMETRIC_CONSISTENCY_MIN_INLIERS)
    {
        const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    std::sort(best.indices.begin(), best.indices.end(),
              [&](int64_t lhs, int64_t rhs)
              {
                  return score_values[static_cast<std::size_t>(lhs)] > score_values[static_cast<std::size_t>(rhs)];
              });
    if (static_cast<int64_t>(best.indices.size()) > GEOMETRIC_CONSISTENCY_MAX_OUTPUT_MATCHES)
    {
        best.indices.resize(static_cast<std::size_t>(GEOMETRIC_CONSISTENCY_MAX_OUTPUT_MATCHES));
    }
    auto selected = selectMatches(cpu_matches, scores, best.indices);
    return cleanupAffineResidualMatches(features_a, features_b, selected.first, selected.second);
}

std::pair<torch::Tensor, torch::Tensor> filterSparseGeometryConsistentMatches(const FeatureSet& features_a,
                                                                              const FeatureSet& features_b,
                                                                              const torch::Tensor& matches,
                                                                              const torch::Tensor& scores)
{
    const auto filter_mode = sparseGeometryFilter();
    if (filter_mode == SparseGeometryFilter::Local)
    {
        return filterLocalDisplacementConsistentMatches(features_a, features_b, matches, scores);
    }
    auto projective = filterProjectiveConsistentMatches(features_a, features_b, matches, scores);
    if (filter_mode != SparseGeometryFilter::Adaptive || !matches.defined() ||
        matches.size(0) > LOCAL_DISPLACEMENT_CONSISTENCY_MAX_ADAPTIVE_CANDIDATES)
    {
        return projective;
    }
    auto local = filterLocalDisplacementConsistentMatches(features_a, features_b, matches, scores);
    if (shouldPreferLocalDisplacementGeometry(projective.first.size(0), local.first.size(0)))
    {
        return local;
    }
    return projective;
}

std::pair<torch::Tensor, torch::Tensor> mutualGraphLogitMatches(const torch::Tensor& logits, int64_t keypoint_count_a,
                                                                int64_t keypoint_count_b)
{
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (keypoint_count_a == 0 || keypoint_count_b == 0)
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }

    auto pair_logits =
        logits.index({torch::indexing::Slice(0, keypoint_count_a), torch::indexing::Slice(0, keypoint_count_b)})
            .to(torch::kCPU, torch::kFloat32)
            .contiguous();
    auto row_logits =
        logits.index({torch::indexing::Slice(0, keypoint_count_a), torch::indexing::Slice(0, keypoint_count_b + 1)})
            .to(torch::kCPU, torch::kFloat32)
            .contiguous();
    auto col_logits =
        logits.index({torch::indexing::Slice(0, keypoint_count_a + 1), torch::indexing::Slice(0, keypoint_count_b)})
            .to(torch::kCPU, torch::kFloat32)
            .contiguous();
    const auto row_probs = torch::softmax(row_logits, 1);
    const auto best_ab = row_logits.max(1);
    const auto best_scores = std::get<0>(best_ab);
    const auto target_indices = std::get<1>(best_ab).to(torch::kCPU, torch::kInt64).contiguous();
    const auto best_ba = std::get<1>(col_logits.max(0)).to(torch::kCPU, torch::kInt64).contiguous();

    struct Candidate
    {
        int64_t source = 0;
        int64_t target = 0;
        float score = 0.0F;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(std::min(keypoint_count_a, keypoint_count_b)));
    for (int64_t source = 0; source < keypoint_count_a; ++source)
    {
        const auto target = target_indices.index({source}).item<int64_t>();
        if (target < keypoint_count_b && best_ba.index({target}).item<int64_t>() == source)
        {
            candidates.push_back(Candidate{source, target, row_probs.index({source, target}).item<float>()});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs)
              {
                  return lhs.score > rhs.score;
              });
    if (static_cast<int64_t>(candidates.size()) > GRAPH_GREEDY_MAX_MATCHES)
    {
        candidates.resize(static_cast<std::size_t>(GRAPH_GREEDY_MAX_MATCHES));
    }

    if (candidates.empty())
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    std::vector<int64_t> matches;
    std::vector<float> scores;
    matches.reserve(candidates.size() * 2);
    scores.reserve(candidates.size());
    for (const auto& candidate : candidates)
    {
        matches.push_back(candidate.source);
        matches.push_back(candidate.target);
        scores.push_back(candidate.score);
    }
    return {
        torch::from_blob(matches.data(), {static_cast<int64_t>(scores.size()), 2}, long_options).clone().contiguous(),
        torch::from_blob(scores.data(), {static_cast<int64_t>(scores.size())}, float_options).clone().contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> greedyGraphLogitMatches(const torch::Tensor& logits, int64_t keypoint_count_a,
                                                                int64_t keypoint_count_b)
{
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (keypoint_count_a == 0 || keypoint_count_b == 0)
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }

    auto pair_logits =
        logits.index({torch::indexing::Slice(0, keypoint_count_a), torch::indexing::Slice(0, keypoint_count_b)})
            .to(torch::kCPU, torch::kFloat32)
            .contiguous();
    auto row_dustbin = logits.index({torch::indexing::Slice(0, keypoint_count_a), keypoint_count_b})
                           .to(torch::kCPU, torch::kFloat32)
                           .contiguous();
    auto col_dustbin = logits.index({keypoint_count_a, torch::indexing::Slice(0, keypoint_count_b)})
                           .to(torch::kCPU, torch::kFloat32)
                           .contiguous();
    auto row_logits =
        logits.index({torch::indexing::Slice(0, keypoint_count_a), torch::indexing::Slice(0, keypoint_count_b + 1)})
            .to(torch::kCPU, torch::kFloat32)
            .contiguous();
    const auto row_probs = torch::softmax(row_logits, 1);
    auto flat = pair_logits.reshape({keypoint_count_a * keypoint_count_b});
    auto sorted = flat.sort(-1, true);
    auto sorted_scores = std::get<0>(sorted);
    auto sorted_indices = std::get<1>(sorted).to(torch::kCPU, torch::kInt64).contiguous();

    std::vector<char> used_a(static_cast<size_t>(keypoint_count_a), 0);
    std::vector<char> used_b(static_cast<size_t>(keypoint_count_b), 0);
    std::vector<int64_t> matches;
    std::vector<float> scores;
    const auto max_matches = std::min<int64_t>(GRAPH_GREEDY_MAX_MATCHES, std::min(keypoint_count_a, keypoint_count_b));
    matches.reserve(static_cast<size_t>(max_matches * 2));
    scores.reserve(static_cast<size_t>(max_matches));
    for (int64_t rank = 0; rank < sorted_indices.size(0) && static_cast<int64_t>(scores.size()) < max_matches; ++rank)
    {
        const auto flat_index = sorted_indices.index({rank}).item<int64_t>();
        const auto source = flat_index / keypoint_count_b;
        const auto target = flat_index % keypoint_count_b;
        const auto score = sorted_scores.index({rank}).item<float>();
        if (score <= row_dustbin.index({source}).item<float>() || score <= col_dustbin.index({target}).item<float>())
        {
            continue;
        }
        if (!used_a[static_cast<size_t>(source)] && !used_b[static_cast<size_t>(target)])
        {
            used_a[static_cast<size_t>(source)] = 1;
            used_b[static_cast<size_t>(target)] = 1;
            matches.push_back(source);
            matches.push_back(target);
            scores.push_back(row_probs.index({source, target}).item<float>());
        }
    }
    if (scores.empty())
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    return {
        torch::from_blob(matches.data(), {static_cast<int64_t>(scores.size()), 2}, long_options).clone().contiguous(),
        torch::from_blob(scores.data(), {static_cast<int64_t>(scores.size())}, float_options).clone().contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> relaxedGraphLogitMatches(const torch::Tensor& logits, int64_t keypoint_count_a,
                                                                 int64_t keypoint_count_b)
{
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (keypoint_count_a == 0 || keypoint_count_b == 0)
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }

    auto pair_logits =
        logits.index({torch::indexing::Slice(0, keypoint_count_a), torch::indexing::Slice(0, keypoint_count_b)})
            .to(torch::kCPU, torch::kFloat32)
            .contiguous();
    auto flat = pair_logits.reshape({keypoint_count_a * keypoint_count_b});
    auto sorted = flat.sort(-1, true);
    auto sorted_scores = std::get<0>(sorted);
    auto sorted_indices = std::get<1>(sorted).to(torch::kCPU, torch::kInt64).contiguous();

    std::vector<char> used_a(static_cast<size_t>(keypoint_count_a), 0);
    std::vector<char> used_b(static_cast<size_t>(keypoint_count_b), 0);
    std::vector<int64_t> matches;
    std::vector<float> scores;
    const auto max_matches = std::min<int64_t>(GRAPH_GREEDY_MAX_MATCHES, std::min(keypoint_count_a, keypoint_count_b));
    matches.reserve(static_cast<size_t>(max_matches * 2));
    scores.reserve(static_cast<size_t>(max_matches));
    for (int64_t rank = 0; rank < sorted_indices.size(0) && static_cast<int64_t>(scores.size()) < max_matches; ++rank)
    {
        const auto flat_index = sorted_indices.index({rank}).item<int64_t>();
        const auto source = flat_index / keypoint_count_b;
        const auto target = flat_index % keypoint_count_b;
        if (!used_a[static_cast<size_t>(source)] && !used_b[static_cast<size_t>(target)])
        {
            used_a[static_cast<size_t>(source)] = 1;
            used_b[static_cast<size_t>(target)] = 1;
            matches.push_back(source);
            matches.push_back(target);
            scores.push_back(sorted_scores.index({rank}).item<float>());
        }
    }
    if (scores.empty())
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    return {
        torch::from_blob(matches.data(), {static_cast<int64_t>(scores.size()), 2}, long_options).clone().contiguous(),
        torch::from_blob(scores.data(), {static_cast<int64_t>(scores.size())}, float_options).clone().contiguous()};
}

template <typename GraphMatcherT>
std::pair<torch::Tensor, torch::Tensor> matchSparseFeatures(const FeatureSet& features_a, const FeatureSet& features_b,
                                                            GraphMatcherT& matcher)
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (!features_a.descriptors.defined() || !features_b.descriptors.defined())
    {
        throw std::invalid_argument("descriptors must be defined");
    }
    if (features_a.descriptors.dim() != 2 || features_b.descriptors.dim() != 2)
    {
        throw std::invalid_argument("descriptors must be 2D");
    }
    if (features_a.descriptors.size(1) != features_b.descriptors.size(1))
    {
        throw std::invalid_argument("descriptor dimensions must match");
    }
    if (features_a.descriptors.size(0) == 0 || features_b.descriptors.size(0) == 0)
    {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    const auto debug = matchDebugEnabled();
    const auto descriptor_matches = matchMutualDescriptorFeatures(features_a, features_b, false);

    auto matcher_device = torch::Device(torch::kCPU);
    const auto parameters = matcher.parameters();
    if (!parameters.empty())
    {
        matcher_device = parameters.front().device();
    }
    std::pair<torch::Tensor, torch::Tensor> greedy{torch::empty({0, 2}, long_options),
                                                   torch::empty({0}, float_options)};
    if (shouldUseGraphMatcherForSparseCount(features_a.descriptors.size(0), features_b.descriptors.size(0)))
    {
        const auto output = matcher.forward(features_a.descriptors.to(matcher_device, torch::kFloat32),
                                            features_a.keypoints.to(matcher_device, torch::kFloat32),
                                            features_b.descriptors.to(matcher_device, torch::kFloat32),
                                            features_b.keypoints.to(matcher_device, torch::kFloat32));
        greedy = mutualGraphLogitMatches(output.logits, features_a.descriptors.size(0), features_b.descriptors.size(0));
        if (greedy.first.size(0) < GEOMETRIC_CONSISTENCY_MIN_MATCHES)
        {
            greedy =
                greedyGraphLogitMatches(output.logits, features_a.descriptors.size(0), features_b.descriptors.size(0));
        }
        if (greedy.first.size(0) < GEOMETRIC_CONSISTENCY_MIN_MATCHES)
        {
            greedy =
                relaxedGraphLogitMatches(output.logits, features_a.descriptors.size(0), features_b.descriptors.size(0));
        }
    }
    else if (debug)
    {
        std::cerr << "match debug: graph matcher skipped for sparse counts " << features_a.descriptors.size(0) << "x"
                  << features_b.descriptors.size(0) << '\n';
    }
    auto matches = greedy.first;
    auto scores = greedy.second;
    if (debug)
    {
        std::cerr << "match debug: features_a=" << features_a.descriptors.size(0)
                  << " features_b=" << features_b.descriptors.size(0)
                  << " descriptor_matches=" << descriptor_matches.first.size(0) << " graph_matches=" << matches.size(0)
                  << '\n';
    }
    if (returnRawGraphMatchesForDebug())
    {
        return {matches, scores};
    }
    Timer debug_timer;
    auto descriptor_topk =
        useTopKGeometryForDebug()
            ? matchDescriptorTopKFeatures(features_a, features_b, descriptorTopKCandidatesPerSource(), matcher_device)
            : std::pair<torch::Tensor, torch::Tensor>{torch::empty({0, 2}, long_options),
                                                      torch::empty({0}, float_options)};
    if (debug)
    {
        std::cerr << "match debug timing: descriptor_topk=" << formatSeconds(debug_timer.elapsedSeconds()) << "s\n";
        debug_timer.reset();
    }
    if (descriptor_topk.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_MATCHES)
    {
        if (descriptorTopKProjectiveBeforeRotation())
        {
            auto unique_projective_topk =
                mergeSparseMatchCandidates(descriptor_topk.first, descriptor_topk.second,
                                           torch::empty({0, 2}, long_options), torch::empty({0}, float_options));
            auto topk_projective = filterSparseGeometryConsistentMatches(
                features_a, features_b, unique_projective_topk.first, unique_projective_topk.second);
            if (debug)
            {
                std::cerr << "match debug timing: topk_projective_raw=" << formatSeconds(debug_timer.elapsedSeconds())
                          << "s\n";
                debug_timer.reset();
                std::cerr << "match debug: descriptor_topk_projective_raw_matches=" << topk_projective.first.size(0)
                          << '\n';
            }
            if (topk_projective.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_INLIERS)
            {
                return topk_projective;
            }
        }
        auto topk_rotation =
            filterRotationConsistentMatches(features_a, features_b, descriptor_topk.first, descriptor_topk.second);
        auto unique_rotation =
            mergeSparseMatchCandidates(topk_rotation.first, topk_rotation.second, torch::empty({0, 2}, long_options),
                                       torch::empty({0}, float_options));
        if (debug)
        {
            std::cerr << "match debug timing: topk_rotation=" << formatSeconds(debug_timer.elapsedSeconds()) << "s\n";
            debug_timer.reset();
        }
        if (debug)
        {
            std::cerr << "match debug: descriptor_topk_matches=" << descriptor_topk.first.size(0)
                      << " descriptor_topk_rotation_matches=" << unique_rotation.first.size(0) << '\n';
        }
        if (sparseGeometryFilter() == SparseGeometryFilter::RotationOnly &&
            unique_rotation.first.size(0) < DESCRIPTOR_PROJECTIVE_RESCUE_MAX_BASE_MATCHES)
        {
            auto unique_projective_topk =
                mergeSparseMatchCandidates(descriptor_topk.first, descriptor_topk.second,
                                           torch::empty({0, 2}, long_options), torch::empty({0}, float_options));
            auto projective_rescue = filterSparseGeometryConsistentMatches(
                features_a, features_b, unique_projective_topk.first, unique_projective_topk.second);
            if (debug)
            {
                std::cerr << "match debug: rotation_only_projective_rescue_matches=" << projective_rescue.first.size(0)
                          << '\n';
            }
            if (shouldUseProjectiveTopKRescue(unique_rotation.first.size(0), projective_rescue.first.size(0)))
            {
                return projective_rescue;
            }
        }
        if (shouldReturnRotationOnlyMatches(unique_rotation.first.size(0)))
        {
            if (debug)
            {
                std::cerr << "match debug: returning rotation-only descriptor matches before projective filter\n";
            }
            return unique_rotation;
        }
        if (unique_rotation.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_MATCHES)
        {
            auto topk_consistent = filterSparseGeometryConsistentMatches(features_a, features_b, unique_rotation.first,
                                                                         unique_rotation.second);
            if (debug)
            {
                std::cerr << "match debug timing: topk_projective=" << formatSeconds(debug_timer.elapsedSeconds())
                          << "s\n";
                debug_timer.reset();
            }
            if (debug)
            {
                std::cerr << "match debug: descriptor_topk_geometric_matches=" << topk_consistent.first.size(0) << '\n';
            }
            if (topk_consistent.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_INLIERS)
            {
                if (descriptor_matches.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_MATCHES)
                {
                    auto descriptor_consistent = filterSparseGeometryConsistentMatches(
                        features_a, features_b, descriptor_matches.first, descriptor_matches.second);
                    if (debug)
                    {
                        std::cerr << "match debug: descriptor_mutual_geometric_matches="
                                  << descriptor_consistent.first.size(0) << '\n';
                    }
                    if (descriptor_consistent.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_INLIERS &&
                        shouldPreferMutualDescriptorGeometry(descriptor_consistent.first.size(0),
                                                             topk_consistent.first.size(0)))
                    {
                        return descriptor_consistent;
                    }
                }
                if (topk_consistent.first.size(0) < DESCRIPTOR_PROJECTIVE_RESCUE_MAX_BASE_MATCHES)
                {
                    auto unique_projective_topk = mergeSparseMatchCandidates(
                        descriptor_topk.first, descriptor_topk.second, torch::empty({0, 2}, long_options),
                        torch::empty({0}, float_options));
                    auto projective_rescue = filterSparseGeometryConsistentMatches(
                        features_a, features_b, unique_projective_topk.first, unique_projective_topk.second);
                    if (debug)
                    {
                        std::cerr << "match debug: descriptor_projective_rescue_matches="
                                  << projective_rescue.first.size(0) << '\n';
                    }
                    if (shouldUseProjectiveTopKRescue(topk_consistent.first.size(0), projective_rescue.first.size(0)))
                    {
                        return projective_rescue;
                    }
                }
                if (descriptorTopKCandidatesPerSource() > DESCRIPTOR_CONSERVATIVE_TOPK_CANDIDATES_PER_SOURCE &&
                    topk_consistent.first.size(0) <= DESCRIPTOR_CONSERVATIVE_TOPK_FALLBACK_MAX_BASE_MATCHES)
                {
                    auto conservative_topk = matchDescriptorTopKFeatures(
                        features_a, features_b, DESCRIPTOR_CONSERVATIVE_TOPK_CANDIDATES_PER_SOURCE, matcher_device);
                    auto conservative_rotation = filterRotationConsistentMatches(
                        features_a, features_b, conservative_topk.first, conservative_topk.second);
                    auto unique_conservative_rotation = mergeSparseMatchCandidates(
                        conservative_rotation.first, conservative_rotation.second, torch::empty({0, 2}, long_options),
                        torch::empty({0}, float_options));
                    if (unique_conservative_rotation.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_MATCHES)
                    {
                        auto conservative_consistent = filterSparseGeometryConsistentMatches(
                            features_a, features_b, unique_conservative_rotation.first,
                            unique_conservative_rotation.second);
                        if (debug)
                        {
                            std::cerr << "match debug: descriptor_conservative_topk_geometric_matches="
                                      << conservative_consistent.first.size(0) << '\n';
                        }
                        if (shouldUseConservativeTopKFallback(topk_consistent.first.size(0),
                                                              conservative_consistent.first.size(0)))
                        {
                            return trimLowConfidenceTopKTail(conservative_consistent.first,
                                                             conservative_consistent.second);
                        }
                    }
                }
                if (topk_consistent.first.size(0) <= DESCRIPTOR_WIDE_TOPK_FALLBACK_MAX_BASE_MATCHES)
                {
                    auto wide_topk = matchDescriptorTopKFeatures(
                        features_a, features_b, DESCRIPTOR_WIDE_TOPK_CANDIDATES_PER_SOURCE, matcher_device);
                    auto wide_rotation =
                        filterRotationConsistentMatches(features_a, features_b, wide_topk.first, wide_topk.second);
                    auto unique_wide_rotation = mergeSparseMatchCandidates(wide_rotation.first, wide_rotation.second,
                                                                           torch::empty({0, 2}, long_options),
                                                                           torch::empty({0}, float_options));
                    if (unique_wide_rotation.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_MATCHES)
                    {
                        auto wide_consistent = filterSparseGeometryConsistentMatches(
                            features_a, features_b, unique_wide_rotation.first, unique_wide_rotation.second);
                        if (debug)
                        {
                            std::cerr << "match debug: descriptor_wide_topk_geometric_matches="
                                      << wide_consistent.first.size(0) << '\n';
                        }
                        const auto wide_mean_score =
                            wide_consistent.second.numel() > 0
                                ? wide_consistent.second.to(torch::kCPU, torch::kFloat32).mean().item<double>()
                                : 0.0;
                        if (shouldUseWideTopKFallback(topk_consistent.first.size(0), wide_consistent.first.size(0),
                                                      wide_mean_score))
                        {
                            return wide_consistent;
                        }
                    }
                }
                return trimLowConfidenceTopKTail(topk_consistent.first, topk_consistent.second);
            }
        }
        if (unique_rotation.first.size(0) >= ROTATION_CONSISTENCY_MIN_MATCHES)
        {
            return unique_rotation;
        }
        if (descriptorReciprocalTopKFallback())
        {
            auto reciprocal_topk = matchDescriptorReciprocalTopKFeatures(
                features_a, features_b, descriptorTopKCandidatesPerSource(), matcher_device);
            auto reciprocal_rotation =
                filterRotationConsistentMatches(features_a, features_b, reciprocal_topk.first, reciprocal_topk.second);
            auto unique_reciprocal_rotation =
                mergeSparseMatchCandidates(reciprocal_rotation.first, reciprocal_rotation.second,
                                           torch::empty({0, 2}, long_options), torch::empty({0}, float_options));
            if (unique_reciprocal_rotation.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_MATCHES)
            {
                auto reciprocal_consistent = filterSparseGeometryConsistentMatches(
                    features_a, features_b, unique_reciprocal_rotation.first, unique_reciprocal_rotation.second);
                if (debug)
                {
                    std::cerr << "match debug: descriptor_reciprocal_topk_geometric_matches="
                              << reciprocal_consistent.first.size(0) << '\n';
                }
                if (reciprocal_consistent.first.size(0) > unique_rotation.first.size(0))
                {
                    return reciprocal_consistent;
                }
            }
        }
    }
    if (descriptor_matches.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_MATCHES)
    {
        auto descriptor_consistent = filterSparseGeometryConsistentMatches(
            features_a, features_b, descriptor_matches.first, descriptor_matches.second);
        if (debug)
        {
            std::cerr << "match debug: descriptor_geometric_matches=" << descriptor_consistent.first.size(0) << '\n';
        }
        if (descriptor_consistent.first.size(0) >= GEOMETRIC_CONSISTENCY_MIN_INLIERS)
        {
            return descriptor_consistent;
        }
    }
    if (matches.size(0) == 0)
    {
        return descriptor_matches;
    }
    auto merged = mergeSparseMatchCandidates(matches, scores, descriptor_matches.first, descriptor_matches.second);
    auto geometric = filterSparseGeometryConsistentMatches(features_a, features_b, merged.first, merged.second);
    if (debug)
    {
        std::cerr << "match debug: merged_matches=" << merged.first.size(0)
                  << " geometric_matches=" << geometric.first.size(0) << '\n';
    }
    if (geometric.first.size(0) == 0)
    {
        if (debug)
        {
            std::cerr << "match debug: geometry failed, returning descriptor mutual candidates only\n";
        }
        return descriptor_matches;
    }
    return geometric;
}

template <typename GraphMatcherT>
MatchSet matchFeatureSetsWithMatcher(const FeatureSet& features_a, const FeatureSet& features_b, GraphMatcherT& matcher)
{
    if (!features_a.dense_points.defined() || !features_b.dense_points.defined() ||
        !features_a.dense_confidence.defined() || !features_b.dense_confidence.defined())
    {
        throw std::invalid_argument("dense features must be defined");
    }
    const auto sparse = matchSparseFeatures(features_a, features_b, matcher);
    const int64_t dense_count = std::min(features_a.dense_points.size(0), features_b.dense_points.size(0));
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (dense_count == 0)
    {
        return MatchSet{sparse.first, sparse.second, torch::empty({0, 2}, float_options),
                        torch::empty({0, 2}, float_options), torch::empty({0}, float_options)};
    }

    const auto confidence_a = features_a.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    const auto confidence_b = features_b.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    return MatchSet{sparse.first, sparse.second,
                    features_a.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
                    features_b.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
                    torch::minimum(confidence_a, confidence_b).contiguous()};
}

} // namespace

MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b,
                          PlanetaryGraphMatcherImpl& matcher)
{
    return matchFeatureSetsWithMatcher(features_a, features_b, matcher);
}

MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b,
                          v21::PfmV21GraphMatcherImpl& matcher)
{
    return matchFeatureSetsWithMatcher(features_a, features_b, matcher);
}

MatchSet matchFeatureSets(const FeatureSet& features_a, const FeatureSet& features_b)
{
    if (!features_a.descriptors.defined() || !features_b.descriptors.defined() || features_a.descriptors.dim() != 2 ||
        features_b.descriptors.dim() != 2)
    {
        throw std::invalid_argument("descriptors must be 2D");
    }
    if (!features_a.dense_points.defined() || !features_b.dense_points.defined() ||
        !features_a.dense_confidence.defined() || !features_b.dense_confidence.defined())
    {
        throw std::invalid_argument("dense features must be defined");
    }
    const auto sparse = matchMutualDescriptorFeatures(features_a, features_b);
    const int64_t dense_count = std::min(features_a.dense_points.size(0), features_b.dense_points.size(0));
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (dense_count == 0)
    {
        return MatchSet{sparse.first, sparse.second, torch::empty({0, 2}, float_options),
                        torch::empty({0, 2}, float_options), torch::empty({0}, float_options)};
    }

    const auto confidence_a = features_a.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    const auto confidence_b = features_b.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    return MatchSet{sparse.first, sparse.second,
                    features_a.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
                    features_b.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
                    torch::minimum(confidence_a, confidence_b).contiguous()};
}

namespace testing
{

int64_t geometric_consistency_max_output_matches_for_test()
{
    return GEOMETRIC_CONSISTENCY_MAX_OUTPUT_MATCHES;
}

std::vector<int64_t> geometric_consistency_prefix_sizes_for_test(int64_t candidate_count)
{
    return geometricConsistencyPrefixSizes(candidate_count);
}

int64_t descriptor_topk_candidates_per_source_for_test()
{
    return DESCRIPTOR_TOPK_CANDIDATES_PER_SOURCE;
}

int64_t descriptor_topk_candidates_per_source_for_test_env()
{
    return descriptorTopKCandidatesPerSource();
}

bool descriptor_topk_projective_before_rotation_for_test()
{
    return descriptorTopKProjectiveBeforeRotation();
}

bool descriptor_reciprocal_topk_fallback_for_test()
{
    return descriptorReciprocalTopKFallback();
}

bool sparse_geometry_filter_adaptive_for_test()
{
    return sparseGeometryFilter() == SparseGeometryFilter::Adaptive;
}

bool sparse_geometry_filter_rotation_only_for_test()
{
    return sparseGeometryFilter() == SparseGeometryFilter::RotationOnly;
}

bool sparse_geometry_filter_local_for_test()
{
    return sparseGeometryFilter() == SparseGeometryFilter::Local;
}

bool should_return_rotation_only_matches_for_test(int64_t rotation_matches)
{
    return shouldReturnRotationOnlyMatches(rotation_matches);
}

bool should_prefer_local_displacement_geometry_for_test(int64_t projective_matches, int64_t local_matches)
{
    return shouldPreferLocalDisplacementGeometry(projective_matches, local_matches);
}

std::pair<torch::Tensor, torch::Tensor> merge_sparse_match_candidates_for_test(const torch::Tensor& primary_matches,
                                                                               const torch::Tensor& primary_scores,
                                                                               const torch::Tensor& fallback_matches,
                                                                               const torch::Tensor& fallback_scores)
{
    return mergeSparseMatchCandidates(primary_matches, primary_scores, fallback_matches, fallback_scores);
}

double geometric_candidate_quality_for_test(double score_mean, int64_t inlier_count)
{
    return geometricCandidateQuality(score_mean, inlier_count);
}

double geometric_candidate_quality_for_test(double score_mean, int64_t inlier_count, double source_spread,
                                            double target_spread)
{
    return geometricCandidateQuality(score_mean, inlier_count, source_spread, target_spread);
}

bool should_use_graph_matcher_for_sparse_count_for_test(int64_t keypoint_count_a, int64_t keypoint_count_b)
{
    return shouldUseGraphMatcherForSparseCount(keypoint_count_a, keypoint_count_b);
}

bool should_use_wide_topk_fallback_for_test(int64_t base_matches, int64_t wide_matches, double wide_mean_score)
{
    return shouldUseWideTopKFallback(base_matches, wide_matches, wide_mean_score);
}

bool should_use_projective_topk_rescue_for_test(int64_t base_matches, int64_t projective_matches)
{
    return shouldUseProjectiveTopKRescue(base_matches, projective_matches);
}

bool should_prefer_mutual_descriptor_geometry_for_test(int64_t mutual_matches, int64_t topk_matches)
{
    return shouldPreferMutualDescriptorGeometry(mutual_matches, topk_matches);
}

bool should_use_conservative_topk_fallback_for_test(int64_t base_matches, int64_t conservative_matches)
{
    return shouldUseConservativeTopKFallback(base_matches, conservative_matches);
}

std::pair<torch::Tensor, torch::Tensor> trim_low_confidence_topk_tail_for_test(const torch::Tensor& matches,
                                                                               const torch::Tensor& scores)
{
    return trimLowConfidenceTopKTail(matches, scores);
}

std::pair<torch::Tensor, torch::Tensor> descriptor_reciprocal_topk_matches_for_test(const FeatureSet& features_a,
                                                                                    const FeatureSet& features_b,
                                                                                    int64_t candidates_per_source)
{
    return matchDescriptorReciprocalTopKFeatures(features_a, features_b, candidates_per_source,
                                                 torch::Device(torch::kCPU));
}

torch::Device descriptor_similarity_compute_device_for_test(const torch::Tensor& descriptors_a,
                                                            const torch::Tensor& descriptors_b,
                                                            const torch::Device& compute_device)
{
    return descriptorSimilarityScores(descriptors_a, descriptors_b, compute_device).device();
}

std::pair<torch::Tensor, torch::Tensor> affine_residual_cleanup_matches_for_test(const FeatureSet& features_a,
                                                                                 const FeatureSet& features_b,
                                                                                 const torch::Tensor& matches,
                                                                                 const torch::Tensor& scores)
{
    return cleanupAffineResidualMatches(features_a, features_b, matches, scores);
}

std::pair<torch::Tensor, torch::Tensor>
local_displacement_consistent_matches_for_test(const FeatureSet& features_a, const FeatureSet& features_b,
                                               const torch::Tensor& matches, const torch::Tensor& scores,
                                               double threshold_px, int64_t neighbors, int64_t min_inliers)
{
    return filterLocalDisplacementConsistentMatches(features_a, features_b, matches, scores, threshold_px, neighbors,
                                                    min_inliers);
}

std::pair<torch::Tensor, torch::Tensor> projective_consistent_matches_for_test(const FeatureSet& features_a,
                                                                               const FeatureSet& features_b,
                                                                               const torch::Tensor& matches,
                                                                               const torch::Tensor& scores)
{
    return filterProjectiveConsistentMatches(features_a, features_b, matches, scores);
}

std::pair<torch::Tensor, torch::Tensor> rotation_consistent_matches_for_test(const FeatureSet& features_a,
                                                                             const FeatureSet& features_b,
                                                                             const torch::Tensor& matches,
                                                                             const torch::Tensor& scores)
{
    return filterRotationConsistentMatches(features_a, features_b, matches, scores);
}

std::pair<torch::Tensor, torch::Tensor>
relaxed_graph_logit_matches_for_test(const torch::Tensor& logits, int64_t keypoint_count_a, int64_t keypoint_count_b)
{
    return relaxedGraphLogitMatches(logits, keypoint_count_a, keypoint_count_b);
}

} // namespace testing

} // namespace pfm
