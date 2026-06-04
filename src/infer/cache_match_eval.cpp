#include "infer/cache_match_eval.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/nn/functional/pooling.h>
#include <torch/nn/functional/upsampling.h>

#include "infer/matching_pipeline.h"

namespace pfm
{
namespace
{

void validateSelectionInputs(const torch::Tensor& image, const torch::Tensor& descriptors,
                             const PythonDescriptorGridConfig& config)
{
    if (!image.defined() || image.dim() != 3)
    {
        throw std::invalid_argument("image must have shape CxHxW");
    }
    if (!descriptors.defined() || descriptors.dim() != 4 || descriptors.size(0) != 1)
    {
        throw std::invalid_argument("descriptors must have shape 1xDxHxW");
    }
    if (descriptors.size(1) <= 0 || descriptors.size(2) <= 0 || descriptors.size(3) <= 0)
    {
        throw std::invalid_argument("descriptor dimensions must be positive");
    }
    if (config.max_keypoints <= 0)
    {
        throw std::invalid_argument("max_keypoints must be positive");
    }
    if (!std::isfinite(config.min_intensity))
    {
        throw std::invalid_argument("min_intensity must be finite");
    }
    if (config.texture_fraction < 0.0 || config.texture_fraction > 1.0 ||
        config.weak_texture_fraction < 0.0 || config.weak_texture_fraction > 1.0 ||
        config.texture_fraction + config.weak_texture_fraction > 1.0)
    {
        throw std::invalid_argument("texture fractions must be in [0, 1] and sum to <= 1");
    }
}

torch::Tensor featureToImagePoints(const torch::Tensor& points_xy, int64_t feature_height, int64_t feature_width,
                                   int64_t image_height, int64_t image_width)
{
    if (points_xy.numel() == 0)
    {
        return points_xy.new_empty({0, 2});
    }
    const auto x = points_xy.index({torch::indexing::Slice(), 0}) *
                   (static_cast<double>(std::max<int64_t>(1, image_width - 1)) /
                    static_cast<double>(std::max<int64_t>(1, feature_width - 1)));
    const auto y = points_xy.index({torch::indexing::Slice(), 1}) *
                   (static_cast<double>(std::max<int64_t>(1, image_height - 1)) /
                    static_cast<double>(std::max<int64_t>(1, feature_height - 1)));
    return torch::stack({x, y}, 1);
}

torch::Tensor imageTextureScores(const torch::Tensor& image, const torch::Tensor& points_xy)
{
    if (points_xy.numel() == 0)
    {
        return image.new_empty({0});
    }
    auto base = image.to(torch::kFloat32).mean(0, true).unsqueeze(0);
    auto local_mean = torch::nn::functional::avg_pool2d(
        base, torch::nn::functional::AvgPool2dFuncOptions({5, 5}).stride(1).padding(2).count_include_pad(false));
    auto contrast = (base - local_mean).abs();
    auto dx = (base - torch::roll(base, {1}, {3})).abs();
    auto dy = (base - torch::roll(base, {1}, {2})).abs();
    auto texture = (contrast + dx + dy).contiguous();
    const auto height = texture.size(2);
    const auto width = texture.size(3);
    auto rounded = points_xy.round().to(torch::kLong);
    auto x = rounded.index({torch::indexing::Slice(), 0}).clamp(0, width - 1);
    auto y = rounded.index({torch::indexing::Slice(), 1}).clamp(0, height - 1);
    return texture.index({0, 0, y, x}).to(torch::kFloat32).contiguous();
}

std::vector<int64_t> sortedIndicesByScore(const torch::Tensor& indices, const torch::Tensor& scores, int64_t limit,
                                          bool descending)
{
    auto cpu_indices = indices.to(torch::kCPU, torch::kLong).contiguous();
    auto cpu_scores = scores.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto* index_data = cpu_indices.data_ptr<int64_t>();
    const auto* score_data = cpu_scores.data_ptr<float>();
    std::vector<int64_t> order(static_cast<std::size_t>(cpu_indices.size(0)));
    for (int64_t row = 0; row < cpu_indices.size(0); ++row)
    {
        order[static_cast<std::size_t>(row)] = row;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t left, int64_t right)
                     {
                         return descending ? score_data[left] > score_data[right]
                                           : score_data[left] < score_data[right];
                     });
    if (static_cast<int64_t>(order.size()) > limit)
    {
        order.resize(static_cast<std::size_t>(limit));
    }

    std::vector<int64_t> selected;
    selected.reserve(order.size());
    for (const auto row : order)
    {
        selected.push_back(index_data[row]);
    }
    return selected;
}

torch::Tensor selectedIndicesForPythonGrid(const torch::Tensor& valid_indices, const torch::Tensor& texture_scores,
                                           const PythonDescriptorGridConfig& config)
{
    if (valid_indices.size(0) <= config.max_keypoints)
    {
        return valid_indices.to(torch::kCPU, torch::kLong).contiguous();
    }

    const auto texture_count =
        std::min<int64_t>(config.max_keypoints, static_cast<int64_t>(std::llround(
                                                       static_cast<double>(config.max_keypoints) *
                                                       config.texture_fraction)));
    const auto weak_count =
        std::min<int64_t>(config.max_keypoints - texture_count,
                          static_cast<int64_t>(std::llround(static_cast<double>(config.max_keypoints) *
                                                            config.weak_texture_fraction)));
    const auto uniform_count = config.max_keypoints - texture_count - weak_count;
    std::vector<int64_t> chosen;
    chosen.reserve(static_cast<std::size_t>(config.max_keypoints));
    std::vector<bool> used(static_cast<std::size_t>(texture_scores.size(0)), false);

    auto appendByScore = [&](int64_t count, bool descending)
    {
        if (count <= 0)
        {
            return;
        }
        std::vector<int64_t> available_rows;
        available_rows.reserve(static_cast<std::size_t>(valid_indices.size(0)));
        for (int64_t row = 0; row < valid_indices.size(0); ++row)
        {
            const auto flat_index = valid_indices.index({row}).item<int64_t>();
            if (!used[static_cast<std::size_t>(flat_index)])
            {
                available_rows.push_back(row);
            }
        }
        if (available_rows.empty())
        {
            return;
        }
        auto row_tensor = torch::from_blob(available_rows.data(), {static_cast<int64_t>(available_rows.size())},
                                           torch::TensorOptions().dtype(torch::kLong))
                              .clone();
        auto candidate_indices = valid_indices.index_select(0, row_tensor);
        auto candidate_scores = texture_scores.index_select(0, candidate_indices);
        auto sorted = sortedIndicesByScore(candidate_indices, candidate_scores, count, descending);
        for (const auto flat_index : sorted)
        {
            if (!used[static_cast<std::size_t>(flat_index)])
            {
                chosen.push_back(flat_index);
                used[static_cast<std::size_t>(flat_index)] = true;
            }
        }
    };

    appendByScore(texture_count, true);
    appendByScore(weak_count, false);

    if (uniform_count > 0)
    {
        std::vector<int64_t> remaining;
        remaining.reserve(static_cast<std::size_t>(valid_indices.size(0)));
        for (int64_t row = 0; row < valid_indices.size(0); ++row)
        {
            const auto flat_index = valid_indices.index({row}).item<int64_t>();
            if (!used[static_cast<std::size_t>(flat_index)])
            {
                remaining.push_back(flat_index);
            }
        }
        const auto take = std::min<int64_t>(uniform_count, static_cast<int64_t>(remaining.size()));
        for (int64_t index = 0; index < take; ++index)
        {
            const auto source = take == 1 ? 0
                                          : static_cast<int64_t>(std::llround(
                                                static_cast<double>(index) *
                                                static_cast<double>(remaining.size() - 1) /
                                                static_cast<double>(take - 1)));
            chosen.push_back(remaining[static_cast<std::size_t>(source)]);
        }
    }

    if (chosen.empty())
    {
        return valid_indices.narrow(0, 0, config.max_keypoints).to(torch::kCPU, torch::kLong).contiguous();
    }
    if (static_cast<int64_t>(chosen.size()) > config.max_keypoints)
    {
        chosen.resize(static_cast<std::size_t>(config.max_keypoints));
    }
    return torch::from_blob(chosen.data(), {static_cast<int64_t>(chosen.size())},
                            torch::TensorOptions().dtype(torch::kLong))
        .clone()
        .contiguous();
}

torch::Tensor sampleWarp(const torch::Tensor& warp_a_to_b, const torch::Tensor& points_a_xy)
{
    if (warp_a_to_b.dim() != 3 || warp_a_to_b.size(2) != 2)
    {
        throw std::invalid_argument("warp_a_to_b must have shape HxWx2");
    }
    if (points_a_xy.numel() == 0)
    {
        return points_a_xy.new_empty({0, 2});
    }
    const auto height = warp_a_to_b.size(0);
    const auto width = warp_a_to_b.size(1);
    auto x = points_a_xy.index({torch::indexing::Slice(), 0}).to(torch::kFloat32) *
                 (2.0 / static_cast<double>(std::max<int64_t>(1, width - 1))) -
             1.0;
    auto y = points_a_xy.index({torch::indexing::Slice(), 1}).to(torch::kFloat32) *
                 (2.0 / static_cast<double>(std::max<int64_t>(1, height - 1))) -
             1.0;
    auto grid = torch::stack({x, y}, 1).reshape({1, -1, 1, 2});
    auto warp = warp_a_to_b.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32);
    auto sampled = torch::nn::functional::grid_sample(
        warp, grid,
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(true));
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous();
}

} // namespace

FeatureSet makePythonDescriptorGridFeatureSet(const torch::Tensor& image, const torch::Tensor& descriptors,
                                              const PythonDescriptorGridConfig& config)
{
    validateSelectionInputs(image, descriptors, config);

    const auto image_cpu = image.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto descriptor_cpu = descriptors.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto image_height = image_cpu.size(1);
    const auto image_width = image_cpu.size(2);
    const auto descriptor_height = descriptor_cpu.size(2);
    const auto descriptor_width = descriptor_cpu.size(3);
    auto y = torch::arange(descriptor_height, torch::kFloat32);
    auto x = torch::arange(descriptor_width, torch::kFloat32);
    auto mesh = torch::meshgrid({y, x}, "ij");
    auto keypoints = torch::stack({mesh[1], mesh[0]}, -1).reshape({-1, 2}).contiguous();
    auto image_points =
        featureToImagePoints(keypoints, descriptor_height, descriptor_width, image_height, image_width);
    auto rounded = image_points.round().to(torch::kLong);
    auto sample_x = rounded.index({torch::indexing::Slice(), 0}).clamp(0, image_width - 1);
    auto sample_y = rounded.index({torch::indexing::Slice(), 1}).clamp(0, image_height - 1);
    auto intensity = image_cpu.mean(0).index({sample_y, sample_x}).to(torch::kFloat32);
    auto valid = config.min_intensity > 0.0 ? intensity.gt(config.min_intensity)
                                            : torch::ones_like(intensity, torch::TensorOptions().dtype(torch::kBool));
    auto valid_indices = torch::nonzero(valid).reshape({-1}).to(torch::kCPU, torch::kLong).contiguous();
    auto texture_scores = imageTextureScores(image_cpu, image_points);
    auto selected = selectedIndicesForPythonGrid(valid_indices, texture_scores, config);

    auto flat_descriptors = descriptor_cpu.squeeze(0).permute({1, 2, 0}).reshape(
        {descriptor_height * descriptor_width, descriptor_cpu.size(1)});
    auto selected_keypoints = keypoints.index_select(0, selected).contiguous();
    auto selected_descriptors = flat_descriptors.index_select(0, selected).contiguous();
    auto selected_scores = texture_scores.index_select(0, selected).contiguous();
    const auto count = selected.size(0);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    FeatureSet features{selected_keypoints,
                        selected_scores,
                        selected_descriptors,
                        torch::ones({count}, float_options),
                        torch::zeros({count, 2}, float_options),
                        torch::zeros({count, 2, 2}, float_options),
                        torch::empty({0, 2}, float_options),
                        torch::empty({0}, float_options)};
    features.feature_map_width = descriptor_width;
    features.feature_map_height = descriptor_height;
    return features;
}

CacheRawMutualEvalResult evaluatePythonRawMutualDescriptorMaps(const PairArchiveSample& pair,
                                                               const torch::Tensor& descriptors_a,
                                                               const torch::Tensor& descriptors_b,
                                                               const PythonDescriptorGridConfig& keypoint_config,
                                                               int64_t max_matches, double threshold_px)
{
    if (max_matches <= 0)
    {
        throw std::invalid_argument("max_matches must be positive");
    }
    if (!std::isfinite(threshold_px) || threshold_px < 0.0)
    {
        throw std::invalid_argument("threshold_px must be non-negative and finite");
    }
    auto features_a = makePythonDescriptorGridFeatureSet(pair.view_a, descriptors_a, keypoint_config);
    auto features_b = makePythonDescriptorGridFeatureSet(pair.view_b, descriptors_b, keypoint_config);
    auto matches = matchFeatureSetsPythonRawMutual(features_a, features_b, max_matches);
    CacheRawMutualEvalResult result;
    result.matches = matches.sparse_matches.size(0);
    if (result.matches == 0)
    {
        return result;
    }

    auto sparse_matches = matches.sparse_matches.to(torch::kCPU, torch::kLong).contiguous();
    auto matched_a = features_a.keypoints.index_select(0, sparse_matches.index({torch::indexing::Slice(), 0}));
    auto matched_b = features_b.keypoints.index_select(0, sparse_matches.index({torch::indexing::Slice(), 1}));
    auto points_a = featureToImagePoints(matched_a, descriptors_a.size(2), descriptors_a.size(3), pair.view_a.size(1),
                                         pair.view_a.size(2));
    auto points_b = featureToImagePoints(matched_b, descriptors_b.size(2), descriptors_b.size(3), pair.view_b.size(1),
                                         pair.view_b.size(2));
    auto target_b = sampleWarp(pair.warp_a_to_b.to(torch::kCPU, torch::kFloat32).contiguous(), points_a);
    auto errors = (target_b - points_b).norm(2, 1);
    result.correct = errors.le(threshold_px).sum().item<int64_t>();
    result.wrong = result.matches - result.correct;
    result.precision = result.matches == 0 ? 0.0 : static_cast<double>(result.correct) / result.matches;
    return result;
}

} // namespace pfm
