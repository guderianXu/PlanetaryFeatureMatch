#include "train/trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/nn/functional/upsampling.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>

#include "core/device.h"
#include "core/tensor_utils.h"
#include "core/timer.h"
#include "data/image_dataset.h"
#include "dataloader/async_dataloader.h"
#include "dataloader/sampler.h"
#include "data/synthetic_pair_dataset.h"
#include "data/intensity_mask.h"
#include "data/synthetic_pair.h"
#include "data/synthetic_pair_cache.h"
#include "infer/feature_codec.h"
#include "infer/feature_extractor.h"
#include "infer/matching_pipeline.h"
#include "logging/csv_metric_logger.h"
#include "logging/gpu_metric_provider.h"
#include "logging/progress_logger.h"
#include "losses/losses.h"
#include "models/backbone.h"
#include "models/dense_head.h"
#include "models/planetary_graph_matcher.h"
#include "models/sparse_head.h"
#include "train/training_visualization.h"

namespace pfm {
namespace {

constexpr int64_t INPUT_CHANNELS = 1;
constexpr int64_t MAX_DESCRIPTOR_LOSS_SAMPLES = 256;
constexpr double DESCRIPTOR_DIVERSITY_WEIGHT = 0.1;
constexpr int64_t DESCRIPTOR_NEGATIVE_SAMPLE_COUNT = 63;
constexpr int64_t GRAPH_MATCHING_MAX_QUERIES = 256;
constexpr int64_t GRAPH_MATCHING_MAX_CANDIDATES = 64;
constexpr double GRAPH_MATCHING_POSITIVE_RADIUS_PIXELS = 3.0;
constexpr float OFFSET_LOSS_WEIGHT = 0.2F;
constexpr int64_t SPARSE_FEATURE_CHANNEL_MULTIPLIER = 2;
constexpr std::size_t TRAINING_VISUALIZATION_QUEUE_CAPACITY = 2048;
constexpr std::size_t TRAINING_VISUALIZATION_WORKER_COUNT = 4;
constexpr double TRAINING_MATCH_CORRECT_THRESHOLD_PIXELS = 3.0;
constexpr int64_t MAX_VISUALIZED_SPARSE_MATCH_LINES = 2048;
constexpr int64_t MAX_VISUALIZED_DENSE_MATCH_LINES = 2048;
const std::vector<std::string> TRAINING_CSV_COLUMNS = {
    "loss_total",
    "feature_loss",
    "repeatability_loss",
    "descriptor_loss",
    "matcher_loss",
    "graph_matching_loss",
    "dense_loss",
    "offset_loss",
    "confidence_loss",
    "descriptor_accuracy",
    "descriptor_diversity",
    "offset_error_px",
    "gpu_utilization_percent",
    "gpu_power_watts",
};

enum class MatchLineColor {
    Red,
    Green,
};

struct MatchCorrectnessStats {
    int64_t correct = 0;
    int64_t wrong = 0;
};

void validate_config(const TrainConfig& config) {
    if (config.image_dir.empty()) {
        throw std::invalid_argument("image_dir must not be empty");
    }
    if (config.checkpoint.empty()) {
        throw std::invalid_argument("checkpoint must not be empty");
    }
    if (config.epochs <= 0) {
        throw std::invalid_argument("epochs must be positive");
    }
    if (config.batch_size <= 0) {
        throw std::invalid_argument("batch_size must be positive");
    }
    if (config.base_channels <= 0) {
        throw std::invalid_argument("base_channels must be positive");
    }
    if (config.descriptor_dim <= 0) {
        throw std::invalid_argument("descriptor_dim must be positive");
    }
    if (config.graph_hidden_dim <= 0) {
        throw std::invalid_argument("graph_hidden_dim must be positive");
    }
    if (config.graph_attention_layers <= 0) {
        throw std::invalid_argument("graph_attention_layers must be positive");
    }
    if (config.resize < 0) {
        throw std::invalid_argument("resize must be non-negative");
    }
    if (config.pairs_per_image <= 0) {
        throw std::invalid_argument("pairs_per_image must be positive");
    }
    (void)parse_synthetic_pair_augmentation_profile(config.augmentation_profile);
    if (config.extreme_pair_ratio < 0.0 || config.extreme_pair_ratio > 1.0) {
        throw std::invalid_argument("extreme_pair_ratio must be between 0 and 1");
    }
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive and finite");
    }
    if (!std::isfinite(config.weight_decay) || config.weight_decay < 0.0) {
        throw std::invalid_argument("weight_decay must be non-negative and finite");
    }
    if (!std::isfinite(config.gradient_clip_norm) || config.gradient_clip_norm < 0.0) {
        throw std::invalid_argument("gradient_clip_norm must be non-negative and finite");
    }
    if (config.dataloader_workers < 0) {
        throw std::invalid_argument("dataloader_workers must be non-negative");
    }
    if (config.prefetch_batches <= 0) {
        throw std::invalid_argument("prefetch_batches must be positive");
    }
    if (config.visualization_samples < 0) {
        throw std::invalid_argument("visualization_samples must be non-negative");
    }
    if (config.max_keypoints <= 0) {
        throw std::invalid_argument("max_keypoints must be positive");
    }
    if (config.min_keypoints < 0) {
        throw std::invalid_argument("min_keypoints must be non-negative");
    }
    if (config.min_keypoints > config.max_keypoints) {
        throw std::invalid_argument("min_keypoints must not exceed max_keypoints");
    }
    if (config.keypoint_grid_rows <= 0) {
        throw std::invalid_argument("keypoint_grid_rows must be positive");
    }
    if (config.keypoint_grid_cols <= 0) {
        throw std::invalid_argument("keypoint_grid_cols must be positive");
    }
    if (config.keypoints_per_cell < 0) {
        throw std::invalid_argument("keypoints_per_cell must be non-negative");
    }
    if (config.nms_radius < 0) {
        throw std::invalid_argument("nms_radius must be non-negative");
    }
    validate_min_keypoint_intensity(config.min_keypoint_intensity);
}

TrainingMetric make_iteration_metric(
    const TrainConfig& config,
    int epoch,
    int iteration,
    int total_iterations,
    int images_seen,
    int total_images,
    double elapsed_seconds,
    const GpuMetrics& gpu_metrics,
    const std::unordered_map<std::string, double>& values
) {
    TrainingMetric metric;
    metric.epoch = epoch;
    metric.total_epochs = config.epochs;
    metric.iteration = iteration;
    metric.total_iterations = total_iterations;
    metric.images_seen = images_seen;
    metric.total_images = total_images;
    metric.learning_rate = config.learning_rate;
    metric.elapsed_seconds = elapsed_seconds;
    metric.values = values;
    if (gpu_metrics.utilization_percent.has_value()) {
        metric.values["gpu_utilization_percent"] = gpu_metrics.utilization_percent.value();
    }
    if (gpu_metrics.power_watts.has_value()) {
        metric.values["gpu_power_watts"] = gpu_metrics.power_watts.value();
    }
    return metric;
}

torch::Tensor ensure_grayscale(const torch::Tensor& image) {
    require_chw_image(image);
    if (channels(image) == INPUT_CHANNELS) {
        return image;
    }
    return image.mean(0, true).contiguous();
}

enum class BatchTensorLayout {
    Hw,
    Chw,
    Hwc
};

int64_t spatial_height(const torch::Tensor& tensor, BatchTensorLayout layout) {
    switch (layout) {
        case BatchTensorLayout::Hw:
        case BatchTensorLayout::Hwc:
            return tensor.size(0);
        case BatchTensorLayout::Chw:
            return tensor.size(1);
    }
    throw std::invalid_argument("unsupported batch tensor layout");
}

int64_t spatial_width(const torch::Tensor& tensor, BatchTensorLayout layout) {
    switch (layout) {
        case BatchTensorLayout::Hw:
        case BatchTensorLayout::Hwc:
            return tensor.size(1);
        case BatchTensorLayout::Chw:
            return tensor.size(2);
    }
    throw std::invalid_argument("unsupported batch tensor layout");
}

torch::Tensor pad_spatial_tensor(
    const torch::Tensor& tensor,
    int64_t target_height,
    int64_t target_width,
    BatchTensorLayout layout
) {
    const auto height = spatial_height(tensor, layout);
    const auto width = spatial_width(tensor, layout);
    if (height == target_height && width == target_width) {
        return tensor.contiguous();
    }
    if (layout == BatchTensorLayout::Hw) {
        auto padded = torch::zeros({target_height, target_width}, tensor.options());
        padded.index_put_({torch::indexing::Slice(0, height), torch::indexing::Slice(0, width)}, tensor);
        return padded.contiguous();
    }
    if (layout == BatchTensorLayout::Hwc) {
        auto padded = torch::zeros({target_height, target_width, tensor.size(2)}, tensor.options());
        padded.index_put_(
            {torch::indexing::Slice(0, height), torch::indexing::Slice(0, width), torch::indexing::Slice()},
            tensor);
        return padded.contiguous();
    }
    auto padded = torch::zeros({tensor.size(0), target_height, target_width}, tensor.options());
    padded.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(0, height), torch::indexing::Slice(0, width)},
        tensor);
    return padded.contiguous();
}

torch::Tensor stack_batch(const std::vector<torch::Tensor>& tensors, BatchTensorLayout layout) {
    int64_t target_height = 0;
    int64_t target_width = 0;
    for (const auto& tensor : tensors) {
        target_height = std::max<int64_t>(target_height, spatial_height(tensor, layout));
        target_width = std::max<int64_t>(target_width, spatial_width(tensor, layout));
    }
    std::vector<torch::Tensor> padded_tensors;
    padded_tensors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        padded_tensors.push_back(pad_spatial_tensor(tensor, target_height, target_width, layout));
    }
    return torch::stack(padded_tensors).contiguous();
}

std::string pair_visualization_stem(std::size_t index) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "pair_%06zu", index);
    return buffer;
}

torch::Tensor mask_to_image(const torch::Tensor& mask) {
    return mask.to(torch::kCPU, torch::kFloat32).unsqueeze(0).contiguous();
}

torch::Tensor make_visualization_mask(const torch::Tensor& image, double min_keypoint_intensity) {
    if (min_keypoint_intensity <= 0.0) {
        return torch::ones(
            {image.size(1), image.size(2)},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    }
    return make_intensity_mask(image.detach().to(torch::kCPU, torch::kFloat32).contiguous(), min_keypoint_intensity);
}

std::pair<int64_t, int64_t> map_feature_point(
    const torch::Tensor& point,
    int64_t map_width,
    int64_t map_height,
    int64_t image_width,
    int64_t image_height
) {
    const auto scale_x = static_cast<float>(image_width) / static_cast<float>(std::max<int64_t>(1, map_width));
    const auto scale_y = static_cast<float>(image_height) / static_cast<float>(std::max<int64_t>(1, map_height));
    const int64_t x = std::min<int64_t>(
        image_width - 1,
        std::max<int64_t>(0, std::llround(point.index({0}).item<float>() * scale_x)));
    const int64_t y = std::min<int64_t>(
        image_height - 1,
        std::max<int64_t>(0, std::llround(point.index({1}).item<float>() * scale_y)));
    return {x, y};
}

torch::Tensor ensure_visualization_rgb(const torch::Tensor& image) {
    auto output = image.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    if (output.size(0) == 1) {
        output = output.repeat({3, 1, 1}).contiguous();
    }
    return output;
}

void draw_image_point(torch::Tensor& output, int64_t x, int64_t y, MatchLineColor color = MatchLineColor::Red) {
    output.index_put_({0, y, x}, color == MatchLineColor::Red ? 1.0F : 0.0F);
    output.index_put_({1, y, x}, color == MatchLineColor::Green ? 1.0F : 0.0F);
    output.index_put_({2, y, x}, 0.0F);
}

void draw_image_line(
    torch::Tensor& output,
    int64_t x0,
    int64_t y0,
    int64_t x1,
    int64_t y1,
    MatchLineColor color = MatchLineColor::Red
) {
    const int64_t steps = std::max<int64_t>(std::abs(x1 - x0), std::abs(y1 - y0));
    for (int64_t step = 0; step <= steps; ++step) {
        const double alpha = steps == 0 ? 0.0 : static_cast<double>(step) / static_cast<double>(steps);
        const int64_t x = std::llround(static_cast<double>(x0) * (1.0 - alpha) + static_cast<double>(x1) * alpha);
        const int64_t y = std::llround(static_cast<double>(y0) * (1.0 - alpha) + static_cast<double>(y1) * alpha);
        draw_image_point(output, x, y, color);
    }
}

void draw_feature_points(
    torch::Tensor& output,
    const torch::Tensor& points,
    int64_t map_width,
    int64_t map_height,
    const torch::Tensor& mask
) {
    auto cpu_points = points.to(torch::kCPU, torch::kFloat32).contiguous();
    auto cpu_mask = mask.to(torch::kCPU, torch::kBool).contiguous();
    for (int64_t index = 0; index < cpu_points.size(0); ++index) {
        const auto [x, y] = map_feature_point(
            cpu_points[index], map_width, map_height, output.size(2), output.size(1));
        if (cpu_mask.index({y, x}).item<bool>()) {
            draw_image_point(output, x, y);
        }
    }
}

torch::Tensor feature_overlay_image(
    const torch::Tensor& image,
    const FeatureSet& features,
    double min_keypoint_intensity
) {
    auto output = ensure_visualization_rgb(image);
    if (features.keypoints.defined() && features.keypoints.numel() > 0) {
        draw_feature_points(
            output,
            features.keypoints,
            features.feature_map_width,
            features.feature_map_height,
            make_visualization_mask(image, min_keypoint_intensity));
    }
    return output;
}

torch::Tensor warp_overlay_image(const SyntheticPair& pair) {
    auto output = pair.view_a.detach().to(torch::kCPU, torch::kFloat32).clone().contiguous();
    auto valid = pair.valid_mask.to(torch::kCPU, torch::kBool).contiguous();
    for (int64_t y = 0; y < valid.size(0); y += 4) {
        for (int64_t x = 0; x < valid.size(1); x += 4) {
            if (valid.index({y, x}).item<bool>()) {
                output.index_put_({0, y, x}, 1.0F);
            }
        }
    }
    return output;
}

bool is_match_correct(
    const torch::Tensor& warp_a_to_b,
    int64_t x_a,
    int64_t y_a,
    int64_t x_b,
    int64_t y_b,
    double correct_threshold_pixels
) {
    if (!warp_a_to_b.defined() || warp_a_to_b.numel() == 0) {
        return false;
    }
    const auto warp = warp_a_to_b.to(torch::kCPU, torch::kFloat32).contiguous();
    if (warp.dim() != 3 || warp.size(2) != 2 || y_a < 0 || y_a >= warp.size(0) || x_a < 0 || x_a >= warp.size(1)) {
        return false;
    }
    const auto expected_x = warp.index({y_a, x_a, 0}).item<float>();
    const auto expected_y = warp.index({y_a, x_a, 1}).item<float>();
    const auto dx = static_cast<double>(x_b) - static_cast<double>(expected_x);
    const auto dy = static_cast<double>(y_b) - static_cast<double>(expected_y);
    return std::sqrt(dx * dx + dy * dy) <= correct_threshold_pixels;
}

void add_match_correctness(MatchCorrectnessStats& stats, bool correct) {
    if (correct) {
        ++stats.correct;
    } else {
        ++stats.wrong;
    }
}

void draw_sparse_match_lines(
    torch::Tensor& output,
    int64_t image_a_width,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b = torch::Tensor(),
    double correct_threshold_pixels = 0.0,
    MatchCorrectnessStats* stats = nullptr
) {
    if (!matches.sparse_matches.defined() || matches.sparse_matches.numel() == 0 || !features_a.keypoints.defined() ||
        !features_b.keypoints.defined()) {
        return;
    }
    auto sparse_matches = matches.sparse_matches.to(torch::kCPU, torch::kLong).contiguous();
    auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const int64_t sparse_count = std::min<int64_t>(sparse_matches.size(0), MAX_VISUALIZED_SPARSE_MATCH_LINES);
    for (int64_t index = 0; index < sparse_count; ++index) {
        const auto index_a = sparse_matches.index({index, 0}).item<int64_t>();
        const auto index_b = sparse_matches.index({index, 1}).item<int64_t>();
        if (index_a < 0 || index_a >= points_a.size(0) || index_b < 0 || index_b >= points_b.size(0)) {
            continue;
        }
        const auto [x_a, y_a] = map_feature_point(
            points_a[index_a],
            features_a.feature_map_width,
            features_a.feature_map_height,
            image_a_width,
            output.size(1));
        const auto [x_b, y_b] = map_feature_point(
            points_b[index_b],
            features_b.feature_map_width,
            features_b.feature_map_height,
            output.size(2) - image_a_width,
            output.size(1));
        const bool correct = is_match_correct(warp_a_to_b, x_a, y_a, x_b, y_b, correct_threshold_pixels);
        if (stats != nullptr) {
            add_match_correctness(*stats, correct);
        }
        draw_image_line(output, x_a, y_a, x_b + image_a_width, y_b, correct ? MatchLineColor::Green : MatchLineColor::Red);
    }
}

void draw_dense_match_lines(
    torch::Tensor& output,
    int64_t image_a_width,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b = torch::Tensor(),
    double correct_threshold_pixels = 0.0,
    MatchCorrectnessStats* stats = nullptr
) {
    if (!matches.points_a.defined() || !matches.points_b.defined() || matches.points_a.numel() == 0) {
        return;
    }
    auto points_a = matches.points_a.to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b = matches.points_b.to(torch::kCPU, torch::kFloat32).contiguous();
    const int64_t image_b_width = output.size(2) - image_a_width;
    const int64_t count = std::min<int64_t>(
        std::min(points_a.size(0), points_b.size(0)), MAX_VISUALIZED_DENSE_MATCH_LINES);
    for (int64_t index = 0; index < count; ++index) {
        const auto [x_a, y_a] = map_feature_point(
            points_a[index],
            features_a.feature_map_width,
            features_a.feature_map_height,
            image_a_width,
            output.size(1));
        const auto [x_b, y_b] = map_feature_point(
            points_b[index],
            features_b.feature_map_width,
            features_b.feature_map_height,
            image_b_width,
            output.size(1));
        const bool correct = is_match_correct(warp_a_to_b, x_a, y_a, x_b, y_b, correct_threshold_pixels);
        if (stats != nullptr) {
            add_match_correctness(*stats, correct);
        }
        draw_image_line(output, x_a, y_a, x_b + image_a_width, y_b, correct ? MatchLineColor::Green : MatchLineColor::Red);
    }
}

MatchCorrectnessStats compute_match_correctness_stats(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b,
    double correct_threshold_pixels
) {
    auto output = torch::zeros({3, warp_a_to_b.size(0), warp_a_to_b.size(1) * 2}, torch::kFloat32);
    MatchCorrectnessStats stats;
    draw_sparse_match_lines(
        output, warp_a_to_b.size(1), features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels, &stats);
    draw_dense_match_lines(
        output, warp_a_to_b.size(1), features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels, &stats);
    return stats;
}

std::string model_match_overlay_text(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const MatchCorrectnessStats& stats
) {
    return "features_a=" + std::to_string(features_a.keypoints.size(0)) + " features_b=" +
           std::to_string(features_b.keypoints.size(0)) + " sparse_matches=" +
           std::to_string(matches.sparse_matches.size(0)) + " dense_matches=" +
           std::to_string(matches.points_a.size(0)) + " correct_matches=" + std::to_string(stats.correct) +
           " wrong_matches=" + std::to_string(stats.wrong);
}

std::string model_match_overlay_text(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b,
    double correct_threshold_pixels
) {
    return model_match_overlay_text(
        features_a,
        features_b,
        matches,
        compute_match_correctness_stats(features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels));
}

torch::Tensor match_overlay_image(
    const torch::Tensor& image_a,
    const torch::Tensor& image_b,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b,
    double correct_threshold_pixels,
    double min_keypoint_intensity,
    MatchCorrectnessStats* stats = nullptr
) {
    auto output_a = feature_overlay_image(image_a, features_a, min_keypoint_intensity);
    auto output_b = feature_overlay_image(image_b, features_b, min_keypoint_intensity);
    auto output = torch::cat({output_a, output_b}, 2).contiguous();
    draw_sparse_match_lines(
        output, image_a.size(2), features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels, stats);
    draw_dense_match_lines(
        output, image_a.size(2), features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels, stats);
    return output;
}

torch::Tensor match_overlay_image(
    const torch::Tensor& image_a,
    const torch::Tensor& image_b,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    double min_keypoint_intensity
) {
    return match_overlay_image(
        image_a, image_b, features_a, features_b, matches, torch::Tensor(), 0.0, min_keypoint_intensity);
}

torch::Tensor limit_training_image_size(const torch::Tensor& image, int64_t resize) {
    const auto height = image.size(1);
    const auto width = image.size(2);
    const auto max_edge = std::max(height, width);
    if (resize == 0 || max_edge <= resize) {
        return image.contiguous();
    }

    const double scale = static_cast<double>(resize) / static_cast<double>(max_edge);
    const int64_t resized_height =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(height) * scale)));
    const int64_t resized_width =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(width) * scale)));
    return torch::nn::functional::interpolate(
               image.unsqueeze(0),
               torch::nn::functional::InterpolateFuncOptions()
                   .size(std::vector<int64_t>{resized_height, resized_width})
                   .mode(torch::kBilinear)
                   .align_corners(false))
        .squeeze(0)
        .contiguous();
}

torch::Tensor make_descriptor_sample_indices(const torch::Tensor& descriptors) {
    const auto spatial_count = descriptors.size(2) * descriptors.size(3);
    const auto sample_count = std::min<int64_t>(spatial_count, MAX_DESCRIPTOR_LOSS_SAMPLES);
    const auto sample_options = torch::TensorOptions().dtype(torch::kLong).device(descriptors.device());
    if (sample_count == spatial_count) {
        return torch::arange(spatial_count, sample_options);
    }
    return torch::linspace(0, spatial_count - 1, sample_count, descriptors.options()).round().to(torch::kLong);
}

torch::Tensor sample_spatial_descriptors(const torch::Tensor& descriptors, const torch::Tensor& sample_indices) {
    const auto batch_size = descriptors.size(0);
    const auto descriptor_dim = descriptors.size(1);
    const auto spatial_count = descriptors.size(2) * descriptors.size(3);
    auto flat = descriptors.permute({0, 2, 3, 1}).reshape({batch_size, spatial_count, descriptor_dim});
    return flat.index_select(1, sample_indices).contiguous();
}

torch::Tensor make_descriptor_target_indices(
    const torch::Tensor& warp,
    const torch::Tensor& sample_indices,
    int64_t descriptor_height,
    int64_t descriptor_width
) {
    using torch::indexing::Slice;

    const auto image_height = warp.size(1);
    const auto image_width = warp.size(2);
    auto source_y = sample_indices / descriptor_width;
    auto source_x = sample_indices.remainder(descriptor_width);
    auto image_x = (source_x * image_width / descriptor_width).clamp(0, image_width - 1);
    auto image_y = (source_y * image_height / descriptor_height).clamp(0, image_height - 1);
    auto flat_image_indices = (image_y * image_width + image_x).to(torch::kLong);
    auto flat_warp = warp.reshape({warp.size(0), image_height * image_width, 2});
    auto sampled_warp = flat_warp.index_select(1, flat_image_indices).round().to(torch::kLong);
    auto target_x = sampled_warp.index({Slice(), Slice(), 0}) * descriptor_width / image_width;
    auto target_y = sampled_warp.index({Slice(), Slice(), 1}) * descriptor_height / image_height;
    target_x = target_x.clamp(0, descriptor_width - 1);
    target_y = target_y.clamp(0, descriptor_height - 1);
    return (target_y * descriptor_width + target_x).to(torch::kLong);
}

torch::Tensor filter_descriptor_sample_indices(
    const torch::Tensor& sample_indices,
    const torch::Tensor& valid_mask,
    int64_t descriptor_height,
    int64_t descriptor_width
) {
    auto mask = valid_mask.to(torch::kFloat32).unsqueeze(1);
    auto resized = torch::nn::functional::interpolate(
        mask,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{descriptor_height, descriptor_width})
            .mode(torch::kNearest));
    auto flat = resized.squeeze(1).reshape({valid_mask.size(0), descriptor_height * descriptor_width});
    auto valid_for_batch = flat.index_select(1, sample_indices).to(torch::kBool).all(0);
    return sample_indices.index({valid_for_batch}).contiguous();
}

torch::Tensor make_descriptor_candidate_indices(const torch::Tensor& target_indices, int64_t spatial_count) {
    const auto negative_count = std::min<int64_t>(DESCRIPTOR_NEGATIVE_SAMPLE_COUNT, spatial_count - 1);
    const auto stride = std::max<int64_t>(1, spatial_count / (negative_count + 1));
    auto offsets = (torch::arange(
                        1,
                        negative_count + 1,
                        torch::TensorOptions().dtype(torch::kLong).device(target_indices.device())) * stride)
                       .reshape({1, 1, negative_count});
    auto negatives = (target_indices.unsqueeze(2) + offsets).remainder(spatial_count);
    return torch::cat({target_indices.unsqueeze(2), negatives}, 2).contiguous();
}

torch::Tensor gather_descriptor_candidates(
    const torch::Tensor& descriptors,
    const torch::Tensor& candidate_indices
) {
    const auto batch_size = descriptors.size(0);
    const auto query_count = candidate_indices.size(1);
    const auto candidate_count = candidate_indices.size(2);
    const auto descriptor_dim = descriptors.size(1);
    const auto spatial_count = descriptors.size(2) * descriptors.size(3);
    auto flat = descriptors.permute({0, 2, 3, 1}).reshape({batch_size, spatial_count, descriptor_dim});
    auto expanded_indices = candidate_indices.reshape({batch_size, query_count * candidate_count, 1})
                                .expand({batch_size, query_count * candidate_count, descriptor_dim});
    return flat.gather(1, expanded_indices).reshape({batch_size, query_count, candidate_count, descriptor_dim}).contiguous();
}

struct DescriptorTrainingMetrics {
    torch::Tensor loss;
    torch::Tensor accuracy;
    torch::Tensor diversity;
};

struct GraphMatchingTrainingMetrics {
    torch::Tensor loss;
    torch::Tensor accuracy;
    int64_t query_count = 0;
    int64_t positive_count = 0;
    int64_t dustbin_count = 0;
};

DescriptorTrainingMetrics make_sparse_descriptor_metrics(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    const auto spatial_count = descriptors_b.size(2) * descriptors_b.size(3);
    auto sample_indices = filter_descriptor_sample_indices(
        make_descriptor_sample_indices(descriptors_a),
        valid_mask,
        descriptors_a.size(2),
        descriptors_a.size(3));
    if (sample_indices.numel() == 0) {
        auto zero = torch::zeros({}, descriptors_a.options());
        return DescriptorTrainingMetrics{zero, zero, zero};
    }
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    auto target_indices = make_descriptor_target_indices(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    auto candidate_indices = make_descriptor_candidate_indices(target_indices, spatial_count);
    auto sampled_b = gather_descriptor_candidates(descriptors_b, candidate_indices);
    auto target = torch::zeros(
        {descriptors_a.size(0), sample_indices.size(0)},
        torch::TensorOptions().dtype(torch::kLong).device(descriptors_a.device()));
    auto loss = descriptor_candidate_cross_entropy_loss(sampled_a, sampled_b, target);
    auto normalized_a = sampled_a / sampled_a.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    auto normalized_b = sampled_b / sampled_b.pow(2).sum(3, true).clamp_min(1.0e-12).sqrt();
    auto logits = (normalized_a.unsqueeze(2) * normalized_b).sum(3);
    auto predictions = logits.argmax(2);
    auto accuracy = predictions.eq(target).to(torch::kFloat32).mean();
    auto diversity = descriptor_diversity_loss(sampled_a);
    return DescriptorTrainingMetrics{loss, accuracy, diversity};
}


torch::Tensor assign_graph_matching_targets(
    const torch::Tensor& keypoints_a,
    const torch::Tensor& keypoints_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double positive_radius_pixels
) {
    const auto dustbin = keypoints_b.size(0);
    auto targets = torch::full(
        {keypoints_a.size(0)},
        dustbin,
        torch::TensorOptions().dtype(torch::kLong).device(keypoints_a.device()));
    if (keypoints_a.size(0) == 0 || keypoints_b.size(0) == 0) {
        return targets;
    }

    auto points_a_cpu = keypoints_a.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b_cpu = keypoints_b.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    auto warp_cpu = warp.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    auto mask_cpu = valid_mask.detach().to(torch::kCPU, torch::kBool).contiguous();
    std::vector<int64_t> labels(static_cast<size_t>(keypoints_a.size(0)), dustbin);
    const auto radius_sq = positive_radius_pixels * positive_radius_pixels;

    for (int64_t index = 0; index < points_a_cpu.size(0); ++index) {
        const auto x = static_cast<int64_t>(std::llround(points_a_cpu.index({index, 0}).item<float>()));
        const auto y = static_cast<int64_t>(std::llround(points_a_cpu.index({index, 1}).item<float>()));
        if (y < 0 || y >= warp_cpu.size(1) || x < 0 || x >= warp_cpu.size(2)) {
            continue;
        }
        if (!mask_cpu.index({0, y, x}).item<bool>()) {
            continue;
        }
        const auto expected_x = warp_cpu.index({0, y, x, 0}).item<float>();
        const auto expected_y = warp_cpu.index({0, y, x, 1}).item<float>();
        const auto target_x = static_cast<int64_t>(std::llround(expected_x));
        const auto target_y = static_cast<int64_t>(std::llround(expected_y));
        if (target_y < 0 || target_y >= mask_cpu.size(1) || target_x < 0 || target_x >= mask_cpu.size(2)) {
            continue;
        }
        if (!mask_cpu.index({0, target_y, target_x}).item<bool>()) {
            continue;
        }
        int64_t best = dustbin;
        double best_distance = radius_sq;
        for (int64_t candidate = 0; candidate < points_b_cpu.size(0); ++candidate) {
            const auto dx = static_cast<double>(points_b_cpu.index({candidate, 0}).item<float>()) - expected_x;
            const auto dy = static_cast<double>(points_b_cpu.index({candidate, 1}).item<float>()) - expected_y;
            const auto distance = dx * dx + dy * dy;
            if (distance <= best_distance) {
                best_distance = distance;
                best = candidate;
            }
        }
        labels[static_cast<size_t>(index)] = best;
    }

    return torch::tensor(labels, targets.options());
}

torch::Tensor make_graph_candidate_indices(
    const torch::Tensor& target_indices,
    int64_t keypoint_count_b,
    int64_t max_candidates
) {
    const auto dustbin = keypoint_count_b;
    std::vector<int64_t> candidates;
    candidates.reserve(static_cast<size_t>(std::max<int64_t>(1, max_candidates)));

    auto targets_cpu = target_indices.detach().to(torch::kCPU, torch::kLong).contiguous();
    for (int64_t index = 0; index < targets_cpu.numel(); ++index) {
        const auto label = targets_cpu[index].item<int64_t>();
        if (label >= 0 && label < keypoint_count_b &&
            std::find(candidates.begin(), candidates.end(), label) == candidates.end()) {
            candidates.push_back(label);
        }
    }

    for (int64_t candidate = 0;
         candidate < keypoint_count_b && static_cast<int64_t>(candidates.size()) < max_candidates - 1;
         ++candidate) {
        if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
            candidates.push_back(candidate);
        }
    }

    candidates.push_back(dustbin);
    return torch::tensor(candidates, torch::TensorOptions().dtype(torch::kLong).device(target_indices.device()));
}

torch::Tensor scale_feature_keypoints_to_image(
    const torch::Tensor& keypoints,
    int64_t feature_width,
    int64_t feature_height,
    int64_t image_width,
    int64_t image_height
) {
    if (feature_width <= 0 || feature_height <= 0) {
        return keypoints;
    }
    auto scale = torch::tensor(
        {static_cast<float>(image_width) / static_cast<float>(feature_width),
         static_cast<float>(image_height) / static_cast<float>(feature_height)},
        keypoints.options());
    return keypoints * scale;
}

GraphMatchingTrainingMetrics make_keypoint_graph_matching_metrics(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    auto zero = torch::zeros({}, warp.options());
    if (!features_a.keypoints.defined() || !features_b.keypoints.defined() ||
        features_a.keypoints.size(0) == 0 || features_b.keypoints.size(0) == 0) {
        return GraphMatchingTrainingMetrics{zero, zero, 0, 0, 0};
    }

    const auto query_count = std::min<int64_t>(features_a.keypoints.size(0), GRAPH_MATCHING_MAX_QUERIES);
    auto query_indices = torch::arange(
        query_count,
        torch::TensorOptions().dtype(torch::kLong).device(features_a.keypoints.device()));
    auto keypoints_a = features_a.keypoints.index_select(0, query_indices).to(warp.device());
    auto descriptors_a = features_a.descriptors.index_select(0, query_indices).to(warp.device());
    auto keypoints_b = features_b.keypoints.to(warp.device());
    auto descriptors_b = features_b.descriptors.to(warp.device());

    auto image_keypoints_a = scale_feature_keypoints_to_image(
        keypoints_a,
        features_a.feature_map_width,
        features_a.feature_map_height,
        warp.size(2),
        warp.size(1));
    auto image_keypoints_b = scale_feature_keypoints_to_image(
        keypoints_b,
        features_b.feature_map_width,
        features_b.feature_map_height,
        warp.size(2),
        warp.size(1));
    auto target_full = assign_graph_matching_targets(
        image_keypoints_a,
        image_keypoints_b,
        warp,
        valid_mask,
        GRAPH_MATCHING_POSITIVE_RADIUS_PIXELS);
    auto candidate_indices = make_graph_candidate_indices(
        target_full,
        keypoints_b.size(0),
        std::min<int64_t>(GRAPH_MATCHING_MAX_CANDIDATES, keypoints_b.size(0) + 1));

    auto candidate_keypoints = keypoints_b.index_select(0, candidate_indices.narrow(0, 0, candidate_indices.size(0) - 1));
    auto candidate_descriptors = descriptors_b.index_select(0, candidate_indices.narrow(0, 0, candidate_indices.size(0) - 1));
    auto remapped_targets = torch::full(
        {target_full.size(0)},
        candidate_indices.size(0) - 1,
        torch::TensorOptions().dtype(torch::kLong).device(target_full.device()));
    for (int64_t col = 0; col < candidate_indices.size(0) - 1; ++col) {
        remapped_targets.index_put_({target_full == candidate_indices[col]}, col);
    }

    auto output = graph_matcher.forward(descriptors_a, keypoints_a, candidate_descriptors, candidate_keypoints);
    auto loss = graph_matching_cross_entropy_loss(output.logits, remapped_targets);
    auto predictions = output.logits.narrow(0, 0, remapped_targets.size(0)).argmax(1);
    auto accuracy = predictions.eq(remapped_targets).to(torch::kFloat32).mean();
    const auto dustbin_label = candidate_indices.size(0) - 1;
    const auto dustbin_count = remapped_targets.eq(dustbin_label).sum().item<int64_t>();
    return GraphMatchingTrainingMetrics{
        loss,
        accuracy,
        remapped_targets.size(0),
        remapped_targets.size(0) - dustbin_count,
        dustbin_count};
}

torch::Tensor make_keypoint_graph_matching_loss(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    return make_keypoint_graph_matching_metrics(graph_matcher, features_a, features_b, warp, valid_mask).loss;
}

torch::Tensor make_sparse_descriptor_loss(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    return make_sparse_descriptor_metrics(descriptors_a, descriptors_b, warp, valid_mask).loss;
}

torch::Tensor make_graph_matching_loss(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    auto sample_indices = filter_descriptor_sample_indices(
        make_descriptor_sample_indices(descriptors_a),
        valid_mask,
        descriptors_a.size(2),
        descriptors_a.size(3));
    if (sample_indices.numel() == 0) {
        return torch::zeros({}, descriptors_a.options());
    }
    auto target_spatial = make_descriptor_target_indices(
        warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    const auto batch_size = descriptors_a.size(0);
    const auto sample_count = sample_indices.size(0);
    const auto descriptor_dim = descriptors_b.size(1);
    const auto spatial_count = descriptors_b.size(2) * descriptors_b.size(3);
    auto flat_b = descriptors_b.permute({0, 2, 3, 1}).contiguous().reshape({batch_size, spatial_count, descriptor_dim});
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    std::vector<torch::Tensor> sampled_b_batches;
    sampled_b_batches.reserve(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        sampled_b_batches.push_back(flat_b[b].index_select(0, target_spatial[b]));
    }
    auto sampled_b = torch::stack(sampled_b_batches, 0);
    auto target_columns = torch::arange(sample_count, sample_indices.options());
    std::vector<torch::Tensor> losses;
    losses.reserve(static_cast<size_t>(batch_size));
    auto keypoints_a = torch::stack(
        {sample_indices.remainder(descriptors_a.size(3)).to(descriptors_a.dtype()),
         (sample_indices / descriptors_a.size(3)).to(descriptors_a.dtype())},
        1);
    auto desc_w = static_cast<int64_t>(descriptors_b.size(3));
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto kp_b = torch::stack(
            {target_spatial[batch].remainder(desc_w).to(descriptors_b.dtype()),
             (target_spatial[batch] / desc_w).to(descriptors_b.dtype())},
            1);
        auto output = graph_matcher.forward(sampled_a[batch], keypoints_a, sampled_b[batch], kp_b);
        losses.push_back(graph_matching_cross_entropy_loss(output.logits, target_columns));
    }
    return torch::stack(losses).mean();
}

torch::Tensor resize_mask_for_heatmap(const torch::Tensor& valid_mask, const torch::Tensor& heatmap) {
    auto mask = valid_mask.to(heatmap.dtype()).unsqueeze(1);
    return torch::nn::functional::interpolate(
        mask,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{heatmap.size(2), heatmap.size(3)})
            .mode(torch::kNearest));
}

torch::Tensor warp_mask_to_view_b(const torch::Tensor& view_b_mask, const torch::Tensor& warp) {
    using torch::indexing::Slice;

    auto grid = warp.to(torch::kFloat32).contiguous();
    grid.index_put_({Slice(), Slice(), Slice(), 0},
                    grid.index({Slice(), Slice(), Slice(), 0}) /
                            static_cast<double>(std::max<int64_t>(1, view_b_mask.size(2) - 1)) * 2.0 -
                        1.0);
    grid.index_put_({Slice(), Slice(), Slice(), 1},
                    grid.index({Slice(), Slice(), Slice(), 1}) /
                            static_cast<double>(std::max<int64_t>(1, view_b_mask.size(1) - 1)) * 2.0 -
                        1.0);
    return torch::nn::functional::grid_sample(
               view_b_mask.unsqueeze(1).to(torch::kFloat32),
               grid,
               torch::nn::functional::GridSampleFuncOptions()
                   .mode(torch::kNearest)
                   .padding_mode(torch::kZeros)
                   .align_corners(true))
        .squeeze(1)
        .gt(0.0);
}

torch::Tensor make_training_valid_mask(
    const torch::Tensor& view_a,
    const torch::Tensor& view_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double min_keypoint_intensity
) {
    std::vector<torch::Tensor> masks_a;
    std::vector<torch::Tensor> masks_b;
    masks_a.reserve(static_cast<size_t>(view_a.size(0)));
    masks_b.reserve(static_cast<size_t>(view_b.size(0)));
    for (int64_t index = 0; index < view_a.size(0); ++index) {
        masks_a.push_back(make_intensity_mask(view_a[index], min_keypoint_intensity));
        masks_b.push_back(make_intensity_mask(view_b[index], min_keypoint_intensity));
    }
    const auto mask_a = torch::stack(masks_a).to(valid_mask.device()).to(torch::kBool);
    const auto mask_b = torch::stack(masks_b).to(valid_mask.device()).to(torch::kBool);
    return valid_mask.to(torch::kBool).logical_and(mask_a).logical_and(warp_mask_to_view_b(mask_b, warp));
}

torch::Tensor warp_heatmap_for_repeatability(const torch::Tensor& heatmap, const torch::Tensor& warp) {
    using torch::indexing::Slice;

    auto resized_warp = torch::nn::functional::interpolate(
        warp.permute({0, 3, 1, 2}),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{heatmap.size(2), heatmap.size(3)})
            .mode(torch::kBilinear)
            .align_corners(false));
    auto grid = resized_warp.permute({0, 2, 3, 1}).to(heatmap.dtype()).contiguous();
    grid.index_put_({Slice(), Slice(), Slice(), 0},
                    grid.index({Slice(), Slice(), Slice(), 0}) /
                            static_cast<double>(std::max<int64_t>(1, warp.size(2) - 1)) * 2.0 -
                        1.0);
    grid.index_put_({Slice(), Slice(), Slice(), 1},
                    grid.index({Slice(), Slice(), Slice(), 1}) /
                            static_cast<double>(std::max<int64_t>(1, warp.size(1) - 1)) * 2.0 -
                        1.0);
    return torch::nn::functional::grid_sample(
        heatmap,
        grid,
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(true));
}

torch::Tensor resize_offsets_for_dense_head(const torch::Tensor& warp, const torch::Tensor& offsets) {
    using torch::indexing::Slice;

    auto source_grid = make_xy_grid(warp.size(1), warp.size(2), warp.device()).unsqueeze(0).to(warp.dtype());
    auto displacement = (warp - source_grid).permute({0, 3, 1, 2}).to(offsets.dtype()).contiguous();
    auto resized = torch::nn::functional::interpolate(
        displacement,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{offsets.size(2), offsets.size(3)})
            .mode(torch::kBilinear)
            .align_corners(false));
    resized.index_put_({Slice(), Slice(0, 1), Slice(), Slice()},
                       resized.index({Slice(), Slice(0, 1), Slice(), Slice()}) / static_cast<double>(offsets.size(3)));
    resized.index_put_({Slice(), Slice(1, 2), Slice(), Slice()},
                       resized.index({Slice(), Slice(1, 2), Slice(), Slice()}) / static_cast<double>(offsets.size(2)));
    return resized;
}

struct TrainModules {
    Backbone backbone{nullptr};
    SparseHead sparse_head{nullptr};
    DenseHead dense_head{nullptr};
    PlanetaryGraphMatcher graph_matcher{nullptr};
};

TrainModules make_modules(const TrainConfig& config, torch::Device device) {
    TrainModules modules;
    modules.backbone = Backbone(INPUT_CHANNELS, config.base_channels);
    modules.sparse_head = SparseHead(config.base_channels * SPARSE_FEATURE_CHANNEL_MULTIPLIER, config.descriptor_dim);
    modules.dense_head = DenseHead(config.base_channels);
    modules.graph_matcher = PlanetaryGraphMatcher(
        config.descriptor_dim,
        config.graph_hidden_dim,
        config.graph_attention_layers);
    modules.backbone->to(device);
    modules.sparse_head->to(device);
    modules.dense_head->to(device);
    modules.graph_matcher->to(device);
    modules.backbone->train();
    modules.sparse_head->train();
    modules.dense_head->train();
    modules.graph_matcher->train();
    return modules;
}

FeatureDecodeConfig make_training_decode_config(const TrainConfig& config) {
    FeatureDecodeConfig decode_config;
    decode_config.max_keypoints = config.max_keypoints;
    decode_config.min_keypoints = config.min_keypoints;
    decode_config.keypoint_grid_rows = config.keypoint_grid_rows;
    decode_config.keypoint_grid_cols = config.keypoint_grid_cols;
    decode_config.keypoints_per_cell = config.keypoints_per_cell;
    decode_config.nms_radius = config.nms_radius;
    return decode_config;
}

torch::Tensor resize_dense_confidence_for_sparse_decode(
    const torch::Tensor& dense_confidence,
    const torch::Tensor& sparse_heatmap
) {
    return torch::nn::functional::interpolate(
        dense_confidence,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{sparse_heatmap.size(2), sparse_heatmap.size(3)})
            .mode(torch::kNearest));
}

FeatureSet decode_training_features(
    const torch::Tensor& view,
    const SparseHeadOutput& sparse,
    const torch::Tensor& dense_confidence,
    const TrainConfig& config
) {
    RawFeatureMaps maps{
        sparse.heatmap.detach().cpu().contiguous(),
        sparse.descriptors.detach().cpu().contiguous(),
        sparse.scale.detach().cpu().contiguous(),
        sparse.orientation.detach().cpu().contiguous(),
        sparse.affine.detach().cpu().contiguous(),
        dense_confidence.detach().cpu().contiguous()};
    auto features = decode_feature_maps(
        maps,
        make_training_decode_config(config),
        make_visualization_mask(view, config.min_keypoint_intensity));
    features.feature_map_width = maps.heatmap.size(3);
    features.feature_map_height = maps.heatmap.size(2);
    return features;
}

std::vector<torch::Tensor> module_parameters(TrainModules& modules) {
    std::vector<torch::Tensor> parameters;
    for (auto& parameter : modules.backbone->parameters()) {
        parameters.push_back(parameter);
    }
    for (auto& parameter : modules.sparse_head->parameters()) {
        parameters.push_back(parameter);
    }
    for (auto& parameter : modules.dense_head->parameters()) {
        parameters.push_back(parameter);
    }
    for (auto& parameter : modules.graph_matcher->parameters()) {
        parameters.push_back(parameter);
    }
    return parameters;
}

SyntheticPairConfig make_default_pair_config() {
    SyntheticPairConfig pair_config;
    pair_config.noise_sigma = 0.01F;
    return pair_config;
}

std::vector<SyntheticPair> make_synthetic_pairs_from_batch(
    const torch::Tensor& batch,
    const std::vector<int64_t>& source_indices,
    const std::vector<int64_t>& variant_indices,
    const SyntheticPairConfig& pair_config
) {
    std::vector<SyntheticPair> pairs;
    pairs.reserve(static_cast<size_t>(batch.size(0)));
    for (int64_t index = 0; index < batch.size(0); ++index) {
        auto variant_config = pair_config;
        variant_config.source_index = source_indices[static_cast<std::size_t>(index)];
        variant_config.variant_index = variant_indices[static_cast<std::size_t>(index)];
        pairs.push_back(make_synthetic_pair(batch[index], variant_config));
    }
    return pairs;
}

SyntheticPair move_pair_to_device(const SyntheticPair& pair, torch::Device device) {
    return SyntheticPair{
        pair.view_a.to(device),
        pair.view_b.to(device),
        pair.warp_a_to_b.to(device),
        pair.valid_mask.to(device)};
}

std::vector<SyntheticPair> load_cached_pairs(
    const SyntheticPairCacheDataset& cache_dataset,
    std::size_t offset,
    std::size_t end,
    torch::Device device
) {
    std::vector<SyntheticPair> pairs;
    pairs.reserve(end - offset);
    for (std::size_t index = offset; index < end; ++index) {
        pairs.push_back(move_pair_to_device(cache_dataset.load(index), device));
    }
    return pairs;
}

struct TrainingBatchForward {
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp;
    torch::Tensor valid_mask;
    SparseHeadOutput sparse_a;
    SparseHeadOutput sparse_b;
    torch::Tensor dense_confidence;
    torch::Tensor dense_offsets;
};

struct TrainingLossComponents {
    torch::Tensor total;
    torch::Tensor repeatability;
    torch::Tensor descriptor;
    torch::Tensor graph_matching;
    torch::Tensor offset;
    torch::Tensor confidence;
    torch::Tensor descriptor_accuracy;
    torch::Tensor descriptor_diversity;
    torch::Tensor offset_error;
    TrainingBatchForward forward;
};

struct TrainingDiagnosticSnapshot {
    TrainConfig config;
    int epoch = 0;
    std::size_t pair_index = 0;
    SyntheticPair pair;
    SparseHeadOutput sparse_a;
    SparseHeadOutput sparse_b;
    torch::Tensor dense_confidence;
};

torch::Tensor weighted_total_training_loss(
    const torch::Tensor& repeatability,
    const torch::Tensor& descriptor,
    const torch::Tensor& graph_matching,
    const torch::Tensor& offset,
    const torch::Tensor& confidence,
    const torch::Tensor& descriptor_diversity
) {
    return repeatability + descriptor + descriptor_diversity * DESCRIPTOR_DIVERSITY_WEIGHT +
           graph_matching + offset * OFFSET_LOSS_WEIGHT + confidence;
}

torch::Tensor offset_pixel_error(const torch::Tensor& offsets, const torch::Tensor& target_offsets, const torch::Tensor& mask) {
    using torch::indexing::Slice;

    auto pixel_delta = offsets - target_offsets;
    pixel_delta.index_put_({Slice(), Slice(0, 1), Slice(), Slice()},
                           pixel_delta.index({Slice(), Slice(0, 1), Slice(), Slice()}) * offsets.size(3));
    pixel_delta.index_put_({Slice(), Slice(1, 2), Slice(), Slice()},
                           pixel_delta.index({Slice(), Slice(1, 2), Slice(), Slice()}) * offsets.size(2));
    auto error = pixel_delta.pow(2).sum(1, true).sqrt();
    auto mask_float = mask.to(offsets.dtype());
    auto denom = mask_float.sum();
    if (denom.item<double>() <= 0.0) {
        return torch::zeros({}, offsets.options());
    }
    return (error * mask_float).sum() / denom;
}

TrainingLossComponents training_loss_from_pairs(
    TrainModules& modules,
    const std::vector<SyntheticPair>& pairs,
    double min_keypoint_intensity
) {
    std::vector<torch::Tensor> views_a;
    std::vector<torch::Tensor> views_b;
    std::vector<torch::Tensor> warps;
    std::vector<torch::Tensor> valid_masks;
    views_a.reserve(pairs.size());
    views_b.reserve(pairs.size());
    warps.reserve(pairs.size());
    valid_masks.reserve(pairs.size());

    for (const auto& pair : pairs) {
        views_a.push_back(pair.view_a);
        views_b.push_back(pair.view_b);
        warps.push_back(pair.warp_a_to_b);
        valid_masks.push_back(pair.valid_mask);
    }

    const auto view_a = stack_batch(views_a, BatchTensorLayout::Chw);
    const auto view_b = stack_batch(views_b, BatchTensorLayout::Chw);
    const auto warp = stack_batch(warps, BatchTensorLayout::Hwc);
    const auto valid_mask = make_training_valid_mask(
        view_a,
        view_b,
        warp,
        stack_batch(valid_masks, BatchTensorLayout::Hw),
        min_keypoint_intensity);

    const auto feature_pyramid_a = modules.backbone->forward(view_a);
    const auto feature_pyramid_b = modules.backbone->forward(view_b);
    const auto dense_features_a = feature_pyramid_a.front();
    const auto dense_features_b = feature_pyramid_b.front();
    const auto sparse_a = modules.sparse_head->forward(feature_pyramid_a[1]);
    const auto sparse_b = modules.sparse_head->forward(feature_pyramid_b[1]);
    const auto dense = modules.dense_head->forward(dense_features_a, dense_features_b);
    const auto sparse_mask = resize_mask_for_heatmap(valid_mask, sparse_a.heatmap);
    const auto dense_mask = resize_mask_for_heatmap(valid_mask, dense.confidence);
    const auto target_offsets = resize_offsets_for_dense_head(warp, dense.offsets);

    auto repeatability = repeatability_loss(sparse_a.heatmap, warp_heatmap_for_repeatability(sparse_b.heatmap, warp), sparse_mask);
    auto descriptor = make_sparse_descriptor_metrics(sparse_a.descriptors, sparse_b.descriptors, warp, valid_mask);
    std::vector<torch::Tensor> graph_losses;
    graph_losses.reserve(static_cast<size_t>(view_a.size(0)));
    auto decode_config = TrainConfig{};
    decode_config.max_keypoints = 1024;
    decode_config.min_keypoints = 0;
    decode_config.nms_radius = 4;
    decode_config.min_keypoint_intensity = min_keypoint_intensity;
    for (int64_t batch = 0; batch < view_a.size(0); ++batch) {
        SparseHeadOutput sparse_a_item{
            sparse_a.heatmap.index({batch}).unsqueeze(0),
            sparse_a.descriptors.index({batch}).unsqueeze(0),
            sparse_a.scale.index({batch}).unsqueeze(0),
            sparse_a.orientation.index({batch}).unsqueeze(0),
            sparse_a.affine.index({batch}).unsqueeze(0)};
        SparseHeadOutput sparse_b_item{
            sparse_b.heatmap.index({batch}).unsqueeze(0),
            sparse_b.descriptors.index({batch}).unsqueeze(0),
            sparse_b.scale.index({batch}).unsqueeze(0),
            sparse_b.orientation.index({batch}).unsqueeze(0),
            sparse_b.affine.index({batch}).unsqueeze(0)};
        auto features_a = decode_training_features(
            view_a.index({batch}),
            sparse_a_item,
            resize_dense_confidence_for_sparse_decode(dense.confidence.index({batch}).unsqueeze(0), sparse_a_item.heatmap),
            decode_config);
        auto features_b = decode_training_features(
            view_b.index({batch}),
            sparse_b_item,
            resize_dense_confidence_for_sparse_decode(dense.confidence.index({batch}).unsqueeze(0), sparse_b_item.heatmap),
            decode_config);
        graph_losses.push_back(make_keypoint_graph_matching_loss(
            *modules.graph_matcher,
            features_a,
            features_b,
            warp.index({batch}).unsqueeze(0),
            valid_mask.index({batch}).unsqueeze(0)));
    }
    auto graph_matching = graph_losses.empty() ? torch::zeros({}, sparse_a.descriptors.options()) : torch::stack(graph_losses).mean();
    auto offset = masked_smooth_l1_loss(dense.offsets, target_offsets, dense_mask);
    auto confidence = confidence_bce_loss(dense.confidence, dense_mask);
    auto offset_error = offset_pixel_error(dense.offsets, target_offsets, dense_mask);
    return TrainingLossComponents{weighted_total_training_loss(
                                      repeatability,
                                      descriptor.loss,
                                      graph_matching,
                                      offset,
                                      confidence,
                                      descriptor.diversity),
                                  repeatability,
                                  descriptor.loss,
                                  graph_matching,
                                  offset,
                                  confidence,
                                  descriptor.accuracy,
                                  descriptor.diversity,
                                  offset_error,
                                  TrainingBatchForward{
                                      view_a,
                                      view_b,
                                      warp,
                                      valid_mask,
                                      sparse_a,
                                      sparse_b,
                                      dense.confidence,
                                      dense.offsets}};
}

bool should_enqueue_training_visualization(std::size_t enqueued_count, std::size_t visualization_limit) {
    return enqueued_count < visualization_limit;
}

bool should_use_online_dataloader(const TrainConfig& config) {
    return config.dataloader_workers > 0 && config.synthetic_pair_cache_dir.empty();
}

DatasetSplit make_training_dataset_split(std::size_t total_images, const TrainConfig& config) {
    const auto validation_ratio = config.val_ratio;
    const auto train_ratio = config.train_ratio;
    const auto test_ratio = 1.0 - train_ratio - validation_ratio;
    auto split = make_train_validation_test_split(
        total_images,
        train_ratio,
        validation_ratio,
        test_ratio,
        static_cast<uint64_t>(config.split_seed),
        true);
    if (split.train.empty() && total_images > 0) {
        if (!split.validation.empty()) {
            split.train.push_back(split.validation.front());
            split.validation.erase(split.validation.begin());
        } else if (!split.test.empty()) {
            split.train.push_back(split.test.front());
            split.test.erase(split.test.begin());
        }
    }
    return split;
}

std::filesystem::path epoch_visualization_dir(const TrainConfig& config, int epoch) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "epoch_%06d", epoch);
    return std::filesystem::path(config.visualization_dir) / buffer;
}

std::filesystem::path static_visualization_dir(const TrainConfig& config) {
    return std::filesystem::path(config.visualization_dir) / "static";
}

std::shared_ptr<std::pair<FeatureSet, FeatureSet>> decode_training_diagnostic_features(
    const TrainingDiagnosticSnapshot& snapshot
) {
    return std::make_shared<std::pair<FeatureSet, FeatureSet>>(
        decode_training_features(snapshot.pair.view_a, snapshot.sparse_a, snapshot.dense_confidence, snapshot.config),
        decode_training_features(snapshot.pair.view_b, snapshot.sparse_b, snapshot.dense_confidence, snapshot.config));
}

void render_static_training_diagnostics(const TrainingDiagnosticSnapshot& snapshot) {
    const auto stem = pair_visualization_stem(snapshot.pair_index);
    const auto static_dir = static_visualization_dir(snapshot.config);
    const auto valid_count = snapshot.pair.valid_mask.to(torch::kFloat32).sum().item<int64_t>();
    const auto total_count = snapshot.pair.valid_mask.numel();

    writeVisualizationImage(static_dir / (stem + "_view_a.png"), snapshot.pair.view_a, "view_a");
    writeVisualizationImage(static_dir / (stem + "_view_b.png"), snapshot.pair.view_b, "view_b");
    writeVisualizationImage(static_dir / (stem + "_valid_mask.png"), mask_to_image(snapshot.pair.valid_mask),
                            "valid=" + std::to_string(valid_count) + "/" + std::to_string(total_count));
    writeVisualizationImage(static_dir / (stem + "_warp_matches.png"), warp_overlay_image(snapshot.pair),
                            "matches=" + std::to_string(valid_count) + " valid=" + std::to_string(valid_count) + "/" +
                                std::to_string(total_count));
}

void render_feature_training_diagnostics(
    const TrainingDiagnosticSnapshot& snapshot,
    const std::shared_ptr<std::pair<FeatureSet, FeatureSet>>& features
) {
    const auto stem = pair_visualization_stem(snapshot.pair_index);
    const auto output_dir = epoch_visualization_dir(snapshot.config, snapshot.epoch);

    writeVisualizationImage(output_dir / (stem + "_features_a.png"),
                            feature_overlay_image(snapshot.pair.view_a, features->first, snapshot.config.min_keypoint_intensity),
                            "features=" + std::to_string(features->first.keypoints.size(0)));
    writeVisualizationImage(output_dir / (stem + "_features_b.png"),
                            feature_overlay_image(snapshot.pair.view_b, features->second, snapshot.config.min_keypoint_intensity),
                            "features=" + std::to_string(features->second.keypoints.size(0)));
}

void render_model_match_training_diagnostics(
    const TrainingDiagnosticSnapshot& snapshot,
    const std::shared_ptr<std::pair<FeatureSet, FeatureSet>>& features
) {
    const auto stem = pair_visualization_stem(snapshot.pair_index);
    const auto output_dir = epoch_visualization_dir(snapshot.config, snapshot.epoch);
    const auto matches = matchFeatureSets(features->first, features->second);
    MatchCorrectnessStats stats;
    auto image = match_overlay_image(
        snapshot.pair.view_a,
        snapshot.pair.view_b,
        features->first,
        features->second,
        matches,
        snapshot.pair.warp_a_to_b,
        TRAINING_MATCH_CORRECT_THRESHOLD_PIXELS,
        snapshot.config.min_keypoint_intensity,
        &stats);

    writeVisualizationImage(output_dir / (stem + "_model_matches.png"),
                            image,
                            model_match_overlay_text(features->first, features->second, matches, stats));
}

TrainingDiagnosticSnapshot make_training_diagnostic_snapshot(
    const TrainConfig& config,
    int epoch,
    std::size_t pair_index,
    std::size_t batch_index,
    const SyntheticPair& pair,
    const TrainingBatchForward& forward
) {
    const auto dense_confidence = torch::nn::functional::interpolate(
        forward.dense_confidence.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{forward.sparse_a.heatmap.size(2), forward.sparse_a.heatmap.size(3)})
            .mode(torch::kNearest));
    return TrainingDiagnosticSnapshot{
        config,
        epoch,
        pair_index,
        SyntheticPair{
            pair.view_a.detach().to(torch::kCPU).contiguous(),
            pair.view_b.detach().to(torch::kCPU).contiguous(),
            pair.warp_a_to_b.detach().to(torch::kCPU).contiguous(),
            pair.valid_mask.detach().to(torch::kCPU).contiguous()},
        SparseHeadOutput{
            forward.sparse_a.heatmap.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous(),
            forward.sparse_a.descriptors.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous(),
            forward.sparse_a.scale.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous(),
            forward.sparse_a.orientation.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous(),
            forward.sparse_a.affine.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous()},
        SparseHeadOutput{
            forward.sparse_b.heatmap.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous(),
            forward.sparse_b.descriptors.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous(),
            forward.sparse_b.scale.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous(),
            forward.sparse_b.orientation.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous(),
            forward.sparse_b.affine.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0).contiguous()},
        dense_confidence.contiguous()};
}

void enqueue_training_diagnostics(
    AsyncVisualizationWriter& writer,
    const TrainConfig& config,
    int epoch,
    std::size_t pair_index,
    std::size_t batch_index,
    const SyntheticPair& pair,
    const TrainingBatchForward& forward
) {
    auto snapshot = make_training_diagnostic_snapshot(config, epoch, pair_index, batch_index, pair, forward);
    if (epoch == 1) {
        writer.enqueueJob([snapshot]() { render_static_training_diagnostics(snapshot); });
    }
    writer.enqueueJob([snapshot]() {
        const auto features = decode_training_diagnostic_features(snapshot);
        render_feature_training_diagnostics(snapshot, features);
    });
    writer.enqueueJob([snapshot = std::move(snapshot)]() {
        const auto features = decode_training_diagnostic_features(snapshot);
        render_model_match_training_diagnostics(snapshot, features);
    });
}

AugmentationProfile to_augmentation_profile(SyntheticPairAugmentationProfile profile) {
    switch (profile) {
        case SyntheticPairAugmentationProfile::Mixed:
            return AugmentationProfile::Mixed;
        case SyntheticPairAugmentationProfile::Mild:
            return AugmentationProfile::Mild;
        case SyntheticPairAugmentationProfile::Medium:
            return AugmentationProfile::Medium;
        case SyntheticPairAugmentationProfile::Hard:
            return AugmentationProfile::Hard;
        case SyntheticPairAugmentationProfile::Extreme:
            return AugmentationProfile::Extreme;
    }
    return AugmentationProfile::Mixed;
}

ImagePairAugmentationConfig make_online_pair_config(const TrainConfig& config) {
    ImagePairAugmentationConfig pair_config;
    pair_config.profile = to_augmentation_profile(parse_synthetic_pair_augmentation_profile(config.augmentation_profile));
    pair_config.extreme_pair_ratio = config.extreme_pair_ratio;
    return pair_config;
}

DataLoaderOptions make_dataloader_options(const TrainConfig& config) {
    DataLoaderOptions options;
    options.batch_size = static_cast<size_t>(config.batch_size);
    options.worker_count = static_cast<size_t>(config.dataloader_workers);
    options.prefetch_batches = static_cast<size_t>(config.prefetch_batches);
    options.pin_memory = config.pin_memory;
    return options;
}

TensorBatchCollator make_synthetic_pair_collator() {
    return TensorBatchCollator({
        {"view_a", TensorLayout::Chw},
        {"view_b", TensorLayout::Chw},
        {"warp_a_to_b", TensorLayout::Hwc},
        {"valid_mask", TensorLayout::Hw},
    });
}

std::vector<SyntheticPair> pairs_from_tensor_batch(TensorBatch batch, torch::Device device) {
    auto view_a = batch.at("view_a").to(device);
    auto view_b = batch.at("view_b").to(device);
    auto warp_a_to_b = batch.at("warp_a_to_b").to(device);
    auto valid_mask = batch.at("valid_mask").to(device);
    std::vector<SyntheticPair> pairs;
    pairs.reserve(static_cast<std::size_t>(view_a.size(0)));
    for (int64_t index = 0; index < view_a.size(0); ++index) {
        pairs.push_back(SyntheticPair{
            view_a.index({index}).contiguous(),
            view_b.index({index}).contiguous(),
            warp_a_to_b.index({index}).contiguous(),
            valid_mask.index({index}).contiguous(),
        });
    }
    return pairs;
}

SyntheticPairCacheConfig make_cache_config(const TrainConfig& config, std::size_t epoch_size) {
    SyntheticPairCacheConfig cache_config;
    cache_config.cache_dir = config.synthetic_pair_cache_dir;
    cache_config.resize = config.resize;
    cache_config.pair_count = epoch_size;
    cache_config.pairs_per_image = static_cast<std::size_t>(config.pairs_per_image);
    cache_config.pair_config = make_default_pair_config();
    cache_config.pair_config.augmentation_profile = parse_synthetic_pair_augmentation_profile(config.augmentation_profile);
    cache_config.pair_config.extreme_pair_ratio = config.extreme_pair_ratio;
    cache_config.rebuild = config.synthetic_pair_cache_rebuild;
    return cache_config;
}

void move_modules_to_device(TrainModules& modules, torch::Device device) {
    modules.backbone->to(device);
    modules.sparse_head->to(device);
    modules.dense_head->to(device);
    modules.graph_matcher->to(device);
}

void save_checkpoint(const TrainConfig& config, TrainModules& modules) {
    move_modules_to_device(modules, torch::Device(torch::kCPU));
    torch::serialize::OutputArchive archive;
    torch::serialize::OutputArchive config_archive;
    config_archive.write("checkpoint_version", torch::tensor({2}, torch::kInt64));
    config_archive.write("base_channels", torch::tensor({config.base_channels}, torch::kInt64));
    config_archive.write("descriptor_dim", torch::tensor({config.descriptor_dim}, torch::kInt64));
    config_archive.write("graph_hidden_dim", torch::tensor({config.graph_hidden_dim}, torch::kInt64));
    config_archive.write("graph_attention_layers", torch::tensor({config.graph_attention_layers}, torch::kInt64));
    config_archive.write("input_channels", torch::tensor({INPUT_CHANNELS}, torch::kInt64));
    archive.write("config", config_archive);

    torch::serialize::OutputArchive backbone_archive;
    torch::serialize::OutputArchive sparse_head_archive;
    torch::serialize::OutputArchive dense_head_archive;
    torch::serialize::OutputArchive graph_matcher_archive;
    modules.backbone->save(backbone_archive);
    modules.sparse_head->save(sparse_head_archive);
    modules.dense_head->save(dense_head_archive);
    modules.graph_matcher->save(graph_matcher_archive);
    archive.write("backbone", backbone_archive);
    archive.write("sparse_head", sparse_head_archive);
    archive.write("dense_head", dense_head_archive);
    archive.write("graph_matcher", graph_matcher_archive);
    archive.save_to(config.checkpoint);
}

}  // namespace

namespace testing {

torch::Tensor resize_offsets_for_dense_head_for_test(const torch::Tensor& warp, const torch::Tensor& offsets) {
    return resize_offsets_for_dense_head(warp, offsets);
}

torch::Tensor make_sparse_descriptor_loss_for_test(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask) {
    return make_sparse_descriptor_loss(descriptors_a, descriptors_b, warp, valid_mask);
}

torch::Tensor make_graph_matching_loss_for_test(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    return make_graph_matching_loss(graph_matcher, descriptors_a, descriptors_b, warp, valid_mask);
}


torch::Tensor assign_graph_matching_targets_for_test(
    const torch::Tensor& keypoints_a,
    const torch::Tensor& keypoints_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double positive_radius_pixels
) {
    return assign_graph_matching_targets(keypoints_a, keypoints_b, warp, valid_mask, positive_radius_pixels);
}

torch::Tensor make_graph_candidate_indices_for_test(
    const torch::Tensor& target_indices,
    int64_t keypoint_count_b,
    int64_t max_candidates
) {
    return make_graph_candidate_indices(target_indices, keypoint_count_b, max_candidates);
}

torch::Tensor make_keypoint_graph_matching_loss_for_test(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    return make_keypoint_graph_matching_loss(graph_matcher, features_a, features_b, warp, valid_mask);
}

torch::Tensor make_descriptor_sample_indices_for_test(const torch::Tensor& descriptors) {
    return make_descriptor_sample_indices(descriptors);
}

torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count) {
    return make_descriptor_candidate_indices(target_indices, spatial_count);
}

torch::Tensor limit_training_image_size_for_test(const torch::Tensor& image, int64_t max_edge) {
    return limit_training_image_size(image, max_edge);
}

torch::Tensor stack_chw_batch_for_test(const std::vector<torch::Tensor>& tensors) {
    return stack_batch(tensors, BatchTensorLayout::Chw);
}

torch::Tensor stack_hw_batch_for_test(const std::vector<torch::Tensor>& tensors) {
    return stack_batch(tensors, BatchTensorLayout::Hw);
}

torch::Tensor stack_hwc_batch_for_test(const std::vector<torch::Tensor>& tensors) {
    return stack_batch(tensors, BatchTensorLayout::Hwc);
}

torch::Tensor weighted_total_training_loss_for_test(
    const torch::Tensor& repeatability,
    const torch::Tensor& descriptor,
    const torch::Tensor& offset,
    const torch::Tensor& confidence,
    const torch::Tensor& descriptor_diversity
) {
    return weighted_total_training_loss(
        repeatability,
        descriptor,
        torch::zeros({}, descriptor.options()),
        offset,
        confidence,
        descriptor_diversity);
}

torch::Tensor warp_heatmap_for_repeatability_for_test(const torch::Tensor& heatmap, const torch::Tensor& warp) {
    return warp_heatmap_for_repeatability(heatmap, warp);
}

torch::Tensor make_training_valid_mask_for_test(
    const torch::Tensor& view_a,
    const torch::Tensor& view_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double min_keypoint_intensity
) {
    return make_training_valid_mask(view_a, view_b, warp, valid_mask, min_keypoint_intensity);
}

torch::Tensor training_warp_overlay_image_for_test(const SyntheticPair& pair) {
    return warp_overlay_image(pair);
}

torch::Tensor training_feature_overlay_image_for_test(
    const torch::Tensor& image,
    const FeatureSet& features,
    double min_keypoint_intensity
) {
    return feature_overlay_image(image, features, min_keypoint_intensity);
}

torch::Tensor training_match_overlay_image_for_test(
    const torch::Tensor& image_a,
    const torch::Tensor& image_b,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches
) {
    return match_overlay_image(image_a, image_b, features_a, features_b, matches, 0.0);
}

torch::Tensor training_match_overlay_image_for_test(
    const torch::Tensor& image_a,
    const torch::Tensor& image_b,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b,
    double correct_threshold_pixels
) {
    return match_overlay_image(
        image_a, image_b, features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels, 0.0);
}

std::string training_model_match_overlay_text_for_test(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b,
    double correct_threshold_pixels
) {
    return model_match_overlay_text(features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels);
}

bool should_enqueue_training_visualization_for_test(std::size_t enqueued_count, std::size_t visualization_limit) {
    return should_enqueue_training_visualization(enqueued_count, visualization_limit);
}

bool should_use_online_dataloader_for_test(const TrainConfig& config) {
    return should_use_online_dataloader(config);
}

std::vector<std::size_t> make_training_image_indices_for_test(
    std::size_t total_images,
    const TrainConfig& config
) {
    return make_training_dataset_split(total_images, config).train;
}

std::vector<std::size_t> make_validation_image_indices_for_test(
    std::size_t total_images,
    const TrainConfig& config
) {
    return make_training_dataset_split(total_images, config).validation;
}

}  // namespace testing

TrainResult train_model(const TrainConfig& config) {
    validate_config(config);
    Timer total_timer;
    int64_t completed_batches = 0;
    double accumulated_batch_seconds = 0.0;
    const auto device = resolve_compute_device(config.device);
    ImageDataset dataset(config.image_dir);
    auto modules = make_modules(config, device);
    auto parameters = module_parameters(modules);
    auto optimizer_options = torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay);
    auto optimizer = torch::optim::AdamW(parameters, optimizer_options);
    std::unique_ptr<CsvMetricLogger> csv_logger;
    std::unique_ptr<GpuMetricProvider> gpu_metric_provider;
    if (!config.log_csv.empty()) {
        csv_logger = std::make_unique<CsvMetricLogger>(config.log_csv, TRAINING_CSV_COLUMNS);
        gpu_metric_provider = makeDefaultGpuMetricProvider();
    }

    ConsoleProgressLogger progress_logger(std::cout, 30);

    TrainResult result;
    double first_loss = 0.0;
    double last_loss = 0.0;
    bool has_loss = false;

    const auto total_images = dataset.size();
    const auto dataset_split = make_training_dataset_split(total_images, config);
    const auto& train_indices = dataset_split.train;
    const auto& validation_indices = dataset_split.validation;
    const auto train_images = train_indices.size();
    const auto val_images = validation_indices.size();
    const auto train_epoch_size = train_images * static_cast<std::size_t>(config.pairs_per_image);
    const auto epoch_size = train_epoch_size;
    const auto lr_max = config.learning_rate;
    const auto lr_min = lr_max * 0.01;
    const auto total_iterations = static_cast<double>(config.epochs) *
        static_cast<double>(train_epoch_size) / static_cast<double>(config.batch_size);
    int64_t global_step = 0;
    std::unique_ptr<AsyncVisualizationWriter> visualization_writer;
    const std::size_t visualization_limit = config.visualization_samples_all
        ? epoch_size
        : std::min<std::size_t>(static_cast<std::size_t>(config.visualization_samples), epoch_size);
    if (!config.visualization_dir.empty() && visualization_limit > 0) {
        std::cout << "training visualization: dir=" << config.visualization_dir
                  << " samples=" << (config.visualization_samples_all ? "all" : std::to_string(config.visualization_samples))
                  << " max_keypoints=" << config.max_keypoints
                  << " min_keypoints=" << config.min_keypoints
                  << " keypoint_grid=" << config.keypoint_grid_rows << 'x' << config.keypoint_grid_cols
                  << " keypoints_per_cell=" << config.keypoints_per_cell
                  << " nms_radius=" << config.nms_radius
                  << " async_queue=" << TRAINING_VISUALIZATION_QUEUE_CAPACITY
                  << " async_workers=" << TRAINING_VISUALIZATION_WORKER_COUNT << '\n';
        visualization_writer = std::make_unique<AsyncVisualizationWriter>(
            TRAINING_VISUALIZATION_QUEUE_CAPACITY, TRAINING_VISUALIZATION_WORKER_COUNT);
    }
    auto pair_config = make_default_pair_config();
    pair_config.augmentation_profile = parse_synthetic_pair_augmentation_profile(config.augmentation_profile);
    pair_config.extreme_pair_ratio = config.extreme_pair_ratio;
    if (!config.synthetic_pair_cache_dir.empty()) {
        prepare_synthetic_pair_cache(dataset, make_cache_config(config, epoch_size));
    }
    std::unique_ptr<SyntheticPairCacheDataset> cache_dataset;
    if (!config.synthetic_pair_cache_dir.empty()) {
        cache_dataset = std::make_unique<SyntheticPairCacheDataset>(config.synthetic_pair_cache_dir);
    }
    std::unique_ptr<AsyncDataLoader> online_loader;
    if (should_use_online_dataloader(config)) {
        std::vector<torch::Tensor> images;
        images.reserve(train_images);
        for (std::size_t index = 0; index < train_images; ++index) {
            images.push_back(limit_training_image_size(ensure_grayscale(dataset.load(train_indices[index])), config.resize));
        }
        auto online_dataset = std::make_shared<SyntheticPairTensorDataset>(
            std::move(images), static_cast<size_t>(config.pairs_per_image), make_online_pair_config(config));
        online_loader = std::make_unique<AsyncDataLoader>(
            online_dataset,
            std::make_unique<SequentialSampler>(online_dataset->size()),
            make_synthetic_pair_collator(),
            make_dataloader_options(config));
    }

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        Timer epoch_timer;
        if (online_loader) {
            online_loader->reset();
        }
        for (std::size_t offset = 0; offset < epoch_size; offset += static_cast<std::size_t>(config.batch_size)) {
            Timer batch_timer;
            const auto batch_end = offset + static_cast<std::size_t>(config.batch_size);
            const auto end = std::min<std::size_t>(epoch_size, batch_end);
            std::vector<SyntheticPair> pairs;
            if (online_loader) {
                auto batch = online_loader->next();
                if (!batch.has_value()) {
                    throw std::runtime_error("online dataloader exhausted before epoch end");
                }
                pairs = pairs_from_tensor_batch(std::move(batch.value()), device);
            } else if (cache_dataset) {
                pairs = load_cached_pairs(*cache_dataset, offset, end, device);
            } else {
                std::vector<torch::Tensor> images;
                std::vector<int64_t> source_indices;
                std::vector<int64_t> variant_indices;
                images.reserve(end - offset);
                source_indices.reserve(end - offset);
                variant_indices.reserve(end - offset);
                for (std::size_t index = offset; index < end; ++index) {
                    const auto source_index = train_indices[index % train_images];
                    images.push_back(limit_training_image_size(
                        ensure_grayscale(dataset.load(source_index)),
                        config.resize));
                    source_indices.push_back(static_cast<int64_t>(source_index));
                    variant_indices.push_back(static_cast<int64_t>(index / train_images));
                }
                pairs = make_synthetic_pairs_from_batch(
                    stack_batch(images, BatchTensorLayout::Chw).to(device), source_indices, variant_indices, pair_config);
            }

            auto loss = training_loss_from_pairs(modules, pairs, config.min_keypoint_intensity);
            if (visualization_writer && offset < visualization_limit) {
                for (std::size_t pair_offset = 0; pair_offset < pairs.size(); ++pair_offset) {
                    const auto pair_index = offset + pair_offset;
                    if (!should_enqueue_training_visualization(pair_index, visualization_limit)) {
                        break;
                    }
                    enqueue_training_diagnostics(
                        *visualization_writer, config, epoch + 1, pair_index, pair_offset, pairs[pair_offset], loss.forward);
                }
            }

            optimizer.zero_grad();
            loss.total.backward();
            if (config.gradient_clip_norm > 0.0) {
                torch::nn::utils::clip_grad_norm_(parameters, config.gradient_clip_norm);
            }
            optimizer.step();

            ++global_step;
            const auto progress = static_cast<double>(global_step) / total_iterations;
            const auto cos_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + std::cos(3.14159265358979323846 * progress));
            for (auto& group : optimizer.param_groups()) {
                group.options().set_lr(cos_lr);
            }

            last_loss = loss.total.detach().item<double>();
            const auto repeatability_loss_value = loss.repeatability.detach().item<double>();
            const auto descriptor_loss_value = loss.descriptor.detach().item<double>();
            const auto graph_matching_loss_value = loss.graph_matching.detach().item<double>();
            const auto descriptor_accuracy_value = loss.descriptor_accuracy.detach().item<double>();
            const auto descriptor_diversity_value = loss.descriptor_diversity.detach().item<double>();
            const auto offset_loss_value = loss.offset.detach().item<double>();
            const auto offset_error_value = loss.offset_error.detach().item<double>();
            const auto confidence_loss_value = loss.confidence.detach().item<double>();
            const auto feature_loss_value = repeatability_loss_value + descriptor_loss_value;
            const auto dense_loss_value = offset_loss_value * OFFSET_LOSS_WEIGHT + confidence_loss_value;
            const double batch_seconds = batch_timer.elapsedSeconds();
            accumulated_batch_seconds += batch_seconds;
            ++completed_batches;
            const auto iteration = static_cast<int>((offset / static_cast<std::size_t>(config.batch_size)) + 1);
            const auto total_iterations = static_cast<int>(
                (epoch_size + static_cast<std::size_t>(config.batch_size) - 1) /
                static_cast<std::size_t>(config.batch_size));
            const auto metric_values = std::unordered_map<std::string, double>{
                {"loss_total", last_loss},
                {"feature_loss", feature_loss_value},
                {"repeatability_loss", repeatability_loss_value},
                {"descriptor_loss", descriptor_loss_value},
                {"matcher_loss", graph_matching_loss_value},
                {"graph_matching_loss", graph_matching_loss_value},
                {"dense_loss", dense_loss_value},
                {"offset_loss", offset_loss_value},
                {"confidence_loss", confidence_loss_value},
                {"descriptor_accuracy", descriptor_accuracy_value},
                {"descriptor_diversity", descriptor_diversity_value},
                {"offset_error_px", offset_error_value},
            };
            if (csv_logger) {
                const auto gpu_metrics = gpu_metric_provider ? gpu_metric_provider->sample() : GpuMetrics{};
                csv_logger->logIteration(make_iteration_metric(
                    config,
                    epoch + 1,
                    iteration,
                    total_iterations,
                    static_cast<int>(end),
                    static_cast<int>(epoch_size),
                    total_timer.elapsedSeconds(),
                    gpu_metrics,
                    metric_values));
            }
            TrainingMetric iter_metric;
            iter_metric.epoch = static_cast<int>(epoch + 1);
            iter_metric.total_epochs = static_cast<int>(config.epochs);
            iter_metric.iteration = static_cast<int>((offset / static_cast<std::size_t>(config.batch_size)) + 1);
            iter_metric.total_iterations = static_cast<int>(
                (epoch_size + static_cast<std::size_t>(config.batch_size) - 1) /
                static_cast<std::size_t>(config.batch_size));
            iter_metric.images_seen = static_cast<int>(end);
            iter_metric.total_images = static_cast<int>(epoch_size);
            iter_metric.learning_rate = cos_lr;
            iter_metric.elapsed_seconds = total_timer.elapsedSeconds();
            iter_metric.values["loss_total"] = last_loss;
            iter_metric.values["matcher_loss"] = graph_matching_loss_value;
            iter_metric.values["dense_loss"] = dense_loss_value;
            iter_metric.values["offset_error_px"] = offset_error_value;
            iter_metric.values["descriptor_accuracy"] = descriptor_accuracy_value;
            iter_metric.values["descriptor_diversity"] = descriptor_diversity_value;
            iter_metric.values["feature_loss"] = feature_loss_value;
            iter_metric.values["repeatability_loss"] = repeatability_loss_value;
            iter_metric.values["descriptor_loss"] = descriptor_loss_value;
            progress_logger.logIteration(iter_metric);
            if (!has_loss) {
                first_loss = last_loss;
                has_loss = true;
            }
        }
        if (csv_logger) {
            TrainingMetric epoch_metric;
            epoch_metric.epoch = epoch + 1;
            epoch_metric.total_epochs = config.epochs;
            epoch_metric.elapsed_seconds = epoch_timer.elapsedSeconds();
            csv_logger->logEpochSummary(epoch_metric);
        }
        {
            TrainingMetric epoch_metric;
            epoch_metric.epoch = static_cast<int>(epoch + 1);
            epoch_metric.total_epochs = static_cast<int>(config.epochs);
            epoch_metric.elapsed_seconds = epoch_timer.elapsedSeconds();
            progress_logger.logEpochSummary(epoch_metric);
        }
        if (val_images > 0) {
            torch::NoGradGuard no_grad;
            modules.dense_head->eval();
            modules.sparse_head->eval();
            modules.backbone->eval();
            modules.graph_matcher->eval();
            double val_total = 0.0;
            int64_t val_batches = 0;
            const auto val_epoch_size = val_images * static_cast<std::size_t>(config.pairs_per_image);
            for (std::size_t val_offset = 0; val_offset < val_epoch_size;
                 val_offset += static_cast<std::size_t>(config.batch_size)) {
                const auto val_end = std::min<std::size_t>(val_epoch_size,
                    val_offset + static_cast<std::size_t>(config.batch_size));
                std::vector<torch::Tensor> val_img_batch;
                std::vector<int64_t> val_src_indices;
                std::vector<int64_t> val_var_indices;
                val_img_batch.reserve(val_end - val_offset);
                val_src_indices.reserve(val_end - val_offset);
                val_var_indices.reserve(val_end - val_offset);
                for (std::size_t idx = val_offset; idx < val_end; ++idx) {
                    const auto src = validation_indices[idx % val_images];
                    val_img_batch.push_back(limit_training_image_size(
                        ensure_grayscale(dataset.load(src)), config.resize));
                    val_src_indices.push_back(static_cast<int64_t>(src));
                    val_var_indices.push_back(static_cast<int64_t>(idx / val_images));
                }
                auto val_pairs = make_synthetic_pairs_from_batch(
                    stack_batch(val_img_batch, BatchTensorLayout::Chw).to(device),
                    val_src_indices, val_var_indices, pair_config);
                auto val_loss = training_loss_from_pairs(modules, val_pairs, config.min_keypoint_intensity);
                val_total += val_loss.total.detach().item<double>();
                ++val_batches;
            }
            const auto val_avg = val_batches > 0 ? val_total / static_cast<double>(val_batches) : 0.0;
            if (val_avg < result.best_val_loss) {
                result.best_val_loss = val_avg;
            }
            std::cout << "val loss=" << val_avg << " best=" << result.best_val_loss << '\n';
            modules.dense_head->train();
            modules.sparse_head->train();
            modules.backbone->train();
            modules.graph_matcher->train();
        }
        ++result.epochs_completed;
    }

    result.initial_loss = first_loss;
    result.final_loss = last_loss;
    result.total_time_seconds = total_timer.elapsedSeconds();
    result.avg_batch_time_seconds = completed_batches == 0
        ? 0.0
        : accumulated_batch_seconds / static_cast<double>(completed_batches);
    if (visualization_writer) {
        visualization_writer->join();
    }
    if (csv_logger) {
        csv_logger->flush();
    }
    save_checkpoint(config, modules);
    return result;
}

bool checkpoint_can_load(const std::string& checkpoint) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(checkpoint);
        torch::serialize::InputArchive config_archive;
        archive.read("config", config_archive);
        torch::Tensor base_channels;
        torch::Tensor descriptor_dim;
        torch::Tensor input_channels;
        config_archive.read("base_channels", base_channels);
        config_archive.read("descriptor_dim", descriptor_dim);
        config_archive.read("input_channels", input_channels);
        torch::serialize::InputArchive graph_matcher_archive;
        archive.read("graph_matcher", graph_matcher_archive);
        return base_channels.defined() && descriptor_dim.defined() && input_channels.defined();
    } catch (const c10::Error&) {
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace pfm
