#include "train/trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <torch/nn/functional/upsampling.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>

#include "core/device.h"
#include "core/tensor_utils.h"
#include "core/timer.h"
#include "data/image_dataset.h"
#include "data/pair_archive_dataset.h"
#include "data/synthetic_pair.h"
#include "data/synthetic_pair_cache.h"
#include "data/synthetic_pair_dataset.h"
#include "dataloader/async_dataloader.h"
#include "dataloader/sampler.h"
#include "feature_io/feature_codec.h"
#include "image/intensity_mask.h"
#include "infer/feature_extractor.h"
#include "infer/matching_pipeline.h"
#include "logging/csv_metric_logger.h"
#include "logging/gpu_metric_provider.h"
#include "logging/progress_logger.h"
#include "losses/losses.h"
#include "models/head_outputs.h"
#include "models/pfm_model_v21.h"
#include "models/planetary_graph_matcher.h"
#include "optim/amp.h"
#include "optim/descriptor_similarity.h"
#include "train/training_visualization.h"

namespace pfm
{
namespace
{

constexpr int64_t INPUT_CHANNELS = 1;
constexpr int64_t MAX_DESCRIPTOR_LOSS_SAMPLES = 1024;
// 下面的权重按训练目标分组排列：检测器、描述子、图匹配、稠密分支和日志/可视化控制。
constexpr double REPEATABILITY_LOSS_WEIGHT = 100.0;
constexpr double REPEATABLE_SALIENCY_TARGET_WEIGHT = 8.0;
constexpr double REPEATABLE_KEYPOINT_TARGET_WEIGHT = 25.0;
constexpr double WARP_ALIGNED_KEYPOINT_TARGET_WEIGHT = 20.0;
constexpr double WARP_ALIGNED_KEYPOINT_PEAK_WEIGHT = 160.0;
constexpr double DECODED_KEYPOINT_REPEATABILITY_WEIGHT = 400.0;
constexpr double REPEATABLE_KEYPOINT_TARGET_FRACTION = 0.03;
constexpr int64_t REPEATABLE_KEYPOINT_TARGET_MIN_COUNT = 32;
constexpr int64_t REPEATABLE_KEYPOINT_TARGET_MAX_COUNT = 2048;
constexpr double REPEATABLE_GRID_KEYPOINT_TARGET_WEIGHT = 150.0;
constexpr int64_t REPEATABLE_GRID_KEYPOINT_TARGET_ROWS = 32;
constexpr int64_t REPEATABLE_GRID_KEYPOINT_TARGET_COLS = 32;
constexpr double DESCRIPTOR_DIVERSITY_WEIGHT = 5.0;
constexpr double DESCRIPTOR_MAP_STD_TARGET = 0.25;
constexpr double DESCRIPTOR_MAP_COVARIANCE_WEIGHT = 0.5;
constexpr double DESCRIPTOR_MAP_UNIFORMITY_WEIGHT = 2.0;
constexpr int64_t DESCRIPTOR_MAP_UNIFORMITY_MAX_SAMPLES = 512;
constexpr int64_t DESCRIPTOR_NEGATIVE_SAMPLE_COUNT = 127;
constexpr double DENSE_DESCRIPTOR_HARD_NEGATIVE_MARGIN = 0.25;
constexpr double DENSE_DESCRIPTOR_HARD_NEGATIVE_WEIGHT = 8.0;
constexpr double DENSE_DESCRIPTOR_HARD_NEGATIVE_EXCLUSION_RADIUS = 2.0;
constexpr int64_t DENSE_DESCRIPTOR_TOPK_NEGATIVES = 16;
constexpr double DENSE_DESCRIPTOR_REVERSE_HARD_NEGATIVE_WEIGHT = 0.25;
constexpr int64_t DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT = 16;
constexpr double WARP_DESCRIPTOR_CONTRASTIVE_WEIGHT = 3.0;
constexpr double DIRECT_FULL_MAP_DESCRIPTOR_WEIGHT = 1.5;
constexpr double CROSS_BATCH_DESCRIPTOR_CONTRASTIVE_WEIGHT = 0.0;
constexpr double POSITIVE_DESCRIPTOR_ALIGNMENT_WEIGHT = 80.0;
constexpr double PATCH_DESCRIPTOR_ALIGNMENT_WEIGHT = 40.0;
constexpr double DESCRIPTOR_LOCAL_CE_WEIGHT = 0.25;
constexpr double DESCRIPTOR_GLOBAL_CE_WEIGHT = 0.25;
constexpr double SUPERVISED_DESCRIPTOR_RANKING_WEIGHT = 160.0;
constexpr double SUPERVISED_DESCRIPTOR_RANKING_MARGIN = 0.35;
constexpr double SUPERVISED_DESCRIPTOR_SOFT_RANK_WEIGHT = 0.35;
constexpr double SUPERVISED_DESCRIPTOR_TAIL_RANK_WEIGHT = 0.2;
constexpr float SUPERVISED_DESCRIPTOR_TAIL_RANK_START = 16.0F;
constexpr int64_t SUPERVISED_DESCRIPTOR_TOPK_NEGATIVES = 64;
constexpr float SUPERVISED_DESCRIPTOR_RANKING_LOGIT_SCALE = 20.0F;
constexpr float SUPERVISED_DESCRIPTOR_SOFT_RANK_SCALE = 20.0F;
constexpr double SAMPLED_DESCRIPTOR_DECORRELATION_WEIGHT = 20.0;
constexpr double SAMPLED_DESCRIPTOR_DECORRELATION_MARGIN = 0.25;
constexpr int64_t SAMPLED_DESCRIPTOR_DECORRELATION_TOPK = 16;
constexpr int64_t SAMPLED_DESCRIPTOR_DECORRELATION_EXCLUSION_RADIUS = 2;
constexpr double PAIRWISE_TEXTURE_TEACHER_WEIGHT = 0.0;
constexpr double ROTATION_INVARIANT_TEXTURE_TARGET_WEIGHT = 20.0;
constexpr double ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT = 0.5;
constexpr double DESCRIPTOR_FINETUNE_ANCHOR_WEIGHT = 800.0;
constexpr float CROSS_BATCH_DESCRIPTOR_LOGIT_SCALE = 30.0F;
constexpr double KEYPOINT_DESCRIPTOR_LOSS_WEIGHT = 8.0;
constexpr double LEARNED_KEYPOINT_DESCRIPTOR_LOSS_WEIGHT = 64.0;
constexpr double SUPERVISED_KEYPOINT_DESCRIPTOR_LOSS_WEIGHT = 4.0;
constexpr double KEYPOINT_DESCRIPTOR_MARGIN = 0.2;
constexpr double KEYPOINT_DESCRIPTOR_MARGIN_WEIGHT = 12.0;
constexpr double KEYPOINT_DESCRIPTOR_POSITIVE_ALIGNMENT_WEIGHT = 40.0;
constexpr double KEYPOINT_DESCRIPTOR_FALSE_NEGATIVE_RADIUS_PIXELS = 12.0;
constexpr int64_t KEYPOINT_DESCRIPTOR_MAX_QUERIES = 1024;
constexpr double KEYPOINT_DENSE_DESCRIPTOR_LOSS_WEIGHT = 1.0;
constexpr double LEARNED_KEYPOINT_DENSE_DESCRIPTOR_LOSS_WEIGHT = 64.0;
constexpr double KEYPOINT_DENSE_DESCRIPTOR_MARGIN = 0.25;
constexpr double KEYPOINT_DENSE_DESCRIPTOR_MARGIN_WEIGHT = 8.0;
constexpr double KEYPOINT_DENSE_DESCRIPTOR_POSITIVE_ALIGNMENT_WEIGHT = 20.0;
constexpr int64_t KEYPOINT_DENSE_DESCRIPTOR_MAX_QUERIES = 1024;
constexpr double KEYPOINT_PATCH_DESCRIPTOR_ALIGNMENT_WEIGHT = 24.0;
constexpr int64_t KEYPOINT_PATCH_DESCRIPTOR_KERNEL = 3;
constexpr double WARPED_KEYPOINT_DESCRIPTOR_CONTRASTIVE_WEIGHT = 160.0;
constexpr double ORIENTATION_LOSS_WEIGHT = 0.0;
constexpr double GRAPH_MATCHING_LOSS_WEIGHT = 4.0;
constexpr double LEARNED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT = 4.0;
constexpr double WARP_COMPLETED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT = 2.0;
constexpr double SUPERVISED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT = 1.0;
constexpr double HEATMAP_SELECTION_WEIGHT = 200.0;
constexpr double HEATMAP_TARGET_MEAN = 0.03;
constexpr double HEATMAP_BINARY_WEIGHT = 0.05;
constexpr int64_t GRAPH_MATCHING_MAX_QUERIES = 512;
constexpr int64_t GRAPH_DENSE_MATCHING_MAX_SAMPLES = 256;
constexpr int64_t GRAPH_DENSE_MATCHING_GRID_ROWS = 8;
constexpr int64_t GRAPH_DENSE_MATCHING_GRID_COLS = 8;
constexpr float HEATMAP_TARGET_POSITIVE_WEIGHT = 8.0F;
constexpr double GRAPH_MATCHING_POSITIVE_RADIUS_PIXELS = 6.0;
constexpr float GRAPH_POSITIVE_MARGIN = 1.0F;
constexpr float GRAPH_POSITIVE_MARGIN_WEIGHT = 2.0F;
constexpr float OFFSET_LOSS_WEIGHT = 0.0F;
constexpr float CONFIDENCE_LOSS_WEIGHT = 0.0F;
constexpr int64_t SPARSE_FEATURE_CHANNEL_MULTIPLIER = 2;
constexpr std::size_t TRAINING_VISUALIZATION_QUEUE_CAPACITY = 2048;
constexpr std::size_t TRAINING_VISUALIZATION_WORKER_COUNT = 4;
constexpr double TRAINING_MATCH_CORRECT_THRESHOLD_PIXELS = 3.0;
constexpr double PI = 3.14159265358979323846;
constexpr int64_t MAX_VISUALIZED_SPARSE_MATCH_LINES = 2048;
constexpr int64_t MAX_VISUALIZED_DENSE_MATCH_LINES = 2048;
constexpr int TRAINING_METRIC_LOG_INTERVAL = 1;
constexpr int TRAINING_DECODE_MAX_KEYPOINTS = 2048;
constexpr int64_t DESCRIPTOR_SIMILARITY_QUERY_CHUNK = 128;
constexpr int64_t TRAINING_KEYPOINT_LOSS_BATCH_ITEMS = 4;
const std::vector<std::string> TRAINING_CSV_COLUMNS = {
    "loss_total",
    "feature_loss",
    "repeatability_loss",
    "descriptor_loss",
    "orientation_loss",
    "matcher_loss",
    "graph_matching_loss",
    "graph_matching_accuracy",
    "graph_positive_fraction",
    "graph_positive_count",
    "graph_query_count",
    "graph_features_a",
    "graph_features_b",
    "learned_graph_matching_accuracy",
    "learned_graph_positive_fraction",
    "learned_graph_positive_count",
    "learned_graph_query_count",
    "dense_loss",
    "offset_loss",
    "confidence_loss",
    "descriptor_accuracy",
    "descriptor_positive_score",
    "descriptor_hard_negative_score",
    "descriptor_positive_margin",
    "descriptor_positive_rank",
    "keypoint_descriptor_accuracy",
    "keypoint_descriptor_positive_margin",
    "keypoint_descriptor_positive_rank",
    "descriptor_diversity",
    "offset_error_px",
    "gpu_utilization_percent",
    "gpu_power_watts",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_memory_free_mb",
};

int64_t descriptor_broad_far_negative_count_for_progress(double progress)
{
    // 广域 hard negative 在课程后半段逐步打开，早期先让模型学稳局部正样本。
    if (!std::isfinite(progress) || progress <= 0.25)
    {
        return 0;
    }
    if (progress < 0.50)
    {
        return std::max<int64_t>(1, DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT / 8);
    }
    if (progress < 0.75)
    {
        return std::max<int64_t>(1, DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT / 4);
    }
    return std::max<int64_t>(1, DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT / 2);
}

enum class MatchLineColor
{
    Red,
    Green,
};

struct MatchCorrectnessStats
{
    int64_t correct = 0;
    int64_t wrong = 0;
};

struct TrainingCacheSpec
{
    std::string cache_dir;
    std::optional<std::size_t> pair_index;
};

enum class TrainingProfile
{
    Smoke,
    Detector,
    Descriptor,
    Graph,
    Full,
    PythonCompare,
};

TrainingProfile parse_training_profile(const std::string& value)
{
    // training_profile 决定启用哪些损失分支，便于 smoke、descriptor-only、graph-only 等分阶段实验复用同一训练入口。
    if (value == "smoke")
    {
        return TrainingProfile::Smoke;
    }
    if (value == "detector")
    {
        return TrainingProfile::Detector;
    }
    if (value == "descriptor")
    {
        return TrainingProfile::Descriptor;
    }
    if (value == "graph")
    {
        return TrainingProfile::Graph;
    }
    if (value == "full")
    {
        return TrainingProfile::Full;
    }
    if (value == "python-compare")
    {
        return TrainingProfile::PythonCompare;
    }
    throw std::invalid_argument(
        "training_profile must be one of smoke, detector, descriptor, graph, full, or legacy alias python-compare");
}

int64_t training_profile_id(TrainingProfile profile)
{
    switch (profile)
    {
    case TrainingProfile::Smoke:
        return 0;
    case TrainingProfile::Detector:
        return 1;
    case TrainingProfile::Descriptor:
        return 2;
    case TrainingProfile::Graph:
        return 3;
    case TrainingProfile::Full:
        return 4;
    case TrainingProfile::PythonCompare:
        return 5;
    }
    return 5;
}

bool training_profile_uses_detector_targets(TrainingProfile profile)
{
    return profile == TrainingProfile::Detector || profile == TrainingProfile::Descriptor ||
           profile == TrainingProfile::Graph || profile == TrainingProfile::Full;
}

bool training_profile_uses_descriptor_losses(TrainingProfile profile)
{
    return profile == TrainingProfile::Descriptor || profile == TrainingProfile::Full ||
           profile == TrainingProfile::PythonCompare;
}

bool training_profile_uses_graph_losses(TrainingProfile profile)
{
    return profile == TrainingProfile::Graph || profile == TrainingProfile::Full ||
           profile == TrainingProfile::PythonCompare;
}

bool training_profile_uses_dense_pair_loss(TrainingProfile)
{
    return false;
}

bool training_profile_uses_python_aligned_pair_loss(TrainingProfile profile)
{
    return profile == TrainingProfile::Full || profile == TrainingProfile::PythonCompare;
}

bool training_profile_uses_dense_quality_forward(TrainingProfile profile)
{
    return profile != TrainingProfile::Smoke && !training_profile_uses_python_aligned_pair_loss(profile);
}

void validate_config(const TrainConfig& config)
{
    // 训练入口的参数校验集中在这里，后续构建模型、DataLoader 和优化器时不再重复检查。
    if (config.image_dir.empty() && config.pair_cache_dirs.empty())
    {
        throw std::invalid_argument("image_dir or pair_cache_dirs must not be empty");
    }
    if (config.checkpoint.empty())
    {
        throw std::invalid_argument("checkpoint must not be empty");
    }
    if (!config.init_checkpoint.empty() && !std::filesystem::exists(config.init_checkpoint))
    {
        throw std::invalid_argument("init_checkpoint does not exist: " + config.init_checkpoint);
    }
    const int finetune_modes = (config.descriptor_only_finetune ? 1 : 0) +
                               (config.viewpoint_head_only_finetune ? 1 : 0) + (config.graph_only_finetune ? 1 : 0);
    if (finetune_modes > 1)
    {
        throw std::invalid_argument(
            "descriptor_only_finetune, viewpoint_head_only_finetune, and graph_only_finetune are mutually exclusive");
    }
    if (config.epochs <= 0)
    {
        throw std::invalid_argument("epochs must be positive");
    }
    if (config.batch_size <= 0)
    {
        throw std::invalid_argument("batch_size must be positive");
    }
    if (config.base_channels <= 0)
    {
        throw std::invalid_argument("base_channels must be positive");
    }
    if (config.descriptor_dim <= 0)
    {
        throw std::invalid_argument("descriptor_dim must be positive");
    }
    if (config.graph_hidden_dim <= 0)
    {
        throw std::invalid_argument("graph_hidden_dim must be positive");
    }
    if (config.graph_attention_layers <= 0)
    {
        throw std::invalid_argument("graph_attention_layers must be positive");
    }
    if (config.graph_keypoint_meta_dim <= 0)
    {
        throw std::invalid_argument("graph_keypoint_meta_dim must be positive");
    }
    (void)parse_training_profile(config.training_profile);
    if (config.samples_per_pair <= 0)
    {
        throw std::invalid_argument("samples_per_pair must be positive");
    }
    if (!std::isfinite(config.synthetic_loss_weight) || config.synthetic_loss_weight < 0.0)
    {
        throw std::invalid_argument("synthetic_loss_weight must be non-negative and finite");
    }
    if (!std::isfinite(config.graph_matcher_loss_weight) || config.graph_matcher_loss_weight < 0.0)
    {
        throw std::invalid_argument("graph_matcher_loss_weight must be non-negative and finite");
    }
    if (!std::isfinite(config.graph_matcher_accept_weight) || config.graph_matcher_accept_weight < 0.0)
    {
        throw std::invalid_argument("graph_matcher_accept_weight must be non-negative and finite");
    }
    if (config.graph_matcher_accept_negative_topk < 0)
    {
        throw std::invalid_argument("graph_matcher_accept_negative_topk must be non-negative");
    }
    if (config.graph_matcher_no_match_points < 0)
    {
        throw std::invalid_argument("graph_matcher_no_match_points must be non-negative");
    }
    if (!std::isfinite(config.graph_matcher_no_match_min_distance) ||
        config.graph_matcher_no_match_min_distance < 0.0)
    {
        throw std::invalid_argument("graph_matcher_no_match_min_distance must be non-negative and finite");
    }
    if (config.graph_matcher_train_max_attention_layers < 0)
    {
        throw std::invalid_argument("graph_matcher_train_max_attention_layers must be non-negative");
    }
    if (!std::isfinite(config.graph_matcher_train_max_attention_work_fraction) ||
        config.graph_matcher_train_max_attention_work_fraction < 0.0 ||
        config.graph_matcher_train_max_attention_work_fraction > 1.0)
    {
        throw std::invalid_argument("graph_matcher_train_max_attention_work_fraction must be in [0, 1]");
    }
    if (!std::isfinite(config.graph_matcher_train_width_keep_ratio) ||
        config.graph_matcher_train_width_keep_ratio <= 0.0 || config.graph_matcher_train_width_keep_ratio > 1.0)
    {
        throw std::invalid_argument("graph_matcher_train_width_keep_ratio must be in (0, 1]");
    }
    if (!std::isfinite(config.graph_matcher_prune_ranking_weight) ||
        config.graph_matcher_prune_ranking_weight < 0.0)
    {
        throw std::invalid_argument("graph_matcher_prune_ranking_weight must be non-negative and finite");
    }
    if (!std::isfinite(config.graph_matcher_prune_ranking_margin) ||
        config.graph_matcher_prune_ranking_margin < 0.0)
    {
        throw std::invalid_argument("graph_matcher_prune_ranking_margin must be non-negative and finite");
    }
    if (!std::isfinite(config.graph_matcher_stop_confidence_weight) ||
        config.graph_matcher_stop_confidence_weight < 0.0)
    {
        throw std::invalid_argument("graph_matcher_stop_confidence_weight must be non-negative and finite");
    }
    if (!std::isfinite(config.graph_matcher_stop_confidence_margin) ||
        config.graph_matcher_stop_confidence_margin < 0.0)
    {
        throw std::invalid_argument("graph_matcher_stop_confidence_margin must be non-negative and finite");
    }
    if (!std::isfinite(config.training_texture_blend_weight) || config.training_texture_blend_weight < 0.0)
    {
        throw std::invalid_argument("training_texture_blend_weight must be non-negative and finite");
    }
    if (!std::isfinite(config.temperature) || config.temperature <= 0.0)
    {
        throw std::invalid_argument("temperature must be positive and finite");
    }
    if (config.resize < 0)
    {
        throw std::invalid_argument("resize must be non-negative");
    }
    if (config.training_crop_size < 0)
    {
        throw std::invalid_argument("training_crop_size must be non-negative");
    }
    if (config.pairs_per_image <= 0)
    {
        throw std::invalid_argument("pairs_per_image must be positive");
    }
    if (config.max_train_batches < 0)
    {
        throw std::invalid_argument("max_train_batches must be non-negative");
    }
    (void)parse_synthetic_pair_augmentation_profile(config.augmentation_profile);
    if (!std::isfinite(config.rotation_step_degrees) || config.rotation_step_degrees <= 0.0)
    {
        throw std::invalid_argument("rotation_step_degrees must be positive and finite");
    }
    if (config.extreme_pair_ratio < 0.0 || config.extreme_pair_ratio > 1.0)
    {
        throw std::invalid_argument("extreme_pair_ratio must be between 0 and 1");
    }
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0)
    {
        throw std::invalid_argument("learning_rate must be positive and finite");
    }
    if (config.lr_warmup_steps < 0)
    {
        throw std::invalid_argument("lr_warmup_steps must be non-negative");
    }
    if (!std::isfinite(config.min_learning_rate_ratio) || config.min_learning_rate_ratio < 0.0 ||
        config.min_learning_rate_ratio > 1.0)
    {
        throw std::invalid_argument("min_learning_rate_ratio must be finite and between 0 and 1");
    }
    if (!std::isfinite(config.weight_decay) || config.weight_decay < 0.0)
    {
        throw std::invalid_argument("weight_decay must be non-negative and finite");
    }
    if (!std::isfinite(config.gradient_clip_norm) || config.gradient_clip_norm < 0.0)
    {
        throw std::invalid_argument("gradient_clip_norm must be non-negative and finite");
    }
    if (config.seed < 0)
    {
        throw std::invalid_argument("seed must be non-negative");
    }
    if (config.dataloader_workers < 0)
    {
        throw std::invalid_argument("dataloader_workers must be non-negative");
    }
    for (const auto& cache_dir : config.extra_synthetic_pair_cache_dirs)
    {
        if (cache_dir.empty())
        {
            throw std::invalid_argument("extra_synthetic_pair_cache_dirs must not contain empty paths");
        }
    }
    for (const auto& cache_dir : config.hard_synthetic_pair_cache_dirs)
    {
        if (cache_dir.empty())
        {
            throw std::invalid_argument("hard_synthetic_pair_cache_dirs must not contain empty paths");
        }
    }
    for (const auto& cache_dir : config.pair_cache_dirs)
    {
        if (cache_dir.empty())
        {
            throw std::invalid_argument("pair_cache_dirs must not contain empty paths");
        }
    }
    if (config.pair_cache_limit < 0)
    {
        throw std::invalid_argument("pair_cache_limit must be non-negative");
    }
    if (config.pair_memory_cache_size < 0)
    {
        throw std::invalid_argument("pair_memory_cache_size must be non-negative");
    }
    if (config.hard_synthetic_pair_cache_repeats <= 0)
    {
        throw std::invalid_argument("hard_synthetic_pair_cache_repeats must be positive");
    }
    for (const auto index : config.hard_synthetic_pair_cache_indices)
    {
        if (index < 0)
        {
            throw std::invalid_argument("hard_synthetic_pair_cache_indices must be non-negative");
        }
    }
    if (config.prefetch_batches <= 0)
    {
        throw std::invalid_argument("prefetch_batches must be positive");
    }
    if (config.cache_only && config.synthetic_pair_cache_dir.empty())
    {
        throw std::invalid_argument("cache_only requires synthetic_pair_cache_dir");
    }
    if (config.visualization_samples < 0)
    {
        throw std::invalid_argument("visualization_samples must be non-negative");
    }
    if (config.max_keypoints <= 0)
    {
        throw std::invalid_argument("max_keypoints must be positive");
    }
    if (config.min_keypoints < 0)
    {
        throw std::invalid_argument("min_keypoints must be non-negative");
    }
    if (config.min_keypoints > config.max_keypoints)
    {
        throw std::invalid_argument("min_keypoints must not exceed max_keypoints");
    }
    if (config.keypoint_grid_rows <= 0)
    {
        throw std::invalid_argument("keypoint_grid_rows must be positive");
    }
    if (config.keypoint_grid_cols <= 0)
    {
        throw std::invalid_argument("keypoint_grid_cols must be positive");
    }
    if (config.keypoints_per_cell < 0)
    {
        throw std::invalid_argument("keypoints_per_cell must be non-negative");
    }
    if (config.nms_radius < 0)
    {
        throw std::invalid_argument("nms_radius must be non-negative");
    }
    validate_min_keypoint_intensity(config.min_keypoint_intensity);
}

double training_learning_rate_for_step(const TrainConfig& config, int64_t step, int64_t total_steps)
{
    // 支持可选 warmup + cosine decay，min_learning_rate_ratio 控制最低学习率。
    if (total_steps <= 0)
    {
        return config.learning_rate;
    }
    step = std::max<int64_t>(0, std::min<int64_t>(step, total_steps - 1));
    if (config.lr_warmup_steps > 0 && step < config.lr_warmup_steps)
    {
        const double warmup_progress = static_cast<double>(step + 1) / static_cast<double>(config.lr_warmup_steps);
        return config.learning_rate * std::min(1.0, warmup_progress);
    }

    const auto warmup_steps = std::min<int64_t>(static_cast<int64_t>(config.lr_warmup_steps), total_steps);
    const auto decay_steps = std::max<int64_t>(1, total_steps - warmup_steps);
    const auto decay_step = std::max<int64_t>(0, step - warmup_steps);
    const double progress = decay_steps <= 1 ? 0.0
                                             : static_cast<double>(std::min<int64_t>(decay_step, decay_steps - 1)) /
                                                   static_cast<double>(decay_steps - 1);
    const double lr_max = config.learning_rate;
    const double lr_min = lr_max * config.min_learning_rate_ratio;
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + std::cos(PI * progress));
}

void set_optimizer_learning_rate(torch::optim::Optimizer& optimizer, double learning_rate)
{
    for (auto& group : optimizer.param_groups())
    {
        group.options().set_lr(learning_rate);
    }
}

TrainingMetric make_iteration_metric(const TrainConfig& config, int epoch, int iteration, int total_iterations,
                                     int images_seen, int total_images, double learning_rate, double elapsed_seconds,
                                     const GpuMetrics& gpu_metrics,
                                     const std::unordered_map<std::string, double>& values)
{
    TrainingMetric metric;
    metric.epoch = epoch;
    metric.total_epochs = config.epochs;
    metric.iteration = iteration;
    metric.total_iterations = total_iterations;
    metric.images_seen = images_seen;
    metric.total_images = total_images;
    metric.learning_rate = learning_rate;
    metric.elapsed_seconds = elapsed_seconds;
    metric.values = values;
    if (gpu_metrics.utilization_percent.has_value())
    {
        metric.values["gpu_utilization_percent"] = gpu_metrics.utilization_percent.value();
    }
    if (gpu_metrics.power_watts.has_value())
    {
        metric.values["gpu_power_watts"] = gpu_metrics.power_watts.value();
    }
    if (gpu_metrics.memory_used_mb.has_value())
    {
        metric.values["gpu_memory_used_mb"] = gpu_metrics.memory_used_mb.value();
    }
    if (gpu_metrics.memory_total_mb.has_value())
    {
        metric.values["gpu_memory_total_mb"] = gpu_metrics.memory_total_mb.value();
    }
    if (gpu_metrics.memory_free_mb.has_value())
    {
        metric.values["gpu_memory_free_mb"] = gpu_metrics.memory_free_mb.value();
    }
    return metric;
}

bool is_no_valid_correspondence_error(const std::runtime_error& exc)
{
    return std::string(exc.what()) == "no valid correspondences sampled";
}

torch::Tensor ensure_grayscale(const torch::Tensor& image)
{
    require_chw_image(image);
    if (channels(image) == INPUT_CHANNELS)
    {
        return image;
    }
    return image.mean(0, true).contiguous();
}

enum class BatchTensorLayout
{
    Hw,
    Chw,
    Hwc
};

int64_t spatial_height(const torch::Tensor& tensor, BatchTensorLayout layout)
{
    switch (layout)
    {
    case BatchTensorLayout::Hw:
    case BatchTensorLayout::Hwc:
        return tensor.size(0);
    case BatchTensorLayout::Chw:
        return tensor.size(1);
    }
    throw std::invalid_argument("unsupported batch tensor layout");
}

int64_t spatial_width(const torch::Tensor& tensor, BatchTensorLayout layout)
{
    switch (layout)
    {
    case BatchTensorLayout::Hw:
    case BatchTensorLayout::Hwc:
        return tensor.size(1);
    case BatchTensorLayout::Chw:
        return tensor.size(2);
    }
    throw std::invalid_argument("unsupported batch tensor layout");
}

torch::Tensor pad_spatial_tensor(const torch::Tensor& tensor, int64_t target_height, int64_t target_width,
                                 BatchTensorLayout layout)
{
    const auto height = spatial_height(tensor, layout);
    const auto width = spatial_width(tensor, layout);
    if (height == target_height && width == target_width)
    {
        return tensor.contiguous();
    }
    if (layout == BatchTensorLayout::Hw)
    {
        auto padded = torch::zeros({target_height, target_width}, tensor.options());
        padded.index_put_({torch::indexing::Slice(0, height), torch::indexing::Slice(0, width)}, tensor);
        return padded.contiguous();
    }
    if (layout == BatchTensorLayout::Hwc)
    {
        auto padded = torch::zeros({target_height, target_width, tensor.size(2)}, tensor.options());
        padded.index_put_(
            {torch::indexing::Slice(0, height), torch::indexing::Slice(0, width), torch::indexing::Slice()}, tensor);
        return padded.contiguous();
    }
    auto padded = torch::zeros({tensor.size(0), target_height, target_width}, tensor.options());
    padded.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, height), torch::indexing::Slice(0, width)},
                      tensor);
    return padded.contiguous();
}

torch::Tensor stack_batch(const std::vector<torch::Tensor>& tensors, BatchTensorLayout layout)
{
    int64_t target_height = 0;
    int64_t target_width = 0;
    for (const auto& tensor : tensors)
    {
        target_height = std::max<int64_t>(target_height, spatial_height(tensor, layout));
        target_width = std::max<int64_t>(target_width, spatial_width(tensor, layout));
    }
    std::vector<torch::Tensor> padded_tensors;
    padded_tensors.reserve(tensors.size());
    for (const auto& tensor : tensors)
    {
        padded_tensors.push_back(pad_spatial_tensor(tensor, target_height, target_width, layout));
    }
    return torch::stack(padded_tensors).contiguous();
}

std::string pair_visualization_stem(std::size_t index)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "pair_%06zu", index);
    return buffer;
}

torch::Tensor mask_to_image(const torch::Tensor& mask)
{
    return mask.to(torch::kCPU, torch::kFloat32).unsqueeze(0).contiguous();
}

torch::Tensor make_visualization_mask(const torch::Tensor& image, double min_keypoint_intensity)
{
    if (min_keypoint_intensity <= 0.0)
    {
        return torch::ones({image.size(1), image.size(2)},
                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    }
    return make_intensity_mask(image.detach().to(torch::kCPU, torch::kFloat32).contiguous(), min_keypoint_intensity);
}

std::pair<int64_t, int64_t> map_feature_point(const torch::Tensor& point, int64_t map_width, int64_t map_height,
                                              int64_t image_width, int64_t image_height)
{
    const auto scale_x = static_cast<float>(image_width) / static_cast<float>(std::max<int64_t>(1, map_width));
    const auto scale_y = static_cast<float>(image_height) / static_cast<float>(std::max<int64_t>(1, map_height));
    const int64_t x = std::min<int64_t>(image_width - 1,
                                        std::max<int64_t>(0, std::llround(point.index({0}).item<float>() * scale_x)));
    const int64_t y = std::min<int64_t>(image_height - 1,
                                        std::max<int64_t>(0, std::llround(point.index({1}).item<float>() * scale_y)));
    return {x, y};
}

torch::Tensor ensure_visualization_rgb(const torch::Tensor& image)
{
    auto output = image.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    if (output.size(0) == 1)
    {
        output = output.repeat({3, 1, 1}).contiguous();
    }
    return output;
}

void draw_image_point(torch::Tensor& output, int64_t x, int64_t y, MatchLineColor color = MatchLineColor::Red)
{
    output.index_put_({0, y, x}, color == MatchLineColor::Red ? 1.0F : 0.0F);
    output.index_put_({1, y, x}, color == MatchLineColor::Green ? 1.0F : 0.0F);
    output.index_put_({2, y, x}, 0.0F);
}

void draw_image_line(torch::Tensor& output, int64_t x0, int64_t y0, int64_t x1, int64_t y1,
                     MatchLineColor color = MatchLineColor::Red)
{
    const int64_t steps = std::max<int64_t>(std::abs(x1 - x0), std::abs(y1 - y0));
    for (int64_t step = 0; step <= steps; ++step)
    {
        const double alpha = steps == 0 ? 0.0 : static_cast<double>(step) / static_cast<double>(steps);
        const int64_t x = std::llround(static_cast<double>(x0) * (1.0 - alpha) + static_cast<double>(x1) * alpha);
        const int64_t y = std::llround(static_cast<double>(y0) * (1.0 - alpha) + static_cast<double>(y1) * alpha);
        draw_image_point(output, x, y, color);
    }
}

void draw_feature_points(torch::Tensor& output, const torch::Tensor& points, int64_t map_width, int64_t map_height,
                         const torch::Tensor& mask)
{
    auto cpu_points = points.to(torch::kCPU, torch::kFloat32).contiguous();
    auto cpu_mask = mask.to(torch::kCPU, torch::kBool).contiguous();
    for (int64_t index = 0; index < cpu_points.size(0); ++index)
    {
        const auto [x, y] = map_feature_point(cpu_points[index], map_width, map_height, output.size(2), output.size(1));
        if (cpu_mask.index({y, x}).item<bool>())
        {
            draw_image_point(output, x, y);
        }
    }
}

torch::Tensor feature_overlay_image(const torch::Tensor& image, const FeatureSet& features,
                                    double min_keypoint_intensity)
{
    auto output = ensure_visualization_rgb(image);
    if (features.keypoints.defined() && features.keypoints.numel() > 0)
    {
        draw_feature_points(output, features.keypoints, features.feature_map_width, features.feature_map_height,
                            make_visualization_mask(image, min_keypoint_intensity));
    }
    return output;
}

torch::Tensor warp_overlay_image(const SyntheticPair& pair)
{
    auto output = pair.view_a.detach().to(torch::kCPU, torch::kFloat32).clone().contiguous();
    auto valid = pair.valid_mask.to(torch::kCPU, torch::kBool).contiguous();
    for (int64_t y = 0; y < valid.size(0); y += 4)
    {
        for (int64_t x = 0; x < valid.size(1); x += 4)
        {
            if (valid.index({y, x}).item<bool>())
            {
                output.index_put_({0, y, x}, 1.0F);
            }
        }
    }
    return output;
}

bool is_match_correct(const torch::Tensor& warp_a_to_b, int64_t x_a, int64_t y_a, int64_t x_b, int64_t y_b,
                      double correct_threshold_pixels)
{
    if (!warp_a_to_b.defined() || warp_a_to_b.numel() == 0)
    {
        return false;
    }
    const auto warp = warp_a_to_b.to(torch::kCPU, torch::kFloat32).contiguous();
    if (warp.dim() != 3 || warp.size(2) != 2 || y_a < 0 || y_a >= warp.size(0) || x_a < 0 || x_a >= warp.size(1))
    {
        return false;
    }
    const auto expected_x = warp.index({y_a, x_a, 0}).item<float>();
    const auto expected_y = warp.index({y_a, x_a, 1}).item<float>();
    const auto dx = static_cast<double>(x_b) - static_cast<double>(expected_x);
    const auto dy = static_cast<double>(y_b) - static_cast<double>(expected_y);
    return std::sqrt(dx * dx + dy * dy) <= correct_threshold_pixels;
}

void add_match_correctness(MatchCorrectnessStats& stats, bool correct)
{
    if (correct)
    {
        ++stats.correct;
    }
    else
    {
        ++stats.wrong;
    }
}

void draw_sparse_match_lines(torch::Tensor& output, int64_t image_a_width, const FeatureSet& features_a,
                             const FeatureSet& features_b, const MatchSet& matches,
                             const torch::Tensor& warp_a_to_b = torch::Tensor(), double correct_threshold_pixels = 0.0,
                             MatchCorrectnessStats* stats = nullptr)
{
    if (!matches.sparse_matches.defined() || matches.sparse_matches.numel() == 0 || !features_a.keypoints.defined() ||
        !features_b.keypoints.defined())
    {
        return;
    }
    auto sparse_matches = matches.sparse_matches.to(torch::kCPU, torch::kLong).contiguous();
    auto points_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const int64_t sparse_count = std::min<int64_t>(sparse_matches.size(0), MAX_VISUALIZED_SPARSE_MATCH_LINES);
    for (int64_t index = 0; index < sparse_count; ++index)
    {
        const auto index_a = sparse_matches.index({index, 0}).item<int64_t>();
        const auto index_b = sparse_matches.index({index, 1}).item<int64_t>();
        if (index_a < 0 || index_a >= points_a.size(0) || index_b < 0 || index_b >= points_b.size(0))
        {
            continue;
        }
        const auto [x_a, y_a] = map_feature_point(points_a[index_a], features_a.feature_map_width,
                                                  features_a.feature_map_height, image_a_width, output.size(1));
        const auto [x_b, y_b] =
            map_feature_point(points_b[index_b], features_b.feature_map_width, features_b.feature_map_height,
                              output.size(2) - image_a_width, output.size(1));
        const bool correct = is_match_correct(warp_a_to_b, x_a, y_a, x_b, y_b, correct_threshold_pixels);
        if (stats != nullptr)
        {
            add_match_correctness(*stats, correct);
        }
        draw_image_line(output, x_a, y_a, x_b + image_a_width, y_b,
                        correct ? MatchLineColor::Green : MatchLineColor::Red);
    }
}

void draw_dense_match_lines(torch::Tensor& output, int64_t image_a_width, const FeatureSet& features_a,
                            const FeatureSet& features_b, const MatchSet& matches,
                            const torch::Tensor& warp_a_to_b = torch::Tensor(), double correct_threshold_pixels = 0.0,
                            MatchCorrectnessStats* stats = nullptr)
{
    if (!matches.points_a.defined() || !matches.points_b.defined() || matches.points_a.numel() == 0)
    {
        return;
    }
    auto points_a = matches.points_a.to(torch::kCPU, torch::kFloat32).contiguous();
    auto points_b = matches.points_b.to(torch::kCPU, torch::kFloat32).contiguous();
    const int64_t image_b_width = output.size(2) - image_a_width;
    const int64_t count =
        std::min<int64_t>(std::min(points_a.size(0), points_b.size(0)), MAX_VISUALIZED_DENSE_MATCH_LINES);
    for (int64_t index = 0; index < count; ++index)
    {
        const auto [x_a, y_a] = map_feature_point(points_a[index], features_a.feature_map_width,
                                                  features_a.feature_map_height, image_a_width, output.size(1));
        const auto [x_b, y_b] = map_feature_point(points_b[index], features_b.feature_map_width,
                                                  features_b.feature_map_height, image_b_width, output.size(1));
        const bool correct = is_match_correct(warp_a_to_b, x_a, y_a, x_b, y_b, correct_threshold_pixels);
        if (stats != nullptr)
        {
            add_match_correctness(*stats, correct);
        }
        draw_image_line(output, x_a, y_a, x_b + image_a_width, y_b,
                        correct ? MatchLineColor::Green : MatchLineColor::Red);
    }
}

MatchCorrectnessStats compute_match_correctness_stats(const FeatureSet& features_a, const FeatureSet& features_b,
                                                      const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                                      double correct_threshold_pixels)
{
    auto output = torch::zeros({3, warp_a_to_b.size(0), warp_a_to_b.size(1) * 2}, torch::kFloat32);
    MatchCorrectnessStats stats;
    draw_sparse_match_lines(output, warp_a_to_b.size(1), features_a, features_b, matches, warp_a_to_b,
                            correct_threshold_pixels, &stats);
    draw_dense_match_lines(output, warp_a_to_b.size(1), features_a, features_b, matches, warp_a_to_b,
                           correct_threshold_pixels, &stats);
    return stats;
}

std::string model_match_overlay_text(const FeatureSet& features_a, const FeatureSet& features_b,
                                     const MatchSet& matches, const MatchCorrectnessStats& stats)
{
    return "features_a=" + std::to_string(features_a.keypoints.size(0)) +
           " features_b=" + std::to_string(features_b.keypoints.size(0)) +
           " sparse_matches=" + std::to_string(matches.sparse_matches.size(0)) +
           " dense_matches=" + std::to_string(matches.points_a.size(0)) +
           " correct_matches=" + std::to_string(stats.correct) + " wrong_matches=" + std::to_string(stats.wrong);
}

std::string model_match_overlay_text(const FeatureSet& features_a, const FeatureSet& features_b,
                                     const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                     double correct_threshold_pixels)
{
    return model_match_overlay_text(
        features_a, features_b, matches,
        compute_match_correctness_stats(features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels));
}

torch::Tensor match_overlay_image(const torch::Tensor& image_a, const torch::Tensor& image_b,
                                  const FeatureSet& features_a, const FeatureSet& features_b, const MatchSet& matches,
                                  const torch::Tensor& warp_a_to_b, double correct_threshold_pixels,
                                  double min_keypoint_intensity, MatchCorrectnessStats* stats = nullptr)
{
    auto output_a = feature_overlay_image(image_a, features_a, min_keypoint_intensity);
    auto output_b = feature_overlay_image(image_b, features_b, min_keypoint_intensity);
    auto output = torch::cat({output_a, output_b}, 2).contiguous();
    draw_sparse_match_lines(output, image_a.size(2), features_a, features_b, matches, warp_a_to_b,
                            correct_threshold_pixels, stats);
    draw_dense_match_lines(output, image_a.size(2), features_a, features_b, matches, warp_a_to_b,
                           correct_threshold_pixels, stats);
    return output;
}

torch::Tensor match_overlay_image(const torch::Tensor& image_a, const torch::Tensor& image_b,
                                  const FeatureSet& features_a, const FeatureSet& features_b, const MatchSet& matches,
                                  double min_keypoint_intensity)
{
    return match_overlay_image(image_a, image_b, features_a, features_b, matches, torch::Tensor(), 0.0,
                               min_keypoint_intensity);
}

torch::Tensor limit_training_image_size(const torch::Tensor& image, int64_t resize)
{
    const auto height = image.size(1);
    const auto width = image.size(2);
    const auto max_edge = std::max(height, width);
    if (resize == 0 || max_edge <= resize)
    {
        return image.contiguous();
    }

    const double scale = static_cast<double>(resize) / static_cast<double>(max_edge);
    const int64_t resized_height =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(height) * scale)));
    const int64_t resized_width =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(width) * scale)));
    return torch::nn::functional::interpolate(image.unsqueeze(0),
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{resized_height, resized_width})
                                                  .mode(torch::kBilinear)
                                                  .align_corners(false))
        .squeeze(0)
        .contiguous();
}

torch::Tensor resize_training_warp(const torch::Tensor& warp, int64_t target_height, int64_t target_width,
                                   int64_t source_b_height, int64_t source_b_width, int64_t target_b_height,
                                   int64_t target_b_width)
{
    const auto source_height = warp.size(0);
    const auto source_width = warp.size(1);
    if (source_height == target_height && source_width == target_width && source_b_height == target_b_height &&
        source_b_width == target_b_width)
    {
        return warp.contiguous();
    }

    auto resized = torch::nn::functional::interpolate(warp.permute({2, 0, 1}).unsqueeze(0),
                                                      torch::nn::functional::InterpolateFuncOptions()
                                                          .size(std::vector<int64_t>{target_height, target_width})
                                                          .mode(torch::kBilinear)
                                                          .align_corners(true))
                       .squeeze(0)
                       .permute({1, 2, 0})
                       .contiguous();
    resized.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0},
                       resized.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) *
                           (static_cast<double>(target_b_width - 1) /
                            static_cast<double>(std::max<int64_t>(1, source_b_width - 1))));
    resized.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1},
                       resized.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) *
                           (static_cast<double>(target_b_height - 1) /
                            static_cast<double>(std::max<int64_t>(1, source_b_height - 1))));
    return resized;
}

torch::Tensor resize_training_valid_mask(const torch::Tensor& valid_mask, int64_t target_height, int64_t target_width)
{
    if (valid_mask.size(0) == target_height && valid_mask.size(1) == target_width)
    {
        return valid_mask.contiguous();
    }

    return torch::nn::functional::interpolate(valid_mask.to(torch::kFloat32).unsqueeze(0).unsqueeze(0),
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{target_height, target_width})
                                                  .mode(torch::kArea))
        .squeeze(0)
        .squeeze(0)
        .gt(0.0)
        .to(valid_mask.dtype())
        .contiguous();
}

int64_t clamped_crop_origin(const torch::Tensor& center, int64_t crop_size, int64_t full_size)
{
    if (crop_size >= full_size)
    {
        return 0;
    }
    const auto center_value = center.detach().to(torch::kCPU, torch::kFloat32).item<float>();
    const auto origin =
        static_cast<int64_t>(std::nearbyint(center_value - static_cast<double>(crop_size - 1) * 0.5));
    return std::max<int64_t>(0, std::min<int64_t>(origin, full_size - crop_size));
}

std::optional<at::Generator> make_training_random_generator(const torch::Device& device, uint64_t seed)
{
    if (device.is_cuda())
    {
        auto generator = at::cuda::detail::createCUDAGenerator(device.index());
        generator.set_current_seed(seed);
        return generator;
    }
    return at::detail::createCPUGenerator(seed);
}

torch::Tensor randint_with_training_generator(int64_t high, at::IntArrayRef size, const torch::TensorOptions& options,
                                              std::optional<at::Generator>& generator)
{
    if (!generator.has_value())
    {
        return torch::randint(high, size, options);
    }
    return at::randint(high, size, *generator, options);
}

torch::Tensor randperm_with_training_generator(int64_t count, const torch::TensorOptions& options,
                                               std::optional<at::Generator>& generator)
{
    if (!generator.has_value())
    {
        return torch::randperm(count, options);
    }
    return at::randperm(count, *generator, options);
}

int64_t uniform_crop_origin(int64_t crop_size, int64_t full_size, const torch::Device& device,
                            std::optional<at::Generator>& generator)
{
    if (crop_size >= full_size)
    {
        return 0;
    }
    const auto limit = full_size - crop_size + 1;
    return randint_with_training_generator(limit, {1}, torch::TensorOptions().dtype(torch::kLong).device(device),
                                           generator)
        .item<int64_t>();
}

SyntheticPair crop_training_pair(const SyntheticPair& pair, int64_t crop_size, std::optional<at::Generator>& generator)
{
    using torch::indexing::Slice;

    if (crop_size <= 0)
    {
        return pair;
    }
    const auto height_a = pair.view_a.size(1);
    const auto width_a = pair.view_a.size(2);
    const auto height_b = pair.view_b.size(1);
    const auto width_b = pair.view_b.size(2);
    const auto crop_h_a = std::min<int64_t>(crop_size, height_a);
    const auto crop_w_a = std::min<int64_t>(crop_size, width_a);
    const auto crop_h_b = std::min<int64_t>(crop_size, height_b);
    const auto crop_w_b = std::min<int64_t>(crop_size, width_b);
    if (crop_h_a == height_a && crop_w_a == width_a && crop_h_b == height_b && crop_w_b == width_b)
    {
        return pair;
    }

    auto finite_full_warp = torch::isfinite(pair.warp_a_to_b).all(-1);
    auto valid_full = pair.valid_mask.to(torch::kBool).logical_and(finite_full_warp);
    int64_t ax0 = 0;
    int64_t ay0 = 0;
    if (valid_full.any().item<bool>())
    {
        auto valid_yx = torch::nonzero(valid_full);
        auto selected_index = randint_with_training_generator(
                                  valid_yx.size(0), {1},
                                  torch::TensorOptions().dtype(torch::kLong).device(valid_yx.device()), generator)
                                  .item<int64_t>();
        auto selected_yx = valid_yx.index({selected_index});
        ax0 = clamped_crop_origin(selected_yx.index({1}).to(torch::kFloat32), crop_w_a, width_a);
        ay0 = clamped_crop_origin(selected_yx.index({0}).to(torch::kFloat32), crop_h_a, height_a);
    }
    else
    {
        ax0 = uniform_crop_origin(crop_w_a, width_a, pair.view_a.device(), generator);
        ay0 = uniform_crop_origin(crop_h_a, height_a, pair.view_a.device(), generator);
    }
    const auto ax1 = ax0 + crop_w_a;
    const auto ay1 = ay0 + crop_h_a;

    auto warp_crop_full_b = pair.warp_a_to_b.index({Slice(ay0, ay1), Slice(ax0, ax1), Slice()});
    auto valid_crop = pair.valid_mask.index({Slice(ay0, ay1), Slice(ax0, ax1)}).to(torch::kBool).clone();
    auto finite_warp = torch::isfinite(warp_crop_full_b).all(-1);
    auto valid_for_center = valid_crop.logical_and(finite_warp);
    int64_t bx0 = 0;
    int64_t by0 = 0;
    if (valid_for_center.any().item<bool>())
    {
        auto center_b = warp_crop_full_b.index({valid_for_center}).mean(0);
        bx0 = clamped_crop_origin(center_b.index({0}), crop_w_b, width_b);
        by0 = clamped_crop_origin(center_b.index({1}), crop_h_b, height_b);
    }
    else
    {
        bx0 = std::max<int64_t>(0, std::min<int64_t>(ax0, width_b - crop_w_b));
        by0 = std::max<int64_t>(0, std::min<int64_t>(ay0, height_b - crop_h_b));
    }
    const auto bx1 = bx0 + crop_w_b;
    const auto by1 = by0 + crop_h_b;

    auto warp = warp_crop_full_b.clone();
    warp.index_put_({Slice(), Slice(), 0}, warp.index({Slice(), Slice(), 0}) - static_cast<double>(bx0));
    warp.index_put_({Slice(), Slice(), 1}, warp.index({Slice(), Slice(), 1}) - static_cast<double>(by0));
    valid_crop = valid_crop.logical_and(finite_warp)
                     .logical_and(warp.index({Slice(), Slice(), 0}).ge(0.0))
                     .logical_and(warp.index({Slice(), Slice(), 0}).le(static_cast<double>(crop_w_b - 1)))
                     .logical_and(warp.index({Slice(), Slice(), 1}).ge(0.0))
                     .logical_and(warp.index({Slice(), Slice(), 1}).le(static_cast<double>(crop_h_b - 1)));
    return SyntheticPair{pair.view_a.index({Slice(), Slice(ay0, ay1), Slice(ax0, ax1)}).contiguous(),
                         pair.view_b.index({Slice(), Slice(by0, by1), Slice(bx0, bx1)}).contiguous(),
                         warp.contiguous(), valid_crop.contiguous()};
}

SyntheticPair limit_training_pair_size(const SyntheticPair& pair, int64_t resize)
{
    const auto original_height = pair.view_a.size(1);
    const auto original_width = pair.view_a.size(2);
    const auto original_b_height = pair.view_b.size(1);
    const auto original_b_width = pair.view_b.size(2);
    auto view_a = limit_training_image_size(pair.view_a, resize);
    auto view_b = limit_training_image_size(pair.view_b, resize);
    const auto target_height = view_a.size(1);
    const auto target_width = view_a.size(2);
    const auto target_b_height = view_b.size(1);
    const auto target_b_width = view_b.size(2);
    if (target_height == original_height && target_width == original_width && target_b_height == original_b_height &&
        target_b_width == original_b_width)
    {
        return SyntheticPair{view_a, view_b, pair.warp_a_to_b.contiguous(), pair.valid_mask.contiguous()};
    }

    return SyntheticPair{view_a,
                         view_b,
                         resize_training_warp(pair.warp_a_to_b, target_height, target_width, original_b_height,
                                              original_b_width, target_b_height, target_b_width),
                         resize_training_valid_mask(pair.valid_mask, target_height, target_width)};
}

SyntheticPair prepare_training_pair_size(const SyntheticPair& pair, int64_t training_crop_size, int64_t resize,
                                         std::optional<at::Generator>& generator)
{
    return limit_training_pair_size(crop_training_pair(pair, training_crop_size, generator), resize);
}


torch::Tensor make_descriptor_sample_indices(const torch::Tensor& descriptors)
{
    const auto spatial_count = descriptors.size(2) * descriptors.size(3);
    const auto sample_count = std::min<int64_t>(spatial_count, MAX_DESCRIPTOR_LOSS_SAMPLES);
    const auto sample_options = torch::TensorOptions().dtype(torch::kLong).device(descriptors.device());
    if (sample_count == spatial_count)
    {
        return torch::arange(spatial_count, sample_options);
    }
    return torch::randperm(spatial_count, sample_options).narrow(0, 0, sample_count).contiguous();
}

torch::Tensor sample_spatial_descriptors(const torch::Tensor& descriptors, const torch::Tensor& sample_indices)
{
    const auto batch_size = descriptors.size(0);
    const auto descriptor_dim = descriptors.size(1);
    const auto spatial_count = descriptors.size(2) * descriptors.size(3);
    auto flat = descriptors.permute({0, 2, 3, 1}).reshape({batch_size, spatial_count, descriptor_dim});
    return flat.index_select(1, sample_indices).contiguous();
}

torch::Tensor make_descriptor_target_indices(const torch::Tensor& warp, const torch::Tensor& sample_indices,
                                             int64_t descriptor_height, int64_t descriptor_width)
{
    using torch::indexing::Slice;

    const auto image_height = warp.size(1);
    const auto image_width = warp.size(2);
    auto source_y = torch::floor_divide(sample_indices, descriptor_width).to(torch::kFloat32);
    auto source_x = sample_indices.remainder(descriptor_width).to(torch::kFloat32);
    auto image_x = ((source_x + 0.5F) * static_cast<float>(image_width) / static_cast<float>(descriptor_width) - 0.5F)
                       .round()
                       .clamp(0, image_width - 1);
    auto image_y = ((source_y + 0.5F) * static_cast<float>(image_height) / static_cast<float>(descriptor_height) - 0.5F)
                       .round()
                       .clamp(0, image_height - 1);
    auto flat_image_indices = (image_y * image_width + image_x).to(torch::kLong);
    auto flat_warp = warp.reshape({warp.size(0), image_height * image_width, 2});
    auto sampled_warp = flat_warp.index_select(1, flat_image_indices);
    auto target_x = ((sampled_warp.index({Slice(), Slice(), 0}) + 0.5F) * static_cast<float>(descriptor_width) /
                         static_cast<float>(image_width) -
                     0.5F)
                        .round()
                        .clamp(0, descriptor_width - 1);
    auto target_y = ((sampled_warp.index({Slice(), Slice(), 1}) + 0.5F) * static_cast<float>(descriptor_height) /
                         static_cast<float>(image_height) -
                     0.5F)
                        .round()
                        .clamp(0, descriptor_height - 1);
    return (target_y * descriptor_width + target_x).to(torch::kLong);
}

torch::Tensor make_descriptor_target_coordinates(const torch::Tensor& warp, const torch::Tensor& sample_indices,
                                                 int64_t descriptor_height, int64_t descriptor_width)
{
    using torch::indexing::Slice;

    const auto image_height = warp.size(1);
    const auto image_width = warp.size(2);
    auto source_y = torch::floor_divide(sample_indices, descriptor_width).to(torch::kFloat32);
    auto source_x = sample_indices.remainder(descriptor_width).to(torch::kFloat32);
    auto image_x = ((source_x + 0.5F) * static_cast<float>(image_width) / static_cast<float>(descriptor_width) - 0.5F)
                       .clamp(0, image_width - 1);
    auto image_y = ((source_y + 0.5F) * static_cast<float>(image_height) / static_cast<float>(descriptor_height) - 0.5F)
                       .clamp(0, image_height - 1);

    auto grid_x =
        image_width > 1 ? image_x / static_cast<float>(image_width - 1) * 2.0F - 1.0F : torch::zeros_like(image_x);
    auto grid_y =
        image_height > 1 ? image_y / static_cast<float>(image_height - 1) * 2.0F - 1.0F : torch::zeros_like(image_y);
    auto grid = torch::stack({grid_x, grid_y}, 1)
                    .reshape({1, sample_indices.size(0), 1, 2})
                    .expand({warp.size(0), sample_indices.size(0), 1, 2})
                    .contiguous();
    auto sampled_warp = torch::nn::functional::grid_sample(warp.permute({0, 3, 1, 2}).contiguous(), grid,
                                                           torch::nn::functional::GridSampleFuncOptions()
                                                               .mode(torch::kBilinear)
                                                               .padding_mode(torch::kBorder)
                                                               .align_corners(true))
                            .squeeze(3)
                            .permute({0, 2, 1});
    auto target_x = (sampled_warp.index({Slice(), Slice(), 0}) + 0.5F) * static_cast<float>(descriptor_width) /
                        static_cast<float>(image_width) -
                    0.5F;
    auto target_y = (sampled_warp.index({Slice(), Slice(), 1}) + 0.5F) * static_cast<float>(descriptor_height) /
                        static_cast<float>(image_height) -
                    0.5F;
    target_x = target_x.clamp(0, descriptor_width - 1);
    target_y = target_y.clamp(0, descriptor_height - 1);
    return torch::stack({target_x, target_y}, 2).contiguous();
}

torch::Tensor sample_warped_descriptors(const torch::Tensor& descriptors, const torch::Tensor& target_coordinates)
{
    const auto descriptor_height = descriptors.size(2);
    const auto descriptor_width = descriptors.size(3);
    auto target_x = target_coordinates.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto target_y = target_coordinates.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    auto grid_x = descriptor_width > 1 ? target_x / static_cast<float>(descriptor_width - 1) * 2.0F - 1.0F
                                       : torch::zeros_like(target_x);
    auto grid_y = descriptor_height > 1 ? target_y / static_cast<float>(descriptor_height - 1) * 2.0F - 1.0F
                                        : torch::zeros_like(target_y);
    auto grid = torch::stack({grid_x, grid_y}, 2).unsqueeze(2).contiguous();
    return torch::nn::functional::grid_sample(descriptors, grid,
                                              torch::nn::functional::GridSampleFuncOptions()
                                                  .mode(torch::kBilinear)
                                                  .padding_mode(torch::kBorder)
                                                  .align_corners(true))
        .squeeze(3)
        .permute({0, 2, 1})
        .contiguous();
}

torch::Tensor make_descriptor_valid_mask(const torch::Tensor& valid_mask, int64_t descriptor_height,
                                         int64_t descriptor_width)
{
    auto mask = valid_mask.to(torch::kFloat32).unsqueeze(1);
    auto resized =
        torch::nn::functional::interpolate(mask, torch::nn::functional::InterpolateFuncOptions()
                                                     .size(std::vector<int64_t>{descriptor_height, descriptor_width})
                                                     .mode(torch::kNearest));
    return resized.squeeze(1).reshape({valid_mask.size(0), descriptor_height * descriptor_width}).to(torch::kBool);
}

torch::Tensor filter_descriptor_sample_indices(const torch::Tensor& sample_indices, const torch::Tensor& valid_mask,
                                               int64_t descriptor_height, int64_t descriptor_width)
{
    auto flat = make_descriptor_valid_mask(valid_mask, descriptor_height, descriptor_width);
    auto valid_for_batch = flat.index_select(1, sample_indices).to(torch::kBool).all(0);
    return sample_indices.index({valid_for_batch}).contiguous();
}

torch::Tensor make_balanced_descriptor_sample_indices(const torch::Tensor& valid_mask, int64_t descriptor_height,
                                                      int64_t descriptor_width, int64_t max_samples)
{
    auto mask = valid_mask.to(torch::kFloat32).unsqueeze(1);
    auto resized =
        torch::nn::functional::interpolate(mask, torch::nn::functional::InterpolateFuncOptions()
                                                     .size(std::vector<int64_t>{descriptor_height, descriptor_width})
                                                     .mode(torch::kNearest));
    auto valid = resized.squeeze(1).to(torch::kBool).all(0).contiguous();
    const auto long_options = torch::TensorOptions().dtype(torch::kLong).device(valid.device());
    auto global_indices = torch::arange(descriptor_height * descriptor_width, long_options)
                              .reshape({descriptor_height, descriptor_width});
    const auto cell_count = GRAPH_DENSE_MATCHING_GRID_ROWS * GRAPH_DENSE_MATCHING_GRID_COLS;
    const auto per_cell = std::max<int64_t>(1, (max_samples + cell_count - 1) / cell_count);
    std::vector<torch::Tensor> selected_cells;
    selected_cells.reserve(static_cast<std::size_t>(cell_count));
    for (int64_t row = 0; row < GRAPH_DENSE_MATCHING_GRID_ROWS; ++row)
    {
        const auto y0 = descriptor_height * row / GRAPH_DENSE_MATCHING_GRID_ROWS;
        const auto y1 = descriptor_height * (row + 1) / GRAPH_DENSE_MATCHING_GRID_ROWS;
        for (int64_t col = 0; col < GRAPH_DENSE_MATCHING_GRID_COLS; ++col)
        {
            const auto x0 = descriptor_width * col / GRAPH_DENSE_MATCHING_GRID_COLS;
            const auto x1 = descriptor_width * (col + 1) / GRAPH_DENSE_MATCHING_GRID_COLS;
            auto cell_valid =
                valid.index({torch::indexing::Slice(y0, y1), torch::indexing::Slice(x0, x1)}).reshape({-1});
            auto cell_indices = global_indices.index({torch::indexing::Slice(y0, y1), torch::indexing::Slice(x0, x1)})
                                    .reshape({-1})
                                    .index({cell_valid});
            if (cell_indices.numel() == 0)
            {
                continue;
            }
            if (cell_indices.size(0) > per_cell)
            {
                auto order = torch::randperm(cell_indices.size(0), long_options).narrow(0, 0, per_cell);
                cell_indices = cell_indices.index_select(0, order);
            }
            selected_cells.push_back(cell_indices);
        }
    }
    if (selected_cells.empty())
    {
        return torch::empty({0}, long_options);
    }
    auto selected = torch::cat(selected_cells, 0).contiguous();
    if (selected.size(0) > max_samples)
    {
        selected = selected.index_select(0, torch::randperm(selected.size(0), long_options).narrow(0, 0, max_samples));
    }
    return selected.contiguous();
}

torch::Tensor make_descriptor_candidate_indices(const torch::Tensor& target_indices, int64_t spatial_count,
                                                int64_t descriptor_width,
                                                const torch::Tensor& candidate_valid_mask = torch::Tensor(),
                                                int64_t broad_far_negative_count = DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT)
{
    const auto negative_count = std::min<int64_t>(DESCRIPTOR_NEGATIVE_SAMPLE_COUNT, spatial_count - 1);
    if (negative_count <= 0)
    {
        return target_indices.unsqueeze(2).contiguous();
    }
    const auto long_options = torch::TensorOptions().dtype(torch::kLong).device(target_indices.device());
    auto offsets = torch::arange(1, spatial_count, long_options).reshape({1, 1, spatial_count - 1});
    auto all_negatives = (target_indices.unsqueeze(2) + offsets).remainder(spatial_count);
    auto negatives = all_negatives.narrow(2, 0, negative_count);
    if (descriptor_width > 0 && spatial_count % descriptor_width == 0 && spatial_count >= 32)
    {
        constexpr int64_t exclusion_radius = 2;
        constexpr int64_t near_ring_radius = 6;
        constexpr int64_t max_excluded_nearby = (exclusion_radius * 2 + 1) * (exclusion_radius * 2 + 1) - 1;
        const auto far_negative_count =
            std::min<int64_t>(negative_count, std::max<int64_t>(1, spatial_count - 1 - max_excluded_nearby));
        const auto descriptor_height = spatial_count / descriptor_width;
        const auto effective_broad_far_negative_count =
            std::clamp<int64_t>(broad_far_negative_count, 0, DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT);
        auto gcd = [](int64_t lhs, int64_t rhs)
        {
            lhs = lhs < 0 ? -lhs : lhs;
            rhs = rhs < 0 ? -rhs : rhs;
            while (rhs != 0)
            {
                const auto next = lhs % rhs;
                lhs = rhs;
                rhs = next;
            }
            return std::max<int64_t>(1, lhs);
        };
        auto broad_stride =
            std::max<int64_t>(1, static_cast<int64_t>(std::floor(static_cast<double>(spatial_count) * 0.61803398875)));
        while (broad_stride < spatial_count && gcd(broad_stride, spatial_count) != 1)
        {
            ++broad_stride;
        }
        if (broad_stride >= spatial_count)
        {
            broad_stride = 1;
        }
        auto targets_cpu = target_indices.to(torch::kCPU).contiguous();
        auto valid_cpu = candidate_valid_mask.defined()
                             ? candidate_valid_mask.to(torch::kCPU, torch::kBool).contiguous()
                             : torch::Tensor();
        auto targets = targets_cpu.accessor<int64_t, 2>();
        const bool has_valid_mask = valid_cpu.defined() && valid_cpu.dim() == 2 &&
                                    valid_cpu.size(0) == target_indices.size(0) && valid_cpu.size(1) == spatial_count;
        const auto* valid_data = has_valid_mask ? valid_cpu.data_ptr<bool>() : nullptr;
        std::vector<std::vector<int64_t>> rows;
        rows.reserve(static_cast<size_t>(target_indices.numel()));
        int64_t effective_negative_count = far_negative_count;
        for (int64_t batch = 0; batch < target_indices.size(0); ++batch)
        {
            for (int64_t query = 0; query < target_indices.size(1); ++query)
            {
                const auto target = targets[batch][query];
                const auto target_x = target % descriptor_width;
                const auto target_y = target / descriptor_width;
                std::vector<int64_t> row_indices;
                row_indices.reserve(static_cast<size_t>(far_negative_count));
                auto add_candidate = [&](int64_t candidate)
                {
                    if (candidate < 0 || candidate >= spatial_count || candidate == target)
                    {
                        return;
                    }
                    if (std::find(row_indices.begin(), row_indices.end(), candidate) != row_indices.end())
                    {
                        return;
                    }
                    if (has_valid_mask && !valid_data[batch * spatial_count + candidate])
                    {
                        return;
                    }
                    const auto candidate_x = candidate % descriptor_width;
                    const auto candidate_y = candidate / descriptor_width;
                    const auto dx = candidate_x - target_x;
                    const auto dy = candidate_y - target_y;
                    if (dx * dx + dy * dy <= exclusion_radius * exclusion_radius)
                    {
                        return;
                    }
                    row_indices.push_back(candidate);
                };
                for (int64_t radius = exclusion_radius + 1;
                     radius <= near_ring_radius && static_cast<int64_t>(row_indices.size()) < far_negative_count;
                     ++radius)
                {
                    for (int64_t dy = -radius;
                         dy <= radius && static_cast<int64_t>(row_indices.size()) < far_negative_count; ++dy)
                    {
                        for (int64_t dx = -radius;
                             dx <= radius && static_cast<int64_t>(row_indices.size()) < far_negative_count; ++dx)
                        {
                            const auto abs_dx = dx < 0 ? -dx : dx;
                            const auto abs_dy = dy < 0 ? -dy : dy;
                            if (std::max(abs_dx, abs_dy) != radius)
                            {
                                continue;
                            }
                            const auto distance_sq = dx * dx + dy * dy;
                            if (distance_sq <= exclusion_radius * exclusion_radius ||
                                distance_sq > near_ring_radius * near_ring_radius)
                            {
                                continue;
                            }
                            const auto candidate_x = target_x + dx;
                            const auto candidate_y = target_y + dy;
                            if (candidate_x < 0 || candidate_x >= descriptor_width || candidate_y < 0 ||
                                candidate_y >= descriptor_height)
                            {
                                continue;
                            }
                            add_candidate(candidate_y * descriptor_width + candidate_x);
                        }
                    }
                }
                const auto broad_start_count = static_cast<int64_t>(row_indices.size());
                const bool restrict_global_fill =
                    effective_broad_far_negative_count < DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT;
                const auto row_negative_limit =
                    restrict_global_fill
                        ? std::min<int64_t>(far_negative_count, broad_start_count + effective_broad_far_negative_count)
                        : far_negative_count;
                for (int64_t step = 1;
                     step < spatial_count && static_cast<int64_t>(row_indices.size()) < row_negative_limit &&
                     static_cast<int64_t>(row_indices.size()) < broad_start_count + effective_broad_far_negative_count;
                     ++step)
                {
                    add_candidate((target + step * broad_stride) % spatial_count);
                }
                for (int64_t candidate = 0;
                     candidate < spatial_count && static_cast<int64_t>(row_indices.size()) < row_negative_limit;
                     ++candidate)
                {
                    add_candidate(candidate);
                }
                for (int64_t offset = 1;
                     !has_valid_mask && static_cast<int64_t>(row_indices.size()) < row_negative_limit; ++offset)
                {
                    const auto candidate = (target + offset) % spatial_count;
                    if (candidate == target ||
                        std::find(row_indices.begin(), row_indices.end(), candidate) != row_indices.end())
                    {
                        continue;
                    }
                    row_indices.push_back(candidate);
                }
                effective_negative_count =
                    std::min<int64_t>(effective_negative_count, static_cast<int64_t>(row_indices.size()));
                rows.push_back(std::move(row_indices));
            }
        }
        if (effective_negative_count <= 0)
        {
            negatives = torch::empty({target_indices.size(0), target_indices.size(1), 0},
                                     torch::TensorOptions().dtype(torch::kLong).device(target_indices.device()));
            return torch::cat({target_indices.unsqueeze(2), negatives}, 2).contiguous();
        }
        std::vector<int64_t> far_indices;
        far_indices.reserve(static_cast<size_t>(target_indices.numel() * effective_negative_count));
        for (const auto& row : rows)
        {
            far_indices.insert(far_indices.end(), row.begin(), row.begin() + effective_negative_count);
        }
        negatives = torch::from_blob(far_indices.data(),
                                     {target_indices.size(0), target_indices.size(1), effective_negative_count},
                                     torch::TensorOptions().dtype(torch::kLong))
                        .clone()
                        .to(target_indices.device());
    }
    return torch::cat({target_indices.unsqueeze(2), negatives}, 2).contiguous();
}

torch::Tensor gather_descriptor_candidates(const torch::Tensor& descriptors, const torch::Tensor& candidate_indices)
{
    const auto batch_size = descriptors.size(0);
    const auto query_count = candidate_indices.size(1);
    const auto candidate_count = candidate_indices.size(2);
    const auto descriptor_dim = descriptors.size(1);
    const auto spatial_count = descriptors.size(2) * descriptors.size(3);
    auto flat = descriptors.permute({0, 2, 3, 1}).reshape({batch_size, spatial_count, descriptor_dim});
    auto expanded_indices = candidate_indices.reshape({batch_size, query_count * candidate_count, 1})
                                .expand({batch_size, query_count * candidate_count, descriptor_dim});
    return flat.gather(1, expanded_indices)
        .reshape({batch_size, query_count, candidate_count, descriptor_dim})
        .contiguous();
}

torch::Tensor descriptor_pair_similarity_scores(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b)
{
    return cyclicDescriptorSimilarityScoresChunked(descriptors_a, descriptors_b, DESCRIPTOR_SIMILARITY_QUERY_CHUNK);
}

torch::Tensor descriptor_candidate_similarity_scores(const torch::Tensor& descriptors_a,
                                                     const torch::Tensor& candidate_descriptors)
{
    if (descriptors_a.dim() != 3 || candidate_descriptors.dim() != 4 ||
        descriptors_a.size(0) != candidate_descriptors.size(0) ||
        descriptors_a.size(1) != candidate_descriptors.size(1) ||
        descriptors_a.size(2) != candidate_descriptors.size(3))
    {
        throw std::invalid_argument("descriptor candidate similarity inputs must have shape BxNxD and BxNxKxD");
    }
    const auto batch_size = descriptors_a.size(0);
    const auto query_count = descriptors_a.size(1);
    const auto candidate_count = candidate_descriptors.size(2);
    const auto descriptor_dim = descriptors_a.size(2);
    auto queries = descriptors_a.reshape({batch_size * query_count, 1, descriptor_dim});
    auto candidates = candidate_descriptors.reshape({batch_size * query_count, candidate_count, descriptor_dim});
    auto normalized_queries = queries / queries.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    auto normalized_candidates = candidates / candidates.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    return torch::bmm(normalized_queries, normalized_candidates.transpose(1, 2))
        .reshape({batch_size, query_count, candidate_count});
}

torch::Tensor make_strict_descriptor_cross_entropy_loss(const torch::Tensor& descriptors_a,
                                                        const torch::Tensor& descriptors_b,
                                                        const torch::Tensor& target_indices)
{
    if (descriptors_a.dim() != 3 || descriptors_b.dim() != 3 || descriptors_a.size(0) != descriptors_b.size(0) ||
        descriptors_a.size(2) != descriptors_b.size(2) || target_indices.dim() != 2 ||
        target_indices.size(0) != descriptors_a.size(0) || target_indices.size(1) != descriptors_a.size(1) ||
        target_indices.dtype() != torch::kLong)
    {
        throw std::invalid_argument("strict descriptor CE inputs must have BxNxD, BxMxD, and BxN long shapes");
    }
    const auto candidate_count = descriptors_b.size(1);
    if (target_indices.numel() > 0 &&
        (target_indices.lt(0).any().item<bool>() || target_indices.ge(candidate_count).any().item<bool>()))
    {
        throw std::invalid_argument("strict descriptor CE target indices are out of range");
    }
    auto normalized_a = descriptors_a / descriptors_a.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    auto normalized_b = descriptors_b / descriptors_b.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    auto logits = torch::bmm(normalized_a, normalized_b.transpose(1, 2)) * SUPERVISED_DESCRIPTOR_RANKING_LOGIT_SCALE;
    return torch::nn::functional::cross_entropy(
        logits.reshape({descriptors_a.size(0) * descriptors_a.size(1), candidate_count}),
        target_indices.reshape({descriptors_a.size(0) * descriptors_a.size(1)}));
}

torch::Tensor make_descriptor_map_hard_negative_loss(const torch::Tensor& query_descriptors,
                                                     const torch::Tensor& candidate_map_descriptors,
                                                     const torch::Tensor& target_indices)
{
    if (query_descriptors.dim() != 3 || candidate_map_descriptors.dim() != 4 || target_indices.dim() != 2 ||
        target_indices.size(0) != query_descriptors.size(0) || target_indices.size(1) != query_descriptors.size(1) ||
        candidate_map_descriptors.size(0) != query_descriptors.size(0) ||
        candidate_map_descriptors.size(1) != query_descriptors.size(2))
    {
        throw std::invalid_argument(
            "descriptor map hard negative inputs must have BxNxD, BxDxHxW, and BxN target shapes");
    }
    const auto spatial_count = candidate_map_descriptors.size(2) * candidate_map_descriptors.size(3);
    if (query_descriptors.size(1) == 0 || spatial_count < 2)
    {
        return torch::zeros({}, query_descriptors.options());
    }

    const auto batch_size = candidate_map_descriptors.size(0);
    const auto descriptor_dim = candidate_map_descriptors.size(1);
    const auto descriptor_width = candidate_map_descriptors.size(3);
    const auto flat_candidates =
        candidate_map_descriptors.permute({0, 2, 3, 1}).reshape({batch_size, spatial_count, descriptor_dim});
    auto scores = descriptor_pair_similarity_scores(query_descriptors, flat_candidates);
    auto positive_scores = scores.gather(2, target_indices.unsqueeze(2)).squeeze(2);

    auto spatial_indices = torch::arange(spatial_count, target_indices.options());
    auto candidate_x = spatial_indices.remainder(descriptor_width).reshape({1, 1, spatial_count});
    auto candidate_y = torch::floor_divide(spatial_indices, descriptor_width).reshape({1, 1, spatial_count});
    auto target_x = target_indices.remainder(descriptor_width).unsqueeze(2);
    auto target_y = torch::floor_divide(target_indices, descriptor_width).unsqueeze(2);
    const auto exclusion_radius_sq =
        DENSE_DESCRIPTOR_HARD_NEGATIVE_EXCLUSION_RADIUS * DENSE_DESCRIPTOR_HARD_NEGATIVE_EXCLUSION_RADIUS;
    auto dx = (candidate_x - target_x).to(torch::kFloat32);
    auto dy = (candidate_y - target_y).to(torch::kFloat32);
    auto near_positive = (dx.pow(2) + dy.pow(2)).le(exclusion_radius_sq);

    auto negative_scores = scores.masked_fill(near_positive, -std::numeric_limits<float>::infinity());
    const auto hard_count = std::min<int64_t>(DENSE_DESCRIPTOR_TOPK_NEGATIVES, negative_scores.size(2));
    auto hard_negatives = std::get<0>(negative_scores.topk(hard_count, 2));
    auto finite_mask = torch::isfinite(hard_negatives);
    if (!finite_mask.any().item<bool>())
    {
        return torch::zeros({}, query_descriptors.options());
    }
    auto margin = torch::relu(hard_negatives - positive_scores.unsqueeze(2) + DENSE_DESCRIPTOR_HARD_NEGATIVE_MARGIN);
    auto finite_margin = margin.index({finite_mask});
    auto strongest_margin = std::get<0>(margin.masked_fill(~finite_mask, 0.0F).max(2)).mean();
    return strongest_margin * 0.5 + finite_margin.mean() * 0.5;
}

torch::Tensor make_dense_descriptor_hard_negative_loss(const torch::Tensor& descriptors_a,
                                                       const torch::Tensor& descriptors_b,
                                                       const torch::Tensor& sample_indices,
                                                       const torch::Tensor& target_indices)
{
    if (sample_indices.numel() == 0)
    {
        return torch::zeros({}, descriptors_a.options());
    }
    const auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    return make_descriptor_map_hard_negative_loss(sampled_a, descriptors_b, target_indices);
}

torch::Tensor make_bidirectional_dense_descriptor_hard_negative_loss(const torch::Tensor& descriptors_a,
                                                                     const torch::Tensor& descriptors_b,
                                                                     const torch::Tensor& sample_indices,
                                                                     const torch::Tensor& target_indices)
{
    if (sample_indices.numel() == 0)
    {
        return torch::zeros({}, descriptors_a.options());
    }
    auto forward =
        make_dense_descriptor_hard_negative_loss(descriptors_a, descriptors_b, sample_indices, target_indices);
    auto target_candidate_indices = target_indices.unsqueeze(2);
    auto positive_b = gather_descriptor_candidates(descriptors_b, target_candidate_indices).squeeze(2);
    auto source_indices =
        sample_indices.to(target_indices.device(), torch::kLong).unsqueeze(0).expand_as(target_indices).contiguous();
    auto reverse = make_descriptor_map_hard_negative_loss(positive_b, descriptors_a, source_indices);
    return forward + reverse * DENSE_DESCRIPTOR_REVERSE_HARD_NEGATIVE_WEIGHT;
}

torch::Tensor make_descriptor_finetune_anchor_loss(const torch::Tensor& current_a, const torch::Tensor& current_b,
                                                   const torch::Tensor& anchor_a, const torch::Tensor& anchor_b,
                                                   const torch::Tensor& valid_mask)
{
    if (current_a.dim() != 4 || current_b.dim() != 4 || anchor_a.sizes() != current_a.sizes() ||
        anchor_b.sizes() != current_b.sizes() || current_a.sizes() != current_b.sizes() || valid_mask.dim() != 3 ||
        valid_mask.size(0) != current_a.size(0))
    {
        throw std::invalid_argument(
            "descriptor finetune anchor inputs must have matching BxDxHxW descriptors and BxHxW mask");
    }
    auto mask = valid_mask.to(current_a.device(), torch::kFloat32).unsqueeze(1);
    if (mask.size(2) != current_a.size(2) || mask.size(3) != current_a.size(3))
    {
        mask = torch::nn::functional::interpolate(mask,
                                                  torch::nn::functional::InterpolateFuncOptions()
                                                      .size(std::vector<int64_t>{current_a.size(2), current_a.size(3)})
                                                      .mode(torch::kNearest));
    }
    mask = mask.gt(0.0F).to(current_a.dtype());
    const auto denom = mask.sum() * 2.0;
    if (denom.item<double>() <= 0.0)
    {
        return torch::zeros({}, current_a.options());
    }
    auto normalize = [](const torch::Tensor& descriptors)
    {
        return torch::nn::functional::normalize(descriptors,
                                                torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12));
    };
    const auto current_a_normalized = normalize(current_a);
    const auto current_b_normalized = normalize(current_b);
    const auto anchor_a_normalized = normalize(anchor_a.to(current_a.device(), current_a.dtype()));
    const auto anchor_b_normalized = normalize(anchor_b.to(current_b.device(), current_b.dtype()));
    const auto drift_a = 1.0F - (current_a_normalized * anchor_a_normalized).sum(1, true);
    const auto drift_b = 1.0F - (current_b_normalized * anchor_b_normalized).sum(1, true);
    return ((drift_a + drift_b) * mask).sum() / denom.clamp_min(1.0);
}

torch::Tensor make_warp_descriptor_contrastive_loss(const torch::Tensor& descriptors_a,
                                                    const torch::Tensor& descriptors_b, const torch::Tensor& warp,
                                                    const torch::Tensor& valid_mask)
{
    auto sample_indices = make_balanced_descriptor_sample_indices(valid_mask, descriptors_a.size(2),
                                                                  descriptors_a.size(3), MAX_DESCRIPTOR_LOSS_SAMPLES);
    if (sample_indices.numel() < 2)
    {
        return torch::zeros({}, descriptors_a.options());
    }

    const auto batch_size = descriptors_a.size(0);
    const auto sample_count = sample_indices.size(0);
    const auto descriptor_dim = descriptors_b.size(1);
    const auto spatial_count_b = descriptors_b.size(2) * descriptors_b.size(3);
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    (void)descriptor_dim;
    (void)spatial_count_b;
    auto target_coordinates =
        make_descriptor_target_coordinates(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    auto positive_b = sample_warped_descriptors(descriptors_b, target_coordinates);
    auto targets =
        torch::arange(sample_count, torch::TensorOptions().dtype(torch::kLong).device(descriptors_a.device()))
            .unsqueeze(0)
            .expand({batch_size, sample_count});

    auto logits_ab = descriptor_pair_similarity_scores(sampled_a, positive_b) * 20.0F;
    auto logits_ba = descriptor_pair_similarity_scores(positive_b, sampled_a) * 20.0F;
    auto loss_ab = torch::nn::functional::cross_entropy(logits_ab.reshape({batch_size * sample_count, sample_count}),
                                                        targets.reshape({batch_size * sample_count}));
    auto loss_ba = torch::nn::functional::cross_entropy(logits_ba.reshape({batch_size * sample_count, sample_count}),
                                                        targets.reshape({batch_size * sample_count}));
    return (loss_ab + loss_ba) * 0.5;
}

torch::Tensor make_direct_full_map_descriptor_loss(const torch::Tensor& descriptors_a,
                                                   const torch::Tensor& descriptors_b, const torch::Tensor& warp,
                                                   const torch::Tensor& valid_mask)
{
    auto sample_indices = make_balanced_descriptor_sample_indices(valid_mask, descriptors_a.size(2),
                                                                  descriptors_a.size(3), MAX_DESCRIPTOR_LOSS_SAMPLES);
    const auto spatial_count_b = descriptors_b.size(2) * descriptors_b.size(3);
    if (sample_indices.numel() == 0 || spatial_count_b < 2)
    {
        return torch::zeros({}, descriptors_a.options());
    }

    const auto batch_size = descriptors_a.size(0);
    const auto sample_count = sample_indices.size(0);
    const auto descriptor_dim = descriptors_b.size(1);
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    auto target_indices =
        make_descriptor_target_indices(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    auto flat_b = descriptors_b.permute({0, 2, 3, 1}).reshape({batch_size, spatial_count_b, descriptor_dim});
    auto logits = descriptor_pair_similarity_scores(sampled_a, flat_b) * 20.0F;
    return torch::nn::functional::cross_entropy(logits.reshape({batch_size * sample_count, spatial_count_b}),
                                                target_indices.reshape({batch_size * sample_count}));
}

torch::Tensor make_cross_batch_descriptor_contrastive_loss(const torch::Tensor& sampled_a,
                                                           const torch::Tensor& positive_b)
{
    if (sampled_a.dim() != 3 || positive_b.dim() != 3 || !sampled_a.sizes().equals(positive_b.sizes()))
    {
        throw std::invalid_argument("sampled_a and positive_b must have matching BxNxD descriptor shapes");
    }
    const auto batch_size = sampled_a.size(0);
    const auto sample_count = sampled_a.size(1);
    const auto descriptor_dim = sampled_a.size(2);
    const auto total_count = batch_size * sample_count;
    if (total_count < 2)
    {
        return torch::zeros({}, sampled_a.options());
    }

    auto queries = sampled_a.reshape({total_count, descriptor_dim});
    auto keys = positive_b.reshape({total_count, descriptor_dim});
    queries = queries / queries.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
    keys = keys / keys.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
    auto logits = queries.matmul(keys.transpose(0, 1)) * CROSS_BATCH_DESCRIPTOR_LOGIT_SCALE;
    auto targets = torch::arange(total_count, torch::TensorOptions().dtype(torch::kLong).device(sampled_a.device()));
    auto loss_ab = torch::nn::functional::cross_entropy(logits, targets);
    auto loss_ba = torch::nn::functional::cross_entropy(logits.transpose(0, 1), targets);
    return (loss_ab + loss_ba) * 0.5;
}

torch::Tensor make_positive_descriptor_alignment_loss(const torch::Tensor& sampled_a, const torch::Tensor& positive_b)
{
    if (sampled_a.dim() != 3 || positive_b.dim() != 3 || !sampled_a.sizes().equals(positive_b.sizes()))
    {
        throw std::invalid_argument("sampled_a and positive_b must have matching BxNxD descriptor shapes");
    }
    auto normalized_a = sampled_a / sampled_a.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    auto normalized_b = positive_b / positive_b.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    return (1.0F - (normalized_a * normalized_b).sum(2)).mean();
}

torch::Tensor make_patch_descriptor_alignment_loss(const torch::Tensor& descriptors_a,
                                                   const torch::Tensor& descriptors_b, const torch::Tensor& warp,
                                                   const torch::Tensor& valid_mask)
{
    auto sample_indices = make_balanced_descriptor_sample_indices(valid_mask, descriptors_a.size(2),
                                                                  descriptors_a.size(3), MAX_DESCRIPTOR_LOSS_SAMPLES);
    if (sample_indices.numel() == 0)
    {
        return torch::zeros({}, descriptors_a.options());
    }

    auto target_coordinates =
        make_descriptor_target_coordinates(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    auto make_scale_loss = [&](int64_t kernel_size)
    {
        const auto padding = kernel_size / 2;
        auto patch_a =
            torch::avg_pool2d(descriptors_a, {kernel_size, kernel_size}, {1, 1}, {padding, padding}, false, true);
        auto patch_b =
            torch::avg_pool2d(descriptors_b, {kernel_size, kernel_size}, {1, 1}, {padding, padding}, false, true);
        patch_a = patch_a / patch_a.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
        patch_b = patch_b / patch_b.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
        auto sampled_a = sample_spatial_descriptors(patch_a, sample_indices);
        auto positive_b = sample_warped_descriptors(patch_b, target_coordinates);
        return make_positive_descriptor_alignment_loss(sampled_a, positive_b);
    };
    return make_scale_loss(3) * 0.5 + make_scale_loss(5) * 0.5;
}

torch::Tensor make_supervised_descriptor_ranking_loss(const torch::Tensor& sampled_a, const torch::Tensor& candidate_b)
{
    if (sampled_a.dim() != 3 || candidate_b.dim() != 4 || sampled_a.size(0) != candidate_b.size(0) ||
        sampled_a.size(1) != candidate_b.size(1) || sampled_a.size(2) != candidate_b.size(3))
    {
        throw std::invalid_argument("sampled_a and candidate_b must have matching BxNxD and BxNxKxD shapes");
    }
    if (candidate_b.size(2) < 2)
    {
        return torch::zeros({}, sampled_a.options());
    }

    auto scores = descriptor_candidate_similarity_scores(sampled_a, candidate_b);
    auto positive_scores = scores.narrow(2, 0, 1);
    auto negative_scores = scores.narrow(2, 1, scores.size(2) - 1);
    auto margin = torch::relu(negative_scores - positive_scores + SUPERVISED_DESCRIPTOR_RANKING_MARGIN);
    const auto hard_count = std::min<int64_t>(SUPERVISED_DESCRIPTOR_TOPK_NEGATIVES, margin.size(2));
    auto hard_negatives = std::get<0>(margin.topk(hard_count, 2)).mean();
    auto all_negatives = margin.mean();
    auto targets = torch::zeros({sampled_a.size(0), sampled_a.size(1)},
                                torch::TensorOptions().dtype(torch::kLong).device(sampled_a.device()));
    auto ranking_ce =
        torch::nn::functional::cross_entropy((scores * SUPERVISED_DESCRIPTOR_RANKING_LOGIT_SCALE)
                                                 .reshape({sampled_a.size(0) * sampled_a.size(1), scores.size(2)}),
                                             targets.reshape({sampled_a.size(0) * sampled_a.size(1)}));
    auto soft_rank =
        1.0F + torch::sigmoid((negative_scores - positive_scores) * SUPERVISED_DESCRIPTOR_SOFT_RANK_SCALE).sum(2);
    auto soft_rank_loss = torch::log(soft_rank).mean();
    auto tail_rank_loss = torch::log1p(torch::relu(soft_rank - SUPERVISED_DESCRIPTOR_TAIL_RANK_START)).mean();
    return ranking_ce * 0.4 + hard_negatives * 0.45 + all_negatives * 0.15 +
           soft_rank_loss * SUPERVISED_DESCRIPTOR_SOFT_RANK_WEIGHT +
           tail_rank_loss * SUPERVISED_DESCRIPTOR_TAIL_RANK_WEIGHT;
}

torch::Tensor make_descriptor_map_regularization_loss(const torch::Tensor& descriptors)
{
    if (!descriptors.defined() || descriptors.dim() != 4)
    {
        throw std::invalid_argument("descriptors must have shape BxCxHxW");
    }
    const auto channel_count = descriptors.size(1);
    const auto sample_count = descriptors.size(0) * descriptors.size(2) * descriptors.size(3);
    if (channel_count < 2 || sample_count < 2)
    {
        return torch::zeros({}, descriptors.options());
    }

    auto flat = descriptors.permute({0, 2, 3, 1}).reshape({sample_count, channel_count});
    flat = flat / flat.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
    auto centered = flat - flat.mean(0, true);
    auto stddev = centered.pow(2).mean(0).add(1.0e-4).sqrt();
    auto variance_loss = torch::relu(DESCRIPTOR_MAP_STD_TARGET - stddev).mean();

    auto covariance = centered.transpose(0, 1).matmul(centered) / static_cast<float>(sample_count - 1);
    auto eye = torch::eye(channel_count, descriptors.options()).to(torch::kBool);
    auto covariance_loss = covariance.pow(2).masked_select(eye.logical_not()).mean();

    const auto uniformity_samples = std::min<int64_t>(sample_count, DESCRIPTOR_MAP_UNIFORMITY_MAX_SAMPLES);
    auto uniformity_flat = flat;
    if (sample_count > uniformity_samples)
    {
        auto indices = torch::linspace(0, sample_count - 1, uniformity_samples,
                                       torch::TensorOptions().dtype(torch::kFloat32).device(descriptors.device()))
                           .round()
                           .to(torch::kLong);
        uniformity_flat = flat.index_select(0, indices);
    }
    auto similarity = uniformity_flat.matmul(uniformity_flat.transpose(0, 1));
    auto sample_eye = torch::eye(uniformity_flat.size(0), descriptors.options()).to(torch::kBool);
    auto uniformity_loss = similarity.pow(2).masked_select(sample_eye.logical_not()).mean();
    return variance_loss + covariance_loss * DESCRIPTOR_MAP_COVARIANCE_WEIGHT +
           uniformity_loss * DESCRIPTOR_MAP_UNIFORMITY_WEIGHT;
}

torch::Tensor make_sampled_descriptor_decorrelation_loss(const torch::Tensor& sampled_descriptors,
                                                         const torch::Tensor& sample_indices, int64_t descriptor_width)
{
    if (sampled_descriptors.dim() != 3 || sample_indices.dim() != 1 ||
        sampled_descriptors.size(1) != sample_indices.size(0))
    {
        throw std::invalid_argument(
            "sampled descriptor decorrelation inputs must have BxNxD descriptors and N indices");
    }
    const auto sample_count = sampled_descriptors.size(1);
    if (sample_count < 2 || descriptor_width <= 0)
    {
        return torch::zeros({}, sampled_descriptors.options());
    }
    auto normalized = sampled_descriptors / sampled_descriptors.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    auto scores = torch::bmm(normalized, normalized.transpose(1, 2));

    auto x = sample_indices.remainder(descriptor_width);
    auto y = torch::floor_divide(sample_indices, descriptor_width);
    auto dx = (x.reshape({sample_count, 1}) - x.reshape({1, sample_count})).abs();
    auto dy = (y.reshape({sample_count, 1}) - y.reshape({1, sample_count})).abs();
    auto nearby = dx.le(SAMPLED_DESCRIPTOR_DECORRELATION_EXCLUSION_RADIUS)
                      .logical_and(dy.le(SAMPLED_DESCRIPTOR_DECORRELATION_EXCLUSION_RADIUS));
    auto mask = nearby.unsqueeze(0).expand({sampled_descriptors.size(0), sample_count, sample_count});
    auto negatives = scores.masked_fill(mask, -std::numeric_limits<float>::infinity());
    const auto hard_count = std::min<int64_t>(SAMPLED_DESCRIPTOR_DECORRELATION_TOPK, sample_count);
    auto hard_scores = std::get<0>(negatives.topk(hard_count, 2));
    auto finite = torch::isfinite(hard_scores);
    if (!finite.any().item<bool>())
    {
        return torch::zeros({}, sampled_descriptors.options());
    }
    auto margin = torch::relu(hard_scores - SAMPLED_DESCRIPTOR_DECORRELATION_MARGIN);
    return margin.index({finite}).mean();
}

torch::Tensor make_rotation_invariant_texture_target(const torch::Tensor& image, int64_t descriptor_height,
                                                     int64_t descriptor_width, int64_t descriptor_dim)
{
    auto base = image;
    if (base.size(1) != 1)
    {
        base = base.mean(1, true);
    }

    std::vector<torch::Tensor> channels;
    channels.reserve(40);
    channels.push_back(base);
    const auto height = base.size(2);
    const auto width = base.size(3);
    const auto options = base.options();
    auto y = torch::arange(height, options).view({1, 1, height, 1});
    auto x = torch::arange(width, options).view({1, 1, 1, width});
    const auto center_y = (static_cast<double>(height) - 1.0) * 0.5;
    const auto center_x = (static_cast<double>(width) - 1.0) * 0.5;
    const auto max_radius = std::max(1.0, std::hypot(center_x, center_y));
    auto radius = torch::sqrt((x - center_x).pow(2) + (y - center_y).pow(2)) / max_radius;
    radius = radius.expand({base.size(0), 1, height, width}).contiguous();
    channels.push_back(radius);
    channels.push_back(radius.pow(2));
    channels.push_back(base * radius);
    for (const int64_t kernel : {3, 7, 15, 31})
    {
        auto blur = torch::avg_pool2d(base, {kernel, kernel}, {1, 1}, {kernel / 2, kernel / 2}, false, true);
        channels.push_back(blur);
        channels.push_back((base - blur).abs());
    }
    auto dx = (base - torch::roll(base, {1}, {3})).abs();
    auto dy = (base - torch::roll(base, {1}, {2})).abs();
    auto gradient = dx + dy;
    for (const int64_t kernel : {3, 7, 11})
    {
        channels.push_back(
            torch::avg_pool2d(gradient, {kernel, kernel}, {1, 1}, {kernel / 2, kernel / 2}, false, true));
    }
    for (const int64_t ring_radius : {1, 2, 4, 8})
    {
        std::vector<torch::Tensor> diffs;
        diffs.reserve(8);
        for (const auto& offset : std::vector<std::pair<int64_t, int64_t>>{{-ring_radius, 0},
                                                                           {ring_radius, 0},
                                                                           {0, -ring_radius},
                                                                           {0, ring_radius},
                                                                           {-ring_radius, -ring_radius},
                                                                           {-ring_radius, ring_radius},
                                                                           {ring_radius, -ring_radius},
                                                                           {ring_radius, ring_radius}})
        {
            diffs.push_back((base - torch::roll(base, {offset.first, offset.second}, {2, 3})).abs());
        }
        auto ring = torch::stack(diffs, 1);
        channels.push_back(ring.mean(1));
        channels.push_back(std::get<0>(ring.max(1)));
        auto centered_ring = ring - ring.mean(1, true);
        channels.push_back(centered_ring.pow(2).mean(1).sqrt());
        channels.push_back(ring.mean(1) * radius);
    }
    channels.push_back(gradient * radius);
    auto target = torch::cat(channels, 1);
    target =
        torch::nn::functional::interpolate(target, torch::nn::functional::InterpolateFuncOptions()
                                                       .size(std::vector<int64_t>{descriptor_height, descriptor_width})
                                                       .mode(torch::kBilinear)
                                                       .align_corners(false));
    auto centered = target - target.mean({2, 3}, true);
    auto scaled = centered / centered.pow(2).mean({2, 3}, true).add(1.0e-4).sqrt();
    if (scaled.size(1) < descriptor_dim)
    {
        const auto repeat_count = (descriptor_dim + scaled.size(1) - 1) / scaled.size(1);
        scaled = scaled.repeat({1, repeat_count, 1, 1});
    }
    target = scaled.narrow(1, 0, descriptor_dim).contiguous();
    return target / target.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
}

torch::Tensor make_texture_target_descriptor_loss(const torch::Tensor& descriptors, const torch::Tensor& image,
                                                  const torch::Tensor& valid_mask)
{
    auto target =
        make_rotation_invariant_texture_target(image, descriptors.size(2), descriptors.size(3), descriptors.size(1));
    auto mask =
        torch::nn::functional::interpolate(valid_mask.to(descriptors.dtype()).unsqueeze(1),
                                           torch::nn::functional::InterpolateFuncOptions()
                                               .size(std::vector<int64_t>{descriptors.size(2), descriptors.size(3)})
                                               .mode(torch::kNearest));
    auto similarity = (descriptors * target).sum(1, true);
    auto denom = mask.sum().clamp_min(1.0F);
    return ((1.0F - similarity) * mask).sum() / denom;
}

torch::Tensor blend_rotation_invariant_texture_descriptor(const torch::Tensor& descriptors, const torch::Tensor& image)
{
    auto target =
        make_rotation_invariant_texture_target(image, descriptors.size(2), descriptors.size(3), descriptors.size(1));
    auto blended = descriptors + target * ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT;
    return blended / blended.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
}

torch::Tensor canonicalize_descriptor_map_by_orientation(const torch::Tensor& descriptors,
                                                         const torch::Tensor& orientation)
{
    if (!descriptors.defined() || !orientation.defined() || descriptors.dim() != 4 || orientation.dim() != 4 ||
        descriptors.size(1) < 4 || descriptors.size(1) % 4 != 0 || orientation.size(0) != descriptors.size(0) ||
        orientation.size(1) < 2)
    {
        return descriptors;
    }

    auto orientation_map = orientation.to(descriptors.device(), torch::kFloat32);
    if (orientation_map.size(2) != descriptors.size(2) || orientation_map.size(3) != descriptors.size(3))
    {
        orientation_map = torch::nn::functional::interpolate(
            orientation_map, torch::nn::functional::InterpolateFuncOptions()
                                 .size(std::vector<int64_t>{descriptors.size(2), descriptors.size(3)})
                                 .mode(torch::kBilinear)
                                 .align_corners(false));
    }

    auto axis_x = orientation_map.narrow(1, 0, 1);
    auto axis_y = orientation_map.narrow(1, 1, 1);
    auto axis_norm = (axis_x * axis_x + axis_y * axis_y).sqrt();
    auto turns = torch::round(torch::atan2(axis_y, axis_x) / (PI * 0.5)).to(torch::kLong).remainder(4);
    turns = torch::where(turns.lt(0), turns + 4, turns);
    turns = torch::where(axis_norm.gt(1.0e-6F), turns, torch::zeros_like(turns));

    const auto group_channels = descriptors.size(1) / 4;
    auto canonical = torch::zeros_like(descriptors);
    for (int64_t turn = 0; turn < 4; ++turn)
    {
        auto shifted = turn == 0 ? descriptors : torch::roll(descriptors, {-turn * group_channels}, {1});
        canonical = canonical + shifted * turns.eq(turn).to(descriptors.dtype());
    }
    return canonical.contiguous();
}

torch::Tensor make_pairwise_texture_teacher_descriptor_loss(const torch::Tensor& descriptors_a,
                                                            const torch::Tensor& descriptors_b,
                                                            const torch::Tensor& view_a, const torch::Tensor& warp,
                                                            const torch::Tensor& valid_mask)
{
    auto sample_indices = filter_descriptor_sample_indices(make_descriptor_sample_indices(descriptors_a), valid_mask,
                                                           descriptors_a.size(2), descriptors_a.size(3));
    if (sample_indices.numel() == 0)
    {
        return torch::zeros({}, descriptors_a.options());
    }

    auto teacher_a = make_rotation_invariant_texture_target(view_a, descriptors_a.size(2), descriptors_a.size(3),
                                                            descriptors_a.size(1));
    auto sampled_teacher = sample_spatial_descriptors(teacher_a, sample_indices);
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    auto target_coordinates =
        make_descriptor_target_coordinates(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    auto positive_b = sample_warped_descriptors(descriptors_b, target_coordinates);

    auto normalize = [](const torch::Tensor& tensor)
    {
        return tensor / tensor.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    };
    sampled_teacher = normalize(sampled_teacher);
    sampled_a = normalize(sampled_a);
    positive_b = normalize(positive_b);
    auto loss_a = 1.0F - (sampled_a * sampled_teacher).sum(2);
    auto loss_b = 1.0F - (positive_b * sampled_teacher).sum(2);
    return (loss_a + loss_b).mean() * 0.5;
}

struct DescriptorTrainingMetrics
{
    torch::Tensor loss;
    torch::Tensor accuracy;
    torch::Tensor positive_score;
    torch::Tensor hard_negative_score;
    torch::Tensor positive_margin;
    torch::Tensor positive_rank;
    torch::Tensor diversity;
};

DescriptorTrainingMetrics make_zero_descriptor_training_metrics(const torch::Tensor& reference)
{
    auto zero = torch::zeros({}, reference.options());
    return DescriptorTrainingMetrics{zero, zero, zero, zero, zero, zero, zero};
}

struct GraphMatchingTrainingMetrics
{
    torch::Tensor loss;
    DescriptorTrainingMetrics sparse_descriptor;
    torch::Tensor accuracy;
    int64_t query_count = 0;
    int64_t positive_count = 0;
    int64_t dustbin_count = 0;
    int64_t features_a_count = 0;
    int64_t features_b_count = 0;
};

DescriptorTrainingMetrics
make_sparse_descriptor_metrics(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                               const torch::Tensor& warp, const torch::Tensor& valid_mask,
                               const torch::Tensor& candidate_valid_mask = torch::Tensor(),
                               int64_t broad_far_negative_count = DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT)
{
    if (descriptors_a.size(0) > 1)
    {
        std::vector<DescriptorTrainingMetrics> batch_metrics;
        batch_metrics.reserve(static_cast<size_t>(descriptors_a.size(0)));
        for (int64_t batch = 0; batch < descriptors_a.size(0); ++batch)
        {
            auto candidate_mask = candidate_valid_mask.defined()
                                      ? candidate_valid_mask.index({batch}).unsqueeze(0).contiguous()
                                      : torch::Tensor();
            batch_metrics.push_back(make_sparse_descriptor_metrics(
                descriptors_a.index({batch}).unsqueeze(0), descriptors_b.index({batch}).unsqueeze(0),
                warp.index({batch}).unsqueeze(0), valid_mask.index({batch}).unsqueeze(0), candidate_mask,
                broad_far_negative_count));
        }
        auto stack_metric = [](const std::vector<DescriptorTrainingMetrics>& metrics, auto member)
        {
            std::vector<torch::Tensor> values;
            values.reserve(metrics.size());
            for (const auto& metric : metrics)
            {
                values.push_back(member(metric));
            }
            return torch::stack(values).mean();
        };
        return DescriptorTrainingMetrics{stack_metric(batch_metrics,
                                                      [](const DescriptorTrainingMetrics& metric)
                                                      {
                                                          return metric.loss;
                                                      }),
                                         stack_metric(batch_metrics,
                                                      [](const DescriptorTrainingMetrics& metric)
                                                      {
                                                          return metric.accuracy;
                                                      }),
                                         stack_metric(batch_metrics,
                                                      [](const DescriptorTrainingMetrics& metric)
                                                      {
                                                          return metric.positive_score;
                                                      }),
                                         stack_metric(batch_metrics,
                                                      [](const DescriptorTrainingMetrics& metric)
                                                      {
                                                          return metric.hard_negative_score;
                                                      }),
                                         stack_metric(batch_metrics,
                                                      [](const DescriptorTrainingMetrics& metric)
                                                      {
                                                          return metric.positive_margin;
                                                      }),
                                         stack_metric(batch_metrics,
                                                      [](const DescriptorTrainingMetrics& metric)
                                                      {
                                                          return metric.positive_rank;
                                                      }),
                                         stack_metric(batch_metrics,
                                                      [](const DescriptorTrainingMetrics& metric)
                                                      {
                                                          return metric.diversity;
                                                      })};
    }
    const auto spatial_count = descriptors_b.size(2) * descriptors_b.size(3);
    auto sample_indices = make_balanced_descriptor_sample_indices(valid_mask, descriptors_a.size(2),
                                                                  descriptors_a.size(3), MAX_DESCRIPTOR_LOSS_SAMPLES);
    if (sample_indices.numel() == 0)
    {
        auto zero = torch::zeros({}, descriptors_a.options());
        return DescriptorTrainingMetrics{zero, zero, zero, zero, zero, zero, zero};
    }
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    auto target_indices =
        make_descriptor_target_indices(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    auto candidate_indices = make_descriptor_candidate_indices(target_indices, spatial_count, descriptors_b.size(3),
                                                               candidate_valid_mask, broad_far_negative_count);
    auto sampled_b = gather_descriptor_candidates(descriptors_b, candidate_indices);
    auto target_coordinates =
        make_descriptor_target_coordinates(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    auto positive_b = sample_warped_descriptors(descriptors_b, target_coordinates);
    if (sampled_b.size(2) > 1)
    {
        sampled_b = torch::cat({positive_b.unsqueeze(2), sampled_b.narrow(2, 1, sampled_b.size(2) - 1)}, 2);
    }
    else
    {
        sampled_b = positive_b.unsqueeze(2);
    }
    auto target = torch::zeros({descriptors_a.size(0), sample_indices.size(0)},
                               torch::TensorOptions().dtype(torch::kLong).device(descriptors_a.device()));
    auto local_loss = descriptor_candidate_cross_entropy_loss(sampled_a, sampled_b, target);
    auto global_targets =
        torch::arange(sample_indices.size(0), torch::TensorOptions().dtype(torch::kLong).device(descriptors_a.device()))
            .unsqueeze(0)
            .expand({descriptors_a.size(0), sample_indices.size(0)});
    auto global_ab = make_strict_descriptor_cross_entropy_loss(sampled_a, positive_b, global_targets);
    auto global_ba = make_strict_descriptor_cross_entropy_loss(positive_b, sampled_a, global_targets);
    auto dense_hard_negative =
        make_dense_descriptor_hard_negative_loss(descriptors_a, descriptors_b, sample_indices, target_indices);
    auto warp_contrastive = make_warp_descriptor_contrastive_loss(descriptors_a, descriptors_b, warp, valid_mask);
    auto direct_full_map = make_direct_full_map_descriptor_loss(descriptors_a, descriptors_b, warp, valid_mask);
    auto cross_batch_contrastive = make_cross_batch_descriptor_contrastive_loss(sampled_a, positive_b);
    auto positive_alignment = make_positive_descriptor_alignment_loss(sampled_a, positive_b);
    auto patch_alignment = make_patch_descriptor_alignment_loss(descriptors_a, descriptors_b, warp, valid_mask);
    auto supervised_ranking = make_supervised_descriptor_ranking_loss(sampled_a, sampled_b);
    auto sampled_decorrelation = torch::zeros({}, descriptors_a.options());
    if constexpr (SAMPLED_DESCRIPTOR_DECORRELATION_WEIGHT > 0.0)
    {
        sampled_decorrelation =
            (make_sampled_descriptor_decorrelation_loss(sampled_a, sample_indices, descriptors_a.size(3)) +
             make_sampled_descriptor_decorrelation_loss(positive_b, sample_indices, descriptors_a.size(3))) *
            0.5;
    }
    auto loss =
        local_loss * DESCRIPTOR_LOCAL_CE_WEIGHT + (global_ab + global_ba) * (0.5 * DESCRIPTOR_GLOBAL_CE_WEIGHT) +
        dense_hard_negative * DENSE_DESCRIPTOR_HARD_NEGATIVE_WEIGHT +
        warp_contrastive * WARP_DESCRIPTOR_CONTRASTIVE_WEIGHT + direct_full_map * DIRECT_FULL_MAP_DESCRIPTOR_WEIGHT +
        cross_batch_contrastive * CROSS_BATCH_DESCRIPTOR_CONTRASTIVE_WEIGHT +
        positive_alignment * POSITIVE_DESCRIPTOR_ALIGNMENT_WEIGHT +
        patch_alignment * PATCH_DESCRIPTOR_ALIGNMENT_WEIGHT +
        supervised_ranking * SUPERVISED_DESCRIPTOR_RANKING_WEIGHT +
        sampled_decorrelation * SAMPLED_DESCRIPTOR_DECORRELATION_WEIGHT;
    auto logits = descriptor_candidate_similarity_scores(sampled_a, sampled_b);
    auto predictions = logits.argmax(2);
    auto accuracy = predictions.eq(target).to(torch::kFloat32).mean();
    auto positive_scores = logits.narrow(2, 0, 1).squeeze(2);
    torch::Tensor hard_negative_scores;
    torch::Tensor positive_ranks;
    if (logits.size(2) > 1)
    {
        auto negative_scores = logits.narrow(2, 1, logits.size(2) - 1);
        hard_negative_scores = std::get<0>(negative_scores.max(2));
        positive_ranks = negative_scores.gt(positive_scores.unsqueeze(2)).to(torch::kFloat32).sum(2) + 1.0F;
    }
    else
    {
        hard_negative_scores = positive_scores;
        positive_ranks = torch::ones_like(positive_scores);
    }
    auto positive_score = positive_scores.mean();
    auto hard_negative_score = hard_negative_scores.mean();
    auto positive_margin = (positive_scores - hard_negative_scores).mean();
    auto positive_rank = positive_ranks.mean();
    auto diversity = (descriptor_diversity_loss(sampled_a) + descriptor_diversity_loss(positive_b)) * 0.5 +
                     (make_descriptor_map_regularization_loss(descriptors_a) +
                      make_descriptor_map_regularization_loss(descriptors_b)) *
                         0.5;
    return DescriptorTrainingMetrics{loss,          accuracy, positive_score, hard_negative_score, positive_margin,
                                     positive_rank, diversity};
}

torch::Tensor assign_graph_matching_targets(const torch::Tensor& keypoints_a, const torch::Tensor& keypoints_b,
                                            const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                            double positive_radius_pixels)
{
    const auto dustbin = keypoints_b.size(0);
    auto targets = torch::full({keypoints_a.size(0)}, dustbin,
                               torch::TensorOptions().dtype(torch::kLong).device(keypoints_a.device()));
    if (keypoints_a.size(0) == 0 || keypoints_b.size(0) == 0)
    {
        return targets;
    }

    const auto image_height = warp.size(1);
    const auto image_width = warp.size(2);
    auto points_a = keypoints_a.to(warp.device(), torch::kFloat32).contiguous();
    auto points_b = keypoints_b.to(warp.device(), torch::kFloat32).contiguous();
    auto x = points_a.index({torch::indexing::Slice(), 0}).round().to(torch::kLong);
    auto y = points_a.index({torch::indexing::Slice(), 1}).round().to(torch::kLong);
    auto source_in_bounds = x.ge(0).logical_and(x.lt(image_width)).logical_and(y.ge(0)).logical_and(y.lt(image_height));
    auto x_safe = x.clamp(0, image_width - 1);
    auto y_safe = y.clamp(0, image_height - 1);
    auto source_linear = y_safe * image_width + x_safe;
    auto source_valid = valid_mask.to(warp.device(), torch::kBool)
                            .reshape({image_height * image_width})
                            .index_select(0, source_linear)
                            .logical_and(source_in_bounds);

    auto sampled_warp = warp.reshape({image_height * image_width, 2}).index_select(0, source_linear);
    auto target_x = sampled_warp.index({torch::indexing::Slice(), 0}).round().to(torch::kLong);
    auto target_y = sampled_warp.index({torch::indexing::Slice(), 1}).round().to(torch::kLong);
    auto target_in_bounds = target_x.ge(0)
                                .logical_and(target_x.lt(image_width))
                                .logical_and(target_y.ge(0))
                                .logical_and(target_y.lt(image_height));

    auto delta = points_b.unsqueeze(0) - sampled_warp.unsqueeze(1);
    auto distances = delta.pow(2).sum(2);
    auto min_result = distances.min(1);
    auto best_distance = std::get<0>(min_result);
    auto best_index = std::get<1>(min_result).to(torch::kLong);
    auto positive = source_valid.logical_and(target_in_bounds)
                        .logical_and(best_distance.le(positive_radius_pixels * positive_radius_pixels));
    return torch::where(positive, best_index, targets.to(warp.device())).to(keypoints_a.device(), torch::kLong);
}

torch::Tensor make_graph_candidate_indices(const torch::Tensor& target_indices, int64_t keypoint_count_b,
                                           int64_t max_candidates)
{
    const auto dustbin = keypoint_count_b;
    std::vector<int64_t> candidates;
    candidates.reserve(static_cast<size_t>(std::max<int64_t>(1, max_candidates)));

    auto targets_cpu = target_indices.detach().to(torch::kCPU, torch::kLong).contiguous();
    for (int64_t index = 0; index < targets_cpu.numel(); ++index)
    {
        const auto label = targets_cpu[index].item<int64_t>();
        if (label >= 0 && label < keypoint_count_b &&
            std::find(candidates.begin(), candidates.end(), label) == candidates.end())
        {
            candidates.push_back(label);
        }
    }

    for (int64_t candidate = 0;
         candidate < keypoint_count_b && static_cast<int64_t>(candidates.size()) < max_candidates - 1; ++candidate)
    {
        if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end())
        {
            candidates.push_back(candidate);
        }
    }

    candidates.push_back(dustbin);
    return torch::tensor(candidates, torch::TensorOptions().dtype(torch::kLong).device(target_indices.device()));
}

torch::Tensor make_graph_training_query_indices(const torch::Tensor& target_indices, int64_t keypoint_count_b,
                                                int64_t max_queries)
{
    if (max_queries <= 0 || target_indices.numel() == 0)
    {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(target_indices.device()));
    }

    const auto positive_mask = target_indices.ge(0).logical_and(target_indices.lt(keypoint_count_b));
    auto positive_indices = torch::nonzero(positive_mask).flatten();
    auto background_indices = torch::nonzero(positive_mask.logical_not()).flatten();
    const auto reserved_background =
        background_indices.size(0) > 0
            ? std::min<int64_t>(background_indices.size(0), std::max<int64_t>(1, max_queries / 4))
            : 0;
    const auto positive_keep = std::min<int64_t>(positive_indices.size(0), max_queries - reserved_background);
    if (positive_keep >= max_queries)
    {
        return positive_indices.narrow(0, 0, max_queries).contiguous();
    }
    const auto background_keep = std::min<int64_t>(background_indices.size(0), max_queries - positive_keep);
    if (background_keep <= 0)
    {
        return positive_indices.narrow(0, 0, positive_keep).contiguous();
    }
    return torch::cat({positive_indices.narrow(0, 0, positive_keep), background_indices.narrow(0, 0, background_keep)},
                      0)
        .contiguous();
}

torch::Tensor scale_feature_keypoints_to_image(const torch::Tensor& keypoints, int64_t feature_width,
                                               int64_t feature_height, int64_t image_width, int64_t image_height)
{
    if (feature_width <= 0 || feature_height <= 0)
    {
        return keypoints;
    }
    auto scale = torch::tensor({static_cast<float>(image_width) / static_cast<float>(feature_width),
                                static_cast<float>(image_height) / static_cast<float>(feature_height)},
                               keypoints.options());
    return (keypoints + 0.5F) * scale - 0.5F;
}

torch::Tensor make_keypoint_descriptor_target_coordinates(const torch::Tensor& image_keypoints_a,
                                                          const torch::Tensor& warp, int64_t descriptor_height,
                                                          int64_t descriptor_width)
{
    using torch::indexing::Slice;

    const auto image_height = warp.size(1);
    const auto image_width = warp.size(2);
    auto image_x = image_keypoints_a.index({Slice(), 0}).clamp(0, image_width - 1);
    auto image_y = image_keypoints_a.index({Slice(), 1}).clamp(0, image_height - 1);
    auto grid_x =
        image_width > 1 ? image_x / static_cast<float>(image_width - 1) * 2.0F - 1.0F : torch::zeros_like(image_x);
    auto grid_y =
        image_height > 1 ? image_y / static_cast<float>(image_height - 1) * 2.0F - 1.0F : torch::zeros_like(image_y);
    auto grid = torch::stack({grid_x, grid_y}, 1)
                    .reshape({1, image_keypoints_a.size(0), 1, 2})
                    .expand({warp.size(0), image_keypoints_a.size(0), 1, 2})
                    .contiguous();
    auto sampled_warp = torch::nn::functional::grid_sample(warp.permute({0, 3, 1, 2}).contiguous(), grid,
                                                           torch::nn::functional::GridSampleFuncOptions()
                                                               .mode(torch::kBilinear)
                                                               .padding_mode(torch::kBorder)
                                                               .align_corners(true))
                            .squeeze(3)
                            .permute({0, 2, 1});
    auto target_x = (sampled_warp.index({Slice(), Slice(), 0}) + 0.5F) * static_cast<float>(descriptor_width) /
                        static_cast<float>(image_width) -
                    0.5F;
    auto target_y = (sampled_warp.index({Slice(), Slice(), 1}) + 0.5F) * static_cast<float>(descriptor_height) /
                        static_cast<float>(image_height) -
                    0.5F;
    target_x = target_x.clamp(0, descriptor_width - 1);
    target_y = target_y.clamp(0, descriptor_height - 1);
    return torch::stack({target_x, target_y}, 2).contiguous();
}

DescriptorTrainingMetrics make_keypoint_descriptor_metrics(const FeatureSet& features_a, const FeatureSet& features_b,
                                                           const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    auto zero = torch::zeros({}, warp.options());
    if (!features_a.keypoints.defined() || !features_b.keypoints.defined() || features_a.keypoints.size(0) == 0 ||
        features_b.keypoints.size(0) == 0)
    {
        return DescriptorTrainingMetrics{zero, zero, zero, zero, zero, zero, zero};
    }

    const auto query_count = std::min<int64_t>(features_a.keypoints.size(0), KEYPOINT_DESCRIPTOR_MAX_QUERIES);
    auto query_indices =
        torch::arange(query_count, torch::TensorOptions().dtype(torch::kLong).device(features_a.keypoints.device()));
    auto keypoints_a = features_a.keypoints.index_select(0, query_indices).to(warp.device());
    auto descriptors_a = features_a.descriptors.index_select(0, query_indices).to(warp.device());
    auto keypoints_b = features_b.keypoints.to(warp.device());
    auto descriptors_b = features_b.descriptors.to(warp.device());

    auto image_keypoints_a = scale_feature_keypoints_to_image(
        keypoints_a, features_a.feature_map_width, features_a.feature_map_height, warp.size(2), warp.size(1));
    auto image_keypoints_b = scale_feature_keypoints_to_image(
        keypoints_b, features_b.feature_map_width, features_b.feature_map_height, warp.size(2), warp.size(1));
    auto target_full = assign_graph_matching_targets(image_keypoints_a, image_keypoints_b, warp, valid_mask,
                                                     GRAPH_MATCHING_POSITIVE_RADIUS_PIXELS);
    const auto positive_mask = target_full.lt(keypoints_b.size(0));
    if (!positive_mask.any().item<bool>())
    {
        return DescriptorTrainingMetrics{zero, zero, zero, zero, zero, zero, zero};
    }

    auto positive_descriptors_a = descriptors_a.index({positive_mask});
    auto positive_targets = target_full.index({positive_mask});
    auto strict_ce_loss = make_strict_descriptor_cross_entropy_loss(
        positive_descriptors_a.unsqueeze(0), descriptors_b.unsqueeze(0), positive_targets.unsqueeze(0));
    auto scores =
        descriptor_pair_similarity_scores(positive_descriptors_a.unsqueeze(0), descriptors_b.unsqueeze(0)).squeeze(0);
    auto row_indices = torch::arange(positive_targets.size(0), positive_targets.options());
    auto positive_scores = scores.index({row_indices, positive_targets});
    auto predictions = scores.argmax(1);
    auto accuracy = predictions.eq(positive_targets).to(torch::kFloat32).mean();
    auto positive_score = positive_scores.mean();
    auto diversity = (descriptor_diversity_loss(positive_descriptors_a.unsqueeze(0)) +
                      descriptor_diversity_loss(descriptors_b.unsqueeze(0))) *
                     0.5;
    if (descriptors_b.size(0) < 2)
    {
        return DescriptorTrainingMetrics{strict_ce_loss,
                                         accuracy,
                                         positive_score,
                                         positive_score,
                                         torch::zeros({}, scores.options()),
                                         torch::ones({}, scores.options()),
                                         diversity};
    }

    auto positive_source_indices = torch::nonzero(positive_mask).flatten();
    auto positive_image_keypoints_a =
        image_keypoints_a.index_select(0, positive_source_indices).round().to(torch::kLong);
    auto positive_source_linear =
        positive_image_keypoints_a.index({torch::indexing::Slice(), 1}).clamp(0, warp.size(1) - 1) * warp.size(2) +
        positive_image_keypoints_a.index({torch::indexing::Slice(), 0}).clamp(0, warp.size(2) - 1);
    auto warped_positive_targets =
        warp.reshape({warp.size(1) * warp.size(2), 2}).index_select(0, positive_source_linear);
    auto candidate_delta = image_keypoints_b.unsqueeze(0) - warped_positive_targets.unsqueeze(1);
    auto false_negative_mask = candidate_delta.pow(2).sum(2).le(KEYPOINT_DESCRIPTOR_FALSE_NEGATIVE_RADIUS_PIXELS *
                                                                KEYPOINT_DESCRIPTOR_FALSE_NEGATIVE_RADIUS_PIXELS);
    false_negative_mask.index_put_({row_indices, positive_targets}, false);
    auto candidate_available = false_negative_mask.logical_not();
    candidate_available.index_put_({row_indices, positive_targets}, false);
    auto has_available_negative = candidate_available.any(1);
    if (has_available_negative.logical_not().any().item<bool>())
    {
        false_negative_mask.index_put_({has_available_negative.logical_not()}, false);
    }

    auto masked_logits = scores * 20.0F;
    masked_logits = masked_logits.masked_fill(false_negative_mask, -std::numeric_limits<float>::infinity());
    auto ce_loss = torch::nn::functional::cross_entropy(masked_logits, positive_targets);
    auto negative_scores = scores.masked_fill(false_negative_mask, -std::numeric_limits<float>::infinity());
    negative_scores.index_put_({row_indices, positive_targets}, -std::numeric_limits<float>::infinity());
    auto hardest_negative = std::get<0>(negative_scores.max(1));
    auto margin_loss = torch::relu(hardest_negative - positive_scores + KEYPOINT_DESCRIPTOR_MARGIN).mean();
    auto alignment_loss = (1.0F - positive_scores).mean();
    auto hard_negative_score = hardest_negative.mean();
    auto positive_ranks = negative_scores.gt(positive_scores.unsqueeze(1)).to(torch::kFloat32).sum(1) + 1.0F;
    return DescriptorTrainingMetrics{ce_loss + margin_loss * KEYPOINT_DESCRIPTOR_MARGIN_WEIGHT +
                                         alignment_loss * KEYPOINT_DESCRIPTOR_POSITIVE_ALIGNMENT_WEIGHT,
                                     accuracy,
                                     positive_score,
                                     hard_negative_score,
                                     (positive_scores - hardest_negative).mean(),
                                     positive_ranks.mean(),
                                     diversity};
}

torch::Tensor make_keypoint_descriptor_loss(const FeatureSet& features_a, const FeatureSet& features_b,
                                            const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_keypoint_descriptor_metrics(features_a, features_b, warp, valid_mask).loss;
}

torch::Tensor make_keypoint_dense_descriptor_loss(const FeatureSet& features_a, const torch::Tensor& descriptors_b,
                                                  const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    auto zero = torch::zeros({}, warp.options());
    if (!features_a.keypoints.defined() || !features_a.descriptors.defined() || features_a.keypoints.size(0) == 0 ||
        descriptors_b.size(2) * descriptors_b.size(3) < 2)
    {
        return zero;
    }

    const auto query_count = std::min<int64_t>(features_a.keypoints.size(0), KEYPOINT_DENSE_DESCRIPTOR_MAX_QUERIES);
    auto query_indices =
        torch::arange(query_count, torch::TensorOptions().dtype(torch::kLong).device(features_a.keypoints.device()));
    auto keypoints_a = features_a.keypoints.index_select(0, query_indices).to(warp.device());
    auto descriptors_a = features_a.descriptors.index_select(0, query_indices).to(warp.device());
    auto image_keypoints_a = scale_feature_keypoints_to_image(
        keypoints_a, features_a.feature_map_width, features_a.feature_map_height, warp.size(2), warp.size(1));

    const auto image_width = warp.size(2);
    const auto image_height = warp.size(1);
    const auto descriptor_width = descriptors_b.size(3);
    const auto descriptor_height = descriptors_b.size(2);
    auto source_x =
        image_keypoints_a.index({torch::indexing::Slice(), 0}).round().to(torch::kLong).clamp(0, image_width - 1);
    auto source_y =
        image_keypoints_a.index({torch::indexing::Slice(), 1}).round().to(torch::kLong).clamp(0, image_height - 1);
    auto source_linear = source_y * image_width + source_x;
    auto flat_valid = valid_mask.reshape({image_height * image_width}).to(torch::kBool);
    auto source_valid = flat_valid.index_select(0, source_linear);
    auto sampled_warp = warp.reshape({image_height * image_width, 2}).index_select(0, source_linear);
    auto target_x_float = sampled_warp.index({torch::indexing::Slice(), 0});
    auto target_y_float = sampled_warp.index({torch::indexing::Slice(), 1});
    auto target_in_bounds = target_x_float.ge(0.0F)
                                .logical_and(target_x_float.le(static_cast<float>(image_width - 1)))
                                .logical_and(target_y_float.ge(0.0F))
                                .logical_and(target_y_float.le(static_cast<float>(image_height - 1)));
    auto target_x_image = target_x_float.round().to(torch::kLong).clamp(0, image_width - 1);
    auto target_y_image = target_y_float.round().to(torch::kLong).clamp(0, image_height - 1);
    auto positive_mask = source_valid.logical_and(target_in_bounds);
    if (!positive_mask.any().item<bool>())
    {
        return zero;
    }

    auto positive_descriptors_a = descriptors_a.index({positive_mask});
    auto target_x = torch::floor_divide(target_x_image.index({positive_mask}) * descriptor_width, image_width)
                        .clamp(0, descriptor_width - 1);
    auto target_y = torch::floor_divide(target_y_image.index({positive_mask}) * descriptor_height, image_height)
                        .clamp(0, descriptor_height - 1);
    auto target_indices = (target_y * descriptor_width + target_x).to(torch::kLong);
    const auto spatial_count = descriptor_width * descriptor_height;
    auto flat_b =
        descriptors_b.squeeze(0).permute({1, 2, 0}).reshape({spatial_count, descriptors_b.size(1)}).unsqueeze(0);
    auto scores = descriptor_pair_similarity_scores(positive_descriptors_a.unsqueeze(0), flat_b).squeeze(0);
    auto logits = scores * 20.0F;
    auto ce_loss = torch::nn::functional::cross_entropy(logits, target_indices);

    auto row_indices = torch::arange(target_indices.size(0), target_indices.options());
    auto positive_scores = scores.index({row_indices, target_indices});
    auto spatial_indices = torch::arange(spatial_count, target_indices.options());
    auto candidate_x = spatial_indices.remainder(descriptor_width).reshape({1, spatial_count});
    auto candidate_y = torch::floor_divide(spatial_indices, descriptor_width).reshape({1, spatial_count});
    auto dx = (candidate_x - target_x.unsqueeze(1)).to(torch::kFloat32);
    auto dy = (candidate_y - target_y.unsqueeze(1)).to(torch::kFloat32);
    const auto exclusion_radius_sq =
        DENSE_DESCRIPTOR_HARD_NEGATIVE_EXCLUSION_RADIUS * DENSE_DESCRIPTOR_HARD_NEGATIVE_EXCLUSION_RADIUS;
    auto near_positive = (dx.pow(2) + dy.pow(2)).le(exclusion_radius_sq);
    auto negative_scores = scores.masked_fill(near_positive, -std::numeric_limits<float>::infinity());
    auto hardest_negative = std::get<0>(negative_scores.max(1));
    auto finite_mask = torch::isfinite(hardest_negative);
    if (!finite_mask.any().item<bool>())
    {
        return ce_loss;
    }
    auto margin_loss = torch::relu(hardest_negative - positive_scores + KEYPOINT_DENSE_DESCRIPTOR_MARGIN);
    auto alignment_loss = (1.0F - positive_scores).mean();
    return ce_loss + margin_loss.index({finite_mask}).mean() * KEYPOINT_DENSE_DESCRIPTOR_MARGIN_WEIGHT +
           alignment_loss * KEYPOINT_DENSE_DESCRIPTOR_POSITIVE_ALIGNMENT_WEIGHT;
}

torch::Tensor make_keypoint_patch_descriptor_alignment_loss(const FeatureSet& features_a,
                                                            const torch::Tensor& descriptors_b,
                                                            const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    auto zero = torch::zeros({}, warp.options());
    if (!features_a.keypoints.defined() || !features_a.descriptors.defined() || features_a.keypoints.size(0) == 0 ||
        descriptors_b.size(0) == 0 || descriptors_b.size(2) == 0 || descriptors_b.size(3) == 0)
    {
        return zero;
    }

    const auto query_count = std::min<int64_t>(features_a.keypoints.size(0), KEYPOINT_DENSE_DESCRIPTOR_MAX_QUERIES);
    auto query_indices =
        torch::arange(query_count, torch::TensorOptions().dtype(torch::kLong).device(features_a.keypoints.device()));
    auto keypoints_a = features_a.keypoints.index_select(0, query_indices).to(warp.device());
    auto descriptors_a = features_a.descriptors.index_select(0, query_indices).to(warp.device());
    auto image_keypoints_a = scale_feature_keypoints_to_image(
        keypoints_a, features_a.feature_map_width, features_a.feature_map_height, warp.size(2), warp.size(1));

    const auto image_width = warp.size(2);
    const auto image_height = warp.size(1);
    const auto descriptor_width = descriptors_b.size(3);
    const auto descriptor_height = descriptors_b.size(2);
    auto source_x =
        image_keypoints_a.index({torch::indexing::Slice(), 0}).round().to(torch::kLong).clamp(0, image_width - 1);
    auto source_y =
        image_keypoints_a.index({torch::indexing::Slice(), 1}).round().to(torch::kLong).clamp(0, image_height - 1);
    auto source_linear = source_y * image_width + source_x;
    auto flat_valid = valid_mask.reshape({image_height * image_width}).to(torch::kBool);
    auto source_valid = flat_valid.index_select(0, source_linear);
    auto sampled_warp = warp.reshape({image_height * image_width, 2}).index_select(0, source_linear);
    auto target_x_float = sampled_warp.index({torch::indexing::Slice(), 0});
    auto target_y_float = sampled_warp.index({torch::indexing::Slice(), 1});
    auto target_in_bounds = target_x_float.ge(0.0F)
                                .logical_and(target_x_float.le(static_cast<float>(image_width - 1)))
                                .logical_and(target_y_float.ge(0.0F))
                                .logical_and(target_y_float.le(static_cast<float>(image_height - 1)));
    auto positive_mask = source_valid.logical_and(target_in_bounds);
    if (!positive_mask.any().item<bool>())
    {
        return zero;
    }

    auto positive_descriptors_a = torch::nn::functional::normalize(
        descriptors_a.index({positive_mask}), torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto normalized_descriptors_b =
        torch::nn::functional::normalize(descriptors_b, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto target_coordinates = make_keypoint_descriptor_target_coordinates(image_keypoints_a.index({positive_mask}),
                                                                          warp, descriptor_height, descriptor_width);
    auto positive_descriptors_b = sample_warped_descriptors(normalized_descriptors_b, target_coordinates).squeeze(0);
    auto positive_scores = (positive_descriptors_a * positive_descriptors_b).sum(1);
    return (1.0F - positive_scores).mean();
}

torch::Tensor make_warped_keypoint_descriptor_contrastive_loss(const FeatureSet& features_a,
                                                               const torch::Tensor& descriptors_b,
                                                               const torch::Tensor& warp,
                                                               const torch::Tensor& valid_mask)
{
    auto zero = torch::zeros({}, warp.options());
    if (!features_a.keypoints.defined() || !features_a.descriptors.defined() || features_a.keypoints.size(0) == 0 ||
        descriptors_b.size(0) == 0 || descriptors_b.size(2) == 0 || descriptors_b.size(3) == 0)
    {
        return zero;
    }

    const auto query_count = std::min<int64_t>(features_a.keypoints.size(0), KEYPOINT_DESCRIPTOR_MAX_QUERIES);
    auto query_indices =
        torch::arange(query_count, torch::TensorOptions().dtype(torch::kLong).device(features_a.keypoints.device()));
    auto keypoints_a = features_a.keypoints.index_select(0, query_indices).to(warp.device());
    auto descriptors_a = features_a.descriptors.index_select(0, query_indices).to(warp.device());
    auto image_keypoints_a = scale_feature_keypoints_to_image(
        keypoints_a, features_a.feature_map_width, features_a.feature_map_height, warp.size(2), warp.size(1));

    const auto image_width = warp.size(2);
    const auto image_height = warp.size(1);
    auto source_x =
        image_keypoints_a.index({torch::indexing::Slice(), 0}).round().to(torch::kLong).clamp(0, image_width - 1);
    auto source_y =
        image_keypoints_a.index({torch::indexing::Slice(), 1}).round().to(torch::kLong).clamp(0, image_height - 1);
    auto source_linear = source_y * image_width + source_x;
    auto source_valid =
        valid_mask.reshape({image_height * image_width}).to(warp.device(), torch::kBool).index_select(0, source_linear);
    auto sampled_warp = warp.reshape({image_height * image_width, 2}).index_select(0, source_linear);
    auto target_x_float = sampled_warp.index({torch::indexing::Slice(), 0});
    auto target_y_float = sampled_warp.index({torch::indexing::Slice(), 1});
    auto target_in_bounds = target_x_float.ge(0.0F)
                                .logical_and(target_x_float.le(static_cast<float>(image_width - 1)))
                                .logical_and(target_y_float.ge(0.0F))
                                .logical_and(target_y_float.le(static_cast<float>(image_height - 1)));
    auto positive_mask = source_valid.logical_and(target_in_bounds);
    if (!positive_mask.any().item<bool>())
    {
        return zero;
    }

    auto positive_descriptors_a = descriptors_a.index({positive_mask});
    auto positive_keypoints_a = image_keypoints_a.index({positive_mask});
    auto normalized_descriptors_b =
        torch::nn::functional::normalize(descriptors_b, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto target_coordinates = make_keypoint_descriptor_target_coordinates(positive_keypoints_a, warp,
                                                                          descriptors_b.size(2), descriptors_b.size(3));
    auto positive_descriptors_b = sample_warped_descriptors(normalized_descriptors_b, target_coordinates).squeeze(0);
    positive_descriptors_a = torch::nn::functional::normalize(
        positive_descriptors_a, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    if (positive_descriptors_a.size(0) < 2)
    {
        auto positive_scores = (positive_descriptors_a * positive_descriptors_b).sum(1);
        return (1.0F - positive_scores).mean();
    }
    auto targets =
        torch::arange(positive_descriptors_a.size(0), torch::TensorOptions().dtype(torch::kLong).device(warp.device()));
    auto scores =
        descriptor_pair_similarity_scores(positive_descriptors_a.unsqueeze(0), positive_descriptors_b.unsqueeze(0))
            .squeeze(0) *
        20.0F;
    auto loss_ab = torch::nn::functional::cross_entropy(scores, targets);
    auto loss_ba = torch::nn::functional::cross_entropy(scores.transpose(0, 1).contiguous(), targets);
    auto row_indices = torch::arange(positive_descriptors_a.size(0), targets.options());
    auto positive_scores = scores.index({row_indices, targets}) / 20.0F;
    auto alignment = (1.0F - positive_scores).mean();
    return (loss_ab + loss_ba) * 0.5F + alignment * 8.0F;
}

template <typename GraphMatcherT>
GraphMatchingTrainingMetrics
make_keypoint_graph_matching_metrics(GraphMatcherT& graph_matcher, const FeatureSet& features_a,
                                     const FeatureSet& features_b, const torch::Tensor& warp,
                                     const torch::Tensor& valid_mask)
{
    auto zero = torch::zeros({}, warp.options());
    if (!features_a.keypoints.defined() || !features_b.keypoints.defined() || features_a.keypoints.size(0) == 0 ||
        features_b.keypoints.size(0) == 0)
    {
        return GraphMatchingTrainingMetrics{zero,
                                            DescriptorTrainingMetrics{zero, zero, zero, zero, zero, zero, zero},
                                            zero,
                                            0,
                                            0,
                                            0,
                                            features_a.keypoints.defined() ? features_a.keypoints.size(0) : 0,
                                            features_b.keypoints.defined() ? features_b.keypoints.size(0) : 0};
    }

    auto keypoints_b = features_b.keypoints.to(warp.device());
    auto descriptors_b = features_b.descriptors.to(warp.device());

    auto all_keypoints_a = features_a.keypoints.to(warp.device());
    auto image_keypoints_a_all = scale_feature_keypoints_to_image(
        all_keypoints_a, features_a.feature_map_width, features_a.feature_map_height, warp.size(2), warp.size(1));
    auto image_keypoints_b = scale_feature_keypoints_to_image(
        keypoints_b, features_b.feature_map_width, features_b.feature_map_height, warp.size(2), warp.size(1));
    auto target_full_all = assign_graph_matching_targets(image_keypoints_a_all, image_keypoints_b, warp, valid_mask,
                                                         GRAPH_MATCHING_POSITIVE_RADIUS_PIXELS);
    auto query_indices =
        make_graph_training_query_indices(target_full_all, keypoints_b.size(0), GRAPH_MATCHING_MAX_QUERIES);
    if (query_indices.size(0) == 0)
    {
        return GraphMatchingTrainingMetrics{zero,
                                            DescriptorTrainingMetrics{zero, zero, zero, zero, zero, zero, zero},
                                            zero,
                                            0,
                                            0,
                                            0,
                                            features_a.keypoints.size(0),
                                            features_b.keypoints.size(0)};
    }
    auto feature_query_indices = query_indices.to(features_a.keypoints.device());
    auto keypoints_a = features_a.keypoints.index_select(0, feature_query_indices).to(warp.device());
    auto descriptors_a = features_a.descriptors.index_select(0, feature_query_indices).to(warp.device());
    auto target_full = target_full_all.index_select(0, query_indices);
    auto output = graph_matcher.forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b);
    auto row_logits = output.logits.narrow(0, 0, target_full.size(0));
    const auto positive_mask = target_full.lt(keypoints_b.size(0));
    const auto dustbin_mask = target_full.eq(keypoints_b.size(0));
    auto positive_indices = torch::nonzero(positive_mask).flatten();
    auto dustbin_indices = torch::nonzero(dustbin_mask).flatten();
    torch::Tensor loss_indices;
    if (positive_indices.numel() > 0)
    {
        const auto dustbin_keep = std::min<int64_t>(dustbin_indices.size(0), positive_indices.size(0));
        loss_indices = dustbin_keep > 0 ? torch::cat({positive_indices, dustbin_indices.narrow(0, 0, dustbin_keep)}, 0)
                                        : positive_indices;
    }
    else
    {
        const auto dustbin_keep = std::min<int64_t>(dustbin_indices.size(0), GRAPH_MATCHING_MAX_QUERIES / 4);
        loss_indices =
            dustbin_keep > 0 ? dustbin_indices.narrow(0, 0, dustbin_keep) : torch::empty({0}, target_full.options());
    }
    auto row_loss = loss_indices.numel() > 0
                        ? torch::nn::functional::cross_entropy(row_logits.index_select(0, loss_indices),
                                                               target_full.index_select(0, loss_indices))
                        : zero;
    auto column_loss = zero;
    auto positive_margin_loss = zero;
    if (positive_indices.numel() > 0)
    {
        auto positive_targets = target_full.index_select(0, positive_indices);
        auto positive_row_logits = row_logits.index_select(0, positive_indices);
        auto row_indices = torch::arange(positive_indices.size(0), positive_targets.options());
        auto positive_logits = positive_row_logits.index({row_indices, positive_targets});
        auto negative_logits = positive_row_logits.narrow(1, 0, keypoints_b.size(0)).clone();
        negative_logits.index_put_({row_indices, positive_targets}, -std::numeric_limits<float>::infinity());
        auto hardest_negative = std::get<0>(negative_logits.max(1));
        auto dustbin_logits = positive_row_logits.index({torch::indexing::Slice(), keypoints_b.size(0)});
        auto finite_negative = torch::isfinite(hardest_negative);
        auto margin_terms = torch::relu(dustbin_logits - positive_logits + GRAPH_POSITIVE_MARGIN);
        if (finite_negative.any().template item<bool>())
        {
            margin_terms = margin_terms + torch::relu(hardest_negative.index({finite_negative}) -
                                                      positive_logits.index({finite_negative}) + GRAPH_POSITIVE_MARGIN)
                                              .mean();
        }
        positive_margin_loss = margin_terms.mean();

        auto targets_cpu =
            target_full.index_select(0, positive_indices).detach().to(torch::kCPU, torch::kLong).contiguous();
        std::vector<char> seen_targets(static_cast<std::size_t>(keypoints_b.size(0)), 0);
        std::vector<int64_t> unique_offsets;
        unique_offsets.reserve(static_cast<std::size_t>(targets_cpu.numel()));
        for (int64_t offset = 0; offset < targets_cpu.numel(); ++offset)
        {
            const auto target = targets_cpu.index({offset}).template item<int64_t>();
            if (target >= 0 && target < keypoints_b.size(0) && !seen_targets[static_cast<std::size_t>(target)])
            {
                seen_targets[static_cast<std::size_t>(target)] = 1;
                unique_offsets.push_back(offset);
            }
        }
        if (!unique_offsets.empty())
        {
            auto keep = torch::from_blob(unique_offsets.data(), {static_cast<int64_t>(unique_offsets.size())},
                                         torch::TensorOptions().dtype(torch::kLong))
                            .clone()
                            .to(target_full.device());
            auto column_indices = target_full.index_select(0, positive_indices).index_select(0, keep);
            auto column_targets = positive_indices.index_select(0, keep);
            auto column_logits =
                output.logits.index({torch::indexing::Slice(0, keypoints_a.size(0) + 1), column_indices})
                    .transpose(0, 1)
                    .contiguous();
            column_loss = torch::nn::functional::cross_entropy(column_logits, column_targets);
        }
    }
    auto loss = row_loss + column_loss * 0.5F + positive_margin_loss * GRAPH_POSITIVE_MARGIN_WEIGHT;
    auto sparse_descriptor = make_keypoint_descriptor_metrics(features_a, features_b, warp, valid_mask);
    auto predictions = row_logits.argmax(1);
    auto accuracy =
        positive_mask.any().template item<bool>()
            ? predictions.index({positive_mask}).eq(target_full.index({positive_mask})).to(torch::kFloat32).mean()
            : zero;
    const auto dustbin_label = keypoints_b.size(0);
    const auto dustbin_count = target_full.eq(dustbin_label).sum().template item<int64_t>();
    return GraphMatchingTrainingMetrics{loss,
                                        sparse_descriptor,
                                        accuracy,
                                        target_full.size(0),
                                        target_full.size(0) - dustbin_count,
                                        dustbin_count,
                                        features_a.keypoints.size(0),
                                        features_b.keypoints.size(0)};
}

std::pair<FeatureSet, FeatureSet> make_supervised_graph_feature_pair(const SparseHeadOutput& sparse_a,
                                                                     const SparseHeadOutput& sparse_b,
                                                                     const torch::Tensor& warp,
                                                                     const torch::Tensor& valid_mask)
{
    const auto descriptor_height = sparse_a.descriptors.size(2);
    const auto descriptor_width = sparse_a.descriptors.size(3);
    auto sample_indices = make_balanced_descriptor_sample_indices(valid_mask, descriptor_height, descriptor_width,
                                                                  GRAPH_MATCHING_MAX_QUERIES);
    const auto descriptor_options = sparse_a.descriptors.options();
    const auto point_options = descriptor_options.dtype(torch::kFloat32);
    auto empty_points = torch::empty({0, 2}, point_options);
    auto empty_descriptors = torch::empty({0, sparse_a.descriptors.size(1)}, descriptor_options);
    auto empty_scores = torch::empty({0}, point_options);
    FeatureSet empty{empty_points,
                     empty_scores,
                     empty_descriptors,
                     torch::empty({0}, point_options),
                     torch::empty({0}, point_options),
                     torch::empty({0, 4}, point_options),
                     empty_points,
                     empty_scores,
                     descriptor_width,
                     descriptor_height};
    if (sample_indices.numel() == 0)
    {
        return {empty, empty};
    }

    auto keypoint_y = torch::floor_divide(sample_indices, descriptor_width).to(torch::kFloat32);
    auto keypoint_x = sample_indices.remainder(descriptor_width).to(torch::kFloat32);
    auto keypoints_a = torch::stack({keypoint_x, keypoint_y}, 1).contiguous();
    auto keypoints_b = make_descriptor_target_coordinates(warp, sample_indices, descriptor_height, descriptor_width)
                           .squeeze(0)
                           .contiguous();
    auto descriptors_a = sample_spatial_descriptors(sparse_a.descriptors, sample_indices).squeeze(0);
    auto descriptors_b = sample_warped_descriptors(sparse_b.descriptors, keypoints_b.unsqueeze(0)).squeeze(0);
    descriptors_a = descriptors_a / descriptors_a.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
    descriptors_b = descriptors_b / descriptors_b.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
    auto scores = torch::ones({sample_indices.size(0)}, point_options);
    auto scale = torch::ones({sample_indices.size(0)}, point_options);
    auto orientation = torch::zeros({sample_indices.size(0)}, point_options);
    auto affine = torch::zeros({sample_indices.size(0), 4}, point_options);

    FeatureSet features_a{keypoints_a, scores,           descriptors_a,    scale, orientation, affine, keypoints_a,
                          scores,      descriptor_width, descriptor_height};
    FeatureSet features_b{keypoints_b, scores,           descriptors_b,    scale, orientation, affine, keypoints_b,
                          scores,      descriptor_width, descriptor_height};
    return {features_a, features_b};
}

std::pair<FeatureSet, FeatureSet> make_warp_completed_keypoint_feature_pair(const FeatureSet& features_a,
                                                                            const torch::Tensor& descriptors_b,
                                                                            const torch::Tensor& warp,
                                                                            const torch::Tensor& valid_mask)
{
    const auto descriptor_height = descriptors_b.size(2);
    const auto descriptor_width = descriptors_b.size(3);
    const auto descriptor_options = descriptors_b.options();
    const auto point_options = descriptor_options.dtype(torch::kFloat32);
    auto make_empty = [&]()
    {
        auto empty_points = torch::empty({0, 2}, point_options);
        auto empty_descriptors = torch::empty({0, descriptors_b.size(1)}, descriptor_options);
        auto empty_scores = torch::empty({0}, point_options);
        return FeatureSet{empty_points,
                          empty_scores,
                          empty_descriptors,
                          torch::empty({0}, point_options),
                          torch::empty({0}, point_options),
                          torch::empty({0, 4}, point_options),
                          empty_points,
                          empty_scores,
                          descriptor_width,
                          descriptor_height};
    };
    if (!features_a.keypoints.defined() || !features_a.descriptors.defined() || features_a.keypoints.size(0) == 0 ||
        descriptors_b.size(0) == 0 || descriptor_height == 0 || descriptor_width == 0)
    {
        auto empty = make_empty();
        return {empty, empty};
    }

    auto keypoints_a = features_a.keypoints.to(warp.device());
    auto descriptors_a = features_a.descriptors.to(warp.device());
    auto image_keypoints_a = scale_feature_keypoints_to_image(
        keypoints_a, features_a.feature_map_width, features_a.feature_map_height, warp.size(2), warp.size(1));
    const auto image_width = warp.size(2);
    const auto image_height = warp.size(1);
    auto source_x =
        image_keypoints_a.index({torch::indexing::Slice(), 0}).round().to(torch::kLong).clamp(0, image_width - 1);
    auto source_y =
        image_keypoints_a.index({torch::indexing::Slice(), 1}).round().to(torch::kLong).clamp(0, image_height - 1);
    auto source_linear = source_y * image_width + source_x;
    auto source_valid =
        valid_mask.reshape({image_height * image_width}).to(warp.device(), torch::kBool).index_select(0, source_linear);
    if (!source_valid.any().item<bool>())
    {
        auto empty = make_empty();
        return {empty, empty};
    }

    auto kept_keypoints_a = keypoints_a.index({source_valid}).contiguous();
    auto kept_descriptors_a = descriptors_a.index({source_valid}).contiguous();
    auto kept_scores_a = features_a.scores.defined() && features_a.scores.size(0) == features_a.keypoints.size(0)
                             ? features_a.scores.to(warp.device()).index({source_valid}).contiguous()
                             : torch::ones({kept_keypoints_a.size(0)}, point_options.device(warp.device()));
    auto kept_image_keypoints_a = image_keypoints_a.index({source_valid}).contiguous();
    auto keypoints_b =
        make_keypoint_descriptor_target_coordinates(kept_image_keypoints_a, warp, descriptor_height, descriptor_width)
            .squeeze(0)
            .contiguous();
    auto sampled_descriptors_b =
        sample_warped_descriptors(descriptors_b, keypoints_b.unsqueeze(0)).squeeze(0).contiguous();
    kept_descriptors_a =
        torch::nn::functional::normalize(kept_descriptors_a, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    sampled_descriptors_b = torch::nn::functional::normalize(sampled_descriptors_b,
                                                             torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto scores_b = torch::ones({keypoints_b.size(0)}, point_options.device(warp.device()));
    auto scale = torch::ones({keypoints_b.size(0)}, point_options.device(warp.device()));
    auto orientation = torch::zeros({keypoints_b.size(0)}, point_options.device(warp.device()));
    auto affine = torch::zeros({keypoints_b.size(0), 4}, point_options.device(warp.device()));

    FeatureSet completed_a{kept_keypoints_a,
                           kept_scores_a,
                           kept_descriptors_a,
                           scale,
                           orientation,
                           affine,
                           kept_keypoints_a,
                           kept_scores_a,
                           features_a.feature_map_width,
                           features_a.feature_map_height};
    FeatureSet completed_b{keypoints_b,      scores_b,         sampled_descriptors_b, scale,
                           orientation,      affine,           keypoints_b,           scores_b,
                           descriptor_width, descriptor_height};
    return {completed_a, completed_b};
}

template <typename GraphMatcherT>
torch::Tensor make_keypoint_graph_matching_loss(GraphMatcherT& graph_matcher, const FeatureSet& features_a,
                                                const FeatureSet& features_b, const torch::Tensor& warp,
                                                const torch::Tensor& valid_mask)
{
    return make_keypoint_graph_matching_metrics(graph_matcher, features_a, features_b, warp, valid_mask).loss;
}

torch::Tensor make_sparse_descriptor_loss(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                                          const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_sparse_descriptor_metrics(descriptors_a, descriptors_b, warp, valid_mask).loss;
}

template <typename GraphMatcherT>
torch::Tensor make_graph_matching_loss(GraphMatcherT& graph_matcher, const torch::Tensor& descriptors_a,
                                       const torch::Tensor& descriptors_b, const torch::Tensor& warp,
                                       const torch::Tensor& valid_mask)
{
    auto sample_indices = make_balanced_descriptor_sample_indices(
        valid_mask, descriptors_a.size(2), descriptors_a.size(3), GRAPH_DENSE_MATCHING_MAX_SAMPLES);
    if (sample_indices.numel() == 0)
    {
        return torch::zeros({}, descriptors_a.options());
    }
    auto target_spatial =
        make_descriptor_target_indices(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    const auto batch_size = descriptors_a.size(0);
    const auto sample_count = sample_indices.size(0);
    const auto descriptor_dim = descriptors_b.size(1);
    const auto spatial_count = descriptors_b.size(2) * descriptors_b.size(3);
    auto flat_b = descriptors_b.permute({0, 2, 3, 1}).contiguous().reshape({batch_size, spatial_count, descriptor_dim});
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    std::vector<torch::Tensor> sampled_b_batches;
    sampled_b_batches.reserve(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b)
    {
        sampled_b_batches.push_back(flat_b[b].index_select(0, target_spatial[b]));
    }
    auto sampled_b = torch::stack(sampled_b_batches, 0);
    auto target_columns = torch::arange(sample_count, sample_indices.options());
    std::vector<torch::Tensor> losses;
    losses.reserve(static_cast<size_t>(batch_size));
    auto keypoints_a =
        torch::stack({sample_indices.remainder(descriptors_a.size(3)).to(descriptors_a.dtype()),
                      torch::floor_divide(sample_indices, descriptors_a.size(3)).to(descriptors_a.dtype())},
                     1);
    auto desc_w = static_cast<int64_t>(descriptors_b.size(3));
    for (int64_t batch = 0; batch < batch_size; ++batch)
    {
        auto kp_b = torch::stack({target_spatial[batch].remainder(desc_w).to(descriptors_b.dtype()),
                                  (target_spatial[batch] / desc_w).to(descriptors_b.dtype())},
                                 1);
        auto output = graph_matcher.forward(sampled_a[batch], keypoints_a, sampled_b[batch], kp_b);
        losses.push_back(graph_matching_cross_entropy_loss(output.logits, target_columns));
    }
    return torch::stack(losses).mean();
}

torch::Tensor resize_mask_for_heatmap(const torch::Tensor& valid_mask, const torch::Tensor& heatmap)
{
    auto mask = valid_mask.to(heatmap.dtype()).unsqueeze(1);
    return torch::nn::functional::interpolate(mask, torch::nn::functional::InterpolateFuncOptions()
                                                        .size(std::vector<int64_t>{heatmap.size(2), heatmap.size(3)})
                                                        .mode(torch::kNearest));
}

torch::Tensor make_batch_intensity_mask(const torch::Tensor& batch, double min_keypoint_intensity)
{
    std::vector<torch::Tensor> masks;
    masks.reserve(static_cast<size_t>(batch.size(0)));
    for (int64_t index = 0; index < batch.size(0); ++index)
    {
        masks.push_back(make_intensity_mask(batch[index], min_keypoint_intensity));
    }
    return torch::stack(masks).to(batch.device());
}

torch::Tensor estimate_warp_x_axis(const torch::Tensor& warp)
{
    const auto options = warp.options().dtype(torch::kFloat32);
    if (warp.size(2) < 2)
    {
        auto fallback = torch::zeros({warp.size(0), 2}, options);
        fallback.index_put_({torch::indexing::Slice(), 0}, 1.0F);
        return fallback;
    }

    auto x_axis = warp.index({torch::indexing::Slice(), 0, 1, torch::indexing::Slice()}) -
                  warp.index({torch::indexing::Slice(), 0, 0, torch::indexing::Slice()});
    return x_axis / x_axis.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
}

torch::Tensor orientation_target_loss(const torch::Tensor& orientation, const torch::Tensor& target_vectors,
                                      const torch::Tensor& mask)
{
    auto mask_float = mask.to(orientation.dtype());
    auto denom = mask_float.sum();
    if (denom.item<double>() <= 0.0)
    {
        return torch::zeros({}, orientation.options());
    }
    auto target = target_vectors.to(orientation.device(), orientation.dtype()).reshape({orientation.size(0), 2, 1, 1});
    target = target / target.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
    auto cosine = (orientation * target).sum(1, true).clamp(-1.0, 1.0);
    return ((1.0 - cosine) * mask_float).sum() / denom;
}

torch::Tensor make_orientation_supervision_loss(const SparseHeadOutput& sparse_a, const SparseHeadOutput& sparse_b,
                                                const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                const torch::Tensor& warp, double min_keypoint_intensity)
{
    auto mask_a =
        resize_mask_for_heatmap(make_batch_intensity_mask(view_a, min_keypoint_intensity), sparse_a.orientation);
    auto mask_b =
        resize_mask_for_heatmap(make_batch_intensity_mask(view_b, min_keypoint_intensity), sparse_b.orientation);
    auto target_a = torch::zeros({view_a.size(0), 2}, warp.options().dtype(torch::kFloat32));
    target_a.index_put_({torch::indexing::Slice(), 0}, 1.0F);
    auto target_b = estimate_warp_x_axis(warp);
    return (orientation_target_loss(sparse_a.orientation, target_a, mask_a) +
            orientation_target_loss(sparse_b.orientation, target_b, mask_b)) *
           0.5;
}

torch::Tensor warp_mask_to_view_b(const torch::Tensor& view_b_mask, const torch::Tensor& warp)
{
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
    return torch::nn::functional::grid_sample(view_b_mask.unsqueeze(1).to(torch::kFloat32), grid,
                                              torch::nn::functional::GridSampleFuncOptions()
                                                  .mode(torch::kNearest)
                                                  .padding_mode(torch::kZeros)
                                                  .align_corners(true))
        .squeeze(1)
        .gt(0.0);
}

torch::Tensor make_training_valid_mask(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                       const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                       double min_keypoint_intensity)
{
    std::vector<torch::Tensor> masks_a;
    std::vector<torch::Tensor> masks_b;
    masks_a.reserve(static_cast<size_t>(view_a.size(0)));
    masks_b.reserve(static_cast<size_t>(view_b.size(0)));
    for (int64_t index = 0; index < view_a.size(0); ++index)
    {
        masks_a.push_back(make_intensity_mask(view_a[index], min_keypoint_intensity));
        masks_b.push_back(make_intensity_mask(view_b[index], min_keypoint_intensity));
    }
    const auto mask_a = torch::stack(masks_a).to(valid_mask.device()).to(torch::kBool);
    const auto mask_b = torch::stack(masks_b).to(valid_mask.device()).to(torch::kBool);
    return valid_mask.to(torch::kBool).logical_and(mask_a).logical_and(warp_mask_to_view_b(mask_b, warp));
}

torch::Tensor make_pair_loss_valid_mask(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                        const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                        double min_keypoint_intensity, TrainingProfile training_profile)
{
    if (training_profile_uses_python_aligned_pair_loss(training_profile))
    {
        return valid_mask.to(torch::kBool).contiguous();
    }
    return make_training_valid_mask(view_a, view_b, warp, valid_mask, min_keypoint_intensity);
}

torch::Tensor warp_heatmap_for_repeatability(const torch::Tensor& heatmap, const torch::Tensor& warp)
{
    using torch::indexing::Slice;

    auto resized_warp = torch::nn::functional::interpolate(
        warp.permute({0, 3, 1, 2}), torch::nn::functional::InterpolateFuncOptions()
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
    return torch::nn::functional::grid_sample(heatmap, grid,
                                              torch::nn::functional::GridSampleFuncOptions()
                                                  .mode(torch::kBilinear)
                                                  .padding_mode(torch::kZeros)
                                                  .align_corners(true));
}

torch::Tensor resize_offsets_for_dense_head(const torch::Tensor& warp, const torch::Tensor& offsets)
{
    using torch::indexing::Slice;

    auto source_grid = make_xy_grid(warp.size(1), warp.size(2), warp.device()).unsqueeze(0).to(warp.dtype());
    auto displacement = (warp - source_grid).permute({0, 3, 1, 2}).to(offsets.dtype()).contiguous();
    auto resized = torch::nn::functional::interpolate(displacement,
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

struct TrainModules
{
    v21::PfmV21Backbone backbone{nullptr};
    v21::PfmV21DualFPNLite dual_fpn{nullptr};
    v21::PfmV21SparseHead sparse_head{nullptr};
    v21::PfmV21TextureDescriptorAdapter texture_adapter{nullptr};
    v21::PfmV21DescriptorFusionAdapter descriptor_fusion{nullptr};
    v21::PfmV21DenseHead dense_head{nullptr};
    v21::PfmV21QualityHead quality_head{nullptr};
    v21::PfmV21SemiDenseCandidateBranch semi_dense_branch{nullptr};
    v21::PfmV21GraphMatcher graph_matcher{nullptr};
};

void set_all_modules_train(TrainModules& modules)
{
    modules.backbone->train();
    modules.dual_fpn->train();
    modules.sparse_head->train();
    modules.texture_adapter->train();
    modules.descriptor_fusion->train();
    modules.dense_head->train();
    modules.quality_head->train();
    modules.semi_dense_branch->train();
    modules.graph_matcher->train();
}

TrainModules make_modules(const TrainConfig& config, torch::Device device)
{
    TrainModules modules;
    modules.backbone = v21::PfmV21Backbone(INPUT_CHANNELS, config.base_channels);
    modules.dual_fpn = v21::PfmV21DualFPNLite(config.base_channels);
    modules.sparse_head = v21::PfmV21SparseHead(config.base_channels * 2, config.descriptor_dim);
    modules.texture_adapter = v21::PfmV21TextureDescriptorAdapter(config.descriptor_dim);
    modules.descriptor_fusion = v21::PfmV21DescriptorFusionAdapter(config.descriptor_dim);
    modules.dense_head = v21::PfmV21DenseHead(config.base_channels);
    modules.quality_head = v21::PfmV21QualityHead(config.descriptor_dim);
    modules.semi_dense_branch = v21::PfmV21SemiDenseCandidateBranch(config.descriptor_dim);
    modules.graph_matcher = v21::PfmV21GraphMatcher(config.descriptor_dim, config.graph_hidden_dim,
                                                    config.graph_attention_layers, config.graph_keypoint_meta_dim);
    modules.backbone->to(device);
    modules.dual_fpn->to(device);
    modules.sparse_head->to(device);
    modules.texture_adapter->to(device);
    modules.descriptor_fusion->to(device);
    modules.dense_head->to(device);
    modules.quality_head->to(device);
    modules.semi_dense_branch->to(device);
    modules.graph_matcher->to(device);
    set_all_modules_train(modules);
    return modules;
}

SparseHeadOutput adapt_v21_sparse_output(v21::PfmV21SparseHeadOutput output)
{
    return SparseHeadOutput{output.heatmap, output.descriptors, output.scale, output.orientation, output.affine};
}

DenseHeadOutput adapt_v21_dense_output(v21::PfmV21DenseHeadOutput output)
{
    return DenseHeadOutput{output.confidence, output.offsets};
}

torch::Tensor resize_confidence_for_sparse_output(const torch::Tensor& confidence, const torch::Tensor& heatmap)
{
    if (confidence.size(2) == heatmap.size(2) && confidence.size(3) == heatmap.size(3))
    {
        return confidence;
    }
    return torch::nn::functional::interpolate(confidence,
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{heatmap.size(2), heatmap.size(3)})
                                                  .mode(torch::kNearest));
}

SparseHeadOutput finalize_v21_sparse_output(TrainModules& modules, v21::PfmV21SparseHeadOutput output,
                                            const torch::Tensor& image, const torch::Tensor& dense_confidence,
                                            double texture_blend_weight = ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT)
{
    auto sparse = adapt_v21_sparse_output(std::move(output));
    auto texture = v21::makeRotationInvariantTextureDescriptor(image, sparse.descriptors.size(2),
                                                               sparse.descriptors.size(3), sparse.descriptors.size(1));
    texture = modules.texture_adapter->forward(texture);
    sparse.descriptors = modules.descriptor_fusion->forward(sparse.descriptors, texture, texture_blend_weight);
    const auto texture_saliency =
        v21::makeRotationInvariantTextureSaliency(image, sparse.heatmap.size(2), sparse.heatmap.size(3));
    const auto confidence = resize_confidence_for_sparse_output(dense_confidence, sparse.heatmap);
    const auto quality =
        modules.quality_head->forward(sparse.descriptors, sparse.heatmap, texture_saliency, confidence);
    sparse.heatmap = (sparse.heatmap * quality).clamp(0.0, 1.0);
    return sparse;
}

SparseHeadOutput finalize_v21_python_aligned_sparse_output(TrainModules& modules, v21::PfmV21SparseHeadOutput output,
                                                           const torch::Tensor& image,
                                                           double texture_blend_weight)
{
    auto sparse = adapt_v21_sparse_output(std::move(output));
    auto texture = v21::makeRotationInvariantTextureDescriptor(image, sparse.descriptors.size(2),
                                                               sparse.descriptors.size(3), sparse.descriptors.size(1));
    texture = modules.texture_adapter->forward(texture);
    sparse.descriptors = modules.descriptor_fusion->forward(sparse.descriptors, texture, texture_blend_weight);
    return sparse;
}

DenseHeadOutput make_zero_dense_output_like_sparse(const SparseHeadOutput& sparse)
{
    return DenseHeadOutput{
        torch::zeros({sparse.heatmap.size(0), 1, sparse.heatmap.size(2), sparse.heatmap.size(3)},
                     sparse.heatmap.options()),
        torch::zeros({sparse.heatmap.size(0), 2, sparse.heatmap.size(2), sparse.heatmap.size(3)},
                     sparse.heatmap.options())};
}

FeatureDecodeConfig make_training_decode_config(const TrainConfig& config)
{
    FeatureDecodeConfig decode_config;
    decode_config.max_keypoints = std::min(config.max_keypoints, TRAINING_DECODE_MAX_KEYPOINTS);
    decode_config.min_keypoints = std::min(config.min_keypoints, decode_config.max_keypoints);
    decode_config.keypoint_grid_rows = config.keypoint_grid_rows;
    decode_config.keypoint_grid_cols = config.keypoint_grid_cols;
    decode_config.keypoints_per_cell = config.keypoints_per_cell;
    decode_config.nms_radius = config.nms_radius;
    return decode_config;
}

FeatureSet decode_training_features(const torch::Tensor& view, const SparseHeadOutput& sparse,
                                    const torch::Tensor& dense_confidence, const TrainConfig& config)
{
    RawFeatureMaps maps{sparse.heatmap.detach().cpu().contiguous(), sparse.descriptors.detach().cpu().contiguous(),
                        sparse.scale.detach().cpu().contiguous(),   sparse.orientation.detach().cpu().contiguous(),
                        sparse.affine.detach().cpu().contiguous(),  dense_confidence.detach().cpu().contiguous()};
    auto features = decode_feature_maps(maps, make_training_decode_config(config),
                                        make_visualization_mask(view, config.min_keypoint_intensity));
    features.feature_map_width = maps.heatmap.size(3);
    features.feature_map_height = maps.heatmap.size(2);
    return features;
}

torch::Tensor make_texture_saliency_target(const torch::Tensor& image, int64_t target_height, int64_t target_width);

torch::Tensor make_training_decode_mask_gpu(const torch::Tensor& image, double min_keypoint_intensity)
{
    auto intensity = image.to(torch::kFloat32).mean(0).contiguous();
    auto bright = intensity.ge(min_keypoint_intensity);
    if (min_keypoint_intensity <= 0.0 || intensity.size(0) < 7 || intensity.size(1) < 7)
    {
        return bright;
    }

    const int64_t kernel = 7;
    auto local_support = torch::nn::functional::avg_pool2d(
        bright.to(torch::kFloat32).reshape({1, 1, intensity.size(0), intensity.size(1)}),
        torch::nn::functional::AvgPool2dFuncOptions({kernel, kernel})
            .stride(1)
            .padding(kernel / 2)
            .count_include_pad(false));
    auto local_mean = torch::nn::functional::avg_pool2d(intensity.reshape({1, 1, intensity.size(0), intensity.size(1)}),
                                                        torch::nn::functional::AvgPool2dFuncOptions({kernel, kernel})
                                                            .stride(1)
                                                            .padding(kernel / 2)
                                                            .count_include_pad(false));
    local_support = local_support.reshape({intensity.size(0), intensity.size(1)}).ge(0.25);
    local_mean = local_mean.reshape({intensity.size(0), intensity.size(1)}).ge(min_keypoint_intensity);
    return bright.logical_and(local_support).logical_and(local_mean);
}

FeatureSet decode_training_features_fast(const torch::Tensor& view, const SparseHeadOutput& sparse,
                                         const TrainConfig& config)
{
    const auto height = sparse.heatmap.size(2);
    const auto width = sparse.heatmap.size(3);
    auto mask = make_training_decode_mask_gpu(view, config.min_keypoint_intensity)
                    .to(sparse.heatmap.device(), torch::kFloat32)
                    .reshape({1, 1, view.size(1), view.size(2)});
    if (view.size(1) != height || view.size(2) != width)
    {
        mask = torch::nn::functional::interpolate(mask, torch::nn::functional::InterpolateFuncOptions()
                                                            .size(std::vector<int64_t>{height, width})
                                                            .mode(torch::kNearest));
    }

    const auto spatial_count = height * width;
    auto saliency = make_texture_saliency_target(view.unsqueeze(0).to(sparse.heatmap.device(), sparse.heatmap.dtype()),
                                                 height, width);
    auto score_map = (sparse.heatmap * 0.5F + saliency * 0.5F).reshape({height, width});
    auto valid_map = mask.reshape({height, width}).gt(0.0);
    if (config.nms_radius > 0)
    {
        const auto kernel = static_cast<int64_t>(config.nms_radius) * 2 + 1;
        auto local_max =
            torch::max_pool2d(score_map.reshape({1, 1, height, width}), {kernel, kernel}, {1, 1},
                              {static_cast<int64_t>(config.nms_radius), static_cast<int64_t>(config.nms_radius)});
        valid_map = valid_map.logical_and(score_map.eq(local_max.reshape({height, width})));
    }
    const auto cell_count = std::max<int64_t>(1, config.keypoint_grid_rows * config.keypoint_grid_cols);
    const auto per_cell =
        config.keypoints_per_cell > 0
            ? static_cast<int64_t>(config.keypoints_per_cell)
            : std::max<int64_t>(1, (static_cast<int64_t>(config.max_keypoints) + cell_count - 1) / cell_count);
    std::vector<torch::Tensor> selected_scores;
    std::vector<torch::Tensor> selected_indices;
    for (int row = 0; row < config.keypoint_grid_rows; ++row)
    {
        const auto y0 = static_cast<int64_t>(row) * height / config.keypoint_grid_rows;
        const auto y1 = static_cast<int64_t>(row + 1) * height / config.keypoint_grid_rows;
        if (y1 <= y0)
        {
            continue;
        }
        for (int col = 0; col < config.keypoint_grid_cols; ++col)
        {
            const auto x0 = static_cast<int64_t>(col) * width / config.keypoint_grid_cols;
            const auto x1 = static_cast<int64_t>(col + 1) * width / config.keypoint_grid_cols;
            if (x1 <= x0)
            {
                continue;
            }
            auto cell_scores = score_map.index({torch::indexing::Slice(y0, y1), torch::indexing::Slice(x0, x1)})
                                   .reshape({(y1 - y0) * (x1 - x0)});
            auto cell_valid = valid_map.index({torch::indexing::Slice(y0, y1), torch::indexing::Slice(x0, x1)})
                                  .reshape({(y1 - y0) * (x1 - x0)});
            auto masked_cell_scores =
                cell_scores.masked_fill(cell_valid.logical_not(), -std::numeric_limits<float>::infinity());
            const auto k = std::min<int64_t>(per_cell, masked_cell_scores.numel());
            auto topk = masked_cell_scores.topk(k);
            auto top_scores = std::get<0>(topk);
            auto local_indices = std::get<1>(topk).to(torch::kLong);
            auto finite = torch::isfinite(top_scores);
            if (!finite.any().item<bool>())
            {
                continue;
            }
            top_scores = top_scores.index({finite}).contiguous();
            local_indices = local_indices.index({finite}).contiguous();
            auto local_x = local_indices.remainder(x1 - x0);
            auto local_y = torch::floor_divide(local_indices, x1 - x0);
            auto global_indices = (local_y + y0) * width + (local_x + x0);
            selected_scores.push_back(top_scores);
            selected_indices.push_back(global_indices.to(torch::kLong).contiguous());
        }
    }
    torch::Tensor top_scores;
    torch::Tensor top_indices;
    if (selected_indices.empty())
    {
        top_scores = torch::empty({0}, sparse.heatmap.options());
        top_indices = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(sparse.heatmap.device()));
    }
    else
    {
        top_scores = torch::cat(selected_scores, 0);
        top_indices = torch::cat(selected_indices, 0);
        if (top_indices.size(0) > config.max_keypoints)
        {
            auto keep_topk = top_scores.topk(config.max_keypoints);
            auto keep = std::get<1>(keep_topk).to(torch::kLong);
            top_scores = std::get<0>(keep_topk).contiguous();
            top_indices = top_indices.index_select(0, keep).contiguous();
        }
    }

    auto xs = top_indices.remainder(width).to(torch::kFloat32);
    auto ys = torch::floor_divide(top_indices, width).to(torch::kFloat32);
    auto keypoints = torch::stack({xs, ys}, 1).contiguous();
    auto descriptor_map = sparse.descriptors;
    if (descriptor_map.size(2) != height || descriptor_map.size(3) != width)
    {
        descriptor_map =
            torch::nn::functional::interpolate(descriptor_map, torch::nn::functional::InterpolateFuncOptions()
                                                                   .size(std::vector<int64_t>{height, width})
                                                                   .mode(torch::kBilinear)
                                                                   .align_corners(false));
    }
    descriptor_map = torch::nn::functional::normalize(
        descriptor_map, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12));
    auto flat_descriptors =
        descriptor_map.squeeze(0).permute({1, 2, 0}).reshape({spatial_count, descriptor_map.size(1)});
    auto safe_indices = top_indices.clamp(0, spatial_count - 1);
    auto descriptors = flat_descriptors.index_select(0, safe_indices).contiguous();
    return FeatureSet{keypoints,
                      top_scores,
                      descriptors,
                      torch::empty({top_indices.size(0)}, descriptors.options()),
                      torch::empty({top_indices.size(0), 2}, descriptors.options()),
                      torch::empty({top_indices.size(0), 2, 2}, descriptors.options()),
                      torch::empty({0, 2}, descriptors.options()),
                      torch::empty({0}, descriptors.options()),
                      width,
                      height};
}

template <typename ModuleHolderT>
void append_module_parameters(ModuleHolderT& module, std::vector<torch::Tensor>& parameters)
{
    for (auto& parameter : module->parameters())
    {
        if (parameter.requires_grad())
        {
            parameters.push_back(parameter);
        }
    }
}

template <typename ModuleHolderT>
void append_trainable_parameter_names(const std::string& prefix, ModuleHolderT& module, std::vector<std::string>& names)
{
    for (auto& named_parameter : module->named_parameters())
    {
        if (named_parameter.value().requires_grad())
        {
            names.push_back(prefix + "." + named_parameter.key());
        }
    }
}

std::vector<torch::Tensor> module_parameters(TrainModules& modules)
{
    std::vector<torch::Tensor> parameters;
    append_module_parameters(modules.backbone, parameters);
    append_module_parameters(modules.dual_fpn, parameters);
    append_module_parameters(modules.sparse_head, parameters);
    append_module_parameters(modules.texture_adapter, parameters);
    append_module_parameters(modules.descriptor_fusion, parameters);
    append_module_parameters(modules.dense_head, parameters);
    append_module_parameters(modules.quality_head, parameters);
    append_module_parameters(modules.semi_dense_branch, parameters);
    append_module_parameters(modules.graph_matcher, parameters);
    return parameters;
}

std::vector<std::string> trainable_parameter_names(TrainModules& modules)
{
    std::vector<std::string> names;
    append_trainable_parameter_names("backbone", modules.backbone, names);
    append_trainable_parameter_names("dual_fpn", modules.dual_fpn, names);
    append_trainable_parameter_names("sparse_head", modules.sparse_head, names);
    append_trainable_parameter_names("texture_adapter", modules.texture_adapter, names);
    append_trainable_parameter_names("descriptor_fusion", modules.descriptor_fusion, names);
    append_trainable_parameter_names("dense_head", modules.dense_head, names);
    append_trainable_parameter_names("quality_head", modules.quality_head, names);
    append_trainable_parameter_names("semi_dense_branch", modules.semi_dense_branch, names);
    append_trainable_parameter_names("graph_matcher", modules.graph_matcher, names);
    return names;
}

void set_module_trainable(torch::nn::Module& module, bool trainable)
{
    for (auto& parameter : module.parameters())
    {
        parameter.set_requires_grad(trainable);
    }
}

bool is_sparse_descriptor_parameter(const std::string& name)
{
    return name.rfind("descriptors.", 0) == 0 || name.rfind("descriptor_", 0) == 0;
}

bool is_sparse_viewpoint_descriptor_parameter(const std::string& name)
{
    return name.rfind("descriptor_viewpoint_", 0) == 0;
}

bool is_sparse_context_parameter(const std::string& name)
{
    return name.rfind("context.", 0) == 0 || name.rfind("descriptor_context.", 0) == 0 ||
           name.rfind("geometry_context.", 0) == 0;
}

bool is_sparse_keypoint_parameter(const std::string& name)
{
    return name.rfind("heatmap.", 0) == 0 || name.rfind("keypoint_context.", 0) == 0 ||
           name.rfind("keypoint_offsets.", 0) == 0;
}

bool is_sparse_geometry_parameter(const std::string& name)
{
    return name.rfind("scale.", 0) == 0 || name.rfind("orientation.", 0) == 0 ||
           name.rfind("affine.", 0) == 0 || name.rfind("geometry_context.", 0) == 0;
}

bool is_python_compare_sparse_parameter(const std::string& name, const TrainConfig& config)
{
    bool trainable = !config.freeze_descriptor_head && is_sparse_descriptor_parameter(name);
    trainable = trainable || (config.train_sparse_context && is_sparse_context_parameter(name));
    trainable = trainable || (config.train_keypoint_head && is_sparse_keypoint_parameter(name));
    trainable = trainable || (config.train_geometry_head && is_sparse_geometry_parameter(name));
    return trainable;
}

bool has_explicit_finetune_mode(const TrainConfig& config)
{
    return config.descriptor_only_finetune || config.viewpoint_head_only_finetune || config.graph_only_finetune;
}

void apply_descriptor_only_finetune(TrainModules& modules, const TrainConfig& config)
{
    if (!config.descriptor_only_finetune)
    {
        return;
    }
    set_module_trainable(*modules.backbone, false);
    set_module_trainable(*modules.dual_fpn, false);
    set_module_trainable(*modules.dense_head, false);
    set_module_trainable(*modules.quality_head, false);
    set_module_trainable(*modules.semi_dense_branch, false);
    for (auto& named_parameter : modules.sparse_head->named_parameters())
    {
        named_parameter.value().set_requires_grad(is_sparse_descriptor_parameter(named_parameter.key()));
    }
    set_module_trainable(*modules.texture_adapter, true);
    set_module_trainable(*modules.descriptor_fusion, true);
    set_module_trainable(*modules.graph_matcher, false);
}

void apply_viewpoint_head_only_finetune(TrainModules& modules, const TrainConfig& config)
{
    if (!config.viewpoint_head_only_finetune)
    {
        return;
    }
    set_module_trainable(*modules.backbone, false);
    set_module_trainable(*modules.dual_fpn, false);
    set_module_trainable(*modules.dense_head, false);
    set_module_trainable(*modules.texture_adapter, false);
    set_module_trainable(*modules.descriptor_fusion, false);
    set_module_trainable(*modules.quality_head, false);
    set_module_trainable(*modules.semi_dense_branch, false);
    for (auto& named_parameter : modules.sparse_head->named_parameters())
    {
        named_parameter.value().set_requires_grad(is_sparse_viewpoint_descriptor_parameter(named_parameter.key()));
    }
    set_module_trainable(*modules.graph_matcher, false);
}

void keep_descriptor_only_frozen_modules_eval(TrainModules& modules, const TrainConfig& config)
{
    if (!config.descriptor_only_finetune && !config.viewpoint_head_only_finetune)
    {
        return;
    }
    modules.backbone->eval();
    modules.dual_fpn->eval();
    modules.dense_head->eval();
    modules.quality_head->eval();
    modules.semi_dense_branch->eval();
    modules.graph_matcher->eval();
}

void apply_graph_only_finetune(TrainModules& modules, const TrainConfig& config)
{
    if (!config.graph_only_finetune)
    {
        return;
    }
    set_module_trainable(*modules.backbone, false);
    set_module_trainable(*modules.dual_fpn, false);
    set_module_trainable(*modules.sparse_head, false);
    set_module_trainable(*modules.texture_adapter, false);
    set_module_trainable(*modules.descriptor_fusion, false);
    set_module_trainable(*modules.dense_head, false);
    set_module_trainable(*modules.quality_head, false);
    set_module_trainable(*modules.semi_dense_branch, false);
    set_module_trainable(*modules.graph_matcher, true);
}

void keep_graph_only_frozen_modules_eval(TrainModules& modules, const TrainConfig& config)
{
    if (!config.graph_only_finetune)
    {
        return;
    }
    modules.backbone->eval();
    modules.dual_fpn->eval();
    modules.sparse_head->eval();
    modules.texture_adapter->eval();
    modules.descriptor_fusion->eval();
    modules.dense_head->eval();
    modules.quality_head->eval();
    modules.semi_dense_branch->eval();
}

void set_all_modules_trainable(TrainModules& modules, bool trainable)
{
    set_module_trainable(*modules.backbone, trainable);
    set_module_trainable(*modules.dual_fpn, trainable);
    set_module_trainable(*modules.sparse_head, trainable);
    set_module_trainable(*modules.texture_adapter, trainable);
    set_module_trainable(*modules.descriptor_fusion, trainable);
    set_module_trainable(*modules.dense_head, trainable);
    set_module_trainable(*modules.quality_head, trainable);
    set_module_trainable(*modules.semi_dense_branch, trainable);
    set_module_trainable(*modules.graph_matcher, trainable);
}

void set_all_modules_eval(TrainModules& modules)
{
    modules.backbone->eval();
    modules.dual_fpn->eval();
    modules.sparse_head->eval();
    modules.texture_adapter->eval();
    modules.descriptor_fusion->eval();
    modules.dense_head->eval();
    modules.quality_head->eval();
    modules.semi_dense_branch->eval();
    modules.graph_matcher->eval();
}

void apply_training_profile_module_mode(TrainModules& modules, const TrainConfig& config)
{
    if (training_profile_uses_python_aligned_pair_loss(parse_training_profile(config.training_profile)))
    {
        // Python load_pytorch_state() leaves the model in eval mode and the training loop updates parameters in that
        // mode. Keep BatchNorm statistics frozen and GraphMatcher dropout disabled for apples-to-apples C++ training.
        set_all_modules_eval(modules);
    }
}

void apply_python_compare_trainable_selection(TrainModules& modules, const TrainConfig& config)
{
    const bool use_python_compare_profile =
        parse_training_profile(config.training_profile) == TrainingProfile::PythonCompare;
    if (!use_python_compare_profile || has_explicit_finetune_mode(config))
    {
        return;
    }

    // Mirrors python/pfm_pytorch_training.py::descriptor_parameters for python-compare runs.
    set_all_modules_trainable(modules, false);
    set_module_trainable(*modules.backbone, config.train_backbone);
    set_module_trainable(*modules.dual_fpn, config.train_dual_fpn);
    for (auto& named_parameter : modules.sparse_head->named_parameters())
    {
        named_parameter.value().set_requires_grad(is_python_compare_sparse_parameter(named_parameter.key(), config));
    }
    set_module_trainable(*modules.texture_adapter, config.train_texture_adapter);
    set_module_trainable(*modules.descriptor_fusion, config.train_descriptor_fusion);
    set_module_trainable(*modules.quality_head, config.train_quality_head);
    set_module_trainable(*modules.graph_matcher, config.train_graph_matcher);
}

void apply_trainable_parameter_selection(TrainModules& modules, const TrainConfig& config)
{
    apply_python_compare_trainable_selection(modules, config);
    apply_descriptor_only_finetune(modules, config);
    apply_viewpoint_head_only_finetune(modules, config);
    apply_graph_only_finetune(modules, config);
}

int64_t count_trainable_parameter_values(const std::vector<torch::Tensor>& parameters)
{
    int64_t count = 0;
    for (const auto& parameter : parameters)
    {
        count += parameter.numel();
    }
    return count;
}

double effective_python_compare_graph_loss_weight(const TrainConfig& config)
{
    return (config.train_graph_matcher || config.graph_only_finetune) ? config.graph_matcher_loss_weight : 0.0;
}

bool should_use_descriptor_finetune_anchor(const TrainConfig& config)
{
    return (config.descriptor_only_finetune || config.viewpoint_head_only_finetune) && !config.init_checkpoint.empty();
}

SyntheticPairConfig make_default_pair_config()
{
    SyntheticPairConfig pair_config;
    pair_config.noise_sigma = 0.01F;
    return pair_config;
}

std::vector<SyntheticPair> make_synthetic_pairs_from_batch(const torch::Tensor& batch,
                                                           const std::vector<int64_t>& source_indices,
                                                           const std::vector<int64_t>& variant_indices,
                                                           const SyntheticPairConfig& pair_config)
{
    std::vector<SyntheticPair> pairs;
    pairs.reserve(static_cast<size_t>(batch.size(0)));
    for (int64_t index = 0; index < batch.size(0); ++index)
    {
        auto variant_config = pair_config;
        variant_config.source_index = source_indices[static_cast<std::size_t>(index)];
        variant_config.variant_index = variant_indices[static_cast<std::size_t>(index)];
        pairs.push_back(make_synthetic_pair(batch[index], variant_config));
    }
    return pairs;
}

SyntheticPair move_pair_to_device(const SyntheticPair& pair, torch::Device device)
{
    return SyntheticPair{pair.view_a.to(device), pair.view_b.to(device), pair.warp_a_to_b.to(device),
                         pair.valid_mask.to(device)};
}

class CompositeSyntheticPairCacheDataset
{
  public:
    explicit CompositeSyntheticPairCacheDataset(const std::vector<TrainingCacheSpec>& cache_specs)
    {
        if (cache_specs.empty())
        {
            throw std::invalid_argument("cache_specs must not be empty");
        }
        std::size_t offset = 0;
        for (const auto& cache_spec : cache_specs)
        {
            auto dataset = std::make_unique<SyntheticPairCacheDataset>(cache_spec.cache_dir);
            const auto dataset_size = dataset->size();
            if (dataset_size == 0)
            {
                continue;
            }
            if (cache_spec.pair_index.has_value())
            {
                if (*cache_spec.pair_index >= dataset_size)
                {
                    throw std::invalid_argument("hard synthetic pair cache index is out of range: " +
                                                std::to_string(*cache_spec.pair_index));
                }
                offset += 1;
            }
            else
            {
                offset += dataset_size;
            }
            _offsets.push_back(offset);
            _explicit_pair_indices.push_back(cache_spec.pair_index);
            _datasets.push_back(std::move(dataset));
        }
        if (_datasets.empty())
        {
            throw std::invalid_argument("combined synthetic pair cache is empty");
        }
    }

    std::size_t size() const
    {
        return _offsets.back();
    }

    SyntheticPair load(std::size_t index) const
    {
        if (index >= size())
        {
            throw std::out_of_range("synthetic pair cache index out of range");
        }
        const auto dataset_index =
            static_cast<std::size_t>(std::upper_bound(_offsets.begin(), _offsets.end(), index) - _offsets.begin());
        const auto previous_offset = dataset_index == 0 ? 0 : _offsets[dataset_index - 1];
        if (_explicit_pair_indices[dataset_index].has_value())
        {
            return _datasets[dataset_index]->load(*_explicit_pair_indices[dataset_index]);
        }
        return _datasets[dataset_index]->load(index - previous_offset);
    }

  private:
    std::vector<std::unique_ptr<SyntheticPairCacheDataset>> _datasets;
    std::vector<std::size_t> _offsets;
    std::vector<std::optional<std::size_t>> _explicit_pair_indices;
};

class CompositeTensorDataset : public TensorDataset
{
  public:
    explicit CompositeTensorDataset(std::vector<std::shared_ptr<TensorDataset>> datasets)
        : _datasets(std::move(datasets))
    {
        if (_datasets.empty())
        {
            throw std::invalid_argument("datasets must not be empty");
        }
        std::size_t offset = 0;
        for (const auto& dataset : _datasets)
        {
            if (!dataset)
            {
                throw std::invalid_argument("datasets must not contain null entries");
            }
            offset += dataset->size();
            _offsets.push_back(offset);
        }
        if (offset == 0)
        {
            throw std::invalid_argument("combined tensor dataset is empty");
        }
    }

    size_t size() const override
    {
        return _offsets.back();
    }

    TensorBatch get(size_t index) override
    {
        if (index >= size())
        {
            throw std::out_of_range("combined tensor dataset index out of range");
        }
        const auto dataset_index =
            static_cast<std::size_t>(std::upper_bound(_offsets.begin(), _offsets.end(), index) - _offsets.begin());
        const auto previous_offset = dataset_index == 0 ? 0 : _offsets[dataset_index - 1];
        return _datasets[dataset_index]->get(index - previous_offset);
    }

  private:
    std::vector<std::shared_ptr<TensorDataset>> _datasets;
    std::vector<std::size_t> _offsets;
};

TensorBatch cached_pair_to_tensor_batch(const SyntheticPair& pair)
{
    TensorBatch batch;
    batch["view_a"] = pair.view_a;
    batch["view_b"] = pair.view_b;
    batch["warp_a_to_b"] = pair.warp_a_to_b;
    batch["valid_mask"] = pair.valid_mask;
    return batch;
}

SyntheticPair pair_archive_sample_to_synthetic_pair(const PairArchiveSample& sample)
{
    return SyntheticPair{sample.view_a, sample.view_b, sample.warp_a_to_b, sample.valid_mask};
}

class PairArchiveTensorDataset : public TensorDataset
{
  public:
    PairArchiveTensorDataset(PairArchiveDatasetConfig config, std::size_t memory_cache_size)
        : _dataset(std::move(config)), _memory_cache_size(memory_cache_size)
    {
    }

    size_t size() const override
    {
        return _dataset.size();
    }

    TensorBatch get(size_t index) override
    {
        return cached_pair_to_tensor_batch(loadPair(index));
    }

  private:
    struct CacheEntry
    {
        SyntheticPair pair;
        std::list<size_t>::iterator lru_iterator;
    };

    SyntheticPair loadPair(size_t index)
    {
        if (_memory_cache_size == 0)
        {
            return pair_archive_sample_to_synthetic_pair(_dataset.load(index));
        }

        {
            std::lock_guard<std::mutex> lock(_cache_mutex);
            auto cached = _cache.find(index);
            if (cached != _cache.end())
            {
                _lru.splice(_lru.begin(), _lru, cached->second.lru_iterator);
                return cached->second.pair;
            }
        }

        auto loaded_pair = pair_archive_sample_to_synthetic_pair(_dataset.load(index));
        {
            std::lock_guard<std::mutex> lock(_cache_mutex);
            auto cached = _cache.find(index);
            if (cached != _cache.end())
            {
                _lru.splice(_lru.begin(), _lru, cached->second.lru_iterator);
                return cached->second.pair;
            }

            _lru.push_front(index);
            _cache.emplace(index, CacheEntry{loaded_pair, _lru.begin()});
            while (_cache.size() > _memory_cache_size)
            {
                const auto evicted_index = _lru.back();
                _lru.pop_back();
                _cache.erase(evicted_index);
            }
        }
        return loaded_pair;
    }

    PairArchiveDataset _dataset;
    std::size_t _memory_cache_size = 0;
    std::mutex _cache_mutex;
    std::list<size_t> _lru;
    std::unordered_map<size_t, CacheEntry> _cache;
};

class IndexedSyntheticPairCacheTensorDataset : public TensorDataset
{
  public:
    IndexedSyntheticPairCacheTensorDataset(std::string cache_dir, std::vector<std::size_t> indices)
        : _cache(std::move(cache_dir)), _indices(std::move(indices))
    {
        if (_indices.empty())
        {
            throw std::invalid_argument("indexed synthetic pair cache must contain at least one index");
        }
        for (const auto index : _indices)
        {
            if (index >= _cache.size())
            {
                throw std::invalid_argument("indexed synthetic pair cache index is out of range: " +
                                            std::to_string(index));
            }
        }
    }

    size_t size() const override
    {
        return _indices.size();
    }

    TensorBatch get(size_t index) override
    {
        if (index >= _indices.size())
        {
            throw std::out_of_range("indexed synthetic pair cache tensor index out of range");
        }
        return cached_pair_to_tensor_batch(_cache.load(_indices[index]));
    }

  private:
    SyntheticPairCacheDataset _cache;
    std::vector<std::size_t> _indices;
};

std::vector<SyntheticPair> load_cached_pairs(const CompositeSyntheticPairCacheDataset& cache_dataset,
                                             std::size_t offset, std::size_t end, torch::Device device,
                                             int64_t training_crop_size, int64_t resize,
                                             std::optional<at::Generator>& generator)
{
    std::vector<SyntheticPair> pairs;
    pairs.reserve(end - offset);
    for (std::size_t index = offset; index < end; ++index)
    {
        pairs.push_back(prepare_training_pair_size(move_pair_to_device(cache_dataset.load(index), device),
                                                   training_crop_size, resize, generator));
    }
    return pairs;
}

int64_t training_variant_index_for_pair(std::size_t pair_index, std::size_t train_image_count, int epoch,
                                        int pairs_per_image)
{
    if (train_image_count == 0 || pairs_per_image <= 0)
    {
        return 0;
    }
    return static_cast<int64_t>(pair_index / train_image_count) +
           static_cast<int64_t>(epoch) * static_cast<int64_t>(pairs_per_image);
}

struct TrainingBatchForward
{
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp;
    torch::Tensor valid_mask;
    SparseHeadOutput sparse_a;
    SparseHeadOutput sparse_b;
    torch::Tensor dense_confidence;
    torch::Tensor dense_offsets;
};

struct TrainingLossComponents
{
    torch::Tensor total;
    torch::Tensor repeatability;
    torch::Tensor descriptor;
    torch::Tensor orientation;
    torch::Tensor graph_matching;
    torch::Tensor offset;
    torch::Tensor confidence;
    torch::Tensor descriptor_accuracy;
    torch::Tensor graph_matching_accuracy;
    torch::Tensor graph_positive_fraction;
    torch::Tensor graph_positive_count;
    torch::Tensor graph_query_count;
    torch::Tensor graph_features_a_count;
    torch::Tensor graph_features_b_count;
    torch::Tensor learned_graph_matching_accuracy;
    torch::Tensor learned_graph_positive_fraction;
    torch::Tensor learned_graph_positive_count;
    torch::Tensor learned_graph_query_count;
    torch::Tensor descriptor_positive_score;
    torch::Tensor descriptor_hard_negative_score;
    torch::Tensor descriptor_positive_margin;
    torch::Tensor descriptor_positive_rank;
    torch::Tensor keypoint_descriptor_accuracy;
    torch::Tensor keypoint_descriptor_positive_margin;
    torch::Tensor keypoint_descriptor_positive_rank;
    torch::Tensor descriptor_diversity;
    torch::Tensor offset_error;
    TrainingBatchForward forward;
};

struct TrainingDiagnosticSnapshot
{
    TrainConfig config;
    int epoch = 0;
    std::size_t pair_index = 0;
    SyntheticPair pair;
    SparseHeadOutput sparse_a;
    SparseHeadOutput sparse_b;
    torch::Tensor dense_confidence;
};

struct PythonCompareSample
{
    torch::Tensor points_a;
    torch::Tensor points_b;
};

torch::Tensor normalize_python_descriptor_rows(const torch::Tensor& descriptors)
{
    auto finite = torch::nan_to_num(descriptors, 0.0, 0.0, 0.0);
    return finite / finite.norm(2, 1, true).clamp_min(1.0e-3);
}

torch::Tensor scale_image_points_to_feature_grid(const torch::Tensor& points, int64_t image_height, int64_t image_width,
                                                 int64_t feature_height, int64_t feature_width)
{
    if (points.numel() == 0)
    {
        return points.new_empty({0, 2});
    }
    auto x = points.index({torch::indexing::Slice(), 0}) *
             (static_cast<double>(std::max<int64_t>(1, feature_width - 1)) /
              static_cast<double>(std::max<int64_t>(1, image_width - 1)));
    auto y = points.index({torch::indexing::Slice(), 1}) *
             (static_cast<double>(std::max<int64_t>(1, feature_height - 1)) /
              static_cast<double>(std::max<int64_t>(1, image_height - 1)));
    return torch::stack({x, y}, 1).contiguous();
}

torch::Tensor sample_python_descriptor_points(const torch::Tensor& descriptor_map, const torch::Tensor& points)
{
    if (points.numel() == 0)
    {
        return descriptor_map.new_empty({0, descriptor_map.size(1)});
    }
    const auto height = descriptor_map.size(2);
    const auto width = descriptor_map.size(3);
    auto x = points.index({torch::indexing::Slice(), 0});
    auto y = points.index({torch::indexing::Slice(), 1});
    auto grid_x = width > 1 ? x / static_cast<double>(width - 1) * 2.0 - 1.0 : torch::zeros_like(x);
    auto grid_y = height > 1 ? y / static_cast<double>(height - 1) * 2.0 - 1.0 : torch::zeros_like(y);
    auto grid = torch::stack({grid_x, grid_y}, 1).reshape({1, points.size(0), 1, 2}).contiguous();
    return torch::nn::functional::grid_sample(descriptor_map, grid,
                                              torch::nn::functional::GridSampleFuncOptions()
                                                  .mode(torch::kBilinear)
                                                  .padding_mode(torch::kZeros)
                                                  .align_corners(true))
        .squeeze(0)
        .squeeze(-1)
        .transpose(0, 1)
        .contiguous();
}

torch::Tensor center_intensity_for_points(const torch::Tensor& image, const torch::Tensor& points)
{
    if (points.numel() == 0)
    {
        return image.new_empty({0});
    }
    auto intensity = image.mean(0);
    const auto height = intensity.size(0);
    const auto width = intensity.size(1);
    auto rounded = points.round().to(torch::kLong);
    auto x = rounded.index({torch::indexing::Slice(), 0}).clamp(0, width - 1);
    auto y = rounded.index({torch::indexing::Slice(), 1}).clamp(0, height - 1);
    return intensity.index({y, x});
}

PythonCompareSample sample_python_compare_correspondences(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                          const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                                          int64_t batch, int64_t feature_height,
                                                          int64_t feature_width, int64_t count, double min_intensity,
                                                          std::optional<at::Generator>& generator)
{
    const auto image_height = warp.size(1);
    const auto image_width = warp.size(2);
    const auto target_height = view_b.size(2);
    const auto target_width = view_b.size(3);
    auto warp_item = warp.index({batch});
    auto valid = valid_mask.index({batch}).to(torch::kBool)
                     .logical_and(torch::isfinite(warp_item.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                                                    0})))
                     .logical_and(torch::isfinite(warp_item.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                                                    1})))
                     .logical_and(warp_item.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).ge(0.0))
                     .logical_and(warp_item.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).le(
                         static_cast<double>(target_width - 1)))
                     .logical_and(warp_item.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).ge(0.0))
                     .logical_and(warp_item.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).le(
                         static_cast<double>(target_height - 1)));
    auto valid_indices = torch::nonzero(valid.reshape({image_height * image_width})).flatten();
    if (valid_indices.numel() == 0)
    {
        return PythonCompareSample{warp.new_empty({0, 2}), warp.new_empty({0, 2})};
    }
    auto y = torch::floor_divide(valid_indices, image_width).to(warp.dtype());
    auto x = valid_indices.remainder(image_width).to(warp.dtype());
    auto points_a_image = torch::stack({x, y}, 1).contiguous();
    auto points_b_image = warp_item.reshape({image_height * image_width, 2}).index_select(0, valid_indices).contiguous();
    if (min_intensity > 0.0 && points_a_image.numel() > 0)
    {
        auto textured =
            center_intensity_for_points(view_a.index({batch}), points_a_image).gt(min_intensity).logical_and(
                center_intensity_for_points(view_b.index({batch}), points_b_image).gt(min_intensity));
        points_a_image = points_a_image.index({textured});
        points_b_image = points_b_image.index({textured});
    }
    if (points_a_image.numel() == 0)
    {
        return PythonCompareSample{warp.new_empty({0, 2}), warp.new_empty({0, 2})};
    }
    const auto take = std::min<int64_t>(count, points_a_image.size(0));
    auto order = randperm_with_training_generator(points_a_image.size(0), valid_indices.options(), generator)
                     .narrow(0, 0, take);
    points_a_image = points_a_image.index_select(0, order);
    points_b_image = points_b_image.index_select(0, order);
    return PythonCompareSample{
        scale_image_points_to_feature_grid(points_a_image, image_height, image_width, feature_height, feature_width),
        scale_image_points_to_feature_grid(points_b_image, image_height, image_width, feature_height, feature_width)};
}

torch::Tensor sample_python_compare_unmatched_feature_points(int64_t feature_height, int64_t feature_width,
                                                             const torch::Tensor& reference_points, int64_t count,
                                                             double min_distance,
                                                             std::optional<at::Generator>& generator)
{
    if (count <= 0)
    {
        return reference_points.new_empty({0, 2});
    }
    if (feature_height <= 0 || feature_width <= 0)
    {
        throw std::invalid_argument("feature size must be positive");
    }
    const int64_t total = feature_height * feature_width;
    if (total <= 0)
    {
        return reference_points.new_empty({0, 2});
    }

    const int64_t candidate_count = std::min<int64_t>(total, std::max<int64_t>(count * 32, count + 128));
    auto flat = randperm_with_training_generator(
                    total, torch::TensorOptions().dtype(torch::kLong).device(reference_points.device()), generator)
                    .narrow(0, 0, candidate_count);
    auto y = torch::floor_divide(flat, feature_width).to(torch::kFloat32);
    auto x = flat.remainder(feature_width).to(torch::kFloat32);
    auto candidates = torch::stack({x, y}, 1);
    if (reference_points.numel() > 0 && min_distance > 0.0)
    {
        auto refs = reference_points.to(torch::TensorOptions().device(candidates.device()).dtype(torch::kFloat32));
        auto distances = torch::cdist(candidates, refs);
        auto keep = std::get<0>(distances.min(1)).ge(min_distance);
        candidates = candidates.index({keep});
    }
    const auto take = std::min<int64_t>(count, candidates.size(0));
    return candidates.narrow(0, 0, take).contiguous();
}

DescriptorTrainingMetrics make_python_compare_descriptor_metrics(const torch::Tensor& desc_a,
                                                                 const torch::Tensor& desc_b, double temperature)
{
    auto normalized_a = normalize_python_descriptor_rows(desc_a);
    auto normalized_b = normalize_python_descriptor_rows(desc_b);
    auto similarity = torch::matmul(normalized_a, normalized_b.transpose(0, 1));
    auto targets = torch::arange(desc_a.size(0), torch::TensorOptions().dtype(torch::kLong).device(desc_a.device()));
    auto logits = similarity / temperature;
    auto loss_ab = torch::nn::functional::cross_entropy(logits, targets);
    auto loss_ba = torch::nn::functional::cross_entropy(logits.transpose(0, 1).contiguous(), targets);
    auto loss = (loss_ab + loss_ba) * 0.5;
    auto top1 = similarity.argmax(1).eq(targets).to(torch::kFloat32).mean();
    auto sorted = similarity.argsort(1, true);
    auto ranks = sorted.eq(targets.unsqueeze(1)).to(torch::kInt64).argmax(1).to(torch::kFloat32) + 1.0F;
    auto positive = similarity.diag().mean();
    auto negative = torch::zeros({}, similarity.options());
    if (desc_a.size(0) > 1)
    {
        auto off_diagonal = torch::eye(desc_a.size(0), torch::TensorOptions().dtype(torch::kBool).device(desc_a.device()))
                                .logical_not();
        negative = similarity.index({off_diagonal}).mean();
    }
    return DescriptorTrainingMetrics{loss, top1, positive, negative, positive - negative, ranks.mean(),
                                     torch::zeros({}, desc_a.options())};
}

torch::Tensor prepare_python_compare_keypoints_for_embedding(const torch::Tensor& points, int64_t meta_dim)
{
    auto prepared = points.to(torch::TensorOptions().device(points.device()).dtype(torch::kFloat32));
    if (prepared.size(0) == 0)
    {
        return prepared.new_empty({0, meta_dim});
    }
    auto min_xy = std::get<0>(prepared.min(0, true));
    auto max_xy = std::get<0>(prepared.max(0, true));
    auto center = (min_xy + max_xy) * 0.5;
    auto span = std::get<0>((max_xy - min_xy).max(1, true)).clamp_min(1.0e-6);
    auto centered = (prepared - center) * 2.0 / span;
    auto radius = centered.pow(2).sum(1, true).sqrt();
    auto legacy = torch::cat(std::vector<torch::Tensor>{radius, radius.pow(2)}, 1);
    auto spatial = torch::cat(std::vector<torch::Tensor>{centered, legacy}, 1);
    if (meta_dim <= spatial.size(1))
    {
        return spatial.index({torch::indexing::Slice(), torch::indexing::Slice(0, meta_dim)}).contiguous();
    }
    return torch::cat(std::vector<torch::Tensor>{spatial, spatial.new_zeros({spatial.size(0), meta_dim - spatial.size(1)})},
                      1)
        .contiguous();
}

torch::Tensor make_python_compare_graph_metadata(const torch::Tensor& points, int64_t meta_dim)
{
    auto base = prepare_python_compare_keypoints_for_embedding(points, std::max<int64_t>(meta_dim, 4));
    const auto count = points.size(0);
    auto score = points.new_full({count, 1}, 1.0);
    auto scale = points.new_zeros({count, 1});
    auto orientation =
        torch::cat(std::vector<torch::Tensor>{points.new_full({count, 1}, 1.0), points.new_zeros({count, 1})}, 1);
    auto affine = torch::cat(std::vector<torch::Tensor>{points.new_full({count, 1}, 1.0), points.new_zeros({count, 2}),
                                                        points.new_full({count, 1}, 1.0)},
                             1);
    auto quality = points.new_full({count, 1}, 1.0);
    auto contrast = points.new_zeros({count, 1});
    auto uncertainty = points.new_zeros({count, 1});
    auto metadata = torch::cat(std::vector<torch::Tensor>{
                                   base.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}), score, scale,
                                   orientation, affine, quality, contrast, uncertainty},
                               1);
    if (metadata.size(1) >= meta_dim)
    {
        return metadata.index({torch::indexing::Slice(), torch::indexing::Slice(0, meta_dim)}).contiguous();
    }
    return torch::cat(std::vector<torch::Tensor>{metadata, metadata.new_zeros({count, meta_dim - metadata.size(1)})}, 1)
        .contiguous();
}

torch::Tensor binary_cross_entropy_with_logits_mean(const torch::Tensor& logits, const torch::Tensor& targets)
{
    auto zeros = torch::zeros_like(logits);
    auto positive = torch::maximum(logits, zeros);
    auto stable = positive - logits * targets + torch::log1p(torch::exp(-logits.abs()));
    return stable.mean();
}

torch::Tensor make_python_compare_graph_acceptance_loss(const v21::PfmV21GraphMatcherOutput& output,
                                                        const torch::Tensor& desc_a,
                                                        const torch::Tensor& desc_b, int64_t positive_count,
                                                        int64_t negative_topk)
{
    if (!output.accept_logits.defined() || positive_count <= 0)
    {
        return output.logits.new_zeros({});
    }
    const auto count = std::min<int64_t>({positive_count, output.accept_logits.size(0), output.accept_logits.size(1),
                                          desc_a.size(0), desc_b.size(0)});
    if (count <= 0)
    {
        return output.logits.new_zeros({});
    }

    std::vector<torch::Tensor> terms;
    auto accept_square =
        output.accept_logits.index({torch::indexing::Slice(0, count), torch::indexing::Slice(0, count)});
    auto diag_logits = accept_square.diagonal();
    terms.push_back(binary_cross_entropy_with_logits_mean(diag_logits, torch::ones_like(diag_logits)));
    if (count > 1 && negative_topk > 0)
    {
        auto normalized_a = torch::nn::functional::normalize(
            desc_a.narrow(0, 0, count), torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        auto normalized_b = torch::nn::functional::normalize(
            desc_b.narrow(0, 0, count), torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        auto similarity = torch::matmul(normalized_a, normalized_b.transpose(0, 1));
        auto diagonal_mask =
            torch::eye(count, torch::TensorOptions().dtype(torch::kBool).device(similarity.device()));
        auto masked_similarity = similarity.masked_fill(diagonal_mask, -std::numeric_limits<float>::infinity());
        const auto k = std::min<int64_t>(negative_topk, count - 1);
        auto hard_indices = std::get<1>(masked_similarity.topk(k, 1));
        auto hard_logits = accept_square.gather(1, hard_indices);
        terms.push_back(binary_cross_entropy_with_logits_mean(hard_logits, torch::zeros_like(hard_logits)));
    }
    if (output.accept_logits.size(1) > count)
    {
        auto no_match_ab = output.accept_logits.narrow(0, 0, count)
                                .narrow(1, count, output.accept_logits.size(1) - count);
        if (no_match_ab.numel() > 0)
        {
            terms.push_back(binary_cross_entropy_with_logits_mean(no_match_ab, torch::zeros_like(no_match_ab)));
        }
    }
    if (output.accept_logits.size(0) > count)
    {
        auto no_match_ba = output.accept_logits.narrow(0, count, output.accept_logits.size(0) - count)
                                .narrow(1, 0, count);
        if (no_match_ba.numel() > 0)
        {
            terms.push_back(binary_cross_entropy_with_logits_mean(no_match_ba, torch::zeros_like(no_match_ba)));
        }
    }
    return torch::stack(terms).mean();
}

torch::Tensor make_python_compare_graph_prune_ranking_loss(const v21::PfmV21GraphMatcherOutput& output,
                                                           int64_t positive_count, double margin)
{
    if (!output.accept_logits.defined() || positive_count <= 0)
    {
        return output.logits.new_zeros({});
    }
    const auto count = std::min<int64_t>({positive_count, output.accept_logits.size(0), output.accept_logits.size(1)});
    if (count <= 0)
    {
        return output.logits.new_zeros({});
    }

    std::vector<torch::Tensor> terms;
    const auto margin_tensor = torch::full({}, static_cast<float>(margin), output.accept_logits.options());
    auto positive_square = output.accept_logits.narrow(0, 0, count).narrow(1, 0, count);
    auto positive_logits = positive_square.diagonal();
    if (count > 1)
    {
        auto diagonal_mask =
            torch::eye(count, torch::TensorOptions().dtype(torch::kBool).device(positive_square.device()));
        auto masked = positive_square.masked_fill(diagonal_mask, -std::numeric_limits<float>::infinity());
        auto row_hard = std::get<0>(masked.max(1));
        auto col_hard = std::get<0>(masked.max(0));
        terms.push_back(torch::relu(margin_tensor - positive_logits + row_hard).mean());
        terms.push_back(torch::relu(margin_tensor - positive_logits + col_hard).mean());
    }
    auto positive_anchor = positive_logits.mean();
    if (output.accept_logits.size(0) > count)
    {
        auto negative_rows =
            output.accept_logits.narrow(0, count, output.accept_logits.size(0) - count).narrow(1, 0, count);
        if (negative_rows.numel() > 0)
        {
            auto negative_scores = std::get<0>(negative_rows.max(1));
            terms.push_back(torch::relu(margin_tensor - positive_anchor + negative_scores).mean());
        }
    }
    if (output.accept_logits.size(1) > count)
    {
        auto negative_cols =
            output.accept_logits.narrow(0, 0, count).narrow(1, count, output.accept_logits.size(1) - count);
        if (negative_cols.numel() > 0)
        {
            auto negative_scores = std::get<0>(negative_cols.max(0));
            terms.push_back(torch::relu(margin_tensor - positive_anchor + negative_scores).mean());
        }
    }
    if (terms.empty())
    {
        return output.logits.new_zeros({});
    }
    return torch::stack(terms).mean();
}

torch::Tensor binary_cross_entropy_probability_mean(const torch::Tensor& probabilities, const torch::Tensor& targets)
{
    const auto clamped = probabilities.clamp(1.0e-6, 1.0 - 1.0e-6);
    return -(targets * clamped.log() + (1.0 - targets) * (1.0 - clamped).log()).mean();
}

torch::Tensor make_python_compare_assignment_confidence(const torch::Tensor& pair_logits)
{
    if (pair_logits.numel() == 0)
    {
        return pair_logits.new_zeros({});
    }
    auto row_confidence = std::get<0>(torch::softmax(pair_logits, 1).max(1));
    auto column_confidence = std::get<0>(torch::softmax(pair_logits, 0).max(0));
    return torch::minimum(row_confidence.mean(), column_confidence.mean());
}

torch::Tensor make_python_compare_graph_stop_confidence_loss(const v21::PfmV21GraphMatcherOutput& output,
                                                             int64_t positive_count, double safe_margin)
{
    const auto count =
        std::min<int64_t>({positive_count, output.logits.size(0) - 1, output.logits.size(1) - 1});
    if (count <= 1)
    {
        return output.logits.new_zeros({});
    }

    auto pair_logits = output.logits.narrow(0, 0, count).narrow(1, 0, count);
    auto diagonal_mask = torch::eye(count, torch::TensorOptions().dtype(torch::kBool).device(pair_logits.device()));
    auto positive_logits = pair_logits.diagonal();
    auto masked = pair_logits.masked_fill(diagonal_mask, -std::numeric_limits<float>::infinity());
    auto row_hard = std::get<0>(masked.max(1));
    auto col_hard = std::get<0>(masked.max(0));
    auto margin_tensor = torch::full({}, static_cast<float>(safe_margin), output.logits.options());
    auto safe_rows = positive_logits - row_hard >= margin_tensor;
    auto safe_cols = positive_logits - col_hard >= margin_tensor;
    auto target = safe_rows.logical_and(safe_cols).to(pair_logits.dtype()).mean().detach();
    auto confidence = make_python_compare_assignment_confidence(pair_logits);
    return binary_cross_entropy_probability_mean(confidence, target);
}

int64_t resolve_python_compare_graph_attention_budget(int64_t max_attention_layers, bool random_attention_layers,
                                                      const torch::Device& device,
                                                      std::optional<at::Generator>* generator)
{
    if (!random_attention_layers)
    {
        return max_attention_layers;
    }
    const auto layer_limit = std::max<int64_t>(1, max_attention_layers);
    auto options = torch::TensorOptions().dtype(torch::kLong).device(device);
    std::optional<at::Generator> fallback_generator;
    auto& active_generator = generator == nullptr ? fallback_generator : *generator;
    return randint_with_training_generator(layer_limit, {1}, options, active_generator).item<int64_t>() + 1;
}

template <typename GraphMatcherT>
torch::Tensor make_python_compare_graph_loss(GraphMatcherT& graph_matcher, const torch::Tensor& desc_a,
                                             const torch::Tensor& desc_b, const torch::Tensor& points_a,
                                             const torch::Tensor& points_b, int64_t meta_dim,
                                             double accept_weight = 0.0, int64_t accept_negative_topk = 8,
                                             double prune_ranking_weight = 0.0, double prune_ranking_margin = 0.25,
                                             double stop_confidence_weight = 0.0,
                                             double stop_confidence_margin = 0.5,
                                             int64_t max_attention_layers = 0,
                                             bool random_attention_layers = false,
                                             std::optional<at::Generator>* generator = nullptr,
                                             int64_t positive_count_override = -1,
                                             double width_keep_ratio = 1.0,
                                             int64_t* selected_positive_count = nullptr,
                                             double max_attention_work_fraction = 1.0)
{
    if (desc_a.size(0) == 0 || desc_b.size(0) == 0)
    {
        if (selected_positive_count != nullptr)
        {
            *selected_positive_count = 0;
        }
        return torch::zeros({}, desc_a.options());
    }
    if (!std::isfinite(max_attention_work_fraction) || max_attention_work_fraction < 0.0 ||
        max_attention_work_fraction > 1.0)
    {
        throw std::invalid_argument("graph matcher max_attention_work_fraction must be in [0, 1]");
    }
    if (!std::isfinite(width_keep_ratio) || width_keep_ratio <= 0.0 || width_keep_ratio > 1.0)
    {
        throw std::invalid_argument("graph matcher width_keep_ratio must be in (0, 1]");
    }
    if (points_a.size(0) < desc_a.size(0) || points_b.size(0) < desc_b.size(0))
    {
        throw std::invalid_argument("python compare graph loss points must cover all descriptors");
    }
    const auto inferred_positive_count = std::min<int64_t>(desc_a.size(0), desc_b.size(0));
    auto count = positive_count_override >= 0
                     ? std::min<int64_t>(positive_count_override, inferred_positive_count)
                     : inferred_positive_count;
    if (count <= 0)
    {
        if (selected_positive_count != nullptr)
        {
            *selected_positive_count = 0;
        }
        return torch::zeros({}, desc_a.options());
    }
    std::optional<at::Generator> fallback_generator;
    auto& active_generator = generator == nullptr ? fallback_generator : *generator;
    auto active_desc_a = desc_a;
    auto active_desc_b = desc_b;
    auto active_points_a = points_a.narrow(0, 0, desc_a.size(0));
    auto active_points_b = points_b.narrow(0, 0, desc_b.size(0));
    if (width_keep_ratio < 1.0 && count > 1)
    {
        const auto requested_keep =
            static_cast<int64_t>(std::ceil(static_cast<double>(count) * width_keep_ratio));
        const auto keep_count = std::max<int64_t>(1, std::min<int64_t>(count, requested_keep));
        if (keep_count < count)
        {
            auto index_options = torch::TensorOptions().dtype(torch::kLong).device(desc_a.device());
            auto keep_indices =
                std::get<0>(randperm_with_training_generator(count, index_options, active_generator)
                                .narrow(0, 0, keep_count)
                                .sort());
            auto keep_indices_b = keep_indices.to(desc_b.device());
            auto keep_points_a = keep_indices.to(points_a.device());
            auto keep_points_b = keep_indices.to(points_b.device());
            auto positive_desc_a = desc_a.narrow(0, 0, count).index_select(0, keep_indices);
            auto positive_desc_b = desc_b.narrow(0, 0, count).index_select(0, keep_indices_b);
            auto positive_points_a = points_a.narrow(0, 0, count).index_select(0, keep_points_a);
            auto positive_points_b = points_b.narrow(0, 0, count).index_select(0, keep_points_b);
            active_desc_a = desc_a.size(0) > count
                                ? torch::cat({positive_desc_a, desc_a.narrow(0, count, desc_a.size(0) - count)}, 0)
                                : positive_desc_a;
            active_desc_b = desc_b.size(0) > count
                                ? torch::cat({positive_desc_b, desc_b.narrow(0, count, desc_b.size(0) - count)}, 0)
                                : positive_desc_b;
            active_points_a =
                points_a.size(0) > count
                    ? torch::cat({positive_points_a, points_a.narrow(0, count, points_a.size(0) - count)}, 0)
                    : positive_points_a;
            active_points_b =
                points_b.size(0) > count
                    ? torch::cat({positive_points_b, points_b.narrow(0, count, points_b.size(0) - count)}, 0)
                    : positive_points_b;
            count = keep_count;
        }
    }
    if (selected_positive_count != nullptr)
    {
        *selected_positive_count = count;
    }
    auto meta_a =
        make_python_compare_graph_metadata(active_points_a.narrow(0, 0, active_desc_a.size(0)), meta_dim)
            .to(active_desc_a.device());
    auto meta_b =
        make_python_compare_graph_metadata(active_points_b.narrow(0, 0, active_desc_b.size(0)), meta_dim)
            .to(active_desc_b.device());
    const auto attention_budget = resolve_python_compare_graph_attention_budget(
        max_attention_layers, random_attention_layers, active_desc_a.device(), &active_generator);
    auto output = graph_matcher.forward(active_desc_a, meta_a, active_desc_b, meta_b, false, -1.0, -1.0,
                                        attention_budget, max_attention_work_fraction);
    auto targets = torch::arange(count, torch::TensorOptions().dtype(torch::kLong).device(output.logits.device()));
    auto row_loss = torch::nn::functional::cross_entropy(output.logits.narrow(0, 0, count), targets);
    auto col_logits =
        output.logits.index({torch::indexing::Slice(), torch::indexing::Slice(0, count)}).transpose(0, 1).contiguous();
    auto col_loss = torch::nn::functional::cross_entropy(col_logits, targets);
    auto loss = (row_loss + col_loss) * 0.5;
    if (accept_weight > 0.0)
    {
        loss = loss + static_cast<float>(accept_weight) *
                          make_python_compare_graph_acceptance_loss(output, active_desc_a, active_desc_b, count,
                                                                    accept_negative_topk);
    }
    if (prune_ranking_weight > 0.0)
    {
        loss = loss + static_cast<float>(prune_ranking_weight) *
                          make_python_compare_graph_prune_ranking_loss(output, count, prune_ranking_margin);
    }
    if (stop_confidence_weight > 0.0)
    {
        loss = loss + static_cast<float>(stop_confidence_weight) *
                          make_python_compare_graph_stop_confidence_loss(output, count, stop_confidence_margin);
    }
    return loss;
}

TrainingLossComponents make_python_compare_training_loss(TrainModules& modules, const torch::Tensor& view_a,
                                                         const torch::Tensor& view_b, const torch::Tensor& warp,
                                                         const torch::Tensor& valid_mask, SparseHeadOutput sparse_a,
                                                         SparseHeadOutput sparse_b, const DenseHeadOutput& dense,
                                                         const TrainConfig& config,
                                                         std::optional<at::Generator>& generator)
{
    auto zero = sparse_a.descriptors.sum() * 0.0;
    std::vector<torch::Tensor> descriptor_losses;
    std::vector<torch::Tensor> graph_losses;
    std::vector<torch::Tensor> accuracies;
    std::vector<torch::Tensor> positive_scores;
    std::vector<torch::Tensor> negative_scores;
    std::vector<torch::Tensor> margins;
    std::vector<torch::Tensor> ranks;
    std::vector<torch::Tensor> counts;
    const auto batch_size = view_a.size(0);
    const auto graph_loss_weight = effective_python_compare_graph_loss_weight(config);
    const bool use_graph_loss = graph_loss_weight > 0.0;
    descriptor_losses.reserve(static_cast<size_t>(batch_size));
    graph_losses.reserve(static_cast<size_t>(batch_size));
    for (int64_t batch = 0; batch < batch_size; ++batch)
    {
        auto sample = sample_python_compare_correspondences(view_a, view_b, warp, valid_mask, batch,
                                                            sparse_a.descriptors.size(2), sparse_a.descriptors.size(3),
                                                            config.samples_per_pair, config.min_keypoint_intensity,
                                                            generator);
        if (sample.points_a.size(0) == 0)
        {
            continue;
        }
        auto desc_a = sample_python_descriptor_points(sparse_a.descriptors.index({batch}).unsqueeze(0), sample.points_a);
        auto desc_b = sample_python_descriptor_points(sparse_b.descriptors.index({batch}).unsqueeze(0), sample.points_b);
        auto descriptor_metrics = make_python_compare_descriptor_metrics(desc_a, desc_b, config.temperature);
        descriptor_losses.push_back(descriptor_metrics.loss);
        accuracies.push_back(descriptor_metrics.accuracy);
        positive_scores.push_back(descriptor_metrics.positive_score);
        negative_scores.push_back(descriptor_metrics.hard_negative_score);
        margins.push_back(descriptor_metrics.positive_margin);
        ranks.push_back(descriptor_metrics.positive_rank);
        counts.push_back(torch::full({}, static_cast<float>(sample.points_a.size(0)), sparse_a.descriptors.options()));
        if (use_graph_loss)
        {
            auto graph_desc_a = normalize_python_descriptor_rows(desc_a);
            auto graph_desc_b = normalize_python_descriptor_rows(desc_b);
            auto graph_points_a = sample.points_a;
            auto graph_points_b = sample.points_b;
            if (config.graph_matcher_no_match_points > 0 && config.graph_matcher_accept_weight > 0.0)
            {
                auto no_match_a = sample_python_compare_unmatched_feature_points(
                    sparse_a.descriptors.size(2), sparse_a.descriptors.size(3), sample.points_a,
                    config.graph_matcher_no_match_points, config.graph_matcher_no_match_min_distance, generator);
                auto no_match_b = sample_python_compare_unmatched_feature_points(
                    sparse_b.descriptors.size(2), sparse_b.descriptors.size(3), sample.points_b,
                    config.graph_matcher_no_match_points, config.graph_matcher_no_match_min_distance, generator);
                if (no_match_a.numel() > 0)
                {
                    auto no_match_desc_a = sample_python_descriptor_points(
                        sparse_a.descriptors.index({batch}).unsqueeze(0), no_match_a.to(sample.points_a.device()));
                    graph_desc_a = torch::cat({graph_desc_a, normalize_python_descriptor_rows(no_match_desc_a)}, 0);
                    graph_points_a = torch::cat({graph_points_a, no_match_a.to(sample.points_a.device())}, 0);
                }
                if (no_match_b.numel() > 0)
                {
                    auto no_match_desc_b = sample_python_descriptor_points(
                        sparse_b.descriptors.index({batch}).unsqueeze(0), no_match_b.to(sample.points_b.device()));
                    graph_desc_b = torch::cat({graph_desc_b, normalize_python_descriptor_rows(no_match_desc_b)}, 0);
                    graph_points_b = torch::cat({graph_points_b, no_match_b.to(sample.points_b.device())}, 0);
                }
            }
            const auto train_max_attention_layers =
                config.graph_matcher_train_random_attention_layers && config.graph_matcher_train_max_attention_layers <= 0
                    ? config.graph_attention_layers
                    : config.graph_matcher_train_max_attention_layers;
            graph_losses.push_back(make_python_compare_graph_loss(
                *modules.graph_matcher, graph_desc_a, graph_desc_b, graph_points_a, graph_points_b,
                config.graph_keypoint_meta_dim, config.graph_matcher_accept_weight,
                config.graph_matcher_accept_negative_topk, config.graph_matcher_prune_ranking_weight,
                config.graph_matcher_prune_ranking_margin, config.graph_matcher_stop_confidence_weight,
                config.graph_matcher_stop_confidence_margin, train_max_attention_layers,
                config.graph_matcher_train_random_attention_layers, &generator, sample.points_a.size(0),
                config.graph_matcher_train_width_keep_ratio, nullptr,
                config.graph_matcher_train_max_attention_work_fraction));
        }
    }
    if (descriptor_losses.empty() && graph_losses.empty())
    {
        throw std::runtime_error("no valid correspondences sampled");
    }
    auto descriptor_loss = descriptor_losses.empty() ? zero : torch::stack(descriptor_losses).mean();
    auto graph_loss = graph_losses.empty() ? zero : torch::stack(graph_losses).mean();
    auto total = descriptor_loss * config.synthetic_loss_weight + graph_loss * graph_loss_weight;
    auto mean_or_zero = [&](const std::vector<torch::Tensor>& values)
    {
        return values.empty() ? zero : torch::stack(values).mean();
    };
    auto count = mean_or_zero(counts);
    return TrainingLossComponents{total,
                                  zero,
                                  descriptor_loss,
                                  zero,
                                  graph_loss,
                                  zero,
                                  zero,
                                  mean_or_zero(accuracies),
                                  mean_or_zero(accuracies),
                                  torch::ones_like(count),
                                  count,
                                  count,
                                  count,
                                  count,
                                  mean_or_zero(accuracies),
                                  torch::ones_like(count),
                                  count,
                                  count,
                                  mean_or_zero(positive_scores),
                                  mean_or_zero(negative_scores),
                                  mean_or_zero(margins),
                                  mean_or_zero(ranks),
                                  mean_or_zero(accuracies),
                                  mean_or_zero(margins),
                                  mean_or_zero(ranks),
                                  zero,
                                  zero,
                                  TrainingBatchForward{view_a, view_b, warp, valid_mask, sparse_a, sparse_b,
                                                       dense.confidence, dense.offsets}};
}

torch::Tensor weighted_total_training_loss(const torch::Tensor& repeatability, const torch::Tensor& descriptor,
                                           const torch::Tensor& orientation, const torch::Tensor& graph_matching,
                                           const torch::Tensor& offset, const torch::Tensor& confidence,
                                           const torch::Tensor& descriptor_diversity)
{
    return repeatability * REPEATABILITY_LOSS_WEIGHT + descriptor + orientation * ORIENTATION_LOSS_WEIGHT +
           descriptor_diversity * DESCRIPTOR_DIVERSITY_WEIGHT + graph_matching * GRAPH_MATCHING_LOSS_WEIGHT +
           offset * OFFSET_LOSS_WEIGHT + confidence * CONFIDENCE_LOSS_WEIGHT;
}

torch::Tensor make_heatmap_selection_loss(const torch::Tensor& heatmap, const torch::Tensor& mask)
{
    auto mask_float = mask.to(heatmap.dtype());
    if (mask_float.sizes() != heatmap.sizes())
    {
        mask_float = mask_float.expand_as(heatmap);
    }
    auto denom = mask_float.sum().clamp_min(1.0F);
    auto selected_mean = (heatmap * mask_float).sum() / denom;
    auto mean_loss = (selected_mean - HEATMAP_TARGET_MEAN) * (selected_mean - HEATMAP_TARGET_MEAN);
    auto binary_loss = (heatmap * (1.0F - heatmap) * mask_float).sum() / denom;
    return (mean_loss + binary_loss * HEATMAP_BINARY_WEIGHT) * HEATMAP_SELECTION_WEIGHT;
}

torch::Tensor make_texture_saliency_target(const torch::Tensor& image, int64_t target_height, int64_t target_width)
{
    auto base = image;
    if (base.size(1) != 1)
    {
        base = base.mean(1, true);
    }
    auto blur = torch::avg_pool2d(base, {15, 15}, {1, 1}, {7, 7}, false, true);
    auto contrast = (base - blur).abs();
    auto dx = (base - torch::roll(base, {1}, {3})).abs();
    auto dy = (base - torch::roll(base, {1}, {2})).abs();
    auto saliency = torch::avg_pool2d(contrast + dx + dy, {5, 5}, {1, 1}, {2, 2}, false, true);
    saliency = torch::nn::functional::interpolate(saliency, torch::nn::functional::InterpolateFuncOptions()
                                                                .size(std::vector<int64_t>{target_height, target_width})
                                                                .mode(torch::kBilinear)
                                                                .align_corners(false));
    auto flat = saliency.reshape({saliency.size(0), saliency.size(1), saliency.size(2) * saliency.size(3)});
    auto min_value = std::get<0>(flat.min(2, true)).reshape({saliency.size(0), saliency.size(1), 1, 1});
    auto max_value = std::get<0>(flat.max(2, true)).reshape({saliency.size(0), saliency.size(1), 1, 1});
    return (saliency - min_value) / (max_value - min_value).clamp_min(1.0e-6);
}

torch::Tensor make_repeatable_saliency_target_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b,
                                                   const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                   const torch::Tensor& warp, const torch::Tensor& mask)
{
    auto mask_float = resize_mask_for_heatmap(mask, heatmap_a).to(heatmap_a.dtype());
    auto saliency_a = make_texture_saliency_target(view_a, heatmap_a.size(2), heatmap_a.size(3));
    auto saliency_b = make_texture_saliency_target(view_b, heatmap_b.size(2), heatmap_b.size(3));
    auto warped_saliency_b = warp_heatmap_for_repeatability(saliency_b, warp);
    auto target = torch::sqrt((saliency_a * warped_saliency_b).clamp_min(0.0F));
    target = target * mask_float;
    auto denom = mask_float.sum().clamp_min(1.0F);
    auto warped_heatmap_b = warp_heatmap_for_repeatability(heatmap_b, warp);
    auto loss_a = ((heatmap_a - target).pow(2) * mask_float).sum() / denom;
    auto loss_b = ((warped_heatmap_b - target).pow(2) * mask_float).sum() / denom;
    return (loss_a + loss_b) * 0.5F;
}

torch::Tensor make_repeatable_keypoint_target(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                              const torch::Tensor& warp, const torch::Tensor& mask,
                                              int64_t target_height, int64_t target_width, bool dilate_targets = true,
                                              bool enforce_min_count = true)
{
    auto mask_float = resize_mask_for_heatmap(mask, torch::zeros({mask.size(0), 1, target_height, target_width},
                                                                 mask.options().dtype(torch::kFloat32)))
                          .to(torch::kFloat32);
    auto saliency_a = make_texture_saliency_target(view_a, target_height, target_width);
    auto saliency_b = make_texture_saliency_target(view_b, target_height, target_width);
    auto warped_saliency_b = warp_heatmap_for_repeatability(saliency_b, warp);
    auto repeatable = torch::sqrt((saliency_a * warped_saliency_b).clamp_min(0.0F)) * mask_float;
    if (!enforce_min_count)
    {
        auto intensity_a = view_a.size(1) == 1 ? view_a : view_a.mean(1, true);
        auto intensity_b = view_b.size(1) == 1 ? view_b : view_b.mean(1, true);
        intensity_a =
            torch::nn::functional::interpolate(intensity_a, torch::nn::functional::InterpolateFuncOptions()
                                                                .size(std::vector<int64_t>{target_height, target_width})
                                                                .mode(torch::kBilinear)
                                                                .align_corners(false));
        intensity_b =
            torch::nn::functional::interpolate(intensity_b, torch::nn::functional::InterpolateFuncOptions()
                                                                .size(std::vector<int64_t>{target_height, target_width})
                                                                .mode(torch::kBilinear)
                                                                .align_corners(false));
        auto warped_intensity_b = warp_heatmap_for_repeatability(intensity_b, warp);
        repeatable =
            repeatable + torch::sqrt((intensity_a * warped_intensity_b).clamp_min(0.0F)) * mask_float * 1.0e-3F;
    }
    auto target = torch::zeros_like(repeatable);
    const auto spatial_count = target_height * target_width;
    for (int64_t batch = 0; batch < repeatable.size(0); ++batch)
    {
        auto flat_scores = repeatable.index({batch, 0}).reshape({spatial_count});
        auto flat_mask = mask_float.index({batch, 0}).reshape({spatial_count}).gt(0.0);
        const auto valid_count = flat_mask.sum().item<int64_t>();
        if (valid_count <= 0)
        {
            continue;
        }
        const auto requested =
            static_cast<int64_t>(std::llround(static_cast<double>(valid_count) * REPEATABLE_KEYPOINT_TARGET_FRACTION));
        const auto lower_bound = enforce_min_count ? REPEATABLE_KEYPOINT_TARGET_MIN_COUNT : 1;
        const auto upper_bound =
            enforce_min_count ? REPEATABLE_KEYPOINT_TARGET_MAX_COUNT : std::max<int64_t>(1, requested);
        const auto k = std::min<int64_t>(
            valid_count,
            std::max<int64_t>(1, std::min<int64_t>(upper_bound, std::max<int64_t>(lower_bound, requested))));
        auto masked_scores = flat_scores.masked_fill(flat_mask.logical_not(), -std::numeric_limits<float>::infinity());
        auto topk = masked_scores.topk(k);
        auto indices = std::get<1>(topk).to(torch::kLong);
        auto selected = torch::zeros({spatial_count}, repeatable.options());
        selected.index_put_({indices}, 1.0F);
        target.index_put_({batch, 0}, selected.reshape({target_height, target_width}));
    }
    if (dilate_targets && target_height >= 3 && target_width >= 3)
    {
        target = torch::max_pool2d(target, {3, 3}, {1, 1}, {1, 1});
    }
    return target * mask_float;
}

torch::Tensor make_heatmap_correspondence_target_loss(const torch::Tensor& heatmap_a,
                                                      const torch::Tensor& heatmap_b_at_a, const torch::Tensor& target,
                                                      const torch::Tensor& mask)
{
    auto mask_float = mask.to(heatmap_a.dtype());
    if (mask_float.sizes() != heatmap_a.sizes())
    {
        mask_float = mask_float.expand_as(heatmap_a);
    }
    auto target_float = target.to(heatmap_a.dtype());
    if (target_float.sizes() != heatmap_a.sizes())
    {
        target_float = target_float.expand_as(heatmap_a);
    }
    auto denom = mask_float.sum().clamp_min(1.0F);
    auto stable_a = heatmap_a.clamp(1.0e-4, 1.0F - 1.0e-4);
    auto stable_b = heatmap_b_at_a.clamp(1.0e-4, 1.0F - 1.0e-4);
    auto bce_a = -(target_float * stable_a.log() + (1.0F - target_float) * (1.0F - stable_a).log());
    auto bce_b = -(target_float * stable_b.log() + (1.0F - target_float) * (1.0F - stable_b).log());
    auto positive_weight = 1.0F + target_float * (HEATMAP_TARGET_POSITIVE_WEIGHT - 1.0F);
    return ((bce_a + bce_b) * 0.5F * positive_weight * mask_float).sum() / denom;
}

torch::Tensor make_heatmap_positive_target_loss(const torch::Tensor& heatmap, const torch::Tensor& target,
                                                const torch::Tensor& mask)
{
    auto mask_float = mask.to(heatmap.dtype());
    if (mask_float.sizes() != heatmap.sizes())
    {
        mask_float = mask_float.expand_as(heatmap);
    }
    auto target_float = target.to(heatmap.dtype());
    if (target_float.sizes() != heatmap.sizes())
    {
        target_float = target_float.expand_as(heatmap);
    }
    auto positive_mask = target_float.gt(0.0F).to(heatmap.dtype()) * mask_float;
    auto positive_count = positive_mask.sum();
    if (positive_count.item<float>() <= 0.0F)
    {
        return torch::zeros({}, heatmap.options());
    }
    auto stable = heatmap.clamp(1.0e-4F, 1.0F - 1.0e-4F);
    return -(stable.log() * positive_mask).sum() / positive_count.clamp_min(1.0F);
}

torch::Tensor make_repeatable_grid_keypoint_target(const torch::Tensor& mask, const torch::Tensor& heatmap)
{
    auto mask_float = resize_mask_for_heatmap(mask, heatmap).to(torch::kFloat32);
    auto target = torch::zeros_like(mask_float);
    const auto height = heatmap.size(2);
    const auto width = heatmap.size(3);
    for (int64_t batch = 0; batch < heatmap.size(0); ++batch)
    {
        for (int64_t row = 0; row < REPEATABLE_GRID_KEYPOINT_TARGET_ROWS; ++row)
        {
            const auto y0 = row * height / REPEATABLE_GRID_KEYPOINT_TARGET_ROWS;
            const auto y1 = std::max<int64_t>(y0 + 1, (row + 1) * height / REPEATABLE_GRID_KEYPOINT_TARGET_ROWS);
            for (int64_t col = 0; col < REPEATABLE_GRID_KEYPOINT_TARGET_COLS; ++col)
            {
                const auto x0 = col * width / REPEATABLE_GRID_KEYPOINT_TARGET_COLS;
                const auto x1 = std::max<int64_t>(x0 + 1, (col + 1) * width / REPEATABLE_GRID_KEYPOINT_TARGET_COLS);
                auto cell_mask = mask_float.index({batch, 0, torch::indexing::Slice(y0, std::min<int64_t>(y1, height)),
                                                   torch::indexing::Slice(x0, std::min<int64_t>(x1, width))});
                if (!cell_mask.any().item<bool>())
                {
                    continue;
                }
                const auto y = std::min<int64_t>(height - 1, (y0 + y1 - 1) / 2);
                const auto x = std::min<int64_t>(width - 1, (x0 + x1 - 1) / 2);
                if (mask_float.index({batch, 0, y, x}).item<float>() > 0.0F)
                {
                    target.index_put_({batch, 0, y, x}, 1.0F);
                }
                else
                {
                    auto flat = cell_mask.reshape({cell_mask.numel()});
                    auto best = std::get<1>(flat.max(0)).item<int64_t>();
                    const auto cell_width = cell_mask.size(1);
                    target.index_put_({batch, 0, y0 + best / cell_width, x0 + best % cell_width}, 1.0F);
                }
            }
        }
    }
    if (height >= 3 && width >= 3)
    {
        target = torch::max_pool2d(target, {3, 3}, {1, 1}, {1, 1});
    }
    return target * mask_float;
}

torch::Tensor make_repeatable_keypoint_target_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b,
                                                   const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                   const torch::Tensor& warp, const torch::Tensor& mask)
{
    auto target =
        make_repeatable_keypoint_target(view_a, view_b, warp, mask, heatmap_a.size(2), heatmap_a.size(3)).detach();
    auto mask_float = resize_mask_for_heatmap(mask, heatmap_a).to(heatmap_a.dtype());
    auto warped_heatmap_b = warp_heatmap_for_repeatability(heatmap_b, warp);
    return make_heatmap_correspondence_target_loss(heatmap_a, warped_heatmap_b, target, mask_float);
}

std::pair<torch::Tensor, torch::Tensor>
make_warp_aligned_keypoint_targets(const torch::Tensor& view_a, const torch::Tensor& view_b, const torch::Tensor& warp,
                                   const torch::Tensor& mask, int64_t target_height, int64_t target_width)
{
    auto target_a =
        make_repeatable_keypoint_target(view_a, view_b, warp, mask, target_height, target_width, false, false);
    auto target_b = torch::zeros_like(target_a);
    const auto spatial_count = target_height * target_width;
    for (int64_t batch = 0; batch < target_a.size(0); ++batch)
    {
        auto selected = torch::nonzero(target_a.index({batch, 0}).reshape({spatial_count}).gt(0.0)).flatten();
        if (selected.numel() == 0)
        {
            continue;
        }
        auto coordinates = make_descriptor_target_coordinates(warp.index({batch}).unsqueeze(0),
                                                              selected.to(torch::kLong), target_height, target_width)
                               .squeeze(0);
        auto target_x =
            coordinates.index({torch::indexing::Slice(), 0}).round().to(torch::kLong).clamp(0, target_width - 1);
        auto target_y =
            coordinates.index({torch::indexing::Slice(), 1}).round().to(torch::kLong).clamp(0, target_height - 1);
        auto target_indices = target_y * target_width + target_x;
        auto flat = torch::zeros({spatial_count}, target_b.options());
        flat.index_put_({target_indices}, 1.0F);
        target_b.index_put_({batch, 0}, flat.reshape({target_height, target_width}));
    }
    return {target_a, target_b};
}

torch::Tensor make_warp_aligned_keypoint_target_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b,
                                                     const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                     const torch::Tensor& warp, const torch::Tensor& mask)
{
    auto targets = make_warp_aligned_keypoint_targets(view_a, view_b, warp, mask, heatmap_a.size(2), heatmap_a.size(3));
    auto mask_a = resize_mask_for_heatmap(mask, heatmap_a).to(heatmap_a.dtype());
    auto mask_b = torch::ones_like(mask_a);
    auto loss_a = make_heatmap_correspondence_target_loss(heatmap_a, heatmap_a, targets.first.detach(), mask_a);
    auto loss_b = make_heatmap_correspondence_target_loss(heatmap_b, heatmap_b, targets.second.detach(), mask_b);
    return (loss_a + loss_b) * 0.5F;
}

torch::Tensor make_warp_aligned_keypoint_peak_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b,
                                                   const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                   const torch::Tensor& warp, const torch::Tensor& mask)
{
    auto targets = make_warp_aligned_keypoint_targets(view_a, view_b, warp, mask, heatmap_a.size(2), heatmap_a.size(3));
    auto mask_a = resize_mask_for_heatmap(mask, heatmap_a).to(heatmap_a.dtype());
    auto mask_b = torch::ones_like(mask_a);
    auto loss_a = make_heatmap_positive_target_loss(heatmap_a, targets.first.detach(), mask_a);
    auto loss_b = make_heatmap_positive_target_loss(heatmap_b, targets.second.detach(), mask_b);
    return (loss_a + loss_b) * 0.5F;
}

torch::Tensor make_repeatable_grid_keypoint_target_loss(const torch::Tensor& heatmap_a, const torch::Tensor& heatmap_b,
                                                        const torch::Tensor& warp, const torch::Tensor& mask)
{
    auto target = make_repeatable_grid_keypoint_target(mask, heatmap_a).detach();
    auto mask_float = resize_mask_for_heatmap(mask, heatmap_a).to(heatmap_a.dtype());
    auto warped_heatmap_b = warp_heatmap_for_repeatability(heatmap_b, warp);
    return make_heatmap_correspondence_target_loss(heatmap_a, warped_heatmap_b, target, mask_float);
}

torch::Tensor make_decoded_keypoint_repeatability_loss(const FeatureSet& features_a, const torch::Tensor& heatmap_b,
                                                       const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    auto zero = torch::zeros({}, heatmap_b.options());
    if (!features_a.keypoints.defined() || features_a.keypoints.size(0) == 0 || heatmap_b.size(2) == 0 ||
        heatmap_b.size(3) == 0)
    {
        return zero;
    }

    const auto image_width = warp.size(2);
    const auto image_height = warp.size(1);
    auto keypoints_a = features_a.keypoints.to(warp.device(), torch::kFloat32);
    auto image_keypoints_a = scale_feature_keypoints_to_image(keypoints_a, features_a.feature_map_width,
                                                              features_a.feature_map_height, image_width, image_height);
    auto source_x = image_keypoints_a.index({torch::indexing::Slice(), 0}).round().to(torch::kLong);
    auto source_y = image_keypoints_a.index({torch::indexing::Slice(), 1}).round().to(torch::kLong);
    auto source_in_bounds = source_x.ge(0)
                                .logical_and(source_x.lt(image_width))
                                .logical_and(source_y.ge(0))
                                .logical_and(source_y.lt(image_height));
    auto source_x_safe = source_x.clamp(0, image_width - 1);
    auto source_y_safe = source_y.clamp(0, image_height - 1);
    auto source_linear = source_y_safe * image_width + source_x_safe;
    auto source_valid = valid_mask.reshape({image_height * image_width})
                            .to(warp.device(), torch::kBool)
                            .index_select(0, source_linear)
                            .logical_and(source_in_bounds);
    auto sampled_warp = warp.reshape({image_height * image_width, 2}).index_select(0, source_linear);
    auto target_x_image = sampled_warp.index({torch::indexing::Slice(), 0});
    auto target_y_image = sampled_warp.index({torch::indexing::Slice(), 1});
    auto target_in_bounds = target_x_image.ge(0.0F)
                                .logical_and(target_x_image.le(static_cast<float>(image_width - 1)))
                                .logical_and(target_y_image.ge(0.0F))
                                .logical_and(target_y_image.le(static_cast<float>(image_height - 1)));
    auto positive_mask = source_valid.logical_and(target_in_bounds);
    if (!positive_mask.any().item<bool>())
    {
        return zero;
    }

    auto target_x = (target_x_image.index({positive_mask}) + 0.5F) * static_cast<float>(heatmap_b.size(3)) /
                        static_cast<float>(image_width) -
                    0.5F;
    auto target_y = (target_y_image.index({positive_mask}) + 0.5F) * static_cast<float>(heatmap_b.size(2)) /
                        static_cast<float>(image_height) -
                    0.5F;
    auto grid_x = heatmap_b.size(3) > 1 ? target_x / static_cast<float>(heatmap_b.size(3) - 1) * 2.0F - 1.0F
                                        : torch::zeros_like(target_x);
    auto grid_y = heatmap_b.size(2) > 1 ? target_y / static_cast<float>(heatmap_b.size(2) - 1) * 2.0F - 1.0F
                                        : torch::zeros_like(target_y);
    auto grid = torch::stack({grid_x, grid_y}, 1)
                    .reshape({1, target_x.size(0), 1, 2})
                    .to(heatmap_b.device(), heatmap_b.dtype())
                    .contiguous();
    auto sampled_heatmap = torch::nn::functional::grid_sample(heatmap_b, grid,
                                                              torch::nn::functional::GridSampleFuncOptions()
                                                                  .mode(torch::kBilinear)
                                                                  .padding_mode(torch::kZeros)
                                                                  .align_corners(true))
                               .reshape({target_x.size(0)});
    return -sampled_heatmap.clamp(1.0e-4F, 1.0F).log().mean();
}

torch::Tensor offset_pixel_error(const torch::Tensor& offsets, const torch::Tensor& target_offsets,
                                 const torch::Tensor& mask)
{
    using torch::indexing::Slice;

    auto pixel_delta = offsets - target_offsets;
    pixel_delta.index_put_({Slice(), Slice(0, 1), Slice(), Slice()},
                           pixel_delta.index({Slice(), Slice(0, 1), Slice(), Slice()}) * offsets.size(3));
    pixel_delta.index_put_({Slice(), Slice(1, 2), Slice(), Slice()},
                           pixel_delta.index({Slice(), Slice(1, 2), Slice(), Slice()}) * offsets.size(2));
    auto error = pixel_delta.pow(2).sum(1, true).sqrt();
    auto mask_float = mask.to(offsets.dtype());
    auto denom = mask_float.sum();
    if (denom.item<double>() <= 0.0)
    {
        return torch::zeros({}, offsets.options());
    }
    return (error * mask_float).sum() / denom;
}

std::vector<torch::Tensor> float_feature_pyramid(std::vector<torch::Tensor> pyramid);
SparseHeadOutput float_sparse_output(SparseHeadOutput output);
DenseHeadOutput float_dense_output(DenseHeadOutput output);

TrainingLossComponents
training_loss_from_pairs(TrainModules& modules, const std::vector<SyntheticPair>& pairs, const TrainConfig& config,
                         TrainModules* descriptor_anchor_modules = nullptr,
                         int64_t descriptor_broad_far_negative_count = DESCRIPTOR_BROAD_FAR_NEGATIVE_COUNT,
                         std::optional<at::Generator>* training_generator = nullptr)
{
    const auto training_profile = parse_training_profile(config.training_profile);
    const bool use_detector_targets = training_profile_uses_detector_targets(training_profile);
    const bool use_descriptor_losses = training_profile_uses_descriptor_losses(training_profile);
    const bool use_graph_losses = training_profile_uses_graph_losses(training_profile);
    const bool use_dense_pair_loss = training_profile_uses_dense_pair_loss(training_profile);

    std::vector<torch::Tensor> views_a;
    std::vector<torch::Tensor> views_b;
    std::vector<torch::Tensor> warps;
    std::vector<torch::Tensor> valid_masks;
    views_a.reserve(pairs.size());
    views_b.reserve(pairs.size());
    warps.reserve(pairs.size());
    valid_masks.reserve(pairs.size());

    for (const auto& pair : pairs)
    {
        views_a.push_back(pair.view_a);
        views_b.push_back(pair.view_b);
        warps.push_back(pair.warp_a_to_b);
        valid_masks.push_back(pair.valid_mask);
    }

    const auto view_a = stack_batch(views_a, BatchTensorLayout::Chw);
    const auto view_b = stack_batch(views_b, BatchTensorLayout::Chw);
    const auto warp = stack_batch(warps, BatchTensorLayout::Hwc);
    const auto raw_valid_mask = stack_batch(valid_masks, BatchTensorLayout::Hw);
    const auto valid_mask =
        make_pair_loss_valid_mask(view_a, view_b, warp, raw_valid_mask, config.min_keypoint_intensity, training_profile);

    std::vector<torch::Tensor> feature_pyramid_a;
    std::vector<torch::Tensor> feature_pyramid_b;
    SparseHeadOutput sparse_a;
    SparseHeadOutput sparse_b;
    DenseHeadOutput dense;
    {
        const auto use_amp = view_a.is_cuda() && !training_profile_uses_python_aligned_pair_loss(training_profile);
        AmpAutocastGuard autocast_guard(use_amp, c10::DeviceType::CUDA, at::kBFloat16);
        feature_pyramid_a = modules.backbone->forward(view_a);
        feature_pyramid_b = modules.backbone->forward(view_b);
        const auto fpn_a = modules.dual_fpn->forward(feature_pyramid_a);
        const auto fpn_b = modules.dual_fpn->forward(feature_pyramid_b);
        const auto dense_features_a = feature_pyramid_a.front();
        const auto dense_features_b = feature_pyramid_b.front();
        auto raw_sparse_a = modules.sparse_head->forward(fpn_a.first, fpn_a.second);
        auto raw_sparse_b = modules.sparse_head->forward(fpn_b.first, fpn_b.second);
        if (training_profile == TrainingProfile::Smoke)
        {
            sparse_a = adapt_v21_sparse_output(std::move(raw_sparse_a));
            sparse_b = adapt_v21_sparse_output(std::move(raw_sparse_b));
            dense = make_zero_dense_output_like_sparse(sparse_a);
        }
        else if (!training_profile_uses_dense_quality_forward(training_profile))
        {
            if (config.train_blended_descriptors)
            {
                sparse_a = finalize_v21_python_aligned_sparse_output(
                    modules, std::move(raw_sparse_a), view_a, config.training_texture_blend_weight);
                sparse_b = finalize_v21_python_aligned_sparse_output(
                    modules, std::move(raw_sparse_b), view_b, config.training_texture_blend_weight);
            }
            else
            {
                sparse_a = adapt_v21_sparse_output(std::move(raw_sparse_a));
                sparse_b = adapt_v21_sparse_output(std::move(raw_sparse_b));
            }
            dense = make_zero_dense_output_like_sparse(sparse_a);
        }
        else
        {
            const auto dense_a_self = modules.dense_head->forward(dense_features_a, dense_features_a);
            const auto dense_b_self = modules.dense_head->forward(dense_features_b, dense_features_b);
            const auto texture_blend_weight =
                training_profile_uses_python_aligned_pair_loss(training_profile) ? 1.0
                                                                                 : ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT;
            sparse_a =
                finalize_v21_sparse_output(modules, std::move(raw_sparse_a), view_a, dense_a_self.confidence,
                                           texture_blend_weight);
            sparse_b =
                finalize_v21_sparse_output(modules, std::move(raw_sparse_b), view_b, dense_b_self.confidence,
                                           texture_blend_weight);
            if (use_dense_pair_loss)
            {
                dense = adapt_v21_dense_output(modules.dense_head->forward(dense_features_a, dense_features_b));
            }
            else
            {
                dense = DenseHeadOutput{dense_a_self.confidence,
                                        torch::zeros({dense_a_self.confidence.size(0), 2,
                                                      dense_a_self.confidence.size(2), dense_a_self.confidence.size(3)},
                                                     dense_a_self.confidence.options())};
            }
        }
    }
    feature_pyramid_a = float_feature_pyramid(std::move(feature_pyramid_a));
    feature_pyramid_b = float_feature_pyramid(std::move(feature_pyramid_b));
    sparse_a = float_sparse_output(std::move(sparse_a));
    sparse_b = float_sparse_output(std::move(sparse_b));
    dense = float_dense_output(std::move(dense));
    if (config.descriptor_orientation_canonicalization &&
        !training_profile_uses_python_aligned_pair_loss(training_profile))
    {
        sparse_a.descriptors = canonicalize_descriptor_map_by_orientation(sparse_a.descriptors, sparse_a.orientation);
        sparse_b.descriptors = canonicalize_descriptor_map_by_orientation(sparse_b.descriptors, sparse_b.orientation);
    }
    if (training_profile_uses_python_aligned_pair_loss(training_profile))
    {
        std::optional<at::Generator> fallback_generator;
        auto& generator = training_generator != nullptr ? *training_generator : fallback_generator;
        return make_python_compare_training_loss(modules, view_a, view_b, warp, valid_mask, std::move(sparse_a),
                                                 std::move(sparse_b), dense, config, generator);
    }
    const auto sparse_mask = resize_mask_for_heatmap(valid_mask, sparse_a.heatmap);
    const auto dense_mask = resize_mask_for_heatmap(valid_mask, dense.confidence);
    const auto target_offsets = resize_offsets_for_dense_head(warp, dense.offsets);
    auto descriptor_anchor = torch::zeros({}, sparse_a.descriptors.options());
    if (descriptor_anchor_modules != nullptr && use_descriptor_losses)
    {
        torch::NoGradGuard no_grad;
        const auto use_amp = view_a.is_cuda();
        SparseHeadOutput anchor_sparse_a;
        SparseHeadOutput anchor_sparse_b;
        {
            AmpAutocastGuard autocast_guard(use_amp, c10::DeviceType::CUDA, at::kBFloat16);
            auto anchor_pyramid_a = descriptor_anchor_modules->backbone->forward(view_a);
            auto anchor_pyramid_b = descriptor_anchor_modules->backbone->forward(view_b);
            const auto anchor_fpn_a = descriptor_anchor_modules->dual_fpn->forward(anchor_pyramid_a);
            const auto anchor_fpn_b = descriptor_anchor_modules->dual_fpn->forward(anchor_pyramid_b);
            auto anchor_raw_sparse_a =
                descriptor_anchor_modules->sparse_head->forward(anchor_fpn_a.first, anchor_fpn_a.second);
            auto anchor_raw_sparse_b =
                descriptor_anchor_modules->sparse_head->forward(anchor_fpn_b.first, anchor_fpn_b.second);
            const auto anchor_dense_a =
                descriptor_anchor_modules->dense_head->forward(anchor_pyramid_a.front(), anchor_pyramid_a.front());
            const auto anchor_dense_b =
                descriptor_anchor_modules->dense_head->forward(anchor_pyramid_b.front(), anchor_pyramid_b.front());
            anchor_sparse_a = finalize_v21_sparse_output(*descriptor_anchor_modules, std::move(anchor_raw_sparse_a),
                                                         view_a, anchor_dense_a.confidence);
            anchor_sparse_b = finalize_v21_sparse_output(*descriptor_anchor_modules, std::move(anchor_raw_sparse_b),
                                                         view_b, anchor_dense_b.confidence);
        }
        anchor_sparse_a = float_sparse_output(std::move(anchor_sparse_a));
        anchor_sparse_b = float_sparse_output(std::move(anchor_sparse_b));
        if (config.descriptor_orientation_canonicalization)
        {
            anchor_sparse_a.descriptors =
                canonicalize_descriptor_map_by_orientation(anchor_sparse_a.descriptors, anchor_sparse_a.orientation);
            anchor_sparse_b.descriptors =
                canonicalize_descriptor_map_by_orientation(anchor_sparse_b.descriptors, anchor_sparse_b.orientation);
        }
        descriptor_anchor =
            make_descriptor_finetune_anchor_loss(sparse_a.descriptors, sparse_b.descriptors,
                                                 anchor_sparse_a.descriptors, anchor_sparse_b.descriptors, valid_mask);
    }

    auto zero = torch::zeros({}, sparse_a.descriptors.options());
    auto repeatability = zero;
    if (training_profile == TrainingProfile::Smoke)
    {
        repeatability =
            repeatability_loss(sparse_a.heatmap, warp_heatmap_for_repeatability(sparse_b.heatmap, warp), sparse_mask) +
            (make_heatmap_selection_loss(sparse_a.heatmap, sparse_mask) +
             make_heatmap_selection_loss(sparse_b.heatmap, sparse_mask)) *
                0.5;
    }
    else if (use_detector_targets)
    {
        repeatability =
            repeatability_loss(sparse_a.heatmap, warp_heatmap_for_repeatability(sparse_b.heatmap, warp), sparse_mask) +
            (make_heatmap_selection_loss(sparse_a.heatmap, sparse_mask) +
             make_heatmap_selection_loss(sparse_b.heatmap, sparse_mask)) *
                0.5 +
            make_repeatable_saliency_target_loss(sparse_a.heatmap, sparse_b.heatmap, view_a, view_b, warp, valid_mask) *
                REPEATABLE_SALIENCY_TARGET_WEIGHT +
            make_repeatable_keypoint_target_loss(sparse_a.heatmap, sparse_b.heatmap, view_a, view_b, warp, valid_mask) *
                REPEATABLE_KEYPOINT_TARGET_WEIGHT +
            make_warp_aligned_keypoint_target_loss(sparse_a.heatmap, sparse_b.heatmap, view_a, view_b, warp,
                                                   valid_mask) *
                WARP_ALIGNED_KEYPOINT_TARGET_WEIGHT +
            make_warp_aligned_keypoint_peak_loss(sparse_a.heatmap, sparse_b.heatmap, view_a, view_b, warp, valid_mask) *
                WARP_ALIGNED_KEYPOINT_PEAK_WEIGHT +
            make_repeatable_grid_keypoint_target_loss(sparse_a.heatmap, sparse_b.heatmap, warp, valid_mask) *
                REPEATABLE_GRID_KEYPOINT_TARGET_WEIGHT;
    }
    auto orientation =
        training_profile == TrainingProfile::Full
            ? make_orientation_supervision_loss(sparse_a, sparse_b, view_a, view_b, warp, config.min_keypoint_intensity)
            : zero;
    auto descriptor = make_zero_descriptor_training_metrics(sparse_a.descriptors);
    auto texture_descriptor = zero;
    auto pairwise_texture_teacher = zero;
    if (use_descriptor_losses)
    {
        descriptor = make_sparse_descriptor_metrics(sparse_a.descriptors, sparse_b.descriptors, warp, valid_mask,
                                                    torch::Tensor(), descriptor_broad_far_negative_count);
        texture_descriptor = (make_texture_target_descriptor_loss(sparse_a.descriptors, view_a, valid_mask) +
                              make_texture_target_descriptor_loss(sparse_b.descriptors, view_b, valid_mask)) *
                             0.5;
        if constexpr (PAIRWISE_TEXTURE_TEACHER_WEIGHT > 0.0)
        {
            pairwise_texture_teacher = make_pairwise_texture_teacher_descriptor_loss(
                sparse_a.descriptors, sparse_b.descriptors, view_a, warp, valid_mask);
        }
    }
    std::vector<torch::Tensor> graph_losses;
    double graph_loss_weight_sum = 0.0;
    std::vector<torch::Tensor> graph_accuracies;
    std::vector<torch::Tensor> graph_positive_fractions;
    std::vector<torch::Tensor> graph_positive_counts;
    std::vector<torch::Tensor> graph_query_counts;
    std::vector<torch::Tensor> graph_features_a_counts;
    std::vector<torch::Tensor> graph_features_b_counts;
    std::vector<torch::Tensor> learned_graph_accuracies;
    std::vector<torch::Tensor> learned_graph_positive_fractions;
    std::vector<torch::Tensor> learned_graph_positive_counts;
    std::vector<torch::Tensor> learned_graph_query_counts;
    std::vector<torch::Tensor> sparse_keypoint_descriptor_losses;
    std::vector<torch::Tensor> decoded_keypoint_repeatability_losses;
    std::vector<torch::Tensor> keypoint_descriptor_accuracies;
    std::vector<torch::Tensor> keypoint_descriptor_margins;
    std::vector<torch::Tensor> keypoint_descriptor_ranks;
    const auto keypoint_loss_batch_items = std::min<int64_t>(view_a.size(0), TRAINING_KEYPOINT_LOSS_BATCH_ITEMS);
    graph_losses.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    graph_accuracies.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    graph_positive_fractions.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    graph_positive_counts.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    graph_query_counts.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    graph_features_a_counts.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    graph_features_b_counts.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    learned_graph_accuracies.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    learned_graph_positive_fractions.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    learned_graph_positive_counts.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    learned_graph_query_counts.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    sparse_keypoint_descriptor_losses.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    decoded_keypoint_repeatability_losses.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    keypoint_descriptor_accuracies.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    keypoint_descriptor_margins.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    keypoint_descriptor_ranks.reserve(static_cast<size_t>(keypoint_loss_batch_items));
    auto decode_config = config;
    if (use_graph_losses)
    {
        for (int64_t batch = 0; batch < keypoint_loss_batch_items; ++batch)
        {
            SparseHeadOutput sparse_a_item{
                sparse_a.heatmap.index({batch}).unsqueeze(0), sparse_a.descriptors.index({batch}).unsqueeze(0),
                sparse_a.scale.index({batch}).unsqueeze(0), sparse_a.orientation.index({batch}).unsqueeze(0),
                sparse_a.affine.index({batch}).unsqueeze(0)};
            SparseHeadOutput sparse_b_item{
                sparse_b.heatmap.index({batch}).unsqueeze(0), sparse_b.descriptors.index({batch}).unsqueeze(0),
                sparse_b.scale.index({batch}).unsqueeze(0), sparse_b.orientation.index({batch}).unsqueeze(0),
                sparse_b.affine.index({batch}).unsqueeze(0)};
            auto features_a = decode_training_features_fast(view_a.index({batch}), sparse_a_item, decode_config);
            auto features_b = decode_training_features_fast(view_b.index({batch}), sparse_b_item, decode_config);
            const auto keypoint_metrics = make_keypoint_graph_matching_metrics(
                *modules.graph_matcher, features_a, features_b, warp.index({batch}).unsqueeze(0),
                valid_mask.index({batch}).unsqueeze(0));
            auto supervised_features = make_supervised_graph_feature_pair(
                sparse_a_item, sparse_b_item, warp.index({batch}).unsqueeze(0), valid_mask.index({batch}).unsqueeze(0));
            const auto supervised_keypoint_metrics = make_keypoint_graph_matching_metrics(
                *modules.graph_matcher, supervised_features.first, supervised_features.second,
                warp.index({batch}).unsqueeze(0), valid_mask.index({batch}).unsqueeze(0));
            auto warp_completed_features = make_warp_completed_keypoint_feature_pair(
                features_a, sparse_b_item.descriptors, warp.index({batch}).unsqueeze(0),
                valid_mask.index({batch}).unsqueeze(0));
            const auto warp_completed_keypoint_metrics = make_keypoint_graph_matching_metrics(
                *modules.graph_matcher, warp_completed_features.first, warp_completed_features.second,
                warp.index({batch}).unsqueeze(0), valid_mask.index({batch}).unsqueeze(0));
            if (training_profile == TrainingProfile::Full)
            {
                decoded_keypoint_repeatability_losses.push_back(make_decoded_keypoint_repeatability_loss(
                    features_a, sparse_b_item.heatmap, warp.index({batch}).unsqueeze(0),
                    valid_mask.index({batch}).unsqueeze(0)));
            }
            graph_losses.push_back(keypoint_metrics.loss * LEARNED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT);
            graph_loss_weight_sum += LEARNED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT;
            graph_losses.push_back(warp_completed_keypoint_metrics.loss *
                                   WARP_COMPLETED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT);
            graph_loss_weight_sum += WARP_COMPLETED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT;
            graph_losses.push_back(supervised_keypoint_metrics.loss * SUPERVISED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT);
            graph_loss_weight_sum += SUPERVISED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT;
            graph_accuracies.push_back(keypoint_metrics.accuracy);
            graph_accuracies.push_back(supervised_keypoint_metrics.accuracy);
            graph_accuracies.push_back(warp_completed_keypoint_metrics.accuracy);
            const auto count_options =
                torch::TensorOptions().dtype(torch::kFloat32).device(sparse_a.descriptors.device());
            auto make_fraction = [&](const GraphMatchingTrainingMetrics& metrics)
            {
                return torch::full({},
                                   metrics.query_count > 0 ? static_cast<float>(metrics.positive_count) /
                                                                 static_cast<float>(metrics.query_count)
                                                           : 0.0F,
                                   count_options);
            };
            auto record_graph_counts = [&](const GraphMatchingTrainingMetrics& metrics)
            {
                graph_positive_fractions.push_back(make_fraction(metrics));
                graph_positive_counts.push_back(
                    torch::full({}, static_cast<float>(metrics.positive_count), count_options));
                graph_query_counts.push_back(torch::full({}, static_cast<float>(metrics.query_count), count_options));
                graph_features_a_counts.push_back(
                    torch::full({}, static_cast<float>(metrics.features_a_count), count_options));
                graph_features_b_counts.push_back(
                    torch::full({}, static_cast<float>(metrics.features_b_count), count_options));
            };
            record_graph_counts(keypoint_metrics);
            record_graph_counts(supervised_keypoint_metrics);
            record_graph_counts(warp_completed_keypoint_metrics);
            learned_graph_accuracies.push_back(keypoint_metrics.accuracy);
            learned_graph_positive_fractions.push_back(make_fraction(keypoint_metrics));
            learned_graph_positive_counts.push_back(
                torch::full({}, static_cast<float>(keypoint_metrics.positive_count), count_options));
            learned_graph_query_counts.push_back(
                torch::full({}, static_cast<float>(keypoint_metrics.query_count), count_options));
            if (training_profile == TrainingProfile::Full)
            {
                auto keypoint_dense_descriptor = make_keypoint_dense_descriptor_loss(
                    features_a, sparse_b_item.descriptors, warp.index({batch}).unsqueeze(0),
                    valid_mask.index({batch}).unsqueeze(0));
                auto keypoint_patch_descriptor = make_keypoint_patch_descriptor_alignment_loss(
                    features_a, sparse_b_item.descriptors, warp.index({batch}).unsqueeze(0),
                    valid_mask.index({batch}).unsqueeze(0));
                auto warped_keypoint_descriptor = make_warped_keypoint_descriptor_contrastive_loss(
                    features_a, sparse_b_item.descriptors, warp.index({batch}).unsqueeze(0),
                    valid_mask.index({batch}).unsqueeze(0));
                sparse_keypoint_descriptor_losses.push_back(
                    keypoint_metrics.sparse_descriptor.loss * LEARNED_KEYPOINT_DESCRIPTOR_LOSS_WEIGHT +
                    keypoint_dense_descriptor * LEARNED_KEYPOINT_DENSE_DESCRIPTOR_LOSS_WEIGHT +
                    keypoint_patch_descriptor * KEYPOINT_PATCH_DESCRIPTOR_ALIGNMENT_WEIGHT +
                    warped_keypoint_descriptor * WARPED_KEYPOINT_DESCRIPTOR_CONTRASTIVE_WEIGHT);
                sparse_keypoint_descriptor_losses.push_back(warp_completed_keypoint_metrics.sparse_descriptor.loss *
                                                            LEARNED_KEYPOINT_DESCRIPTOR_LOSS_WEIGHT);
                sparse_keypoint_descriptor_losses.push_back(supervised_keypoint_metrics.sparse_descriptor.loss *
                                                            SUPERVISED_KEYPOINT_DESCRIPTOR_LOSS_WEIGHT);
            }
            auto record_keypoint_descriptor_metrics = [&](const GraphMatchingTrainingMetrics& metrics)
            {
                if (metrics.positive_count <= 0)
                {
                    return;
                }
                keypoint_descriptor_accuracies.push_back(metrics.sparse_descriptor.accuracy);
                keypoint_descriptor_margins.push_back(metrics.sparse_descriptor.positive_margin);
                keypoint_descriptor_ranks.push_back(metrics.sparse_descriptor.positive_rank);
            };
            record_keypoint_descriptor_metrics(keypoint_metrics);
            record_keypoint_descriptor_metrics(warp_completed_keypoint_metrics);
        }
    }
    auto dense_graph_matching = use_graph_losses
                                    ? make_graph_matching_loss(*modules.graph_matcher, sparse_a.descriptors,
                                                               sparse_b.descriptors, warp, valid_mask)
                                    : zero;
    auto keypoint_graph_matching = graph_losses.empty() || graph_loss_weight_sum <= 0.0
                                       ? torch::zeros({}, sparse_a.descriptors.options())
                                       : torch::stack(graph_losses).sum() / graph_loss_weight_sum;
    auto graph_matching_accuracy = graph_accuracies.empty() ? torch::zeros({}, sparse_a.descriptors.options())
                                                            : torch::stack(graph_accuracies).mean();
    auto graph_positive_fraction = graph_positive_fractions.empty() ? torch::zeros({}, sparse_a.descriptors.options())
                                                                    : torch::stack(graph_positive_fractions).mean();
    auto graph_positive_count = graph_positive_counts.empty() ? torch::zeros({}, sparse_a.descriptors.options())
                                                              : torch::stack(graph_positive_counts).mean();
    auto graph_query_count = graph_query_counts.empty() ? torch::zeros({}, sparse_a.descriptors.options())
                                                        : torch::stack(graph_query_counts).mean();
    auto graph_features_a_count = graph_features_a_counts.empty() ? torch::zeros({}, sparse_a.descriptors.options())
                                                                  : torch::stack(graph_features_a_counts).mean();
    auto graph_features_b_count = graph_features_b_counts.empty() ? torch::zeros({}, sparse_a.descriptors.options())
                                                                  : torch::stack(graph_features_b_counts).mean();
    auto learned_graph_matching_accuracy = learned_graph_accuracies.empty()
                                               ? torch::zeros({}, sparse_a.descriptors.options())
                                               : torch::stack(learned_graph_accuracies).mean();
    auto learned_graph_positive_fraction = learned_graph_positive_fractions.empty()
                                               ? torch::zeros({}, sparse_a.descriptors.options())
                                               : torch::stack(learned_graph_positive_fractions).mean();
    auto learned_graph_positive_count = learned_graph_positive_counts.empty()
                                            ? torch::zeros({}, sparse_a.descriptors.options())
                                            : torch::stack(learned_graph_positive_counts).mean();
    auto learned_graph_query_count = learned_graph_query_counts.empty()
                                         ? torch::zeros({}, sparse_a.descriptors.options())
                                         : torch::stack(learned_graph_query_counts).mean();
    auto graph_matching = use_graph_losses ? dense_graph_matching + keypoint_graph_matching : zero;
    auto sparse_keypoint_descriptor = sparse_keypoint_descriptor_losses.empty()
                                          ? torch::zeros({}, sparse_a.descriptors.options())
                                          : torch::stack(sparse_keypoint_descriptor_losses).mean();
    auto keypoint_descriptor_accuracy = keypoint_descriptor_accuracies.empty()
                                            ? torch::zeros({}, sparse_a.descriptors.options())
                                            : torch::stack(keypoint_descriptor_accuracies).mean();
    auto keypoint_descriptor_margin = keypoint_descriptor_margins.empty()
                                          ? torch::zeros({}, sparse_a.descriptors.options())
                                          : torch::stack(keypoint_descriptor_margins).mean();
    auto keypoint_descriptor_rank = keypoint_descriptor_ranks.empty() ? torch::zeros({}, sparse_a.descriptors.options())
                                                                      : torch::stack(keypoint_descriptor_ranks).mean();
    auto decoded_keypoint_repeatability = decoded_keypoint_repeatability_losses.empty()
                                              ? torch::zeros({}, sparse_a.descriptors.options())
                                              : torch::stack(decoded_keypoint_repeatability_losses).mean();
    if (training_profile == TrainingProfile::Full)
    {
        repeatability = repeatability + decoded_keypoint_repeatability * DECODED_KEYPOINT_REPEATABILITY_WEIGHT;
    }
    auto offset = use_dense_pair_loss ? masked_smooth_l1_loss(dense.offsets, target_offsets, dense_mask) : zero;
    auto confidence = use_dense_pair_loss ? confidence_bce_loss(dense.confidence, dense_mask) : zero;
    auto offset_error = use_dense_pair_loss ? offset_pixel_error(dense.offsets, target_offsets, dense_mask) : zero;
    auto descriptor_total = descriptor.loss + sparse_keypoint_descriptor +
                            texture_descriptor * ROTATION_INVARIANT_TEXTURE_TARGET_WEIGHT +
                            pairwise_texture_teacher * PAIRWISE_TEXTURE_TEACHER_WEIGHT +
                            descriptor_anchor * DESCRIPTOR_FINETUNE_ANCHOR_WEIGHT;
    return TrainingLossComponents{
        weighted_total_training_loss(repeatability, descriptor_total, orientation, graph_matching, offset, confidence,
                                     descriptor.diversity),
        repeatability,
        descriptor_total,
        orientation,
        graph_matching,
        offset,
        confidence,
        descriptor.accuracy,
        graph_matching_accuracy,
        graph_positive_fraction,
        graph_positive_count,
        graph_query_count,
        graph_features_a_count,
        graph_features_b_count,
        learned_graph_matching_accuracy,
        learned_graph_positive_fraction,
        learned_graph_positive_count,
        learned_graph_query_count,
        descriptor.positive_score,
        descriptor.hard_negative_score,
        descriptor.positive_margin,
        descriptor.positive_rank,
        keypoint_descriptor_accuracy,
        keypoint_descriptor_margin,
        keypoint_descriptor_rank,
        descriptor.diversity,
        offset_error,
        TrainingBatchForward{view_a, view_b, warp, valid_mask, sparse_a, sparse_b, dense.confidence, dense.offsets}};
}

bool should_enqueue_training_visualization(std::size_t enqueued_count, std::size_t visualization_limit)
{
    return enqueued_count < visualization_limit;
}

bool should_use_online_dataloader(const TrainConfig& config)
{
    return config.dataloader_workers > 0 && config.synthetic_pair_cache_dir.empty() &&
           config.extra_synthetic_pair_cache_dirs.empty() && config.hard_synthetic_pair_cache_dirs.empty() &&
           config.pair_cache_dirs.empty() && !config.augmentation_curriculum;
}

DatasetSplit make_training_dataset_split(std::size_t total_images, const TrainConfig& config)
{
    const auto validation_ratio = config.val_ratio;
    const auto train_ratio = config.train_ratio;
    const auto test_ratio = 1.0 - train_ratio - validation_ratio;
    auto split = make_train_validation_test_split(total_images, train_ratio, validation_ratio, test_ratio,
                                                  static_cast<uint64_t>(config.split_seed), true);
    if (split.train.empty() && total_images > 0)
    {
        if (!split.validation.empty())
        {
            split.train.push_back(split.validation.front());
            split.validation.erase(split.validation.begin());
        }
        else if (!split.test.empty())
        {
            split.train.push_back(split.test.front());
            split.test.erase(split.test.begin());
        }
    }
    return split;
}

std::filesystem::path epoch_visualization_dir(const TrainConfig& config, int epoch)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "epoch_%06d", epoch);
    return std::filesystem::path(config.visualization_dir) / buffer;
}

std::filesystem::path static_visualization_dir(const TrainConfig& config)
{
    return std::filesystem::path(config.visualization_dir) / "static";
}

std::shared_ptr<std::pair<FeatureSet, FeatureSet>>
decode_training_diagnostic_features(const TrainingDiagnosticSnapshot& snapshot)
{
    return std::make_shared<std::pair<FeatureSet, FeatureSet>>(
        decode_training_features(snapshot.pair.view_a, snapshot.sparse_a, snapshot.dense_confidence, snapshot.config),
        decode_training_features(snapshot.pair.view_b, snapshot.sparse_b, snapshot.dense_confidence, snapshot.config));
}

void render_static_training_diagnostics(const TrainingDiagnosticSnapshot& snapshot)
{
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

void render_feature_training_diagnostics(const TrainingDiagnosticSnapshot& snapshot,
                                         const std::shared_ptr<std::pair<FeatureSet, FeatureSet>>& features)
{
    const auto stem = pair_visualization_stem(snapshot.pair_index);
    const auto output_dir = epoch_visualization_dir(snapshot.config, snapshot.epoch);

    writeVisualizationImage(
        output_dir / (stem + "_features_a.png"),
        feature_overlay_image(snapshot.pair.view_a, features->first, snapshot.config.min_keypoint_intensity),
        "features=" + std::to_string(features->first.keypoints.size(0)));
    writeVisualizationImage(
        output_dir / (stem + "_features_b.png"),
        feature_overlay_image(snapshot.pair.view_b, features->second, snapshot.config.min_keypoint_intensity),
        "features=" + std::to_string(features->second.keypoints.size(0)));
}

void render_model_match_training_diagnostics(const TrainingDiagnosticSnapshot& snapshot,
                                             const std::shared_ptr<std::pair<FeatureSet, FeatureSet>>& features)
{
    const auto stem = pair_visualization_stem(snapshot.pair_index);
    const auto output_dir = epoch_visualization_dir(snapshot.config, snapshot.epoch);
    const auto matches = matchFeatureSets(features->first, features->second);
    MatchCorrectnessStats stats;
    auto image = match_overlay_image(snapshot.pair.view_a, snapshot.pair.view_b, features->first, features->second,
                                     matches, snapshot.pair.warp_a_to_b, TRAINING_MATCH_CORRECT_THRESHOLD_PIXELS,
                                     snapshot.config.min_keypoint_intensity, &stats);

    writeVisualizationImage(output_dir / (stem + "_model_matches.png"), image,
                            model_match_overlay_text(features->first, features->second, matches, stats));
}

TrainingDiagnosticSnapshot make_training_diagnostic_snapshot(const TrainConfig& config, int epoch,
                                                             std::size_t pair_index, std::size_t batch_index,
                                                             const SyntheticPair& pair,
                                                             const TrainingBatchForward& forward)
{
    const auto dense_confidence = torch::nn::functional::interpolate(
        forward.dense_confidence.index({static_cast<int64_t>(batch_index)}).detach().to(torch::kCPU).unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{forward.sparse_a.heatmap.size(2), forward.sparse_a.heatmap.size(3)})
            .mode(torch::kNearest));
    return TrainingDiagnosticSnapshot{
        config,
        epoch,
        pair_index,
        SyntheticPair{pair.view_a.detach().to(torch::kCPU).contiguous(),
                      pair.view_b.detach().to(torch::kCPU).contiguous(),
                      pair.warp_a_to_b.detach().to(torch::kCPU).contiguous(),
                      pair.valid_mask.detach().to(torch::kCPU).contiguous()},
        SparseHeadOutput{forward.sparse_a.heatmap.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous(),
                         forward.sparse_a.descriptors.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous(),
                         forward.sparse_a.scale.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous(),
                         forward.sparse_a.orientation.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous(),
                         forward.sparse_a.affine.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous()},
        SparseHeadOutput{forward.sparse_b.heatmap.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous(),
                         forward.sparse_b.descriptors.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous(),
                         forward.sparse_b.scale.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous(),
                         forward.sparse_b.orientation.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous(),
                         forward.sparse_b.affine.index({static_cast<int64_t>(batch_index)})
                             .detach()
                             .to(torch::kCPU)
                             .unsqueeze(0)
                             .contiguous()},
        dense_confidence.contiguous()};
}

void enqueue_training_diagnostics(AsyncVisualizationWriter& writer, const TrainConfig& config, int epoch,
                                  std::size_t pair_index, std::size_t batch_index, const SyntheticPair& pair,
                                  const TrainingBatchForward& forward)
{
    auto snapshot = make_training_diagnostic_snapshot(config, epoch, pair_index, batch_index, pair, forward);
    if (epoch == 1)
    {
        writer.enqueueJob(
            [snapshot]()
            {
                render_static_training_diagnostics(snapshot);
            });
    }
    writer.enqueueJob(
        [snapshot]()
        {
            const auto features = decode_training_diagnostic_features(snapshot);
            render_feature_training_diagnostics(snapshot, features);
        });
    writer.enqueueJob(
        [snapshot = std::move(snapshot)]()
        {
            const auto features = decode_training_diagnostic_features(snapshot);
            render_model_match_training_diagnostics(snapshot, features);
        });
}

AugmentationProfile to_augmentation_profile(SyntheticPairAugmentationProfile profile)
{
    switch (profile)
    {
    case SyntheticPairAugmentationProfile::Mixed:
        return AugmentationProfile::Mixed;
    case SyntheticPairAugmentationProfile::RotationOnly:
        return AugmentationProfile::RotationOnly;
    case SyntheticPairAugmentationProfile::Mild:
        return AugmentationProfile::Mild;
    case SyntheticPairAugmentationProfile::Medium:
        return AugmentationProfile::Medium;
    case SyntheticPairAugmentationProfile::Hard:
        return AugmentationProfile::Hard;
    case SyntheticPairAugmentationProfile::Extreme:
        return AugmentationProfile::Extreme;
    case SyntheticPairAugmentationProfile::Viewpoint:
        return AugmentationProfile::Viewpoint;
    case SyntheticPairAugmentationProfile::CompoundViewpoint:
        return AugmentationProfile::CompoundViewpoint;
    }
    return AugmentationProfile::Mixed;
}

SyntheticPairAugmentationProfile effective_augmentation_profile_for_epoch(const TrainConfig& config, int epoch)
{
    const auto requested = parse_synthetic_pair_augmentation_profile(config.augmentation_profile);
    if (!config.augmentation_curriculum)
    {
        return requested;
    }
    const auto first_stage_epochs = std::max(1, config.epochs / 3);
    const auto second_stage_end = std::max(first_stage_epochs + 1, (config.epochs * 2) / 3);
    if (epoch < first_stage_epochs)
    {
        return SyntheticPairAugmentationProfile::Mixed;
    }
    if (epoch < second_stage_end)
    {
        return SyntheticPairAugmentationProfile::Viewpoint;
    }
    return requested;
}

std::string augmentation_profile_name(SyntheticPairAugmentationProfile profile)
{
    switch (profile)
    {
    case SyntheticPairAugmentationProfile::Mixed:
        return "mixed";
    case SyntheticPairAugmentationProfile::RotationOnly:
        return "rotation-only";
    case SyntheticPairAugmentationProfile::Mild:
        return "mild";
    case SyntheticPairAugmentationProfile::Medium:
        return "medium";
    case SyntheticPairAugmentationProfile::Hard:
        return "hard";
    case SyntheticPairAugmentationProfile::Extreme:
        return "extreme";
    case SyntheticPairAugmentationProfile::Viewpoint:
        return "viewpoint";
    case SyntheticPairAugmentationProfile::CompoundViewpoint:
        return "compound-viewpoint";
    }
    return "mixed";
}

ImagePairAugmentationConfig make_online_pair_config(const TrainConfig& config)
{
    ImagePairAugmentationConfig pair_config;
    pair_config.profile = to_augmentation_profile(effective_augmentation_profile_for_epoch(config, 0));
    pair_config.extreme_pair_ratio = config.extreme_pair_ratio;
    pair_config.rotation_step_degrees = static_cast<float>(config.rotation_step_degrees);
    return pair_config;
}

DataLoaderOptions make_dataloader_options(const TrainConfig& config)
{
    DataLoaderOptions options;
    options.batch_size = static_cast<size_t>(config.batch_size);
    options.worker_count = static_cast<size_t>(config.dataloader_workers);
    options.prefetch_batches = static_cast<size_t>(config.prefetch_batches);
    options.pin_memory = config.pin_memory;
    return options;
}

std::unique_ptr<Sampler> make_cache_training_sampler(std::size_t count, const TrainConfig& config)
{
    return std::make_unique<ShuffleSampler>(count, static_cast<uint64_t>(config.seed + 1701));
}

std::vector<std::string> make_training_cache_dirs(const TrainConfig& config)
{
    std::vector<std::string> cache_dirs;
    if (!config.synthetic_pair_cache_dir.empty())
    {
        cache_dirs.push_back(config.synthetic_pair_cache_dir);
    }
    cache_dirs.insert(cache_dirs.end(), config.extra_synthetic_pair_cache_dirs.begin(),
                      config.extra_synthetic_pair_cache_dirs.end());
    for (const auto& hard_cache_dir : config.hard_synthetic_pair_cache_dirs)
    {
        for (int repeat = 0; repeat < config.hard_synthetic_pair_cache_repeats; ++repeat)
        {
            cache_dirs.push_back(hard_cache_dir);
        }
    }
    return cache_dirs;
}

std::vector<TrainingCacheSpec> make_training_cache_specs(const TrainConfig& config)
{
    std::vector<TrainingCacheSpec> cache_specs;
    if (!config.synthetic_pair_cache_dir.empty())
    {
        cache_specs.push_back({config.synthetic_pair_cache_dir, std::nullopt});
    }
    for (const auto& cache_dir : config.extra_synthetic_pair_cache_dirs)
    {
        cache_specs.push_back({cache_dir, std::nullopt});
    }
    for (const auto& hard_cache_dir : config.hard_synthetic_pair_cache_dirs)
    {
        for (int repeat = 0; repeat < config.hard_synthetic_pair_cache_repeats; ++repeat)
        {
            if (config.hard_synthetic_pair_cache_indices.empty())
            {
                cache_specs.push_back({hard_cache_dir, std::nullopt});
            }
            else
            {
                for (const auto index : config.hard_synthetic_pair_cache_indices)
                {
                    cache_specs.push_back({hard_cache_dir, static_cast<std::size_t>(index)});
                }
            }
        }
    }
    return cache_specs;
}

std::vector<std::string> describe_training_cache_specs(const TrainConfig& config)
{
    std::vector<std::string> entries;
    for (const auto& spec : make_training_cache_specs(config))
    {
        entries.push_back(spec.cache_dir + ":" +
                          (spec.pair_index.has_value() ? std::to_string(*spec.pair_index) : "*"));
    }
    return entries;
}

TensorBatchCollator make_synthetic_pair_collator()
{
    return TensorBatchCollator({
        {"view_a", TensorLayout::Chw},
        {"view_b", TensorLayout::Chw},
        {"warp_a_to_b", TensorLayout::Hwc},
        {"valid_mask", TensorLayout::Hw},
    });
}

std::vector<torch::Tensor> float_feature_pyramid(std::vector<torch::Tensor> pyramid)
{
    for (auto& feature : pyramid)
    {
        feature = feature.to(torch::kFloat32);
    }
    return pyramid;
}

SparseHeadOutput float_sparse_output(SparseHeadOutput output)
{
    output.heatmap = output.heatmap.to(torch::kFloat32);
    output.descriptors = output.descriptors.to(torch::kFloat32);
    output.scale = output.scale.to(torch::kFloat32);
    output.orientation = output.orientation.to(torch::kFloat32);
    output.affine = output.affine.to(torch::kFloat32);
    return output;
}

DenseHeadOutput float_dense_output(DenseHeadOutput output)
{
    output.confidence = output.confidence.to(torch::kFloat32);
    output.offsets = output.offsets.to(torch::kFloat32);
    return output;
}

std::vector<SyntheticPair> pairs_from_tensor_batch(TensorBatch batch, torch::Device device, int64_t training_crop_size,
                                                   int64_t resize, std::optional<at::Generator>& generator)
{
    auto view_a = batch.at("view_a").to(device);
    auto view_b = batch.at("view_b").to(device);
    auto warp_a_to_b = batch.at("warp_a_to_b").to(device);
    auto valid_mask = batch.at("valid_mask").to(device);
    std::vector<SyntheticPair> pairs;
    pairs.reserve(static_cast<std::size_t>(view_a.size(0)));
    for (int64_t index = 0; index < view_a.size(0); ++index)
    {
        pairs.push_back(prepare_training_pair_size(
            SyntheticPair{view_a.index({index}).contiguous(), view_b.index({index}).contiguous(),
                          warp_a_to_b.index({index}).contiguous(), valid_mask.index({index}).contiguous()},
            training_crop_size, resize, generator));
    }
    return pairs;
}

SyntheticPairCacheConfig make_cache_config(const TrainConfig& config, std::size_t epoch_size)
{
    SyntheticPairCacheConfig cache_config;
    cache_config.cache_dir = config.synthetic_pair_cache_dir;
    cache_config.resize = config.resize;
    cache_config.pair_count = epoch_size;
    cache_config.pairs_per_image = static_cast<std::size_t>(config.pairs_per_image);
    cache_config.pair_config = make_default_pair_config();
    cache_config.pair_config.augmentation_profile =
        parse_synthetic_pair_augmentation_profile(config.augmentation_profile);
    cache_config.pair_config.extreme_pair_ratio = config.extreme_pair_ratio;
    cache_config.pair_config.rotation_step_degrees = static_cast<float>(config.rotation_step_degrees);
    cache_config.rebuild = config.synthetic_pair_cache_rebuild;
    return cache_config;
}

void move_modules_to_device(TrainModules& modules, torch::Device device)
{
    modules.backbone->to(device);
    modules.dual_fpn->to(device);
    modules.sparse_head->to(device);
    modules.texture_adapter->to(device);
    modules.descriptor_fusion->to(device);
    modules.dense_head->to(device);
    modules.quality_head->to(device);
    modules.semi_dense_branch->to(device);
    modules.graph_matcher->to(device);
}

int64_t read_checkpoint_config_int64(torch::serialize::InputArchive& config_archive, const char* name)
{
    torch::Tensor tensor;
    config_archive.read(name, tensor);
    if (!tensor.defined() || tensor.numel() != 1)
    {
        throw std::invalid_argument(std::string("checkpoint config missing ") + name);
    }
    return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

int64_t read_checkpoint_config_int64_or(torch::serialize::InputArchive& config_archive, const char* name,
                                        int64_t fallback)
{
    try
    {
        return read_checkpoint_config_int64(config_archive, name);
    }
    catch (const c10::Error&)
    {
        return fallback;
    }
}

void load_checkpoint_into_modules(const std::string& checkpoint, const TrainConfig& config, TrainModules& modules)
{
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive config_archive;
    archive.read("config", config_archive);

    const auto input_channels = read_checkpoint_config_int64(config_archive, "input_channels");
    const auto base_channels = read_checkpoint_config_int64(config_archive, "base_channels");
    const auto descriptor_dim = read_checkpoint_config_int64(config_archive, "descriptor_dim");
    const auto graph_hidden_dim =
        read_checkpoint_config_int64_or(config_archive, "graph_hidden_dim", std::max<int64_t>(32, descriptor_dim));
    const auto graph_attention_layers = read_checkpoint_config_int64_or(config_archive, "graph_attention_layers", 1);
    const auto graph_keypoint_meta_dim = read_checkpoint_config_int64_or(config_archive, "graph_keypoint_meta_dim", 16);
    if (input_channels != INPUT_CHANNELS || base_channels != config.base_channels ||
        descriptor_dim != config.descriptor_dim || graph_hidden_dim != config.graph_hidden_dim ||
        graph_attention_layers != config.graph_attention_layers ||
        graph_keypoint_meta_dim != config.graph_keypoint_meta_dim)
    {
        throw std::invalid_argument("init_checkpoint architecture does not match training config");
    }

    torch::serialize::InputArchive backbone_archive;
    torch::serialize::InputArchive dual_fpn_archive;
    torch::serialize::InputArchive sparse_head_archive;
    torch::serialize::InputArchive texture_adapter_archive;
    torch::serialize::InputArchive descriptor_fusion_archive;
    torch::serialize::InputArchive dense_head_archive;
    torch::serialize::InputArchive quality_head_archive;
    torch::serialize::InputArchive semi_dense_branch_archive;
    torch::serialize::InputArchive graph_matcher_archive;
    archive.read("backbone", backbone_archive);
    archive.read("dual_fpn", dual_fpn_archive);
    archive.read("sparse_head", sparse_head_archive);
    archive.read("texture_adapter", texture_adapter_archive);
    archive.read("descriptor_fusion", descriptor_fusion_archive);
    archive.read("dense_head", dense_head_archive);
    archive.read("quality_head", quality_head_archive);
    archive.read("semi_dense_branch", semi_dense_branch_archive);
    archive.read("graph_matcher", graph_matcher_archive);
    modules.backbone->load(backbone_archive);
    modules.backbone->sanitizeNonfiniteState();
    modules.dual_fpn->load(dual_fpn_archive);
    modules.sparse_head->load(sparse_head_archive);
    modules.texture_adapter->load(texture_adapter_archive);
    modules.descriptor_fusion->load(descriptor_fusion_archive);
    modules.dense_head->load(dense_head_archive);
    modules.quality_head->load(quality_head_archive);
    modules.semi_dense_branch->load(semi_dense_branch_archive);
    modules.graph_matcher->load(graph_matcher_archive);
}

void save_checkpoint(const TrainConfig& config, TrainModules& modules)
{
    move_modules_to_device(modules, torch::Device(torch::kCPU));
    modules.backbone->sanitizeNonfiniteState();
    torch::serialize::OutputArchive archive;
    torch::serialize::OutputArchive config_archive;
    config_archive.write("checkpoint_version", torch::tensor({3}, torch::kInt64));
    config_archive.write("base_channels", torch::tensor({config.base_channels}, torch::kInt64));
    config_archive.write("descriptor_dim", torch::tensor({config.descriptor_dim}, torch::kInt64));
    config_archive.write("graph_hidden_dim", torch::tensor({config.graph_hidden_dim}, torch::kInt64));
    config_archive.write("graph_attention_layers", torch::tensor({config.graph_attention_layers}, torch::kInt64));
    config_archive.write("graph_keypoint_meta_dim", torch::tensor({config.graph_keypoint_meta_dim}, torch::kInt64));
    config_archive.write("seed", torch::tensor({config.seed}, torch::kInt64));
    config_archive.write(
        "training_profile_id",
        torch::tensor({training_profile_id(parse_training_profile(config.training_profile))}, torch::kInt64));
    config_archive.write("input_channels", torch::tensor({INPUT_CHANNELS}, torch::kInt64));
    archive.write("config", config_archive);

    torch::serialize::OutputArchive backbone_archive;
    torch::serialize::OutputArchive dual_fpn_archive;
    torch::serialize::OutputArchive sparse_head_archive;
    torch::serialize::OutputArchive texture_adapter_archive;
    torch::serialize::OutputArchive descriptor_fusion_archive;
    torch::serialize::OutputArchive dense_head_archive;
    torch::serialize::OutputArchive quality_head_archive;
    torch::serialize::OutputArchive semi_dense_branch_archive;
    torch::serialize::OutputArchive graph_matcher_archive;
    modules.backbone->save(backbone_archive);
    modules.dual_fpn->save(dual_fpn_archive);
    modules.sparse_head->save(sparse_head_archive);
    modules.texture_adapter->save(texture_adapter_archive);
    modules.descriptor_fusion->save(descriptor_fusion_archive);
    modules.dense_head->save(dense_head_archive);
    modules.quality_head->save(quality_head_archive);
    modules.semi_dense_branch->save(semi_dense_branch_archive);
    modules.graph_matcher->save(graph_matcher_archive);
    archive.write("backbone", backbone_archive);
    archive.write("dual_fpn", dual_fpn_archive);
    archive.write("sparse_head", sparse_head_archive);
    archive.write("texture_adapter", texture_adapter_archive);
    archive.write("descriptor_fusion", descriptor_fusion_archive);
    archive.write("dense_head", dense_head_archive);
    archive.write("quality_head", quality_head_archive);
    archive.write("semi_dense_branch", semi_dense_branch_archive);
    archive.write("graph_matcher", graph_matcher_archive);
    archive.save_to(config.checkpoint);
}

} // namespace

namespace testing
{

torch::Tensor resize_offsets_for_dense_head_for_test(const torch::Tensor& warp, const torch::Tensor& offsets)
{
    return resize_offsets_for_dense_head(warp, offsets);
}

torch::Tensor make_sparse_descriptor_loss_for_test(const torch::Tensor& descriptors_a,
                                                   const torch::Tensor& descriptors_b, const torch::Tensor& warp,
                                                   const torch::Tensor& valid_mask)
{
    return make_sparse_descriptor_loss(descriptors_a, descriptors_b, warp, valid_mask);
}

torch::Tensor make_dense_descriptor_hard_negative_loss_for_test(const torch::Tensor& descriptors_a,
                                                                const torch::Tensor& descriptors_b,
                                                                const torch::Tensor& warp,
                                                                const torch::Tensor& valid_mask)
{
    auto sample_indices = filter_descriptor_sample_indices(make_descriptor_sample_indices(descriptors_a), valid_mask,
                                                           descriptors_a.size(2), descriptors_a.size(3));
    auto target_indices =
        make_descriptor_target_indices(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    return make_dense_descriptor_hard_negative_loss(descriptors_a, descriptors_b, sample_indices, target_indices);
}

torch::Tensor make_bidirectional_dense_descriptor_hard_negative_loss_for_test(const torch::Tensor& descriptors_a,
                                                                              const torch::Tensor& descriptors_b,
                                                                              const torch::Tensor& warp,
                                                                              const torch::Tensor& valid_mask)
{
    auto sample_indices = filter_descriptor_sample_indices(make_descriptor_sample_indices(descriptors_a), valid_mask,
                                                           descriptors_a.size(2), descriptors_a.size(3));
    auto target_indices =
        make_descriptor_target_indices(warp, sample_indices, descriptors_b.size(2), descriptors_b.size(3));
    return make_bidirectional_dense_descriptor_hard_negative_loss(descriptors_a, descriptors_b, sample_indices,
                                                                  target_indices);
}

torch::Tensor make_descriptor_finetune_anchor_loss_for_test(const torch::Tensor& current_a,
                                                            const torch::Tensor& current_b,
                                                            const torch::Tensor& anchor_a,
                                                            const torch::Tensor& anchor_b,
                                                            const torch::Tensor& valid_mask)
{
    return make_descriptor_finetune_anchor_loss(current_a, current_b, anchor_a, anchor_b, valid_mask);
}

torch::Tensor make_warp_descriptor_contrastive_loss_for_test(const torch::Tensor& descriptors_a,
                                                             const torch::Tensor& descriptors_b,
                                                             const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_warp_descriptor_contrastive_loss(descriptors_a, descriptors_b, warp, valid_mask);
}

torch::Tensor make_direct_full_map_descriptor_loss_for_test(const torch::Tensor& descriptors_a,
                                                            const torch::Tensor& descriptors_b,
                                                            const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_direct_full_map_descriptor_loss(descriptors_a, descriptors_b, warp, valid_mask);
}

torch::Tensor make_descriptor_map_regularization_loss_for_test(const torch::Tensor& descriptors)
{
    return make_descriptor_map_regularization_loss(descriptors);
}

torch::Tensor make_descriptor_target_coordinates_for_test(const torch::Tensor& warp,
                                                          const torch::Tensor& sample_indices,
                                                          int64_t descriptor_height, int64_t descriptor_width)
{
    return make_descriptor_target_coordinates(warp, sample_indices, descriptor_height, descriptor_width);
}

torch::Tensor sample_warped_descriptors_for_test(const torch::Tensor& descriptors,
                                                 const torch::Tensor& target_coordinates)
{
    return sample_warped_descriptors(descriptors, target_coordinates);
}

torch::Tensor make_graph_matching_loss_for_test(PlanetaryGraphMatcherImpl& graph_matcher,
                                                const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                                                const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_graph_matching_loss(graph_matcher, descriptors_a, descriptors_b, warp, valid_mask);
}

torch::Tensor assign_graph_matching_targets_for_test(const torch::Tensor& keypoints_a, const torch::Tensor& keypoints_b,
                                                     const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                                     double positive_radius_pixels)
{
    return assign_graph_matching_targets(keypoints_a, keypoints_b, warp, valid_mask, positive_radius_pixels);
}

torch::Tensor make_graph_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t keypoint_count_b,
                                                    int64_t max_candidates)
{
    return make_graph_candidate_indices(target_indices, keypoint_count_b, max_candidates);
}

torch::Tensor make_graph_training_query_indices_for_test(const torch::Tensor& target_indices, int64_t keypoint_count_b,
                                                         int64_t max_queries)
{
    return make_graph_training_query_indices(target_indices, keypoint_count_b, max_queries);
}

torch::Tensor make_keypoint_graph_matching_loss_for_test(PlanetaryGraphMatcherImpl& graph_matcher,
                                                         const FeatureSet& features_a, const FeatureSet& features_b,
                                                         const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_keypoint_graph_matching_loss(graph_matcher, features_a, features_b, warp, valid_mask);
}

torch::Tensor make_keypoint_descriptor_loss_for_test(const FeatureSet& features_a, const FeatureSet& features_b,
                                                     const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_keypoint_descriptor_loss(features_a, features_b, warp, valid_mask);
}

torch::Tensor make_keypoint_descriptor_metric_tensor_for_test(const FeatureSet& features_a,
                                                              const FeatureSet& features_b, const torch::Tensor& warp,
                                                              const torch::Tensor& valid_mask)
{
    const auto metrics = make_keypoint_descriptor_metrics(features_a, features_b, warp, valid_mask);
    return torch::stack({metrics.loss, metrics.accuracy, metrics.positive_score, metrics.hard_negative_score,
                         metrics.positive_margin, metrics.positive_rank, metrics.diversity});
}

torch::Tensor make_keypoint_dense_descriptor_loss_for_test(const FeatureSet& features_a,
                                                           const torch::Tensor& descriptors_b,
                                                           const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_keypoint_dense_descriptor_loss(features_a, descriptors_b, warp, valid_mask);
}

torch::Tensor make_keypoint_patch_descriptor_alignment_loss_for_test(const FeatureSet& features_a,
                                                                     const torch::Tensor& descriptors_b,
                                                                     const torch::Tensor& warp,
                                                                     const torch::Tensor& valid_mask)
{
    return make_keypoint_patch_descriptor_alignment_loss(features_a, descriptors_b, warp, valid_mask);
}

torch::Tensor make_warped_keypoint_descriptor_contrastive_loss_for_test(const FeatureSet& features_a,
                                                                        const torch::Tensor& descriptors_b,
                                                                        const torch::Tensor& warp,
                                                                        const torch::Tensor& valid_mask)
{
    return make_warped_keypoint_descriptor_contrastive_loss(features_a, descriptors_b, warp, valid_mask);
}

torch::Tensor make_decoded_keypoint_repeatability_loss_for_test(const FeatureSet& features_a,
                                                                const torch::Tensor& heatmap_b,
                                                                const torch::Tensor& warp,
                                                                const torch::Tensor& valid_mask)
{
    return make_decoded_keypoint_repeatability_loss(features_a, heatmap_b, warp, valid_mask);
}

std::pair<FeatureSet, FeatureSet> make_warp_completed_keypoint_feature_pair_for_test(const FeatureSet& features_a,
                                                                                     const torch::Tensor& descriptors_b,
                                                                                     const torch::Tensor& warp,
                                                                                     const torch::Tensor& valid_mask)
{
    return make_warp_completed_keypoint_feature_pair(features_a, descriptors_b, warp, valid_mask);
}

torch::Tensor scale_feature_keypoints_to_image_for_test(const torch::Tensor& keypoints, int64_t feature_width,
                                                        int64_t feature_height, int64_t image_width,
                                                        int64_t image_height)
{
    return scale_feature_keypoints_to_image(keypoints, feature_width, feature_height, image_width, image_height);
}

torch::Tensor make_orientation_supervision_loss_for_test(const SparseHeadOutput& sparse_a,
                                                         const SparseHeadOutput& sparse_b, const torch::Tensor& view_a,
                                                         const torch::Tensor& view_b, const torch::Tensor& warp,
                                                         double min_keypoint_intensity)
{
    return make_orientation_supervision_loss(sparse_a, sparse_b, view_a, view_b, warp, min_keypoint_intensity);
}

torch::Tensor make_descriptor_sample_indices_for_test(const torch::Tensor& descriptors)
{
    return make_descriptor_sample_indices(descriptors);
}

torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count)
{
    const auto root = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(spatial_count))));
    const auto inferred_width = root * root == spatial_count ? root : spatial_count;
    return make_descriptor_candidate_indices(target_indices, spatial_count, inferred_width);
}

torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count,
                                                         int64_t broad_far_negative_count)
{
    const auto root = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(spatial_count))));
    const auto inferred_width = root * root == spatial_count ? root : spatial_count;
    return make_descriptor_candidate_indices(target_indices, spatial_count, inferred_width, torch::Tensor(),
                                             broad_far_negative_count);
}

torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count,
                                                         const torch::Tensor& candidate_valid_mask)
{
    const auto root = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(spatial_count))));
    const auto inferred_width = root * root == spatial_count ? root : spatial_count;
    return make_descriptor_candidate_indices(target_indices, spatial_count, inferred_width, candidate_valid_mask);
}

int64_t descriptor_broad_far_negative_count_for_progress_for_test(double progress)
{
    return descriptor_broad_far_negative_count_for_progress(progress);
}

torch::Tensor make_supervised_descriptor_ranking_loss_for_test(const torch::Tensor& sampled_a,
                                                               const torch::Tensor& candidate_b)
{
    return make_supervised_descriptor_ranking_loss(sampled_a, candidate_b);
}

torch::Tensor make_sampled_descriptor_decorrelation_loss_for_test(const torch::Tensor& sampled_descriptors,
                                                                  const torch::Tensor& sample_indices,
                                                                  int64_t descriptor_width)
{
    return make_sampled_descriptor_decorrelation_loss(sampled_descriptors, sample_indices, descriptor_width);
}

torch::Tensor make_positive_descriptor_alignment_loss_for_test(const torch::Tensor& sampled_a,
                                                               const torch::Tensor& positive_b)
{
    return make_positive_descriptor_alignment_loss(sampled_a, positive_b);
}

torch::Tensor make_patch_descriptor_alignment_loss_for_test(const torch::Tensor& descriptors_a,
                                                            const torch::Tensor& descriptors_b,
                                                            const torch::Tensor& warp, const torch::Tensor& valid_mask)
{
    return make_patch_descriptor_alignment_loss(descriptors_a, descriptors_b, warp, valid_mask);
}

torch::Tensor descriptor_candidate_similarity_scores_for_test(const torch::Tensor& descriptors_a,
                                                              const torch::Tensor& candidate_descriptors)
{
    return descriptor_candidate_similarity_scores(descriptors_a, candidate_descriptors);
}

torch::Tensor make_strict_descriptor_cross_entropy_loss_for_test(const torch::Tensor& descriptors_a,
                                                                 const torch::Tensor& descriptors_b,
                                                                 const torch::Tensor& target_indices)
{
    return make_strict_descriptor_cross_entropy_loss(descriptors_a, descriptors_b, target_indices);
}

torch::Tensor blend_rotation_invariant_texture_descriptor_for_test(const torch::Tensor& descriptors,
                                                                   const torch::Tensor& image)
{
    return blend_rotation_invariant_texture_descriptor(descriptors, image);
}

torch::Tensor canonicalize_descriptor_map_by_orientation_for_test(const torch::Tensor& descriptors,
                                                                  const torch::Tensor& orientation)
{
    return canonicalize_descriptor_map_by_orientation(descriptors, orientation);
}

double descriptor_texture_teacher_weight_for_test()
{
    return PAIRWISE_TEXTURE_TEACHER_WEIGHT;
}

double descriptor_texture_target_weight_for_test()
{
    return ROTATION_INVARIANT_TEXTURE_TARGET_WEIGHT;
}

double descriptor_texture_blend_weight_for_test()
{
    return ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT;
}

double descriptor_finetune_anchor_weight_for_test()
{
    return DESCRIPTOR_FINETUNE_ANCHOR_WEIGHT;
}

int64_t descriptor_negative_sample_count_for_test()
{
    return DESCRIPTOR_NEGATIVE_SAMPLE_COUNT;
}

double descriptor_global_ce_weight_for_test()
{
    return DESCRIPTOR_GLOBAL_CE_WEIGHT;
}

int64_t supervised_descriptor_topk_negatives_for_test()
{
    return SUPERVISED_DESCRIPTOR_TOPK_NEGATIVES;
}

double supervised_descriptor_soft_rank_weight_for_test()
{
    return SUPERVISED_DESCRIPTOR_SOFT_RANK_WEIGHT;
}

double supervised_descriptor_tail_rank_weight_for_test()
{
    return SUPERVISED_DESCRIPTOR_TAIL_RANK_WEIGHT;
}

double learned_keypoint_graph_loss_weight_for_test()
{
    return LEARNED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT;
}

double warp_completed_keypoint_graph_loss_weight_for_test()
{
    return WARP_COMPLETED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT;
}

double supervised_keypoint_graph_loss_weight_for_test()
{
    return SUPERVISED_KEYPOINT_GRAPH_MATCHING_LOSS_WEIGHT;
}

int64_t training_variant_index_for_pair_for_test(std::size_t pair_index, std::size_t train_image_count, int epoch,
                                                 int pairs_per_image)
{
    return training_variant_index_for_pair(pair_index, train_image_count, epoch, pairs_per_image);
}

torch::Tensor limit_training_image_size_for_test(const torch::Tensor& image, int64_t max_edge)
{
    return limit_training_image_size(image, max_edge);
}

SyntheticPair limit_training_pair_size_for_test(const SyntheticPair& pair, int64_t max_edge)
{
    return limit_training_pair_size(pair, max_edge);
}

SyntheticPair crop_training_pair_with_seed_for_test(const SyntheticPair& pair, int64_t crop_size, uint64_t seed)
{
    auto generator = make_training_random_generator(pair.view_a.device(), seed);
    return crop_training_pair(pair, crop_size, generator);
}

torch::Tensor stack_chw_batch_for_test(const std::vector<torch::Tensor>& tensors)
{
    return stack_batch(tensors, BatchTensorLayout::Chw);
}

torch::Tensor stack_hw_batch_for_test(const std::vector<torch::Tensor>& tensors)
{
    return stack_batch(tensors, BatchTensorLayout::Hw);
}

torch::Tensor stack_hwc_batch_for_test(const std::vector<torch::Tensor>& tensors)
{
    return stack_batch(tensors, BatchTensorLayout::Hwc);
}

torch::Tensor make_cache_training_sample_indices_for_test(std::size_t count, const TrainConfig& config)
{
    auto sampler = make_cache_training_sampler(count, config);
    auto indices = sampler->indices();
    return torch::tensor(std::vector<int64_t>(indices.begin(), indices.end()),
                         torch::TensorOptions().dtype(torch::kLong));
}

std::vector<std::string> make_training_cache_dirs_for_test(const TrainConfig& config)
{
    return make_training_cache_dirs(config);
}

std::vector<std::string> make_training_cache_entries_for_test(const TrainConfig& config)
{
    return describe_training_cache_specs(config);
}

torch::Tensor weighted_total_training_loss_for_test(const torch::Tensor& repeatability, const torch::Tensor& descriptor,
                                                    const torch::Tensor& offset, const torch::Tensor& confidence,
                                                    const torch::Tensor& descriptor_diversity)
{
    return weighted_total_training_loss(repeatability, descriptor, torch::zeros({}, descriptor.options()),
                                        torch::zeros({}, descriptor.options()), offset, confidence,
                                        descriptor_diversity);
}

torch::Tensor warp_heatmap_for_repeatability_for_test(const torch::Tensor& heatmap, const torch::Tensor& warp)
{
    return warp_heatmap_for_repeatability(heatmap, warp);
}

torch::Tensor make_heatmap_correspondence_target_loss_for_test(const torch::Tensor& heatmap_a,
                                                               const torch::Tensor& heatmap_b_at_a,
                                                               const torch::Tensor& target, const torch::Tensor& mask)
{
    return make_heatmap_correspondence_target_loss(heatmap_a, heatmap_b_at_a, target, mask);
}

torch::Tensor make_heatmap_positive_target_loss_for_test(const torch::Tensor& heatmap, const torch::Tensor& target,
                                                         const torch::Tensor& mask)
{
    return make_heatmap_positive_target_loss(heatmap, target, mask);
}

torch::Tensor make_training_valid_mask_for_test(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                                double min_keypoint_intensity)
{
    return make_training_valid_mask(view_a, view_b, warp, valid_mask, min_keypoint_intensity);
}

torch::Tensor make_pair_loss_valid_mask_for_test(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                 const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                                 double min_keypoint_intensity,
                                                 const std::string& training_profile)
{
    return make_pair_loss_valid_mask(view_a, view_b, warp, valid_mask, min_keypoint_intensity,
                                     parse_training_profile(training_profile));
}

torch::Tensor make_warp_aligned_keypoint_targets_for_test(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                          const torch::Tensor& warp, const torch::Tensor& mask,
                                                          int64_t target_height, int64_t target_width)
{
    auto targets = make_warp_aligned_keypoint_targets(view_a, view_b, warp, mask, target_height, target_width);
    return torch::cat({targets.first, targets.second}, 0);
}

FeatureSet decode_training_features_fast_for_test(const torch::Tensor& view, const SparseHeadOutput& sparse,
                                                  const TrainConfig& config)
{
    return decode_training_features_fast(view, sparse, config);
}

torch::Tensor training_warp_overlay_image_for_test(const SyntheticPair& pair)
{
    return warp_overlay_image(pair);
}

torch::Tensor training_feature_overlay_image_for_test(const torch::Tensor& image, const FeatureSet& features,
                                                      double min_keypoint_intensity)
{
    return feature_overlay_image(image, features, min_keypoint_intensity);
}

torch::Tensor training_match_overlay_image_for_test(const torch::Tensor& image_a, const torch::Tensor& image_b,
                                                    const FeatureSet& features_a, const FeatureSet& features_b,
                                                    const MatchSet& matches)
{
    return match_overlay_image(image_a, image_b, features_a, features_b, matches, 0.0);
}

torch::Tensor training_match_overlay_image_for_test(const torch::Tensor& image_a, const torch::Tensor& image_b,
                                                    const FeatureSet& features_a, const FeatureSet& features_b,
                                                    const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                                    double correct_threshold_pixels)
{
    return match_overlay_image(image_a, image_b, features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels,
                               0.0);
}

std::string training_model_match_overlay_text_for_test(const FeatureSet& features_a, const FeatureSet& features_b,
                                                       const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                                       double correct_threshold_pixels)
{
    return model_match_overlay_text(features_a, features_b, matches, warp_a_to_b, correct_threshold_pixels);
}

bool should_enqueue_training_visualization_for_test(std::size_t enqueued_count, std::size_t visualization_limit)
{
    return should_enqueue_training_visualization(enqueued_count, visualization_limit);
}

bool should_use_online_dataloader_for_test(const TrainConfig& config)
{
    return should_use_online_dataloader(config);
}

std::string effective_augmentation_profile_for_epoch_for_test(const TrainConfig& config, int epoch)
{
    return augmentation_profile_name(effective_augmentation_profile_for_epoch(config, epoch));
}

std::vector<std::size_t> make_training_image_indices_for_test(std::size_t total_images, const TrainConfig& config)
{
    return make_training_dataset_split(total_images, config).train;
}

std::vector<std::size_t> make_validation_image_indices_for_test(std::size_t total_images, const TrainConfig& config)
{
    return make_training_dataset_split(total_images, config).validation;
}

double training_learning_rate_for_step_for_test(const TrainConfig& config, int64_t step, int64_t total_steps)
{
    return training_learning_rate_for_step(config, step, total_steps);
}

bool training_profile_uses_dense_quality_forward_for_test(const std::string& training_profile)
{
    return training_profile_uses_dense_quality_forward(parse_training_profile(training_profile));
}

std::vector<std::string> trainable_parameter_names_for_config_for_test(const TrainConfig& config)
{
    auto modules = make_modules(config, torch::kCPU);
    apply_trainable_parameter_selection(modules, config);
    return trainable_parameter_names(modules);
}

torch::Tensor make_python_compare_graph_loss_for_test(v21::PfmV21GraphMatcherImpl& graph_matcher,
                                                      const torch::Tensor& desc_a, const torch::Tensor& desc_b,
                                                      const torch::Tensor& points_a, const torch::Tensor& points_b,
                                                      int64_t meta_dim)
{
    return make_python_compare_graph_loss(graph_matcher, desc_a, desc_b, points_a, points_b, meta_dim);
}

torch::Tensor make_python_compare_graph_loss_for_test(v21::PfmV21GraphMatcherImpl& graph_matcher,
                                                      const torch::Tensor& desc_a, const torch::Tensor& desc_b,
                                                      const torch::Tensor& points_a, const torch::Tensor& points_b,
                                                      int64_t meta_dim, double accept_weight)
{
    return make_python_compare_graph_loss(graph_matcher, desc_a, desc_b, points_a, points_b, meta_dim, accept_weight);
}

torch::Tensor make_python_compare_graph_loss_for_test(v21::PfmV21GraphMatcherImpl& graph_matcher,
                                                      const torch::Tensor& desc_a, const torch::Tensor& desc_b,
                                                      const torch::Tensor& points_a, const torch::Tensor& points_b,
                                                      int64_t meta_dim, double accept_weight,
                                                      double prune_ranking_weight, double prune_ranking_margin)
{
    return make_python_compare_graph_loss(graph_matcher, desc_a, desc_b, points_a, points_b, meta_dim, accept_weight,
                                          8, prune_ranking_weight, prune_ranking_margin);
}

torch::Tensor make_python_compare_graph_loss_for_test(v21::PfmV21GraphMatcherImpl& graph_matcher,
                                                      const torch::Tensor& desc_a, const torch::Tensor& desc_b,
                                                      const torch::Tensor& points_a, const torch::Tensor& points_b,
                                                      int64_t meta_dim, double accept_weight,
                                                      double prune_ranking_weight, double prune_ranking_margin,
                                                      double stop_confidence_weight)
{
    return make_python_compare_graph_loss(graph_matcher, desc_a, desc_b, points_a, points_b, meta_dim, accept_weight,
                                          8, prune_ranking_weight, prune_ranking_margin, stop_confidence_weight, 0.5);
}

torch::Tensor make_python_compare_graph_loss_with_attention_budget_for_test(
    v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim, int64_t max_attention_layers)
{
    return make_python_compare_graph_loss(graph_matcher, desc_a, desc_b, points_a, points_b, meta_dim, 0.0, 8, 0.0,
                                          0.25, 0.0, 0.5, max_attention_layers);
}

torch::Tensor make_python_compare_graph_loss_with_random_attention_budget_for_test(
    v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim, int64_t max_attention_layers,
    uint64_t seed)
{
    auto generator = make_training_random_generator(desc_a.device(), seed);
    return make_python_compare_graph_loss(graph_matcher, desc_a, desc_b, points_a, points_b, meta_dim, 0.0, 8, 0.0,
                                          0.25, 0.0, 0.5, max_attention_layers, true, &generator);
}

torch::Tensor make_python_compare_graph_loss_with_attention_work_fraction_for_test(
    v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim, double max_attention_work_fraction)
{
    return make_python_compare_graph_loss(graph_matcher, desc_a, desc_b, points_a, points_b, meta_dim, 0.0, 8, 0.0,
                                          0.25, 0.0, 0.5, 0, false, nullptr, -1, 1.0, nullptr,
                                          max_attention_work_fraction);
}

std::pair<torch::Tensor, int64_t> make_python_compare_graph_loss_with_width_keep_ratio_for_test(
    v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim, double width_keep_ratio,
    uint64_t seed)
{
    auto generator = make_training_random_generator(desc_a.device(), seed);
    int64_t selected_positive_count = 0;
    auto loss = make_python_compare_graph_loss(graph_matcher, desc_a, desc_b, points_a, points_b, meta_dim, 0.0, 8,
                                               0.0, 0.25, 0.0, 0.5, 0, false, &generator, -1, width_keep_ratio,
                                               &selected_positive_count);
    return {loss, selected_positive_count};
}

} // namespace testing

TrainResult train_model(const TrainConfig& config)
{
    validate_config(config);
    Timer total_timer;
    int64_t completed_batches = 0;
    double accumulated_batch_seconds = 0.0;
    const auto device = resolve_compute_device(config.device);
    torch::manual_seed(static_cast<uint64_t>(config.seed));
    std::unique_ptr<ImageDataset> dataset;
    if (!config.image_dir.empty())
    {
        // 原始图像数据集只在在线合成或生成 synthetic cache 时需要；纯 pair archive 训练可以不提供。
        dataset = std::make_unique<ImageDataset>(config.image_dir);
    }
    auto modules = make_modules(config, device);
    if (!config.init_checkpoint.empty())
    {
        // 初始化检查点加载后重新切到训练模式，随后再根据微调模式冻结部分模块。
        load_checkpoint_into_modules(config.init_checkpoint, config, modules);
        move_modules_to_device(modules, device);
        set_all_modules_train(modules);
    }
    apply_trainable_parameter_selection(modules, config);
    keep_descriptor_only_frozen_modules_eval(modules, config);
    keep_graph_only_frozen_modules_eval(modules, config);
    apply_training_profile_module_mode(modules, config);
    std::unique_ptr<TrainModules> descriptor_anchor_modules;
    if (should_use_descriptor_finetune_anchor(config))
    {
        // 描述子微调锚点保留一份冻结 teacher，用于约束已有旋转基线不要漂移过大。
        auto anchor_modules = make_modules(config, device);
        load_checkpoint_into_modules(config.init_checkpoint, config, anchor_modules);
        move_modules_to_device(anchor_modules, device);
        set_all_modules_trainable(anchor_modules, false);
        set_all_modules_eval(anchor_modules);
        descriptor_anchor_modules = std::make_unique<TrainModules>(std::move(anchor_modules));
    }
    auto parameters = module_parameters(modules);
    if (parameters.empty())
    {
        throw std::invalid_argument("no trainable parameters selected");
    }
    std::cout << "trainable parameters: tensors=" << parameters.size()
              << " values=" << count_trainable_parameter_values(parameters) << "\n";
    auto optimizer_options = torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay);
    auto optimizer = torch::optim::AdamW(parameters, optimizer_options);
    auto training_generator = make_training_random_generator(device, static_cast<uint64_t>(config.seed));
    std::unique_ptr<CsvMetricLogger> csv_logger;
    std::unique_ptr<GpuMetricProvider> gpu_metric_provider;
    if (!config.log_csv.empty())
    {
        csv_logger = std::make_unique<CsvMetricLogger>(config.log_csv, TRAINING_CSV_COLUMNS);
        gpu_metric_provider = makeDefaultGpuMetricProvider();
    }

    ConsoleProgressLogger progress_logger(std::cout, 30);

    TrainResult result;
    double first_loss = 0.0;
    double last_loss = 0.0;
    bool has_loss = false;

    const auto total_images = dataset ? dataset->size() : 0;
    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> validation_indices;
    if (dataset)
    {
        const auto dataset_split = make_training_dataset_split(total_images, config);
        train_indices = dataset_split.train;
        validation_indices = dataset_split.validation;
    }
    const auto train_images = train_indices.size();
    const auto val_images = validation_indices.size();
    const auto train_epoch_size = train_images * static_cast<std::size_t>(config.pairs_per_image);
    std::size_t epoch_size = train_epoch_size;
    int64_t global_step = 0;
    std::unique_ptr<AsyncVisualizationWriter> visualization_writer;
    auto pair_config = make_default_pair_config();
    pair_config.augmentation_profile = effective_augmentation_profile_for_epoch(config, 0);
    pair_config.extreme_pair_ratio = config.extreme_pair_ratio;
    pair_config.rotation_step_degrees = static_cast<float>(config.rotation_step_degrees);
    if (!config.synthetic_pair_cache_dir.empty())
    {
        // 主 synthetic cache 可在训练前自动准备；配置不匹配时由 cache 模块决定是否重建。
        if (!dataset)
        {
            throw std::invalid_argument("synthetic_pair_cache_dir requires image_dir");
        }
        prepare_synthetic_pair_cache(
            *dataset, make_cache_config(config, dataset->size() * static_cast<std::size_t>(config.pairs_per_image)));
    }
    if (config.cache_only)
    {
        result.total_time_seconds = total_timer.elapsedSeconds();
        std::cout << "cache generation complete: dir=" << config.synthetic_pair_cache_dir
                  << " pairs=" << total_images * static_cast<std::size_t>(config.pairs_per_image)
                  << " elapsed=" << formatSeconds(result.total_time_seconds) << '\n';
        return result;
    }
    const auto cache_specs = make_training_cache_specs(config);

    std::unique_ptr<CompositeSyntheticPairCacheDataset> cache_dataset;
    if (!cache_specs.empty())
    {
        // 旧式 synthetic cache 支持普通 cache、extra cache 和 hard cache 重复采样。
        cache_dataset = std::make_unique<CompositeSyntheticPairCacheDataset>(cache_specs);
        epoch_size = cache_dataset->size();
    }
    std::shared_ptr<TensorDataset> pair_archive_tensor_dataset;
    if (!config.pair_cache_dirs.empty())
    {
        // pair archive cache 是当前仿真数据主入口，多个目录会组合成一个 TensorDataset。
        std::vector<std::shared_ptr<TensorDataset>> tensor_datasets;
        tensor_datasets.reserve(config.pair_cache_dirs.size());
        const auto cache_dir_count = config.pair_cache_dirs.size();
        const auto per_dir_memory_cache_size =
            config.pair_memory_cache_size > 0
                ? std::max<std::size_t>(1, static_cast<std::size_t>(config.pair_memory_cache_size) / cache_dir_count)
                : 0;
        for (const auto& cache_dir : config.pair_cache_dirs)
        {
            PairArchiveDatasetConfig pair_config_archive;
            pair_config_archive.cache_dir = cache_dir;
            pair_config_archive.limit_pairs = config.pair_cache_limit;
            tensor_datasets.push_back(
                std::make_shared<PairArchiveTensorDataset>(std::move(pair_config_archive), per_dir_memory_cache_size));
        }
        pair_archive_tensor_dataset = std::make_shared<CompositeTensorDataset>(std::move(tensor_datasets));
        epoch_size = pair_archive_tensor_dataset->size();
    }
    std::cout << "training data: raw_images=" << total_images << " train_images=" << train_images
              << " validation_images=" << val_images << " cache_entries=" << cache_specs.size()
              << " pair_archive_dirs=" << config.pair_cache_dirs.size() << " pairs_per_epoch=" << epoch_size
              << " pairs_per_image=" << config.pairs_per_image << " training_profile=" << config.training_profile
              << " training_crop_size=" << config.training_crop_size << " seed=" << config.seed
              << " memory_cache_items=" << config.pair_memory_cache_size
              << " dataloader_workers=" << config.dataloader_workers
              << " prefetch_batches=" << config.prefetch_batches
              << '\n';
    if (config.max_train_batches > 0)
    {
        epoch_size = std::min<std::size_t>(epoch_size, static_cast<std::size_t>(config.max_train_batches) *
                                                           static_cast<std::size_t>(config.batch_size));
        std::cout << "training data limited: max_train_batches=" << config.max_train_batches
                  << " effective_pairs_per_epoch=" << epoch_size << '\n';
    }
    std::unique_ptr<AsyncDataLoader> cache_loader;
    if (cache_dataset && config.dataloader_workers > 0)
    {
        // cached synthetic pair 可以异步读取；若没有 worker，则走同步 load_cached_pairs，便于调试。
        std::vector<std::shared_ptr<TensorDataset>> tensor_datasets;
        tensor_datasets.reserve(cache_specs.size());
        for (const auto& cache_spec : cache_specs)
        {
            if (cache_spec.pair_index.has_value())
            {
                tensor_datasets.push_back(std::make_shared<IndexedSyntheticPairCacheTensorDataset>(
                    cache_spec.cache_dir, std::vector<std::size_t>{*cache_spec.pair_index}));
            }
            else
            {
                tensor_datasets.push_back(std::make_shared<SyntheticPairCacheTensorDataset>(cache_spec.cache_dir));
            }
        }
        auto tensor_dataset = std::make_shared<CompositeTensorDataset>(std::move(tensor_datasets));
        cache_loader = std::make_unique<AsyncDataLoader>(
            tensor_dataset, make_cache_training_sampler(tensor_dataset->size(), config), make_synthetic_pair_collator(),
            make_dataloader_options(config));
    }
    std::unique_ptr<AsyncDataLoader> pair_archive_loader;
    if (pair_archive_tensor_dataset)
    {
        // pair archive 默认走异步 DataLoader，保持大规模仿真 cache 读取吞吐。
        pair_archive_loader = std::make_unique<AsyncDataLoader>(
            pair_archive_tensor_dataset, make_cache_training_sampler(pair_archive_tensor_dataset->size(), config),
            make_synthetic_pair_collator(), make_dataloader_options(config));
    }
    std::unique_ptr<AsyncDataLoader> online_loader;
    if (should_use_online_dataloader(config))
    {
        // 在线合成适合小数据 smoke 或无 cache 实验；图像先加载到内存，worker 只负责生成 pair。
        std::vector<torch::Tensor> images;
        images.reserve(train_images);
        for (std::size_t index = 0; index < train_images; ++index)
        {
            images.push_back(
                limit_training_image_size(ensure_grayscale(dataset->load(train_indices[index])), config.resize));
        }
        auto online_dataset = std::make_shared<SyntheticPairTensorDataset>(
            std::move(images), static_cast<size_t>(config.pairs_per_image), make_online_pair_config(config));
        online_loader = std::make_unique<AsyncDataLoader>(
            online_dataset, std::make_unique<SequentialSampler>(online_dataset->size()), make_synthetic_pair_collator(),
            make_dataloader_options(config));
    }
    const auto epoch_iterations = static_cast<int64_t>((epoch_size + static_cast<std::size_t>(config.batch_size) - 1) /
                                                       static_cast<std::size_t>(config.batch_size));
    const auto total_steps = static_cast<int64_t>(config.epochs) * epoch_iterations;
    const auto total_iterations = static_cast<double>(std::max<int64_t>(1, total_steps));
    const std::size_t visualization_limit =
        config.visualization_samples_all
            ? epoch_size
            : std::min<std::size_t>(static_cast<std::size_t>(config.visualization_samples), epoch_size);
    if (!config.visualization_dir.empty() && visualization_limit > 0)
    {
        // 训练可视化异步写出，避免 PNG 编码阻塞主训练循环。
        std::cout << "training visualization: dir=" << config.visualization_dir << " samples="
                  << (config.visualization_samples_all ? "all" : std::to_string(config.visualization_samples))
                  << " max_keypoints=" << config.max_keypoints << " min_keypoints=" << config.min_keypoints
                  << " keypoint_grid=" << config.keypoint_grid_rows << 'x' << config.keypoint_grid_cols
                  << " keypoints_per_cell=" << config.keypoints_per_cell << " nms_radius=" << config.nms_radius
                  << " async_queue=" << TRAINING_VISUALIZATION_QUEUE_CAPACITY
                  << " async_workers=" << TRAINING_VISUALIZATION_WORKER_COUNT << '\n';
        visualization_writer = std::make_unique<AsyncVisualizationWriter>(TRAINING_VISUALIZATION_QUEUE_CAPACITY,
                                                                          TRAINING_VISUALIZATION_WORKER_COUNT);
    }

    for (int epoch = 0; epoch < config.epochs; ++epoch)
    {
        Timer epoch_timer;
        // augmentation curriculum 在每个 epoch 开始时更新，cache 数据不受在线增强档位影响。
        pair_config.augmentation_profile = effective_augmentation_profile_for_epoch(config, epoch);
        if (epoch > 0)
        {
            if (online_loader)
            {
                online_loader->reset();
            }
            if (cache_loader)
            {
                cache_loader->reset();
            }
            if (pair_archive_loader)
            {
                pair_archive_loader->reset();
            }
        }
        for (std::size_t offset = 0; offset < epoch_size; offset += static_cast<std::size_t>(config.batch_size))
        {
            Timer batch_timer;
            const auto batch_end = offset + static_cast<std::size_t>(config.batch_size);
            const auto end = std::min<std::size_t>(epoch_size, batch_end);
            std::vector<SyntheticPair> pairs;
            if (online_loader)
            {
                // 数据源优先级：在线 loader、pair archive loader、synthetic cache loader、同步
                // cache、最后原图在线合成。
                auto batch = online_loader->next();
                if (!batch.has_value())
                {
                    throw std::runtime_error("online dataloader exhausted before epoch end");
                }
                pairs = pairs_from_tensor_batch(std::move(batch.value()), device, config.training_crop_size,
                                                config.resize, training_generator);
            }
            else if (pair_archive_loader)
            {
                auto batch = pair_archive_loader->next();
                if (!batch.has_value())
                {
                    throw std::runtime_error("pair archive dataloader exhausted before epoch end");
                }
                pairs = pairs_from_tensor_batch(std::move(batch.value()), device, config.training_crop_size,
                                                config.resize, training_generator);
            }
            else if (cache_loader)
            {
                auto batch = cache_loader->next();
                if (!batch.has_value())
                {
                    throw std::runtime_error("cache dataloader exhausted before epoch end");
                }
                pairs = pairs_from_tensor_batch(std::move(batch.value()), device, config.training_crop_size,
                                                config.resize, training_generator);
            }
            else if (cache_dataset)
            {
                pairs = load_cached_pairs(*cache_dataset, offset, end, device, config.training_crop_size,
                                          config.resize, training_generator);
            }
            else
            {
                std::vector<torch::Tensor> images;
                std::vector<int64_t> source_indices;
                std::vector<int64_t> variant_indices;
                images.reserve(end - offset);
                source_indices.reserve(end - offset);
                variant_indices.reserve(end - offset);
                for (std::size_t index = offset; index < end; ++index)
                {
                    const auto source_index = train_indices[index % train_images];
                    images.push_back(
                        limit_training_image_size(ensure_grayscale(dataset->load(source_index)), config.resize));
                    source_indices.push_back(static_cast<int64_t>(source_index));
                    variant_indices.push_back(
                        training_variant_index_for_pair(index, train_images, epoch, config.pairs_per_image));
                }
                pairs = make_synthetic_pairs_from_batch(stack_batch(images, BatchTensorLayout::Chw).to(device),
                                                        source_indices, variant_indices, pair_config);
            }

            const auto curriculum_progress =
                total_iterations > 1.0 ? static_cast<double>(global_step) / std::max(1.0, total_iterations - 1.0) : 1.0;
            const auto descriptor_broad_far_negative_count =
                descriptor_broad_far_negative_count_for_progress(curriculum_progress);
            const auto current_learning_rate = training_learning_rate_for_step(config, global_step, total_steps);
            set_optimizer_learning_rate(optimizer, current_learning_rate);
            // training_loss_from_pairs 内部同时计算检测器、描述子、图匹配和稠密分支损失，并返回诊断指标。
            TrainingLossComponents loss;
            try
            {
                loss = training_loss_from_pairs(modules, pairs, config, descriptor_anchor_modules.get(),
                                                descriptor_broad_far_negative_count, &training_generator);
            }
            catch (const std::runtime_error& exc)
            {
                if (!is_no_valid_correspondence_error(exc))
                {
                    throw;
                }
                const auto iteration = static_cast<int>((offset / static_cast<std::size_t>(config.batch_size)) + 1);
                std::cerr << "training batch skipped: no valid correspondences sampled epoch=" << (epoch + 1)
                          << " iteration=" << iteration << '\n';
                optimizer.zero_grad();
                continue;
            }
            if (!torch::isfinite(loss.total.detach()).item<bool>())
            {
                optimizer.zero_grad();
                continue;
            }
            if (visualization_writer && offset < visualization_limit)
            {
                // 可视化使用本 batch 前向结果，不额外跑模型；只对配置要求的样本排队。
                for (std::size_t pair_offset = 0; pair_offset < pairs.size(); ++pair_offset)
                {
                    const auto pair_index = offset + pair_offset;
                    if (!should_enqueue_training_visualization(pair_index, visualization_limit))
                    {
                        break;
                    }
                    enqueue_training_diagnostics(*visualization_writer, config, epoch + 1, pair_index, pair_offset,
                                                 pairs[pair_offset], loss.forward);
                }
            }

            optimizer.zero_grad();
            loss.total.backward();
            if (config.gradient_clip_norm > 0.0)
            {
                // 梯度裁剪对 hard negative 和图匹配损失的尖峰较重要，避免偶发难例打爆优化器。
                torch::nn::utils::clip_grad_norm_(parameters, config.gradient_clip_norm);
            }
            optimizer.step();

            ++global_step;

            const double batch_seconds = batch_timer.elapsedSeconds();
            accumulated_batch_seconds += batch_seconds;
            ++completed_batches;
            const auto iteration = static_cast<int>((offset / static_cast<std::size_t>(config.batch_size)) + 1);
            const auto should_report = iteration == 1 || iteration == static_cast<int>(epoch_iterations) ||
                                       iteration % TRAINING_METRIC_LOG_INTERVAL == 0;
            if (should_report)
            {
                // 控制台和 CSV 使用同一批标量，保证人工观察和离线曲线一致。
                last_loss = loss.total.detach().item<double>();
                const auto repeatability_loss_value = loss.repeatability.detach().item<double>();
                const auto descriptor_loss_value = loss.descriptor.detach().item<double>();
                const auto orientation_loss_value = loss.orientation.detach().item<double>();
                const auto graph_matching_loss_value = loss.graph_matching.detach().item<double>();
                const auto descriptor_accuracy_value = loss.descriptor_accuracy.detach().item<double>();
                const auto graph_matching_accuracy_value = loss.graph_matching_accuracy.detach().item<double>();
                const auto graph_positive_fraction_value = loss.graph_positive_fraction.detach().item<double>();
                const auto graph_positive_count_value = loss.graph_positive_count.detach().item<double>();
                const auto graph_query_count_value = loss.graph_query_count.detach().item<double>();
                const auto graph_features_a_count_value = loss.graph_features_a_count.detach().item<double>();
                const auto graph_features_b_count_value = loss.graph_features_b_count.detach().item<double>();
                const auto learned_graph_matching_accuracy_value =
                    loss.learned_graph_matching_accuracy.detach().item<double>();
                const auto learned_graph_positive_fraction_value =
                    loss.learned_graph_positive_fraction.detach().item<double>();
                const auto learned_graph_positive_count_value =
                    loss.learned_graph_positive_count.detach().item<double>();
                const auto learned_graph_query_count_value = loss.learned_graph_query_count.detach().item<double>();
                const auto descriptor_positive_score_value = loss.descriptor_positive_score.detach().item<double>();
                const auto descriptor_hard_negative_score_value =
                    loss.descriptor_hard_negative_score.detach().item<double>();
                const auto descriptor_positive_margin_value = loss.descriptor_positive_margin.detach().item<double>();
                const auto descriptor_positive_rank_value = loss.descriptor_positive_rank.detach().item<double>();
                const auto keypoint_descriptor_accuracy_value =
                    loss.keypoint_descriptor_accuracy.detach().item<double>();
                const auto keypoint_descriptor_positive_margin_value =
                    loss.keypoint_descriptor_positive_margin.detach().item<double>();
                const auto keypoint_descriptor_positive_rank_value =
                    loss.keypoint_descriptor_positive_rank.detach().item<double>();
                const auto descriptor_diversity_value = loss.descriptor_diversity.detach().item<double>();
                const auto offset_loss_value = loss.offset.detach().item<double>();
                const auto offset_error_value = loss.offset_error.detach().item<double>();
                const auto confidence_loss_value = loss.confidence.detach().item<double>();
                const auto feature_loss_value =
                    repeatability_loss_value + descriptor_loss_value + orientation_loss_value;
                const auto dense_loss_value =
                    offset_loss_value * OFFSET_LOSS_WEIGHT + confidence_loss_value * CONFIDENCE_LOSS_WEIGHT;
                const auto metric_values = std::unordered_map<std::string, double>{
                    {"loss_total", last_loss},
                    {"feature_loss", feature_loss_value},
                    {"repeatability_loss", repeatability_loss_value},
                    {"descriptor_loss", descriptor_loss_value},
                    {"orientation_loss", orientation_loss_value},
                    {"matcher_loss", graph_matching_loss_value},
                    {"graph_matching_loss", graph_matching_loss_value},
                    {"graph_matching_accuracy", graph_matching_accuracy_value},
                    {"graph_positive_fraction", graph_positive_fraction_value},
                    {"graph_positive_count", graph_positive_count_value},
                    {"graph_query_count", graph_query_count_value},
                    {"graph_features_a", graph_features_a_count_value},
                    {"graph_features_b", graph_features_b_count_value},
                    {"learned_graph_matching_accuracy", learned_graph_matching_accuracy_value},
                    {"learned_graph_positive_fraction", learned_graph_positive_fraction_value},
                    {"learned_graph_positive_count", learned_graph_positive_count_value},
                    {"learned_graph_query_count", learned_graph_query_count_value},
                    {"dense_loss", dense_loss_value},
                    {"offset_loss", offset_loss_value},
                    {"confidence_loss", confidence_loss_value},
                    {"descriptor_accuracy", descriptor_accuracy_value},
                    {"descriptor_positive_score", descriptor_positive_score_value},
                    {"descriptor_hard_negative_score", descriptor_hard_negative_score_value},
                    {"descriptor_positive_margin", descriptor_positive_margin_value},
                    {"descriptor_positive_rank", descriptor_positive_rank_value},
                    {"keypoint_descriptor_accuracy", keypoint_descriptor_accuracy_value},
                    {"keypoint_descriptor_positive_margin", keypoint_descriptor_positive_margin_value},
                    {"keypoint_descriptor_positive_rank", keypoint_descriptor_positive_rank_value},
                    {"descriptor_diversity", descriptor_diversity_value},
                    {"offset_error_px", offset_error_value},
                };
                if (csv_logger)
                {
                    const auto gpu_metrics = gpu_metric_provider ? gpu_metric_provider->sample() : GpuMetrics{};
                    csv_logger->logIteration(make_iteration_metric(
                        config, epoch + 1, iteration, static_cast<int>(epoch_iterations), static_cast<int>(end),
                        static_cast<int>(epoch_size), current_learning_rate, total_timer.elapsedSeconds(), gpu_metrics,
                        metric_values));
                }
                TrainingMetric iter_metric;
                iter_metric.epoch = static_cast<int>(epoch + 1);
                iter_metric.total_epochs = static_cast<int>(config.epochs);
                iter_metric.iteration = iteration;
                iter_metric.total_iterations = static_cast<int>(epoch_iterations);
                iter_metric.images_seen = static_cast<int>(end);
                iter_metric.total_images = static_cast<int>(epoch_size);
                iter_metric.learning_rate = current_learning_rate;
                iter_metric.elapsed_seconds = total_timer.elapsedSeconds();
                iter_metric.values["loss_total"] = last_loss;
                iter_metric.values["matcher_loss"] = graph_matching_loss_value;
                iter_metric.values["graph_matching_accuracy"] = graph_matching_accuracy_value;
                iter_metric.values["graph_positive_fraction"] = graph_positive_fraction_value;
                iter_metric.values["graph_positive_count"] = graph_positive_count_value;
                iter_metric.values["graph_query_count"] = graph_query_count_value;
                iter_metric.values["graph_features_a"] = graph_features_a_count_value;
                iter_metric.values["graph_features_b"] = graph_features_b_count_value;
                iter_metric.values["learned_graph_matching_accuracy"] = learned_graph_matching_accuracy_value;
                iter_metric.values["learned_graph_positive_fraction"] = learned_graph_positive_fraction_value;
                iter_metric.values["learned_graph_positive_count"] = learned_graph_positive_count_value;
                iter_metric.values["learned_graph_query_count"] = learned_graph_query_count_value;
                iter_metric.values["dense_loss"] = dense_loss_value;
                iter_metric.values["offset_error_px"] = offset_error_value;
                iter_metric.values["descriptor_accuracy"] = descriptor_accuracy_value;
                iter_metric.values["descriptor_positive_score"] = descriptor_positive_score_value;
                iter_metric.values["descriptor_hard_negative_score"] = descriptor_hard_negative_score_value;
                iter_metric.values["descriptor_positive_margin"] = descriptor_positive_margin_value;
                iter_metric.values["descriptor_positive_rank"] = descriptor_positive_rank_value;
                iter_metric.values["keypoint_descriptor_accuracy"] = keypoint_descriptor_accuracy_value;
                iter_metric.values["keypoint_descriptor_positive_margin"] = keypoint_descriptor_positive_margin_value;
                iter_metric.values["keypoint_descriptor_positive_rank"] = keypoint_descriptor_positive_rank_value;
                iter_metric.values["descriptor_diversity"] = descriptor_diversity_value;
                iter_metric.values["feature_loss"] = feature_loss_value;
                iter_metric.values["repeatability_loss"] = repeatability_loss_value;
                iter_metric.values["descriptor_loss"] = descriptor_loss_value;
                iter_metric.values["orientation_loss"] = orientation_loss_value;
                progress_logger.logIteration(iter_metric);
                if (!has_loss)
                {
                    first_loss = last_loss;
                    has_loss = true;
                }
            }
        }
        if (csv_logger)
        {
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
        if (val_images > 0)
        {
            // 验证集仍用合成 pair，但关闭梯度并切到 eval，统计平均 total loss。
            torch::NoGradGuard no_grad;
            set_all_modules_eval(modules);
            double val_total = 0.0;
            int64_t val_batches = 0;
            const auto val_epoch_size = val_images * static_cast<std::size_t>(config.pairs_per_image);
            for (std::size_t val_offset = 0; val_offset < val_epoch_size;
                 val_offset += static_cast<std::size_t>(config.batch_size))
            {
                const auto val_end =
                    std::min<std::size_t>(val_epoch_size, val_offset + static_cast<std::size_t>(config.batch_size));
                std::vector<torch::Tensor> val_img_batch;
                std::vector<int64_t> val_src_indices;
                std::vector<int64_t> val_var_indices;
                val_img_batch.reserve(val_end - val_offset);
                val_src_indices.reserve(val_end - val_offset);
                val_var_indices.reserve(val_end - val_offset);
                for (std::size_t idx = val_offset; idx < val_end; ++idx)
                {
                    const auto src = validation_indices[idx % val_images];
                    val_img_batch.push_back(
                        limit_training_image_size(ensure_grayscale(dataset->load(src)), config.resize));
                    val_src_indices.push_back(static_cast<int64_t>(src));
                    val_var_indices.push_back(static_cast<int64_t>(idx / val_images));
                }
                auto val_pairs =
                    make_synthetic_pairs_from_batch(stack_batch(val_img_batch, BatchTensorLayout::Chw).to(device),
                                                    val_src_indices, val_var_indices, pair_config);
                TrainingLossComponents val_loss;
                try
                {
                    val_loss = training_loss_from_pairs(modules, val_pairs, config, descriptor_anchor_modules.get());
                }
                catch (const std::runtime_error& exc)
                {
                    if (!is_no_valid_correspondence_error(exc))
                    {
                        throw;
                    }
                    std::cerr << "validation batch skipped: no valid correspondences sampled epoch=" << (epoch + 1)
                              << " offset=" << val_offset << '\n';
                    continue;
                }
                if (!torch::isfinite(val_loss.total.detach()).item<bool>())
                {
                    continue;
                }
                val_total += val_loss.total.detach().item<double>();
                ++val_batches;
            }
            const auto val_avg = val_batches > 0 ? val_total / static_cast<double>(val_batches) : 0.0;
            if (val_avg < result.best_val_loss)
            {
                result.best_val_loss = val_avg;
            }
            std::cout << "val loss=" << val_avg << " best=" << result.best_val_loss << '\n';
            set_all_modules_train(modules);
            keep_descriptor_only_frozen_modules_eval(modules, config);
            keep_graph_only_frozen_modules_eval(modules, config);
            apply_training_profile_module_mode(modules, config);
        }
        ++result.epochs_completed;
        // 每个 epoch 后写检查点，保存为 CPU 可加载 archive，便于中断后继续训练或推理。
        save_checkpoint(config, modules);
        move_modules_to_device(modules, device);
    }

    result.initial_loss = first_loss;
    result.final_loss = last_loss;
    result.total_time_seconds = total_timer.elapsedSeconds();
    result.avg_batch_time_seconds =
        completed_batches == 0 ? 0.0 : accumulated_batch_seconds / static_cast<double>(completed_batches);
    if (visualization_writer)
    {
        visualization_writer->join();
    }
    if (csv_logger)
    {
        csv_logger->flush();
    }
    return result;
}

bool checkpoint_can_load(const std::string& checkpoint)
{
    try
    {
        // 只做轻量 archive/config 检查，不完整实例化模型，供 CLI 快速判断 checkpoint 是否可用。
        torch::serialize::InputArchive archive;
        archive.load_from(checkpoint);
        torch::serialize::InputArchive config_archive;
        archive.read("config", config_archive);
        torch::Tensor base_channels;
        torch::Tensor descriptor_dim;
        torch::Tensor input_channels;
        torch::Tensor graph_keypoint_meta_dim;
        config_archive.read("base_channels", base_channels);
        config_archive.read("descriptor_dim", descriptor_dim);
        config_archive.read("input_channels", input_channels);
        config_archive.read("graph_keypoint_meta_dim", graph_keypoint_meta_dim);
        torch::serialize::InputArchive dual_fpn_archive;
        torch::serialize::InputArchive texture_adapter_archive;
        torch::serialize::InputArchive descriptor_fusion_archive;
        torch::serialize::InputArchive quality_head_archive;
        torch::serialize::InputArchive semi_dense_branch_archive;
        torch::serialize::InputArchive graph_matcher_archive;
        archive.read("dual_fpn", dual_fpn_archive);
        archive.read("texture_adapter", texture_adapter_archive);
        archive.read("descriptor_fusion", descriptor_fusion_archive);
        archive.read("quality_head", quality_head_archive);
        archive.read("semi_dense_branch", semi_dense_branch_archive);
        archive.read("graph_matcher", graph_matcher_archive);
        return base_channels.defined() && descriptor_dim.defined() && input_channels.defined() &&
               graph_keypoint_meta_dim.defined();
    }
    catch (const c10::Error&)
    {
        return false;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

} // namespace pfm
