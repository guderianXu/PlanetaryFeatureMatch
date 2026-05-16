#include "train/trainer.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/nn/functional/upsampling.h>
#include <torch/torch.h>

#include "core/device.h"
#include "core/tensor_utils.h"
#include "data/image_dataset.h"
#include "data/intensity_mask.h"
#include "data/synthetic_pair.h"
#include "data/synthetic_pair_cache.h"
#include "losses/losses.h"
#include "models/backbone.h"
#include "models/dense_head.h"
#include "models/sparse_head.h"

namespace pfm {
namespace {

constexpr int64_t INPUT_CHANNELS = 1;
constexpr int64_t MAX_DESCRIPTOR_LOSS_SAMPLES = 1024;
constexpr int64_t DESCRIPTOR_NEGATIVE_SAMPLE_COUNT = 63;
constexpr float OFFSET_LOSS_WEIGHT = 0.2F;
constexpr int64_t SPARSE_FEATURE_CHANNEL_MULTIPLIER = 2;

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
    validate_min_keypoint_intensity(config.min_keypoint_intensity);
}

torch::Tensor ensure_grayscale(const torch::Tensor& image) {
    require_chw_image(image);
    if (channels(image) == INPUT_CHANNELS) {
        return image;
    }
    return image.mean(0, true).contiguous();
}

torch::Tensor stack_batch(const std::vector<torch::Tensor>& images) {
    return torch::stack(images).contiguous();
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

torch::Tensor make_sparse_descriptor_loss(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask
) {
    return make_sparse_descriptor_metrics(descriptors_a, descriptors_b, warp, valid_mask).loss;
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
};

TrainModules make_modules(const TrainConfig& config, torch::Device device) {
    TrainModules modules;
    modules.backbone = Backbone(INPUT_CHANNELS, config.base_channels);
    modules.sparse_head = SparseHead(config.base_channels * SPARSE_FEATURE_CHANNEL_MULTIPLIER, config.descriptor_dim);
    modules.dense_head = DenseHead(config.base_channels);
    modules.backbone->to(device);
    modules.sparse_head->to(device);
    modules.dense_head->to(device);
    modules.backbone->train();
    modules.sparse_head->train();
    modules.dense_head->train();
    return modules;
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

struct TrainingLossComponents {
    torch::Tensor total;
    torch::Tensor repeatability;
    torch::Tensor descriptor;
    torch::Tensor offset;
    torch::Tensor confidence;
    torch::Tensor descriptor_accuracy;
    torch::Tensor descriptor_diversity;
    torch::Tensor offset_error;
};

torch::Tensor weighted_total_training_loss(
    const torch::Tensor& repeatability,
    const torch::Tensor& descriptor,
    const torch::Tensor& offset,
    const torch::Tensor& confidence,
    const torch::Tensor& descriptor_diversity
) {
    (void)descriptor_diversity;
    return repeatability + descriptor + offset * OFFSET_LOSS_WEIGHT + confidence;
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

    const auto view_a = stack_batch(views_a);
    const auto view_b = stack_batch(views_b);
    const auto warp = stack_batch(warps);
    const auto valid_mask = make_training_valid_mask(
        view_a,
        view_b,
        warp,
        stack_batch(valid_masks),
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
    auto offset = masked_l1_loss(dense.offsets, target_offsets, dense_mask);
    auto confidence = confidence_bce_loss(dense.confidence, dense_mask);
    auto offset_error = offset_pixel_error(dense.offsets, target_offsets, dense_mask);
    return TrainingLossComponents{weighted_total_training_loss(
                                      repeatability,
                                      descriptor.loss,
                                      offset,
                                      confidence,
                                      descriptor.diversity),
                                  repeatability,
                                  descriptor.loss,
                                  offset,
                                  confidence,
                                  descriptor.accuracy,
                                  descriptor.diversity,
                                  offset_error};
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
}

void save_checkpoint(const TrainConfig& config, TrainModules& modules) {
    move_modules_to_device(modules, torch::Device(torch::kCPU));
    torch::serialize::OutputArchive archive;
    torch::serialize::OutputArchive config_archive;
    config_archive.write("base_channels", torch::tensor({config.base_channels}, torch::kInt64));
    config_archive.write("descriptor_dim", torch::tensor({config.descriptor_dim}, torch::kInt64));
    config_archive.write("input_channels", torch::tensor({INPUT_CHANNELS}, torch::kInt64));
    archive.write("config", config_archive);

    torch::serialize::OutputArchive backbone_archive;
    torch::serialize::OutputArchive sparse_head_archive;
    torch::serialize::OutputArchive dense_head_archive;
    modules.backbone->save(backbone_archive);
    modules.sparse_head->save(sparse_head_archive);
    modules.dense_head->save(dense_head_archive);
    archive.write("backbone", backbone_archive);
    archive.write("sparse_head", sparse_head_archive);
    archive.write("dense_head", dense_head_archive);
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

torch::Tensor make_descriptor_sample_indices_for_test(const torch::Tensor& descriptors) {
    return make_descriptor_sample_indices(descriptors);
}

torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count) {
    return make_descriptor_candidate_indices(target_indices, spatial_count);
}

torch::Tensor limit_training_image_size_for_test(const torch::Tensor& image, int64_t max_edge) {
    return limit_training_image_size(image, max_edge);
}

torch::Tensor weighted_total_training_loss_for_test(
    const torch::Tensor& repeatability,
    const torch::Tensor& descriptor,
    const torch::Tensor& offset,
    const torch::Tensor& confidence,
    const torch::Tensor& descriptor_diversity
) {
    return weighted_total_training_loss(repeatability, descriptor, offset, confidence, descriptor_diversity);
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

}  // namespace testing

TrainResult train_model(const TrainConfig& config) {
    validate_config(config);
    const auto device = resolve_compute_device(config.device);
    ImageDataset dataset(config.image_dir);
    auto modules = make_modules(config, device);
    auto optimizer = torch::optim::Adam(module_parameters(modules), torch::optim::AdamOptions(config.learning_rate));

    TrainResult result;
    double first_loss = 0.0;
    double last_loss = 0.0;
    bool has_loss = false;

    const auto epoch_size = dataset.size() * static_cast<std::size_t>(config.pairs_per_image);
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

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        for (std::size_t offset = 0; offset < epoch_size; offset += static_cast<std::size_t>(config.batch_size)) {
            const auto batch_end = offset + static_cast<std::size_t>(config.batch_size);
            const auto end = std::min<std::size_t>(epoch_size, batch_end);
            std::vector<SyntheticPair> pairs;
            if (cache_dataset) {
                pairs = load_cached_pairs(*cache_dataset, offset, end, device);
            } else {
                std::vector<torch::Tensor> images;
                std::vector<int64_t> source_indices;
                std::vector<int64_t> variant_indices;
                images.reserve(end - offset);
                source_indices.reserve(end - offset);
                variant_indices.reserve(end - offset);
                for (std::size_t index = offset; index < end; ++index) {
                    const auto source_index = index % dataset.size();
                    images.push_back(limit_training_image_size(
                        ensure_grayscale(dataset.load(source_index)),
                        config.resize));
                    source_indices.push_back(static_cast<int64_t>(source_index));
                    variant_indices.push_back(static_cast<int64_t>(index / dataset.size()));
                }
                pairs = make_synthetic_pairs_from_batch(stack_batch(images).to(device), source_indices, variant_indices, pair_config);
            }

            auto loss = training_loss_from_pairs(modules, pairs, config.min_keypoint_intensity);
            optimizer.zero_grad();
            loss.total.backward();
            optimizer.step();

            last_loss = loss.total.detach().item<double>();
            const auto repeatability_loss_value = loss.repeatability.detach().item<double>();
            const auto descriptor_loss_value = loss.descriptor.detach().item<double>();
            const auto descriptor_accuracy_value = loss.descriptor_accuracy.detach().item<double>();
            const auto descriptor_diversity_value = loss.descriptor_diversity.detach().item<double>();
            const auto offset_loss_value = loss.offset.detach().item<double>();
            const auto offset_error_value = loss.offset_error.detach().item<double>();
            const auto confidence_loss_value = loss.confidence.detach().item<double>();
            std::cout << "train progress: epoch=" << epoch + 1 << '/' << config.epochs
                      << " batch=" << (offset / static_cast<std::size_t>(config.batch_size)) + 1 << '/'
                      << (epoch_size + static_cast<std::size_t>(config.batch_size) - 1) /
                             static_cast<std::size_t>(config.batch_size)
                      << " images=" << end << '/' << epoch_size << " loss=" << last_loss
                      << " repeatability=" << repeatability_loss_value
                      << " descriptor=" << descriptor_loss_value
                      << " descriptor_accuracy=" << descriptor_accuracy_value
                      << " descriptor_diversity=" << descriptor_diversity_value
                      << " offset=" << offset_loss_value
                      << " offset_error=" << offset_error_value
                      << " confidence=" << confidence_loss_value << '\n';
            if (!has_loss) {
                first_loss = last_loss;
                has_loss = true;
            }
        }
        ++result.epochs_completed;
    }

    result.initial_loss = first_loss;
    result.final_loss = last_loss;
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
        return base_channels.defined() && descriptor_dim.defined() && input_channels.defined();
    } catch (const c10::Error&) {
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace pfm
