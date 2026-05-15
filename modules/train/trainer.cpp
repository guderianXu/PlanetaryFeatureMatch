#include "train/trainer.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/nn/functional/upsampling.h>
#include <torch/torch.h>

#include "core/tensor_utils.h"
#include "data/image_dataset.h"
#include "data/synthetic_pair.h"
#include "losses/losses.h"
#include "models/backbone.h"
#include "models/dense_head.h"
#include "models/sparse_head.h"

namespace pfm {
namespace {

constexpr int64_t INPUT_CHANNELS = 1;
constexpr int64_t MAX_DESCRIPTOR_LOSS_SAMPLES = 1024;
constexpr int64_t MAX_TRAINING_IMAGE_EDGE = 64;
constexpr std::size_t MAX_TRAINING_IMAGES_PER_EPOCH = 2;

void validate_config(const TrainConfig& config) {
    if (config.image_dir.empty()) {
        throw std::invalid_argument("image_dir must not be empty");
    }
    if (config.checkpoint.empty()) {
        throw std::invalid_argument("checkpoint must not be empty");
    }
    if (config.device != "cpu") {
        throw std::invalid_argument("only cpu device is supported");
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
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive and finite");
    }
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

torch::Tensor limit_training_image_size(const torch::Tensor& image) {
    const auto height = image.size(1);
    const auto width = image.size(2);
    const auto max_edge = std::max(height, width);
    if (max_edge <= MAX_TRAINING_IMAGE_EDGE) {
        return image.contiguous();
    }

    const double scale = static_cast<double>(MAX_TRAINING_IMAGE_EDGE) / static_cast<double>(max_edge);
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
    return torch::linspace(0, spatial_count - 1, sample_count, sample_options);
}

torch::Tensor sample_spatial_descriptors(const torch::Tensor& descriptors, const torch::Tensor& sample_indices) {
    const auto batch_size = descriptors.size(0);
    const auto descriptor_dim = descriptors.size(1);
    const auto spatial_count = descriptors.size(2) * descriptors.size(3);
    auto flat = descriptors.permute({0, 2, 3, 1}).reshape({batch_size, spatial_count, descriptor_dim});
    return flat.index_select(1, sample_indices).contiguous();
}

torch::Tensor make_sparse_descriptor_loss(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b) {
    auto sample_indices = make_descriptor_sample_indices(descriptors_a);
    auto sampled_a = sample_spatial_descriptors(descriptors_a, sample_indices);
    auto sampled_b = sample_spatial_descriptors(descriptors_b, sample_indices);
    auto target = torch::arange(sample_indices.size(0), sample_indices.options())
                      .unsqueeze(0)
                      .expand({descriptors_a.size(0), sample_indices.size(0)});
    return descriptor_cross_entropy_loss(sampled_a, sampled_b, target);
}

torch::Tensor resize_mask_for_heatmap(const torch::Tensor& valid_mask, const torch::Tensor& heatmap) {
    auto mask = valid_mask.to(heatmap.dtype()).unsqueeze(1);
    return torch::nn::functional::interpolate(
        mask,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{heatmap.size(2), heatmap.size(3)})
            .mode(torch::kNearest));
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
    auto x_scale = static_cast<double>(offsets.size(3)) / static_cast<double>(displacement.size(3));
    auto y_scale = static_cast<double>(offsets.size(2)) / static_cast<double>(displacement.size(2));
    resized.index_put_({Slice(), Slice(0, 1), Slice(), Slice()},
                       resized.index({Slice(), Slice(0, 1), Slice(), Slice()}) * x_scale);
    resized.index_put_({Slice(), Slice(1, 2), Slice(), Slice()},
                       resized.index({Slice(), Slice(1, 2), Slice(), Slice()}) * y_scale);
    return resized;
}

struct TrainModules {
    Backbone backbone{nullptr};
    SparseHead sparse_head{nullptr};
    DenseHead dense_head{nullptr};
};

TrainModules make_modules(const TrainConfig& config) {
    TrainModules modules;
    modules.backbone = Backbone(INPUT_CHANNELS, config.base_channels);
    modules.sparse_head = SparseHead(config.base_channels, config.descriptor_dim);
    modules.dense_head = DenseHead(config.base_channels);
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

torch::Tensor training_loss(TrainModules& modules, const torch::Tensor& batch) {
    SyntheticPairConfig pair_config;
    pair_config.translation_x = 1.0F;
    pair_config.translation_y = 0.0F;
    pair_config.brightness_delta = 0.02F;
    pair_config.contrast_scale = 0.98F;

    std::vector<torch::Tensor> views_a;
    std::vector<torch::Tensor> views_b;
    std::vector<torch::Tensor> warps;
    std::vector<torch::Tensor> valid_masks;
    views_a.reserve(static_cast<size_t>(batch.size(0)));
    views_b.reserve(static_cast<size_t>(batch.size(0)));
    warps.reserve(static_cast<size_t>(batch.size(0)));
    valid_masks.reserve(static_cast<size_t>(batch.size(0)));

    for (int64_t index = 0; index < batch.size(0); ++index) {
        auto pair = make_synthetic_pair(batch[index], pair_config);
        views_a.push_back(pair.view_a);
        views_b.push_back(pair.view_b);
        warps.push_back(pair.warp_a_to_b);
        valid_masks.push_back(pair.valid_mask);
    }

    const auto view_a = stack_batch(views_a);
    const auto view_b = stack_batch(views_b);
    const auto warp = stack_batch(warps);
    const auto valid_mask = stack_batch(valid_masks);

    const auto features_a = modules.backbone->forward(view_a).front();
    const auto features_b = modules.backbone->forward(view_b).front();
    const auto sparse_a = modules.sparse_head->forward(features_a);
    const auto sparse_b = modules.sparse_head->forward(features_b);
    const auto dense = modules.dense_head->forward(features_a, features_b);
    const auto heatmap_mask = resize_mask_for_heatmap(valid_mask, sparse_a.heatmap);
    const auto target_offsets = resize_offsets_for_dense_head(warp, dense.offsets);

    return repeatability_loss(sparse_a.heatmap, sparse_b.heatmap, heatmap_mask) +
           make_sparse_descriptor_loss(sparse_a.descriptors, sparse_b.descriptors) +
           masked_l1_loss(dense.offsets, target_offsets, heatmap_mask) +
           confidence_bce_loss(dense.confidence, heatmap_mask);
}

void save_checkpoint(const TrainConfig& config, TrainModules& modules) {
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
    const torch::Tensor& descriptors_b) {
    return make_sparse_descriptor_loss(descriptors_a, descriptors_b);
}

torch::Tensor make_descriptor_sample_indices_for_test(const torch::Tensor& descriptors) {
    return make_descriptor_sample_indices(descriptors);
}

torch::Tensor limit_training_image_size_for_test(const torch::Tensor& image) {
    return limit_training_image_size(image);
}

}  // namespace testing

TrainResult train_model(const TrainConfig& config) {
    validate_config(config);
    ImageDataset dataset(config.image_dir);
    auto modules = make_modules(config);
    auto optimizer = torch::optim::Adam(module_parameters(modules), torch::optim::AdamOptions(config.learning_rate));

    TrainResult result;
    double first_loss = 0.0;
    double last_loss = 0.0;
    bool has_loss = false;

    const auto epoch_size = std::min<std::size_t>(dataset.size(), MAX_TRAINING_IMAGES_PER_EPOCH);
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        for (std::size_t offset = 0; offset < epoch_size; offset += static_cast<std::size_t>(config.batch_size)) {
            const auto batch_end = offset + static_cast<std::size_t>(config.batch_size);
            const auto end = std::min<std::size_t>(epoch_size, batch_end);
            std::vector<torch::Tensor> images;
            images.reserve(end - offset);
            for (std::size_t index = offset; index < end; ++index) {
                images.push_back(limit_training_image_size(ensure_grayscale(dataset.load(index))));
            }

            auto batch = stack_batch(images);
            auto loss = training_loss(modules, batch);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            last_loss = loss.detach().item<double>();
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
