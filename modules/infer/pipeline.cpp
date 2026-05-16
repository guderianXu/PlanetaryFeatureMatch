#include "infer/pipeline.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/nn/functional/upsampling.h>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "core/device.h"
#include "data/image_io.h"
#include "data/intensity_mask.h"
#include "infer/eval_pipeline.h"
#include "infer/feature_codec.h"
#include "infer/feature_extractor.h"
#include "infer/match_codec.h"
#include "infer/matching_pipeline.h"
#include "infer/visualization.h"
#include "models/backbone.h"
#include "models/dense_head.h"
#include "models/sparse_head.h"
#include "train/trainer.h"

namespace pfm {
namespace {

constexpr int64_t SPARSE_FEATURE_CHANNEL_MULTIPLIER = 2;

bool require_path(const std::string& value, const char* option_name) {
    if (!value.empty()) {
        return true;
    }

    std::cerr << "missing required option " << option_name << '\n';
    return false;
}

int64_t read_config_value(torch::serialize::InputArchive& config_archive, const char* name) {
    torch::Tensor tensor;
    config_archive.read(name, tensor);
    if (!tensor.defined() || tensor.numel() != 1) {
        throw std::invalid_argument(std::string("checkpoint config missing ") + name);
    }
    return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

struct CheckpointConfig {
    int64_t input_channels = 1;
    int64_t base_channels = 8;
    int64_t descriptor_dim = 32;
};

CheckpointConfig load_checkpoint_config(const std::string& checkpoint) {
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive config_archive;
    archive.read("config", config_archive);

    CheckpointConfig config;
    config.input_channels = read_config_value(config_archive, "input_channels");
    config.base_channels = read_config_value(config_archive, "base_channels");
    config.descriptor_dim = read_config_value(config_archive, "descriptor_dim");
    return config;
}

torch::Tensor adapt_image_channels(const torch::Tensor& image, int64_t input_channels) {
    if (image.size(0) == input_channels) {
        return image;
    }
    if (input_channels == 1) {
        return image.mean(0, true).contiguous();
    }
    throw std::invalid_argument("image channel count does not match checkpoint input_channels");
}

struct InferenceModules {
    Backbone backbone{nullptr};
    SparseHead sparse_head{nullptr};
    DenseHead dense_head{nullptr};
};

InferenceModules load_inference_modules(
    const std::string& checkpoint,
    const CheckpointConfig& config,
    torch::Device device
) {
    InferenceModules modules;
    modules.backbone = Backbone(config.input_channels, config.base_channels);
    modules.sparse_head = SparseHead(config.base_channels * SPARSE_FEATURE_CHANNEL_MULTIPLIER, config.descriptor_dim);
    modules.dense_head = DenseHead(config.base_channels);

    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive backbone_archive;
    torch::serialize::InputArchive sparse_head_archive;
    torch::serialize::InputArchive dense_head_archive;
    archive.read("backbone", backbone_archive);
    archive.read("sparse_head", sparse_head_archive);
    archive.read("dense_head", dense_head_archive);
    modules.backbone->load(backbone_archive);
    modules.sparse_head->load(sparse_head_archive);
    modules.dense_head->load(dense_head_archive);

    modules.backbone->to(device);
    modules.sparse_head->to(device);
    modules.dense_head->to(device);
    modules.backbone->eval();
    modules.sparse_head->eval();
    modules.dense_head->eval();
    return modules;
}

RawFeatureMaps run_mvp_model(
    const torch::Tensor& image,
    InferenceModules& modules,
    const CheckpointConfig& config,
    torch::Device device
) {
    torch::NoGradGuard no_grad;
    const auto input = adapt_image_channels(image, config.input_channels).unsqueeze(0).contiguous().to(device);
    const auto feature_pyramid = modules.backbone->forward(input);
    const auto sparse = modules.sparse_head->forward(feature_pyramid[1]);
    const auto dense = modules.dense_head->forward(feature_pyramid.front(), feature_pyramid.front());
    const auto dense_confidence = torch::nn::functional::interpolate(
        dense.confidence,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{sparse.heatmap.size(2), sparse.heatmap.size(3)})
            .mode(torch::kNearest));
    return RawFeatureMaps{
        sparse.heatmap.detach().cpu().contiguous(),
        sparse.descriptors.detach().cpu().contiguous(),
        sparse.scale.detach().cpu().contiguous(),
        sparse.orientation.detach().cpu().contiguous(),
        sparse.affine.detach().cpu().contiguous(),
        dense_confidence.detach().cpu().contiguous()};
}

struct ExtractedFeatureSet {
    FeatureSet features;
    int64_t feature_map_width = 0;
    int64_t feature_map_height = 0;
};

ExtractedFeatureSet extract_feature_set(
    const std::string& image_path,
    InferenceModules& modules,
    const CheckpointConfig& checkpoint_config,
    torch::Device device,
    int max_keypoints,
    double semi_dense_threshold,
    double min_keypoint_intensity
) {
    const auto image = load_image_tensor(image_path);
    const auto maps = run_mvp_model(image, modules, checkpoint_config, device);
    const auto intensity_mask = make_intensity_mask(image, min_keypoint_intensity).to(torch::kCPU);
    return ExtractedFeatureSet{
        decode_feature_maps(maps, max_keypoints, semi_dense_threshold, intensity_mask),
        maps.heatmap.size(3),
        maps.heatmap.size(2)};
}

bool inference_checkpoint_can_load(const std::string& checkpoint) {
    try {
        const auto checkpoint_config = load_checkpoint_config(checkpoint);
        (void)load_inference_modules(checkpoint, checkpoint_config, torch::Device(torch::kCPU));
        return true;
    } catch (const c10::Error&) {
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

void copy_file_contents(const std::string& source, const std::string& destination) {
    std::ifstream input(source, std::ios::binary);
    if (!input) {
        throw std::invalid_argument("failed to open checkpoint for export: " + source);
    }
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::invalid_argument("failed to open export output: " + destination);
    }
    output << input.rdbuf();
    if (!output) {
        throw std::invalid_argument("failed to write export output: " + destination);
    }
}

}  // namespace

int run_train_command(const CliOptions& options) {
    if (!require_path(options.image_dir, "--image-dir") || !require_path(options.checkpoint, "--checkpoint")) {
        return 1;
    }

    try {
        TrainConfig config;
        config.image_dir = options.image_dir;
        config.checkpoint = options.checkpoint;
        config.device = options.device;
        config.epochs = options.epochs;
        config.batch_size = options.batch_size;
        config.resize = options.resize;
        config.pairs_per_image = options.pairs_per_image;
        config.augmentation_profile = options.augmentation_profile;
        config.extreme_pair_ratio = options.extreme_pair_ratio;
        config.synthetic_pair_cache_dir = options.synthetic_pair_cache_dir;
        config.synthetic_pair_cache_rebuild = options.synthetic_pair_cache_rebuild;
        config.min_keypoint_intensity = options.min_keypoint_intensity;
        const auto result = train_model(config);
        std::cout << "training complete: epochs=" << result.epochs_completed << " final_loss=" << result.final_loss << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "train failed: " << error.what() << '\n';
        return 1;
    }
}

int run_extract_command(const CliOptions& options) {
    if (!require_path(options.image, "--image") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output")) {
        return 1;
    }

    try {
        if (!checkpoint_can_load(options.checkpoint)) {
            std::cerr << "extract failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config, device);
        const auto extracted = extract_feature_set(
            options.image,
            modules,
            checkpoint_config,
            device,
            options.max_keypoints,
            options.semi_dense_threshold,
            options.min_keypoint_intensity
        );
        save_feature_set(extracted.features, options.output);
        if (!options.visualization_dir.empty()) {
            (void)save_feature_visualization(
                options.image,
                extracted.features,
                options.visualization_dir,
                extracted.feature_map_width,
                extracted.feature_map_height);
        }
        std::cout << "extraction complete: features=" << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "extract failed: " << error.what() << '\n';
        return 1;
    }
}

int run_match_command(const CliOptions& options) {
    if (!require_path(options.image_a, "--image-a") || !require_path(options.image_b, "--image-b") ||
        !require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output")) {
        return 1;
    }

    try {
        if (!checkpoint_can_load(options.checkpoint)) {
            std::cerr << "match failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config, device);
        const auto extracted_a = extract_feature_set(
            options.image_a,
            modules,
            checkpoint_config,
            device,
            options.max_keypoints,
            options.semi_dense_threshold,
            options.min_keypoint_intensity
        );
        const auto extracted_b = extract_feature_set(
            options.image_b,
            modules,
            checkpoint_config,
            device,
            options.max_keypoints,
            options.semi_dense_threshold,
            options.min_keypoint_intensity
        );
        const auto match_set = matchFeatureSets(extracted_a.features, extracted_b.features);
        save_match_set(match_set, options.output);
        if (!options.visualization_dir.empty()) {
            (void)save_match_visualization(
                options.image_a,
                options.image_b,
                match_set,
                options.visualization_dir,
                extracted_a.feature_map_width,
                extracted_a.feature_map_height,
                extracted_b.feature_map_width,
                extracted_b.feature_map_height);
        }
        std::cout << "matching complete: matches=" << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "match failed: " << error.what() << '\n';
        return 1;
    }
}

int run_eval_command(const CliOptions& options) {
    if (!require_path(options.pairs, "--pairs") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output")) {
        return 1;
    }

    try {
        if (!checkpoint_can_load(options.checkpoint)) {
            std::cerr << "eval failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto pairs = loadEvalPairs(options.pairs);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config, device);

        std::vector<std::pair<FeatureSet, FeatureSet>> feature_sets;
        std::vector<MatchSet> match_sets;
        feature_sets.reserve(pairs.size());
        match_sets.reserve(pairs.size());
        for (const auto& pair : pairs) {
            auto extracted_a = extract_feature_set(
                pair.first,
                modules,
                checkpoint_config,
                device,
                options.max_keypoints,
                options.semi_dense_threshold,
                options.min_keypoint_intensity
            );
            auto extracted_b = extract_feature_set(
                pair.second,
                modules,
                checkpoint_config,
                device,
                options.max_keypoints,
                options.semi_dense_threshold,
                options.min_keypoint_intensity
            );
            match_sets.push_back(matchFeatureSets(extracted_a.features, extracted_b.features));
            feature_sets.push_back(std::make_pair(std::move(extracted_a.features), std::move(extracted_b.features)));
        }

        saveEvalReport(options.output, aggregateEvalReport(feature_sets, match_sets));
        std::cout << "evaluation complete: report=" << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "eval failed: " << error.what() << '\n';
        return 1;
    }
}

int run_export_command(const CliOptions& options) {
    if (!require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output")) {
        return 1;
    }

    try {
        if (!inference_checkpoint_can_load(options.checkpoint)) {
            std::cerr << "export failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        copy_file_contents(options.checkpoint, options.output);
        if (!inference_checkpoint_can_load(options.output)) {
            std::cerr << "export failed: exported checkpoint cannot load: " << options.output << '\n';
            return 1;
        }
        std::cout << "export complete: checkpoint=" << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "export failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace pfm
