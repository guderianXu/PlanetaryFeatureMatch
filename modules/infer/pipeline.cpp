#include "infer/pipeline.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/serialize.h>
#include <torch/torch.h>

#include "data/image_io.h"
#include "infer/feature_codec.h"
#include "infer/feature_extractor.h"
#include "models/backbone.h"
#include "models/dense_head.h"
#include "models/sparse_head.h"
#include "train/trainer.h"

namespace pfm {
namespace {

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

InferenceModules load_inference_modules(const std::string& checkpoint, const CheckpointConfig& config) {
    InferenceModules modules;
    modules.backbone = Backbone(config.input_channels, config.base_channels);
    modules.sparse_head = SparseHead(config.base_channels, config.descriptor_dim);
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

    modules.backbone->eval();
    modules.sparse_head->eval();
    modules.dense_head->eval();
    return modules;
}

RawFeatureMaps run_mvp_model(const torch::Tensor& image, InferenceModules& modules, const CheckpointConfig& config) {
    torch::NoGradGuard no_grad;
    const auto input = adapt_image_channels(image, config.input_channels).unsqueeze(0).contiguous();
    const auto feature = modules.backbone->forward(input).front();
    const auto sparse = modules.sparse_head->forward(feature);
    const auto dense = modules.dense_head->forward(feature, feature);
    return RawFeatureMaps{
        sparse.heatmap.detach().cpu().contiguous(),
        sparse.descriptors.detach().cpu().contiguous(),
        sparse.scale.detach().cpu().contiguous(),
        sparse.orientation.detach().cpu().contiguous(),
        sparse.affine.detach().cpu().contiguous(),
        dense.confidence.detach().cpu().contiguous()};
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
        if (options.device != "cpu") {
            throw std::invalid_argument("only cpu device is supported");
        }
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config);
        const auto image = load_image_tensor(options.image);
        const auto maps = run_mvp_model(image, modules, checkpoint_config);
        const auto feature_set = decode_feature_maps(maps, options.max_keypoints, options.semi_dense_threshold);
        save_feature_set(feature_set, options.output);
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

    std::cerr << "match command is implemented in Task 8\n";
    return 1;
}

int run_eval_command(const CliOptions& options) {
    if (!require_path(options.pairs, "--pairs") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output")) {
        return 1;
    }

    std::cerr << "eval command is implemented in Task 8\n";
    return 1;
}

int run_export_command(const CliOptions& options) {
    if (!require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output")) {
        return 1;
    }

    try {
        if (!checkpoint_can_load(options.checkpoint)) {
            std::cerr << "export failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        copy_file_contents(options.checkpoint, options.output);
        if (!checkpoint_can_load(options.output)) {
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
