#include "infer/pipeline.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/serialize.h>
#include <torch/torch.h>

#include "data/image_io.h"
#include "infer/feature_codec.h"
#include "infer/feature_extractor.h"
#include "infer/match_codec.h"
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

FeatureSet extract_feature_set(
    const std::string& image_path,
    InferenceModules& modules,
    const CheckpointConfig& checkpoint_config,
    int max_keypoints,
    double semi_dense_threshold
) {
    const auto image = load_image_tensor(image_path);
    const auto maps = run_mvp_model(image, modules, checkpoint_config);
    return decode_feature_maps(maps, max_keypoints, semi_dense_threshold);
}

bool inference_checkpoint_can_load(const std::string& checkpoint) {
    try {
        const auto checkpoint_config = load_checkpoint_config(checkpoint);
        (void)load_inference_modules(checkpoint, checkpoint_config);
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

std::pair<torch::Tensor, torch::Tensor> match_sparse_features(
    const FeatureSet& features_a,
    const FeatureSet& features_b
) {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (!features_a.descriptors.defined() || !features_b.descriptors.defined()) {
        throw std::invalid_argument("descriptors must be defined");
    }
    if (features_a.descriptors.dim() != 2 || features_b.descriptors.dim() != 2) {
        throw std::invalid_argument("descriptors must be 2D");
    }
    if (features_a.descriptors.size(1) != features_b.descriptors.size(1)) {
        throw std::invalid_argument("descriptor dimensions must match");
    }
    if (features_a.descriptors.size(0) == 0 || features_b.descriptors.size(0) == 0) {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }

    const auto descriptors_a = torch::nn::functional::normalize(
        features_a.descriptors.to(torch::kCPU, torch::kFloat32),
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12)
    );
    const auto descriptors_b = torch::nn::functional::normalize(
        features_b.descriptors.to(torch::kCPU, torch::kFloat32),
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12)
    );
    const auto similarity = torch::matmul(descriptors_a, descriptors_b.transpose(0, 1));
    const auto best_b = std::get<1>(torch::max(similarity, 1)).to(torch::kCPU, torch::kInt64).contiguous();
    const auto best_score = std::get<0>(torch::max(similarity, 1)).to(torch::kCPU, torch::kFloat32).contiguous();
    const auto best_a = std::get<1>(torch::max(similarity, 0)).to(torch::kCPU, torch::kInt64).contiguous();

    std::vector<int64_t> match_indices;
    std::vector<float> match_scores;
    auto best_b_accessor = best_b.accessor<int64_t, 1>();
    auto best_a_accessor = best_a.accessor<int64_t, 1>();
    auto score_accessor = best_score.accessor<float, 1>();
    for (int64_t index_a = 0; index_a < best_b.size(0); ++index_a) {
        const int64_t index_b = best_b_accessor[index_a];
        if (best_a_accessor[index_b] == index_a) {
            match_indices.push_back(index_a);
            match_indices.push_back(index_b);
            match_scores.push_back(score_accessor[index_a]);
        }
    }

    const int64_t match_count = static_cast<int64_t>(match_scores.size());
    if (match_count == 0) {
        return {torch::empty({0, 2}, long_options), torch::empty({0}, float_options)};
    }
    return {
        torch::from_blob(match_indices.data(), {match_count, 2}, long_options).clone().contiguous(),
        torch::from_blob(match_scores.data(), {match_count}, float_options).clone().contiguous()};
}

MatchSet match_features(const FeatureSet& features_a, const FeatureSet& features_b) {
    if (!features_a.dense_points.defined() || !features_b.dense_points.defined() ||
        !features_a.dense_confidence.defined() || !features_b.dense_confidence.defined()) {
        throw std::invalid_argument("dense features must be defined");
    }
    const auto sparse = match_sparse_features(features_a, features_b);
    const int64_t dense_count = std::min(features_a.dense_points.size(0), features_b.dense_points.size(0));
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (dense_count == 0) {
        return MatchSet{
            sparse.first,
            sparse.second,
            torch::empty({0, 2}, float_options),
            torch::empty({0, 2}, float_options),
            torch::empty({0}, float_options)};
    }

    const auto confidence_a = features_a.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    const auto confidence_b = features_b.dense_confidence.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count);
    return MatchSet{
        sparse.first,
        sparse.second,
        features_a.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
        features_b.dense_points.to(torch::kCPU, torch::kFloat32).narrow(0, 0, dense_count).contiguous(),
        torch::minimum(confidence_a, confidence_b).contiguous()};
}

std::vector<std::pair<std::string, std::string>> load_pairs(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument("failed to open pairs file: " + path);
    }

    std::vector<std::pair<std::string, std::string>> pairs;
    std::string image_a;
    std::string image_b;
    while (input >> image_a >> image_b) {
        pairs.push_back(std::make_pair(image_a, image_b));
    }
    if (pairs.empty()) {
        throw std::invalid_argument("pairs file is empty: " + path);
    }
    return pairs;
}

float tensor_average_or_zero(const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return 0.0F;
    }
    return tensor.to(torch::kCPU, torch::kFloat32).mean().item<float>();
}

void save_eval_report(
    const std::string& path,
    double average_matches,
    double average_sparse_score,
    double average_dense_confidence,
    double semi_dense_coverage
) {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::serialize::OutputArchive archive;
    archive.write("average_matches", torch::tensor({static_cast<float>(average_matches)}, options));
    archive.write("average_sparse_score", torch::tensor({static_cast<float>(average_sparse_score)}, options));
    archive.write("average_dense_confidence", torch::tensor({static_cast<float>(average_dense_confidence)}, options));
    archive.write("semi_dense_coverage", torch::tensor({static_cast<float>(semi_dense_coverage)}, options));
    archive.save_to(path);
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
        const auto feature_set = extract_feature_set(
            options.image,
            modules,
            checkpoint_config,
            options.max_keypoints,
            options.semi_dense_threshold
        );
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

    try {
        if (!checkpoint_can_load(options.checkpoint)) {
            std::cerr << "match failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        if (options.device != "cpu") {
            throw std::invalid_argument("only cpu device is supported");
        }
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config);
        const auto features_a = extract_feature_set(
            options.image_a,
            modules,
            checkpoint_config,
            options.max_keypoints,
            options.semi_dense_threshold
        );
        const auto features_b = extract_feature_set(
            options.image_b,
            modules,
            checkpoint_config,
            options.max_keypoints,
            options.semi_dense_threshold
        );
        save_match_set(match_features(features_a, features_b), options.output);
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
        if (options.device != "cpu") {
            throw std::invalid_argument("only cpu device is supported");
        }
        const auto pairs = load_pairs(options.pairs);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config);

        double total_matches = 0.0;
        double total_sparse_score = 0.0;
        double total_dense_confidence = 0.0;
        double total_coverage = 0.0;
        for (const auto& pair : pairs) {
            const auto features_a = extract_feature_set(
                pair.first,
                modules,
                checkpoint_config,
                options.max_keypoints,
                options.semi_dense_threshold
            );
            const auto features_b = extract_feature_set(
                pair.second,
                modules,
                checkpoint_config,
                options.max_keypoints,
                options.semi_dense_threshold
            );
            const auto matches = match_features(features_a, features_b);
            total_matches += static_cast<double>(matches.sparse_matches.size(0));
            total_sparse_score += static_cast<double>(tensor_average_or_zero(matches.sparse_scores));
            total_dense_confidence += static_cast<double>(tensor_average_or_zero(matches.confidence));
            const int64_t dense_base = std::max<int64_t>(features_a.dense_points.size(0), 1);
            total_coverage += static_cast<double>(matches.points_a.size(0)) / static_cast<double>(dense_base);
        }

        const double pair_count = static_cast<double>(pairs.size());
        save_eval_report(
            options.output,
            total_matches / pair_count,
            total_sparse_score / pair_count,
            total_dense_confidence / pair_count,
            total_coverage / pair_count
        );
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
