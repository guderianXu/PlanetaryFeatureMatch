#include "infer/pipeline.h"

#include <algorithm>
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
#include "core/timer.h"
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
#include "models/planetary_graph_matcher.h"
#include "models/sparse_head.h"
#include "train/trainer.h"

namespace pfm {
namespace {

constexpr int64_t SPARSE_FEATURE_CHANNEL_MULTIPLIER = 2;
constexpr double ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT = 4.0;

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

int64_t read_optional_config_value(
    torch::serialize::InputArchive& config_archive,
    const char* name,
    int64_t fallback
) {
    try {
        return read_config_value(config_archive, name);
    } catch (const c10::Error&) {
        return fallback;
    }
}

struct CheckpointConfig {
    int64_t input_channels = 1;
    int64_t base_channels = 8;
    int64_t descriptor_dim = 32;
    int64_t graph_hidden_dim = 32;
    int64_t graph_attention_layers = 1;
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
    config.graph_hidden_dim = read_optional_config_value(
        config_archive,
        "graph_hidden_dim",
        std::max<int64_t>(32, config.descriptor_dim));
    config.graph_attention_layers = read_optional_config_value(config_archive, "graph_attention_layers", 1);
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

torch::Tensor make_rotation_invariant_texture_descriptor(
    const torch::Tensor& image,
    int64_t descriptor_height,
    int64_t descriptor_width,
    int64_t descriptor_dim
) {
    auto base = image;
    if (base.size(1) != 1) {
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
    for (const int64_t kernel : {3, 7, 15, 31}) {
        auto blur = torch::avg_pool2d(
            base,
            {kernel, kernel},
            {1, 1},
            {kernel / 2, kernel / 2},
            false,
            true);
        channels.push_back(blur);
        channels.push_back((base - blur).abs());
    }
    auto dx = (base - torch::roll(base, {1}, {3})).abs();
    auto dy = (base - torch::roll(base, {1}, {2})).abs();
    auto gradient = dx + dy;
    for (const int64_t kernel : {3, 7, 11}) {
        channels.push_back(torch::avg_pool2d(
            gradient,
            {kernel, kernel},
            {1, 1},
            {kernel / 2, kernel / 2},
            false,
            true));
    }
    for (const int64_t ring_radius : {1, 2, 4, 8}) {
        std::vector<torch::Tensor> diffs;
        diffs.reserve(8);
        for (const auto& offset : std::vector<std::pair<int64_t, int64_t>>{
                 {-ring_radius, 0},
                 {ring_radius, 0},
                 {0, -ring_radius},
                 {0, ring_radius},
                 {-ring_radius, -ring_radius},
                 {-ring_radius, ring_radius},
                 {ring_radius, -ring_radius},
                 {ring_radius, ring_radius}}) {
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
    target = torch::nn::functional::interpolate(
        target,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{descriptor_height, descriptor_width})
            .mode(torch::kBilinear)
            .align_corners(false));
    auto centered = target - target.mean({2, 3}, true);
    auto scaled = centered / centered.pow(2).mean({2, 3}, true).add(1.0e-4).sqrt();
    if (scaled.size(1) < descriptor_dim) {
        const auto repeat_count = (descriptor_dim + scaled.size(1) - 1) / scaled.size(1);
        scaled = scaled.repeat({1, repeat_count, 1, 1});
    }
    target = scaled.narrow(1, 0, descriptor_dim).contiguous();
    return target / target.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
}

torch::Tensor blend_rotation_invariant_texture_descriptor(
    const torch::Tensor& descriptors,
    const torch::Tensor& image
) {
    auto target = make_rotation_invariant_texture_descriptor(
        image,
        descriptors.size(2),
        descriptors.size(3),
        descriptors.size(1));
    auto blended = descriptors + target * ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT;
    return blended / blended.pow(2).sum(1, true).clamp_min(1.0e-12).sqrt();
}

torch::Tensor make_rotation_invariant_texture_saliency(
    const torch::Tensor& image,
    int64_t target_height,
    int64_t target_width
) {
    auto base = image;
    if (base.size(1) != 1) {
        base = base.mean(1, true);
    }
    auto blur = torch::avg_pool2d(base, {15, 15}, {1, 1}, {7, 7}, false, true);
    auto contrast = (base - blur).abs();
    auto dx = (base - torch::roll(base, {1}, {3})).abs();
    auto dy = (base - torch::roll(base, {1}, {2})).abs();
    auto saliency = contrast + dx + dy;
    saliency = torch::avg_pool2d(saliency, {5, 5}, {1, 1}, {2, 2}, false, true);
    saliency = torch::nn::functional::interpolate(
        saliency,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{target_height, target_width})
            .mode(torch::kBilinear)
            .align_corners(false));
    auto flat = saliency.reshape({saliency.size(0), saliency.size(1), saliency.size(2) * saliency.size(3)});
    auto min_value = std::get<0>(flat.min(2, true)).reshape({saliency.size(0), saliency.size(1), 1, 1});
    auto max_value = std::get<0>(flat.max(2, true)).reshape({saliency.size(0), saliency.size(1), 1, 1});
    return (saliency - min_value) / (max_value - min_value).clamp_min(1.0e-6);
}

struct InferenceModules {
    Backbone backbone{nullptr};
    SparseHead sparse_head{nullptr};
    DenseHead dense_head{nullptr};
    PlanetaryGraphMatcher graph_matcher{nullptr};
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
    modules.graph_matcher = PlanetaryGraphMatcher(
        config.descriptor_dim,
        config.graph_hidden_dim,
        config.graph_attention_layers);

    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive backbone_archive;
    torch::serialize::InputArchive sparse_head_archive;
    torch::serialize::InputArchive dense_head_archive;
    torch::serialize::InputArchive graph_matcher_archive;
    archive.read("backbone", backbone_archive);
    archive.read("sparse_head", sparse_head_archive);
    archive.read("dense_head", dense_head_archive);
    archive.read("graph_matcher", graph_matcher_archive);
    modules.backbone->load(backbone_archive);
    modules.sparse_head->load(sparse_head_archive);
    modules.dense_head->load(dense_head_archive);
    modules.graph_matcher->load(graph_matcher_archive);

    modules.backbone->to(device);
    modules.sparse_head->to(device);
    modules.dense_head->to(device);
    modules.graph_matcher->to(device);
    modules.backbone->eval();
    modules.sparse_head->eval();
    modules.dense_head->eval();
    modules.graph_matcher->eval();
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
    const auto descriptors = blend_rotation_invariant_texture_descriptor(sparse.descriptors, input);
    const auto heatmap = make_rotation_invariant_texture_saliency(input, sparse.heatmap.size(2), sparse.heatmap.size(3));
    const auto dense = modules.dense_head->forward(feature_pyramid.front(), feature_pyramid.front());
    const auto dense_confidence = torch::nn::functional::interpolate(
        dense.confidence,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{sparse.heatmap.size(2), sparse.heatmap.size(3)})
            .mode(torch::kNearest));
    return RawFeatureMaps{
        heatmap.detach().cpu().contiguous(),
        descriptors.detach().cpu().contiguous(),
        sparse.scale.detach().cpu().contiguous(),
        sparse.orientation.detach().cpu().contiguous(),
        sparse.affine.detach().cpu().contiguous(),
        dense_confidence.detach().cpu().contiguous()};
}

struct ExtractionTiming {
    double image_load_seconds = 0.0;
    double model_forward_seconds = 0.0;
    double decode_seconds = 0.0;
};

struct ExtractedFeatureSet {
    FeatureSet features;
    int64_t feature_map_width = 0;
    int64_t feature_map_height = 0;
    ExtractionTiming timing;
};

MatchSet filterMatchMode(const MatchSet& match_set, const std::string& match_mode) {
    if (match_mode == "both") {
        return match_set;
    }

    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (match_mode == "sparse") {
        return MatchSet{
            match_set.sparse_matches,
            match_set.sparse_scores,
            torch::empty({0, 2}, float_options),
            torch::empty({0, 2}, float_options),
            torch::empty({0}, float_options)};
    }
    if (match_mode == "dense") {
        const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        return MatchSet{
            torch::empty({0, 2}, long_options),
            torch::empty({0}, float_options),
            match_set.points_a,
            match_set.points_b,
            match_set.confidence};
    }
    throw std::invalid_argument("match mode must be sparse, dense, or both");
}

FeatureDecodeConfig makeFeatureDecodeConfig(const CliOptions& options) {
    FeatureDecodeConfig config;
    config.max_keypoints = options.max_keypoints;
    config.min_keypoints = options.min_keypoints;
    config.semi_dense_threshold = options.semi_dense_threshold;
    config.keypoint_grid_rows = options.keypoint_grid_rows;
    config.keypoint_grid_cols = options.keypoint_grid_cols;
    config.keypoints_per_cell = options.keypoints_per_cell;
    config.nms_radius = options.nms_radius;
    return config;
}

ExtractedFeatureSet extract_feature_set(
    const std::string& image_path,
    InferenceModules& modules,
    const CheckpointConfig& checkpoint_config,
    torch::Device device,
    const FeatureDecodeConfig& decode_config,
    double min_keypoint_intensity
) {
    ExtractionTiming timing;
    Timer image_timer;
    const auto image = load_image_tensor(image_path);
    timing.image_load_seconds = image_timer.elapsedSeconds();

    Timer forward_timer;
    const auto maps = run_mvp_model(image, modules, checkpoint_config, device);
    timing.model_forward_seconds = forward_timer.elapsedSeconds();

    Timer decode_timer;
    const auto intensity_mask = make_intensity_mask(image, min_keypoint_intensity).to(torch::kCPU);
    auto features = decode_feature_maps(maps, decode_config, intensity_mask);
    timing.decode_seconds = decode_timer.elapsedSeconds();
    features.feature_map_width = maps.heatmap.size(3);
    features.feature_map_height = maps.heatmap.size(2);

    return ExtractedFeatureSet{std::move(features), maps.heatmap.size(3), maps.heatmap.size(2), timing};
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
        config.base_channels = options.base_channels;
        config.descriptor_dim = options.descriptor_dim;
        config.graph_hidden_dim = options.graph_hidden_dim;
        config.graph_attention_layers = options.graph_attention_layers;
        config.pairs_per_image = options.pairs_per_image;
        config.learning_rate = options.learning_rate;
        config.weight_decay = options.weight_decay;
        config.augmentation_profile = options.augmentation_profile;
        config.extreme_pair_ratio = options.extreme_pair_ratio;
        config.rotation_step_degrees = options.rotation_step_degrees;
        config.train_ratio = options.train_ratio;
        config.val_ratio = options.val_ratio;
        config.split_seed = options.split_seed;
        config.synthetic_pair_cache_dir = options.synthetic_pair_cache_dir;
        config.log_csv = options.log_csv;
        config.dataloader_workers = options.dataloader_workers;
        config.prefetch_batches = options.prefetch_batches;
        config.pin_memory = options.pin_memory;
        config.synthetic_pair_cache_rebuild = options.synthetic_pair_cache_rebuild;
        config.visualization_dir = options.visualization_dir;
        config.visualization_samples = options.visualization_samples;
        config.visualization_samples_all = options.visualization_samples_all;
        config.max_keypoints = options.max_keypoints;
        config.min_keypoints = options.min_keypoints;
        config.keypoint_grid_rows = options.keypoint_grid_rows;
        config.keypoint_grid_cols = options.keypoint_grid_cols;
        config.keypoints_per_cell = options.keypoints_per_cell;
        config.nms_radius = options.nms_radius;
        config.min_keypoint_intensity = options.min_keypoint_intensity;
        const auto result = train_model(config);
        std::cout << "training complete: epochs=" << result.epochs_completed
                  << " final_loss=" << result.final_loss
                  << " total_time=" << formatSeconds(result.total_time_seconds) << "s"
                  << " avg_batch_time=" << formatSeconds(result.avg_batch_time_seconds) << "s\n";
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
        Timer total_timer;
        if (!checkpoint_can_load(options.checkpoint)) {
            std::cerr << "extract failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        const auto decode_config = makeFeatureDecodeConfig(options);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config, device);
        const auto extracted = extract_feature_set(
            options.image,
            modules,
            checkpoint_config,
            device,
            decode_config,
            options.min_keypoint_intensity
        );
        Timer save_timer;
        save_feature_set(extracted.features, options.output);
        const auto save_seconds = save_timer.elapsedSeconds();
        double visualization_seconds = 0.0;
        if (!options.visualization_dir.empty()) {
            Timer visualization_timer;
            (void)save_feature_visualization(
                options.image,
                extracted.features,
                options.visualization_dir,
                extracted.feature_map_width,
                extracted.feature_map_height);
            visualization_seconds = visualization_timer.elapsedSeconds();
        }
        std::cout << "extraction complete: features=" << options.output
                  << " sparse_features=" << extracted.features.keypoints.size(0)
                  << " dense_features=" << extracted.features.dense_points.size(0)
                  << " elapsed=" << formatSeconds(total_timer.elapsedSeconds()) << "s"
                  << " image_load=" << formatSeconds(extracted.timing.image_load_seconds) << "s"
                  << " model_forward=" << formatSeconds(extracted.timing.model_forward_seconds) << "s"
                  << " decode=" << formatSeconds(extracted.timing.decode_seconds) << "s"
                  << " save=" << formatSeconds(save_seconds) << "s"
                  << " visualization=" << formatSeconds(visualization_seconds) << "s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "extract failed: " << error.what() << '\n';
        return 1;
    }
}

int run_match_command(const CliOptions& options) {
    const bool use_feature_files = !options.feature_a.empty() || !options.feature_b.empty();
    if (!require_path(options.output, "--output")) {
        return 1;
    }
    if (use_feature_files) {
        if (!require_path(options.feature_a, "--feature-a") || !require_path(options.feature_b, "--feature-b") ||
            !require_path(options.checkpoint, "--checkpoint")) {
            return 1;
        }
    } else if (!require_path(options.image_a, "--image-a") || !require_path(options.image_b, "--image-b") ||
               !require_path(options.checkpoint, "--checkpoint")) {
        return 1;
    }

    try {
        Timer total_timer;
        if (!checkpoint_can_load(options.checkpoint)) {
            std::cerr << "match failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config, device);
        ExtractedFeatureSet extracted_a;
        ExtractedFeatureSet extracted_b;
        if (use_feature_files) {
            extracted_a.features = load_feature_set(options.feature_a);
            extracted_b.features = load_feature_set(options.feature_b);
            extracted_a.feature_map_width = extracted_a.features.feature_map_width;
            extracted_a.feature_map_height = extracted_a.features.feature_map_height;
            extracted_b.feature_map_width = extracted_b.features.feature_map_width;
            extracted_b.feature_map_height = extracted_b.features.feature_map_height;
        } else {
            const auto decode_config = makeFeatureDecodeConfig(options);
            extracted_a = extract_feature_set(
                options.image_a,
                modules,
                checkpoint_config,
                device,
                decode_config,
                options.min_keypoint_intensity
            );
            extracted_b = extract_feature_set(
                options.image_b,
                modules,
                checkpoint_config,
                device,
                decode_config,
                options.min_keypoint_intensity
            );
        }
        const auto extract_a_seconds = extracted_a.timing.image_load_seconds +
                                       extracted_a.timing.model_forward_seconds +
                                       extracted_a.timing.decode_seconds;
        const auto extract_b_seconds = extracted_b.timing.image_load_seconds +
                                       extracted_b.timing.model_forward_seconds +
                                       extracted_b.timing.decode_seconds;
        Timer match_timer;
        const auto match_set = filterMatchMode(
            matchFeatureSets(extracted_a.features, extracted_b.features, *modules.graph_matcher),
            options.match_mode);
        const auto match_seconds = match_timer.elapsedSeconds();
        Timer save_timer;
        save_match_set(match_set, options.output);
        const auto save_seconds = save_timer.elapsedSeconds();
        double visualization_seconds = 0.0;
        if (!options.visualization_dir.empty()) {
            Timer visualization_timer;
            if (extracted_a.feature_map_width > 0 && extracted_a.feature_map_height > 0 &&
                extracted_b.feature_map_width > 0 && extracted_b.feature_map_height > 0) {
                (void)save_match_visualization(
                    options.image_a,
                    options.image_b,
                    extracted_a.features,
                    extracted_b.features,
                    match_set,
                    options.visualization_dir,
                    extracted_a.feature_map_width,
                    extracted_a.feature_map_height,
                    extracted_b.feature_map_width,
                    extracted_b.feature_map_height);
            } else {
                (void)save_match_visualization(options.image_a, options.image_b, match_set, options.visualization_dir);
            }
            visualization_seconds = visualization_timer.elapsedSeconds();
        }
        std::cout << "matching complete: matches=" << options.output
                  << " features_a=" << extracted_a.features.keypoints.size(0)
                  << " features_b=" << extracted_b.features.keypoints.size(0)
                  << " sparse_matches=" << match_set.sparse_matches.size(0)
                  << " dense_matches=" << match_set.points_a.size(0)
                  << " elapsed=" << formatSeconds(total_timer.elapsedSeconds()) << "s"
                  << " extract_a=" << formatSeconds(extract_a_seconds) << "s"
                  << " extract_b=" << formatSeconds(extract_b_seconds) << "s"
                  << " match_time=" << formatSeconds(match_seconds) << "s"
                  << " save=" << formatSeconds(save_seconds) << "s"
                  << " visualization=" << formatSeconds(visualization_seconds) << "s\n";
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
        Timer total_timer;
        if (!checkpoint_can_load(options.checkpoint)) {
            std::cerr << "eval failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto pairs = loadEvalPairs(options.pairs);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        const auto decode_config = makeFeatureDecodeConfig(options);
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
                decode_config,
                options.min_keypoint_intensity
            );
            auto extracted_b = extract_feature_set(
                pair.second,
                modules,
                checkpoint_config,
                device,
                decode_config,
                options.min_keypoint_intensity
            );
            match_sets.push_back(matchFeatureSets(extracted_a.features, extracted_b.features, *modules.graph_matcher));
            feature_sets.push_back(std::make_pair(std::move(extracted_a.features), std::move(extracted_b.features)));
        }

        const auto report = aggregateEvalReport(feature_sets, match_sets);
        saveEvalReport(options.output, report);
        const auto elapsed = total_timer.elapsedSeconds();
        const auto avg_pair_time = pairs.empty() ? 0.0 : elapsed / static_cast<double>(pairs.size());
        std::cout << "evaluation complete: report=" << options.output
                  << " pairs=" << pairs.size()
                  << " half_turn_consistency=" << report.half_turn_consistency
                  << " half_turn_mean_error=" << report.half_turn_mean_error
                  << " elapsed=" << formatSeconds(elapsed) << "s"
                  << " avg_pair_time=" << formatSeconds(avg_pair_time) << "s\n";
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
        Timer total_timer;
        if (!inference_checkpoint_can_load(options.checkpoint)) {
            std::cerr << "export failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        copy_file_contents(options.checkpoint, options.output);
        if (!inference_checkpoint_can_load(options.output)) {
            std::cerr << "export failed: exported checkpoint cannot load: " << options.output << '\n';
            return 1;
        }
        std::cout << "export complete: checkpoint=" << options.output
                  << " elapsed=" << formatSeconds(total_timer.elapsedSeconds()) << "s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "export failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace pfm
