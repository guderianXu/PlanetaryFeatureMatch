#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "CLI11.hpp"

#include "cli/commands.h"
#include "infer/pipeline.h"

namespace pfm {

namespace {

void parseVisualizationSamples(CliOptions& options) {
    if (options.visualization_samples_option == "all") {
        options.visualization_samples_all = true;
        return;
    }
    std::size_t consumed = 0;
    int samples = 0;
    try {
        samples = std::stoi(options.visualization_samples_option, &consumed);
    } catch (const std::exception&) {
        throw CLI::ValidationError("--visualization-samples", "expected non-negative integer or all");
    }
    if (consumed != options.visualization_samples_option.size() || samples < 0) {
        throw CLI::ValidationError("--visualization-samples", "expected non-negative integer or all");
    }
    options.visualization_samples = samples;
    options.visualization_samples_all = false;
}

}  // namespace

std::unique_ptr<CLI::App> build_cli_app(CliOptions& options) {
    auto app = std::make_unique<CLI::App>("Planetary feature matching");
    app->require_subcommand(1);
    app->get_formatter()->enable_footer_formatting(false);
    app->footer(
        "\nCommon command options:\n"
        "  train --image-dir images --checkpoint model.pt [--epochs 1] [--batch-size 1] [--device cpu] "
        "[--resize 512] [--pairs-per-image 1] [--augmentation-profile mixed] "
        "[--extreme-pair-ratio 0.2] [--synthetic-pair-cache-dir build/pair_cache] [--log-csv metrics.csv] "
        "[--dataloader-workers 0] [--prefetch-batches 2] [--pin-memory] "
        "[--synthetic-pair-cache-rebuild] [--visualization-dir vis] [--visualization-samples 4] "
        "[--min-keypoint-intensity 0.0] [--max-keypoints 1024] [--min-keypoints 0] [--keypoint-grid-rows 8] "
        "[--keypoint-grid-cols 8] [--keypoints-per-cell 0] [--nms-radius 4]\n"
        "  extract --image a.tif --checkpoint model.pt --output features.pt [--device cpu] "
        "[--max-keypoints 1024] [--min-keypoints 0] [--semi-dense-threshold 0.5] [--visualization-dir vis] "
        "[--min-keypoint-intensity 0.0] [--keypoint-grid-rows 8] [--keypoint-grid-cols 8] "
        "[--keypoints-per-cell 0] [--nms-radius 4]\n"
        "  match --image-a a.tif --image-b b.tif --checkpoint model.pt --output matches.pt [--device cpu] "
        "[--feature-a a_features.pt] [--feature-b b_features.pt] [--match-mode both] "
        "[--max-keypoints 1024] [--min-keypoints 0] [--semi-dense-threshold 0.5] [--visualization-dir vis] "
        "[--min-keypoint-intensity 0.0] [--keypoint-grid-rows 8] [--keypoint-grid-cols 8] "
        "[--keypoints-per-cell 0] [--nms-radius 4]\n"
        "  eval --pairs pairs.txt --checkpoint model.pt --output report.pt [--device cpu] "
        "[--max-keypoints 1024] [--min-keypoints 0] [--semi-dense-threshold 0.5] [--min-keypoint-intensity 0.0] "
        "[--keypoint-grid-rows 8] [--keypoint-grid-cols 8] [--keypoints-per-cell 0] [--nms-radius 4]\n"
        "  export --checkpoint model.pt --output exported.pt\n"
        "\nUse '<subcommand> --help' for detailed descriptions."
    );

    CLI::App* train = app->add_subcommand("train", "Train a feature matching model");
    train->add_option("--image-dir", options.image_dir, "Training image directory")->required();
    train->add_option("--checkpoint", options.checkpoint, "Model checkpoint path")->required();
    train->add_option("--pairs", options.pairs, "Training pair list");
    train->add_option("--config", options.config, "Training configuration");
    train->add_option("--output", options.output, "Output checkpoint path");
    train->add_option("--device", options.device, "Compute device");
    train->add_option("--epochs", options.epochs, "Training epochs");
    train->add_option("--batch-size", options.batch_size, "Training batch size");
    train->add_option("--resize", options.resize, "Resize training image max edge; use 0 to keep original size");
    train->add_option("--base-channels", options.base_channels, "Backbone base channel count")
        ->check(CLI::PositiveNumber);
    train->add_option("--descriptor-dim", options.descriptor_dim, "Descriptor dimension")
        ->check(CLI::PositiveNumber);
    train->add_option("--graph-hidden-dim", options.graph_hidden_dim, "Graph matcher hidden dimension")
        ->check(CLI::PositiveNumber);
    train->add_option("--learning-rate", options.learning_rate, "Initial learning rate");
    train->add_option("--weight-decay", options.weight_decay, "AdamW weight decay");
    train->add_option("--graph-attention-layers", options.graph_attention_layers, "Graph matcher attention layer count")
        ->check(CLI::PositiveNumber);
    train->add_option("--pairs-per-image", options.pairs_per_image, "Synthetic training pairs generated per source image");
    train->add_option("--augmentation-profile",
                      options.augmentation_profile,
                      "Synthetic augmentation profile: mixed, mild, medium, hard, or extreme");
    train->add_option("--extreme-pair-ratio",
                      options.extreme_pair_ratio,
                      "Extreme pair ratio used by mixed augmentation profile");
    train->add_option("--train-ratio", options.train_ratio, "Training split ratio")
        ->check(CLI::Range(0.1, 1.0));
    train->add_option("--val-ratio", options.val_ratio, "Validation split ratio")
        ->check(CLI::Range(0.0, 0.5));
    train->add_option("--split-seed", options.split_seed, "Random seed for train/val split");
    train->add_option("--synthetic-pair-cache-dir",
                      options.synthetic_pair_cache_dir,
                      "Directory for cached synthetic training pairs");
    train->add_option("--log-csv", options.log_csv, "CSV path for per-iteration training metrics");
    train->add_option("--dataloader-workers", options.dataloader_workers, "Online synthetic pair dataloader worker count")
        ->check(CLI::NonNegativeNumber);
    train->add_option("--prefetch-batches", options.prefetch_batches, "Async dataloader prefetch batch count")
        ->check(CLI::PositiveNumber);
    train->add_flag("--pin-memory", options.pin_memory, "Pin CPU dataloader batches before device transfer");
    train->add_option("--visualization-dir", options.visualization_dir, "Directory for training diagnostic PNG output");
    train->add_option("--visualization-samples",
                      options.visualization_samples_option,
                      "Training diagnostic sample count or all");
    train->add_option("--min-keypoint-intensity",
                      options.min_keypoint_intensity,
                      "Minimum normalized image intensity for keypoint supervision")
        ->check(CLI::Range(0.0, 1.0));
    train->add_option("--max-keypoints", options.max_keypoints, "Maximum sparse keypoints for training visualization")
        ->check(CLI::PositiveNumber);
    train->add_option("--min-keypoints", options.min_keypoints, "Soft minimum sparse keypoints for training visualization")
        ->check(CLI::NonNegativeNumber);
    train->add_option("--keypoint-grid-rows", options.keypoint_grid_rows, "Training visualization sparse keypoint grid rows")
        ->check(CLI::PositiveNumber);
    train->add_option("--keypoint-grid-cols", options.keypoint_grid_cols, "Training visualization sparse keypoint grid columns")
        ->check(CLI::PositiveNumber);
    train->add_option("--keypoints-per-cell",
                      options.keypoints_per_cell,
                      "Training visualization sparse keypoints per grid cell; 0 derives from max-keypoints")
        ->check(CLI::NonNegativeNumber);
    train->add_option("--nms-radius", options.nms_radius, "Training visualization sparse keypoint NMS radius")
        ->check(CLI::NonNegativeNumber);
    train->add_flag("--synthetic-pair-cache-rebuild",
                    options.synthetic_pair_cache_rebuild,
                    "Rebuild cached synthetic training pairs");
    train->callback([&options]() {
        parseVisualizationSamples(options);
        options.command = Command::Train;
    });

    CLI::App* extract = app->add_subcommand("extract", "Extract image features");
    extract->add_option("--image", options.image, "Input image path")->required();
    extract->add_option("--checkpoint", options.checkpoint, "Model checkpoint path")->required();
    extract->add_option("--output", options.output, "Output feature path")->required();
    extract->add_option("--device", options.device, "Compute device");
    extract->add_option("--max-keypoints", options.max_keypoints, "Maximum sparse keypoints");
    extract->add_option("--min-keypoints", options.min_keypoints, "Soft minimum sparse keypoints")
        ->check(CLI::NonNegativeNumber);
    extract->add_option("--semi-dense-threshold", options.semi_dense_threshold, "Semi-dense confidence threshold");
    extract->add_option("--visualization-dir", options.visualization_dir, "Directory for feature visualization PNG output");
    extract->add_option("--min-keypoint-intensity",
                        options.min_keypoint_intensity,
                        "Minimum normalized image intensity for output keypoints")
        ->check(CLI::Range(0.0, 1.0));
    extract->add_option("--keypoint-grid-rows", options.keypoint_grid_rows, "Sparse keypoint grid rows")
        ->check(CLI::PositiveNumber);
    extract->add_option("--keypoint-grid-cols", options.keypoint_grid_cols, "Sparse keypoint grid columns")
        ->check(CLI::PositiveNumber);
    extract->add_option("--keypoints-per-cell",
                        options.keypoints_per_cell,
                        "Sparse keypoints per grid cell; 0 derives from max-keypoints")
        ->check(CLI::NonNegativeNumber);
    extract->add_option("--nms-radius", options.nms_radius, "Sparse keypoint NMS radius in feature-map pixels")
        ->check(CLI::NonNegativeNumber);
    extract->callback([&options]() { options.command = Command::Extract; });

    CLI::App* match = app->add_subcommand("match", "Match two images or two pre-extracted feature files");
    match->add_option("--image-a", options.image_a, "First image path");
    match->add_option("--image-b", options.image_b, "Second image path");
    match->add_option("--feature-a", options.feature_a, "First pre-extracted feature file");
    match->add_option("--feature-b", options.feature_b, "Second pre-extracted feature file");
    match->add_option("--checkpoint", options.checkpoint, "Model checkpoint path");
    match->add_option("--output", options.output, "Output matches path")->required();
    match->add_option("--device", options.device, "Compute device");
    match->add_option("--match-mode", options.match_mode, "Match output mode: sparse, dense, or both")
        ->check(CLI::IsMember({"sparse", "dense", "both"}));
    match->add_option("--max-keypoints", options.max_keypoints, "Maximum sparse keypoints");
    match->add_option("--min-keypoints", options.min_keypoints, "Soft minimum sparse keypoints")
        ->check(CLI::NonNegativeNumber);
    match->add_option("--semi-dense-threshold", options.semi_dense_threshold, "Semi-dense confidence threshold");
    match->add_option("--visualization-dir", options.visualization_dir, "Directory for match visualization PNG output");
    match->add_option("--min-keypoint-intensity",
                      options.min_keypoint_intensity,
                      "Minimum normalized image intensity for output keypoints")
        ->check(CLI::Range(0.0, 1.0));
    match->add_option("--keypoint-grid-rows", options.keypoint_grid_rows, "Sparse keypoint grid rows")
        ->check(CLI::PositiveNumber);
    match->add_option("--keypoint-grid-cols", options.keypoint_grid_cols, "Sparse keypoint grid columns")
        ->check(CLI::PositiveNumber);
    match->add_option("--keypoints-per-cell",
                      options.keypoints_per_cell,
                      "Sparse keypoints per grid cell; 0 derives from max-keypoints")
        ->check(CLI::NonNegativeNumber);
    match->add_option("--nms-radius", options.nms_radius, "Sparse keypoint NMS radius in feature-map pixels")
        ->check(CLI::NonNegativeNumber);
    match->callback([&options]() { options.command = Command::Match; });

    CLI::App* eval = app->add_subcommand("eval", "Evaluate feature matching results");
    eval->add_option("--pairs", options.pairs, "Evaluation pair list")->required();
    eval->add_option("--checkpoint", options.checkpoint, "Model checkpoint path")->required();
    eval->add_option("--output", options.output, "Output report path")->required();
    eval->add_option("--device", options.device, "Compute device");
    eval->add_option("--max-keypoints", options.max_keypoints, "Maximum sparse keypoints");
    eval->add_option("--min-keypoints", options.min_keypoints, "Soft minimum sparse keypoints")
        ->check(CLI::NonNegativeNumber);
    eval->add_option("--semi-dense-threshold", options.semi_dense_threshold, "Semi-dense confidence threshold");
    eval->add_option("--min-keypoint-intensity",
                     options.min_keypoint_intensity,
                     "Minimum normalized image intensity for output keypoints")
        ->check(CLI::Range(0.0, 1.0));
    eval->add_option("--keypoint-grid-rows", options.keypoint_grid_rows, "Sparse keypoint grid rows")
        ->check(CLI::PositiveNumber);
    eval->add_option("--keypoint-grid-cols", options.keypoint_grid_cols, "Sparse keypoint grid columns")
        ->check(CLI::PositiveNumber);
    eval->add_option("--keypoints-per-cell",
                     options.keypoints_per_cell,
                     "Sparse keypoints per grid cell; 0 derives from max-keypoints")
        ->check(CLI::NonNegativeNumber);
    eval->add_option("--nms-radius", options.nms_radius, "Sparse keypoint NMS radius in feature-map pixels")
        ->check(CLI::NonNegativeNumber);
    eval->callback([&options]() { options.command = Command::Eval; });

    CLI::App* export_command = app->add_subcommand("export", "Export a trained model");
    export_command->add_option("--checkpoint", options.checkpoint, "Model checkpoint path")->required();
    export_command->add_option("--output", options.output, "Output model path")->required();
    export_command->callback([&options]() { options.command = Command::Export; });

    return app;
}

CliOptions parse_cli(const std::vector<std::string>& args) {
    CliOptions options;
    std::unique_ptr<CLI::App> app = build_cli_app(options);

    std::vector<std::string> parse_args = args;
    if (!parse_args.empty()) {
        parse_args.erase(parse_args.begin());
    }
    std::reverse(parse_args.begin(), parse_args.end());
    app->parse(parse_args);
    return options;
}

int run_cli(int argc, char** argv) {
    CliOptions options;
    std::unique_ptr<CLI::App> app = build_cli_app(options);

    try {
        app->parse(argc, argv);
        switch (options.command) {
            case Command::Train:
                return run_train_command(options);
            case Command::Extract:
                return run_extract_command(options);
            case Command::Match:
                return run_match_command(options);
            case Command::Eval:
                return run_eval_command(options);
            case Command::Export:
                return run_export_command(options);
            case Command::None:
                std::cerr << "missing command\n";
                return 1;
        }
        return 1;
    } catch (const CLI::ParseError& error) {
        return app->exit(error);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

}  // namespace pfm
