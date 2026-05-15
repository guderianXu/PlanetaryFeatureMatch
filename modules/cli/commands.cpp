#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "CLI11.hpp"

#include "cli/commands.h"
#include "infer/pipeline.h"

namespace pfm {

std::unique_ptr<CLI::App> build_cli_app(CliOptions& options) {
    auto app = std::make_unique<CLI::App>("Planetary feature matching");
    app->require_subcommand(1);

    CLI::App* train = app->add_subcommand("train", "Train a feature matching model");
    train->add_option("--image-dir", options.image_dir, "Training image directory")->required();
    train->add_option("--checkpoint", options.checkpoint, "Model checkpoint path")->required();
    train->add_option("--pairs", options.pairs, "Training pair list");
    train->add_option("--config", options.config, "Training configuration");
    train->add_option("--output", options.output, "Output checkpoint path");
    train->add_option("--device", options.device, "Compute device");
    train->add_option("--epochs", options.epochs, "Training epochs");
    train->add_option("--batch-size", options.batch_size, "Training batch size");
    train->callback([&options]() { options.command = Command::Train; });

    CLI::App* extract = app->add_subcommand("extract", "Extract image features");
    extract->add_option("--image", options.image, "Input image path")->required();
    extract->add_option("--checkpoint", options.checkpoint, "Model checkpoint path")->required();
    extract->add_option("--output", options.output, "Output feature path")->required();
    extract->add_option("--device", options.device, "Compute device");
    extract->add_option("--max-keypoints", options.max_keypoints, "Maximum sparse keypoints");
    extract->add_option("--semi-dense-threshold", options.semi_dense_threshold, "Semi-dense confidence threshold");
    extract->callback([&options]() { options.command = Command::Extract; });

    CLI::App* match = app->add_subcommand("match", "Match two images");
    match->add_option("--image-a", options.image_a, "First image path")->required();
    match->add_option("--image-b", options.image_b, "Second image path")->required();
    match->add_option("--checkpoint", options.checkpoint, "Model checkpoint path")->required();
    match->add_option("--output", options.output, "Output matches path")->required();
    match->add_option("--device", options.device, "Compute device");
    match->add_option("--max-keypoints", options.max_keypoints, "Maximum sparse keypoints");
    match->add_option("--semi-dense-threshold", options.semi_dense_threshold, "Semi-dense confidence threshold");
    match->callback([&options]() { options.command = Command::Match; });

    CLI::App* eval = app->add_subcommand("eval", "Evaluate feature matching results");
    eval->add_option("--pairs", options.pairs, "Evaluation pair list")->required();
    eval->add_option("--checkpoint", options.checkpoint, "Model checkpoint path")->required();
    eval->add_option("--output", options.output, "Output report path")->required();
    eval->add_option("--device", options.device, "Compute device");
    eval->add_option("--max-keypoints", options.max_keypoints, "Maximum sparse keypoints");
    eval->add_option("--semi-dense-threshold", options.semi_dense_threshold, "Semi-dense confidence threshold");
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
