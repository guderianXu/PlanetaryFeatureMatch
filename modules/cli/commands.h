#pragma once

#include <memory>
#include <string>
#include <vector>

#include "CLI11.hpp"

namespace pfm {

/// Supported command line subcommands.
enum class Command {
    None,
    Train,
    Extract,
    Match,
    Eval,
    Export,
};

/// Parsed command line options for pfm commands.
struct CliOptions {
    Command command = Command::None;
    std::string image_dir;
    std::string image;
    std::string image_a;
    std::string image_b;
    std::string pairs;
    std::string checkpoint;
    std::string config;
    std::string output;
    std::string visualization_dir;
    std::string device = "cpu";
    int max_keypoints = 1024;
    double semi_dense_threshold = 0.5;
    double min_keypoint_intensity = 0.0;
    int epochs = 1;
    int batch_size = 1;
    int resize = 512;
    int pairs_per_image = 1;
    std::string augmentation_profile = "mixed";
    double extreme_pair_ratio = 0.2;
    std::string synthetic_pair_cache_dir;
    bool synthetic_pair_cache_rebuild = false;
};

/// Parse command line arguments into CLI options.
/// @param args Command line arguments including the program name.
/// @return Parsed command line options.
/// @throws CLI::ParseError when CLI11 rejects the command line.
CliOptions parse_cli(const std::vector<std::string>& args);

/// Build the CLI11 application and bind parse results into options.
/// @param options Parsed options object populated by CLI11 callbacks and option bindings.
/// @return Configured CLI11 application.
std::unique_ptr<CLI::App> build_cli_app(CliOptions& options);

/// Run the pfm command line parser.
/// @param argc Argument count from main.
/// @param argv Argument values from main.
/// @return Zero on success, nonzero on parse or runtime errors.
int run_cli(int argc, char** argv);

}  // namespace pfm
