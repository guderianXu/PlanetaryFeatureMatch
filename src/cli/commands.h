#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CLI11.hpp"

namespace pfm
{

/// Supported command line subcommands.
enum class Command
{
    None,
    Train,
    Extract,
    Match,
    Eval,
    Export,
};

/// Parsed command line options for pfm commands.
struct CliOptions
{
    Command command = Command::None;
    std::string image_dir;
    std::string image;
    std::string image_a;
    std::string image_b;
    std::string feature_a;
    std::string feature_b;
    std::string pairs;
    std::string checkpoint;
    std::string init_checkpoint;
    std::string config;
    std::string output;
    std::string warp_a_to_b;
    std::string visualization_dir;
    std::string visualization_samples_option = "4";
    int visualization_samples = 4;
    bool visualization_samples_all = false;
    std::string device = "cpu";
    int max_keypoints = 1024;
    int min_keypoints = 0;
    double semi_dense_threshold = 0.5;
    double min_keypoint_intensity = 0.08;
    double match_correct_threshold_pixels = 5.0;
    int keypoint_grid_rows = 8;
    int keypoint_grid_cols = 8;
    int keypoints_per_cell = 0;
    int nms_radius = 4;
    int descriptor_pool_radius = 0;
    bool disable_descriptor_orientation_canonicalization = false;
    std::string match_mode = "sparse";
    std::string sparse_geometry_filter;
    int epochs = 1;
    int batch_size = 1;
    int resize = 512;
    int base_channels = 32;
    int descriptor_dim = 128;
    int graph_hidden_dim = 256;
    int graph_attention_layers = 6;
    int graph_keypoint_meta_dim = 16;
    bool full_v21 = false;
    std::string training_profile = "full";
    int pairs_per_image = 1;
    int max_train_batches = 0;
    double learning_rate = 3.0e-4;
    int lr_warmup_steps = 0;
    double min_learning_rate_ratio = 0.01;
    double weight_decay = 5.0e-4;
    std::string augmentation_profile = "mixed";
    bool augmentation_curriculum = false;
    double extreme_pair_ratio = 0.2;
    double rotation_step_degrees = 15.0;
    double train_ratio = 1.0;
    double val_ratio = 0.0;
    int split_seed = 42;
    std::string synthetic_pair_cache_dir;
    std::vector<std::string> extra_synthetic_pair_cache_dirs;
    std::vector<std::string> hard_synthetic_pair_cache_dirs;
    std::vector<std::string> pair_cache_dirs;
    int64_t pair_cache_limit = 0;
    int hard_synthetic_pair_cache_repeats = 3;
    std::vector<int64_t> hard_synthetic_pair_cache_indices;
    bool cache_only = false;
    std::string log_csv;
    int dataloader_workers = 0;
    int prefetch_batches = 2;
    bool pin_memory = false;
    bool descriptor_only_finetune = false;
    bool viewpoint_head_only_finetune = false;
    bool graph_only_finetune = false;
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

} // namespace pfm
