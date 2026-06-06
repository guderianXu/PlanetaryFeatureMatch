#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CLI11.hpp"

namespace pfm
{

/// 支持的命令行子命令。
enum class Command
{
    None,
    Train,
    Extract,
    Match,
    Eval,
    Export,
};

/// pfm 命令解析后的选项集合。
struct CliOptions
{
    /// 已选择的子命令。
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
    std::string sparse_match_strategy = "learned";
    int max_matches = 512;
    std::string graph_inference_preset = "off";
    double graph_width_prune_min_score = -1.0;
    double graph_early_stop_min_confidence = -1.0;
    double graph_min_accept_probability = -1.0;
    int graph_max_attention_layers = 0;
    double graph_max_attention_work_fraction = 1.0;
    double graph_width_prune_keep_ratio = 1.0;
    std::string graph_fallback_mode = "geometry";
    int epochs = 1;
    int batch_size = 1;
    int resize = 512;
    int training_crop_size = 0;
    int base_channels = 32;
    int descriptor_dim = 128;
    int graph_hidden_dim = 256;
    int graph_attention_layers = 6;
    int graph_keypoint_meta_dim = 16;
    bool full_v21 = false;
    std::string training_profile = "full";
    int samples_per_pair = 512;
    double synthetic_loss_weight = 0.1;
    double graph_matcher_loss_weight = 1.0;
    std::string graph_matcher_metadata_mode = "full";
    double graph_matcher_accept_weight = 0.2;
    int graph_matcher_accept_negative_topk = 8;
    int graph_matcher_no_match_points = 0;
    double graph_matcher_no_match_min_distance = 4.0;
    int graph_matcher_train_max_attention_layers = 0;
    bool graph_matcher_train_random_attention_layers = false;
    double graph_matcher_train_max_attention_work_fraction = 1.0;
    double graph_matcher_train_width_keep_ratio = 1.0;
    double graph_matcher_prune_ranking_weight = 0.1;
    double graph_matcher_prune_ranking_margin = 0.25;
    double graph_matcher_stop_confidence_weight = 0.05;
    double graph_matcher_stop_confidence_margin = 0.5;
    double graph_matcher_raw_preservation_weight = 0.0;
    double graph_matcher_raw_preservation_margin = 1.0;
    double graph_matcher_raw_preservation_raw_margin = 0.05;
    double graph_matcher_hard_negative_dustbin_weight = 0.0;
    int graph_matcher_hard_negative_dustbin_topk = 8;
    double graph_matcher_hard_negative_dustbin_margin = 0.25;
    double graph_matcher_hard_negative_dustbin_spatial_min_distance = 0.0;
    bool train_backbone = false;
    bool train_dual_fpn = false;
    bool freeze_descriptor_head = false;
    bool train_sparse_context = false;
    bool train_keypoint_head = false;
    bool train_geometry_head = false;
    bool train_blended_descriptors = false;
    bool train_texture_adapter = false;
    bool train_descriptor_fusion = false;
    bool train_quality_head = false;
    bool train_graph_matcher = false;
    double training_texture_blend_weight = 1.0;
    double temperature = 0.07;
    int pairs_per_image = 1;
    int max_train_batches = 0;
    double learning_rate = 3.0e-4;
    int lr_warmup_steps = 0;
    double min_learning_rate_ratio = 0.01;
    double weight_decay = 5.0e-4;
    double gradient_clip_norm = 1.0;
    std::string augmentation_profile = "mixed";
    bool augmentation_curriculum = false;
    double extreme_pair_ratio = 0.2;
    double rotation_step_degrees = 15.0;
    double train_ratio = 1.0;
    double val_ratio = 0.0;
    int seed = 1234;
    int split_seed = 42;
    std::string synthetic_pair_cache_dir;
    std::vector<std::string> extra_synthetic_pair_cache_dirs;
    std::vector<std::string> hard_synthetic_pair_cache_dirs;
    std::vector<std::string> pair_cache_dirs;
    int64_t pair_cache_limit = 0;
    int64_t pair_memory_cache_size = 0;
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

/// 将命令行参数解析为 CLI 选项。
/// @param args 包含程序名在内的命令行参数。
/// @return 解析后的命令行选项。
/// @throws CLI::ParseError 当 CLI11 拒绝命令行参数时抛出。
CliOptions parse_cli(const std::vector<std::string>& args);

/// 构建 CLI11 应用，并把解析结果绑定到 options。
/// @param options 由 CLI11 回调和选项绑定填充的解析结果对象。
/// @return 已配置的 CLI11 应用。
std::unique_ptr<CLI::App> build_cli_app(CliOptions& options);

/// 运行 pfm 命令行解析器。
/// @param argc main 传入的参数数量。
/// @param argv main 传入的参数值。
/// @return 成功返回 0；解析或运行失败返回非 0。
int run_cli(int argc, char** argv);

} // namespace pfm
