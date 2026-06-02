#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace pfm
{

struct TrainConfig
{
    std::string image_dir;
    std::string checkpoint;
    std::string init_checkpoint;
    std::string device = "gpu";
    int epochs = 1;
    int batch_size = 1;
    int base_channels = 32;
    int descriptor_dim = 128;
    int graph_hidden_dim = 256;
    int graph_attention_layers = 6;
    int graph_keypoint_meta_dim = 16;
    std::string training_profile = "full";
    int resize = 512;
    int pairs_per_image = 1;
    int max_train_batches = 0;
    std::string augmentation_profile = "mixed";
    bool augmentation_curriculum = false;
    double extreme_pair_ratio = 0.2;
    double rotation_step_degrees = 15.0;
    std::string synthetic_pair_cache_dir;
    std::vector<std::string> extra_synthetic_pair_cache_dirs;
    std::vector<std::string> hard_synthetic_pair_cache_dirs;
    std::vector<std::string> pair_cache_dirs;
    int64_t pair_cache_limit = 0;
    int hard_synthetic_pair_cache_repeats = 3;
    std::vector<int64_t> hard_synthetic_pair_cache_indices;
    bool cache_only = false;
    std::string log_csv;
    bool synthetic_pair_cache_rebuild = false;
    std::string visualization_dir;
    int visualization_samples = 4;
    bool visualization_samples_all = false;
    int max_keypoints = 1024;
    int min_keypoints = 0;
    int keypoint_grid_rows = 8;
    int keypoint_grid_cols = 8;
    int keypoints_per_cell = 0;
    int nms_radius = 4;
    double min_keypoint_intensity = 0.08;
    double learning_rate = 3.0e-4;
    int lr_warmup_steps = 0;
    double min_learning_rate_ratio = 0.01;
    double weight_decay = 5.0e-4;
    double gradient_clip_norm = 1.0;
    double train_ratio = 1.0;
    double val_ratio = 0.0;
    int split_seed = 42;
    int dataloader_workers = 0;
    int prefetch_batches = 2;
    bool pin_memory = false;
    bool descriptor_only_finetune = false;
    bool viewpoint_head_only_finetune = false;
    bool graph_only_finetune = false;
    bool descriptor_orientation_canonicalization = true;
};

struct TrainResult
{
    int epochs_completed = 0;
    double initial_loss = 0.0;
    double final_loss = 0.0;
    double best_val_loss = std::numeric_limits<double>::max();
    double total_time_seconds = 0.0;
    double avg_batch_time_seconds = 0.0;
};

/// 按配置训练真实影像 MVP 模型，并保存 checkpoint。
/// @param config 训练图像目录、checkpoint 路径、计算设备、数据限制、缓存设置和优化器设置。
/// @return 已完成 epoch 数，以及首次和最终观测到的训练 loss。
/// @throws std::invalid_argument 当路径、数值参数或请求的设备非法时抛出。
TrainResult train_model(const TrainConfig& config);

/// 检查训练 checkpoint 能否作为 LibTorch archive 加载。
/// @param checkpoint 由 train_model 写出的 checkpoint 文件路径。
/// @return archive 可加载且包含必需配置张量时返回 true，否则返回 false。
bool checkpoint_can_load(const std::string& checkpoint);

} // namespace pfm
