#pragma once

#include <limits>
#include <string>

namespace pfm {

struct TrainConfig {
    std::string image_dir;
    std::string checkpoint;
    std::string device = "cpu";
    int epochs = 1;
    int batch_size = 1;
    int base_channels = 32;
    int descriptor_dim = 128;
    int graph_hidden_dim = 256;
    int graph_attention_layers = 6;
    int resize = 512;
    int pairs_per_image = 1;
    std::string augmentation_profile = "mixed";
    double extreme_pair_ratio = 0.2;
    double rotation_step_degrees = 15.0;
    std::string synthetic_pair_cache_dir;
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
    double min_keypoint_intensity = 0.0;
    double learning_rate = 3.0e-4;
    double weight_decay = 5.0e-4;
    double gradient_clip_norm = 1.0;
    double train_ratio = 0.8;
    double val_ratio = 0.1;
    int split_seed = 42;
    int dataloader_workers = 0;
    int prefetch_batches = 2;
    bool pin_memory = false;
};

struct TrainResult {
    int epochs_completed = 0;
    double initial_loss = 0.0;
    double final_loss = 0.0;
    double best_val_loss = std::numeric_limits<double>::max();
    double total_time_seconds = 0.0;
    double avg_batch_time_seconds = 0.0;
};

/// Trains the real-image MVP model for the configured number of epochs and saves a checkpoint.
/// @param config Training image directory, checkpoint path, compute device, data limits, cache settings, and optimizer settings.
/// @return Completed epoch count with first and final observed training losses.
/// @throws std::invalid_argument if paths, numeric settings, or the requested device are invalid.
TrainResult train_model(const TrainConfig& config);

/// Checks whether a training checkpoint can be loaded as a LibTorch archive.
/// @param checkpoint Path to a checkpoint file written by train_model.
/// @return True when the archive loads and contains required config tensors; false otherwise.
bool checkpoint_can_load(const std::string& checkpoint);

}  // namespace pfm
