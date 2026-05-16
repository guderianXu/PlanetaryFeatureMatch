#pragma once

#include <string>

namespace pfm {

struct TrainConfig {
    std::string image_dir;
    std::string checkpoint;
    std::string device = "cpu";
    int epochs = 1;
    int batch_size = 1;
    int base_channels = 8;
    int descriptor_dim = 32;
    int resize = 512;
    int pairs_per_image = 1;
    std::string augmentation_profile = "mixed";
    double extreme_pair_ratio = 0.2;
    std::string synthetic_pair_cache_dir;
    bool synthetic_pair_cache_rebuild = false;
    double min_keypoint_intensity = 0.0;
    double learning_rate = 1.0e-3;
};

struct TrainResult {
    int epochs_completed = 0;
    double initial_loss = 0.0;
    double final_loss = 0.0;
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
