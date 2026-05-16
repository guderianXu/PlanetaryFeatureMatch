#pragma once

#include <filesystem>
#include <string>

#include "infer/feature_codec.h"
#include "infer/match_codec.h"

namespace pfm {

/// Saves a PNG overlay showing extracted feature points on the input image.
/// @param image_path Source image path used as the visualization background.
/// @param feature_set Decoded feature tensors with keypoints in image coordinate order {x, y}.
/// @param visualization_dir Directory where the PNG will be written; created if missing.
/// @return Path to the written PNG file.
/// @throws std::invalid_argument if the image cannot be read or the PNG cannot be written.
std::filesystem::path save_feature_visualization(
    const std::string& image_path,
    const FeatureSet& feature_set,
    const std::string& visualization_dir
);

/// Saves a PNG overlay after scaling feature-map coordinates to source image pixels.
/// @param image_path Source image path used as the visualization background.
/// @param feature_set Decoded feature tensors with keypoints in feature-map coordinates.
/// @param visualization_dir Directory where the PNG will be written; created if missing.
/// @param feature_map_width Width of the feature map that produced keypoints.
/// @param feature_map_height Height of the feature map that produced keypoints.
/// @return Path to the written PNG file.
/// @throws std::invalid_argument if dimensions are invalid, the image cannot be read, or the PNG cannot be written.
std::filesystem::path save_feature_visualization(
    const std::string& image_path,
    const FeatureSet& feature_set,
    const std::string& visualization_dir,
    int64_t feature_map_width,
    int64_t feature_map_height
);

/// Saves a PNG side-by-side overlay showing matched points between two images.
/// @param image_a_path First source image path.
/// @param image_b_path Second source image path.
/// @param match_set Match tensors with points in image coordinate order {x, y} and confidence scores.
/// @param visualization_dir Directory where the PNG will be written; created if missing.
/// @return Path to the written PNG file.
/// @throws std::invalid_argument if either image cannot be read or the PNG cannot be written.
std::filesystem::path save_match_visualization(
    const std::string& image_a_path,
    const std::string& image_b_path,
    const MatchSet& match_set,
    const std::string& visualization_dir
);

/// Saves a PNG match overlay after scaling feature-map points to source image pixels.
/// @param image_a_path First source image path.
/// @param image_b_path Second source image path.
/// @param match_set Match tensors with points in feature-map coordinates.
/// @param visualization_dir Directory where the PNG will be written; created if missing.
/// @param feature_map_a_width Width of the first feature map.
/// @param feature_map_a_height Height of the first feature map.
/// @param feature_map_b_width Width of the second feature map.
/// @param feature_map_b_height Height of the second feature map.
/// @return Path to the written PNG file.
/// @throws std::invalid_argument if dimensions are invalid, either image cannot be read, or the PNG cannot be written.
std::filesystem::path save_match_visualization(
    const std::string& image_a_path,
    const std::string& image_b_path,
    const MatchSet& match_set,
    const std::string& visualization_dir,
    int64_t feature_map_a_width,
    int64_t feature_map_a_height,
    int64_t feature_map_b_width,
    int64_t feature_map_b_height
);

}  // namespace pfm
