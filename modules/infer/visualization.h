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

}  // namespace pfm
