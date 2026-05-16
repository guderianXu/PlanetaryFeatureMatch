#pragma once

#include <torch/torch.h>

#include "infer/feature_codec.h"

namespace pfm {

struct RawFeatureMaps {
    torch::Tensor heatmap;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
    torch::Tensor dense_confidence;
};

/// Controls sparse keypoint and semi-dense feature decoding.
struct FeatureDecodeConfig {
    /// Maximum sparse keypoints to return.
    int max_keypoints = 1024;
    /// Minimum dense confidence value for semi-dense output.
    double semi_dense_threshold = 0.5;
    /// Sparse keypoint grid row count.
    int keypoint_grid_rows = 8;
    /// Sparse keypoint grid column count.
    int keypoint_grid_cols = 8;
    /// Sparse keypoints per grid cell; 0 derives the value from max_keypoints.
    int keypoints_per_cell = 0;
    /// Sparse keypoint non-maximum suppression radius in feature-map pixels.
    int nms_radius = 4;
};

/// Decodes sparse and dense feature tensors from raw network output maps.
/// @param maps Raw heatmap, descriptor, scale, orientation, affine, and dense confidence maps; each must be
/// defined CPU 4D tensors with batch size 1, matching positive spatial sizes, and valid channel counts.
/// @param max_keypoints Maximum number of sparse heatmap locations to return; must be positive.
/// @param semi_dense_threshold Minimum dense confidence value for dense point output.
/// @return FeatureSet containing contiguous CPU float tensors decoded in feature-map coordinates.
/// @throws std::invalid_argument if maps are undefined, non-CPU, non-4D, have batch size other than 1, have
/// invalid channel counts, have mismatched or non-positive spatial sizes, or max_keypoints is not positive.
FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold);

/// Decodes sparse and dense feature tensors while suppressing invalid image locations.
/// @param maps Raw heatmap, descriptor, scale, orientation, affine, and dense confidence maps.
/// @param max_keypoints Maximum number of sparse heatmap locations to return; must be positive.
/// @param semi_dense_threshold Minimum dense confidence value for dense point output.
/// @param intensity_mask Optional H x W CPU mask in image coordinates; nonzero values are valid.
/// @return FeatureSet containing only valid sparse and dense locations.
/// @throws std::invalid_argument if maps, arguments, or mask shape/device are invalid.
FeatureSet decode_feature_maps(
    const RawFeatureMaps& maps,
    int max_keypoints,
    double semi_dense_threshold,
    const torch::Tensor& intensity_mask
);

/// Decodes sparse and dense feature tensors using an explicit decode configuration.
/// @param maps Raw heatmap, descriptor, scale, orientation, affine, and dense confidence maps.
/// @param config Sparse keypoint and semi-dense decoding parameters; counts must be valid and non-negative where
/// documented by FeatureDecodeConfig.
/// @param intensity_mask Optional H x W CPU mask in image coordinates; nonzero values are valid.
/// @return FeatureSet containing sparse locations after local NMS and dense locations above the configured threshold.
/// @throws std::invalid_argument if maps, config, or mask shape/device are invalid.
FeatureSet decode_feature_maps(
    const RawFeatureMaps& maps,
    const FeatureDecodeConfig& config,
    const torch::Tensor& intensity_mask
);

}  // namespace pfm
