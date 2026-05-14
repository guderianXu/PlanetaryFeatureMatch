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

/// Decodes sparse and dense feature tensors from raw network output maps.
/// @param maps Raw heatmap, descriptor, scale, orientation, affine, and dense confidence maps; each must be
/// defined 4D tensors with batch size 1.
/// @param max_keypoints Maximum number of sparse heatmap locations to return.
/// @param semi_dense_threshold Minimum dense confidence value for dense point output.
/// @return FeatureSet containing contiguous float tensors decoded in feature-map coordinates.
/// @throws std::invalid_argument if maps are undefined, non-4D, batch size is not 1, or max_keypoints is not positive.
FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold);

}  // namespace pfm
