#pragma once

#include <vector>

#include "augment/image_pair_augmentor.h"
#include "dataloader/dataset.h"

namespace pfm {

class SyntheticPairTensorDataset : public TensorDataset {
public:
    /// Creates an online synthetic pair dataset.
    /// @param images Source CHW float tensors.
    /// @param pairs_per_image Number of generated pairs for each source image.
    /// @param config Base augmentation configuration.
    /// @throws std::invalid_argument if images is empty or pairs_per_image is zero.
    SyntheticPairTensorDataset(
        std::vector<torch::Tensor> images,
        size_t pairs_per_image,
        ImagePairAugmentationConfig config);

    /// Returns total generated pair count.
    /// @return images.size() multiplied by pairs_per_image.
    size_t size() const override;

    /// Generates one synthetic pair sample.
    /// @param index Dataset index.
    /// @return Tensor batch with view_a, view_b, warp_a_to_b, and valid_mask.
    /// @throws std::out_of_range if index is invalid.
    TensorBatch get(size_t index) override;

private:
    std::vector<torch::Tensor> _images;
    size_t _pairs_per_image;
    ImagePairAugmentationConfig _config;
};

}  // namespace pfm
