#include "data/synthetic_pair_dataset.h"

#include <stdexcept>
#include <utility>

namespace pfm {

SyntheticPairTensorDataset::SyntheticPairTensorDataset(
    std::vector<torch::Tensor> images,
    size_t pairs_per_image,
    ImagePairAugmentationConfig config)
    : _images(std::move(images)), _pairs_per_image(pairs_per_image), _config(config) {
    if (_images.empty()) {
        throw std::invalid_argument("synthetic pair tensor dataset requires at least one image");
    }
    if (_pairs_per_image == 0) {
        throw std::invalid_argument("pairs per image must be positive");
    }
}

size_t SyntheticPairTensorDataset::size() const {
    return _images.size() * _pairs_per_image;
}

TensorBatch SyntheticPairTensorDataset::get(size_t index) {
    if (index >= size()) {
        throw std::out_of_range("synthetic pair tensor dataset index out of range");
    }

    const auto source_index = index / _pairs_per_image;
    const auto variant_index = index % _pairs_per_image;
    auto config = _config;
    config.source_index = static_cast<int64_t>(source_index);
    config.variant_index = static_cast<int64_t>(variant_index);
    const auto sample = ImagePairAugmentor(config).augment(_images[source_index]);

    TensorBatch batch;
    batch["view_a"] = sample.view_a;
    batch["view_b"] = sample.view_b;
    batch["warp_a_to_b"] = sample.warp_a_to_b;
    batch["valid_mask"] = sample.valid_mask;
    return batch;
}

}  // namespace pfm
