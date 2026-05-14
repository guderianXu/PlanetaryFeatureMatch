#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace pfm {

class ImageDataset {
public:
    /// Builds an image dataset from direct child image files in a directory.
    ///
    /// Supported extensions are .png, .jpg, .jpeg, .tif, and .tiff, matched case-insensitively.
    /// Paths are sorted lexicographically for deterministic traversal.
    ///
    /// @param image_dir Directory containing image files.
    /// @throws std::invalid_argument if image_dir is not a directory or contains no supported images.
    explicit ImageDataset(const std::string& image_dir);

    /// Returns the number of supported image files in the dataset.
    ///
    /// @return Number of image paths collected from the directory.
    std::size_t size() const;

    /// Returns the sorted image path at an index.
    ///
    /// @param index Zero-based dataset index.
    /// @return Reference to the stored image path string.
    /// @throws std::out_of_range if index is not less than size().
    const std::string& path(std::size_t index) const;

    /// Loads one image sample as a tensor.
    ///
    /// @param index Zero-based dataset index.
    /// @return Contiguous CHW float32 image tensor normalized to [0, 1].
    /// @throws std::out_of_range if index is not less than size().
    /// @throws std::invalid_argument if the image cannot be loaded or has unsupported format.
    torch::Tensor load(std::size_t index) const;

private:
    std::vector<std::string> _paths;
};

}  // namespace pfm
