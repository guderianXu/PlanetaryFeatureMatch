#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm {

/// Loads an image file into a contiguous CHW float32 tensor normalized to [0, 1].
///
/// Supports 1-channel grayscale, 3-channel BGR, and 4-channel BGRA images with 8-bit or 16-bit unsigned depth.
/// Color images are returned in RGB channel order.
///
/// @param path Filesystem path to the image file.
/// @return Contiguous tensor with shape {C, H, W}, dtype float32, and values in [0, 1].
/// @throws std::invalid_argument if the image cannot be loaded or has unsupported channels/depth.
torch::Tensor load_image_tensor(const std::string& path);

}  // namespace pfm
