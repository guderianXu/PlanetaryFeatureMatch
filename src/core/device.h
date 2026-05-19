#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm {

/// Resolve a compute device string such as cpu, cuda, or cuda:N.
/// @param requested Device string from CLI or configuration.
/// @return Resolved LibTorch device.
/// @throws std::invalid_argument when the string is invalid or the requested CUDA device is unavailable.
torch::Device resolve_compute_device(const std::string& requested);

}  // namespace pfm
