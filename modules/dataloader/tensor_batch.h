#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

namespace pfm {

using TensorBatch = std::unordered_map<std::string, torch::Tensor>;

enum class TensorLayout {
    Hw,
    Chw,
    Hwc
};

/// Copies every tensor in a batch to the requested device.
/// \param batch Source tensor batch.
/// \param device Target torch device.
/// \param non_blocking Whether to request non-blocking copies.
/// \return Batch with copied tensors.
TensorBatch moveBatchToDevice(const TensorBatch& batch, const torch::Device& device, bool non_blocking);

/// Copies every CPU tensor in a batch to pinned memory.
/// \param batch Source tensor batch.
/// \return Batch whose tensors use pinned CPU memory when supported.
/// \throws std::runtime_error if pinned allocation fails.
TensorBatch pinTensorBatchMemory(const TensorBatch& batch);

}  // namespace pfm
