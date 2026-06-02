#include <stdexcept>

#include "dataloader/tensor_batch.h"

namespace pfm
{

TensorBatch moveBatchToDevice(const TensorBatch& batch, const torch::Device& device, bool non_blocking)
{
    TensorBatch result;
    result.reserve(batch.size());
    for (const auto& [key, tensor] : batch)
    {
        result[key] = tensor.to(device, tensor.dtype(), non_blocking, false);
    }
    return result;
}

TensorBatch pinTensorBatchMemory(const TensorBatch& batch)
{
    TensorBatch result;
    result.reserve(batch.size());
    try
    {
        for (const auto& [key, tensor] : batch)
        {
            result[key] = tensor.device().is_cpu() ? tensor.pin_memory() : tensor;
        }
    }
    catch (const c10::Error& error)
    {
        throw std::runtime_error(std::string("failed to pin tensor batch memory: ") + error.what());
    }
    return result;
}

} // namespace pfm
