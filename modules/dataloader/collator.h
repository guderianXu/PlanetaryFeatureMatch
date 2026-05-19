#pragma once

#include <string>
#include <utility>
#include <vector>

#include "dataloader/tensor_batch.h"

namespace pfm {

class TensorBatchCollator {
public:
    /// Creates a collator with required keys and tensor layouts.
    /// \param layouts Required key-layout pairs.
    /// \throws std::invalid_argument if layouts is empty.
    explicit TensorBatchCollator(std::vector<std::pair<std::string, TensorLayout>> layouts);

    /// Pads and stacks samples into one batch.
    /// \param samples Unbatched sample maps.
    /// \return Batched tensor map.
    /// \throws std::invalid_argument if samples are empty, keys are missing, or tensor ranks are invalid.
    TensorBatch collate(const std::vector<TensorBatch>& samples) const;

private:
    std::vector<std::pair<std::string, TensorLayout>> _layouts;
};

}  // namespace pfm
