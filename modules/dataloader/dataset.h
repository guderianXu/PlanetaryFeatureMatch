#pragma once

#include <cstddef>

#include "dataloader/tensor_batch.h"

namespace pfm {

class TensorDataset {
public:
    /// Destroys the dataset.
    virtual ~TensorDataset() = default;

    /// Returns the number of samples.
    /// \return Dataset size.
    virtual size_t size() const = 0;

    /// Loads one sample by index.
    /// \param index Sample index.
    /// \return Tensor batch containing one unbatched sample.
    /// \throws std::out_of_range if index is invalid.
    virtual TensorBatch get(size_t index) = 0;
};

}  // namespace pfm
