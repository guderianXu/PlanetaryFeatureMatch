#include <algorithm>
#include <stdexcept>

#include <torch/torch.h>

#include "dataloader/collator.h"

namespace pfm {
namespace {

std::pair<int64_t, int64_t> spatialSize(const torch::Tensor& tensor, TensorLayout layout) {
    if (layout == TensorLayout::Hw) {
        if (tensor.dim() != 2) {
            throw std::invalid_argument("HW tensor must have rank 2");
        }
        return {tensor.size(0), tensor.size(1)};
    }
    if (layout == TensorLayout::Chw) {
        if (tensor.dim() != 3) {
            throw std::invalid_argument("CHW tensor must have rank 3");
        }
        return {tensor.size(1), tensor.size(2)};
    }
    if (layout == TensorLayout::Hwc) {
        if (tensor.dim() != 3) {
            throw std::invalid_argument("HWC tensor must have rank 3");
        }
        return {tensor.size(0), tensor.size(1)};
    }
    throw std::invalid_argument("unsupported tensor layout");
}

int64_t nonSpatialSize(const torch::Tensor& tensor, TensorLayout layout) {
    if (layout == TensorLayout::Hw) {
        return 0;
    }
    if (layout == TensorLayout::Chw) {
        return tensor.size(0);
    }
    if (layout == TensorLayout::Hwc) {
        return tensor.size(2);
    }
    throw std::invalid_argument("unsupported tensor layout");
}

torch::Tensor padToSpatialSize(const torch::Tensor& tensor, TensorLayout layout, int64_t height, int64_t width) {
    const auto current = spatialSize(tensor, layout);
    const auto pad_h = height - current.first;
    const auto pad_w = width - current.second;
    if (pad_h < 0 || pad_w < 0) {
        throw std::invalid_argument("target spatial size cannot be smaller than tensor size");
    }
    if (layout == TensorLayout::Hwc) {
        return torch::constant_pad_nd(tensor, {0, 0, 0, pad_w, 0, pad_h}, 0);
    }
    return torch::constant_pad_nd(tensor, {0, pad_w, 0, pad_h}, 0);
}

}  // namespace

TensorBatchCollator::TensorBatchCollator(std::vector<std::pair<std::string, TensorLayout>> layouts)
    : _layouts(std::move(layouts)) {
    if (_layouts.empty()) {
        throw std::invalid_argument("tensor batch collator requires at least one layout");
    }
}

TensorBatch TensorBatchCollator::collate(const std::vector<TensorBatch>& samples) const {
    if (samples.empty()) {
        throw std::invalid_argument("cannot collate an empty sample list");
    }

    TensorBatch result;
    for (const auto& [key, layout] : _layouts) {
        int64_t height = 0;
        int64_t width = 0;
        int64_t expected_non_spatial_size = -1;
        for (const auto& sample : samples) {
            const auto it = sample.find(key);
            if (it == sample.end()) {
                throw std::invalid_argument("tensor batch sample is missing required key: " + key);
            }
            const auto size = spatialSize(it->second, layout);
            const auto current_non_spatial_size = nonSpatialSize(it->second, layout);
            if (expected_non_spatial_size < 0) {
                expected_non_spatial_size = current_non_spatial_size;
            } else if (current_non_spatial_size != expected_non_spatial_size) {
                throw std::invalid_argument("tensor batch sample has mismatched non-spatial size for key: " + key);
            }
            height = std::max(height, size.first);
            width = std::max(width, size.second);
        }

        std::vector<torch::Tensor> padded;
        padded.reserve(samples.size());
        for (const auto& sample : samples) {
            padded.push_back(padToSpatialSize(sample.at(key), layout, height, width));
        }
        result[key] = torch::stack(padded, 0).contiguous();
    }
    return result;
}

}  // namespace pfm
