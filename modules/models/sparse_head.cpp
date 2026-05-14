#include "models/sparse_head.h"

#include <stdexcept>
#include <string>

namespace pfm {
namespace {

void require_positive_channels(int64_t channels, const char* name) {
    if (channels <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

torch::Tensor normalize_channels(const torch::Tensor& tensor) {
    return tensor / tensor.norm(2, 1, true).clamp_min(1.0e-12);
}

}  // namespace

SparseHeadImpl::SparseHeadImpl(int64_t input_channels, int64_t descriptor_dim)
    : _input_channels(input_channels), _descriptor_dim(descriptor_dim) {
    require_positive_channels(_input_channels, "input_channels");
    require_positive_channels(_descriptor_dim, "descriptor_dim");

    _heatmap = register_module("heatmap", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
    _descriptors = register_module(
        "descriptors", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _descriptor_dim, 1)));
    _scale = register_module("scale", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
    _orientation = register_module("orientation", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 2, 1)));
    _affine = register_module("affine", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 4, 1)));
}

SparseHeadOutput SparseHeadImpl::forward(const torch::Tensor& feature) {
    if (!feature.defined()) {
        throw std::invalid_argument("feature tensor is undefined");
    }
    if (feature.dim() != 4) {
        throw std::invalid_argument("feature tensor must have shape BxCxHxW");
    }
    if (feature.size(1) != _input_channels) {
        throw std::invalid_argument("feature tensor channel count does not match sparse head");
    }

    auto heatmap = torch::sigmoid(_heatmap->forward(feature));
    auto descriptors = normalize_channels(_descriptors->forward(feature));
    auto scale = torch::softplus(_scale->forward(feature)) + 1.0e-3;
    auto orientation = normalize_channels(_orientation->forward(feature));
    auto affine = _affine->forward(feature);
    return SparseHeadOutput{heatmap, descriptors, scale, orientation, affine};
}

}  // namespace pfm
