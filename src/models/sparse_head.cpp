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

    _context = register_module(
        "context",
        torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _input_channels, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _input_channels, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))));
    _heatmap = register_module("heatmap", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
    _descriptors = register_module(
        "descriptors",
        torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _input_channels, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _descriptor_dim, 1))));
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

    auto context = _context->forward(feature);
    auto heatmap = torch::sigmoid(_heatmap->forward(context));
    auto descriptors = normalize_channels(_descriptors->forward(context));
    auto scale = torch::softplus(_scale->forward(context)) + 1.0e-3;
    auto orientation = normalize_channels(_orientation->forward(context));
    auto affine = _affine->forward(context);
    return SparseHeadOutput{heatmap, descriptors, scale, orientation, affine};
}

}  // namespace pfm
