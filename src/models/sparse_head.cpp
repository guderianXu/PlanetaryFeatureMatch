#include "models/sparse_head.h"

#include <stdexcept>
#include <string>
#include <vector>

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

torch::Tensor rotate_feature_map(const torch::Tensor& tensor, int64_t turns) {
    const auto normalized_turns = ((turns % 4) + 4) % 4;
    if (normalized_turns == 0) {
        return tensor;
    }
    return torch::rot90(tensor, normalized_turns, {2, 3}).contiguous();
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
    _descriptor_skip = register_module(
        "descriptor_skip",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _descriptor_dim, 1)));
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
    std::vector<torch::Tensor> heatmap_logits;
    std::vector<torch::Tensor> descriptor_logits;
    heatmap_logits.reserve(4);
    descriptor_logits.reserve(4);
    heatmap_logits.push_back(_heatmap->forward(context));
    descriptor_logits.push_back(_descriptors->forward(context) + _descriptor_skip->forward(feature));
    for (int64_t turns = 1; turns < 4; ++turns) {
        auto rotated_feature = rotate_feature_map(feature, turns);
        auto rotated_context = _context->forward(rotated_feature);
        heatmap_logits.push_back(rotate_feature_map(_heatmap->forward(rotated_context), -turns));
        descriptor_logits.push_back(rotate_feature_map(
            _descriptors->forward(rotated_context) + _descriptor_skip->forward(rotated_feature),
            -turns));
    }
    auto heatmap = torch::sigmoid(torch::stack(heatmap_logits, 0).mean(0));
    auto descriptors = normalize_channels(torch::stack(descriptor_logits, 0).mean(0));
    auto scale = torch::softplus(_scale->forward(context)) + 1.0e-3;
    auto orientation = normalize_channels(_orientation->forward(context));
    auto affine = _affine->forward(context);
    return SparseHeadOutput{heatmap, descriptors, scale, orientation, affine};
}

}  // namespace pfm
