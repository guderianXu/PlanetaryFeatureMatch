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

torch::Tensor rotate_spatial(const torch::Tensor& tensor, int64_t turns) {
    return torch::rot90(tensor, turns, {2, 3});
}

torch::Tensor c4_cyclic_descriptors(
    torch::nn::Sequential& descriptor_tower,
    const torch::Tensor& context
) {
    std::vector<torch::Tensor> descriptor_views;
    descriptor_views.reserve(4);
    for (int64_t turns = 0; turns < 4; ++turns) {
        auto rotated_context = rotate_spatial(context, turns);
        auto rotated_descriptors = descriptor_tower->forward(rotated_context);
        descriptor_views.push_back(rotate_spatial(rotated_descriptors, -turns));
    }
    const auto view0 = descriptor_views.at(0);
    const auto view1 = descriptor_views.at(1);
    const auto view2 = descriptor_views.at(2);
    const auto view3 = descriptor_views.at(3);
    if (view0.size(1) < 4 || view0.size(1) % 4 != 0) {
        return (view0 + view1 + view2 + view3) * 0.25;
    }
    const auto group_channels = view0.size(1) / 4;
    return torch::cat(
        {
            view0.slice(1, 0, group_channels),
            view1.slice(1, group_channels, group_channels * 2),
            view2.slice(1, group_channels * 2, group_channels * 3),
            view3.slice(1, group_channels * 3, group_channels * 4),
        },
        1);
}

torch::Tensor canonicalize_c4_descriptor_slots(
    const torch::Tensor& descriptors,
    const torch::Tensor& orientation
) {
    if (descriptors.size(1) < 4 || descriptors.size(1) % 4 != 0) {
        return descriptors;
    }

    constexpr int64_t group_count = 4;
    constexpr float orientation_sharpness = 12.0F;
    const auto group_channels = descriptors.size(1) / group_count;
    const auto grouped = descriptors.reshape(
        {descriptors.size(0), group_count, group_channels, descriptors.size(2), descriptors.size(3)});

    std::vector<torch::Tensor> shifted_groups;
    shifted_groups.reserve(group_count);
    for (int64_t shift = 0; shift < group_count; ++shift) {
        shifted_groups.push_back(torch::roll(grouped, {-shift}, {1}));
    }
    const auto shifted = torch::stack(shifted_groups, 1);

    const auto x = orientation.slice(1, 0, 1);
    const auto y = orientation.slice(1, 1, 2);
    const auto logits = torch::cat({x, y, -x, -y}, 1) * orientation_sharpness;
    const auto weights = torch::softmax(logits, 1).unsqueeze(2).unsqueeze(3);
    return (shifted * weights)
        .sum(1)
        .reshape({descriptors.size(0), descriptors.size(1), descriptors.size(2), descriptors.size(3)});
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
    auto scale = torch::softplus(_scale->forward(context)) + 1.0e-3;
    auto orientation = normalize_channels(_orientation->forward(context));
    auto descriptors = normalize_channels(canonicalize_c4_descriptor_slots(
        c4_cyclic_descriptors(_descriptors, context),
        orientation));
    auto affine = _affine->forward(context);
    return SparseHeadOutput{heatmap, descriptors, scale, orientation, affine};
}

}  // namespace pfm
