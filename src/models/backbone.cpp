#include "models/backbone.h"

#include <stdexcept>

namespace pfm {
namespace {

torch::nn::Sequential make_stage(int64_t input_channels, int64_t output_channels) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, output_channels, 3).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(output_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(output_channels, output_channels, 3).padding(1).bias(false)),
        torch::nn::BatchNorm2d(output_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
}

void require_positive_channels(int64_t channels, const char* name) {
    if (channels <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

}  // namespace

BackboneImpl::BackboneImpl(int64_t input_channels, int64_t base_channels)
    : _input_channels(input_channels), _base_channels(base_channels) {
    require_positive_channels(_input_channels, "input_channels");
    require_positive_channels(_base_channels, "base_channels");

    _stage1 = register_module("stage1", make_stage(_input_channels, _base_channels));
    _stage2 = register_module("stage2", make_stage(_base_channels, _base_channels * 2));
    _stage3 = register_module("stage3", make_stage(_base_channels * 2, _base_channels * 4));
    _stage4 = register_module("stage4", make_stage(_base_channels * 4, _base_channels * 8));
}

std::vector<torch::Tensor> BackboneImpl::forward(const torch::Tensor& x) {
    if (!x.defined()) {
        throw std::invalid_argument("input tensor is undefined");
    }
    if (x.dim() != 4) {
        throw std::invalid_argument("input tensor must have shape BxCxHxW");
    }
    if (x.size(1) != _input_channels) {
        throw std::invalid_argument("input tensor channel count does not match backbone");
    }

    auto y1 = _stage1->forward(x);
    auto y2 = _stage2->forward(y1);
    auto y3 = _stage3->forward(y2);
    auto y4 = _stage4->forward(y3);
    return {y1, y2, y3, y4};
}

void BackboneImpl::sanitize_nonfinite_state() {
    for (auto& item : named_buffers(/*recurse=*/true)) {
        auto tensor = item.value();
        if (!tensor.defined() || !tensor.is_floating_point()) {
            continue;
        }
        auto finite = torch::isfinite(tensor);
        if (finite.all().item<bool>()) {
            continue;
        }
        if (item.key().find("running_var") != std::string::npos) {
            tensor.masked_fill_(finite.logical_not(), 1.0);
        } else {
            tensor.masked_fill_(finite.logical_not(), 0.0);
        }
    }
}

}  // namespace pfm
