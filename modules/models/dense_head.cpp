#include "models/dense_head.h"

#include <stdexcept>
#include <string>

namespace pfm {
namespace {

void require_positive_channels(int64_t channels, const char* name) {
    if (channels <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

}  // namespace

DenseHeadImpl::DenseHeadImpl(int64_t feature_channels) : _feature_channels(feature_channels) {
    require_positive_channels(_feature_channels, "feature_channels");

    const int64_t input_channels = _feature_channels * 3;
    _predictor = register_module(
        "predictor",
        torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, _feature_channels, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_feature_channels, 3, 1))));
}

DenseHeadOutput DenseHeadImpl::forward(const torch::Tensor& feature_a, const torch::Tensor& feature_b) {
    if (!feature_a.defined() || !feature_b.defined()) {
        throw std::invalid_argument("feature tensors must be defined");
    }
    if (feature_a.dim() != 4 || feature_b.dim() != 4) {
        throw std::invalid_argument("feature tensors must have shape BxCxHxW");
    }
    if (feature_a.sizes() != feature_b.sizes()) {
        throw std::invalid_argument("feature tensors must have matching shapes");
    }
    if (feature_a.size(1) != _feature_channels) {
        throw std::invalid_argument("feature tensor channel count does not match dense head");
    }

    auto pair_feature = torch::cat({feature_a, feature_b, torch::abs(feature_a - feature_b)}, 1);
    auto prediction = _predictor->forward(pair_feature);

    using torch::indexing::Slice;
    auto confidence = torch::sigmoid(prediction.index({Slice(), Slice(0, 1), Slice(), Slice()}));
    auto offsets = prediction.index({Slice(), Slice(1, 3), Slice(), Slice()});
    return DenseHeadOutput{confidence, offsets};
}

}  // namespace pfm
