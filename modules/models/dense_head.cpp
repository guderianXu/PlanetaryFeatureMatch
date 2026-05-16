#include "models/dense_head.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor_utils.h"

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

    const int64_t input_channels = _feature_channels * 3 + 2;
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

    auto coordinates = make_xy_grid(feature_a.size(2), feature_a.size(3), feature_a.device()).to(feature_a.dtype());
    coordinates.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0},
                           coordinates.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) /
                                   std::max<int64_t>(1, feature_a.size(3) - 1) * 2.0 -
                               1.0);
    coordinates.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1},
                           coordinates.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) /
                                   std::max<int64_t>(1, feature_a.size(2) - 1) * 2.0 -
                               1.0);
    auto coordinate_channels = coordinates.permute({2, 0, 1}).unsqueeze(0).expand({feature_a.size(0), 2, feature_a.size(2), feature_a.size(3)});
    auto pair_feature = torch::cat({feature_a, feature_b, torch::abs(feature_a - feature_b), coordinate_channels}, 1);
    auto prediction = _predictor->forward(pair_feature);

    using torch::indexing::Slice;
    auto confidence = torch::sigmoid(prediction.index({Slice(), Slice(0, 1), Slice(), Slice()}));
    auto offsets = prediction.index({Slice(), Slice(1, 3), Slice(), Slice()});
    return DenseHeadOutput{confidence, offsets};
}

}  // namespace pfm
