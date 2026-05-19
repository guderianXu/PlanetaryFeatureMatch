#include "models/dense_head.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor_utils.h"

namespace pfm {
namespace {

constexpr int64_t CORRELATION_RADIUS = 4;
constexpr int64_t CORRELATION_CHANNELS = (CORRELATION_RADIUS * 2 + 1) * (CORRELATION_RADIUS * 2 + 1);

void require_positive_channels(int64_t channels, const char* name) {
    if (channels <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

torch::Tensor shifted_feature(const torch::Tensor& feature, int64_t dy, int64_t dx) {
    using torch::indexing::Slice;

    auto shifted = torch::zeros_like(feature);
    const auto height = feature.size(2);
    const auto width = feature.size(3);
    const auto source_y0 = std::max<int64_t>(0, -dy);
    const auto source_y1 = std::min<int64_t>(height, height - dy);
    const auto source_x0 = std::max<int64_t>(0, -dx);
    const auto source_x1 = std::min<int64_t>(width, width - dx);
    if (source_y0 >= source_y1 || source_x0 >= source_x1) {
        return shifted;
    }
    const auto target_y0 = source_y0 + dy;
    const auto target_y1 = source_y1 + dy;
    const auto target_x0 = source_x0 + dx;
    const auto target_x1 = source_x1 + dx;
    shifted.index_put_(
        {Slice(), Slice(), Slice(target_y0, target_y1), Slice(target_x0, target_x1)},
        feature.index({Slice(), Slice(), Slice(source_y0, source_y1), Slice(source_x0, source_x1)}));
    return shifted;
}

torch::Tensor local_correlation(const torch::Tensor& feature_a, const torch::Tensor& feature_b) {
    std::vector<torch::Tensor> correlations;
    correlations.reserve(CORRELATION_CHANNELS);
    for (int64_t dy = -CORRELATION_RADIUS; dy <= CORRELATION_RADIUS; ++dy) {
        for (int64_t dx = -CORRELATION_RADIUS; dx <= CORRELATION_RADIUS; ++dx) {
            correlations.push_back((feature_a * shifted_feature(feature_b, dy, dx)).mean(1, true));
        }
    }
    return torch::cat(correlations, 1);
}

}  // namespace

DenseHeadImpl::DenseHeadImpl(int64_t feature_channels) : _feature_channels(feature_channels) {
    require_positive_channels(_feature_channels, "feature_channels");

    _correlation_projection = register_module(
        "correlation_projection",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(CORRELATION_CHANNELS, _feature_channels, 1)));
    const int64_t input_channels = _feature_channels * 4 + 2;
    _predictor = register_module(
        "predictor",
        torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, _feature_channels * 2, 3).padding(1)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_feature_channels * 2, _feature_channels * 2, 3).padding(1)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_feature_channels * 2, _feature_channels, 3).padding(1)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true)),
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
    auto correlation = _correlation_projection->forward(local_correlation(feature_a, feature_b));
    auto pair_feature = torch::cat({feature_a, feature_b, torch::abs(feature_a - feature_b), correlation, coordinate_channels}, 1);
    auto prediction = _predictor->forward(pair_feature);

    using torch::indexing::Slice;
    auto confidence = torch::sigmoid(prediction.index({Slice(), Slice(0, 1), Slice(), Slice()}));
    auto offsets = prediction.index({Slice(), Slice(1, 3), Slice(), Slice()});
    return DenseHeadOutput{confidence, offsets};
}

}  // namespace pfm
