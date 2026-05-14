#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "infer/feature_extractor.h"

namespace pfm {
namespace {

void validateMap(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    if (tensor.dim() != 4) {
        throw std::invalid_argument(std::string(name) + " must be 4D");
    }
    if (tensor.size(0) != 1) {
        throw std::invalid_argument(std::string(name) + " batch size must be 1");
    }
}

void validateRawMaps(const RawFeatureMaps& maps) {
    validateMap(maps.heatmap, "heatmap");
    validateMap(maps.descriptors, "descriptors");
    validateMap(maps.scale, "scale");
    validateMap(maps.orientation, "orientation");
    validateMap(maps.affine, "affine");
    validateMap(maps.dense_confidence, "dense_confidence");
}

void appendPoint(std::vector<float>& points, int64_t y, int64_t x) {
    points.push_back(static_cast<float>(x));
    points.push_back(static_cast<float>(y));
}

}  // namespace

FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold) {
    if (max_keypoints <= 0) {
        throw std::invalid_argument("max_keypoints must be positive");
    }
    validateRawMaps(maps);

    const auto heatmap = maps.heatmap.to(torch::kFloat32).contiguous();
    const auto descriptors = maps.descriptors.to(torch::kFloat32).contiguous();
    const auto scale = maps.scale.to(torch::kFloat32).contiguous();
    const auto orientation = maps.orientation.to(torch::kFloat32).contiguous();
    const auto affine = maps.affine.to(torch::kFloat32).contiguous();
    const auto dense_confidence_map = maps.dense_confidence.to(torch::kFloat32).contiguous();

    const int64_t width = heatmap.size(3);
    const int64_t sparse_count = std::min<int64_t>(max_keypoints, heatmap.numel());
    const auto topk = torch::topk(heatmap.flatten(), sparse_count);
    const auto topk_values = std::get<0>(topk).contiguous();
    const auto topk_indices = std::get<1>(topk).to(torch::kLong).contiguous();

    std::vector<float> sparse_points;
    std::vector<float> sparse_descriptors;
    std::vector<float> sparse_scale;
    std::vector<float> sparse_orientation;
    std::vector<float> sparse_affine;
    sparse_points.reserve(static_cast<size_t>(sparse_count * 2));
    sparse_descriptors.reserve(static_cast<size_t>(sparse_count * descriptors.size(1)));
    sparse_scale.reserve(static_cast<size_t>(sparse_count));
    sparse_orientation.reserve(static_cast<size_t>(sparse_count));
    sparse_affine.reserve(static_cast<size_t>(sparse_count * 4));

    auto index_accessor = topk_indices.accessor<int64_t, 1>();
    for (int64_t i = 0; i < sparse_count; ++i) {
        const int64_t index = index_accessor[i];
        const int64_t y = index / width;
        const int64_t x = index % width;
        appendPoint(sparse_points, y, x);
        for (int64_t channel = 0; channel < descriptors.size(1); ++channel) {
            sparse_descriptors.push_back(descriptors.index({0, channel, y, x}).item<float>());
        }
        sparse_scale.push_back(scale.index({0, 0, y, x}).item<float>());
        sparse_orientation.push_back(orientation.index({0, 0, y, x}).item<float>());
        for (int64_t channel = 0; channel < affine.size(1); ++channel) {
            sparse_affine.push_back(affine.index({0, channel, y, x}).item<float>());
        }
    }

    std::vector<float> dense_points;
    std::vector<float> dense_confidence;
    const int64_t dense_height = dense_confidence_map.size(2);
    const int64_t dense_width = dense_confidence_map.size(3);
    for (int64_t y = 0; y < dense_height; ++y) {
        for (int64_t x = 0; x < dense_width; ++x) {
            const float confidence = dense_confidence_map.index({0, 0, y, x}).item<float>();
            if (confidence >= semi_dense_threshold) {
                appendPoint(dense_points, y, x);
                dense_confidence.push_back(confidence);
            }
        }
    }

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
    return FeatureSet{
        torch::from_blob(sparse_points.data(), {sparse_count, 2}, tensor_options).clone().contiguous(),
        topk_values.clone().contiguous(),
        torch::from_blob(sparse_descriptors.data(), {sparse_count, descriptors.size(1)}, tensor_options).clone().contiguous(),
        torch::from_blob(sparse_scale.data(), {sparse_count}, tensor_options).clone().contiguous(),
        torch::from_blob(sparse_orientation.data(), {sparse_count}, tensor_options).clone().contiguous(),
        torch::from_blob(sparse_affine.data(), {sparse_count, 2, 2}, tensor_options).clone().contiguous(),
        torch::from_blob(dense_points.data(), {static_cast<int64_t>(dense_confidence.size()), 2}, tensor_options).clone().contiguous(),
        torch::from_blob(dense_confidence.data(), {static_cast<int64_t>(dense_confidence.size())}, tensor_options)
            .clone()
            .contiguous()};
}

}  // namespace pfm
