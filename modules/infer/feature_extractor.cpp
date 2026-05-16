#include <algorithm>
#include <limits>
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
    if (!tensor.device().is_cpu()) {
        throw std::invalid_argument(std::string(name) + " must be a CPU tensor");
    }
    if (tensor.dim() != 4) {
        throw std::invalid_argument(std::string(name) + " must be 4D");
    }
    if (tensor.size(0) != 1) {
        throw std::invalid_argument(std::string(name) + " batch size must be 1");
    }
}

void validateSpatialSize(const torch::Tensor& tensor, const char* name, int64_t height, int64_t width) {
    if (tensor.size(2) != height || tensor.size(3) != width) {
        throw std::invalid_argument(std::string(name) + " spatial size must match heatmap");
    }
}

void validateRawMaps(const RawFeatureMaps& maps) {
    validateMap(maps.heatmap, "heatmap");
    validateMap(maps.descriptors, "descriptors");
    validateMap(maps.scale, "scale");
    validateMap(maps.orientation, "orientation");
    validateMap(maps.affine, "affine");
    validateMap(maps.dense_confidence, "dense_confidence");

    const int64_t height = maps.heatmap.size(2);
    const int64_t width = maps.heatmap.size(3);
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("heatmap height and width must be positive");
    }
    if (maps.heatmap.size(1) != 1) {
        throw std::invalid_argument("heatmap channels must be 1");
    }
    if (maps.descriptors.size(1) <= 0) {
        throw std::invalid_argument("descriptors channels must be positive");
    }
    if (maps.scale.size(1) < 1) {
        throw std::invalid_argument("scale channels must be at least 1");
    }
    if (maps.orientation.size(1) != 2) {
        throw std::invalid_argument("orientation channels must be 2");
    }
    if (maps.affine.size(1) != 4) {
        throw std::invalid_argument("affine channels must be 4");
    }
    if (maps.dense_confidence.size(1) != 1) {
        throw std::invalid_argument("dense_confidence channels must be 1");
    }

    validateSpatialSize(maps.descriptors, "descriptors", height, width);
    validateSpatialSize(maps.scale, "scale", height, width);
    validateSpatialSize(maps.orientation, "orientation", height, width);
    validateSpatialSize(maps.affine, "affine", height, width);
    validateSpatialSize(maps.dense_confidence, "dense_confidence", height, width);
}

void appendPoint(std::vector<float>& points, int64_t y, int64_t x) {
    points.push_back(static_cast<float>(x));
    points.push_back(static_cast<float>(y));
}

torch::Tensor prepare_decode_mask(const torch::Tensor& mask, int64_t height, int64_t width) {
    if (!mask.defined()) {
        return torch::ones({height, width}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    }
    if (!mask.device().is_cpu()) {
        throw std::invalid_argument("intensity_mask must be a CPU tensor");
    }
    if (mask.dim() != 2) {
        throw std::invalid_argument("intensity_mask must be 2D");
    }
    auto float_mask = mask.to(torch::kFloat32).unsqueeze(0).unsqueeze(0);
    if (mask.size(0) != height || mask.size(1) != width) {
        float_mask = torch::nn::functional::interpolate(
            float_mask,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{height, width})
                .mode(torch::kNearest));
    }
    return float_mask.squeeze().gt(0.0).contiguous();
}

}  // namespace

FeatureSet decode_feature_maps(const RawFeatureMaps& maps, int max_keypoints, double semi_dense_threshold) {
    return decode_feature_maps(maps, max_keypoints, semi_dense_threshold, torch::Tensor());
}

FeatureSet decode_feature_maps(
    const RawFeatureMaps& maps,
    int max_keypoints,
    double semi_dense_threshold,
    const torch::Tensor& intensity_mask
) {
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

    const int64_t height = heatmap.size(2);
    const int64_t width = heatmap.size(3);
    const auto valid_mask = prepare_decode_mask(intensity_mask, height, width);
    const auto valid_flat = valid_mask.flatten();
    const int64_t valid_count = valid_flat.to(torch::kLong).sum().item<int64_t>();
    const int64_t sparse_count = std::min<int64_t>(max_keypoints, valid_count);
    const auto masked_heatmap = heatmap.flatten().masked_fill(valid_flat.logical_not(), -std::numeric_limits<float>::infinity());
    const auto topk = sparse_count == 0
        ? std::make_tuple(torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32)),
                          torch::empty({0}, torch::TensorOptions().dtype(torch::kLong)))
        : torch::topk(masked_heatmap, sparse_count);
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
    sparse_orientation.reserve(static_cast<size_t>(sparse_count * 2));
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
        for (int64_t channel = 0; channel < orientation.size(1); ++channel) {
            sparse_orientation.push_back(orientation.index({0, channel, y, x}).item<float>());
        }
        for (int64_t channel = 0; channel < affine.size(1); ++channel) {
            sparse_affine.push_back(affine.index({0, channel, y, x}).item<float>());
        }
    }

    std::vector<float> dense_points;
    std::vector<float> dense_confidence;
    const int64_t dense_height = dense_confidence_map.size(2);
    const int64_t dense_width = dense_confidence_map.size(3);
    const auto dense_valid_mask = prepare_decode_mask(intensity_mask, dense_height, dense_width);
    for (int64_t y = 0; y < dense_height; ++y) {
        for (int64_t x = 0; x < dense_width; ++x) {
            const float confidence = dense_confidence_map.index({0, 0, y, x}).item<float>();
            if (confidence >= semi_dense_threshold && dense_valid_mask.index({y, x}).item<bool>()) {
                appendPoint(dense_points, y, x);
                dense_confidence.push_back(confidence);
            }
        }
    }

    const auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const int64_t dense_count = static_cast<int64_t>(dense_confidence.size());
    const auto dense_points_tensor = dense_count == 0
        ? torch::empty({0, 2}, tensor_options)
        : torch::from_blob(dense_points.data(), {dense_count, 2}, tensor_options).clone().contiguous();
    const auto dense_confidence_tensor = dense_count == 0
        ? torch::empty({0}, tensor_options)
        : torch::from_blob(dense_confidence.data(), {dense_count}, tensor_options).clone().contiguous();

    return FeatureSet{
        torch::from_blob(sparse_points.data(), {sparse_count, 2}, tensor_options).clone().contiguous(),
        topk_values.to(torch::kCPU, torch::kFloat32).contiguous(),
        torch::from_blob(sparse_descriptors.data(), {sparse_count, descriptors.size(1)}, tensor_options).clone().contiguous(),
        torch::from_blob(sparse_scale.data(), {sparse_count}, tensor_options).clone().contiguous(),
        torch::from_blob(sparse_orientation.data(), {sparse_count, orientation.size(1)}, tensor_options).clone().contiguous(),
        torch::from_blob(sparse_affine.data(), {sparse_count, 2, 2}, tensor_options).clone().contiguous(),
        dense_points_tensor,
        dense_confidence_tensor};
}

}  // namespace pfm
