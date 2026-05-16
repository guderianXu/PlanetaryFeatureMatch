#include <algorithm>
#include <cmath>
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

void validateDecodeConfig(const FeatureDecodeConfig& config) {
    if (config.max_keypoints <= 0) {
        throw std::invalid_argument("max_keypoints must be positive");
    }
    if (config.keypoint_grid_rows <= 0) {
        throw std::invalid_argument("keypoint_grid_rows must be positive");
    }
    if (config.keypoint_grid_cols <= 0) {
        throw std::invalid_argument("keypoint_grid_cols must be positive");
    }
    if (config.keypoints_per_cell < 0) {
        throw std::invalid_argument("keypoints_per_cell must be non-negative");
    }
    if (config.nms_radius < 0) {
        throw std::invalid_argument("nms_radius must be non-negative");
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

struct SparseCandidate {
    int64_t y = 0;
    int64_t x = 0;
    float score = 0.0F;
};

bool isSuppressedBySelected(
    const std::vector<SparseCandidate>& selected,
    const SparseCandidate& candidate,
    int radius
) {
    for (const auto& point : selected) {
        if (std::abs(point.y - candidate.y) <= radius && std::abs(point.x - candidate.x) <= radius) {
            return true;
        }
    }
    return false;
}

std::vector<SparseCandidate> makeNmsCandidates(
    const torch::Tensor& heatmap,
    const torch::Tensor& valid_mask,
    int nms_radius
) {
    std::vector<SparseCandidate> candidates;
    for (int64_t y = 0; y < heatmap.size(2); ++y) {
        for (int64_t x = 0; x < heatmap.size(3); ++x) {
            if (valid_mask.index({y, x}).item<bool>()) {
                candidates.push_back(SparseCandidate{y, x, heatmap.index({0, 0, y, x}).item<float>()});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const SparseCandidate& lhs, const SparseCandidate& rhs) {
        if (lhs.score == rhs.score) {
            if (lhs.y == rhs.y) {
                return lhs.x < rhs.x;
            }
            return lhs.y < rhs.y;
        }
        return lhs.score > rhs.score;
    });

    std::vector<SparseCandidate> selected;
    selected.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (!isSuppressedBySelected(selected, candidate, nms_radius)) {
            selected.push_back(candidate);
        }
    }
    return selected;
}

int resolvedKeypointsPerCell(const FeatureDecodeConfig& config) {
    if (config.keypoints_per_cell > 0) {
        return config.keypoints_per_cell;
    }
    const int cell_count = config.keypoint_grid_rows * config.keypoint_grid_cols;
    return std::max(1, (config.max_keypoints + cell_count - 1) / cell_count);
}

int64_t cellStart(int64_t extent, int cell_index, int cell_count) {
    return extent * cell_index / cell_count;
}

bool candidateInCell(
    const SparseCandidate& candidate,
    int row,
    int col,
    const FeatureDecodeConfig& config,
    int64_t height,
    int64_t width
) {
    const auto y0 = cellStart(height, row, config.keypoint_grid_rows);
    const auto y1 = cellStart(height, row + 1, config.keypoint_grid_rows);
    const auto x0 = cellStart(width, col, config.keypoint_grid_cols);
    const auto x1 = cellStart(width, col + 1, config.keypoint_grid_cols);
    return candidate.y >= y0 && candidate.y < y1 && candidate.x >= x0 && candidate.x < x1;
}

bool sameCandidate(const SparseCandidate& lhs, const SparseCandidate& rhs) {
    return lhs.y == rhs.y && lhs.x == rhs.x;
}

bool containsCandidate(const std::vector<SparseCandidate>& candidates, const SparseCandidate& candidate) {
    return std::any_of(candidates.begin(), candidates.end(), [&](const SparseCandidate& selected) {
        return sameCandidate(selected, candidate);
    });
}

bool hasHigherScoreThanPosition(const SparseCandidate& lhs, const SparseCandidate& rhs) {
    if (lhs.score == rhs.score) {
        if (lhs.y == rhs.y) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    }
    return lhs.score > rhs.score;
}

std::vector<SparseCandidate> selectGridBalancedCandidates(
    const std::vector<SparseCandidate>& candidates,
    const FeatureDecodeConfig& config,
    int64_t height,
    int64_t width
) {
    std::vector<SparseCandidate> grid_candidates;
    const int per_cell = resolvedKeypointsPerCell(config);
    for (int row = 0; row < config.keypoint_grid_rows; ++row) {
        for (int col = 0; col < config.keypoint_grid_cols; ++col) {
            int taken = 0;
            for (const auto& candidate : candidates) {
                if (taken >= per_cell) {
                    break;
                }
                if (candidateInCell(candidate, row, col, config, height, width)) {
                    grid_candidates.push_back(candidate);
                    ++taken;
                }
            }
        }
    }
    std::sort(grid_candidates.begin(), grid_candidates.end(), hasHigherScoreThanPosition);

    std::vector<SparseCandidate> selected;
    selected.reserve(static_cast<size_t>(std::min<int64_t>(config.max_keypoints, candidates.size())));
    for (const auto& candidate : grid_candidates) {
        if (static_cast<int>(selected.size()) >= config.max_keypoints) {
            break;
        }
        selected.push_back(candidate);
    }
    for (const auto& candidate : candidates) {
        if (static_cast<int>(selected.size()) >= config.max_keypoints) {
            break;
        }
        if (!containsCandidate(selected, candidate)) {
            selected.push_back(candidate);
        }
    }
    return selected;
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
    FeatureDecodeConfig config;
    config.max_keypoints = max_keypoints;
    config.semi_dense_threshold = semi_dense_threshold;
    config.keypoint_grid_rows = 1;
    config.keypoint_grid_cols = 1;
    config.keypoints_per_cell = max_keypoints;
    config.nms_radius = 0;
    return decode_feature_maps(maps, config, torch::Tensor());
}

FeatureSet decode_feature_maps(
    const RawFeatureMaps& maps,
    int max_keypoints,
    double semi_dense_threshold,
    const torch::Tensor& intensity_mask
) {
    FeatureDecodeConfig config;
    config.max_keypoints = max_keypoints;
    config.semi_dense_threshold = semi_dense_threshold;
    config.keypoint_grid_rows = 1;
    config.keypoint_grid_cols = 1;
    config.keypoints_per_cell = max_keypoints;
    config.nms_radius = 0;
    return decode_feature_maps(maps, config, intensity_mask);
}

FeatureSet decode_feature_maps(
    const RawFeatureMaps& maps,
    const FeatureDecodeConfig& config,
    const torch::Tensor& intensity_mask
) {
    validateDecodeConfig(config);
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
    const auto nms_candidates = makeNmsCandidates(heatmap, valid_mask, config.nms_radius);
    const auto selected_candidates = selectGridBalancedCandidates(nms_candidates, config, height, width);
    const int64_t sparse_count = static_cast<int64_t>(selected_candidates.size());

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

    std::vector<float> sparse_scores;
    sparse_scores.reserve(static_cast<size_t>(sparse_count));
    for (const auto& candidate : selected_candidates) {
        const int64_t y = candidate.y;
        const int64_t x = candidate.x;
        appendPoint(sparse_points, y, x);
        sparse_scores.push_back(candidate.score);
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
            if (confidence >= config.semi_dense_threshold && dense_valid_mask.index({y, x}).item<bool>()) {
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

    const auto sparse_points_tensor = torch::from_blob(
        sparse_points.data(), {sparse_count, 2}, tensor_options).clone().contiguous();
    const auto sparse_scores_tensor = torch::from_blob(
        sparse_scores.data(), {sparse_count}, tensor_options).clone().contiguous();
    const auto sparse_descriptors_tensor = torch::from_blob(
        sparse_descriptors.data(), {sparse_count, descriptors.size(1)}, tensor_options).clone().contiguous();
    const auto sparse_scale_tensor = torch::from_blob(
        sparse_scale.data(), {sparse_count}, tensor_options).clone().contiguous();
    const auto sparse_orientation_tensor = torch::from_blob(
        sparse_orientation.data(), {sparse_count, orientation.size(1)}, tensor_options).clone().contiguous();
    const auto sparse_affine_tensor = torch::from_blob(
        sparse_affine.data(), {sparse_count, 2, 2}, tensor_options).clone().contiguous();

    return FeatureSet{
        sparse_points_tensor,
        sparse_scores_tensor,
        sparse_descriptors_tensor,
        sparse_scale_tensor,
        sparse_orientation_tensor,
        sparse_affine_tensor,
        dense_points_tensor,
        dense_confidence_tensor};
}

}  // namespace pfm
