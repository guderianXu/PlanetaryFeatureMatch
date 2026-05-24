#include <algorithm>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "infer/feature_extractor.h"

namespace pfm {
namespace {

constexpr float DESCRIPTOR_POOL_CENTER_WEIGHT = 8.0F;
constexpr float DESCRIPTOR_POOL_AXIS_WEIGHT = 1.0F;
constexpr float DESCRIPTOR_POOL_DIAGONAL_WEIGHT = 0.5F;
constexpr double PI = 3.14159265358979323846;

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
    if (config.min_keypoints < 0) {
        throw std::invalid_argument("min_keypoints must be non-negative");
    }
    if (config.min_keypoints > config.max_keypoints) {
        throw std::invalid_argument("min_keypoints must not exceed max_keypoints");
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
    if (config.descriptor_pool_radius < 0) {
        throw std::invalid_argument("descriptor_pool_radius must be non-negative");
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
    const int64_t height = heatmap.size(2);
    const int64_t width = heatmap.size(3);
    const auto* heatmap_data = heatmap.data_ptr<float>();
    const auto* mask_data = valid_mask.data_ptr<bool>();
    candidates.reserve(static_cast<std::size_t>(height * width));
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            const auto offset = y * width + x;
            if (mask_data[offset]) {
                candidates.push_back(SparseCandidate{y, x, heatmap_data[offset]});
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

void appendCandidateUntilLimit(
    std::vector<SparseCandidate>& selected,
    const SparseCandidate& candidate,
    int max_keypoints
) {
    if (static_cast<int>(selected.size()) < max_keypoints && !containsCandidate(selected, candidate)) {
        selected.push_back(candidate);
    }
}

std::vector<SparseCandidate> selectGridBalancedCandidates(
    const std::vector<SparseCandidate>& candidates,
    const std::vector<SparseCandidate>& unfiltered_candidates,
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
        appendCandidateUntilLimit(selected, candidate, config.max_keypoints);
    }
    for (const auto& candidate : unfiltered_candidates) {
        if (static_cast<int>(selected.size()) >= config.min_keypoints ||
            static_cast<int>(selected.size()) >= config.max_keypoints) {
            break;
        }
        appendCandidateUntilLimit(selected, candidate, config.max_keypoints);
    }
    return selected;
}

void appendRawDescriptor(
    std::vector<float>& output,
    const float* descriptor_data,
    int64_t descriptor_channels,
    int64_t height,
    int64_t width,
    int64_t y,
    int64_t x
) {
    const auto spatial_offset = y * width + x;
    for (int64_t channel = 0; channel < descriptor_channels; ++channel) {
        output.push_back(descriptor_data[channel * height * width + spatial_offset]);
    }
}

void addBilinearDescriptorSample(
    std::vector<float>& accumulator,
    float& weight_sum,
    const float* descriptor_data,
    const bool* mask_data,
    int64_t descriptor_channels,
    int64_t height,
    int64_t width,
    float sample_y,
    float sample_x,
    float sample_weight
) {
    if (!std::isfinite(sample_y) || !std::isfinite(sample_x) || sample_weight <= 0.0F ||
        sample_y < 0.0F || sample_x < 0.0F ||
        sample_y > static_cast<float>(height - 1) || sample_x > static_cast<float>(width - 1)) {
        return;
    }

    const auto y0 = static_cast<int64_t>(std::floor(sample_y));
    const auto x0 = static_cast<int64_t>(std::floor(sample_x));
    const auto y1 = std::min<int64_t>(height - 1, y0 + 1);
    const auto x1 = std::min<int64_t>(width - 1, x0 + 1);
    const auto dy = sample_y - static_cast<float>(y0);
    const auto dx = sample_x - static_cast<float>(x0);
    const std::vector<std::pair<int64_t, float>> y_weights =
        y0 == y1
            ? std::vector<std::pair<int64_t, float>>{{y0, 1.0F}}
            : std::vector<std::pair<int64_t, float>>{{y0, 1.0F - dy}, {y1, dy}};
    const std::vector<std::pair<int64_t, float>> x_weights =
        x0 == x1
            ? std::vector<std::pair<int64_t, float>>{{x0, 1.0F}}
            : std::vector<std::pair<int64_t, float>>{{x0, 1.0F - dx}, {x1, dx}};

    for (const auto& [yy, yw] : y_weights) {
        for (const auto& [xx, xw] : x_weights) {
            const auto spatial_offset = yy * width + xx;
            if (!mask_data[spatial_offset]) {
                continue;
            }
            const auto weight = sample_weight * yw * xw;
            if (weight <= 0.0F) {
                continue;
            }
            for (int64_t channel = 0; channel < descriptor_channels; ++channel) {
                accumulator[static_cast<std::size_t>(channel)] +=
                    descriptor_data[channel * height * width + spatial_offset] * weight;
            }
            weight_sum += weight;
        }
    }
}

void appendOrientationPooledDescriptor(
    std::vector<float>& output,
    const float* descriptor_data,
    const float* orientation_data,
    const bool* mask_data,
    int64_t descriptor_channels,
    int64_t height,
    int64_t width,
    int64_t y,
    int64_t x,
    int descriptor_pool_radius
) {
    if (descriptor_pool_radius <= 0) {
        appendRawDescriptor(output, descriptor_data, descriptor_channels, height, width, y, x);
        return;
    }

    std::vector<float> accumulator(static_cast<std::size_t>(descriptor_channels), 0.0F);
    float weight_sum = 0.0F;
    const auto spatial_offset = y * width + x;
    addBilinearDescriptorSample(
        accumulator,
        weight_sum,
        descriptor_data,
        mask_data,
        descriptor_channels,
        height,
        width,
        static_cast<float>(y),
        static_cast<float>(x),
        DESCRIPTOR_POOL_CENTER_WEIGHT);

    float axis_x = orientation_data[spatial_offset];
    float axis_y = orientation_data[height * width + spatial_offset];
    const auto axis_norm = std::sqrt(axis_x * axis_x + axis_y * axis_y);
    if (!std::isfinite(axis_norm) || axis_norm <= 1.0e-6F) {
        axis_x = 1.0F;
        axis_y = 0.0F;
    } else {
        axis_x /= axis_norm;
        axis_y /= axis_norm;
    }
    const auto ortho_x = -axis_y;
    const auto ortho_y = axis_x;
    for (int radius = 1; radius <= descriptor_pool_radius; ++radius) {
        const auto step = static_cast<float>(radius);
        const auto axis_weight = DESCRIPTOR_POOL_AXIS_WEIGHT / step;
        addBilinearDescriptorSample(
            accumulator,
            weight_sum,
            descriptor_data,
            mask_data,
            descriptor_channels,
            height,
            width,
            static_cast<float>(y) + axis_y * step,
            static_cast<float>(x) + axis_x * step,
            axis_weight);
        addBilinearDescriptorSample(
            accumulator,
            weight_sum,
            descriptor_data,
            mask_data,
            descriptor_channels,
            height,
            width,
            static_cast<float>(y) - axis_y * step,
            static_cast<float>(x) - axis_x * step,
            axis_weight);
        addBilinearDescriptorSample(
            accumulator,
            weight_sum,
            descriptor_data,
            mask_data,
            descriptor_channels,
            height,
            width,
            static_cast<float>(y) + ortho_y * step,
            static_cast<float>(x) + ortho_x * step,
            axis_weight);
        addBilinearDescriptorSample(
            accumulator,
            weight_sum,
            descriptor_data,
            mask_data,
            descriptor_channels,
            height,
            width,
            static_cast<float>(y) - ortho_y * step,
            static_cast<float>(x) - ortho_x * step,
            axis_weight);
        if (descriptor_pool_radius > 1) {
            const auto diagonal_weight = DESCRIPTOR_POOL_DIAGONAL_WEIGHT / step;
            addBilinearDescriptorSample(
                accumulator,
                weight_sum,
                descriptor_data,
                mask_data,
                descriptor_channels,
                height,
                width,
                static_cast<float>(y) + (axis_y + ortho_y) * step,
                static_cast<float>(x) + (axis_x + ortho_x) * step,
                diagonal_weight);
            addBilinearDescriptorSample(
                accumulator,
                weight_sum,
                descriptor_data,
                mask_data,
                descriptor_channels,
                height,
                width,
                static_cast<float>(y) + (axis_y - ortho_y) * step,
                static_cast<float>(x) + (axis_x - ortho_x) * step,
                diagonal_weight);
            addBilinearDescriptorSample(
                accumulator,
                weight_sum,
                descriptor_data,
                mask_data,
                descriptor_channels,
                height,
                width,
                static_cast<float>(y) + (-axis_y + ortho_y) * step,
                static_cast<float>(x) + (-axis_x + ortho_x) * step,
                diagonal_weight);
            addBilinearDescriptorSample(
                accumulator,
                weight_sum,
                descriptor_data,
                mask_data,
                descriptor_channels,
                height,
                width,
                static_cast<float>(y) - (axis_y + ortho_y) * step,
                static_cast<float>(x) - (axis_x + ortho_x) * step,
                diagonal_weight);
        }
    }

    if (weight_sum <= 0.0F) {
        appendRawDescriptor(output, descriptor_data, descriptor_channels, height, width, y, x);
        return;
    }
    float descriptor_norm = 0.0F;
    for (const auto value : accumulator) {
        descriptor_norm += value * value;
    }
    descriptor_norm = std::sqrt(descriptor_norm);
    if (!std::isfinite(descriptor_norm) || descriptor_norm <= 1.0e-12F) {
        appendRawDescriptor(output, descriptor_data, descriptor_channels, height, width, y, x);
        return;
    }
    for (auto value : accumulator) {
        output.push_back(value / descriptor_norm);
    }
}

int64_t predictedOrientationQuarterTurn(
    const float* orientation_data,
    int64_t height,
    int64_t width,
    int64_t y,
    int64_t x
) {
    const auto spatial_offset = y * width + x;
    const auto axis_x = orientation_data[spatial_offset];
    const auto axis_y = orientation_data[height * width + spatial_offset];
    const auto axis_norm = std::sqrt(axis_x * axis_x + axis_y * axis_y);
    if (!std::isfinite(axis_norm) || axis_norm <= 1.0e-6F) {
        return 0;
    }
    const auto angle = std::atan2(static_cast<double>(axis_y), static_cast<double>(axis_x));
    auto turns = static_cast<int64_t>(std::llround(angle / (PI * 0.5)));
    turns %= 4;
    if (turns < 0) {
        turns += 4;
    }
    return turns;
}

void canonicalizeLastDescriptorOrientation(
    std::vector<float>& descriptors,
    int64_t descriptor_channels,
    int64_t quarter_turns
) {
    if (quarter_turns == 0 || descriptor_channels < 4 || descriptor_channels % 4 != 0 ||
        static_cast<int64_t>(descriptors.size()) < descriptor_channels) {
        return;
    }
    const auto group_channels = descriptor_channels / 4;
    const auto left_shift = (quarter_turns * group_channels) % descriptor_channels;
    const auto begin = descriptors.size() - static_cast<std::size_t>(descriptor_channels);
    std::vector<float> original(
        descriptors.begin() + static_cast<std::ptrdiff_t>(begin),
        descriptors.end());
    for (int64_t channel = 0; channel < descriptor_channels; ++channel) {
        descriptors[begin + static_cast<std::size_t>(channel)] =
            original[static_cast<std::size_t>((channel + left_shift) % descriptor_channels)];
    }
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
    return float_mask.reshape({height, width}).gt(0.0).contiguous();
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
    const auto unfiltered_candidates = makeNmsCandidates(heatmap, valid_mask, 0);
    const auto nms_candidates = makeNmsCandidates(heatmap, valid_mask, config.nms_radius);
    const auto selected_candidates = selectGridBalancedCandidates(nms_candidates, unfiltered_candidates, config, height, width);
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
    const auto descriptor_channels = descriptors.size(1);
    const auto* descriptor_data = descriptors.data_ptr<float>();
    const auto* scale_data = scale.data_ptr<float>();
    const auto* orientation_data = orientation.data_ptr<float>();
    const auto* affine_data = affine.data_ptr<float>();
    const auto* valid_mask_data = valid_mask.data_ptr<bool>();
    for (const auto& candidate : selected_candidates) {
        const int64_t y = candidate.y;
        const int64_t x = candidate.x;
        appendPoint(sparse_points, y, x);
        sparse_scores.push_back(candidate.score);
        const auto spatial_offset = y * width + x;
        appendOrientationPooledDescriptor(
            sparse_descriptors,
            descriptor_data,
            orientation_data,
            valid_mask_data,
            descriptor_channels,
            height,
            width,
            y,
            x,
            config.descriptor_pool_radius);
        if (config.descriptor_orientation_canonicalization) {
            canonicalizeLastDescriptorOrientation(
                sparse_descriptors,
                descriptor_channels,
                predictedOrientationQuarterTurn(orientation_data, height, width, y, x));
        }
        sparse_scale.push_back(scale_data[spatial_offset]);
        for (int64_t channel = 0; channel < orientation.size(1); ++channel) {
            sparse_orientation.push_back(orientation_data[channel * height * width + spatial_offset]);
        }
        for (int64_t channel = 0; channel < affine.size(1); ++channel) {
            sparse_affine.push_back(affine_data[channel * height * width + spatial_offset]);
        }
    }

    std::vector<float> dense_points;
    std::vector<float> dense_confidence;
    const int64_t dense_height = dense_confidence_map.size(2);
    const int64_t dense_width = dense_confidence_map.size(3);
    const auto dense_valid_mask = prepare_decode_mask(intensity_mask, dense_height, dense_width);
    const auto* dense_confidence_data = dense_confidence_map.data_ptr<float>();
    const auto* dense_mask_data = dense_valid_mask.data_ptr<bool>();
    for (int64_t y = 0; y < dense_height; ++y) {
        for (int64_t x = 0; x < dense_width; ++x) {
            const auto offset = y * dense_width + x;
            const float confidence = dense_confidence_data[offset];
            if (confidence >= config.semi_dense_threshold && dense_mask_data[offset]) {
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
