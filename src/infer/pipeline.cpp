#include "infer/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/nn/functional/upsampling.h>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "core/device.h"
#include "core/timer.h"
#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"
#include "image/image_io.h"
#include "image/intensity_mask.h"
#include "infer/eval_pipeline.h"
#include "infer/feature_extractor.h"
#include "infer/match_metrics.h"
#include "infer/matching_pipeline.h"
#include "infer/visualization.h"
#include "models/head_outputs.h"
#include "models/pfm_model_v21.h"
#include "models/planetary_graph_matcher.h"
#include "train/trainer.h"

namespace pfm
{
namespace
{

// 推理阶段的阈值按“先快后稳”组织：常规输出不足时才进入更贵的高密度解码或纹理融合重跑。
constexpr int64_t SPARSE_FEATURE_CHANNEL_MULTIPLIER = 2;
constexpr double ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT = 1.0;
constexpr int64_t DESCRIPTOR_GRID_FALLBACK_MIN_SPARSE_MATCHES = 16;
constexpr int64_t DESCRIPTOR_GRID_FALLBACK_MAX_BASE_MATCHES = 7;
constexpr int64_t ADAPTIVE_HIGH_DENSITY_MIN_KEYPOINTS = 1500;
constexpr int64_t ADAPTIVE_HIGH_DENSITY_LOW_BASE_MAX_MATCHES = 16;
constexpr int64_t ADAPTIVE_HIGH_DENSITY_LOW_BASE_MIN_MATCHES = 100;
constexpr int64_t ADAPTIVE_HIGH_DENSITY_LOW_BASE_MODERATE_MAX_BASE_MATCHES = 12;
constexpr int64_t ADAPTIVE_HIGH_DENSITY_LOW_BASE_MODERATE_MIN_MATCHES = 32;
constexpr int64_t ADAPTIVE_HIGH_DENSITY_LOW_BASE_MODERATE_MAX_GAIN = 5;
constexpr int64_t ADAPTIVE_HIGH_DENSITY_STRONG_BASE_MIN_MATCHES = 200;
constexpr int64_t ADAPTIVE_HIGH_DENSITY_STRONG_MIN_GAIN = 20;
constexpr int64_t ROTATION_ONLY_FAST_PATH_MIN_SPARSE_MATCHES = 32;
constexpr double ALTERNATE_TEXTURE_BLEND_WEIGHT = 0.0;
constexpr int64_t ALTERNATE_TEXTURE_BLEND_MIN_SPARSE_MATCHES = 30;
constexpr int64_t ALTERNATE_TEXTURE_BLEND_MIN_GAIN_NUMERATOR = 2;
constexpr double BALANCED_TEXTURE_BLEND_WEIGHT = 1.0;
constexpr int64_t BALANCED_TEXTURE_BLEND_MIN_BASE_MATCHES = 30;
constexpr int64_t BALANCED_TEXTURE_BLEND_MAX_GAIN = 5;
constexpr double PI = 3.14159265358979323846;

bool require_path(const std::string& value, const char* option_name)
{
    if (!value.empty())
    {
        return true;
    }

    std::cerr << "missing required option " << option_name << '\n';
    return false;
}

int64_t descriptorOrientationQuarterTurn(const float* orientation_data, int64_t spatial_count, int64_t index)
{
    const auto axis_x = orientation_data[index];
    const auto axis_y = orientation_data[spatial_count + index];
    const auto axis_norm = std::sqrt(axis_x * axis_x + axis_y * axis_y);
    if (!std::isfinite(axis_norm) || axis_norm <= 1.0e-6F)
    {
        return 0;
    }
    const auto angle = std::atan2(static_cast<double>(axis_y), static_cast<double>(axis_x));
    auto turns = static_cast<int64_t>(std::llround(angle / (PI * 0.5)));
    turns %= 4;
    if (turns < 0)
    {
        turns += 4;
    }
    return turns;
}

void canonicalizeDescriptorRowsByOrientation(torch::Tensor& descriptors, const torch::Tensor& orientation,
                                             const std::vector<int64_t>& spatial_indices, int64_t spatial_count)
{
    if (!descriptors.defined() || descriptors.size(1) < 4 || descriptors.size(1) % 4 != 0 || !orientation.defined() ||
        orientation.size(1) != 2 || static_cast<int64_t>(spatial_indices.size()) != descriptors.size(0))
    {
        return;
    }
    auto descriptor_rows = descriptors.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto orientation_cpu = orientation.to(torch::kCPU, torch::kFloat32).contiguous();
    auto* descriptor_data = descriptor_rows.data_ptr<float>();
    const auto* orientation_data = orientation_cpu.data_ptr<float>();
    const auto descriptor_channels = descriptor_rows.size(1);
    const auto group_channels = descriptor_channels / 4;
    std::vector<float> row(static_cast<std::size_t>(descriptor_channels), 0.0F);
    for (int64_t row_index = 0; row_index < descriptor_rows.size(0); ++row_index)
    {
        const auto turns = descriptorOrientationQuarterTurn(orientation_data, spatial_count,
                                                            spatial_indices[static_cast<std::size_t>(row_index)]);
        if (turns == 0)
        {
            continue;
        }
        const auto left_shift = (turns * group_channels) % descriptor_channels;
        auto* descriptor_row = descriptor_data + row_index * descriptor_channels;
        std::copy(descriptor_row, descriptor_row + descriptor_channels, row.begin());
        for (int64_t channel = 0; channel < descriptor_channels; ++channel)
        {
            descriptor_row[channel] = row[static_cast<std::size_t>((channel + left_shift) % descriptor_channels)];
        }
    }
    descriptors = descriptor_rows;
}

double rotationInvariantTextureBlendWeight()
{
    const char* value = std::getenv("PFM_TEXTURE_BLEND_WEIGHT");
    if (value == nullptr)
    {
        return ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT;
    }
    try
    {
        const auto parsed = std::stod(value);
        if (std::isfinite(parsed) && parsed >= 0.0)
        {
            return parsed;
        }
    }
    catch (const std::exception&)
    {
    }
    return ROTATION_INVARIANT_TEXTURE_BLEND_WEIGHT;
}

bool sparseGeometryFilterRotationOnlyRequested()
{
    const char* value = std::getenv("PFM_SPARSE_GEOMETRY_FILTER");
    if (value == nullptr)
    {
        return false;
    }
    const std::string mode(value);
    return mode == "rotation" || mode == "rotation-only";
}

bool shouldSkipExpensiveSparseAlternates(int64_t sparse_matches)
{
    return sparseGeometryFilterRotationOnlyRequested() && sparse_matches >= ROTATION_ONLY_FAST_PATH_MIN_SPARSE_MATCHES;
}

int64_t read_config_value(torch::serialize::InputArchive& config_archive, const char* name)
{
    torch::Tensor tensor;
    config_archive.read(name, tensor);
    if (!tensor.defined() || tensor.numel() != 1)
    {
        throw std::invalid_argument(std::string("checkpoint config missing ") + name);
    }
    return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

int64_t read_optional_config_value(torch::serialize::InputArchive& config_archive, const char* name, int64_t fallback)
{
    try
    {
        return read_config_value(config_archive, name);
    }
    catch (const c10::Error&)
    {
        return fallback;
    }
}

struct CheckpointConfig
{
    int64_t input_channels = 1;
    int64_t base_channels = 8;
    int64_t descriptor_dim = 32;
    int64_t graph_hidden_dim = 32;
    int64_t graph_attention_layers = 1;
    int64_t graph_keypoint_meta_dim = 16;
};

class ScopedEnvironmentOverride
{
  public:
    ScopedEnvironmentOverride(std::string name, const std::string& value)
        : _name(std::move(name)), _active(!value.empty())
    {
        if (!_active)
        {
            return;
        }
        const char* old_value = std::getenv(_name.c_str());
        if (old_value != nullptr)
        {
            _had_old_value = true;
            _old_value = old_value;
        }
        setenv(_name.c_str(), value.c_str(), 1);
    }

    ScopedEnvironmentOverride(const ScopedEnvironmentOverride&) = delete;
    ScopedEnvironmentOverride& operator=(const ScopedEnvironmentOverride&) = delete;

    ~ScopedEnvironmentOverride()
    {
        if (!_active)
        {
            return;
        }
        if (_had_old_value)
        {
            setenv(_name.c_str(), _old_value.c_str(), 1);
        }
        else
        {
            unsetenv(_name.c_str());
        }
    }

  private:
    std::string _name;
    bool _active = false;
    bool _had_old_value = false;
    std::string _old_value;
};

CheckpointConfig load_checkpoint_config(const std::string& checkpoint)
{
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive config_archive;
    archive.read("config", config_archive);

    CheckpointConfig config;
    config.input_channels = read_config_value(config_archive, "input_channels");
    config.base_channels = read_config_value(config_archive, "base_channels");
    config.descriptor_dim = read_config_value(config_archive, "descriptor_dim");
    config.graph_hidden_dim =
        read_optional_config_value(config_archive, "graph_hidden_dim", std::max<int64_t>(32, config.descriptor_dim));
    config.graph_attention_layers = read_optional_config_value(config_archive, "graph_attention_layers", 1);
    config.graph_keypoint_meta_dim = read_optional_config_value(config_archive, "graph_keypoint_meta_dim", 16);
    return config;
}

torch::Tensor adapt_image_channels(const torch::Tensor& image, int64_t input_channels)
{
    if (image.size(0) == input_channels)
    {
        return image;
    }
    if (input_channels == 1)
    {
        return image.mean(0, true).contiguous();
    }
    throw std::invalid_argument("image channel count does not match checkpoint input_channels");
}

torch::Tensor make_rotation_invariant_texture_saliency(const torch::Tensor& image, int64_t target_height,
                                                       int64_t target_width)
{
    // 用低频差分和一阶梯度构造旋转不敏感纹理热力图，作为推理阶段已学习热力图的稳定补充。
    auto base = image;
    if (base.size(1) != 1)
    {
        base = base.mean(1, true);
    }
    auto blur = torch::avg_pool2d(base, {15, 15}, {1, 1}, {7, 7}, false, true);
    auto contrast = (base - blur).abs();
    auto dx = (base - torch::roll(base, {1}, {3})).abs();
    auto dy = (base - torch::roll(base, {1}, {2})).abs();
    auto saliency = contrast + dx + dy;
    saliency = torch::avg_pool2d(saliency, {5, 5}, {1, 1}, {2, 2}, false, true);
    saliency = torch::nn::functional::interpolate(saliency, torch::nn::functional::InterpolateFuncOptions()
                                                                .size(std::vector<int64_t>{target_height, target_width})
                                                                .mode(torch::kBilinear)
                                                                .align_corners(false));
    auto flat = saliency.reshape({saliency.size(0), saliency.size(1), saliency.size(2) * saliency.size(3)});
    auto min_value = std::get<0>(flat.min(2, true)).reshape({saliency.size(0), saliency.size(1), 1, 1});
    auto max_value = std::get<0>(flat.max(2, true)).reshape({saliency.size(0), saliency.size(1), 1, 1});
    return (saliency - min_value) / (max_value - min_value).clamp_min(1.0e-6);
}

torch::Tensor make_inference_decode_heatmap(const torch::Tensor& image, const torch::Tensor& learned_heatmap)
{
    return make_rotation_invariant_texture_saliency(image, learned_heatmap.size(2), learned_heatmap.size(3));
}

struct InferenceModules
{
    v21::PfmV21Backbone backbone{nullptr};
    v21::PfmV21DualFPNLite dual_fpn{nullptr};
    v21::PfmV21SparseHead sparse_head{nullptr};
    v21::PfmV21TextureDescriptorAdapter texture_adapter{nullptr};
    v21::PfmV21DescriptorFusionAdapter descriptor_fusion{nullptr};
    v21::PfmV21DenseHead dense_head{nullptr};
    v21::PfmV21QualityHead quality_head{nullptr};
    v21::PfmV21SemiDenseCandidateBranch semi_dense_branch{nullptr};
    v21::PfmV21GraphMatcher graph_matcher{nullptr};
};

InferenceModules load_inference_modules(const std::string& checkpoint, const CheckpointConfig& config,
                                        torch::Device device)
{
    InferenceModules modules;
    modules.backbone = v21::PfmV21Backbone(config.input_channels, config.base_channels);
    modules.dual_fpn = v21::PfmV21DualFPNLite(config.base_channels);
    modules.sparse_head = v21::PfmV21SparseHead(config.base_channels * 2, config.descriptor_dim);
    modules.texture_adapter = v21::PfmV21TextureDescriptorAdapter(config.descriptor_dim);
    modules.descriptor_fusion = v21::PfmV21DescriptorFusionAdapter(config.descriptor_dim);
    modules.dense_head = v21::PfmV21DenseHead(config.base_channels);
    modules.quality_head = v21::PfmV21QualityHead(config.descriptor_dim);
    modules.semi_dense_branch = v21::PfmV21SemiDenseCandidateBranch(config.descriptor_dim);
    modules.graph_matcher = v21::PfmV21GraphMatcher(config.descriptor_dim, config.graph_hidden_dim,
                                                    config.graph_attention_layers, config.graph_keypoint_meta_dim);

    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive backbone_archive;
    torch::serialize::InputArchive dual_fpn_archive;
    torch::serialize::InputArchive sparse_head_archive;
    torch::serialize::InputArchive texture_adapter_archive;
    torch::serialize::InputArchive descriptor_fusion_archive;
    torch::serialize::InputArchive dense_head_archive;
    torch::serialize::InputArchive quality_head_archive;
    torch::serialize::InputArchive semi_dense_branch_archive;
    torch::serialize::InputArchive graph_matcher_archive;
    archive.read("backbone", backbone_archive);
    archive.read("dual_fpn", dual_fpn_archive);
    archive.read("sparse_head", sparse_head_archive);
    archive.read("texture_adapter", texture_adapter_archive);
    archive.read("descriptor_fusion", descriptor_fusion_archive);
    archive.read("dense_head", dense_head_archive);
    archive.read("quality_head", quality_head_archive);
    archive.read("semi_dense_branch", semi_dense_branch_archive);
    archive.read("graph_matcher", graph_matcher_archive);
    modules.backbone->load(backbone_archive);
    modules.backbone->sanitizeNonfiniteState();
    modules.dual_fpn->load(dual_fpn_archive);
    modules.sparse_head->load(sparse_head_archive);
    modules.texture_adapter->load(texture_adapter_archive);
    modules.descriptor_fusion->load(descriptor_fusion_archive);
    modules.dense_head->load(dense_head_archive);
    modules.quality_head->load(quality_head_archive);
    modules.semi_dense_branch->load(semi_dense_branch_archive);
    modules.graph_matcher->load(graph_matcher_archive);

    modules.backbone->to(device);
    modules.dual_fpn->to(device);
    modules.sparse_head->to(device);
    modules.texture_adapter->to(device);
    modules.descriptor_fusion->to(device);
    modules.dense_head->to(device);
    modules.quality_head->to(device);
    modules.semi_dense_branch->to(device);
    modules.graph_matcher->to(device);
    modules.backbone->eval();
    modules.dual_fpn->eval();
    modules.sparse_head->eval();
    modules.texture_adapter->eval();
    modules.descriptor_fusion->eval();
    modules.dense_head->eval();
    modules.quality_head->eval();
    modules.semi_dense_branch->eval();
    modules.graph_matcher->eval();
    return modules;
}

torch::Tensor resize_dense_confidence_for_heatmap(const torch::Tensor& confidence, const torch::Tensor& heatmap)
{
    if (confidence.size(2) == heatmap.size(2) && confidence.size(3) == heatmap.size(3))
    {
        return confidence;
    }
    return torch::nn::functional::interpolate(confidence,
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{heatmap.size(2), heatmap.size(3)})
                                                  .mode(torch::kNearest));
}

RawFeatureMaps run_mvp_model(const torch::Tensor& image, InferenceModules& modules, const CheckpointConfig& config,
                             torch::Device device, double texture_blend_weight)
{
    // C++ 推理复现 v2.1 Python 结构：backbone/FPN/稀疏头产生候选，纹理描述子再与已学习描述子融合。
    torch::NoGradGuard no_grad;
    const auto input = adapt_image_channels(image, config.input_channels).unsqueeze(0).contiguous().to(device);
    const auto feature_pyramid = modules.backbone->forward(input);
    const auto fpn = modules.dual_fpn->forward(feature_pyramid);
    const auto sparse = modules.sparse_head->forward(fpn.first, fpn.second);
    auto texture = v21::makeRotationInvariantTextureDescriptor(input, sparse.descriptors.size(2),
                                                               sparse.descriptors.size(3), sparse.descriptors.size(1));
    texture = modules.texture_adapter->forward(texture);
    const auto descriptors = modules.descriptor_fusion->forward(sparse.descriptors, texture, texture_blend_weight);
    const auto dense = modules.dense_head->forward(feature_pyramid.front(), feature_pyramid.front());
    const auto dense_confidence = resize_dense_confidence_for_heatmap(dense.confidence, sparse.heatmap);
    const auto texture_saliency =
        v21::makeRotationInvariantTextureSaliency(input, sparse.heatmap.size(2), sparse.heatmap.size(3));
    const auto quality = modules.quality_head->forward(descriptors, sparse.heatmap, texture_saliency, dense_confidence);
    const auto heatmap = (sparse.heatmap * quality).clamp(0.0, 1.0);
    return RawFeatureMaps{heatmap.detach().cpu().contiguous(),       descriptors.detach().cpu().contiguous(),
                          sparse.scale.detach().cpu().contiguous(),  sparse.orientation.detach().cpu().contiguous(),
                          sparse.affine.detach().cpu().contiguous(), dense_confidence.detach().cpu().contiguous()};
}

struct ExtractionTiming
{
    double image_load_seconds = 0.0;
    double model_forward_seconds = 0.0;
    double decode_seconds = 0.0;
};

struct ExtractedFeatureSet
{
    FeatureSet features;
    RawFeatureMaps maps;
    torch::Tensor intensity_mask;
    int64_t feature_map_width = 0;
    int64_t feature_map_height = 0;
    ExtractionTiming timing;
};

FeatureSet make_descriptor_grid_feature_set(const RawFeatureMaps& maps, const FeatureDecodeConfig& config,
                                            const torch::Tensor& intensity_mask)
{
    // 描述子网格回退不依赖热力图响应，直接按网格抽取描述子；用于热力图过稀但描述子仍可匹配的样本。
    const auto descriptor_height = maps.descriptors.size(2);
    const auto descriptor_width = maps.descriptors.size(3);
    const auto max_points = std::max<int>(1, config.max_keypoints);
    const auto grid_rows = std::max<int64_t>(1, static_cast<int64_t>(std::floor(std::sqrt(max_points))));
    const auto grid_cols = std::max<int64_t>(1, (max_points + grid_rows - 1) / grid_rows);
    auto resized_mask =
        intensity_mask.to(torch::kFloat32).reshape({1, 1, intensity_mask.size(0), intensity_mask.size(1)});
    if (intensity_mask.size(0) != descriptor_height || intensity_mask.size(1) != descriptor_width)
    {
        resized_mask = torch::nn::functional::interpolate(
            resized_mask, torch::nn::functional::InterpolateFuncOptions()
                              .size(std::vector<int64_t>{descriptor_height, descriptor_width})
                              .mode(torch::kNearest));
    }
    auto mask = resized_mask.reshape({descriptor_height, descriptor_width}).gt(0.0).to(torch::kCPU);
    auto descriptor_map = torch::nn::functional::normalize(
        maps.descriptors, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12));
    auto flat_descriptors = descriptor_map.squeeze(0).permute({1, 2, 0}).reshape(
        {descriptor_height * descriptor_width, descriptor_map.size(1)});

    std::vector<int64_t> indices;
    indices.reserve(static_cast<std::size_t>(max_points));
    const auto points_per_cell = std::max<int64_t>(1, static_cast<int64_t>(config.keypoints_per_cell));
    for (int64_t row = 0; row < grid_rows && static_cast<int>(indices.size()) < max_points; ++row)
    {
        const auto y0 = row * descriptor_height / grid_rows;
        const auto y1 = std::max<int64_t>(y0 + 1, (row + 1) * descriptor_height / grid_rows);
        for (int64_t col = 0; col < grid_cols && static_cast<int>(indices.size()) < max_points; ++col)
        {
            const auto x0 = col * descriptor_width / grid_cols;
            const auto x1 = std::max<int64_t>(x0 + 1, (col + 1) * descriptor_width / grid_cols);
            std::vector<int64_t> cell_indices;
            const auto center_y = std::min<int64_t>(descriptor_height - 1, (y0 + y1 - 1) / 2);
            const auto center_x = std::min<int64_t>(descriptor_width - 1, (x0 + x1 - 1) / 2);
            auto append_if_valid = [&](int64_t y, int64_t x)
            {
                if (static_cast<int>(indices.size()) + static_cast<int>(cell_indices.size()) >= max_points || y < y0 ||
                    y >= std::min<int64_t>(y1, descriptor_height) || x < x0 ||
                    x >= std::min<int64_t>(x1, descriptor_width) || !mask.index({y, x}).item<bool>())
                {
                    return;
                }
                const auto index = y * descriptor_width + x;
                if (std::find(cell_indices.begin(), cell_indices.end(), index) == cell_indices.end())
                {
                    cell_indices.push_back(index);
                }
            };
            append_if_valid(center_y, center_x);
            for (int64_t sample = 0;
                 sample < points_per_cell && static_cast<int64_t>(cell_indices.size()) < points_per_cell &&
                 static_cast<int>(indices.size()) + static_cast<int>(cell_indices.size()) < max_points;
                 ++sample)
            {
                const auto sample_y =
                    std::min<int64_t>(descriptor_height - 1,
                                      y0 + ((2 * sample + 1) * std::max<int64_t>(1, y1 - y0)) / (2 * points_per_cell));
                const auto sample_x =
                    std::min<int64_t>(descriptor_width - 1,
                                      x0 + ((2 * sample + 1) * std::max<int64_t>(1, x1 - x0)) / (2 * points_per_cell));
                append_if_valid(sample_y, sample_x);
            }
            for (int64_t yy = y0; yy < std::min<int64_t>(y1, descriptor_height) &&
                                  static_cast<int64_t>(cell_indices.size()) < points_per_cell &&
                                  static_cast<int>(indices.size()) + static_cast<int>(cell_indices.size()) < max_points;
                 ++yy)
            {
                for (int64_t xx = x0;
                     xx < std::min<int64_t>(x1, descriptor_width) &&
                     static_cast<int64_t>(cell_indices.size()) < points_per_cell &&
                     static_cast<int>(indices.size()) + static_cast<int>(cell_indices.size()) < max_points;
                     ++xx)
                {
                    append_if_valid(yy, xx);
                }
            }
            for (const auto index : cell_indices)
            {
                if (static_cast<int>(indices.size()) >= max_points)
                {
                    break;
                }
                if (std::find(indices.begin(), indices.end(), index) == indices.end())
                {
                    indices.push_back(index);
                }
            }
        }
    }

    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (indices.empty())
    {
        return FeatureSet{torch::empty({0, 2}, float_options),
                          torch::empty({0}, float_options),
                          torch::empty({0, maps.descriptors.size(1)}, float_options),
                          torch::empty({0}, float_options),
                          torch::empty({0, 2}, float_options),
                          torch::empty({0, 2, 2}, float_options),
                          torch::empty({0, 2}, float_options),
                          torch::empty({0}, float_options),
                          descriptor_width,
                          descriptor_height};
    }
    auto index_tensor = torch::from_blob(indices.data(), {static_cast<int64_t>(indices.size())}, long_options).clone();
    auto xs = index_tensor.remainder(descriptor_width).to(torch::kFloat32);
    auto ys = torch::floor_divide(index_tensor, descriptor_width).to(torch::kFloat32);
    auto keypoints = torch::stack({xs, ys}, 1).contiguous();
    auto descriptors = flat_descriptors.index_select(0, index_tensor).to(torch::kCPU, torch::kFloat32).contiguous();
    if (config.descriptor_orientation_canonicalization)
    {
        canonicalizeDescriptorRowsByOrientation(descriptors, maps.orientation, indices,
                                                descriptor_height * descriptor_width);
    }
    return FeatureSet{keypoints,
                      torch::ones({keypoints.size(0)}, float_options),
                      descriptors,
                      torch::empty({keypoints.size(0)}, float_options),
                      torch::empty({keypoints.size(0), 2}, float_options),
                      torch::empty({keypoints.size(0), 2, 2}, float_options),
                      torch::empty({0, 2}, float_options),
                      torch::empty({0}, float_options),
                      descriptor_width,
                      descriptor_height};
}

MatchSet filterMatchMode(const MatchSet& match_set, const std::string& match_mode)
{
    if (match_mode == "both")
    {
        return match_set;
    }

    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (match_mode == "sparse")
    {
        return MatchSet{match_set.sparse_matches, match_set.sparse_scores, torch::empty({0, 2}, float_options),
                        torch::empty({0, 2}, float_options), torch::empty({0}, float_options)};
    }
    if (match_mode == "dense")
    {
        const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        return MatchSet{torch::empty({0, 2}, long_options), torch::empty({0}, float_options), match_set.points_a,
                        match_set.points_b, match_set.confidence};
    }
    throw std::invalid_argument("match mode must be sparse, dense, or both");
}

bool shouldUseHighDensitySparseMatches(int64_t base_sparse_matches, int64_t high_density_sparse_matches)
{
    // 高密度重解码只在明显增加有效稀疏匹配时替换基础结果，避免单纯增加噪声候选。
    if (base_sparse_matches <= ADAPTIVE_HIGH_DENSITY_LOW_BASE_MAX_MATCHES)
    {
        return high_density_sparse_matches >= ADAPTIVE_HIGH_DENSITY_LOW_BASE_MIN_MATCHES ||
               (base_sparse_matches <= ADAPTIVE_HIGH_DENSITY_LOW_BASE_MODERATE_MAX_BASE_MATCHES &&
                high_density_sparse_matches >= ADAPTIVE_HIGH_DENSITY_LOW_BASE_MODERATE_MIN_MATCHES &&
                high_density_sparse_matches <= base_sparse_matches * ADAPTIVE_HIGH_DENSITY_LOW_BASE_MODERATE_MAX_GAIN);
    }
    return base_sparse_matches >= ADAPTIVE_HIGH_DENSITY_STRONG_BASE_MIN_MATCHES &&
           high_density_sparse_matches >= base_sparse_matches + ADAPTIVE_HIGH_DENSITY_STRONG_MIN_GAIN;
}

bool shouldUseAlternateTextureBlendMatches(int64_t base_sparse_matches, int64_t alternate_sparse_matches)
{
    return alternate_sparse_matches >= ALTERNATE_TEXTURE_BLEND_MIN_SPARSE_MATCHES &&
           alternate_sparse_matches >= base_sparse_matches * ALTERNATE_TEXTURE_BLEND_MIN_GAIN_NUMERATOR;
}

bool shouldUseBalancedTextureBlendMatches(int64_t base_sparse_matches, int64_t alternate_sparse_matches)
{
    return base_sparse_matches >= BALANCED_TEXTURE_BLEND_MIN_BASE_MATCHES &&
           alternate_sparse_matches >= base_sparse_matches &&
           alternate_sparse_matches <= base_sparse_matches + BALANCED_TEXTURE_BLEND_MAX_GAIN;
}

bool shouldUseDescriptorGridFallback(int64_t base_sparse_matches, int64_t grid_sparse_matches)
{
    return base_sparse_matches <= DESCRIPTOR_GRID_FALLBACK_MAX_BASE_MATCHES &&
           grid_sparse_matches >= DESCRIPTOR_GRID_FALLBACK_MIN_SPARSE_MATCHES &&
           grid_sparse_matches > base_sparse_matches;
}

FeatureDecodeConfig makeHighDensityDecodeConfig(FeatureDecodeConfig decode_config)
{
    decode_config.min_keypoints = std::max<int>(decode_config.min_keypoints, ADAPTIVE_HIGH_DENSITY_MIN_KEYPOINTS);
    return decode_config;
}

FeatureSet decode_high_density_feature_set(const ExtractedFeatureSet& extracted, FeatureDecodeConfig decode_config)
{
    // 复用已经前向得到的原始特征图，仅提高 min_keypoints，避免为了补点重复跑网络。
    decode_config = makeHighDensityDecodeConfig(decode_config);
    auto features = decode_feature_maps(extracted.maps, decode_config, extracted.intensity_mask);
    features.feature_map_width = extracted.feature_map_width;
    features.feature_map_height = extracted.feature_map_height;
    return features;
}

FeatureDecodeConfig makeFeatureDecodeConfig(const CliOptions& options)
{
    FeatureDecodeConfig config;
    config.max_keypoints = options.max_keypoints;
    config.min_keypoints = options.min_keypoints;
    config.semi_dense_threshold = options.semi_dense_threshold;
    config.keypoint_grid_rows = options.keypoint_grid_rows;
    config.keypoint_grid_cols = options.keypoint_grid_cols;
    config.keypoints_per_cell = options.keypoints_per_cell;
    config.nms_radius = options.nms_radius;
    config.descriptor_pool_radius = options.descriptor_pool_radius;
    config.descriptor_orientation_canonicalization = !options.disable_descriptor_orientation_canonicalization;
    return config;
}

GraphMatcherInferenceOptions makeGraphMatcherInferenceOptions(const CliOptions& options)
{
    GraphMatcherInferenceOptions graph_options;
    if (options.graph_inference_preset == "off")
    {
        graph_options.width_prune_min_score = -1.0;
        graph_options.early_stop_min_confidence = -1.0;
    }
    else if (options.graph_inference_preset == "fast")
    {
        graph_options.width_prune_min_score = 0.25;
        graph_options.early_stop_min_confidence = 0.85;
    }
    else if (options.graph_inference_preset == "high_precision")
    {
        graph_options.width_prune_min_score = 0.5;
        graph_options.early_stop_min_confidence = 0.85;
    }
    else
    {
        throw std::invalid_argument("graph_inference_preset must be one of: off, fast, high_precision");
    }

    if (options.graph_width_prune_min_score > -1.0)
    {
        graph_options.width_prune_min_score = options.graph_width_prune_min_score;
    }
    if (options.graph_early_stop_min_confidence > -1.0)
    {
        graph_options.early_stop_min_confidence = options.graph_early_stop_min_confidence;
    }
    if (options.graph_min_accept_probability > -1.0)
    {
        graph_options.min_accept_probability = options.graph_min_accept_probability;
    }
    graph_options.max_attention_layers = options.graph_max_attention_layers;
    graph_options.max_attention_work_fraction = options.graph_max_attention_work_fraction;
    graph_options.width_prune_keep_ratio = options.graph_width_prune_keep_ratio;
    if (options.graph_fallback_mode == "geometry")
    {
        graph_options.fallback_mode = GraphMatcherFallbackMode::Geometry;
    }
    else if (options.graph_fallback_mode == "none")
    {
        graph_options.fallback_mode = GraphMatcherFallbackMode::None;
    }
    else
    {
        throw std::invalid_argument("graph_fallback_mode must be one of: geometry, none");
    }
    return graph_options;
}

ExtractedFeatureSet extract_feature_set(const std::string& image_path, InferenceModules& modules,
                                        const CheckpointConfig& checkpoint_config, torch::Device device,
                                        const FeatureDecodeConfig& decode_config, double min_keypoint_intensity,
                                        double texture_blend_weight = rotationInvariantTextureBlendWeight())
{
    // 单张影像提取被匹配/评估/提取命令共用，同时记录分阶段耗时，方便定位慢在 IO、前向还是解码。
    ExtractionTiming timing;
    Timer image_timer;
    const auto image = load_image_tensor(image_path);
    timing.image_load_seconds = image_timer.elapsedSeconds();

    Timer forward_timer;
    const auto maps = run_mvp_model(image, modules, checkpoint_config, device, texture_blend_weight);
    timing.model_forward_seconds = forward_timer.elapsedSeconds();

    Timer decode_timer;
    const auto intensity_mask = make_intensity_mask(image, min_keypoint_intensity).to(torch::kCPU);
    auto features = decode_feature_maps(maps, decode_config, intensity_mask);
    timing.decode_seconds = decode_timer.elapsedSeconds();
    features.feature_map_width = maps.heatmap.size(3);
    features.feature_map_height = maps.heatmap.size(2);

    const auto feature_map_width = maps.heatmap.size(3);
    const auto feature_map_height = maps.heatmap.size(2);
    return ExtractedFeatureSet{std::move(features), std::move(maps),    intensity_mask,
                               feature_map_width,   feature_map_height, timing};
}

bool inference_checkpoint_can_load(const std::string& checkpoint)
{
    try
    {
        const auto checkpoint_config = load_checkpoint_config(checkpoint);
        (void)load_inference_modules(checkpoint, checkpoint_config, torch::Device(torch::kCPU));
        return true;
    }
    catch (const c10::Error&)
    {
        return false;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

void copy_file_contents(const std::string& source, const std::string& destination)
{
    std::ifstream input(source, std::ios::binary);
    if (!input)
    {
        throw std::invalid_argument("failed to open checkpoint for export: " + source);
    }
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw std::invalid_argument("failed to open export output: " + destination);
    }
    output << input.rdbuf();
    if (!output)
    {
        throw std::invalid_argument("failed to write export output: " + destination);
    }
}

} // namespace

int run_train_command(const CliOptions& options)
{
    if ((options.image_dir.empty() && options.pair_cache_dirs.empty()) ||
        !require_path(options.checkpoint, "--checkpoint"))
    {
        if (options.image_dir.empty() && options.pair_cache_dirs.empty())
        {
            std::cerr << "missing required option --image-dir or --pair-cache-dir\n";
        }
        return 1;
    }

    try
    {
        TrainConfig config;
        config.image_dir = options.image_dir;
        config.checkpoint = options.checkpoint;
        config.init_checkpoint = options.init_checkpoint;
        config.device = options.device;
        config.epochs = options.epochs;
        config.batch_size = options.batch_size;
        config.resize = options.resize;
        config.training_crop_size = options.training_crop_size;
        config.base_channels = options.base_channels;
        config.descriptor_dim = options.descriptor_dim;
        config.graph_hidden_dim = options.graph_hidden_dim;
        config.graph_attention_layers = options.graph_attention_layers;
        config.graph_keypoint_meta_dim = options.graph_keypoint_meta_dim;
        config.training_profile = options.training_profile;
        config.samples_per_pair = options.samples_per_pair;
        config.synthetic_loss_weight = options.synthetic_loss_weight;
        config.graph_matcher_loss_weight = options.graph_matcher_loss_weight;
        config.graph_matcher_accept_weight = options.graph_matcher_accept_weight;
        config.graph_matcher_accept_negative_topk = options.graph_matcher_accept_negative_topk;
        config.graph_matcher_no_match_points = options.graph_matcher_no_match_points;
        config.graph_matcher_no_match_min_distance = options.graph_matcher_no_match_min_distance;
        config.graph_matcher_train_max_attention_layers = options.graph_matcher_train_max_attention_layers;
        config.graph_matcher_train_random_attention_layers = options.graph_matcher_train_random_attention_layers;
        config.graph_matcher_prune_ranking_weight = options.graph_matcher_prune_ranking_weight;
        config.graph_matcher_prune_ranking_margin = options.graph_matcher_prune_ranking_margin;
        config.graph_matcher_stop_confidence_weight = options.graph_matcher_stop_confidence_weight;
        config.graph_matcher_stop_confidence_margin = options.graph_matcher_stop_confidence_margin;
        config.train_backbone = options.train_backbone;
        config.train_dual_fpn = options.train_dual_fpn;
        config.freeze_descriptor_head = options.freeze_descriptor_head;
        config.train_sparse_context = options.train_sparse_context;
        config.train_keypoint_head = options.train_keypoint_head;
        config.train_geometry_head = options.train_geometry_head;
        config.train_blended_descriptors = options.train_blended_descriptors;
        config.train_texture_adapter = options.train_texture_adapter;
        config.train_descriptor_fusion = options.train_descriptor_fusion;
        config.train_quality_head = options.train_quality_head;
        config.train_graph_matcher = options.train_graph_matcher;
        config.training_texture_blend_weight = options.training_texture_blend_weight;
        config.temperature = options.temperature;
        config.pairs_per_image = options.pairs_per_image;
        config.max_train_batches = options.max_train_batches;
        config.learning_rate = options.learning_rate;
        config.lr_warmup_steps = options.lr_warmup_steps;
        config.min_learning_rate_ratio = options.min_learning_rate_ratio;
        config.weight_decay = options.weight_decay;
        config.gradient_clip_norm = options.gradient_clip_norm;
        config.augmentation_profile = options.augmentation_profile;
        config.augmentation_curriculum = options.augmentation_curriculum;
        config.extreme_pair_ratio = options.extreme_pair_ratio;
        config.rotation_step_degrees = options.rotation_step_degrees;
        config.train_ratio = options.train_ratio;
        config.val_ratio = options.val_ratio;
        config.seed = options.seed;
        config.split_seed = options.split_seed;
        config.synthetic_pair_cache_dir = options.synthetic_pair_cache_dir;
        config.extra_synthetic_pair_cache_dirs = options.extra_synthetic_pair_cache_dirs;
        config.hard_synthetic_pair_cache_dirs = options.hard_synthetic_pair_cache_dirs;
        config.pair_cache_dirs = options.pair_cache_dirs;
        config.pair_cache_limit = options.pair_cache_limit;
        config.pair_memory_cache_size = options.pair_memory_cache_size;
        config.hard_synthetic_pair_cache_repeats = options.hard_synthetic_pair_cache_repeats;
        config.hard_synthetic_pair_cache_indices = options.hard_synthetic_pair_cache_indices;
        config.cache_only = options.cache_only;
        config.log_csv = options.log_csv;
        config.dataloader_workers = options.dataloader_workers;
        config.prefetch_batches = options.prefetch_batches;
        config.pin_memory = options.pin_memory;
        config.descriptor_only_finetune = options.descriptor_only_finetune;
        config.viewpoint_head_only_finetune = options.viewpoint_head_only_finetune;
        config.graph_only_finetune = options.graph_only_finetune;
        config.descriptor_orientation_canonicalization = !options.disable_descriptor_orientation_canonicalization;
        config.synthetic_pair_cache_rebuild = options.synthetic_pair_cache_rebuild;
        config.visualization_dir = options.visualization_dir;
        config.visualization_samples = options.visualization_samples;
        config.visualization_samples_all = options.visualization_samples_all;
        config.max_keypoints = options.max_keypoints;
        config.min_keypoints = options.min_keypoints;
        config.keypoint_grid_rows = options.keypoint_grid_rows;
        config.keypoint_grid_cols = options.keypoint_grid_cols;
        config.keypoints_per_cell = options.keypoints_per_cell;
        config.nms_radius = options.nms_radius;
        config.min_keypoint_intensity = options.min_keypoint_intensity;
        const auto result = train_model(config);
        std::cout << "training complete: epochs=" << result.epochs_completed << " final_loss=" << result.final_loss
                  << " total_time=" << formatSeconds(result.total_time_seconds) << "s"
                  << " avg_batch_time=" << formatSeconds(result.avg_batch_time_seconds) << "s\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "train failed: " << error.what() << '\n';
        return 1;
    }
}

int run_extract_command(const CliOptions& options)
{
    if (!require_path(options.image, "--image") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output"))
    {
        return 1;
    }

    try
    {
        Timer total_timer;
        if (!checkpoint_can_load(options.checkpoint))
        {
            std::cerr << "extract failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        const auto decode_config = makeFeatureDecodeConfig(options);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config, device);
        const auto extracted = extract_feature_set(options.image, modules, checkpoint_config, device, decode_config,
                                                   options.min_keypoint_intensity);
        Timer save_timer;
        save_feature_set(extracted.features, options.output);
        const auto save_seconds = save_timer.elapsedSeconds();
        double visualization_seconds = 0.0;
        if (!options.visualization_dir.empty())
        {
            Timer visualization_timer;
            (void)save_feature_visualization(options.image, extracted.features, options.visualization_dir,
                                             extracted.feature_map_width, extracted.feature_map_height);
            visualization_seconds = visualization_timer.elapsedSeconds();
        }
        std::cout << "extraction complete: features=" << options.output
                  << " sparse_features=" << extracted.features.keypoints.size(0)
                  << " dense_features=" << extracted.features.dense_points.size(0)
                  << " elapsed=" << formatSeconds(total_timer.elapsedSeconds()) << "s"
                  << " image_load=" << formatSeconds(extracted.timing.image_load_seconds) << "s"
                  << " model_forward=" << formatSeconds(extracted.timing.model_forward_seconds) << "s"
                  << " decode=" << formatSeconds(extracted.timing.decode_seconds) << "s"
                  << " save=" << formatSeconds(save_seconds) << "s"
                  << " visualization=" << formatSeconds(visualization_seconds) << "s\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "extract failed: " << error.what() << '\n';
        return 1;
    }
}

int run_match_command(const CliOptions& options)
{
    const bool use_feature_files = !options.feature_a.empty() || !options.feature_b.empty();
    if (!require_path(options.output, "--output"))
    {
        return 1;
    }
    if (use_feature_files)
    {
        if (!require_path(options.feature_a, "--feature-a") || !require_path(options.feature_b, "--feature-b") ||
            !require_path(options.checkpoint, "--checkpoint"))
        {
            return 1;
        }
    }
    else if (!require_path(options.image_a, "--image-a") || !require_path(options.image_b, "--image-b") ||
             !require_path(options.checkpoint, "--checkpoint"))
    {
        return 1;
    }

    try
    {
        Timer total_timer;
        ScopedEnvironmentOverride sparse_geometry_filter_env("PFM_SPARSE_GEOMETRY_FILTER",
                                                             options.sparse_geometry_filter);
        if (!checkpoint_can_load(options.checkpoint))
        {
            std::cerr << "match failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config, device);
        ExtractedFeatureSet extracted_a;
        ExtractedFeatureSet extracted_b;
        if (use_feature_files)
        {
            // 使用预先保存的 .pt 特征时不重新解码，特征图尺寸来自 archive，用于坐标缩放和可视化。
            extracted_a.features = load_feature_set(options.feature_a);
            extracted_b.features = load_feature_set(options.feature_b);
            extracted_a.feature_map_width = extracted_a.features.feature_map_width;
            extracted_a.feature_map_height = extracted_a.features.feature_map_height;
            extracted_b.feature_map_width = extracted_b.features.feature_map_width;
            extracted_b.feature_map_height = extracted_b.features.feature_map_height;
        }
        else
        {
            const auto decode_config = makeFeatureDecodeConfig(options);
            extracted_a = extract_feature_set(options.image_a, modules, checkpoint_config, device, decode_config,
                                              options.min_keypoint_intensity);
            extracted_b = extract_feature_set(options.image_b, modules, checkpoint_config, device, decode_config,
                                              options.min_keypoint_intensity);
        }
        const auto extract_a_seconds = extracted_a.timing.image_load_seconds +
                                       extracted_a.timing.model_forward_seconds + extracted_a.timing.decode_seconds;
        const auto extract_b_seconds = extracted_b.timing.image_load_seconds +
                                       extracted_b.timing.model_forward_seconds + extracted_b.timing.decode_seconds;
        Timer match_timer;
        auto metric_features_a = extracted_a.features;
        auto metric_features_b = extracted_b.features;
        const bool use_python_raw_mutual = options.sparse_match_strategy == "python-raw-mutual";
        if (!use_python_raw_mutual && options.sparse_match_strategy != "learned")
        {
            throw std::invalid_argument("sparse match strategy must be learned or python-raw-mutual");
        }
        const auto graph_options = makeGraphMatcherInferenceOptions(options);
        auto raw_match_set =
            use_python_raw_mutual
                ? matchFeatureSetsPythonRawMutual(extracted_a.features, extracted_b.features, options.max_matches)
                : matchFeatureSets(extracted_a.features, extracted_b.features, *modules.graph_matcher, graph_options);
        auto match_set = filterMatchMode(raw_match_set, options.match_mode);
        if (!use_python_raw_mutual && !use_feature_files && options.match_mode != "dense" &&
            match_set.sparse_matches.size(0) < DESCRIPTOR_GRID_FALLBACK_MIN_SPARSE_MATCHES &&
            extracted_a.maps.descriptors.defined() && extracted_b.maps.descriptors.defined() &&
            extracted_a.intensity_mask.defined() && extracted_b.intensity_mask.defined())
        {
            // 常规热力图匹配很少时，网格描述子回退可验证是否是关键点检测过稀而不是描述子失效。
            const auto decode_config = makeFeatureDecodeConfig(options);
            auto grid_features_a =
                make_descriptor_grid_feature_set(extracted_a.maps, decode_config, extracted_a.intensity_mask);
            auto grid_features_b =
                make_descriptor_grid_feature_set(extracted_b.maps, decode_config, extracted_b.intensity_mask);
            auto grid_match_set = filterMatchMode(
                matchFeatureSets(grid_features_a, grid_features_b, *modules.graph_matcher, graph_options),
                options.match_mode);
            if (shouldUseDescriptorGridFallback(match_set.sparse_matches.size(0),
                                                grid_match_set.sparse_matches.size(0)))
            {
                match_set = std::move(grid_match_set);
                metric_features_a = std::move(grid_features_a);
                metric_features_b = std::move(grid_features_b);
            }
        }
        const auto skip_expensive_sparse_alternates =
            shouldSkipExpensiveSparseAlternates(match_set.sparse_matches.size(0));
        if (!use_python_raw_mutual && !use_feature_files && options.match_mode != "dense" &&
            !skip_expensive_sparse_alternates &&
            options.min_keypoints < ADAPTIVE_HIGH_DENSITY_MIN_KEYPOINTS &&
            options.max_keypoints >= ADAPTIVE_HIGH_DENSITY_MIN_KEYPOINTS && extracted_a.maps.descriptors.defined() &&
            extracted_b.maps.descriptors.defined())
        {
            // 如果用户允许高上限但默认 min_keypoints 较低，尝试高密度重解码补足弱纹理区域。
            const auto decode_config = makeFeatureDecodeConfig(options);
            auto high_density_features_a = decode_high_density_feature_set(extracted_a, decode_config);
            auto high_density_features_b = decode_high_density_feature_set(extracted_b, decode_config);
            auto high_density_match_set = filterMatchMode(
                matchFeatureSets(high_density_features_a, high_density_features_b, *modules.graph_matcher,
                                 graph_options),
                options.match_mode);
            if (shouldUseHighDensitySparseMatches(match_set.sparse_matches.size(0),
                                                  high_density_match_set.sparse_matches.size(0)))
            {
                match_set = std::move(high_density_match_set);
                metric_features_a = std::move(high_density_features_a);
                metric_features_b = std::move(high_density_features_b);
            }
        }
        if (!use_python_raw_mutual && !use_feature_files && options.match_mode != "dense" &&
            !skip_expensive_sparse_alternates &&
            options.max_keypoints >= ADAPTIVE_HIGH_DENSITY_MIN_KEYPOINTS &&
            rotationInvariantTextureBlendWeight() != ALTERNATE_TEXTURE_BLEND_WEIGHT)
        {
            // 纹理融合权重为 0 的重跑保留纯 learned descriptor 路径，用于纹理先验干扰时的补救。
            const auto decode_config = makeFeatureDecodeConfig(options);
            auto alternate_a = extract_feature_set(options.image_a, modules, checkpoint_config, device, decode_config,
                                                   options.min_keypoint_intensity, ALTERNATE_TEXTURE_BLEND_WEIGHT);
            auto alternate_b = extract_feature_set(options.image_b, modules, checkpoint_config, device, decode_config,
                                                   options.min_keypoint_intensity, ALTERNATE_TEXTURE_BLEND_WEIGHT);
            auto alternate_match_set =
                filterMatchMode(matchFeatureSets(alternate_a.features, alternate_b.features, *modules.graph_matcher,
                                                 graph_options),
                                options.match_mode);
            if (options.min_keypoints < ADAPTIVE_HIGH_DENSITY_MIN_KEYPOINTS && alternate_a.maps.descriptors.defined() &&
                alternate_b.maps.descriptors.defined())
            {
                auto alternate_high_density_a = decode_high_density_feature_set(alternate_a, decode_config);
                auto alternate_high_density_b = decode_high_density_feature_set(alternate_b, decode_config);
                auto alternate_high_density_match_set = filterMatchMode(
                    matchFeatureSets(alternate_high_density_a, alternate_high_density_b, *modules.graph_matcher,
                                     graph_options),
                    options.match_mode);
                if (shouldUseHighDensitySparseMatches(alternate_match_set.sparse_matches.size(0),
                                                      alternate_high_density_match_set.sparse_matches.size(0)))
                {
                    alternate_match_set = std::move(alternate_high_density_match_set);
                    alternate_a.features = std::move(alternate_high_density_a);
                    alternate_b.features = std::move(alternate_high_density_b);
                }
            }
            if (shouldUseAlternateTextureBlendMatches(match_set.sparse_matches.size(0),
                                                      alternate_match_set.sparse_matches.size(0)))
            {
                match_set = std::move(alternate_match_set);
                metric_features_a = std::move(alternate_a.features);
                metric_features_b = std::move(alternate_b.features);
            }
        }
        if (!use_python_raw_mutual && !use_feature_files && options.match_mode != "dense" &&
            !skip_expensive_sparse_alternates &&
            options.max_keypoints >= ADAPTIVE_HIGH_DENSITY_MIN_KEYPOINTS &&
            rotationInvariantTextureBlendWeight() != BALANCED_TEXTURE_BLEND_WEIGHT)
        {
            // 平衡纹理融合只接受小幅增益，避免为了少量匹配替换掉已经稳定的基础结果。
            const auto decode_config = makeFeatureDecodeConfig(options);
            auto balanced_a = extract_feature_set(options.image_a, modules, checkpoint_config, device, decode_config,
                                                  options.min_keypoint_intensity, BALANCED_TEXTURE_BLEND_WEIGHT);
            auto balanced_b = extract_feature_set(options.image_b, modules, checkpoint_config, device, decode_config,
                                                  options.min_keypoint_intensity, BALANCED_TEXTURE_BLEND_WEIGHT);
            auto balanced_match_set = filterMatchMode(
                matchFeatureSets(balanced_a.features, balanced_b.features, *modules.graph_matcher, graph_options),
                options.match_mode);
            if (options.min_keypoints < ADAPTIVE_HIGH_DENSITY_MIN_KEYPOINTS && balanced_a.maps.descriptors.defined() &&
                balanced_b.maps.descriptors.defined())
            {
                auto balanced_high_density_a = decode_high_density_feature_set(balanced_a, decode_config);
                auto balanced_high_density_b = decode_high_density_feature_set(balanced_b, decode_config);
                auto balanced_high_density_match_set = filterMatchMode(
                    matchFeatureSets(balanced_high_density_a, balanced_high_density_b, *modules.graph_matcher,
                                     graph_options),
                    options.match_mode);
                if (shouldUseHighDensitySparseMatches(balanced_match_set.sparse_matches.size(0),
                                                      balanced_high_density_match_set.sparse_matches.size(0)))
                {
                    balanced_match_set = std::move(balanced_high_density_match_set);
                    balanced_a.features = std::move(balanced_high_density_a);
                    balanced_b.features = std::move(balanced_high_density_b);
                }
            }
            if (shouldUseBalancedTextureBlendMatches(match_set.sparse_matches.size(0),
                                                     balanced_match_set.sparse_matches.size(0)))
            {
                match_set = std::move(balanced_match_set);
                metric_features_a = std::move(balanced_a.features);
                metric_features_b = std::move(balanced_b.features);
            }
        }
        const auto match_seconds = match_timer.elapsedSeconds();
        Timer save_timer;
        save_match_set(match_set, options.output);
        const auto save_seconds = save_timer.elapsedSeconds();
        WarpMatchMetrics warp_metrics;
        bool has_warp_metrics = false;
        torch::Tensor warp_a_to_b;
        if (!options.warp_a_to_b.empty())
        {
            // 合成影像对提供真实稠密变形场时，直接输出正确/错误匹配数，供检查点质量门控和回归测试使用。
            warp_a_to_b = load_warp_a_to_b_tensor(options.warp_a_to_b);
            warp_metrics = compute_warp_match_metrics(metric_features_a, metric_features_b, match_set, warp_a_to_b,
                                                      options.match_correct_threshold_pixels);
            has_warp_metrics = true;
        }
        double visualization_seconds = 0.0;
        if (!options.visualization_dir.empty())
        {
            Timer visualization_timer;
            if (metric_features_a.feature_map_width > 0 && metric_features_a.feature_map_height > 0 &&
                metric_features_b.feature_map_width > 0 && metric_features_b.feature_map_height > 0)
            {
                // 有特征图尺寸时按解码坐标缩放到原图，预提取特征和在线提取都能共用这一条路径。
                if (warp_a_to_b.defined())
                {
                    (void)save_match_visualization(
                        options.image_a, options.image_b, metric_features_a, metric_features_b, match_set,
                        options.visualization_dir, metric_features_a.feature_map_width,
                        metric_features_a.feature_map_height, metric_features_b.feature_map_width,
                        metric_features_b.feature_map_height, warp_a_to_b, options.match_correct_threshold_pixels);
                }
                else
                {
                    (void)save_match_visualization(
                        options.image_a, options.image_b, metric_features_a, metric_features_b, match_set,
                        options.visualization_dir, metric_features_a.feature_map_width,
                        metric_features_a.feature_map_height, metric_features_b.feature_map_width,
                        metric_features_b.feature_map_height);
                }
            }
            else
            {
                // 旧 archive 可能没有特征图尺寸，只能按已经写入的点坐标直接可视化。
                if (warp_a_to_b.defined())
                {
                    (void)save_match_visualization(options.image_a, options.image_b, match_set,
                                                   options.visualization_dir, warp_a_to_b,
                                                   options.match_correct_threshold_pixels);
                }
                else
                {
                    (void)save_match_visualization(options.image_a, options.image_b, match_set,
                                                   options.visualization_dir);
                }
            }
            visualization_seconds = visualization_timer.elapsedSeconds();
        }
        std::cout << "matching complete: matches=" << options.output
                  << " features_a=" << metric_features_a.keypoints.size(0)
                  << " features_b=" << metric_features_b.keypoints.size(0)
                  << " sparse_matches=" << match_set.sparse_matches.size(0)
                  << " dense_matches=" << match_set.points_a.size(0)
                  << " graph_layers=" << match_set.graph_executed_layers
                  << " graph_keypoints=" << match_set.graph_kept_keypoints_a << "/" << match_set.graph_input_keypoints_a
                  << "," << match_set.graph_kept_keypoints_b << "/" << match_set.graph_input_keypoints_b
                  << " graph_pruned=" << match_set.graph_pruned_keypoints_a << "/"
                  << match_set.graph_pruned_keypoints_b
                  << " graph_work=" << match_set.graph_attention_work_fraction
                  << " graph_work_units=" << match_set.graph_attention_work_units << "/"
                  << match_set.graph_full_attention_work_units
                  << " correct_matches=" << (has_warp_metrics ? warp_metrics.correct() : 0)
                  << " wrong_matches=" << (has_warp_metrics ? warp_metrics.total() - warp_metrics.correct() : 0)
                  << " match_precision=" << (has_warp_metrics ? warp_metrics.precision() : 0.0)
                  << " elapsed=" << formatSeconds(total_timer.elapsedSeconds()) << "s"
                  << " extract_a=" << formatSeconds(extract_a_seconds) << "s"
                  << " extract_b=" << formatSeconds(extract_b_seconds) << "s"
                  << " match_time=" << formatSeconds(match_seconds) << "s"
                  << " save=" << formatSeconds(save_seconds) << "s"
                  << " visualization=" << formatSeconds(visualization_seconds) << "s\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "match failed: " << error.what() << '\n';
        return 1;
    }
}

int run_eval_command(const CliOptions& options)
{
    if (!require_path(options.pairs, "--pairs") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output"))
    {
        return 1;
    }

    try
    {
        Timer total_timer;
        if (!checkpoint_can_load(options.checkpoint))
        {
            std::cerr << "eval failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        const auto device = resolve_compute_device(options.device);
        const auto pairs = loadEvalPairs(options.pairs);
        const auto checkpoint_config = load_checkpoint_config(options.checkpoint);
        const auto decode_config = makeFeatureDecodeConfig(options);
        auto modules = load_inference_modules(options.checkpoint, checkpoint_config, device);
        const bool use_python_raw_mutual = options.sparse_match_strategy == "python-raw-mutual";
        if (!use_python_raw_mutual && options.sparse_match_strategy != "learned")
        {
            throw std::invalid_argument("sparse match strategy must be learned or python-raw-mutual");
        }
        const auto graph_options = makeGraphMatcherInferenceOptions(options);

        std::vector<std::pair<FeatureSet, FeatureSet>> feature_sets;
        std::vector<MatchSet> match_sets;
        feature_sets.reserve(pairs.size());
        match_sets.reserve(pairs.size());
        for (const auto& pair : pairs)
        {
            auto extracted_a = extract_feature_set(pair.first, modules, checkpoint_config, device, decode_config,
                                                   options.min_keypoint_intensity);
            auto extracted_b = extract_feature_set(pair.second, modules, checkpoint_config, device, decode_config,
                                                   options.min_keypoint_intensity);
            match_sets.push_back(
                use_python_raw_mutual
                    ? matchFeatureSetsPythonRawMutual(extracted_a.features, extracted_b.features, options.max_matches)
                    : matchFeatureSets(extracted_a.features, extracted_b.features, *modules.graph_matcher,
                                       graph_options));
            feature_sets.push_back(std::make_pair(std::move(extracted_a.features), std::move(extracted_b.features)));
        }

        const auto report = aggregateEvalReport(feature_sets, match_sets);
        saveEvalReport(options.output, report);
        const auto elapsed = total_timer.elapsedSeconds();
        const auto avg_pair_time = pairs.empty() ? 0.0 : elapsed / static_cast<double>(pairs.size());
        std::cout << "evaluation complete: report=" << options.output << " pairs=" << pairs.size()
                  << " half_turn_consistency=" << report.half_turn_consistency
                  << " half_turn_mean_error=" << report.half_turn_mean_error
                  << " graph_layers=" << report.average_graph_executed_layers
                  << " graph_keypoints=" << report.average_graph_kept_keypoints_a << "/"
                  << report.average_graph_input_keypoints_a << "," << report.average_graph_kept_keypoints_b << "/"
                  << report.average_graph_input_keypoints_b
                  << " graph_pruned=" << report.graph_pruned_keypoint_fraction
                  << " graph_work=" << report.graph_attention_work_fraction << " elapsed=" << formatSeconds(elapsed)
                  << "s"
                  << " avg_pair_time=" << formatSeconds(avg_pair_time) << "s\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "eval failed: " << error.what() << '\n';
        return 1;
    }
}

int run_export_command(const CliOptions& options)
{
    if (!require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output"))
    {
        return 1;
    }

    try
    {
        Timer total_timer;
        if (!inference_checkpoint_can_load(options.checkpoint))
        {
            std::cerr << "export failed: checkpoint cannot load: " << options.checkpoint << '\n';
            return 1;
        }
        copy_file_contents(options.checkpoint, options.output);
        if (!inference_checkpoint_can_load(options.output))
        {
            std::cerr << "export failed: exported checkpoint cannot load: " << options.output << '\n';
            return 1;
        }
        std::cout << "export complete: checkpoint=" << options.output
                  << " elapsed=" << formatSeconds(total_timer.elapsedSeconds()) << "s\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "export failed: " << error.what() << '\n';
        return 1;
    }
}

namespace testing
{

int64_t descriptor_grid_fallback_min_sparse_matches_for_test()
{
    return DESCRIPTOR_GRID_FALLBACK_MIN_SPARSE_MATCHES;
}

bool should_use_descriptor_grid_fallback_for_test(int64_t base_sparse_matches, int64_t grid_sparse_matches)
{
    return shouldUseDescriptorGridFallback(base_sparse_matches, grid_sparse_matches);
}

bool should_use_high_density_sparse_matches_for_test(int64_t base_sparse_matches, int64_t high_density_sparse_matches)
{
    return shouldUseHighDensitySparseMatches(base_sparse_matches, high_density_sparse_matches);
}

bool should_use_alternate_texture_blend_matches_for_test(int64_t base_sparse_matches, int64_t alternate_sparse_matches)
{
    return shouldUseAlternateTextureBlendMatches(base_sparse_matches, alternate_sparse_matches);
}

bool should_use_balanced_texture_blend_matches_for_test(int64_t base_sparse_matches, int64_t alternate_sparse_matches)
{
    return shouldUseBalancedTextureBlendMatches(base_sparse_matches, alternate_sparse_matches);
}

double rotation_invariant_texture_blend_weight_for_test()
{
    return rotationInvariantTextureBlendWeight();
}

bool sparse_geometry_filter_rotation_only_requested_for_test()
{
    return sparseGeometryFilterRotationOnlyRequested();
}

bool should_skip_expensive_sparse_alternates_for_test(int64_t sparse_matches)
{
    return shouldSkipExpensiveSparseAlternates(sparse_matches);
}

FeatureSet make_descriptor_grid_feature_set_for_test(const RawFeatureMaps& maps, const FeatureDecodeConfig& config,
                                                     const torch::Tensor& intensity_mask)
{
    return make_descriptor_grid_feature_set(maps, config, intensity_mask);
}

torch::Tensor make_inference_decode_heatmap_for_test(const torch::Tensor& image, const torch::Tensor& learned_heatmap)
{
    return make_inference_decode_heatmap(image, learned_heatmap);
}

FeatureDecodeConfig make_high_density_decode_config_for_test(FeatureDecodeConfig decode_config)
{
    return makeHighDensityDecodeConfig(decode_config);
}

GraphMatcherInferenceOptions make_graph_matcher_inference_options_for_test(const CliOptions& options)
{
    return makeGraphMatcherInferenceOptions(options);
}

} // namespace testing

} // namespace pfm
