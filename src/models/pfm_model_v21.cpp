#include "models/pfm_model_v21.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>

namespace pfm::v21
{
namespace
{

constexpr int64_t CORRELATION_RADIUS = 4;
constexpr int64_t CORRELATION_CHANNELS = (CORRELATION_RADIUS * 2 + 1) * (CORRELATION_RADIUS * 2 + 1);

// 当前 C++ v2.1 模型镜像 Python 训练结构，辅助函数尽量保持和 state_dict 命名/形状兼容。
void requirePositive(int64_t value, const char* name)
{
    if (value <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

int64_t groupCount(int64_t channels)
{
    // 分组归一化优先使用较大组数，但保证每组至少两个通道，适配不同 base_channels。
    for (const auto groups : {32, 16, 8, 4, 2})
    {
        if (channels % groups == 0 && channels / groups >= 2)
        {
            return groups;
        }
    }
    return 1;
}

torch::nn::GroupNorm makeNorm(int64_t channels)
{
    return torch::nn::GroupNorm(torch::nn::GroupNormOptions(groupCount(channels), channels));
}

torch::Tensor finiteOrZero(const torch::Tensor& tensor)
{
    return torch::where(torch::isfinite(tensor), tensor, torch::zeros_like(tensor));
}

torch::Tensor normalizeChannelsStable(const torch::Tensor& tensor)
{
    // 非有限值先清零，再做尺度归一化，避免损坏 checkpoint 在推理端继续传播 NaN。
    const auto finite = finiteOrZero(tensor);
    const auto scale = finite.detach().abs().amax({1}, true).clamp_min(1.0);
    const auto scaled = finite / scale;
    return scaled / scaled.norm(2, 1, true).clamp_min(1.0e-3);
}

torch::nn::Sequential makeStage(int64_t input_channels, int64_t output_channels)
{
    return torch::nn::Sequential(
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(input_channels, output_channels, 3).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(output_channels), torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(output_channels, output_channels, 3).padding(1).bias(false)),
        torch::nn::BatchNorm2d(output_channels), torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
}

torch::nn::Sequential makeStageRefinement(int64_t channels)
{
    return torch::nn::Sequential(PfmV21ZeroResidualContextBlock(channels), PfmV21ZeroResidualContextBlock(channels),
                                 PfmV21ZeroResidualContextBlock(channels, 2));
}

void zeroModule(torch::nn::Module& module)
{
    // 零初始化残差支路让新模块初始接近恒等映射，便于加载旧 checkpoint 后稳定微调。
    torch::NoGradGuard no_grad;
    for (auto& parameter : module.parameters())
    {
        parameter.zero_();
    }
}

void zeroConv(torch::nn::Conv2d& module)
{
    torch::NoGradGuard no_grad;
    module->weight.zero_();
    if (module->bias.defined())
    {
        module->bias.zero_();
    }
}

void initConcatIdentityProjection(torch::nn::Conv2d& module, int64_t descriptor_dim)
{
    // 融合适配器输入是 learned/texture 拼接，初始化为直接保留 learned descriptor。
    torch::NoGradGuard no_grad;
    module->weight.zero_();
    if (module->bias.defined())
    {
        module->bias.zero_();
    }
    for (int64_t channel = 0; channel < descriptor_dim; ++channel)
    {
        module->weight.index_put_({channel, channel, 0, 0}, 1.0);
    }
}

void zeroSequentialChild(torch::nn::Sequential& sequence, const std::string& child_name)
{
    for (const auto& child : sequence->named_children())
    {
        if (child.key() == child_name)
        {
            zeroModule(*child.value());
            return;
        }
    }
    throw std::invalid_argument("missing sequential child " + child_name);
}

torch::Tensor rotateFeatureMap(const torch::Tensor& tensor, int64_t turns)
{
    const auto normalized_turns = ((turns % 4) + 4) % 4;
    if (normalized_turns == 0)
    {
        return tensor;
    }
    return torch::rot90(tensor, normalized_turns, {2, 3}).contiguous();
}

torch::Tensor alignDescriptorOrientationChannels(const torch::Tensor& tensor, int64_t turns)
{
    // 描述子通道按四个方向分组，旋转特征图时同步滚动通道，保持方向语义对齐。
    const auto channels = tensor.size(1);
    if (channels < 4 || channels % 4 != 0)
    {
        return tensor;
    }
    const auto shift = channels / 4;
    return torch::roll(tensor, {-turns * shift}, {1});
}

class PfmV21DescriptorResidualBlockImpl : public torch::nn::Module
{
  public:
    explicit PfmV21DescriptorResidualBlockImpl(int64_t channels)
    {
        _conv1 =
            register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
        _conv2 =
            register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
    }

    torch::Tensor forward(const torch::Tensor& x)
    {
        auto hidden = torch::relu(_conv1->forward(x));
        return torch::relu(x + _conv2->forward(hidden));
    }

  private:
    torch::nn::Conv2d _conv1{nullptr};
    torch::nn::Conv2d _conv2{nullptr};
};

TORCH_MODULE(PfmV21DescriptorResidualBlock);

torch::nn::Sequential makeDescriptorTower(int64_t input_channels, int64_t descriptor_dim)
{
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, input_channels, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)), PfmV21DescriptorResidualBlock(input_channels),
        PfmV21DescriptorResidualBlock(input_channels), PfmV21DescriptorResidualBlock(input_channels),
        PfmV21DescriptorResidualBlock(input_channels),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, descriptor_dim, 1)));
}

torch::Tensor makeMultiscaleDescriptorContext(const torch::Tensor& context)
{
    // 局部和较大窗口平均池化提供多尺度纹理上下文，帮助弱纹理区域描述子稳定。
    auto local = torch::avg_pool2d(context, {3, 3}, {1, 1}, {1, 1}, false, false);
    auto wider = torch::avg_pool2d(context, {5, 5}, {1, 1}, {2, 2}, false, false);
    return torch::cat({context, local, wider}, 1);
}

torch::Tensor interpolateTo(const torch::Tensor& tensor, int64_t height, int64_t width, const char* mode)
{
    if (std::string(mode) == "nearest")
    {
        return torch::nn::functional::interpolate(tensor, torch::nn::functional::InterpolateFuncOptions()
                                                              .size(std::vector<int64_t>{height, width})
                                                              .mode(torch::kNearest));
    }
    return torch::nn::functional::interpolate(tensor, torch::nn::functional::InterpolateFuncOptions()
                                                          .size(std::vector<int64_t>{height, width})
                                                          .mode(torch::kBilinear)
                                                          .align_corners(false));
}

torch::Tensor conv1x1ChannelSlice(torch::nn::Conv2d& projection, const torch::Tensor& input, int64_t channel_offset,
                                  bool include_bias)
{
    using torch::indexing::Slice;

    const auto channels = input.size(1);
    const auto weight =
        projection->weight.index({Slice(), Slice(channel_offset, channel_offset + channels), Slice(), Slice()});
    const auto bias = include_bias && projection->bias.defined() ? std::optional<torch::Tensor>(projection->bias)
                                                                 : std::optional<torch::Tensor>();
    return torch::conv2d(input, weight, bias, std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0},
                         std::vector<int64_t>{1, 1}, 1);
}

torch::Tensor applyAnisotropicViewpointProjection(torch::nn::Conv2d& projection, const torch::Tensor& context)
{
    // 横向/纵向大核上下文模拟视角拉伸下的各向异性纹理支持区域。
    const auto channels = context.size(1);
    auto result = conv1x1ChannelSlice(projection, context, 0, true);
    auto horizontal = torch::avg_pool2d(context, {1, 7}, {1, 1}, {0, 3}, false, false);
    result = result + conv1x1ChannelSlice(projection, horizontal, channels, false);
    auto vertical = torch::avg_pool2d(context, {7, 1}, {1, 1}, {3, 0}, false, false);
    result = result + conv1x1ChannelSlice(projection, vertical, channels * 2, false);
    auto wide_horizontal = torch::avg_pool2d(context, {3, 9}, {1, 1}, {1, 4}, false, false);
    result = result + conv1x1ChannelSlice(projection, wide_horizontal, channels * 3, false);
    auto wide_vertical = torch::avg_pool2d(context, {9, 3}, {1, 1}, {4, 1}, false, false);
    return result + conv1x1ChannelSlice(projection, wide_vertical, channels * 4, false);
}

torch::Tensor makeXyGrid(int64_t height, int64_t width, const c10::Device& device, c10::ScalarType dtype)
{
    auto y = torch::arange(height, torch::TensorOptions().device(device).dtype(dtype));
    auto x = torch::arange(width, torch::TensorOptions().device(device).dtype(dtype));
    auto mesh = torch::meshgrid({y, x}, "ij");
    return torch::stack({mesh[1], mesh[0]}, -1);
}

torch::Tensor geometryAwareDescriptorPool(const torch::Tensor& descriptors, const torch::Tensor& orientation,
                                          const torch::Tensor& scale, const torch::Tensor& affine, double radius = 0.75)
{
    using torch::indexing::Slice;

    if (descriptors.dim() != 4)
    {
        throw std::invalid_argument("descriptors must have shape BxDxHxW");
    }
    const auto batch = descriptors.size(0);
    const auto height = descriptors.size(2);
    const auto width = descriptors.size(3);
    if (height <= 1 || width <= 1)
    {
        return normalizeChannelsStable(descriptors);
    }
    auto base_xy = makeXyGrid(height, width, descriptors.device(), descriptors.dtype().toScalarType())
                       .permute({2, 0, 1})
                       .unsqueeze(0)
                       .expand({batch, 2, height, width});
    auto ori = torch::nn::functional::normalize(orientation.to(descriptors.dtype()),
                                                torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-6));
    auto tangent = ori;
    auto normal = torch::stack({-ori.index({Slice(), 1}), ori.index({Slice(), 0})}, 1);
    auto clamped_scale = scale.to(descriptors.dtype()).clamp(0.5, 2.0);
    auto a00 = affine.index({Slice(), Slice(0, 1)}).to(descriptors.dtype());
    auto a01 = affine.index({Slice(), Slice(1, 2)}).to(descriptors.dtype());
    auto a10 = affine.index({Slice(), Slice(2, 3)}).to(descriptors.dtype());
    auto a11 = affine.index({Slice(), Slice(3, 4)}).to(descriptors.dtype());

    auto sample = [&](const torch::Tensor& offset_x, const torch::Tensor& offset_y)
    {
        auto warped_x = a00 * offset_x + a01 * offset_y;
        auto warped_y = a10 * offset_x + a11 * offset_y;
        auto sample_xy = base_xy + torch::cat({warped_x, warped_y}, 1);
        auto grid_x =
            sample_xy.index({Slice(), 0}) * (2.0 / static_cast<double>(std::max<int64_t>(1, width - 1))) - 1.0;
        auto grid_y =
            sample_xy.index({Slice(), 1}) * (2.0 / static_cast<double>(std::max<int64_t>(1, height - 1))) - 1.0;
        auto grid = torch::stack({grid_x, grid_y}, -1);
        return torch::nn::functional::grid_sample(descriptors, grid,
                                                  torch::nn::functional::GridSampleFuncOptions()
                                                      .mode(torch::kBilinear)
                                                      .padding_mode(torch::kBorder)
                                                      .align_corners(true));
    };

    auto step = clamped_scale * radius;
    auto zero = torch::zeros_like(step);
    std::vector<torch::Tensor> samples;
    samples.reserve(5);
    samples.push_back(sample(zero, zero));
    samples.push_back(
        sample(tangent.index({Slice(), Slice(0, 1)}) * step, tangent.index({Slice(), Slice(1, 2)}) * step));
    samples.push_back(
        sample(-tangent.index({Slice(), Slice(0, 1)}) * step, -tangent.index({Slice(), Slice(1, 2)}) * step));
    samples.push_back(sample(normal.index({Slice(), Slice(0, 1)}) * step, normal.index({Slice(), Slice(1, 2)}) * step));
    samples.push_back(
        sample(-normal.index({Slice(), Slice(0, 1)}) * step, -normal.index({Slice(), Slice(1, 2)}) * step));
    auto pooled = torch::stack(samples, 0).mean(0);
    return normalizeChannelsStable(0.5 * descriptors + 0.5 * pooled);
}

torch::Tensor shiftedFeature(const torch::Tensor& feature, int64_t dy, int64_t dx)
{
    using torch::indexing::Slice;

    auto shifted = torch::zeros_like(feature);
    const auto height = feature.size(2);
    const auto width = feature.size(3);
    const auto source_y0 = std::max<int64_t>(0, -dy);
    const auto source_y1 = std::min<int64_t>(height, height - dy);
    const auto source_x0 = std::max<int64_t>(0, -dx);
    const auto source_x1 = std::min<int64_t>(width, width - dx);
    if (source_y0 >= source_y1 || source_x0 >= source_x1)
    {
        return shifted;
    }
    const auto target_y0 = source_y0 + dy;
    const auto target_y1 = source_y1 + dy;
    const auto target_x0 = source_x0 + dx;
    const auto target_x1 = source_x1 + dx;
    shifted.index_put_({Slice(), Slice(), Slice(target_y0, target_y1), Slice(target_x0, target_x1)},
                       feature.index({Slice(), Slice(), Slice(source_y0, source_y1), Slice(source_x0, source_x1)}));
    return shifted;
}

torch::Tensor localCorrelation(const torch::Tensor& feature_a, const torch::Tensor& feature_b)
{
    std::vector<torch::Tensor> correlations;
    correlations.reserve(CORRELATION_CHANNELS);
    for (int64_t dy = -CORRELATION_RADIUS; dy <= CORRELATION_RADIUS; ++dy)
    {
        for (int64_t dx = -CORRELATION_RADIUS; dx <= CORRELATION_RADIUS; ++dx)
        {
            correlations.push_back((feature_a * shiftedFeature(feature_b, dy, dx)).mean(1, true));
        }
    }
    return torch::cat(correlations, 1);
}

torch::Tensor attend(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
                     int64_t hidden_dim)
{
    auto logits = torch::matmul(query, key.transpose(0, 1)) / std::sqrt(static_cast<double>(hidden_dim));
    return torch::matmul(torch::softmax(logits, 1), value);
}

torch::Tensor prepareKeypointsForEmbedding(const torch::Tensor& keypoints, int64_t meta_dim)
{
    requirePositive(meta_dim, "meta_dim");
    auto prepared = keypoints.to(torch::TensorOptions().device(keypoints.device()).dtype(torch::kFloat32));
    if (prepared.size(0) == 0)
    {
        return prepared.new_empty({0, meta_dim});
    }
    auto min_xy = std::get<0>(prepared.min(0, true));
    auto max_xy = std::get<0>(prepared.max(0, true));
    auto center = (min_xy + max_xy) * 0.5;
    auto span = std::get<0>((max_xy - min_xy).max(1, true)).clamp_min(1.0e-6);
    auto centered = (prepared - center) * 2.0 / span;
    auto radius = centered.pow(2).sum(1, true).sqrt();
    if (meta_dim == 1)
    {
        return radius;
    }
    auto legacy = torch::cat({radius, radius.pow(2)}, 1);
    if (meta_dim == 2)
    {
        return legacy;
    }
    auto spatial = torch::cat({centered, legacy}, 1);
    if (meta_dim <= spatial.size(1))
    {
        return spatial.index({torch::indexing::Slice(), torch::indexing::Slice(0, meta_dim)});
    }
    return torch::cat({spatial, spatial.new_zeros({spatial.size(0), meta_dim - spatial.size(1)})}, 1);
}

torch::Tensor metaColumn(const torch::Tensor& meta, int64_t index, double default_value = 0.0)
{
    if (meta.size(1) <= index)
    {
        return meta.new_full({meta.size(0)}, default_value);
    }
    return meta.index({torch::indexing::Slice(), index});
}

void validateConfig(const PfmV21Config& config)
{
    requirePositive(config.input_channels, "input_channels");
    requirePositive(config.base_channels, "base_channels");
    requirePositive(config.descriptor_dim, "descriptor_dim");
    requirePositive(config.graph_hidden_dim, "graph_hidden_dim");
    requirePositive(config.graph_attention_layers, "graph_attention_layers");
    requirePositive(config.graph_keypoint_meta_dim, "graph_keypoint_meta_dim");
}

} // namespace

PfmV21ZeroResidualContextBlockImpl::PfmV21ZeroResidualContextBlockImpl(int64_t channels, int64_t dilation)
{
    requirePositive(channels, "channels");
    requirePositive(dilation, "dilation");
    _conv1 = register_module(
        "conv1", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(channels, channels, 3).padding(dilation).dilation(dilation).bias(false)));
    _norm1 = register_module("norm1", makeNorm(channels));
    _conv2 = register_module(
        "conv2", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(channels, channels, 3).padding(dilation).dilation(dilation).bias(false)));
    _norm2 = register_module("norm2", makeNorm(channels));
    {
        torch::NoGradGuard no_grad;
        _conv2->weight.zero_();
    }
}

torch::Tensor PfmV21ZeroResidualContextBlockImpl::forward(const torch::Tensor& x)
{
    auto hidden = torch::gelu(_norm1->forward(_conv1->forward(x)));
    return x + _norm2->forward(_conv2->forward(hidden));
}

PfmV21BackboneImpl::PfmV21BackboneImpl(int64_t input_channels, int64_t base_channels)
    : _input_channels(input_channels), _base_channels(base_channels)
{
    requirePositive(_input_channels, "input_channels");
    requirePositive(_base_channels, "base_channels");
    _stage1 = register_module("stage1", makeStage(_input_channels, _base_channels));
    _stage2 = register_module("stage2", makeStage(_base_channels, _base_channels * 2));
    _stage3 = register_module("stage3", makeStage(_base_channels * 2, _base_channels * 4));
    _stage4 = register_module("stage4", makeStage(_base_channels * 4, _base_channels * 8));
    _stage1_refine = register_module("stage1_refine", makeStageRefinement(_base_channels));
    _stage2_refine = register_module("stage2_refine", makeStageRefinement(_base_channels * 2));
    _stage3_refine = register_module("stage3_refine", makeStageRefinement(_base_channels * 4));
    _stage4_refine = register_module("stage4_refine", makeStageRefinement(_base_channels * 8));
}

std::vector<torch::Tensor> PfmV21BackboneImpl::forward(const torch::Tensor& x)
{
    if (!x.defined() || x.dim() != 4 || x.size(1) != _input_channels)
    {
        throw std::invalid_argument("input tensor must have shape BxCxHxW with the configured channel count");
    }
    auto clean = finiteOrZero(x);
    auto y1 = _stage1_refine->forward(_stage1->forward(clean));
    auto y2 = _stage2_refine->forward(_stage2->forward(y1));
    auto y3 = _stage3_refine->forward(_stage3->forward(y2));
    auto y4 = _stage4_refine->forward(_stage4->forward(y3));
    return {y1, y2, y3, y4};
}

void PfmV21BackboneImpl::sanitizeNonfiniteState()
{
    for (auto& item : named_buffers(true))
    {
        auto tensor = item.value();
        if (!tensor.defined() || !tensor.is_floating_point())
        {
            continue;
        }
        auto finite = torch::isfinite(tensor);
        if (finite.all().item<bool>())
        {
            continue;
        }
        const auto fill = item.key().find("running_var") != std::string::npos ? 1.0 : 0.0;
        tensor.masked_fill_(finite.logical_not(), fill);
    }
}

PfmV21DualFPNLiteImpl::PfmV21DualFPNLiteImpl(int64_t base_channels) : _base_channels(base_channels)
{
    requirePositive(_base_channels, "base_channels");
    const auto p2_channels = _base_channels * 2;
    _keypoint_from_stage3 = register_module(
        "keypoint_from_stage3", torch::nn::Conv2d(torch::nn::Conv2dOptions(_base_channels * 4, p2_channels, 1)));
    _descriptor_from_stage3 = register_module(
        "descriptor_from_stage3", torch::nn::Conv2d(torch::nn::Conv2dOptions(_base_channels * 4, p2_channels, 1)));
    _descriptor_from_stage4 = register_module(
        "descriptor_from_stage4", torch::nn::Conv2d(torch::nn::Conv2dOptions(_base_channels * 8, p2_channels, 1)));
    _keypoint_refine = register_module("keypoint_refine", PfmV21ZeroResidualContextBlock(p2_channels));
    _descriptor_refine =
        register_module("descriptor_refine", torch::nn::Sequential(PfmV21ZeroResidualContextBlock(p2_channels),
                                                                   PfmV21ZeroResidualContextBlock(p2_channels, 2)));
    zeroConv(_keypoint_from_stage3);
    zeroConv(_descriptor_from_stage3);
    zeroConv(_descriptor_from_stage4);
}

std::pair<torch::Tensor, torch::Tensor> PfmV21DualFPNLiteImpl::forward(const std::vector<torch::Tensor>& features)
{
    if (features.size() < 4)
    {
        throw std::invalid_argument("DualFPNLite requires backbone stages 1..4");
    }
    const auto& stage2 = features[1];
    auto stage3 =
        interpolateTo(_keypoint_from_stage3->forward(features[2]), stage2.size(2), stage2.size(3), "bilinear");
    auto p2_keypoint = _keypoint_refine->forward(stage2 + stage3);
    auto desc_stage3 =
        interpolateTo(_descriptor_from_stage3->forward(features[2]), stage2.size(2), stage2.size(3), "bilinear");
    auto desc_stage4 =
        interpolateTo(_descriptor_from_stage4->forward(features[3]), stage2.size(2), stage2.size(3), "bilinear");
    auto p2_descriptor = _descriptor_refine->forward(stage2 + desc_stage3 + desc_stage4);
    return {p2_keypoint, p2_descriptor};
}

PfmV21SparseHeadImpl::PfmV21SparseHeadImpl(int64_t input_channels, int64_t descriptor_dim)
    : _input_channels(input_channels), _descriptor_dim(descriptor_dim)
{
    requirePositive(_input_channels, "input_channels");
    requirePositive(_descriptor_dim, "descriptor_dim");
    _context = register_module(
        "context", torch::nn::Sequential(
                       torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _input_channels, 3).padding(1)),
                       torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
                       torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _input_channels, 3).padding(1)),
                       torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))));
    _keypoint_context = register_module("keypoint_context", PfmV21ZeroResidualContextBlock(_input_channels));
    _descriptor_context = register_module("descriptor_context", PfmV21ZeroResidualContextBlock(_input_channels));
    _geometry_context = register_module("geometry_context", PfmV21ZeroResidualContextBlock(_input_channels));
    _heatmap = register_module("heatmap", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
    _heatmap_viewpoint_context = register_module(
        "heatmap_viewpoint_context", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 5, 1, 1)));
    _keypoint_offsets =
        register_module("keypoint_offsets", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 2, 1)));
    _descriptors = register_module("descriptors", makeDescriptorTower(_input_channels, _descriptor_dim));
    _descriptor_multiscale = register_module(
        "descriptor_multiscale", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 3, _descriptor_dim, 1)));
    _descriptor_attention = register_module(
        "descriptor_attention", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 3, _descriptor_dim, 1)));
    _descriptor_viewpoint_context =
        register_module("descriptor_viewpoint_context",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 5, _descriptor_dim, 1)));
    _descriptor_viewpoint_attention =
        register_module("descriptor_viewpoint_attention",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 5, _descriptor_dim, 1)));
    _descriptor_orientation_alignment =
        register_module("descriptor_orientation_alignment",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim, _descriptor_dim, 1)));
    _descriptor_dilated_context = register_module(
        "descriptor_dilated_context",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim, _descriptor_dim, 3).padding(2).dilation(2)));
    _descriptor_branch_quality = register_module("descriptor_branch_quality",
                                                 torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim, 1, 1)));
    _descriptor_rotation_fusion =
        register_module("descriptor_rotation_fusion",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim * 2, _descriptor_dim, 1)));
    _descriptor_skip = register_module(
        "descriptor_skip", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _descriptor_dim, 1)));
    _scale = register_module("scale", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
    _orientation = register_module("orientation", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 2, 1)));
    _affine = register_module("affine", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 4, 1)));

    zeroConv(_heatmap_viewpoint_context);
    zeroConv(_keypoint_offsets);
    zeroConv(_descriptor_viewpoint_context);
    zeroConv(_descriptor_viewpoint_attention);
    zeroConv(_descriptor_orientation_alignment);
    zeroConv(_descriptor_dilated_context);
    zeroConv(_descriptor_branch_quality);
    initConcatIdentityProjection(_descriptor_rotation_fusion, _descriptor_dim);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PfmV21SparseHeadImpl::descriptorBranch(const torch::Tensor& keypoint_feature, const torch::Tensor& descriptor_feature)
{
    auto keypoint_shared_context = _context->forward(keypoint_feature);
    auto descriptor_shared_context = _context->forward(descriptor_feature);
    auto keypoint_context = _keypoint_context->forward(keypoint_shared_context);
    auto descriptor_context = _descriptor_context->forward(descriptor_shared_context);
    auto multiscale_context = makeMultiscaleDescriptorContext(descriptor_context);
    auto viewpoint_descriptor = applyAnisotropicViewpointProjection(_descriptor_viewpoint_context, descriptor_context);
    auto viewpoint_gate =
        1.0 + torch::sigmoid(applyAnisotropicViewpointProjection(_descriptor_viewpoint_attention, descriptor_context));
    auto descriptor_base = _descriptors->forward(descriptor_context) +
                           _descriptor_multiscale->forward(multiscale_context) + viewpoint_descriptor * viewpoint_gate +
                           _descriptor_skip->forward(descriptor_feature);
    auto descriptor_gated =
        descriptor_base * (1.0 + torch::sigmoid(_descriptor_attention->forward(multiscale_context)));
    auto heatmap = _heatmap->forward(keypoint_context) +
                   applyAnisotropicViewpointProjection(_heatmap_viewpoint_context, keypoint_context);
    auto keypoint_offsets = torch::tanh(_keypoint_offsets->forward(keypoint_context)) * 0.5;
    auto geometry_context = _geometry_context->forward(descriptor_shared_context);
    return {geometry_context, heatmap, descriptor_gated, keypoint_offsets};
}

PfmV21SparseHeadOutput PfmV21SparseHeadImpl::forward(const torch::Tensor& feature)
{
    return forward(feature, feature);
}

PfmV21SparseHeadOutput PfmV21SparseHeadImpl::forward(const torch::Tensor& feature,
                                                     const torch::Tensor& descriptor_feature)
{
    if (!feature.defined() || feature.dim() != 4 || feature.size(1) != _input_channels)
    {
        throw std::invalid_argument("feature tensor must have shape BxCxHxW with the configured channel count");
    }
    if (!descriptor_feature.defined() || descriptor_feature.dim() != 4 ||
        descriptor_feature.size(1) != _input_channels || descriptor_feature.size(2) != feature.size(2) ||
        descriptor_feature.size(3) != feature.size(3))
    {
        throw std::invalid_argument("descriptor_feature must have shape BxCxHxW matching the keypoint feature grid");
    }

    auto branch = descriptorBranch(feature, descriptor_feature);
    auto geometry_context = std::get<0>(branch);
    auto heatmap_sum = std::get<1>(branch);
    auto descriptor_gated = std::get<2>(branch);
    auto keypoint_offsets = std::get<3>(branch);
    std::vector<torch::Tensor> descriptor_branches;
    descriptor_branches.push_back(descriptor_gated + _descriptor_orientation_alignment->forward(descriptor_gated) +
                                  _descriptor_dilated_context->forward(descriptor_gated));

    for (int64_t turns = 1; turns < 4; ++turns)
    {
        auto rotated_feature = rotateFeatureMap(feature, turns);
        auto rotated_descriptor_feature = rotateFeatureMap(descriptor_feature, turns);
        auto rotated_branch = descriptorBranch(rotated_feature, rotated_descriptor_feature);
        auto rotated_heatmap = std::get<1>(rotated_branch);
        auto rotated_descriptor_gated = std::get<2>(rotated_branch);
        heatmap_sum = heatmap_sum + rotateFeatureMap(rotated_heatmap, -turns);
        auto rotated_descriptor = rotateFeatureMap(rotated_descriptor_gated, -turns);
        auto orientation_aligned = alignDescriptorOrientationChannels(rotated_descriptor, turns);
        descriptor_branches.push_back(rotated_descriptor +
                                      _descriptor_orientation_alignment->forward(orientation_aligned) +
                                      _descriptor_dilated_context->forward(rotated_descriptor));
    }

    auto descriptor_stack = torch::stack(descriptor_branches, 1);
    std::vector<torch::Tensor> quality_logits;
    quality_logits.reserve(descriptor_branches.size());
    for (const auto& descriptor_branch : descriptor_branches)
    {
        quality_logits.push_back(_descriptor_branch_quality->forward(descriptor_branch));
    }
    auto branch_weights = torch::softmax(torch::stack(quality_logits, 1), 1);
    auto descriptor_invariant = (descriptor_stack * branch_weights).sum(1);
    auto descriptor_equivariant = descriptor_branches.front();
    auto descriptor_sum =
        _descriptor_rotation_fusion->forward(torch::cat({descriptor_invariant, descriptor_equivariant}, 1));
    auto heatmap = torch::sigmoid(heatmap_sum / 4.0);
    auto descriptors = normalizeChannelsStable(descriptor_sum);
    auto scale = torch::exp(_scale->forward(geometry_context).clamp(-2.0, 2.0));
    auto orientation = normalizeChannelsStable(_orientation->forward(geometry_context));
    auto affine_delta = torch::tanh(_affine->forward(geometry_context)) * 0.1;
    auto identity = torch::tensor({1.0, 0.0, 0.0, 1.0}, affine_delta.options()).view({1, 4, 1, 1});
    auto affine = identity + affine_delta;
    descriptors = geometryAwareDescriptorPool(descriptors, orientation, scale, affine);
    return PfmV21SparseHeadOutput{heatmap, descriptors, scale, orientation, affine, keypoint_offsets};
}

PfmV21DenseHeadImpl::PfmV21DenseHeadImpl(int64_t feature_channels) : _feature_channels(feature_channels)
{
    requirePositive(_feature_channels, "feature_channels");
    _correlation_projection =
        register_module("correlation_projection",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(CORRELATION_CHANNELS, _feature_channels, 1)));
    const auto input_channels = _feature_channels * 4 + 2;
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

PfmV21DenseHeadOutput PfmV21DenseHeadImpl::forward(const torch::Tensor& feature_a, const torch::Tensor& feature_b)
{
    using torch::indexing::Slice;

    if (!feature_a.defined() || !feature_b.defined() || feature_a.dim() != 4 || feature_b.dim() != 4)
    {
        throw std::invalid_argument("feature tensors must have shape BxCxHxW");
    }
    if (feature_a.sizes() != feature_b.sizes() || feature_a.size(1) != _feature_channels)
    {
        throw std::invalid_argument("feature tensors must have matching shapes and configured channels");
    }
    const auto height = feature_a.size(2);
    const auto width = feature_a.size(3);
    auto coordinates = makeXyGrid(height, width, feature_a.device(), feature_a.dtype().toScalarType());
    coordinates.index_put_({Slice(), Slice(), 0},
                           coordinates.index({Slice(), Slice(), 0}) / std::max<int64_t>(1, width - 1) * 2.0 - 1.0);
    coordinates.index_put_({Slice(), Slice(), 1},
                           coordinates.index({Slice(), Slice(), 1}) / std::max<int64_t>(1, height - 1) * 2.0 - 1.0);
    auto coordinate_channels =
        coordinates.permute({2, 0, 1}).unsqueeze(0).expand({feature_a.size(0), 2, height, width});
    auto correlation = _correlation_projection->forward(localCorrelation(feature_a, feature_b));
    auto pair_feature =
        torch::cat({feature_a, feature_b, torch::abs(feature_a - feature_b), correlation, coordinate_channels}, 1);
    auto prediction = _predictor->forward(pair_feature);
    auto confidence = torch::sigmoid(prediction.index({Slice(), Slice(0, 1), Slice(), Slice()}));
    auto offsets = prediction.index({Slice(), Slice(1, 3), Slice(), Slice()});
    return PfmV21DenseHeadOutput{confidence, offsets};
}

PfmV21GraphAttentionLayerImpl::PfmV21GraphAttentionLayerImpl(int64_t hidden_dim) : _hidden_dim(hidden_dim)
{
    requirePositive(_hidden_dim, "hidden_dim");
    _self_query = register_module("self_query", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _self_key = register_module("self_key", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _self_value = register_module("self_value", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _self_output = register_module("self_output", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _cross_query = register_module("cross_query", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _cross_key = register_module("cross_key", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _cross_value = register_module("cross_value", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _cross_output = register_module("cross_output", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _self_norm = register_module("self_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({_hidden_dim})));
    _cross_norm = register_module("cross_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({_hidden_dim})));
    _feed_forward_norm =
        register_module("feed_forward_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({_hidden_dim})));
    _attention_dropout = register_module("attention_dropout", torch::nn::Dropout(0.1));
    _feed_forward = register_module(
        "feed_forward", torch::nn::Sequential(torch::nn::Linear(_hidden_dim, _hidden_dim * 2), torch::nn::GELU(),
                                              torch::nn::Linear(_hidden_dim * 2, _hidden_dim)));
}

std::pair<torch::Tensor, torch::Tensor> PfmV21GraphAttentionLayerImpl::forward(const torch::Tensor& features_a,
                                                                               const torch::Tensor& features_b)
{
    auto self_a = attend(_self_query->forward(features_a), _self_key->forward(features_a),
                         _self_value->forward(features_a), _hidden_dim);
    auto self_b = attend(_self_query->forward(features_b), _self_key->forward(features_b),
                         _self_value->forward(features_b), _hidden_dim);
    auto refined_a = _self_norm->forward(features_a + _attention_dropout->forward(_self_output->forward(self_a)));
    auto refined_b = _self_norm->forward(features_b + _attention_dropout->forward(_self_output->forward(self_b)));
    auto cross_a = attend(_cross_query->forward(refined_a), _cross_key->forward(refined_b),
                          _cross_value->forward(refined_b), _hidden_dim);
    auto cross_b = attend(_cross_query->forward(refined_b), _cross_key->forward(refined_a),
                          _cross_value->forward(refined_a), _hidden_dim);
    refined_a = _cross_norm->forward(refined_a + _attention_dropout->forward(_cross_output->forward(cross_a)));
    refined_b = _cross_norm->forward(refined_b + _attention_dropout->forward(_cross_output->forward(cross_b)));
    return {_feed_forward_norm->forward(refined_a + _feed_forward->forward(refined_a)),
            _feed_forward_norm->forward(refined_b + _feed_forward->forward(refined_b))};
}

PfmV21GraphMatcherImpl::PfmV21GraphMatcherImpl(int64_t descriptor_dim, int64_t hidden_dim, int64_t attention_layers,
                                               int64_t keypoint_meta_dim, int64_t candidate_topk)
    : _descriptor_dim(descriptor_dim), _hidden_dim(hidden_dim), _attention_layer_count(attention_layers),
      _keypoint_meta_dim(keypoint_meta_dim), _candidate_topk(candidate_topk)
{
    requirePositive(_descriptor_dim, "descriptor_dim");
    requirePositive(_hidden_dim, "hidden_dim");
    requirePositive(_attention_layer_count, "attention_layers");
    requirePositive(_keypoint_meta_dim, "keypoint_meta_dim");
    _descriptor_projection = register_module("descriptor_projection", torch::nn::Linear(_descriptor_dim, _hidden_dim));
    _keypoint_projection = register_module("keypoint_projection", torch::nn::Linear(_keypoint_meta_dim, _hidden_dim));
    _score_projection = register_module("score_projection", torch::nn::Linear(_hidden_dim, _hidden_dim));
    const auto small_hidden = std::max<int64_t>(16, _hidden_dim / 8);
    _geometry_bias =
        register_module("geometry_bias", torch::nn::Sequential(torch::nn::Linear(8, small_hidden), torch::nn::GELU(),
                                                               torch::nn::Linear(small_hidden, 1)));
    _accept_head =
        register_module("accept_head", torch::nn::Sequential(torch::nn::Linear(6, small_hidden), torch::nn::GELU(),
                                                             torch::nn::Linear(small_hidden, 1)));
    _logit_scale = register_parameter("logit_scale", torch::ones({1}) * std::sqrt(static_cast<double>(_hidden_dim)));
    _raw_score_temperature = register_parameter("raw_score_temperature", torch::tensor(0.10F, torch::kFloat32));
    _graph_delta_scale = register_parameter("graph_delta_scale", torch::tensor(0.20F, torch::kFloat32));
    _accept_logit_scale = register_parameter("accept_logit_scale", torch::tensor(0.10F, torch::kFloat32));
    _dustbin_bias = register_parameter("dustbin_bias", torch::zeros({1}));
    _attention_layers = register_module("attention_layers", torch::nn::ModuleList());
    for (int64_t index = 0; index < _attention_layer_count; ++index)
    {
        _attention_layers->push_back(PfmV21GraphAttentionLayer(_hidden_dim));
    }
    zeroSequentialChild(_geometry_bias, "2");
    zeroSequentialChild(_accept_head, "2");
}

torch::Tensor PfmV21GraphMatcherImpl::metadata(const torch::Tensor& keypoints_or_meta) const
{
    if (keypoints_or_meta.dim() != 2)
    {
        throw std::invalid_argument("graph matcher keypoints/meta must have shape NxC");
    }
    if (keypoints_or_meta.size(1) == _keypoint_meta_dim)
    {
        return keypoints_or_meta.to(torch::kFloat32);
    }
    if (keypoints_or_meta.size(1) < 2)
    {
        throw std::invalid_argument("graph matcher keypoints must contain at least x/y");
    }
    return prepareKeypointsForEmbedding(
        keypoints_or_meta.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}), _keypoint_meta_dim);
}

torch::Tensor PfmV21GraphMatcherImpl::geometryCompatibilityBias(const torch::Tensor& meta_a,
                                                                const torch::Tensor& meta_b)
{
    auto ax = metaColumn(meta_a, 0).unsqueeze(1);
    auto ay = metaColumn(meta_a, 1).unsqueeze(1);
    auto bx = metaColumn(meta_b, 0).unsqueeze(0);
    auto by = metaColumn(meta_b, 1).unsqueeze(0);
    auto score_delta = metaColumn(meta_a, 4).unsqueeze(1) - metaColumn(meta_b, 4).unsqueeze(0);
    auto scale_delta = metaColumn(meta_a, 5).unsqueeze(1) - metaColumn(meta_b, 5).unsqueeze(0);
    auto aox = metaColumn(meta_a, 6, 1.0).unsqueeze(1);
    auto aoy = metaColumn(meta_a, 7).unsqueeze(1);
    auto box = metaColumn(meta_b, 6, 1.0).unsqueeze(0);
    auto boy = metaColumn(meta_b, 7).unsqueeze(0);
    auto orientation_cos = (aox * box + aoy * boy).clamp(-1.0, 1.0);
    auto quality_pair = 0.5 * (metaColumn(meta_a, 12, 1.0).unsqueeze(1) + metaColumn(meta_b, 12, 1.0).unsqueeze(0));
    auto contrast_pair = 0.5 * (metaColumn(meta_a, 13).unsqueeze(1) + metaColumn(meta_b, 13).unsqueeze(0));
    auto dx = ax - bx;
    auto dy = ay - by;
    auto features = torch::stack(
        {
            dx,
            dy,
            torch::sqrt(dx.square() + dy.square()).clamp_max(4.0),
            score_delta,
            scale_delta,
            orientation_cos,
            quality_pair,
            contrast_pair,
        },
        -1);
    return _geometry_bias->forward(features).squeeze(-1);
}

torch::Tensor PfmV21GraphMatcherImpl::candidateMask(const torch::Tensor& desc_a, const torch::Tensor& desc_b) const
{
    const auto count_a = desc_a.size(0);
    const auto count_b = desc_b.size(0);
    if (_candidate_topk <= 0 || _candidate_topk >= count_b)
    {
        return torch::ones({count_a, count_b}, torch::TensorOptions().device(desc_a.device()).dtype(torch::kBool));
    }
    auto norm_a = torch::nn::functional::normalize(
        desc_a, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12));
    auto norm_b = torch::nn::functional::normalize(
        desc_b, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12));
    auto similarity = torch::matmul(norm_a, norm_b.transpose(0, 1));
    auto mask = torch::zeros({count_a, count_b}, torch::TensorOptions().device(desc_a.device()).dtype(torch::kBool));
    const auto row_k = std::min<int64_t>(_candidate_topk, count_b);
    auto row_indices = std::get<1>(similarity.topk(row_k, 1));
    mask.scatter_(1, row_indices, true);
    const auto col_k = std::min<int64_t>(_candidate_topk, count_a);
    auto col_indices = std::get<1>(similarity.topk(col_k, 0));
    mask.scatter_(0, col_indices, true);
    return mask;
}

torch::Tensor PfmV21GraphMatcherImpl::acceptanceLogits(const torch::Tensor& raw_similarity,
                                                       const torch::Tensor& graph_delta, const torch::Tensor& meta_a,
                                                       const torch::Tensor& meta_b)
{
    if (raw_similarity.numel() == 0)
    {
        return raw_similarity.new_empty(raw_similarity.sizes());
    }
    torch::Tensor row_margin;
    torch::Tensor col_margin;
    if (raw_similarity.size(1) > 1)
    {
        auto row_top2 = std::get<0>(raw_similarity.topk(2, 1));
        row_margin = (row_top2.index({torch::indexing::Slice(), 0}) - row_top2.index({torch::indexing::Slice(), 1}))
                         .clamp(0.0, 2.0);
    }
    else
    {
        row_margin = raw_similarity.new_zeros({raw_similarity.size(0)});
    }
    if (raw_similarity.size(0) > 1)
    {
        auto col_top2 = std::get<0>(raw_similarity.topk(2, 0));
        col_margin = (col_top2.index({0, torch::indexing::Slice()}) - col_top2.index({1, torch::indexing::Slice()}))
                         .clamp(0.0, 2.0);
    }
    else
    {
        col_margin = raw_similarity.new_zeros({raw_similarity.size(1)});
    }
    auto quality_pair = 0.5 * (metaColumn(meta_a, 12, 1.0).unsqueeze(1) + metaColumn(meta_b, 12, 1.0).unsqueeze(0));
    auto contrast_pair = 0.5 * (metaColumn(meta_a, 13).unsqueeze(1) + metaColumn(meta_b, 13).unsqueeze(0));
    auto features = torch::stack(
        {
            raw_similarity.clamp(-1.0, 1.0),
            row_margin.unsqueeze(1).expand_as(raw_similarity),
            col_margin.unsqueeze(0).expand_as(raw_similarity),
            graph_delta.detach().clamp(-20.0, 20.0) / 20.0,
            quality_pair.clamp(0.0, 1.0),
            contrast_pair.clamp(0.0, 1.0),
        },
        -1);
    return _accept_head->forward(features).squeeze(-1);
}

torch::Tensor PfmV21GraphMatcherImpl::provisionalPairLogits(const torch::Tensor& embed_a,
                                                            const torch::Tensor& embed_b,
                                                            const torch::Tensor& raw_similarity,
                                                            const torch::Tensor& meta_a,
                                                            const torch::Tensor& meta_b)
{
    return provisionalPairOutputs(embed_a, embed_b, raw_similarity, meta_a, meta_b).first;
}

std::pair<torch::Tensor, torch::Tensor> PfmV21GraphMatcherImpl::provisionalPairOutputs(
    const torch::Tensor& embed_a, const torch::Tensor& embed_b, const torch::Tensor& raw_similarity,
    const torch::Tensor& meta_a, const torch::Tensor& meta_b)
{
    auto projected_a = torch::nn::functional::normalize(_score_projection->forward(embed_a),
                                                        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto projected_b = torch::nn::functional::normalize(_score_projection->forward(embed_b),
                                                        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto graph_delta = torch::matmul(projected_a, projected_b.transpose(0, 1)) * _logit_scale.clamp(1.0, 100.0);
    graph_delta = graph_delta + geometryCompatibilityBias(meta_a, meta_b);
    auto accept_logits = acceptanceLogits(raw_similarity, graph_delta, meta_a, meta_b);
    auto raw_temperature = _raw_score_temperature.abs().clamp(0.03, 1.0);
    auto delta_scale = _graph_delta_scale.clamp(0.0, 2.0);
    auto accept_scale = _accept_logit_scale.clamp(0.0, 2.0);
    auto pair_logits = raw_similarity / raw_temperature + delta_scale * graph_delta + accept_scale * accept_logits;
    return {pair_logits, accept_logits};
}

torch::Tensor PfmV21GraphMatcherImpl::assignmentConfidence(const torch::Tensor& pair_logits)
{
    if (pair_logits.numel() == 0)
    {
        return torch::zeros({}, pair_logits.options());
    }
    auto row_confidence = std::get<0>(torch::softmax(pair_logits, 1).max(1));
    auto column_confidence = std::get<0>(torch::softmax(pair_logits, 0).max(0));
    return torch::minimum(row_confidence.mean(), column_confidence.mean());
}

std::pair<torch::Tensor, torch::Tensor> PfmV21GraphMatcherImpl::acceptanceKeepMasks(
    const torch::Tensor& accept_logits, double min_probability)
{
    if (accept_logits.numel() == 0)
    {
        auto keep_a = torch::zeros({accept_logits.size(0)},
                                   torch::TensorOptions().device(accept_logits.device()).dtype(torch::kBool));
        auto keep_b = torch::zeros({accept_logits.size(1)},
                                   torch::TensorOptions().device(accept_logits.device()).dtype(torch::kBool));
        return {keep_a, keep_b};
    }
    auto accept_probability = torch::sigmoid(accept_logits);
    auto keep_a = std::get<0>(accept_probability.max(1)) >= min_probability;
    auto keep_b = std::get<0>(accept_probability.max(0)) >= min_probability;
    return {keep_a, keep_b};
}

std::pair<torch::Tensor, torch::Tensor> PfmV21GraphMatcherImpl::acceptanceTopCountKeepMasks(
    const torch::Tensor& accept_logits, int64_t keep_count_a, int64_t keep_count_b)
{
    if (accept_logits.numel() == 0)
    {
        auto keep_a = torch::zeros({accept_logits.size(0)},
                                   torch::TensorOptions().device(accept_logits.device()).dtype(torch::kBool));
        auto keep_b = torch::zeros({accept_logits.size(1)},
                                   torch::TensorOptions().device(accept_logits.device()).dtype(torch::kBool));
        return {keep_a, keep_b};
    }

    auto accept_probability = torch::sigmoid(accept_logits);
    auto score_a = std::get<0>(accept_probability.max(1));
    auto score_b = std::get<0>(accept_probability.max(0));

    const auto top_mask = [](const torch::Tensor& scores, int64_t keep_count)
    {
        keep_count = std::min<int64_t>(scores.size(0), std::max<int64_t>(1, keep_count));
        if (keep_count >= scores.size(0))
        {
            return torch::ones_like(scores, scores.options().dtype(torch::kBool));
        }
        auto top_indices = std::get<1>(scores.topk(keep_count, 0, true, false));
        auto mask = torch::zeros_like(scores, scores.options().dtype(torch::kBool));
        mask.index_put_({top_indices}, true);
        return mask;
    };

    return {top_mask(score_a, keep_count_a), top_mask(score_b, keep_count_b)};
}

int64_t PfmV21GraphMatcherImpl::lastExecutedAttentionLayers() const
{
    return _last_executed_attention_layers;
}

PfmV21GraphMatcherOutput PfmV21GraphMatcherImpl::forward(const torch::Tensor& descriptors_a,
                                                         const torch::Tensor& keypoints_a,
                                                         const torch::Tensor& descriptors_b,
                                                         const torch::Tensor& keypoints_b,
                                                         bool apply_candidate_mask, double width_prune_min_score,
                                                         double early_stop_min_confidence,
                                                         int64_t max_attention_layers,
                                                         double max_attention_work_fraction,
                                                         double width_prune_keep_ratio)
{
    using torch::indexing::Slice;

    if (width_prune_min_score < -1.0)
    {
        throw std::invalid_argument("width_prune_min_score must be at least -1.0; -1 disables pruning");
    }
    if (early_stop_min_confidence < -1.0)
    {
        throw std::invalid_argument("early_stop_min_confidence must be at least -1.0; -1 disables early stopping");
    }
    if (max_attention_layers < 0)
    {
        throw std::invalid_argument("max_attention_layers must be nonnegative; 0 disables hard layer budget");
    }
    if (!std::isfinite(max_attention_work_fraction) || max_attention_work_fraction < 0.0 ||
        max_attention_work_fraction > 1.0)
    {
        throw std::invalid_argument("max_attention_work_fraction must be in [0, 1]");
    }
    if (!std::isfinite(width_prune_keep_ratio) || width_prune_keep_ratio < 0.0 || width_prune_keep_ratio > 1.0)
    {
        throw std::invalid_argument("width_prune_keep_ratio must be in [0, 1]");
    }
    if (descriptors_a.dim() != 2 || descriptors_b.dim() != 2)
    {
        throw std::invalid_argument("graph matcher descriptors must have shape NxD");
    }
    if (descriptors_a.size(0) != keypoints_a.size(0) || descriptors_b.size(0) != keypoints_b.size(0))
    {
        throw std::invalid_argument("graph matcher descriptor and keypoint counts must match");
    }
    auto desc_a = descriptors_a.to(torch::kFloat32);
    auto desc_b = descriptors_b.to(torch::kFloat32);
    auto kp_a = metadata(keypoints_a).to(desc_a.device());
    auto kp_b = metadata(keypoints_b).to(desc_b.device());
    auto raw_similarity_full = torch::matmul(
        torch::nn::functional::normalize(desc_a,
                                         torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12)),
        torch::nn::functional::normalize(desc_b, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12))
            .transpose(0, 1));

    const bool prune_enabled = width_prune_min_score > -1.0;
    const bool ratio_prune_enabled = width_prune_keep_ratio < 1.0;
    torch::Tensor keep_a;
    torch::Tensor keep_b;
    if (prune_enabled)
    {
        if (raw_similarity_full.numel() == 0)
        {
            keep_a = torch::zeros({desc_a.size(0)}, desc_a.options().dtype(torch::kBool));
            keep_b = torch::zeros({desc_b.size(0)}, desc_b.options().dtype(torch::kBool));
        }
        else
        {
            keep_a = std::get<0>(raw_similarity_full.max(1)) >= width_prune_min_score;
            keep_b = std::get<0>(raw_similarity_full.max(0)) >= width_prune_min_score;
        }
    }
    else
    {
        keep_a = torch::ones({desc_a.size(0)}, desc_a.options().dtype(torch::kBool));
        keep_b = torch::ones({desc_b.size(0)}, desc_b.options().dtype(torch::kBool));
    }

    auto indices_a = torch::nonzero(keep_a).flatten();
    auto indices_b = torch::nonzero(keep_b).flatten();
    auto desc_work_a = desc_a.index_select(0, indices_a);
    auto desc_work_b = desc_b.index_select(0, indices_b);
    auto kp_work_a = kp_a.index_select(0, indices_a);
    auto kp_work_b = kp_b.index_select(0, indices_b);
    const int64_t input_keypoints_a = descriptors_a.size(0);
    const int64_t input_keypoints_b = descriptors_b.size(0);
    const int64_t full_attention_work_units = input_keypoints_a * input_keypoints_b * _attention_layer_count;
    const int64_t max_attention_work_units =
        static_cast<int64_t>(std::floor(static_cast<double>(full_attention_work_units) * max_attention_work_fraction +
                                        1.0e-9));
    const bool work_budget_enabled = max_attention_work_fraction < 1.0;
    const int64_t keep_count_a =
        std::max<int64_t>(1, static_cast<int64_t>(std::ceil(input_keypoints_a * width_prune_keep_ratio)));
    const int64_t keep_count_b =
        std::max<int64_t>(1, static_cast<int64_t>(std::ceil(input_keypoints_b * width_prune_keep_ratio)));
    const bool restore_pruned_logits = prune_enabled || ratio_prune_enabled;
    int64_t attention_work_units = 0;

    torch::Tensor pair_logits;
    torch::Tensor accept_logits;
    if (desc_work_a.size(0) == 0 || desc_work_b.size(0) == 0)
    {
        _last_executed_attention_layers = 0;
        pair_logits = torch::full_like(raw_similarity_full, -1.0e4);
        accept_logits = torch::full_like(raw_similarity_full, -1.0e4);
    }
    else
    {
        auto embed_a =
            torch::relu(_descriptor_projection->forward(desc_work_a) + _keypoint_projection->forward(kp_work_a));
        auto embed_b =
            torch::relu(_descriptor_projection->forward(desc_work_b) + _keypoint_projection->forward(kp_work_b));
        auto raw_similarity = raw_similarity_full.index_select(0, indices_a).index_select(1, indices_b);
        _last_executed_attention_layers = 0;
        for (const auto& layer : *_attention_layers)
        {
            if (max_attention_layers > 0 && _last_executed_attention_layers >= max_attention_layers)
            {
                break;
            }
            const int64_t layer_work_units = embed_a.size(0) * embed_b.size(0);
            if (work_budget_enabled && attention_work_units + layer_work_units > max_attention_work_units)
            {
                break;
            }
            attention_work_units += layer_work_units;
            auto refined = layer->as<PfmV21GraphAttentionLayerImpl>()->forward(embed_a, embed_b);
            embed_a = refined.first;
            embed_b = refined.second;
            ++_last_executed_attention_layers;
            const bool can_run_more_layers =
                _last_executed_attention_layers < _attention_layer_count &&
                (max_attention_layers <= 0 || _last_executed_attention_layers < max_attention_layers);
            const bool can_adapt = can_run_more_layers;
            if (can_adapt && (prune_enabled || ratio_prune_enabled || early_stop_min_confidence > -1.0))
            {
                auto provisional_outputs = provisionalPairOutputs(embed_a, embed_b, raw_similarity, kp_work_a,
                                                                  kp_work_b);
                auto provisional_pair_logits = provisional_outputs.first;
                auto provisional_accept_logits = provisional_outputs.second;
                if (prune_enabled || ratio_prune_enabled)
                {
                    auto keep_work_a = torch::ones({embed_a.size(0)}, embed_a.options().dtype(torch::kBool));
                    auto keep_work_b = torch::ones({embed_b.size(0)}, embed_b.options().dtype(torch::kBool));
                    auto threshold_keep_a = keep_work_a;
                    auto threshold_keep_b = keep_work_b;
                    if (prune_enabled)
                    {
                        auto keep_masks = acceptanceKeepMasks(provisional_accept_logits, width_prune_min_score);
                        threshold_keep_a = keep_masks.first;
                        threshold_keep_b = keep_masks.second;
                    }
                    if (ratio_prune_enabled)
                    {
                        auto keep_masks =
                            acceptanceTopCountKeepMasks(provisional_accept_logits, keep_count_a, keep_count_b);
                        if (prune_enabled && threshold_keep_a.any().item<bool>() &&
                            threshold_keep_b.any().item<bool>())
                        {
                            auto combined_keep_a = threshold_keep_a.logical_and(keep_masks.first);
                            auto combined_keep_b = threshold_keep_b.logical_and(keep_masks.second);
                            if (combined_keep_a.any().item<bool>() && combined_keep_b.any().item<bool>())
                            {
                                keep_work_a = combined_keep_a;
                                keep_work_b = combined_keep_b;
                            }
                            else
                            {
                                keep_work_a = keep_masks.first;
                                keep_work_b = keep_masks.second;
                            }
                        }
                        else
                        {
                            keep_work_a = keep_masks.first;
                            keep_work_b = keep_masks.second;
                        }
                    }
                    else if (prune_enabled)
                    {
                        keep_work_a = threshold_keep_a;
                        keep_work_b = threshold_keep_b;
                    }
                    const bool has_a = keep_work_a.any().item<bool>();
                    const bool has_b = keep_work_b.any().item<bool>();
                    const bool keeps_all = keep_work_a.all().item<bool>() && keep_work_b.all().item<bool>();
                    if (has_a && has_b && !keeps_all)
                    {
                        auto local_indices_a = torch::nonzero(keep_work_a).flatten();
                        auto local_indices_b = torch::nonzero(keep_work_b).flatten();
                        indices_a = indices_a.index_select(0, local_indices_a);
                        indices_b = indices_b.index_select(0, local_indices_b);
                        desc_work_a = desc_work_a.index_select(0, local_indices_a);
                        desc_work_b = desc_work_b.index_select(0, local_indices_b);
                        kp_work_a = kp_work_a.index_select(0, local_indices_a);
                        kp_work_b = kp_work_b.index_select(0, local_indices_b);
                        embed_a = embed_a.index_select(0, local_indices_a);
                        embed_b = embed_b.index_select(0, local_indices_b);
                        raw_similarity = raw_similarity.index_select(0, local_indices_a).index_select(1,
                                                                                                       local_indices_b);
                    }
                }
                if (early_stop_min_confidence > -1.0 &&
                    assignmentConfidence(provisional_pair_logits).item<float>() >= early_stop_min_confidence)
                {
                    break;
                }
            }
        }
        embed_a = torch::nn::functional::normalize(_score_projection->forward(embed_a),
                                                   torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        embed_b = torch::nn::functional::normalize(_score_projection->forward(embed_b),
                                                   torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        auto graph_delta = torch::matmul(embed_a, embed_b.transpose(0, 1)) * _logit_scale.clamp(1.0, 100.0);
        graph_delta = graph_delta + geometryCompatibilityBias(kp_work_a, kp_work_b);
        auto accept_logits_work = acceptanceLogits(raw_similarity, graph_delta, kp_work_a, kp_work_b);
        auto raw_temperature = _raw_score_temperature.abs().clamp(0.03, 1.0);
        auto delta_scale = _graph_delta_scale.clamp(0.0, 2.0);
        auto accept_scale = _accept_logit_scale.clamp(0.0, 2.0);
        auto pair_logits_work =
            raw_similarity / raw_temperature + delta_scale * graph_delta + accept_scale * accept_logits_work;
        if (apply_candidate_mask)
        {
            auto mask = candidateMask(desc_work_a, desc_work_b);
            pair_logits_work = pair_logits_work.masked_fill(mask.logical_not(), -1.0e4);
            accept_logits_work = accept_logits_work.masked_fill(mask.logical_not(), -1.0e4);
        }
        if (restore_pruned_logits)
        {
            pair_logits = torch::full_like(raw_similarity_full, -1.0e4);
            accept_logits = torch::full_like(raw_similarity_full, -1.0e4);
            pair_logits.index_put_({indices_a.unsqueeze(1), indices_b.unsqueeze(0)}, pair_logits_work);
            accept_logits.index_put_({indices_a.unsqueeze(1), indices_b.unsqueeze(0)}, accept_logits_work);
        }
        else
        {
            pair_logits = pair_logits_work;
            accept_logits = accept_logits_work;
        }
    }
    auto logits =
        torch::zeros({descriptors_a.size(0) + 1, descriptors_b.size(0) + 1}, pair_logits.options()) + _dustbin_bias;
    logits.index_put_({Slice(0, descriptors_a.size(0)), Slice(0, descriptors_b.size(0))}, pair_logits);
    auto row_logits = logits.index({Slice(0, descriptors_a.size(0)), Slice()});
    auto row_prob = torch::softmax(row_logits, 1).index({Slice(), Slice(0, descriptors_b.size(0))});
    auto col_prob = torch::softmax(logits.index({Slice(), Slice(0, descriptors_b.size(0))}), 0)
                        .index({Slice(0, descriptors_a.size(0)), Slice()});
    auto dual_scores = row_prob * col_prob;
    auto best = dual_scores.max(1);
    auto best_values = std::get<0>(best);
    auto best_indices = std::get<1>(best);
    auto source_indices = torch::arange(descriptors_a.size(0), best_indices.options());
    auto inlier_mask = best_values.gt(torch::softmax(row_logits, 1).index({Slice(), -1}));
    if (descriptors_a.size(0) > 0 && descriptors_b.size(0) > 0)
    {
        auto reverse_best = std::get<1>(dual_scores.max(0));
        auto mutual_sources = reverse_best.index_select(0, best_indices.clamp(0, descriptors_b.size(0) - 1));
        inlier_mask = inlier_mask.logical_and(mutual_sources.eq(source_indices));
    }
    source_indices = source_indices.index({inlier_mask});
    auto target_indices = best_indices.index({inlier_mask});
    auto probabilities = best_values.index({inlier_mask});
    if (probabilities.numel() > 0 && accept_logits.numel() > 0)
    {
        probabilities = probabilities * torch::sigmoid(accept_logits.index({source_indices, target_indices}));
    }
    auto matches = torch::stack({source_indices, target_indices}, 1).to(torch::kCPU, torch::kInt64).contiguous();
    auto scores = probabilities.to(torch::kCPU, torch::kFloat32).contiguous();
    const int64_t kept_keypoints_a = indices_a.size(0);
    const int64_t kept_keypoints_b = indices_b.size(0);
    const double attention_work_fraction =
        full_attention_work_units == 0 ? 0.0
                                       : static_cast<double>(attention_work_units) /
                                             static_cast<double>(full_attention_work_units);
    return PfmV21GraphMatcherOutput{
        logits.contiguous(),
        matches,
        scores,
        accept_logits.contiguous(),
        _last_executed_attention_layers,
        input_keypoints_a,
        input_keypoints_b,
        kept_keypoints_a,
        kept_keypoints_b,
        std::max<int64_t>(0, input_keypoints_a - kept_keypoints_a),
        std::max<int64_t>(0, input_keypoints_b - kept_keypoints_b),
        attention_work_units,
        full_attention_work_units,
        attention_work_fraction,
    };
}

torch::Tensor makeRotationInvariantTextureDescriptor(const torch::Tensor& image, int64_t descriptor_height,
                                                     int64_t descriptor_width, int64_t descriptor_dim)
{
    using torch::indexing::Slice;

    auto base = finiteOrZero(image);
    if (base.size(1) != 1)
    {
        base = base.mean(1, true);
    }
    std::vector<torch::Tensor> channels{base};
    const auto height = base.size(2);
    const auto width = base.size(3);
    auto y = torch::arange(height, base.options()).view({1, 1, height, 1});
    auto x = torch::arange(width, base.options()).view({1, 1, 1, width});
    const auto center_y = (static_cast<double>(height) - 1.0) * 0.5;
    const auto center_x = (static_cast<double>(width) - 1.0) * 0.5;
    const auto max_radius = std::max(1.0, std::hypot(center_x, center_y));
    auto radius = torch::sqrt((x - center_x).pow(2) + (y - center_y).pow(2)) / max_radius;
    radius = radius.expand({base.size(0), 1, height, width}).contiguous();
    channels.push_back(radius);
    channels.push_back(radius.pow(2));
    channels.push_back(base * radius);

    auto local_mean = torch::avg_pool2d(base, {15, 15}, {1, 1}, {7, 7}, false, false);
    auto local_sq_mean = torch::avg_pool2d(base.square(), {15, 15}, {1, 1}, {7, 7}, false, false);
    auto local_std = (local_sq_mean - local_mean.square()).clamp_min(0.0).sqrt();
    auto local_normalized = (base - local_mean) / local_std.add(1.0e-3);
    channels.push_back(local_normalized);
    channels.push_back(local_std);
    channels.push_back((base - local_mean).abs());
    for (const auto kernel : {3, 7, 15, 31})
    {
        auto blur = torch::avg_pool2d(base, {kernel, kernel}, {1, 1}, {kernel / 2, kernel / 2}, false, false);
        channels.push_back(blur);
        channels.push_back((base - blur).abs());
    }
    auto dog_small = torch::avg_pool2d(base, {3, 3}, {1, 1}, {1, 1}, false, false) -
                     torch::avg_pool2d(base, {7, 7}, {1, 1}, {3, 3}, false, false);
    auto dog_large = torch::avg_pool2d(base, {7, 7}, {1, 1}, {3, 3}, false, false) -
                     torch::avg_pool2d(base, {21, 21}, {1, 1}, {10, 10}, false, false);
    auto laplacian = -4.0 * base + torch::roll(base, {1}, {2}) + torch::roll(base, {-1}, {2}) +
                     torch::roll(base, {1}, {3}) + torch::roll(base, {-1}, {3});
    channels.push_back(dog_small);
    channels.push_back(dog_large);
    channels.push_back(laplacian.abs());
    auto dx = (base - torch::roll(base, {1}, {3})).abs();
    auto dy = (base - torch::roll(base, {1}, {2})).abs();
    auto gradient = dx + dy;
    auto signed_dx = base - torch::roll(base, {1}, {3});
    auto signed_dy = base - torch::roll(base, {1}, {2});
    auto grad_norm = torch::sqrt(signed_dx.square() + signed_dy.square()).clamp_min(1.0e-6);
    channels.push_back(signed_dx / grad_norm);
    channels.push_back(signed_dy / grad_norm);
    channels.push_back(gradient);
    for (const auto kernel : {3, 7, 11})
    {
        channels.push_back(
            torch::avg_pool2d(gradient, {kernel, kernel}, {1, 1}, {kernel / 2, kernel / 2}, false, false));
    }
    for (const auto ring_radius : {1, 2, 4, 8})
    {
        std::vector<torch::Tensor> diffs;
        std::vector<torch::Tensor> signed_diffs;
        const std::array<std::pair<int64_t, int64_t>, 8> offsets{
            std::pair<int64_t, int64_t>{-ring_radius, 0},
            std::pair<int64_t, int64_t>{ring_radius, 0},
            std::pair<int64_t, int64_t>{0, -ring_radius},
            std::pair<int64_t, int64_t>{0, ring_radius},
            std::pair<int64_t, int64_t>{-ring_radius, -ring_radius},
            std::pair<int64_t, int64_t>{-ring_radius, ring_radius},
            std::pair<int64_t, int64_t>{ring_radius, -ring_radius},
            std::pair<int64_t, int64_t>{ring_radius, ring_radius},
        };
        for (const auto& offset : offsets)
        {
            auto shifted = torch::roll(base, {offset.first, offset.second}, {2, 3});
            auto signed_diff = base - shifted;
            signed_diffs.push_back(signed_diff);
            diffs.push_back(signed_diff.abs());
        }
        auto ring = torch::stack(diffs, 1);
        auto signed_ring = torch::stack(signed_diffs, 1);
        auto ring_mean = ring.mean(1);
        channels.push_back(ring_mean);
        channels.push_back(std::get<0>(ring.max(1)));
        auto centered_ring = ring - ring.mean(1, true);
        channels.push_back(centered_ring.pow(2).mean(1).sqrt());
        channels.push_back(ring_mean * radius);
        channels.push_back(torch::tanh(signed_ring * 8.0).mean(1));
        channels.push_back((signed_ring > 0.0).to(base.dtype()).mean(1) * 2.0 - 1.0);
    }
    channels.push_back(gradient * radius);
    auto target = torch::cat(channels, 1);
    target = interpolateTo(target, descriptor_height, descriptor_width, "bilinear");
    auto centered = target - target.mean({2, 3}, true);
    auto scaled = centered / centered.pow(2).mean({2, 3}, true).add(1.0e-4).sqrt();
    if (scaled.size(1) < descriptor_dim)
    {
        const auto repeat_count = (descriptor_dim + scaled.size(1) - 1) / scaled.size(1);
        scaled = scaled.repeat({1, repeat_count, 1, 1});
    }
    target = scaled.index({Slice(), Slice(0, descriptor_dim), Slice(), Slice()}).contiguous();
    return normalizeChannelsStable(target);
}

torch::Tensor makeRotationInvariantTextureSaliency(const torch::Tensor& image, int64_t target_height,
                                                   int64_t target_width)
{
    auto base = image;
    if (base.size(1) != 1)
    {
        base = base.mean(1, true);
    }
    auto blur = torch::avg_pool2d(base, {15, 15}, {1, 1}, {7, 7}, false, false);
    auto contrast = (base - blur).abs();
    auto dx = (base - torch::roll(base, {1}, {3})).abs();
    auto dy = (base - torch::roll(base, {1}, {2})).abs();
    auto saliency = torch::avg_pool2d(contrast + dx + dy, {5, 5}, {1, 1}, {2, 2}, false, false);
    saliency = interpolateTo(saliency, target_height, target_width, "bilinear");
    auto flat = saliency.reshape({saliency.size(0), saliency.size(1), -1});
    auto min_value = std::get<0>(flat.min(2, true)).reshape({saliency.size(0), saliency.size(1), 1, 1});
    auto max_value = std::get<0>(flat.max(2, true)).reshape({saliency.size(0), saliency.size(1), 1, 1});
    return (saliency - min_value) / (max_value - min_value).clamp_min(1.0e-6);
}

PfmV21TextureDescriptorAdapterImpl::PfmV21TextureDescriptorAdapterImpl(int64_t descriptor_dim)
    : _descriptor_dim(descriptor_dim)
{
    requirePositive(_descriptor_dim, "descriptor_dim");
    _residual =
        register_module("residual", torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim, _descriptor_dim, 1)));
    zeroConv(_residual);
}

torch::Tensor PfmV21TextureDescriptorAdapterImpl::forward(const torch::Tensor& texture)
{
    if (texture.dim() != 4 || texture.size(1) != _descriptor_dim)
    {
        throw std::invalid_argument("texture tensor must have shape BxDxHxW with the configured descriptor dimension");
    }
    return normalizeChannelsStable(texture + _residual->forward(texture));
}

PfmV21DescriptorFusionAdapterImpl::PfmV21DescriptorFusionAdapterImpl(int64_t descriptor_dim, int64_t hidden_dim)
    : _descriptor_dim(descriptor_dim),
      _hidden_dim(hidden_dim > 0 ? hidden_dim : std::max<int64_t>(16, descriptor_dim * 2))
{
    requirePositive(_descriptor_dim, "descriptor_dim");
    _input_projection = register_module(
        "input_projection", torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim * 4, _hidden_dim, 1)));
    _context = register_module(
        "context",
        torch::nn::Sequential(
            torch::nn::GELU(), torch::nn::Conv2d(torch::nn::Conv2dOptions(_hidden_dim, _hidden_dim, 3).padding(1)),
            torch::nn::GELU(), torch::nn::Conv2d(torch::nn::Conv2dOptions(_hidden_dim, _hidden_dim, 3).padding(1)),
            torch::nn::GELU()));
    _texture_gate =
        register_module("texture_gate", torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim * 4, 1, 1)));
    _output = register_module("output", torch::nn::Conv2d(torch::nn::Conv2dOptions(_hidden_dim, _descriptor_dim, 1)));
    zeroConv(_texture_gate);
    zeroConv(_output);
}

torch::Tensor PfmV21DescriptorFusionAdapterImpl::forward(const torch::Tensor& learned, const torch::Tensor& texture,
                                                         double blend_weight)
{
    if (learned.sizes() != texture.sizes() || learned.dim() != 4 || learned.size(1) != _descriptor_dim)
    {
        throw std::invalid_argument("learned and texture descriptors must have matching BxDxHxW shapes");
    }
    auto initial_weighted_texture = texture * blend_weight;
    auto gate_features = torch::cat(
        {learned, initial_weighted_texture, learned - initial_weighted_texture, learned * initial_weighted_texture}, 1);
    auto texture_gate = 1.0 + 0.5 * torch::tanh(_texture_gate->forward(gate_features));
    auto weighted_texture = initial_weighted_texture * texture_gate;
    auto base = normalizeChannelsStable(learned + weighted_texture);
    auto features = torch::cat({learned, weighted_texture, learned - weighted_texture, learned * weighted_texture}, 1);
    auto residual = _output->forward(_context->forward(_input_projection->forward(features)));
    return normalizeChannelsStable(base + residual);
}

PfmV21QualityHeadImpl::PfmV21QualityHeadImpl(int64_t descriptor_dim) : _descriptor_dim(descriptor_dim)
{
    requirePositive(_descriptor_dim, "descriptor_dim");
    const auto hidden = std::max<int64_t>(16, _descriptor_dim / 2);
    _predictor = register_module(
        "predictor", torch::nn::Sequential(
                         torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim + 3, hidden, 1)), torch::nn::GELU(),
                         torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden, hidden, 3).padding(1)), torch::nn::GELU(),
                         torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden, 1, 1))));
    zeroSequentialChild(_predictor, "4");
}

torch::Tensor PfmV21QualityHeadImpl::forward(const torch::Tensor& descriptors, const torch::Tensor& heatmap,
                                             const torch::Tensor& texture_saliency,
                                             const torch::Tensor& dense_confidence)
{
    if (descriptors.dim() != 4)
    {
        throw std::invalid_argument("descriptors must have shape BxDxHxW");
    }
    std::vector<torch::Tensor> auxiliaries;
    for (const auto& tensor : {heatmap, texture_saliency, dense_confidence})
    {
        if (tensor.dim() != 4 || tensor.size(0) != descriptors.size(0) || tensor.size(1) != 1)
        {
            throw std::invalid_argument("quality auxiliary maps must have shape Bx1xHxW");
        }
        auto auxiliary = tensor;
        if (auxiliary.size(2) != descriptors.size(2) || auxiliary.size(3) != descriptors.size(3))
        {
            auxiliary = interpolateTo(auxiliary, descriptors.size(2), descriptors.size(3), "bilinear");
        }
        auxiliaries.push_back(auxiliary.to(descriptors.dtype()));
    }
    auto logits = _predictor->forward(torch::cat({descriptors, auxiliaries[0], auxiliaries[1], auxiliaries[2]}, 1));
    return torch::sigmoid(logits + 0.5 * auxiliaries[0] + 0.5 * auxiliaries[1] + 0.25 * auxiliaries[2]);
}

PfmV21SemiDenseCandidateBranchImpl::PfmV21SemiDenseCandidateBranchImpl(int64_t descriptor_dim, int64_t projection_dim,
                                                                       int64_t max_grid)
    : _descriptor_dim(descriptor_dim), _projection_dim(projection_dim), _max_grid(max_grid)
{
    requirePositive(_descriptor_dim, "descriptor_dim");
    requirePositive(_projection_dim, "projection_dim");
    requirePositive(_max_grid, "max_grid");
    _projection = register_module(
        "projection",
        torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim, _projection_dim, 1)),
                              torch::nn::GELU(),
                              torch::nn::Conv2d(torch::nn::Conv2dOptions(_projection_dim, _projection_dim, 1))));
}

torch::Tensor PfmV21SemiDenseCandidateBranchImpl::coarse(const torch::Tensor& descriptors)
{
    if (descriptors.dim() != 4 || descriptors.size(1) != _descriptor_dim)
    {
        throw std::invalid_argument("semi-dense descriptors must have shape BxDxHxW");
    }
    const auto height = descriptors.size(2);
    const auto width = descriptors.size(3);
    const auto target_height = std::min<int64_t>(height, _max_grid);
    const auto target_width = std::min<int64_t>(width, _max_grid);
    auto coarse_descriptors = descriptors;
    if (target_height != height || target_width != width)
    {
        coarse_descriptors = torch::adaptive_avg_pool2d(coarse_descriptors, {target_height, target_width});
    }
    return torch::nn::functional::normalize(_projection->forward(coarse_descriptors),
                                            torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1.0e-12));
}

PfmV21SemiDenseCandidateOutput PfmV21SemiDenseCandidateBranchImpl::forward(const torch::Tensor& descriptors_a,
                                                                           const torch::Tensor& descriptors_b,
                                                                           int64_t max_candidates, double min_score)
{
    if (descriptors_a.size(0) != 1 || descriptors_b.size(0) != 1)
    {
        throw std::invalid_argument("semi-dense candidate branch currently expects single-pair descriptor maps");
    }
    if (max_candidates <= 0)
    {
        auto empty_xy = descriptors_a.new_empty({0, 2});
        return PfmV21SemiDenseCandidateOutput{empty_xy, empty_xy.clone(), descriptors_a.new_empty({0})};
    }
    auto coarse_a = coarse(descriptors_a);
    auto coarse_b = coarse(descriptors_b);
    const auto channels = coarse_a.size(1);
    const auto coarse_ha = coarse_a.size(2);
    const auto coarse_wa = coarse_a.size(3);
    const auto coarse_hb = coarse_b.size(2);
    const auto coarse_wb = coarse_b.size(3);
    auto flat_a = coarse_a.squeeze(0).permute({1, 2, 0}).reshape({-1, channels});
    auto flat_b = coarse_b.squeeze(0).permute({1, 2, 0}).reshape({-1, channels});
    auto logits = torch::matmul(flat_a, flat_b.transpose(0, 1)) / std::sqrt(static_cast<double>(channels));
    auto dual_scores = torch::softmax(logits, 1) * torch::softmax(logits, 0);
    auto flat_scores = dual_scores.reshape({-1});
    const auto candidate_count = std::min<int64_t>(max_candidates, flat_scores.numel());
    if (candidate_count == 0)
    {
        auto empty_xy = descriptors_a.new_empty({0, 2});
        return PfmV21SemiDenseCandidateOutput{empty_xy, empty_xy.clone(), descriptors_a.new_empty({0})};
    }
    auto top = flat_scores.topk(candidate_count);
    auto values = std::get<0>(top);
    auto indices = std::get<1>(top);
    auto keep = values >= min_score;
    values = values.index({keep});
    indices = indices.index({keep});
    if (values.numel() == 0)
    {
        auto empty_xy = descriptors_a.new_empty({0, 2});
        return PfmV21SemiDenseCandidateOutput{empty_xy, empty_xy.clone(), values};
    }
    auto source = torch::div(indices, flat_b.size(0), "floor");
    auto target = indices.remainder(flat_b.size(0));
    auto source_y = torch::div(source, coarse_wa, "floor").to(descriptors_a.dtype());
    auto source_x = source.remainder(coarse_wa).to(descriptors_a.dtype());
    auto target_y = torch::div(target, coarse_wb, "floor").to(descriptors_b.dtype());
    auto target_x = target.remainder(coarse_wb).to(descriptors_b.dtype());
    auto scale_coords = [](const torch::Tensor& x, const torch::Tensor& y, int64_t coarse_h, int64_t coarse_w,
                           int64_t full_h, int64_t full_w)
    {
        auto scaled_x = x;
        auto scaled_y = y;
        if (coarse_w > 1)
        {
            scaled_x =
                scaled_x * static_cast<double>(std::max<int64_t>(1, full_w - 1)) / static_cast<double>(coarse_w - 1);
        }
        if (coarse_h > 1)
        {
            scaled_y =
                scaled_y * static_cast<double>(std::max<int64_t>(1, full_h - 1)) / static_cast<double>(coarse_h - 1);
        }
        return torch::stack({scaled_x, scaled_y}, 1);
    };
    auto keypoints_a =
        scale_coords(source_x, source_y, coarse_ha, coarse_wa, descriptors_a.size(2), descriptors_a.size(3));
    auto keypoints_b =
        scale_coords(target_x, target_y, coarse_hb, coarse_wb, descriptors_b.size(2), descriptors_b.size(3));
    return PfmV21SemiDenseCandidateOutput{keypoints_a.contiguous(), keypoints_b.contiguous(), values.contiguous()};
}

PfmV21FeatureMatcherImpl::PfmV21FeatureMatcherImpl(const PfmV21Config& config) : _config(config)
{
    validateConfig(_config);
    _backbone = register_module("backbone", PfmV21Backbone(_config.input_channels, _config.base_channels));
    _dual_fpn = register_module("dual_fpn", PfmV21DualFPNLite(_config.base_channels));
    _sparse_head = register_module("sparse_head", PfmV21SparseHead(_config.base_channels * 2, _config.descriptor_dim));
    _texture_adapter = register_module("texture_adapter", PfmV21TextureDescriptorAdapter(_config.descriptor_dim));
    _descriptor_fusion = register_module("descriptor_fusion", PfmV21DescriptorFusionAdapter(_config.descriptor_dim));
    _dense_head = register_module("dense_head", PfmV21DenseHead(_config.base_channels));
    _quality_head = register_module("quality_head", PfmV21QualityHead(_config.descriptor_dim));
    _semi_dense_branch = register_module("semi_dense_branch", PfmV21SemiDenseCandidateBranch(_config.descriptor_dim));
    _graph_matcher = register_module(
        "graph_matcher", PfmV21GraphMatcher(_config.descriptor_dim, _config.graph_hidden_dim,
                                            _config.graph_attention_layers, _config.graph_keypoint_meta_dim));
}

const PfmV21Config& PfmV21FeatureMatcherImpl::config() const
{
    return _config;
}

torch::Tensor PfmV21FeatureMatcherImpl::learnedDescriptorMapSingle(const torch::Tensor& image)
{
    auto features = _backbone->forward(image);
    auto fpn = _dual_fpn->forward(features);
    auto sparse = _sparse_head->forward(fpn.first, fpn.second);
    return sparse.descriptors;
}

torch::Tensor PfmV21FeatureMatcherImpl::rawTextureDescriptorMapSingle(const torch::Tensor& image)
{
    if (image.dim() != 4)
    {
        throw std::invalid_argument("image must have shape BxCxHxW");
    }
    const auto descriptor_height = std::max<int64_t>(1, (image.size(2) + 3) / 4);
    const auto descriptor_width = std::max<int64_t>(1, (image.size(3) + 3) / 4);
    return makeRotationInvariantTextureDescriptor(image, descriptor_height, descriptor_width, _config.descriptor_dim);
}

torch::Tensor PfmV21FeatureMatcherImpl::textureDescriptorMapSingle(const torch::Tensor& image)
{
    return _texture_adapter->forward(rawTextureDescriptorMapSingle(image));
}

torch::Tensor PfmV21FeatureMatcherImpl::fuseDescriptorMaps(const torch::Tensor& learned_descriptors,
                                                           const torch::Tensor& image, double texture_blend_weight)
{
    auto texture = _texture_adapter->forward(makeRotationInvariantTextureDescriptor(
        image, learned_descriptors.size(2), learned_descriptors.size(3), learned_descriptors.size(1)));
    return _descriptor_fusion->forward(learned_descriptors, texture, texture_blend_weight);
}

torch::Tensor PfmV21FeatureMatcherImpl::descriptorMapSingle(const torch::Tensor& image, double texture_blend_weight)
{
    auto features = _backbone->forward(image);
    auto fpn = _dual_fpn->forward(features);
    auto sparse = _sparse_head->forward(fpn.first, fpn.second);
    return fuseDescriptorMaps(sparse.descriptors, image, texture_blend_weight);
}

PfmV21RawFeatureMaps PfmV21FeatureMatcherImpl::forwardSingle(const torch::Tensor& image, double texture_blend_weight)
{
    if (image.dim() != 4)
    {
        throw std::invalid_argument("image must have shape BxCxHxW");
    }
    auto features = _backbone->forward(image);
    auto fpn = _dual_fpn->forward(features);
    auto sparse = _sparse_head->forward(fpn.first, fpn.second);
    auto descriptors = fuseDescriptorMaps(sparse.descriptors, image, texture_blend_weight);
    auto texture_saliency = makeRotationInvariantTextureSaliency(image, sparse.heatmap.size(2), sparse.heatmap.size(3));
    auto dense = _dense_head->forward(features[0], features[0]);
    auto dense_confidence = interpolateTo(dense.confidence, sparse.heatmap.size(2), sparse.heatmap.size(3), "nearest");
    auto quality = _quality_head->forward(descriptors, sparse.heatmap, texture_saliency, dense_confidence);
    auto heatmap = (sparse.heatmap * quality).clamp(0.0, 1.0);
    return PfmV21RawFeatureMaps{heatmap,
                                descriptors,
                                sparse.scale,
                                sparse.orientation,
                                sparse.affine,
                                dense_confidence,
                                sparse.keypoint_offsets,
                                quality,
                                texture_saliency};
}

} // namespace pfm::v21
