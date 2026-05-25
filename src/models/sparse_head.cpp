#include "models/sparse_head.h"

#include <array>
#include <optional>
#include <stdexcept>
#include <string>

namespace pfm {
namespace {

void require_positive_channels(int64_t channels, const char* name) {
    if (channels <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

torch::Tensor normalize_channels(const torch::Tensor& tensor) {
    const auto finite = torch::where(torch::isfinite(tensor), tensor, torch::zeros_like(tensor));
    auto scale = std::get<0>(finite.detach().abs().max(1, true)).clamp_min(1.0e-12);
    auto scaled = finite / scale;
    return scaled / scaled.norm(2, 1, true).clamp_min(1.0e-12);
}

torch::Tensor rotate_feature_map(const torch::Tensor& tensor, int64_t turns) {
    const auto normalized_turns = ((turns % 4) + 4) % 4;
    if (normalized_turns == 0) {
        return tensor;
    }
    return torch::rot90(tensor, normalized_turns, {2, 3}).contiguous();
}

torch::Tensor align_descriptor_orientation_channels(const torch::Tensor& tensor, int64_t turns) {
    const auto channels = tensor.size(1);
    if (channels < 4 || channels % 4 != 0) {
        return tensor;
    }
    const auto shift = channels / 4;
    return torch::roll(tensor, {-turns * shift}, {1});
}

class DescriptorResidualBlockImpl : public torch::nn::Module {
public:
    explicit DescriptorResidualBlockImpl(int64_t channels) {
        _conv1 = register_module(
            "conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
        _conv2 = register_module(
            "conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
    }

    torch::Tensor forward(const torch::Tensor& input) {
        auto hidden = torch::relu(_conv1->forward(input));
        return torch::relu(input + _conv2->forward(hidden));
    }

private:
    torch::nn::Conv2d _conv1{nullptr};
    torch::nn::Conv2d _conv2{nullptr};
};

TORCH_MODULE(DescriptorResidualBlock);

torch::nn::Sequential make_descriptor_tower(int64_t input_channels, int64_t descriptor_dim) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, input_channels, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        DescriptorResidualBlock(input_channels),
        DescriptorResidualBlock(input_channels),
        DescriptorResidualBlock(input_channels),
        DescriptorResidualBlock(input_channels),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, descriptor_dim, 1)));
}

torch::Tensor make_multiscale_descriptor_context(const torch::Tensor& context) {
    auto local = torch::avg_pool2d(context, {3, 3}, {1, 1}, {1, 1}, false, true);
    auto wider = torch::avg_pool2d(context, {5, 5}, {1, 1}, {2, 2}, false, true);
    return torch::cat({context, local, wider}, 1);
}

torch::Tensor conv1x1_channel_slice(
    torch::nn::Conv2d& projection,
    const torch::Tensor& input,
    int64_t channel_offset,
    bool include_bias
) {
    using torch::indexing::Slice;

    const auto channels = input.size(1);
    const auto weight = projection->weight.index(
        {Slice(), Slice(channel_offset, channel_offset + channels), Slice(), Slice()});
    const std::optional<torch::Tensor> bias =
        include_bias && projection->bias.defined() ? std::optional<torch::Tensor>(projection->bias) : std::nullopt;
    const std::array<int64_t, 2> stride{1, 1};
    const std::array<int64_t, 2> padding{0, 0};
    const std::array<int64_t, 2> dilation{1, 1};
    return torch::conv2d(input, weight, bias, stride, padding, dilation, 1);
}

torch::Tensor apply_anisotropic_viewpoint_projection(
    torch::nn::Conv2d& projection,
    const torch::Tensor& context
) {
    const auto channels = context.size(1);
    auto result = conv1x1_channel_slice(projection, context, 0, true);
    auto horizontal = torch::avg_pool2d(context, {1, 7}, {1, 1}, {0, 3}, false, true);
    result = result + conv1x1_channel_slice(projection, horizontal, channels, false);
    auto vertical = torch::avg_pool2d(context, {7, 1}, {1, 1}, {3, 0}, false, true);
    result = result + conv1x1_channel_slice(projection, vertical, channels * 2, false);
    auto wide_horizontal = torch::avg_pool2d(context, {3, 9}, {1, 1}, {1, 4}, false, true);
    result = result + conv1x1_channel_slice(projection, wide_horizontal, channels * 3, false);
    auto wide_vertical = torch::avg_pool2d(context, {9, 3}, {1, 1}, {4, 1}, false, true);
    return result + conv1x1_channel_slice(projection, wide_vertical, channels * 4, false);
}

void zero_module(torch::nn::Conv2d& module) {
    module->weight.zero_();
    if (module->bias.defined()) {
        module->bias.zero_();
    }
}

void load_child_archive(torch::serialize::InputArchive& archive, torch::nn::Module& module, const char* name) {
    torch::serialize::InputArchive child_archive;
    archive.read(name, child_archive);
    module.load(child_archive);
}

void load_sequential_child_archive(
    torch::nn::Sequential& sequence,
    torch::serialize::InputArchive& sequence_archive,
    const char* child_name
) {
    torch::serialize::InputArchive child_archive;
    sequence_archive.read(child_name, child_archive);
    for (const auto& child : sequence->named_children()) {
        if (child.key() == child_name) {
            child.value()->load(child_archive);
            return;
        }
    }
    throw std::invalid_argument(std::string("sparse head missing sequential child ") + child_name);
}

void load_sequential_child_archive_as(
    torch::nn::Sequential& sequence,
    torch::serialize::InputArchive& sequence_archive,
    const char* source_child_name,
    const char* target_child_name
) {
    torch::serialize::InputArchive child_archive;
    sequence_archive.read(source_child_name, child_archive);
    for (const auto& child : sequence->named_children()) {
        if (child.key() == target_child_name) {
            child.value()->load(child_archive);
            return;
        }
    }
    throw std::invalid_argument(std::string("sparse head missing sequential child ") + target_child_name);
}

bool archive_has_child(torch::serialize::InputArchive& archive, const char* name) {
    try {
        torch::serialize::InputArchive child_archive;
        archive.read(name, child_archive);
        return true;
    } catch (const c10::Error&) {
        return false;
    }
}

}  // namespace

SparseHeadImpl::SparseHeadImpl(int64_t input_channels, int64_t descriptor_dim)
    : _input_channels(input_channels), _descriptor_dim(descriptor_dim) {
    require_positive_channels(_input_channels, "input_channels");
    require_positive_channels(_descriptor_dim, "descriptor_dim");

    _context = register_module(
        "context",
        torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _input_channels, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _input_channels, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))));
    _heatmap = register_module("heatmap", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
    _heatmap_viewpoint_context = register_module(
        "heatmap_viewpoint_context",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 5, 1, 1)));
    _descriptors = register_module("descriptors", make_descriptor_tower(_input_channels, _descriptor_dim));
    _descriptor_multiscale = register_module(
        "descriptor_multiscale",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 3, _descriptor_dim, 1)));
    _descriptor_attention = register_module(
        "descriptor_attention",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 3, _descriptor_dim, 1)));
    _descriptor_viewpoint_context = register_module(
        "descriptor_viewpoint_context",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 5, _descriptor_dim, 1)));
    _descriptor_viewpoint_attention = register_module(
        "descriptor_viewpoint_attention",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels * 5, _descriptor_dim, 1)));
    _descriptor_orientation_alignment = register_module(
        "descriptor_orientation_alignment",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_descriptor_dim, _descriptor_dim, 1)));
    _descriptor_dilated_context = register_module(
        "descriptor_dilated_context",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(_descriptor_dim, _descriptor_dim, 3)
                .padding(2)
                .dilation(2)));
    {
        torch::NoGradGuard no_grad;
        zero_module(_heatmap_viewpoint_context);
        zero_module(_descriptor_viewpoint_context);
        zero_module(_descriptor_viewpoint_attention);
        zero_module(_descriptor_orientation_alignment);
        zero_module(_descriptor_dilated_context);
    }
    _descriptor_skip = register_module(
        "descriptor_skip",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, _descriptor_dim, 1)));
    _scale = register_module("scale", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
    _orientation = register_module("orientation", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 2, 1)));
    _affine = register_module("affine", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 4, 1)));
}

SparseHeadOutput SparseHeadImpl::forward(const torch::Tensor& feature) {
    if (!feature.defined()) {
        throw std::invalid_argument("feature tensor is undefined");
    }
    if (feature.dim() != 4) {
        throw std::invalid_argument("feature tensor must have shape BxCxHxW");
    }
    if (feature.size(1) != _input_channels) {
        throw std::invalid_argument("feature tensor channel count does not match sparse head");
    }

    auto context = _context->forward(feature);
    auto heatmap_sum = _heatmap->forward(context) + apply_anisotropic_viewpoint_projection(_heatmap_viewpoint_context, context);
    auto multiscale_context = make_multiscale_descriptor_context(context);
    auto viewpoint_descriptor =
        apply_anisotropic_viewpoint_projection(_descriptor_viewpoint_context, context) *
        (1.0F + torch::sigmoid(apply_anisotropic_viewpoint_projection(_descriptor_viewpoint_attention, context)));
    auto descriptor_base =
        _descriptors->forward(context) +
        _descriptor_multiscale->forward(multiscale_context) +
        viewpoint_descriptor +
        _descriptor_skip->forward(feature);
    auto descriptor_gated = descriptor_base * (1.0F + torch::sigmoid(_descriptor_attention->forward(multiscale_context)));
    auto descriptor_sum =
        descriptor_gated +
        _descriptor_orientation_alignment->forward(descriptor_gated) +
        _descriptor_dilated_context->forward(descriptor_gated);
    for (int64_t turns = 1; turns < 4; ++turns) {
        auto rotated_feature = rotate_feature_map(feature, turns);
        auto rotated_context = _context->forward(rotated_feature);
        heatmap_sum = heatmap_sum + rotate_feature_map(
            _heatmap->forward(rotated_context) +
                apply_anisotropic_viewpoint_projection(_heatmap_viewpoint_context, rotated_context),
            -turns);
        auto rotated_multiscale_context = make_multiscale_descriptor_context(rotated_context);
        auto rotated_viewpoint_descriptor =
            apply_anisotropic_viewpoint_projection(_descriptor_viewpoint_context, rotated_context) *
            (1.0F + torch::sigmoid(apply_anisotropic_viewpoint_projection(
                         _descriptor_viewpoint_attention,
                         rotated_context)));
        auto rotated_descriptor_base =
            _descriptors->forward(rotated_context) +
            _descriptor_multiscale->forward(rotated_multiscale_context) +
            rotated_viewpoint_descriptor +
            _descriptor_skip->forward(rotated_feature);
        auto rotated_descriptor_gated =
            rotated_descriptor_base *
            (1.0F + torch::sigmoid(_descriptor_attention->forward(rotated_multiscale_context)));
        auto rotated_descriptor = rotate_feature_map(rotated_descriptor_gated, -turns);
        auto orientation_aligned = align_descriptor_orientation_channels(rotated_descriptor, turns);
        descriptor_sum = descriptor_sum +
            rotated_descriptor +
            _descriptor_orientation_alignment->forward(orientation_aligned) +
            _descriptor_dilated_context->forward(rotated_descriptor);
    }
    auto heatmap = torch::sigmoid(heatmap_sum / 4.0F);
    auto descriptors = normalize_channels(descriptor_sum / 4.0F);
    auto scale = torch::softplus(_scale->forward(context)) + 1.0e-3;
    auto orientation = normalize_channels(_orientation->forward(context));
    auto affine = _affine->forward(context);
    return SparseHeadOutput{heatmap, descriptors, scale, orientation, affine};
}

void SparseHeadImpl::load_compatible(torch::serialize::InputArchive& archive) {
    load_child_archive(archive, *_context, "context");
    load_child_archive(archive, *_heatmap, "heatmap");
    if (archive_has_child(archive, "heatmap_viewpoint_context")) {
        load_child_archive(archive, *_heatmap_viewpoint_context, "heatmap_viewpoint_context");
    }
    torch::serialize::InputArchive descriptors_archive;
    archive.read("descriptors", descriptors_archive);
    if (archive_has_child(descriptors_archive, "6")) {
        _descriptors->load(descriptors_archive);
    } else if (archive_has_child(descriptors_archive, "4")) {
        load_sequential_child_archive(_descriptors, descriptors_archive, "0");
        load_sequential_child_archive(_descriptors, descriptors_archive, "2");
        load_sequential_child_archive(_descriptors, descriptors_archive, "3");
        load_sequential_child_archive_as(_descriptors, descriptors_archive, "4", "6");
    } else {
        load_sequential_child_archive(_descriptors, descriptors_archive, "0");
        load_sequential_child_archive_as(_descriptors, descriptors_archive, "2", "6");
    }
    if (archive_has_child(archive, "descriptor_multiscale")) {
        load_child_archive(archive, *_descriptor_multiscale, "descriptor_multiscale");
    }
    if (archive_has_child(archive, "descriptor_attention")) {
        load_child_archive(archive, *_descriptor_attention, "descriptor_attention");
    }
    if (archive_has_child(archive, "descriptor_viewpoint_context")) {
        load_child_archive(archive, *_descriptor_viewpoint_context, "descriptor_viewpoint_context");
    }
    if (archive_has_child(archive, "descriptor_viewpoint_attention")) {
        load_child_archive(archive, *_descriptor_viewpoint_attention, "descriptor_viewpoint_attention");
    }
    if (archive_has_child(archive, "descriptor_orientation_alignment")) {
        load_child_archive(archive, *_descriptor_orientation_alignment, "descriptor_orientation_alignment");
    }
    if (archive_has_child(archive, "descriptor_dilated_context")) {
        load_child_archive(archive, *_descriptor_dilated_context, "descriptor_dilated_context");
    }
    load_child_archive(archive, *_descriptor_skip, "descriptor_skip");
    load_child_archive(archive, *_scale, "scale");
    load_child_archive(archive, *_orientation, "orientation");
    load_child_archive(archive, *_affine, "affine");
}

namespace testing {

torch::Tensor normalize_sparse_head_channels_for_test(const torch::Tensor& tensor) {
    return normalize_channels(tensor);
}

}  // namespace testing

}  // namespace pfm
