#include "models/planetary_graph_matcher.h"

#include <cmath>
#include <stdexcept>

#include <torch/torch.h>

namespace pfm {
namespace {

void validate_graph_features(
    const torch::Tensor& features_a,
    const torch::Tensor& features_b,
    int64_t hidden_dim
) {
    if (!features_a.defined() || !features_b.defined()) {
        throw std::invalid_argument("graph attention features must be defined");
    }
    if (features_a.dim() != 2 || features_b.dim() != 2) {
        throw std::invalid_argument("graph attention features must have shape {N,H}");
    }
    if (features_a.size(1) != hidden_dim || features_b.size(1) != hidden_dim) {
        throw std::invalid_argument("graph attention hidden dimensions must match");
    }
}

torch::Tensor attend(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    int64_t hidden_dim
) {
    auto logits = torch::matmul(query, key.transpose(0, 1)) / std::sqrt(static_cast<double>(hidden_dim));
    return torch::matmul(torch::softmax(logits, 1), value);
}

void validate_matcher_inputs(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& keypoints_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& keypoints_b,
    int64_t descriptor_dim
) {
    if (!descriptors_a.defined() || !descriptors_b.defined()) {
        throw std::invalid_argument("graph matcher descriptors must be defined");
    }
    if (!keypoints_a.defined() || !keypoints_b.defined()) {
        throw std::invalid_argument("graph matcher keypoints must be defined");
    }
    if (descriptors_a.dim() != 2 || descriptors_b.dim() != 2) {
        throw std::invalid_argument("graph matcher descriptors must have shape {N,D}");
    }
    if (keypoints_a.dim() != 2 || keypoints_b.dim() != 2 || keypoints_a.size(1) != 2 || keypoints_b.size(1) != 2) {
        throw std::invalid_argument("graph matcher keypoints must have shape {N,2}");
    }
    if (descriptors_a.size(0) != keypoints_a.size(0) || descriptors_b.size(0) != keypoints_b.size(0)) {
        throw std::invalid_argument("graph matcher descriptor and keypoint counts must match");
    }
    if (descriptors_a.size(1) != descriptor_dim || descriptors_b.size(1) != descriptor_dim) {
        throw std::invalid_argument("graph matcher descriptor dimensions must match the configured dimension");
    }
}

torch::Tensor prepare_keypoints_for_embedding(const torch::Tensor& keypoints) {
    return keypoints.to(torch::TensorOptions().dtype(torch::kFloat32).device(keypoints.device()));
}

}  // namespace

PlanetaryGraphAttentionLayerImpl::PlanetaryGraphAttentionLayerImpl(int64_t hidden_dim) : _hidden_dim(hidden_dim) {
    if (_hidden_dim <= 0) {
        throw std::invalid_argument("graph attention hidden dimension must be positive");
    }
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
    _feed_forward_norm = register_module(
        "feed_forward_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({_hidden_dim})));
    _attention_dropout = register_module("attention_dropout", torch::nn::Dropout(0.1));
    _feed_forward = register_module(
        "feed_forward",
        torch::nn::Sequential(
            torch::nn::Linear(_hidden_dim, _hidden_dim * 2),
            torch::nn::GELU(),
            torch::nn::Linear(_hidden_dim * 2, _hidden_dim)));
}

std::pair<torch::Tensor, torch::Tensor> PlanetaryGraphAttentionLayerImpl::forward(
    const torch::Tensor& features_a,
    const torch::Tensor& features_b
) {
    validate_graph_features(features_a, features_b, _hidden_dim);
    auto self_a = attend(_self_query(features_a), _self_key(features_a), _self_value(features_a), _hidden_dim);
    auto self_b = attend(_self_query(features_b), _self_key(features_b), _self_value(features_b), _hidden_dim);
    auto refined_a = _self_norm(features_a + _attention_dropout(_self_output(self_a)));
    auto refined_b = _self_norm(features_b + _attention_dropout(_self_output(self_b)));
    auto cross_a = attend(_cross_query(refined_a), _cross_key(refined_b), _cross_value(refined_b), _hidden_dim);
    auto cross_b = attend(_cross_query(refined_b), _cross_key(refined_a), _cross_value(refined_a), _hidden_dim);
    refined_a = _cross_norm(refined_a + _attention_dropout(_cross_output(cross_a)));
    refined_b = _cross_norm(refined_b + _attention_dropout(_cross_output(cross_b)));
    return {
        _feed_forward_norm(refined_a + _feed_forward->forward(refined_a)),
        _feed_forward_norm(refined_b + _feed_forward->forward(refined_b))};
}

PlanetaryGraphMatcherImpl::PlanetaryGraphMatcherImpl(
    int64_t descriptor_dim,
    int64_t hidden_dim,
    int64_t attention_layers
)
    : _descriptor_dim(descriptor_dim), _hidden_dim(hidden_dim), _attention_layer_count(attention_layers) {
    if (_descriptor_dim <= 0 || _hidden_dim <= 0 || _attention_layer_count <= 0) {
        throw std::invalid_argument("graph matcher dimensions and attention layer count must be positive");
    }
    _descriptor_projection = register_module("descriptor_projection", torch::nn::Linear(_descriptor_dim, _hidden_dim));
    _keypoint_projection = register_module("keypoint_projection", torch::nn::Linear(2, _hidden_dim));
    _score_projection = register_module("score_projection", torch::nn::Linear(_hidden_dim, _hidden_dim));
    _logit_scale = register_parameter("logit_scale", torch::ones({1}) * std::sqrt(static_cast<double>(_hidden_dim)));
    _dustbin_bias = register_parameter("dustbin_bias", torch::zeros({1}));
    _attention_layers = register_module("attention_layers", torch::nn::ModuleList());
    for (int64_t index = 0; index < _attention_layer_count; ++index) {
        _attention_layers->push_back(PlanetaryGraphAttentionLayer(_hidden_dim));
    }
}

PlanetaryGraphMatcherOutput PlanetaryGraphMatcherImpl::forward(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& keypoints_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& keypoints_b
) {
    validate_matcher_inputs(descriptors_a, keypoints_a, descriptors_b, keypoints_b, _descriptor_dim);
    const auto options = descriptors_a.options().dtype(torch::kFloat32);
    const auto desc_a = descriptors_a.to(options);
    const auto desc_b = descriptors_b.to(options);
    auto kp_a = prepare_keypoints_for_embedding(keypoints_a).to(desc_a.device());
    auto kp_b = prepare_keypoints_for_embedding(keypoints_b).to(desc_b.device());
    auto embed_a = torch::relu(_descriptor_projection(desc_a) + _keypoint_projection(kp_a));
    auto embed_b = torch::relu(_descriptor_projection(desc_b) + _keypoint_projection(kp_b));
    for (const auto& layer : *_attention_layers) {
        auto refined = layer->as<PlanetaryGraphAttentionLayerImpl>()->forward(embed_a, embed_b);
        embed_a = refined.first;
        embed_b = refined.second;
    }
    embed_a = torch::nn::functional::normalize(_score_projection(embed_a), torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    embed_b = torch::nn::functional::normalize(_score_projection(embed_b), torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    auto pair_logits = torch::matmul(embed_a, embed_b.transpose(0, 1)) * _logit_scale.clamp(1.0, 100.0);
    auto logits = torch::zeros({descriptors_a.size(0) + 1, descriptors_b.size(0) + 1}, pair_logits.options()) + _dustbin_bias;
    logits.index_put_({torch::indexing::Slice(0, descriptors_a.size(0)), torch::indexing::Slice(0, descriptors_b.size(0))}, pair_logits);

    auto row_logits = logits.index({torch::indexing::Slice(0, descriptors_a.size(0)), torch::indexing::Slice()});
    const auto best = torch::max(row_logits, 1);
    const auto best_indices = std::get<1>(best);
    auto source_indices = torch::arange(descriptors_a.size(0), best_indices.options());
    auto reverse_best = std::get<1>(pair_logits.max(0));
    auto inlier_mask = best_indices.lt(descriptors_b.size(0));
    if (descriptors_a.size(0) > 0 && descriptors_b.size(0) > 0) {
        auto mutual_sources = reverse_best.index_select(0, best_indices.clamp(0, descriptors_b.size(0) - 1));
        inlier_mask = inlier_mask.logical_and(mutual_sources.eq(source_indices));
    }
    source_indices = source_indices.index({inlier_mask});
    auto target_indices = best_indices.index({inlier_mask});
    auto probabilities = std::get<0>(torch::softmax(row_logits, 1).max(1)).index({inlier_mask});
    auto matches = torch::stack({source_indices, target_indices}, 1).to(torch::kCPU, torch::kInt64).contiguous();
    return PlanetaryGraphMatcherOutput{logits.contiguous(), matches, probabilities.to(torch::kCPU, torch::kFloat32).contiguous()};
}

int64_t PlanetaryGraphMatcherImpl::attentionLayerCount() const {
    return _attention_layer_count;
}

}  // namespace pfm
