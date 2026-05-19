#pragma once

#include <utility>

#include <torch/torch.h>

namespace pfm {

struct PlanetaryGraphMatcherOutput {
    torch::Tensor logits;
    torch::Tensor matches;
    torch::Tensor scores;
};

class PlanetaryGraphAttentionLayerImpl : public torch::nn::Module {
public:
    /// Creates one self-attention plus cross-attention graph refinement layer.
    /// @param hidden_dim Internal feature dimension for both graphs.
    /// @throws std::invalid_argument if hidden_dim is not positive.
    explicit PlanetaryGraphAttentionLayerImpl(int64_t hidden_dim);

    /// Refines two sparse feature graphs with self-attention and cross-attention.
    /// @param features_a First graph tensor with shape {Na,H}.
    /// @param features_b Second graph tensor with shape {Nb,H}.
    /// @return Refined feature tensors for graph A and graph B.
    /// @throws std::invalid_argument if tensor ranks or hidden dimensions are invalid.
    std::pair<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& features_a,
        const torch::Tensor& features_b
    );

private:
    int64_t _hidden_dim;
    torch::nn::Linear _self_query{nullptr};
    torch::nn::Linear _self_key{nullptr};
    torch::nn::Linear _self_value{nullptr};
    torch::nn::Linear _self_output{nullptr};
    torch::nn::Linear _cross_query{nullptr};
    torch::nn::Linear _cross_key{nullptr};
    torch::nn::Linear _cross_value{nullptr};
    torch::nn::Linear _cross_output{nullptr};
    torch::nn::LayerNorm _self_norm{nullptr};
    torch::nn::LayerNorm _cross_norm{nullptr};
    torch::nn::LayerNorm _feed_forward_norm{nullptr};
    torch::nn::Sequential _feed_forward{nullptr};
    torch::nn::Dropout _attention_dropout{nullptr};
};

TORCH_MODULE(PlanetaryGraphAttentionLayer);

class PlanetaryGraphMatcherImpl : public torch::nn::Module {
public:
    /// Creates a learned graph matcher for planetary image features.
    /// @param descriptor_dim Descriptor channel count for each sparse feature.
    /// @param hidden_dim Internal embedding dimension used by the matcher.
    /// @param attention_layers Number of self/cross attention refinement layers.
    /// @throws std::invalid_argument if descriptor_dim, hidden_dim, or attention_layers is not positive.
    PlanetaryGraphMatcherImpl(int64_t descriptor_dim, int64_t hidden_dim, int64_t attention_layers = 1);

    /// Predicts learned match logits and confidence-filter-free match candidates.
    /// @param descriptors_a Descriptor tensor with shape {Na,D}.
    /// @param keypoints_a Keypoint tensor with shape {Na,2}.
    /// @param descriptors_b Descriptor tensor with shape {Nb,D}.
    /// @param keypoints_b Keypoint tensor with shape {Nb,2}.
    /// @return Logits with dustbin shape {(Na+1),(Nb+1)}, matches {Na,2}, and scores {Na}.
    /// @throws std::invalid_argument if tensor ranks, descriptor dimensions, or keypoint dimensions are invalid.
    PlanetaryGraphMatcherOutput forward(
        const torch::Tensor& descriptors_a,
        const torch::Tensor& keypoints_a,
        const torch::Tensor& descriptors_b,
        const torch::Tensor& keypoints_b
    );

    /// Returns the number of attention refinement layers configured for this matcher.
    /// @return Positive attention layer count.
    int64_t attentionLayerCount() const;

private:
    int64_t _descriptor_dim;
    int64_t _hidden_dim;
    int64_t _attention_layer_count;
    torch::nn::Linear _descriptor_projection{nullptr};
    torch::nn::Linear _keypoint_projection{nullptr};
    torch::nn::Linear _score_projection{nullptr};
    torch::Tensor _logit_scale;
    torch::Tensor _dustbin_bias;
    torch::nn::ModuleList _attention_layers{nullptr};
};

TORCH_MODULE(PlanetaryGraphMatcher);

}  // namespace pfm
