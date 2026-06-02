#pragma once

#include <utility>

#include <torch/torch.h>

namespace pfm
{

struct PlanetaryGraphMatcherOutput
{
    torch::Tensor logits;
    torch::Tensor matches;
    torch::Tensor scores;
};

class PlanetaryGraphAttentionLayerImpl : public torch::nn::Module
{
  public:
    /// 创建一层包含 self-attention 和 cross-attention 的 graph refinement layer。
    /// @param hidden_dim 两个 graph 共用的内部特征维度。
    /// @throws std::invalid_argument 当 hidden_dim 非正时抛出。
    explicit PlanetaryGraphAttentionLayerImpl(int64_t hidden_dim);

    /// 使用 self-attention 和 cross-attention 对两组 sparse feature graph 做 refinement。
    /// @param features_a 形状为 {Na,H} 的第一组 graph 张量。
    /// @param features_b 形状为 {Nb,H} 的第二组 graph 张量。
    /// @return refinement 后的 graph A 和 graph B 特征张量。
    /// @throws std::invalid_argument 当张量 rank 或 hidden 维度非法时抛出。
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& features_a, const torch::Tensor& features_b);

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

class PlanetaryGraphMatcherImpl : public torch::nn::Module
{
  public:
    /// 为行星影像特征创建 learned graph matcher。
    /// @param descriptor_dim 每个 sparse feature 的 descriptor 通道数。
    /// @param hidden_dim matcher 使用的内部 embedding 维度。
    /// @param attention_layers self/cross attention refinement 层数。
    /// @throws std::invalid_argument 当 descriptor_dim、hidden_dim 或 attention_layers 非正时抛出。
    PlanetaryGraphMatcherImpl(int64_t descriptor_dim, int64_t hidden_dim, int64_t attention_layers = 1);

    /// 预测 learned match logits 和未做置信度过滤的候选匹配。
    /// @param descriptors_a 形状为 {Na,D} 的 descriptor 张量。
    /// @param keypoints_a 形状为 {Na,2} 的关键点张量。
    /// @param descriptors_b 形状为 {Nb,D} 的 descriptor 张量。
    /// @param keypoints_b 形状为 {Nb,2} 的关键点张量。
    /// @return 带 dustbin 的 {(Na+1),(Nb+1)} logits、{Na,2} matches 和 {Na} scores。
    /// @throws std::invalid_argument 当张量 rank、descriptor 维度或关键点维度非法时抛出。
    PlanetaryGraphMatcherOutput forward(const torch::Tensor& descriptors_a, const torch::Tensor& keypoints_a,
                                        const torch::Tensor& descriptors_b, const torch::Tensor& keypoints_b);

    /// 返回当前 matcher 配置的 attention refinement 层数。
    /// @return 正数 attention 层数。
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

} // namespace pfm
