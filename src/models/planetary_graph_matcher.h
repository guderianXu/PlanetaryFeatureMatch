#pragma once

#include <utility>

#include <torch/torch.h>

namespace pfm
{

struct PlanetaryGraphMatcherOutput
{
    /// 带 dustbin 行列的匹配 logits。
    torch::Tensor logits;
    /// 未经过外部几何过滤的匹配索引。
    torch::Tensor matches;
    /// 匹配置信度分数。
    torch::Tensor scores;
};

class PlanetaryGraphAttentionLayerImpl : public torch::nn::Module
{
  public:
    /// 创建一层包含自注意力和交叉注意力的图特征精化层。
    /// @param hidden_dim 两个图共用的内部特征维度。
    /// @throws std::invalid_argument 当 hidden_dim 非正时抛出。
    explicit PlanetaryGraphAttentionLayerImpl(int64_t hidden_dim);

    /// 使用自注意力和交叉注意力精化两组稀疏特征图。
    /// @param features_a 形状为 {Na,H} 的第一组图特征张量。
    /// @param features_b 形状为 {Nb,H} 的第二组图特征张量。
    /// @return 精化后的图 A 和图 B 特征张量。
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
    /// 为行星影像特征创建已学习的图匹配器。
    /// @param descriptor_dim 每个稀疏特征的描述子通道数。
    /// @param hidden_dim 匹配器使用的内部嵌入维度。
    /// @param attention_layers 自注意力/交叉注意力精化层数。
    /// @throws std::invalid_argument 当 descriptor_dim、hidden_dim 或 attention_layers 非正时抛出。
    PlanetaryGraphMatcherImpl(int64_t descriptor_dim, int64_t hidden_dim, int64_t attention_layers = 1);

    /// 预测已学习匹配 logits 和未做置信度过滤的候选匹配。
    /// @param descriptors_a 形状为 {Na,D} 的描述子张量。
    /// @param keypoints_a 形状为 {Na,2} 的关键点张量。
    /// @param descriptors_b 形状为 {Nb,D} 的描述子张量。
    /// @param keypoints_b 形状为 {Nb,2} 的关键点张量。
    /// @return 带 dustbin 的 {(Na+1),(Nb+1)} logits、{Na,2} matches 和 {Na} scores。
    /// @throws std::invalid_argument 当张量 rank、描述子维度或关键点维度非法时抛出。
    PlanetaryGraphMatcherOutput forward(const torch::Tensor& descriptors_a, const torch::Tensor& keypoints_a,
                                        const torch::Tensor& descriptors_b, const torch::Tensor& keypoints_b);

    /// 返回当前匹配器配置的注意力精化层数。
    /// @return 正数注意力层数。
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
