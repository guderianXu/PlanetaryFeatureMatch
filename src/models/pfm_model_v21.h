#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace pfm::v21
{

constexpr double INFERENCE_TEXTURE_BLEND_WEIGHT = 1.0;

struct PfmV21Config
{
    int64_t input_channels = 1;
    int64_t base_channels = 64;
    int64_t descriptor_dim = 256;
    int64_t graph_hidden_dim = 512;
    int64_t graph_attention_layers = 8;
    int64_t graph_keypoint_meta_dim = 16;
};

struct PfmV21SparseHeadOutput
{
    torch::Tensor heatmap;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
    torch::Tensor keypoint_offsets;
};

struct PfmV21DenseHeadOutput
{
    torch::Tensor confidence;
    torch::Tensor offsets;
};

struct PfmV21GraphMatcherOutput
{
    torch::Tensor logits;
    torch::Tensor matches;
    torch::Tensor scores;
    torch::Tensor accept_logits;
};

struct PfmV21SemiDenseCandidateOutput
{
    torch::Tensor keypoints_a;
    torch::Tensor keypoints_b;
    torch::Tensor scores;
};

struct PfmV21RawFeatureMaps
{
    torch::Tensor heatmap;
    torch::Tensor descriptors;
    torch::Tensor scale;
    torch::Tensor orientation;
    torch::Tensor affine;
    torch::Tensor dense_confidence;
    torch::Tensor keypoint_offsets;
    torch::Tensor quality;
    torch::Tensor local_contrast;
};

class PfmV21ZeroResidualContextBlockImpl : public torch::nn::Module
{
  public:
    /// 创建零初始化的残差上下文块。
    /// @param channels 输入和输出通道数。
    /// @param dilation 两个空间卷积使用的 dilation。
    /// @throws std::invalid_argument 当 channels 或 dilation 非正时抛出。
    explicit PfmV21ZeroResidualContextBlockImpl(int64_t channels, int64_t dilation = 1);

    /// 对 NCHW 张量应用残差上下文块。
    /// @param x BxCxHxW 输入张量。
    /// @return 与 x 同形状的输出张量。
    torch::Tensor forward(const torch::Tensor& x);

  private:
    torch::nn::Conv2d _conv1{nullptr};
    torch::nn::GroupNorm _norm1{nullptr};
    torch::nn::Conv2d _conv2{nullptr};
    torch::nn::GroupNorm _norm2{nullptr};
};

TORCH_MODULE(PfmV21ZeroResidualContextBlock);

class PfmV21BackboneImpl : public torch::nn::Module
{
  public:
    /// 创建当前 Python backbone 的 C++ 镜像实现。
    /// @param input_channels 输入图像通道数。
    /// @param base_channels 第一层 backbone stage 的通道数。
    /// @throws std::invalid_argument 当 input_channels 或 base_channels 非正时抛出。
    PfmV21BackboneImpl(int64_t input_channels, int64_t base_channels);

    /// 计算四级经过 refine 的 stride-2 特征。
    /// @param x BxCxHxW 输入张量。
    /// @return stride 分别为 2、4、8、16 的特征 stage。
    std::vector<torch::Tensor> forward(const torch::Tensor& x);

    /// 替换归一化层中的非有限 buffer，保持 checkpoint 加载兼容性。
    void sanitizeNonfiniteState();

  private:
    int64_t _input_channels;
    int64_t _base_channels;
    torch::nn::Sequential _stage1{nullptr};
    torch::nn::Sequential _stage2{nullptr};
    torch::nn::Sequential _stage3{nullptr};
    torch::nn::Sequential _stage4{nullptr};
    torch::nn::Sequential _stage1_refine{nullptr};
    torch::nn::Sequential _stage2_refine{nullptr};
    torch::nn::Sequential _stage3_refine{nullptr};
    torch::nn::Sequential _stage4_refine{nullptr};
};

TORCH_MODULE(PfmV21Backbone);

class PfmV21DualFPNLiteImpl : public torch::nn::Module
{
  public:
    /// 为关键点和 descriptor 创建独立的 1/4 分辨率 FPN 分支。
    /// @param base_channels backbone 基础通道数。
    /// @throws std::invalid_argument 当 base_channels 非正时抛出。
    explicit PfmV21DualFPNLiteImpl(int64_t base_channels);

    /// 从 backbone 多级特征生成关键点和 descriptor 特征图。
    /// @param features backbone 的第 1 到第 4 级 stage。
    /// @return 两个 Bx(2*base_channels)xHxW 张量。
    std::pair<torch::Tensor, torch::Tensor> forward(const std::vector<torch::Tensor>& features);

  private:
    int64_t _base_channels;
    torch::nn::Conv2d _keypoint_from_stage3{nullptr};
    torch::nn::Conv2d _descriptor_from_stage3{nullptr};
    torch::nn::Conv2d _descriptor_from_stage4{nullptr};
    PfmV21ZeroResidualContextBlock _keypoint_refine{nullptr};
    torch::nn::Sequential _descriptor_refine{nullptr};
};

TORCH_MODULE(PfmV21DualFPNLite);

class PfmV21SparseHeadImpl : public torch::nn::Module
{
  public:
    /// 创建当前 Python sparse head 的 C++ 镜像实现。
    /// @param input_channels 输入特征通道数。
    /// @param descriptor_dim 输出 descriptor 通道数。
    /// @throws std::invalid_argument 当 input_channels 或 descriptor_dim 非正时抛出。
    PfmV21SparseHeadImpl(int64_t input_channels, int64_t descriptor_dim);

    /// 从共享特征张量预测 sparse maps。
    /// @param feature 关键点和 descriptor 共享的输入特征图。
    /// @return 与 Python `SparseHeadOutput` 对齐的 sparse head 输出。
    PfmV21SparseHeadOutput forward(const torch::Tensor& feature);

    /// 从独立的关键点和 descriptor 特征张量预测 sparse maps。
    /// @param feature 关键点特征图。
    /// @param descriptor_feature 同一网格上的 descriptor 特征图。
    /// @return 与 Python `SparseHeadOutput` 对齐的 sparse head 输出。
    PfmV21SparseHeadOutput forward(const torch::Tensor& feature, const torch::Tensor& descriptor_feature);

  private:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    descriptorBranch(const torch::Tensor& keypoint_feature, const torch::Tensor& descriptor_feature);

    int64_t _input_channels;
    int64_t _descriptor_dim;
    torch::nn::Sequential _context{nullptr};
    PfmV21ZeroResidualContextBlock _keypoint_context{nullptr};
    PfmV21ZeroResidualContextBlock _descriptor_context{nullptr};
    PfmV21ZeroResidualContextBlock _geometry_context{nullptr};
    torch::nn::Conv2d _heatmap{nullptr};
    torch::nn::Conv2d _heatmap_viewpoint_context{nullptr};
    torch::nn::Conv2d _keypoint_offsets{nullptr};
    torch::nn::Sequential _descriptors{nullptr};
    torch::nn::Conv2d _descriptor_multiscale{nullptr};
    torch::nn::Conv2d _descriptor_attention{nullptr};
    torch::nn::Conv2d _descriptor_viewpoint_context{nullptr};
    torch::nn::Conv2d _descriptor_viewpoint_attention{nullptr};
    torch::nn::Conv2d _descriptor_orientation_alignment{nullptr};
    torch::nn::Conv2d _descriptor_dilated_context{nullptr};
    torch::nn::Conv2d _descriptor_branch_quality{nullptr};
    torch::nn::Conv2d _descriptor_rotation_fusion{nullptr};
    torch::nn::Conv2d _descriptor_skip{nullptr};
    torch::nn::Conv2d _scale{nullptr};
    torch::nn::Conv2d _orientation{nullptr};
    torch::nn::Conv2d _affine{nullptr};
};

TORCH_MODULE(PfmV21SparseHead);

class PfmV21DenseHeadImpl : public torch::nn::Module
{
  public:
    /// 创建当前 Python dense head 的 C++ 镜像实现。
    /// @param feature_channels 每张 dense 特征图的通道数。
    /// @throws std::invalid_argument 当 feature_channels 非正时抛出。
    explicit PfmV21DenseHeadImpl(int64_t feature_channels);

    /// 从一对 dense 特征预测置信度和偏移量。
    /// @param feature_a 第一张 BxCxHxW 特征张量。
    /// @param feature_b 第二张 BxCxHxW 特征张量。
    /// @return dense 置信度和 x/y 偏移量。
    PfmV21DenseHeadOutput forward(const torch::Tensor& feature_a, const torch::Tensor& feature_b);

  private:
    int64_t _feature_channels;
    torch::nn::Conv2d _correlation_projection{nullptr};
    torch::nn::Sequential _predictor{nullptr};
};

TORCH_MODULE(PfmV21DenseHead);

class PfmV21GraphAttentionLayerImpl : public torch::nn::Module
{
  public:
    /// 创建一层 self/cross attention graph layer。
    /// @param hidden_dim graph 内部特征维度。
    /// @throws std::invalid_argument 当 hidden_dim 非正时抛出。
    explicit PfmV21GraphAttentionLayerImpl(int64_t hidden_dim);

    /// 对两组关键点 graph 特征做 refinement。
    /// @param features_a NxH 的 graph A 特征。
    /// @param features_b MxH 的 graph B 特征。
    /// @return refinement 后的两组 graph 特征。
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
    torch::nn::Dropout _attention_dropout{nullptr};
    torch::nn::Sequential _feed_forward{nullptr};
};

TORCH_MODULE(PfmV21GraphAttentionLayer);

class PfmV21GraphMatcherImpl : public torch::nn::Module
{
  public:
    /// 创建当前 Python graph matcher 的 C++ 镜像实现。
    /// @param descriptor_dim descriptor 通道数。
    /// @param hidden_dim graph 内部隐藏维度。
    /// @param attention_layers graph attention 层数。
    /// @param keypoint_meta_dim keypoint projection 接收的元数据维度。
    /// @param candidate_topk 候选匹配剪枝的 top-k。
    /// @throws std::invalid_argument 当维度或层数非正时抛出。
    PfmV21GraphMatcherImpl(int64_t descriptor_dim, int64_t hidden_dim, int64_t attention_layers = 1,
                           int64_t keypoint_meta_dim = 2, int64_t candidate_topk = 64);

    /// 在 sparse descriptors 和关键点元数据上运行 graph matching。
    /// @param descriptors_a NxD 的 A 视图 descriptors。
    /// @param keypoints_a NxC 的 A 视图关键点或元数据。
    /// @param descriptors_b MxD 的 B 视图 descriptors。
    /// @param keypoints_b MxC 的 B 视图关键点或元数据。
    /// @return 匹配 logits、互选匹配、概率和接受 logits。
    PfmV21GraphMatcherOutput forward(const torch::Tensor& descriptors_a, const torch::Tensor& keypoints_a,
                                     const torch::Tensor& descriptors_b, const torch::Tensor& keypoints_b);

  private:
    torch::Tensor metadata(const torch::Tensor& keypoints_or_meta) const;
    torch::Tensor geometryCompatibilityBias(const torch::Tensor& meta_a, const torch::Tensor& meta_b);
    torch::Tensor candidateMask(const torch::Tensor& desc_a, const torch::Tensor& desc_b) const;
    torch::Tensor acceptanceLogits(const torch::Tensor& raw_similarity, const torch::Tensor& graph_delta,
                                   const torch::Tensor& meta_a, const torch::Tensor& meta_b);

    int64_t _descriptor_dim;
    int64_t _hidden_dim;
    int64_t _attention_layer_count;
    int64_t _keypoint_meta_dim;
    int64_t _candidate_topk;
    torch::nn::Linear _descriptor_projection{nullptr};
    torch::nn::Linear _keypoint_projection{nullptr};
    torch::nn::Linear _score_projection{nullptr};
    torch::nn::Sequential _geometry_bias{nullptr};
    torch::nn::Sequential _accept_head{nullptr};
    torch::Tensor _logit_scale;
    torch::Tensor _raw_score_temperature;
    torch::Tensor _graph_delta_scale;
    torch::Tensor _accept_logit_scale;
    torch::Tensor _dustbin_bias;
    torch::nn::ModuleList _attention_layers{nullptr};
};

TORCH_MODULE(PfmV21GraphMatcher);

class PfmV21TextureDescriptorAdapterImpl : public torch::nn::Module
{
  public:
    /// 创建可训练的残差纹理 descriptor adapter。
    /// @param descriptor_dim descriptor 通道数。
    /// @throws std::invalid_argument 当 descriptor_dim 非正时抛出。
    explicit PfmV21TextureDescriptorAdapterImpl(int64_t descriptor_dim);

    /// 应用残差适配和通道归一化。
    /// @param texture BxDxHxW 纹理 descriptor 图。
    /// @return 适配后的纹理 descriptor 图。
    torch::Tensor forward(const torch::Tensor& texture);

  private:
    int64_t _descriptor_dim;
    torch::nn::Conv2d _residual{nullptr};
};

TORCH_MODULE(PfmV21TextureDescriptorAdapter);

class PfmV21DescriptorFusionAdapterImpl : public torch::nn::Module
{
  public:
    /// 创建 learned descriptor 和解析纹理 descriptor 的融合 adapter。
    /// @param descriptor_dim descriptor 通道数。
    /// @param hidden_dim 可选隐藏通道数；非正值使用 Python 默认设置。
    /// @throws std::invalid_argument 当 descriptor_dim 非正时抛出。
    explicit PfmV21DescriptorFusionAdapterImpl(int64_t descriptor_dim, int64_t hidden_dim = 0);

    /// 融合 learned descriptor 和纹理 descriptor。
    /// @param learned learned descriptor 图。
    /// @param texture 解析纹理 descriptor 图。
    /// @param blend_weight 解析 descriptor 融合权重。
    /// @return 融合并归一化后的 descriptor 图。
    torch::Tensor forward(const torch::Tensor& learned, const torch::Tensor& texture, double blend_weight);

  private:
    int64_t _descriptor_dim;
    int64_t _hidden_dim;
    torch::nn::Conv2d _input_projection{nullptr};
    torch::nn::Sequential _context{nullptr};
    torch::nn::Conv2d _texture_gate{nullptr};
    torch::nn::Conv2d _output{nullptr};
};

TORCH_MODULE(PfmV21DescriptorFusionAdapter);

class PfmV21QualityHeadImpl : public torch::nn::Module
{
  public:
    /// 创建 descriptor/关键点可靠性 head。
    /// @param descriptor_dim descriptor 通道数。
    /// @throws std::invalid_argument 当 descriptor_dim 非正时抛出。
    explicit PfmV21QualityHeadImpl(int64_t descriptor_dim);

    /// 从 descriptors 和辅助图预测可靠性。
    /// @param descriptors BxDxHxW 融合 descriptor 图。
    /// @param heatmap Bx1xHxW sparse heatmap。
    /// @param texture_saliency Bx1xHxW 纹理显著性图。
    /// @param dense_confidence Bx1xHxW dense 置信度图。
    /// @return Bx1xHxW 可靠性图。
    torch::Tensor forward(const torch::Tensor& descriptors, const torch::Tensor& heatmap,
                          const torch::Tensor& texture_saliency, const torch::Tensor& dense_confidence);

  private:
    int64_t _descriptor_dim;
    torch::nn::Sequential _predictor{nullptr};
};

TORCH_MODULE(PfmV21QualityHead);

class PfmV21SemiDenseCandidateBranchImpl : public torch::nn::Module
{
  public:
    /// 创建弱纹理场景下的 semi-dense 候选分支。
    /// @param descriptor_dim descriptor 通道数。
    /// @param projection_dim projection 通道数。
    /// @param max_grid coarse grid 的最大边长。
    /// @throws std::invalid_argument 当任意维度非正时抛出。
    PfmV21SemiDenseCandidateBranchImpl(int64_t descriptor_dim, int64_t projection_dim = 64, int64_t max_grid = 32);

    /// 从 descriptor 图生成 detector-free 候选匹配点对。
    /// @param descriptors_a A 视图 descriptor 图。
    /// @param descriptors_b B 视图 descriptor 图。
    /// @param max_candidates 最大候选数量。
    /// @param min_score 最小 dual-softmax 分数。
    /// @return 候选关键点和分数。
    PfmV21SemiDenseCandidateOutput forward(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                                           int64_t max_candidates, double min_score = 0.0);

  private:
    torch::Tensor coarse(const torch::Tensor& descriptors);

    int64_t _descriptor_dim;
    int64_t _projection_dim;
    int64_t _max_grid;
    torch::nn::Sequential _projection{nullptr};
};

TORCH_MODULE(PfmV21SemiDenseCandidateBranch);

class PfmV21FeatureMatcherImpl : public torch::nn::Module
{
  public:
    /// 创建完整的当前 Python 模型 C++ 镜像实现。
    /// @param config 来自 Python checkpoint config 的网络结构维度。
    /// @throws std::invalid_argument 当任意配置维度非正时抛出。
    explicit PfmV21FeatureMatcherImpl(const PfmV21Config& config = {});

    /// 返回不可变的网络结构配置。
    /// @return 模型结构配置。
    const PfmV21Config& config() const;

    /// 计算解析纹理融合前的 learned descriptor 图。
    /// @param image BxCxHxW 图像张量。
    /// @return learned descriptor 图。
    torch::Tensor learnedDescriptorMapSingle(const torch::Tensor& image);

    /// 计算原始解析纹理 descriptor 图。
    /// @param image BxCxHxW 图像张量。
    /// @return 1/4 分辨率的原始解析 descriptor 图。
    torch::Tensor rawTextureDescriptorMapSingle(const torch::Tensor& image);

    /// 计算适配后的解析纹理 descriptor 图。
    /// @param image BxCxHxW 图像张量。
    /// @return 适配后的解析 descriptor 图。
    torch::Tensor textureDescriptorMapSingle(const torch::Tensor& image);

    /// 融合 learned descriptors 和解析纹理 descriptors。
    /// @param learned_descriptors learned descriptor 图。
    /// @param image 源图像张量。
    /// @param texture_blend_weight 解析 descriptor 融合权重。
    /// @return 融合后的 descriptor 图。
    torch::Tensor fuseDescriptorMaps(const torch::Tensor& learned_descriptors, const torch::Tensor& image,
                                     double texture_blend_weight = INFERENCE_TEXTURE_BLEND_WEIGHT);

    /// 为单个图像 batch 计算融合 descriptor 图。
    /// @param image BxCxHxW 图像张量。
    /// @param texture_blend_weight 解析 descriptor 融合权重。
    /// @return 融合后的 descriptor 图。
    torch::Tensor descriptorMapSingle(const torch::Tensor& image,
                                      double texture_blend_weight = INFERENCE_TEXTURE_BLEND_WEIGHT);

    /// 计算 Python `forward_single` 暴露的全部原始特征图。
    /// @param image BxCxHxW 图像张量。
    /// @param texture_blend_weight 解析 descriptor 融合权重。
    /// @return 原始特征图集合。
    PfmV21RawFeatureMaps forwardSingle(const torch::Tensor& image,
                                       double texture_blend_weight = INFERENCE_TEXTURE_BLEND_WEIGHT);

  private:
    PfmV21Config _config;
    PfmV21Backbone _backbone{nullptr};
    PfmV21DualFPNLite _dual_fpn{nullptr};
    PfmV21SparseHead _sparse_head{nullptr};
    PfmV21TextureDescriptorAdapter _texture_adapter{nullptr};
    PfmV21DescriptorFusionAdapter _descriptor_fusion{nullptr};
    PfmV21DenseHead _dense_head{nullptr};
    PfmV21QualityHead _quality_head{nullptr};
    PfmV21SemiDenseCandidateBranch _semi_dense_branch{nullptr};
    PfmV21GraphMatcher _graph_matcher{nullptr};
};

TORCH_MODULE(PfmV21FeatureMatcher);

/// 构建当前 Python 版本对应的解析纹理 descriptor 图。
/// @param image 输入图像张量。
/// @param descriptor_height 目标 descriptor 图高度。
/// @param descriptor_width 目标 descriptor 图宽度。
/// @param descriptor_dim 目标 descriptor 通道数。
/// @return 归一化后的 descriptor 张量。
torch::Tensor makeRotationInvariantTextureDescriptor(const torch::Tensor& image, int64_t descriptor_height,
                                                     int64_t descriptor_width, int64_t descriptor_dim);

/// 构建当前 Python 版本对应的解析纹理显著性图。
/// @param image 输入图像张量。
/// @param target_height 目标图高度。
/// @param target_width 目标图宽度。
/// @return 归一化后的显著性图。
torch::Tensor makeRotationInvariantTextureSaliency(const torch::Tensor& image, int64_t target_height,
                                                   int64_t target_width);

} // namespace pfm::v21
