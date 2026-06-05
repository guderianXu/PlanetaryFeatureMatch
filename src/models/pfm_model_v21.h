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
    /// 输入图像通道数。
    int64_t input_channels = 1;
    /// backbone 基础通道数。
    int64_t base_channels = 64;
    /// 描述子通道数。
    int64_t descriptor_dim = 256;
    /// 图匹配器隐藏层维度。
    int64_t graph_hidden_dim = 512;
    /// 图匹配器注意力层数。
    int64_t graph_attention_layers = 8;
    /// 图匹配器关键点元数据维度。
    int64_t graph_keypoint_meta_dim = 16;
};

struct PfmV21SparseHeadOutput
{
    /// 稀疏关键点热力图。
    torch::Tensor heatmap;
    /// 稀疏描述子图。
    torch::Tensor descriptors;
    /// 关键点尺度图。
    torch::Tensor scale;
    /// 关键点方向图。
    torch::Tensor orientation;
    /// 局部仿射参数图。
    torch::Tensor affine;
    /// 关键点亚像素偏移图。
    torch::Tensor keypoint_offsets;
};

struct PfmV21DenseHeadOutput
{
    /// 稠密匹配置信度图。
    torch::Tensor confidence;
    /// 稠密匹配偏移图。
    torch::Tensor offsets;
};

struct PfmV21GraphMatcherOutput
{
    /// 带 dustbin 行列的匹配 logits。
    torch::Tensor logits;
    /// 互选匹配索引。
    torch::Tensor matches;
    /// 互选匹配概率分数。
    torch::Tensor scores;
    /// 每个候选匹配的接受 logits。
    torch::Tensor accept_logits;
    /// 实际执行的图注意力层数。
    int64_t executed_layers = 0;
    /// 输入 A 视图关键点数。
    int64_t input_keypoints_a = 0;
    /// 输入 B 视图关键点数。
    int64_t input_keypoints_b = 0;
    /// 自适应剪枝后保留的 A 视图关键点数。
    int64_t kept_keypoints_a = 0;
    /// 自适应剪枝后保留的 B 视图关键点数。
    int64_t kept_keypoints_b = 0;
    /// 自适应剪枝移除的 A 视图关键点数。
    int64_t pruned_keypoints_a = 0;
    /// 自适应剪枝移除的 B 视图关键点数。
    int64_t pruned_keypoints_b = 0;
    /// 实际执行的注意力点对计算量代理，按每层 N_a * N_b 累加。
    int64_t attention_work_units = 0;
    /// 不早停、不剪枝时的满层注意力点对计算量代理。
    int64_t full_attention_work_units = 0;
    /// 实际计算量占满计算量的比例。
    double attention_work_fraction = 0.0;
};

struct PfmV21SemiDenseCandidateOutput
{
    /// A 视图半稠密候选关键点。
    torch::Tensor keypoints_a;
    /// B 视图半稠密候选关键点。
    torch::Tensor keypoints_b;
    /// 候选点对分数。
    torch::Tensor scores;
};

struct PfmV21RawFeatureMaps
{
    /// 质量校正后的稀疏关键点热力图。
    torch::Tensor heatmap;
    /// 融合后的描述子图。
    torch::Tensor descriptors;
    /// 关键点尺度图。
    torch::Tensor scale;
    /// 关键点方向图。
    torch::Tensor orientation;
    /// 局部仿射参数图。
    torch::Tensor affine;
    /// 稠密匹配置信度图。
    torch::Tensor dense_confidence;
    /// 关键点亚像素偏移图。
    torch::Tensor keypoint_offsets;
    /// 描述子/关键点可靠性图。
    torch::Tensor quality;
    /// 局部纹理对比度图。
    torch::Tensor local_contrast;
};

class PfmV21ZeroResidualContextBlockImpl : public torch::nn::Module
{
  public:
    /// 创建零初始化的残差上下文块。
    /// @param channels 输入和输出通道数。
    /// @param dilation 两个空间卷积使用的空洞率。
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
    /// @param base_channels 第一层 backbone 阶段的通道数。
    /// @throws std::invalid_argument 当 input_channels 或 base_channels 非正时抛出。
    PfmV21BackboneImpl(int64_t input_channels, int64_t base_channels);

    /// 计算四级经过精化的 stride-2 特征。
    /// @param x BxCxHxW 输入张量。
    /// @return stride 分别为 2、4、8、16 的特征阶段。
    std::vector<torch::Tensor> forward(const torch::Tensor& x);

    /// 替换归一化层中的非有限 buffer，保持检查点加载兼容性。
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
    /// 为关键点和描述子创建独立的 1/4 分辨率 FPN 分支。
    /// @param base_channels backbone 基础通道数。
    /// @throws std::invalid_argument 当 base_channels 非正时抛出。
    explicit PfmV21DualFPNLiteImpl(int64_t base_channels);

    /// 从 backbone 多级特征生成关键点和描述子特征图。
    /// @param features backbone 的第 1 到第 4 级特征阶段。
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
    /// 创建当前 Python 稀疏头的 C++ 镜像实现。
    /// @param input_channels 输入特征通道数。
    /// @param descriptor_dim 输出描述子通道数。
    /// @throws std::invalid_argument 当 input_channels 或 descriptor_dim 非正时抛出。
    PfmV21SparseHeadImpl(int64_t input_channels, int64_t descriptor_dim);

    /// 从共享特征张量预测稀疏输出图。
    /// @param feature 关键点和描述子共享的输入特征图。
    /// @return 与 Python `SparseHeadOutput` 对齐的稀疏头输出。
    PfmV21SparseHeadOutput forward(const torch::Tensor& feature);

    /// 从独立的关键点和描述子特征张量预测稀疏输出图。
    /// @param feature 关键点特征图。
    /// @param descriptor_feature 同一网格上的描述子特征图。
    /// @return 与 Python `SparseHeadOutput` 对齐的稀疏头输出。
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
    /// 创建当前 Python 稠密头的 C++ 镜像实现。
    /// @param feature_channels 每张稠密特征图的通道数。
    /// @throws std::invalid_argument 当 feature_channels 非正时抛出。
    explicit PfmV21DenseHeadImpl(int64_t feature_channels);

    /// 从一对稠密特征预测置信度和偏移量。
    /// @param feature_a 第一张 BxCxHxW 特征张量。
    /// @param feature_b 第二张 BxCxHxW 特征张量。
    /// @return 稠密置信度和 x/y 偏移量。
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
    /// 创建一层自注意力/交叉注意力图层。
    /// @param hidden_dim 图内部特征维度。
    /// @throws std::invalid_argument 当 hidden_dim 非正时抛出。
    explicit PfmV21GraphAttentionLayerImpl(int64_t hidden_dim);

    /// 对两组关键点图特征做精化。
    /// @param features_a NxH 的图 A 特征。
    /// @param features_b MxH 的图 B 特征。
    /// @return 精化后的两组图特征。
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
    /// 创建当前 Python 图匹配器的 C++ 镜像实现。
    /// @param descriptor_dim 描述子通道数。
    /// @param hidden_dim 图内部隐藏维度。
    /// @param attention_layers 图注意力层数。
    /// @param keypoint_meta_dim 关键点投影接收的元数据维度。
    /// @param candidate_topk 候选匹配剪枝的 top-k。
    /// @throws std::invalid_argument 当维度或层数非正时抛出。
    PfmV21GraphMatcherImpl(int64_t descriptor_dim, int64_t hidden_dim, int64_t attention_layers = 1,
                           int64_t keypoint_meta_dim = 2, int64_t candidate_topk = 64);

    /// 在稀疏描述子和关键点元数据上运行图匹配。
    /// @param descriptors_a NxD 的 A 视图描述子。
    /// @param keypoints_a NxC 的 A 视图关键点或元数据。
    /// @param descriptors_b MxD 的 B 视图描述子。
    /// @param keypoints_b MxC 的 B 视图关键点或元数据。
    /// @param apply_candidate_mask 是否应用候选 top-k 剪枝；训练监督 loss 可关闭以避免随机初期屏蔽真值。
    /// @param width_prune_min_score LightGlue 风格宽度剪枝阈值；-1 表示关闭。
    /// @param early_stop_min_confidence LightGlue 风格深度提前停止阈值；-1 表示关闭。
    /// @param max_attention_layers LightGlue 风格深度预算硬上限；0 表示执行完整层数。
    /// @return 匹配 logits、互选匹配、概率和接受 logits。
    PfmV21GraphMatcherOutput forward(const torch::Tensor& descriptors_a, const torch::Tensor& keypoints_a,
                                     const torch::Tensor& descriptors_b, const torch::Tensor& keypoints_b,
                                     bool apply_candidate_mask = true, double width_prune_min_score = -1.0,
                                     double early_stop_min_confidence = -1.0, int64_t max_attention_layers = 0);

    /// 返回上一次 forward 实际执行的 attention 层数，便于调试 early stopping。
    int64_t lastExecutedAttentionLayers() const;

  private:
    torch::Tensor metadata(const torch::Tensor& keypoints_or_meta) const;
    torch::Tensor geometryCompatibilityBias(const torch::Tensor& meta_a, const torch::Tensor& meta_b);
    torch::Tensor candidateMask(const torch::Tensor& desc_a, const torch::Tensor& desc_b) const;
    torch::Tensor acceptanceLogits(const torch::Tensor& raw_similarity, const torch::Tensor& graph_delta,
                                   const torch::Tensor& meta_a, const torch::Tensor& meta_b);
    torch::Tensor provisionalPairLogits(const torch::Tensor& embed_a, const torch::Tensor& embed_b,
                                        const torch::Tensor& raw_similarity, const torch::Tensor& meta_a,
                                        const torch::Tensor& meta_b);
    std::pair<torch::Tensor, torch::Tensor> provisionalPairOutputs(const torch::Tensor& embed_a,
                                                                   const torch::Tensor& embed_b,
                                                                   const torch::Tensor& raw_similarity,
                                                                   const torch::Tensor& meta_a,
                                                                   const torch::Tensor& meta_b);
    static torch::Tensor assignmentConfidence(const torch::Tensor& pair_logits);
    static std::pair<torch::Tensor, torch::Tensor> acceptanceKeepMasks(const torch::Tensor& accept_logits,
                                                                       double min_probability);

    int64_t _descriptor_dim;
    int64_t _hidden_dim;
    int64_t _attention_layer_count;
    int64_t _keypoint_meta_dim;
    int64_t _candidate_topk;
    int64_t _last_executed_attention_layers = 0;
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
    /// 创建可训练的残差纹理描述子适配器。
    /// @param descriptor_dim 描述子通道数。
    /// @throws std::invalid_argument 当 descriptor_dim 非正时抛出。
    explicit PfmV21TextureDescriptorAdapterImpl(int64_t descriptor_dim);

    /// 应用残差适配和通道归一化。
    /// @param texture BxDxHxW 纹理描述子图。
    /// @return 适配后的纹理描述子图。
    torch::Tensor forward(const torch::Tensor& texture);

  private:
    int64_t _descriptor_dim;
    torch::nn::Conv2d _residual{nullptr};
};

TORCH_MODULE(PfmV21TextureDescriptorAdapter);

class PfmV21DescriptorFusionAdapterImpl : public torch::nn::Module
{
  public:
    /// 创建已学习描述子和解析纹理描述子的融合适配器。
    /// @param descriptor_dim 描述子通道数。
    /// @param hidden_dim 可选隐藏通道数；非正值使用 Python 默认设置。
    /// @throws std::invalid_argument 当 descriptor_dim 非正时抛出。
    explicit PfmV21DescriptorFusionAdapterImpl(int64_t descriptor_dim, int64_t hidden_dim = 0);

    /// 融合已学习描述子和纹理描述子。
    /// @param learned 已学习描述子图。
    /// @param texture 解析纹理描述子图。
    /// @param blend_weight 解析描述子融合权重。
    /// @return 融合并归一化后的描述子图。
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
    /// 创建描述子/关键点可靠性头。
    /// @param descriptor_dim 描述子通道数。
    /// @throws std::invalid_argument 当 descriptor_dim 非正时抛出。
    explicit PfmV21QualityHeadImpl(int64_t descriptor_dim);

    /// 从描述子和辅助图预测可靠性。
    /// @param descriptors BxDxHxW 融合描述子图。
    /// @param heatmap Bx1xHxW 稀疏热力图。
    /// @param texture_saliency Bx1xHxW 纹理显著性图。
    /// @param dense_confidence Bx1xHxW 稠密置信度图。
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
    /// 创建弱纹理场景下的半稠密候选分支。
    /// @param descriptor_dim 描述子通道数。
    /// @param projection_dim 投影通道数。
    /// @param max_grid 粗网格最大边长。
    /// @throws std::invalid_argument 当任意维度非正时抛出。
    PfmV21SemiDenseCandidateBranchImpl(int64_t descriptor_dim, int64_t projection_dim = 64, int64_t max_grid = 32);

    /// 从描述子图生成无检测器候选匹配点对。
    /// @param descriptors_a A 视图描述子图。
    /// @param descriptors_b B 视图描述子图。
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

    /// 计算解析纹理融合前的已学习描述子图。
    /// @param image BxCxHxW 图像张量。
    /// @return 已学习描述子图。
    torch::Tensor learnedDescriptorMapSingle(const torch::Tensor& image);

    /// 计算原始解析纹理描述子图。
    /// @param image BxCxHxW 图像张量。
    /// @return 1/4 分辨率的原始解析描述子图。
    torch::Tensor rawTextureDescriptorMapSingle(const torch::Tensor& image);

    /// 计算适配后的解析纹理描述子图。
    /// @param image BxCxHxW 图像张量。
    /// @return 适配后的解析描述子图。
    torch::Tensor textureDescriptorMapSingle(const torch::Tensor& image);

    /// 融合已学习描述子和解析纹理描述子。
    /// @param learned_descriptors 已学习描述子图。
    /// @param image 源图像张量。
    /// @param texture_blend_weight 解析描述子融合权重。
    /// @return 融合后的描述子图。
    torch::Tensor fuseDescriptorMaps(const torch::Tensor& learned_descriptors, const torch::Tensor& image,
                                     double texture_blend_weight = INFERENCE_TEXTURE_BLEND_WEIGHT);

    /// 为单个图像 batch 计算融合 descriptor 图。
    /// @param image BxCxHxW 图像张量。
    /// @param texture_blend_weight 解析描述子融合权重。
    /// @return 融合后的描述子图。
    torch::Tensor descriptorMapSingle(const torch::Tensor& image,
                                      double texture_blend_weight = INFERENCE_TEXTURE_BLEND_WEIGHT);

    /// 计算 Python `forward_single` 暴露的全部原始特征图。
    /// @param image BxCxHxW 图像张量。
    /// @param texture_blend_weight 解析描述子融合权重。
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

/// 构建当前 Python 版本对应的解析纹理描述子图。
/// @param image 输入图像张量。
/// @param descriptor_height 目标描述子图高度。
/// @param descriptor_width 目标描述子图宽度。
/// @param descriptor_dim 目标描述子通道数。
/// @return 归一化后的描述子张量。
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
