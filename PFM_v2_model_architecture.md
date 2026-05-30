# PFM v2.1 完整特征提取与匹配模型架构设计稿

这是面向行星影像跨视角、局部畸变、强光照差异、旋转、拍摄角度变化、弱纹理和重复地貌纹理的 PFM v2.1 设计稿。

实现状态更新（2026-05-27）：

```text
PyTorch 主路径已经一次性落地 v2.1-full 主架构。
C++/LibTorch 正式推理端尚未同步完整 v2.1。
旧 v2-lite checkpoint 仍按自身配置加载，不会静默升级成 full 256。
```

已落地代码：

```text
python/pfm_model.py:
  DualFPNLite
  SparseHead separate P2_kp / P2_desc
  geometry-aware descriptor pooling
  QualityHead
  illumination-robust texture descriptor + adaptive fusion
  SemiDenseCandidateBranch coarse detector-free candidates
  GraphMatcherV2 rich metadata / top-k pruning / geometry bias / dual-softmax

python/pytorch_cache_match_eval.py:
  GraphMatcher metadata sampled from RawFeatureMaps
  sparse + semi-dense candidates merged before GraphMatcher
  graph matcher weak-output raw descriptor fallback merge

python/pfm_pytorch_training.py:
  train-backbone / train-dual-fpn / train-geometry-head / train-quality-head / train-graph-matcher
  graph matcher correspondence CE loss
```

PFM v2.1 的重点不再是简单堆更大的特征维度或更强的 GraphMatcher，而是让描述子本身具备：

```text
光照鲁棒性
几何归一化能力
弱纹理适应能力
重复地貌判别能力
可靠 no-match / dustbin 拒配能力
```

核心定位：

```text
PFM v2.1 是一个面向行星影像极端成像差异的
geometry-aware、illumination-robust、texture-adaptive
local feature matching network。
```

## 总体流程

```text
Image A/B
  ↓
BackboneV2 + Dual-FPN-Lite
  ├─ P2_kp: local-detail feature
  └─ P2_desc: multi-scale context feature
  ↓
SparseHeadV2
  ├─ KeypointHead
  │    ├─ heatmap
  │    └─ subpixel offset
  │
  ├─ GeometryHead
  │    ├─ log_scale
  │    ├─ orientation unit vector
  │    └─ residual affine: I + 0.1 * tanh(ΔA)
  │
  ├─ DescriptorHead
  │    ├─ C4 branch descriptors
  │    ├─ branch-quality attention
  │    ├─ rotation consistency training
  │    └─ optional geometry-aware local sampling
  │
  ├─ IlluminationRobustTextureBranch
  │    ├─ local normalized intensity
  │    ├─ gradient / DoG / LoG
  │    ├─ rank/census-like statistics
  │    └─ radial ring contrast
  │
  └─ AdaptiveTextureFusion
       ├─ texture quality estimation
       ├─ spatial texture gate
       └─ L2 normalized fused descriptor
  ↓
Feature Decode
  ├─ NMS
  ├─ grid-balanced sampling
  ├─ validity/local-contrast mask
  ├─ texture-adaptive quota
  └─ subpixel refinement
  ↓
Optional semi-dense coarse candidates for weak texture
  ↓
GraphMatcherV2
  ├─ descriptor projection
  ├─ x/y + geometry + quality encoding
  ├─ self/cross attention
  ├─ top-k sparse candidate pruning
  ├─ dual-softmax or Sinkhorn
  └─ dustbin/no-match
  ↓
Sparse matches + confidence
```

## 1. 需要保留的 v2 基础模块

PFM v2 已有方向中值得保留的部分：

```text
BackboneV2 + FPN-Lite
SparseHeadV2
subpixel offset
C4 attention descriptor
GeometryHead
analytic texture descriptor
TextureFusion
GraphMatcherV2
raw_descriptor + graph_matcher 双报告
```

但这些模块需要升级成解决行星影像困难匹配的主机制：

```text
GeometryHead 不只作为 metadata，而要进入 descriptor canonicalization。
TextureFusion 不只做特征拼接，而要成为光照鲁棒自适应融合。
GraphMatcher 不应掩盖 descriptor 问题，训练报告必须同时保留 raw descriptor 口径。
```

## 2. BackboneV2 与 Dual-FPN-Lite

输入为 `1 x H x W` 灰度影像。

建议保留两个配置：

```text
PFM-v2-Lite:
  base_channels = 48
  descriptor_dim = 192
  graph_hidden_dim = 384
  graph_attention_layers = 4 或 6

PFM-v2-Full:
  base_channels = 64
  descriptor_dim = 256
  graph_hidden_dim = 512
  graph_attention_layers = 8
```

开发和 ablation 先用 Lite。只有 Lite 证明有效后，再上 Full。

Full 版特征金字塔：

```text
Stage1: 1   → 64   stride=2   输出 H/2  x W/2
Stage2: 64  → 128  stride=2   输出 H/4  x W/4
Stage3: 128 → 256  stride=2   输出 H/8  x W/8
Stage4: 256 → 512  stride=2   输出 H/16 x W/16
```

每个 stage 从浅层 `Conv-BN-ReLU` 改为：

```text
Downsample Conv-GN-GELU
+ ResidualLocalBlock x2
+ DilatedContextBlock
```

优先使用 GroupNorm / InstanceNorm，而不是强依赖 BatchNorm。原因是 1024 crop + GraphMatcher 下 batch size 通常较小，BN 统计不稳定。

Dual-FPN-Lite：

```text
P2_kp   = Stage2 + light upsample(Stage3)
P2_desc = Stage2 + upsample(Stage3) + upsample(Stage4)
```

原因：

```text
KeypointHead 需要空间定位精确，少用过粗上下文。
DescriptorHead 需要上下文鲁棒，多用中高层语义和几何上下文。
```

后续 keypoint 和 descriptor 都保持 `1/4` 分辨率。

## 3. SparseHeadV2

SparseHead 拆成三个相对独立的分支，减少训练目标互相干扰。

### 3.1 KeypointHead

```text
P2_kp
 → keypoint_context
 → heatmap: 1 x H/4 x W/4
 → keypoint_offset: 2 x H/4 x W/4
```

`keypoint_offset = tanh(offset) * 0.5`，表示当前 feature cell 内的亚像素偏移。decode 时关键点坐标从：

```text
(x, y)
```

变为：

```text
(x + dx, y + dy)
```

KeypointHead 的训练重点是重复性，而不是单图显著性：

```text
同一地貌位置在不同光照、视角、旋转、尺度下应重复检测。
```

### 3.2 GeometryHead

输出形式：

```text
log_scale:       1 x H/4 x W/4
orientation_raw: 2 x H/4 x W/4
affine_delta:    4 x H/4 x W/4
```

decode 规则：

```text
scale = exp(clamp(log_scale, min=-s, max=s))
orientation = normalize(orientation_raw, dim=channel)
affine = I + 0.1 * tanh(affine_delta)
```

不要直接输出线性 scale，避免负尺度或数值爆炸。affine 用 residual identity，避免不可逆、过大 shear 或采样区域乱飞。

正则：

```text
L_affine_reg = ||A - I||^2
```

GeometryHead 必须进入 descriptor normalization，而不是只作为 GraphMatcher metadata。

最小改动版：

```text
P2_desc
+ keypoint location
+ predicted orientation
→ rotate local sampling grid
→ pool descriptor
```

完整版本：

```text
keypoint k:
  loc = (x, y)
  theta = atan2(ori_y, ori_x)
  scale = exp(log_scale)
  A = I + 0.1 * tanh(ΔA)

  grid = affine_grid(theta, scale, A)
  local_patch = grid_sample(P2_desc, grid)
  desc_k = descriptor_pool(local_patch)
```

### 3.3 DescriptorHead

输入：

```text
P2_desc
```

C4 不再直接平均，而是由每个旋转分支自身产生质量权重：

```text
desc_0   = tower(P2_desc)
desc_90  = unrotate(tower(rot90(P2_desc)))
desc_180 = unrotate(tower(rot180(P2_desc)))
desc_270 = unrotate(tower(rot270(P2_desc)))

quality_0   = Conv(desc_0)
quality_90  = Conv(desc_90)
quality_180 = Conv(desc_180)
quality_270 = Conv(desc_270)

weights = softmax([quality_0, quality_90, quality_180, quality_270], dim=rotation)

desc_inv = w0*desc_0 + w90*desc_90 + w180*desc_180 + w270*desc_270
desc_inv = L2Norm(desc_inv)
```

保留 rotation-invariant 和 rotation-sensitive 两种信息：

```text
desc_inv = C4 attention fused descriptor
desc_eq  = normal descriptor without C4 fusion

learned_desc = Conv([desc_inv, desc_eq])
learned_desc = L2Norm(learned_desc)
```

这样同时保留：

```text
旋转鲁棒性
+ 方向判别性
```

输出：

```text
learned_desc: descriptor_dim x H/4 x W/4
```

## 4. IlluminationRobustTextureBranch

Texture branch 从“纹理拼接”升级成“光照鲁棒分支”。它不应主要依赖绝对灰度，而应强调：

```text
局部相对强度
局部排序关系
梯度方向结构
多尺度响应
环形邻域相对统计
局部对比度
```

建议 texture channels：

```text
texture_channels = [
  local_normalized_intensity,
  local_contrast,
  gradient_magnitude,
  gradient_orientation_sin,
  gradient_orientation_cos,
  DoG_small,
  DoG_large,
  LoG_response,
  rank_transform_like_feature,
  census_like_binary_or_soft_feature,
  radial_ring_mean_diff,
  radial_ring_contrast,
  annular_gradient_statistics,
  shadow_edge_response
]
```

关键构造：

```text
local_normalized_intensity = (I - local_mean) / (local_std + eps)
ring_contrast = mean(inner_ring) - mean(outer_ring)
rank/census-like = 比较中心点与邻域点的相对大小
DoG/LoG = 弱化整体亮度，强调结构变化
```

目标是让 texture descriptor 更像光照鲁棒的行星纹理描述子，而不是简单灰度统计。

## 5. AdaptiveTextureFusion

从固定全局 `texture_weight` 改成局部自适应 gate。

当前粗略形式：

```text
final_desc = L2Norm(learned_desc + texture_weight * texture_desc + fusion_residual)
```

建议形式：

```text
texture_quality = QualityHead(texture_channels)

texture_gate = sigmoid(Conv([
    learned_desc,
    texture_desc,
    local_contrast,
    gradient_energy,
    texture_entropy,
    texture_quality
]))

fusion_input = [
  learned_desc,
  texture_gate * texture_desc,
  learned_desc - texture_desc,
  learned_desc * texture_desc
]

fusion_residual = Conv/GELU/Conv/GELU/Conv(fusion_input)

final_desc = L2Norm(
    learned_desc
    + texture_gate * texture_desc
    + fusion_residual
)
```

初版建议使用：

```text
texture_gate: 1 x H/4 x W/4
```

不要一开始就做 256 通道 gate，避免训练不稳。

局部区域策略：

```text
撞击坑边缘、沟槽边缘：更多依赖 learned descriptor
弱纹理平原：更多依赖 analytic texture + 多尺度上下文
强阴影边界：更多依赖 gradient / rank / local contrast
重复 crater 区：更多依赖 learned descriptor + GraphMatcher
低对比度区域：更多依赖 local normalization + texture prior
过曝/极暗区域：降低 keypoint / descriptor confidence
```

## 6. Feature Decode

从 heatmap 中选点：

```text
heatmap NMS
+ grid-balanced sampling
+ validity/local-contrast mask
+ texture-adaptive quota
+ subpixel offset refinement
```

取消简单 `min intensity mask` 作为主过滤逻辑。强阴影区域可能仍有稳定边界或纹理；真正应该过滤的是：

```text
无效值区域
饱和区域
极低局部对比度区域
边界 padding 区域
明显噪声区域
```

每个 grid cell 保留不同类型候选：

```text
top heatmap keypoints
top gradient/structure keypoints
top low-texture semi-dense candidates
```

每个关键点包含：

```text
(x, y)
score
descriptor[descriptor_dim]
log_scale / scale
orientation
residual affine
texture_quality
local_contrast
descriptor_uncertainty 或 reliability
```

## 7. Weak-texture Semi-dense Fallback

Sparse keypoint pipeline 对强纹理区域有效，但弱纹理行星区域可能选不出稳定点。v2.1 加一个轻量 semi-dense candidate branch：

```text
P2_desc_A, P2_desc_B
→ coarse correlation / dual-softmax
→ top-K semi-dense candidate matches
→ merge with sparse keypoint matches
→ GraphMatcherV2 refine
```

最终候选变成：

```text
Sparse keypoint candidates
+ Semi-dense weak-texture candidates
→ GraphMatcherV2
```

该模块主要针对：

```text
弱纹理平原
阴影内部
低对比度区域
重复纹理区域
keypoint 数量不足的场景
```

## 8. GraphMatcherV2

GraphMatcherV2 不应只是更大，而要几何感知、质量感知，并做候选稀疏化。

输入：

```text
features_a: Na x descriptor_dim
features_b: Nb x descriptor_dim

keypoint_meta:
  x_norm
  y_norm
  radius
  radius^2
  score
  log_scale
  orientation_x
  orientation_y
  affine_00
  affine_01
  affine_10
  affine_11
  texture_quality
  local_contrast
  descriptor_uncertainty
```

节点编码：

```text
node = descriptor_projection(desc)
     + keypoint_meta_projection(meta)
```

Graph attention：

```text
Self-Attention on A
Self-Attention on B
Cross-Attention A ↔ B
FFN
重复 L 层
```

### 8.1 Top-k Candidate Pruning

GraphMatcher 前先用 raw descriptor 相似度取候选：

```text
raw descriptor similarity
→ 每个点取 top-k candidates, k=32 或 64
→ GraphMatcher 只在候选边上推理
```

目标：

```text
把 O(Na * Nb) 降到近似 O(N * k)
减少重复纹理误匹配
提升训练稳定性
```

### 8.2 Pairwise Geometry Compatibility Bias

匹配 logits 加局部几何兼容项：

```text
logit_ij = desc_score_ij + geo_bias_ij
```

其中：

```text
geo_bias_ij = MLP([
  Δx_norm,
  Δy_norm,
  scale_i - scale_j,
  cos(theta_i - theta_j),
  affine_similarity,
  quality_i,
  quality_j
])
```

该项帮助处理：

```text
旋转差异
尺度差异
局部仿射差异
重复纹理误匹配
```

### 8.3 Assignment 输出

初版：

```text
dual-softmax + dustbin
```

形式：

```text
P_ij = softmax(logits, dim=row)_ij * softmax(logits, dim=col)_ij
```

再做：

```text
mutual check
dustbin filtering
score threshold
margin threshold
```

完整版后续再考虑：

```text
Sinkhorn / optimal transport
```

## 9. QualityHead

增加轻量质量分支：

```text
QualityHead:
  texture_quality
  illumination_confidence
  keypoint_reliability
  descriptor_uncertainty
```

输出：

```text
quality: 1 x H/4 x W/4
```

用途：

```text
Feature Decode 时调整采样
TextureFusion 时作为 gate 输入
GraphMatcher 时作为 node confidence
最终 score 时参与置信度
```

示例：

```text
final_keypoint_score =
  heatmap_score
  * descriptor_quality
  * texture_quality
  * validity_mask
```

## 10. 训练阶段

v2.1 模块多，不建议端到端一次性全开。

### Stage A：只训 backbone + keypoint + learned descriptor

关闭：

```text
TextureFusion
GeometryHead 主损失
GraphMatcher
```

训练：

```text
L = L_kp + L_offset + L_desc + L_repeat + L_rot
```

目标：

```text
raw_descriptor matching 必须先能工作
```

### Stage B：加入 illumination-robust TextureFusion

训练：

```text
analytic texture adapter
texture gate
fusion residual
descriptor head 后半部分
```

评估：

```text
raw learned descriptor
raw fused descriptor
```

确认 texture branch 是提升，不是污染。

### Stage C：加入 GeometryHead + canonical sampling

训练：

```text
orientation
log_scale
affine residual
geometry-aware descriptor pooling
```

重点评估：

```text
rotation subset
scale subset
viewpoint / affine subset
```

### Stage D：加入 GraphMatcherV2

冻结或小学习率训练 extractor：

```text
extractor lr = 1e-5 ~ 3e-5
matcher lr = 3e-4 ~ 1e-3
```

目标：

```text
GraphMatcher 不要把已经可用的 descriptor 压坏
```

### Stage E：端到端微调

全部打开，但控制 loss 权重：

```text
L = L_kp
  + L_offset
  + L_desc
  + L_repeat
  + L_rot
  + L_photo
  + L_geo
  + L_match
  + L_reg
```

不要一开始让 `L_match` 完全主导，否则 GraphMatcher 会补偿 descriptor 缺陷，导致特征提取器本身不强。

## 11. Loss 设计

loss 围绕五类困难变化设计：

```text
光照变化
旋转变化
尺度变化
跨视角 / 仿射畸变
弱纹理 / 重复纹理
```

### 11.1 Keypoint Repeatability Loss

给一张图像做已知变换：

```text
I_a → T(I_a) = I_b
```

其中 T 包括：

```text
rotation
scale
affine
perspective
illumination transform
shadow simulation
blur / noise
```

模型输出：

```text
heatmap_a
heatmap_b
```

将 heatmap_a warp 到 b 坐标系：

```text
warp(heatmap_a, T) ≈ heatmap_b
```

损失：

```text
L_repeat = BCE / MSE / focal consistency
```

### 11.2 Subpixel Offset Loss

有已知几何变换 T 时：

```text
p_b_gt = T(p_a)
cell_b = floor(p_b_gt / stride)
offset_b_gt = p_b_gt / stride - cell_b
```

损失：

```text
L_offset = SmoothL1(offset_pred, offset_gt)
```

没有显式 GT 时，用一致性约束：

```text
warp(kp_a + offset_a) 与 kp_b + offset_b 的距离
```

### 11.3 Photometric Consistency Loss

对同一张图做强光照变换：

```text
I_b = PhotoTransform(I_a)
```

包括：

```text
gamma
local contrast
shadow simulation
directional shading
histogram shift
blur / noise
```

要求：

```text
desc_a(p) ≈ desc_b(p)
```

损失：

```text
L_photo = 1 - cosine(desc_a(p), desc_b(p))
```

### 11.4 Rotation Consistency Loss

```text
I_b = Rotate(I_a, θ)
p_b = Rθ(p_a)
desc_a(p_a) ≈ desc_b(p_b)
```

损失：

```text
L_rot = 1 - cosine(desc_a, desc_b)
```

同时监督 orientation：

```text
theta_pred_b ≈ theta_pred_a + θ
```

### 11.5 Affine Consistency Loss

```text
I_b = AffineWarp(I_a, A)
p_b = A(p_a)
```

要求：

```text
descriptor consistency
keypoint repeatability
offset consistency
affine prediction consistency
```

### 11.6 Hard Negative Descriptor Loss

行星影像重复地貌多，普通 InfoNCE 不够。需要 hard negative mining：

```text
positive = same physical / warped location
hard negative = nearby but not same location, similar texture
```

可用：

```text
InfoNCE + hardest in-batch negative
Circle loss
Triplet loss with hard negative
```

目标是区分：

```text
相似 crater 但不同位置
相似沟槽但不同结构
相似阴影边界但不同地貌
```

### 11.7 GraphMatcher Assignment Loss

如果有 GT match matrix：

```text
L_match = cross_entropy(assignment_matrix, gt_assignment)
```

无匹配点监督到 dustbin：

```text
unmatched keypoint → dustbin
```

## 12. 行星影像专用 Augmentation Pipeline

不要照搬自然图像分类增强。建议专门做：

```text
Photometric:
  gamma
  local contrast
  histogram shift
  shadow cast simulation
  directional shading
  low sun angle simulation
  blur
  sensor noise
  compression artifacts

Geometric:
  rotation: 0°~360°
  scale: 0.5~2.0
  affine shear
  local perspective
  mild elastic / terrain-like warp

Resolution:
  downsample-upsample
  anisotropic blur
  sensor resolution mismatch

Mask:
  no-data mask
  border mask
  saturation mask
```

训练时生成成对样本：

```text
I_a
I_b = GeoTransform(PhotoTransform(I_a))
known correspondence from transform
```

## 13. 评估与训练报告

不要只报一个总分。按困难类型拆分评估：

```text
Set A: illumination difference
Set B: rotation
Set C: scale difference
Set D: viewpoint / affine distortion
Set E: weak texture
Set F: repetitive terrain
Set G: mixed hard cases
```

每个子集报告：

```text
repeatability
matching precision
matching recall
F1
number of correct matches
inlier ratio
mean localization error
graph matcher accepted ratio
dustbin accuracy
```

报告必须保留双口径：

```text
raw_descriptor:
  descriptor mutual nearest + min_margin
  用于诊断 descriptor 本身

graph_matcher:
  正式 GraphMatcher 输出
  用于判断真实推理效果
```

再增加：

```text
extractor_only + simple MNN
extractor + graph matcher
extractor + graph matcher + optional geometry filter
```

现象诊断：

| 现象 | 说明 |
| --- | --- |
| raw_descriptor 好，graph_matcher 差 | matcher 训练或阈值有问题 |
| raw_descriptor 差，graph_matcher 好 | matcher 在补偿 descriptor，不利于证明特征本身 |
| 两者都差 | extractor / texture / loss 有问题 |
| illumination 子集差 | texture branch 和 photometric loss 不够 |
| rotation 子集差 | C4 或 orientation loss 不够 |
| weak texture 子集差 | sparse detector 失败，需要 semi-dense fallback |
| repetitive terrain 差 | descriptor 判别性或 hard negative 不够 |

## 14. 立刻优先改的 8 个点

按优先级排序：

1. **TextureFusion 改成 adaptive gate**

```text
gate = sigmoid(Conv([learned_desc, texture_desc, texture_quality]))

final_desc = L2Norm(
    learned_desc
    + gate * texture_desc
    + fusion_residual
)
```

2. **texture descriptor 改成 illumination-robust descriptor**

增加：

```text
local normalized intensity
DoG / LoG
rank / census-like
gradient orientation
radial contrast
local contrast
```

3. **GeometryHead 接入 descriptor sampling**

至少做：

```text
orientation-aware descriptor pooling
```

后续完整做：

```text
scale + orientation + affine canonical sampling
```

4. **C4 attention 权重由各分支 descriptor 产生**

```text
quality_i = Conv(desc_i)
weights = softmax([quality_0, quality_90, quality_180, quality_270])
```

并加入 rotation consistency loss。

5. **GraphMatcher meta 加 x/y 和 texture quality**

至少包含：

```text
x_norm
y_norm
score
log_scale
orientation_x
orientation_y
texture_quality
local_contrast
```

`radius/radius²` 可以保留，但不能替代 x/y。

6. **GraphMatcher 加 top-k candidate pruning**

```text
raw descriptor similarity
→ top-32 或 top-64 candidates
→ GraphMatcher
```

7. **Feature Decode 取消 min intensity mask，改成 validity/local contrast mask**

不要因为暗就删点。过滤无效值、过曝、极低局部对比、边界和噪声。

8. **加 weak-texture semi-dense fallback**

```text
coarse correlation / dual-softmax
→ top-K semi-dense candidates
→ GraphMatcher refine
```

## 15. 最终设计重点

PFM v2.1 不应该继续朝“更大 SuperGlue”方向走。优势应该是：

```text
行星影像专用光照鲁棒 texture prior
+ 几何归一化 descriptor
+ C4 / rotation consistency
+ 弱纹理 fallback
+ graph-level no-match filtering
```

三条核心原则：

1. 光照问题靠 illumination-robust texture branch + photometric consistency loss，不是只靠更深 backbone。
2. 跨视角和畸变问题靠 GeometryHead 参与 descriptor canonicalization，不是只把 geometry 当 metadata。
3. 弱纹理问题靠 sparse keypoint + semi-dense fallback，不是只提高 heatmap 阈值或增加关键点数量。
