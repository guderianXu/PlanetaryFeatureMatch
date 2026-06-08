#!/usr/bin/env python3
"""PyTorch 版 PFM 主模型。

这个文件保留模型结构、checkpoint 读写和 Python/C++ 对齐所需的公共类。
不要继续把训练 loss、数据读取、报告生成或一次性实验逻辑塞进这里；这些逻辑应该拆到
`pfm_pytorch_training.py` 的子模块或 `scripts/` 下面。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from pfm_model_descriptors import geometry_aware_descriptor_pool, make_xy_grid, normalize_channels_stable

INFERENCE_TEXTURE_BLEND_WEIGHT = 1.0


@dataclass(frozen=True)
class CheckpointConfig:
    input_channels: int
    base_channels: int
    descriptor_dim: int
    graph_hidden_dim: int
    graph_attention_layers: int
    graph_keypoint_meta_dim: int = 2


@dataclass(frozen=True)
class SparseHeadOutput:
    heatmap: torch.Tensor
    descriptors: torch.Tensor
    scale: torch.Tensor
    orientation: torch.Tensor
    affine: torch.Tensor
    keypoint_offsets: torch.Tensor
    matchability: torch.Tensor
    descriptor_uncertainty: torch.Tensor
    no_match_prior: torch.Tensor


@dataclass(frozen=True)
class DenseHeadOutput:
    confidence: torch.Tensor
    offsets: torch.Tensor


@dataclass(frozen=True)
class GraphMatcherOutput:
    logits: torch.Tensor
    matches: torch.Tensor
    scores: torch.Tensor
    accept_logits: torch.Tensor | None = None
    executed_layers: int = 0
    input_keypoints_a: int = 0
    input_keypoints_b: int = 0
    kept_keypoints_a: int = 0
    kept_keypoints_b: int = 0
    pruned_keypoints_a: int = 0
    pruned_keypoints_b: int = 0
    attention_work_units: int = 0
    full_attention_work_units: int = 0
    attention_work_fraction: float = 0.0


@dataclass(frozen=True)
class SemiDenseCandidateOutput:
    keypoints_a: torch.Tensor
    keypoints_b: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class RawFeatureMaps:
    heatmap: torch.Tensor
    descriptors: torch.Tensor
    scale: torch.Tensor
    orientation: torch.Tensor
    affine: torch.Tensor
    dense_confidence: torch.Tensor
    keypoint_offsets: torch.Tensor
    quality: torch.Tensor
    local_contrast: torch.Tensor
    matchability: torch.Tensor | None = None
    descriptor_uncertainty: torch.Tensor | None = None
    no_match_prior: torch.Tensor | None = None


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


def _make_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels), channels)


def _make_stage(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


def _checkpoint_tensor_call(enabled: bool, function, *inputs: torch.Tensor):
    if not enabled or not torch.is_grad_enabled():
        return function(*inputs)
    return activation_checkpoint(function, *inputs, use_reentrant=False)


def _checkpoint_module(enabled: bool, module: nn.Module, *inputs: torch.Tensor):
    return _checkpoint_tensor_call(enabled, lambda *args: module(*args), *inputs)


class ZeroResidualContextBlock(nn.Module):
    """Residual local/dilated context block initialized as an exact no-op."""

    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.norm1 = _make_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.norm2 = _make_norm(channels)
        with torch.no_grad():
            self.conv2.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.conv1(x)))
        return x + self.norm2(self.conv2(hidden))


def _make_stage_refinement(channels: int) -> nn.Sequential:
    return nn.Sequential(
        ZeroResidualContextBlock(channels),
        ZeroResidualContextBlock(channels),
        ZeroResidualContextBlock(channels, dilation=2),
    )


class Backbone(nn.Module):
    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels <= 0:
            raise ValueError("input_channels and base_channels must be positive")
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.stage1 = _make_stage(input_channels, base_channels)
        self.stage2 = _make_stage(base_channels, base_channels * 2)
        self.stage3 = _make_stage(base_channels * 2, base_channels * 4)
        self.stage4 = _make_stage(base_channels * 4, base_channels * 8)
        self.stage1_refine = _make_stage_refinement(base_channels)
        self.stage2_refine = _make_stage_refinement(base_channels * 2)
        self.stage3_refine = _make_stage_refinement(base_channels * 4)
        self.stage4_refine = _make_stage_refinement(base_channels * 8)

    def forward(self, x: torch.Tensor, *, activation_checkpointing: bool = False) -> list[torch.Tensor]:
        if x.dim() != 4 or x.size(1) != self.input_channels:
            raise ValueError("input tensor must have shape BxCxHxW with the configured channel count")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y1 = _checkpoint_module(activation_checkpointing, self.stage1_refine, self.stage1(x))
        y2 = _checkpoint_module(activation_checkpointing, self.stage2_refine, self.stage2(y1))
        y3 = _checkpoint_module(activation_checkpointing, self.stage3_refine, self.stage3(y2))
        y4 = _checkpoint_module(activation_checkpointing, self.stage4_refine, self.stage4(y3))
        return [y1, y2, y3, y4]

    def sanitize_nonfinite_state(self) -> None:
        with torch.no_grad():
            for name, tensor in list(self.named_buffers(recurse=True)):
                if not tensor.is_floating_point():
                    continue
                finite = torch.isfinite(tensor)
                if bool(finite.all()):
                    continue
                fill = 1.0 if "running_var" in name else 0.0
                tensor.masked_fill_(~finite, fill)


class DualFPNLite(nn.Module):
    """Build separate 1/4-resolution features for keypoints and descriptors."""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        p2_channels = base_channels * 2
        self.keypoint_from_stage3 = nn.Conv2d(base_channels * 4, p2_channels, 1)
        self.descriptor_from_stage3 = nn.Conv2d(base_channels * 4, p2_channels, 1)
        self.descriptor_from_stage4 = nn.Conv2d(base_channels * 8, p2_channels, 1)
        self.keypoint_refine = ZeroResidualContextBlock(p2_channels)
        self.descriptor_refine = nn.Sequential(
            ZeroResidualContextBlock(p2_channels),
            ZeroResidualContextBlock(p2_channels, dilation=2),
        )
        _zero_module(self.keypoint_from_stage3)
        _zero_module(self.descriptor_from_stage3)
        _zero_module(self.descriptor_from_stage4)

    def _forward_tensors(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del stage1
        stage3_keypoint = F.interpolate(
            self.keypoint_from_stage3(stage3),
            size=stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p2_keypoint = self.keypoint_refine(stage2 + stage3_keypoint)
        desc_stage3 = F.interpolate(
            self.descriptor_from_stage3(stage3),
            size=stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        desc_stage4 = F.interpolate(
            self.descriptor_from_stage4(stage4),
            size=stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p2_descriptor = self.descriptor_refine(stage2 + desc_stage3 + desc_stage4)
        return p2_keypoint, p2_descriptor

    def forward(
        self,
        features: list[torch.Tensor],
        *,
        activation_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(features) < 4:
            raise ValueError("DualFPNLite requires backbone stages 1..4")
        return _checkpoint_tensor_call(
            activation_checkpointing,
            self._forward_tensors,
            features[0],
            features[1],
            features[2],
            features[3],
        )


def _normalize_channels(tensor: torch.Tensor) -> torch.Tensor:
    return normalize_channels_stable(tensor)


class DescriptorResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(x))
        return torch.relu(x + self.conv2(hidden))


def _make_descriptor_tower(input_channels: int, descriptor_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, input_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        DescriptorResidualBlock(input_channels),
        DescriptorResidualBlock(input_channels),
        DescriptorResidualBlock(input_channels),
        DescriptorResidualBlock(input_channels),
        nn.Conv2d(input_channels, descriptor_dim, 1),
    )


def make_multiscale_descriptor_context(context: torch.Tensor) -> torch.Tensor:
    local = F.avg_pool2d(context, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    wider = F.avg_pool2d(context, kernel_size=5, stride=1, padding=2, count_include_pad=False)
    return torch.cat([context, local, wider], dim=1)


def _conv1x1_channel_slice(
    projection: nn.Conv2d,
    x: torch.Tensor,
    channel_offset: int,
    include_bias: bool,
) -> torch.Tensor:
    channels = x.size(1)
    weight = projection.weight[:, channel_offset : channel_offset + channels, :, :]
    bias = projection.bias if include_bias else None
    return F.conv2d(x, weight, bias=bias)


def _apply_anisotropic_viewpoint_projection(projection: nn.Conv2d, context: torch.Tensor) -> torch.Tensor:
    channels = context.size(1)
    result = _conv1x1_channel_slice(projection, context, 0, True)
    horizontal = F.avg_pool2d(context, kernel_size=(1, 7), stride=1, padding=(0, 3), count_include_pad=False)
    result = result + _conv1x1_channel_slice(projection, horizontal, channels, False)
    vertical = F.avg_pool2d(context, kernel_size=(7, 1), stride=1, padding=(3, 0), count_include_pad=False)
    result = result + _conv1x1_channel_slice(projection, vertical, channels * 2, False)
    wide_horizontal = F.avg_pool2d(context, kernel_size=(3, 9), stride=1, padding=(1, 4), count_include_pad=False)
    result = result + _conv1x1_channel_slice(projection, wide_horizontal, channels * 3, False)
    wide_vertical = F.avg_pool2d(context, kernel_size=(9, 3), stride=1, padding=(4, 1), count_include_pad=False)
    return result + _conv1x1_channel_slice(projection, wide_vertical, channels * 4, False)


def _zero_module(module: nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


class SparseHead(nn.Module):
    """稀疏特征头：同时预测关键点、descriptor 和局部几何。

    当前版本已经删除旧的 C4 旋转分支。descriptor 的旋转鲁棒性不再靠 0/90/180/270
    离散旋转枚举，而是由 `orientation/scale/affine` 预测出的连续局部几何驱动
    `geometry_aware_descriptor_pool()` 完成。
    """

    def __init__(self, input_channels: int, descriptor_dim: int) -> None:
        super().__init__()
        if input_channels <= 0 or descriptor_dim <= 0:
            raise ValueError("input_channels and descriptor_dim must be positive")
        self.input_channels = input_channels
        self.descriptor_dim = descriptor_dim
        self.context = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.keypoint_context = ZeroResidualContextBlock(input_channels)
        self.descriptor_context = ZeroResidualContextBlock(input_channels)
        self.geometry_context = ZeroResidualContextBlock(input_channels)
        self.heatmap = nn.Conv2d(input_channels, 1, 1)
        self.heatmap_viewpoint_context = nn.Conv2d(input_channels * 5, 1, 1)
        self.keypoint_offsets = nn.Conv2d(input_channels, 2, 1)
        self.descriptors = _make_descriptor_tower(input_channels, descriptor_dim)
        self.descriptor_multiscale = nn.Conv2d(input_channels * 3, descriptor_dim, 1)
        self.descriptor_attention = nn.Conv2d(input_channels * 3, descriptor_dim, 1)
        self.descriptor_viewpoint_context = nn.Conv2d(input_channels * 5, descriptor_dim, 1)
        self.descriptor_viewpoint_attention = nn.Conv2d(input_channels * 5, descriptor_dim, 1)
        self.descriptor_dilated_context = nn.Conv2d(descriptor_dim, descriptor_dim, 3, padding=2, dilation=2)
        _zero_module(self.heatmap_viewpoint_context)
        _zero_module(self.keypoint_offsets)
        _zero_module(self.descriptor_viewpoint_context)
        _zero_module(self.descriptor_viewpoint_attention)
        _zero_module(self.descriptor_dilated_context)
        self.descriptor_skip = nn.Conv2d(input_channels, descriptor_dim, 1)
        self.scale = nn.Conv2d(input_channels, 1, 1)
        self.orientation = nn.Conv2d(input_channels, 2, 1)
        self.affine = nn.Conv2d(input_channels, 4, 1)
        self.matchability = nn.Conv2d(input_channels, 1, 1)
        self.descriptor_uncertainty = nn.Conv2d(input_channels, 1, 1)
        self.no_match_prior = nn.Conv2d(input_channels, 1, 1)
        _zero_module(self.matchability)
        _zero_module(self.descriptor_uncertainty)
        _zero_module(self.no_match_prior)

    def _descriptor_branch(
        self,
        keypoint_feature: torch.Tensor,
        descriptor_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if descriptor_feature is None:
            descriptor_feature = keypoint_feature
        keypoint_shared_context = self.context(keypoint_feature)
        descriptor_shared_context = self.context(descriptor_feature)
        keypoint_context = self.keypoint_context(keypoint_shared_context)
        descriptor_context = self.descriptor_context(descriptor_shared_context)
        multiscale_context = make_multiscale_descriptor_context(descriptor_context)
        viewpoint_descriptor = _apply_anisotropic_viewpoint_projection(self.descriptor_viewpoint_context, descriptor_context)
        viewpoint_gate = 1.0 + torch.sigmoid(
            _apply_anisotropic_viewpoint_projection(self.descriptor_viewpoint_attention, descriptor_context)
        )
        descriptor_base = (
            self.descriptors(descriptor_context)
            + self.descriptor_multiscale(multiscale_context)
            + viewpoint_descriptor * viewpoint_gate
            + self.descriptor_skip(descriptor_feature)
        )
        descriptor_gated = descriptor_base * (1.0 + torch.sigmoid(self.descriptor_attention(multiscale_context)))
        heatmap = self.heatmap(keypoint_context) + _apply_anisotropic_viewpoint_projection(
            self.heatmap_viewpoint_context,
            keypoint_context,
        )
        keypoint_offsets = torch.tanh(self.keypoint_offsets(keypoint_context)) * 0.5
        geometry_context = self.geometry_context(descriptor_shared_context)
        return geometry_context, heatmap, descriptor_gated, keypoint_offsets

    def forward(
        self,
        feature: torch.Tensor,
        descriptor_feature: torch.Tensor | None = None,
        *,
        activation_checkpointing: bool = False,
    ) -> SparseHeadOutput:
        if feature.dim() != 4 or feature.size(1) != self.input_channels:
            raise ValueError("feature tensor must have shape BxCxHxW with the configured channel count")
        if descriptor_feature is not None and (
            descriptor_feature.dim() != 4
            or descriptor_feature.size(1) != self.input_channels
            or descriptor_feature.shape[-2:] != feature.shape[-2:]
        ):
            raise ValueError("descriptor_feature must have shape BxCxHxW matching the keypoint feature grid")
        descriptor_input = feature if descriptor_feature is None else descriptor_feature
        geometry_context, heatmap_sum, descriptor_gated, keypoint_offsets = _checkpoint_tensor_call(
            activation_checkpointing,
            self._descriptor_branch,
            feature,
            descriptor_input,
        )
        descriptor_sum = descriptor_gated + _checkpoint_module(
            activation_checkpointing,
            self.descriptor_dilated_context,
            descriptor_gated,
        )
        heatmap = torch.sigmoid(heatmap_sum)
        descriptors = _normalize_channels(descriptor_sum)

        # 这三个几何头不是给外部直接显示用的，而是给 descriptor canonical pooling 用：
        # scale 控制采样半径，orientation 给出局部主方向，affine 捕捉斜视/局部剪切。
        scale = torch.exp(torch.clamp(self.scale(geometry_context), min=-2.0, max=2.0))
        orientation = _normalize_channels(self.orientation(geometry_context))
        affine_delta = torch.tanh(self.affine(geometry_context)) * 0.1
        identity = affine_delta.new_tensor([1.0, 0.0, 0.0, 1.0]).view(1, 4, 1, 1)
        affine = identity + affine_delta

        # 用连续几何做 canonical pooling，替代旧 C4 分支，输出仍保持 dense descriptor map。
        matchability = torch.sigmoid(self.matchability(geometry_context))
        descriptor_uncertainty = torch.sigmoid(self.descriptor_uncertainty(geometry_context))
        no_match_prior = torch.sigmoid(self.no_match_prior(geometry_context))
        descriptors = _checkpoint_tensor_call(
            activation_checkpointing,
            geometry_aware_descriptor_pool,
            descriptors,
            orientation,
            scale,
            affine,
        )
        return SparseHeadOutput(
            heatmap,
            descriptors,
            scale,
            orientation,
            affine,
            keypoint_offsets,
            matchability,
            descriptor_uncertainty,
            no_match_prior,
        )


def _shifted_feature(feature: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    shifted = torch.zeros_like(feature)
    height = feature.size(2)
    width = feature.size(3)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    if source_y0 >= source_y1 or source_x0 >= source_x1:
        return shifted
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    shifted[:, :, target_y0:target_y1, target_x0:target_x1] = feature[:, :, source_y0:source_y1, source_x0:source_x1]
    return shifted


def local_correlation(feature_a: torch.Tensor, feature_b: torch.Tensor, radius: int = 4) -> torch.Tensor:
    correlations = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            correlations.append((feature_a * _shifted_feature(feature_b, dy, dx)).mean(dim=1, keepdim=True))
    return torch.cat(correlations, dim=1)


class DenseHead(nn.Module):
    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        self.feature_channels = feature_channels
        self.correlation_projection = nn.Conv2d(81, feature_channels, 1)
        input_channels = feature_channels * 4 + 2
        self.predictor = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feature_channels * 2, feature_channels * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feature_channels * 2, feature_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feature_channels, 3, 1),
        )

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> DenseHeadOutput:
        if feature_a.dim() != 4 or feature_b.dim() != 4 or feature_a.shape != feature_b.shape:
            raise ValueError("feature tensors must have matching BxCxHxW shapes")
        if feature_a.size(1) != self.feature_channels:
            raise ValueError("feature tensor channel count does not match dense head")
        height = feature_a.size(2)
        width = feature_a.size(3)
        coordinates = make_xy_grid(height, width, device=feature_a.device, dtype=feature_a.dtype)
        coordinates[..., 0] = coordinates[..., 0] / max(1, width - 1) * 2.0 - 1.0
        coordinates[..., 1] = coordinates[..., 1] / max(1, height - 1) * 2.0 - 1.0
        coordinate_channels = coordinates.permute(2, 0, 1).unsqueeze(0).expand(feature_a.size(0), 2, height, width)
        correlation = self.correlation_projection(local_correlation(feature_a, feature_b))
        pair_feature = torch.cat([feature_a, feature_b, torch.abs(feature_a - feature_b), correlation, coordinate_channels], dim=1)
        prediction = self.predictor(pair_feature)
        confidence = torch.sigmoid(prediction[:, 0:1])
        offsets = prediction[:, 1:3]
        return DenseHeadOutput(confidence, offsets)


class DescriptorMatcher(nn.Module):
    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.descriptor_dim = descriptor_dim

    def forward(self, desc_a: torch.Tensor, desc_b: torch.Tensor) -> torch.Tensor:
        if desc_a.dim() != 3 or desc_b.dim() != 3:
            raise ValueError("descriptor tensors must have shapes BxNxD and BxMxD")
        if desc_a.size(0) != desc_b.size(0) or desc_a.size(2) != self.descriptor_dim or desc_b.size(2) != self.descriptor_dim:
            raise ValueError("descriptor tensor dimensions do not match matcher")
        normalized_a = F.normalize(desc_a, p=2, dim=2, eps=1.0e-12)
        normalized_b = F.normalize(desc_b, p=2, dim=2, eps=1.0e-12)
        return torch.bmm(normalized_a, normalized_b.transpose(1, 2)) / math.sqrt(float(self.descriptor_dim))


def prepare_keypoints_for_embedding(keypoints: torch.Tensor, *, meta_dim: int = 2) -> torch.Tensor:
    if meta_dim <= 0:
        raise ValueError("meta_dim must be positive")
    prepared = keypoints.to(dtype=torch.float32)
    if prepared.size(0) == 0:
        return prepared.new_empty((0, meta_dim))
    min_xy = prepared.min(dim=0, keepdim=True).values
    max_xy = prepared.max(dim=0, keepdim=True).values
    center = (min_xy + max_xy) * 0.5
    span = (max_xy - min_xy).max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
    centered = (prepared - center) * 2.0 / span
    radius = centered.pow(2).sum(dim=1, keepdim=True).sqrt()
    if meta_dim == 1:
        return radius
    legacy = torch.cat([radius, radius.pow(2)], dim=1)
    if meta_dim == 2:
        return legacy
    spatial = torch.cat([centered, legacy], dim=1)
    if meta_dim <= spatial.size(1):
        return spatial[:, :meta_dim]
    return torch.cat([spatial, spatial.new_zeros((spatial.size(0), meta_dim - spatial.size(1)))], dim=1)


def prepare_graph_keypoint_metadata(
    keypoints: torch.Tensor,
    *,
    meta_dim: int,
    scores: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    orientation: torch.Tensor | None = None,
    affine: torch.Tensor | None = None,
    quality: torch.Tensor | None = None,
    local_contrast: torch.Tensor | None = None,
    matchability: torch.Tensor | None = None,
    descriptor_uncertainty: torch.Tensor | None = None,
    no_match_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build v2.1 GraphMatcher metadata from selected sparse keypoints."""

    if meta_dim <= 0:
        raise ValueError("meta_dim must be positive")
    base = prepare_keypoints_for_embedding(keypoints, meta_dim=max(meta_dim, 4))
    count = keypoints.size(0)
    device = keypoints.device
    dtype = torch.float32

    def vector(value: torch.Tensor | None, default: float, width: int = 1) -> torch.Tensor:
        if value is None:
            return torch.full((count, width), float(default), dtype=dtype, device=device)
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 1:
            value = value.unsqueeze(1)
        if value.size(0) != count:
            raise ValueError("metadata vectors must have one row per keypoint")
        if value.size(1) < width:
            value = torch.cat([value, value.new_full((count, width - value.size(1)), float(default))], dim=1)
        return value[:, :width]

    score_column = vector(scores, 1.0)
    scale_column = vector(scale, 1.0).clamp_min(1.0e-4).log()
    orientation_columns = F.normalize(vector(orientation, 0.0, width=2), p=2, dim=1, eps=1.0e-6)
    if orientation is None:
        orientation_columns[:, 0] = 1.0
        orientation_columns[:, 1] = 0.0
    affine_columns = vector(affine, 0.0, width=4)
    if affine is None:
        affine_columns[:, 0] = 1.0
        affine_columns[:, 3] = 1.0
    quality_column = vector(quality, 1.0)
    matchability_column = vector(matchability, 1.0) if matchability is not None else quality_column
    contrast_column = vector(local_contrast, 0.0)
    uncertainty_column = (
        vector(descriptor_uncertainty, 0.0)
        if descriptor_uncertainty is not None
        else (1.0 - quality_column).clamp(0.0, 1.0)
    )
    no_match_column = vector(no_match_prior, 0.0)
    metadata = torch.cat(
        [
            base[:, :4],
            score_column,
            scale_column,
            orientation_columns,
            affine_columns,
            matchability_column,
            contrast_column,
            uncertainty_column,
            no_match_column,
        ],
        dim=1,
    )
    if metadata.size(1) >= meta_dim:
        return metadata[:, :meta_dim].contiguous()
    return torch.cat([metadata, metadata.new_zeros((count, meta_dim - metadata.size(1)))], dim=1).contiguous()


def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    logits = query @ key.transpose(0, 1) / math.sqrt(float(hidden_dim))
    return torch.softmax(logits, dim=1) @ value


class PlanetaryGraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.self_query = nn.Linear(hidden_dim, hidden_dim)
        self.self_key = nn.Linear(hidden_dim, hidden_dim)
        self.self_value = nn.Linear(hidden_dim, hidden_dim)
        self.self_output = nn.Linear(hidden_dim, hidden_dim)
        self.cross_query = nn.Linear(hidden_dim, hidden_dim)
        self.cross_key = nn.Linear(hidden_dim, hidden_dim)
        self.cross_value = nn.Linear(hidden_dim, hidden_dim)
        self.cross_output = nn.Linear(hidden_dim, hidden_dim)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(0.1)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self_a = _attend(self.self_query(features_a), self.self_key(features_a), self.self_value(features_a), self.hidden_dim)
        self_b = _attend(self.self_query(features_b), self.self_key(features_b), self.self_value(features_b), self.hidden_dim)
        refined_a = self.self_norm(features_a + self.attention_dropout(self.self_output(self_a)))
        refined_b = self.self_norm(features_b + self.attention_dropout(self.self_output(self_b)))
        cross_a = _attend(self.cross_query(refined_a), self.cross_key(refined_b), self.cross_value(refined_b), self.hidden_dim)
        cross_b = _attend(self.cross_query(refined_b), self.cross_key(refined_a), self.cross_value(refined_a), self.hidden_dim)
        refined_a = self.cross_norm(refined_a + self.attention_dropout(self.cross_output(cross_a)))
        refined_b = self.cross_norm(refined_b + self.attention_dropout(self.cross_output(cross_b)))
        return (
            self.feed_forward_norm(refined_a + self.feed_forward(refined_a)),
            self.feed_forward_norm(refined_b + self.feed_forward(refined_b)),
        )


class PlanetaryGraphMatcher(nn.Module):
    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        attention_layers: int = 1,
        keypoint_meta_dim: int = 2,
        candidate_topk: int = 64,
    ) -> None:
        super().__init__()
        if descriptor_dim <= 0 or hidden_dim <= 0 or attention_layers <= 0 or keypoint_meta_dim <= 0:
            raise ValueError("descriptor_dim, hidden_dim, attention_layers, and keypoint_meta_dim must be positive")
        self.descriptor_dim = descriptor_dim
        self.hidden_dim = hidden_dim
        self.keypoint_meta_dim = keypoint_meta_dim
        self.candidate_topk = int(candidate_topk)
        self.descriptor_projection = nn.Linear(descriptor_dim, hidden_dim)
        self.keypoint_projection = nn.Linear(keypoint_meta_dim, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_bias = nn.Sequential(
            nn.Linear(8, max(16, hidden_dim // 8)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 8), 1),
        )
        self.accept_head = nn.Sequential(
            nn.Linear(6, max(16, hidden_dim // 8)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 8), 1),
        )
        self.logit_scale = nn.Parameter(torch.ones(1) * math.sqrt(float(hidden_dim)))
        self.raw_score_temperature = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.graph_delta_scale = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.accept_logit_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.dustbin_bias = nn.Parameter(torch.zeros(1))
        self.last_executed_attention_layers = 0
        self.attention_layers = nn.ModuleList([PlanetaryGraphAttentionLayer(hidden_dim) for _ in range(attention_layers)])
        _zero_module(self.geometry_bias[-1])
        _zero_module(self.accept_head[-1])

    def _metadata(self, keypoints_or_meta: torch.Tensor) -> torch.Tensor:
        if keypoints_or_meta.dim() != 2:
            raise ValueError("graph matcher keypoints/meta must have shape NxC")
        if keypoints_or_meta.size(1) == self.keypoint_meta_dim:
            return keypoints_or_meta.to(dtype=torch.float32)
        if keypoints_or_meta.size(1) < 2:
            raise ValueError("graph matcher keypoints must contain at least x/y")
        return prepare_keypoints_for_embedding(keypoints_or_meta[:, :2], meta_dim=self.keypoint_meta_dim)

    @staticmethod
    def _metadata_column(meta: torch.Tensor, index: int, default: float = 0.0) -> torch.Tensor:
        if meta.size(1) <= index:
            return meta.new_full((meta.size(0),), default)
        return meta[:, index]

    @classmethod
    def _metadata_reliability_score(cls, meta: torch.Tensor) -> torch.Tensor:
        matchability = cls._metadata_column(meta, 12, 0.5).clamp(0.0, 1.0)
        uncertainty = cls._metadata_column(meta, 14, 0.5).clamp(0.0, 1.0)
        no_match_prior = cls._metadata_column(meta, 15, 0.5).clamp(0.0, 1.0)
        return (matchability - 0.5) - (uncertainty - 0.5) - (no_match_prior - 0.5)

    @classmethod
    def _pair_reliability_bias(cls, meta_a: torch.Tensor, meta_b: torch.Tensor) -> torch.Tensor:
        score_a = cls._metadata_reliability_score(meta_a)
        score_b = cls._metadata_reliability_score(meta_b)
        return 0.5 * (score_a[:, None] + score_b[None, :])

    def _geometry_compatibility_bias(self, meta_a: torch.Tensor, meta_b: torch.Tensor) -> torch.Tensor:
        def column(meta: torch.Tensor, index: int, default: float = 0.0) -> torch.Tensor:
            if meta.size(1) <= index:
                return meta.new_full((meta.size(0),), default)
            return meta[:, index]

        ax = column(meta_a, 0)[:, None]
        ay = column(meta_a, 1)[:, None]
        bx = column(meta_b, 0)[None, :]
        by = column(meta_b, 1)[None, :]
        score_delta = column(meta_a, 4)[:, None] - column(meta_b, 4)[None, :]
        scale_delta = column(meta_a, 5)[:, None] - column(meta_b, 5)[None, :]
        aox = column(meta_a, 6, 1.0)[:, None]
        aoy = column(meta_a, 7)[:, None]
        box = column(meta_b, 6, 1.0)[None, :]
        boy = column(meta_b, 7)[None, :]
        orientation_cos = (aox * box + aoy * boy).clamp(-1.0, 1.0)
        quality_pair = 0.5 * (column(meta_a, 12, 1.0)[:, None] + column(meta_b, 12, 1.0)[None, :])
        contrast_pair = 0.5 * (column(meta_a, 13)[:, None] + column(meta_b, 13)[None, :])
        dx = ax - bx
        dy = ay - by
        features = torch.stack(
            [
                dx,
                dy,
                torch.sqrt(dx.square() + dy.square()).clamp_max(4.0),
                score_delta,
                scale_delta,
                orientation_cos,
                quality_pair,
                contrast_pair,
            ],
            dim=-1,
        )
        return self.geometry_bias(features).squeeze(-1)

    def _candidate_mask(self, desc_a: torch.Tensor, desc_b: torch.Tensor) -> torch.Tensor:
        count_a = desc_a.size(0)
        count_b = desc_b.size(0)
        if self.candidate_topk <= 0 or self.candidate_topk >= count_b:
            return torch.ones(count_a, count_b, dtype=torch.bool, device=desc_a.device)
        similarity = F.normalize(desc_a, p=2, dim=1, eps=1.0e-12) @ F.normalize(desc_b, p=2, dim=1, eps=1.0e-12).T
        mask = torch.zeros(count_a, count_b, dtype=torch.bool, device=desc_a.device)
        row_k = min(self.candidate_topk, count_b)
        row_indices = similarity.topk(row_k, dim=1).indices
        mask.scatter_(1, row_indices, True)
        col_k = min(self.candidate_topk, count_a)
        col_indices = similarity.topk(col_k, dim=0).indices
        mask.scatter_(0, col_indices, True)
        return mask

    def _acceptance_logits(
        self,
        raw_similarity: torch.Tensor,
        graph_delta: torch.Tensor,
        meta_a: torch.Tensor,
        meta_b: torch.Tensor,
    ) -> torch.Tensor:
        if raw_similarity.numel() == 0:
            return raw_similarity.new_empty(raw_similarity.shape)

        def column(meta: torch.Tensor, index: int, default: float = 0.0) -> torch.Tensor:
            if meta.size(1) <= index:
                return meta.new_full((meta.size(0),), default)
            return meta[:, index]

        if raw_similarity.size(1) > 1:
            row_top2 = raw_similarity.topk(2, dim=1).values
            row_margin = (row_top2[:, 0] - row_top2[:, 1]).clamp(0.0, 2.0)
        else:
            row_margin = raw_similarity.new_zeros((raw_similarity.size(0),))
        if raw_similarity.size(0) > 1:
            col_top2 = raw_similarity.topk(2, dim=0).values
            col_margin = (col_top2[0] - col_top2[1]).clamp(0.0, 2.0)
        else:
            col_margin = raw_similarity.new_zeros((raw_similarity.size(1),))
        quality_pair = 0.5 * (column(meta_a, 12, 1.0)[:, None] + column(meta_b, 12, 1.0)[None, :])
        contrast_pair = 0.5 * (column(meta_a, 13)[:, None] + column(meta_b, 13)[None, :])
        features = torch.stack(
            [
                raw_similarity.clamp(-1.0, 1.0),
                row_margin[:, None].expand_as(raw_similarity),
                col_margin[None, :].expand_as(raw_similarity),
                graph_delta.detach().clamp(-20.0, 20.0) / 20.0,
                quality_pair.clamp(0.0, 1.0),
                contrast_pair.clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        return self.accept_head(features).squeeze(-1) + self._pair_reliability_bias(meta_a, meta_b)

    def _provisional_pair_logits(
        self,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
        raw_similarity: torch.Tensor,
        meta_a: torch.Tensor,
        meta_b: torch.Tensor,
    ) -> torch.Tensor:
        pair_logits, _ = self._provisional_pair_outputs(embed_a, embed_b, raw_similarity, meta_a, meta_b)
        return pair_logits

    def _provisional_pair_outputs(
        self,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
        raw_similarity: torch.Tensor,
        meta_a: torch.Tensor,
        meta_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_a = F.normalize(self.score_projection(embed_a), p=2, dim=1)
        projected_b = F.normalize(self.score_projection(embed_b), p=2, dim=1)
        graph_delta = (projected_a @ projected_b.transpose(0, 1)) * self.logit_scale.clamp(1.0, 100.0)
        graph_delta = graph_delta + self._geometry_compatibility_bias(meta_a, meta_b)
        accept_logits = self._acceptance_logits(raw_similarity, graph_delta, meta_a, meta_b)
        raw_temperature = self.raw_score_temperature.abs().clamp(0.03, 1.0)
        delta_scale = self.graph_delta_scale.clamp(0.0, 2.0)
        accept_scale = self.accept_logit_scale.clamp(0.0, 2.0)
        pair_logits = raw_similarity / raw_temperature + delta_scale * graph_delta + accept_scale * accept_logits
        return pair_logits, accept_logits

    @staticmethod
    def _assignment_confidence(pair_logits: torch.Tensor) -> torch.Tensor:
        if pair_logits.numel() == 0:
            return pair_logits.new_tensor(0.0)
        row_confidence = torch.softmax(pair_logits, dim=1).max(dim=1).values
        column_confidence = torch.softmax(pair_logits, dim=0).max(dim=0).values
        return torch.minimum(row_confidence.mean(), column_confidence.mean())

    @staticmethod
    def _acceptance_keep_masks(accept_logits: torch.Tensor, min_probability: float) -> tuple[torch.Tensor, torch.Tensor]:
        if accept_logits.numel() == 0:
            keep_a = torch.zeros(accept_logits.size(0), dtype=torch.bool, device=accept_logits.device)
            keep_b = torch.zeros(accept_logits.size(1), dtype=torch.bool, device=accept_logits.device)
            return keep_a, keep_b
        accept_probability = torch.sigmoid(accept_logits)
        keep_a = accept_probability.max(dim=1).values >= float(min_probability)
        keep_b = accept_probability.max(dim=0).values >= float(min_probability)
        return keep_a, keep_b

    @staticmethod
    def _acceptance_top_count_keep_masks(
        accept_logits: torch.Tensor,
        keep_count_a: int,
        keep_count_b: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if accept_logits.numel() == 0:
            keep_a = torch.zeros(accept_logits.size(0), dtype=torch.bool, device=accept_logits.device)
            keep_b = torch.zeros(accept_logits.size(1), dtype=torch.bool, device=accept_logits.device)
            return keep_a, keep_b
        accept_probability = torch.sigmoid(accept_logits)
        score_a = accept_probability.max(dim=1).values
        score_b = accept_probability.max(dim=0).values

        def top_mask(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
            keep_count = min(int(scores.numel()), max(1, int(keep_count)))
            if keep_count >= int(scores.numel()):
                return torch.ones_like(scores, dtype=torch.bool)
            top_indices = torch.topk(scores, keep_count, largest=True, sorted=False).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[top_indices] = True
            return mask

        return top_mask(score_a, keep_count_a), top_mask(score_b, keep_count_b)

    def forward(
        self,
        descriptors_a: torch.Tensor,
        keypoints_a: torch.Tensor,
        descriptors_b: torch.Tensor,
        keypoints_b: torch.Tensor,
        apply_candidate_mask: bool = True,
        width_prune_min_score: float = -1.0,
        early_stop_min_confidence: float = -1.0,
        max_attention_layers: int = 0,
        max_attention_work_fraction: float = 1.0,
        width_prune_keep_ratio: float = 1.0,
    ) -> GraphMatcherOutput:
        if early_stop_min_confidence < -1.0:
            raise ValueError("early_stop_min_confidence must be at least -1.0; -1 disables early stopping")
        if max_attention_layers < 0:
            raise ValueError("max_attention_layers must be nonnegative; 0 disables hard layer budget")
        if max_attention_work_fraction < 0.0 or max_attention_work_fraction > 1.0:
            raise ValueError("max_attention_work_fraction must be in [0, 1]")
        if (
            not math.isfinite(float(width_prune_keep_ratio))
            or width_prune_keep_ratio < 0.0
            or width_prune_keep_ratio > 1.0
        ):
            raise ValueError("width_prune_keep_ratio must be in [0, 1]")
        if descriptors_a.dim() != 2 or descriptors_b.dim() != 2:
            raise ValueError("graph matcher descriptors must have shape NxD")
        if descriptors_a.size(0) != keypoints_a.size(0) or descriptors_b.size(0) != keypoints_b.size(0):
            raise ValueError("graph matcher descriptor and keypoint counts must match")
        desc_a = descriptors_a.to(dtype=torch.float32)
        desc_b = descriptors_b.to(dtype=torch.float32)
        kp_a = self._metadata(keypoints_a).to(device=desc_a.device)
        kp_b = self._metadata(keypoints_b).to(device=desc_b.device)
        raw_similarity_full = F.normalize(desc_a, p=2, dim=1, eps=1.0e-12) @ F.normalize(desc_b, p=2, dim=1, eps=1.0e-12).T
        prune_enabled = float(width_prune_min_score) > -1.0
        ratio_prune_enabled = float(width_prune_keep_ratio) < 1.0
        if prune_enabled:
            if raw_similarity_full.numel() == 0:
                keep_a = torch.zeros(desc_a.size(0), dtype=torch.bool, device=desc_a.device)
                keep_b = torch.zeros(desc_b.size(0), dtype=torch.bool, device=desc_b.device)
            else:
                keep_a = raw_similarity_full.max(dim=1).values >= float(width_prune_min_score)
                keep_b = raw_similarity_full.max(dim=0).values >= float(width_prune_min_score)
        else:
            keep_a = torch.ones(desc_a.size(0), dtype=torch.bool, device=desc_a.device)
            keep_b = torch.ones(desc_b.size(0), dtype=torch.bool, device=desc_b.device)
        indices_a = keep_a.nonzero(as_tuple=False).flatten()
        indices_b = keep_b.nonzero(as_tuple=False).flatten()
        desc_work_a = desc_a.index_select(0, indices_a)
        desc_work_b = desc_b.index_select(0, indices_b)
        kp_work_a = kp_a.index_select(0, indices_a)
        kp_work_b = kp_b.index_select(0, indices_b)
        input_keypoints_a = int(descriptors_a.size(0))
        input_keypoints_b = int(descriptors_b.size(0))
        full_attention_work_units = input_keypoints_a * input_keypoints_b * len(self.attention_layers)
        max_attention_work_units = int(math.floor(full_attention_work_units * float(max_attention_work_fraction) + 1.0e-9))
        work_budget_enabled = float(max_attention_work_fraction) < 1.0
        keep_count_a = max(1, int(math.ceil(input_keypoints_a * float(width_prune_keep_ratio))))
        keep_count_b = max(1, int(math.ceil(input_keypoints_b * float(width_prune_keep_ratio))))
        restore_pruned_logits = prune_enabled or ratio_prune_enabled
        attention_work_units = 0
        if desc_work_a.size(0) == 0 or desc_work_b.size(0) == 0:
            self.last_executed_attention_layers = 0
            pair_logits = raw_similarity_full.new_full(raw_similarity_full.shape, -1.0e4)
            accept_logits = raw_similarity_full.new_full(raw_similarity_full.shape, -1.0e4)
        else:
            embed_a = torch.relu(self.descriptor_projection(desc_work_a) + self.keypoint_projection(kp_work_a))
            embed_b = torch.relu(self.descriptor_projection(desc_work_b) + self.keypoint_projection(kp_work_b))
            raw_similarity = raw_similarity_full.index_select(0, indices_a).index_select(1, indices_b)
            self.last_executed_attention_layers = 0
            for layer in self.attention_layers:
                if max_attention_layers > 0 and self.last_executed_attention_layers >= int(max_attention_layers):
                    break
                layer_work_units = int(embed_a.size(0)) * int(embed_b.size(0))
                if work_budget_enabled and attention_work_units + layer_work_units > max_attention_work_units:
                    break
                attention_work_units += layer_work_units
                embed_a, embed_b = layer(embed_a, embed_b)
                self.last_executed_attention_layers += 1
                can_run_more_layers = self.last_executed_attention_layers < len(self.attention_layers) and (
                    max_attention_layers <= 0 or self.last_executed_attention_layers < int(max_attention_layers)
                )
                can_adapt = can_run_more_layers
                if can_adapt and (prune_enabled or ratio_prune_enabled or early_stop_min_confidence > -1.0):
                    provisional_pair_logits, provisional_accept_logits = self._provisional_pair_outputs(
                        embed_a,
                        embed_b,
                        raw_similarity,
                        kp_work_a,
                        kp_work_b,
                    )
                    if prune_enabled or ratio_prune_enabled:
                        keep_work_a = torch.ones(embed_a.size(0), dtype=torch.bool, device=embed_a.device)
                        keep_work_b = torch.ones(embed_b.size(0), dtype=torch.bool, device=embed_b.device)
                        threshold_keep_a = keep_work_a
                        threshold_keep_b = keep_work_b
                        if prune_enabled:
                            threshold_keep_a, threshold_keep_b = self._acceptance_keep_masks(
                                provisional_accept_logits,
                                float(width_prune_min_score),
                            )
                        if ratio_prune_enabled:
                            ratio_keep_a, ratio_keep_b = self._acceptance_top_count_keep_masks(
                                provisional_accept_logits,
                                keep_count_a,
                                keep_count_b,
                            )
                            if prune_enabled and bool(threshold_keep_a.any()) and bool(threshold_keep_b.any()):
                                combined_keep_a = threshold_keep_a & ratio_keep_a
                                combined_keep_b = threshold_keep_b & ratio_keep_b
                                if bool(combined_keep_a.any()) and bool(combined_keep_b.any()):
                                    keep_work_a = combined_keep_a
                                    keep_work_b = combined_keep_b
                                else:
                                    keep_work_a = ratio_keep_a
                                    keep_work_b = ratio_keep_b
                            else:
                                keep_work_a = ratio_keep_a
                                keep_work_b = ratio_keep_b
                        elif prune_enabled:
                            keep_work_a = threshold_keep_a
                            keep_work_b = threshold_keep_b
                        if bool(keep_work_a.any()) and bool(keep_work_b.any()) and (
                            not bool(keep_work_a.all()) or not bool(keep_work_b.all())
                        ):
                            local_indices_a = keep_work_a.nonzero(as_tuple=False).flatten()
                            local_indices_b = keep_work_b.nonzero(as_tuple=False).flatten()
                            indices_a = indices_a.index_select(0, local_indices_a)
                            indices_b = indices_b.index_select(0, local_indices_b)
                            desc_work_a = desc_work_a.index_select(0, local_indices_a)
                            desc_work_b = desc_work_b.index_select(0, local_indices_b)
                            kp_work_a = kp_work_a.index_select(0, local_indices_a)
                            kp_work_b = kp_work_b.index_select(0, local_indices_b)
                            embed_a = embed_a.index_select(0, local_indices_a)
                            embed_b = embed_b.index_select(0, local_indices_b)
                            raw_similarity = raw_similarity.index_select(0, local_indices_a).index_select(
                                1,
                                local_indices_b,
                            )
                    if early_stop_min_confidence > -1.0 and bool(
                        self._assignment_confidence(provisional_pair_logits).ge(float(early_stop_min_confidence))
                    ):
                        break
            embed_a = F.normalize(self.score_projection(embed_a), p=2, dim=1)
            embed_b = F.normalize(self.score_projection(embed_b), p=2, dim=1)
            graph_delta = (embed_a @ embed_b.transpose(0, 1)) * self.logit_scale.clamp(1.0, 100.0)
            graph_delta = graph_delta + self._geometry_compatibility_bias(kp_work_a, kp_work_b)
            accept_logits_work = self._acceptance_logits(raw_similarity, graph_delta, kp_work_a, kp_work_b)
            raw_temperature = self.raw_score_temperature.abs().clamp(0.03, 1.0)
            delta_scale = self.graph_delta_scale.clamp(0.0, 2.0)
            accept_scale = self.accept_logit_scale.clamp(0.0, 2.0)
            pair_logits_work = raw_similarity / raw_temperature + delta_scale * graph_delta + accept_scale * accept_logits_work
            if apply_candidate_mask:
                candidate_mask = self._candidate_mask(desc_work_a, desc_work_b)
                pair_logits_work = pair_logits_work.masked_fill(~candidate_mask, -1.0e4)
                accept_logits_work = accept_logits_work.masked_fill(~candidate_mask, -1.0e4)
            if restore_pruned_logits:
                pair_logits = raw_similarity_full.new_full(raw_similarity_full.shape, -1.0e4)
                accept_logits = raw_similarity_full.new_full(raw_similarity_full.shape, -1.0e4)
                pair_logits[indices_a[:, None], indices_b[None, :]] = pair_logits_work
                accept_logits[indices_a[:, None], indices_b[None, :]] = accept_logits_work
            else:
                pair_logits = pair_logits_work
                accept_logits = accept_logits_work
        logits = torch.zeros(
            descriptors_a.size(0) + 1,
            descriptors_b.size(0) + 1,
            dtype=pair_logits.dtype,
            device=pair_logits.device,
        ) + self.dustbin_bias
        if pair_logits.numel() > 0:
            pair_logits = pair_logits + self._pair_reliability_bias(kp_a, kp_b)
        logits[: descriptors_a.size(0), : descriptors_b.size(0)] = pair_logits
        if descriptors_a.size(0) > 0:
            logits[: descriptors_a.size(0), descriptors_b.size(0)] = (
                logits[: descriptors_a.size(0), descriptors_b.size(0)] - self._metadata_reliability_score(kp_a)
            )
        if descriptors_b.size(0) > 0:
            logits[descriptors_a.size(0), : descriptors_b.size(0)] = (
                logits[descriptors_a.size(0), : descriptors_b.size(0)] - self._metadata_reliability_score(kp_b)
            )
        row_logits = logits[: descriptors_a.size(0), :]
        row_prob = torch.softmax(logits[: descriptors_a.size(0), :], dim=1)[:, : descriptors_b.size(0)]
        col_prob = torch.softmax(logits[:, : descriptors_b.size(0)], dim=0)[: descriptors_a.size(0), :]
        dual_scores = row_prob * col_prob
        best_values, best_indices = dual_scores.max(dim=1)
        source_indices = torch.arange(descriptors_a.size(0), device=best_indices.device)
        inlier_mask = best_values.gt(torch.softmax(row_logits, dim=1)[:, -1])
        if descriptors_a.size(0) > 0 and descriptors_b.size(0) > 0:
            reverse_best = dual_scores.max(dim=0).indices
            mutual_sources = reverse_best.index_select(0, best_indices.clamp(0, descriptors_b.size(0) - 1))
            inlier_mask = inlier_mask & mutual_sources.eq(source_indices)
        source_indices = source_indices[inlier_mask]
        target_indices = best_indices[inlier_mask]
        probabilities = best_values[inlier_mask]
        if probabilities.numel() > 0 and accept_logits.numel() > 0:
            probabilities = probabilities * torch.sigmoid(accept_logits[source_indices, target_indices])
        matches = torch.stack([source_indices, target_indices], dim=1).to(device="cpu", dtype=torch.long).contiguous()
        scores = probabilities.to(device="cpu", dtype=torch.float32).contiguous()
        kept_keypoints_a = int(indices_a.numel())
        kept_keypoints_b = int(indices_b.numel())
        attention_work_fraction = (
            0.0 if full_attention_work_units == 0 else attention_work_units / full_attention_work_units
        )
        return GraphMatcherOutput(
            logits.contiguous(),
            matches,
            scores,
            accept_logits.contiguous(),
            executed_layers=int(self.last_executed_attention_layers),
            input_keypoints_a=input_keypoints_a,
            input_keypoints_b=input_keypoints_b,
            kept_keypoints_a=kept_keypoints_a,
            kept_keypoints_b=kept_keypoints_b,
            pruned_keypoints_a=max(0, input_keypoints_a - kept_keypoints_a),
            pruned_keypoints_b=max(0, input_keypoints_b - kept_keypoints_b),
            attention_work_units=attention_work_units,
            full_attention_work_units=full_attention_work_units,
            attention_work_fraction=float(attention_work_fraction),
        )


def make_rotation_invariant_texture_descriptor(
    image: torch.Tensor,
    descriptor_height: int,
    descriptor_width: int,
    descriptor_dim: int,
) -> torch.Tensor:
    base = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if base.size(1) != 1:
        base = base.mean(dim=1, keepdim=True)
    channels = [base]
    height = base.size(2)
    width = base.size(3)
    y = torch.arange(height, device=base.device, dtype=base.dtype).view(1, 1, height, 1)
    x = torch.arange(width, device=base.device, dtype=base.dtype).view(1, 1, 1, width)
    center_y = (height - 1.0) * 0.5
    center_x = (width - 1.0) * 0.5
    max_radius = max(1.0, math.hypot(center_x, center_y))
    radius = torch.sqrt((x - center_x).pow(2) + (y - center_y).pow(2)) / max_radius
    radius = radius.expand(base.size(0), 1, height, width).contiguous()
    channels.extend([radius, radius.pow(2), base * radius])
    local_mean = F.avg_pool2d(base, kernel_size=15, stride=1, padding=7, count_include_pad=False)
    local_sq_mean = F.avg_pool2d(base.square(), kernel_size=15, stride=1, padding=7, count_include_pad=False)
    local_std = (local_sq_mean - local_mean.square()).clamp_min(0.0).sqrt()
    local_normalized = (base - local_mean) / local_std.add(1.0e-3)
    channels.extend([local_normalized, local_std, (base - local_mean).abs()])
    for kernel in (3, 7, 15, 31):
        blur = F.avg_pool2d(base, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        channels.extend([blur, (base - blur).abs()])
    dog_small = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1, count_include_pad=False) - F.avg_pool2d(
        base,
        kernel_size=7,
        stride=1,
        padding=3,
        count_include_pad=False,
    )
    dog_large = F.avg_pool2d(base, kernel_size=7, stride=1, padding=3, count_include_pad=False) - F.avg_pool2d(
        base,
        kernel_size=21,
        stride=1,
        padding=10,
        count_include_pad=False,
    )
    laplacian = (
        -4.0 * base
        + torch.roll(base, shifts=1, dims=2)
        + torch.roll(base, shifts=-1, dims=2)
        + torch.roll(base, shifts=1, dims=3)
        + torch.roll(base, shifts=-1, dims=3)
    )
    channels.extend([dog_small, dog_large, laplacian.abs()])
    dx = (base - torch.roll(base, shifts=1, dims=3)).abs()
    dy = (base - torch.roll(base, shifts=1, dims=2)).abs()
    gradient = dx + dy
    signed_dx = base - torch.roll(base, shifts=1, dims=3)
    signed_dy = base - torch.roll(base, shifts=1, dims=2)
    grad_norm = torch.sqrt(signed_dx.square() + signed_dy.square()).clamp_min(1.0e-6)
    channels.extend([signed_dx / grad_norm, signed_dy / grad_norm, gradient])
    for kernel in (3, 7, 11):
        channels.append(F.avg_pool2d(gradient, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False))
    for ring_radius in (1, 2, 4, 8):
        diffs = []
        signed_diffs = []
        for dy_offset, dx_offset in (
            (-ring_radius, 0),
            (ring_radius, 0),
            (0, -ring_radius),
            (0, ring_radius),
            (-ring_radius, -ring_radius),
            (-ring_radius, ring_radius),
            (ring_radius, -ring_radius),
            (ring_radius, ring_radius),
        ):
            shifted = torch.roll(base, shifts=(dy_offset, dx_offset), dims=(2, 3))
            signed = base - shifted
            signed_diffs.append(signed)
            diffs.append(signed.abs())
        ring = torch.stack(diffs, dim=1)
        signed_ring = torch.stack(signed_diffs, dim=1)
        ring_mean = ring.mean(dim=1)
        channels.append(ring_mean)
        channels.append(ring.max(dim=1).values)
        centered_ring = ring - ring.mean(dim=1, keepdim=True)
        channels.append(centered_ring.pow(2).mean(dim=1).sqrt())
        channels.append(ring_mean * radius)
        channels.append(torch.tanh(signed_ring * 8.0).mean(dim=1))
        channels.append((signed_ring > 0.0).to(base.dtype).mean(dim=1) * 2.0 - 1.0)
    channels.append(gradient * radius)
    target = torch.cat(channels, dim=1)
    target = F.interpolate(target, size=(descriptor_height, descriptor_width), mode="bilinear", align_corners=False)
    centered = target - target.mean(dim=(2, 3), keepdim=True)
    scaled = centered / centered.pow(2).mean(dim=(2, 3), keepdim=True).add(1.0e-4).sqrt()
    if scaled.size(1) < descriptor_dim:
        repeat_count = (descriptor_dim + scaled.size(1) - 1) // scaled.size(1)
        scaled = scaled.repeat(1, repeat_count, 1, 1)
    target = scaled[:, :descriptor_dim].contiguous()
    return _normalize_channels(target)


def blend_rotation_invariant_texture_descriptor(
    descriptors: torch.Tensor,
    image: torch.Tensor,
    blend_weight: float = INFERENCE_TEXTURE_BLEND_WEIGHT,
) -> torch.Tensor:
    target = make_rotation_invariant_texture_descriptor(image, descriptors.size(2), descriptors.size(3), descriptors.size(1))
    return _normalize_channels(descriptors + target * blend_weight)


class TextureDescriptorAdapter(nn.Module):
    """Trainable residual adapter for the analytic texture descriptor."""

    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.descriptor_dim = descriptor_dim
        self.residual = nn.Conv2d(descriptor_dim, descriptor_dim, 1)
        _zero_module(self.residual)

    def forward(self, texture: torch.Tensor) -> torch.Tensor:
        if texture.dim() != 4 or texture.size(1) != self.descriptor_dim:
            raise ValueError("texture tensor must have shape BxDxHxW with the configured descriptor dimension")
        return _normalize_channels(texture + self.residual(texture))


class DescriptorFusionAdapter(nn.Module):
    """Higher-capacity residual fusion for learned and texture descriptors."""

    def __init__(self, descriptor_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.descriptor_dim = descriptor_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(16, descriptor_dim * 2)
        self.input_projection = nn.Conv2d(descriptor_dim * 4, self.hidden_dim, 1)
        self.context = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.texture_gate = nn.Conv2d(descriptor_dim * 4, 1, 1)
        self.output = nn.Conv2d(self.hidden_dim, descriptor_dim, 1)
        _zero_module(self.texture_gate)
        _zero_module(self.output)

    def forward(self, learned: torch.Tensor, texture: torch.Tensor, *, blend_weight: float) -> torch.Tensor:
        if learned.shape != texture.shape or learned.dim() != 4 or learned.size(1) != self.descriptor_dim:
            raise ValueError("learned and texture descriptors must have matching BxDxHxW shapes")
        initial_weighted_texture = texture * float(blend_weight)
        gate_features = torch.cat(
            [
                learned,
                initial_weighted_texture,
                learned - initial_weighted_texture,
                learned * initial_weighted_texture,
            ],
            dim=1,
        )
        texture_gate = 1.0 + 0.5 * torch.tanh(self.texture_gate(gate_features))
        weighted_texture = initial_weighted_texture * texture_gate
        base = _normalize_channels(learned + weighted_texture)
        features = torch.cat(
            [
                learned,
                weighted_texture,
                learned - weighted_texture,
                learned * weighted_texture,
            ],
            dim=1,
        )
        residual = self.output(self.context(self.input_projection(features)))
        return _normalize_channels(base + residual)


def make_rotation_invariant_texture_saliency(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    base = image
    if base.size(1) != 1:
        base = base.mean(dim=1, keepdim=True)
    blur = F.avg_pool2d(base, kernel_size=15, stride=1, padding=7, count_include_pad=False)
    contrast = (base - blur).abs()
    dx = (base - torch.roll(base, shifts=1, dims=3)).abs()
    dy = (base - torch.roll(base, shifts=1, dims=2)).abs()
    saliency = F.avg_pool2d(contrast + dx + dy, kernel_size=5, stride=1, padding=2, count_include_pad=False)
    saliency = F.interpolate(saliency, size=(target_height, target_width), mode="bilinear", align_corners=False)
    flat = saliency.reshape(saliency.size(0), saliency.size(1), -1)
    min_value = flat.min(dim=2, keepdim=True).values.reshape(saliency.size(0), saliency.size(1), 1, 1)
    max_value = flat.max(dim=2, keepdim=True).values.reshape(saliency.size(0), saliency.size(1), 1, 1)
    return (saliency - min_value) / (max_value - min_value).clamp_min(1.0e-6)


class QualityHead(nn.Module):
    """Estimate descriptor/keypoint reliability from fused descriptors and local image structure."""

    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        hidden = max(16, descriptor_dim // 2)
        self.predictor = nn.Sequential(
            nn.Conv2d(descriptor_dim + 3, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        _zero_module(self.predictor[-1])

    def forward(
        self,
        descriptors: torch.Tensor,
        heatmap: torch.Tensor,
        texture_saliency: torch.Tensor,
        dense_confidence: torch.Tensor,
    ) -> torch.Tensor:
        if descriptors.dim() != 4:
            raise ValueError("descriptors must have shape BxDxHxW")
        auxiliaries = []
        for tensor in (heatmap, texture_saliency, dense_confidence):
            if tensor.dim() != 4 or tensor.size(0) != descriptors.size(0) or tensor.size(1) != 1:
                raise ValueError("quality auxiliary maps must have shape Bx1xHxW")
            if tensor.shape[-2:] != descriptors.shape[-2:]:
                tensor = F.interpolate(tensor, size=descriptors.shape[-2:], mode="bilinear", align_corners=False)
            auxiliaries.append(tensor.to(descriptors.dtype))
        logits = self.predictor(torch.cat([descriptors, *auxiliaries], dim=1))
        return torch.sigmoid(logits + 0.5 * auxiliaries[0] + 0.5 * auxiliaries[1] + 0.25 * auxiliaries[2])


class SemiDenseCandidateBranch(nn.Module):
    """Coarse detector-free candidate branch for weak-texture regions."""

    def __init__(self, descriptor_dim: int, projection_dim: int = 64, max_grid: int = 32) -> None:
        super().__init__()
        if descriptor_dim <= 0 or projection_dim <= 0 or max_grid <= 0:
            raise ValueError("descriptor_dim, projection_dim and max_grid must be positive")
        self.descriptor_dim = descriptor_dim
        self.projection_dim = projection_dim
        self.max_grid = max_grid
        self.projection = nn.Sequential(
            nn.Conv2d(descriptor_dim, projection_dim, 1),
            nn.GELU(),
            nn.Conv2d(projection_dim, projection_dim, 1),
        )

    def _coarse(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.dim() != 4 or descriptors.size(1) != self.descriptor_dim:
            raise ValueError("semi-dense descriptors must have shape BxDxHxW")
        height, width = descriptors.shape[-2:]
        target_height = min(height, self.max_grid)
        target_width = min(width, self.max_grid)
        coarse = descriptors
        if (target_height, target_width) != (height, width):
            coarse = F.adaptive_avg_pool2d(coarse, (target_height, target_width))
        return F.normalize(self.projection(coarse), p=2, dim=1, eps=1.0e-12)

    def forward(
        self,
        descriptors_a: torch.Tensor,
        descriptors_b: torch.Tensor,
        *,
        max_candidates: int,
        min_score: float = 0.0,
    ) -> SemiDenseCandidateOutput:
        if descriptors_a.size(0) != 1 or descriptors_b.size(0) != 1:
            raise ValueError("semi-dense candidate branch currently expects single-pair descriptor maps")
        if max_candidates <= 0:
            empty_xy = descriptors_a.new_empty((0, 2))
            empty_scores = descriptors_a.new_empty((0,))
            return SemiDenseCandidateOutput(empty_xy, empty_xy.clone(), empty_scores)
        coarse_a = self._coarse(descriptors_a)
        coarse_b = self._coarse(descriptors_b)
        _, channels, coarse_ha, coarse_wa = coarse_a.shape
        _, _, coarse_hb, coarse_wb = coarse_b.shape
        flat_a = coarse_a.squeeze(0).permute(1, 2, 0).reshape(-1, channels)
        flat_b = coarse_b.squeeze(0).permute(1, 2, 0).reshape(-1, channels)
        logits = flat_a @ flat_b.T / math.sqrt(float(channels))
        dual_scores = torch.softmax(logits, dim=1) * torch.softmax(logits, dim=0)
        flat_scores = dual_scores.reshape(-1)
        candidate_count = min(int(max_candidates), int(flat_scores.numel()))
        if candidate_count == 0:
            empty_xy = descriptors_a.new_empty((0, 2))
            return SemiDenseCandidateOutput(empty_xy, empty_xy.clone(), descriptors_a.new_empty((0,)))
        values, indices = flat_scores.topk(candidate_count)
        keep = values >= float(min_score)
        values = values[keep]
        indices = indices[keep]
        if values.numel() == 0:
            empty_xy = descriptors_a.new_empty((0, 2))
            return SemiDenseCandidateOutput(empty_xy, empty_xy.clone(), values)
        source = torch.div(indices, flat_b.size(0), rounding_mode="floor")
        target = indices.remainder(flat_b.size(0))
        source_y = torch.div(source, coarse_wa, rounding_mode="floor").to(descriptors_a.dtype)
        source_x = source.remainder(coarse_wa).to(descriptors_a.dtype)
        target_y = torch.div(target, coarse_wb, rounding_mode="floor").to(descriptors_b.dtype)
        target_x = target.remainder(coarse_wb).to(descriptors_b.dtype)

        def scale_coords(x: torch.Tensor, y: torch.Tensor, coarse_h: int, coarse_w: int, full_h: int, full_w: int) -> torch.Tensor:
            if coarse_w > 1:
                x = x * float(max(1, full_w - 1)) / float(coarse_w - 1)
            if coarse_h > 1:
                y = y * float(max(1, full_h - 1)) / float(coarse_h - 1)
            return torch.stack([x, y], dim=1)

        keypoints_a = scale_coords(source_x, source_y, coarse_ha, coarse_wa, descriptors_a.size(2), descriptors_a.size(3))
        keypoints_b = scale_coords(target_x, target_y, coarse_hb, coarse_wb, descriptors_b.size(2), descriptors_b.size(3))
        return SemiDenseCandidateOutput(keypoints_a.contiguous(), keypoints_b.contiguous(), values.contiguous())


class PlanetaryFeatureMatcher(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 1,
        base_channels: int = 64,
        descriptor_dim: int = 256,
        graph_hidden_dim: int = 512,
        graph_attention_layers: int = 8,
        graph_keypoint_meta_dim: int = 16,
    ) -> None:
        super().__init__()
        self.config = CheckpointConfig(
            input_channels,
            base_channels,
            descriptor_dim,
            graph_hidden_dim,
            graph_attention_layers,
            graph_keypoint_meta_dim,
        )
        self.backbone = Backbone(input_channels, base_channels)
        self.dual_fpn = DualFPNLite(base_channels)
        self.sparse_head = SparseHead(base_channels * 2, descriptor_dim)
        self.texture_adapter = TextureDescriptorAdapter(descriptor_dim)
        self.descriptor_fusion = DescriptorFusionAdapter(descriptor_dim)
        self.dense_head = DenseHead(base_channels)
        self.quality_head = QualityHead(descriptor_dim)
        self.semi_dense_branch = SemiDenseCandidateBranch(descriptor_dim)
        self.graph_matcher = PlanetaryGraphMatcher(
            descriptor_dim,
            graph_hidden_dim,
            graph_attention_layers,
            graph_keypoint_meta_dim,
        )

    def learned_descriptor_map_single(
        self,
        image: torch.Tensor,
        *,
        activation_checkpointing: bool = False,
    ) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("image must have shape BxCxHxW")
        features = self.backbone(image, activation_checkpointing=activation_checkpointing)
        p2_keypoint, p2_descriptor = self.dual_fpn(features, activation_checkpointing=activation_checkpointing)
        sparse = self.sparse_head(p2_keypoint, p2_descriptor, activation_checkpointing=activation_checkpointing)
        return sparse.descriptors

    def raw_texture_descriptor_map_single(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("image must have shape BxCxHxW")
        descriptor_height = max(1, (image.size(2) + 3) // 4)
        descriptor_width = max(1, (image.size(3) + 3) // 4)
        return make_rotation_invariant_texture_descriptor(
            image,
            descriptor_height,
            descriptor_width,
            self.config.descriptor_dim,
        )

    def texture_descriptor_map_single(self, image: torch.Tensor) -> torch.Tensor:
        return self.texture_adapter(self.raw_texture_descriptor_map_single(image))

    def fuse_descriptor_maps(
        self,
        learned_descriptors: torch.Tensor,
        image: torch.Tensor,
        *,
        texture_blend_weight: float = INFERENCE_TEXTURE_BLEND_WEIGHT,
    ) -> torch.Tensor:
        texture = self.texture_adapter(
            make_rotation_invariant_texture_descriptor(
                image,
                learned_descriptors.size(2),
                learned_descriptors.size(3),
                learned_descriptors.size(1),
            )
        )
        return self.descriptor_fusion(learned_descriptors, texture, blend_weight=texture_blend_weight)

    def descriptor_map_single(
        self,
        image: torch.Tensor,
        *,
        texture_blend_weight: float = INFERENCE_TEXTURE_BLEND_WEIGHT,
        activation_checkpointing: bool = False,
    ) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("image must have shape BxCxHxW")
        features = self.backbone(image, activation_checkpointing=activation_checkpointing)
        p2_keypoint, p2_descriptor = self.dual_fpn(features, activation_checkpointing=activation_checkpointing)
        sparse = self.sparse_head(p2_keypoint, p2_descriptor, activation_checkpointing=activation_checkpointing)
        return self.fuse_descriptor_maps(sparse.descriptors, image, texture_blend_weight=texture_blend_weight)

    def forward_single(
        self,
        image: torch.Tensor,
        *,
        texture_blend_weight: float = INFERENCE_TEXTURE_BLEND_WEIGHT,
        activation_checkpointing: bool = False,
    ) -> RawFeatureMaps:
        if image.dim() != 4:
            raise ValueError("image must have shape BxCxHxW")
        features = self.backbone(image, activation_checkpointing=activation_checkpointing)
        p2_keypoint, p2_descriptor = self.dual_fpn(features, activation_checkpointing=activation_checkpointing)
        sparse = self.sparse_head(p2_keypoint, p2_descriptor, activation_checkpointing=activation_checkpointing)
        descriptors = self.fuse_descriptor_maps(sparse.descriptors, image, texture_blend_weight=texture_blend_weight)
        texture_saliency = make_rotation_invariant_texture_saliency(image, sparse.heatmap.size(2), sparse.heatmap.size(3))
        dense = self.dense_head(features[0], features[0])
        dense_confidence = F.interpolate(dense.confidence, size=sparse.heatmap.shape[-2:], mode="nearest")
        quality = self.quality_head(descriptors, sparse.heatmap, texture_saliency, dense_confidence)
        heatmap = (sparse.heatmap * quality).clamp(0.0, 1.0)
        return RawFeatureMaps(
            heatmap,
            descriptors,
            sparse.scale,
            sparse.orientation,
            sparse.affine,
            dense_confidence,
            sparse.keypoint_offsets,
            quality,
            texture_saliency,
            sparse.matchability,
            sparse.descriptor_uncertainty,
            sparse.no_match_prior,
        )


def _read_int_from_state(state: dict[str, torch.Tensor], key: str, fallback: int | None = None) -> int:
    tensor = state.get(key)
    if tensor is None:
        if fallback is None:
            raise KeyError(f"checkpoint config missing {key}")
        return fallback
    return int(tensor.reshape(-1)[0].item())


def checkpoint_config_from_state_dict(state: dict[str, torch.Tensor]) -> CheckpointConfig:
    descriptor_dim = _read_int_from_state(state, "config.descriptor_dim")
    return CheckpointConfig(
        input_channels=_read_int_from_state(state, "config.input_channels"),
        base_channels=_read_int_from_state(state, "config.base_channels"),
        descriptor_dim=descriptor_dim,
        graph_hidden_dim=_read_int_from_state(state, "config.graph_hidden_dim", max(32, descriptor_dim)),
        graph_attention_layers=_read_int_from_state(state, "config.graph_attention_layers", 1),
        graph_keypoint_meta_dim=_read_int_from_state(state, "config.graph_keypoint_meta_dim", 2),
    )


def _with_default_compatible_state(
    model: PlanetaryFeatureMatcher,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    defaults = model.state_dict()
    patched = {key: value for key, value in state.items() if key in defaults}
    for key, value in defaults.items():
        if key not in patched:
            patched[key] = value.detach().clone()
    return patched


def load_libtorch_checkpoint(
    checkpoint: Path | str,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[PlanetaryFeatureMatcher, CheckpointConfig]:
    archive = torch.jit.load(str(checkpoint), map_location=device)
    raw_state = dict(archive.state_dict())
    config = checkpoint_config_from_state_dict(raw_state)
    model = PlanetaryFeatureMatcher(
        input_channels=config.input_channels,
        base_channels=config.base_channels,
        descriptor_dim=config.descriptor_dim,
        graph_hidden_dim=config.graph_hidden_dim,
        graph_attention_layers=config.graph_attention_layers,
        graph_keypoint_meta_dim=config.graph_keypoint_meta_dim,
    ).to(device)
    model_state = {key: value for key, value in raw_state.items() if not key.startswith("config.")}
    model_state = _with_default_compatible_state(model, model_state)
    result = model.load_state_dict(model_state, strict=strict)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(f"checkpoint load mismatch: {result}")
    model.backbone.sanitize_nonfinite_state()
    model.eval()
    return model, config


def load_pytorch_state(
    checkpoint: Path | str,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[PlanetaryFeatureMatcher, CheckpointConfig]:
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    config_dict = payload["config"]
    config = CheckpointConfig(
        input_channels=int(config_dict["input_channels"]),
        base_channels=int(config_dict["base_channels"]),
        descriptor_dim=int(config_dict["descriptor_dim"]),
        graph_hidden_dim=int(config_dict["graph_hidden_dim"]),
        graph_attention_layers=int(config_dict["graph_attention_layers"]),
        graph_keypoint_meta_dim=int(config_dict.get("graph_keypoint_meta_dim", 2)),
    )
    model = PlanetaryFeatureMatcher(
        input_channels=config.input_channels,
        base_channels=config.base_channels,
        descriptor_dim=config.descriptor_dim,
        graph_hidden_dim=config.graph_hidden_dim,
        graph_attention_layers=config.graph_attention_layers,
        graph_keypoint_meta_dim=config.graph_keypoint_meta_dim,
    ).to(device)
    model_state = _with_default_compatible_state(model, payload["model"])
    result = model.load_state_dict(model_state, strict=strict)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(f"pytorch state load mismatch: {result}")
    model.eval()
    return model, config


def interpolate_pytorch_state_payloads(
    first: dict,
    second: dict,
    *,
    alpha: float,
) -> dict:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if first["config"] != second["config"]:
        raise ValueError("cannot interpolate checkpoints with different configs")
    first_state = first["model"]
    second_state = second["model"]
    if first_state.keys() != second_state.keys():
        raise ValueError("checkpoint model state keys differ")
    mixed_state = {}
    for key, first_tensor in first_state.items():
        second_tensor = second_state[key]
        if first_tensor.shape != second_tensor.shape:
            raise ValueError(f"checkpoint tensor shape differs for {key}")
        if first_tensor.is_floating_point() and second_tensor.is_floating_point():
            mixed_state[key] = first_tensor * (1.0 - alpha) + second_tensor.to(first_tensor.device) * alpha
        else:
            mixed_state[key] = first_tensor.clone()
    return {
        "config": dict(first["config"]),
        "model": mixed_state,
        "interpolation": {
            "alpha": float(alpha),
            "first_source": first.get("source_checkpoint", ""),
            "second_source": second.get("source_checkpoint", ""),
        },
    }
