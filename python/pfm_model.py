#!/usr/bin/env python3
"""PyTorch 版 PFM 主模型。

这个文件保留模型结构、checkpoint 读写和 Python/C++ 对齐所需的公共类。
不要继续把训练 loss、数据读取、报告生成或一次性实验逻辑塞进这里；这些逻辑应该拆到
`pfm_pytorch_training.py` 的子模块或 `scripts/` 下面。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from pfm_model_descriptors import geometry_aware_descriptor_pool, make_xy_grid, normalize_channels_stable

INFERENCE_TEXTURE_BLEND_WEIGHT = 1.0
DESCRIPTOR_GEOMETRY_MODES = ("full", "orientation_scale", "plain")
DESCRIPTOR_GEOMETRY_SAFETY_SCHEDULES = ("off", "phase4")
QUALITY_SCORE_MODES = ("soft", "multiply", "raw")
MATCHER_RELIABILITY_PAIR_BIAS_MODES = ("full", "off")
MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES = ("full", "matchability", "off")
MATCHER_FINAL_ACCEPT_SCORE_MODES = ("multiply", "none", "add")
MATCHER_ACCEPT_ASSIGNMENT_MODES = ("add", "off")
PAIR_ACCEPT_CONTEXT_FEATURE_NAMES = (
    "score_min",
    "score_mean",
    "score_median",
    "score_max",
    "score_std",
    "accept_probability_min",
    "accept_probability_mean",
    "accept_probability_median",
    "accept_probability_max",
    "accept_probability_std",
    "pair_logit_min",
    "pair_logit_mean",
    "pair_logit_median",
    "pair_logit_max",
    "pair_logit_std",
    "raw_similarity_min",
    "raw_similarity_mean",
    "raw_similarity_median",
    "raw_similarity_max",
    "raw_similarity_std",
    "matched_raw_similarity_min",
    "matched_raw_similarity_mean",
    "matched_raw_similarity_median",
    "matched_raw_similarity_max",
    "matched_raw_similarity_std",
    "matched_raw_margin_min",
    "matched_raw_margin_mean",
    "matched_raw_margin_median",
    "matched_raw_margin_max",
    "matched_raw_margin_std",
    "matched_raw_margin_low_fraction",
    "match_count_ratio",
    "kept_keypoints_a_ratio",
    "kept_keypoints_b_ratio",
    "match_dx_median",
    "match_dy_median",
    "match_dx_mad",
    "match_dy_mad",
    "match_displacement_median",
    "match_displacement_mad",
    "match_projective_residual_valid",
    "match_projective_residual_median",
    "match_projective_residual_p90",
)


@dataclass(frozen=True)
class CheckpointConfig:
    input_channels: int
    base_channels: int
    descriptor_dim: int
    graph_hidden_dim: int
    graph_attention_layers: int
    graph_keypoint_meta_dim: int = 2
    descriptor_geometry_mode: str = "full"
    quality_score_mode: str = "soft"
    matcher_reliability_pair_bias_mode: str = "off"
    matcher_reliability_dustbin_bias_mode: str = "off"
    matcher_final_accept_score_mode: str = "none"
    matcher_geometry_bias_scale: float = 1.0
    matcher_accept_assignment_mode: str = "add"
    matcher_final_accept_score_alpha: float = 0.05
    matcher_geometry_bias_clamp: float = 2.0
    matcher_attention_residual_gate_init: float = 1.0
    matcher_candidate_topk: int = 256
    descriptor_geometry_blend_weight: float = 1.0
    descriptor_scale_log_clamp_min: float = -2.0
    descriptor_scale_log_clamp_max: float = 2.0


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
    pair_accept_logit: torch.Tensor | None = None
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
        self.keypoint_from_stage1 = nn.Conv2d(base_channels, p2_channels, 1)
        self.keypoint_from_stage3 = nn.Conv2d(base_channels * 4, p2_channels, 1)
        self.descriptor_from_stage3 = nn.Conv2d(base_channels * 4, p2_channels, 1)
        self.descriptor_from_stage4 = nn.Conv2d(base_channels * 8, p2_channels, 1)
        self.keypoint_refine = ZeroResidualContextBlock(p2_channels)
        self.descriptor_refine = nn.Sequential(
            ZeroResidualContextBlock(p2_channels),
            ZeroResidualContextBlock(p2_channels, dilation=2),
        )
        _zero_module(self.keypoint_from_stage1)
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
        stage1_keypoint = F.interpolate(
            self.keypoint_from_stage1(stage1),
            size=stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        stage3_keypoint = F.interpolate(
            self.keypoint_from_stage3(stage3),
            size=stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p2_keypoint = self.keypoint_refine(stage2 + stage1_keypoint + stage3_keypoint)
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


def apply_quality_score_mode(heatmap: torch.Tensor, quality: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "raw":
        return heatmap.clamp(0.0, 1.0)
    if mode == "multiply":
        return (heatmap * quality).clamp(0.0, 1.0)
    if mode == "soft":
        return (heatmap * (0.5 + 0.5 * quality)).clamp(0.0, 1.0)
    raise ValueError(f"quality score mode must be one of {QUALITY_SCORE_MODES}")


def descriptor_geometry_safety_for_progress(
    schedule: str,
    progress: float,
) -> tuple[float, float, float] | None:
    """Return descriptor geometry blend/clamp settings for the named training schedule."""

    if schedule == "off":
        return None
    if schedule != "phase4":
        raise ValueError(f"descriptor geometry safety schedule must be one of {DESCRIPTOR_GEOMETRY_SAFETY_SCHEDULES}")
    if not math.isfinite(float(progress)):
        raise ValueError("progress must be finite")
    clipped = min(1.0, max(0.0, float(progress)))
    if clipped < 0.2:
        blend_weight = 0.0
    elif clipped < 0.6:
        blend_weight = 0.3 * ((clipped - 0.2) / 0.4)
    else:
        blend_weight = 0.3 + 0.2 * ((clipped - 0.6) / 0.4)
    clamp_abs = 0.7 if clipped < 0.6 else 1.2
    return (float(blend_weight), -float(clamp_abs), float(clamp_abs))


class SparseHead(nn.Module):
    """稀疏特征头：同时预测关键点、descriptor 和局部几何。

    当前版本已经删除旧的 C4 旋转分支。descriptor 的旋转鲁棒性不再靠 0/90/180/270
    离散旋转枚举，而是由 `orientation/scale/affine` 预测出的连续局部几何驱动
    `geometry_aware_descriptor_pool()` 完成。
    """

    def __init__(
        self,
        input_channels: int,
        descriptor_dim: int,
        *,
        descriptor_geometry_mode: str = "full",
        descriptor_geometry_blend_weight: float = 1.0,
        descriptor_scale_log_clamp_min: float = -2.0,
        descriptor_scale_log_clamp_max: float = 2.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or descriptor_dim <= 0:
            raise ValueError("input_channels and descriptor_dim must be positive")
        if descriptor_geometry_mode not in DESCRIPTOR_GEOMETRY_MODES:
            raise ValueError(f"descriptor_geometry_mode must be one of {DESCRIPTOR_GEOMETRY_MODES}")
        if (
            not math.isfinite(float(descriptor_geometry_blend_weight))
            or descriptor_geometry_blend_weight < 0.0
            or descriptor_geometry_blend_weight > 1.0
        ):
            raise ValueError("descriptor_geometry_blend_weight must be in [0, 1]")
        if (
            not math.isfinite(float(descriptor_scale_log_clamp_min))
            or not math.isfinite(float(descriptor_scale_log_clamp_max))
            or descriptor_scale_log_clamp_min > descriptor_scale_log_clamp_max
        ):
            raise ValueError("descriptor scale log clamp bounds must be finite and ordered")
        self.input_channels = input_channels
        self.descriptor_dim = descriptor_dim
        self.descriptor_geometry_mode = descriptor_geometry_mode
        self.descriptor_geometry_blend_weight = float(descriptor_geometry_blend_weight)
        self.descriptor_scale_log_clamp_min = float(descriptor_scale_log_clamp_min)
        self.descriptor_scale_log_clamp_max = float(descriptor_scale_log_clamp_max)
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
        scale = torch.exp(
            torch.clamp(
                self.scale(geometry_context),
                min=float(self.descriptor_scale_log_clamp_min),
                max=float(self.descriptor_scale_log_clamp_max),
            )
        )
        orientation = _normalize_channels(self.orientation(geometry_context))
        affine_delta = torch.tanh(self.affine(geometry_context)) * 0.1
        identity = affine_delta.new_tensor([1.0, 0.0, 0.0, 1.0]).view(1, 4, 1, 1)
        affine = identity + affine_delta

        # 用连续几何做 canonical pooling，替代旧 C4 分支，输出仍保持 dense descriptor map。
        matchability = torch.sigmoid(self.matchability(geometry_context))
        descriptor_uncertainty = torch.sigmoid(self.descriptor_uncertainty(geometry_context))
        no_match_prior = torch.sigmoid(self.no_match_prior(geometry_context))
        if self.descriptor_geometry_mode == "full":
            pooling_affine = affine
        elif self.descriptor_geometry_mode == "orientation_scale":
            pooling_affine = identity.expand_as(affine)
        elif self.descriptor_geometry_mode == "plain":
            pooling_affine = None
        else:
            raise ValueError(f"unsupported descriptor geometry mode: {self.descriptor_geometry_mode}")
        if pooling_affine is not None and self.descriptor_geometry_blend_weight > 0.0:
            pooled_descriptors = _checkpoint_tensor_call(
                activation_checkpointing,
                geometry_aware_descriptor_pool,
                descriptors,
                orientation,
                scale,
                pooling_affine,
            )
            if self.descriptor_geometry_blend_weight >= 1.0:
                descriptors = pooled_descriptors
            else:
                blend_weight = float(self.descriptor_geometry_blend_weight)
                descriptors = _normalize_channels((1.0 - blend_weight) * descriptors + blend_weight * pooled_descriptors)
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
    def __init__(self, hidden_dim: int, *, residual_gate_init: float = 1.0) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(float(residual_gate_init)):
            raise ValueError("residual_gate_init must be finite")
        self.hidden_dim = hidden_dim
        gate_init = float(max(0.0, min(1.0, residual_gate_init)))
        self.self_residual_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.cross_residual_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.feed_forward_residual_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
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

    @staticmethod
    def _clamped_gate(gate: torch.Tensor) -> torch.Tensor:
        return gate.to(dtype=torch.float32).clamp(0.0, 1.0)

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self_a = _attend(self.self_query(features_a), self.self_key(features_a), self.self_value(features_a), self.hidden_dim)
        self_b = _attend(self.self_query(features_b), self.self_key(features_b), self.self_value(features_b), self.hidden_dim)
        self_candidate_a = self.self_norm(features_a + self.attention_dropout(self.self_output(self_a)))
        self_candidate_b = self.self_norm(features_b + self.attention_dropout(self.self_output(self_b)))
        self_gate = self._clamped_gate(self.self_residual_gate).to(device=features_a.device, dtype=features_a.dtype)
        refined_a = features_a + self_gate * (self_candidate_a - features_a)
        refined_b = features_b + self_gate * (self_candidate_b - features_b)
        cross_a = _attend(self.cross_query(refined_a), self.cross_key(refined_b), self.cross_value(refined_b), self.hidden_dim)
        cross_b = _attend(self.cross_query(refined_b), self.cross_key(refined_a), self.cross_value(refined_a), self.hidden_dim)
        cross_candidate_a = self.cross_norm(refined_a + self.attention_dropout(self.cross_output(cross_a)))
        cross_candidate_b = self.cross_norm(refined_b + self.attention_dropout(self.cross_output(cross_b)))
        cross_gate = self._clamped_gate(self.cross_residual_gate).to(device=features_a.device, dtype=features_a.dtype)
        refined_a = refined_a + cross_gate * (cross_candidate_a - refined_a)
        refined_b = refined_b + cross_gate * (cross_candidate_b - refined_b)
        feed_candidate_a = self.feed_forward_norm(refined_a + self.feed_forward(refined_a))
        feed_candidate_b = self.feed_forward_norm(refined_b + self.feed_forward(refined_b))
        feed_gate = self._clamped_gate(self.feed_forward_residual_gate).to(device=features_a.device, dtype=features_a.dtype)
        return (
            refined_a + feed_gate * (feed_candidate_a - refined_a),
            refined_b + feed_gate * (feed_candidate_b - refined_b),
        )


class PlanetaryGraphMatcher(nn.Module):
    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        attention_layers: int = 1,
        keypoint_meta_dim: int = 2,
        candidate_topk: int = 256,
        reliability_pair_bias_mode: str = "off",
        reliability_dustbin_bias_mode: str = "off",
        final_accept_score_mode: str = "none",
        geometry_bias_scale: float = 1.0,
        accept_assignment_mode: str = "add",
        final_accept_score_alpha: float = 0.05,
        geometry_bias_clamp: float = 2.0,
        attention_residual_gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        if descriptor_dim <= 0 or hidden_dim <= 0 or attention_layers <= 0 or keypoint_meta_dim <= 0:
            raise ValueError("descriptor_dim, hidden_dim, attention_layers, and keypoint_meta_dim must be positive")
        if candidate_topk < 0:
            raise ValueError("candidate_topk must be nonnegative")
        if reliability_pair_bias_mode not in MATCHER_RELIABILITY_PAIR_BIAS_MODES:
            raise ValueError(f"reliability_pair_bias_mode must be one of {MATCHER_RELIABILITY_PAIR_BIAS_MODES}")
        if reliability_dustbin_bias_mode not in MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES:
            raise ValueError(f"reliability_dustbin_bias_mode must be one of {MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES}")
        if final_accept_score_mode not in MATCHER_FINAL_ACCEPT_SCORE_MODES:
            raise ValueError(f"final_accept_score_mode must be one of {MATCHER_FINAL_ACCEPT_SCORE_MODES}")
        if accept_assignment_mode not in MATCHER_ACCEPT_ASSIGNMENT_MODES:
            raise ValueError(f"accept_assignment_mode must be one of {MATCHER_ACCEPT_ASSIGNMENT_MODES}")
        if not math.isfinite(float(geometry_bias_scale)):
            raise ValueError("geometry_bias_scale must be finite")
        if not math.isfinite(float(final_accept_score_alpha)) or final_accept_score_alpha < 0.0:
            raise ValueError("final_accept_score_alpha must be finite and nonnegative")
        if not math.isfinite(float(geometry_bias_clamp)) or geometry_bias_clamp < 0.0:
            raise ValueError("geometry_bias_clamp must be finite and nonnegative")
        if not math.isfinite(float(attention_residual_gate_init)):
            raise ValueError("attention_residual_gate_init must be finite")
        self.descriptor_dim = descriptor_dim
        self.hidden_dim = hidden_dim
        self.keypoint_meta_dim = keypoint_meta_dim
        self.candidate_topk = int(candidate_topk)
        self.reliability_pair_bias_mode = reliability_pair_bias_mode
        self.reliability_dustbin_bias_mode = reliability_dustbin_bias_mode
        self.final_accept_score_mode = final_accept_score_mode
        self.geometry_bias_scale = float(geometry_bias_scale)
        self.accept_assignment_mode = accept_assignment_mode
        self.final_accept_score_alpha = float(final_accept_score_alpha)
        self.geometry_bias_clamp = float(geometry_bias_clamp)
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
        self.pair_accept_head = nn.Sequential(
            nn.Linear(8, max(16, hidden_dim // 8)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 8), 1),
        )
        self.pair_accept_context_head = nn.Sequential(
            nn.LayerNorm(len(PAIR_ACCEPT_CONTEXT_FEATURE_NAMES)),
            nn.Linear(len(PAIR_ACCEPT_CONTEXT_FEATURE_NAMES), max(16, hidden_dim // 8)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 8), 1),
        )
        self.logit_scale = nn.Parameter(torch.ones(1) * math.sqrt(float(hidden_dim)))
        self.raw_score_temperature = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.graph_delta_scale = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
        self.accept_logit_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.dustbin_bias = nn.Parameter(torch.zeros(1))
        self.last_executed_attention_layers = 0
        self.attention_layers = nn.ModuleList(
            [
                PlanetaryGraphAttentionLayer(
                    hidden_dim,
                    residual_gate_init=attention_residual_gate_init,
                )
                for _ in range(attention_layers)
            ]
        )
        _zero_module(self.geometry_bias[-1])
        _zero_module(self.accept_head[-1])
        _zero_module(self.pair_accept_head[-1])
        _zero_module(self.pair_accept_context_head[-1])

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
    def _metadata_reliability_score(cls, meta: torch.Tensor, *, mode: str = "full") -> torch.Tensor:
        if mode in ("off", "full", "matchability"):
            return meta.new_zeros((meta.size(0),))
        raise ValueError(f"metadata reliability mode must be one of {MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES}")

    def _pair_reliability_bias(self, meta_a: torch.Tensor, meta_b: torch.Tensor) -> torch.Tensor:
        if self.reliability_pair_bias_mode == "off":
            return meta_a.new_zeros((meta_a.size(0), meta_b.size(0)))
        score_a = self._metadata_reliability_score(meta_a, mode="full")
        score_b = self._metadata_reliability_score(meta_b, mode="full")
        return 0.5 * (score_a[:, None] + score_b[None, :])

    def _dustbin_reliability_score(self, meta: torch.Tensor) -> torch.Tensor:
        return self._metadata_reliability_score(meta, mode=self.reliability_dustbin_bias_mode)

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
        if self.geometry_bias_scale == 0.0:
            return features.new_zeros(features.shape[:-1])
        bias = self.geometry_bias(features).squeeze(-1) * float(self.geometry_bias_scale)
        if self.geometry_bias_clamp > 0.0:
            bias = bias.clamp(-float(self.geometry_bias_clamp), float(self.geometry_bias_clamp))
        return bias

    def _candidate_mask(
        self,
        desc_a: torch.Tensor,
        desc_b: torch.Tensor,
        meta_a: torch.Tensor | None = None,
        meta_b: torch.Tensor | None = None,
        *,
        candidate_topk: int | None = None,
        positive_pair_count: int = 0,
    ) -> torch.Tensor:
        count_a = desc_a.size(0)
        count_b = desc_b.size(0)
        topk = self.candidate_topk if candidate_topk is None else int(candidate_topk)
        if topk <= 0 or topk >= count_b:
            return torch.ones(count_a, count_b, dtype=torch.bool, device=desc_a.device)
        similarity = F.normalize(desc_a, p=2, dim=1, eps=1.0e-12) @ F.normalize(desc_b, p=2, dim=1, eps=1.0e-12).T
        if meta_a is not None and meta_b is not None and self.geometry_bias_scale != 0.0:
            geometry_candidate_bias = self._geometry_compatibility_bias(meta_a, meta_b).detach().to(similarity.dtype)
            similarity = similarity + 0.10 * geometry_candidate_bias.clamp(-1.0, 1.0)
        mask = torch.zeros(count_a, count_b, dtype=torch.bool, device=desc_a.device)
        row_k = min(topk, count_b)
        row_indices = similarity.topk(row_k, dim=1).indices
        mask.scatter_(1, row_indices, True)
        col_k = min(topk, count_a)
        col_indices = similarity.topk(col_k, dim=0).indices
        mask.scatter_(0, col_indices, True)
        diagonal_count = min(int(positive_pair_count), count_a, count_b)
        if diagonal_count > 0:
            diagonal_indices = torch.arange(diagonal_count, device=desc_a.device)
            mask[diagonal_indices, diagonal_indices] = True
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

    @staticmethod
    def _tensor_stats(values: torch.Tensor, *, empty_like: torch.Tensor) -> tuple[torch.Tensor, ...]:
        zero = empty_like.new_zeros(())
        if values.numel() == 0:
            return zero, zero, zero, zero, zero
        values = torch.nan_to_num(values.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return (
            values.min(),
            values.mean(),
            values.median(),
            values.max(),
            values.std(unbiased=False) if values.numel() > 1 else zero,
        )

    def _legacy_pair_acceptance_features(
        self,
        pair_logits: torch.Tensor,
        accept_logits: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        input_keypoints_a: int,
        input_keypoints_b: int,
        kept_keypoints_a: int,
        kept_keypoints_b: int,
    ) -> torch.Tensor:
        accept_probability = torch.sigmoid(accept_logits).to(torch.float32)
        score_values = probabilities.to(torch.float32)
        valid_pair_logits = pair_logits.to(torch.float32)
        valid_pair_logits = valid_pair_logits[torch.isfinite(valid_pair_logits) & valid_pair_logits.gt(-9999.0)]
        match_count = float(score_values.numel())
        keypoint_count = float(max(1, input_keypoints_a + input_keypoints_b))
        zero = pair_logits.new_zeros((), dtype=torch.float32)

        def bounded(value: torch.Tensor) -> torch.Tensor:
            return torch.tanh(value.to(torch.float32) / 8.0)

        return torch.stack(
            [
                bounded(score_values.mean()) if score_values.numel() else zero,
                bounded(score_values.max()) if score_values.numel() else zero,
                accept_probability.mean(),
                accept_probability.max(),
                bounded(valid_pair_logits.mean()) if valid_pair_logits.numel() else zero,
                torch.tensor(match_count / keypoint_count, dtype=torch.float32, device=pair_logits.device),
                torch.tensor(
                    kept_keypoints_a / max(1, input_keypoints_a),
                    dtype=torch.float32,
                    device=pair_logits.device,
                ),
                torch.tensor(
                    kept_keypoints_b / max(1, input_keypoints_b),
                    dtype=torch.float32,
                    device=pair_logits.device,
                ),
            ]
        )

    def _pair_acceptance_context_features(
        self,
        pair_logits: torch.Tensor,
        accept_logits: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        raw_similarity: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        meta_a: torch.Tensor,
        meta_b: torch.Tensor,
        input_keypoints_a: int,
        input_keypoints_b: int,
        kept_keypoints_a: int,
        kept_keypoints_b: int,
    ) -> torch.Tensor:
        base = pair_logits.to(torch.float32)
        valid_logit_mask = torch.isfinite(base) & base.gt(-9999.0)
        valid_pair_logits = base[valid_logit_mask]
        valid_accept = torch.sigmoid(accept_logits.to(torch.float32))[valid_logit_mask]
        score_values = probabilities.to(torch.float32)
        score_stats = self._tensor_stats(score_values, empty_like=base)
        accept_stats = self._tensor_stats(valid_accept, empty_like=base)
        pair_logit_stats = self._tensor_stats(valid_pair_logits, empty_like=base)
        raw_values = raw_similarity.to(torch.float32)
        valid_raw_similarity = raw_values[torch.isfinite(raw_values)]
        raw_similarity_stats = self._tensor_stats(valid_raw_similarity, empty_like=base)

        if source_indices.numel() > 0 and target_indices.numel() > 0 and raw_values.numel() > 0:
            raw_source_indices = source_indices.to(device=raw_values.device, dtype=torch.long)
            raw_target_indices = target_indices.to(device=raw_values.device, dtype=torch.long)
            matched_raw_similarity = raw_values[raw_source_indices, raw_target_indices]
            if raw_values.size(1) > 1:
                row_top2 = raw_values.topk(2, dim=1).values
                row_margin = (row_top2[:, 0] - row_top2[:, 1]).clamp(-2.0, 2.0)
            else:
                row_margin = raw_values.new_zeros((raw_values.size(0),))
            if raw_values.size(0) > 1:
                col_top2 = raw_values.topk(2, dim=0).values
                col_margin = (col_top2[0] - col_top2[1]).clamp(-2.0, 2.0)
            else:
                col_margin = raw_values.new_zeros((raw_values.size(1),))
            matched_raw_margin = torch.minimum(
                row_margin.index_select(0, raw_source_indices),
                col_margin.index_select(0, raw_target_indices),
            )
        else:
            matched_raw_similarity = raw_values.new_empty((0,))
            matched_raw_margin = raw_values.new_empty((0,))
        matched_raw_similarity_stats = self._tensor_stats(matched_raw_similarity, empty_like=base)
        matched_raw_margin_stats = self._tensor_stats(matched_raw_margin, empty_like=base)
        if matched_raw_margin.numel() > 0:
            matched_raw_margin_low_fraction = matched_raw_margin.le(0.05).to(torch.float32).mean()
        else:
            matched_raw_margin_low_fraction = base.new_zeros(())

        match_count = float(score_values.numel())
        keypoint_count = float(max(1, input_keypoints_a + input_keypoints_b))
        match_count_ratio = base.new_tensor(match_count / keypoint_count)
        kept_a_ratio = base.new_tensor(kept_keypoints_a / max(1, input_keypoints_a))
        kept_b_ratio = base.new_tensor(kept_keypoints_b / max(1, input_keypoints_b))

        zero = base.new_zeros(())
        if source_indices.numel() > 0 and target_indices.numel() > 0 and meta_a.size(1) >= 2 and meta_b.size(1) >= 2:
            source_indices = source_indices.to(device=meta_a.device, dtype=torch.long)
            target_indices = target_indices.to(device=meta_b.device, dtype=torch.long)
            source_xy = meta_a.index_select(0, source_indices)[:, :2].to(torch.float32)
            target_xy = meta_b.index_select(0, target_indices)[:, :2].to(torch.float32)
            displacement = target_xy - source_xy
            dx = displacement[:, 0]
            dy = displacement[:, 1]
            distance = torch.linalg.vector_norm(displacement, dim=1)
            dx_median = dx.median()
            dy_median = dy.median()
            distance_median = distance.median()
            dx_mad = (dx - dx_median).abs().median()
            dy_mad = (dy - dy_median).abs().median()
            distance_mad = (distance - distance_median).abs().median()
            projective_valid, projective_median, projective_p90 = self._projective_residual_stats(
                source_xy,
                target_xy,
                empty_like=base,
            )
        else:
            dx_median = dy_median = dx_mad = dy_mad = distance_median = distance_mad = zero
            projective_valid = projective_median = projective_p90 = zero

        features = torch.stack(
            [
                *score_stats,
                *accept_stats,
                *pair_logit_stats,
                *raw_similarity_stats,
                *matched_raw_similarity_stats,
                *matched_raw_margin_stats,
                matched_raw_margin_low_fraction,
                match_count_ratio,
                kept_a_ratio,
                kept_b_ratio,
                dx_median,
                dy_median,
                dx_mad,
                dy_mad,
                distance_median,
                distance_mad,
                projective_valid,
                projective_median,
                projective_p90,
            ]
        )
        return torch.nan_to_num(features, nan=0.0, posinf=1.0e4, neginf=-1.0e4)

    @staticmethod
    def _normalize_projective_points(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centroid = points.mean(dim=0)
        centered = points - centroid
        mean_distance = torch.linalg.vector_norm(centered, dim=1).mean().clamp_min(1.0e-6)
        scale = points.new_tensor(math.sqrt(2.0)) / mean_distance
        normalized = centered * scale
        transform = points.new_tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform[0, 0] = scale
        transform[1, 1] = scale
        transform[0, 2] = -scale * centroid[0]
        transform[1, 2] = -scale * centroid[1]
        return normalized, transform

    @classmethod
    def _projective_residual_stats(
        cls,
        source_xy: torch.Tensor,
        target_xy: torch.Tensor,
        *,
        empty_like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_xy.size(0) < 4 or target_xy.size(0) < 4:
            zero = empty_like.new_zeros(())
            return zero, zero, zero

        source = source_xy.to(torch.float32)
        target = target_xy.to(torch.float32)
        source_norm, source_transform = cls._normalize_projective_points(source)
        target_norm, target_transform = cls._normalize_projective_points(target)
        x = source_norm[:, 0]
        y = source_norm[:, 1]
        u = target_norm[:, 0]
        v = target_norm[:, 1]
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        rows_u = torch.stack((-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u), dim=1)
        rows_v = torch.stack((zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v), dim=1)
        system = torch.stack((rows_u, rows_v), dim=1).reshape(-1, 9)
        try:
            _u, _s, vh = torch.linalg.svd(system, full_matrices=False)
        except RuntimeError:
            zero = empty_like.new_zeros(())
            return zero, zero, zero
        homography_norm = vh[-1].reshape(3, 3)
        if not torch.isfinite(homography_norm).all():
            zero = empty_like.new_zeros(())
            return zero, zero, zero
        homography = torch.linalg.inv(target_transform) @ homography_norm @ source_transform
        homogeneous_source = torch.cat([source, torch.ones((source.size(0), 1), dtype=source.dtype, device=source.device)], dim=1)
        projected = (homography @ homogeneous_source.transpose(0, 1)).transpose(0, 1)
        denominator = projected[:, 2:3]
        valid = denominator.abs().gt(1.0e-6).squeeze(1)
        if not bool(valid.any()):
            zero = empty_like.new_zeros(())
            return zero, zero, zero
        projected_xy = projected[valid, :2] / denominator[valid]
        residual = torch.linalg.vector_norm(projected_xy - target[valid], dim=1)
        if residual.numel() == 0 or not torch.isfinite(residual).all():
            zero = empty_like.new_zeros(())
            return zero, zero, zero
        valid_ratio = residual.new_tensor(float(residual.numel()) / float(source.size(0)))
        median = residual.median()
        p90 = torch.quantile(residual, 0.9) if residual.numel() > 1 else residual[0]
        return valid_ratio, median, p90

    def _pair_acceptance_logit(
        self,
        pair_logits: torch.Tensor,
        accept_logits: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        raw_similarity: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        meta_a: torch.Tensor,
        meta_b: torch.Tensor,
        input_keypoints_a: int,
        input_keypoints_b: int,
        kept_keypoints_a: int,
        kept_keypoints_b: int,
    ) -> torch.Tensor:
        if pair_logits.numel() == 0:
            return self.dustbin_bias.new_zeros(())
        legacy_features = self._legacy_pair_acceptance_features(
            pair_logits,
            accept_logits,
            probabilities,
            input_keypoints_a=input_keypoints_a,
            input_keypoints_b=input_keypoints_b,
            kept_keypoints_a=kept_keypoints_a,
            kept_keypoints_b=kept_keypoints_b,
        )
        context_features = self._pair_acceptance_context_features(
            pair_logits,
            accept_logits,
            probabilities,
            raw_similarity=raw_similarity,
            source_indices=source_indices,
            target_indices=target_indices,
            meta_a=meta_a,
            meta_b=meta_b,
            input_keypoints_a=input_keypoints_a,
            input_keypoints_b=input_keypoints_b,
            kept_keypoints_a=kept_keypoints_a,
            kept_keypoints_b=kept_keypoints_b,
        )
        return self.pair_accept_head(legacy_features).squeeze() + self.pair_accept_context_head(context_features).squeeze()

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
        pair_logits = raw_similarity / raw_temperature + delta_scale * graph_delta
        if self.accept_assignment_mode == "add":
            pair_logits = pair_logits + accept_scale * accept_logits
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
        candidate_topk: int | None = None,
        positive_pair_count_for_mask: int = 0,
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
            pair_logits_work = raw_similarity / raw_temperature + delta_scale * graph_delta
            if self.accept_assignment_mode == "add":
                pair_logits_work = pair_logits_work + accept_scale * accept_logits_work
            if apply_candidate_mask:
                candidate_mask = self._candidate_mask(
                    desc_work_a,
                    desc_work_b,
                    kp_work_a,
                    kp_work_b,
                    candidate_topk=candidate_topk,
                    positive_pair_count=positive_pair_count_for_mask,
                )
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
                logits[: descriptors_a.size(0), descriptors_b.size(0)] - self._dustbin_reliability_score(kp_a)
            )
        if descriptors_b.size(0) > 0:
            logits[descriptors_a.size(0), : descriptors_b.size(0)] = (
                logits[descriptors_a.size(0), : descriptors_b.size(0)] - self._dustbin_reliability_score(kp_b)
            )
        count_a = descriptors_a.size(0)
        count_b = descriptors_b.size(0)
        if count_a > 0 and count_b > 0:
            margin_scores = (
                logits[:count_a, :count_b]
                - logits[:count_a, count_b][:, None]
                - logits[count_a, :count_b][None, :]
            )
            best_values, best_indices = margin_scores.max(dim=1)
            source_indices = torch.arange(count_a, device=best_indices.device)
            reverse_best = margin_scores.max(dim=0).indices
            mutual_sources = reverse_best.index_select(0, best_indices)
            inlier_mask = best_values.gt(0.0) & mutual_sources.eq(source_indices)
            source_indices = source_indices[inlier_mask]
            target_indices = best_indices[inlier_mask]
            probabilities = best_values[inlier_mask]
        else:
            source_indices = torch.empty(0, dtype=torch.long, device=logits.device)
            target_indices = torch.empty(0, dtype=torch.long, device=logits.device)
            probabilities = logits.new_empty((0,))
        if (
            self.final_accept_score_mode == "multiply"
            and probabilities.numel() > 0
            and accept_logits.numel() > 0
        ):
            probabilities = probabilities * torch.sigmoid(accept_logits[source_indices, target_indices])
        elif (
            self.final_accept_score_mode == "add"
            and probabilities.numel() > 0
            and accept_logits.numel() > 0
        ):
            probabilities = (
                probabilities
                + float(self.final_accept_score_alpha) * torch.sigmoid(accept_logits[source_indices, target_indices])
            )
        matches = torch.stack([source_indices, target_indices], dim=1).to(device="cpu", dtype=torch.long).contiguous()
        scores = probabilities.to(device="cpu", dtype=torch.float32).contiguous()
        kept_keypoints_a = int(indices_a.numel())
        kept_keypoints_b = int(indices_b.numel())
        attention_work_fraction = (
            0.0 if full_attention_work_units == 0 else attention_work_units / full_attention_work_units
        )
        pair_accept_logit = self._pair_acceptance_logit(
            pair_logits,
            accept_logits,
            probabilities,
            raw_similarity=raw_similarity_full,
            source_indices=source_indices,
            target_indices=target_indices,
            meta_a=kp_a,
            meta_b=kp_b,
            input_keypoints_a=input_keypoints_a,
            input_keypoints_b=input_keypoints_b,
            kept_keypoints_a=kept_keypoints_a,
            kept_keypoints_b=kept_keypoints_b,
        )
        return GraphMatcherOutput(
            logits.contiguous(),
            matches,
            scores,
            accept_logits.contiguous(),
            pair_accept_logit=pair_accept_logit,
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
        descriptor_geometry_mode: str = "full",
        quality_score_mode: str = "soft",
        matcher_reliability_pair_bias_mode: str = "off",
        matcher_reliability_dustbin_bias_mode: str = "off",
        matcher_final_accept_score_mode: str = "none",
        matcher_geometry_bias_scale: float = 1.0,
        matcher_accept_assignment_mode: str = "add",
        matcher_final_accept_score_alpha: float = 0.05,
        matcher_geometry_bias_clamp: float = 2.0,
        matcher_attention_residual_gate_init: float = 1.0,
        matcher_candidate_topk: int = 256,
        descriptor_geometry_blend_weight: float = 1.0,
        descriptor_scale_log_clamp_min: float = -2.0,
        descriptor_scale_log_clamp_max: float = 2.0,
    ) -> None:
        super().__init__()
        if descriptor_geometry_mode not in DESCRIPTOR_GEOMETRY_MODES:
            raise ValueError(f"descriptor_geometry_mode must be one of {DESCRIPTOR_GEOMETRY_MODES}")
        if quality_score_mode not in QUALITY_SCORE_MODES:
            raise ValueError(f"quality_score_mode must be one of {QUALITY_SCORE_MODES}")
        if matcher_reliability_pair_bias_mode not in MATCHER_RELIABILITY_PAIR_BIAS_MODES:
            raise ValueError(f"matcher_reliability_pair_bias_mode must be one of {MATCHER_RELIABILITY_PAIR_BIAS_MODES}")
        if matcher_reliability_dustbin_bias_mode not in MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES:
            raise ValueError(
                f"matcher_reliability_dustbin_bias_mode must be one of {MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES}"
            )
        if matcher_final_accept_score_mode not in MATCHER_FINAL_ACCEPT_SCORE_MODES:
            raise ValueError(f"matcher_final_accept_score_mode must be one of {MATCHER_FINAL_ACCEPT_SCORE_MODES}")
        if matcher_accept_assignment_mode not in MATCHER_ACCEPT_ASSIGNMENT_MODES:
            raise ValueError(f"matcher_accept_assignment_mode must be one of {MATCHER_ACCEPT_ASSIGNMENT_MODES}")
        if not math.isfinite(float(matcher_geometry_bias_scale)):
            raise ValueError("matcher_geometry_bias_scale must be finite")
        if not math.isfinite(float(matcher_final_accept_score_alpha)) or matcher_final_accept_score_alpha < 0.0:
            raise ValueError("matcher_final_accept_score_alpha must be finite and nonnegative")
        if not math.isfinite(float(matcher_geometry_bias_clamp)) or matcher_geometry_bias_clamp < 0.0:
            raise ValueError("matcher_geometry_bias_clamp must be finite and nonnegative")
        if not math.isfinite(float(matcher_attention_residual_gate_init)):
            raise ValueError("matcher_attention_residual_gate_init must be finite")
        if matcher_candidate_topk < 0:
            raise ValueError("matcher_candidate_topk must be nonnegative")
        if (
            not math.isfinite(float(descriptor_geometry_blend_weight))
            or descriptor_geometry_blend_weight < 0.0
            or descriptor_geometry_blend_weight > 1.0
        ):
            raise ValueError("descriptor_geometry_blend_weight must be in [0, 1]")
        if (
            not math.isfinite(float(descriptor_scale_log_clamp_min))
            or not math.isfinite(float(descriptor_scale_log_clamp_max))
            or descriptor_scale_log_clamp_min > descriptor_scale_log_clamp_max
        ):
            raise ValueError("descriptor scale log clamp bounds must be finite and ordered")
        self.config = CheckpointConfig(
            input_channels,
            base_channels,
            descriptor_dim,
            graph_hidden_dim,
            graph_attention_layers,
            graph_keypoint_meta_dim,
            descriptor_geometry_mode,
            quality_score_mode,
            matcher_reliability_pair_bias_mode,
            matcher_reliability_dustbin_bias_mode,
            matcher_final_accept_score_mode,
            float(matcher_geometry_bias_scale),
            matcher_accept_assignment_mode,
            float(matcher_final_accept_score_alpha),
            float(matcher_geometry_bias_clamp),
            float(matcher_attention_residual_gate_init),
            int(matcher_candidate_topk),
            float(descriptor_geometry_blend_weight),
            float(descriptor_scale_log_clamp_min),
            float(descriptor_scale_log_clamp_max),
        )
        self.backbone = Backbone(input_channels, base_channels)
        self.dual_fpn = DualFPNLite(base_channels)
        self.sparse_head = SparseHead(
            base_channels * 2,
            descriptor_dim,
            descriptor_geometry_mode=descriptor_geometry_mode,
            descriptor_geometry_blend_weight=descriptor_geometry_blend_weight,
            descriptor_scale_log_clamp_min=descriptor_scale_log_clamp_min,
            descriptor_scale_log_clamp_max=descriptor_scale_log_clamp_max,
        )
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
            reliability_pair_bias_mode=matcher_reliability_pair_bias_mode,
            reliability_dustbin_bias_mode=matcher_reliability_dustbin_bias_mode,
            final_accept_score_mode=matcher_final_accept_score_mode,
            geometry_bias_scale=matcher_geometry_bias_scale,
            accept_assignment_mode=matcher_accept_assignment_mode,
            final_accept_score_alpha=matcher_final_accept_score_alpha,
            geometry_bias_clamp=matcher_geometry_bias_clamp,
            attention_residual_gate_init=matcher_attention_residual_gate_init,
            candidate_topk=matcher_candidate_topk,
        )

    def set_descriptor_geometry_mode(self, mode: str) -> None:
        if mode not in DESCRIPTOR_GEOMETRY_MODES:
            raise ValueError(f"descriptor geometry mode must be one of {DESCRIPTOR_GEOMETRY_MODES}")
        self.sparse_head.descriptor_geometry_mode = mode
        self.config = replace(self.config, descriptor_geometry_mode=mode)

    def set_descriptor_geometry_safety(
        self,
        *,
        blend_weight: float | None = None,
        scale_log_clamp_min: float | None = None,
        scale_log_clamp_max: float | None = None,
    ) -> None:
        resolved_blend = (
            self.config.descriptor_geometry_blend_weight
            if blend_weight is None
            else float(blend_weight)
        )
        resolved_min = (
            self.config.descriptor_scale_log_clamp_min
            if scale_log_clamp_min is None
            else float(scale_log_clamp_min)
        )
        resolved_max = (
            self.config.descriptor_scale_log_clamp_max
            if scale_log_clamp_max is None
            else float(scale_log_clamp_max)
        )
        if not math.isfinite(resolved_blend) or resolved_blend < 0.0 or resolved_blend > 1.0:
            raise ValueError("descriptor geometry blend weight must be in [0, 1]")
        if not math.isfinite(resolved_min) or not math.isfinite(resolved_max) or resolved_min > resolved_max:
            raise ValueError("descriptor scale log clamp bounds must be finite and ordered")
        self.sparse_head.descriptor_geometry_blend_weight = resolved_blend
        self.sparse_head.descriptor_scale_log_clamp_min = resolved_min
        self.sparse_head.descriptor_scale_log_clamp_max = resolved_max
        self.config = replace(
            self.config,
            descriptor_geometry_blend_weight=resolved_blend,
            descriptor_scale_log_clamp_min=resolved_min,
            descriptor_scale_log_clamp_max=resolved_max,
        )

    def set_quality_score_mode(self, mode: str) -> None:
        if mode not in QUALITY_SCORE_MODES:
            raise ValueError(f"quality score mode must be one of {QUALITY_SCORE_MODES}")
        self.config = replace(self.config, quality_score_mode=mode)

    def set_matcher_calibration(
        self,
        *,
        reliability_pair_bias_mode: str | None = None,
        reliability_dustbin_bias_mode: str | None = None,
        final_accept_score_mode: str | None = None,
        geometry_bias_scale: float | None = None,
        accept_assignment_mode: str | None = None,
        final_accept_score_alpha: float | None = None,
        geometry_bias_clamp: float | None = None,
        attention_residual_gate_init: float | None = None,
        attention_residual_gate_start_layer: int = 1,
        candidate_topk: int | None = None,
    ) -> None:
        pair_mode = self.config.matcher_reliability_pair_bias_mode if reliability_pair_bias_mode is None else reliability_pair_bias_mode
        dustbin_mode = (
            self.config.matcher_reliability_dustbin_bias_mode
            if reliability_dustbin_bias_mode is None
            else reliability_dustbin_bias_mode
        )
        accept_mode = self.config.matcher_final_accept_score_mode if final_accept_score_mode is None else final_accept_score_mode
        geometry_scale = (
            self.config.matcher_geometry_bias_scale if geometry_bias_scale is None else float(geometry_bias_scale)
        )
        accept_assignment = (
            self.config.matcher_accept_assignment_mode
            if accept_assignment_mode is None
            else accept_assignment_mode
        )
        accept_alpha = (
            self.config.matcher_final_accept_score_alpha
            if final_accept_score_alpha is None
            else float(final_accept_score_alpha)
        )
        geometry_clamp = (
            self.config.matcher_geometry_bias_clamp
            if geometry_bias_clamp is None
            else float(geometry_bias_clamp)
        )
        residual_gate_init = (
            None if attention_residual_gate_init is None else float(attention_residual_gate_init)
        )
        candidate_count = self.config.matcher_candidate_topk if candidate_topk is None else int(candidate_topk)
        if pair_mode not in MATCHER_RELIABILITY_PAIR_BIAS_MODES:
            raise ValueError(f"reliability_pair_bias_mode must be one of {MATCHER_RELIABILITY_PAIR_BIAS_MODES}")
        if dustbin_mode not in MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES:
            raise ValueError(f"reliability_dustbin_bias_mode must be one of {MATCHER_RELIABILITY_DUSTBIN_BIAS_MODES}")
        if accept_mode not in MATCHER_FINAL_ACCEPT_SCORE_MODES:
            raise ValueError(f"final_accept_score_mode must be one of {MATCHER_FINAL_ACCEPT_SCORE_MODES}")
        if accept_assignment not in MATCHER_ACCEPT_ASSIGNMENT_MODES:
            raise ValueError(f"accept_assignment_mode must be one of {MATCHER_ACCEPT_ASSIGNMENT_MODES}")
        if not math.isfinite(float(geometry_scale)):
            raise ValueError("geometry_bias_scale must be finite")
        if not math.isfinite(float(accept_alpha)) or accept_alpha < 0.0:
            raise ValueError("final_accept_score_alpha must be finite and nonnegative")
        if not math.isfinite(float(geometry_clamp)) or geometry_clamp < 0.0:
            raise ValueError("geometry_bias_clamp must be finite and nonnegative")
        if residual_gate_init is not None and not math.isfinite(float(residual_gate_init)):
            raise ValueError("attention_residual_gate_init must be finite")
        if int(attention_residual_gate_start_layer) < 1:
            raise ValueError("attention_residual_gate_start_layer must be at least 1")
        if candidate_count < 0:
            raise ValueError("candidate_topk must be nonnegative")
        self.graph_matcher.reliability_pair_bias_mode = pair_mode
        self.graph_matcher.reliability_dustbin_bias_mode = dustbin_mode
        self.graph_matcher.final_accept_score_mode = accept_mode
        self.graph_matcher.geometry_bias_scale = float(geometry_scale)
        self.graph_matcher.accept_assignment_mode = accept_assignment
        self.graph_matcher.final_accept_score_alpha = float(accept_alpha)
        self.graph_matcher.geometry_bias_clamp = float(geometry_clamp)
        self.graph_matcher.candidate_topk = int(candidate_count)
        if residual_gate_init is not None:
            clamped_gate_init = float(max(0.0, min(1.0, residual_gate_init)))
            start_index = int(attention_residual_gate_start_layer) - 1
            for index, layer in enumerate(self.graph_matcher.attention_layers):
                if index < start_index:
                    continue
                layer.self_residual_gate.data.fill_(clamped_gate_init)
                layer.cross_residual_gate.data.fill_(clamped_gate_init)
                layer.feed_forward_residual_gate.data.fill_(clamped_gate_init)
        self.config = replace(
            self.config,
            matcher_reliability_pair_bias_mode=pair_mode,
            matcher_reliability_dustbin_bias_mode=dustbin_mode,
            matcher_final_accept_score_mode=accept_mode,
            matcher_geometry_bias_scale=float(geometry_scale),
            matcher_accept_assignment_mode=accept_assignment,
            matcher_final_accept_score_alpha=float(accept_alpha),
            matcher_geometry_bias_clamp=float(geometry_clamp),
            matcher_attention_residual_gate_init=(
                self.config.matcher_attention_residual_gate_init
                if residual_gate_init is None
                else float(residual_gate_init)
            ),
            matcher_candidate_topk=int(candidate_count),
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
        if self.config.quality_score_mode == "raw":
            dense_confidence = torch.ones_like(sparse.heatmap)
            quality = torch.ones_like(sparse.heatmap)
        else:
            dense = self.dense_head(features[0], features[0])
            dense_confidence = F.interpolate(dense.confidence, size=sparse.heatmap.shape[-2:], mode="nearest")
            quality = self.quality_head(descriptors, sparse.heatmap, texture_saliency, dense_confidence)
        heatmap = apply_quality_score_mode(sparse.heatmap, quality, mode=self.config.quality_score_mode)
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
        matcher_candidate_topk=_read_int_from_state(state, "config.matcher_candidate_topk", 256),
    )


def _with_default_compatible_state(
    model: PlanetaryFeatureMatcher,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    defaults = model.state_dict()
    patched = {
        key: value
        for key, value in state.items()
        if key in defaults and tuple(value.shape) == tuple(defaults[key].shape)
    }
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
        descriptor_geometry_mode=config.descriptor_geometry_mode,
        quality_score_mode=config.quality_score_mode,
        matcher_reliability_pair_bias_mode=config.matcher_reliability_pair_bias_mode,
        matcher_reliability_dustbin_bias_mode=config.matcher_reliability_dustbin_bias_mode,
        matcher_final_accept_score_mode=config.matcher_final_accept_score_mode,
        matcher_geometry_bias_scale=config.matcher_geometry_bias_scale,
        matcher_accept_assignment_mode=config.matcher_accept_assignment_mode,
        matcher_final_accept_score_alpha=config.matcher_final_accept_score_alpha,
        matcher_geometry_bias_clamp=config.matcher_geometry_bias_clamp,
        matcher_attention_residual_gate_init=config.matcher_attention_residual_gate_init,
        matcher_candidate_topk=config.matcher_candidate_topk,
        descriptor_geometry_blend_weight=config.descriptor_geometry_blend_weight,
        descriptor_scale_log_clamp_min=config.descriptor_scale_log_clamp_min,
        descriptor_scale_log_clamp_max=config.descriptor_scale_log_clamp_max,
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
    graph_hidden_dim: int | None = None,
    graph_attention_layers: int | None = None,
) -> tuple[PlanetaryFeatureMatcher, CheckpointConfig]:
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    config_dict = payload["config"]
    resolved_graph_hidden_dim = (
        int(config_dict["graph_hidden_dim"]) if graph_hidden_dim is None else int(graph_hidden_dim)
    )
    resolved_graph_attention_layers = (
        int(config_dict["graph_attention_layers"]) if graph_attention_layers is None else int(graph_attention_layers)
    )
    if resolved_graph_hidden_dim <= 0:
        raise ValueError("graph_hidden_dim must be positive")
    if resolved_graph_attention_layers <= 0:
        raise ValueError("graph_attention_layers must be positive")
    config = CheckpointConfig(
        input_channels=int(config_dict["input_channels"]),
        base_channels=int(config_dict["base_channels"]),
        descriptor_dim=int(config_dict["descriptor_dim"]),
        graph_hidden_dim=resolved_graph_hidden_dim,
        graph_attention_layers=resolved_graph_attention_layers,
        graph_keypoint_meta_dim=int(config_dict.get("graph_keypoint_meta_dim", 2)),
        descriptor_geometry_mode=str(config_dict.get("descriptor_geometry_mode", "full")),
        quality_score_mode=str(config_dict.get("quality_score_mode", "soft")),
        matcher_reliability_pair_bias_mode=str(config_dict.get("matcher_reliability_pair_bias_mode", "off")),
        matcher_reliability_dustbin_bias_mode=str(config_dict.get("matcher_reliability_dustbin_bias_mode", "off")),
        matcher_final_accept_score_mode=str(config_dict.get("matcher_final_accept_score_mode", "none")),
        matcher_geometry_bias_scale=float(config_dict.get("matcher_geometry_bias_scale", 1.0)),
        matcher_accept_assignment_mode=str(config_dict.get("matcher_accept_assignment_mode", "add")),
        matcher_final_accept_score_alpha=float(config_dict.get("matcher_final_accept_score_alpha", 0.05)),
        matcher_geometry_bias_clamp=float(config_dict.get("matcher_geometry_bias_clamp", 2.0)),
        matcher_attention_residual_gate_init=float(config_dict.get("matcher_attention_residual_gate_init", 1.0)),
        matcher_candidate_topk=int(config_dict.get("matcher_candidate_topk", 256)),
        descriptor_geometry_blend_weight=float(config_dict.get("descriptor_geometry_blend_weight", 1.0)),
        descriptor_scale_log_clamp_min=float(config_dict.get("descriptor_scale_log_clamp_min", -2.0)),
        descriptor_scale_log_clamp_max=float(config_dict.get("descriptor_scale_log_clamp_max", 2.0)),
    )
    model = PlanetaryFeatureMatcher(
        input_channels=config.input_channels,
        base_channels=config.base_channels,
        descriptor_dim=config.descriptor_dim,
        graph_hidden_dim=config.graph_hidden_dim,
        graph_attention_layers=config.graph_attention_layers,
        graph_keypoint_meta_dim=config.graph_keypoint_meta_dim,
        descriptor_geometry_mode=config.descriptor_geometry_mode,
        quality_score_mode=config.quality_score_mode,
        matcher_reliability_pair_bias_mode=config.matcher_reliability_pair_bias_mode,
        matcher_reliability_dustbin_bias_mode=config.matcher_reliability_dustbin_bias_mode,
        matcher_final_accept_score_mode=config.matcher_final_accept_score_mode,
        matcher_geometry_bias_scale=config.matcher_geometry_bias_scale,
        matcher_accept_assignment_mode=config.matcher_accept_assignment_mode,
        matcher_final_accept_score_alpha=config.matcher_final_accept_score_alpha,
        matcher_geometry_bias_clamp=config.matcher_geometry_bias_clamp,
        matcher_attention_residual_gate_init=config.matcher_attention_residual_gate_init,
        matcher_candidate_topk=config.matcher_candidate_topk,
        descriptor_geometry_blend_weight=config.descriptor_geometry_blend_weight,
        descriptor_scale_log_clamp_min=config.descriptor_scale_log_clamp_min,
        descriptor_scale_log_clamp_max=config.descriptor_scale_log_clamp_max,
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
