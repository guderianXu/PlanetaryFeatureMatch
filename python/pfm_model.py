#!/usr/bin/env python3
"""PyTorch implementation of the current C++/LibTorch PFM model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

INFERENCE_TEXTURE_BLEND_WEIGHT = 1.0


@dataclass(frozen=True)
class CheckpointConfig:
    input_channels: int
    base_channels: int
    descriptor_dim: int
    graph_hidden_dim: int
    graph_attention_layers: int


@dataclass(frozen=True)
class SparseHeadOutput:
    heatmap: torch.Tensor
    descriptors: torch.Tensor
    scale: torch.Tensor
    orientation: torch.Tensor
    affine: torch.Tensor


@dataclass(frozen=True)
class DenseHeadOutput:
    confidence: torch.Tensor
    offsets: torch.Tensor


@dataclass(frozen=True)
class GraphMatcherOutput:
    logits: torch.Tensor
    matches: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class RawFeatureMaps:
    heatmap: torch.Tensor
    descriptors: torch.Tensor
    scale: torch.Tensor
    orientation: torch.Tensor
    affine: torch.Tensor
    dense_confidence: torch.Tensor


def _make_stage(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() != 4 or x.size(1) != self.input_channels:
            raise ValueError("input tensor must have shape BxCxHxW with the configured channel count")
        y1 = self.stage1(x)
        y2 = self.stage2(y1)
        y3 = self.stage3(y2)
        y4 = self.stage4(y3)
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


def normalize_channels_stable(tensor: torch.Tensor) -> torch.Tensor:
    finite = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    scale = finite.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0e-12)
    scaled = finite / scale
    return scaled / scaled.norm(p=2, dim=1, keepdim=True).clamp_min(1.0e-12)


def _normalize_channels(tensor: torch.Tensor) -> torch.Tensor:
    return normalize_channels_stable(tensor)


def _rotate_feature_map(tensor: torch.Tensor, turns: int) -> torch.Tensor:
    turns = turns % 4
    if turns == 0:
        return tensor
    return torch.rot90(tensor, turns, dims=(2, 3)).contiguous()


def _align_descriptor_orientation_channels(tensor: torch.Tensor, turns: int) -> torch.Tensor:
    channels = tensor.size(1)
    if channels < 4 or channels % 4 != 0:
        return tensor
    shift = channels // 4
    return torch.roll(tensor, shifts=-turns * shift, dims=1)


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
        self.heatmap = nn.Conv2d(input_channels, 1, 1)
        self.heatmap_viewpoint_context = nn.Conv2d(input_channels * 5, 1, 1)
        self.descriptors = _make_descriptor_tower(input_channels, descriptor_dim)
        self.descriptor_multiscale = nn.Conv2d(input_channels * 3, descriptor_dim, 1)
        self.descriptor_attention = nn.Conv2d(input_channels * 3, descriptor_dim, 1)
        self.descriptor_viewpoint_context = nn.Conv2d(input_channels * 5, descriptor_dim, 1)
        self.descriptor_viewpoint_attention = nn.Conv2d(input_channels * 5, descriptor_dim, 1)
        self.descriptor_orientation_alignment = nn.Conv2d(descriptor_dim, descriptor_dim, 1)
        self.descriptor_dilated_context = nn.Conv2d(descriptor_dim, descriptor_dim, 3, padding=2, dilation=2)
        _zero_module(self.heatmap_viewpoint_context)
        _zero_module(self.descriptor_viewpoint_context)
        _zero_module(self.descriptor_viewpoint_attention)
        _zero_module(self.descriptor_orientation_alignment)
        _zero_module(self.descriptor_dilated_context)
        self.descriptor_skip = nn.Conv2d(input_channels, descriptor_dim, 1)
        self.scale = nn.Conv2d(input_channels, 1, 1)
        self.orientation = nn.Conv2d(input_channels, 2, 1)
        self.affine = nn.Conv2d(input_channels, 4, 1)

    def _descriptor_branch(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context(feature)
        multiscale_context = make_multiscale_descriptor_context(context)
        viewpoint_descriptor = _apply_anisotropic_viewpoint_projection(self.descriptor_viewpoint_context, context)
        viewpoint_gate = 1.0 + torch.sigmoid(
            _apply_anisotropic_viewpoint_projection(self.descriptor_viewpoint_attention, context)
        )
        descriptor_base = (
            self.descriptors(context)
            + self.descriptor_multiscale(multiscale_context)
            + viewpoint_descriptor * viewpoint_gate
            + self.descriptor_skip(feature)
        )
        descriptor_gated = descriptor_base * (1.0 + torch.sigmoid(self.descriptor_attention(multiscale_context)))
        heatmap = self.heatmap(context) + _apply_anisotropic_viewpoint_projection(
            self.heatmap_viewpoint_context,
            context,
        )
        return context, heatmap, descriptor_gated

    def forward(self, feature: torch.Tensor) -> SparseHeadOutput:
        if feature.dim() != 4 or feature.size(1) != self.input_channels:
            raise ValueError("feature tensor must have shape BxCxHxW with the configured channel count")
        context, heatmap_sum, descriptor_gated = self._descriptor_branch(feature)
        descriptor_sum = (
            descriptor_gated
            + self.descriptor_orientation_alignment(descriptor_gated)
            + self.descriptor_dilated_context(descriptor_gated)
        )
        for turns in range(1, 4):
            rotated_feature = _rotate_feature_map(feature, turns)
            _, rotated_heatmap, rotated_descriptor_gated = self._descriptor_branch(rotated_feature)
            heatmap_sum = heatmap_sum + _rotate_feature_map(rotated_heatmap, -turns)
            rotated_descriptor = _rotate_feature_map(rotated_descriptor_gated, -turns)
            orientation_aligned = _align_descriptor_orientation_channels(rotated_descriptor, turns)
            descriptor_sum = (
                descriptor_sum
                + rotated_descriptor
                + self.descriptor_orientation_alignment(orientation_aligned)
                + self.descriptor_dilated_context(rotated_descriptor)
            )
        heatmap = torch.sigmoid(heatmap_sum / 4.0)
        descriptors = _normalize_channels(descriptor_sum / 4.0)
        scale = F.softplus(self.scale(context)) + 1.0e-3
        orientation = _normalize_channels(self.orientation(context))
        affine = self.affine(context)
        return SparseHeadOutput(heatmap, descriptors, scale, orientation, affine)


def make_xy_grid(height: int, width: int, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack([x.to(dtype), y.to(dtype)], dim=-1)


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


def prepare_keypoints_for_embedding(keypoints: torch.Tensor) -> torch.Tensor:
    prepared = keypoints.to(dtype=torch.float32)
    if prepared.size(0) == 0:
        return prepared
    min_xy = prepared.min(dim=0, keepdim=True).values
    max_xy = prepared.max(dim=0, keepdim=True).values
    center = (min_xy + max_xy) * 0.5
    span = (max_xy - min_xy).max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
    centered = (prepared - center) * 2.0 / span
    radius = centered.pow(2).sum(dim=1, keepdim=True).sqrt()
    return torch.cat([radius, radius.pow(2)], dim=1)


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
    def __init__(self, descriptor_dim: int, hidden_dim: int, attention_layers: int = 1) -> None:
        super().__init__()
        if descriptor_dim <= 0 or hidden_dim <= 0 or attention_layers <= 0:
            raise ValueError("descriptor_dim, hidden_dim, and attention_layers must be positive")
        self.descriptor_dim = descriptor_dim
        self.hidden_dim = hidden_dim
        self.descriptor_projection = nn.Linear(descriptor_dim, hidden_dim)
        self.keypoint_projection = nn.Linear(2, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, hidden_dim)
        self.logit_scale = nn.Parameter(torch.ones(1) * math.sqrt(float(hidden_dim)))
        self.dustbin_bias = nn.Parameter(torch.zeros(1))
        self.attention_layers = nn.ModuleList([PlanetaryGraphAttentionLayer(hidden_dim) for _ in range(attention_layers)])

    def forward(
        self,
        descriptors_a: torch.Tensor,
        keypoints_a: torch.Tensor,
        descriptors_b: torch.Tensor,
        keypoints_b: torch.Tensor,
    ) -> GraphMatcherOutput:
        if descriptors_a.dim() != 2 or descriptors_b.dim() != 2:
            raise ValueError("graph matcher descriptors must have shape NxD")
        if keypoints_a.dim() != 2 or keypoints_b.dim() != 2 or keypoints_a.size(1) != 2 or keypoints_b.size(1) != 2:
            raise ValueError("graph matcher keypoints must have shape Nx2")
        if descriptors_a.size(0) != keypoints_a.size(0) or descriptors_b.size(0) != keypoints_b.size(0):
            raise ValueError("graph matcher descriptor and keypoint counts must match")
        desc_a = descriptors_a.to(dtype=torch.float32)
        desc_b = descriptors_b.to(dtype=torch.float32)
        kp_a = prepare_keypoints_for_embedding(keypoints_a).to(device=desc_a.device)
        kp_b = prepare_keypoints_for_embedding(keypoints_b).to(device=desc_b.device)
        embed_a = torch.relu(self.descriptor_projection(desc_a) + self.keypoint_projection(kp_a))
        embed_b = torch.relu(self.descriptor_projection(desc_b) + self.keypoint_projection(kp_b))
        for layer in self.attention_layers:
            embed_a, embed_b = layer(embed_a, embed_b)
        embed_a = F.normalize(self.score_projection(embed_a), p=2, dim=1)
        embed_b = F.normalize(self.score_projection(embed_b), p=2, dim=1)
        pair_logits = (embed_a @ embed_b.transpose(0, 1)) * self.logit_scale.clamp(1.0, 100.0)
        logits = torch.zeros(
            descriptors_a.size(0) + 1,
            descriptors_b.size(0) + 1,
            dtype=pair_logits.dtype,
            device=pair_logits.device,
        ) + self.dustbin_bias
        logits[: descriptors_a.size(0), : descriptors_b.size(0)] = pair_logits
        row_logits = logits[: descriptors_a.size(0), :]
        best_values, best_indices = row_logits.max(dim=1)
        source_indices = torch.arange(descriptors_a.size(0), device=best_indices.device)
        inlier_mask = best_indices.lt(descriptors_b.size(0))
        if descriptors_a.size(0) > 0 and descriptors_b.size(0) > 0:
            reverse_best = pair_logits.max(dim=0).indices
            mutual_sources = reverse_best.index_select(0, best_indices.clamp(0, descriptors_b.size(0) - 1))
            inlier_mask = inlier_mask & mutual_sources.eq(source_indices)
        source_indices = source_indices[inlier_mask]
        target_indices = best_indices[inlier_mask]
        probabilities = torch.softmax(row_logits, dim=1).max(dim=1).values[inlier_mask]
        matches = torch.stack([source_indices, target_indices], dim=1).to(device="cpu", dtype=torch.long).contiguous()
        scores = probabilities.to(device="cpu", dtype=torch.float32).contiguous()
        return GraphMatcherOutput(logits.contiguous(), matches, scores)


def make_rotation_invariant_texture_descriptor(
    image: torch.Tensor,
    descriptor_height: int,
    descriptor_width: int,
    descriptor_dim: int,
) -> torch.Tensor:
    base = image
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
    for kernel in (3, 7, 15, 31):
        blur = F.avg_pool2d(base, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        channels.extend([blur, (base - blur).abs()])
    dx = (base - torch.roll(base, shifts=1, dims=3)).abs()
    dy = (base - torch.roll(base, shifts=1, dims=2)).abs()
    gradient = dx + dy
    for kernel in (3, 7, 11):
        channels.append(F.avg_pool2d(gradient, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False))
    for ring_radius in (1, 2, 4, 8):
        diffs = []
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
            diffs.append((base - torch.roll(base, shifts=(dy_offset, dx_offset), dims=(2, 3))).abs())
        ring = torch.stack(diffs, dim=1)
        ring_mean = ring.mean(dim=1)
        channels.append(ring_mean)
        channels.append(ring.max(dim=1).values)
        centered_ring = ring - ring.mean(dim=1, keepdim=True)
        channels.append(centered_ring.pow(2).mean(dim=1).sqrt())
        channels.append(ring_mean * radius)
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


class PlanetaryFeatureMatcher(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 1,
        base_channels: int = 48,
        descriptor_dim: int = 192,
        graph_hidden_dim: int = 384,
        graph_attention_layers: int = 8,
    ) -> None:
        super().__init__()
        self.config = CheckpointConfig(input_channels, base_channels, descriptor_dim, graph_hidden_dim, graph_attention_layers)
        self.backbone = Backbone(input_channels, base_channels)
        self.sparse_head = SparseHead(base_channels * 2, descriptor_dim)
        self.dense_head = DenseHead(base_channels)
        self.graph_matcher = PlanetaryGraphMatcher(descriptor_dim, graph_hidden_dim, graph_attention_layers)

    def learned_descriptor_map_single(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("image must have shape BxCxHxW")
        features = self.backbone(image)
        sparse = self.sparse_head(features[1])
        return sparse.descriptors

    def texture_descriptor_map_single(self, image: torch.Tensor) -> torch.Tensor:
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

    def descriptor_map_single(
        self,
        image: torch.Tensor,
        *,
        texture_blend_weight: float = INFERENCE_TEXTURE_BLEND_WEIGHT,
    ) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("image must have shape BxCxHxW")
        features = self.backbone(image)
        sparse = self.sparse_head(features[1])
        return blend_rotation_invariant_texture_descriptor(sparse.descriptors, image, texture_blend_weight)

    def forward_single(
        self,
        image: torch.Tensor,
        *,
        texture_blend_weight: float = INFERENCE_TEXTURE_BLEND_WEIGHT,
    ) -> RawFeatureMaps:
        if image.dim() != 4:
            raise ValueError("image must have shape BxCxHxW")
        features = self.backbone(image)
        sparse = self.sparse_head(features[1])
        descriptors = blend_rotation_invariant_texture_descriptor(sparse.descriptors, image, texture_blend_weight)
        heatmap = make_rotation_invariant_texture_saliency(image, sparse.heatmap.size(2), sparse.heatmap.size(3))
        dense = self.dense_head(features[0], features[0])
        dense_confidence = F.interpolate(dense.confidence, size=sparse.heatmap.shape[-2:], mode="nearest")
        return RawFeatureMaps(heatmap, descriptors, sparse.scale, sparse.orientation, sparse.affine, dense_confidence)


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
    )


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
    ).to(device)
    model_state = {key: value for key, value in raw_state.items() if not key.startswith("config.")}
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
    )
    model = PlanetaryFeatureMatcher(
        input_channels=config.input_channels,
        base_channels=config.base_channels,
        descriptor_dim=config.descriptor_dim,
        graph_hidden_dim=config.graph_hidden_dim,
        graph_attention_layers=config.graph_attention_layers,
    ).to(device)
    result = model.load_state_dict(payload["model"], strict=strict)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(f"pytorch state load mismatch: {result}")
    model.eval()
    return model, config
