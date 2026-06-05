"""训练用光照扰动和局部对比预处理。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from patch_descriptor_training import SyntheticPair


@dataclass(frozen=True)
class PhotometricAugmentConfig:
    enabled: bool = False
    probability: float = 1.0
    brightness: float = 0.0
    contrast: float = 0.0
    gamma: float = 0.0
    shadow: float = 0.0
    noise: float = 0.0


def rand_uniform(generator: torch.Generator, low: float, high: float) -> float:
    if high <= low:
        return float(low)
    value = torch.rand((), generator=generator, dtype=torch.float32).item()
    return float(low + (high - low) * value)


def make_shadow_map(image: torch.Tensor, config: PhotometricAugmentConfig, generator: torch.Generator) -> torch.Tensor:
    _, height, width = image.shape
    device = image.device
    dtype = image.dtype
    strength = rand_uniform(generator, 0.0, max(0.0, float(config.shadow)))
    if strength <= 0.0:
        return image.new_ones((1, height, width))

    if rand_uniform(generator, 0.0, 1.0) < 0.5:
        axis = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device).view(1, 1, width)
    else:
        axis = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device).view(1, height, 1)

    if rand_uniform(generator, 0.0, 1.0) < 0.5:
        axis = 1.0 - axis
    side_profile = axis.expand(1, height, width)

    center = rand_uniform(generator, 0.2, 0.8)
    band_width = rand_uniform(generator, 0.18, 0.45)
    band_profile = (1.0 - (axis - center).abs() / max(band_width, 1.0e-3)).clamp(0.0, 1.0).expand(
        1,
        height,
        width,
    )
    mix = rand_uniform(generator, 0.0, 1.0)
    profile = (mix * side_profile + (1.0 - mix) * band_profile).clamp(0.0, 1.0)
    return (1.0 - strength * profile).clamp(1.0 - strength, 1.0)


def augment_single_view(
    image: torch.Tensor,
    config: PhotometricAugmentConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    augmented = image.to(torch.float32).clamp(0.0, 1.0)
    if rand_uniform(generator, 0.0, 1.0) > max(0.0, min(1.0, float(config.probability))):
        return augmented

    contrast = max(0.0, float(config.contrast))
    if contrast > 0.0:
        contrast_scale = rand_uniform(generator, max(0.05, 1.0 - contrast), 1.0 + contrast)
        mean = augmented.mean(dim=(-2, -1), keepdim=True)
        augmented = (augmented - mean) * contrast_scale + mean

    gamma = max(0.0, float(config.gamma))
    if gamma > 0.0:
        # gamma 用指数采样，暗化和亮化的比例更对称。
        gamma_value = math.exp(rand_uniform(generator, -gamma, gamma))
        augmented = augmented.clamp(0.0, 1.0).pow(gamma_value)

    brightness = max(0.0, float(config.brightness))
    if brightness > 0.0:
        augmented = augmented + rand_uniform(generator, -brightness, brightness)

    if config.shadow > 0.0:
        augmented = augmented * make_shadow_map(augmented, config, generator)

    noise = max(0.0, float(config.noise))
    if noise > 0.0:
        augmented = augmented + torch.randn(
            augmented.shape,
            generator=generator,
            dtype=augmented.dtype,
            device=augmented.device,
        ) * rand_uniform(generator, 0.0, noise)

    return augmented.nan_to_num(0.0, 0.0, 0.0).clamp(0.0, 1.0).contiguous()


def apply_photometric_augmentation(
    pair: SyntheticPair,
    config: PhotometricAugmentConfig,
    *,
    seed: int,
) -> SyntheticPair:
    if not config.enabled:
        return pair
    generator_a = torch.Generator(device=pair.view_a.device)
    generator_b = torch.Generator(device=pair.view_b.device)
    generator_a.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    generator_b.manual_seed((int(seed) + 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF)
    return SyntheticPair(
        view_a=augment_single_view(pair.view_a, config, generator_a),
        view_b=augment_single_view(pair.view_b, config, generator_b),
        warp_a_to_b=pair.warp_a_to_b,
        valid_mask=pair.valid_mask,
    )


def make_illumination_consistency_pair(
    pair: SyntheticPair,
    config: PhotometricAugmentConfig,
    *,
    seed: int,
) -> SyntheticPair:
    if not config.enabled:
        return pair
    return apply_photometric_augmentation(pair, config, seed=seed)


def make_illumination_match_pair(
    pair: SyntheticPair,
    config: PhotometricAugmentConfig,
    *,
    seed: int,
    changed_view: str = "b",
) -> SyntheticPair:
    """生成同一几何关系下的单侧光照扰动匹配训练样本。"""
    if not config.enabled:
        return pair
    if changed_view not in {"a", "b", "both"}:
        raise ValueError("changed_view must be one of: a, b, both")

    generator_a = torch.Generator(device=pair.view_a.device)
    generator_b = torch.Generator(device=pair.view_b.device)
    generator_a.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    generator_b.manual_seed((int(seed) + 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF)
    view_a = pair.view_a
    view_b = pair.view_b
    if changed_view in {"a", "both"}:
        view_a = augment_single_view(pair.view_a, config, generator_a)
    if changed_view in {"b", "both"}:
        view_b = augment_single_view(pair.view_b, config, generator_b)
    return SyntheticPair(
        view_a=view_a.contiguous(),
        view_b=view_b.contiguous(),
        warp_a_to_b=pair.warp_a_to_b,
        valid_mask=pair.valid_mask,
    )


def local_contrast_single_view(view: torch.Tensor, *, strength: float, kernel_size: int) -> torch.Tensor:
    if strength <= 0.0:
        return view
    kernel = max(3, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    image = view.to(torch.float32).clamp(0.0, 1.0)
    batch = image.unsqueeze(0)
    mean = F.avg_pool2d(batch, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    high_pass = torch.clamp((batch - mean) * 0.75 + 0.5, 0.0, 1.0)
    normalized = (1.0 - float(strength)) * batch + float(strength) * high_pass
    return normalized.squeeze(0).clamp(0.0, 1.0).contiguous()


def apply_local_contrast_normalization(
    pair: SyntheticPair,
    *,
    strength: float,
    kernel_size: int = 31,
) -> SyntheticPair:
    if strength <= 0.0:
        return pair
    return SyntheticPair(
        view_a=local_contrast_single_view(pair.view_a, strength=strength, kernel_size=kernel_size),
        view_b=local_contrast_single_view(pair.view_b, strength=strength, kernel_size=kernel_size),
        warp_a_to_b=pair.warp_a_to_b,
        valid_mask=pair.valid_mask,
    )


def apply_training_transforms(
    pair: SyntheticPair,
    *,
    photometric_config: PhotometricAugmentConfig,
    seed: int,
    input_local_contrast: bool,
    local_contrast_strength: float,
    local_contrast_kernel: int,
) -> SyntheticPair:
    transformed = apply_photometric_augmentation(pair, photometric_config, seed=seed)
    if input_local_contrast:
        transformed = apply_local_contrast_normalization(
            transformed,
            strength=local_contrast_strength,
            kernel_size=local_contrast_kernel,
        )
    return transformed
