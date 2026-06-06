"""PFM 描述子几何处理工具。

这个模块只放和 dense descriptor 后处理有关的纯张量函数：
- 通道归一化，避免 NaN/Inf 或极小范数把训练打崩。
- 构造像素坐标网格，供 dense head 和 descriptor pooling 共用。
- 按模型预测的方向、尺度和局部仿射矩阵做 canonical pooling。

原来这些函数混在 `pfm_model.py` 里，导致主模型文件过长，也不方便单独测试。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def normalize_channels_stable(tensor: torch.Tensor) -> torch.Tensor:
    """对通道维做稳定 L2 归一化。

    这里先用通道最大绝对值缩放，再做 L2 normalize。目的不是改变特征含义，
    而是避免少量异常值或全零 descriptor 让后面的相似度、loss 产生非有限值。
    """

    finite = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    scale = finite.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    scaled = finite / scale
    return scaled / scaled.norm(p=2, dim=1, keepdim=True).clamp_min(1.0e-3)


def make_xy_grid(height: int, width: int, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """生成 HxWx2 的像素坐标网格，最后一维为 x/y。"""

    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack([x.to(dtype), y.to(dtype)], dim=-1)


def geometry_aware_descriptor_pool(
    descriptors: torch.Tensor,
    orientation: torch.Tensor,
    scale: torch.Tensor,
    affine: torch.Tensor,
    *,
    radius: float = 0.75,
) -> torch.Tensor:
    """用预测局部几何在 canonical 邻域内聚合 descriptor。

    这个函数替代旧的 C4 旋转分支。旧方法只枚举 0/90/180/270 四个离散方向，
    对真实轨道影像里的任意旋转、斜视和局部尺度变化适应性弱。现在改为：

    1. `orientation` 给出每个像素的主方向，形成切向 tangent。
    2. tangent 旋转 90 度得到法向 normal。
    3. `scale` 控制采样半径，`affine` 控制局部椭圆/剪切形变。
    4. 在中心、切向、法向和对角方向采样，再加权融合。

    输出仍保持 BxDxHxW，便于 C++/Python 两边继续共用同一模型契约。
    """

    if descriptors.dim() != 4:
        raise ValueError("descriptors must have shape BxDxHxW")
    if orientation.shape[:2] != (descriptors.size(0), 2) or orientation.shape[-2:] != descriptors.shape[-2:]:
        raise ValueError("orientation must have shape Bx2xHxW matching descriptors")
    if scale.shape[:2] != (descriptors.size(0), 1) or scale.shape[-2:] != descriptors.shape[-2:]:
        raise ValueError("scale must have shape Bx1xHxW matching descriptors")
    if affine.shape[:2] != (descriptors.size(0), 4) or affine.shape[-2:] != descriptors.shape[-2:]:
        raise ValueError("affine must have shape Bx4xHxW matching descriptors")
    batch, _, height, width = descriptors.shape
    if height <= 1 or width <= 1:
        return normalize_channels_stable(descriptors)

    # base_xy 是每个 dense descriptor 所在的原始像素坐标。后续所有采样偏移都加在这个坐标上。
    base_xy = make_xy_grid(height, width, device=descriptors.device, dtype=descriptors.dtype)
    base_xy = base_xy.permute(2, 0, 1).unsqueeze(0).expand(batch, 2, height, width)

    # orientation 是网络预测出来的主方向。归一化后直接作为局部坐标系的切向轴。
    ori = F.normalize(orientation.to(descriptors.dtype), p=2, dim=1, eps=1.0e-6)
    tangent = ori
    normal = torch.stack([-ori[:, 1], ori[:, 0]], dim=1)

    # 尺度限制在一个保守范围，避免训练早期 scale 发散后采样到过远区域。
    clamped_scale = scale.to(descriptors.dtype).clamp(0.5, 2.0)
    a00, a01, a10, a11 = [affine[:, index : index + 1].to(descriptors.dtype) for index in range(4)]

    def sample(offset_x: torch.Tensor, offset_y: torch.Tensor) -> torch.Tensor:
        """把局部 offset 经 affine 变换后，用 grid_sample 读取 descriptor。"""

        warped_x = a00 * offset_x + a01 * offset_y
        warped_y = a10 * offset_x + a11 * offset_y
        sample_xy = base_xy + torch.cat([warped_x, warped_y], dim=1)

        # grid_sample 需要 [-1, 1] 归一化坐标；padding 用 border，避免边缘点直接变成 0。
        grid_x = sample_xy[:, 0] * (2.0 / float(max(1, width - 1))) - 1.0
        grid_y = sample_xy[:, 1] * (2.0 / float(max(1, height - 1))) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return F.grid_sample(descriptors, grid, mode="bilinear", padding_mode="border", align_corners=True)

    step = clamped_scale * float(radius)
    diagonal_step = step * 0.70710678118
    zero = torch.zeros_like(step)
    tangent_x = tangent[:, 0:1]
    tangent_y = tangent[:, 1:2]
    normal_x = normal[:, 0:1]
    normal_y = normal[:, 1:2]

    # 权重按“中心最高、切向次之、法向和对角补充”的顺序设计。
    # 这样能保留原始 descriptor 的局部判别性，同时让方向和尺度预测真实参与训练。
    offsets = [
        (zero, zero, 0.30),
        (tangent_x * step, tangent_y * step, 0.16),
        (-tangent_x * step, -tangent_y * step, 0.16),
        (normal_x * step, normal_y * step, 0.09),
        (-normal_x * step, -normal_y * step, 0.09),
        ((tangent_x + normal_x) * diagonal_step, (tangent_y + normal_y) * diagonal_step, 0.05),
        (-(tangent_x + normal_x) * diagonal_step, -(tangent_y + normal_y) * diagonal_step, 0.05),
        ((tangent_x - normal_x) * diagonal_step, (tangent_y - normal_y) * diagonal_step, 0.05),
        (-(tangent_x - normal_x) * diagonal_step, -(tangent_y - normal_y) * diagonal_step, 0.05),
    ]
    pooled = sum(sample(dx, dy) * weight for dx, dy, weight in offsets)
    return normalize_channels_stable(0.45 * descriptors + 0.55 * pooled)
