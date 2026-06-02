#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

namespace pfm
{

struct AffineTransform
{
    std::array<float, 6> matrix;

    /// 构造单位仿射变换。
    static AffineTransform identity();

    /// 构造二维平移变换。
    static AffineTransform translation(float tx, float ty);

    /// 构造以 (cx, cy) 为中心的缩放与旋转变换。
    static AffineTransform scale_rotate(float scale, float radians, float cx, float cy);
};

struct ProjectiveTransform
{
    std::array<float, 9> matrix;

    /// 构造单位单应/投影变换。
    static ProjectiveTransform identity();

    /// 将 2x3 仿射矩阵提升为 3x3 投影矩阵。
    static ProjectiveTransform from_affine(const AffineTransform& transform);

    /// 返回投影变换逆矩阵。
    /// @throws std::invalid_argument 当矩阵奇异不可逆时抛出。
    ProjectiveTransform inverse() const;
};

/// 生成从源图像像素到目标图像像素的稠密仿射 warp field。
torch::Tensor dense_warp_field(int64_t height, int64_t width, const AffineTransform& transform, torch::Device device);

/// 生成从源图像像素到目标图像像素的稠密投影 warp field。
torch::Tensor dense_warp_field(int64_t height, int64_t width, const ProjectiveTransform& transform,
                               torch::Device device);

/// 根据 warp field 判断目标坐标是否落在目标图像范围内。
torch::Tensor valid_warp_mask(const torch::Tensor& field, int64_t target_height, int64_t target_width);

/// 对 Nx2 点集应用仿射变换。
torch::Tensor warp_points(const torch::Tensor& points, const AffineTransform& transform);

/// 对 Nx2 点集应用投影变换，并执行齐次坐标归一化。
torch::Tensor warp_points(const torch::Tensor& points, const ProjectiveTransform& transform);

} // namespace pfm
