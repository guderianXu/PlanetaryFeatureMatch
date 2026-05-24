#pragma once

#include <array>
#include <cstdint>
#include <torch/torch.h>

namespace pfm {

struct AffineTransform {
    std::array<float, 6> matrix;

    static AffineTransform identity();
    static AffineTransform translation(float tx, float ty);
    static AffineTransform scale_rotate(float scale, float radians, float cx, float cy);
};

struct ProjectiveTransform {
    std::array<float, 9> matrix;

    static ProjectiveTransform identity();
    static ProjectiveTransform from_affine(const AffineTransform& transform);
    ProjectiveTransform inverse() const;
};

torch::Tensor dense_warp_field(int64_t height, int64_t width, const AffineTransform& transform, torch::Device device);
torch::Tensor dense_warp_field(int64_t height, int64_t width, const ProjectiveTransform& transform, torch::Device device);
torch::Tensor valid_warp_mask(const torch::Tensor& field, int64_t target_height, int64_t target_width);
torch::Tensor warp_points(const torch::Tensor& points, const AffineTransform& transform);
torch::Tensor warp_points(const torch::Tensor& points, const ProjectiveTransform& transform);

}  // namespace pfm
