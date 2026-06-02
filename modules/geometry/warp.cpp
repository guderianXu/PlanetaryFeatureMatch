#include "geometry/warp.h"

#include <cmath>
#include <stdexcept>

#include "core/tensor_utils.h"

namespace pfm
{

namespace
{

void require_point_tensor(const torch::Tensor& points)
{
    if (!points.defined())
    {
        throw std::invalid_argument("points tensor is undefined");
    }
    if (points.dim() != 2 || points.size(1) != 2)
    {
        throw std::invalid_argument("points tensor must have shape Nx2");
    }
    if (points.scalar_type() != torch::kFloat32)
    {
        throw std::invalid_argument("points tensor must be float32");
    }
}

void require_warp_field(const torch::Tensor& field)
{
    if (!field.defined())
    {
        throw std::invalid_argument("warp field tensor is undefined");
    }
    if (field.dim() != 3 || field.size(2) != 2)
    {
        throw std::invalid_argument("warp field tensor must have shape HxWx2");
    }
    if (field.scalar_type() != torch::kFloat32)
    {
        throw std::invalid_argument("warp field tensor must be float32");
    }
}

} // namespace

AffineTransform AffineTransform::identity()
{
    return AffineTransform{{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F}};
}

AffineTransform AffineTransform::translation(float tx, float ty)
{
    return AffineTransform{{1.0F, 0.0F, tx, 0.0F, 1.0F, ty}};
}

AffineTransform AffineTransform::scale_rotate(float scale, float radians, float cx, float cy)
{
    const float c = std::cos(radians) * scale;
    const float s = std::sin(radians) * scale;
    return AffineTransform{{
        c,
        -s,
        cx - c * cx + s * cy,
        s,
        c,
        cy - s * cx - c * cy,
    }};
}

ProjectiveTransform ProjectiveTransform::identity()
{
    return ProjectiveTransform{{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F}};
}

ProjectiveTransform ProjectiveTransform::from_affine(const AffineTransform& transform)
{
    return ProjectiveTransform{{
        transform.matrix[0],
        transform.matrix[1],
        transform.matrix[2],
        transform.matrix[3],
        transform.matrix[4],
        transform.matrix[5],
        0.0F,
        0.0F,
        1.0F,
    }};
}

ProjectiveTransform ProjectiveTransform::inverse() const
{
    const auto& m = matrix;
    const float c00 = m[4] * m[8] - m[5] * m[7];
    const float c01 = -(m[3] * m[8] - m[5] * m[6]);
    const float c02 = m[3] * m[7] - m[4] * m[6];
    const float c10 = -(m[1] * m[8] - m[2] * m[7]);
    const float c11 = m[0] * m[8] - m[2] * m[6];
    const float c12 = -(m[0] * m[7] - m[1] * m[6]);
    const float c20 = m[1] * m[5] - m[2] * m[4];
    const float c21 = -(m[0] * m[5] - m[2] * m[3]);
    const float c22 = m[0] * m[4] - m[1] * m[3];
    const float determinant = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if (std::abs(determinant) < 1.0e-8F)
    {
        throw std::invalid_argument("projective transform is singular");
    }
    const float inv_det = 1.0F / determinant;
    return ProjectiveTransform{{
        c00 * inv_det,
        c10 * inv_det,
        c20 * inv_det,
        c01 * inv_det,
        c11 * inv_det,
        c21 * inv_det,
        c02 * inv_det,
        c12 * inv_det,
        c22 * inv_det,
    }};
}

torch::Tensor dense_warp_field(int64_t h, int64_t w, const AffineTransform& transform, torch::Device device)
{
    auto grid = make_xy_grid(h, w, device);
    return warp_points(grid.reshape({h * w, 2}), transform).reshape({h, w, 2});
}

torch::Tensor dense_warp_field(int64_t h, int64_t w, const ProjectiveTransform& transform, torch::Device device)
{
    auto grid = make_xy_grid(h, w, device);
    return warp_points(grid.reshape({h * w, 2}), transform).reshape({h, w, 2});
}

torch::Tensor valid_warp_mask(const torch::Tensor& field, int64_t target_height, int64_t target_width)
{
    require_warp_field(field);
    if (target_height <= 0 || target_width <= 0)
    {
        throw std::invalid_argument("target dimensions must be positive");
    }

    using torch::indexing::Slice;
    auto x = field.index({Slice(), Slice(), 0});
    auto y = field.index({Slice(), Slice(), 1});
    return (x >= 0.0F) & (x <= static_cast<float>(target_width - 1)) & (y >= 0.0F) &
           (y <= static_cast<float>(target_height - 1));
}

torch::Tensor warp_points(const torch::Tensor& points, const AffineTransform& transform)
{
    require_point_tensor(points);

    using torch::indexing::Slice;
    auto x = points.index({Slice(), 0});
    auto y = points.index({Slice(), 1});
    auto warped_x = x * transform.matrix[0] + y * transform.matrix[1] + transform.matrix[2];
    auto warped_y = x * transform.matrix[3] + y * transform.matrix[4] + transform.matrix[5];
    return torch::stack({warped_x, warped_y}, 1);
}

torch::Tensor warp_points(const torch::Tensor& points, const ProjectiveTransform& transform)
{
    require_point_tensor(points);

    using torch::indexing::Slice;
    const auto& m = transform.matrix;
    auto x = points.index({Slice(), 0});
    auto y = points.index({Slice(), 1});
    auto denominator = x * m[6] + y * m[7] + m[8];
    auto warped_x = (x * m[0] + y * m[1] + m[2]) / denominator;
    auto warped_y = (x * m[3] + y * m[4] + m[5]) / denominator;
    return torch::stack({warped_x, warped_y}, 1);
}

} // namespace pfm
