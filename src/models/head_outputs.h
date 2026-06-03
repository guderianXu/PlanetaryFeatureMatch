#pragma once

#include <torch/torch.h>

namespace pfm
{

struct SparseHeadOutput
{
    /// 稀疏关键点热力图。
    torch::Tensor heatmap;
    /// 稀疏描述子特征图。
    torch::Tensor descriptors;
    /// 关键点尺度预测图。
    torch::Tensor scale;
    /// 关键点方向预测图。
    torch::Tensor orientation;
    /// 关键点局部仿射参数预测图。
    torch::Tensor affine;
};

struct DenseHeadOutput
{
    /// 稠密匹配置信度图。
    torch::Tensor confidence;
    /// 稠密匹配 x/y 偏移图。
    torch::Tensor offsets;
};

} // namespace pfm
