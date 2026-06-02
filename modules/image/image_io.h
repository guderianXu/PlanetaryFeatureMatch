#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm
{

/// 将图像文件读取为连续的 C x H x W float32 张量，并归一化到 [0, 1]。
///
/// 支持 8-bit/16-bit 无符号灰度图、BGR 三通道图和 BGRA 四通道图。
/// OpenCV 读取彩色图时默认是 BGR/BGRA，这里统一转换为 RGB，避免训练和推理颜色顺序漂移。
///
/// @param path 图像文件路径。
/// @return 连续内存张量，shape 为 {C, H, W}，dtype 为 float32，值域为 [0, 1]。
/// @throws std::invalid_argument 当图像无法加载、通道数不支持或位深不支持时抛出。
torch::Tensor load_image_tensor(const std::string& path);

} // namespace pfm
