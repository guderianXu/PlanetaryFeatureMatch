#pragma once

#include <string>

#include <torch/torch.h>

namespace pfm
{

/// 解析训练或推理配置中的计算设备字符串，例如 cpu、cuda 或 cuda:N。
/// @param requested 来自 CLI 或配置文件的设备字符串。
/// @return LibTorch 可直接使用的设备对象。
/// @throws std::invalid_argument 当字符串非法或请求的 CUDA 设备不可用时抛出。
torch::Device resolve_compute_device(const std::string& requested);

} // namespace pfm
