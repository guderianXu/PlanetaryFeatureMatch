#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace pfm
{

class ImageDataset
{
  public:
    /// 从目录直属图像文件构建数据集。
    ///
    /// 支持 .png、.jpg、.jpeg、.tif 和 .tiff，扩展名大小写不敏感。
    /// 路径按字典序排序，保证训练和测试遍历顺序可复现。
    ///
    /// @param image_dir 图像文件所在目录。
    /// @throws std::invalid_argument 当 image_dir 不是目录或没有支持的图像文件时抛出。
    explicit ImageDataset(const std::string& image_dir);

    /// 返回数据集中支持的图像文件数量。
    ///
    /// @return 从目录收集到的图像路径数量。
    std::size_t size() const;

    /// 返回排序后的第 index 个图像路径。
    ///
    /// @param index 从 0 开始的数据集索引。
    /// @return 内部保存的图像路径字符串引用。
    /// @throws std::out_of_range 当 index 不小于 size() 时抛出。
    const std::string& path(std::size_t index) const;

    /// 读取单张图像并转换为模型输入张量。
    ///
    /// @param index 从 0 开始的数据集索引。
    /// @return 连续内存的 C x H x W float32 图像张量，值域归一化到 [0, 1]。
    /// @throws std::out_of_range 当 index 不小于 size() 时抛出。
    /// @throws std::invalid_argument 当图像无法加载或格式不支持时抛出。
    torch::Tensor load(std::size_t index) const;

  private:
    std::vector<std::string> _paths;
};

} // namespace pfm
