#include "data/image_dataset.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <cctype>

#include "image/image_io.h"

namespace pfm
{

namespace
{

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch)
                   {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

bool is_supported_extension(const std::filesystem::path& path)
{
    // 数据集扫描只根据扩展名做快速过滤，真正的格式校验留给 image 模块读取阶段。
    const std::string extension = to_lower(path.extension().string());
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".tif" ||
           extension == ".tiff";
}

} // namespace

ImageDataset::ImageDataset(const std::string& image_dir)
{
    const std::filesystem::path dir_path(image_dir);
    if (!std::filesystem::is_directory(dir_path))
    {
        throw std::invalid_argument("image dataset directory does not exist: " + image_dir);
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir_path))
    {
        if (entry.is_regular_file() && is_supported_extension(entry.path()))
        {
            _paths.push_back(entry.path().string());
        }
    }
    // 排序是训练可复现性的基础，避免不同文件系统枚举顺序影响 split 和采样。
    std::sort(_paths.begin(), _paths.end());

    if (_paths.empty())
    {
        throw std::invalid_argument("image dataset directory contains no supported images: " + image_dir);
    }
}

std::size_t ImageDataset::size() const
{
    return _paths.size();
}

const std::string& ImageDataset::path(std::size_t index) const
{
    if (index >= _paths.size())
    {
        throw std::out_of_range("image dataset index out of range");
    }
    return _paths[index];
}

torch::Tensor ImageDataset::load(std::size_t index) const
{
    return load_image_tensor(path(index));
}

} // namespace pfm
