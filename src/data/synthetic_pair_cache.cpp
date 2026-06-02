#include "data/synthetic_pair_cache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/nn/functional/upsampling.h>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "core/tensor_utils.h"

namespace pfm
{
namespace
{

constexpr int64_t CACHE_FORMAT_VERSION = 1;
constexpr int64_t INPUT_CHANNELS = 1;

std::filesystem::path cache_path(const std::string& cache_dir)
{
    return std::filesystem::path(cache_dir);
}

std::string pair_stem(std::size_t index)
{
    std::ostringstream stream;
    stream << "pair_" << std::setw(6) << std::setfill('0') << index;
    return stream.str();
}

std::string source_dir_name(std::size_t source_index, const std::string& source_stem)
{
    std::ostringstream stream;
    stream << "source_" << std::setw(6) << std::setfill('0') << source_index << "_" << source_stem;
    return stream.str();
}

std::string source_dir_prefix(std::size_t source_index)
{
    std::ostringstream stream;
    stream << "source_" << std::setw(6) << std::setfill('0') << source_index << "_";
    return stream.str();
}

std::filesystem::path source_dir_path(const std::string& cache_dir, const ImageDataset& dataset,
                                      std::size_t source_index)
{
    const auto stem = std::filesystem::path(dataset.path(source_index)).stem().string();
    return cache_path(cache_dir) / source_dir_name(source_index, stem);
}

std::filesystem::path find_source_dir_path(const std::string& cache_dir, std::size_t source_index)
{
    const auto prefix = source_dir_prefix(source_index);
    const auto root = cache_path(cache_dir);
    if (!std::filesystem::exists(root))
    {
        return root / ("source_" + std::to_string(source_index));
    }
    for (const auto& entry : std::filesystem::directory_iterator(root))
    {
        if (!entry.is_directory())
        {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0)
        {
            return entry.path();
        }
    }
    return root / ("source_" + std::to_string(source_index));
}

std::filesystem::path pair_pt_path(const std::string& cache_dir, std::size_t index, std::size_t source_count)
{
    const auto source_index = index % std::max<std::size_t>(1, source_count);
    return find_source_dir_path(cache_dir, source_index) / (pair_stem(index) + ".pt");
}

std::filesystem::path pair_pt_path(const std::string& cache_dir, const ImageDataset& dataset, std::size_t index)
{
    const auto source_index = index % dataset.size();
    return source_dir_path(cache_dir, dataset, source_index) / (pair_stem(index) + ".pt");
}

std::filesystem::path pair_view_path(const std::string& cache_dir, const ImageDataset& dataset, std::size_t index,
                                     const char* view_name)
{
    const auto source_index = index % dataset.size();
    return source_dir_path(cache_dir, dataset, source_index) / (pair_stem(index) + "_" + view_name + ".png");
}

std::string angle_suffix(float degrees)
{
    auto rounded = static_cast<int64_t>(std::llround(std::fmod(degrees, 360.0F)));
    if (rounded < 0)
    {
        rounded += 360;
    }
    std::ostringstream stream;
    stream << std::setw(3) << std::setfill('0') << rounded;
    return stream.str();
}

std::filesystem::path named_rotation_image_path(const std::string& cache_dir, const ImageDataset& dataset,
                                                std::size_t source_index, float degrees)
{
    const auto stem = std::filesystem::path(dataset.path(source_index)).stem().string();
    return source_dir_path(cache_dir, dataset, source_index) / (stem + "_" + angle_suffix(degrees) + ".tif");
}

std::filesystem::path named_rotation_pair_path(const std::string& cache_dir, const ImageDataset& dataset,
                                               std::size_t source_index, float degrees)
{
    const auto stem = std::filesystem::path(dataset.path(source_index)).stem().string();
    return source_dir_path(cache_dir, dataset, source_index) / (stem + "_" + angle_suffix(degrees) + ".pt");
}

std::filesystem::path source_view_a_path(const std::string& cache_dir, const ImageDataset& dataset,
                                         std::size_t source_index)
{
    std::ostringstream stream;
    stream << "source_" << std::setw(6) << std::setfill('0') << source_index << "_view_a.png";
    return source_dir_path(cache_dir, dataset, source_index) / stream.str();
}

torch::Tensor read_tensor(torch::serialize::InputArchive& archive, const char* name)
{
    torch::Tensor tensor;
    archive.read(name, tensor);
    if (!tensor.defined())
    {
        throw std::invalid_argument(std::string("synthetic pair cache missing ") + name);
    }
    return tensor;
}

int64_t read_int64(torch::serialize::InputArchive& archive, const char* name)
{
    auto tensor = read_tensor(archive, name);
    if (tensor.numel() != 1)
    {
        throw std::invalid_argument(std::string("synthetic pair cache invalid ") + name);
    }
    return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

float read_float(torch::serialize::InputArchive& archive, const char* name)
{
    auto tensor = read_tensor(archive, name);
    if (tensor.numel() != 1)
    {
        throw std::invalid_argument(std::string("synthetic pair cache invalid ") + name);
    }
    return tensor.to(torch::kCPU, torch::kFloat32).reshape({1}).item<float>();
}

void write_int64(torch::serialize::OutputArchive& archive, const char* name, int64_t value)
{
    archive.write(name, torch::tensor({value}, torch::kInt64));
}

void write_float(torch::serialize::OutputArchive& archive, const char* name, float value)
{
    archive.write(name, torch::tensor({value}, torch::kFloat32));
}

bool float_matches(float left, float right)
{
    return std::abs(left - right) <= 1.0e-6F;
}

torch::Tensor ensure_grayscale(const torch::Tensor& image)
{
    require_chw_image(image);
    if (channels(image) == INPUT_CHANNELS)
    {
        return image.contiguous();
    }
    return image.mean(0, true).contiguous();
}

torch::Tensor limit_training_image_size(const torch::Tensor& image, int64_t resize)
{
    const auto height = image.size(1);
    const auto width = image.size(2);
    const auto max_edge = std::max(height, width);
    if (resize == 0 || max_edge <= resize)
    {
        return image.contiguous();
    }

    const double scale = static_cast<double>(resize) / static_cast<double>(max_edge);
    const int64_t resized_height =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(height) * scale)));
    const int64_t resized_width =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(width) * scale)));
    return torch::nn::functional::interpolate(image.unsqueeze(0),
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{resized_height, resized_width})
                                                  .mode(torch::kBilinear)
                                                  .align_corners(false))
        .squeeze(0)
        .contiguous();
}

void validate_config(const ImageDataset&, const SyntheticPairCacheConfig& config)
{
    if (config.cache_dir.empty())
    {
        throw std::invalid_argument("synthetic pair cache_dir must not be empty");
    }
    if (config.resize < 0)
    {
        throw std::invalid_argument("synthetic pair resize must be non-negative");
    }
    if (config.pair_count == 0)
    {
        throw std::invalid_argument("synthetic pair cache pair_count must be positive");
    }
    if (config.pairs_per_image == 0)
    {
        throw std::invalid_argument("synthetic pair cache pairs_per_image must be positive");
    }
}

void save_view_png(const torch::Tensor& view, const std::filesystem::path& path)
{
    auto image = (view.detach().cpu().clamp(0.0, 1.0).squeeze(0) * 255.0).to(torch::kUInt8).contiguous();
    cv::Mat mat(static_cast<int>(image.size(0)), static_cast<int>(image.size(1)), CV_8UC1, image.data_ptr<uint8_t>());
    if (!cv::imwrite(path.string(), mat))
    {
        throw std::invalid_argument("failed to write synthetic pair cache png: " + path.string());
    }
}

void save_view_tif(const torch::Tensor& view, const std::filesystem::path& path)
{
    auto image = (view.detach().cpu().clamp(0.0, 1.0).squeeze(0) * 65535.0).to(torch::kUInt16).contiguous();
    cv::Mat mat(static_cast<int>(image.size(0)), static_cast<int>(image.size(1)), CV_16UC1, image.data_ptr<uint16_t>());
    if (!cv::imwrite(path.string(), mat))
    {
        throw std::invalid_argument("failed to write synthetic pair cache tif: " + path.string());
    }
}

void write_pair_archive(const SyntheticPair& pair, std::size_t source_index, const std::filesystem::path& path)
{
    torch::serialize::OutputArchive archive;
    write_int64(archive, "format_version", CACHE_FORMAT_VERSION);
    write_int64(archive, "source_index", static_cast<int64_t>(source_index));
    archive.write("view_a", pair.view_a.detach().cpu().to(torch::kFloat32).contiguous());
    archive.write("view_b", pair.view_b.detach().cpu().to(torch::kFloat32).contiguous());
    archive.write("warp_a_to_b", pair.warp_a_to_b.detach().cpu().to(torch::kFloat32).contiguous());
    archive.write("valid_mask", pair.valid_mask.detach().cpu().contiguous());
    try
    {
        archive.save_to(path.string());
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

void write_manifest(const SyntheticPairCacheConfig& config)
{
    torch::serialize::OutputArchive archive;
    write_int64(archive, "format_version", CACHE_FORMAT_VERSION);
    write_int64(archive, "pair_count", static_cast<int64_t>(config.pair_count));
    write_int64(archive, "pairs_per_image", static_cast<int64_t>(config.pairs_per_image));
    write_int64(archive, "source_count", static_cast<int64_t>(config.source_count));
    write_int64(archive, "resize", config.resize);
    write_float(archive, "translation_x", config.pair_config.translation_x);
    write_float(archive, "translation_y", config.pair_config.translation_y);
    write_float(archive, "rotation_degrees", config.pair_config.rotation_degrees);
    write_float(archive, "scale", config.pair_config.scale);
    write_float(archive, "brightness_delta", config.pair_config.brightness_delta);
    write_float(archive, "contrast_scale", config.pair_config.contrast_scale);
    write_float(archive, "noise_sigma", config.pair_config.noise_sigma);
    write_float(archive, "rotation_step_degrees", config.pair_config.rotation_step_degrees);
    write_int64(archive, "augmentation_profile", static_cast<int64_t>(config.pair_config.augmentation_profile));
    write_float(archive, "extreme_pair_ratio", static_cast<float>(config.pair_config.extreme_pair_ratio));
    try
    {
        archive.save_to((cache_path(config.cache_dir) / "manifest.pt").string());
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

bool manifest_matches(const SyntheticPairCacheConfig& config)
{
    try
    {
        torch::serialize::InputArchive archive;
        archive.load_from((cache_path(config.cache_dir) / "manifest.pt").string());
        return read_int64(archive, "format_version") == CACHE_FORMAT_VERSION &&
               read_int64(archive, "pair_count") == static_cast<int64_t>(config.pair_count) &&
               read_int64(archive, "pairs_per_image") == static_cast<int64_t>(config.pairs_per_image) &&
               read_int64(archive, "source_count") == static_cast<int64_t>(config.source_count) &&
               read_int64(archive, "resize") == config.resize &&
               float_matches(read_float(archive, "translation_x"), config.pair_config.translation_x) &&
               float_matches(read_float(archive, "translation_y"), config.pair_config.translation_y) &&
               float_matches(read_float(archive, "rotation_degrees"), config.pair_config.rotation_degrees) &&
               float_matches(read_float(archive, "scale"), config.pair_config.scale) &&
               float_matches(read_float(archive, "brightness_delta"), config.pair_config.brightness_delta) &&
               float_matches(read_float(archive, "contrast_scale"), config.pair_config.contrast_scale) &&
               float_matches(read_float(archive, "noise_sigma"), config.pair_config.noise_sigma) &&
               float_matches(read_float(archive, "rotation_step_degrees"), config.pair_config.rotation_step_degrees) &&
               read_int64(archive, "augmentation_profile") ==
                   static_cast<int64_t>(config.pair_config.augmentation_profile) &&
               float_matches(read_float(archive, "extreme_pair_ratio"),
                             static_cast<float>(config.pair_config.extreme_pair_ratio));
    }
    catch (const c10::Error&)
    {
        return false;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool cache_files_exist(const SyntheticPairCacheConfig& config)
{
    if (!std::filesystem::exists(cache_path(config.cache_dir) / "manifest.pt"))
    {
        return false;
    }
    const auto source_count = std::max<std::size_t>(1, config.source_count);
    for (std::size_t index = 0; index < config.pair_count; ++index)
    {
        const auto source_index = index % source_count;
        const auto source_dir = find_source_dir_path(config.cache_dir, source_index);
        std::ostringstream view_a_name;
        view_a_name << "source_" << std::setw(6) << std::setfill('0') << source_index << "_view_a.png";
        if (!std::filesystem::exists(pair_pt_path(config.cache_dir, index, source_count)) ||
            !std::filesystem::exists(source_dir / view_a_name.str()) ||
            !std::filesystem::exists(source_dir / (pair_stem(index) + "_view_b.png")))
        {
            return false;
        }
    }
    return true;
}

bool cache_is_complete(const SyntheticPairCacheConfig& config)
{
    return cache_files_exist(config) && manifest_matches(config);
}

std::filesystem::path moved_aside_cache_path(const std::filesystem::path& path)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        auto candidate = path;
        candidate += ".rebuild_old_";
        candidate += std::to_string(stamp);
        candidate += "_";
        candidate += std::to_string(attempt);
        if (!std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }
    throw std::invalid_argument("failed to choose synthetic pair cache rebuild path");
}

void generate_cache(const ImageDataset& dataset, const SyntheticPairCacheConfig& config)
{
    const auto root = cache_path(config.cache_dir);
    if (config.rebuild && std::filesystem::exists(root))
    {
        std::error_code error;
        std::filesystem::rename(root, moved_aside_cache_path(root), error);
        if (error)
        {
            throw std::invalid_argument("failed to move old synthetic pair cache for rebuild: " + error.message());
        }
    }
    std::filesystem::create_directories(root);
    for (std::size_t source_index = 0; source_index < dataset.size(); ++source_index)
    {
        std::filesystem::create_directories(source_dir_path(config.cache_dir, dataset, source_index));
    }
    for (std::size_t index = 0; index < config.pair_count; ++index)
    {
        const auto source_index = index % dataset.size();
        const auto variant_index = index / dataset.size();
        auto image = limit_training_image_size(ensure_grayscale(dataset.load(source_index)), config.resize);
        auto pair_config = config.pair_config;
        pair_config.source_index = static_cast<int64_t>(source_index);
        pair_config.variant_index = static_cast<int64_t>(variant_index);
        auto pair = make_synthetic_pair(image, pair_config);
        write_pair_archive(pair, source_index, pair_pt_path(config.cache_dir, dataset, index));
        if (variant_index == 0)
        {
            save_view_png(pair.view_a, source_view_a_path(config.cache_dir, dataset, source_index));
        }
        save_view_png(pair.view_b, pair_view_path(config.cache_dir, dataset, index, "view_b"));
        if (pair_config.augmentation_profile == SyntheticPairAugmentationProfile::RotationOnly)
        {
            const auto degrees = static_cast<float>(variant_index) * pair_config.rotation_step_degrees;
            save_view_tif(pair.view_b, named_rotation_image_path(config.cache_dir, dataset, source_index, degrees));
            write_pair_archive(pair, source_index,
                               named_rotation_pair_path(config.cache_dir, dataset, source_index, degrees));
        }
    }
    write_manifest(config);
}

} // namespace

void prepare_synthetic_pair_cache(const ImageDataset& dataset, const SyntheticPairCacheConfig& config)
{
    auto resolved_config = config;
    resolved_config.source_count = dataset.size();
    validate_config(dataset, resolved_config);
    if (!resolved_config.rebuild && cache_is_complete(resolved_config))
    {
        return;
    }
    generate_cache(dataset, resolved_config);
}

SyntheticPairCacheDataset::SyntheticPairCacheDataset(std::string cache_dir) : _cache_dir(std::move(cache_dir))
{
    try
    {
        torch::serialize::InputArchive archive;
        archive.load_from((cache_path(_cache_dir) / "manifest.pt").string());
        if (read_int64(archive, "format_version") != CACHE_FORMAT_VERSION)
        {
            throw std::invalid_argument("synthetic pair cache format version is unsupported");
        }
        _pair_count = static_cast<std::size_t>(read_int64(archive, "pair_count"));
        _pairs_per_image = static_cast<std::size_t>(read_int64(archive, "pairs_per_image"));
        _source_count = static_cast<std::size_t>(read_int64(archive, "source_count"));
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

std::size_t SyntheticPairCacheDataset::size() const
{
    return _pair_count;
}

SyntheticPair SyntheticPairCacheDataset::load(std::size_t index) const
{
    if (index >= _pair_count)
    {
        throw std::out_of_range("synthetic pair cache index out of range");
    }

    try
    {
        torch::serialize::InputArchive archive;
        archive.load_from(pair_pt_path(_cache_dir, index, _source_count).string());
        if (read_int64(archive, "format_version") != CACHE_FORMAT_VERSION)
        {
            throw std::invalid_argument("synthetic pair cache pair format version is unsupported");
        }
        return SyntheticPair{read_tensor(archive, "view_a").to(torch::kCPU, torch::kFloat32).contiguous(),
                             read_tensor(archive, "view_b").to(torch::kCPU, torch::kFloat32).contiguous(),
                             read_tensor(archive, "warp_a_to_b").to(torch::kCPU, torch::kFloat32).contiguous(),
                             read_tensor(archive, "valid_mask").cpu().contiguous()};
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

SyntheticPairCacheTensorDataset::SyntheticPairCacheTensorDataset(std::string cache_dir) : _cache(std::move(cache_dir))
{
}

size_t SyntheticPairCacheTensorDataset::size() const
{
    return _cache.size();
}

TensorBatch SyntheticPairCacheTensorDataset::get(size_t index)
{
    const auto pair = _cache.load(index);
    TensorBatch batch;
    batch["view_a"] = pair.view_a;
    batch["view_b"] = pair.view_b;
    batch["warp_a_to_b"] = pair.warp_a_to_b;
    batch["valid_mask"] = pair.valid_mask;
    return batch;
}

} // namespace pfm
