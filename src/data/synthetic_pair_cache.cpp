#include "data/synthetic_pair_cache.h"

#include <algorithm>
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

namespace pfm {
namespace {

constexpr int64_t CACHE_FORMAT_VERSION = 1;
constexpr int64_t INPUT_CHANNELS = 1;

std::filesystem::path cache_path(const std::string& cache_dir) {
    return std::filesystem::path(cache_dir);
}

std::string pair_stem(std::size_t index) {
    std::ostringstream stream;
    stream << "pair_" << std::setw(6) << std::setfill('0') << index;
    return stream.str();
}

std::filesystem::path pair_pt_path(const std::string& cache_dir, std::size_t index) {
    return cache_path(cache_dir) / (pair_stem(index) + ".pt");
}

std::filesystem::path pair_view_path(const std::string& cache_dir, std::size_t index, const char* view_name) {
    return cache_path(cache_dir) / (pair_stem(index) + "_" + view_name + ".png");
}

std::filesystem::path source_view_a_path(const std::string& cache_dir, std::size_t source_index) {
    std::ostringstream stream;
    stream << "source_" << std::setw(6) << std::setfill('0') << source_index << "_view_a.png";
    return cache_path(cache_dir) / stream.str();
}

torch::Tensor read_tensor(torch::serialize::InputArchive& archive, const char* name) {
    torch::Tensor tensor;
    archive.read(name, tensor);
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string("synthetic pair cache missing ") + name);
    }
    return tensor;
}

int64_t read_int64(torch::serialize::InputArchive& archive, const char* name) {
    auto tensor = read_tensor(archive, name);
    if (tensor.numel() != 1) {
        throw std::invalid_argument(std::string("synthetic pair cache invalid ") + name);
    }
    return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

float read_float(torch::serialize::InputArchive& archive, const char* name) {
    auto tensor = read_tensor(archive, name);
    if (tensor.numel() != 1) {
        throw std::invalid_argument(std::string("synthetic pair cache invalid ") + name);
    }
    return tensor.to(torch::kCPU, torch::kFloat32).reshape({1}).item<float>();
}

void write_int64(torch::serialize::OutputArchive& archive, const char* name, int64_t value) {
    archive.write(name, torch::tensor({value}, torch::kInt64));
}

void write_float(torch::serialize::OutputArchive& archive, const char* name, float value) {
    archive.write(name, torch::tensor({value}, torch::kFloat32));
}

bool float_matches(float left, float right) {
    return std::abs(left - right) <= 1.0e-6F;
}

torch::Tensor ensure_grayscale(const torch::Tensor& image) {
    require_chw_image(image);
    if (channels(image) == INPUT_CHANNELS) {
        return image.contiguous();
    }
    return image.mean(0, true).contiguous();
}

torch::Tensor limit_training_image_size(const torch::Tensor& image, int64_t resize) {
    const auto height = image.size(1);
    const auto width = image.size(2);
    const auto max_edge = std::max(height, width);
    if (resize == 0 || max_edge <= resize) {
        return image.contiguous();
    }

    const double scale = static_cast<double>(resize) / static_cast<double>(max_edge);
    const int64_t resized_height =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(height) * scale)));
    const int64_t resized_width =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(width) * scale)));
    return torch::nn::functional::interpolate(
               image.unsqueeze(0),
               torch::nn::functional::InterpolateFuncOptions()
                   .size(std::vector<int64_t>{resized_height, resized_width})
                   .mode(torch::kBilinear)
                   .align_corners(false))
        .squeeze(0)
        .contiguous();
}

void validate_config(const ImageDataset&, const SyntheticPairCacheConfig& config) {
    if (config.cache_dir.empty()) {
        throw std::invalid_argument("synthetic pair cache_dir must not be empty");
    }
    if (config.resize < 0) {
        throw std::invalid_argument("synthetic pair resize must be non-negative");
    }
    if (config.pair_count == 0) {
        throw std::invalid_argument("synthetic pair cache pair_count must be positive");
    }
    if (config.pairs_per_image == 0) {
        throw std::invalid_argument("synthetic pair cache pairs_per_image must be positive");
    }
}

void save_view_png(const torch::Tensor& view, const std::filesystem::path& path) {
    auto image = (view.detach().cpu().clamp(0.0, 1.0).squeeze(0) * 255.0).to(torch::kUInt8).contiguous();
    cv::Mat mat(
        static_cast<int>(image.size(0)),
        static_cast<int>(image.size(1)),
        CV_8UC1,
        image.data_ptr<uint8_t>());
    if (!cv::imwrite(path.string(), mat)) {
        throw std::invalid_argument("failed to write synthetic pair cache png: " + path.string());
    }
}

void write_pair_archive(
    const SyntheticPair& pair,
    std::size_t source_index,
    const std::filesystem::path& path
) {
    torch::serialize::OutputArchive archive;
    write_int64(archive, "format_version", CACHE_FORMAT_VERSION);
    write_int64(archive, "source_index", static_cast<int64_t>(source_index));
    archive.write("view_a", pair.view_a.detach().cpu().to(torch::kFloat32).contiguous());
    archive.write("view_b", pair.view_b.detach().cpu().to(torch::kFloat32).contiguous());
    archive.write("warp_a_to_b", pair.warp_a_to_b.detach().cpu().to(torch::kFloat32).contiguous());
    archive.write("valid_mask", pair.valid_mask.detach().cpu().contiguous());
    try {
        archive.save_to(path.string());
    } catch (const c10::Error& e) {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

void write_manifest(const SyntheticPairCacheConfig& config) {
    torch::serialize::OutputArchive archive;
    write_int64(archive, "format_version", CACHE_FORMAT_VERSION);
    write_int64(archive, "pair_count", static_cast<int64_t>(config.pair_count));
    write_int64(archive, "pairs_per_image", static_cast<int64_t>(config.pairs_per_image));
    write_int64(archive, "resize", config.resize);
    write_float(archive, "translation_x", config.pair_config.translation_x);
    write_float(archive, "translation_y", config.pair_config.translation_y);
    write_float(archive, "rotation_degrees", config.pair_config.rotation_degrees);
    write_float(archive, "scale", config.pair_config.scale);
    write_float(archive, "brightness_delta", config.pair_config.brightness_delta);
    write_float(archive, "contrast_scale", config.pair_config.contrast_scale);
    write_float(archive, "noise_sigma", config.pair_config.noise_sigma);
    write_int64(archive,
                "augmentation_profile",
                static_cast<int64_t>(config.pair_config.augmentation_profile));
    write_float(archive, "extreme_pair_ratio", static_cast<float>(config.pair_config.extreme_pair_ratio));
    try {
        archive.save_to((cache_path(config.cache_dir) / "manifest.pt").string());
    } catch (const c10::Error& e) {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

bool manifest_matches(const SyntheticPairCacheConfig& config) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from((cache_path(config.cache_dir) / "manifest.pt").string());
        return read_int64(archive, "format_version") == CACHE_FORMAT_VERSION &&
               read_int64(archive, "pair_count") == static_cast<int64_t>(config.pair_count) &&
               read_int64(archive, "pairs_per_image") == static_cast<int64_t>(config.pairs_per_image) &&
               read_int64(archive, "resize") == config.resize &&
               float_matches(read_float(archive, "translation_x"), config.pair_config.translation_x) &&
               float_matches(read_float(archive, "translation_y"), config.pair_config.translation_y) &&
               float_matches(read_float(archive, "rotation_degrees"), config.pair_config.rotation_degrees) &&
               float_matches(read_float(archive, "scale"), config.pair_config.scale) &&
               float_matches(read_float(archive, "brightness_delta"), config.pair_config.brightness_delta) &&
               float_matches(read_float(archive, "contrast_scale"), config.pair_config.contrast_scale) &&
               float_matches(read_float(archive, "noise_sigma"), config.pair_config.noise_sigma) &&
               read_int64(archive, "augmentation_profile") ==
                   static_cast<int64_t>(config.pair_config.augmentation_profile) &&
               float_matches(read_float(archive, "extreme_pair_ratio"),
                             static_cast<float>(config.pair_config.extreme_pair_ratio));
    } catch (const c10::Error&) {
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

bool cache_files_exist(const SyntheticPairCacheConfig& config) {
    if (!std::filesystem::exists(cache_path(config.cache_dir) / "manifest.pt")) {
        return false;
    }
    const auto source_count = std::max<std::size_t>(1, config.pair_count / config.pairs_per_image);
    for (std::size_t index = 0; index < config.pair_count; ++index) {
        const auto source_index = index % source_count;
        if (!std::filesystem::exists(pair_pt_path(config.cache_dir, index)) ||
            !std::filesystem::exists(source_view_a_path(config.cache_dir, source_index)) ||
            !std::filesystem::exists(pair_view_path(config.cache_dir, index, "view_b"))) {
            return false;
        }
    }
    return true;
}

bool cache_is_complete(const SyntheticPairCacheConfig& config) {
    return cache_files_exist(config) && manifest_matches(config);
}

void generate_cache(const ImageDataset& dataset, const SyntheticPairCacheConfig& config) {
    std::filesystem::create_directories(cache_path(config.cache_dir));
    for (std::size_t index = 0; index < config.pair_count; ++index) {
        const auto source_index = index % dataset.size();
        const auto variant_index = index / dataset.size();
        auto image = limit_training_image_size(ensure_grayscale(dataset.load(source_index)), config.resize);
        auto pair_config = config.pair_config;
        pair_config.source_index = static_cast<int64_t>(source_index);
        pair_config.variant_index = static_cast<int64_t>(variant_index);
        auto pair = make_synthetic_pair(image, pair_config);
        write_pair_archive(pair, source_index, pair_pt_path(config.cache_dir, index));
        if (variant_index == 0) {
            save_view_png(pair.view_a, source_view_a_path(config.cache_dir, source_index));
        }
        save_view_png(pair.view_b, pair_view_path(config.cache_dir, index, "view_b"));
    }
    write_manifest(config);
}

}  // namespace

void prepare_synthetic_pair_cache(const ImageDataset& dataset, const SyntheticPairCacheConfig& config) {
    validate_config(dataset, config);
    if (!config.rebuild && cache_is_complete(config)) {
        return;
    }
    generate_cache(dataset, config);
}

SyntheticPairCacheDataset::SyntheticPairCacheDataset(std::string cache_dir) : _cache_dir(std::move(cache_dir)) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from((cache_path(_cache_dir) / "manifest.pt").string());
        if (read_int64(archive, "format_version") != CACHE_FORMAT_VERSION) {
            throw std::invalid_argument("synthetic pair cache format version is unsupported");
        }
        _pair_count = static_cast<std::size_t>(read_int64(archive, "pair_count"));
    } catch (const c10::Error& e) {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

std::size_t SyntheticPairCacheDataset::size() const {
    return _pair_count;
}

SyntheticPair SyntheticPairCacheDataset::load(std::size_t index) const {
    if (index >= _pair_count) {
        throw std::out_of_range("synthetic pair cache index out of range");
    }

    try {
        torch::serialize::InputArchive archive;
        archive.load_from(pair_pt_path(_cache_dir, index).string());
        if (read_int64(archive, "format_version") != CACHE_FORMAT_VERSION) {
            throw std::invalid_argument("synthetic pair cache pair format version is unsupported");
        }
        return SyntheticPair{
            read_tensor(archive, "view_a").to(torch::kCPU, torch::kFloat32).contiguous(),
            read_tensor(archive, "view_b").to(torch::kCPU, torch::kFloat32).contiguous(),
            read_tensor(archive, "warp_a_to_b").to(torch::kCPU, torch::kFloat32).contiguous(),
            read_tensor(archive, "valid_mask").cpu().contiguous()};
    } catch (const c10::Error& e) {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

}  // namespace pfm
