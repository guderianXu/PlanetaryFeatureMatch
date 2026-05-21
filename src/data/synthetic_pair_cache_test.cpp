#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "data/image_dataset.h"
#include "data/synthetic_pair_cache.h"
#include "tests/test_harness.h"

namespace {

class TempCacheDirectory {
public:
    explicit TempCacheDirectory(const std::string& stem) {
        const auto suffix = std::to_string(static_cast<long long>(getpid())) + "_" +
                            std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempCacheDirectory() {
        std::error_code ignored;
        const auto cache_dir = _path / "pair_cache";
        if (std::filesystem::exists(cache_dir, ignored)) {
            for (const auto& entry : std::filesystem::directory_iterator(cache_dir)) {
                std::filesystem::remove(entry.path(), ignored);
            }
            std::filesystem::remove(cache_dir, ignored);
        }
        for (const auto& entry : std::filesystem::directory_iterator(_path)) {
            std::filesystem::remove(entry.path(), ignored);
        }
        std::filesystem::remove(_path, ignored);
    }

    const std::filesystem::path& path() const {
        return _path;
    }

    std::filesystem::path file(const std::string& name) const {
        return _path / name;
    }

private:
    std::filesystem::path _path;
};

void require_image_written(const std::filesystem::path& path, int offset) {
    cv::Mat image(20, 24, CV_8UC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 5 + y * 9 + offset) % 256);
        }
    }
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

pfm::ImageDataset make_dataset(const TempCacheDirectory& temp_dir) {
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 17);
    return pfm::ImageDataset(temp_dir.path().string());
}

pfm::SyntheticPairCacheConfig make_cache_config(const TempCacheDirectory& temp_dir, std::size_t pair_count) {
    pfm::SyntheticPairCacheConfig config;
    config.cache_dir = temp_dir.file("pair_cache").string();
    config.resize = 16;
    config.pair_count = pair_count;
    config.pair_config.translation_x = 1.0F;
    config.pair_config.translation_y = 0.0F;
    config.pair_config.brightness_delta = 0.02F;
    config.pair_config.contrast_scale = 0.98F;
    return config;
}

std::filesystem::path source_dir(const pfm::SyntheticPairCacheConfig& config, std::size_t source_index, const char* stem) {
    std::ostringstream stream;
    stream << "source_" << std::setw(6) << std::setfill('0') << source_index << "_" << stem;
    return std::filesystem::path(config.cache_dir) / stream.str();
}

}  // namespace

static void synthetic_pair_cache_generates_pt_manifest_and_png_views() {
    TempCacheDirectory temp_dir("pfm_pair_cache_generate");
    auto dataset = make_dataset(temp_dir);
    const auto config = make_cache_config(temp_dir, 2);

    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.cache_dir) / "manifest.pt"));
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 0, "image_a") / "pair_000000.pt"));
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 0, "image_a") / "source_000000_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 0, "image_a") / "pair_000000_view_b.png"));
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 1, "image_b") / "pair_000001.pt"));
}

static void synthetic_pair_cache_writes_named_rotation_tif_and_correspondence() {
    TempCacheDirectory temp_dir("pfm_pair_cache_rotation_named");
    auto dataset = make_dataset(temp_dir);
    auto config = make_cache_config(temp_dir, 4);
    config.pairs_per_image = 2;
    config.pair_config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::RotationOnly;
    config.pair_config.rotation_step_degrees = 30.0F;

    pfm::prepare_synthetic_pair_cache(dataset, config);

    const auto cache_dir = std::filesystem::path(config.cache_dir);
    const auto image_a_dir = source_dir(config, 0, "image_a");
    PFM_REQUIRE(std::filesystem::exists(image_a_dir / "image_a_000.tif"));
    PFM_REQUIRE(std::filesystem::exists(image_a_dir / "image_a_030.tif"));
    PFM_REQUIRE(std::filesystem::exists(image_a_dir / "image_a_030.pt"));

    pfm::SyntheticPairCacheDataset cache_dataset(config.cache_dir);
    const auto identity_pair = cache_dataset.load(0);
    const auto rotated_pair = cache_dataset.load(2);
    auto xy = torch::meshgrid(
        {torch::arange(identity_pair.warp_a_to_b.size(0), torch::kFloat32),
         torch::arange(identity_pair.warp_a_to_b.size(1), torch::kFloat32)},
        "ij");
    const auto grid = torch::stack({xy[1], xy[0]}, 2);
    PFM_REQUIRE(torch::allclose(identity_pair.warp_a_to_b, grid, 1.0e-4, 1.0e-4));
    PFM_REQUIRE(!torch::allclose(rotated_pair.warp_a_to_b, grid, 1.0e-4, 1.0e-4));
}

static void synthetic_pair_cache_dataset_loads_cached_pair_shapes() {
    TempCacheDirectory temp_dir("pfm_pair_cache_load");
    auto dataset = make_dataset(temp_dir);
    const auto config = make_cache_config(temp_dir, 1);
    pfm::prepare_synthetic_pair_cache(dataset, config);

    pfm::SyntheticPairCacheDataset cache_dataset(config.cache_dir);
    auto pair = cache_dataset.load(0);

    PFM_REQUIRE(cache_dataset.size() == 1);
    PFM_REQUIRE(pair.view_a.sizes() == torch::IntArrayRef({1, 13, 16}));
    PFM_REQUIRE(pair.view_b.sizes() == torch::IntArrayRef({1, 13, 16}));
    PFM_REQUIRE(pair.warp_a_to_b.sizes() == torch::IntArrayRef({13, 16, 2}));
    PFM_REQUIRE(pair.valid_mask.sizes() == torch::IntArrayRef({13, 16}));
    PFM_REQUIRE(pair.view_a.device().is_cpu());
}

static void synthetic_pair_cache_generates_multiple_pairs_per_source_image() {
    TempCacheDirectory temp_dir("pfm_pair_cache_multiple_pairs");
    auto dataset = make_dataset(temp_dir);
    const auto config = make_cache_config(temp_dir, 4);

    pfm::prepare_synthetic_pair_cache(dataset, config);

    pfm::SyntheticPairCacheDataset cache_dataset(config.cache_dir);
    PFM_REQUIRE(cache_dataset.size() == 4);
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 0, "image_a") / "pair_000002.pt"));
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 1, "image_b") / "pair_000003.pt"));
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 1, "image_b") / "source_000001_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 1, "image_b") / "pair_000003_view_b.png"));
}

static void synthetic_pair_cache_varies_multiple_pairs_from_same_source_image() {
    TempCacheDirectory temp_dir("pfm_pair_cache_varied_pairs");
    auto dataset = make_dataset(temp_dir);
    const auto config = make_cache_config(temp_dir, 4);

    pfm::prepare_synthetic_pair_cache(dataset, config);

    pfm::SyntheticPairCacheDataset cache_dataset(config.cache_dir);
    auto first_source_first_pair = cache_dataset.load(0);
    auto first_source_second_pair = cache_dataset.load(2);
    PFM_REQUIRE(!torch::allclose(first_source_first_pair.warp_a_to_b, first_source_second_pair.warp_a_to_b));
    PFM_REQUIRE(torch::allclose(first_source_first_pair.view_a, first_source_second_pair.view_a));
}

static void synthetic_pair_cache_saves_one_view_a_png_per_source_image() {
    TempCacheDirectory temp_dir("pfm_pair_cache_single_view_a");
    auto dataset = make_dataset(temp_dir);
    const auto config = make_cache_config(temp_dir, 4);

    pfm::prepare_synthetic_pair_cache(dataset, config);

    std::size_t view_a_count = 0;
    for (const auto& root_entry : std::filesystem::directory_iterator(config.cache_dir)) {
        if (!root_entry.is_directory()) {
            continue;
        }
        for (const auto& entry : std::filesystem::directory_iterator(root_entry.path())) {
        if (entry.path().filename().string().find("view_a.png") != std::string::npos) {
            ++view_a_count;
        }
        }
    }
    PFM_REQUIRE(view_a_count == dataset.size());
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 0, "image_a") / "source_000000_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(source_dir(config, 1, "image_b") / "source_000001_view_a.png"));
}

static void synthetic_pair_cache_reuses_complete_matching_cache() {
    TempCacheDirectory temp_dir("pfm_pair_cache_reuse");
    auto dataset = make_dataset(temp_dir);
    const auto config = make_cache_config(temp_dir, 1);
    pfm::prepare_synthetic_pair_cache(dataset, config);
    const auto pair_path = source_dir(config, 0, "image_a") / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) == first_write_time);
}

static void synthetic_pair_cache_rebuilds_when_manifest_config_changes() {
    TempCacheDirectory temp_dir("pfm_pair_cache_config_rebuild");
    auto dataset = make_dataset(temp_dir);
    auto config = make_cache_config(temp_dir, 1);
    pfm::prepare_synthetic_pair_cache(dataset, config);
    const auto pair_path = source_dir(config, 0, "image_a") / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.resize = 12;
    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) != first_write_time);
}

static void synthetic_pair_cache_rebuilds_when_geometric_config_changes() {
    TempCacheDirectory temp_dir("pfm_pair_cache_geometric_rebuild");
    auto dataset = make_dataset(temp_dir);
    auto config = make_cache_config(temp_dir, 1);
    pfm::prepare_synthetic_pair_cache(dataset, config);
    const auto pair_path = source_dir(config, 0, "image_a") / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.pair_config.rotation_degrees = 12.0F;
    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) != first_write_time);
}

static void synthetic_pair_cache_rebuilds_when_pair_file_is_missing() {
    TempCacheDirectory temp_dir("pfm_pair_cache_missing_rebuild");
    auto dataset = make_dataset(temp_dir);
    const auto config = make_cache_config(temp_dir, 1);
    pfm::prepare_synthetic_pair_cache(dataset, config);
    const auto pair_path = source_dir(config, 0, "image_a") / "pair_000000.pt";
    PFM_REQUIRE(std::filesystem::remove(pair_path));

    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::exists(pair_path));
}

static void synthetic_pair_cache_rebuilds_when_augmentation_profile_changes() {
    TempCacheDirectory temp_dir("pfm_pair_cache_profile_rebuild");
    auto dataset = make_dataset(temp_dir);
    auto config = make_cache_config(temp_dir, 2);
    config.pair_config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Mild;
    pfm::prepare_synthetic_pair_cache(dataset, config);
    const auto pair_path = source_dir(config, 0, "image_a") / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.pair_config.augmentation_profile = pfm::SyntheticPairAugmentationProfile::Extreme;
    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) != first_write_time);
}

static void synthetic_pair_cache_rebuilds_when_extreme_pair_ratio_changes() {
    TempCacheDirectory temp_dir("pfm_pair_cache_ratio_rebuild");
    auto dataset = make_dataset(temp_dir);
    auto config = make_cache_config(temp_dir, 2);
    config.pair_config.extreme_pair_ratio = 0.2;
    pfm::prepare_synthetic_pair_cache(dataset, config);
    const auto pair_path = source_dir(config, 0, "image_a") / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.pair_config.extreme_pair_ratio = 0.6;
    pfm::prepare_synthetic_pair_cache(dataset, config);

    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) != first_write_time);
}

void register_synthetic_pair_cache_tests() {
    register_test("synthetic_pair_cache_generates_pt_manifest_and_png_views",
                  synthetic_pair_cache_generates_pt_manifest_and_png_views);
    register_test(
        "synthetic_pair_cache_writes_named_rotation_tif_and_correspondence",
        synthetic_pair_cache_writes_named_rotation_tif_and_correspondence);
    register_test("synthetic_pair_cache_dataset_loads_cached_pair_shapes",
                  synthetic_pair_cache_dataset_loads_cached_pair_shapes);
    register_test("synthetic_pair_cache_generates_multiple_pairs_per_source_image",
                  synthetic_pair_cache_generates_multiple_pairs_per_source_image);
    register_test("synthetic_pair_cache_varies_multiple_pairs_from_same_source_image",
                  synthetic_pair_cache_varies_multiple_pairs_from_same_source_image);
    register_test("synthetic_pair_cache_saves_one_view_a_png_per_source_image",
                  synthetic_pair_cache_saves_one_view_a_png_per_source_image);
    register_test("synthetic_pair_cache_reuses_complete_matching_cache",
                  synthetic_pair_cache_reuses_complete_matching_cache);
    register_test("synthetic_pair_cache_rebuilds_when_manifest_config_changes",
                  synthetic_pair_cache_rebuilds_when_manifest_config_changes);
    register_test("synthetic_pair_cache_rebuilds_when_geometric_config_changes",
                  synthetic_pair_cache_rebuilds_when_geometric_config_changes);
    register_test("synthetic_pair_cache_rebuilds_when_pair_file_is_missing",
                  synthetic_pair_cache_rebuilds_when_pair_file_is_missing);
    register_test("synthetic_pair_cache_rebuilds_when_augmentation_profile_changes",
                  synthetic_pair_cache_rebuilds_when_augmentation_profile_changes);
    register_test("synthetic_pair_cache_rebuilds_when_extreme_pair_ratio_changes",
                  synthetic_pair_cache_rebuilds_when_extreme_pair_ratio_changes);
}
