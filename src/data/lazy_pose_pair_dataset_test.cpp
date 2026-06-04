#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <unistd.h>

#include "data/lazy_pose_pair_dataset.h"
#include "dataloader/async_dataloader.h"
#include "dataloader/collator.h"
#include "dataloader/sampler.h"
#include "tests/test_harness.h"

namespace
{

class TempLazyPosePairDirectory
{
  public:
    explicit TempLazyPosePairDirectory(const std::string& name)
    {
        const auto root = std::filesystem::temp_directory_path();
        path = root / (name + "_" + std::to_string(::getpid()) + "_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
    }

    std::filesystem::path path;
};

void writeTextFile(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    stream << text;
}

void writeTsai(const std::filesystem::path& path)
{
    writeTextFile(path,
                  "fu = 1\n"
                  "fv = 1\n"
                  "cu = 0.5\n"
                  "cv = 0.5\n"
                  "C = 0 0 0\n"
                  "R = 1 0 0 0 1 0 0 0 1\n");
}

void writeGrayTif(const std::filesystem::path& path, int rows, int cols, uint8_t base)
{
    std::filesystem::create_directories(path.parent_path());
    cv::Mat image(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            image.at<uint8_t>(y, x) = static_cast<uint8_t>(base + y * cols + x);
        }
    }
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

void writeDepthTif(const std::filesystem::path& path, int rows, int cols, float value)
{
    std::filesystem::create_directories(path.parent_path());
    cv::Mat image(rows, cols, CV_32FC1, cv::Scalar(value));
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

std::string renderManifestHeader()
{
    return "pose_id,base_id,variant,split,lon_deg,lat_deg,tsai_path,image_path,depth_path,chunk_index\n";
}

std::string renderManifestRow(const std::string& pose_id, const std::string& base_id, const std::string& variant,
                              const std::string& split, const std::filesystem::path& tsai_path,
                              const std::filesystem::path& image_path, const std::filesystem::path& depth_path)
{
    return pose_id + "," + base_id + "," + variant + "," + split + ",45.0,-10.0," + tsai_path.string() + "," +
           image_path.string() + "," + depth_path.string() + ",7\n";
}

pfm::LazyPosePairDatasetConfig makeTinyDatasetFiles(TempLazyPosePairDirectory& temp, int base_count)
{
    std::string manifest = renderManifestHeader();
    std::string uint8_manifest = "source_path,uint8_path\nsource_path,uint8_path\n";
    for (int base_index = 0; base_index < base_count; ++base_index)
    {
        const auto stem = std::string("b") + std::to_string(base_index);
        for (const auto variant : {std::string("nadir"), std::string("small_01")})
        {
            const auto pose_id = stem + "_" + variant;
            const auto tsai = temp.path / "tsai" / (pose_id + ".tsai");
            const auto render = temp.path / "render" / (pose_id + ".tif");
            const auto uint8 = temp.path / "uint8" / (pose_id + ".tif");
            const auto depth = temp.path / "depth" / (pose_id + ".tif");
            writeTsai(tsai);
            writeGrayTif(render, 6, 6, variant == "nadir" ? 10 : 80);
            writeGrayTif(uint8, 6, 6, variant == "nadir" ? 20 : 90);
            writeDepthTif(depth, 6, 6, 1.0F);
            manifest += renderManifestRow(pose_id, stem, variant, "train", tsai, render, depth);
            uint8_manifest += render.string() + "," + uint8.string() + "\n";
        }
    }
    const auto render_manifest = temp.path / "render_manifest.csv";
    const auto uint8_manifest_path = temp.path / "images_u8" / "uint8_manifest.csv";
    writeTextFile(render_manifest, manifest);
    writeTextFile(uint8_manifest_path, uint8_manifest);

    pfm::LazyPosePairDatasetConfig config;
    config.render_manifest = render_manifest;
    config.uint8_manifest = uint8_manifest_path;
    config.target_variants = {"small_01"};
    config.crop_size = 4;
    return config;
}

void lazy_pose_pair_manifest_builds_default_nadir_targets()
{
    TempLazyPosePairDirectory temp("pfm_lazy_pose_manifest");
    const auto nadir_tsai = temp.path / "tsai" / "nadir.tsai";
    const auto small_tsai = temp.path / "tsai" / "small.tsai";
    const auto mid_tsai = temp.path / "tsai" / "mid.tsai";
    const auto ignored_tsai = temp.path / "tsai" / "ignored.tsai";
    const auto nadir_image = temp.path / "render" / "nadir.tif";
    const auto small_image = temp.path / "render" / "small.tif";
    const auto mid_image = temp.path / "render" / "mid.tif";
    const auto ignored_image = temp.path / "render" / "ignored.tif";
    const auto nadir_u8 = temp.path / "uint8" / "nadir.tif";
    const auto small_u8 = temp.path / "uint8" / "small.tif";
    const auto mid_u8 = temp.path / "uint8" / "mid.tif";
    const auto nadir_depth = temp.path / "depth" / "nadir.tif";
    const auto small_depth = temp.path / "depth" / "small.tif";
    const auto mid_depth = temp.path / "depth" / "mid.tif";
    const auto ignored_depth = temp.path / "depth" / "ignored.tif";
    for (const auto& path : {nadir_tsai, small_tsai, mid_tsai, ignored_tsai, nadir_image, small_image, mid_image,
                            ignored_image, nadir_u8, small_u8, mid_u8, nadir_depth, small_depth, mid_depth,
                            ignored_depth})
    {
        writeTextFile(path, "x");
    }
    const auto render_manifest = temp.path / "render_manifest.csv";
    writeTextFile(render_manifest,
                  renderManifestHeader() +
                      renderManifestRow("pose_nadir", "base_001", "nadir", "train", nadir_tsai, nadir_image,
                                        nadir_depth) +
                      renderManifestRow("pose_mid", "base_001", "mid_01", "train", mid_tsai, mid_image, mid_depth) +
                      renderManifestRow("pose_small", "base_001", "small_01", "train", small_tsai, small_image,
                                        small_depth) +
                      renderManifestRow("pose_other", "base_001", "other", "train", ignored_tsai, ignored_image,
                                        ignored_depth));
    const auto uint8_manifest = temp.path / "images_u8" / "uint8_manifest.csv";
    writeTextFile(uint8_manifest, "source_path,uint8_path\nsource_path,uint8_path\n" + nadir_image.string() + "," +
                                      nadir_u8.string() + "\n" + small_image.string() + "," + small_u8.string() +
                                      "\n" + mid_image.string() + "," + mid_u8.string() + "\n");

    const auto uint8_paths = pfm::readPoseUint8Manifest(uint8_manifest);
    const auto records = pfm::readPoseRenderManifest(render_manifest, uint8_paths);
    const auto specs = pfm::buildLazyPosePairSpecs(records, pfm::LazyPosePairBuildOptions{});

    PFM_REQUIRE(uint8_paths.size() == 3);
    PFM_REQUIRE(records.size() == 4);
    PFM_REQUIRE(specs.size() == 2);
    PFM_REQUIRE(specs[0].base_id == "base_001");
    PFM_REQUIRE(specs[0].reference_variant == "nadir");
    PFM_REQUIRE(specs[0].target_variant == "small_01");
    PFM_REQUIRE(specs[0].reference_image_path == nadir_u8);
    PFM_REQUIRE(specs[0].target_image_path == small_u8);
    PFM_REQUIRE(specs[1].target_variant == "mid_01");
}

void lazy_pose_pair_project_warp_identity_camera_returns_pixel_grid()
{
    TempLazyPosePairDirectory temp("pfm_lazy_pose_warp");
    const auto tsai = temp.path / "camera.tsai";
    writeTsai(tsai);
    const auto camera = pfm::parsePoseTsaiCamera(tsai);
    const auto depth = torch::ones({3, 4}, torch::kFloat32);

    const auto result = pfm::projectPoseDepthWarp(depth, depth, camera, camera, 0.1, 0.001);

    PFM_REQUIRE(result.warp_a_to_b.sizes() == torch::IntArrayRef({3, 4, 2}));
    PFM_REQUIRE(result.valid_mask.sizes() == torch::IntArrayRef({3, 4}));
    PFM_REQUIRE(result.valid_mask.all().item<bool>());
    PFM_REQUIRE_CLOSE(result.warp_a_to_b.index({0, 0, 0}).item<float>(), 0.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(result.warp_a_to_b.index({0, 0, 1}).item<float>(), 0.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(result.warp_a_to_b.index({2, 3, 0}).item<float>(), 3.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(result.warp_a_to_b.index({2, 3, 1}).item<float>(), 2.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(result.valid_pair_fraction, 1.0, 1.0e-12);
}

void lazy_pose_pair_project_warp_uses_projected_b_depth_for_visibility()
{
    TempLazyPosePairDirectory temp("pfm_lazy_pose_depth_consistency");
    const auto tsai = temp.path / "camera.tsai";
    writeTsai(tsai);
    auto camera_a = pfm::parsePoseTsaiCamera(tsai);
    camera_a.cu = 1.5;
    camera_a.cv = 1.5;
    auto camera_b = camera_a;
    camera_b.center = {0.0, 0.0, -1.0};
    const auto depth_a = torch::ones({3, 3}, torch::kFloat32);
    const auto consistent_depth_b = torch::ones({3, 3}, torch::kFloat32) * 2.0F;

    const auto result = pfm::projectPoseDepthWarp(depth_a, consistent_depth_b, camera_a, camera_b, 0.1, 0.001);

    PFM_REQUIRE(result.valid_mask.index({1, 1}).item<bool>());
    PFM_REQUIRE(result.valid_pixels > 0);
    PFM_REQUIRE(result.valid_pair_fraction > 0.0);
}

void lazy_pose_pair_dataset_loads_cropped_uint8_pair()
{
    TempLazyPosePairDirectory temp("pfm_lazy_pose_dataset");
    auto config = makeTinyDatasetFiles(temp, 1);
    pfm::LazyPosePairDataset dataset(config);

    const auto sample = dataset.get(0);

    PFM_REQUIRE(dataset.size() == 1);
    PFM_REQUIRE(sample.at("view_a").sizes() == torch::IntArrayRef({1, 4, 4}));
    PFM_REQUIRE(sample.at("view_b").sizes() == torch::IntArrayRef({1, 4, 4}));
    PFM_REQUIRE(sample.at("warp_a_to_b").sizes() == torch::IntArrayRef({4, 4, 2}));
    PFM_REQUIRE(sample.at("valid_mask").sizes() == torch::IntArrayRef({4, 4}));
    PFM_REQUIRE(sample.at("valid_mask").all().item<bool>());
    PFM_REQUIRE_CLOSE(sample.at("warp_a_to_b").index({0, 0, 0}).item<float>(), 0.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(sample.at("warp_a_to_b").index({3, 3, 1}).item<float>(), 3.0F, 1.0e-5F);
    PFM_REQUIRE(sample.at("view_a").index({0, 0, 0}).item<float>() > 0.0F);
}

void lazy_pose_pair_dataset_feeds_async_dataloader_prefetch()
{
    TempLazyPosePairDirectory temp("pfm_lazy_pose_loader");
    auto config = makeTinyDatasetFiles(temp, 2);
    auto dataset = std::make_shared<pfm::LazyPosePairDataset>(config);
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 2;
    options.prefetch_batches = 2;
    pfm::AsyncDataLoader loader(dataset, std::make_unique<pfm::SequentialSampler>(dataset->size()),
                                pfm::makeLazyPosePairCollator(), options);

    auto batch = loader.next();
    PFM_REQUIRE(batch.has_value());
    PFM_REQUIRE(batch->at("view_a").sizes() == torch::IntArrayRef({2, 1, 4, 4}));
    PFM_REQUIRE(batch->at("warp_a_to_b").sizes() == torch::IntArrayRef({2, 4, 4, 2}));
    PFM_REQUIRE(!loader.next().has_value());
}

} // namespace

void register_lazy_pose_pair_dataset_tests()
{
    register_test("lazy_pose_pair_manifest_builds_default_nadir_targets",
                  lazy_pose_pair_manifest_builds_default_nadir_targets);
    register_test("lazy_pose_pair_project_warp_identity_camera_returns_pixel_grid",
                  lazy_pose_pair_project_warp_identity_camera_returns_pixel_grid);
    register_test("lazy_pose_pair_project_warp_uses_projected_b_depth_for_visibility",
                  lazy_pose_pair_project_warp_uses_projected_b_depth_for_visibility);
    register_test("lazy_pose_pair_dataset_loads_cropped_uint8_pair",
                  lazy_pose_pair_dataset_loads_cropped_uint8_pair);
    register_test("lazy_pose_pair_dataset_feeds_async_dataloader_prefetch",
                  lazy_pose_pair_dataset_feeds_async_dataloader_prefetch);
}
