#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "tests/test_harness.h"
#include "train/trainer.h"

namespace pfm::testing {

torch::Tensor resize_offsets_for_dense_head_for_test(const torch::Tensor& warp, const torch::Tensor& offsets);
torch::Tensor make_sparse_descriptor_loss_for_test(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b);
torch::Tensor make_descriptor_sample_indices_for_test(const torch::Tensor& descriptors);
torch::Tensor limit_training_image_size_for_test(const torch::Tensor& image);

}  // namespace pfm::testing

namespace {

class TempTrainingDirectory {
public:
    explicit TempTrainingDirectory(const std::string& stem) {
        const auto suffix = std::to_string(static_cast<long long>(getpid())) + "_" +
                            std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempTrainingDirectory() {
        for (const auto& file_path : _files) {
            std::remove(file_path.string().c_str());
        }
        std::filesystem::remove(_path);
    }

    const std::filesystem::path& path() const {
        return _path;
    }

    std::filesystem::path file(const std::string& name) {
        auto file_path = _path / name;
        _files.push_back(file_path);
        return file_path;
    }

private:
    std::filesystem::path _path;
    std::vector<std::filesystem::path> _files;
};

void require_image_written(const std::filesystem::path& path, int offset) {
    cv::Mat image(32, 32, CV_8UC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 7 + y * 11 + offset) % 256);
        }
    }
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

pfm::TrainConfig tiny_config(const TempTrainingDirectory& temp_dir) {
    pfm::TrainConfig config;
    config.image_dir = temp_dir.path().string();
    config.checkpoint = (temp_dir.path() / "checkpoint.pt").string();
    config.epochs = 1;
    config.batch_size = 1;
    config.base_channels = 2;
    config.descriptor_dim = 4;
    config.learning_rate = 1.0e-3;
    return config;
}

}  // namespace

static void trainer_one_epoch_saves_loadable_checkpoint() {
    TempTrainingDirectory temp_dir("pfm_trainer_checkpoint");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.initial_loss > 0.0);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
}

static void trainer_missing_image_dir_throws_invalid_argument() {
    TempTrainingDirectory temp_dir("pfm_trainer_missing_dir");
    auto config = tiny_config(temp_dir);
    config.image_dir = (temp_dir.path() / "missing").string();
    temp_dir.file("checkpoint.pt");

    PFM_REQUIRE_INVALID_ARG(pfm::train_model(config));
}

static void trainer_invalid_numeric_parameters_throw_invalid_argument() {
    TempTrainingDirectory temp_dir("pfm_trainer_invalid_numeric");
    require_image_written(temp_dir.file("image.png"), 0);
    auto config = tiny_config(temp_dir);
    temp_dir.file("checkpoint.pt");

    auto invalid_epochs = config;
    invalid_epochs.epochs = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_epochs));

    auto invalid_batch = config;
    invalid_batch.batch_size = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_batch));

    auto invalid_base_channels = config;
    invalid_base_channels.base_channels = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_base_channels));

    auto invalid_descriptor_dim = config;
    invalid_descriptor_dim.descriptor_dim = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_descriptor_dim));

    auto invalid_learning_rate = config;
    invalid_learning_rate.learning_rate = 0.0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_learning_rate));
}

static void trainer_resizes_dense_warp_as_local_offsets() {
    auto xy = torch::meshgrid({torch::arange(32, torch::kFloat32), torch::arange(32, torch::kFloat32)}, "ij");
    auto warp = torch::stack({xy[1] + 1.0F, xy[0]}, 2).unsqueeze(0);
    auto offsets = torch::zeros({1, 2, 16, 16}, torch::kFloat32);

    auto target = pfm::testing::resize_offsets_for_dense_head_for_test(warp, offsets);
    auto expected_x = torch::full({16, 16}, 0.5F);
    auto expected_y = torch::zeros({16, 16});

    PFM_REQUIRE(target.sizes() == offsets.sizes());
    PFM_REQUIRE(torch::allclose(target.index({0, 0}), expected_x, 1.0e-6, 1.0e-6));
    PFM_REQUIRE(torch::allclose(target.index({0, 1}), expected_y, 1.0e-6, 1.0e-6));
}

static void trainer_bounds_descriptor_loss_spatial_samples() {
    const int64_t height = 80;
    const int64_t width = 80;
    auto grid = torch::arange(height * width, torch::kFloat32).reshape({1, 1, height, width});
    auto descriptors_a = torch::cat({grid, grid + 1.0F, grid + 2.0F, grid + 3.0F}, 1);
    auto descriptors_b = descriptors_a.clone();

    auto sample_indices = pfm::testing::make_descriptor_sample_indices_for_test(descriptors_a);
    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b);

    PFM_REQUIRE(sample_indices.size(0) == 1024);
    PFM_REQUIRE(sample_indices[0].item<int64_t>() == 0);
    PFM_REQUIRE(sample_indices[-1].item<int64_t>() == height * width - 1);
    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(loss.dim() == 0);
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
}

static void trainer_limits_large_training_image_edge() {
    auto image = torch::zeros({1, 900, 600}, torch::kFloat32);

    auto resized = pfm::testing::limit_training_image_size_for_test(image);

    PFM_REQUIRE(resized.sizes() == torch::IntArrayRef({1, 64, 43}));
    PFM_REQUIRE(resized.is_contiguous());
}

void register_trainer_tests() {
    register_test("trainer_one_epoch_saves_loadable_checkpoint", trainer_one_epoch_saves_loadable_checkpoint);
    register_test("trainer_missing_image_dir_throws_invalid_argument",
                  trainer_missing_image_dir_throws_invalid_argument);
    register_test("trainer_invalid_numeric_parameters_throw_invalid_argument",
                  trainer_invalid_numeric_parameters_throw_invalid_argument);
    register_test("trainer_resizes_dense_warp_as_local_offsets",
                  trainer_resizes_dense_warp_as_local_offsets);
    register_test("trainer_bounds_descriptor_loss_spatial_samples",
                  trainer_bounds_descriptor_loss_spatial_samples);
    register_test("trainer_limits_large_training_image_edge", trainer_limits_large_training_image_edge);
}
