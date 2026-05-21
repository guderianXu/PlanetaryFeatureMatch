#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "data/synthetic_pair_cache.h"
#include "dataloader/sampler.h"
#include "infer/feature_codec.h"
#include "infer/match_codec.h"
#include "models/planetary_graph_matcher.h"
#include "models/sparse_head.h"
#include "tests/test_harness.h"
#include "train/trainer.h"

namespace pfm::testing {

torch::Tensor resize_offsets_for_dense_head_for_test(const torch::Tensor& warp, const torch::Tensor& offsets);
torch::Tensor make_sparse_descriptor_loss_for_test(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
torch::Tensor make_dense_descriptor_hard_negative_loss_for_test(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
torch::Tensor make_warp_descriptor_contrastive_loss_for_test(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
torch::Tensor make_direct_full_map_descriptor_loss_for_test(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
torch::Tensor make_descriptor_map_regularization_loss_for_test(const torch::Tensor& descriptors);
torch::Tensor make_descriptor_target_coordinates_for_test(
    const torch::Tensor& warp,
    const torch::Tensor& sample_indices,
    int64_t descriptor_height,
    int64_t descriptor_width);
torch::Tensor sample_warped_descriptors_for_test(
    const torch::Tensor& descriptors,
    const torch::Tensor& target_coordinates);
torch::Tensor make_graph_matching_loss_for_test(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
torch::Tensor assign_graph_matching_targets_for_test(
    const torch::Tensor& keypoints_a,
    const torch::Tensor& keypoints_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double positive_radius_pixels);
torch::Tensor make_graph_candidate_indices_for_test(
    const torch::Tensor& target_indices,
    int64_t keypoint_count_b,
    int64_t max_candidates);
torch::Tensor make_keypoint_graph_matching_loss_for_test(
    PlanetaryGraphMatcherImpl& graph_matcher,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
torch::Tensor make_keypoint_descriptor_loss_for_test(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
torch::Tensor make_keypoint_dense_descriptor_loss_for_test(
    const FeatureSet& features_a,
    const torch::Tensor& descriptors_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask);
torch::Tensor make_orientation_supervision_loss_for_test(
    const SparseHeadOutput& sparse_a,
    const SparseHeadOutput& sparse_b,
    const torch::Tensor& view_a,
    const torch::Tensor& view_b,
    const torch::Tensor& warp,
    double min_keypoint_intensity);
torch::Tensor make_descriptor_sample_indices_for_test(const torch::Tensor& descriptors);
torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count);
int64_t training_variant_index_for_pair_for_test(
    std::size_t pair_index,
    std::size_t train_image_count,
    int epoch,
    int pairs_per_image);
torch::Tensor limit_training_image_size_for_test(const torch::Tensor& image, int64_t max_edge);
torch::Tensor stack_chw_batch_for_test(const std::vector<torch::Tensor>& tensors);
torch::Tensor stack_hw_batch_for_test(const std::vector<torch::Tensor>& tensors);
torch::Tensor stack_hwc_batch_for_test(const std::vector<torch::Tensor>& tensors);
torch::Tensor weighted_total_training_loss_for_test(
    const torch::Tensor& repeatability,
    const torch::Tensor& descriptor,
    const torch::Tensor& offset,
    const torch::Tensor& confidence,
    const torch::Tensor& descriptor_diversity = torch::tensor(0.0F));
torch::Tensor warp_heatmap_for_repeatability_for_test(const torch::Tensor& heatmap, const torch::Tensor& warp);
torch::Tensor make_training_valid_mask_for_test(
    const torch::Tensor& view_a,
    const torch::Tensor& view_b,
    const torch::Tensor& warp,
    const torch::Tensor& valid_mask,
    double min_keypoint_intensity);
torch::Tensor training_warp_overlay_image_for_test(const SyntheticPair& pair);
torch::Tensor training_feature_overlay_image_for_test(
    const torch::Tensor& image,
    const FeatureSet& features,
    double min_keypoint_intensity);
torch::Tensor training_match_overlay_image_for_test(
    const torch::Tensor& image_a,
    const torch::Tensor& image_b,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches);
torch::Tensor training_match_overlay_image_for_test(
    const torch::Tensor& image_a,
    const torch::Tensor& image_b,
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b,
    double correct_threshold_pixels);
std::string training_model_match_overlay_text_for_test(
    const FeatureSet& features_a,
    const FeatureSet& features_b,
    const MatchSet& matches,
    const torch::Tensor& warp_a_to_b,
    double correct_threshold_pixels);
bool should_enqueue_training_visualization_for_test(std::size_t enqueued_count, std::size_t visualization_limit);
bool should_use_online_dataloader_for_test(const pfm::TrainConfig& config);
std::vector<std::size_t> make_training_image_indices_for_test(
    std::size_t total_images,
    const pfm::TrainConfig& config);
std::vector<std::size_t> make_validation_image_indices_for_test(
    std::size_t total_images,
    const pfm::TrainConfig& config);

}  // namespace pfm::testing

namespace {

struct CoutCapture {
    std::ostringstream stream;
    std::streambuf* old = nullptr;

    CoutCapture() : old(std::cout.rdbuf(stream.rdbuf())) {}
    CoutCapture(const CoutCapture&) = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;
    CoutCapture(CoutCapture&&) = delete;
    CoutCapture& operator=(CoutCapture&&) = delete;

    ~CoutCapture() noexcept {
        try {
            if (old != nullptr) {
                std::cout.rdbuf(old);
            }
        } catch (...) {
        }
    }

    std::string str() const { return stream.str(); }
};

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
        std::error_code ignored;
        const auto cache_dir = _path / "pair_cache";
        if (std::filesystem::exists(cache_dir, ignored)) {
            for (const auto& entry : std::filesystem::directory_iterator(cache_dir)) {
                std::filesystem::remove(entry.path(), ignored);
            }
            std::filesystem::remove(cache_dir, ignored);
        }
        std::filesystem::remove(_path, ignored);
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

void require_sized_image_written(const std::filesystem::path& path, int height, int width, int offset) {
    cv::Mat image(height, width, CV_8UC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 7 + y * 11 + offset) % 256);
        }
    }
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

int64_t read_checkpoint_config_value(const std::string& checkpoint, const char* name) {
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive config_archive;
    archive.read("config", config_archive);
    torch::Tensor value;
    config_archive.read(name, value);
    PFM_REQUIRE(value.defined());
    return value.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
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
    config.train_ratio = 1.0;
    config.val_ratio = 0.0;
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

static void trainer_default_config_uses_larger_model_settings() {
    pfm::TrainConfig config;

    PFM_REQUIRE(config.base_channels == 32);
    PFM_REQUIRE(config.descriptor_dim == 128);
    PFM_REQUIRE(config.graph_hidden_dim == 256);
    PFM_REQUIRE(config.graph_attention_layers == 6);
    PFM_REQUIRE_CLOSE(config.learning_rate, 3.0e-4, 1.0e-9);
    PFM_REQUIRE_CLOSE(config.weight_decay, 5.0e-4, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.gradient_clip_norm, 1.0, 1.0e-12);
    PFM_REQUIRE(config.dataloader_workers == 0);
    PFM_REQUIRE(config.prefetch_batches == 2);
    PFM_REQUIRE(!config.pin_memory);
}

static void trainer_checkpoint_saves_graph_matcher_architecture_config() {
    TempTrainingDirectory temp_dir("pfm_trainer_checkpoint_architecture");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.graph_hidden_dim = 16;
    config.graph_attention_layers = 3;
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(read_checkpoint_config_value(config.checkpoint, "checkpoint_version") == 2);
    PFM_REQUIRE(read_checkpoint_config_value(config.checkpoint, "graph_hidden_dim") == 16);
    PFM_REQUIRE(read_checkpoint_config_value(config.checkpoint, "graph_attention_layers") == 3);
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

    auto invalid_resize = config;
    invalid_resize.resize = -1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_resize));

    auto invalid_pairs_per_image = config;
    invalid_pairs_per_image.pairs_per_image = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_pairs_per_image));
}

static void trainer_invalid_device_throws_invalid_argument() {
    TempTrainingDirectory temp_dir("pfm_trainer_invalid_device");
    require_image_written(temp_dir.file("image.png"), 0);
    auto config = tiny_config(temp_dir);
    config.device = "cuda:abc";
    temp_dir.file("checkpoint.pt");

    PFM_REQUIRE_INVALID_ARG(pfm::train_model(config));
}

static void trainer_cuda_device_is_strictly_validated() {
    if (torch::cuda::is_available()) {
        return;
    }

    TempTrainingDirectory temp_dir("pfm_trainer_cuda_unavailable");
    require_image_written(temp_dir.file("image.png"), 0);
    auto config = tiny_config(temp_dir);
    config.device = "cuda";
    temp_dir.file("checkpoint.pt");

    PFM_REQUIRE_INVALID_ARG(pfm::train_model(config));
}

static void trainer_cuda_one_epoch_saves_cpu_loadable_checkpoint_when_available() {
    if (!torch::cuda::is_available()) {
        return;
    }

    TempTrainingDirectory temp_dir("pfm_trainer_cuda_checkpoint");
    require_image_written(temp_dir.file("image.png"), 0);
    auto config = tiny_config(temp_dir);
    config.device = "cuda";
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
}

static void trainer_resizes_dense_warp_as_normalized_local_offsets() {
    auto xy = torch::meshgrid({torch::arange(32, torch::kFloat32), torch::arange(32, torch::kFloat32)}, "ij");
    auto warp = torch::stack({xy[1] + 16.0F, xy[0]}, 2).unsqueeze(0);
    auto offsets = torch::zeros({1, 2, 16, 16}, torch::kFloat32);

    auto target = pfm::testing::resize_offsets_for_dense_head_for_test(warp, offsets);
    auto expected_x = torch::ones({16, 16});
    auto expected_y = torch::zeros({16, 16});

    PFM_REQUIRE(target.sizes() == offsets.sizes());
    PFM_REQUIRE(torch::allclose(target.index({0, 0}), expected_x, 1.0e-6, 1.0e-6));
    PFM_REQUIRE(torch::allclose(target.index({0, 1}), expected_y, 1.0e-6, 1.0e-6));
}

static void trainer_repeatability_uses_warped_heatmap_correspondence() {
    auto heatmap_b = torch::tensor({{{{0.0F, 1.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto warp_a_to_b = torch::tensor({{{{1.0F, 0.0F}, {1.0F, 0.0F}}, {{1.0F, 1.0F}, {1.0F, 1.0F}}}}, torch::kFloat32);

    auto heatmap_b_at_a = pfm::testing::warp_heatmap_for_repeatability_for_test(heatmap_b, warp_a_to_b);

    PFM_REQUIRE(torch::allclose(heatmap_b_at_a.index({0, 0, 0, 0}), torch::tensor(1.0F), 1.0e-6, 1.0e-6));
}

static void trainer_descriptor_loss_uses_warped_correspondence() {
    auto descriptors_a = torch::tensor({{{{1.0F, 0.0F}}, {{0.0F, 1.0F}}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{{0.0F, 1.0F}}, {{1.0F, 0.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{1.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::ones({1, 1, 2}, torch::kBool);

    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() < 0.4F);
}

static void trainer_descriptor_loss_ignores_invalid_warp_targets() {
    auto descriptors_a = torch::tensor({{{{1.0F, 0.0F}}, {{0.0F, 1.0F}}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{{0.0F, 1.0F}}, {{1.0F, 0.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{0.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::tensor({{{false, true}}}, torch::kBool);

    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() < 0.4F);
}

static void trainer_descriptor_loss_penalizes_globally_collapsed_descriptors() {
    auto descriptors_a = torch::eye(16, torch::kFloat32).narrow(0, 0, 4).transpose(0, 1).reshape({1, 16, 1, 4});
    auto descriptors_b = descriptors_a.clone();
    auto collapsed_a = torch::ones({1, 16, 1, 4}, torch::kFloat32);
    auto collapsed_b = torch::ones({1, 16, 1, 4}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto distinctive_loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);
    auto collapsed_loss = pfm::testing::make_sparse_descriptor_loss_for_test(collapsed_a, collapsed_b, warp, valid_mask);

    PFM_REQUIRE(collapsed_loss.item<float>() > distinctive_loss.item<float>() + 1.0F);
}

static void trainer_dense_descriptor_hard_negative_loss_scans_full_map() {
    auto descriptors = torch::eye(81, torch::kFloat32).narrow(0, 0, 80);
    auto clean_a = descriptors.transpose(0, 1).reshape({1, 81, 1, 80});
    auto clean_b = clean_a.clone();
    auto hard_rows = descriptors.clone();
    hard_rows.index_put_({torch::indexing::Slice(40, 80)}, descriptors.index({torch::indexing::Slice(0, 40)}));
    auto hard_b = hard_rows.transpose(0, 1).reshape({1, 81, 1, 80});
    auto warp = torch::zeros({1, 1, 80, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(80, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 80}, torch::kBool);

    auto clean_loss = pfm::testing::make_dense_descriptor_hard_negative_loss_for_test(
        clean_a,
        clean_b,
        warp,
        valid_mask);
    auto hard_loss = pfm::testing::make_dense_descriptor_hard_negative_loss_for_test(
        clean_a,
        hard_b,
        warp,
        valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < 1.0e-4F);
    PFM_REQUIRE(hard_loss.item<float>() > clean_loss.item<float>() + 0.1F);
}

static torch::Tensor make_cyclic_safe_descriptor_row(int64_t width, int64_t group_shift = 0) {
    auto descriptors = torch::zeros({1, 16, 1, width}, torch::kFloat32);
    const auto group = ((group_shift % 4) + 4) % 4;
    for (int64_t x = 0; x < width; ++x) {
        descriptors.index_put_({0, group * 4 + x, 0, x}, 1.0F);
    }
    return descriptors;
}

static torch::Tensor make_cyclic_safe_flipped_descriptor_row(int64_t width, int64_t group_shift = 0) {
    auto descriptors = torch::zeros({1, 16, 1, width}, torch::kFloat32);
    const auto group = ((group_shift % 4) + 4) % 4;
    for (int64_t x = 0; x < width; ++x) {
        descriptors.index_put_({0, group * 4 + x, 0, width - 1 - x}, 1.0F);
    }
    return descriptors;
}

static torch::Tensor make_cyclic_safe_descriptor_grid(
    int64_t height,
    int64_t width,
    int64_t group_shift = 0,
    bool half_turn_spatial = false
) {
    auto descriptors = torch::zeros({1, 16, height, width}, torch::kFloat32);
    const auto group = ((group_shift % 4) + 4) % 4;
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            const auto identity = y * width + x;
            const auto target_y = half_turn_spatial ? height - 1 - y : y;
            const auto target_x = half_turn_spatial ? width - 1 - x : x;
            descriptors.index_put_({0, group * 4 + identity, target_y, target_x}, 1.0F);
        }
    }
    return descriptors;
}

static void trainer_warp_descriptor_contrastive_loss_uses_half_turn_correspondence() {
    auto descriptors_a = make_cyclic_safe_descriptor_grid(2, 2);
    auto correct_b = make_cyclic_safe_descriptor_grid(2, 2, 0, true);
    auto same_position_b = make_cyclic_safe_descriptor_grid(2, 2, 0, false);
    auto warp = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    warp.index_put_({0, 0, 0, 0}, 1.0F);
    warp.index_put_({0, 0, 0, 1}, 1.0F);
    warp.index_put_({0, 0, 1, 0}, 0.0F);
    warp.index_put_({0, 0, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 0, 0}, 1.0F);
    warp.index_put_({0, 1, 0, 1}, 0.0F);
    warp.index_put_({0, 1, 1, 0}, 0.0F);
    warp.index_put_({0, 1, 1, 1}, 0.0F);
    auto valid_mask = torch::ones({1, 2, 2}, torch::kBool);

    auto correct_loss = pfm::testing::make_warp_descriptor_contrastive_loss_for_test(
        descriptors_a, correct_b, warp, valid_mask);
    auto same_position_loss = pfm::testing::make_warp_descriptor_contrastive_loss_for_test(
        descriptors_a, same_position_b, warp, valid_mask);

    PFM_REQUIRE(same_position_loss.item<float>() > correct_loss.item<float>() + 5.0F);
}

static void trainer_warp_descriptor_contrastive_loss_accepts_cyclic_descriptor_shift() {
    auto descriptors_a = make_cyclic_safe_descriptor_row(4);
    auto descriptors_b = make_cyclic_safe_descriptor_row(4, 1);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto loss = pfm::testing::make_warp_descriptor_contrastive_loss_for_test(
        descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() > 1.0F);
}

static void trainer_direct_full_map_descriptor_loss_penalizes_global_distractor() {
    auto descriptors_a = make_cyclic_safe_descriptor_row(4);
    auto clean_b = make_cyclic_safe_flipped_descriptor_row(4, 0);
    auto hard_b = clean_b.clone();
    hard_b.index_put_(
        {0, torch::indexing::Slice(), 0, 0},
        clean_b.index({0, torch::indexing::Slice(), 0, 3}));

    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, 0, 0}, 3.0F);
    warp.index_put_({0, 0, 1, 0}, 2.0F);
    warp.index_put_({0, 0, 2, 0}, 1.0F);
    warp.index_put_({0, 0, 3, 0}, 0.0F);
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto clean_loss = pfm::testing::make_direct_full_map_descriptor_loss_for_test(
        descriptors_a, clean_b, warp, valid_mask);
    auto hard_loss = pfm::testing::make_direct_full_map_descriptor_loss_for_test(
        descriptors_a, hard_b, warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < 0.1F);
    PFM_REQUIRE(hard_loss.item<float>() > clean_loss.item<float>() + 0.1F);
}

static void trainer_direct_full_map_descriptor_loss_accepts_cyclic_descriptor_shift() {
    auto descriptors_a = make_cyclic_safe_descriptor_row(4);
    auto descriptors_b = make_cyclic_safe_descriptor_row(4, 1);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto loss = pfm::testing::make_direct_full_map_descriptor_loss_for_test(
        descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() > 1.0F);
}

static void trainer_descriptor_targets_use_cell_centers_for_warp_coordinates() {
    auto warp = torch::zeros({1, 4, 4, 2}, torch::kFloat32);
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 0},
                    torch::arange(4, torch::kFloat32).reshape({1, 4}).expand({4, 4}));
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 1},
                    torch::arange(4, torch::kFloat32).reshape({4, 1}).expand({4, 4}));
    auto sample_indices = torch::tensor({0, 3}, torch::kLong);

    auto coordinates = pfm::testing::make_descriptor_target_coordinates_for_test(warp, sample_indices, 1, 4);

    PFM_REQUIRE_CLOSE(coordinates.index({0, 0, 0}).item<float>(), 0.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(coordinates.index({0, 1, 0}).item<float>(), 3.0F, 1.0e-5F);
}

static void trainer_warped_descriptor_sampling_preserves_subpixel_correspondence() {
    auto descriptors = torch::zeros({1, 2, 1, 4}, torch::kFloat32);
    descriptors.index_put_({0, 0, 0, 1}, 1.0F);
    descriptors.index_put_({0, 1, 0, 2}, 1.0F);
    auto target_coordinates = torch::tensor({{{1.5F, 0.0F}}}, torch::kFloat32);

    auto sampled = pfm::testing::sample_warped_descriptors_for_test(descriptors, target_coordinates);

    PFM_REQUIRE_CLOSE(sampled.index({0, 0, 0}).item<float>(), 0.5F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(sampled.index({0, 0, 1}).item<float>(), 0.5F, 1.0e-5F);
}

static void trainer_descriptor_map_regularization_penalizes_spatial_collapse() {
    auto collapsed = torch::ones({1, 8, 4, 4}, torch::kFloat32);
    auto diverse = torch::zeros({1, 8, 4, 4}, torch::kFloat32);
    for (int64_t y = 0; y < 4; ++y) {
        for (int64_t x = 0; x < 4; ++x) {
            diverse.index_put_({0, (y * 4 + x) % 8, y, x}, 1.0F);
        }
    }

    auto collapsed_loss = pfm::testing::make_descriptor_map_regularization_loss_for_test(collapsed);
    auto diverse_loss = pfm::testing::make_descriptor_map_regularization_loss_for_test(diverse);

    PFM_REQUIRE(collapsed_loss.item<float>() > diverse_loss.item<float>() + 0.2F);
}

static pfm::FeatureSet make_keypoint_descriptor_feature_set(
    const torch::Tensor& keypoints,
    const torch::Tensor& descriptors
) {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    return pfm::FeatureSet{
        keypoints.clone().to(torch::kFloat32),
        torch::ones({keypoints.size(0)}, float_options),
        descriptors.clone().to(torch::kFloat32),
        torch::ones({keypoints.size(0)}, float_options),
        torch::zeros({keypoints.size(0), 2}, float_options),
        torch::zeros({keypoints.size(0), 2, 2}, float_options),
        torch::empty({0, 2}, float_options),
        torch::empty({0}, float_options),
        4,
        1};
}

static void trainer_keypoint_descriptor_loss_uses_sparse_keypoint_hard_negatives() {
    auto keypoints = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, torch::kFloat32);
    auto descriptors = torch::eye(16, torch::kFloat32).narrow(0, 0, 4);
    auto collapsed = torch::ones({4, 16}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto distinctive_loss = pfm::testing::make_keypoint_descriptor_loss_for_test(
        make_keypoint_descriptor_feature_set(keypoints, descriptors),
        make_keypoint_descriptor_feature_set(keypoints, descriptors),
        warp,
        valid_mask);
    auto collapsed_loss = pfm::testing::make_keypoint_descriptor_loss_for_test(
        make_keypoint_descriptor_feature_set(keypoints, collapsed),
        make_keypoint_descriptor_feature_set(keypoints, collapsed),
        warp,
        valid_mask);

    PFM_REQUIRE(collapsed_loss.item<float>() > distinctive_loss.item<float>() + 1.0F);
}

static void trainer_keypoint_descriptor_loss_penalizes_hardest_negative_margin() {
    auto keypoints = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32);
    auto query_descriptors = torch::tensor(
        {{1.0F, 0.0F}, {0.0F, 1.0F}, {-1.0F, 0.0F}},
        torch::kFloat32);
    auto easy_b = torch::tensor(
        {{1.0F, 0.0F}, {0.0F, 1.0F}, {-1.0F, 0.0F}},
        torch::kFloat32);
    auto hard_b = torch::tensor(
        {{1.0F, 0.0F}, {0.99F, 0.01F}, {-1.0F, 0.0F}},
        torch::kFloat32);
    auto warp = torch::zeros({1, 1, 3, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(3, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 3}, torch::kBool);

    auto easy_loss = pfm::testing::make_keypoint_descriptor_loss_for_test(
        make_keypoint_descriptor_feature_set(keypoints, query_descriptors),
        make_keypoint_descriptor_feature_set(keypoints, easy_b),
        warp,
        valid_mask);
    auto hard_loss = pfm::testing::make_keypoint_descriptor_loss_for_test(
        make_keypoint_descriptor_feature_set(keypoints, query_descriptors),
        make_keypoint_descriptor_feature_set(keypoints, hard_b),
        warp,
        valid_mask);

    PFM_REQUIRE(hard_loss.item<float>() > easy_loss.item<float>() + 0.1F);
}

static void trainer_keypoint_dense_descriptor_loss_uses_warp_target_in_full_map() {
    auto features_a = make_keypoint_descriptor_feature_set(
        torch::tensor({{0.0F, 0.0F}}, torch::kFloat32),
        torch::tensor({{1.0F, 0.0F}}, torch::kFloat32));
    features_a.feature_map_width = 4;
    features_a.feature_map_height = 1;
    auto clean_b = torch::zeros({1, 2, 1, 4}, torch::kFloat32);
    clean_b.index_put_({0, 0, 0, 0}, 1.0F);
    clean_b.index_put_({0, 1, 0, torch::indexing::Slice(1, 4)}, 1.0F);
    auto hard_b = clean_b.clone();
    hard_b.index_put_({0, 0, 0, 3}, 1.0F);
    hard_b.index_put_({0, 1, 0, 3}, 0.0F);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto clean_loss = pfm::testing::make_keypoint_dense_descriptor_loss_for_test(features_a, clean_b, warp, valid_mask);
    auto hard_loss = pfm::testing::make_keypoint_dense_descriptor_loss_for_test(features_a, hard_b, warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < 0.1F);
    PFM_REQUIRE(hard_loss.item<float>() > clean_loss.item<float>() + 0.5F);
}

static pfm::SparseHeadOutput make_sparse_orientation_output(const torch::Tensor& orientation) {
    return pfm::SparseHeadOutput{
        torch::empty({0}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32),
        torch::empty({0}, torch::kFloat32),
        orientation.clone().to(torch::kFloat32),
        torch::empty({0}, torch::kFloat32)};
}

static void trainer_orientation_supervision_uses_warp_rotation() {
    auto orientation_a = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    orientation_a.index_put_({0, 0, torch::indexing::Slice(), torch::indexing::Slice()}, 1.0F);
    auto correct_b = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    correct_b.index_put_({0, 1, torch::indexing::Slice(), torch::indexing::Slice()}, 1.0F);
    auto wrong_b = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    wrong_b.index_put_({0, 0, torch::indexing::Slice(), torch::indexing::Slice()}, -1.0F);

    auto view = torch::ones({1, 1, 4, 4}, torch::kFloat32);
    auto warp = torch::zeros({1, 4, 4, 2}, torch::kFloat32);
    for (int64_t x = 0; x < 4; ++x) {
        warp.index_put_({0, torch::indexing::Slice(), x, 1}, static_cast<float>(x));
    }

    auto correct_loss = pfm::testing::make_orientation_supervision_loss_for_test(
        make_sparse_orientation_output(orientation_a),
        make_sparse_orientation_output(correct_b),
        view,
        view,
        warp,
        0.05);
    auto wrong_loss = pfm::testing::make_orientation_supervision_loss_for_test(
        make_sparse_orientation_output(orientation_a),
        make_sparse_orientation_output(wrong_b),
        view,
        view,
        warp,
        0.05);

    PFM_REQUIRE(correct_loss.item<float>() < 1.0e-4F);
    PFM_REQUIRE(wrong_loss.item<float>() > correct_loss.item<float>() + 0.25F);
}

static void trainer_keypoint_descriptor_loss_covers_more_than_graph_query_limit() {
    const int64_t count = 300;
    auto keypoints = torch::stack(
        {torch::arange(count, torch::kFloat32), torch::zeros({count}, torch::kFloat32)},
        1);
    auto descriptors_a = torch::zeros({count, 2}, torch::kFloat32);
    auto descriptors_b = torch::zeros({count, 2}, torch::kFloat32);
    descriptors_a.index_put_({torch::indexing::Slice(), 0}, 1.0F);
    descriptors_b.index_put_({torch::indexing::Slice(), 0}, 1.0F);
    descriptors_a.index_put_({count - 1, 0}, 0.0F);
    descriptors_a.index_put_({count - 1, 1}, 1.0F);
    descriptors_b.index_put_({count - 1, 0}, 0.0F);
    descriptors_b.index_put_({count - 1, 1}, 1.0F);

    auto features_a = make_keypoint_descriptor_feature_set(keypoints, descriptors_a);
    auto features_b = make_keypoint_descriptor_feature_set(keypoints, descriptors_b);
    features_a.feature_map_width = count;
    features_b.feature_map_width = count;
    features_b.descriptors.set_requires_grad(true);
    auto warp = torch::zeros({1, 1, count, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(count, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, count}, torch::kBool);

    auto loss = pfm::testing::make_keypoint_descriptor_loss_for_test(features_a, features_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(features_b.descriptors.grad().defined());
    PFM_REQUIRE(features_b.descriptors.grad().index({count - 1}).abs().sum().item<float>() > 0.0F);
}

static void trainer_graph_matching_loss_trains_graph_matcher_parameters() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    auto descriptors_a = torch::tensor({{{{1.0F, 0.0F}}, {{0.0F, 1.0F}}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{{0.0F, 1.0F}}, {{1.0F, 0.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{1.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::ones({1, 1, 2}, torch::kBool);

    auto loss = pfm::testing::make_graph_matching_loss_for_test(*matcher, descriptors_a, descriptors_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(matcher->parameters().front().grad().defined());
    PFM_REQUIRE(matcher->parameters().front().grad().abs().sum().item<float>() > 0.0F);
}

static void trainer_graph_matching_loss_is_finite_with_many_descriptors() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    auto descriptors_a = torch::ones({1, 2, 1, 1026}, torch::kFloat32);
    auto descriptors_b = torch::ones({1, 2, 1, 1026}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 1026, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, 1025.0F);
    auto valid_mask = torch::ones({1, 1, 1026}, torch::kBool);

    auto loss = pfm::testing::make_graph_matching_loss_for_test(*matcher, descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
}


static void trainer_keypoint_graph_targets_use_warped_nearest_b_keypoint() {
    auto keypoints_a = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{5.0F, 1.0F}, {7.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto targets = pfm::testing::assign_graph_matching_targets_for_test(
        keypoints_a, keypoints_b, warp, valid_mask, 2.0);

    PFM_REQUIRE(targets.sizes() == std::vector<int64_t>({2}));
    PFM_REQUIRE(targets[0].item<int64_t>() == 0);
    PFM_REQUIRE(targets[1].item<int64_t>() == 1);
}

static void trainer_keypoint_graph_targets_use_dustbin_for_unmatched_keypoints() {
    auto keypoints_a = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{6.0F, 6.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto targets = pfm::testing::assign_graph_matching_targets_for_test(
        keypoints_a, keypoints_b, warp, valid_mask, 1.0);

    PFM_REQUIRE(targets[0].item<int64_t>() == 1);
    PFM_REQUIRE(targets[1].item<int64_t>() == 1);
}

static void trainer_keypoint_graph_targets_use_dustbin_for_invalid_target_pixels() {
    auto keypoints_a = torch::tensor({{1.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{5.0F, 1.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);
    valid_mask.index_put_({0, 1, 5}, false);

    auto targets = pfm::testing::assign_graph_matching_targets_for_test(
        keypoints_a, keypoints_b, warp, valid_mask, 2.0);

    PFM_REQUIRE(targets[0].item<int64_t>() == 1);
}

static void trainer_graph_candidates_include_positives_once_and_dustbin_last() {
    auto target_indices = torch::tensor({0, 2, 2, 5}, torch::kLong);

    auto candidates = pfm::testing::make_graph_candidate_indices_for_test(target_indices, 5, 6);

    PFM_REQUIRE(candidates.size(0) == 6);
    PFM_REQUIRE(candidates[-1].item<int64_t>() == 5);
    PFM_REQUIRE((candidates == 0).sum().item<int64_t>() == 1);
    PFM_REQUIRE((candidates == 2).sum().item<int64_t>() == 1);
    PFM_REQUIRE((candidates == 5).sum().item<int64_t>() == 1);
}

static void trainer_keypoint_graph_matching_loss_trains_graph_matcher_parameters() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    pfm::FeatureSet features_a;
    features_a.keypoints = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    features_a.scores = torch::tensor({1.0F, 0.9F}, torch::kFloat32);
    features_a.descriptors = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    features_a.feature_map_width = 8;
    features_a.feature_map_height = 8;
    pfm::FeatureSet features_b;
    features_b.keypoints = torch::tensor({{5.0F, 1.0F}, {7.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32);
    features_b.scores = torch::tensor({1.0F, 0.9F, 0.1F}, torch::kFloat32);
    features_b.descriptors = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {0.5F, 0.5F}}, torch::kFloat32);
    features_b.feature_map_width = 8;
    features_b.feature_map_height = 8;
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto loss = pfm::testing::make_keypoint_graph_matching_loss_for_test(
        *matcher, features_a, features_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
    PFM_REQUIRE(matcher->parameters().front().grad().defined());
    PFM_REQUIRE(matcher->parameters().front().grad().abs().sum().item<float>() > 0.0F);
}

static void trainer_keypoint_graph_matching_loss_uses_full_b_candidate_set() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    pfm::FeatureSet features_a;
    features_a.keypoints = torch::tensor({{1.0F, 1.0F}}, torch::kFloat32);
    features_a.scores = torch::tensor({1.0F}, torch::kFloat32);
    features_a.descriptors = torch::tensor({{1.0F, 0.0F}}, torch::kFloat32);
    features_a.feature_map_width = 80;
    features_a.feature_map_height = 8;

    std::vector<float> keypoint_values;
    std::vector<float> descriptor_values;
    keypoint_values.reserve(70 * 2);
    descriptor_values.reserve(70 * 2);
    for (int index = 0; index < 70; ++index) {
        keypoint_values.push_back(static_cast<float>(index + 5));
        keypoint_values.push_back(1.0F);
        descriptor_values.push_back(index == 0 ? 1.0F : 0.0F);
        descriptor_values.push_back(index == 0 ? 0.0F : 1.0F);
    }

    pfm::FeatureSet features_b;
    features_b.keypoints = torch::from_blob(keypoint_values.data(), {70, 2}, torch::kFloat32).clone();
    features_b.scores = torch::ones({70}, torch::kFloat32);
    features_b.descriptors = torch::from_blob(descriptor_values.data(), {70, 2}, torch::kFloat32).clone();
    features_b.descriptors.set_requires_grad(true);
    features_b.feature_map_width = 80;
    features_b.feature_map_height = 8;

    auto warp = torch::zeros({1, 8, 80, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 80}, torch::kBool);

    auto loss = pfm::testing::make_keypoint_graph_matching_loss_for_test(
        *matcher, features_a, features_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(features_b.descriptors.grad().defined());
    PFM_REQUIRE(features_b.descriptors.grad().index({69}).abs().sum().item<float>() > 0.0F);
}

static void trainer_stacks_variable_spatial_training_tensors_with_padding() {
    auto chw = pfm::testing::stack_chw_batch_for_test(
        {torch::ones({1, 2, 3}, torch::kFloat32), torch::ones({1, 3, 2}, torch::kFloat32) * 2.0F});
    PFM_REQUIRE(chw.sizes() == std::vector<int64_t>({2, 1, 3, 3}));
    PFM_REQUIRE_CLOSE(chw.index({0, 0, 1, 2}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(chw.index({0, 0, 2, 2}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(chw.index({1, 0, 2, 1}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(chw.index({1, 0, 2, 2}).item<float>(), 0.0F, 1.0e-6F);

    auto hw = pfm::testing::stack_hw_batch_for_test(
        {torch::ones({2, 3}, torch::kBool), torch::zeros({3, 2}, torch::kBool)});
    PFM_REQUIRE(hw.sizes() == std::vector<int64_t>({2, 3, 3}));
    PFM_REQUIRE(hw.index({0, 1, 2}).item<bool>());
    PFM_REQUIRE(!hw.index({0, 2, 2}).item<bool>());

    auto warp_a = torch::ones({2, 3, 2}, torch::kFloat32);
    auto warp_b = torch::ones({3, 2, 2}, torch::kFloat32) * 2.0F;
    auto hwc = pfm::testing::stack_hwc_batch_for_test({warp_a, warp_b});
    PFM_REQUIRE(hwc.sizes() == std::vector<int64_t>({2, 3, 3, 2}));
    PFM_REQUIRE_CLOSE(hwc.index({0, 1, 2, 1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(hwc.index({0, 2, 2, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(hwc.index({1, 2, 1, 1}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(hwc.index({1, 2, 2, 1}).item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_training_valid_mask_requires_bright_source_and_target_pixels() {
    auto view_a = torch::tensor({{{{0.0F, 1.0F}, {1.0F, 1.0F}}}}, torch::kFloat32);
    auto view_b = torch::tensor({{{{1.0F, 0.0F}, {1.0F, 1.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{0.0F, 0.0F}, {1.0F, 0.0F}}, {{0.0F, 1.0F}, {1.0F, 1.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::ones({1, 2, 2}, torch::kBool);

    auto masked = pfm::testing::make_training_valid_mask_for_test(view_a, view_b, warp, valid_mask, 0.5);
    auto expected = torch::tensor({{{false, false}, {true, true}}}, torch::kBool);

    PFM_REQUIRE(torch::equal(masked, expected));
}

static void trainer_descriptor_candidates_do_not_repeat_positive_target() {
    auto target_indices = torch::tensor({{0, 4}}, torch::kLong);

    auto candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, 5);

    PFM_REQUIRE(candidates.size(2) == 5);
    PFM_REQUIRE(candidates.index({0, 0, 0}).item<int64_t>() == 0);
    PFM_REQUIRE(candidates.index({0, 1, 0}).item<int64_t>() == 4);
    PFM_REQUIRE(candidates.index({0, 0, 1}).item<int64_t>() != 0);
    PFM_REQUIRE(candidates.index({0, 0, 2}).item<int64_t>() != 0);
    PFM_REQUIRE(candidates.index({0, 0, 3}).item<int64_t>() != 0);
    PFM_REQUIRE(candidates.index({0, 0, 4}).item<int64_t>() != 0);
    PFM_REQUIRE(candidates.index({0, 1, 1}).item<int64_t>() != 4);
    PFM_REQUIRE(candidates.index({0, 1, 2}).item<int64_t>() != 4);
    PFM_REQUIRE(candidates.index({0, 1, 3}).item<int64_t>() != 4);
    PFM_REQUIRE(candidates.index({0, 1, 4}).item<int64_t>() != 4);
}

static void trainer_bounds_descriptor_loss_spatial_samples() {
    const int64_t height = 80;
    const int64_t width = 80;
    auto grid = torch::arange(height * width, torch::kFloat32).reshape({1, 1, height, width});
    auto descriptors_a = torch::cat({grid, grid + 1.0F, grid + 2.0F, grid + 3.0F}, 1);
    auto descriptors_b = descriptors_a.clone();
    auto xy = torch::meshgrid({torch::arange(height, torch::kFloat32), torch::arange(width, torch::kFloat32)}, "ij");
    auto warp = torch::stack({xy[1], xy[0]}, 2).unsqueeze(0);
    auto valid_mask = torch::ones({1, height, width}, torch::kBool);

    auto sample_indices = pfm::testing::make_descriptor_sample_indices_for_test(descriptors_a);
    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(sample_indices.size(0) == 1024);
    PFM_REQUIRE(sample_indices.min().item<int64_t>() >= 0);
    PFM_REQUIRE(sample_indices.max().item<int64_t>() < height * width);
    auto sorted_indices = std::get<0>(sample_indices.sort());
    PFM_REQUIRE(sorted_indices.slice(0, 1).ne(sorted_indices.slice(0, 0, -1)).all().item<bool>());
    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(loss.dim() == 0);
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
}

static void trainer_resizes_large_training_image() {
    auto image = torch::zeros({1, 900, 600}, torch::kFloat32);

    auto resized = pfm::testing::limit_training_image_size_for_test(image, 64);

    PFM_REQUIRE(resized.sizes() == torch::IntArrayRef({1, 64, 43}));
    PFM_REQUIRE(resized.is_contiguous());
}

static void trainer_uses_configured_resize() {
    auto image = torch::zeros({1, 900, 600}, torch::kFloat32);

    auto resized = pfm::testing::limit_training_image_size_for_test(image, 300);

    PFM_REQUIRE(resized.sizes() == torch::IntArrayRef({1, 300, 200}));
    PFM_REQUIRE(resized.is_contiguous());
}

static void trainer_training_and_validation_indices_use_dataloader_split() {
    pfm::TrainConfig config;
    config.train_ratio = 0.6;
    config.val_ratio = 0.2;
    config.split_seed = 7;

    const auto train = pfm::testing::make_training_image_indices_for_test(10, config);
    const auto validation = pfm::testing::make_validation_image_indices_for_test(10, config);
    const auto split = pfm::make_train_validation_test_split(10, 0.6, 0.2, 0.2, 7, true);

    PFM_REQUIRE(train == split.train);
    PFM_REQUIRE(validation == split.validation);
}

static void trainer_variant_indices_advance_across_epochs() {
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(0, 1, 0, 8) == 0);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(7, 1, 0, 8) == 7);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(0, 1, 1, 8) == 8);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(7, 1, 1, 8) == 15);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(0, 2, 1, 8) == 8);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(2, 2, 1, 8) == 9);
}

static void trainer_total_loss_downweights_dense_offset_pixels() {
    auto repeatability = torch::tensor(1.0F);
    auto descriptor = torch::tensor(2.0F);
    auto offset = torch::tensor(30.0F);
    auto confidence = torch::tensor(4.0F);

    auto loss = pfm::testing::weighted_total_training_loss_for_test(repeatability, descriptor, offset, confidence);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 2.0F, 1.0e-6F);
}

static void trainer_total_loss_penalizes_descriptor_spatial_collapse() {
    auto repeatability = torch::tensor(1.0F);
    auto descriptor = torch::tensor(2.0F);
    auto offset = torch::tensor(3.0F);
    auto confidence = torch::tensor(4.0F);
    auto descriptor_diversity = torch::tensor(1.0F);

    auto loss = pfm::testing::weighted_total_training_loss_for_test(
        repeatability,
        descriptor,
        offset,
        confidence,
        descriptor_diversity);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 102.0F, 1.0e-6F);
}

static void trainer_progress_reports_loss_components() {
    TempTrainingDirectory temp_dir("pfm_trainer_loss_components");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    CoutCapture capture;
    auto result = pfm::train_model(config);
    const auto output = capture.str();

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(output.find("loss=") != std::string::npos);
    PFM_REQUIRE(output.find("match=") != std::string::npos);
    PFM_REQUIRE(output.find("feat=") != std::string::npos);
    PFM_REQUIRE(output.find("dense=") != std::string::npos);
    PFM_REQUIRE(output.find("off=") != std::string::npos);
    PFM_REQUIRE(output.find("epoch summary") != std::string::npos);
}

static void trainer_reports_epoch_and_batch_timing() {
    TempTrainingDirectory temp_dir("pfm_trainer_timing");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    CoutCapture capture;
    const auto result = pfm::train_model(config);
    const auto output = capture.str();

    PFM_REQUIRE(result.total_time_seconds > 0.0);
    PFM_REQUIRE(result.avg_batch_time_seconds > 0.0);
    PFM_REQUIRE(output.find("elapsed=") != std::string::npos);
}

static void trainer_writes_csv_metric_log() {
    TempTrainingDirectory temp_dir("pfm_trainer_csv_metrics");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    config.log_csv = (temp_dir.path() / "metrics.csv").string();
    temp_dir.file("checkpoint.pt");

    const auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    std::ifstream input(config.log_csv);
    std::string header;
    std::string row;
    std::getline(input, header);
    std::getline(input, row);
    PFM_REQUIRE(header.find("loss_total") != std::string::npos);
    PFM_REQUIRE(header.find("graph_matching_loss") != std::string::npos);
    PFM_REQUIRE(header.find("offset_error_px") != std::string::npos);
    PFM_REQUIRE(row.find("1,1,1,2") == 0);
}

static void trainer_uses_online_dataloader_when_workers_requested() {
    pfm::TrainConfig config;
    config.dataloader_workers = 2;
    config.synthetic_pair_cache_dir.clear();

    PFM_REQUIRE(pfm::testing::should_use_online_dataloader_for_test(config));
}

static void trainer_trains_with_online_dataloader_workers() {
    TempTrainingDirectory temp_dir("pfm_trainer_online_dataloader");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    config.dataloader_workers = 2;
    config.prefetch_batches = 2;
    temp_dir.file("checkpoint.pt");

    const auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
}

static void trainer_trains_full_dataset() {
    TempTrainingDirectory temp_dir("pfm_trainer_full_dataset");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    require_image_written(temp_dir.file("image_c.png"), 73);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
}

static void trainer_with_synthetic_pair_cache_writes_cache_and_checkpoint() {
    TempTrainingDirectory temp_dir("pfm_trainer_pair_cache");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) / "manifest.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) / "source_000000_image_a" / "pair_000000.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) / "source_000000_image_a" / "source_000000_view_a.png"));
}

static void trainer_pairs_per_image_expands_cached_training_pairs() {
    TempTrainingDirectory temp_dir("pfm_trainer_pairs_per_image_cache");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    config.pairs_per_image = 2;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) / "source_000001_image_b" / "pair_000003.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) / "source_000001_image_b" / "source_000001_view_a.png"));
}

static void trainer_reuses_existing_synthetic_pair_cache() {
    TempTrainingDirectory temp_dir("pfm_trainer_pair_cache_reuse");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    config.resize = 32;
    temp_dir.file("checkpoint.pt");
    (void)pfm::train_model(config);
    const auto pair_path = std::filesystem::path(config.synthetic_pair_cache_dir) / "source_000000_image_a" / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.checkpoint = (temp_dir.path() / "checkpoint_2.pt").string();
    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) == first_write_time);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
}

static void trainer_rebuilds_synthetic_pair_cache_when_requested() {
    TempTrainingDirectory temp_dir("pfm_trainer_pair_cache_rebuild");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    config.resize = 32;
    temp_dir.file("checkpoint.pt");
    (void)pfm::train_model(config);
    const auto pair_path = std::filesystem::path(config.synthetic_pair_cache_dir) / "source_000000_image_a" / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.checkpoint = (temp_dir.path() / "checkpoint_2.pt").string();
    config.synthetic_pair_cache_rebuild = true;
    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) != first_write_time);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
}

static float cached_pair_mean_displacement(const std::string& cache_dir, std::size_t index) {
    pfm::SyntheticPairCacheDataset cache_dataset(cache_dir);
    const auto pair = cache_dataset.load(index);
    const auto xy = torch::meshgrid(
        {torch::arange(pair.warp_a_to_b.size(0), torch::kFloat32),
         torch::arange(pair.warp_a_to_b.size(1), torch::kFloat32)},
        "ij");
    const auto grid = torch::stack({xy[1], xy[0]}, 2);
    return (pair.warp_a_to_b - grid).norm(2, 2).mean().item<float>();
}

static void trainer_forwards_augmentation_profile_to_cached_pairs() {
    TempTrainingDirectory temp_dir("pfm_trainer_profile_forwarding");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto mild_config = tiny_config(temp_dir);
    mild_config.synthetic_pair_cache_dir = (temp_dir.path() / "mild_cache").string();
    mild_config.augmentation_profile = "mild";
    mild_config.pairs_per_image = 2;
    mild_config.resize = 32;
    temp_dir.file("mild_checkpoint.pt");
    (void)pfm::train_model(mild_config);

    auto extreme_config = tiny_config(temp_dir);
    extreme_config.checkpoint = (temp_dir.path() / "extreme_checkpoint.pt").string();
    extreme_config.synthetic_pair_cache_dir = (temp_dir.path() / "extreme_cache").string();
    extreme_config.augmentation_profile = "extreme";
    extreme_config.pairs_per_image = 2;
    extreme_config.resize = 32;
    temp_dir.file("extreme_checkpoint.pt");
    (void)pfm::train_model(extreme_config);

    PFM_REQUIRE(cached_pair_mean_displacement(extreme_config.synthetic_pair_cache_dir, 2) >
                cached_pair_mean_displacement(mild_config.synthetic_pair_cache_dir, 2) * 2.0F);
}

static void trainer_visualization_writes_expected_pngs_for_sampled_pair() {
    TempTrainingDirectory temp_dir("pfm_trainer_visualization_sampled");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.visualization_dir = (temp_dir.path() / "train_vis").string();
    config.visualization_samples = 1;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    (void)pfm::train_model(config);
    const auto vis_dir = std::filesystem::path(config.visualization_dir);

    const auto static_dir = vis_dir / "static";
    const auto epoch_dir = vis_dir / "epoch_000001";
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_view_b.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_valid_mask.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_warp_matches.png"));
    PFM_REQUIRE(!std::filesystem::exists(epoch_dir / "pair_000000_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000000_features_a.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000000_features_b.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000000_model_matches.png"));
    PFM_REQUIRE(!std::filesystem::exists(static_dir / "pair_000001_view_a.png"));
}

static void trainer_visualization_all_writes_every_pair() {
    TempTrainingDirectory temp_dir("pfm_trainer_visualization_all");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.visualization_dir = (temp_dir.path() / "train_vis").string();
    config.visualization_samples_all = true;
    config.pairs_per_image = 2;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    (void)pfm::train_model(config);
    const auto vis_dir = std::filesystem::path(config.visualization_dir);

    const auto static_dir = vis_dir / "static";
    const auto epoch_dir = vis_dir / "epoch_000001";
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000001_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000002_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000003_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000000_model_matches.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000003_model_matches.png"));
}

static void trainer_visualization_zero_samples_writes_no_pngs() {
    TempTrainingDirectory temp_dir("pfm_trainer_visualization_zero");
    require_image_written(temp_dir.file("image_a.png"), 0);
    auto config = tiny_config(temp_dir);
    config.visualization_dir = (temp_dir.path() / "train_vis").string();
    config.visualization_samples = 0;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    (void)pfm::train_model(config);

    PFM_REQUIRE(!std::filesystem::exists(std::filesystem::path(config.visualization_dir) / "epoch_000001"));
}

static void trainer_visualization_writes_sampled_pair_for_each_epoch() {
    TempTrainingDirectory temp_dir("pfm_trainer_visualization_each_epoch");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.epochs = 2;
    config.visualization_dir = (temp_dir.path() / "train_vis").string();
    config.visualization_samples = 1;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    (void)pfm::train_model(config);
    const auto vis_dir = std::filesystem::path(config.visualization_dir);

    PFM_REQUIRE(std::filesystem::exists(vis_dir / "epoch_000001" / "pair_000000_model_matches.png"));
    PFM_REQUIRE(std::filesystem::exists(vis_dir / "epoch_000002" / "pair_000000_model_matches.png"));
    PFM_REQUIRE(!std::filesystem::exists(vis_dir / "epoch_000002" / "pair_000001_view_a.png"));
}

static void trainer_visualization_warp_overlay_does_not_mutate_source_pair() {
    pfm::SyntheticPair pair;
    pair.view_a = torch::zeros({1, 8, 8}, torch::kFloat32);
    pair.view_b = torch::zeros({1, 8, 8}, torch::kFloat32);
    pair.warp_a_to_b = torch::zeros({8, 8, 2}, torch::kFloat32);
    pair.valid_mask = torch::ones({8, 8}, torch::kFloat32);

    (void)pfm::testing::training_warp_overlay_image_for_test(pair);

    PFM_REQUIRE_CLOSE(pair.view_a.max().item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_visualization_feature_overlay_suppresses_dark_pixels() {
    auto image = torch::zeros({1, 8, 8}, torch::kFloat32);
    image.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(4, 8)}, 0.5F);
    pfm::FeatureSet features;
    features.keypoints = torch::tensor({{1.0F, 3.0F}, {5.0F, 3.0F}}, torch::kFloat32);
    features.feature_map_width = 8;
    features.feature_map_height = 8;

    const auto overlay = pfm::testing::training_feature_overlay_image_for_test(image, features, 0.05);

    PFM_REQUIRE(overlay.size(0) == 3);
    PFM_REQUIRE_CLOSE(overlay.index({0, 3, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 3, 5}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({1, 3, 5}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({2, 3, 5}).item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_visualization_model_matches_uses_side_by_side_canvas() {
    auto image_a = torch::zeros({1, 16, 16}, torch::kFloat32);
    auto image_b = torch::zeros({1, 16, 16}, torch::kFloat32);
    pfm::FeatureSet features_a;
    features_a.keypoints = torch::tensor({{1.0F, 1.0F}}, torch::kFloat32);
    features_a.feature_map_width = 8;
    features_a.feature_map_height = 8;
    pfm::FeatureSet features_b;
    features_b.keypoints = torch::tensor({{6.0F, 6.0F}}, torch::kFloat32);
    features_b.feature_map_width = 8;
    features_b.feature_map_height = 8;
    pfm::MatchSet matches;
    matches.sparse_matches = torch::tensor({{0, 0}}, torch::kLong);
    matches.points_a = torch::tensor({{1.0F, 6.0F}}, torch::kFloat32);
    matches.points_b = torch::tensor({{6.0F, 6.0F}}, torch::kFloat32);
    matches.confidence = torch::ones({1}, torch::kFloat32);

    const auto overlay = pfm::testing::training_match_overlay_image_for_test(
        image_a, image_b, features_a, features_b, matches);

    PFM_REQUIRE(overlay.size(1) == 16);
    PFM_REQUIRE(overlay.size(2) == 32);
    PFM_REQUIRE_CLOSE(overlay.index({0, 2, 2}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 12, 28}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 12, 2}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 12, 28}).item<float>(), 1.0F, 1.0e-6F);
}

static pfm::FeatureSet make_match_color_features(const torch::Tensor& keypoints) {
    pfm::FeatureSet features;
    features.keypoints = keypoints;
    features.feature_map_width = 8;
    features.feature_map_height = 8;
    return features;
}

static torch::Tensor make_match_color_warp() {
    auto warp = torch::zeros({8, 8, 2}, torch::kFloat32);
    warp.index_put_({1, 1, 0}, 6.0F);
    warp.index_put_({1, 1, 1}, 1.0F);
    warp.index_put_({6, 1, 0}, 1.0F);
    warp.index_put_({6, 1, 1}, 6.0F);
    return warp;
}

static void trainer_visualization_model_matches_colors_sparse_correct_and_wrong_lines() {
    auto image_a = torch::zeros({1, 8, 8}, torch::kFloat32);
    auto image_b = torch::zeros({1, 8, 8}, torch::kFloat32);
    const auto features_a = make_match_color_features(torch::tensor({{1.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32));
    const auto features_b = make_match_color_features(torch::tensor({{6.0F, 1.0F}, {6.0F, 6.0F}}, torch::kFloat32));
    pfm::MatchSet matches;
    matches.sparse_matches = torch::tensor({{0, 0}, {1, 1}}, torch::kLong);

    const auto overlay = pfm::testing::training_match_overlay_image_for_test(
        image_a, image_b, features_a, features_b, matches, make_match_color_warp(), 1.0);

    PFM_REQUIRE_CLOSE(overlay.index({0, 1, 10}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({1, 1, 10}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({2, 1, 10}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 6, 10}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({1, 6, 10}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({2, 6, 10}).item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_visualization_model_matches_text_includes_correct_and_wrong_counts() {
    const auto features_a = make_match_color_features(torch::tensor({{1.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32));
    const auto features_b = make_match_color_features(torch::tensor({{6.0F, 1.0F}, {6.0F, 6.0F}}, torch::kFloat32));
    pfm::MatchSet matches;
    matches.sparse_matches = torch::tensor({{0, 0}, {1, 1}}, torch::kLong);
    matches.points_a = torch::tensor({{1.0F, 1.0F}}, torch::kFloat32);
    matches.points_b = torch::tensor({{6.0F, 1.0F}}, torch::kFloat32);

    const auto text = pfm::testing::training_model_match_overlay_text_for_test(
        features_a, features_b, matches, make_match_color_warp(), 1.0);

    PFM_REQUIRE(text.find("correct_matches=2") != std::string::npos);
    PFM_REQUIRE(text.find("wrong_matches=1") != std::string::npos);
}

void register_trainer_tests() {
    register_test("trainer_one_epoch_saves_loadable_checkpoint", trainer_one_epoch_saves_loadable_checkpoint);
    register_test("trainer_default_config_uses_larger_model_settings",
                  trainer_default_config_uses_larger_model_settings);
    register_test("trainer_checkpoint_saves_graph_matcher_architecture_config",
                  trainer_checkpoint_saves_graph_matcher_architecture_config);
    register_test("trainer_missing_image_dir_throws_invalid_argument",
                  trainer_missing_image_dir_throws_invalid_argument);
    register_test("trainer_invalid_numeric_parameters_throw_invalid_argument",
                  trainer_invalid_numeric_parameters_throw_invalid_argument);
    register_test("trainer_invalid_device_throws_invalid_argument", trainer_invalid_device_throws_invalid_argument);
    register_test("trainer_cuda_device_is_strictly_validated", trainer_cuda_device_is_strictly_validated);
    register_test(
        "trainer_cuda_one_epoch_saves_cpu_loadable_checkpoint_when_available",
        trainer_cuda_one_epoch_saves_cpu_loadable_checkpoint_when_available);
    register_test("trainer_resizes_dense_warp_as_normalized_local_offsets",
                  trainer_resizes_dense_warp_as_normalized_local_offsets);
    register_test("trainer_repeatability_uses_warped_heatmap_correspondence",
                  trainer_repeatability_uses_warped_heatmap_correspondence);
    register_test("trainer_descriptor_loss_uses_warped_correspondence",
                  trainer_descriptor_loss_uses_warped_correspondence);
    register_test("trainer_descriptor_loss_ignores_invalid_warp_targets",
                  trainer_descriptor_loss_ignores_invalid_warp_targets);
    register_test("trainer_descriptor_loss_penalizes_globally_collapsed_descriptors",
                  trainer_descriptor_loss_penalizes_globally_collapsed_descriptors);
    register_test("trainer_dense_descriptor_hard_negative_loss_scans_full_map",
                  trainer_dense_descriptor_hard_negative_loss_scans_full_map);
    register_test(
        "trainer_warp_descriptor_contrastive_loss_uses_half_turn_correspondence",
        trainer_warp_descriptor_contrastive_loss_uses_half_turn_correspondence);
    register_test(
        "trainer_warp_descriptor_contrastive_loss_rejects_cyclic_descriptor_shift",
        trainer_warp_descriptor_contrastive_loss_accepts_cyclic_descriptor_shift);
    register_test(
        "trainer_direct_full_map_descriptor_loss_penalizes_global_distractor",
        trainer_direct_full_map_descriptor_loss_penalizes_global_distractor);
    register_test(
        "trainer_direct_full_map_descriptor_loss_rejects_cyclic_descriptor_shift",
        trainer_direct_full_map_descriptor_loss_accepts_cyclic_descriptor_shift);
    register_test(
        "trainer_descriptor_targets_use_cell_centers_for_warp_coordinates",
        trainer_descriptor_targets_use_cell_centers_for_warp_coordinates);
    register_test(
        "trainer_warped_descriptor_sampling_preserves_subpixel_correspondence",
        trainer_warped_descriptor_sampling_preserves_subpixel_correspondence);
    register_test(
        "trainer_descriptor_map_regularization_penalizes_spatial_collapse",
        trainer_descriptor_map_regularization_penalizes_spatial_collapse);
    register_test("trainer_keypoint_descriptor_loss_uses_sparse_keypoint_hard_negatives",
                  trainer_keypoint_descriptor_loss_uses_sparse_keypoint_hard_negatives);
    register_test("trainer_keypoint_descriptor_loss_penalizes_hardest_negative_margin",
                  trainer_keypoint_descriptor_loss_penalizes_hardest_negative_margin);
    register_test("trainer_keypoint_dense_descriptor_loss_uses_warp_target_in_full_map",
                  trainer_keypoint_dense_descriptor_loss_uses_warp_target_in_full_map);
    register_test("trainer_orientation_supervision_uses_warp_rotation",
                  trainer_orientation_supervision_uses_warp_rotation);
    register_test("trainer_keypoint_descriptor_loss_covers_more_than_graph_query_limit",
                  trainer_keypoint_descriptor_loss_covers_more_than_graph_query_limit);
    register_test("trainer_graph_matching_loss_trains_graph_matcher_parameters",
                  trainer_graph_matching_loss_trains_graph_matcher_parameters);
    register_test("trainer_graph_matching_loss_is_finite_with_many_descriptors",
                  trainer_graph_matching_loss_is_finite_with_many_descriptors);
    register_test(
        "trainer keypoint graph targets use warped nearest b keypoint",
        trainer_keypoint_graph_targets_use_warped_nearest_b_keypoint);
    register_test(
        "trainer keypoint graph targets use dustbin for unmatched keypoints",
        trainer_keypoint_graph_targets_use_dustbin_for_unmatched_keypoints);
    register_test(
        "trainer keypoint graph targets use dustbin for invalid target pixels",
        trainer_keypoint_graph_targets_use_dustbin_for_invalid_target_pixels);
    register_test(
        "trainer graph candidates include positives once and dustbin last",
        trainer_graph_candidates_include_positives_once_and_dustbin_last);
    register_test(
        "trainer keypoint graph matching loss trains graph matcher parameters",
        trainer_keypoint_graph_matching_loss_trains_graph_matcher_parameters);
    register_test(
        "trainer keypoint graph matching loss uses full b candidate set",
        trainer_keypoint_graph_matching_loss_uses_full_b_candidate_set);
    register_test("trainer_stacks_variable_spatial_training_tensors_with_padding",
                  trainer_stacks_variable_spatial_training_tensors_with_padding);
    register_test("trainer_training_valid_mask_requires_bright_source_and_target_pixels",
                  trainer_training_valid_mask_requires_bright_source_and_target_pixels);
    register_test("trainer_descriptor_candidates_do_not_repeat_positive_target",
                  trainer_descriptor_candidates_do_not_repeat_positive_target);
    register_test("trainer_bounds_descriptor_loss_spatial_samples",
                  trainer_bounds_descriptor_loss_spatial_samples);
    register_test("trainer_total_loss_downweights_dense_offset_pixels",
                  trainer_total_loss_downweights_dense_offset_pixels);
    register_test("trainer_total_loss_penalizes_descriptor_spatial_collapse",
                  trainer_total_loss_penalizes_descriptor_spatial_collapse);
    register_test("trainer_progress_reports_loss_components", trainer_progress_reports_loss_components);
    register_test("trainer_reports_epoch_and_batch_timing", trainer_reports_epoch_and_batch_timing);
    register_test("trainer_writes_csv_metric_log", trainer_writes_csv_metric_log);
    register_test("trainer_uses_online_dataloader_when_workers_requested",
                  trainer_uses_online_dataloader_when_workers_requested);
    register_test("trainer_trains_with_online_dataloader_workers", trainer_trains_with_online_dataloader_workers);
    register_test("trainer_resizes_large_training_image", trainer_resizes_large_training_image);
    register_test("trainer_uses_configured_resize", trainer_uses_configured_resize);
    register_test("trainer_training_and_validation_indices_use_dataloader_split",
                  trainer_training_and_validation_indices_use_dataloader_split);
    register_test("trainer_variant_indices_advance_across_epochs",
                  trainer_variant_indices_advance_across_epochs);
    register_test("trainer_trains_full_dataset", trainer_trains_full_dataset);
    register_test("trainer_with_synthetic_pair_cache_writes_cache_and_checkpoint",
                  trainer_with_synthetic_pair_cache_writes_cache_and_checkpoint);
    register_test("trainer_pairs_per_image_expands_cached_training_pairs",
                  trainer_pairs_per_image_expands_cached_training_pairs);
    register_test("trainer_reuses_existing_synthetic_pair_cache", trainer_reuses_existing_synthetic_pair_cache);
    register_test("trainer_rebuilds_synthetic_pair_cache_when_requested",
                  trainer_rebuilds_synthetic_pair_cache_when_requested);
    register_test("trainer_forwards_augmentation_profile_to_cached_pairs",
                  trainer_forwards_augmentation_profile_to_cached_pairs);
    register_test("trainer_visualization_writes_expected_pngs_for_sampled_pair",
                  trainer_visualization_writes_expected_pngs_for_sampled_pair);
    register_test("trainer_visualization_all_writes_every_pair", trainer_visualization_all_writes_every_pair);
    register_test("trainer_visualization_zero_samples_writes_no_pngs",
                  trainer_visualization_zero_samples_writes_no_pngs);
    register_test("trainer_visualization_writes_sampled_pair_for_each_epoch",
                  trainer_visualization_writes_sampled_pair_for_each_epoch);
    register_test("trainer_visualization_warp_overlay_does_not_mutate_source_pair",
                  trainer_visualization_warp_overlay_does_not_mutate_source_pair);
    register_test("trainer_visualization_feature_overlay_suppresses_dark_pixels",
                  trainer_visualization_feature_overlay_suppresses_dark_pixels);
    register_test("trainer_visualization_model_matches_uses_side_by_side_canvas",
                  trainer_visualization_model_matches_uses_side_by_side_canvas);
    register_test(
        "trainer_visualization_model_matches_colors_sparse_correct_and_wrong_lines",
        trainer_visualization_model_matches_colors_sparse_correct_and_wrong_lines);
    register_test(
        "trainer_visualization_model_matches_text_includes_correct_and_wrong_counts",
        trainer_visualization_model_matches_text_includes_correct_and_wrong_counts);
}
