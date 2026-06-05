#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cctype>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <unistd.h>

#include "cli/commands.h"
#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"
#include "infer/feature_extractor.h"
#include "infer/matching_pipeline.h"
#include "infer/pipeline.h"
#include "tests/test_harness.h"
#include "train/trainer.h"

namespace pfm::testing
{
int64_t descriptor_grid_fallback_min_sparse_matches_for_test();
bool should_use_descriptor_grid_fallback_for_test(int64_t base_sparse_matches, int64_t grid_sparse_matches);
bool should_use_high_density_sparse_matches_for_test(int64_t base_sparse_matches, int64_t high_density_sparse_matches);
bool should_use_alternate_texture_blend_matches_for_test(int64_t base_sparse_matches, int64_t alternate_sparse_matches);
bool should_use_balanced_texture_blend_matches_for_test(int64_t base_sparse_matches, int64_t alternate_sparse_matches);
double rotation_invariant_texture_blend_weight_for_test();
bool sparse_geometry_filter_rotation_only_requested_for_test();
bool should_skip_expensive_sparse_alternates_for_test(int64_t sparse_matches);
FeatureSet make_descriptor_grid_feature_set_for_test(const RawFeatureMaps& maps, const FeatureDecodeConfig& config,
                                                     const torch::Tensor& intensity_mask);
torch::Tensor make_inference_decode_heatmap_for_test(const torch::Tensor& image, const torch::Tensor& learned_heatmap);
FeatureDecodeConfig make_high_density_decode_config_for_test(FeatureDecodeConfig decode_config);
GraphMatcherInferenceOptions make_graph_matcher_inference_options_for_test(const CliOptions& options);
} // namespace pfm::testing

namespace
{

struct CoutCapture
{
    std::ostringstream stream;
    std::streambuf* old = nullptr;

    CoutCapture() : old(std::cout.rdbuf(stream.rdbuf()))
    {
    }
    CoutCapture(const CoutCapture&) = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;
    CoutCapture(CoutCapture&&) = delete;
    CoutCapture& operator=(CoutCapture&&) = delete;

    ~CoutCapture() noexcept
    {
        try
        {
            if (old != nullptr)
            {
                std::cout.rdbuf(old);
            }
        }
        catch (...)
        {
        }
    }

    std::string str() const
    {
        return stream.str();
    }
};

class TempPipelineDirectory
{
  public:
    explicit TempPipelineDirectory(const std::string& stem)
    {
        const auto suffix =
            std::to_string(static_cast<long long>(getpid())) + "_" + std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempPipelineDirectory()
    {
        for (const auto& file_path : _files)
        {
            std::remove(file_path.string().c_str());
        }
        std::error_code ignored;
        const auto cache_dir = _path / "pair_cache";
        if (std::filesystem::exists(cache_dir, ignored))
        {
            for (const auto& entry : std::filesystem::directory_iterator(cache_dir))
            {
                std::filesystem::remove(entry.path(), ignored);
            }
            std::filesystem::remove(cache_dir, ignored);
        }
        std::filesystem::remove(_path, ignored);
    }

    const std::filesystem::path& path() const
    {
        return _path;
    }

    std::filesystem::path file(const std::string& name)
    {
        auto file_path = _path / name;
        _files.push_back(file_path);
        return file_path;
    }

  private:
    std::filesystem::path _path;
    std::vector<std::filesystem::path> _files;
};

void write_test_image(const std::filesystem::path& path, int offset)
{
    cv::Mat image(32, 32, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            image.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 5 + y * 13 + offset) % 256);
        }
    }
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

void write_scaled_match_image(const std::filesystem::path& path)
{
    cv::Mat image(32, 40, CV_8UC1, cv::Scalar(40));
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

bool has_yellow_pixel_near(const cv::Mat& image, int center_x, int center_y)
{
    for (int y = std::max(0, center_y - 3); y <= std::min(image.rows - 1, center_y + 3); ++y)
    {
        for (int x = std::max(0, center_x - 3); x <= std::min(image.cols - 1, center_x + 3); ++x)
        {
            const auto pixel = image.at<cv::Vec3b>(y, x);
            if (pixel[0] < 80 && pixel[1] > 150 && pixel[2] > 150)
            {
                return true;
            }
        }
    }
    return false;
}

bool has_formatted_seconds_after_label(const std::string& output, const std::string& label)
{
    const auto start = output.find(label);
    if (start == std::string::npos)
    {
        return false;
    }
    auto index = start + label.size();
    if (index >= output.size() || !std::isdigit(static_cast<unsigned char>(output[index])))
    {
        return false;
    }
    while (index < output.size() && std::isdigit(static_cast<unsigned char>(output[index])))
    {
        ++index;
    }
    if (index >= output.size() || output[index] != '.')
    {
        return false;
    }
    ++index;
    int decimal_count = 0;
    while (index < output.size() && std::isdigit(static_cast<unsigned char>(output[index])))
    {
        ++index;
        ++decimal_count;
    }
    return decimal_count >= 3 && index < output.size() && output[index] == 's';
}

void write_text_file(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream output(path);
    PFM_REQUIRE(static_cast<bool>(output));
    output << text;
}

torch::Tensor make_identity_warp(int64_t height, int64_t width)
{
    auto warp = torch::zeros({height, width, 2}, torch::kFloat32);
    auto values = warp.accessor<float, 3>();
    for (int64_t y = 0; y < height; ++y)
    {
        for (int64_t x = 0; x < width; ++x)
        {
            values[y][x][0] = static_cast<float>(x);
            values[y][x][1] = static_cast<float>(y);
        }
    }
    return warp;
}

void write_pair_archive_for_training(const std::filesystem::path& path)
{
    std::filesystem::create_directories(path.parent_path());
    torch::jit::Module module("PairArchive");
    module.register_attribute("view_a", c10::TensorType::get(), torch::ones({1, 32, 32}, torch::kFloat32));
    module.register_attribute("view_b", c10::TensorType::get(), torch::ones({1, 32, 32}, torch::kFloat32) * 0.9F);
    module.register_attribute("warp_a_to_b", c10::TensorType::get(), make_identity_warp(32, 32));
    module.register_attribute("valid_mask", c10::TensorType::get(), torch::ones({32, 32}, torch::kBool));
    module.save(path.string());
}

pfm::CliOptions make_train_options(TempPipelineDirectory& temp_dir)
{
    pfm::CliOptions options;
    options.image_dir = temp_dir.path().string();
    options.checkpoint = temp_dir.file("checkpoint.pt").string();
    options.epochs = 1;
    options.batch_size = 1;
    options.device = "cpu";
    options.train_ratio = 1.0;
    options.val_ratio = 0.0;
    return options;
}

std::string write_checkpoint(TempPipelineDirectory& temp_dir)
{
    write_test_image(temp_dir.file("train_a.png"), 3);
    write_test_image(temp_dir.file("train_b.png"), 41);

    pfm::TrainConfig config;
    config.image_dir = temp_dir.path().string();
    config.checkpoint = temp_dir.file("source_checkpoint.pt").string();
    config.epochs = 1;
    config.batch_size = 1;
    config.base_channels = 2;
    config.descriptor_dim = 4;
    config.graph_hidden_dim = 16;
    config.graph_attention_layers = 1;
    config.graph_keypoint_meta_dim = 16;
    pfm::train_model(config);
    return config.checkpoint;
}

std::string write_config_only_checkpoint(TempPipelineDirectory& temp_dir)
{
    const auto checkpoint = temp_dir.file("config_only.pt");
    torch::serialize::OutputArchive archive;
    torch::serialize::OutputArchive config_archive;
    config_archive.write("base_channels", torch::tensor({2}, torch::kInt64));
    config_archive.write("descriptor_dim", torch::tensor({4}, torch::kInt64));
    config_archive.write("input_channels", torch::tensor({1}, torch::kInt64));
    archive.write("config", config_archive);
    archive.save_to(checkpoint.string());
    return checkpoint.string();
}

pfm::FeatureSet makeTestFeatureSet(float offset)
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    return pfm::FeatureSet{torch::tensor({{0.0F + offset, 0.0F}, {1.0F + offset, 1.0F}}, float_options),
                           torch::tensor({0.9F, 0.8F}, float_options),
                           torch::tensor({{1.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F, 0.0F}}, float_options),
                           torch::ones({2}, float_options),
                           torch::zeros({2}, float_options),
                           torch::zeros({2, 2, 2}, float_options),
                           torch::tensor({{2.0F + offset, 2.0F}, {3.0F + offset, 3.0F}}, float_options),
                           torch::tensor({0.7F, 0.6F}, float_options)};
}

pfm::FeatureSet makeScaledFeatureSet()
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto features = pfm::FeatureSet{torch::tensor({{5.0F, 4.0F}}, float_options),
                                    torch::tensor({0.9F}, float_options),
                                    torch::tensor({{1.0F, 0.0F, 0.0F, 0.0F}}, float_options),
                                    torch::ones({1}, float_options),
                                    torch::zeros({1}, float_options),
                                    torch::zeros({1, 2, 2}, float_options),
                                    torch::empty({0, 2}, float_options),
                                    torch::empty({0}, float_options)};
    features.feature_map_width = 10;
    features.feature_map_height = 8;
    return features;
}

pfm::FeatureSet makePythonRawMutualFeatureSetA()
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    return pfm::FeatureSet{torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, float_options),
                           torch::ones({4}, float_options),
                           torch::tensor({{1.0F, 0.0F, 0.0F},
                                          {0.0F, 1.0F, 0.0F},
                                          {0.0F, 0.0F, 1.0F},
                                          {0.6F, 0.8F, 0.0F}},
                                         float_options),
                           torch::ones({4}, float_options),
                           torch::zeros({4, 2}, float_options),
                           torch::zeros({4, 2, 2}, float_options),
                           torch::empty({0, 2}, float_options),
                           torch::empty({0}, float_options)};
}

pfm::FeatureSet makePythonRawMutualFeatureSetB()
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    return pfm::FeatureSet{torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, float_options),
                           torch::ones({4}, float_options),
                           torch::tensor({{0.0F, 0.03F, 0.9995F},
                                          {0.8F, 0.2F, 0.0F},
                                          {0.2F, 0.7F, 0.0F},
                                          {0.4F, 0.8F, 0.0F}},
                                         float_options),
                           torch::ones({4}, float_options),
                           torch::zeros({4, 2}, float_options),
                           torch::zeros({4, 2, 2}, float_options),
                           torch::empty({0, 2}, float_options),
                           torch::empty({0}, float_options)};
}

} // namespace

static void pipeline_train_writes_loadable_checkpoint()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train");
    write_test_image(temp_dir.file("image.png"), 11);
    auto options = make_train_options(temp_dir);

    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    PFM_REQUIRE(std::filesystem::exists(options.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(options.checkpoint));
}

static void pipeline_train_accepts_pair_archive_cache_without_image_dir()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train_pair_archive");
    const auto cache_dir = temp_dir.path() / "archive_cache" / "train";
    write_pair_archive_for_training(cache_dir / "source_000001" / "pair_000001_identity.pt");

    pfm::CliOptions options;
    options.checkpoint = temp_dir.file("checkpoint.pt").string();
    options.epochs = 1;
    options.batch_size = 1;
    options.device = "cpu";
    options.base_channels = 2;
    options.descriptor_dim = 4;
    options.graph_hidden_dim = 16;
    options.graph_attention_layers = 1;
    options.graph_keypoint_meta_dim = 16;
    options.resize = 0;
    options.training_profile = "smoke";
    options.pair_cache_dirs.push_back(cache_dir.string());
    options.pair_cache_limit = 1;

    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    PFM_REQUIRE(std::filesystem::exists(options.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(options.checkpoint));
}

static void pipeline_train_rejects_invalid_training_limits()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train_invalid_limits");
    write_test_image(temp_dir.file("image.png"), 11);

    auto invalid_resize = make_train_options(temp_dir);
    invalid_resize.resize = -1;
    PFM_REQUIRE(pfm::run_train_command(invalid_resize) != 0);
    PFM_REQUIRE(!std::filesystem::exists(invalid_resize.checkpoint));
}

static void pipeline_train_writes_synthetic_pair_cache()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train_pair_cache");
    write_test_image(temp_dir.file("image_a.png"), 11);
    write_test_image(temp_dir.file("image_b.png"), 23);
    auto options = make_train_options(temp_dir);
    options.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    options.resize = 32;

    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    PFM_REQUIRE(std::filesystem::exists(options.checkpoint));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(options.synthetic_pair_cache_dir) / "manifest.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(options.synthetic_pair_cache_dir) /
                                        "source_000000_image_a" / "pair_000000_view_b.png"));
}

static void pipeline_train_cache_only_writes_cache_without_checkpoint()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train_cache_only");
    write_test_image(temp_dir.file("image_a.png"), 11);
    write_test_image(temp_dir.file("image_b.png"), 23);
    auto options = make_train_options(temp_dir);
    options.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    options.cache_only = true;
    options.resize = 32;

    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    PFM_REQUIRE(!std::filesystem::exists(options.checkpoint));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(options.synthetic_pair_cache_dir) / "manifest.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(options.synthetic_pair_cache_dir) /
                                        "source_000000_image_a" / "pair_000000_view_b.png"));
}

static void pipeline_train_prints_total_and_average_batch_time()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train_timing");
    write_test_image(temp_dir.file("image_a.png"), 11);
    write_test_image(temp_dir.file("image_b.png"), 23);
    auto options = make_train_options(temp_dir);
    options.checkpoint = temp_dir.file("timed_model.pt").string();

    CoutCapture capture;
    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    const auto output = capture.str();

    PFM_REQUIRE(has_formatted_seconds_after_label(output, "total_time="));
    PFM_REQUIRE(has_formatted_seconds_after_label(output, "avg_batch_time="));
}

static void pipeline_train_prints_visualization_note()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train_visualization_note");
    write_test_image(temp_dir.file("image_a.png"), 11);
    write_test_image(temp_dir.file("image_b.png"), 23);
    auto options = make_train_options(temp_dir);
    options.visualization_dir = (temp_dir.path() / "train_vis").string();
    options.visualization_samples = 6;
    options.max_keypoints = 2048;
    options.min_keypoints = 512;
    options.keypoint_grid_rows = 4;
    options.keypoint_grid_cols = 6;
    options.keypoints_per_cell = 8;
    options.nms_radius = 2;

    CoutCapture capture;
    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    const auto output = capture.str();

    PFM_REQUIRE(output.find("training visualization: dir=" + options.visualization_dir) != std::string::npos);
    PFM_REQUIRE(output.find("samples=6") != std::string::npos);
    PFM_REQUIRE(output.find("max_keypoints=2048") != std::string::npos);
    PFM_REQUIRE(output.find("min_keypoints=512") != std::string::npos);
    PFM_REQUIRE(output.find("keypoint_grid=4x6") != std::string::npos);
    PFM_REQUIRE(output.find("keypoints_per_cell=8") != std::string::npos);
    PFM_REQUIRE(output.find("nms_radius=2") != std::string::npos);
    PFM_REQUIRE(output.find("async_queue=2048") != std::string::npos);
    PFM_REQUIRE(output.find("async_workers=4") != std::string::npos);
}

static void pipeline_train_accepts_min_keypoint_intensity()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train_intensity_mask");
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(255));
    PFM_REQUIRE(cv::imwrite(temp_dir.file("image.png").string(), mat));
    auto options = make_train_options(temp_dir);
    options.min_keypoint_intensity = 0.5;
    options.resize = 32;

    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    PFM_REQUIRE(std::filesystem::exists(options.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(options.checkpoint));
}

static void pipeline_train_forwards_pairs_per_image_to_cache_generation()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_train_pairs_per_image");
    write_test_image(temp_dir.file("image_a.png"), 11);
    write_test_image(temp_dir.file("image_b.png"), 23);
    auto options = make_train_options(temp_dir);
    options.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    options.pairs_per_image = 2;
    options.augmentation_profile = "extreme";
    options.extreme_pair_ratio = 0.4;
    options.min_keypoint_intensity = 0.0;
    options.resize = 32;

    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(options.synthetic_pair_cache_dir) /
                                        "source_000001_image_b" / "pair_000003.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(options.synthetic_pair_cache_dir) /
                                        "source_000001_image_b" / "pair_000003_view_b.png"));
}

static void pipeline_extract_writes_loadable_feature_file()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_extract");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("extract.png");
    write_test_image(image, 83);
    const auto output = temp_dir.file("features.pt");

    pfm::CliOptions options;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 128;
    options.semi_dense_threshold = 0.0;
    options.device = "cpu";
    options.visualization_dir = (temp_dir.path() / "vis").string();

    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto features = pfm::load_feature_set(options.output);
    PFM_REQUIRE(features.keypoints.defined());
    PFM_REQUIRE(features.keypoints.size(0) > 0);
    PFM_REQUIRE(features.keypoints.size(0) <= 64);
    PFM_REQUIRE(std::filesystem::exists(temp_dir.path() / "vis" / "extract_features.png"));
    PFM_REQUIRE(features.descriptors.defined());
    PFM_REQUIRE(features.descriptors.size(0) == features.keypoints.size(0));
}

static void pipeline_extract_uses_keypoint_distribution_options()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_decode_distribution");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("distributed_extract.png");
    write_test_image(image, 97);
    const auto output = temp_dir.file("features.pt");

    pfm::CliOptions options;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 64;
    options.semi_dense_threshold = 0.0;
    options.keypoint_grid_rows = 1;
    options.keypoint_grid_cols = 1;
    options.keypoints_per_cell = 64;
    options.nms_radius = 100;
    options.device = "cpu";

    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto features = pfm::load_feature_set(options.output);

    PFM_REQUIRE(features.keypoints.size(0) == 1);
}

static void pipeline_extract_prints_stage_timing()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_extract_timing");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("timed_extract.png");
    write_test_image(image, 101);

    pfm::CliOptions options;
    options.command = pfm::Command::Extract;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = temp_dir.file("features.pt").string();
    options.visualization_dir = temp_dir.file("vis").string();
    options.device = "cpu";

    CoutCapture capture;
    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto output = capture.str();

    PFM_REQUIRE(has_formatted_seconds_after_label(output, "elapsed="));
    PFM_REQUIRE(has_formatted_seconds_after_label(output, "image_load="));
    PFM_REQUIRE(has_formatted_seconds_after_label(output, "model_forward="));
    PFM_REQUIRE(has_formatted_seconds_after_label(output, "decode="));
    PFM_REQUIRE(has_formatted_seconds_after_label(output, "save="));
    PFM_REQUIRE(has_formatted_seconds_after_label(output, "visualization="));
    PFM_REQUIRE(output.find("sparse_features=") != std::string::npos);
    PFM_REQUIRE(output.find("dense_features=") != std::string::npos);
}

static void pipeline_extract_filters_keypoints_below_min_intensity()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_extract_intensity_mask");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("masked_extract.png");
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(0));
    mat(cv::Rect(20, 20, 8, 8)).setTo(cv::Scalar(255));
    PFM_REQUIRE(cv::imwrite(image.string(), mat));
    const auto output = temp_dir.file("features.pt");

    pfm::CliOptions options;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 16;
    options.semi_dense_threshold = 0.0;
    options.device = "cpu";
    options.min_keypoint_intensity = 0.5;

    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto features = pfm::load_feature_set(options.output);
    for (int64_t index = 0; index < features.keypoints.size(0); ++index)
    {
        PFM_REQUIRE(features.keypoints.index({index, 0}).item<float>() >= 5.0F);
        PFM_REQUIRE(features.keypoints.index({index, 0}).item<float>() <= 6.0F);
        PFM_REQUIRE(features.keypoints.index({index, 1}).item<float>() >= 5.0F);
        PFM_REQUIRE(features.keypoints.index({index, 1}).item<float>() <= 6.0F);
    }
}

static void pipeline_export_writes_loadable_checkpoint()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_export");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto output = temp_dir.file("exported.pt");

    pfm::CliOptions options;
    options.checkpoint = checkpoint;
    options.output = output.string();

    PFM_REQUIRE(pfm::run_export_command(options) == 0);
    PFM_REQUIRE(std::filesystem::exists(options.output));
    PFM_REQUIRE(pfm::checkpoint_can_load(options.output));
}

static void pipeline_export_rejects_config_only_checkpoint()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_export_config_only");
    const auto checkpoint = write_config_only_checkpoint(temp_dir);
    const auto output = temp_dir.file("exported_config_only.pt");

    pfm::CliOptions options;
    options.checkpoint = checkpoint;
    options.output = output.string();

    PFM_REQUIRE(pfm::run_export_command(options) != 0);
    PFM_REQUIRE(!std::filesystem::exists(options.output));
}

static void pipeline_match_writes_match_file()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_match");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image_a = temp_dir.file("match_a.png");
    const auto image_b = temp_dir.file("match_b.png");
    const auto output = temp_dir.file("matches.pt");
    write_test_image(image_a, 17);
    write_test_image(image_b, 29);

    pfm::CliOptions options;
    options.image_a = image_a.string();
    options.image_b = image_b.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 8;
    options.semi_dense_threshold = 0.0;
    options.device = "cpu";
    options.visualization_dir = (temp_dir.path() / "vis").string();

    PFM_REQUIRE(pfm::run_match_command(options) == 0);
    const auto matches = pfm::load_match_set(options.output);
    PFM_REQUIRE(matches.sparse_matches.defined());
    PFM_REQUIRE(matches.sparse_matches.scalar_type() == torch::kInt64);
    PFM_REQUIRE(matches.sparse_matches.dim() == 2);
    PFM_REQUIRE(matches.sparse_matches.size(1) == 2);
    PFM_REQUIRE(std::filesystem::exists(temp_dir.path() / "vis" / "match_a__match_b_matches.png"));
    PFM_REQUIRE(matches.sparse_scores.defined());
    PFM_REQUIRE(matches.sparse_scores.dim() == 1);
    PFM_REQUIRE(matches.sparse_scores.size(0) == matches.sparse_matches.size(0));
    PFM_REQUIRE(matches.points_a.defined());
    PFM_REQUIRE(matches.points_b.defined());
    PFM_REQUIRE(matches.confidence.defined());
    PFM_REQUIRE(matches.points_a.dim() == 2);
    PFM_REQUIRE(matches.points_b.dim() == 2);
    PFM_REQUIRE(matches.points_a.size(1) == 2);
    PFM_REQUIRE(matches.points_b.size(1) == 2);
    PFM_REQUIRE(matches.confidence.dim() == 1);
    PFM_REQUIRE(matches.confidence.size(0) == matches.points_a.size(0));
}

static void pipelineMatchUsesPrecomputedFeatureFilesWithCheckpointMatcher()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_match_feature_files");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto feature_a = temp_dir.file("feature_a.pt");
    const auto feature_b = temp_dir.file("feature_b.pt");
    const auto output = temp_dir.file("matches.pt");
    pfm::save_feature_set(makeTestFeatureSet(0.0F), feature_a.string());
    pfm::save_feature_set(makeTestFeatureSet(1.0F), feature_b.string());

    pfm::CliOptions options;
    options.feature_a = feature_a.string();
    options.feature_b = feature_b.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.match_mode = "sparse";

    PFM_REQUIRE(pfm::run_match_command(options) == 0);
    const auto matches = pfm::load_match_set(options.output);
    PFM_REQUIRE(matches.sparse_matches.size(0) == 2);
    PFM_REQUIRE(matches.points_a.size(0) == 0);
    PFM_REQUIRE(matches.points_b.size(0) == 0);
    PFM_REQUIRE(matches.confidence.size(0) == 0);
}

static void pipelineMatchCanUsePythonRawMutualSparseStrategy()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_match_python_raw_mutual");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto feature_a = temp_dir.file("feature_a.pt");
    const auto feature_b = temp_dir.file("feature_b.pt");
    const auto output = temp_dir.file("matches.pt");
    pfm::save_feature_set(makePythonRawMutualFeatureSetA(), feature_a.string());
    pfm::save_feature_set(makePythonRawMutualFeatureSetB(), feature_b.string());

    pfm::CliOptions options;
    options.feature_a = feature_a.string();
    options.feature_b = feature_b.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.match_mode = "sparse";
    options.sparse_match_strategy = "python-raw-mutual";
    options.max_matches = 2;

    PFM_REQUIRE(pfm::run_match_command(options) == 0);
    const auto matches = pfm::load_match_set(options.output);
    const auto expected = torch::tensor({{2, 0}, {3, 3}}, torch::kInt64);
    PFM_REQUIRE(torch::equal(matches.sparse_matches, expected));
    PFM_REQUIRE(matches.sparse_scores.size(0) == 2);
    PFM_REQUIRE(matches.points_a.size(0) == 0);
    PFM_REQUIRE(matches.points_b.size(0) == 0);
    PFM_REQUIRE(matches.confidence.size(0) == 0);
}

static void pipelineMatchSparseModeWritesSparseVisualization()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_match_sparse_visualization");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image_a = temp_dir.file("sparse_vis_a.png");
    const auto image_b = temp_dir.file("sparse_vis_b.png");
    write_test_image(image_a, 17);
    write_test_image(image_b, 29);

    pfm::CliOptions options;
    options.image_a = image_a.string();
    options.image_b = image_b.string();
    options.checkpoint = checkpoint;
    options.output = temp_dir.file("matches.pt").string();
    options.match_mode = "sparse";
    options.max_keypoints = 8;
    options.semi_dense_threshold = 0.0;
    options.visualization_dir = (temp_dir.path() / "vis").string();
    options.device = "cpu";

    PFM_REQUIRE(pfm::run_match_command(options) == 0);
    const auto matches = pfm::load_match_set(options.output);
    PFM_REQUIRE(matches.points_a.size(0) == 0);
    PFM_REQUIRE(std::filesystem::exists(temp_dir.path() / "vis" / "sparse_vis_a__sparse_vis_b_matches.png"));
}

static void pipelineMatchFeatureFilesScaleSparseVisualization()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_match_feature_files_scaled_vis");
    const auto image_a = temp_dir.file("feature_vis_a.png");
    const auto image_b = temp_dir.file("feature_vis_b.png");
    const auto feature_a = temp_dir.file("feature_a.pt");
    const auto feature_b = temp_dir.file("feature_b.pt");
    write_scaled_match_image(image_a);
    write_scaled_match_image(image_b);
    const auto checkpoint = write_checkpoint(temp_dir);
    pfm::save_feature_set(makeScaledFeatureSet(), feature_a.string());
    pfm::save_feature_set(makeScaledFeatureSet(), feature_b.string());

    pfm::CliOptions options;
    options.image_a = image_a.string();
    options.image_b = image_b.string();
    options.feature_a = feature_a.string();
    options.feature_b = feature_b.string();
    options.checkpoint = checkpoint;
    options.output = temp_dir.file("matches.pt").string();
    options.match_mode = "sparse";
    options.visualization_dir = (temp_dir.path() / "vis").string();

    PFM_REQUIRE(pfm::run_match_command(options) == 0);
    const auto output = cv::imread((temp_dir.path() / "vis" / "feature_vis_a__feature_vis_b_matches.png").string());
    PFM_REQUIRE(has_yellow_pixel_near(output, 20, 16));
    PFM_REQUIRE(has_yellow_pixel_near(output, 40 + 20, 16));
    PFM_REQUIRE(!has_yellow_pixel_near(output, 5, 4));
}

static void pipelineMatchDenseModeOmitsSparseMatches()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_match_dense_mode");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto feature_a = temp_dir.file("feature_a.pt");
    const auto feature_b = temp_dir.file("feature_b.pt");
    const auto output = temp_dir.file("matches.pt");
    pfm::save_feature_set(makeTestFeatureSet(0.0F), feature_a.string());
    pfm::save_feature_set(makeTestFeatureSet(1.0F), feature_b.string());

    pfm::CliOptions options;
    options.feature_a = feature_a.string();
    options.feature_b = feature_b.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.match_mode = "dense";

    PFM_REQUIRE(pfm::run_match_command(options) == 0);
    const auto matches = pfm::load_match_set(options.output);
    PFM_REQUIRE(matches.sparse_matches.size(0) == 0);
    PFM_REQUIRE(matches.sparse_scores.size(0) == 0);
    PFM_REQUIRE(matches.points_a.size(0) == 2);
    PFM_REQUIRE(matches.points_b.size(0) == 2);
    PFM_REQUIRE(matches.confidence.size(0) == 2);
}

static void pipeline_inference_decode_heatmap_prefers_texture_saliency()
{
    auto image = torch::zeros({1, 1, 16, 16}, torch::kFloat32);
    image.index_put_({0, 0, torch::indexing::Slice(), 8}, 1.0F);
    auto learned_heatmap = torch::ones({1, 1, 8, 8}, torch::kFloat32);

    const auto heatmap = pfm::testing::make_inference_decode_heatmap_for_test(image, learned_heatmap);

    PFM_REQUIRE(heatmap.sizes() == learned_heatmap.sizes());
    PFM_REQUIRE(heatmap.min().item<float>() < 0.05F);
    PFM_REQUIRE(heatmap.max().item<float>() > 0.95F);
}

static void pipeline_descriptor_grid_fallback_handles_low_quality_sparse_sets()
{
    PFM_REQUIRE(pfm::testing::descriptor_grid_fallback_min_sparse_matches_for_test() >= 16);
    PFM_REQUIRE(pfm::testing::should_use_descriptor_grid_fallback_for_test(0, 24));
    PFM_REQUIRE(pfm::testing::should_use_descriptor_grid_fallback_for_test(3, 24));
}

static void pipeline_descriptor_grid_fallback_preserves_existing_geometric_solution()
{
    PFM_REQUIRE(!pfm::testing::should_use_descriptor_grid_fallback_for_test(8, 24));
    PFM_REQUIRE(!pfm::testing::should_use_descriptor_grid_fallback_for_test(10, 29));
}

static void pipeline_descriptor_grid_feature_set_uses_multiple_valid_points_per_cell()
{
    pfm::RawFeatureMaps maps{torch::ones({1, 1, 8, 8}, torch::kFloat32), torch::ones({1, 4, 8, 8}, torch::kFloat32),
                             torch::ones({1, 1, 8, 8}, torch::kFloat32), torch::zeros({1, 2, 8, 8}, torch::kFloat32),
                             torch::ones({1, 4, 8, 8}, torch::kFloat32), torch::ones({1, 1, 8, 8}, torch::kFloat32)};
    auto mask = torch::zeros({8, 8}, torch::kUInt8);
    mask.index_put_({torch::indexing::Slice(0, 2), torch::indexing::Slice(0, 2)}, 1);
    mask.index_put_({torch::indexing::Slice(0, 2), torch::indexing::Slice(2, 4)}, 1);

    pfm::FeatureDecodeConfig one_per_cell;
    one_per_cell.max_keypoints = 16;
    one_per_cell.keypoints_per_cell = 1;
    const auto sparse = pfm::testing::make_descriptor_grid_feature_set_for_test(maps, one_per_cell, mask);

    pfm::FeatureDecodeConfig two_per_cell = one_per_cell;
    two_per_cell.keypoints_per_cell = 2;
    const auto denser = pfm::testing::make_descriptor_grid_feature_set_for_test(maps, two_per_cell, mask);

    PFM_REQUIRE(sparse.keypoints.size(0) == 2);
    PFM_REQUIRE(denser.keypoints.size(0) == 4);
    for (int64_t row = 0; row < denser.keypoints.size(0); ++row)
    {
        const auto x = static_cast<int64_t>(denser.keypoints.index({row, 0}).item<float>());
        const auto y = static_cast<int64_t>(denser.keypoints.index({row, 1}).item<float>());
        PFM_REQUIRE(mask.index({y, x}).item<uint8_t>() != 0);
    }
}

static void pipeline_adaptive_high_density_requires_large_sparse_gain()
{
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(15, 42));
    PFM_REQUIRE(pfm::testing::should_use_high_density_sparse_matches_for_test(9, 39));
    PFM_REQUIRE(pfm::testing::should_use_high_density_sparse_matches_for_test(11, 54));
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(10, 67));
    PFM_REQUIRE(pfm::testing::should_use_high_density_sparse_matches_for_test(16, 123));
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(16, 57));
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(17, 123));
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(8, 10));
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(8, 57));
    PFM_REQUIRE(pfm::testing::should_use_high_density_sparse_matches_for_test(7, 124));
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(39, 86));
    PFM_REQUIRE(pfm::testing::should_use_high_density_sparse_matches_for_test(220, 420));
    PFM_REQUIRE(pfm::testing::should_use_high_density_sparse_matches_for_test(214, 378));
    PFM_REQUIRE(pfm::testing::should_use_high_density_sparse_matches_for_test(220, 248));
    PFM_REQUIRE(pfm::testing::should_use_high_density_sparse_matches_for_test(214, 237));
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(156, 180));
    PFM_REQUIRE(!pfm::testing::should_use_high_density_sparse_matches_for_test(124, 124));
}

static void pipeline_adaptive_high_density_preserves_decode_nms_radius()
{
    pfm::FeatureDecodeConfig config;
    config.min_keypoints = 0;
    config.nms_radius = 2;

    const auto high_density = pfm::testing::make_high_density_decode_config_for_test(config);

    PFM_REQUIRE(high_density.min_keypoints >= 1500);
    PFM_REQUIRE(high_density.nms_radius == 2);
}

static void pipeline_alternate_texture_blend_requires_decisive_match_gain()
{
    PFM_REQUIRE(pfm::testing::should_use_alternate_texture_blend_matches_for_test(7, 37));
    PFM_REQUIRE(pfm::testing::should_use_alternate_texture_blend_matches_for_test(41, 108));
    PFM_REQUIRE(!pfm::testing::should_use_alternate_texture_blend_matches_for_test(60, 9));
    PFM_REQUIRE(!pfm::testing::should_use_alternate_texture_blend_matches_for_test(40, 36));
    PFM_REQUIRE(!pfm::testing::should_use_alternate_texture_blend_matches_for_test(8, 16));
}

static void pipeline_balanced_texture_blend_requires_stable_base_and_small_gain()
{
    PFM_REQUIRE(pfm::testing::should_use_balanced_texture_blend_matches_for_test(40, 44));
    PFM_REQUIRE(!pfm::testing::should_use_balanced_texture_blend_matches_for_test(8, 32));
    PFM_REQUIRE(!pfm::testing::should_use_balanced_texture_blend_matches_for_test(41, 108));
    PFM_REQUIRE(!pfm::testing::should_use_balanced_texture_blend_matches_for_test(60, 29));
    PFM_REQUIRE(!pfm::testing::should_use_balanced_texture_blend_matches_for_test(40, 36));
}

static void pipeline_default_texture_blend_uses_empirical_rotation_setting()
{
    unsetenv("PFM_TEXTURE_BLEND_WEIGHT");
    PFM_REQUIRE(pfm::testing::rotation_invariant_texture_blend_weight_for_test() == 1.0);

    setenv("PFM_TEXTURE_BLEND_WEIGHT", "0.25", 1);
    PFM_REQUIRE(pfm::testing::rotation_invariant_texture_blend_weight_for_test() == 0.25);

    setenv("PFM_TEXTURE_BLEND_WEIGHT", "invalid", 1);
    PFM_REQUIRE(pfm::testing::rotation_invariant_texture_blend_weight_for_test() == 1.0);
    unsetenv("PFM_TEXTURE_BLEND_WEIGHT");
}

static void pipeline_graph_inference_fast_preset_sets_lightglue_thresholds()
{
    pfm::CliOptions options;
    options.graph_inference_preset = "fast";

    const auto graph_options = pfm::testing::make_graph_matcher_inference_options_for_test(options);

    PFM_REQUIRE_CLOSE(graph_options.width_prune_min_score, 0.25, 1.0e-12);
    PFM_REQUIRE_CLOSE(graph_options.early_stop_min_confidence, 0.85, 1.0e-12);
}

static void pipeline_graph_inference_high_precision_preset_sets_filter_threshold()
{
    pfm::CliOptions options;
    options.graph_inference_preset = "high_precision";

    const auto graph_options = pfm::testing::make_graph_matcher_inference_options_for_test(options);

    PFM_REQUIRE_CLOSE(graph_options.width_prune_min_score, 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(graph_options.early_stop_min_confidence, 0.85, 1.0e-12);
}

static void pipeline_graph_inference_preset_allows_numeric_override()
{
    pfm::CliOptions options;
    options.graph_inference_preset = "fast";
    options.graph_width_prune_min_score = 0.7;
    options.graph_min_accept_probability = 0.65;
    options.graph_max_attention_layers = 2;
    options.graph_fallback_mode = "none";

    const auto graph_options = pfm::testing::make_graph_matcher_inference_options_for_test(options);

    PFM_REQUIRE_CLOSE(graph_options.width_prune_min_score, 0.7, 1.0e-12);
    PFM_REQUIRE_CLOSE(graph_options.early_stop_min_confidence, 0.85, 1.0e-12);
    PFM_REQUIRE_CLOSE(graph_options.min_accept_probability, 0.65, 1.0e-12);
    PFM_REQUIRE(graph_options.max_attention_layers == 2);
    PFM_REQUIRE(graph_options.fallback_mode == pfm::GraphMatcherFallbackMode::None);
}

static void pipeline_graph_inference_off_preset_disables_lightglue_thresholds()
{
    pfm::CliOptions options;
    options.graph_inference_preset = "off";

    const auto graph_options = pfm::testing::make_graph_matcher_inference_options_for_test(options);

    PFM_REQUIRE_CLOSE(graph_options.width_prune_min_score, -1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(graph_options.early_stop_min_confidence, -1.0, 1.0e-12);
}

static void pipeline_graph_inference_unknown_preset_throws()
{
    pfm::CliOptions options;
    options.graph_inference_preset = "unknown";

    PFM_REQUIRE_THROWS_AS((void)pfm::testing::make_graph_matcher_inference_options_for_test(options),
                          std::invalid_argument);
}

static void pipeline_rotation_only_geometry_skips_expensive_sparse_alternates_after_good_base()
{
    unsetenv("PFM_SPARSE_GEOMETRY_FILTER");
    PFM_REQUIRE(!pfm::testing::sparse_geometry_filter_rotation_only_requested_for_test());
    PFM_REQUIRE(!pfm::testing::should_skip_expensive_sparse_alternates_for_test(169));

    setenv("PFM_SPARSE_GEOMETRY_FILTER", "rotation-only", 1);
    PFM_REQUIRE(pfm::testing::sparse_geometry_filter_rotation_only_requested_for_test());
    PFM_REQUIRE(!pfm::testing::should_skip_expensive_sparse_alternates_for_test(31));
    PFM_REQUIRE(pfm::testing::should_skip_expensive_sparse_alternates_for_test(32));
    PFM_REQUIRE(pfm::testing::should_skip_expensive_sparse_alternates_for_test(169));
    unsetenv("PFM_SPARSE_GEOMETRY_FILTER");
}

static void pipeline_eval_writes_report_archive()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_eval");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image_a = temp_dir.file("eval_a.png");
    const auto image_b = temp_dir.file("eval_b.png");
    const auto pairs = temp_dir.file("pairs.txt");
    const auto output = temp_dir.file("report.pt");
    write_test_image(image_a, 71);
    write_test_image(image_b, 91);
    write_text_file(pairs, image_a.string() + " " + image_b.string() + "\n");

    pfm::CliOptions options;
    options.pairs = pairs.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 8;
    options.semi_dense_threshold = 0.0;
    options.device = "cpu";

    PFM_REQUIRE(pfm::run_eval_command(options) == 0);
    torch::serialize::InputArchive archive;
    archive.load_from(options.output);
    torch::Tensor average_matches;
    archive.read("average_matches", average_matches);
    PFM_REQUIRE(average_matches.defined());
    PFM_REQUIRE(average_matches.numel() == 1);
    PFM_REQUIRE(average_matches.to(torch::kCPU, torch::kFloat32).reshape({1}).item<float>() >= 0.0F);
}

static void pipelineEvalCanUsePythonRawMutualSparseStrategy()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_eval_python_raw_mutual");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image_a = temp_dir.file("eval_raw_a.png");
    const auto image_b = temp_dir.file("eval_raw_b.png");
    const auto pairs = temp_dir.file("pairs.txt");
    const auto output = temp_dir.file("report.pt");
    write_test_image(image_a, 71);
    write_test_image(image_b, 91);
    write_text_file(pairs, image_a.string() + " " + image_b.string() + "\n");

    pfm::CliOptions options;
    options.pairs = pairs.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 8;
    options.max_matches = 1;
    options.sparse_match_strategy = "python-raw-mutual";
    options.semi_dense_threshold = 0.0;
    options.device = "cpu";

    PFM_REQUIRE(pfm::run_eval_command(options) == 0);
    torch::serialize::InputArchive archive;
    archive.load_from(options.output);
    torch::Tensor average_matches;
    archive.read("average_matches", average_matches);
    PFM_REQUIRE_CLOSE(average_matches.to(torch::kCPU, torch::kFloat32).reshape({1}).item<float>(), 1.0F, 1.0e-6);
}

static void pipeline_match_eval_and_export_print_timing()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_command_timing");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image_a = temp_dir.file("timed_match_a.png");
    const auto image_b = temp_dir.file("timed_match_b.png");
    write_test_image(image_a, 17);
    write_test_image(image_b, 29);

    pfm::CliOptions match_options;
    match_options.command = pfm::Command::Match;
    match_options.image_a = image_a.string();
    match_options.image_b = image_b.string();
    match_options.checkpoint = checkpoint;
    match_options.output = temp_dir.file("matches.pt").string();
    match_options.visualization_dir = temp_dir.file("match_vis").string();
    match_options.device = "cpu";

    CoutCapture match_capture;
    PFM_REQUIRE(pfm::run_match_command(match_options) == 0);
    const auto match_output = match_capture.str();
    PFM_REQUIRE(has_formatted_seconds_after_label(match_output, "elapsed="));
    PFM_REQUIRE(has_formatted_seconds_after_label(match_output, "extract_a="));
    PFM_REQUIRE(has_formatted_seconds_after_label(match_output, "extract_b="));
    PFM_REQUIRE(has_formatted_seconds_after_label(match_output, "match_time="));
    PFM_REQUIRE(has_formatted_seconds_after_label(match_output, "save="));
    PFM_REQUIRE(has_formatted_seconds_after_label(match_output, "visualization="));
    PFM_REQUIRE(match_output.find("features_a=") != std::string::npos);
    PFM_REQUIRE(match_output.find("features_b=") != std::string::npos);
    PFM_REQUIRE(match_output.find("sparse_matches=") != std::string::npos);
    PFM_REQUIRE(match_output.find("dense_matches=") != std::string::npos);
    PFM_REQUIRE(match_output.find("graph_layers=") != std::string::npos);
    PFM_REQUIRE(match_output.find("graph_keypoints=") != std::string::npos);
    PFM_REQUIRE(match_output.find("graph_pruned=") != std::string::npos);
    PFM_REQUIRE(match_output.find("graph_work=") != std::string::npos);

    const auto pairs_path = temp_dir.file("pairs.txt");
    write_text_file(pairs_path, match_options.image_a + " " + match_options.image_b + "\n");
    pfm::CliOptions eval_options;
    eval_options.command = pfm::Command::Eval;
    eval_options.pairs = pairs_path.string();
    eval_options.checkpoint = checkpoint;
    eval_options.output = temp_dir.file("report.pt").string();
    eval_options.device = "cpu";

    CoutCapture eval_capture;
    PFM_REQUIRE(pfm::run_eval_command(eval_options) == 0);
    const auto eval_output = eval_capture.str();
    PFM_REQUIRE(eval_output.find("pairs=1") != std::string::npos);
    PFM_REQUIRE(eval_output.find("graph_layers=") != std::string::npos);
    PFM_REQUIRE(eval_output.find("graph_pruned=") != std::string::npos);
    PFM_REQUIRE(eval_output.find("graph_work=") != std::string::npos);
    PFM_REQUIRE(has_formatted_seconds_after_label(eval_output, "elapsed="));
    PFM_REQUIRE(has_formatted_seconds_after_label(eval_output, "avg_pair_time="));

    pfm::CliOptions export_options;
    export_options.command = pfm::Command::Export;
    export_options.checkpoint = checkpoint;
    export_options.output = temp_dir.file("exported.pt").string();

    CoutCapture export_capture;
    PFM_REQUIRE(pfm::run_export_command(export_options) == 0);
    PFM_REQUIRE(has_formatted_seconds_after_label(export_capture.str(), "elapsed="));
}

static void pipeline_extract_rejects_invalid_device()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_extract_invalid_device");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("extract.png");
    const auto output = temp_dir.file("features.pt");
    write_test_image(image, 83);

    pfm::CliOptions options;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.device = "cuda:abc";

    PFM_REQUIRE(pfm::run_extract_command(options) != 0);
    PFM_REQUIRE(!std::filesystem::exists(options.output));
}

static void pipeline_match_rejects_invalid_device()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_match_invalid_device");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image_a = temp_dir.file("match_a.png");
    const auto image_b = temp_dir.file("match_b.png");
    const auto output = temp_dir.file("matches.pt");
    write_test_image(image_a, 17);
    write_test_image(image_b, 29);

    pfm::CliOptions options;
    options.image_a = image_a.string();
    options.image_b = image_b.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.device = "cuda:abc";

    PFM_REQUIRE(pfm::run_match_command(options) != 0);
    PFM_REQUIRE(!std::filesystem::exists(options.output));
}

static void pipeline_eval_rejects_invalid_device()
{
    TempPipelineDirectory temp_dir("pfm_pipeline_eval_invalid_device");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image_a = temp_dir.file("eval_a.png");
    const auto image_b = temp_dir.file("eval_b.png");
    const auto pairs = temp_dir.file("pairs.txt");
    const auto output = temp_dir.file("report.pt");
    write_test_image(image_a, 71);
    write_test_image(image_b, 91);
    {
        std::ofstream stream(pairs);
        stream << image_a.string() << ' ' << image_b.string() << '\n';
    }

    pfm::CliOptions options;
    options.pairs = pairs.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.device = "cuda:abc";

    PFM_REQUIRE(pfm::run_eval_command(options) != 0);
    PFM_REQUIRE(!std::filesystem::exists(options.output));
}

static void pipeline_cuda_device_is_strictly_validated_when_unavailable()
{
    if (torch::cuda::is_available())
    {
        return;
    }

    TempPipelineDirectory temp_dir("pfm_pipeline_cuda_unavailable");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("extract.png");
    const auto output = temp_dir.file("features.pt");
    write_test_image(image, 83);

    pfm::CliOptions options;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.device = "cuda";

    PFM_REQUIRE(pfm::run_extract_command(options) != 0);
    PFM_REQUIRE(!std::filesystem::exists(options.output));
}

static void pipeline_cuda_extract_writes_cpu_feature_file_when_available()
{
    if (!torch::cuda::is_available())
    {
        return;
    }

    TempPipelineDirectory temp_dir("pfm_pipeline_cuda_extract");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("extract.png");
    const auto output = temp_dir.file("features.pt");
    write_test_image(image, 83);

    pfm::CliOptions options;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 8;
    options.semi_dense_threshold = 0.0;
    options.device = "cuda";

    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto features = pfm::load_feature_set(options.output);
    PFM_REQUIRE(features.keypoints.device().is_cpu());
    PFM_REQUIRE(features.descriptors.device().is_cpu());
    PFM_REQUIRE(features.dense_points.device().is_cpu());
    PFM_REQUIRE(features.dense_confidence.device().is_cpu());
}

void register_pipeline_tests()
{
    register_test("pipeline_train_writes_loadable_checkpoint", pipeline_train_writes_loadable_checkpoint);
    register_test("pipeline_train_accepts_pair_archive_cache_without_image_dir",
                  pipeline_train_accepts_pair_archive_cache_without_image_dir);
    register_test("pipeline_train_rejects_invalid_training_limits", pipeline_train_rejects_invalid_training_limits);
    register_test("pipeline_train_writes_synthetic_pair_cache", pipeline_train_writes_synthetic_pair_cache);
    register_test("pipeline_train_cache_only_writes_cache_without_checkpoint",
                  pipeline_train_cache_only_writes_cache_without_checkpoint);
    register_test("pipeline_train_prints_total_and_average_batch_time",
                  pipeline_train_prints_total_and_average_batch_time);
    register_test("pipeline_train_prints_visualization_note", pipeline_train_prints_visualization_note);
    register_test("pipeline_train_accepts_min_keypoint_intensity", pipeline_train_accepts_min_keypoint_intensity);
    register_test("pipeline_train_forwards_pairs_per_image_to_cache_generation",
                  pipeline_train_forwards_pairs_per_image_to_cache_generation);
    register_test("pipeline_extract_writes_loadable_feature_file", pipeline_extract_writes_loadable_feature_file);
    register_test("pipeline_extract_uses_keypoint_distribution_options",
                  pipeline_extract_uses_keypoint_distribution_options);
    register_test("pipeline_extract_prints_stage_timing", pipeline_extract_prints_stage_timing);
    register_test("pipeline_extract_filters_keypoints_below_min_intensity",
                  pipeline_extract_filters_keypoints_below_min_intensity);
    register_test("pipeline_export_writes_loadable_checkpoint", pipeline_export_writes_loadable_checkpoint);
    register_test("pipeline_export_rejects_config_only_checkpoint", pipeline_export_rejects_config_only_checkpoint);
    register_test("pipeline_match_writes_match_file", pipeline_match_writes_match_file);
    register_test("pipeline_match_uses_precomputed_feature_files_with_checkpoint_matcher",
                  pipelineMatchUsesPrecomputedFeatureFilesWithCheckpointMatcher);
    register_test("pipeline_match_can_use_python_raw_mutual_sparse_strategy",
                  pipelineMatchCanUsePythonRawMutualSparseStrategy);
    register_test("pipeline_match_sparse_mode_writes_sparse_visualization",
                  pipelineMatchSparseModeWritesSparseVisualization);
    register_test("pipeline_match_feature_files_scale_sparse_visualization",
                  pipelineMatchFeatureFilesScaleSparseVisualization);
    register_test("pipeline_match_dense_mode_omits_sparse_matches", pipelineMatchDenseModeOmitsSparseMatches);
    register_test("pipeline_inference_decode_heatmap_prefers_texture_saliency",
                  pipeline_inference_decode_heatmap_prefers_texture_saliency);
    register_test("pipeline_descriptor_grid_fallback_handles_low_quality_sparse_sets",
                  pipeline_descriptor_grid_fallback_handles_low_quality_sparse_sets);
    register_test("pipeline_descriptor_grid_fallback_preserves_existing_geometric_solution",
                  pipeline_descriptor_grid_fallback_preserves_existing_geometric_solution);
    register_test("pipeline_descriptor_grid_feature_set_uses_multiple_valid_points_per_cell",
                  pipeline_descriptor_grid_feature_set_uses_multiple_valid_points_per_cell);
    register_test("pipeline_adaptive_high_density_requires_large_sparse_gain",
                  pipeline_adaptive_high_density_requires_large_sparse_gain);
    register_test("pipeline_adaptive_high_density_preserves_decode_nms_radius",
                  pipeline_adaptive_high_density_preserves_decode_nms_radius);
    register_test("pipeline_alternate_texture_blend_requires_decisive_match_gain",
                  pipeline_alternate_texture_blend_requires_decisive_match_gain);
    register_test("pipeline_balanced_texture_blend_requires_stable_base_and_small_gain",
                  pipeline_balanced_texture_blend_requires_stable_base_and_small_gain);
    register_test("pipeline_default_texture_blend_uses_empirical_rotation_setting",
                  pipeline_default_texture_blend_uses_empirical_rotation_setting);
    register_test("pipeline_graph_inference_fast_preset_sets_lightglue_thresholds",
                  pipeline_graph_inference_fast_preset_sets_lightglue_thresholds);
    register_test("pipeline_graph_inference_high_precision_preset_sets_filter_threshold",
                  pipeline_graph_inference_high_precision_preset_sets_filter_threshold);
    register_test("pipeline_graph_inference_preset_allows_numeric_override",
                  pipeline_graph_inference_preset_allows_numeric_override);
    register_test("pipeline_graph_inference_off_preset_disables_lightglue_thresholds",
                  pipeline_graph_inference_off_preset_disables_lightglue_thresholds);
    register_test("pipeline_graph_inference_unknown_preset_throws",
                  pipeline_graph_inference_unknown_preset_throws);
    register_test("pipeline_rotation_only_geometry_skips_expensive_sparse_alternates_after_good_base",
                  pipeline_rotation_only_geometry_skips_expensive_sparse_alternates_after_good_base);
    register_test("pipeline_eval_writes_report_archive", pipeline_eval_writes_report_archive);
    register_test("pipeline_eval_can_use_python_raw_mutual_sparse_strategy",
                  pipelineEvalCanUsePythonRawMutualSparseStrategy);
    register_test("pipeline_match_eval_and_export_print_timing", pipeline_match_eval_and_export_print_timing);
    register_test("pipeline_extract_rejects_invalid_device", pipeline_extract_rejects_invalid_device);
    register_test("pipeline_match_rejects_invalid_device", pipeline_match_rejects_invalid_device);
    register_test("pipeline_eval_rejects_invalid_device", pipeline_eval_rejects_invalid_device);
    register_test("pipeline_cuda_device_is_strictly_validated_when_unavailable",
                  pipeline_cuda_device_is_strictly_validated_when_unavailable);
    register_test("pipeline_cuda_extract_writes_cpu_feature_file_when_available",
                  pipeline_cuda_extract_writes_cpu_feature_file_when_available);
}
