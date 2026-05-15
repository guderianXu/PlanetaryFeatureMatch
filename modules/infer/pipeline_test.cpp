#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cli/commands.h"
#include "infer/feature_codec.h"
#include "infer/pipeline.h"
#include "tests/test_harness.h"
#include "train/trainer.h"

namespace {

class TempPipelineDirectory {
public:
    explicit TempPipelineDirectory(const std::string& stem) {
        const auto suffix = std::to_string(static_cast<long long>(getpid())) + "_" +
                            std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempPipelineDirectory() {
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

void write_test_image(const std::filesystem::path& path, int offset) {
    cv::Mat image(32, 32, CV_8UC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 5 + y * 13 + offset) % 256);
        }
    }
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

pfm::CliOptions make_train_options(TempPipelineDirectory& temp_dir) {
    pfm::CliOptions options;
    options.image_dir = temp_dir.path().string();
    options.checkpoint = temp_dir.file("checkpoint.pt").string();
    options.epochs = 1;
    options.batch_size = 1;
    options.device = "cpu";
    return options;
}

std::string write_checkpoint(TempPipelineDirectory& temp_dir) {
    write_test_image(temp_dir.file("train_a.png"), 3);
    write_test_image(temp_dir.file("train_b.png"), 41);

    pfm::TrainConfig config;
    config.image_dir = temp_dir.path().string();
    config.checkpoint = temp_dir.file("source_checkpoint.pt").string();
    config.epochs = 1;
    config.batch_size = 1;
    config.base_channels = 2;
    config.descriptor_dim = 4;
    pfm::train_model(config);
    return config.checkpoint;
}

}  // namespace

static void pipeline_train_writes_loadable_checkpoint() {
    TempPipelineDirectory temp_dir("pfm_pipeline_train");
    write_test_image(temp_dir.file("image.png"), 11);
    auto options = make_train_options(temp_dir);

    PFM_REQUIRE(pfm::run_train_command(options) == 0);
    PFM_REQUIRE(std::filesystem::exists(options.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(options.checkpoint));
}

static void pipeline_extract_writes_loadable_feature_file() {
    TempPipelineDirectory temp_dir("pfm_pipeline_extract");
    const auto checkpoint = write_checkpoint(temp_dir);
    const auto image = temp_dir.file("extract.png");
    write_test_image(image, 83);
    const auto output = temp_dir.file("features.pt");

    pfm::CliOptions options;
    options.image = image.string();
    options.checkpoint = checkpoint;
    options.output = output.string();
    options.max_keypoints = 8;
    options.semi_dense_threshold = 0.0;
    options.device = "cpu";

    PFM_REQUIRE(pfm::run_extract_command(options) == 0);
    const auto features = pfm::load_feature_set(options.output);
    PFM_REQUIRE(features.keypoints.defined());
    PFM_REQUIRE(features.keypoints.size(0) > 0);
    PFM_REQUIRE(features.descriptors.defined());
    PFM_REQUIRE(features.descriptors.size(0) == features.keypoints.size(0));
}

static void pipeline_export_writes_loadable_checkpoint() {
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

static void pipeline_match_is_deferred_to_task_8() {
    pfm::CliOptions options;
    options.image_a = "a.png";
    options.image_b = "b.png";
    options.checkpoint = "checkpoint.pt";
    options.output = "matches.json";

    PFM_REQUIRE(pfm::run_match_command(options) != 0);
}

static void pipeline_eval_is_deferred_to_task_8() {
    pfm::CliOptions options;
    options.pairs = "pairs.txt";
    options.checkpoint = "checkpoint.pt";
    options.output = "report.json";

    PFM_REQUIRE(pfm::run_eval_command(options) != 0);
}

void register_pipeline_tests() {
    register_test("pipeline_train_writes_loadable_checkpoint", pipeline_train_writes_loadable_checkpoint);
    register_test("pipeline_extract_writes_loadable_feature_file", pipeline_extract_writes_loadable_feature_file);
    register_test("pipeline_export_writes_loadable_checkpoint", pipeline_export_writes_loadable_checkpoint);
    register_test("pipeline_match_is_deferred_to_task_8", pipeline_match_is_deferred_to_task_8);
    register_test("pipeline_eval_is_deferred_to_task_8", pipeline_eval_is_deferred_to_task_8);
}
