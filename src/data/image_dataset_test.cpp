#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "data/image_dataset.h"
#include "tests/test_harness.h"

namespace {

class TempImageDirectory {
public:
    explicit TempImageDirectory(const std::string& stem) {
        const auto suffix = std::to_string(static_cast<long long>(getpid())) + "_" +
                            std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempImageDirectory() {
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

void require_image_written(const std::filesystem::path& path, const cv::Mat& image) {
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

void write_text_file(const std::filesystem::path& path) {
    FILE* file = std::fopen(path.string().c_str(), "wb");
    PFM_REQUIRE(file != nullptr);
    PFM_REQUIRE(std::fputs("ignored", file) >= 0);
    PFM_REQUIRE(std::fclose(file) == 0);
}

}  // namespace

static void image_dataset_filters_supported_extensions_and_sorts() {
    TempImageDirectory temp_dir("pfm_image_dataset_filter");
    cv::Mat image(1, 1, CV_8UC1, cv::Scalar(127));

    require_image_written(temp_dir.file("b.PNG"), image);
    require_image_written(temp_dir.file("a.jpg"), image);
    write_text_file(temp_dir.file("ignore.txt"));

    pfm::ImageDataset dataset(temp_dir.path().string());

    PFM_REQUIRE(dataset.size() == 2);
    PFM_REQUIRE(dataset.path(0).find("a.jpg") != std::string::npos);
    PFM_REQUIRE(dataset.path(1).find("b.PNG") != std::string::npos);
}

static void image_dataset_loads_tensor_sample() {
    TempImageDirectory temp_dir("pfm_image_dataset_load");
    cv::Mat image(2, 2, CV_8UC1, cv::Scalar(64));
    require_image_written(temp_dir.file("image.png"), image);

    pfm::ImageDataset dataset(temp_dir.path().string());
    auto tensor = dataset.load(0);

    PFM_REQUIRE(tensor.sizes() == torch::IntArrayRef({1, 2, 2}));
}

static void image_dataset_rejects_empty_directory() {
    TempImageDirectory temp_dir("pfm_image_dataset_empty");

    PFM_REQUIRE_INVALID_ARG(pfm::ImageDataset(temp_dir.path().string()));
}

static void image_dataset_rejects_out_of_range_path_index() {
    TempImageDirectory temp_dir("pfm_image_dataset_out_of_range");
    cv::Mat image(1, 1, CV_8UC1, cv::Scalar(127));
    require_image_written(temp_dir.file("image.png"), image);
    pfm::ImageDataset dataset(temp_dir.path().string());

    PFM_REQUIRE_THROWS_AS(dataset.path(1), std::out_of_range);
}

void register_image_dataset_tests() {
    register_test("image_dataset_filters_supported_extensions_and_sorts",
                  image_dataset_filters_supported_extensions_and_sorts);
    register_test("image_dataset_loads_tensor_sample", image_dataset_loads_tensor_sample);
    register_test("image_dataset_rejects_empty_directory", image_dataset_rejects_empty_directory);
    register_test("image_dataset_rejects_out_of_range_path_index", image_dataset_rejects_out_of_range_path_index);
}
