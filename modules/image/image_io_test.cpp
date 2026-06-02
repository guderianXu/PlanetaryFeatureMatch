#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <unistd.h>

#include "image/image_io.h"
#include "tests/test_harness.h"

namespace
{

class TempImageFile
{
  public:
    explicit TempImageFile(const std::string& stem)
    {
        const auto suffix =
            std::to_string(static_cast<long long>(getpid())) + "_" + std::to_string(std::random_device{}());
        _path = (std::filesystem::temp_directory_path() / (stem + "_" + suffix + ".png")).string();
    }

    ~TempImageFile()
    {
        std::remove(_path.c_str());
    }

    const std::string& path() const
    {
        return _path;
    }

  private:
    std::string _path;
};

void require_image_written(const std::string& path, const cv::Mat& image)
{
    PFM_REQUIRE(cv::imwrite(path, image));
}

} // namespace

static void load_u8_grayscale_image_returns_chw_float_tensor()
{
    TempImageFile temp_file("pfm_image_io_u8_grayscale");
    const std::string& path = temp_file.path();
    cv::Mat image(2, 3, CV_8UC1);
    image.at<uint8_t>(0, 0) = 0;
    image.at<uint8_t>(0, 1) = 64;
    image.at<uint8_t>(0, 2) = 128;
    image.at<uint8_t>(1, 0) = 192;
    image.at<uint8_t>(1, 1) = 224;
    image.at<uint8_t>(1, 2) = 255;
    require_image_written(path, image);

    auto tensor = pfm::load_image_tensor(path);

    PFM_REQUIRE(tensor.sizes() == torch::IntArrayRef({1, 2, 3}));
    PFM_REQUIRE(tensor.scalar_type() == torch::kFloat32);
    PFM_REQUIRE(tensor.is_contiguous());
    PFM_REQUIRE_CLOSE(tensor.index({0, 0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(tensor.index({0, 1, 2}).item<float>(), 1.0F, 1.0e-6F);
}

static void load_u16_grayscale_image_returns_unit_tensor()
{
    TempImageFile temp_file("pfm_image_io_u16_grayscale");
    const std::string& path = temp_file.path();
    cv::Mat image(1, 2, CV_16UC1);
    image.at<uint16_t>(0, 0) = 0;
    image.at<uint16_t>(0, 1) = 65535;
    require_image_written(path, image);

    auto tensor = pfm::load_image_tensor(path);

    PFM_REQUIRE(tensor.sizes() == torch::IntArrayRef({1, 1, 2}));
    PFM_REQUIRE_CLOSE(tensor.index({0, 0, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void load_u8_color_image_converts_bgr_to_rgb()
{
    TempImageFile temp_file("pfm_image_io_u8_color");
    const std::string& path = temp_file.path();
    cv::Mat image(1, 1, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    require_image_written(path, image);

    auto tensor = pfm::load_image_tensor(path);

    PFM_REQUIRE(tensor.sizes() == torch::IntArrayRef({3, 1, 1}));
    PFM_REQUIRE(tensor.is_contiguous());
    PFM_REQUIRE_CLOSE(tensor.index({0, 0, 0}).item<float>(), 30.0F / 255.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(tensor.index({1, 0, 0}).item<float>(), 20.0F / 255.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(tensor.index({2, 0, 0}).item<float>(), 10.0F / 255.0F, 1.0e-6F);
}

static void load_u8_bgra_image_drops_alpha_and_returns_rgb()
{
    TempImageFile temp_file("pfm_image_io_u8_bgra");
    const std::string& path = temp_file.path();
    cv::Mat image(1, 1, CV_8UC4);
    image.at<cv::Vec4b>(0, 0) = cv::Vec4b(10, 20, 30, 40);
    require_image_written(path, image);

    auto tensor = pfm::load_image_tensor(path);

    PFM_REQUIRE(tensor.sizes() == torch::IntArrayRef({3, 1, 1}));
    PFM_REQUIRE(tensor.is_contiguous());
    PFM_REQUIRE_CLOSE(tensor.index({0, 0, 0}).item<float>(), 30.0F / 255.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(tensor.index({1, 0, 0}).item<float>(), 20.0F / 255.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(tensor.index({2, 0, 0}).item<float>(), 10.0F / 255.0F, 1.0e-6F);
}

static void load_missing_image_throws()
{
    TempImageFile temp_file("pfm_missing_image");
    const std::string& path = temp_file.path();
    PFM_REQUIRE(!std::filesystem::exists(path));

    PFM_REQUIRE_INVALID_ARG(pfm::load_image_tensor(path));
}

void register_image_io_tests()
{
    register_test("load_u8_grayscale_image_returns_chw_float_tensor", load_u8_grayscale_image_returns_chw_float_tensor);
    register_test("load_u16_grayscale_image_returns_unit_tensor", load_u16_grayscale_image_returns_unit_tensor);
    register_test("load_u8_color_image_converts_bgr_to_rgb", load_u8_color_image_converts_bgr_to_rgb);
    register_test("load_u8_bgra_image_drops_alpha_and_returns_rgb", load_u8_bgra_image_drops_alpha_and_returns_rgb);
    register_test("load_missing_image_throws", load_missing_image_throws);
}
