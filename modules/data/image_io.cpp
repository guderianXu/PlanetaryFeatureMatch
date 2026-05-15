#include "data/image_io.h"

#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pfm {

namespace {

cv::Mat convert_to_rgb_if_needed(const cv::Mat& image) {
    if (image.channels() == 1) {
        return image;
    }

    cv::Mat rgb_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        return rgb_image;
    }
    if (image.channels() == 4) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGRA2RGB);
        return rgb_image;
    }

    throw std::invalid_argument("image must have 1, 3, or 4 channels");
}

float depth_scale(int depth) {
    if (depth == CV_8U) {
        return 1.0F / 255.0F;
    }
    if (depth == CV_16U) {
        return 1.0F / 65535.0F;
    }
    throw std::invalid_argument("image depth must be 8-bit or 16-bit unsigned");
}

}  // namespace

torch::Tensor load_image_tensor(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::invalid_argument("image could not be loaded: " + path);
    }

    const float scale = depth_scale(image.depth());
    cv::Mat rgb_image = convert_to_rgb_if_needed(image);
    cv::Mat contiguous_image = rgb_image.isContinuous() ? rgb_image : rgb_image.clone();

    auto options = torch::TensorOptions().device(torch::kCPU);
    if (contiguous_image.depth() == CV_8U) {
        options = options.dtype(torch::kUInt8);
    } else {
        options = options.dtype(torch::kUInt16);
    }

    const int64_t height = contiguous_image.rows;
    const int64_t width = contiguous_image.cols;
    const int64_t channels = contiguous_image.channels();
    auto tensor = torch::from_blob(contiguous_image.data, {height, width, channels}, options).clone();
    return (tensor.permute({2, 0, 1}).to(torch::kFloat32) * scale).contiguous();
}

}  // namespace pfm
