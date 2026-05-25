#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>
#include <torch/serialize.h>

#include "data/image_io.h"
#include "infer/match_metrics.h"

namespace {

struct Options {
    std::string image_a;
    std::string image_b;
    std::string warp;
    double min_intensity = 0.08;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto require_value = [&](const char* name) -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (arg == "--image-a") {
            options.image_a = require_value("--image-a");
        } else if (arg == "--image-b") {
            options.image_b = require_value("--image-b");
        } else if (arg == "--warp-a-to-b") {
            options.warp = require_value("--warp-a-to-b");
        } else if (arg == "--min-intensity") {
            options.min_intensity = std::stod(require_value("--min-intensity"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: pfm_warp_image_diagnostics --image-a a.png --image-b b.png "
                         "--warp-a-to-b pair.pt [--min-intensity 0.08]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.image_a.empty() || options.image_b.empty() || options.warp.empty()) {
        throw std::invalid_argument("image-a, image-b, and warp-a-to-b are required");
    }
    if (!std::isfinite(options.min_intensity) || options.min_intensity < 0.0 || options.min_intensity > 1.0) {
        throw std::invalid_argument("min-intensity must be in [0, 1]");
    }
    return options;
}

torch::Tensor grayscale(const torch::Tensor& image) {
    if (image.dim() != 3) {
        throw std::invalid_argument("image tensor must have CHW shape");
    }
    if (image.size(0) == 1) {
        return image.squeeze(0).contiguous();
    }
    return image.mean(0).contiguous();
}

torch::Tensor read_archive_tensor(const std::string& path, const char* name) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::Tensor tensor;
    archive.read(name, tensor);
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string("archive tensor is missing: ") + name);
    }
    return tensor.to(torch::kCPU, torch::kFloat32).contiguous();
}

double mean_abs_diff_or_negative_one(const torch::Tensor& lhs, const torch::Tensor& rhs) {
    auto left = lhs.to(torch::kCPU, torch::kFloat32).contiguous();
    auto right = rhs.to(torch::kCPU, torch::kFloat32).contiguous();
    if (left.sizes() != right.sizes()) {
        return -1.0;
    }
    return (left - right).abs().mean().item<double>();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        auto image_a = grayscale(pfm::load_image_tensor(options.image_a)).to(torch::kCPU, torch::kFloat32).contiguous();
        auto image_b = grayscale(pfm::load_image_tensor(options.image_b)).to(torch::kCPU, torch::kFloat32).contiguous();
        auto warp = pfm::load_warp_a_to_b_tensor(options.warp).to(torch::kCPU, torch::kFloat32).contiguous();
        auto archive_a = grayscale(read_archive_tensor(options.warp, "view_a"));
        auto archive_b = grayscale(read_archive_tensor(options.warp, "view_b"));
        if (warp.size(0) != image_a.size(0) || warp.size(1) != image_a.size(1) ||
            image_b.size(0) != image_a.size(0) || image_b.size(1) != image_a.size(1)) {
            throw std::invalid_argument("image and warp sizes must match");
        }

        const auto height = image_a.size(0);
        const auto width = image_a.size(1);
        const auto* a = image_a.data_ptr<float>();
        const auto* b = image_b.data_ptr<float>();
        const auto* w = warp.data_ptr<float>();
        int64_t total_pixels = height * width;
        int64_t source_bright = 0;
        int64_t warp_in_bounds = 0;
        int64_t bright_source_in_bounds = 0;
        int64_t bright_source_to_bright_target = 0;
        double abs_diff_sum = 0.0;
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                const auto offset = y * width + x;
                const bool bright_a = a[offset] > options.min_intensity;
                if (bright_a) {
                    ++source_bright;
                }
                const float target_x = w[offset * 2];
                const float target_y = w[offset * 2 + 1];
                const bool in_bounds =
                    target_x >= 0.0F && target_x <= static_cast<float>(width - 1) &&
                    target_y >= 0.0F && target_y <= static_cast<float>(height - 1);
                if (!in_bounds) {
                    continue;
                }
                ++warp_in_bounds;
                if (!bright_a) {
                    continue;
                }
                ++bright_source_in_bounds;
                const auto ix = std::min<int64_t>(width - 1, std::max<int64_t>(0, std::llround(target_x)));
                const auto iy = std::min<int64_t>(height - 1, std::max<int64_t>(0, std::llround(target_y)));
                const auto target = b[iy * width + ix];
                if (target > options.min_intensity) {
                    ++bright_source_to_bright_target;
                }
                abs_diff_sum += std::abs(static_cast<double>(a[offset]) - static_cast<double>(target));
            }
        }
        const auto ratio = [](int64_t numerator, int64_t denominator) {
            return denominator == 0 ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator);
        };
        std::cout << "total_pixels=" << total_pixels
                  << " source_bright=" << source_bright
                  << " warp_in_bounds=" << warp_in_bounds
                  << " bright_source_in_bounds=" << bright_source_in_bounds
                  << " bright_source_to_bright_target=" << bright_source_to_bright_target
                  << " in_bounds_fraction=" << ratio(warp_in_bounds, total_pixels)
                  << " bright_transfer_fraction=" << ratio(bright_source_to_bright_target, bright_source_in_bounds)
                  << " mean_bright_abs_diff="
                  << (bright_source_in_bounds == 0 ? 0.0 : abs_diff_sum / static_cast<double>(bright_source_in_bounds))
                  << " archive_view_a_png_abs_diff=" << mean_abs_diff_or_negative_one(archive_a, image_a)
                  << " archive_view_b_png_abs_diff=" << mean_abs_diff_or_negative_one(archive_b, image_b)
                  << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "warp image diagnostics failed: " << e.what() << '\n';
        return 1;
    }
}
