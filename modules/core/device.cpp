#include "core/device.h"

#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>

namespace pfm {
namespace {

bool is_digits(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    for (const auto character : value) {
        if (!std::isdigit(static_cast<unsigned char>(character))) {
            return false;
        }
    }
    return true;
}

int parse_cuda_index(const std::string& requested) {
    constexpr auto CUDA_PREFIX_SIZE = 5;
    const auto index_text = requested.substr(CUDA_PREFIX_SIZE);
    if (!is_digits(index_text)) {
        throw std::invalid_argument("invalid device: " + requested);
    }

    const auto index = std::stoll(index_text);
    if (index > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("cuda device index out of range: " + requested);
    }
    return static_cast<int>(index);
}

torch::Device resolve_cuda_device(int index, const std::string& requested) {
    if (!torch::cuda::is_available()) {
        throw std::invalid_argument("cuda requested but CUDA is not available: " + requested);
    }
    const auto device_count = torch::cuda::device_count();
    if (index < 0 || index >= device_count) {
        throw std::invalid_argument("cuda device index out of range: " + requested);
    }
    return torch::Device(torch::kCUDA, index);
}

}  // namespace

torch::Device resolve_compute_device(const std::string& requested) {
    if (requested == "cpu") {
        return torch::Device(torch::kCPU);
    }
    if (requested == "cuda") {
        return resolve_cuda_device(0, requested);
    }
    if (requested.rfind("cuda:", 0) == 0) {
        return resolve_cuda_device(parse_cuda_index(requested), requested);
    }
    throw std::invalid_argument("invalid device: " + requested);
}

}  // namespace pfm
