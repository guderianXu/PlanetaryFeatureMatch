#include "tests/test_harness.h"

#include <torch/torch.h>

#include "core/tensor_utils.h"

static void require_invalid_image(const torch::Tensor& tensor) {
    bool thrown = false;
    try {
        pfm::require_chw_image(tensor);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    PFM_REQUIRE(thrown);
}

static void chw_float_tensor_is_accepted() {
    auto tensor = torch::zeros({1, 8, 9}, torch::kFloat32);
    pfm::require_chw_image(tensor);
    PFM_REQUIRE(pfm::height(tensor) == 8);
    PFM_REQUIRE(pfm::width(tensor) == 9);
}

static void channels_returns_channel_count() {
    auto tensor = torch::zeros({3, 4, 5}, torch::kFloat32);
    PFM_REQUIRE(pfm::channels(tensor) == 3);
}

static void xy_grid_contains_x_y_coordinates() {
    auto grid = pfm::make_xy_grid(2, 3, torch::kCPU);
    PFM_REQUIRE(grid.sizes() == torch::IntArrayRef({2, 3, 2}));
    PFM_REQUIRE_CLOSE(grid.index({1, 2, 0}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(grid.index({1, 2, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void hw_tensor_is_rejected() {
    auto tensor = torch::zeros({8, 9}, torch::kFloat32);
    require_invalid_image(tensor);
}

static void invalid_dtype_is_rejected() {
    auto tensor = torch::zeros({1, 2, 3}, torch::kFloat64);
    require_invalid_image(tensor);
}

static void invalid_channel_count_is_rejected() {
    auto tensor = torch::zeros({2, 4, 5}, torch::kFloat32);
    require_invalid_image(tensor);
}

static void undefined_tensor_is_rejected() {
    require_invalid_image(torch::Tensor());
}

static void zero_height_is_rejected() {
    auto tensor = torch::zeros({1, 0, 3}, torch::kFloat32);
    require_invalid_image(tensor);
}

static void zero_width_is_rejected() {
    auto tensor = torch::zeros({1, 2, 0}, torch::kFloat32);
    require_invalid_image(tensor);
}

void register_tensor_utils_tests() {
    register_test("tensor utils accept CHW float image", chw_float_tensor_is_accepted);
    register_test("tensor utils return channel count", channels_returns_channel_count);
    register_test("tensor utils make xy grid with coordinates", xy_grid_contains_x_y_coordinates);
    register_test("tensor utils reject HW tensor", hw_tensor_is_rejected);
    register_test("tensor utils reject invalid dtype", invalid_dtype_is_rejected);
    register_test("tensor utils reject invalid channel count", invalid_channel_count_is_rejected);
    register_test("tensor utils reject undefined tensor", undefined_tensor_is_rejected);
    register_test("tensor utils reject zero height", zero_height_is_rejected);
    register_test("tensor utils reject zero width", zero_width_is_rejected);
}
