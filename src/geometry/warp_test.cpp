#include "tests/test_harness.h"

#include <torch/torch.h>

#include "geometry/warp.h"

static void identity_warp_maps_pixels_to_themselves() {
    auto transform = pfm::AffineTransform::identity();

    auto field = pfm::dense_warp_field(3, 4, transform, torch::kCPU);

    PFM_REQUIRE(field.sizes() == torch::IntArrayRef({3, 4, 2}));
    PFM_REQUIRE_CLOSE(field.index({2, 3, 0}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(field.index({2, 3, 1}).item<float>(), 2.0F, 1.0e-6F);
}

static void translation_warp_offsets_coordinates() {
    auto transform = pfm::AffineTransform::translation(2.0F, -1.0F);

    auto field = pfm::dense_warp_field(3, 4, transform, torch::kCPU);

    PFM_REQUIRE_CLOSE(field.index({1, 1, 0}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(field.index({1, 1, 1}).item<float>(), 0.0F, 1.0e-6F);
}

static void valid_mask_rejects_out_of_bounds_coordinates() {
    auto transform = pfm::AffineTransform::translation(10.0F, 0.0F);
    auto field = pfm::dense_warp_field(3, 4, transform, torch::kCPU);

    auto mask = pfm::valid_warp_mask(field, 3, 4);

    PFM_REQUIRE(mask.scalar_type() == torch::kBool);
    PFM_REQUIRE(mask.sizes() == torch::IntArrayRef({3, 4}));
    PFM_REQUIRE(mask.sum().item<int64_t>() == 0);
}

void register_warp_tests() {
    register_test("identity warp maps pixels to themselves", identity_warp_maps_pixels_to_themselves);
    register_test("translation warp offsets coordinates", translation_warp_offsets_coordinates);
    register_test("valid mask rejects out of bounds coordinates", valid_mask_rejects_out_of_bounds_coordinates);
}
