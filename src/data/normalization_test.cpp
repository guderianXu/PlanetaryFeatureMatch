#include "tests/test_harness.h"

#include <torch/torch.h>

#include "data/normalization.h"

static void uint8_tensor_normalizes_to_unit_range() {
    auto tensor = torch::tensor({{{0, 255}}}, torch::kUInt8);

    auto normalized = pfm::normalize_u8(tensor);

    PFM_REQUIRE(normalized.scalar_type() == torch::kFloat32);
    PFM_REQUIRE(normalized.sizes() == torch::IntArrayRef({1, 1, 2}));
    PFM_REQUIRE_CLOSE(normalized.index({0, 0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(normalized.index({0, 0, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void uint16_tensor_normalizes_to_unit_range() {
    auto tensor = torch::tensor({{{0, 65535}}}, torch::kUInt16);

    auto normalized = pfm::normalize_u16(tensor);

    PFM_REQUIRE(normalized.scalar_type() == torch::kFloat32);
    PFM_REQUIRE(normalized.sizes() == torch::IntArrayRef({1, 1, 2}));
    PFM_REQUIRE_CLOSE(normalized.index({0, 0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(normalized.index({0, 0, 1}).item<float>(), 1.0F, 1.0e-6F);
}

static void local_contrast_preserves_shape() {
    auto tensor = torch::ones({1, 8, 8}, torch::kFloat32) * 0.5F;

    auto normalized = pfm::local_contrast_normalize(tensor, 3);

    PFM_REQUIRE(normalized.sizes() == tensor.sizes());
    PFM_REQUIRE(torch::isfinite(normalized).all().item<bool>());
}

void register_normalization_tests() {
    register_test("uint8 tensor normalizes to unit range", uint8_tensor_normalizes_to_unit_range);
    register_test("uint16 tensor normalizes to unit range", uint16_tensor_normalizes_to_unit_range);
    register_test("local contrast preserves shape", local_contrast_preserves_shape);
}
