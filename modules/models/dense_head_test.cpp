#include "tests/test_harness.h"

#include <torch/torch.h>

#include "models/dense_head.h"

static void dense_head_outputs_coarse_confidence_and_offsets() {
    pfm::DenseHead head(64);
    auto feature_a = torch::randn({2, 64, 8, 8}, torch::kFloat32);
    auto feature_b = torch::randn({2, 64, 8, 8}, torch::kFloat32);

    auto output = head->forward(feature_a, feature_b);

    PFM_REQUIRE(output.confidence.sizes() == torch::IntArrayRef({2, 1, 8, 8}));
    PFM_REQUIRE(output.offsets.sizes() == torch::IntArrayRef({2, 2, 8, 8}));
}

void register_dense_head_tests() {
    register_test("dense head outputs coarse confidence and offsets", dense_head_outputs_coarse_confidence_and_offsets);
}
