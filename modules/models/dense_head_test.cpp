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

static void dense_head_uses_coordinate_channels_for_position_dependent_offsets() {
    torch::manual_seed(1);
    pfm::DenseHead head(1);
    auto feature_a = torch::zeros({1, 1, 4, 4}, torch::kFloat32);
    auto feature_b = torch::zeros({1, 1, 4, 4}, torch::kFloat32);

    auto output = head->forward(feature_a, feature_b);

    PFM_REQUIRE(output.offsets.index({0, 0}).std().item<float>() > 1.0e-5F ||
                output.offsets.index({0, 1}).std().item<float>() > 1.0e-5F);
}

void register_dense_head_tests() {
    register_test("dense head outputs coarse confidence and offsets", dense_head_outputs_coarse_confidence_and_offsets);
    register_test("dense head uses coordinate channels for position dependent offsets",
                  dense_head_uses_coordinate_channels_for_position_dependent_offsets);
}
