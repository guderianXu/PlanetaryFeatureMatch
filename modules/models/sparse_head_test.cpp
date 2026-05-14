#include "tests/test_harness.h"

#include <torch/torch.h>

#include "models/sparse_head.h"

static void sparse_head_outputs_expected_maps() {
    pfm::SparseHead head(16, 64);
    auto feature = torch::randn({2, 16, 32, 32}, torch::kFloat32);

    auto output = head->forward(feature);

    PFM_REQUIRE(output.heatmap.sizes() == torch::IntArrayRef({2, 1, 32, 32}));
    PFM_REQUIRE(output.descriptors.sizes() == torch::IntArrayRef({2, 64, 32, 32}));
    PFM_REQUIRE(output.scale.sizes() == torch::IntArrayRef({2, 1, 32, 32}));
    PFM_REQUIRE(output.orientation.sizes() == torch::IntArrayRef({2, 2, 32, 32}));
    PFM_REQUIRE(output.affine.sizes() == torch::IntArrayRef({2, 4, 32, 32}));
}

void register_sparse_head_tests() {
    register_test("sparse head outputs expected maps", sparse_head_outputs_expected_maps);
}
