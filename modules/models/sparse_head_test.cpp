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

static void sparse_head_descriptors_use_local_context() {
    pfm::SparseHead head(4, 8);
    bool found_local_descriptor_kernel = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptors.0.weight") {
            found_local_descriptor_kernel = true;
            PFM_REQUIRE(parameter.value().size(2) == 3);
            PFM_REQUIRE(parameter.value().size(3) == 3);
        }
    }

    PFM_REQUIRE(found_local_descriptor_kernel);
}

void register_sparse_head_tests() {
    register_test("sparse head outputs expected maps", sparse_head_outputs_expected_maps);
    register_test("sparse head descriptors use local context", sparse_head_descriptors_use_local_context);
}
