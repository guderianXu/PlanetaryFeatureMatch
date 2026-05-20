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

static void sparse_head_uses_shared_context_tower() {
    pfm::SparseHead head(32, 128);
    bool found_first_context = false;
    bool found_second_context = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "context.0.weight") {
            found_first_context = true;
            PFM_REQUIRE(parameter.value().size(2) == 3);
            PFM_REQUIRE(parameter.value().size(3) == 3);
        }
        if (parameter.key() == "context.2.weight") {
            found_second_context = true;
            PFM_REQUIRE(parameter.value().size(2) == 3);
            PFM_REQUIRE(parameter.value().size(3) == 3);
        }
    }

    PFM_REQUIRE(found_first_context);
    PFM_REQUIRE(found_second_context);
}

static void sparse_head_rotation_pooled_descriptors_stay_normalized() {
    pfm::SparseHead head(4, 8);
    auto feature = torch::rand({1, 4, 16, 16}, torch::kFloat32);

    auto descriptors = head->forward(feature).descriptors;
    auto norms = descriptors.norm(2, 1);

    PFM_REQUIRE(torch::isfinite(descriptors).all().item<bool>());
    PFM_REQUIRE(torch::allclose(norms, torch::ones_like(norms), 1.0e-5, 1.0e-5));
}

static void sparse_head_uses_cyclic_descriptor_slots() {
    pfm::SparseHead head(4, 8);
    bool found_descriptor_output = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptors.2.weight") {
            found_descriptor_output = true;
            PFM_REQUIRE(parameter.value().size(0) == 8);
            PFM_REQUIRE(parameter.value().size(1) == 4);
            PFM_REQUIRE(parameter.value().size(2) == 1);
            PFM_REQUIRE(parameter.value().size(3) == 1);
        }
    }

    PFM_REQUIRE(found_descriptor_output);
}

void register_sparse_head_tests() {
    register_test("sparse head outputs expected maps", sparse_head_outputs_expected_maps);
    register_test("sparse head descriptors use local context", sparse_head_descriptors_use_local_context);
    register_test("sparse head uses shared context tower", sparse_head_uses_shared_context_tower);
    register_test("sparse head rotation pooled descriptors stay normalized",
                  sparse_head_rotation_pooled_descriptors_stay_normalized);
    register_test("sparse head uses cyclic descriptor slots", sparse_head_uses_cyclic_descriptor_slots);
}
