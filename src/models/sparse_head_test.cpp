#include "tests/test_harness.h"

#include <filesystem>

#include <torch/torch.h>

#include "models/sparse_head.h"

namespace {

class LegacySparseHeadImpl : public torch::nn::Module {
public:
    LegacySparseHeadImpl(int64_t input_channels, int64_t descriptor_dim) {
        register_module(
            "context",
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, input_channels, 3).padding(1)),
                torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, input_channels, 3).padding(1)),
                torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))));
        register_module("heatmap", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 1, 1)));
        register_module(
            "descriptors",
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, input_channels, 3).padding(1)),
                torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, descriptor_dim, 1))));
        register_module("descriptor_skip", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, descriptor_dim, 1)));
        register_module("scale", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 1, 1)));
        register_module("orientation", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 2, 1)));
        register_module("affine", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 4, 1)));
    }
};

TORCH_MODULE(LegacySparseHead);

}  // namespace

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

static void sparse_head_descriptors_use_deeper_local_context() {
    pfm::SparseHead head(4, 8);
    int local_descriptor_layers = 0;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key().rfind("descriptors.", 0) == 0 &&
            parameter.key().find(".weight") != std::string::npos &&
            parameter.value().dim() == 4 &&
            parameter.value().size(2) == 3 &&
            parameter.value().size(3) == 3) {
            ++local_descriptor_layers;
        }
    }

    PFM_REQUIRE(local_descriptor_layers >= 2);
}

static void sparse_head_descriptors_use_residual_blocks() {
    pfm::SparseHead head(4, 8);
    bool found_first_residual = false;
    bool found_second_residual = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptors.2.conv2.weight") {
            found_first_residual = true;
            PFM_REQUIRE(parameter.value().size(2) == 3);
            PFM_REQUIRE(parameter.value().size(3) == 3);
        }
        if (parameter.key() == "descriptors.3.conv2.weight") {
            found_second_residual = true;
            PFM_REQUIRE(parameter.value().size(2) == 3);
            PFM_REQUIRE(parameter.value().size(3) == 3);
        }
    }

    PFM_REQUIRE(found_first_residual);
    PFM_REQUIRE(found_second_residual);
}

static void sparse_head_descriptors_use_four_residual_blocks() {
    pfm::SparseHead head(4, 8);
    int residual_blocks = 0;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key().rfind("descriptors.", 0) == 0 &&
            parameter.key().find(".conv2.weight") != std::string::npos) {
            ++residual_blocks;
            PFM_REQUIRE(parameter.value().size(2) == 3);
            PFM_REQUIRE(parameter.value().size(3) == 3);
        }
    }

    PFM_REQUIRE(residual_blocks >= 4);
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

static void sparse_head_descriptors_stay_normalized() {
    pfm::SparseHead head(4, 8);
    auto feature = torch::rand({1, 4, 16, 16}, torch::kFloat32);

    auto descriptors = head->forward(feature).descriptors;
    auto norms = descriptors.norm(2, 1);

    PFM_REQUIRE(torch::isfinite(descriptors).all().item<bool>());
    PFM_REQUIRE(torch::allclose(norms, torch::ones_like(norms), 1.0e-5, 1.0e-5));
}

static void sparse_head_descriptor_projection_has_requested_channels() {
    pfm::SparseHead head(4, 8);
    bool found_descriptor_output = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptors.6.weight") {
            found_descriptor_output = true;
            PFM_REQUIRE(parameter.value().size(0) == 8);
            PFM_REQUIRE(parameter.value().size(1) == 4);
            PFM_REQUIRE(parameter.value().size(2) == 1);
            PFM_REQUIRE(parameter.value().size(3) == 1);
        }
    }

    PFM_REQUIRE(found_descriptor_output);
}

static void sparse_head_descriptors_use_multiscale_projection() {
    pfm::SparseHead head(4, 8);
    bool found_multiscale = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptor_multiscale.weight") {
            found_multiscale = true;
            PFM_REQUIRE(parameter.value().size(0) == 8);
            PFM_REQUIRE(parameter.value().size(1) == 12);
            PFM_REQUIRE(parameter.value().size(2) == 1);
            PFM_REQUIRE(parameter.value().size(3) == 1);
        }
    }

    PFM_REQUIRE(found_multiscale);
}

static void sparse_head_descriptors_use_attention_gate() {
    pfm::SparseHead head(4, 8);
    bool found_attention = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptor_attention.weight") {
            found_attention = true;
            PFM_REQUIRE(parameter.value().size(0) == 8);
            PFM_REQUIRE(parameter.value().size(1) == 12);
            PFM_REQUIRE(parameter.value().size(2) == 1);
            PFM_REQUIRE(parameter.value().size(3) == 1);
        }
    }

    PFM_REQUIRE(found_attention);
}

static void sparse_head_descriptors_use_anisotropic_viewpoint_context() {
    pfm::SparseHead head(4, 8);
    bool found_heatmap_viewpoint_context = false;
    bool found_viewpoint_context = false;
    bool found_viewpoint_attention = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "heatmap_viewpoint_context.weight") {
            found_heatmap_viewpoint_context = true;
            PFM_REQUIRE(parameter.value().size(0) == 1);
            PFM_REQUIRE(parameter.value().size(1) == 20);
            PFM_REQUIRE(parameter.value().size(2) == 1);
            PFM_REQUIRE(parameter.value().size(3) == 1);
        }
        if (parameter.key() == "descriptor_viewpoint_context.weight") {
            found_viewpoint_context = true;
            PFM_REQUIRE(parameter.value().size(0) == 8);
            PFM_REQUIRE(parameter.value().size(1) == 20);
            PFM_REQUIRE(parameter.value().size(2) == 1);
            PFM_REQUIRE(parameter.value().size(3) == 1);
        }
        if (parameter.key() == "descriptor_viewpoint_attention.weight") {
            found_viewpoint_attention = true;
            PFM_REQUIRE(parameter.value().size(0) == 8);
            PFM_REQUIRE(parameter.value().size(1) == 20);
            PFM_REQUIRE(parameter.value().size(2) == 1);
            PFM_REQUIRE(parameter.value().size(3) == 1);
        }
    }

    PFM_REQUIRE(found_heatmap_viewpoint_context);
    PFM_REQUIRE(found_viewpoint_context);
    PFM_REQUIRE(found_viewpoint_attention);
}

static void sparse_head_viewpoint_context_starts_as_residual_noop() {
    pfm::SparseHead head(4, 8);
    bool found_heatmap_weight = false;
    bool found_heatmap_bias = false;
    bool found_context_weight = false;
    bool found_context_bias = false;
    bool found_attention_weight = false;
    bool found_attention_bias = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "heatmap_viewpoint_context.weight") {
            found_heatmap_weight = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
        if (parameter.key() == "heatmap_viewpoint_context.bias") {
            found_heatmap_bias = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
        if (parameter.key() == "descriptor_viewpoint_context.weight") {
            found_context_weight = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
        if (parameter.key() == "descriptor_viewpoint_context.bias") {
            found_context_bias = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
        if (parameter.key() == "descriptor_viewpoint_attention.weight") {
            found_attention_weight = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
        if (parameter.key() == "descriptor_viewpoint_attention.bias") {
            found_attention_bias = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
    }

    PFM_REQUIRE(found_heatmap_weight);
    PFM_REQUIRE(found_heatmap_bias);
    PFM_REQUIRE(found_context_weight);
    PFM_REQUIRE(found_context_bias);
    PFM_REQUIRE(found_attention_weight);
    PFM_REQUIRE(found_attention_bias);
}

static void sparse_head_descriptors_use_orientation_alignment_projection() {
    pfm::SparseHead head(4, 8);
    bool found_alignment = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptor_orientation_alignment.weight") {
            found_alignment = true;
            PFM_REQUIRE(parameter.value().size(0) == 8);
            PFM_REQUIRE(parameter.value().size(1) == 8);
            PFM_REQUIRE(parameter.value().size(2) == 1);
            PFM_REQUIRE(parameter.value().size(3) == 1);
        }
    }

    PFM_REQUIRE(found_alignment);
}

static void sparse_head_descriptors_use_dilated_context_residual() {
    pfm::SparseHead head(4, 8);
    bool found_dilated_weight = false;
    bool found_dilated_bias = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptor_dilated_context.weight") {
            found_dilated_weight = true;
            PFM_REQUIRE(parameter.value().size(0) == 8);
            PFM_REQUIRE(parameter.value().size(1) == 8);
            PFM_REQUIRE(parameter.value().size(2) == 3);
            PFM_REQUIRE(parameter.value().size(3) == 3);
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
        if (parameter.key() == "descriptor_dilated_context.bias") {
            found_dilated_bias = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
    }

    PFM_REQUIRE(found_dilated_weight);
    PFM_REQUIRE(found_dilated_bias);
}

static void sparse_head_orientation_alignment_starts_as_residual_noop() {
    pfm::SparseHead head(4, 8);
    bool found_alignment_weight = false;
    bool found_alignment_bias = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptor_orientation_alignment.weight") {
            found_alignment_weight = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
        if (parameter.key() == "descriptor_orientation_alignment.bias") {
            found_alignment_bias = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
    }

    PFM_REQUIRE(found_alignment_weight);
    PFM_REQUIRE(found_alignment_bias);
}

static void sparse_head_loads_legacy_descriptor_tower_checkpoint() {
    LegacySparseHead legacy_head(4, 8);
    torch::Tensor legacy_projection_weight;
    torch::Tensor legacy_projection_bias;
    {
        torch::NoGradGuard no_grad;
        for (const auto& parameter : legacy_head->named_parameters()) {
            if (parameter.key() == "descriptors.2.weight") {
                parameter.value().fill_(0.25F);
                legacy_projection_weight = parameter.value().detach().clone();
            }
            if (parameter.key() == "descriptors.2.bias") {
                parameter.value().fill_(0.125F);
                legacy_projection_bias = parameter.value().detach().clone();
            }
        }
    }
    torch::serialize::OutputArchive output_archive;
    legacy_head->save(output_archive);
    const auto checkpoint = std::filesystem::temp_directory_path() / "pfm_legacy_sparse_head.pt";
    output_archive.save_to(checkpoint.string());

    torch::serialize::InputArchive input_archive;
    input_archive.load_from(checkpoint.string());
    pfm::SparseHead head(4, 8);

    head->load_compatible(input_archive);
    auto output = head->forward(torch::rand({1, 4, 8, 8}, torch::kFloat32));

    bool migrated_projection_weight = false;
    bool migrated_projection_bias = false;
    bool has_new_residual_context = false;
    bool has_new_multiscale_context = false;
    bool has_new_attention_context = false;
    bool has_new_viewpoint_context = false;
    for (const auto& parameter : head->named_parameters()) {
        if (parameter.key() == "descriptors.6.weight") {
            migrated_projection_weight = torch::allclose(parameter.value(), legacy_projection_weight);
        }
        if (parameter.key() == "descriptors.6.bias") {
            migrated_projection_bias = torch::allclose(parameter.value(), legacy_projection_bias);
        }
        if (parameter.key() == "descriptors.2.conv1.weight") {
            has_new_residual_context = true;
        }
        if (parameter.key() == "descriptor_multiscale.weight") {
            has_new_multiscale_context = true;
        }
        if (parameter.key() == "descriptor_attention.weight") {
            has_new_attention_context = true;
        }
        if (parameter.key() == "descriptor_viewpoint_context.weight") {
            has_new_viewpoint_context = true;
            PFM_REQUIRE(torch::allclose(parameter.value(), torch::zeros_like(parameter.value())));
        }
    }

    PFM_REQUIRE(migrated_projection_weight);
    PFM_REQUIRE(migrated_projection_bias);
    PFM_REQUIRE(has_new_residual_context);
    PFM_REQUIRE(has_new_multiscale_context);
    PFM_REQUIRE(has_new_attention_context);
    PFM_REQUIRE(has_new_viewpoint_context);
    PFM_REQUIRE(torch::isfinite(output.descriptors).all().item<bool>());
    PFM_REQUIRE(output.descriptors.sizes() == torch::IntArrayRef({1, 8, 8, 8}));
    std::filesystem::remove(checkpoint);
}

static void sparse_head_heatmap_is_finite_for_rotated_inputs() {
    pfm::SparseHead head(4, 8);
    head->eval();
    auto feature = torch::rand({1, 4, 17, 17}, torch::kFloat32);

    const auto original = head->forward(feature).heatmap;
    const auto rotated = head->forward(torch::rot90(feature, 1, {2, 3})).heatmap;

    PFM_REQUIRE(torch::isfinite(original).all().item<bool>());
    PFM_REQUIRE(torch::isfinite(rotated).all().item<bool>());
    PFM_REQUIRE(original.min().item<float>() >= 0.0F);
    PFM_REQUIRE(original.max().item<float>() <= 1.0F);
    PFM_REQUIRE(rotated.min().item<float>() >= 0.0F);
    PFM_REQUIRE(rotated.max().item<float>() <= 1.0F);
}

void register_sparse_head_tests() {
    register_test("sparse head outputs expected maps", sparse_head_outputs_expected_maps);
    register_test("sparse head descriptors use local context", sparse_head_descriptors_use_local_context);
    register_test("sparse head descriptors use deeper local context",
                  sparse_head_descriptors_use_deeper_local_context);
    register_test("sparse head descriptors use residual blocks",
                  sparse_head_descriptors_use_residual_blocks);
    register_test("sparse head descriptors use four residual blocks",
                  sparse_head_descriptors_use_four_residual_blocks);
    register_test("sparse head uses shared context tower", sparse_head_uses_shared_context_tower);
    register_test("sparse head descriptors stay normalized", sparse_head_descriptors_stay_normalized);
    register_test("sparse head descriptor projection has requested channels",
                  sparse_head_descriptor_projection_has_requested_channels);
    register_test("sparse head descriptors use multiscale projection",
                  sparse_head_descriptors_use_multiscale_projection);
    register_test("sparse head descriptors use attention gate",
                  sparse_head_descriptors_use_attention_gate);
    register_test("sparse head descriptors use anisotropic viewpoint context",
                  sparse_head_descriptors_use_anisotropic_viewpoint_context);
    register_test("sparse head viewpoint context starts as residual noop",
                  sparse_head_viewpoint_context_starts_as_residual_noop);
    register_test("sparse head descriptors use orientation alignment projection",
                  sparse_head_descriptors_use_orientation_alignment_projection);
    register_test("sparse head descriptors use dilated context residual",
                  sparse_head_descriptors_use_dilated_context_residual);
    register_test("sparse head orientation alignment starts as residual noop",
                  sparse_head_orientation_alignment_starts_as_residual_noop);
    register_test("sparse head loads legacy descriptor tower checkpoint",
                  sparse_head_loads_legacy_descriptor_tower_checkpoint);
    register_test("sparse head heatmap is finite for rotated inputs",
                  sparse_head_heatmap_is_finite_for_rotated_inputs);
}
