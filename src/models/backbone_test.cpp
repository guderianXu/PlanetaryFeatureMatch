#include "tests/test_harness.h"

#include <limits>

#include <torch/torch.h>

#include "models/backbone.h"

namespace {

int64_t parameter_count(const pfm::Backbone& model) {
    int64_t count = 0;
    for (const auto& parameter : model->parameters()) {
        count += parameter.numel();
    }
    return count;
}

}  // namespace

static void backbone_returns_four_scales() {
    pfm::Backbone model(1, 16);
    auto x = torch::randn({2, 1, 64, 64}, torch::kFloat32);

    auto outputs = model->forward(x);

    PFM_REQUIRE(outputs.size() == 4);
    PFM_REQUIRE(outputs[0].sizes() == torch::IntArrayRef({2, 16, 32, 32}));
    PFM_REQUIRE(outputs[1].sizes() == torch::IntArrayRef({2, 32, 16, 16}));
    PFM_REQUIRE(outputs[2].sizes() == torch::IntArrayRef({2, 64, 8, 8}));
    PFM_REQUIRE(outputs[3].sizes() == torch::IntArrayRef({2, 128, 4, 4}));
}

static void backbone_uses_refinement_convolutions_in_each_stage() {
    pfm::Backbone model(1, 32);

    PFM_REQUIRE(parameter_count(model) > 390000);
}

static void backbone_sanitizes_nonfinite_batchnorm_buffers_for_eval() {
    pfm::Backbone model(1, 8);
    for (auto& item : model->named_buffers(/*recurse=*/true)) {
        if (item.key().find("running_mean") != std::string::npos ||
            item.key().find("running_var") != std::string::npos) {
            item.value().fill_(std::numeric_limits<float>::quiet_NaN());
        }
    }

    model->sanitize_nonfinite_state();
    model->eval();
    auto outputs = model->forward(torch::randn({1, 1, 32, 32}, torch::kFloat32));

    for (const auto& output : outputs) {
        PFM_REQUIRE(torch::isfinite(output).all().item<bool>());
    }
    for (const auto& item : model->named_buffers(/*recurse=*/true)) {
        if (item.key().find("running_mean") != std::string::npos ||
            item.key().find("running_var") != std::string::npos) {
            PFM_REQUIRE(torch::isfinite(item.value()).all().item<bool>());
        }
    }
}

void register_backbone_tests() {
    register_test("backbone returns four scales", backbone_returns_four_scales);
    register_test("backbone uses refinement convolutions in each stage",
                  backbone_uses_refinement_convolutions_in_each_stage);
    register_test("backbone sanitizes nonfinite batchnorm buffers for eval",
                  backbone_sanitizes_nonfinite_batchnorm_buffers_for_eval);
}
