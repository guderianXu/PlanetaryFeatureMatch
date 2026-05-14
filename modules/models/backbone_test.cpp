#include "tests/test_harness.h"

#include <torch/torch.h>

#include "models/backbone.h"

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

void register_backbone_tests() {
    register_test("backbone returns four scales", backbone_returns_four_scales);
}
