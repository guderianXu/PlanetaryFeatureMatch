#include <torch/torch.h>

#include "models/planetary_graph_matcher.h"
#include "tests/test_harness.h"

static void planetary_graph_matcher_outputs_match_logits_with_dustbin() {
    pfm::PlanetaryGraphMatcher matcher(4, 8);
    auto descriptors_a = torch::randn({3, 4});
    auto descriptors_b = torch::randn({5, 4});
    auto keypoints_a = torch::randn({3, 2});
    auto keypoints_b = torch::randn({5, 2});

    const auto output = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b);

    PFM_REQUIRE(output.logits.sizes() == std::vector<int64_t>({4, 6}));
    PFM_REQUIRE(output.matches.sizes() == std::vector<int64_t>({3, 2}));
    PFM_REQUIRE(output.scores.sizes() == std::vector<int64_t>({3}));
}

static void planetary_graph_matcher_rejects_descriptor_dimension_mismatch() {
    pfm::PlanetaryGraphMatcher matcher(4, 8);
    auto descriptors_a = torch::randn({3, 4});
    auto descriptors_b = torch::randn({5, 6});
    auto keypoints_a = torch::randn({3, 2});
    auto keypoints_b = torch::randn({5, 2});

    PFM_REQUIRE_INVALID_ARG(matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b));
}

static void planetary_graph_matcher_uses_attention_layers() {
    pfm::PlanetaryGraphMatcher matcher(4, 8, 2);
    auto descriptors_a = torch::randn({3, 4});
    auto descriptors_b = torch::randn({5, 4});
    auto keypoints_a = torch::randn({3, 2});
    auto keypoints_b = torch::randn({5, 2});

    const auto output = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b);

    PFM_REQUIRE(output.logits.sizes() == std::vector<int64_t>({4, 6}));
    PFM_REQUIRE(matcher->attentionLayerCount() == 2);
}

static void planetary_graph_matcher_attention_layer_uses_norm_and_ffn() {
    pfm::PlanetaryGraphMatcher matcher(4, 8, 1);
    bool found_norm = false;
    bool found_feed_forward = false;
    for (const auto& parameter : matcher->named_parameters()) {
        if (parameter.key().find("self_norm") != std::string::npos) {
            found_norm = true;
        }
        if (parameter.key().find("feed_forward") != std::string::npos) {
            found_feed_forward = true;
        }
    }

    PFM_REQUIRE(found_norm);
    PFM_REQUIRE(found_feed_forward);
}

static void graph_matcher_keypoints_affect_logits() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    matcher->eval();
    auto descriptors_a = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_a = torch::tensor({{8.0F, 8.0F}, {24.0F, 24.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{8.0F, 8.0F}, {24.0F, 24.0F}}, torch::kFloat32);
    auto shifted_keypoints_b = torch::tensor({{24.0F, 24.0F}, {8.0F, 8.0F}}, torch::kFloat32);

    const auto original = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b).logits;
    const auto shifted = matcher->forward(descriptors_a, keypoints_a, descriptors_b, shifted_keypoints_b).logits;

    PFM_REQUIRE(!torch::allclose(original, shifted));
}

void register_planetary_graph_matcher_tests() {
    register_test("planetary_graph_matcher_outputs_match_logits_with_dustbin",
                  planetary_graph_matcher_outputs_match_logits_with_dustbin);
    register_test("planetary_graph_matcher_rejects_descriptor_dimension_mismatch",
                  planetary_graph_matcher_rejects_descriptor_dimension_mismatch);
    register_test("graph matcher keypoints affect logits", graph_matcher_keypoints_affect_logits);
    register_test("planetary_graph_matcher_uses_attention_layers",
                  planetary_graph_matcher_uses_attention_layers);
    register_test("planetary_graph_matcher_attention_layer_uses_norm_and_ffn",
                  planetary_graph_matcher_attention_layer_uses_norm_and_ffn);
}
