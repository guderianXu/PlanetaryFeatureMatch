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
    PFM_REQUIRE(output.matches.dim() == 2);
    PFM_REQUIRE(output.matches.size(1) == 2);
    PFM_REQUIRE(output.matches.size(0) <= 3);
    PFM_REQUIRE(output.scores.size(0) == output.matches.size(0));
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
    auto descriptors_a = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 1.0F}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_a = torch::tensor({{0.0F, 0.0F}, {5.0F, 5.0F}, {10.0F, 10.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{0.0F, 0.0F}, {5.0F, 5.0F}, {10.0F, 10.0F}}, torch::kFloat32);
    auto shifted_keypoints_b = torch::tensor({{0.0F, 0.0F}, {0.0F, 10.0F}, {10.0F, 10.0F}}, torch::kFloat32);

    const auto original = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b).logits;
    const auto shifted = matcher->forward(descriptors_a, keypoints_a, descriptors_b, shifted_keypoints_b).logits;

    PFM_REQUIRE(!torch::allclose(original, shifted));
}

static void graph_matcher_keypoint_logits_are_half_turn_invariant() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    matcher->eval();
    auto descriptors_a = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 1.0F}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_a = torch::tensor({{0.0F, 0.0F}, {5.0F, 5.0F}, {10.0F, 10.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{2.0F, 0.0F}, {5.0F, 5.0F}, {8.0F, 10.0F}}, torch::kFloat32);
    auto rotated_a = torch::tensor({{10.0F, 10.0F}, {5.0F, 5.0F}, {0.0F, 0.0F}}, torch::kFloat32);
    auto rotated_b = torch::tensor({{8.0F, 10.0F}, {5.0F, 5.0F}, {2.0F, 0.0F}}, torch::kFloat32);

    const auto original = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b).logits;
    const auto rotated = matcher->forward(descriptors_a, rotated_a, descriptors_b, rotated_b).logits;

    PFM_REQUIRE(torch::allclose(original, rotated, 1.0e-4, 1.0e-4));
}

static void graph_matcher_keypoint_logits_are_scale_invariant() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    matcher->eval();
    auto descriptors_a = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_a = torch::tensor({{8.0F, 8.0F}, {24.0F, 24.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{8.0F, 8.0F}, {24.0F, 24.0F}}, torch::kFloat32);

    const auto original = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b).logits;
    const auto scaled = matcher->forward(descriptors_a, keypoints_a * 10.0F, descriptors_b, keypoints_b * 10.0F).logits;

    PFM_REQUIRE(torch::allclose(original, scaled, 1.0e-4, 1.0e-4));
}

static void graph_matcher_filters_dustbin_sparse_matches() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    matcher->eval();
    auto descriptors_a = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_a = torch::tensor({{8.0F, 8.0F}, {24.0F, 24.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{8.0F, 8.0F}, {24.0F, 24.0F}}, torch::kFloat32);
    for (auto& parameter : matcher->parameters()) {
        parameter.detach().zero_();
    }
    matcher->named_parameters()["dustbin_bias"].detach().fill_(10.0F);

    const auto output = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b);

    PFM_REQUIRE(output.matches.sizes() == std::vector<int64_t>({0, 2}));
    PFM_REQUIRE(output.scores.sizes() == std::vector<int64_t>({0}));
}

static void graph_matcher_keeps_only_mutual_sparse_matches() {
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    matcher->eval();
    auto descriptors_a = torch::tensor({{1.0F, 0.0F}, {1.0F, 0.0F}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{1.0F, 0.0F}}, torch::kFloat32);
    auto keypoints_a = torch::tensor({{8.0F, 8.0F}, {8.0F, 8.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{8.0F, 8.0F}}, torch::kFloat32);
    for (auto& parameter : matcher->parameters()) {
        parameter.detach().fill_(0.1F);
    }
    matcher->named_parameters()["dustbin_bias"].detach().fill_(-10.0F);

    const auto output = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b);

    PFM_REQUIRE(output.matches.size(0) == 1);
    PFM_REQUIRE(output.matches.index({0, 1}).item<int64_t>() == 0);
}

void register_planetary_graph_matcher_tests() {
    register_test("planetary_graph_matcher_outputs_match_logits_with_dustbin",
                  planetary_graph_matcher_outputs_match_logits_with_dustbin);
    register_test("planetary_graph_matcher_rejects_descriptor_dimension_mismatch",
                  planetary_graph_matcher_rejects_descriptor_dimension_mismatch);
    register_test("graph matcher keypoints affect logits", graph_matcher_keypoints_affect_logits);
    register_test("graph matcher keypoint logits are half turn invariant",
                  graph_matcher_keypoint_logits_are_half_turn_invariant);
    register_test("graph matcher keypoint logits are scale invariant", graph_matcher_keypoint_logits_are_scale_invariant);
    register_test("graph matcher filters dustbin sparse matches", graph_matcher_filters_dustbin_sparse_matches);
    register_test("graph matcher keeps only mutual sparse matches", graph_matcher_keeps_only_mutual_sparse_matches);
    register_test("planetary_graph_matcher_uses_attention_layers",
                  planetary_graph_matcher_uses_attention_layers);
    register_test("planetary_graph_matcher_attention_layer_uses_norm_and_ffn",
                  planetary_graph_matcher_attention_layer_uses_norm_and_ffn);
}
