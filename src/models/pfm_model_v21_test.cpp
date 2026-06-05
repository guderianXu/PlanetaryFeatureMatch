#include <string>
#include <vector>

#include <torch/torch.h>

#include "models/pfm_model_v21.h"
#include "tests/test_harness.h"

namespace
{

bool hasStateKey(const pfm::v21::PfmV21FeatureMatcher& model, const std::string& key)
{
    for (const auto& parameter : model->named_parameters(true))
    {
        if (parameter.key() == key)
        {
            return true;
        }
    }
    for (const auto& buffer : model->named_buffers(true))
    {
        if (buffer.key() == key)
        {
            return true;
        }
    }
    return false;
}

pfm::v21::PfmV21Config smallConfig()
{
    pfm::v21::PfmV21Config config;
    config.input_channels = 1;
    config.base_channels = 4;
    config.descriptor_dim = 8;
    config.graph_hidden_dim = 16;
    config.graph_attention_layers = 1;
    config.graph_keypoint_meta_dim = 16;
    return config;
}

static void pfm_v21_model_exposes_python_state_keys()
{
    auto model = pfm::v21::PfmV21FeatureMatcher(smallConfig());

    PFM_REQUIRE(hasStateKey(model, "backbone.stage1_refine.0.conv2.weight"));
    PFM_REQUIRE(hasStateKey(model, "dual_fpn.keypoint_from_stage3.weight"));
    PFM_REQUIRE(hasStateKey(model, "sparse_head.keypoint_context.conv1.weight"));
    PFM_REQUIRE(hasStateKey(model, "sparse_head.keypoint_offsets.weight"));
    PFM_REQUIRE(hasStateKey(model, "texture_adapter.residual.weight"));
    PFM_REQUIRE(hasStateKey(model, "descriptor_fusion.input_projection.weight"));
    PFM_REQUIRE(hasStateKey(model, "quality_head.predictor.0.weight"));
    PFM_REQUIRE(hasStateKey(model, "semi_dense_branch.projection.0.weight"));
    PFM_REQUIRE(hasStateKey(model, "graph_matcher.geometry_bias.0.weight"));
    PFM_REQUIRE(hasStateKey(model, "graph_matcher.accept_head.0.weight"));
    PFM_REQUIRE(hasStateKey(model, "graph_matcher.raw_score_temperature"));
    PFM_REQUIRE(hasStateKey(model, "graph_matcher.attention_layers.0.self_query.weight"));
}

static void pfm_v21_forward_single_returns_python_feature_shapes()
{
    auto model = pfm::v21::PfmV21FeatureMatcher(smallConfig());
    model->eval();
    torch::NoGradGuard no_grad;

    const auto output = model->forwardSingle(torch::rand({1, 1, 32, 32}, torch::kFloat32));

    PFM_REQUIRE(output.heatmap.sizes() == torch::IntArrayRef({1, 1, 8, 8}));
    PFM_REQUIRE(output.descriptors.sizes() == torch::IntArrayRef({1, 8, 8, 8}));
    PFM_REQUIRE(output.scale.sizes() == torch::IntArrayRef({1, 1, 8, 8}));
    PFM_REQUIRE(output.orientation.sizes() == torch::IntArrayRef({1, 2, 8, 8}));
    PFM_REQUIRE(output.affine.sizes() == torch::IntArrayRef({1, 4, 8, 8}));
    PFM_REQUIRE(output.dense_confidence.sizes() == torch::IntArrayRef({1, 1, 8, 8}));
    PFM_REQUIRE(output.keypoint_offsets.sizes() == torch::IntArrayRef({1, 2, 8, 8}));
    PFM_REQUIRE(output.quality.sizes() == torch::IntArrayRef({1, 1, 8, 8}));
    PFM_REQUIRE(output.local_contrast.sizes() == torch::IntArrayRef({1, 1, 8, 8}));
    PFM_REQUIRE(torch::isfinite(output.descriptors).all().item<bool>());
}

static void pfm_v21_graph_matcher_accepts_full_metadata()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16);
    matcher->eval();
    torch::NoGradGuard no_grad;

    auto descriptors_a = torch::randn({3, 8}, torch::kFloat32);
    auto descriptors_b = torch::randn({4, 8}, torch::kFloat32);
    auto metadata_a = torch::randn({3, 16}, torch::kFloat32);
    auto metadata_b = torch::randn({4, 16}, torch::kFloat32);

    const auto output = matcher->forward(descriptors_a, metadata_a, descriptors_b, metadata_b);

    PFM_REQUIRE(output.logits.sizes() == torch::IntArrayRef({4, 5}));
    PFM_REQUIRE(output.accept_logits.sizes() == torch::IntArrayRef({3, 4}));
    PFM_REQUIRE(output.matches.dim() == 2);
    PFM_REQUIRE(output.matches.size(1) == 2);
    PFM_REQUIRE(output.scores.dim() == 1);
    PFM_REQUIRE(torch::isfinite(output.logits).all().item<bool>());
    PFM_REQUIRE(torch::isfinite(output.accept_logits).all().item<bool>());
}

static void pfm_v21_graph_matcher_scalar_parameters_match_python_shapes()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16);
    bool found_raw_temperature = false;
    bool found_delta_scale = false;
    bool found_accept_scale = false;

    for (const auto& parameter : matcher->named_parameters(true))
    {
        if (parameter.key() == "raw_score_temperature")
        {
            found_raw_temperature = true;
            PFM_REQUIRE(parameter.value().dim() == 0);
        }
        if (parameter.key() == "graph_delta_scale")
        {
            found_delta_scale = true;
            PFM_REQUIRE(parameter.value().dim() == 0);
        }
        if (parameter.key() == "accept_logit_scale")
        {
            found_accept_scale = true;
            PFM_REQUIRE(parameter.value().dim() == 0);
        }
    }

    PFM_REQUIRE(found_raw_temperature);
    PFM_REQUIRE(found_delta_scale);
    PFM_REQUIRE(found_accept_scale);
}

static void pfm_v21_graph_matcher_can_disable_candidate_mask_for_training_loss()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 1);
    matcher->eval();
    torch::NoGradGuard no_grad;

    auto descriptors_a = torch::zeros({3, 8}, torch::kFloat32);
    auto descriptors_b = torch::zeros({3, 8}, torch::kFloat32);
    descriptors_a.index_put_({0, 0}, 1.0F);
    descriptors_a.index_put_({1, 0}, -1.0F);
    descriptors_a.index_put_({2, 0}, -1.0F);
    descriptors_b.index_put_({0, 0}, -1.0F);
    descriptors_b.index_put_({1, 0}, 1.0F);
    descriptors_b.index_put_({2, 0}, 1.0F);
    auto metadata = torch::zeros({3, 16}, torch::kFloat32);

    const auto masked = matcher->forward(descriptors_a, metadata, descriptors_b, metadata);
    const auto unmasked = matcher->forward(descriptors_a, metadata, descriptors_b, metadata, false);

    PFM_REQUIRE(masked.logits.index({0, 0}).item<float>() < -9000.0F);
    PFM_REQUIRE(unmasked.logits.index({0, 0}).item<float>() > -9000.0F);
    PFM_REQUIRE(torch::isfinite(unmasked.logits).all().item<bool>());
}

static void pfm_v21_graph_matcher_width_pruning_restores_full_logits()
{
    using torch::indexing::Slice;

    auto matcher = pfm::v21::PfmV21GraphMatcher(2, 8, 1, 4);
    matcher->eval();
    torch::NoGradGuard no_grad;

    for (auto& parameter : matcher->named_parameters(true))
    {
        if (parameter.key() == "graph_delta_scale")
        {
            parameter.value().fill_(0.0F);
        }
        if (parameter.key() == "accept_logit_scale")
        {
            parameter.value().fill_(0.0F);
        }
        if (parameter.key() == "raw_score_temperature")
        {
            parameter.value().fill_(0.1F);
        }
    }

    auto descriptors_a = torch::zeros({3, 2}, torch::kFloat32);
    auto descriptors_b = torch::zeros({2, 2}, torch::kFloat32);
    descriptors_a.index_put_({0, 0}, 1.0F);
    descriptors_a.index_put_({1, 1}, 1.0F);
    descriptors_a.index_put_({2, 0}, -1.0F);
    descriptors_b.index_put_({0, 0}, 1.0F);
    descriptors_b.index_put_({1, 1}, 1.0F);
    auto keypoints_a = torch::zeros({3, 2}, torch::kFloat32);
    auto keypoints_b = torch::zeros({2, 2}, torch::kFloat32);
    keypoints_a.index_put_({1, 0}, 1.0F);
    keypoints_a.index_put_({2, 0}, 2.0F);
    keypoints_b.index_put_({1, 0}, 1.0F);

    const auto output = matcher->forward(descriptors_a, keypoints_a, descriptors_b, keypoints_b, true, 0.5);

    PFM_REQUIRE(output.logits.sizes() == torch::IntArrayRef({4, 3}));
    PFM_REQUIRE(output.accept_logits.sizes() == torch::IntArrayRef({3, 2}));
    if (output.matches.numel() > 0)
    {
        PFM_REQUIRE((output.matches.index({Slice(), 0}) != 2).all().item<bool>());
    }
    PFM_REQUIRE((output.logits.index({2, Slice(0, 2)}) < -9000.0F).all().item<bool>());
    PFM_REQUIRE((output.accept_logits.index({2, Slice()}) < -9000.0F).all().item<bool>());
}

static void pfm_v21_graph_matcher_width_pruning_uses_layer_acceptance()
{
    using torch::indexing::Slice;

    auto matcher = pfm::v21::PfmV21GraphMatcher(5, 16, 3, 16, 0);
    matcher->eval();
    torch::NoGradGuard no_grad;

    for (auto& parameter : matcher->named_parameters(true))
    {
        if (parameter.key().rfind("accept_head.", 0) == 0)
        {
            parameter.value().zero_();
        }
    }
    for (auto& parameter : matcher->named_parameters(true))
    {
        if (parameter.key() == "accept_head.0.weight")
        {
            parameter.value().index_put_({0, 4}, 10.0F);
        }
        if (parameter.key() == "accept_head.0.bias")
        {
            parameter.value().index_put_({0}, -7.5F);
        }
        if (parameter.key() == "accept_head.2.weight")
        {
            parameter.value().index_put_({0, 0}, 4.0F);
        }
        if (parameter.key() == "accept_head.2.bias")
        {
            parameter.value().index_put_({0}, -4.0F);
        }
    }

    auto descriptors = torch::eye(5, torch::kFloat32);
    auto metadata = torch::zeros({5, 16}, torch::kFloat32);
    metadata.index_put_({Slice(), 12}, 1.0F);
    metadata.index_put_({4, 12}, 0.0F);

    const auto output = matcher->forward(descriptors, metadata, descriptors, metadata, true, 0.8);

    PFM_REQUIRE((output.accept_logits.index({Slice(0, 4), Slice(0, 4)}) > -100.0F).any().item<bool>());
    PFM_REQUIRE((output.accept_logits.index({4, Slice()}) < -9000.0F).all().item<bool>());
    PFM_REQUIRE((output.accept_logits.index({Slice(), 4}) < -9000.0F).all().item<bool>());
}

static void pfm_v21_graph_matcher_can_stop_attention_layers_when_confident()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(2, 8, 3, 4);
    matcher->eval();
    torch::NoGradGuard no_grad;

    auto descriptors = torch::eye(2, torch::kFloat32);
    auto keypoints = torch::zeros({2, 2}, torch::kFloat32);
    keypoints.index_put_({1, 0}, 1.0F);

    const auto output = matcher->forward(descriptors, keypoints, descriptors, keypoints, true, -1.0, 0.0);

    PFM_REQUIRE(output.logits.sizes() == torch::IntArrayRef({3, 3}));
    PFM_REQUIRE(matcher->lastExecutedAttentionLayers() == 1);
}

static void pfm_v21_graph_matcher_early_stop_tolerates_single_uncertain_keypoint()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(5, 8, 3, 4);
    matcher->eval();
    torch::NoGradGuard no_grad;

    for (auto& parameter : matcher->named_parameters(true))
    {
        if (parameter.key() == "graph_delta_scale")
        {
            parameter.value().fill_(0.0F);
        }
        if (parameter.key() == "accept_logit_scale")
        {
            parameter.value().fill_(0.0F);
        }
        if (parameter.key() == "raw_score_temperature")
        {
            parameter.value().fill_(0.03F);
        }
    }

    auto descriptors_a = torch::eye(5, torch::kFloat32);
    auto descriptors_b = torch::eye(5, torch::kFloat32);
    descriptors_a.index_put_({4}, torch::zeros({5}, torch::kFloat32));
    descriptors_b.index_put_({4}, torch::zeros({5}, torch::kFloat32));
    auto keypoints = torch::zeros({5, 2}, torch::kFloat32);
    keypoints.index_put_({1, 0}, 1.0F);
    keypoints.index_put_({2, 0}, 2.0F);
    keypoints.index_put_({3, 0}, 3.0F);
    keypoints.index_put_({4, 0}, 4.0F);

    const auto output = matcher->forward(descriptors_a, keypoints, descriptors_b, keypoints, true, -1.0, 0.8);

    PFM_REQUIRE(output.logits.sizes() == torch::IntArrayRef({6, 6}));
    PFM_REQUIRE(matcher->lastExecutedAttentionLayers() == 1);
}

static void pfm_v21_optimizer_step_updates_parameters()
{
    torch::manual_seed(7);
    auto model = pfm::v21::PfmV21FeatureMatcher(smallConfig());
    model->train();

    std::vector<torch::Tensor> before;
    for (const auto& parameter : model->parameters())
    {
        before.push_back(parameter.detach().clone());
    }

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1.0e-3));
    optimizer.zero_grad();

    const auto output = model->forwardSingle(torch::rand({1, 1, 32, 32}, torch::kFloat32));
    const auto loss = output.heatmap.mean() + output.descriptors.square().mean() + output.quality.mean() +
                      output.dense_confidence.mean();
    PFM_REQUIRE(torch::isfinite(loss).item<bool>());

    loss.backward();
    optimizer.step();

    bool changed = false;
    const auto after = model->parameters();
    for (std::size_t index = 0; index < before.size(); ++index)
    {
        if (!torch::allclose(before[index], after[index].detach()))
        {
            changed = true;
            break;
        }
    }
    PFM_REQUIRE(changed);
}

} // namespace

void register_pfm_model_v21_tests()
{
    register_test("pfm v21 model exposes python state keys", pfm_v21_model_exposes_python_state_keys);
    register_test("pfm v21 forward single returns python feature shapes",
                  pfm_v21_forward_single_returns_python_feature_shapes);
    register_test("pfm v21 graph matcher accepts full metadata", pfm_v21_graph_matcher_accepts_full_metadata);
    register_test("pfm v21 graph matcher scalar parameters match python shapes",
                  pfm_v21_graph_matcher_scalar_parameters_match_python_shapes);
    register_test("pfm v21 graph matcher can disable candidate mask for training loss",
                  pfm_v21_graph_matcher_can_disable_candidate_mask_for_training_loss);
    register_test("pfm v21 graph matcher width pruning restores full logits",
                  pfm_v21_graph_matcher_width_pruning_restores_full_logits);
    register_test("pfm v21 graph matcher width pruning uses layer acceptance",
                  pfm_v21_graph_matcher_width_pruning_uses_layer_acceptance);
    register_test("pfm v21 graph matcher can stop attention layers when confident",
                  pfm_v21_graph_matcher_can_stop_attention_layers_when_confident);
    register_test("pfm v21 graph matcher early stop tolerates single uncertain keypoint",
                  pfm_v21_graph_matcher_early_stop_tolerates_single_uncertain_keypoint);
    register_test("pfm v21 optimizer step updates parameters", pfm_v21_optimizer_step_updates_parameters);
}
