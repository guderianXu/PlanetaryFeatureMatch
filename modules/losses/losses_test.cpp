#include "tests/test_harness.h"

#include <torch/torch.h>

#include "losses/losses.h"

static void repeatability_loss_zero_for_identical_heatmaps() {
    auto heatmap = torch::ones({1, 1, 4, 4}, torch::kFloat32) * 0.5F;
    auto mask = torch::ones({1, 1, 4, 4}, torch::kBool);

    auto loss = pfm::repeatability_loss(heatmap, heatmap, mask);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 0.0F, 1.0e-6F);
}

static void repeatability_loss_uses_fractional_mask_sum() {
    auto heatmap_a = torch::tensor({{{{1.0F, 3.0F}}}}, torch::kFloat32);
    auto heatmap_b = torch::zeros_like(heatmap_a);
    auto mask = torch::tensor({{{{0.25F, 0.25F}}}}, torch::kFloat32);

    auto loss = pfm::repeatability_loss(heatmap_a, heatmap_b, mask);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 5.0F, 1.0e-6F);
}

static void repeatability_loss_returns_zero_for_empty_mask() {
    auto heatmap_a = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    auto heatmap_b = torch::zeros_like(heatmap_a);
    auto mask = torch::zeros({1, 1, 2, 2}, torch::kFloat32);

    auto loss = pfm::repeatability_loss(heatmap_a, heatmap_b, mask);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 0.0F, 1.0e-6F);
}

static void repeatability_loss_rejects_negative_mask_values() {
    auto heatmap_a = torch::ones({1, 1, 1, 2}, torch::kFloat32);
    auto heatmap_b = torch::zeros_like(heatmap_a);
    auto mask = torch::tensor({{{{1.0F, -0.25F}}}}, torch::kFloat32);

    PFM_REQUIRE_THROWS_AS(pfm::repeatability_loss(heatmap_a, heatmap_b, mask), std::invalid_argument);
}

static void descriptor_loss_lower_for_matching_pairs() {
    auto a = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto b = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto labels = torch::tensor({{0, 1}}, torch::kLong);

    auto loss = pfm::descriptor_cross_entropy_loss(a, b, labels);

    PFM_REQUIRE(loss.item<float>() < 0.4F);
}

static void descriptor_loss_separates_many_matching_candidates() {
    auto descriptors = torch::eye(32, torch::kFloat32).unsqueeze(0);
    auto labels = torch::arange(32, torch::kLong).unsqueeze(0);

    auto loss = pfm::descriptor_cross_entropy_loss(descriptors, descriptors, labels);

    PFM_REQUIRE(loss.item<float>() < 0.1F);
}

static void descriptor_candidate_loss_uses_per_query_candidates() {
    auto queries = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto candidates = torch::tensor({{{{1.0F, 0.0F}, {0.0F, 1.0F}}, {{0.0F, 1.0F}, {1.0F, 0.0F}}}}, torch::kFloat32);
    auto labels = torch::zeros({1, 2}, torch::kLong);

    auto loss = pfm::descriptor_candidate_cross_entropy_loss(queries, candidates, labels);

    PFM_REQUIRE(loss.item<float>() < 0.4F);
}

static void descriptor_diversity_loss_penalizes_collapsed_descriptors() {
    auto collapsed = torch::ones({1, 4, 2}, torch::kFloat32);
    auto diverse = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}, {-1.0F, 0.0F}, {0.0F, -1.0F}}}, torch::kFloat32);

    auto collapsed_loss = pfm::descriptor_diversity_loss(collapsed);
    auto diverse_loss = pfm::descriptor_diversity_loss(diverse);

    PFM_REQUIRE(collapsed_loss.item<float>() > 0.9F);
    PFM_REQUIRE(diverse_loss.item<float>() < collapsed_loss.item<float>());
}

static void descriptor_loss_rejects_non_long_labels() {
    auto a = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto b = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto labels = torch::tensor({{0, 1}}, torch::kInt32);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void descriptor_loss_rejects_descriptor_dtype_mismatch() {
    auto a = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto b = torch::tensor({{{1.0, 0.0}, {0.0, 1.0}}}, torch::kFloat64);
    auto labels = torch::tensor({{0, 1}}, torch::kLong);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void descriptor_loss_rejects_descriptor_batch_mismatch() {
    auto a = torch::ones({2, 2, 2}, torch::kFloat32);
    auto b = torch::ones({1, 2, 2}, torch::kFloat32);
    auto labels = torch::zeros({2, 2}, torch::kLong);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void descriptor_loss_rejects_descriptor_dimension_mismatch() {
    auto a = torch::ones({1, 2, 3}, torch::kFloat32);
    auto b = torch::ones({1, 2, 2}, torch::kFloat32);
    auto labels = torch::zeros({1, 2}, torch::kLong);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void descriptor_loss_rejects_wrong_target_shape() {
    auto a = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto b = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto labels = torch::tensor({0, 1}, torch::kLong);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void descriptor_loss_rejects_target_shape_with_wrong_batch() {
    auto a = torch::ones({2, 2, 2}, torch::kFloat32);
    auto b = torch::ones({2, 2, 2}, torch::kFloat32);
    auto labels = torch::zeros({1, 2}, torch::kLong);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void descriptor_loss_rejects_target_shape_with_wrong_count() {
    auto a = torch::ones({1, 2, 2}, torch::kFloat32);
    auto b = torch::ones({1, 2, 2}, torch::kFloat32);
    auto labels = torch::zeros({1, 1}, torch::kLong);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void descriptor_loss_rejects_out_of_range_label() {
    auto a = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto b = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto labels = torch::tensor({{0, 2}}, torch::kLong);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void descriptor_loss_rejects_negative_labels() {
    auto a = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto b = torch::tensor({{{1.0F, 0.0F}, {0.0F, 1.0F}}}, torch::kFloat32);
    auto labels = torch::tensor({{0, -1}}, torch::kLong);

    PFM_REQUIRE_THROWS_AS(pfm::descriptor_cross_entropy_loss(a, b, labels), std::invalid_argument);
}

static void masked_l1_loss_zero_when_mask_empty() {
    auto pred = torch::ones({1, 2, 4, 4}, torch::kFloat32);
    auto target = torch::zeros_like(pred);
    auto mask = torch::zeros({1, 1, 4, 4}, torch::kBool);

    auto loss = pfm::masked_l1_loss(pred, target, mask);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 0.0F, 1.0e-6F);
}

static void masked_l1_loss_averages_channels_for_selected_spatial_position() {
    auto pred = torch::tensor({{{{1.0F, 3.0F}}, {{5.0F, 7.0F}}}}, torch::kFloat32);
    auto target = torch::zeros_like(pred);
    auto mask = torch::tensor({{{{1.0F, 0.0F}}}}, torch::kFloat32);

    auto loss = pfm::masked_l1_loss(pred, target, mask);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 3.0F, 1.0e-6F);
}

static void masked_l1_loss_supports_exact_shape_mask() {
    auto pred = torch::tensor({{{{1.0F, 3.0F}}, {{5.0F, 7.0F}}}}, torch::kFloat32);
    auto target = torch::zeros_like(pred);
    auto mask = torch::tensor({{{{1.0F, 0.0F}}, {{0.5F, 1.5F}}}}, torch::kFloat32);

    auto loss = pfm::masked_l1_loss(pred, target, mask);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 14.0F / 3.0F, 1.0e-6F);
}

static void masked_l1_loss_supports_scalar_mask() {
    auto pred = torch::tensor({{{{1.0F, 3.0F}}, {{5.0F, 7.0F}}}}, torch::kFloat32);
    auto target = torch::ones_like(pred);
    auto mask = torch::tensor(2.0F, torch::kFloat32);

    auto loss = pfm::masked_l1_loss(pred, target, mask);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 3.0F, 1.0e-6F);
}

static void masked_l1_loss_rejects_ambiguous_bhw_mask() {
    auto pred = torch::ones({1, 2, 2, 2}, torch::kFloat32);
    auto target = torch::zeros_like(pred);
    auto mask = torch::ones({1, 2, 2}, torch::kFloat32);

    PFM_REQUIRE_THROWS_AS(pfm::masked_l1_loss(pred, target, mask), std::invalid_argument);
}

static void confidence_bce_loss_returns_finite_positive_scalar_for_matching_shapes() {
    auto confidence = torch::tensor({{0.25F, 0.75F}}, torch::kFloat32);
    auto target = torch::tensor({{0.0F, 1.0F}}, torch::kFloat32);

    auto loss = pfm::confidence_bce_loss(confidence, target);

    PFM_REQUIRE(loss.dim() == 0);
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
    PFM_REQUIRE(loss.item<float>() > 0.0F);
}

static void confidence_bce_loss_supports_scalar_target() {
    auto confidence = torch::tensor({0.25F, 0.75F}, torch::kFloat32);
    auto target = torch::tensor(1.0F, torch::kFloat32);

    auto loss = pfm::confidence_bce_loss(confidence, target);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 0.8369882F, 1.0e-6F);
}

static void confidence_bce_loss_rejects_incompatible_target_shape() {
    auto confidence = torch::ones({1, 2, 2}, torch::kFloat32) * 0.5F;
    auto target = torch::ones({1, 2}, torch::kFloat32);

    PFM_REQUIRE_THROWS_AS(pfm::confidence_bce_loss(confidence, target), std::invalid_argument);
}

void register_loss_tests() {
    register_test("repeatability loss zero for identical heatmaps", repeatability_loss_zero_for_identical_heatmaps);
    register_test("repeatability loss uses fractional mask sum", repeatability_loss_uses_fractional_mask_sum);
    register_test("repeatability loss returns zero for empty mask", repeatability_loss_returns_zero_for_empty_mask);
    register_test("repeatability loss rejects negative mask values", repeatability_loss_rejects_negative_mask_values);
    register_test("descriptor loss lower for matching pairs", descriptor_loss_lower_for_matching_pairs);
    register_test("descriptor loss separates many matching candidates", descriptor_loss_separates_many_matching_candidates);
    register_test("descriptor candidate loss uses per query candidates", descriptor_candidate_loss_uses_per_query_candidates);
    register_test("descriptor diversity loss penalizes collapsed descriptors",
                  descriptor_diversity_loss_penalizes_collapsed_descriptors);
    register_test("descriptor loss rejects non long labels", descriptor_loss_rejects_non_long_labels);
    register_test(
        "descriptor loss rejects descriptor dtype mismatch",
        descriptor_loss_rejects_descriptor_dtype_mismatch);
    register_test(
        "descriptor loss rejects descriptor batch mismatch",
        descriptor_loss_rejects_descriptor_batch_mismatch);
    register_test(
        "descriptor loss rejects descriptor dimension mismatch",
        descriptor_loss_rejects_descriptor_dimension_mismatch);
    register_test("descriptor loss rejects wrong target shape", descriptor_loss_rejects_wrong_target_shape);
    register_test(
        "descriptor loss rejects target shape with wrong batch",
        descriptor_loss_rejects_target_shape_with_wrong_batch);
    register_test(
        "descriptor loss rejects target shape with wrong count",
        descriptor_loss_rejects_target_shape_with_wrong_count);
    register_test("descriptor loss rejects out of range label", descriptor_loss_rejects_out_of_range_label);
    register_test("descriptor loss rejects negative labels", descriptor_loss_rejects_negative_labels);
    register_test("masked l1 loss zero when mask empty", masked_l1_loss_zero_when_mask_empty);
    register_test(
        "masked l1 loss averages channels for selected spatial position",
        masked_l1_loss_averages_channels_for_selected_spatial_position);
    register_test("masked l1 loss supports exact shape mask", masked_l1_loss_supports_exact_shape_mask);
    register_test("masked l1 loss supports scalar mask", masked_l1_loss_supports_scalar_mask);
    register_test("masked l1 loss rejects ambiguous bhw mask", masked_l1_loss_rejects_ambiguous_bhw_mask);
    register_test(
        "confidence bce loss returns finite positive scalar for matching shapes",
        confidence_bce_loss_returns_finite_positive_scalar_for_matching_shapes);
    register_test("confidence bce loss supports scalar target", confidence_bce_loss_supports_scalar_target);
    register_test(
        "confidence bce loss rejects incompatible target shape",
        confidence_bce_loss_rejects_incompatible_target_shape);
}
