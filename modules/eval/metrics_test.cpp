#include "tests/test_harness.h"

#include <torch/torch.h>

#include "eval/metrics.h"

static void matching_precision_counts_matches_within_threshold() {
    auto predicted_a = torch::tensor({{0.0F, 0.0F}, {10.0F, 10.0F}}, torch::kFloat32);
    auto predicted_b = torch::tensor({{1.0F, 0.0F}, {30.0F, 30.0F}}, torch::kFloat32);
    auto expected_b = torch::tensor({{1.0F, 0.0F}, {11.0F, 10.0F}}, torch::kFloat32);

    const auto precision = pfm::matching_precision(predicted_a, predicted_b, expected_b, 1.5F);

    PFM_REQUIRE_CLOSE(precision, 0.5F, 1.0e-6F);
}

static void matching_precision_returns_zero_for_empty_predictions() {
    auto points_a = torch::empty({0, 2}, torch::kFloat32);
    auto predicted_b = torch::empty({0, 2}, torch::kFloat32);
    auto expected_b = torch::empty({0, 2}, torch::kFloat32);

    const auto precision = pfm::matching_precision(points_a, predicted_b, expected_b, 1.0F);

    PFM_REQUIRE_CLOSE(precision, 0.0F, 1.0e-6F);
}

static void matching_precision_accepts_undefined_points_a() {
    torch::Tensor points_a;
    auto predicted_b = torch::tensor({{1.0F, 1.0F}, {3.0F, 3.0F}}, torch::kFloat32);
    auto expected_b = torch::tensor({{1.0F, 1.0F}, {4.0F, 3.0F}}, torch::kFloat32);

    const auto precision = pfm::matching_precision(points_a, predicted_b, expected_b, 1.0F);

    PFM_REQUIRE_CLOSE(precision, 1.0F, 1.0e-6F);
}

static void matching_precision_rejects_shape_mismatch() {
    auto points_a = torch::empty({2, 2}, torch::kFloat32);
    auto predicted_b = torch::empty({2, 2}, torch::kFloat32);
    auto expected_b = torch::empty({3, 2}, torch::kFloat32);

    PFM_REQUIRE_INVALID_ARG(pfm::matching_precision(points_a, predicted_b, expected_b, 1.0F));
}

static void matching_precision_rejects_integer_predicted_b_dtype() {
    auto points_a = torch::empty({2, 2}, torch::kFloat32);
    auto predicted_b = torch::empty({2, 2}, torch::kInt32);
    auto expected_b = torch::empty({2, 2}, torch::kFloat32);

    PFM_REQUIRE_INVALID_ARG(pfm::matching_precision(points_a, predicted_b, expected_b, 1.0F));
}

static void semi_dense_coverage_uses_valid_mask_area() {
    auto confidence = torch::tensor({{0.9F, 0.1F}, {0.8F, 0.7F}}, torch::kFloat32);
    auto valid = torch::tensor({{true, true}, {false, true}}, torch::kBool);

    const auto coverage = pfm::semi_dense_coverage(confidence, valid, 0.75F);

    PFM_REQUIRE_CLOSE(coverage, 1.0F / 3.0F, 1.0e-6F);
}

static void semi_dense_coverage_returns_zero_for_empty_valid_mask() {
    auto confidence = torch::tensor({{0.9F, 0.1F}, {0.8F, 0.7F}}, torch::kFloat32);
    auto valid = torch::zeros({2, 2}, torch::kBool);

    const auto coverage = pfm::semi_dense_coverage(confidence, valid, 0.75F);

    PFM_REQUIRE_CLOSE(coverage, 0.0F, 1.0e-6F);
}

static void semi_dense_coverage_rejects_shape_mismatch() {
    auto confidence = torch::empty({2, 2}, torch::kFloat32);
    auto valid = torch::empty({2, 3}, torch::kBool);

    PFM_REQUIRE_INVALID_ARG(pfm::semi_dense_coverage(confidence, valid, 0.75F));
}

static void semi_dense_coverage_rejects_undefined_confidence() {
    torch::Tensor confidence;
    auto valid = torch::empty({2, 2}, torch::kBool);

    PFM_REQUIRE_INVALID_ARG(pfm::semi_dense_coverage(confidence, valid, 0.75F));
}

static void semi_dense_coverage_rejects_integer_confidence_dtype() {
    auto confidence = torch::empty({2, 2}, torch::kInt32);
    auto valid = torch::empty({2, 2}, torch::kBool);

    PFM_REQUIRE_INVALID_ARG(pfm::semi_dense_coverage(confidence, valid, 0.75F));
}

void register_metric_tests() {
    register_test(
        "matching_precision_counts_matches_within_threshold",
        matching_precision_counts_matches_within_threshold);
    register_test(
        "matching_precision_returns_zero_for_empty_predictions",
        matching_precision_returns_zero_for_empty_predictions);
    register_test(
        "matching_precision_accepts_undefined_points_a",
        matching_precision_accepts_undefined_points_a);
    register_test(
        "matching_precision_rejects_shape_mismatch",
        matching_precision_rejects_shape_mismatch);
    register_test(
        "matching_precision_rejects_integer_predicted_b_dtype",
        matching_precision_rejects_integer_predicted_b_dtype);
    register_test(
        "semi_dense_coverage_uses_valid_mask_area",
        semi_dense_coverage_uses_valid_mask_area);
    register_test(
        "semi_dense_coverage_returns_zero_for_empty_valid_mask",
        semi_dense_coverage_returns_zero_for_empty_valid_mask);
    register_test(
        "semi_dense_coverage_rejects_shape_mismatch",
        semi_dense_coverage_rejects_shape_mismatch);
    register_test(
        "semi_dense_coverage_rejects_undefined_confidence",
        semi_dense_coverage_rejects_undefined_confidence);
    register_test(
        "semi_dense_coverage_rejects_integer_confidence_dtype",
        semi_dense_coverage_rejects_integer_confidence_dtype);
}
