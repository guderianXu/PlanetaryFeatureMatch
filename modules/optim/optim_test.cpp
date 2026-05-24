#include "tests/test_harness.h"

#include <torch/torch.h>

#include "optim/descriptor_similarity.h"

namespace {

torch::Tensor referenceCosineDescriptorSimilarity(
    const torch::Tensor& descriptors_a,
    const torch::Tensor& descriptors_b
) {
    auto normalized_a = descriptors_a / descriptors_a.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    auto normalized_b = descriptors_b / descriptors_b.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    return torch::bmm(normalized_a, normalized_b.transpose(1, 2));
}

void descriptor_similarity_matches_cosine_reference() {
    torch::manual_seed(7);
    auto descriptors_a = torch::randn({2, 5, 16}, torch::kFloat32);
    auto descriptors_b = torch::randn({2, 9, 16}, torch::kFloat32);

    auto actual = pfm::cyclicDescriptorSimilarityScores(descriptors_a, descriptors_b);
    auto expected = referenceCosineDescriptorSimilarity(descriptors_a, descriptors_b);

    PFM_REQUIRE(torch::allclose(actual, expected, 1.0e-5, 1.0e-5));
}

void descriptor_similarity_matches_cosine_for_non_c4_descriptors() {
    torch::manual_seed(11);
    auto descriptors_a = torch::randn({2, 4, 10}, torch::kFloat32);
    auto descriptors_b = torch::randn({2, 6, 10}, torch::kFloat32);
    auto normalized_a = descriptors_a / descriptors_a.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();
    auto normalized_b = descriptors_b / descriptors_b.pow(2).sum(2, true).clamp_min(1.0e-12).sqrt();

    auto actual = pfm::cyclicDescriptorSimilarityScores(descriptors_a, descriptors_b);
    auto expected = torch::bmm(normalized_a, normalized_b.transpose(1, 2));

    PFM_REQUIRE(torch::allclose(actual, expected, 1.0e-5, 1.0e-5));
}

void descriptor_similarity_rejects_arbitrary_cyclic_quarter_shift() {
    auto descriptors_a = torch::tensor({{{1.0F, 0.0F, 0.0F, 0.0F}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{0.0F, 1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F, 0.0F}}}, torch::kFloat32);

    auto actual = pfm::cyclicDescriptorSimilarityScores(descriptors_a, descriptors_b);

    PFM_REQUIRE_CLOSE(actual.index({0, 0, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(actual.index({0, 0, 1}).item<float>(), 0.0F, 1.0e-6F);
}

void chunked_descriptor_similarity_matches_unchunked() {
    torch::manual_seed(19);
    auto descriptors_a = torch::randn({2, 17, 16}, torch::kFloat32);
    auto descriptors_b = torch::randn({2, 23, 16}, torch::kFloat32);

    auto actual = pfm::cyclicDescriptorSimilarityScoresChunked(descriptors_a, descriptors_b, 5);
    auto expected = pfm::cyclicDescriptorSimilarityScores(descriptors_a, descriptors_b);

    PFM_REQUIRE(torch::allclose(actual, expected, 1.0e-5, 1.0e-5));
}

}  // namespace

void register_optim_tests() {
    register_test("descriptor similarity matches cosine reference", descriptor_similarity_matches_cosine_reference);
    register_test(
        "descriptor similarity matches cosine for non c4 descriptors",
        descriptor_similarity_matches_cosine_for_non_c4_descriptors);
    register_test(
        "descriptor similarity rejects arbitrary cyclic quarter shift",
        descriptor_similarity_rejects_arbitrary_cyclic_quarter_shift);
    register_test(
        "chunked descriptor similarity matches unchunked",
        chunked_descriptor_similarity_matches_unchunked);
}
