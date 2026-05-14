#include "tests/test_harness.h"

#include <torch/torch.h>

#include "models/matcher.h"

static void matcher_outputs_sparse_score_matrix() {
    pfm::Matcher matcher(32);
    auto desc_a = torch::randn({2, 5, 32}, torch::kFloat32);
    auto desc_b = torch::randn({2, 7, 32}, torch::kFloat32);

    auto scores = matcher->forward(desc_a, desc_b);

    PFM_REQUIRE(scores.sizes() == torch::IntArrayRef({2, 5, 7}));
}

void register_matcher_tests() {
    register_test("matcher outputs sparse score matrix", matcher_outputs_sparse_score_matrix);
}
