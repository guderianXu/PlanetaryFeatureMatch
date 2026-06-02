#include "infer/checkpoint_gate.h"
#include "tests/test_harness.h"

namespace
{

static void checkpoint_gate_parses_match_cli_metrics()
{
    const auto metrics = pfm::parse_checkpoint_gate_metrics(
        "matching complete: sparse_matches=421 correct_matches=421 wrong_matches=0 match_precision=1 elapsed=0.895s");
    PFM_REQUIRE(metrics.correct_matches == 421);
    PFM_REQUIRE(metrics.wrong_matches == 0);
    PFM_REQUIRE(metrics.total_matches() == 421);
    PFM_REQUIRE_CLOSE(metrics.precision, 1.0, 1.0e-9);
}

static void checkpoint_gate_rejects_too_few_correct_matches()
{
    const pfm::CheckpointGateMetrics metrics{3, 12, 0.2};
    const pfm::CheckpointGateThreshold threshold{400, 0.99};
    const auto decision = pfm::evaluate_checkpoint_gate_metrics(metrics, threshold);
    PFM_REQUIRE(!decision.passed);
    PFM_REQUIRE(decision.reason.find("correct_matches") != std::string::npos);
}

static void checkpoint_gate_rejects_low_precision_even_with_many_matches()
{
    const pfm::CheckpointGateMetrics metrics{421, 20, 0.954648};
    const pfm::CheckpointGateThreshold threshold{400, 0.99};
    const auto decision = pfm::evaluate_checkpoint_gate_metrics(metrics, threshold);
    PFM_REQUIRE(!decision.passed);
    PFM_REQUIRE(decision.reason.find("match_precision") != std::string::npos);
}

static void checkpoint_gate_accepts_metrics_that_meet_thresholds()
{
    const pfm::CheckpointGateMetrics metrics{421, 0, 1.0};
    const pfm::CheckpointGateThreshold threshold{400, 0.99};
    const auto decision = pfm::evaluate_checkpoint_gate_metrics(metrics, threshold);
    PFM_REQUIRE(decision.passed);
}

} // namespace

void register_checkpoint_gate_tests()
{
    register_test("checkpoint_gate_parses_match_cli_metrics", checkpoint_gate_parses_match_cli_metrics);
    register_test("checkpoint_gate_rejects_too_few_correct_matches", checkpoint_gate_rejects_too_few_correct_matches);
    register_test("checkpoint_gate_rejects_low_precision_even_with_many_matches",
                  checkpoint_gate_rejects_low_precision_even_with_many_matches);
    register_test("checkpoint_gate_accepts_metrics_that_meet_thresholds",
                  checkpoint_gate_accepts_metrics_that_meet_thresholds);
}
