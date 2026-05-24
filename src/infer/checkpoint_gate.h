#pragma once

#include <cstdint>
#include <string>

namespace pfm {

struct CheckpointGateMetrics {
    int64_t correct_matches = 0;
    int64_t wrong_matches = 0;
    double precision = 0.0;

    int64_t total_matches() const;
};

struct CheckpointGateThreshold {
    int64_t min_correct_matches = 0;
    double min_precision = 0.0;
};

struct CheckpointGateDecision {
    bool passed = false;
    std::string reason;
};

CheckpointGateMetrics parse_checkpoint_gate_metrics(const std::string& match_output);

CheckpointGateDecision evaluate_checkpoint_gate_metrics(
    const CheckpointGateMetrics& metrics,
    const CheckpointGateThreshold& threshold);

}  // namespace pfm
