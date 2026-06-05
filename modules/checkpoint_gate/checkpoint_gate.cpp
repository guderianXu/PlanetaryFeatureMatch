#include "checkpoint_gate/checkpoint_gate.h"

#include <regex>
#include <sstream>
#include <stdexcept>

namespace pfm
{

int64_t CheckpointGateMetrics::total_matches() const
{
    return correct_matches + wrong_matches;
}

namespace
{

int64_t parseIntField(const std::string& text, const char* name)
{
    const std::regex pattern(std::string(name) + "=([-+]?[0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        throw std::invalid_argument(std::string("missing match metric field: ") + name);
    }
    return std::stoll(match[1].str());
}

double parseDoubleField(const std::string& text, const char* name)
{
    const std::regex pattern(std::string(name) + "=([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        throw std::invalid_argument(std::string("missing match metric field: ") + name);
    }
    return std::stod(match[1].str());
}

double parseOptionalDoubleField(const std::string& text, const char* name, double fallback)
{
    const std::regex pattern(std::string(name) + "=([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        return fallback;
    }
    return std::stod(match[1].str());
}

} // namespace

CheckpointGateMetrics parse_checkpoint_gate_metrics(const std::string& match_output)
{
    return CheckpointGateMetrics{parseIntField(match_output, "correct_matches"),
                                 parseIntField(match_output, "wrong_matches"),
                                 parseDoubleField(match_output, "match_precision"),
                                 parseOptionalDoubleField(match_output, "graph_work", 0.0)};
}

CheckpointGateDecision evaluate_checkpoint_gate_metrics(const CheckpointGateMetrics& metrics,
                                                        const CheckpointGateThreshold& threshold)
{
    if (metrics.correct_matches < threshold.min_correct_matches)
    {
        std::ostringstream reason;
        reason << "correct_matches " << metrics.correct_matches << " below required " << threshold.min_correct_matches;
        return CheckpointGateDecision{false, reason.str()};
    }
    if (metrics.precision < threshold.min_precision)
    {
        std::ostringstream reason;
        reason << "match_precision " << metrics.precision << " below required " << threshold.min_precision;
        return CheckpointGateDecision{false, reason.str()};
    }
    if (metrics.graph_attention_work_fraction > threshold.max_graph_attention_work_fraction)
    {
        std::ostringstream reason;
        reason << "graph_work " << metrics.graph_attention_work_fraction << " above required "
               << threshold.max_graph_attention_work_fraction;
        return CheckpointGateDecision{false, reason.str()};
    }
    return CheckpointGateDecision{true, "passed"};
}

} // namespace pfm
