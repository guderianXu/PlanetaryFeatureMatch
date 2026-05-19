#pragma once

#include <iosfwd>

#include "logging/training_metric.h"

namespace pfm {

class ConsoleProgressLogger : public TrainingMetricLogger {
public:
    /// Creates a console progress logger.
    /// @param stream Output stream.
    /// @param bar_width Progress bar width in characters.
    ConsoleProgressLogger(std::ostream& stream, int bar_width);

    /// Logs one training iteration as a progress line.
    /// @param metric Metric record to log.
    void logIteration(const TrainingMetric& metric) override;

    /// Logs one epoch summary line.
    /// @param metric Metric record to log.
    void logEpochSummary(const TrainingMetric& metric) override;

    /// Flushes the output stream.
    void flush() override;

private:
    std::ostream& _stream;
    int _bar_width;
};

}  // namespace pfm
