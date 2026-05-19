#pragma once

#include <memory>
#include <vector>

#include "logging/training_metric.h"

namespace pfm {

class MetricLoggerGroup : public TrainingMetricLogger {
public:
    /// Adds a logger sink owned by this group.
    /// @param logger Logger to add.
    /// @throws std::invalid_argument if logger is null.
    void addLogger(std::unique_ptr<TrainingMetricLogger> logger);

    /// Forwards an iteration metric to all loggers.
    /// @param metric Metric record to log.
    void logIteration(const TrainingMetric& metric) override;

    /// Forwards an epoch summary metric to all loggers.
    /// @param metric Metric record to log.
    void logEpochSummary(const TrainingMetric& metric) override;

    /// Flushes all loggers.
    void flush() override;

private:
    std::vector<std::unique_ptr<TrainingMetricLogger>> _loggers;
};

}  // namespace pfm
