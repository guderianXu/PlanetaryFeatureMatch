#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace pfm
{

struct GpuMetrics
{
    std::optional<double> utilization_percent;
    std::optional<double> power_watts;
};

struct TrainingMetric
{
    int epoch = 0;
    int total_epochs = 0;
    int iteration = 0;
    int total_iterations = 0;
    int images_seen = 0;
    int total_images = 0;
    double learning_rate = 0.0;
    double elapsed_seconds = 0.0;
    std::unordered_map<std::string, double> values;
};

class TrainingMetricLogger
{
  public:
    /// Destroys the logger.
    virtual ~TrainingMetricLogger() = default;

    /// Logs one training iteration.
    /// @param metric Metric record to log.
    virtual void logIteration(const TrainingMetric& metric) = 0;

    /// Logs one epoch summary.
    /// @param metric Metric record to log.
    virtual void logEpochSummary(const TrainingMetric& metric) = 0;

    /// Flushes buffered output.
    virtual void flush() = 0;
};

} // namespace pfm
