#include "logging/metric_logger_group.h"

#include <stdexcept>
#include <utility>

namespace pfm
{

void MetricLoggerGroup::addLogger(std::unique_ptr<TrainingMetricLogger> logger)
{
    if (!logger)
    {
        throw std::invalid_argument("metric logger group cannot add a null logger");
    }
    _loggers.push_back(std::move(logger));
}

void MetricLoggerGroup::logIteration(const TrainingMetric& metric)
{
    for (const auto& logger : _loggers)
    {
        logger->logIteration(metric);
    }
}

void MetricLoggerGroup::logEpochSummary(const TrainingMetric& metric)
{
    for (const auto& logger : _loggers)
    {
        logger->logEpochSummary(metric);
    }
}

void MetricLoggerGroup::flush()
{
    for (const auto& logger : _loggers)
    {
        logger->flush();
    }
}

} // namespace pfm
