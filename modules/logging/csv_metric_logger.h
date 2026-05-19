#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "logging/training_metric.h"

namespace pfm {

class CsvMetricLogger : public TrainingMetricLogger {
public:
    /// Opens a CSV metric logger.
    /// @param path Output CSV path.
    /// @param value_columns Metric value columns appended after fixed columns.
    /// @throws std::runtime_error if the file cannot be opened.
    CsvMetricLogger(std::string path, std::vector<std::string> value_columns);

    /// Writes one CSV row for a training iteration.
    /// @param metric Metric record to log.
    void logIteration(const TrainingMetric& metric) override;

    /// Ignores epoch summary records.
    /// @param metric Metric record to log.
    void logEpochSummary(const TrainingMetric& metric) override;

    /// Flushes the CSV file.
    void flush() override;

private:
    void writeHeader();

    std::ofstream _output;
    std::vector<std::string> _value_columns;
    bool _header_written = false;
};

}  // namespace pfm
