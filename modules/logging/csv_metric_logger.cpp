#include "logging/csv_metric_logger.h"

#include <stdexcept>
#include <utility>

namespace pfm {
namespace {

void writeFixedColumns(std::ostream& output, const TrainingMetric& metric) {
    output << metric.epoch << ',' << metric.total_epochs << ',' << metric.iteration << ','
           << metric.total_iterations << ',' << metric.images_seen << ',' << metric.total_images << ','
           << metric.learning_rate << ',' << metric.elapsed_seconds;
}

}  // namespace

CsvMetricLogger::CsvMetricLogger(std::string path, std::vector<std::string> value_columns)
    : _output(path), _value_columns(std::move(value_columns)) {
    if (!_output.is_open()) {
        throw std::runtime_error("failed to open CSV metric log: " + path);
    }
}

void CsvMetricLogger::logIteration(const TrainingMetric& metric) {
    if (!_header_written) {
        writeHeader();
    }
    writeFixedColumns(_output, metric);
    for (const auto& column : _value_columns) {
        _output << ',';
        const auto it = metric.values.find(column);
        if (it != metric.values.end()) {
            _output << it->second;
        }
    }
    _output << '\n';
}

void CsvMetricLogger::logEpochSummary(const TrainingMetric&) {}

void CsvMetricLogger::flush() {
    _output.flush();
}

void CsvMetricLogger::writeHeader() {
    _output << "epoch,total_epochs,iteration,total_iterations,images_seen,total_images,learning_rate,elapsed_seconds";
    for (const auto& column : _value_columns) {
        _output << ',' << column;
    }
    _output << '\n';
    _header_written = true;
}

}  // namespace pfm
