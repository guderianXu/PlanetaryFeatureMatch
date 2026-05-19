#include "logging/progress_logger.h"

#include <algorithm>
#include <ostream>
#include <string>

namespace pfm {
namespace {

double metricValue(const TrainingMetric& metric, const std::string& key) {
    const auto it = metric.values.find(key);
    return it == metric.values.end() ? 0.0 : it->second;
}

}  // namespace

ConsoleProgressLogger::ConsoleProgressLogger(std::ostream& stream, int bar_width)
    : _stream(stream), _bar_width(std::max(1, bar_width)) {}

void ConsoleProgressLogger::logIteration(const TrainingMetric& metric) {
    const int filled = metric.total_iterations > 0
                           ? std::min(_bar_width, metric.iteration * _bar_width / metric.total_iterations)
                           : 0;
    _stream << '\r' << "epoch " << metric.epoch << '/' << metric.total_epochs << " [";
    for (int index = 0; index < _bar_width; ++index) {
        _stream << (index < filled ? '=' : '-');
    }
    _stream << "] " << metric.iteration << '/' << metric.total_iterations
            << " loss=" << metricValue(metric, "loss_total")
            << " match=" << metricValue(metric, "matcher_loss")
            << " feat=" << metricValue(metric, "feature_loss")
            << " rep=" << metricValue(metric, "repeatability_loss")
            << " dense=" << metricValue(metric, "dense_loss")
            << " off=" << metricValue(metric, "offset_error_px")
            << " desc_acc=" << metricValue(metric, "descriptor_accuracy")
            << " div=" << metricValue(metric, "descriptor_diversity")
            << " lr=" << metric.learning_rate;
    const auto gpu_util = metric.values.find("gpu_utilization_percent");
    if (gpu_util != metric.values.end()) {
        _stream << " gpu=" << gpu_util->second << '%';
    }
    const auto power = metric.values.find("gpu_power_watts");
    if (power != metric.values.end()) {
        _stream << " power=" << power->second << 'W';
    }
    _stream.flush();
}

void ConsoleProgressLogger::logEpochSummary(const TrainingMetric& metric) {
    _stream << "\nepoch summary: epoch=" << metric.epoch << '/' << metric.total_epochs
            << " elapsed=" << metric.elapsed_seconds << "s\n";
    _stream.flush();
}

void ConsoleProgressLogger::flush() {
    _stream.flush();
}

}  // namespace pfm
