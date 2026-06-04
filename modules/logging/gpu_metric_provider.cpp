#include "logging/gpu_metric_provider.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace pfm
{
namespace
{

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> values;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        values.push_back(trim(item));
    }
    return values;
}

std::optional<double> parseOptionalDouble(const std::string& value)
{
    const auto text = trim(value);
    if (text.empty() || text == "[N/A]" || text == "N/A" || text == "Not Supported")
    {
        return std::nullopt;
    }
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str())
    {
        return std::nullopt;
    }
    return parsed;
}

class NvidiaSmiGpuMetricProvider : public GpuMetricProvider
{
  public:
    GpuMetrics sample() override
    {
        const char* command =
            "nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.total,memory.free "
            "--format=csv,noheader,nounits -i 0 2>/dev/null";
        std::array<char, 256> buffer{};
        std::string output;
        FILE* pipe = popen(command, "r");
        if (pipe == nullptr)
        {
            return GpuMetrics{};
        }
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
        {
            output += buffer.data();
        }
        const auto status = pclose(pipe);
        if (status != 0 || output.empty())
        {
            return GpuMetrics{};
        }

        const auto line_end = output.find_first_of("\r\n");
        const auto line = line_end == std::string::npos ? output : output.substr(0, line_end);
        const auto values = splitCsvLine(line);
        if (values.size() < 5)
        {
            return GpuMetrics{};
        }

        GpuMetrics result;
        result.utilization_percent = parseOptionalDouble(values[0]);
        result.power_watts = parseOptionalDouble(values[1]);
        result.memory_used_mb = parseOptionalDouble(values[2]);
        result.memory_total_mb = parseOptionalDouble(values[3]);
        result.memory_free_mb = parseOptionalDouble(values[4]);
        return result;
    }
};

class DefaultGpuMetricProvider : public GpuMetricProvider
{
  public:
    DefaultGpuMetricProvider()
        : _primary(makeNvmlGpuMetricProvider()), _fallback(std::make_unique<NvidiaSmiGpuMetricProvider>())
    {
    }

    GpuMetrics sample() override
    {
        auto metrics = _primary ? _primary->sample() : GpuMetrics{};
        if (hasAnyMetric(metrics))
        {
            return metrics;
        }
        return _fallback ? _fallback->sample() : GpuMetrics{};
    }

  private:
    static bool hasAnyMetric(const GpuMetrics& metrics)
    {
        return metrics.utilization_percent.has_value() || metrics.power_watts.has_value() ||
               metrics.memory_used_mb.has_value() || metrics.memory_total_mb.has_value() ||
               metrics.memory_free_mb.has_value();
    }

    std::unique_ptr<GpuMetricProvider> _primary;
    std::unique_ptr<GpuMetricProvider> _fallback;
};

} // namespace

GpuMetrics NullGpuMetricProvider::sample()
{
    return GpuMetrics{};
}

std::unique_ptr<GpuMetricProvider> makeDefaultGpuMetricProvider()
{
    return std::make_unique<DefaultGpuMetricProvider>();
}

} // namespace pfm
