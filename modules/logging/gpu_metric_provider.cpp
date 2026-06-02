#include "logging/gpu_metric_provider.h"

namespace pfm
{

GpuMetrics NullGpuMetricProvider::sample()
{
    return GpuMetrics{};
}

std::unique_ptr<GpuMetricProvider> makeDefaultGpuMetricProvider()
{
    return makeNvmlGpuMetricProvider();
}

} // namespace pfm
