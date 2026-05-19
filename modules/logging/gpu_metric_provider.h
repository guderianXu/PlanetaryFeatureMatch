#pragma once

#include <memory>

#include "logging/training_metric.h"

namespace pfm {

class GpuMetricProvider {
public:
    /// Destroys the provider.
    virtual ~GpuMetricProvider() = default;

    /// Samples current GPU metrics.
    /// @return Current utilization and power values when available.
    virtual GpuMetrics sample() = 0;
};

class NullGpuMetricProvider : public GpuMetricProvider {
public:
    /// Samples no GPU metrics.
    /// @return Empty metric values.
    GpuMetrics sample() override;
};

/// Creates an NVML provider when available for this build, otherwise a null provider.
/// @return GPU metric provider instance.
std::unique_ptr<GpuMetricProvider> makeNvmlGpuMetricProvider();

/// Creates the default GPU metric provider for this build.
/// @return GPU metric provider instance.
std::unique_ptr<GpuMetricProvider> makeDefaultGpuMetricProvider();

}  // namespace pfm
