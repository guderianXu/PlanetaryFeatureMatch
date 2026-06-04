#include "logging/gpu_metric_provider.h"

#ifdef PFM_HAS_NVML
#include <nvml.h>
#endif

#include <memory>

namespace pfm
{

#ifdef PFM_HAS_NVML
namespace
{

double bytesToMiB(unsigned long long bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace

class NvmlGpuMetricProvider : public GpuMetricProvider
{
  public:
    NvmlGpuMetricProvider()
    {
        _available = nvmlInit_v2() == NVML_SUCCESS;
        if (_available)
        {
            _available = nvmlDeviceGetHandleByIndex_v2(0, &_device) == NVML_SUCCESS;
        }
    }

    ~NvmlGpuMetricProvider() override
    {
        if (_available)
        {
            nvmlShutdown();
        }
    }

    GpuMetrics sample() override
    {
        if (!_available)
        {
            return GpuMetrics{};
        }

        GpuMetrics result;
        nvmlUtilization_t utilization{};
        if (nvmlDeviceGetUtilizationRates(_device, &utilization) == NVML_SUCCESS)
        {
            result.utilization_percent = static_cast<double>(utilization.gpu);
        }

        unsigned int milliwatts = 0;
        if (nvmlDeviceGetPowerUsage(_device, &milliwatts) == NVML_SUCCESS)
        {
            result.power_watts = static_cast<double>(milliwatts) / 1000.0;
        }

        nvmlMemory_t memory{};
        if (nvmlDeviceGetMemoryInfo(_device, &memory) == NVML_SUCCESS)
        {
            result.memory_used_mb = bytesToMiB(memory.used);
            result.memory_total_mb = bytesToMiB(memory.total);
            result.memory_free_mb = bytesToMiB(memory.free);
        }
        return result;
    }

  private:
    bool _available = false;
    nvmlDevice_t _device{};
};
#endif

std::unique_ptr<GpuMetricProvider> makeNvmlGpuMetricProvider()
{
#ifdef PFM_HAS_NVML
    return std::make_unique<NvmlGpuMetricProvider>();
#else
    return std::make_unique<NullGpuMetricProvider>();
#endif
}

} // namespace pfm
