#pragma once

#include <ATen/core/ScalarType.h>
#include <c10/core/DeviceType.h>

namespace pfm
{

class AmpAutocastGuard
{
  public:
    /// Enables autocast for the lifetime of this guard when requested.
    /// @param enabled Whether autocast should be enabled.
    /// @param device_type Device type, normally CUDA.
    /// @param dtype Autocast dtype.
    AmpAutocastGuard(bool enabled, c10::DeviceType device_type, at::ScalarType dtype);
    AmpAutocastGuard(const AmpAutocastGuard&) = delete;
    AmpAutocastGuard& operator=(const AmpAutocastGuard&) = delete;
    AmpAutocastGuard(AmpAutocastGuard&&) = delete;
    AmpAutocastGuard& operator=(AmpAutocastGuard&&) = delete;
    ~AmpAutocastGuard();

  private:
    bool _enabled = false;
    c10::DeviceType _device_type;
    at::ScalarType _previous_dtype;
    bool _previous_enabled = false;
};

} // namespace pfm
