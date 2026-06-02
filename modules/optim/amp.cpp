#include "optim/amp.h"

#include <ATen/autocast_mode.h>

namespace pfm
{

AmpAutocastGuard::AmpAutocastGuard(bool enabled, c10::DeviceType device_type, at::ScalarType dtype)
    : _enabled(enabled), _device_type(device_type), _previous_dtype(at::autocast::get_autocast_dtype(device_type)),
      _previous_enabled(at::autocast::is_autocast_enabled(device_type))
{
    if (_enabled)
    {
        at::autocast::set_autocast_dtype(_device_type, dtype);
        at::autocast::set_autocast_enabled(_device_type, true);
    }
}

AmpAutocastGuard::~AmpAutocastGuard()
{
    if (_enabled)
    {
        at::autocast::set_autocast_enabled(_device_type, _previous_enabled);
        at::autocast::set_autocast_dtype(_device_type, _previous_dtype);
    }
}

} // namespace pfm
