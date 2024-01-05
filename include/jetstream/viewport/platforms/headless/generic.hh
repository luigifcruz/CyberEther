#ifndef JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_GENERIC_HH
#define JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_GENERIC_HH

#include "jetstream/backend/base.hh"
#include "jetstream/viewport/adapters/generic.hh"
#include "jetstream/viewport/plugins/endpoint.hh"

namespace Jetstream::Viewport {

template<Device DeviceId>
class Headless : Adapter<DeviceId> {};

}  // namespace Jetstream::Viewport

#endif
