#ifndef JETSTREAM_VIEWPORT_PLATFORM_IOS_GENERIC_HH
#define JETSTREAM_VIEWPORT_PLATFORM_IOS_GENERIC_HH

#include "jetstream/backend/base.hh"
#include "jetstream/viewport/adapters/generic.hh"

namespace Jetstream::Viewport {

template<Device DeviceId>
class iOS : Adapter<DeviceId> {};

}  // namespace Jetstream::Viewport

#endif
