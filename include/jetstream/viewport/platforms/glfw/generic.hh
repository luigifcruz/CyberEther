#ifndef JETSTREAM_VIEWPORT_PLATFORM_GLFW_GENERIC_HH
#define JETSTREAM_VIEWPORT_PLATFORM_GLFW_GENERIC_HH

#include "jetstream/backend/base.hh"
#include "jetstream/viewport/adapters/generic.hh"

#include "jetstream/render/tools/imgui_impl_glfw.h"

namespace Jetstream::Viewport {

template<Device DeviceId>
class GLFW : Adapter<DeviceId> {};

}  // namespace Jetstream::Viewport

#endif
