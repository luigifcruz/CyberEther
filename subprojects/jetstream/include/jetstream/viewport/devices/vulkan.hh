#ifndef JETSTREAM_VIEWPORT_DEVICES_VULKAN_HH
#define JETSTREAM_VIEWPORT_DEVICES_VULKAN_HH

#include "jetstream/viewport/generic.hh"

namespace Jetstream::Viewport {

template<>
class Provider<Device::Vulkan> : public Generic {
 public:
    using Generic::Generic;
    
    virtual void* nextDrawable() = 0;
};

}  // namespace Jetstream::Viewport

#endif
