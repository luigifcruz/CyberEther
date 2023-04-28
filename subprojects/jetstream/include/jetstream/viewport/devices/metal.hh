#ifndef JETSTREAM_VIEWPORT_DEVICES_METAL_HH
#define JETSTREAM_VIEWPORT_DEVICES_METAL_HH

#include "jetstream/viewport/generic.hh"

namespace Jetstream::Viewport {

template<>
class Provider<Device::Metal> : public Generic {
 public:
    using Generic::Generic;
    
    virtual void* nextDrawable() = 0;
};

}  // namespace Jetstream::Viewport

#endif
