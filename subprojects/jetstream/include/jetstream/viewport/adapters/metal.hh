#ifndef JETSTREAM_VIEWPORT_ADAPTER_METAL_HH
#define JETSTREAM_VIEWPORT_ADAPTER_METAL_HH

#include "jetstream/viewport/adapters/generic.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Viewport {

template<>
class Adapter<Device::Metal> : public Generic {
 public:
    using Generic::Generic;
    
    virtual void* nextDrawable() = 0;
};

}  // namespace Jetstream::Viewport

#endif
