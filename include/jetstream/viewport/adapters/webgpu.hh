#ifndef JETSTREAM_VIEWPORT_ADAPTER_WEBGPU_HH
#define JETSTREAM_VIEWPORT_ADAPTER_WEBGPU_HH

#include "jetstream/viewport/adapters/generic.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Viewport {

template<>
class Adapter<Device::WebGPU> : public Generic {
 public:
    using Generic::Generic;

    virtual Result createSwapchain() = 0;
    virtual Result destroySwapchain() = 0;

    virtual Result nextDrawable() = 0;
    virtual Result commitDrawable(wgpu::TextureView& framebufferTexture) = 0;
};

}  // namespace Jetstream::Viewport

#endif
