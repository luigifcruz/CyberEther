#ifndef JETSTREAM_VIEWPORT_ADAPTER_METAL_HH
#define JETSTREAM_VIEWPORT_ADAPTER_METAL_HH

#include "jetstream/viewport/adapters/generic.hh"
#include "jetstream/backend/base.hh"

#include <functional>

namespace Jetstream::Viewport {

template<>
class Adapter<DeviceType::Metal> : public Generic {
 public:
    using Generic::Generic;

    virtual MTL::Texture* nextDrawable(CA::MetalDrawable*& drawable) = 0;
    virtual Extent2D<U64> getSwapchainExtent() const = 0;

    using CaptureCallback = std::function<Result(MTL::CommandBuffer*, MTL::Texture*)>;

    void setCaptureCallback(CaptureCallback callback);
    Result encodeCapture(MTL::CommandBuffer* commandBuffer, MTL::Texture* texture);

 protected:
    virtual void setCaptureEnabled(bool enabled);

 private:
    CaptureCallback captureCallback;
};

}  // namespace Jetstream::Viewport

#endif
