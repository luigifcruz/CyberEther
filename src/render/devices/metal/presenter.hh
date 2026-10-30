#ifndef JETSTREAM_RENDER_METAL_PRESENTER_HH
#define JETSTREAM_RENDER_METAL_PRESENTER_HH

#include "jetstream/render/base/present.hh"
#include "jetstream/render/devices/metal/window.hh"

namespace Jetstream::Render {

struct WindowImp<DeviceType::Metal>::Presenter {
    Result create(MTL::PixelFormat pixelFormat);
    void destroy();
    void encode(MTL::RenderCommandEncoder* encoder,
                const PresentList& list,
                const WindowMetrics& metrics,
                const Extent2D<U64>& target);

 private:
    MTL::RenderPipelineState* straight = nullptr;
    MTL::RenderPipelineState* premultiplied = nullptr;
    MTL::SamplerState* sampler = nullptr;
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_METAL_PRESENTER_HH
