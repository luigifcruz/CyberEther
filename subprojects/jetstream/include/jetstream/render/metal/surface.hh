#ifndef JETSTREAM_RENDER_METAL_SURFACE_HH
#define JETSTREAM_RENDER_METAL_SURFACE_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class SurfaceImp<Device::Metal> : public Surface {
 public:
    explicit SurfaceImp(const Config& config);

    const Size2D<U64>& size(const Size2D<U64>& size);

 protected:
    const Result create();
    const Result destroy();
    const Result draw(MTL::CommandBuffer* commandBuffer);

 private:
    MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;
    std::shared_ptr<TextureImp<Device::Metal>> framebuffer;
    std::vector<std::shared_ptr<ProgramImp<Device::Metal>>> programs;

    const Result createFramebuffer();
    const Result destroyFramebuffer();

    friend class WindowImp<Device::Metal>;
};

}  // namespace Jetstream::Render

#endif
