#ifndef JETSTREAM_RENDER_METAL_SURFACE_HH
#define JETSTREAM_RENDER_METAL_SURFACE_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class SurfaceImp<DeviceType::Metal> : public Surface {
 public:
    explicit SurfaceImp(const Config& config);

    Result create() override;
    Result destroy() override;

    const Extent2D<U64>& size(const Extent2D<U64>& size) override;

 protected:
    Result draw(MTL::CommandBuffer* commandBuffer);

 private:
    Extent2D<U64> requestedSize;
    MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;
    std::shared_ptr<TextureImp<DeviceType::Metal>> framebuffer;
    std::shared_ptr<TextureImp<DeviceType::Metal>> framebufferResolve;
    std::vector<std::shared_ptr<ProgramImp<DeviceType::Metal>>> programs;
    std::vector<std::shared_ptr<KernelImp<DeviceType::Metal>>> kernels;
    std::vector<std::shared_ptr<BufferImp<DeviceType::Metal>>> buffers;

    Result createFramebuffer();
    Result destroyFramebuffer();

    friend class WindowImp<DeviceType::Metal>;
};

}  // namespace Jetstream::Render

#endif
