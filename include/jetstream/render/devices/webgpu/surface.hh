#ifndef JETSTREAM_RENDER_WEBGPU_SURFACE_HH
#define JETSTREAM_RENDER_WEBGPU_SURFACE_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class SurfaceImp<Device::WebGPU> : public Surface {
 public:
    explicit SurfaceImp(const Config& config);

    const Extent2D<U64>& size(const Extent2D<U64>& size);

 protected:
    Result create();
    Result destroy();
    Result draw(wgpu::CommandEncoder& commandEncoder);

 private:
    std::shared_ptr<TextureImp<Device::WebGPU>> framebuffer;
    std::vector<std::shared_ptr<ProgramImp<Device::WebGPU>>> programs;
    std::vector<std::shared_ptr<KernelImp<Device::WebGPU>>> kernels;
    std::vector<std::shared_ptr<BufferImp<Device::WebGPU>>> buffers;
    Extent2D<U64> requestedSize;

    Result createFramebuffer();
    Result destroyFramebuffer();

    friend class WindowImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
