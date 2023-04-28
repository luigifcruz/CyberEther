#ifndef JETSTREAM_RENDER_VULKAN_SURFACE_HH
#define JETSTREAM_RENDER_VULKAN_SURFACE_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class SurfaceImp<Device::Vulkan> : public Surface {
 public:
    explicit SurfaceImp(const Config& config);

    const Size2D<U64>& size(const Size2D<U64>& size);

 protected:
    Result create();
    Result destroy();
    Result draw(VkCommandBuffer* commandBuffer);

 private:
    // MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;
    std::shared_ptr<TextureImp<Device::Vulkan>> framebuffer;
    std::vector<std::shared_ptr<ProgramImp<Device::Vulkan>>> programs;

    Result createFramebuffer();
    Result destroyFramebuffer();

    friend class WindowImp<Device::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
