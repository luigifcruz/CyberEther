#ifndef JETSTREAM_RENDER_VULKAN_SURFACE_HH
#define JETSTREAM_RENDER_VULKAN_SURFACE_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class SurfaceImp<Device::Vulkan> : public Surface {
 public:
    explicit SurfaceImp(const Config& config);

    const Extent2D<U64>& size(const Extent2D<U64>& size);

 protected:
    Result create();
    Result encode(VkCommandBuffer& commandBuffer);
    Result destroy();

 private:
    std::shared_ptr<TextureImp<Device::Vulkan>> framebufferResolve;
    std::shared_ptr<TextureImp<Device::Vulkan>> framebuffer;
    VkFramebuffer framebufferObject;
    VkRenderPass renderPass;
    Extent2D<U64> requestedSize;
      
    std::vector<std::shared_ptr<ProgramImp<Device::Vulkan>>> programs;
    std::vector<std::shared_ptr<KernelImp<Device::Vulkan>>> kernels;
    std::vector<std::shared_ptr<BufferImp<Device::Vulkan>>> buffers;

    friend class WindowImp<Device::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
