#ifndef JETSTREAM_RENDER_VULKAN_SURFACE_HH
#define JETSTREAM_RENDER_VULKAN_SURFACE_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class SurfaceImp<DeviceType::Vulkan> : public Surface {
 public:
    explicit SurfaceImp(const Config& config);

    Result create() override;
    Result destroy() override;

    const Extent2D<U64>& size(const Extent2D<U64>& size) override;

 protected:
    Result encode(VkCommandBuffer& commandBuffer);

 private:
    std::shared_ptr<TextureImp<DeviceType::Vulkan>> framebufferResolve;
    std::shared_ptr<TextureImp<DeviceType::Vulkan>> framebuffer;
    VkFramebuffer framebufferObject;
    VkRenderPass renderPass;
    Extent2D<U64> requestedSize;

    std::vector<std::shared_ptr<ProgramImp<DeviceType::Vulkan>>> programs;
    std::vector<std::shared_ptr<KernelImp<DeviceType::Vulkan>>> kernels;
    std::vector<std::shared_ptr<BufferImp<DeviceType::Vulkan>>> buffers;

    friend class WindowImp<DeviceType::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
