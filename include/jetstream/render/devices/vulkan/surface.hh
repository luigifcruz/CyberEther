#ifndef JETSTREAM_RENDER_VULKAN_SURFACE_HH
#define JETSTREAM_RENDER_VULKAN_SURFACE_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API SurfaceImp<DeviceType::Vulkan> : public Surface {
 public:
    explicit SurfaceImp(const Config& config);

    Result create() override;
    Result destroy() override;

    const Extent2D<U64>& size(const Extent2D<U64>& size) override;

 protected:
    Result reorderPrograms(const std::vector<std::shared_ptr<Program>>& ordered) override;
    bool appendable() const override;
    Result validateAppend(const std::vector<std::shared_ptr<Program>>& programs,
                          const std::vector<std::shared_ptr<Kernel>>& kernels) const override;
    Result createAppended(const std::vector<std::shared_ptr<Program>>& programs,
                          const std::vector<std::shared_ptr<Kernel>>& kernels) override;
    Result prepare();
    Result encode(VkCommandBuffer& commandBuffer, bool visible = true);
    void commit();

 private:
    std::shared_ptr<TextureImp<DeviceType::Vulkan>> framebufferResolve;
    std::shared_ptr<TextureImp<DeviceType::Vulkan>> framebuffer;
    VkFramebuffer framebufferObject = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    Extent2D<U64> requestedSize;
    bool framebufferChanged = false;

    std::vector<std::shared_ptr<ProgramImp<DeviceType::Vulkan>>> programs;
    std::vector<std::shared_ptr<KernelImp<DeviceType::Vulkan>>> kernels;
    std::vector<std::shared_ptr<BufferImp<DeviceType::Vulkan>>> buffers;

    friend class WindowImp<DeviceType::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
