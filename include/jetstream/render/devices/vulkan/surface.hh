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

 protected:
    Result sequenceSteps(const std::vector<Step>& steps) override;
    bool appendable() const override;
    Result validateAppend(const std::vector<std::shared_ptr<Program>>& programs,
                          const std::vector<std::shared_ptr<Kernel>>& kernels) const override;
    Result createAppended(const std::vector<std::shared_ptr<Program>>& programs,
                          const std::vector<std::shared_ptr<Kernel>>& kernels) override;
    bool created() const override;
    Result rebuildFramebuffer() override;
    Result encode(VkCommandBuffer& commandBuffer, bool visible = true);
    void commit();

    bool commandsEncoded = false;

 private:
    std::shared_ptr<TextureImp<DeviceType::Vulkan>> framebufferResolve;
    std::shared_ptr<TextureImp<DeviceType::Vulkan>> framebuffer;
    VkFramebuffer framebufferObject = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;

    struct Ordered {
        std::shared_ptr<ProgramImp<DeviceType::Vulkan>> program;
        std::optional<Draw::Range> range;
        std::optional<ScissorRect> scissor;
    };

    std::vector<std::shared_ptr<ProgramImp<DeviceType::Vulkan>>> programs;
    std::vector<Ordered> ordered;
    std::vector<std::shared_ptr<KernelImp<DeviceType::Vulkan>>> kernels;
    std::vector<std::shared_ptr<BufferImp<DeviceType::Vulkan>>> buffers;

    friend class WindowImp<DeviceType::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
