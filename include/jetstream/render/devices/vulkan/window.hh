#ifndef JETSTREAM_RENDER_VULKAN_WINDOW_HH
#define JETSTREAM_RENDER_VULKAN_WINDOW_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/base.hh"

namespace Jetstream::Render {

template<>
class WindowImp<Device::Vulkan> : public Window {
 public:
    explicit WindowImp(const Config& config,
                       std::shared_ptr<Viewport::Adapter<Device::Vulkan>>& viewport);

    const Stats& stats() const override;
    void drawDebugMessage() const override;

    constexpr Device device() const override {
        return Device::Vulkan;
    };

 protected:
    Result bindBuffer(const std::shared_ptr<Buffer>& buffer) override;
    Result unbindBuffer(const std::shared_ptr<Buffer>& buffer) override;

    Result bindTexture(const std::shared_ptr<Texture>& texture) override;
    Result unbindTexture(const std::shared_ptr<Texture>& texture) override;

    Result bindSurface(const std::shared_ptr<Surface>& surface) override;
    Result unbindSurface(const std::shared_ptr<Surface>& surface) override;

    Result underlyingCreate() override;
    Result underlyingDestroy() override;

    Result underlyingBegin() override;
    Result underlyingEnd() override;

    Result underlyingSynchronize() override;

 private:
    Stats statsData;
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    VkRenderPass renderPass;
    VkCommandPool commandPool;
    VkCommandBuffer currentCommandBuffer;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;

    template<typename T>
    Result bindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container);

    template<typename T>
    Result unbindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container);

    std::vector<std::shared_ptr<BufferImp<Device::Vulkan>>> buffers;
    std::vector<std::shared_ptr<TextureImp<Device::Vulkan>>> textures;
    std::vector<std::shared_ptr<SurfaceImp<Device::Vulkan>>> surfaces;

    std::shared_ptr<Viewport::Adapter<Device::Vulkan>> viewport;

    Result createImgui();
    Result destroyImgui();
    Result createFramebuffer();
    Result destroyFramebuffer();
    Result beginImgui();
    Result endImgui();
    Result recreate();
    Result createSynchronizationObjects();
    Result destroySynchronizationObjects();
};

}  // namespace Jetstream::Render

#endif
