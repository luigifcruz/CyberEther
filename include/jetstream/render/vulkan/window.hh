#ifndef JETSTREAM_RENDER_VULKAN_WINDOW_HH
#define JETSTREAM_RENDER_VULKAN_WINDOW_HH

#include "jetstream/render/tools/imgui_impl_vulkan.h"

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

    Result create() override;
    Result destroy() override;

    Result begin() override;
    Result end() override;

    const Stats& stats() const override;
    void drawDebugMessage() const override;

    constexpr Device device() const override {
        return Device::Vulkan;
    };

    Result bind(const std::shared_ptr<Surface>& surface) override;
    Result unbind(const std::shared_ptr<Surface>& surface) override;

 private:
    Stats statsData;
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    VkRenderPass renderPass;
    VkCommandPool commandPool;
    VkCommandBufferBeginInfo commandBufferBeginInfo;
    VkCommandBuffer currentCommandBuffer;
    VkRenderPassBeginInfo renderPassBeginInfo;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;

    std::vector<std::shared_ptr<SurfaceImp<Device::Vulkan>>> surfaces;
    std::shared_ptr<Viewport::Adapter<Device::Vulkan>> viewport;

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
    Result recreate();
    Result createSynchronizationObjects();
    Result destroySynchronizationObjects();
};

}  // namespace Jetstream::Render

#endif
