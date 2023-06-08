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
                       std::shared_ptr<Viewport::Provider<Device::Vulkan>>& viewport);

    Result create();
    Result destroy();
    Result begin();
    Result end();

    const Stats& stats() const;
    void drawDebugMessage() const;

    constexpr Device device() const {
        return Device::Vulkan;
    };

    Result bind(const std::shared_ptr<Surface>& surface);

 private:
    Stats statsData;
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    std::vector<std::shared_ptr<SurfaceImp<Device::Vulkan>>> surfaces;
    std::shared_ptr<Viewport::Provider<Device::Vulkan>> viewport;

    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    VkCommandBufferBeginInfo commandBufferBeginInfo;
    VkCommandBuffer currentCommandBuffer;
    VkRenderPassBeginInfo renderPassBeginInfo;
    VkRenderPass renderPass;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    std::vector<VkCommandBuffer> commandBuffers;

    Result createSynchronizationObjects();
    Result destroySynchronizationObjects();
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
    Result recreate();
};

}  // namespace Jetstream::Render

#endif
