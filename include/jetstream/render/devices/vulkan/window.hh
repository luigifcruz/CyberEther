#ifndef JETSTREAM_RENDER_VULKAN_WINDOW_HH
#define JETSTREAM_RENDER_VULKAN_WINDOW_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/base.hh"

namespace Jetstream::Render {

template<>
class WindowImp<DeviceType::Vulkan> : public Window {
 public:
    explicit WindowImp(const Config& config,
                       const std::shared_ptr<Viewport::Adapter<DeviceType::Vulkan>>& viewport);

    const Stats& stats() const override;
    std::string info() const override;

    constexpr DeviceType device() const override {
        return DeviceType::Vulkan;
    };

 protected:
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
    size_t currentFrame = 0;
    ImGuiStyle* style = nullptr;
    VkRenderPass renderPass;
    VkCommandPool commandPool;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    std::vector<std::shared_ptr<SurfaceImp<DeviceType::Vulkan>>> surfaces;

    std::shared_ptr<Viewport::Adapter<DeviceType::Vulkan>> viewport;

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
