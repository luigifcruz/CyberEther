#ifndef JETSTREAM_RENDER_VULKAN_WINDOW_HH
#define JETSTREAM_RENDER_VULKAN_WINDOW_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/render/devices/vulkan/transfer.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/base.hh"

#include <memory>

namespace Jetstream::Render {

template<>
class JETSTREAM_API WindowImp<DeviceType::Vulkan> : public Window {
 public:
    explicit WindowImp(const Config& config,
                       const std::shared_ptr<Viewport::Adapter<DeviceType::Vulkan>>& viewport);

    ~WindowImp() override;

    bool supportsCapture() const override {
        return true;
    }

    std::string info() const override;

    constexpr DeviceType device() const override {
        return DeviceType::Vulkan;
    };

    bool supportsEraseBlending() const override {
        return true;
    }

 protected:
    Result bindSurface(const std::shared_ptr<Surface>& surface) override;
    Result unbindSurface(const std::shared_ptr<Surface>& surface) override;

    Result underlyingCreate() override;
    Result underlyingDestroy() override;

    Result underlyingBegin() override;
    Result underlyingEnd() override;
    Result underlyingCancel() override;

    Result underlyingSynchronize() override;
    Result underlyingRecreate() override;

    Viewport::Generic* underlyingViewport() const override { return viewport.get(); }

 private:
    struct Presenter;

    size_t currentFrame = 0;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    TransferImp<DeviceType::Vulkan> transferEncoder;
    std::vector<std::shared_ptr<SurfaceImp<DeviceType::Vulkan>>> surfaces;
    bool commandBuffersAllocated = false;
    std::unique_ptr<Presenter> presenter;
    bool windowCreated = false;

    std::shared_ptr<Viewport::Adapter<DeviceType::Vulkan>> viewport;

    Result createPresenter();
    Result encodePresent();
    Result createFramebuffer();
    Result destroyFramebuffer();
    Result createSynchronizationObjects();
    Result destroySynchronizationObjects();
};

}  // namespace Jetstream::Render

#endif
