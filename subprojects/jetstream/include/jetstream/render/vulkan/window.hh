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
    // CA::MetalDrawable* drawable = nullptr;
    // MTL::CommandQueue* commandQueue = nullptr;
    // MTL::CommandBuffer* commandBuffer = nullptr;
    // MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;
    std::vector<std::shared_ptr<SurfaceImp<Device::Vulkan>>> surfaces;
    std::shared_ptr<Viewport::Provider<Device::Vulkan>> viewport;

    VkRenderPass renderPass;
    VkCommandPool commandPool;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    std::vector<VkCommandBuffer> commandBuffers;

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

}  // namespace Jetstream::Render

#endif
