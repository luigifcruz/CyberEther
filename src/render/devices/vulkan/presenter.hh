#ifndef JETSTREAM_RENDER_VULKAN_PRESENTER_HH
#define JETSTREAM_RENDER_VULKAN_PRESENTER_HH

#include "jetstream/render/base/present.hh"
#include "jetstream/render/devices/vulkan/window.hh"

#include <vector>

namespace Jetstream::Render {

struct WindowImp<DeviceType::Vulkan>::Presenter {
    Result create(VkRenderPass renderPass, U64 framesInFlight);
    void destroy();
    Result encode(VkCommandBuffer commandBuffer,
                  U64 frameIndex,
                  const PresentList& list,
                  const WindowMetrics& metrics,
                  const Extent2D<U64>& target);

 private:
    struct Frame {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        U64 capacity = 0;
    };

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline straight = VK_NULL_HANDLE;
    VkPipeline premultiplied = VK_NULL_HANDLE;
    std::vector<Frame> frames;

    Result reserve(Frame& frame, U64 vertices);
    void release(Frame& frame);
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_VULKAN_PRESENTER_HH
