#ifndef JETSTREAM_VIEWPORT_DEVICES_VULKAN_HH
#define JETSTREAM_VIEWPORT_DEVICES_VULKAN_HH

#include "jetstream/viewport/generic.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Viewport {

template<>
class Provider<Device::Vulkan> : public Generic {
 public:
    using Generic::Generic;

    virtual Result createSwapchain() = 0;
    virtual Result destroySwapchain() = 0;

    virtual const U32& currentDrawableIndex() const = 0;
    virtual Result nextDrawable(VkSemaphore& semaphore) = 0;
    virtual Result commitDrawable(std::vector<VkSemaphore>& semaphores) = 0;
    
    virtual const VkFormat& getSwapchainImageFormat() const = 0;
    virtual std::vector<VkImageView>& getSwapchainImageViews() = 0;
    virtual const VkExtent2D& getSwapchainExtent() const = 0;
};

}  // namespace Jetstream::Viewport

#endif
