#ifndef JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_VULKAN_HH
#define JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_VULKAN_HH

#include <chrono>

#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/viewport/platforms/headless/generic.hh"

namespace Jetstream::Viewport {

template<>
class Headless<Device::Vulkan> : public Adapter<Device::Vulkan> {
 public:
    explicit Headless(const Config& config);
    virtual ~Headless();

    std::string_view prettyName() const {
        return "Headless (Vulkan)";
    }

    std::string_view name() const {
        return "headless";
    }

    constexpr Device device() const {
        return Device::Vulkan;
    }

    constexpr const U32& currentDrawableIndex() const {
        return _currentDrawableIndex;
    }

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();
    F32 scale(const F32& scale) const;
    
    Result createSwapchain();
    Result destroySwapchain();

    Result pollEvents();
    bool keepRunning();
    Result nextDrawable(VkSemaphore& semaphore);
    Result commitDrawable(std::vector<VkSemaphore>& semaphores);

    const VkFormat& getSwapchainImageFormat() const;
    std::vector<VkImageView>& getSwapchainImageViews();
    const VkExtent2D& getSwapchainExtent() const;

 private:
    VkImage swapchainImage;
    std::vector<VkImageView> swapchainImageViews;
    VkDeviceMemory swapchainMemory;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;
    U32 _currentDrawableIndex;
    Endpoint endpoint;
    std::chrono::high_resolution_clock::time_point lastTime;
};

}  // namespace Jetstream::Viewport

#endif
