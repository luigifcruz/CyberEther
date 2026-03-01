#ifndef JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_VULKAN_HH
#define JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_VULKAN_HH

#include <atomic>
#include <chrono>

#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/viewport/platforms/headless/generic.hh"

namespace Jetstream::Viewport {

template<>
class Headless<DeviceType::Vulkan> : public Adapter<DeviceType::Vulkan> {
 public:
    explicit Headless(const Config& config);
    virtual ~Headless();

    std::string id() const {
        return "headless";
    }

    std::string name() const {
        return "Headless (Vulkan)";
    }

    constexpr DeviceType device() const {
        return DeviceType::Vulkan;
    };

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

    Result waitEvents();
    Result pollEvents();
    bool keepRunning();
    Result nextDrawable(VkSemaphore& semaphore);
    Result commitDrawable(std::vector<VkSemaphore>& semaphores);

    const VkFormat& getSwapchainImageFormat() const;
    VkImageView& getSwapchainImageView(const U64& index);
    VkImage getSwapchainImage(const U64& index);
    U32 getSwapchainImageViewsCount() const;
    Extent2D<U64> getSwapchainExtent() const;

 private:
    static constexpr U32 kImageCount = 2;

    U32 _currentDrawableIndex = 0;
    VkFormat imageFormat;
    VkExtent2D extent;

    std::array<VkImage, kImageCount> images;
    std::array<VkDeviceMemory, kImageCount> imageMemory;
    std::array<VkImageView, kImageCount> imageViews;

    std::atomic<bool> running{true};
    std::chrono::steady_clock::time_point lastTime;
};

}  // namespace Jetstream::Viewport

#endif
