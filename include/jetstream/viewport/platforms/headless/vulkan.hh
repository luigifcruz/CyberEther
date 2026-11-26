#ifndef JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_VULKAN_HH
#define JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_VULKAN_HH

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

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

    Extent2D<F32> displaySize() const;
    Extent2D<F32> framebufferSize() const;

    Result createSwapchain();
    Result destroySwapchain();

    Result waitEvents();
    Result pollEvents();
    bool keepRunning();
    bool retainedPresent() const;
    Result resize(const Extent2D<U64>& size, F32 scale);
    Result waitForPresent();
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
    VkExtent2D extent = {};
    F32 scale = 1.0f;
    VkExtent2D requestedExtent = {};
    F32 requestedScale = 1.0f;
    std::atomic<bool> resizeRequested{false};
    mutable std::mutex extentMutex;

    std::array<VkImage, kImageCount> images = {};
    std::array<VkDeviceMemory, kImageCount> imageMemory = {};
    std::array<VkImageView, kImageCount> imageViews = {};

    std::atomic<bool> running{true};
    std::chrono::steady_clock::time_point lastTime;
};

}  // namespace Jetstream::Viewport

#endif
