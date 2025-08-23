#ifndef JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_VULKAN_HH
#define JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_VULKAN_HH

#include <chrono>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>

#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/viewport/platforms/headless/generic.hh"

namespace Jetstream::Viewport {

template<>
class Headless<Device::Vulkan> : public Adapter<Device::Vulkan> {
 public:
    explicit Headless(const Config& config);
    virtual ~Headless();

    std::string id() const {
        return "headless";
    }

    std::string name() const {
        return "Headless (Vulkan)";
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

    Result waitEvents();
    Result pollEvents();
    bool keepRunning();
    Result nextDrawable(VkSemaphore& semaphore);
    Result commitDrawable(std::vector<VkSemaphore>& semaphores);

    const VkFormat& getSwapchainImageFormat() const;
    VkImageView& getSwapchainImageView(const U64& index);
    U32 getSwapchainImageViewsCount() const;
    const VkExtent2D& getSwapchainExtent() const;

 private:
    const static U32 MAX_FRAMES_IN_FLIGHT = 2;

    std::array<Tensor<Device::Vulkan, U8>, MAX_FRAMES_IN_FLIGHT> stagingBuffers;

    std::chrono::steady_clock::time_point lastTime;
    std::array<VkImage, MAX_FRAMES_IN_FLIGHT> swapchainImages;
    std::array<VkImageView, MAX_FRAMES_IN_FLIGHT> swapchainImageViews;
    std::array<VkDeviceMemory, MAX_FRAMES_IN_FLIGHT> swapchainMemory;
    std::array<void*, MAX_FRAMES_IN_FLIGHT> swapchainMemoryMapped;
    std::array<std::atomic_flag, MAX_FRAMES_IN_FLIGHT> swapchainEvents;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> swapchainFences;
    VkCommandPool swapchainCommandPool;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> swapchainCommandBuffers;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;
    U32 _currentDrawableIndex;

    Remote remote;
    std::queue<U64> endpointFrameSubmissionQueue;
    std::mutex endpointFrameSubmissionMutex;
    std::condition_variable endpointFrameSubmissionCondition;
    std::thread endpointFrameSubmissionThread;
    bool endpointFrameSubmissionRunning;
    std::optional<Result> endpointFrameSubmissionResult;

    void endpointFrameSubmissionLoop();
};

}  // namespace Jetstream::Viewport

#endif
