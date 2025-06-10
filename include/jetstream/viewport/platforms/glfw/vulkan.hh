#ifndef JETSTREAM_VIEWPORT_PLATFORM_GLFW_VULKAN_HH
#define JETSTREAM_VIEWPORT_PLATFORM_GLFW_VULKAN_HH

#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/viewport/platforms/glfw/generic.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

template<>
class GLFW<Device::Vulkan> : public Adapter<Device::Vulkan> {
 public:
    explicit GLFW(const Config& config);
    virtual ~GLFW();

    std::string id() const {
        return "glfw";
    }

    std::string name() const {
        return "GLFW (Vulkan)";
    }

    constexpr Device device() const {
        return Device::Vulkan;
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
    U32 getSwapchainImageViewsCount() const;
    const VkExtent2D& getSwapchainExtent() const;

 private:
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    GLFWwindow* window = nullptr;
    bool framebufferDidResize = false;
    U32 _currentDrawableIndex;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;

    SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice& device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
};

}  // namespace Jetstream::Viewport

#endif
