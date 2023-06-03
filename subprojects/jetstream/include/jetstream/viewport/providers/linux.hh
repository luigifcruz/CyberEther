#ifndef JETSTREAM_VIEWPORT_LINUX_HH
#define JETSTREAM_VIEWPORT_LINUX_HH

#include "jetstream/viewport/devices/vulkan.hh"
#include "jetstream/render/tools/imgui_impl_glfw.h"
#include "jetstream/backend/base.hh"

#include <GLFW/glfw3.h>

namespace Jetstream::Viewport {

class Linux : public Provider<Device::Vulkan> {
 public:
    explicit Linux(const Config& config);
    virtual ~Linux();

    const std::string name() const {
        return "Linux (GLFW)";
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
    
    Result pollEvents();
    bool keepRunning();
    Result nextDrawable(VkSemaphore& semaphore);
    Result commitDrawable(std::vector<VkSemaphore>& semaphores);

    static std::shared_ptr<Linux> Factory(const Config& config) {
        return std::make_shared<Linux>(config);
    }

    // TODO: Maybe protect those?
    const VkFormat& getSwapchainImageFormat() const;
    std::vector<VkImageView>& getSwapchainImageViews();
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

    Result createSwapchain();
    Result createImageView();
    SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice& device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
};

}  // namespace Jetstream::Viewport

#endif
