#ifndef JETSTREAM_VIEWPORT_LINUX_HH
#define JETSTREAM_VIEWPORT_LINUX_HH

#include "jetstream/viewport/generic.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/tools/imgui_impl_glfw.h"

#include <GLFW/glfw3.h>

namespace Jetstream::Viewport {

class Linux : public Generic {
 public:
    explicit Linux(const Config& config);
    virtual ~Linux();

    const std::string name() const {
        return "Linux (GLFW)";
    }

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();

    void* nextDrawable();
    
    Result pollEvents();
    bool keepRunning();

    static std::shared_ptr<Linux> Factory(const Config& config) {
        return std::make_shared<Linux>(config);
    }

 private:
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    GLFWwindow* window = nullptr;
    bool framebufferDidResize = false;
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

}

#endif
