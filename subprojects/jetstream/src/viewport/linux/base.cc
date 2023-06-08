#include "jetstream/viewport/providers/linux.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Viewport {

Linux::Linux(const Config& config) : Provider(config) {
    JST_DEBUG("Creating Linux viewport.");
}

Linux::~Linux() {
    JST_DEBUG("Destroying Linux viewport.");
}

Result Linux::create() {
    if (!glfwInit()) {
        return Result::ERROR;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height, 
        config.title.c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        return Result::ERROR;
    }

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    auto& instance = Backend::State<Device::Vulkan>()->getInstance();
    JST_VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface), [&]{
        JST_FATAL("[VULKAN] GLFW failed to create window surface.");     
    });

    JST_CHECK(createSwapchain());

    return Result::SUCCESS;
}

Result Linux::destroy() {
    JST_CHECK(destroySwapchain());

    auto& instance = Backend::State<Device::Vulkan>()->getInstance();
    vkDestroySurfaceKHR(instance, surface, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();

    return Result::SUCCESS;
}

Result Linux::createSwapchain() {
    JST_DEBUG("Creating swapchain.");

    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Create swapchain.

    auto swapchainSupport = querySwapChainSupport(physicalDevice);
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapchainSupport.capabilities);

    U32 imageCount = swapchainSupport.capabilities.minImageCount + 1;
    if (swapchainSupport.capabilities.maxImageCount > 0 && imageCount > swapchainSupport.capabilities.maxImageCount) {
        imageCount = swapchainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicFamily.value(), indices.presentFamily.value()};

    if (indices.graphicFamily != indices.presentFamily ||
        indices.graphicFamily != indices.computeFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    JST_VK_CHECK(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain), [&]{
        JST_FATAL("[VULKAN] Can't create swapchain.");
    });

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent = extent;

    // Create image views.

    swapchainImageViews.resize(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapchainImageFormat;

        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        JST_VK_CHECK(vkCreateImageView(device, &createInfo, NULL, &swapchainImageViews[i]), [&]{
            JST_FATAL("[VULKAN] Failed to create image view."); 
        });
    }

    return Result::SUCCESS;
}

Result Linux::destroySwapchain() {
    JST_DEBUG("Destroying swapchain.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    for (auto& swapchainImageView : swapchainImageViews) {
        vkDestroyImageView(device, swapchainImageView, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);   

    return Result::SUCCESS;
}

Result Linux::createImgui() {
    ImGui_ImplGlfw_InitForVulkan(window, true);

    return Result::SUCCESS;
}

Result Linux::destroyImgui() {
    ImGui_ImplGlfw_Shutdown();

    return Result::SUCCESS;
}

Result Linux::nextDrawable(VkSemaphore& semaphore) {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    const VkResult result = vkAcquireNextImageKHR(device,
                                                  swapchain,
                                                  UINT64_MAX,
                                                  semaphore,
                                                  VK_NULL_HANDLE,
                                                  &_currentDrawableIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return Result::RECREATE;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        return Result::ERROR;
    }

    ImGui_ImplGlfw_NewFrame();

    return Result::SUCCESS;
}

Result Linux::commitDrawable(std::vector<VkSemaphore>& semaphores) {
    VkSwapchainKHR swapchains[] = {swapchain};

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = semaphores.data();
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &_currentDrawableIndex;

    auto& presentQueue = Backend::State<Device::Vulkan>()->getPresentQueue();
    const VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR ||
        result == VK_SUBOPTIMAL_KHR || 
        framebufferDidResize) {
        framebufferDidResize = false;
        return Result::RECREATE;
    } else if (result != VK_SUCCESS) {
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Linux::SwapChainSupportDetails Linux::querySwapChainSupport(const VkPhysicalDevice& device) {
    U32 formatCount;
    U32 presentModeCount;
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR Linux::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto &availableFormat : availableFormats) {
        if (availableFormat.format     == VK_FORMAT_B8G8R8A8_UNORM &&
            availableFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR Linux::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    if (config.vsync) {
        for (const auto &availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                JST_DEBUG("[VULKAN] Swap mailbox presentation mode is available.");
                return availablePresentMode;
            }
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Linux::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<U32>(width),
            static_cast<U32>(height)
        };

        actualExtent.width = std::max(capabilities.minImageExtent.width,
                                      std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height,
                                       std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }        
}

void Linux::framebufferResizeCallback(GLFWwindow *window, int width, int height) {
    reinterpret_cast<Linux*>(glfwGetWindowUserPointer(window))->framebufferDidResize = true;
}

const VkFormat& Linux::getSwapchainImageFormat() const {
    return swapchainImageFormat;
}

std::vector<VkImageView>& Linux::getSwapchainImageViews() {
    return swapchainImageViews;
}

const VkExtent2D& Linux::getSwapchainExtent() const {
    return swapchainExtent;       
}

Result Linux::pollEvents() {
    glfwWaitEvents();

    return Result::SUCCESS;
}

bool Linux::keepRunning() {
    return !glfwWindowShouldClose(window);
}

}  // namespace Jetstream::Viewport 
