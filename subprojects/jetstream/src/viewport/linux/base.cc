#include "jetstream/viewport/linux.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Viewport {

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
        if (availableFormat.format     == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR Linux::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
    for (const auto &availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            JST_DEBUG("[VULKAN] Swap mailbox presentation mode is available.");
            return availablePresentMode;
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
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }        
}

Result Linux::createSwapchain() {
    JST_DEBUG("[VULKAN] Creating swapchain.");

    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

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

    // TODO: Might give problems in the future.
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

    return Result::SUCCESS;
}

void Linux::framebufferResizeCallback(GLFWwindow *window, int width, int height) {
    reinterpret_cast<Linux*>(glfwGetWindowUserPointer(window))->framebufferDidResize = true;
}

Linux::Linux(const Config& config) : Generic(config) {
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
    glfwWindowHint(GLFW_DOUBLEBUFFER, config.vsync);

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
    glfwDestroyWindow(window);
    glfwTerminate();

    return Result::SUCCESS;
}

Result Linux::createImgui() {
    ImGui_ImplGlfw_InitForOther(window, true);

    return Result::SUCCESS;
}

Result Linux::destroyImgui() {
    ImGui_ImplGlfw_Shutdown();

    return Result::SUCCESS;
}

void* Linux::nextDrawable() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    ImGui_ImplGlfw_NewFrame();

    return nullptr;
}

Result Linux::pollEvents() {
    glfwWaitEvents();

    return Result::SUCCESS;
}

bool Linux::keepRunning() {
    return !glfwWindowShouldClose(window);
}

}  // namespace Jetstream::Viewport 

