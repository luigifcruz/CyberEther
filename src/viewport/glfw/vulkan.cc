#include <csignal>

#include "jetstream/viewport/platforms/glfw/vulkan.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#include <GLFW/glfw3.h>

#include "tools/imgui_impl_glfw.h"

static void PrintGLFWError(int, const char* description) {
    JST_FATAL("[VULKAN] GLFW error: {}", description);
}

static bool keepRunningFlag;

namespace Jetstream::Viewport {

using Implementation = GLFW<Device::Vulkan>;

Implementation::GLFW(const Config& config) : Adapter(config) {
    JST_DEBUG("[VULKAN] Creating GLFW viewport.");
}

Implementation::~GLFW() {
    JST_DEBUG("[VULKAN] Destroying GLFW viewport.");
}

Result Implementation::create() {
    // Register signal handler.

    keepRunningFlag = true;
    std::signal(SIGINT, [](int){
        if (!keepRunningFlag) {
            JST_DEBUG("[METAL] Exiting via SIGINT...");
            std::exit(0);
        }
        keepRunningFlag = false;
    });

    // Check if we are running in headless mode.
    JST_ASSERT(!Backend::State<Device::Vulkan>()->headless(), "Headless mode is not supported.");

    // Initialize and configure GLFW.

#ifdef GLFW_PLATFORM_WAYLAND
    if (Backend::WindowMightBeWayland()) {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    }
#endif

    if (!glfwInit()) {
        JST_ERROR("[VULKAN] Failed to initialize GLFW.");
        return Result::ERROR;
    }

    glfwSetErrorCallback(&PrintGLFWError);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height,
        config.title.c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        JST_ERROR("[VULKAN] Failed to create window with GLFW.");
        return Result::ERROR;
    }

    // Initialize variables.
    swapchain = VK_NULL_HANDLE;

    // Create surface.

    auto& instance = Backend::State<Device::Vulkan>()->getInstance();
    JST_VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface), [&]{
        JST_ERROR("[VULKAN] GLFW failed to create window surface.");
    });

    // Create swapchain.
    JST_CHECK(createSwapchain());

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& instance = Backend::State<Device::Vulkan>()->getInstance();

    // Destroy swapchain.

    for (auto& swapchainImageView : swapchainImageViews) {
        vkDestroyImageView(device, swapchainImageView, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    // Destroy surface.

    vkDestroySurfaceKHR(instance, surface, nullptr);

    // Destroy GLFW.

    glfwDestroyWindow(window);
    glfwTerminate();

    return Result::SUCCESS;
}

Result Implementation::createSwapchain() {
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Save current swapchain.

    VkSwapchainKHR oldSwapchain = swapchain;
    swapchain = VK_NULL_HANDLE;

    // Create swapchain.

    auto swapchainSupport = querySwapChainSupport(physicalDevice);
    if (swapchainSupport.formats.empty()) {
        JST_ERROR("[VULKAN] No supported swapchain formats.");
        return Result::ERROR;
    }
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
    createInfo.oldSwapchain = oldSwapchain;

    JST_VK_CHECK(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain), [&]{
        JST_ERROR("[VULKAN] Can't create swapchain.");
    });

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent = extent;
    _currentDrawableIndex = 0;

    // Destroy old swapchain.

    if (oldSwapchain) {
        vkDestroySwapchainKHR(device, oldSwapchain, nullptr);

        for (auto& swapchainImageView : swapchainImageViews) {
            vkDestroyImageView(device, swapchainImageView, nullptr);
        }
    }

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

        JST_VK_CHECK(vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]), [&]{
            JST_ERROR("[VULKAN] Failed to create image view.");
        });
    }

    return Result::SUCCESS;
}

Result Implementation::destroySwapchain() {
    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    ImGui_ImplGlfw_InitForVulkan(window, true);

    return Result::SUCCESS;
}

F32 Implementation::scale(const F32& scale) const {
    I32 w_width, w_height;
    glfwGetWindowSize(window, &w_width, &w_height);

    I32 f_width, f_height;
    glfwGetFramebufferSize(window, &f_width, &f_height);

    F32 x, y;
    glfwGetWindowContentScale(window, &x, &y);

    // This is a X11/Windows fix. If the Framebuffer and Window sizes are the same
    // but the Scale is different than 1, it means HIDPI is necessary.
    if ((w_width == f_width) and (x > 1.0)) {
        return scale * x;
    } else {
        return scale;
    }
}

Result Implementation::destroyImgui() {
    ImGui_ImplGlfw_Shutdown();

    return Result::SUCCESS;
}

Result Implementation::nextDrawable(VkSemaphore& semaphore) {
    // Check if framebuffer size is different from swapchain extent.

    int fb_width, fb_height;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);

    int sf_width = static_cast<int>(swapchainExtent.width);
    int sf_height = static_cast<int>(swapchainExtent.height);
    if (fb_width > 0 && fb_height > 0 && (fb_width != sf_width || fb_height != sf_height)) {
        return Result::RECREATE;
    }

    // Get next drawable.

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    const VkResult result = vkAcquireNextImageKHR(device,
                                                  swapchain,
                                                  UINT64_MAX,
                                                  semaphore,
                                                  VK_NULL_HANDLE,
                                                  &_currentDrawableIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR ||
        result == VK_ERROR_SURFACE_LOST_KHR ||
        result == VK_SUBOPTIMAL_KHR) {
        return Result::RECREATE;
    }

    if (result != VK_SUCCESS) {
        return Result::ERROR;
    }

    ImGui_ImplGlfw_NewFrame();

    return Result::SUCCESS;
}

Result Implementation::commitDrawable(std::vector<VkSemaphore>& semaphores) {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.pWaitSemaphores = semaphores.data();
    presentInfo.waitSemaphoreCount = semaphores.size();

    VkSwapchainKHR swapchains[] = {swapchain};
    presentInfo.pSwapchains = swapchains;
    presentInfo.swapchainCount = 1;

    presentInfo.pImageIndices = &_currentDrawableIndex;

    VkResult result = VK_SUCCESS;
    auto& presentQueue = Backend::State<Device::Vulkan>()->getPresentQueue();

    {
        std::scoped_lock lock(frameScopeMutex);
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
    }

    if (result == VK_ERROR_OUT_OF_DATE_KHR ||
        result == VK_ERROR_SURFACE_LOST_KHR ||
        result == VK_SUBOPTIMAL_KHR) {
        return Result::RECREATE;
    }

    if (result != VK_SUCCESS) {
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Implementation::SwapChainSupportDetails Implementation::querySwapChainSupport(const VkPhysicalDevice& device) {
    U32 formatCount;
    U32 presentModeCount;
    SwapChainSupportDetails details;

    JST_VK_CHECK_THROW(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities), [&]{
        JST_FATAL("[VULKAN] Failed to get surface capabilities.")
    });
    JST_VK_CHECK_THROW(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr), [&]{
        JST_FATAL("[VULKAN] Failed to get surface formats.")
    });
    JST_VK_CHECK_THROW(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr), [&]{
        JST_FATAL("[VULKAN] Failed to get surface present modes.")
    });

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        JST_VK_CHECK_THROW(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data()), [&]{
            JST_FATAL("[VULKAN] Failed to get surface formats.")
        });
    }

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        JST_VK_CHECK_THROW(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data()), [&]{
            JST_FATAL("[VULKAN] Failed to get surface present modes.")
        });
    }

    return details;
}

VkSurfaceFormatKHR Implementation::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto &availableFormat : availableFormats) {
        if (availableFormat.format     == VK_FORMAT_B8G8R8A8_UNORM &&
            availableFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR Implementation::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    if (config.vsync) {
        // TODO: Re-evaluate if we want MAILBOX.
#ifndef JST_OS_WINDOWS
        for (const auto &availablePresentMode : availablePresentModes) {
            // HACK: Mailbox is not currently supported on Wayland.
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR &&!Backend::WindowMightBeWayland()) {
                JST_DEBUG("[VULKAN] Swap mailbox presentation mode is available.");
                return availablePresentMode;
            }
        }
#endif
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
}

VkExtent2D Implementation::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        I32 width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<U32>(width),
            static_cast<U32>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width,
                                        capabilities.minImageExtent.width,
                                        capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height,
                                         capabilities.minImageExtent.height,
                                         capabilities.maxImageExtent.height);

        return actualExtent;
}

const VkFormat& Implementation::getSwapchainImageFormat() const {
    return swapchainImageFormat;
}

VkImageView& Implementation::getSwapchainImageView(const U64& index) {
    return swapchainImageViews[index];
}

U32 Implementation::getSwapchainImageViewsCount() const {
    return swapchainImageViews.size();
}

const VkExtent2D& Implementation::getSwapchainExtent() const {
    return swapchainExtent;
}

Result Implementation::waitEvents() {
    {
        std::scoped_lock lock(frameScopeMutex);
        glfwPollEvents();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    {
        std::scoped_lock lock(frameScopeMutex);
        glfwPollEvents();
    }
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return (!glfwWindowShouldClose(window)) && keepRunningFlag;
}

}  // namespace Jetstream::Viewport
