#include <csignal>

#include "jetstream/viewport/platforms/headless/vulkan.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#include "imgui.h"

static bool keepRunningFlag;

namespace Jetstream::Viewport {

using Implementation = Headless<DeviceType::Vulkan>;

Implementation::Headless(const Config& config) : Adapter(config) {
    JST_DEBUG("[HEADLESS-VULKAN] Creating headless viewport.");
}

Implementation::~Headless() {
    JST_DEBUG("[HEADLESS-VULKAN] Destroying headless viewport.");
}

Result Implementation::create() {
    // Register signal handler.

    keepRunningFlag = true;
    std::signal(SIGINT, [](int){
        if (!keepRunningFlag) {
            JST_DEBUG("[HEADLESS-VULKAN] Exiting via SIGINT...");
            std::exit(0);
        }
        keepRunningFlag = false;
    });

    // Set extent from config.

    extent.width = static_cast<U32>(config.size.x);
    extent.height = static_cast<U32>(config.size.y);

    // Initialize timing.

    lastTime = std::chrono::steady_clock::now();

    // Create off-screen images.

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<DeviceType::Vulkan>()->getPhysicalDevice();

    imageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    for (U32 i = 0; i < kImageCount; i++) {
        // Create image.
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = imageFormat;
        imageInfo.extent.width = extent.width;
        imageInfo.extent.height = extent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        JST_VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &images[i]), [&]{
            JST_ERROR("[HEADLESS-VULKAN] Failed to create image.");
        });

        // Allocate memory.

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, images[i], &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = Backend::FindMemoryType(
            physicalDevice,
            memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        JST_VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory[i]), [&]{
            JST_ERROR("[HEADLESS-VULKAN] Failed to allocate image memory.");
        });

        JST_VK_CHECK(vkBindImageMemory(device, images[i], imageMemory[i], 0), [&]{
            JST_ERROR("[HEADLESS-VULKAN] Failed to bind image memory.");
        });

        // Create image view.

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = imageFormat;
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        JST_VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &imageViews[i]), [&]{
            JST_ERROR("[HEADLESS-VULKAN] Failed to create image view.");
        });
    }

    _currentDrawableIndex = 0;

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    for (U32 i = 0; i < kImageCount; i++) {
        if (imageViews[i]) {
            vkDestroyImageView(device, imageViews[i], nullptr);
        }
        if (images[i]) {
            vkDestroyImage(device, images[i], nullptr);
        }
        if (imageMemory[i]) {
            vkFreeMemory(device, imageMemory[i], nullptr);
        }
    }

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    return Result::SUCCESS;
}

F32 Implementation::scale(const F32& scale) const {
    return scale;
}

Result Implementation::destroyImgui() {
    return Result::SUCCESS;
}

Result Implementation::nextDrawable(VkSemaphore& semaphore) {
    // Cycle through images.

    _currentDrawableIndex = (_currentDrawableIndex + 1) % kImageCount;

    // Ensure that we don't run too fast.

    auto currentTime = std::chrono::steady_clock::now();
    auto deltaTime = std::chrono::duration<F64>(currentTime - lastTime).count();
    const F64 targetDeltaTime = 1.0 / config.framerate;

    if (deltaTime < targetDeltaTime) {
        auto sleepTime = std::chrono::duration<F64>(targetDeltaTime - deltaTime);
        auto endSleepTime = currentTime + std::chrono::duration_cast<std::chrono::steady_clock::duration>(sleepTime);

        while (std::chrono::steady_clock::now() < endSleepTime) {
            std::this_thread::yield();
        }

        currentTime = std::chrono::steady_clock::now();
        deltaTime = std::chrono::duration<F64>(currentTime - lastTime).count();
    }

    lastTime = currentTime;

    // Set ImGui state for headless.

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    io.DeltaTime = deltaTime;

    // Dummy signal semaphore.

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 0;
    submitInfo.pCommandBuffers = nullptr;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semaphore;

    auto& graphicsQueue = Backend::State<DeviceType::Vulkan>()->getGraphicsQueue();
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr);

    return Result::SUCCESS;
}

Result Implementation::commitDrawable(std::vector<VkSemaphore>&) {
    return Result::SUCCESS;
}

const VkFormat& Implementation::getSwapchainImageFormat() const {
    return imageFormat;
}

VkImageView& Implementation::getSwapchainImageView(const U64& index) {
    return imageViews[index];
}

VkImage Implementation::getSwapchainImage(const U64& index) {
    return images[index];
}

U32 Implementation::getSwapchainImageViewsCount() const {
    return kImageCount;
}

Extent2D<U64> Implementation::getSwapchainExtent() const {
    return {extent.width, extent.height};
}

Result Implementation::waitEvents() {
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return keepRunningFlag && running.load();
}

Result Implementation::createSwapchain() {
    return Result::SUCCESS;
}

Result Implementation::destroySwapchain() {
    return Result::SUCCESS;
}

}  // namespace Jetstream::Viewport
