#include <csignal>

#include "jetstream/viewport/platforms/headless/vulkan.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

static bool keepRunningFlag;

namespace Jetstream::Viewport {

using Implementation = Headless<Device::Vulkan>;

Implementation::Headless(const Config& config) : Adapter(config) {
    JST_DEBUG("[VULKAN] Creating Headless viewport.");
}

Implementation::~Headless() {
    JST_DEBUG("[VULKAN] Destroying Headless viewport.");
}

Result Implementation::create() {
    // Register signal handler.

    keepRunningFlag = true;
    std::signal(SIGINT, [](int){
        if (!keepRunningFlag) {
            exit(0);
        }
        keepRunningFlag = false;
    });

    // Check if we are running in headless mode.
    JST_ASSERT(Backend::State<Device::Vulkan>()->headless());

    // Initialize variables.

    endpointFrameSubmissionRunning = false;
    _currentDrawableIndex = 0;
    swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    lastTime = std::chrono::steady_clock::now();

    // Create endpoint.

    JST_CHECK(endpoint.create(config));
    JST_CHECK(createSwapchain());

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_CHECK(endpoint.destroy());
    JST_CHECK(destroySwapchain());

    return Result::SUCCESS;
}

Result Implementation::createSwapchain() {
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();
    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    const auto& unified = Backend::State<Device::Vulkan>()->hasUnifiedMemory();

    // Create extent.

    swapchainExtent = {
        static_cast<U32>(config.size.width),
        static_cast<U32>(config.size.height)
    };

    // Create image.

    for (U32 i = 0; i < swapchainImages.size(); i++) {
        VkImageCreateInfo imageCreateInfo = {};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.extent.width = config.size.width;
        imageCreateInfo.extent.height = config.size.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.format = swapchainImageFormat;
        imageCreateInfo.tiling = (unified) ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        JST_VK_CHECK(vkCreateImage(device, &imageCreateInfo, nullptr, &swapchainImages[i]), [&]{
            JST_ERROR("[VULKAN] Failed to create swapchain image.");   
        });
    }

    // Allocate backing memory.

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(device, swapchainImages[0], &memoryRequirements);

    VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (unified) {
        memoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        memoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                memoryRequirements.memoryTypeBits,
                                                                memoryProperties);

    for (U32 i = 0; i < swapchainMemory.size(); i++) {
        JST_VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &swapchainMemory[i]), [&]{
            JST_ERROR("[VULKAN] Failed to allocate swapchain image memory.");
        });

        JST_VK_CHECK(vkBindImageMemory(device, swapchainImages[i], swapchainMemory[i], 0), [&]{
            JST_ERROR("[VULKAN] Failed to bind memory to the swapchain image.");
        });

        if (unified) {
            JST_VK_CHECK_THROW(vkMapMemory(device, swapchainMemory[i], 0, memoryRequirements.size, 0, &swapchainMemoryMapped[i]), [&]{
                JST_FATAL("[VULKAN] Failed to map swapchain buffer memory.");
            });
        }
    }

    // Create staging buffer in case of non-unified system.

    for (U32 i = 0; (i < swapchainStagingBuffers.size()) && !unified; i++) {
        VkBufferCreateInfo bufferCreateInfo = {};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = memoryRequirements.size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        JST_VK_CHECK(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &swapchainStagingBuffers[i]), [&]{
            JST_ERROR("[VULKAN] Can't create staging buffer.");
        });

        VkMemoryRequirements stagingMemoryRequirements;
        vkGetBufferMemoryRequirements(device, swapchainStagingBuffers[i], &stagingMemoryRequirements);

        VkMemoryAllocateInfo stagingMemoryAllocateInfo = {};
        stagingMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        stagingMemoryAllocateInfo.allocationSize = stagingMemoryRequirements.size;
        stagingMemoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                                  stagingMemoryRequirements.memoryTypeBits,
                                                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        JST_VK_CHECK(vkAllocateMemory(device, &stagingMemoryAllocateInfo, nullptr, &swapchainStagingMemory[i]), [&]{
            JST_ERROR("[VULKAN] Failed to allocate staging buffer memory.");
        });

        JST_VK_CHECK(vkBindBufferMemory(device, swapchainStagingBuffers[i], swapchainStagingMemory[i], 0), [&]{
            JST_ERROR("[VULKAN] Failed to bind memory to the staging buffer.");
        });

        JST_VK_CHECK_THROW(vkMapMemory(device, swapchainStagingMemory[i], 0, stagingMemoryRequirements.size, 0, &swapchainMemoryMapped[i]), [&]{
            JST_FATAL("[VULKAN] Failed to map staging buffer memory.");
        });
    }

    // Create command pool in case of non-unified system.

    if (!unified) {
        Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.queueFamilyIndex = indices.graphicFamily.value();
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        JST_VK_CHECK(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &swapchainCommandPool), [&]{
            JST_ERROR("[VULKAN] Failed to create swapchain command pool.");
        });
    }

    // Create command buffer in case of non-unified system.

   if (!unified) {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = swapchainCommandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = swapchainCommandBuffers.size();

        JST_VK_CHECK(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, swapchainCommandBuffers.data()), [&]{
            JST_ERROR("[VULKAN] Failed to allocate swapchain command buffers.");
        });
    }

    // Create image view.

    for (U32 i = 0; i < swapchainImageViews.size(); i++) {
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
            JST_ERROR("[VULKAN] Failed to create swapchain image view."); 
        });
    }

    // Create events.

    for (U32 i = 0; i < swapchainEvents.size(); i++) {
        swapchainEvents[i].clear();
    }

    // Create fences.

    for (U32 i = 0; i < swapchainFences.size(); i++) {
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        JST_VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &swapchainFences[i]), [&]{
            JST_ERROR("[VULKAN] Failed to create swapchain fence.");
        });
    }

    // Start frame submission worker.

    endpointFrameSubmissionRunning = true;
    endpointFrameSubmissionThread = std::thread(&Implementation::endpointFrameSubmissionLoop, this);

    return Result::SUCCESS;
}

Result Implementation::destroySwapchain() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    const auto& unified = Backend::State<Device::Vulkan>()->hasUnifiedMemory();

    // Destroy swapchain.

    if (!unified) {
        vkFreeCommandBuffers(device, swapchainCommandPool, swapchainCommandBuffers.size(), swapchainCommandBuffers.data());
        vkDestroyCommandPool(device, swapchainCommandPool, nullptr);
    }

    for (U32 i = 0; i < swapchainImageViews.size(); i++) {
        if (!unified) {
            vkUnmapMemory(device, swapchainStagingMemory[i]);
            vkFreeMemory(device, swapchainStagingMemory[i], nullptr);
            vkDestroyBuffer(device, swapchainStagingBuffers[i], nullptr);
        } else {
            vkUnmapMemory(device, swapchainMemory[i]);
        }
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
        vkDestroyImage(device, swapchainImages[i], nullptr);
        vkFreeMemory(device, swapchainMemory[i], nullptr);
        vkDestroyFence(device, swapchainFences[i], nullptr);
    }

    // Stop frame submission worker.

    endpointFrameSubmissionRunning = false;
    endpointFrameSubmissionCondition.notify_one();
    if (endpointFrameSubmissionThread.joinable()) {
        endpointFrameSubmissionThread.join();
    }

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    // TODO: Implement interaction.

    return Result::SUCCESS;
}

F32 Implementation::scale(const F32& scale) const {
    // TODO: Implement HiDPI support.

    return scale;
}

Result Implementation::destroyImgui() {
    return Result::SUCCESS;
}

Result Implementation::nextDrawable(VkSemaphore& semaphore) {
    swapchainEvents[_currentDrawableIndex].wait(true);
    swapchainEvents[_currentDrawableIndex].test_and_set();

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

    // Update ImGui state.
    
    auto& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(config.size.width, config.size.height);
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

    auto& graphicsQueue = Backend::State<Device::Vulkan>()->getGraphicsQueue();
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr);

    return Result::SUCCESS;
}

Result Implementation::commitDrawable(std::vector<VkSemaphore>& semaphores) {
    const auto& unified = Backend::State<Device::Vulkan>()->hasUnifiedMemory();

    // Copy swapchain image to staging buffer if non-unified.

    if (!unified) {
        VkCommandBufferBeginInfo cmdBeginInfo = {};
        cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        JST_VK_CHECK(vkBeginCommandBuffer(swapchainCommandBuffers[_currentDrawableIndex], &cmdBeginInfo), [&]{
            JST_ERROR("[VULKAN] Failed to begin swapchain framebuffer download command-buffer.");
        });

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent.width = config.size.width;
        region.imageExtent.height = config.size.height;
        region.imageExtent.depth = 1;

        vkCmdCopyImageToBuffer(swapchainCommandBuffers[_currentDrawableIndex],
                               swapchainImages[_currentDrawableIndex],
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               swapchainStagingBuffers[_currentDrawableIndex],
                               1,
                               &region);

        JST_VK_CHECK(vkEndCommandBuffer(swapchainCommandBuffers[_currentDrawableIndex]), [&]{
            JST_ERROR("[VULKAN] Failed to end swapchain framebuffer download command-buffer.");
        });
    }

    // Wait framebuffer to be ready.

    const VkPipelineStageFlags waitStage[] = { VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = semaphores.size();
    submitInfo.pWaitSemaphores = semaphores.data();
    submitInfo.pWaitDstStageMask = waitStage;
    submitInfo.commandBufferCount = (unified) ? 0 : 1;
    submitInfo.pCommandBuffers = (unified) ? nullptr : &swapchainCommandBuffers[_currentDrawableIndex];
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    vkResetFences(device, 1, &swapchainFences[_currentDrawableIndex]);

    auto& graphicsQueue = Backend::State<Device::Vulkan>()->getGraphicsQueue();
    JST_VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, swapchainFences[_currentDrawableIndex]), [&]{
        JST_ERROR("[VULKAN] Can't submit headless queue.");            
    });

    // Submit frame to endpoint.

    {
        std::lock_guard<std::mutex> lock(endpointFrameSubmissionMutex);
        endpointFrameSubmissionQueue.push(_currentDrawableIndex);
        endpointFrameSubmissionCondition.notify_one();
    }

    // Update Viewport state. 

    _currentDrawableIndex = (_currentDrawableIndex + 1) % 2;

    return Result::SUCCESS;
}

void Implementation::endpointFrameSubmissionLoop() {
    while (endpointFrameSubmissionRunning) {
        U64 fenceIndex;

        {
            std::unique_lock<std::mutex> lock(endpointFrameSubmissionMutex);
            endpointFrameSubmissionCondition.wait(lock, [&]{ 
                return !endpointFrameSubmissionQueue.empty() || !endpointFrameSubmissionRunning;
            });

            if (!endpointFrameSubmissionRunning && 
                 endpointFrameSubmissionQueue.empty()) {
                return;
            }

            fenceIndex = endpointFrameSubmissionQueue.front();
            endpointFrameSubmissionQueue.pop();
        }

        auto& device = Backend::State<Device::Vulkan>()->getDevice();
        vkWaitForFences(device, 1, &swapchainFences[fenceIndex], true, UINT64_MAX);

        // TODO: Add check.
        endpoint.newFrameHost(static_cast<uint8_t*>(swapchainMemoryMapped[fenceIndex]));

        swapchainEvents[fenceIndex].clear();
        swapchainEvents[fenceIndex].notify_one();
    }
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

Result Implementation::pollEvents() {
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return keepRunningFlag;
}

}  // namespace Jetstream::Viewport 
