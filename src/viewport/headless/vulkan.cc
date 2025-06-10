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
            std::exit(0);
        }
        keepRunningFlag = false;
    });

    // Check if we are running in headless mode.
    JST_ASSERT(Backend::State<Device::Vulkan>()->headless(), "Headless mode is not enabled");

    // Initialize variables.

    endpointFrameSubmissionRunning = false;
    _currentDrawableIndex = 0;
    swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    lastTime = std::chrono::steady_clock::now();

    // Create endpoint.

    JST_CHECK(endpoint.create(config, Device::Vulkan));
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

    // Create extent.

    swapchainExtent = {
        static_cast<U32>(config.size.x),
        static_cast<U32>(config.size.y)
    };

    // Create image.

    for (U32 i = 0; i < swapchainImages.size(); i++) {
        VkImageCreateInfo imageCreateInfo = {};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.extent.width = config.size.x;
        imageCreateInfo.extent.height = config.size.y;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.format = swapchainImageFormat;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
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

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                memoryRequirements.memoryTypeBits,
                                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    for (U32 i = 0; i < swapchainMemory.size(); i++) {
        JST_VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &swapchainMemory[i]), [&]{
            JST_ERROR("[VULKAN] Failed to allocate swapchain image memory.");
        });

        JST_VK_CHECK(vkBindImageMemory(device, swapchainImages[i], swapchainMemory[i], 0), [&]{
            JST_ERROR("[VULKAN] Failed to bind memory to the swapchain image.");
        });
    }

    // Create staging buffer.

    const auto& inputMemoryDevice = endpoint.inputMemoryDevice();

    for (U32 i = 0; i < stagingBuffers.size(); i++) {
        stagingBuffers[i] = Tensor<Device::Vulkan, U8>({config.size.x, config.size.y, 4}, inputMemoryDevice == Device::CPU);

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (inputMemoryDevice == Device::CUDA) {
            swapchainMemoryMapped[i] = MapOn<Device::CUDA>(stagingBuffers[i]).data();
            continue;
        }
#endif
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        if (inputMemoryDevice == Device::CPU) {
            swapchainMemoryMapped[i] = MapOn<Device::CPU>(stagingBuffers[i]).data();
            continue;
        }
#endif
    }

    // Create command pool.

    Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = indices.graphicFamily.value();
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    JST_VK_CHECK(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &swapchainCommandPool), [&]{
        JST_ERROR("[VULKAN] Failed to create swapchain command pool.");
    });

    // Create command buffer.

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = swapchainCommandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = swapchainCommandBuffers.size();

    JST_VK_CHECK(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, swapchainCommandBuffers.data()), [&]{
        JST_ERROR("[VULKAN] Failed to allocate swapchain command buffers.");
    });

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

    // Destroy swapchain.

    vkFreeCommandBuffers(device, swapchainCommandPool, swapchainCommandBuffers.size(), swapchainCommandBuffers.data());
    vkDestroyCommandPool(device, swapchainCommandPool, nullptr);

    for (U32 i = 0; i < swapchainImageViews.size(); i++) {
        stagingBuffers[i] = Tensor<Device::Vulkan, U8>();
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
    return Result::SUCCESS;
}

F32 Implementation::scale(const F32& scale) const {
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
    io.DisplaySize = ImVec2(config.size.x, config.size.y);
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
    // Copy swapchain image to staging buffer.

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
    region.imageExtent.width = config.size.x;
    region.imageExtent.height = config.size.y;
    region.imageExtent.depth = 1;

    vkCmdCopyImageToBuffer(swapchainCommandBuffers[_currentDrawableIndex],
                           swapchainImages[_currentDrawableIndex],
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuffers[_currentDrawableIndex].data(),
                           1,
                           &region);

    JST_VK_CHECK(vkEndCommandBuffer(swapchainCommandBuffers[_currentDrawableIndex]), [&]{
        JST_ERROR("[VULKAN] Failed to end swapchain framebuffer download command-buffer.");
    });

    // Wait framebuffer to be ready.

    const VkPipelineStageFlags waitStage[] = { VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = semaphores.size();
    submitInfo.pWaitSemaphores = semaphores.data();
    submitInfo.pWaitDstStageMask = waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &swapchainCommandBuffers[_currentDrawableIndex];
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

    if (endpointFrameSubmissionResult.has_value()) {
        return endpointFrameSubmissionResult.value();
    }

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

        const auto& result = endpoint.pushNewFrame(swapchainMemoryMapped[fenceIndex]);
        if (result != Result::SUCCESS) {
            endpointFrameSubmissionResult = result;
        }

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

Result Implementation::waitEvents() {
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return keepRunningFlag;
}

}  // namespace Jetstream::Viewport
