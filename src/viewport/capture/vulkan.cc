#include "jetstream/viewport/capture/vulkan.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "jetstream/memory/devices/vulkan/buffer.hh"
#include "jetstream/logger.hh"

#include <array>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace Jetstream::Viewport {

struct FrameCaptureVulkan::Impl {
    static constexpr U32 MAX_FRAMES_IN_FLIGHT = 2;

    Adapter<DeviceType::Vulkan>* vulkanViewport = nullptr;
    VkExtent2D extent = {};
    DeviceType outputDevice = DeviceType::Vulkan;

    std::array<Tensor, MAX_FRAMES_IN_FLIGHT> stagingTensors = {};
    std::array<Tensor, MAX_FRAMES_IN_FLIGHT> outputTensors = {};

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> commandBuffers = {};
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> fences = {};
    std::array<std::atomic_flag, MAX_FRAMES_IN_FLIGHT> frameEvents;

    std::queue<U64> frameQueue;
    std::mutex frameMutex;
    std::condition_variable frameCond;
    std::atomic<bool> running{false};
    std::atomic<bool> stopped{false};

    U32 currentCaptureIndex = 0;
    U64 currentReadyIndex = 0;
};

FrameCaptureVulkan::FrameCaptureVulkan() : pimpl(std::make_unique<Impl>()) {}

FrameCaptureVulkan::~FrameCaptureVulkan() = default;

Result FrameCaptureVulkan::create(Generic* viewport, const DeviceType& outputDevice) {
    JST_DEBUG("[CAPTURE] Creating Vulkan frame capture.");

    pimpl->vulkanViewport = dynamic_cast<Adapter<DeviceType::Vulkan>*>(viewport);
    if (!pimpl->vulkanViewport) {
        JST_ERROR("[CAPTURE] Viewport is not a Vulkan adapter.");
        return Result::ERROR;
    }

    pimpl->outputDevice = outputDevice;

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<DeviceType::Vulkan>()->getPhysicalDevice();

    const auto extent = pimpl->vulkanViewport->getSwapchainExtent();
    pimpl->extent = {static_cast<U32>(extent.x), static_cast<U32>(extent.y)};

    for (U32 i = 0; i < Impl::MAX_FRAMES_IN_FLIGHT; i++) {
        const Shape extent = {
            pimpl->extent.height,
            pimpl->extent.width,
            4
        };

        const Buffer::Config config = {
            .hostAccessible = (pimpl->outputDevice == DeviceType::CPU),
        };
        JST_CHECK(pimpl->stagingTensors[i].create(DeviceType::Vulkan, DataType::U8, extent, config));

        if (pimpl->outputTensors[i].create(pimpl->outputDevice, pimpl->stagingTensors[i]) != Result::SUCCESS) {
            JST_ERROR("[CAPTURE] Staging tensor does not support requested output device.");
            return Result::ERROR;
        }
    }

    Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = indices.graphicFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    JST_VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &pimpl->commandPool), [&]{
        JST_ERROR("[CAPTURE] Failed to create command pool.");
    });

    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = pimpl->commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = Impl::MAX_FRAMES_IN_FLIGHT;

    JST_VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, pimpl->commandBuffers.data()), [&]{
        JST_ERROR("[CAPTURE] Failed to allocate command buffers.");
    });

    for (U32 i = 0; i < Impl::MAX_FRAMES_IN_FLIGHT; i++) {
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        JST_VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &pimpl->fences[i]), [&]{
            JST_ERROR("[CAPTURE] Failed to create fence.");
        });

        pimpl->frameEvents[i].clear();
    }

    pimpl->running = true;
    pimpl->stopped = false;
    pimpl->currentCaptureIndex = 0;

    JST_DEBUG("[CAPTURE] Vulkan frame capture created ({}x{}).", pimpl->extent.width, pimpl->extent.height);

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::destroy() {
    JST_DEBUG("[CAPTURE] Destroying Vulkan frame capture.");

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    vkDeviceWaitIdle(device);

    for (U32 i = 0; i < Impl::MAX_FRAMES_IN_FLIGHT; i++) {
        if (pimpl->fences[i] != VK_NULL_HANDLE) {
            vkDestroyFence(device, pimpl->fences[i], nullptr);
            pimpl->fences[i] = VK_NULL_HANDLE;
        }
    }

    if (pimpl->commandPool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device, pimpl->commandPool, Impl::MAX_FRAMES_IN_FLIGHT, pimpl->commandBuffers.data());
        vkDestroyCommandPool(device, pimpl->commandPool, nullptr);
        pimpl->commandPool = VK_NULL_HANDLE;
    }

    for (U32 i = 0; i < Impl::MAX_FRAMES_IN_FLIGHT; i++) {
        pimpl->stagingTensors[i] = Tensor();
        pimpl->outputTensors[i] = Tensor();
    }

    pimpl->vulkanViewport = nullptr;

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::stop() {
    JST_DEBUG("[CAPTURE] Stopping Vulkan frame capture.");

    pimpl->running = false;
    pimpl->stopped = true;
    pimpl->frameCond.notify_all();

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::captureFrame() {
    if (!pimpl->running || !pimpl->vulkanViewport) {
        return Result::SUCCESS;
    }

    if (pimpl->frameEvents[pimpl->currentCaptureIndex].test()) {
        return Result::SUCCESS;
    }
    pimpl->frameEvents[pimpl->currentCaptureIndex].test_and_set();

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    const bool headless = Backend::State<DeviceType::Vulkan>()->headless();
    const U32 drawableIndex = pimpl->vulkanViewport->currentDrawableIndex();

    vkResetCommandBuffer(pimpl->commandBuffers[pimpl->currentCaptureIndex], 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    JST_VK_CHECK(vkBeginCommandBuffer(pimpl->commandBuffers[pimpl->currentCaptureIndex], &beginInfo), [&]{
        JST_ERROR("[CAPTURE] Failed to begin command buffer.");
    });

    if (!headless) {
        VkImageMemoryBarrier toTransferBarrier = {};
        toTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toTransferBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        toTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferBarrier.image = pimpl->vulkanViewport->getSwapchainImage(drawableIndex);
        toTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toTransferBarrier.subresourceRange.baseMipLevel = 0;
        toTransferBarrier.subresourceRange.levelCount = 1;
        toTransferBarrier.subresourceRange.baseArrayLayer = 0;
        toTransferBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(pimpl->commandBuffers[pimpl->currentCaptureIndex],
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toTransferBarrier);
    }

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent.width = pimpl->extent.width;
    region.imageExtent.height = pimpl->extent.height;
    region.imageExtent.depth = 1;

    auto* vkBackend = static_cast<VulkanBufferBackend*>(
        pimpl->stagingTensors[pimpl->currentCaptureIndex].buffer().backend());

    vkCmdCopyImageToBuffer(pimpl->commandBuffers[pimpl->currentCaptureIndex],
                           pimpl->vulkanViewport->getSwapchainImage(drawableIndex),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           vkBackend->buffer(),
                           1,
                           &region);

    if (!headless) {
        VkImageMemoryBarrier toPresentBarrier = {};
        toPresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toPresentBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toPresentBarrier.dstAccessMask = 0;
        toPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toPresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toPresentBarrier.image = pimpl->vulkanViewport->getSwapchainImage(drawableIndex);
        toPresentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toPresentBarrier.subresourceRange.baseMipLevel = 0;
        toPresentBarrier.subresourceRange.levelCount = 1;
        toPresentBarrier.subresourceRange.baseArrayLayer = 0;
        toPresentBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(pimpl->commandBuffers[pimpl->currentCaptureIndex],
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toPresentBarrier);
    }

    JST_VK_CHECK(vkEndCommandBuffer(pimpl->commandBuffers[pimpl->currentCaptureIndex]), [&]{
        JST_ERROR("[CAPTURE] Failed to end command buffer.");
    });

    vkResetFences(device, 1, &pimpl->fences[pimpl->currentCaptureIndex]);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &pimpl->commandBuffers[pimpl->currentCaptureIndex];

    auto& graphicsQueue = Backend::State<DeviceType::Vulkan>()->getGraphicsQueue();
    JST_VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, pimpl->fences[pimpl->currentCaptureIndex]), [&]{
        JST_ERROR("[CAPTURE] Failed to submit command buffer.");
    });

    {
        std::lock_guard<std::mutex> lock(pimpl->frameMutex);
        pimpl->frameQueue.push(pimpl->currentCaptureIndex);
        pimpl->frameCond.notify_one();
    }

    pimpl->currentCaptureIndex = (pimpl->currentCaptureIndex + 1) % Impl::MAX_FRAMES_IN_FLIGHT;

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::getFrameData(Tensor& tensor) {
    if (pimpl->stopped) {
        return Result::ERROR;
    }

    U64 frameIndex;

    {
        std::unique_lock<std::mutex> lock(pimpl->frameMutex);
        pimpl->frameCond.wait(lock, [&]{
            return !pimpl->frameQueue.empty() || pimpl->stopped;
        });

        if (pimpl->stopped && pimpl->frameQueue.empty()) {
            return Result::ERROR;
        }

        frameIndex = pimpl->frameQueue.front();
        pimpl->frameQueue.pop();
    }

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    vkWaitForFences(device, 1, &pimpl->fences[frameIndex], VK_TRUE, UINT64_MAX);

    pimpl->currentReadyIndex = frameIndex;
    tensor = pimpl->outputTensors[frameIndex];

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::releaseFrame() {
    pimpl->frameEvents[pimpl->currentReadyIndex].clear();
    pimpl->frameEvents[pimpl->currentReadyIndex].notify_one();

    return Result::SUCCESS;
}

}  // namespace Jetstream::Viewport
