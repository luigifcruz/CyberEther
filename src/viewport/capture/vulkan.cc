#include "jetstream/viewport/capture/vulkan.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "jetstream/memory/devices/vulkan/buffer.hh"
#include "jetstream/logger.hh"

#include "vulkan_export.hh"

#include <array>
#include <cstring>
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
    Export exporting = Export::None;

    std::array<Tensor, MAX_FRAMES_IN_FLIGHT> stagingTensors = {};
    std::array<Tensor, MAX_FRAMES_IN_FLIGHT> outputTensors = {};
    std::unique_ptr<FrameExport> exporter;

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

    Result createExporter(U32 family);
    Result waitFrame(U64& index);
};

bool FrameCaptureVulkan::SupportsExport(Export exporting) {
    const auto& state = Backend::State<DeviceType::Vulkan>();
    if (!state->isAvailable()) {
        return false;
    }

    switch (exporting) {
        case Export::None:
            return true;
#if defined(JST_OS_LINUX)
        case Export::OpaqueFd:
            return state->canExportDeviceMemory();
        case Export::DmaBuf:
            return state->canShareFrames() && Device().render != 0;
#endif
#if defined(JST_OS_WINDOWS)
        case Export::D3D11:
            return state->canShareFrames() && Device().luid != 0;
#endif
        default:
            return false;
    }
}

FrameCapture::SharedDevice FrameCaptureVulkan::Device() {
    const auto& state = Backend::State<DeviceType::Vulkan>();

    VkPhysicalDeviceIDProperties id = {};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceProperties2 properties = {};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;

#if defined(JST_OS_LINUX)
    VkPhysicalDeviceDrmPropertiesEXT drm = {};
    drm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT;
    if (state->canShareFrames()) {
        id.pNext = &drm;
    }
#endif

    vkGetPhysicalDeviceProperties2(state->getPhysicalDevice(), &properties);

    SharedDevice device;
    std::memcpy(device.uuid.data(), id.deviceUUID, device.uuid.size());
    if (id.deviceLUIDValid) {
        std::memcpy(&device.luid, id.deviceLUID, sizeof(device.luid));
    }
#if defined(JST_OS_LINUX)
    if (state->canShareFrames() && drm.hasRender) {
        device.render = static_cast<U32>(drm.renderMinor);
    }
#endif

    return device;
}

Result FrameCaptureVulkan::Impl::createExporter(U32 family) {
    switch (exporting) {
#if defined(JST_OS_LINUX)
        case Export::OpaqueFd:
        case Export::DmaBuf:
            exporter = CreateBufferExport(exporting);
            break;
#endif
#if defined(JST_OS_WINDOWS)
        case Export::D3D11:
            exporter = CreateD3D11Export();
            break;
#endif
        default:
            JST_ERROR("[CAPTURE] Frame export is not supported on this platform.");
            return Result::ERROR;
    }

    if (exporter->create(extent, family, MAX_FRAMES_IN_FLIGHT) != Result::SUCCESS) {
        exporter.reset();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

FrameCaptureVulkan::FrameCaptureVulkan(Export exporting) : pimpl(std::make_unique<Impl>()) {
    pimpl->exporting = exporting;
}

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

    Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);
    const U32 family = indices.graphicFamily.value();

    if (pimpl->exporting != Export::None) {
        if (outputDevice != DeviceType::Vulkan || !SupportsExport(pimpl->exporting)) {
            JST_ERROR("[CAPTURE] Vulkan device can't export frames to the encoder.");
            return Result::ERROR;
        }
        JST_CHECK(pimpl->createExporter(family));
    } else {
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
    }

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = family;
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

    JST_DEBUG("[CAPTURE] Vulkan frame capture created ({}x{}, {}).", pimpl->extent.width,
                                                                    pimpl->extent.height,
                                                                    pimpl->exporter ? "exported" : "copied");

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

    if (pimpl->exporter) {
        pimpl->exporter->destroy();
        pimpl->exporter.reset();
    }
    pimpl->vulkanViewport = nullptr;

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::stop() {
    JST_DEBUG("[CAPTURE] Stopping Vulkan frame capture.");

    {
        std::lock_guard<std::mutex> lock(pimpl->frameMutex);
        pimpl->running = false;
        pimpl->stopped = true;
    }
    pimpl->frameCond.notify_all();

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::captureFrame() {
    if (!pimpl->running || !pimpl->vulkanViewport) {
        return Result::SUCCESS;
    }

    const auto swapchainExtent = pimpl->vulkanViewport->getSwapchainExtent();
    if (swapchainExtent.x != pimpl->extent.width || swapchainExtent.y != pimpl->extent.height) {
        return Result::SKIP;
    }

    if (pimpl->frameEvents[pimpl->currentCaptureIndex].test()) {
        return Result::SKIP;
    }
    pimpl->frameEvents[pimpl->currentCaptureIndex].test_and_set();

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    const bool headless = Backend::State<DeviceType::Vulkan>()->headless();
    const U32 drawableIndex = pimpl->vulkanViewport->currentDrawableIndex();
    const VkImage source = pimpl->vulkanViewport->getSwapchainImage(drawableIndex);
    auto& commandBuffer = pimpl->commandBuffers[pimpl->currentCaptureIndex];

    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    JST_VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo), [&]{
        JST_ERROR("[CAPTURE] Failed to begin command buffer.");
    });

    VkImageMemoryBarrier swapchainBarrier = {};
    swapchainBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapchainBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchainBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchainBarrier.image = source;
    swapchainBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapchainBarrier.subresourceRange.levelCount = 1;
    swapchainBarrier.subresourceRange.layerCount = 1;

    if (!headless) {
        swapchainBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        swapchainBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        swapchainBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        swapchainBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &swapchainBarrier);
    }

    if (pimpl->exporter) {
        pimpl->exporter->record(commandBuffer, source, pimpl->currentCaptureIndex);
    } else {
        VkBufferImageCopy region = {};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {pimpl->extent.width, pimpl->extent.height, 1};

        auto* vkBackend = static_cast<VulkanBufferBackend*>(
            pimpl->stagingTensors[pimpl->currentCaptureIndex].buffer().backend());

        vkCmdCopyImageToBuffer(commandBuffer, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               vkBackend->buffer(), 1, &region);
    }

    if (!headless) {
        swapchainBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        swapchainBarrier.dstAccessMask = 0;
        swapchainBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        swapchainBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &swapchainBarrier);
    }

    JST_VK_CHECK(vkEndCommandBuffer(commandBuffer), [&]{
        JST_ERROR("[CAPTURE] Failed to end command buffer.");
    });

    vkResetFences(device, 1, &pimpl->fences[pimpl->currentCaptureIndex]);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

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

Result FrameCaptureVulkan::Impl::waitFrame(U64& index) {
    if (stopped) {
        return Result::ERROR;
    }

    {
        std::unique_lock<std::mutex> lock(frameMutex);
        frameCond.wait(lock, [&]{
            return !frameQueue.empty() || stopped;
        });

        if (stopped && frameQueue.empty()) {
            return Result::ERROR;
        }

        index = frameQueue.front();
        frameQueue.pop();
    }

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    vkWaitForFences(device, 1, &fences[index], VK_TRUE, UINT64_MAX);

    currentReadyIndex = index;

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::getFrameData(Tensor& tensor) {
    if (pimpl->exporter) {
        return Result::ERROR;
    }

    U64 index = 0;
    JST_CHECK(pimpl->waitFrame(index));
    tensor = pimpl->outputTensors[index];

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::getSharedFrame(SharedFrame& frame) {
    if (!pimpl->exporter) {
        return Result::ERROR;
    }

    U64 index = 0;
    JST_CHECK(pimpl->waitFrame(index));
    frame = pimpl->exporter->frame(static_cast<U32>(index));

    return Result::SUCCESS;
}

Result FrameCaptureVulkan::releaseFrame() {
    pimpl->frameEvents[pimpl->currentReadyIndex].clear();
    pimpl->frameEvents[pimpl->currentReadyIndex].notify_one();

    return Result::SUCCESS;
}

}  // namespace Jetstream::Viewport
