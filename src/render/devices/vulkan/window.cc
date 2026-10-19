#include <algorithm>

#include "jetstream/render/devices/vulkan/window.hh"
#include "jetstream/render/devices/vulkan/surface.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#include "presenter.hh"

const size_t MAX_FRAMES_IN_FLIGHT = 2;

namespace Jetstream::Render {

using Implementation = WindowImp<DeviceType::Vulkan>;

Implementation::WindowImp(const Config& config,
                          const std::shared_ptr<Viewport::Adapter<DeviceType::Vulkan>>& viewport)
         : Window(config), viewport(viewport) {
}

Implementation::~WindowImp() = default;

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::Vulkan>>(surface);
    if (_resource->getConfig().composite) {
        surfaces.push_back(_resource);
    } else {
        const auto firstComposite = std::find_if(surfaces.begin(), surfaces.end(), [](const auto& other) {
            return other->getConfig().composite;
        });
        surfaces.insert(firstComposite, _resource);
    }
    return Result::SUCCESS;
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::Vulkan>>(surface);
    surfaces.erase(std::remove(surfaces.begin(), surfaces.end(), _resource), surfaces.end());
    return Result::SUCCESS;
}

Result Implementation::underlyingCreate() {
    JST_DEBUG("[VULKAN] Creating window.");

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    auto& headless = Backend::State<DeviceType::Vulkan>()->headless();
    auto& physicalDevice = Backend::State<DeviceType::Vulkan>()->getPhysicalDevice();

    transferEncoder.create(MAX_FRAMES_IN_FLIGHT);

    // Create render pass.

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = viewport->getSwapchainImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = (headless ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL :
                                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VkRenderPass createdRenderPass = VK_NULL_HANDLE;
    JST_VK_CHECK(vkCreateRenderPass(device,
                                    &renderPassInfo,
                                    nullptr,
                                    &createdRenderPass), [&]{
        JST_ERROR("[VULKAN] Failed to create render pass.");
    });
    renderPass = createdRenderPass;

    // Create command pool.

    Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = indices.graphicFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool createdCommandPool = VK_NULL_HANDLE;
    JST_VK_CHECK(vkCreateCommandPool(device,
                                     &poolInfo,
                                     nullptr,
                                     &createdCommandPool), [&]{
        JST_ERROR("[VULKAN] Failed to create graphics command pool.");
    });
    commandPool = createdCommandPool;

    // Create command buffers.

    commandBuffers.resize(viewport->getSwapchainImageViewsCount());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<U32>(commandBuffers.size());

    JST_VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()), [&]{
        JST_ERROR("[VULKAN] Can't create render command buffers.");
    });
    commandBuffersAllocated = true;

    // Create submodules.

    JST_CHECK(createPresenter());
    JST_CHECK(createFramebuffer());
    JST_CHECK(createSynchronizationObjects());
    this->updateMetrics(*viewport);

    windowCreated = true;
    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[VULKAN] Destroying window.");

    // Release resources.

    Result result = Result::SUCCESS;
    if (windowCreated) {
        result = underlyingSynchronize();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return result;
        }
    }

    destroySynchronizationObjects();
    destroyFramebuffer();
    if (presenter) {
        presenter->destroy();
        presenter.reset();
    }

    transferEncoder.destroy();

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    if (commandBuffersAllocated) {
        vkFreeCommandBuffers(device,
                             commandPool,
                             static_cast<U32>(commandBuffers.size()),
                             commandBuffers.data());
        commandBuffersAllocated = false;
    }
    commandBuffers.clear();
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    if (renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }
    windowCreated = false;

    return result;
}

Result Implementation::underlyingRecreate() {
    JST_CHECK(underlyingSynchronize());

    JST_CHECK(destroySynchronizationObjects());
    JST_CHECK(destroyFramebuffer());
    JST_CHECK(viewport->destroySwapchain());
    JST_CHECK(viewport->createSwapchain());
    JST_CHECK(createFramebuffer());
    JST_CHECK(createSynchronizationObjects());

    return Result::SUCCESS;
}

Result Implementation::createFramebuffer() {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    const auto swapchainExtent = viewport->getSwapchainExtent();

    swapchainFramebuffers.resize(viewport->getSwapchainImageViewsCount());

    for (size_t i = 0; i < swapchainFramebuffers.size(); i++) {
        VkImageView attachments[] = {  viewport->getSwapchainImageView(i) };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapchainExtent.x;
        framebufferInfo.height = swapchainExtent.y;
        framebufferInfo.layers = 1;

        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        JST_VK_CHECK(vkCreateFramebuffer(device,
                                         &framebufferInfo,
                                         nullptr,
                                         &framebuffer), [&]{
            JST_ERROR("[VULKAN] Failed to create swapchain framebuffer.");
        });
        swapchainFramebuffers[i] = framebuffer;
    }

    return Result::SUCCESS;
}

Result Implementation::destroyFramebuffer() {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    for (auto framebuffer : swapchainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    swapchainFramebuffers.clear();

    return Result::SUCCESS;
}

Result Implementation::createPresenter() {
    presenter = std::make_unique<Presenter>();
    const Result result = presenter->create(renderPass, MAX_FRAMES_IN_FLIGHT);
    if (result != Result::SUCCESS) {
        presenter.reset();
    }
    return result;
}

Result Implementation::underlyingBegin() {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    // Wait for a frame to be available.

    JST_VK_CHECK(vkWaitForFences(device,
                                 1,
                                 &inFlightFences[currentFrame],
                                 VK_TRUE,
                                 UINT64_MAX), [&]{
        JST_ERROR("[VULKAN] Failed to wait for an in-flight frame.");
    });

    const auto abortFrame = [&](const Result& abortResult) {
        const Result recovery = underlyingCancel();
        return recovery == Result::SUCCESS ? abortResult : recovery;
    };

    // Refresh command buffer.

    if (vkResetCommandBuffer(commandBuffers[currentFrame], 0) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Can't reset command buffer.");
        return abortFrame(Result::ERROR);
    }

    VkCommandBufferBeginInfo commandBufferBeginInfo = {};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffers[currentFrame], &commandBufferBeginInfo) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Can't begin command buffer.");
        return abortFrame(Result::ERROR);
    }

    for (auto& surface : surfaces) {
        const Result prepareResult = surface->prepare();
        if (prepareResult != Result::SUCCESS && prepareResult != Result::RELOAD) {
            return abortFrame(prepareResult);
        }
    }

    this->updateMetrics(*viewport);
    this->updateInput(*viewport);

    return Result::SUCCESS;
}

Result Implementation::underlyingEnd() {
    Transfer::Batch transfers;

    const auto abortFrame = [&](const Result& result) {
        const Result recovery = underlyingCancel();
        return recovery == Result::SUCCESS ? result : recovery;
    };

    Result result = collectTransfers(transfers);
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        return abortFrame(result);
    }
    if (!transfers.empty()) {
        result = transferEncoder.encode(transfers,
                                        commandBuffers[currentFrame],
                                        currentFrame);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    for (auto& surface : surfaces) {
        const auto& framebuffer = *surface->getConfig().framebuffer;
        result = surface->encode(commandBuffers[currentFrame],
                                 textureVisibleForPresentation(framebuffer));
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    auto pass = beginPresent(transfers);
    if (pass.presenting) {
        const Result acquisition = viewport->nextDrawable(imageAvailableSemaphores[currentFrame]);
        if (acquisition == Result::RECREATE) {
            acquired(pass, Acquire::Recreate);
        } else if (acquisition == Result::SKIP) {
            acquired(pass, Acquire::Dropped);
        } else if (acquisition != Result::SUCCESS) {
            JST_ERROR("[VULKAN] Failed to acquire next viewport drawable.");
            return abortFrame(Result::ERROR);
        }
    }

    if (pass.presenting) {
        result = encodePresent();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Can't end command buffer.");
        return abortFrame(Result::ERROR);
    }

    // Reset synchronization fences and submit to queue.

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    std::vector<VkSemaphore> signalSemaphores;
    if (pass.presenting) {
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitDstStageMask = waitStages;
        signalSemaphores.push_back(renderFinishedSemaphores[currentFrame]);
        submitInfo.pSignalSemaphores = signalSemaphores.data();
        submitInfo.signalSemaphoreCount = static_cast<U32>(signalSemaphores.size());
    }

    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
    submitInfo.commandBufferCount = 1;

    auto& backend = Backend::State<DeviceType::Vulkan>();
    auto& graphicsQueue = backend->getGraphicsQueue();
    auto& device = backend->getDevice();
    if (vkResetFences(device, 1, &inFlightFences[currentFrame]) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Failed to reset an in-flight fence.");
        return abortFrame(Result::ERROR);
    }
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Failed to submit draw command buffer.");
        return abortFrame(Result::ERROR);
    }

    transferEncoder.commit(transfers);
    for (auto& surface : surfaces) {
        surface->commit();
    }
    transfers.commit();

    if (pass.presenting) {
        const auto& presentResult = viewport->commitDrawable(signalSemaphores);
        if (presentResult == Result::RECREATE) {
            acquired(pass, Acquire::Recreate);
        } else if (presentResult != Result::SUCCESS) {
            acquired(pass, Acquire::Recreate);
            const Result recovery = finishPresent(pass);
            return recovery == Result::SKIP ? presentResult : recovery;
        }
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    return finishPresent(pass);
}

Result Implementation::encodePresent() {
    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = swapchainFramebuffers[viewport->currentDrawableIndex()];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    const auto extent = viewport->getSwapchainExtent();
    renderPassBeginInfo.renderArea.extent = {static_cast<U32>(extent.x), static_cast<U32>(extent.y)};

    const auto& background = frameClearColor();
    VkClearValue clearColor = {{{background.r, background.g, background.b, background.a}}};
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearColor;
    vkCmdBeginRenderPass(commandBuffers[currentFrame], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    const Result result = presenter->encode(commandBuffers[currentFrame], currentFrame, framePresentList(), metrics(),
                                            extent);

    vkCmdEndRenderPass(commandBuffers[currentFrame]);

    return result;
}

Result Implementation::underlyingCancel() {
    const bool reset = vkResetCommandBuffer(commandBuffers[currentFrame], 0) == VK_SUCCESS;
    if (!reset) {
        JST_ERROR("[VULKAN] Can't reset cancelled command buffer.");
    }
    const Result recovery = underlyingRecreate();
    return reset ? recovery
                 : (recovery == Result::SUCCESS ? Result::ERROR : recovery);
}

Result Implementation::underlyingSynchronize() {
    JST_VK_CHECK(vkQueueWaitIdle(Backend::State<DeviceType::Vulkan>()->getGraphicsQueue()), [&]{
        JST_ERROR("[VULKAN] Can't synchronize graphics queue.");
    });

    return Result::SUCCESS;
}

Result Implementation::createSynchronizationObjects() {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    currentFrame = 0;
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        if (vkCreateSemaphore(device,
                              &semaphoreInfo,
                              nullptr,
                              &imageAvailable) != VK_SUCCESS) {
            JST_ERROR("[VULKAN] Failed to create synchronization objects.");
            return Result::ERROR;
        }
        imageAvailableSemaphores[i] = imageAvailable;

        VkSemaphore renderFinished = VK_NULL_HANDLE;
        if (vkCreateSemaphore(device,
                              &semaphoreInfo,
                              nullptr,
                              &renderFinished) != VK_SUCCESS) {
            JST_ERROR("[VULKAN] Failed to create synchronization objects.");
            return Result::ERROR;
        }
        renderFinishedSemaphores[i] = renderFinished;

        VkFence inFlight = VK_NULL_HANDLE;
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlight) != VK_SUCCESS) {
            JST_ERROR("[VULKAN] Failed to create synchronization objects.");
            return Result::ERROR;
        }
        inFlightFences[i] = inFlight;
    }

    return Result::SUCCESS;
}

Result Implementation::destroySynchronizationObjects() {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    for (const auto semaphore : renderFinishedSemaphores) {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    for (const auto semaphore : imageAvailableSemaphores) {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    for (const auto fence : inFlightFences) {
        vkDestroyFence(device, fence, nullptr);
    }
    renderFinishedSemaphores.clear();
    imageAvailableSemaphores.clear();
    inFlightFences.clear();
    imagesInFlight.clear();

    return Result::SUCCESS;
}

std::string Implementation::info() const {
    auto& backend = Backend::State<DeviceType::Vulkan>();
    return jst::fmt::format("{} ({:.0f} GB)", backend->getDeviceName(),
                                              (float)backend->getPhysicalMemory() / (1024*1024*1024));
}

}  // namespace Jetstream::Render
