#include "jetstream/render/devices/vulkan/window.hh"
#include "jetstream/render/devices/vulkan/surface.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#include "tools/imgui_impl_vulkan.h"

const size_t MAX_FRAMES_IN_FLIGHT = 2;

namespace Jetstream::Render {

using Implementation = WindowImp<DeviceType::Vulkan>;

Implementation::WindowImp(const Config& config,
                          const std::shared_ptr<Viewport::Adapter<DeviceType::Vulkan>>& viewport)
         : Window(config), viewport(viewport) {
}

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::Vulkan>>(surface);
    surfaces.push_back(_resource);
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

    // Reseting internal variables.

    statsData.droppedFrames = 0;
    statsData.recreatedFrames = 0;
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

    JST_CHECK(createImgui());
    JST_CHECK(createFramebuffer());
    JST_CHECK(createSynchronizationObjects());

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
    if (imguiCreated) {
        const Result imguiResult = destroyImgui();
        if ((result == Result::SUCCESS || result == Result::RELOAD) &&
            imguiResult != Result::SUCCESS && imguiResult != Result::RELOAD) {
            result = imguiResult;
        }
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

Result Implementation::recreate() {
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

Result Implementation::createImgui() {
    JST_DEBUG("[VULKAN] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();
    io->IniFilename = nullptr;

    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    Result viewportResult;
    try {
        viewportResult = viewport->createImgui();
    } catch (...) {
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        throw;
    }
    if (viewportResult != Result::SUCCESS && viewportResult != Result::RELOAD) {
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        return viewportResult;
    }

    try {
        this->updateScalingFactor(*viewport);

        auto& backend = Backend::State<DeviceType::Vulkan>();

        ImGui_ImplVulkan_InitInfo init_info{};
        init_info.Instance = backend->getInstance();
        init_info.PhysicalDevice = backend->getPhysicalDevice();
        init_info.Device = backend->getDevice();
        init_info.QueueFamily = Backend::FindQueueFamilies(backend->getPhysicalDevice()).graphicFamily.value();
        init_info.Queue = backend->getGraphicsQueue();
        init_info.DescriptorPool = backend->getDescriptorPool();
        init_info.RenderPass = renderPass;
        init_info.MinImageCount = static_cast<U32>(viewport->getSwapchainImageViewsCount());
        init_info.ImageCount = static_cast<U32>(viewport->getSwapchainImageViewsCount());
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        ImGui_ImplVulkan_Init(&init_info);
    } catch (...) {
        viewport->destroyImgui();
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        throw;
    }
    imguiCreated = true;

    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("[VULKAN] Destroying ImGui.");

    ImGui_ImplVulkan_Shutdown();
    const Result result = viewport->destroyImgui();
    ImGui::DestroyContext();
    io = nullptr;
    style = nullptr;
    imguiCreated = false;

    return result;
}

Result Implementation::beginImgui() {
    ImGui_ImplVulkan_NewFrame();

    this->updateScalingFactor(*viewport);

    ImGui::NewFrame();

    return Result::SUCCESS;
}

Result Implementation::endImgui() {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffers[currentFrame]);

    return Result::SUCCESS;
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

    // Get next viewport framebuffer.

    const Result& result = viewport->nextDrawable(imageAvailableSemaphores[currentFrame]);

    if (result == Result::SKIP) {
        statsData.droppedFrames += 1;
        return Result::SKIP;
    }

    if (result == Result::RECREATE) {
        statsData.recreatedFrames += 1;
        JST_CHECK(recreate());
        return Result::SKIP;
    }

    if (result != Result::SUCCESS) {
        JST_FATAL("[VULKAN] Failed to acquire next viewport drawable.");
        return Result::ERROR;
    }

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

    // Begin secondary renders.

    const Result beginResult = beginImgui();
    if (beginResult != Result::SUCCESS && beginResult != Result::RELOAD) {
        return abortFrame(beginResult);
    }

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
        result = surface->encode(commandBuffers[currentFrame]);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = swapchainFramebuffers[viewport->currentDrawableIndex()];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    const auto extent = viewport->getSwapchainExtent();
    renderPassBeginInfo.renderArea.extent = {static_cast<U32>(extent.x), static_cast<U32>(extent.y)};

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearColor;
    vkCmdBeginRenderPass(commandBuffers[currentFrame], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    // End secondary renders.

    result = endImgui();
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        return abortFrame(result);
    }

    // End render pass and command buffer.

    vkCmdEndRenderPass(commandBuffers[currentFrame]);

    if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Can't end command buffer.");
        return abortFrame(Result::ERROR);
    }

    // Reset synchronization fences and submit to queue.

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.waitSemaphoreCount = 1;
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
    submitInfo.commandBufferCount = 1;

    std::vector<VkSemaphore> signalSemaphores = { renderFinishedSemaphores[currentFrame] };
    submitInfo.pSignalSemaphores = signalSemaphores.data();
    submitInfo.signalSemaphoreCount = signalSemaphores.size();

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

    // Commit framebuffer to viewport.

    const auto& presentResult = viewport->commitDrawable(signalSemaphores);

    if (presentResult == Result::RECREATE) {
        statsData.recreatedFrames += 1;
        JST_CHECK(recreate());
        return Result::SKIP;
    }

    if (presentResult != Result::SUCCESS) {
        const Result recovery = recreate();
        return recovery == Result::SUCCESS ? presentResult : recovery;
    }

    // Increment frame counter.

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    return Result::SUCCESS;
}

Result Implementation::underlyingCancel() {
    const bool reset = vkResetCommandBuffer(commandBuffers[currentFrame], 0) == VK_SUCCESS;
    if (!reset) {
        JST_ERROR("[VULKAN] Can't reset cancelled command buffer.");
    }
    const Result recovery = recreate();
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

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
