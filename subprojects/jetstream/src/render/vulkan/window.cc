#include "jetstream/render/vulkan/window.hh"
#include "jetstream/render/vulkan/surface.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

const size_t MAX_FRAMES_IN_FLIGHT = 2;

namespace Jetstream::Render {

using Implementation = WindowImp<Device::Vulkan>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Adapter<Device::Vulkan>>& viewport)
         : Window(config), viewport(viewport) {
}

Result Implementation::bind(const std::shared_ptr<Surface>& surface) {
    JST_DEBUG("[VULKAN] Binding surface to window.");

    surfaces.push_back(
        std::dynamic_pointer_cast<SurfaceImp<Device::Vulkan>>(surface)
    );

    return Result::SUCCESS;
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating window.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();

    // Create Render Pass.

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = viewport->getSwapchainImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

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

    JST_VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass), [&]{
        JST_FATAL("[VULKAN] Failed to create render pass.");   
    });

    // Create swapchain framebuffer.

    auto& swapchainImageViews = viewport->getSwapchainImageViews();
    const auto& swapchainExtent = viewport->getSwapchainExtent();

    swapchainFramebuffers.resize(swapchainImageViews.size());

    for (size_t i = 0; i < swapchainImageViews.size(); i++) {
        VkImageView attachments[] = { swapchainImageViews[i] };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapchainExtent.width;
        framebufferInfo.height = swapchainExtent.height;
        framebufferInfo.layers = 1;

        JST_VK_CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]), [&]{
            JST_FATAL("[VULKAN] Failed to create swapchain framebuffer.");
        });
    }

    // Create command pool.

    Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = indices.graphicFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    JST_VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), [&]{
        JST_FATAL("[VULKAN] Failed to create graphics command pool.");
    });

    // Create command buffers.

    commandBuffers.resize(swapchainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<U32>(commandBuffers.size());

    JST_VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()), [&]{
        JST_FATAL("[VULKAN] Can't create render command buffers.");
    });

    // Create children.

    if (config.imgui) {
        JST_CHECK(createImgui());
    }

    for (auto& surface : surfaces) {
        JST_CHECK(surface->create());
    }

    statsData.droppedFrames = 0;
    renderPassBeginInfo = {};
    commandBufferBeginInfo = {};
    JST_CHECK(createSynchronizationObjects());

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying window.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    JST_VK_CHECK(vkQueueWaitIdle(Backend::State<Device::Vulkan>()->getGraphicsQueue()), [&]{
        JST_FATAL("[VULKAN] Can't wait for graphics queue to finish for window destruction.");
    });

    JST_CHECK(destroySynchronizationObjects());

    for (auto& surface : surfaces) {
        JST_CHECK(surface->destroy());
    }

    if (config.imgui) {
        JST_CHECK(destroyImgui());
    } 

    vkFreeCommandBuffers(device, commandPool, commandBuffers.size(), commandBuffers.data());

    vkDestroyCommandPool(device, commandPool, nullptr);

    for (auto framebuffer : swapchainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    vkDestroyRenderPass(device, renderPass, nullptr);
    
    return Result::SUCCESS;
}

Result Implementation::recreate() {
    JST_CHECK(destroy());
    JST_CHECK(viewport->destroySwapchain());
    JST_CHECK(viewport->createSwapchain());
    JST_CHECK(create());

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    JST_DEBUG("[VULKAN] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImNodes::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    JST_CHECK(viewport->createImgui());
    ApplyImGuiTheme(viewport->calculateScale(config.scale));
    ApplyImNodesTheme();

    auto& backend = Backend::State<Device::Vulkan>();
    
    ImGui_ImplVulkan_InitInfo init_info = {
        .Instance = backend->getInstance(),
        .PhysicalDevice = backend->getPhysicalDevice(),
        .Device = backend->getDevice(),
        .QueueFamily = Backend::FindQueueFamilies(backend->getPhysicalDevice()).graphicFamily.value(),
        .Queue = backend->getGraphicsQueue(),
        .PipelineCache = {},
        .DescriptorPool = backend->getDescriptorPool(),
        .Subpass = {},
        .MinImageCount = static_cast<U32>(swapchainFramebuffers.size()),
        .ImageCount = static_cast<U32>(swapchainFramebuffers.size()),
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .Allocator = {},
        .CheckVkResultFn = nullptr
    };
    ImGui_ImplVulkan_Init(&init_info, renderPass); 

    JST_CHECK(Backend::ExecuteOnce(backend->getDevice(),
                                   backend->getComputeQueue(),
                                   backend->getTransferCommandPool(),
        [&](VkCommandBuffer& commandBuffer){
            ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);

            return Result::SUCCESS;
        }
    ));

    ImGui_ImplVulkan_DestroyFontUploadObjects();

    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("[VULKAN] Destroying ImGui.");

    ImGui_ImplVulkan_Shutdown();
    JST_CHECK(viewport->destroyImgui());
    ImNodes::DestroyContext();
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

Result Implementation::beginImgui() {
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    return Result::SUCCESS;
}

Result Implementation::endImgui() {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), currentCommandBuffer);

    return Result::SUCCESS;
}

Result Implementation::begin() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Wait for frame to be available.
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Get next viewport framebuffer.
    const Result& result = viewport->nextDrawable(imageAvailableSemaphores[currentFrame]);

    if (result == Result::SKIP) {
        statsData.droppedFrames += 1;
        return Result::SKIP;
    } else if (result == Result::RECREATE) {
        JST_CHECK(recreate());
        return Result::SKIP;       
    } else if (result != Result::SUCCESS) {
        return result;
    }

    // Wait for fences.
    if (imagesInFlight[viewport->currentDrawableIndex()] != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &imagesInFlight[viewport->currentDrawableIndex()], VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[viewport->currentDrawableIndex()] = inFlightFences[currentFrame]; 

    // Set current command buffer.
    currentCommandBuffer = commandBuffers[viewport->currentDrawableIndex()];

    // Refresh command buffer.
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    JST_VK_CHECK(vkBeginCommandBuffer(currentCommandBuffer, &commandBufferBeginInfo), [&]{
        JST_FATAL("[VULKAN] Can't begin command buffer.");     
    });

    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = swapchainFramebuffers[viewport->currentDrawableIndex()];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    renderPassBeginInfo.renderArea.extent = viewport->getSwapchainExtent();

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearColor;

    for (auto &surface : surfaces) {
        JST_CHECK(surface->encode(currentCommandBuffer));
    }

    vkCmdBeginRenderPass(currentCommandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    if (config.imgui) {
        JST_CHECK(beginImgui());
    }

    return Result::SUCCESS;
}

Result Implementation::end() {
    if (config.imgui) {
        JST_CHECK(endImgui());
    }

    vkCmdEndRenderPass(currentCommandBuffer);

    JST_VK_CHECK(vkEndCommandBuffer(currentCommandBuffer), [&]{
        JST_FATAL("[VULKAN] Can't end command buffer.");
    });

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    std::vector<VkSemaphore> waitSemaphores = { imageAvailableSemaphores[currentFrame] };
    std::vector<VkSemaphore> signalSemaphores = { renderFinishedSemaphores[currentFrame] };

    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[viewport->currentDrawableIndex()];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    auto& graphicsQueue = Backend::State<Device::Vulkan>()->getGraphicsQueue();
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        JST_FATAL("[VULKAN] Failed to submit draw command buffer.");
        return Result::ERROR;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    const auto& result = viewport->commitDrawable(signalSemaphores);
    if (result == Result::RECREATE) {
        return recreate();
    }
    return result;
}

Result Implementation::createSynchronizationObjects() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    currentFrame = 0;
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapchainFramebuffers.size(), nullptr);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            JST_FATAL("[VULKAN] Failed to create synchronization objects.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result Implementation::destroySynchronizationObjects() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    std::fill(imagesInFlight.begin(), imagesInFlight.end(), nullptr);

    return Result::SUCCESS;
}

void Implementation::drawDebugMessage() const {
    auto& backend = Backend::State<Device::Vulkan>();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::TextFormatted("FPS: {:.1f} Hz", io.Framerate);
    ImGui::TextFormatted("Device Name: {}", backend->getDeviceName());
    ImGui::TextFormatted("Has Unified Memory: {}", backend->hasUnifiedMemory() ? "YES" : "NO");
    ImGui::TextFormatted("Physical Memory: {:.0f} GB", (float)backend->getPhysicalMemory() / (1024*1024*1024));
    ImGui::TextFormatted("Processor Count: {}", backend->getTotalProcessorCount());
}

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
