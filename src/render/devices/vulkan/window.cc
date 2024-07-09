#include "jetstream/render/devices/vulkan/window.hh"
#include "jetstream/render/devices/vulkan/surface.hh"
#include "jetstream/render/devices/vulkan/buffer.hh"
#include "jetstream/render/devices/vulkan/texture.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#include "tools/imgui_impl_vulkan.h"

const size_t MAX_FRAMES_IN_FLIGHT = 2;

namespace Jetstream::Render {

using Implementation = WindowImp<Device::Vulkan>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Adapter<Device::Vulkan>>& viewport)
         : Window(config), viewport(viewport) {
}

template<typename T>
Result Implementation::bindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container) {
    // Cast generic resource.
    auto _resource = std::dynamic_pointer_cast<T>(resource);

    // Create the resource.
    JST_CHECK(_resource->create());

    // Add resource to container.
    container.push_back(_resource);

    return Result::SUCCESS;
}

template<typename T>
Result Implementation::unbindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container) {
    // Sychronize all outstanding command buffers.
    JST_VK_CHECK(vkQueueWaitIdle(Backend::State<Device::Vulkan>()->getGraphicsQueue()), [&]{
        JST_ERROR("[VULKAN] Can't wait for queue to complete.");
    });

    // Cast generic resource.
    auto _resource = std::dynamic_pointer_cast<T>(resource);

    // Destroy the resource.
    JST_CHECK(_resource->destroy());

    // Remove resource from container.
    container.erase(std::remove(container.begin(), container.end(), _resource), container.end());

    return Result::SUCCESS;
}

Result Implementation::bindBuffer(const std::shared_ptr<Buffer>& buffer) {
    return bindResource<BufferImp<Device::Vulkan>>(buffer, buffers);
}

Result Implementation::unbindBuffer(const std::shared_ptr<Buffer>& buffer) {
    return unbindResource<BufferImp<Device::Vulkan>>(buffer, buffers);
}

Result Implementation::bindTexture(const std::shared_ptr<Texture>& texture) {
    return bindResource<TextureImp<Device::Vulkan>>(texture, textures);
}

Result Implementation::unbindTexture(const std::shared_ptr<Texture>& texture) {
    return unbindResource<TextureImp<Device::Vulkan>>(texture, textures);
}

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    return bindResource<SurfaceImp<Device::Vulkan>>(surface, surfaces);
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    return unbindResource<SurfaceImp<Device::Vulkan>>(surface, surfaces);
}

Result Implementation::underlyingCreate() {
    JST_DEBUG("[VULKAN] Creating window.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& headless = Backend::State<Device::Vulkan>()->headless();
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();

    // Reseting internal variables.
 
    statsData.droppedFrames = 0;

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

    JST_VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass), [&]{
        JST_ERROR("[VULKAN] Failed to create render pass.");   
    });

    // Create command pool.

    Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = indices.graphicFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    JST_VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), [&]{
        JST_ERROR("[VULKAN] Failed to create graphics command pool.");
    });

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

    // Create submodules.

    JST_CHECK(createImgui());
    JST_CHECK(createFramebuffer());
    JST_CHECK(createSynchronizationObjects());

    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[VULKAN] Destroying window.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    JST_VK_CHECK(vkQueueWaitIdle(Backend::State<Device::Vulkan>()->getGraphicsQueue()), [&]{
        JST_ERROR("[VULKAN] Can't wait for graphics queue to destroy window.");
    });

    JST_CHECK(destroyImgui());

    JST_CHECK(destroySynchronizationObjects());

    if (!buffers.empty() || !textures.empty() || !surfaces.empty()) {
        JST_WARN("[VULKAN] Resources are still bounded to this window "
                 "(buffers={}, textures={}, surfaces={}).", 
                 buffers.size(), textures.size(), surfaces.size());
    }

    vkFreeCommandBuffers(device, commandPool, commandBuffers.size(), commandBuffers.data());

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyRenderPass(device, renderPass, nullptr);

    JST_CHECK(destroyFramebuffer());

    return Result::SUCCESS;
}

Result Implementation::recreate() {
    JST_CHECK(destroyFramebuffer());
    JST_CHECK(viewport->destroySwapchain());
    JST_CHECK(viewport->createSwapchain());
    JST_CHECK(createFramebuffer());

    return Result::SUCCESS;
}

Result Implementation::createFramebuffer() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    const auto& swapchainExtent = viewport->getSwapchainExtent();

    swapchainFramebuffers.resize(viewport->getSwapchainImageViewsCount());

    for (size_t i = 0; i < swapchainFramebuffers.size(); i++) {
        VkImageView attachments[] = {  viewport->getSwapchainImageView(i) };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapchainExtent.width;
        framebufferInfo.height = swapchainExtent.height;
        framebufferInfo.layers = 1;

        JST_VK_CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]), [&]{
            JST_ERROR("[VULKAN] Failed to create swapchain framebuffer.");
        });
    }

    return Result::SUCCESS;
}

Result Implementation::destroyFramebuffer() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    JST_VK_CHECK(vkQueueWaitIdle(Backend::State<Device::Vulkan>()->getGraphicsQueue()), [&]{
        JST_ERROR("[VULKAN] Can't wait for graphics queue to destroy framebuffer.");
    });

    for (auto framebuffer : swapchainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    JST_DEBUG("[VULKAN] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    JST_CHECK(viewport->createImgui());

    this->scaleStyle(*viewport);

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
        .MinImageCount = static_cast<U32>(viewport->getSwapchainImageViewsCount()),
        .ImageCount = static_cast<U32>(viewport->getSwapchainImageViewsCount()),
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .UseDynamicRendering = false,
        .ColorAttachmentFormat = VK_FORMAT_UNDEFINED,
        .Allocator = {},
        .CheckVkResultFn = nullptr,
        .MinAllocationSize = 0,
    };
    ImGui_ImplVulkan_Init(&init_info, renderPass); 

    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("[VULKAN] Destroying ImGui.");

    ImGui_ImplVulkan_Shutdown();
    JST_CHECK(viewport->destroyImgui());
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

Result Implementation::beginImgui() {
    ImGui_ImplVulkan_NewFrame();

    this->scaleStyle(*viewport);

    ImGui::NewFrame();

    return Result::SUCCESS;
}

Result Implementation::endImgui() {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), currentCommandBuffer);

    return Result::SUCCESS;
}

Result Implementation::underlyingBegin() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Wait for a frame to be available.
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

    // Refresh command buffer and begin new render pass.

    VkCommandBufferBeginInfo commandBufferBeginInfo = {};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    JST_VK_CHECK(vkBeginCommandBuffer(currentCommandBuffer, &commandBufferBeginInfo), [&]{
        JST_ERROR("[VULKAN] Can't begin command buffer.");     
    });

    VkRenderPassBeginInfo renderPassBeginInfo = {};
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

    // Begin secondary renders.

    JST_CHECK(beginImgui());

    return Result::SUCCESS;
}

Result Implementation::underlyingEnd() {
    // End secondary renders.

    JST_CHECK(endImgui());

    // End render pass and command buffer.

    vkCmdEndRenderPass(currentCommandBuffer);

    JST_VK_CHECK(vkEndCommandBuffer(currentCommandBuffer), [&]{
        JST_ERROR("[VULKAN] Can't end command buffer.");
    });

    // Reset synchronization fences and submit to queue.

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
        JST_ERROR("[VULKAN] Failed to submit draw command buffer.");
        return Result::ERROR;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    // Commit framebuffer to viewport.

    const auto& result = viewport->commitDrawable(signalSemaphores);
    if (result == Result::RECREATE) {
        return recreate();
    }
    return result;
}

Result Implementation::underlyingSynchronize() {
    JST_VK_CHECK(vkQueueWaitIdle(Backend::State<Device::Vulkan>()->getGraphicsQueue()), [&]{
        JST_ERROR("[VULKAN] Can't synchronize graphics queue.");
    });

    return Result::SUCCESS;
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
            JST_ERROR("[VULKAN] Failed to create synchronization objects.");
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

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Device Name:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{}", backend->getDeviceName());

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Device Memory:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{:.0f} GB", (float)backend->getPhysicalMemory() / (1024*1024*1024));
}

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
