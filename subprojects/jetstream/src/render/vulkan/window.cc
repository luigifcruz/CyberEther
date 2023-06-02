#include "jetstream/render/vulkan/window.hh"
#include "jetstream/render/vulkan/surface.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::Vulkan>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Provider<Device::Vulkan>>& viewport)
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

    JST_CHECK(viewport->create());

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

    // Create children.

    for (auto& surface : surfaces) {
        JST_CHECK(surface->create());
    }

    if (config.imgui) {
        JST_CHECK(createImgui());
    }

    statsData.droppedFrames = 0;

    // Create command pool.

    Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = indices.graphicFamily.value();
    poolInfo.flags = 0;

    JST_VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), [&]{
        JST_FATAL("[VULKAN] Failed to create graphics command pool.");
    });

    // Create command buffers.

    commandBuffers.resize(swapchainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    JST_VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()), [&]{
        JST_FATAL("[VULKAN] Can't create render buffers.");
    });

    for (size_t i = 0; i < commandBuffers.size(); i++) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        JST_VK_CHECK(vkBeginCommandBuffer(commandBuffers[i], &beginInfo), [&]{
            JST_FATAL("[VULKAN] Can't begin command buffer.");     
        });

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapchainFramebuffers[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchainExtent;

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdEndRenderPass(commandBuffers[i]);

        JST_VK_CHECK(vkEndCommandBuffer(commandBuffers[i]), [&]{
            JST_FATAL("[VULKAN] Can't end command buffer.");
        });
    }
    
    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying window.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    for (auto& surface : surfaces) {
        JST_CHECK(surface->destroy());
    }

    if (config.imgui) {
        JST_CHECK(destroyImgui());
    } 

    for (auto framebuffer : swapchainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    vkDestroyRenderPass(device, renderPass, nullptr);
    
    JST_CHECK(viewport->destroy());

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    JST_DEBUG("[VULKAN] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    this->ApplyImGuiTheme(config.scale);

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    JST_CHECK(viewport->createImgui());

    auto& backend = Backend::State<Device::Vulkan>();
    ImGui_ImplVulkan_InitInfo init_info = {
        .Instance = backend->getInstance(),
        .PhysicalDevice = backend->getPhysicalDevice(),
        .Device = backend->getDevice(),
        .QueueFamily = Backend::FindQueueFamilies(backend->getPhysicalDevice()).graphicFamily.value(),
        .Queue = backend->getGraphicsQueue(),
        .PipelineCache = {},
        .DescriptorPool = {},
        .Subpass = {},
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
    ImGui::NewFrame();

    return Result::SUCCESS;
}

Result Implementation::endImgui() {
    ImGui::Render();
    // ImGui_ImplVulkan_RenderDrawData();

    return Result::SUCCESS;
}

Result Implementation::begin() {
    // innerPool = NS::AutoreleasePool::alloc()->init();

    // drawable = static_cast<CA::MetalDrawable*>(viewport->nextDrawable());

    // if (!drawable) {
    //     statsData.droppedFrames += 1;
    //     return Result::SKIP;
    // }

    // auto colorAttachDescriptor = renderPassDescriptor->colorAttachments()->object(0);
    // colorAttachDescriptor->setTexture(drawable->texture());
    // colorAttachDescriptor->setLoadAction(MTL::LoadActionClear);
    // colorAttachDescriptor->setStoreAction(MTL::StoreActionStore);
    // colorAttachDescriptor->setClearColor(MTL::ClearColor(0, 0, 0, 1.0));

    // if (config.imgui) {
    //     JST_CHECK(beginImgui());
    // }

    return Result::SUCCESS;
}

Result Implementation::end() {
    // commandBuffer = commandQueue->commandBuffer();

    // for (auto &surface : surfaces) {
    //     JST_CHECK(surface->draw(commandBuffer));
    // }

    // if (config.imgui) {
    //     JST_CHECK(endImgui());
    // }

    // commandBuffer->presentDrawable(drawable);
    // commandBuffer->commit();
    // commandBuffer->waitUntilCompleted();

    // innerPool->release();

    return Result::SUCCESS;
}

void Implementation::drawDebugMessage() const {
    auto& backend = Backend::State<Device::Vulkan>();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::Text("FPS: %.1f Hz", io.Framerate);
    ImGui::Text("Device Name: %s", backend->getDeviceName().c_str());
    ImGui::Text("Has Unified Memory: %s", backend->hasUnifiedMemory() ? "YES" : "NO");
    ImGui::Text("Physical Memory: %.00f GB", (float)backend->getPhysicalMemory() / (1024*1024*1024));
    ImGui::Text("Processor Count: %lu", backend->getTotalProcessorCount());
}

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
