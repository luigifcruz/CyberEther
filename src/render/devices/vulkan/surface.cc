#include "jetstream/render/devices/vulkan/program.hh"
#include "jetstream/render/devices/vulkan/texture.hh"
#include "jetstream/render/devices/vulkan/surface.hh"
#include "jetstream/render/devices/vulkan/kernel.hh"
#include "jetstream/render/devices/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Render {

using Implementation = SurfaceImp<Device::Vulkan>;

Implementation::SurfaceImp(const Config& config) : Surface(config) {
    framebufferResolve = std::dynamic_pointer_cast<
        TextureImp<Device::Vulkan>>(config.framebuffer);

    if (config.multisampled) {
        auto framebuffer_config = framebufferResolve->getConfig();
        framebuffer_config.multisampled = true;
        framebuffer = std::make_shared<TextureImp<Device::Vulkan>>(framebuffer_config);
    }

    for (auto& program : config.programs) {
        programs.push_back(
            std::dynamic_pointer_cast<ProgramImp<Device::Vulkan>>(program)
        );
    }

    for (auto& kernel : config.kernels) {
        kernels.push_back(
            std::dynamic_pointer_cast<KernelImp<Device::Vulkan>>(kernel)
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating surface.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Create Render Pass.

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = framebufferResolve->getPixelFormat();
    colorAttachment.samples = Backend::State<Device::Vulkan>()->getMultisampling();
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.format = framebufferResolve->getPixelFormat();
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = (config.multisampled) ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResolveRef{};
    colorAttachmentResolveRef.attachment = (config.multisampled) ? 1 : 0;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = (config.multisampled) ? &colorAttachmentRef : &colorAttachmentResolveRef;
    subpass.pResolveAttachments = (config.multisampled) ? &colorAttachmentResolveRef : nullptr;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::vector<VkAttachmentDescription> colorAttachments;
    if (config.multisampled) {
        colorAttachments.push_back(colorAttachment);
    }
    colorAttachments.push_back(colorAttachmentResolve);

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = colorAttachments.size();
    renderPassInfo.pAttachments = colorAttachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    JST_VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass), [&]{
        JST_ERROR("[VULKAN] Failed to create render pass.");   
    });

    for (auto& program : programs) {
        JST_CHECK(program->create(renderPass, (config.multisampled) ? framebuffer : framebufferResolve));
    }

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->create());
    }

    JST_CHECK(framebufferResolve->create());
    if (config.multisampled) {
        JST_CHECK(framebuffer->create());
    }

    std::vector<VkImageView> attachments;
    if (config.multisampled) {
        attachments.push_back(framebuffer->getViewHandle());
    }
    attachments.push_back(framebufferResolve->getViewHandle());

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = attachments.size();
    framebufferInfo.pAttachments = attachments.data();
    framebufferInfo.width = framebufferResolve->size().x;
    framebufferInfo.height = framebufferResolve->size().y;
    framebufferInfo.layers = 1;

    JST_VK_CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebufferObject), [&]{
        JST_ERROR("[VULKAN] Failed to create surface framebuffer.");
    });

    requestedSize = framebufferResolve->size();

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying surface.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->destroy());
    }

    for (auto& program : programs) {
        JST_CHECK(program->destroy());
    }

    vkDestroyFramebuffer(device, framebufferObject, nullptr);

    JST_CHECK(framebufferResolve->destroy());
    if (config.multisampled) {
        JST_CHECK(framebuffer->destroy());
    }

    vkDestroyRenderPass(device, renderPass, nullptr);

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer) {
    if (framebufferResolve->size(requestedSize)) {
        JST_VK_CHECK(vkQueueWaitIdle(Backend::State<Device::Vulkan>()->getGraphicsQueue()), [&]{
            JST_ERROR("[VULKAN] Can't wait for graphics queue to finish for surface destruction.");
        });

        if (config.multisampled) {
            framebuffer->size(requestedSize);
        }

        JST_CHECK(destroy());
        JST_CHECK(create());
    }

    // Encode kernels.

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->encode(commandBuffer));
    }

    // Begin render pass.

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebufferObject;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {
        static_cast<U32>(framebufferResolve->size().x),
        static_cast<U32>(framebufferResolve->size().y)
    };

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 0.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Encode programs.

    for (auto& program : programs) {
        JST_CHECK(program->encode(commandBuffer, renderPass));
    }

    vkCmdEndRenderPass(commandBuffer);

    return Result::SUCCESS;
}

const Extent2D<U64>& Implementation::size(const Extent2D<U64>& size) { 
    if (!framebufferResolve) {
        return NullSize;
    }

    requestedSize = size;

    return framebufferResolve->size();
} 

}  // namespace Jetstream::Render
