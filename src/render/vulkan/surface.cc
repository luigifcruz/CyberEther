#include "jetstream/render/vulkan/program.hh"
#include "jetstream/render/vulkan/texture.hh"
#include "jetstream/render/vulkan/surface.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Render {

using Implementation = SurfaceImp<Device::Vulkan>;

Implementation::SurfaceImp(const Config& config) : Surface(config) {
    framebuffer = std::dynamic_pointer_cast<
        TextureImp<Device::Vulkan>>(config.framebuffer);

    for (auto& program : config.programs) {
        programs.push_back(
            std::dynamic_pointer_cast<ProgramImp<Device::Vulkan>>(program)
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating surface.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Create Render Pass.

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = framebuffer->getPixelFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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

    for (auto& program : programs) {
        JST_CHECK(program->create(renderPass, framebuffer));
    }

    JST_CHECK(framebuffer->create());

    VkImageView attachments[] = { framebuffer->getViewHandle() };

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = framebuffer->size().width;
    framebufferInfo.height = framebuffer->size().height;
    framebufferInfo.layers = 1;

    JST_VK_CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebufferObject), [&]{
        JST_ERROR("[VULKAN] Failed to create surface framebuffer.");
    });

    requestedSize = framebuffer->size();

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying surface.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    for (auto& program : programs) {
        JST_CHECK(program->destroy());
    }

    vkDestroyFramebuffer(device, framebufferObject, nullptr);

    JST_CHECK(framebuffer->destroy());

    vkDestroyRenderPass(device, renderPass, nullptr);

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer) {
    if (framebuffer->size(requestedSize)) {
        JST_VK_CHECK(vkQueueWaitIdle(Backend::State<Device::Vulkan>()->getGraphicsQueue()), [&]{
            JST_ERROR("[VULKAN] Can't wait for graphics queue to finish for surface destruction.");
        });
            
        JST_CHECK(destroy());
        JST_CHECK(create());
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebufferObject;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = VkExtent2D{
            static_cast<U32>(framebuffer->size().width),
            static_cast<U32>(framebuffer->size().height)
        };

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 0.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    for (auto& program : programs) {
        JST_CHECK(program->encode(commandBuffer, renderPass));
    }

    vkCmdEndRenderPass(commandBuffer);

    return Result::SUCCESS;
}

const Size2D<U64>& Implementation::size(const Size2D<U64>& size) { 
    if (!framebuffer) {
        return NullSize;
    }

    requestedSize = size;

    return framebuffer->size();
} 

}  // namespace Jetstream::Render
