#include "jetstream/render/vulkan/vertex.hh"
#include "jetstream/render/vulkan/draw.hh"

namespace Jetstream::Render {

using Implementation = DrawImp<Device::Vulkan>;

Implementation::DrawImp(const Config& config) : Draw(config) {
    buffer = std::dynamic_pointer_cast<VertexImp<Device::Vulkan>>(config.buffer);
}

Result Implementation::create(std::vector<VkVertexInputBindingDescription>& bindingDescription,
                              std::vector<VkVertexInputAttributeDescription>& attributeDescription,
                              VkPipelineInputAssemblyStateCreateInfo& inputAssembly) {
    JST_DEBUG("[VULKAN] Creating draw.");

    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        
    switch (config.mode) {
        case Mode::TRIANGLE_FAN:
            topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
            break;
        case Mode::TRIANGLES:
            topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            break;
        case Mode::LINES:
            topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            break;
        case Mode::LINE_STRIP:
            topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            break;
        case Mode::POINTS:
            topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            break;
    }

    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    JST_CHECK(buffer->create(bindingDescription, attributeDescription));

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer) {
    JST_CHECK(buffer->encode(commandBuffer));

    if (buffer->isBuffered()) {
        vkCmdDrawIndexed(commandBuffer, buffer->getIndicesCount(), 1, 0, 0, 0);
    } else {
        vkCmdDraw(commandBuffer, buffer->getIndicesCount(), 1, 0, 0);
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying draw.");

    JST_CHECK(buffer->destroy());

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
