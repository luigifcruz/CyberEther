#include "jetstream/render/devices/vulkan/vertex.hh"
#include "jetstream/render/devices/vulkan/draw.hh"
#include "jetstream/render/devices/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

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
        case Mode::TRIANGLE_STRIP:
            topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
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

    JST_CHECK(buffer->create(bindingDescription,
                             attributeDescription,
                             config.numberOfDraws,
                             config.numberOfInstances));

    // Create Multi-Draw Indirect Buffer

    if (buffer->isBuffered()) {
        for (uint32_t i = 0; i < config.numberOfDraws; i++) {
            VkDrawIndexedIndirectCommand drawCommand{};
            drawCommand.indexCount = buffer->getIndexCount();
            drawCommand.instanceCount = config.numberOfInstances;
            drawCommand.firstIndex = 0;
            drawCommand.vertexOffset = i * (buffer->getIndexCount() - buffer->getVertexCount());
            drawCommand.firstInstance = i * config.numberOfInstances;

            indexedDrawCommands.push_back(drawCommand);
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = indexedDrawCommands.data();
            cfg.elementByteSize = sizeof(VkDrawIndexedIndirectCommand);
            cfg.size = indexedDrawCommands.size();
            cfg.target = Render::Buffer::Target::INDIRECT;

            indexedIndirectBuffer = std::make_shared<Render::BufferImp<Device::Vulkan>>(cfg);
            indexedIndirectBuffer->create();
        }

        indexedIndirectBuffer->update();
    } else {
        for (uint32_t i = 0; i < config.numberOfDraws; i++) {
            VkDrawIndirectCommand drawCommand{};
            drawCommand.vertexCount = buffer->getVertexCount();
            drawCommand.instanceCount = config.numberOfInstances;
            drawCommand.firstVertex = i * buffer->getVertexCount();
            drawCommand.firstInstance = i * config.numberOfInstances;

            drawCommands.push_back(drawCommand);
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = drawCommands.data();
            cfg.elementByteSize = sizeof(VkDrawIndirectCommand);
            cfg.size = drawCommands.size();
            cfg.target = Render::Buffer::Target::INDIRECT;

            indirectBuffer = std::make_shared<Render::BufferImp<Device::Vulkan>>(cfg);
            indirectBuffer->create();
        }

        indirectBuffer->update();
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying draw.");

    if (buffer->isBuffered()) {
        JST_CHECK(indexedIndirectBuffer->destroy());
        indexedDrawCommands.clear();
    } else {
        JST_CHECK(indirectBuffer->destroy());
        drawCommands.clear();
    }

    JST_CHECK(buffer->destroy());

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer) {
    JST_CHECK(buffer->encode(commandBuffer));

    if (buffer->isBuffered()) {
        vkCmdDrawIndexedIndirect(commandBuffer, indexedIndirectBuffer->getHandle(), 0,
                                 indexedDrawCommands.size(), sizeof(VkDrawIndexedIndirectCommand));
    } else {
        vkCmdDrawIndirect(commandBuffer, indirectBuffer->getHandle(), 0,
                          drawCommands.size(), sizeof(VkDrawIndirectCommand));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
