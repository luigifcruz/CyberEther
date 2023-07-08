#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/render/vulkan/vertex.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Render {

using Implementation = VertexImp<Device::Vulkan>;

Implementation::VertexImp(const Config& config) : Vertex(config) {
    for (const auto& [buffer, stride] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::Vulkan>>(buffer), stride}
        );
    }

    if (config.indices) {
        indices = std::dynamic_pointer_cast<BufferImp<Device::Vulkan>>(config.indices);
    }
}

Result Implementation::create(std::vector<VkVertexInputBindingDescription>& bindingDescription,
                              std::vector<VkVertexInputAttributeDescription>& attributeDescription) {
    JST_DEBUG("[VULKAN] Creating vertex.");

    U32 bindingCount = 0;
    bindingDescription.resize(buffers.size());
    attributeDescription.resize(buffers.size());
    for (const auto& [buffer, stride] : buffers) {
        JST_CHECK(buffer->create());
        vertexCount = buffer->size() / stride;

        bindingDescription[bindingCount].binding = bindingCount;
        bindingDescription[bindingCount].stride = stride * sizeof(F32);
        bindingDescription[bindingCount].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkFormat bindingFormat = VK_FORMAT_UNDEFINED;
        
        switch (stride) {
            case 1:
                bindingFormat = VK_FORMAT_R32_SFLOAT;
                break;       
            case 2:
                bindingFormat = VK_FORMAT_R32G32_SFLOAT;
                break;       
            case 3:
                bindingFormat = VK_FORMAT_R32G32B32_SFLOAT;
                break;       
            case 4:
                bindingFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
                break;       
        }

        attributeDescription[bindingCount].binding = bindingCount;
        attributeDescription[bindingCount].location = bindingCount;
        attributeDescription[bindingCount].format = bindingFormat;
        attributeDescription[bindingCount].offset = 0;

        bindingCount += 1;
    }

    if (indices) {
        JST_CHECK(indices->create());
        vertexCount = indices->size();
    }

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer) {
    U32 bindingCount = 0;
    for (const auto& [buffer, stride] : buffers) {
        const VkBuffer buffers[] = { buffer->getHandle() };
        const VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, bindingCount++, 1, buffers, offsets);
    }

    if (indices) {
        vkCmdBindIndexBuffer(commandBuffer, indices->getHandle(), 0, VK_INDEX_TYPE_UINT32);
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying vertex.");

    for (const auto& [buffer, stride] : buffers) {
        JST_CHECK(buffer->destroy());
    }

    if (indices) {
        JST_CHECK(indices->destroy());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
