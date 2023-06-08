#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/render/vulkan/vertex.hh"

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
                              std::vector<VkVertexInputAttributeDescription>& attributeDescrition) {
    JST_DEBUG("[VULKAN] Creating vertex.");

    U32 bindingCount = 0;
    for (const auto& [buffer, stride] : buffers) {
        JST_CHECK(buffer->create());
        vertexCount = buffer->size() / stride;

        VkVertexInputBindingDescription _bindingDescription{};
        _bindingDescription.binding = bindingCount;
        _bindingDescription.stride = stride * sizeof(F32);
        _bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindingDescription.push_back(_bindingDescription);

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

        VkVertexInputAttributeDescription _attributeDescription{};
        _attributeDescription.binding = bindingCount;
        _attributeDescription.location = bindingCount;
        _attributeDescription.format = bindingFormat;
        _attributeDescription.offset = 0;
        attributeDescrition.push_back(_attributeDescription);

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
