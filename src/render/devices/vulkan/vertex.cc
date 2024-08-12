#include "jetstream/render/devices/vulkan/buffer.hh"
#include "jetstream/render/devices/vulkan/vertex.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Render {

using Implementation = VertexImp<Device::Vulkan>;

Implementation::VertexImp(const Config& config) : Vertex(config) {
    for (const auto& [vertex, stride] : config.vertices) {
        vertices.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::Vulkan>>(vertex), stride}
        );
    }

    for (const auto& [instance, stride] : config.instances) {
        instances.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::Vulkan>>(instance), stride}
        );
    }

    if (config.indices) {
        indices = std::dynamic_pointer_cast<BufferImp<Device::Vulkan>>(config.indices);
    }
}

Result Implementation::create(std::vector<VkVertexInputBindingDescription>& bindingDescription,
                              std::vector<VkVertexInputAttributeDescription>& attributeDescription,
                              const U64& numberOfDraws,
                              const U64&) {
    JST_DEBUG("[VULKAN] Creating vertex.");

    for (const auto& [vertex, stride] : vertices) {
        VkVertexInputBindingDescription binding = {};
        binding.binding = bindingDescription.size();
        binding.stride = stride * sizeof(F32);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindingDescription.push_back(binding);

        for (U32 i = 0; i < stride; i += 4) {
            VkFormat format = VK_FORMAT_UNDEFINED;

            switch (i % 4) {
                case 1:
                    format = VK_FORMAT_R32_SFLOAT;
                    break;
                case 2:
                    format = VK_FORMAT_R32G32_SFLOAT;
                    break;
                case 3:
                    format = VK_FORMAT_R32G32B32_SFLOAT;
                    break;
                case 0:
                    format = VK_FORMAT_R32G32B32A32_SFLOAT;
                    break;
            }

            VkVertexInputAttributeDescription attribute = {};
            attribute.binding = binding.binding;
            attribute.location = attributeDescription.size();
            attribute.format = format;
            attribute.offset = i * sizeof(F32);
            attributeDescription.push_back(attribute);
        }
    }

    for (const auto& [instance, stride] : instances) {
        VkVertexInputBindingDescription binding = {};
        binding.binding = bindingDescription.size();
        binding.stride = stride * sizeof(F32);
        binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
        bindingDescription.push_back(binding);

        for (U32 i = 0; i < stride; i += 4) {
            VkFormat format = VK_FORMAT_UNDEFINED;

            switch (i % 4) {
                case 1:
                    format = VK_FORMAT_R32_SFLOAT;
                    break;
                case 2:
                    format = VK_FORMAT_R32G32_SFLOAT;
                    break;
                case 3:
                    format = VK_FORMAT_R32G32B32_SFLOAT;
                    break;
                case 0:
                    format = VK_FORMAT_R32G32B32A32_SFLOAT;
                    break;
            }

            VkVertexInputAttributeDescription attribute = {};
            attribute.binding = binding.binding;
            attribute.location = attributeDescription.size();
            attribute.format = format;
            attribute.offset = i * sizeof(F32);
            attributeDescription.push_back(attribute);
        }
    }

    const auto& [vertex, stride] = vertices[0];
    vertexCount = vertex->byteSize() / sizeof(F32) / numberOfDraws / stride;

    if (indices) {
        indexCount = indices->size();
    }

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer) {
    // TODO: Is binding multiple buffers at once a good idea?
    //       Not sure if other APIs support this.

    U32 bindingCount = 0;

    for (const auto& [vertex, stride] : vertices) {
        const VkBuffer buffers[] = { vertex->getHandle() };
        const VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, bindingCount++, 1, buffers, offsets);
    }

    for (const auto& [instance, stride] : instances) {
        const VkBuffer buffers[] = { instance->getHandle() };
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

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
