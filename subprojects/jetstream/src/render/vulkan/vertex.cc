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

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating vertex.");

    for (const auto& [buffer, stride] : buffers) {
        JST_CHECK(buffer->create());
        vertexCount = buffer->size() / stride;
    }

    if (indices) {
        JST_CHECK(indices->create());
        vertexCount = indices->size();
    }

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer,
                              const U64& offset) {
    U64 index = offset;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceSize> vertexOffsets;

    for (const auto& [buffer, stride] : buffers) {
        vertexBuffers.push_back(buffer->getHandle());
        vertexOffsets.push_back(index++);
    }

    vkCmdBindVertexBuffers(commandBuffer,
                           0,
                           vertexBuffers.size(),
                           vertexBuffers.data(),
                           vertexOffsets.data());

    if (indices) {
        vkCmdBindIndexBuffer(commandBuffer, indices->getHandle(), 0, VK_INDEX_TYPE_UINT16);
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
