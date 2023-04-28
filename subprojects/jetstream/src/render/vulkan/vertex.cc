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
    JST_DEBUG("Creating Metal vertex.");

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

Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal vertex.");

    for (const auto& [buffer, stride] : buffers) {
        JST_CHECK(buffer->destroy());
    }

    if (indices) {
        JST_CHECK(indices->destroy());
    }

    return Result::SUCCESS;
}

Result Implementation::encode(MTL::RenderCommandEncoder* encoder,
                              const U64& offset) {
    U64 index = offset;

    for (const auto& [buffer, stride] : buffers) {
        encoder->setVertexBuffer(buffer->getHandle(), 0, index++);
    }

    return Result::SUCCESS;
}

const MTL::Buffer* Implementation::getIndexBuffer() {
    return indices->getHandle();
}

}  // namespace Jetstream::Render
