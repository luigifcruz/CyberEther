#include "render/metal/vertex.hpp"

namespace Render {

Metal::Vertex::Vertex(const Config& config, const Metal& instance)
         : Render::Vertex(config), instance(instance) {
    for (const auto& [buffer, stride] : config.buffers) {
        buffers.push_back({std::dynamic_pointer_cast<Metal::Buffer>(buffer), stride});
    }

    if (config.indices) {
        indices = std::dynamic_pointer_cast<Metal::Buffer>(config.indices);
    }
}

Result Metal::Vertex::create() {
    for (const auto& [buffer, stride] : buffers) {
        buffer->create();
        vertex_count = buffer->size() / stride;
    }

    if (indices) {
        indices->create();
        vertex_count = indices->size();
    }

    return Result::SUCCESS;
}

Result Metal::Vertex::destroy() {
    for (const auto& [buffer, stride] : buffers) {
        buffer->destroy();
    }

    if (indices) {
        indices->destroy();
    }

    return Result::SUCCESS;
}

Result Metal::Vertex::encode(MTL::RenderCommandEncoder* encoder,
                             const std::size_t& offset) {
    std::size_t index = offset;

    for (const auto& [buffer, stride] : buffers) {
        encoder->setVertexBuffer(buffer->getHandle(), 0, index++);
    }

    return Result::SUCCESS;
}

}  // namespace Render
