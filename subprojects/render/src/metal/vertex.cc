#include "render/metal/vertex.hpp"

namespace Render {

Result Metal::Vertex::create() {
    MTL::ResourceOptions usage;

    switch (cfg.buffers[0].usage) {
        case Vertex::Buffer::Usage::Dynamic:
            usage = MTL::ResourceStorageModeManaged;
            break;
        case Vertex::Buffer::Usage::Stream:
            usage = MTL::ResourceStorageModeManaged;
            break;
        case Vertex::Buffer::Usage::Static:
            usage = MTL::ResourceOptionCPUCacheModeDefault;
            break;
    }

    for (auto& buffer : cfg.buffers) {
        auto size = buffer.size * sizeof(float);
        auto tmp = inst.device->newBuffer(buffer.data, size, usage);
        vertexBuffers.push_back(tmp);
        vertex_count = buffer.size / buffer.stride;
    }

    if (cfg.indices.size() > 0) {
        indexBuffer = inst.device->newBuffer(cfg.indices.data(),
                cfg.indices.size() * sizeof(uint),
                MTL::ResourceOptionCPUCacheModeDefault);
        vertex_count = cfg.indices.size();
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Vertex::destroy() {
    for (auto& buffer : vertexBuffers) {
        buffer->release();
    }

    if (indexBuffer) {
        indexBuffer->release();
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Vertex::encode(MTL::RenderCommandEncoder* encoder) {
    std::size_t index = 0;

    for (auto& buffer : vertexBuffers) {
        encoder->setVertexBuffer(buffer, 0, index);
        index += 1;
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Vertex::update() {
    for (auto& buffer : vertexBuffers) {
        // TODO: This is bad.
        auto gpuBuff = buffer->contents();
        memcpy(gpuBuff, cfg.buffers[0].data, buffer->length());
        buffer->didModifyRange(NS::Range(0, buffer->length()));
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
