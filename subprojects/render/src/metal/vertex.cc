#include "render/metal/vertex.hpp"

namespace Render {

Result Metal::Vertex::create() {
    for (auto& buffer : cfg.buffers) {
        // TODO: implement modes
        auto tmp = inst.device->newBuffer(buffer.data, buffer.size * sizeof(float),
                MTL::ResourceOptionCPUCacheModeDefault);
        vertexBuffers.push_back(tmp);
        vertex_count = buffer.size / buffer.stride;
    }

    if (cfg.indices.size() > 0) {
        // TODO: implement modes
        indexBuffer = inst.device->newBuffer(cfg.indices.data(),
                cfg.indices.size() * sizeof(uint),
                MTL::ResourceOptionCPUCacheModeDefault);
        vertex_count = cfg.indices.size();
    }

    fmt::print("vertex ok!\n");

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
    /*
    this->begin();
    for (auto& buffer : cfg.buffers) {
        glBindBuffer(GL_ARRAY_BUFFER, buffer.index);
        glBufferSubData(GL_ARRAY_BUFFER, 0, buffer.size * sizeof(float), buffer.data);
    }
    this->end();
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

uint Metal::Vertex::buffered() {
    return cfg.indices.size() != 0;
}

uint Metal::Vertex::count() {
    return vertex_count;
}

} // namespace Render
