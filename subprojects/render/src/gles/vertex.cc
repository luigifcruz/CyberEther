#include "render/gles/vertex.hpp"

namespace Render {

GLES::Vertex::Vertex(const Config& config, const GLES& instance)
         : Render::Vertex(config), instance(instance) {
}

Result GLES::Vertex::create() {
    glGenVertexArrays(1, &vao);

    this->begin();
    int i = 0;
    for (auto& buffer : config.buffers) {
        uint usage = GL_STATIC_DRAW;
        switch (buffer.usage) {
            case Vertex::Buffer::Usage::Dynamic:
                usage = GL_DYNAMIC_DRAW;
                break;
            case Vertex::Buffer::Usage::Stream:
                usage = GL_STREAM_DRAW;
                break;
            case Vertex::Buffer::Usage::Static:
                usage = GL_STATIC_DRAW;
                break;
        }

        glGenBuffers(1, &buffer.index);
        glBindBuffer(GL_ARRAY_BUFFER, buffer.index);
        glBufferData(GL_ARRAY_BUFFER, buffer.size * sizeof(float), buffer.data, usage);
        glVertexAttribPointer(i, buffer.stride, GL_FLOAT, GL_FALSE, buffer.stride * sizeof(float), 0);
        glEnableVertexAttribArray(i++);
        vertex_count = buffer.size / buffer.stride;
    }

    if (config.indices.size() != 0) {
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, config.indices.size() * sizeof(uint),
                config.indices.data(), GL_STATIC_DRAW);
        vertex_count = config.indices.size();
    }
    this->end();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::destroy() {
    bool cudaEnabled = false;

    for (auto& buffer : config.buffers) {
        glDeleteBuffers(1, &buffer.index);
    }

    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::begin() {
    glBindVertexArray(vao);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::end() {
    glBindVertexArray(0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::update() {
    this->begin();
    for (auto& buffer : config.buffers) {
        glBindBuffer(GL_ARRAY_BUFFER, buffer.index);
        glBufferSubData(GL_ARRAY_BUFFER, 0, buffer.size * sizeof(float), buffer.data);
    }
    this->end();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

uint GLES::Vertex::buffered() {
    return config.indices.size() != 0;
}

uint GLES::Vertex::count() {
    return vertex_count;
}

}  // namespace Render
