#include "render/metal/vertex.hpp"

namespace Render {

Result Metal::Vertex::create() {
    /*
    glGenVertexArrays(1, &vao);

    this->begin();
    int i = 0;
    bool cudaEnabled = false;
    for (auto& buffer : cfg.buffers) {
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
        auto ptr = (buffer.cudaInterop) ? nullptr : buffer.data;
        glBufferData(GL_ARRAY_BUFFER, buffer.size * sizeof(float), ptr, usage);
        glVertexAttribPointer(i, buffer.stride, GL_FLOAT, GL_FALSE, buffer.stride * sizeof(float), 0);
        glEnableVertexAttribArray(i++);
        vertex_count = buffer.size / buffer.stride;
    }

    if (cfg.indices.size() != 0) {
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, cfg.indices.size() * sizeof(uint),
                cfg.indices.data(), GL_STATIC_DRAW);
        vertex_count = cfg.indices.size();
    }
    this->end();
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Vertex::destroy() {
    /*
    bool cudaEnabled = false;

    for (auto& buffer : cfg.buffers) {
        glDeleteBuffers(1, &buffer.index);
    }

    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Vertex::begin() {
    //glBindVertexArray(vao);

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Vertex::end() {
    //glBindVertexArray(0);

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
