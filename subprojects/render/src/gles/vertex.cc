#include "render/gles/vertex.hpp"

namespace Render {

GLES::Vertex::Vertex(const Config& config, const GLES& instance)
         : Render::Vertex(config), instance(instance) {
    for (const auto& [buffer, stride] : config.buffers) {
        buffers.push_back({std::dynamic_pointer_cast<GLES::Buffer>(buffer), stride});
    }

    if (config.indices) {
        indices = std::dynamic_pointer_cast<GLES::Buffer>(config.indices);
    }
}

Result GLES::Vertex::create() {
    glGenVertexArrays(1, &vao);

    CHECK(this->begin());

    int i = 0;
    for (const auto& [buffer, stride] : buffers) {
        CHECK(buffer->create());
        CHECK(buffer->begin());
        glVertexAttribPointer(i, stride, GL_FLOAT, GL_FALSE, stride * sizeof(float), 0);
        glEnableVertexAttribArray(i++);
        CHECK(buffer->end());
        vertex_count = buffer->size() / stride;
    }

    if (indices) {
        CHECK(indices->create());
        CHECK(indices->begin());
        vertex_count = indices->size();
    }

    CHECK(this->end());

    if (indices) {
        CHECK(indices->end());
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::destroy() {
    for (const auto& [buffer, stride] : buffers) {
        CHECK(buffer->destroy());
    }

    if (indices) {
        CHECK(indices->destroy());
    }

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

}  // namespace Render
