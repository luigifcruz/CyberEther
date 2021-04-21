#include "render/gles/vertex.hpp"

namespace Render {

const float vertices[] = {
    +1.0f, +1.0f, 0.0f, +0.0f, +0.0f,
    +1.0f, -1.0f, 0.0f, +0.0f, +1.0f,
    -1.0f, -1.0f, 0.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, 0.0f, +1.0f, +0.0f,
};

const uint elements[] = {
    0, 1, 2,
    2, 3, 0
};

Result GLES::Vertex::create() {
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::destroy() {
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::start() {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::end() {
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
