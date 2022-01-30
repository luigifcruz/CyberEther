#include "render/gles/draw.hpp"
#include "render/gles/vertex.hpp"

namespace Render {

GLES::Draw::Draw(const Config& config, const GLES& instance)
         : Render::Draw(config), instance(instance) {
    buffer = std::dynamic_pointer_cast<GLES::Vertex>(config.buffer);
}

Result GLES::Draw::create() {
    CHECK(buffer->create());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Draw::destroy() {
    CHECK(buffer->destroy());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Draw::draw() {
    CHECK(buffer->begin());

    uint mode = GL_TRIANGLES;
    switch (config.mode) {
        case Draw::Mode::TriangleFan:
            mode = GL_TRIANGLE_FAN;
            break;
        case Draw::Mode::Triangles:
            mode = GL_TRIANGLES;
            break;
        case Draw::Mode::Lines:
            mode = GL_LINES;
            break;
        case Draw::Mode::LineStrip:
            mode = GL_LINE_STRIP;
            break;
        case Draw::Mode::Points:
            mode = GL_POINTS;
            break;
    }

    if (buffer->isBuffered()) {
        glDrawElements(mode, buffer->getVertexCount(), GL_UNSIGNED_INT, 0);
    } else {
        glDrawArrays(mode, 0, buffer->getVertexCount());
    }

    CHECK(buffer->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

}  // namespace Render
