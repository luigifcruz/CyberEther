#include "render/gles/draw.hpp"
#include "render/gles/vertex.hpp"

namespace Render {

Result GLES::Draw::create() {
    buffer = std::dynamic_pointer_cast<GLES::Vertex>(cfg.buffer);
    buffer->create();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Draw::destroy() {
    buffer->destroy();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Draw::draw() {
    buffer->begin();

    uint mode = GL_TRIANGLES;
    switch (cfg.mode) {
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
        case Draw::Mode::LineLoop:
            mode = GL_LINE_LOOP;
            break;
    }

    if (buffer->buffered()) {
        glDrawElements(mode, buffer->count(), GL_UNSIGNED_INT, 0);
    } else {
        glDrawArrays(mode, 0, buffer->count());
    }

    buffer->end();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
