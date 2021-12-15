#include "render/metal/draw.hpp"
#include "render/metal/vertex.hpp"

namespace Render {

Result Metal::Draw::create() {
    buffer = std::dynamic_pointer_cast<Metal::Vertex>(cfg.buffer);
    buffer->create();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Draw::destroy() {
    buffer->destroy();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Draw::draw() {
    buffer->begin();

    /*

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

    */

    if (buffer->buffered()) {
        //glDrawElements(mode, buffer->count(), GL_UNSIGNED_INT, 0);
    } else {
        //glDrawArrays(mode, 0, buffer->count());
    }

    buffer->end();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
