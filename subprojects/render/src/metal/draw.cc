#include "render/metal/draw.hpp"
#include "render/metal/vertex.hpp"

namespace Render {

Result Metal::Draw::create() {
    buffer = std::dynamic_pointer_cast<Metal::Vertex>(cfg.buffer);
    buffer->create();
    fmt::print("draw ok!\n");

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Draw::destroy() {
    buffer->destroy();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Draw::encode(MTL::RenderCommandEncoder* encoder) {
    MTL::PrimitiveType mode;

    switch (cfg.mode) {
        case Draw::Mode::TriangleFan:
            mode = MTL::PrimitiveTypeTriangleStrip;
            break;
        case Draw::Mode::Triangles:
            mode = MTL::PrimitiveTypeTriangle;
            break;
        case Draw::Mode::Lines:
            mode = MTL::PrimitiveTypeLine;
            break;
        case Draw::Mode::LineStrip:
            mode = MTL::PrimitiveTypeLineStrip;
            break;
        case Draw::Mode::Points:
            mode = MTL::PrimitiveTypePoint;
            break;
        case Draw::Mode::LineLoop:
            std::cerr << "Not implemented" << std::endl;
            mode = MTL::PrimitiveTypeLine;
            break;
    }

    buffer->encode(encoder);

    if (buffer->buffered()) {
        encoder->drawIndexedPrimitives(mode, (NS::UInteger)buffer->count(),
                MTL::IndexTypeUInt16, buffer->getIndexBuffer(), 0);
    } else {
        encoder->drawPrimitives(mode, (NS::UInteger)0, buffer->count());
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
