#include "render/metal/draw.hpp"
#include "render/metal/vertex.hpp"

namespace Render {

Result Metal::Draw::create() {
    buffer = std::dynamic_pointer_cast<Metal::Vertex>(cfg.buffer);
    CHECK(buffer->create());

    return Result::SUCCESS;
}

Result Metal::Draw::destroy() {
    CHECK(buffer->destroy());

    return Result::SUCCESS;
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
    }

    CHECK(buffer->encode(encoder));

    if (buffer->isBuffered()) {
        encoder->drawIndexedPrimitives(mode, (NS::UInteger)buffer->getVertexCount(),
                MTL::IndexTypeUInt16, buffer->getIndexBuffer(), 0);
    } else {
        encoder->drawPrimitives(mode, (NS::UInteger)0, buffer->getVertexCount());
    }

    return Result::SUCCESS;
}

} // namespace Render
