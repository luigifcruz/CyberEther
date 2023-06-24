#include "jetstream/render/webgpu/vertex.hh"
#include "jetstream/render/webgpu/draw.hh"

namespace Jetstream::Render {

using Implementation = DrawImp<Device::WebGPU>;

Implementation::DrawImp(const Config& config) : Draw(config) {
    buffer = std::dynamic_pointer_cast<VertexImp<Device::Metal>>(config.buffer);
}

Result Implementation::create(MTL::VertexDescriptor* vertDesc, const U64& offset) {
    JST_DEBUG("Creating Metal draw.");

    JST_CHECK(buffer->create(vertDesc, offset));

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal draw.");

    JST_CHECK(buffer->destroy());

    return Result::SUCCESS;
}

Result Implementation::encode(MTL::RenderCommandEncoder* encoder) {
    MTL::PrimitiveType mode;

    switch (config.mode) {
        case Mode::TRIANGLE_FAN:
            mode = MTL::PrimitiveTypeTriangleStrip;
            break;
        case Draw::Mode::TRIANGLES:
            mode = MTL::PrimitiveTypeTriangle;
            break;
        case Draw::Mode::LINES:
            mode = MTL::PrimitiveTypeLine;
            break;
        case Draw::Mode::LINE_STRIP:
            mode = MTL::PrimitiveTypeLineStrip;
            break;
        case Draw::Mode::POINTS:
            mode = MTL::PrimitiveTypePoint;
            break;
    }

    JST_CHECK(buffer->encode(encoder));

    if (buffer->isBuffered()) {
        encoder->drawIndexedPrimitives(mode, (NS::UInteger)buffer->getVertexCount(),
                MTL::IndexTypeUInt32, buffer->getIndexBuffer(), 0);
    } else {
        encoder->drawPrimitives(mode, (NS::UInteger)0, buffer->getVertexCount());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
