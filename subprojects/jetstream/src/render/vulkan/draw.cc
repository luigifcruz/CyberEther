#include "jetstream/render/vulkan/vertex.hh"
#include "jetstream/render/vulkan/draw.hh"

namespace Jetstream::Render {

using Implementation = DrawImp<Device::Vulkan>;

Implementation::DrawImp(const Config& config) : Draw(config) {
    buffer = std::dynamic_pointer_cast<VertexImp<Device::Vulkan>>(config.buffer);
}

Result Implementation::create() {
    JST_DEBUG("Creating Metal draw.");

    JST_CHECK(buffer->create());

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal draw.");

    JST_CHECK(buffer->destroy());

    return Result::SUCCESS;
}

Result Implementation::encode(MTL::RenderCommandEncoder* encoder,
                              const U64& offset) {
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

    JST_CHECK(buffer->encode(encoder, offset));

    if (buffer->isBuffered()) {
        encoder->drawIndexedPrimitives(mode, (NS::UInteger)buffer->getVertexCount(),
                MTL::IndexTypeUInt32, buffer->getIndexBuffer(), 0);
    } else {
        encoder->drawPrimitives(mode, (NS::UInteger)0, buffer->getVertexCount());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
