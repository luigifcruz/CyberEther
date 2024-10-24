#include "jetstream/render/devices/metal/vertex.hh"
#include "jetstream/render/devices/metal/draw.hh"
#include "jetstream/render/devices/metal/buffer.hh"

namespace Jetstream::Render {

using Implementation = DrawImp<Device::Metal>;

Implementation::DrawImp(const Config& config) : Draw(config) {
    buffer = std::dynamic_pointer_cast<VertexImp<Device::Metal>>(config.buffer);
}

Result Implementation::create(MTL::VertexDescriptor* vertDesc, const U64& offset) {
    JST_DEBUG("[Metal] Creating draw.");

    JST_CHECK(buffer->create(vertDesc, 
                             config.numberOfDraws, 
                             config.numberOfInstances, 
                             offset));

    // Create multi-draw indirect buffer.

    if (buffer->isBuffered()) {
        for (U64 i = 0; i < config.numberOfDraws; i++) {
            MTL::DrawIndexedPrimitivesIndirectArguments drawCommand = {};
            drawCommand.indexCount = buffer->getIndexCount();
            drawCommand.instanceCount = config.numberOfInstances;
            drawCommand.indexStart = 0;
            drawCommand.baseVertex = i * (buffer->getIndexCount() - buffer->getVertexCount());
            drawCommand.baseInstance = i * config.numberOfInstances;

            indexedDrawCommands.push_back(drawCommand);
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = indexedDrawCommands.data();
            cfg.elementByteSize = sizeof(MTL::DrawIndexedPrimitivesIndirectArguments);
            cfg.size = indexedDrawCommands.size();
            cfg.target = Render::Buffer::Target::INDIRECT;

            indexedIndirectBuffer = std::make_shared<Render::BufferImp<Device::Metal>>(cfg);
            indexedIndirectBuffer->create();
        }

        indexedIndirectBuffer->update();
    } else {
        for (U64 i = 0; i < config.numberOfDraws; i++) {
            MTL::DrawPrimitivesIndirectArguments drawCommand = {};
            drawCommand.vertexCount = buffer->getVertexCount();
            drawCommand.instanceCount = config.numberOfInstances;
            drawCommand.vertexStart = i * buffer->getVertexCount();
            drawCommand.baseInstance = i * config.numberOfInstances;

            drawCommands.push_back(drawCommand);
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = drawCommands.data();
            cfg.elementByteSize = sizeof(MTL::DrawPrimitivesIndirectArguments);
            cfg.size = drawCommands.size();
            cfg.target = Render::Buffer::Target::INDIRECT;

            indirectBuffer = std::make_shared<Render::BufferImp<Device::Metal>>(cfg);
            indirectBuffer->create();
        }

        indirectBuffer->update();
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[Metal] Destroying draw.");

    if (buffer->isBuffered()) {
        JST_CHECK(indexedIndirectBuffer->destroy());
        indexedDrawCommands.clear();
    } else {
        JST_CHECK(indirectBuffer->destroy());
        drawCommands.clear();
    }

    JST_CHECK(buffer->destroy());

    return Result::SUCCESS;
}

Result Implementation::encode(MTL::RenderCommandEncoder* encoder) {
    MTL::PrimitiveType mode;

    switch (config.mode) {
        case Mode::TRIANGLE_STRIP:
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

    for (U64 i = 0; i < config.numberOfDraws; i++) {
        if (buffer->isBuffered()) {
            encoder->drawIndexedPrimitives(mode,
                                           MTL::IndexTypeUInt32, 
                                           buffer->getIndexBuffer(), 
                                           0,
                                           indexedIndirectBuffer->getHandle(),
                                           i * sizeof(MTL::DrawIndexedPrimitivesIndirectArguments));
        } else {
            encoder->drawPrimitives(mode, 
                                    indirectBuffer->getHandle(),
                                    i * sizeof(MTL::DrawPrimitivesIndirectArguments));
        }
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
