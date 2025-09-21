#include "jetstream/render/devices/webgpu/vertex.hh"
#include "jetstream/render/devices/webgpu/draw.hh"
#include "jetstream/render/devices/webgpu/buffer.hh"

namespace Jetstream::Render {

using Implementation = DrawImp<Device::WebGPU>;

Implementation::DrawImp(const Config& config) : Draw(config) {
    buffer = std::dynamic_pointer_cast<VertexImp<Device::WebGPU>>(config.buffer);
}

Result Implementation::create(WGPURenderPipelineDescriptor& renderDescriptor) {
    JST_DEBUG("[WebGPU] Creating draw.");

    WGPUPrimitiveTopology topology = WGPUPrimitiveTopology_PointList;

    switch (config.mode) {
        case Mode::TRIANGLE_STRIP:
            topology = WGPUPrimitiveTopology_TriangleStrip;
            break;
        case Mode::TRIANGLES:
            topology = WGPUPrimitiveTopology_TriangleList;
            break;
        case Mode::LINES:
            topology = WGPUPrimitiveTopology_LineList;
            break;
        case Mode::LINE_STRIP:
            topology = WGPUPrimitiveTopology_LineStrip;
            break;
        case Mode::POINTS:
            topology = WGPUPrimitiveTopology_PointList;
            break;
    }

    renderDescriptor.primitive.frontFace = WGPUFrontFace_CCW;
    renderDescriptor.primitive.cullMode = WGPUCullMode_None;
    renderDescriptor.primitive.topology = topology;
    renderDescriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;

    JST_CHECK(buffer->create(attributeDescription,
                             vertexLayouts,
                             config.numberOfDraws,
                             config.numberOfInstances));

    renderDescriptor.vertex.bufferCount = vertexLayouts.size();
    renderDescriptor.vertex.buffers = vertexLayouts.data();

    // Create multi-draw indirect buffer.

    if (buffer->isBuffered()) {
        for (U64 i = 0; i < config.numberOfDraws; i++) {
            IndexedDrawCommand drawCommand = {};
            drawCommand.indexCount = buffer->getIndexCount();
            drawCommand.instanceCount = config.numberOfInstances;
            drawCommand.firstIndex = 0;
            drawCommand.baseVertex = i * (buffer->getIndexCount() - buffer->getVertexCount());
            drawCommand.firstInstance = i * config.numberOfInstances;

            indexedDrawCommands.push_back(drawCommand);
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = indexedDrawCommands.data();
            cfg.elementByteSize = sizeof(IndexedDrawCommand);
            cfg.size = indexedDrawCommands.size();
            cfg.target = Render::Buffer::Target::INDIRECT;

            indexedIndirectBuffer = std::make_shared<Render::BufferImp<Device::WebGPU>>(cfg);
            indexedIndirectBuffer->create();
        }

        indexedIndirectBuffer->update();
    } else {
        for (U64 i = 0; i < config.numberOfDraws; i++) {
            DrawCommand drawCommand = {};
            drawCommand.vertexCount = buffer->getVertexCount();
            drawCommand.instanceCount = config.numberOfInstances;
            drawCommand.firstVertex = i * buffer->getVertexCount();
            drawCommand.firstInstance = i * config.numberOfInstances;

            drawCommands.push_back(drawCommand);
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = drawCommands.data();
            cfg.elementByteSize = sizeof(DrawCommand);
            cfg.size = drawCommands.size();
            cfg.target = Render::Buffer::Target::INDIRECT;

            indirectBuffer = std::make_shared<Render::BufferImp<Device::WebGPU>>(cfg);
            indirectBuffer->create();
        }

        indirectBuffer->update();
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying draw.");

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

Result Implementation::encode(WGPURenderPassEncoder& renderPassEncoder) {
    JST_CHECK(buffer->encode(renderPassEncoder));

    // WebGPU doesn't support multi-draw. So we need to call multiple times.
    for (U64 i = 0; i < config.numberOfDraws; i++) {
        if (buffer->isBuffered()) {
            wgpuRenderPassEncoderDrawIndexedIndirect(renderPassEncoder, indexedIndirectBuffer->getHandle(), i * sizeof(IndexedDrawCommand));
        } else {
            wgpuRenderPassEncoderDrawIndirect(renderPassEncoder, indirectBuffer->getHandle(), i * sizeof(DrawCommand));
        }
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
