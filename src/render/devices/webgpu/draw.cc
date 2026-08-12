#include "jetstream/render/devices/webgpu/vertex.hh"
#include "jetstream/render/devices/webgpu/draw.hh"
#include "jetstream/render/devices/webgpu/buffer.hh"

namespace Jetstream::Render {

using Implementation = DrawImp<DeviceType::WebGPU>;

Implementation::DrawImp(const Config& config) : Draw(config) {
    buffer = std::dynamic_pointer_cast<VertexImp<DeviceType::WebGPU>>(config.buffer);
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

    } else {
        for (U64 i = 0; i < config.numberOfDraws; i++) {
            DrawCommand drawCommand = {};
            drawCommand.vertexCount = buffer->getVertexCount();
            drawCommand.instanceCount = config.numberOfInstances;
            drawCommand.firstVertex = i * buffer->getVertexCount();
            drawCommand.firstInstance = i * config.numberOfInstances;

            drawCommands.push_back(drawCommand);
        }

    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying draw.");

    if (buffer->isBuffered()) {
        indexedDrawCommands.clear();
    } else {
        drawCommands.clear();
    }

    JST_CHECK(buffer->destroy());

    return Result::SUCCESS;
}

Result Implementation::encode(WGPURenderPassEncoder& renderPassEncoder) {
    JST_CHECK(buffer->encode(renderPassEncoder));

    // WebGPU doesn't support multi-draw and 'indirect-first-instance' feature
    // may not be available. Use direct draw calls instead of indirect.

    for (U64 i = 0; i < config.numberOfDraws; i++) {
        if (buffer->isBuffered()) {
            const auto& cmd = indexedDrawCommands[i];
            wgpuRenderPassEncoderDrawIndexed(renderPassEncoder,
                                             cmd.indexCount,
                                             cmd.instanceCount,
                                             cmd.firstIndex,
                                             cmd.baseVertex,
                                             cmd.firstInstance);
        } else {
            const auto& cmd = drawCommands[i];
            wgpuRenderPassEncoderDraw(renderPassEncoder,
                                      cmd.vertexCount,
                                      cmd.instanceCount,
                                      cmd.firstVertex,
                                      cmd.firstInstance);
        }
    }

    return Result::SUCCESS;
}

Result Implementation::updateVertexCount(U64 vertexCount) {
    // Check if draw was created
    if (indexedDrawCommands.empty() && drawCommands.empty()) {
        JST_ERROR("[WebGPU] Cannot update vertex count: draw not created yet");
        return Result::ERROR;
    }

    if (buffer->isBuffered()) {
        // Check bounds for indexed drawing
        const U64 maxIndexCount = buffer->getIndexCount();
        if (vertexCount > maxIndexCount) {
            JST_ERROR("[WebGPU] Requested index count ({}) exceeds buffer limit ({})", vertexCount, maxIndexCount);
            return Result::ERROR;
        }

        // Update indexed draw commands
        for (auto& drawCommand : indexedDrawCommands) {
            drawCommand.indexCount = vertexCount;
        }
    } else {
        // Check bounds for non-indexed drawing
        const U64 maxVertexCount = buffer->getVertexCount();
        if (vertexCount > maxVertexCount) {
            JST_ERROR("[WebGPU] Requested vertex count ({}) exceeds buffer limit ({})", vertexCount, maxVertexCount);
            return Result::ERROR;
        }

        // Update non-indexed draw commands
        for (auto& drawCommand : drawCommands) {
            drawCommand.vertexCount = vertexCount;
        }
    }

    return Result::SUCCESS;
}

Result Implementation::updateInstanceCount(U64 instanceCount) {
    if (indexedDrawCommands.empty() && drawCommands.empty()) {
        JST_ERROR("[WebGPU] Cannot update instance count: draw not created yet");
        return Result::ERROR;
    }

    config.numberOfInstances = instanceCount;

    if (buffer->isBuffered()) {
        for (U64 i = 0; i < indexedDrawCommands.size(); ++i) {
            auto& drawCommand = indexedDrawCommands[i];
            drawCommand.instanceCount = instanceCount;
            drawCommand.firstInstance = i * instanceCount;
        }
    } else {
        for (U64 i = 0; i < drawCommands.size(); ++i) {
            auto& drawCommand = drawCommands[i];
            drawCommand.instanceCount = instanceCount;
            drawCommand.firstInstance = i * instanceCount;
        }
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
