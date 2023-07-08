#include "jetstream/render/webgpu/buffer.hh"
#include "jetstream/render/webgpu/vertex.hh"

namespace Jetstream::Render {

using Implementation = VertexImp<Device::WebGPU>;

Implementation::VertexImp(const Config& config) : Vertex(config) {
    for (const auto& [buffer, stride] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::WebGPU>>(buffer), stride}
        );
    }

    if (config.indices) {
        indices = std::dynamic_pointer_cast<BufferImp<Device::WebGPU>>(config.indices);
    }
}

Result Implementation::create(wgpu::RenderPipelineDescriptor& renderDescriptor) {
    JST_DEBUG("[WebGPU] Creating vertex.");

    U32 bindingCount = 0;
    vertexLayouts.resize(buffers.size());
    vertexAttributes.resize(buffers.size());
    for (const auto& [buffer, stride] : buffers) {
        JST_CHECK(buffer->create());
        vertexCount = buffer->size() / stride;

        auto bindingFormat = wgpu::VertexFormat::Undefined;
        
        switch (stride) {
            case 1:
                bindingFormat = wgpu::VertexFormat::Float32;
                break;
            case 2:
                bindingFormat = wgpu::VertexFormat::Float32x2;
                break;
            case 3:
                bindingFormat = wgpu::VertexFormat::Float32x3;
                break;
            case 4:
                bindingFormat = wgpu::VertexFormat::Float32x4;
                break;
        }

        vertexAttributes[bindingCount].format = bindingFormat;
        vertexAttributes[bindingCount].offset = 0;
        vertexAttributes[bindingCount].shaderLocation = bindingCount;

        vertexLayouts[bindingCount].arrayStride = stride * sizeof(F32);
        vertexLayouts[bindingCount].attributeCount = 1;
        vertexLayouts[bindingCount].attributes = &vertexAttributes[bindingCount];

        bindingCount += 1;
    }

    if (indices) {
        JST_CHECK(indices->create());
        vertexCount = indices->size();
    }

    renderDescriptor.vertex.bufferCount = vertexLayouts.size();
    renderDescriptor.vertex.buffers = vertexLayouts.data();

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying vertex.");

    for (const auto& [buffer, stride] : buffers) {
        JST_CHECK(buffer->destroy());
    }

    if (indices) {
        JST_CHECK(indices->destroy());
    }

    return Result::SUCCESS;
}

Result Implementation::encode(wgpu::RenderPassEncoder& renderPassEncoder) {
    U32 bindingCount = 0;
    for (const auto& [buffer, stride] : buffers) {
        renderPassEncoder.SetVertexBuffer(bindingCount++, buffer->getHandle());
    }

    if (indices) {
        renderPassEncoder.SetIndexBuffer(indices->getHandle(), wgpu::IndexFormat::Uint32);
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
