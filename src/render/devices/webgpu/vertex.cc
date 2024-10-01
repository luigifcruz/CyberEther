#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/vertex.hh"

namespace Jetstream::Render {

using Implementation = VertexImp<Device::WebGPU>;

Implementation::VertexImp(const Config& config) : Vertex(config) {
    for (const auto& [vertex, stride] : config.vertices) {
        vertices.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::WebGPU>>(vertex), stride}
        );
    }

    for (const auto& [instance, stride] : config.instances) {
        instances.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::WebGPU>>(instance), stride}
        );
    }

    if (config.indices) {
        indices = std::dynamic_pointer_cast<BufferImp<Device::WebGPU>>(config.indices);
    }
}

Result Implementation::create(std::vector<std::vector<wgpu::VertexAttribute>>& attributeDescription,
                              std::vector<wgpu::VertexBufferLayout>& vertexLayouts,
                              const U64& numberOfDraws,
                              const U64&) {
    JST_DEBUG("[WebGPU] Creating vertex.");

    U32 attributeCount = 0;

    for (const auto& [buffer, stride] : vertices) {
        attributeDescription.emplace_back();
        auto& attributes = attributeDescription.back();

        for (U32 i = 0; i < stride; i += 4) {
            wgpu::VertexAttribute attribute = {};

            switch ((stride - i) % 4) {
                case 1:
                    attribute.format = wgpu::VertexFormat::Float32;
                    break;
                case 2:
                    attribute.format = wgpu::VertexFormat::Float32x2;
                    break;
                case 3:
                    attribute.format = wgpu::VertexFormat::Float32x3;
                    break;
                case 0:
                    attribute.format = wgpu::VertexFormat::Float32x4;
                    break;
            }

            attribute.offset = i * sizeof(F32);
            attribute.shaderLocation = attributeCount++;
            attributes.push_back(attribute);
        }

        wgpu::VertexBufferLayout layout = {};
        layout.arrayStride = stride * sizeof(F32);
        layout.attributeCount = attributes.size();
        layout.attributes = attributes.data();
        layout.stepMode = wgpu::VertexStepMode::Vertex;
        vertexLayouts.push_back(layout);
    }

    for (const auto& [buffer, stride] : instances) {
        attributeDescription.emplace_back();
        auto& attributes = attributeDescription.back();

        for (U32 i = 0; i < stride; i += 4) {
            wgpu::VertexAttribute attribute = {};

            switch ((stride - i) % 4) {
                case 1:
                    attribute.format = wgpu::VertexFormat::Float32;
                    break;
                case 2:
                    attribute.format = wgpu::VertexFormat::Float32x2;
                    break;
                case 3:
                    attribute.format = wgpu::VertexFormat::Float32x3;
                    break;
                case 0:
                    attribute.format = wgpu::VertexFormat::Float32x4;
                    break;
            }

            attribute.offset = i * sizeof(F32);
            attribute.shaderLocation = attributeCount++;
            attributes.push_back(attribute);
        }

        wgpu::VertexBufferLayout layout = {};
        layout.arrayStride = stride * sizeof(F32);
        layout.attributeCount = attributes.size();
        layout.attributes = attributes.data();
        layout.stepMode = wgpu::VertexStepMode::Instance;
        vertexLayouts.push_back(layout);
    }

    const auto& [vertex, stride] = vertices[0];
    vertexCount = vertex->byteSize() / sizeof(F32) / numberOfDraws / stride;

    if (indices) {
        indexCount = indices->size();
    }

    return Result::SUCCESS;
}

Result Implementation::encode(wgpu::RenderPassEncoder& renderPassEncoder) {
    U32 bindingCount = 0;

    for (const auto& [vertex, stride] : vertices) {
        renderPassEncoder.SetVertexBuffer(bindingCount++, vertex->getHandle(), 0, vertex->byteSize());
    }

    for (const auto& [instance, stride] : instances) {
        renderPassEncoder.SetVertexBuffer(bindingCount++, instance->getHandle(), 0, instance->byteSize());
    }

    if (indices) {
        renderPassEncoder.SetIndexBuffer(indices->getHandle(), wgpu::IndexFormat::Uint32, 0, indices->byteSize());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying vertex.");

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
