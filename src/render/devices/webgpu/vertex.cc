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

Result Implementation::create(std::vector<std::vector<WGPUVertexAttribute>>& attributeDescription,
                              std::vector<WGPUVertexBufferLayout>& vertexLayouts,
                              const U64& numberOfDraws,
                              const U64&) {
    JST_DEBUG("[WebGPU] Creating vertex.");

    U32 attributeCount = 0;

    for (const auto& [buffer, stride] : vertices) {
        attributeDescription.emplace_back();
        auto& attributes = attributeDescription.back();

        for (U32 i = 0; i < stride; i += 4) {
            WGPUVertexAttribute attribute = WGPU_VERTEX_ATTRIBUTE_INIT;

            switch ((stride - i) % 4) {
                case 1:
                    attribute.format = WGPUVertexFormat_Float32;
                    break;
                case 2:
                    attribute.format = WGPUVertexFormat_Float32x2;
                    break;
                case 3:
                    attribute.format = WGPUVertexFormat_Float32x3;
                    break;
                case 0:
                    attribute.format = WGPUVertexFormat_Float32x4;
                    break;
            }

            attribute.offset = i * sizeof(F32);
            attribute.shaderLocation = attributeCount++;
            attributes.push_back(attribute);
        }

        WGPUVertexBufferLayout layout = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
        layout.arrayStride = stride * sizeof(F32);
        layout.attributeCount = static_cast<uint32_t>(attributes.size());
        layout.attributes = attributes.data();
        layout.stepMode = WGPUVertexStepMode_Vertex;
        vertexLayouts.push_back(layout);
    }

    for (const auto& [buffer, stride] : instances) {
        attributeDescription.emplace_back();
        auto& attributes = attributeDescription.back();

        for (U32 i = 0; i < stride; i += 4) {
            WGPUVertexAttribute attribute = WGPU_VERTEX_ATTRIBUTE_INIT;

            switch ((stride - i) % 4) {
                case 1:
                    attribute.format = WGPUVertexFormat_Float32;
                    break;
                case 2:
                    attribute.format = WGPUVertexFormat_Float32x2;
                    break;
                case 3:
                    attribute.format = WGPUVertexFormat_Float32x3;
                    break;
                case 0:
                    attribute.format = WGPUVertexFormat_Float32x4;
                    break;
            }

            attribute.offset = i * sizeof(F32);
            attribute.shaderLocation = attributeCount++;
            attributes.push_back(attribute);
        }

        WGPUVertexBufferLayout layout = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
        layout.arrayStride = stride * sizeof(F32);
        layout.attributeCount = static_cast<uint32_t>(attributes.size());
        layout.attributes = attributes.data();
        layout.stepMode = WGPUVertexStepMode_Instance;
        vertexLayouts.push_back(layout);
    }

    const auto& [vertex, stride] = vertices[0];
    vertexCount = vertex->size() / numberOfDraws / stride;

    if (indices) {
        indexCount = indices->size();
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying vertex.");

    return Result::SUCCESS;
}

Result Implementation::encode(WGPURenderPassEncoder& renderPassEncoder) {
    U32 bindingCount = 0;

    for (const auto& [vertex, _] : vertices) {
        wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder,
                                             bindingCount++,
                                             vertex->getHandle(),
                                             0,
                                             vertex->byteSize());
    }

    for (const auto& [instance, _] : instances) {
        wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder,
                                             bindingCount++,
                                             instance->getHandle(),
                                             0,
                                             instance->byteSize());
    }

    if (indices) {
        wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder,
                                            indices->getHandle(),
                                            WGPUIndexFormat_Uint32,
                                            0,
                                            indices->byteSize());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
