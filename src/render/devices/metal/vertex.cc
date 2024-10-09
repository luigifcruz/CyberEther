#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/vertex.hh"

namespace Jetstream::Render {

using Implementation = VertexImp<Device::Metal>;

Implementation::VertexImp(const Config& config) : Vertex(config) {
    for (const auto& [vertex, stride] : config.vertices) {
        vertices.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::Metal>>(vertex), stride}
        );
    }

    for (const auto& [instance, stride] : config.instances) {
        instances.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::Metal>>(instance), stride}
        );
    }

    if (config.indices) {
        indices = std::dynamic_pointer_cast<BufferImp<Device::Metal>>(config.indices);
    }
}

Result Implementation::create(MTL::VertexDescriptor* vertDesc,
                              const U64& numberOfDraws,
                              const U64&,
                              const U64& offset) {
    JST_DEBUG("[Metal] Creating vertex.");

    // Set attachment offset (buffer attachments goes first).

    indexOffset = offset;

    // Set vertex layout.

    U32 attributeCount = 0;
    U32 layoutCount = offset;

    for (const auto& [vertex, stride] : vertices) {
        for (U32 i = 0; i < stride; i += 4) {
            auto attr = vertDesc->attributes()->object(attributeCount++)->init();

            switch ((stride - i) % 4) {
                case 1:
                    attr->setFormat(MTL::VertexFormat::VertexFormatFloat);
                    break;       
                case 2:
                    attr->setFormat(MTL::VertexFormat::VertexFormatFloat2);
                    break;       
                case 3:
                    attr->setFormat(MTL::VertexFormat::VertexFormatFloat3);
                    break;       
                case 0:
                    attr->setFormat(MTL::VertexFormat::VertexFormatFloat4);
                    break;       
            }

            attr->setBufferIndex(layoutCount);
            attr->setOffset(i * sizeof(F32));
        }

        auto layout = vertDesc->layouts()->object(layoutCount++)->init();
        layout->setStride(stride * sizeof(F32));
        layout->setStepRate(1);
        layout->setStepFunction(MTL::VertexStepFunctionPerVertex);
    }

    for (const auto& [instance, stride] : instances) {
        for (U32 i = 0; i < stride; i += 4) {
            auto attr = vertDesc->attributes()->object(attributeCount++)->init();
            
            switch ((stride - i) % 4) {
                case 1:
                    attr->setFormat(MTL::VertexFormat::VertexFormatFloat);
                    break;       
                case 2:
                    attr->setFormat(MTL::VertexFormat::VertexFormatFloat2);
                    break;       
                case 3:
                    attr->setFormat(MTL::VertexFormat::VertexFormatFloat3);
                    break;       
                case 0:
                    attr->setFormat(MTL::VertexFormat::VertexFormatFloat4);
                    break;       
            }

            attr->setBufferIndex(layoutCount);
            attr->setOffset(i * sizeof(F32));
        }

        auto layout = vertDesc->layouts()->object(layoutCount++)->init();
        layout->setStride(stride * sizeof(F32));
        layout->setStepRate(1);
        layout->setStepFunction(MTL::VertexStepFunctionPerInstance);
    }

    const auto& [vertex, stride] = vertices[0];
    vertexCount = vertex->size() / numberOfDraws / stride;

    if (indices) {
        indexCount = indices->size();
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[Metal] Destroying vertex.");

    return Result::SUCCESS;
}

Result Implementation::encode(MTL::RenderCommandEncoder* encoder) {
    U64 index = indexOffset;

    for (const auto& [vertex, _] : vertices) {
        encoder->setVertexBuffer(vertex->getHandle(), 0, index++);
    }

    for (const auto& [instance, _] : instances) {
        encoder->setVertexBuffer(instance->getHandle(), 0, index++);
    }

    return Result::SUCCESS;
}

const MTL::Buffer* Implementation::getIndexBuffer() {
    return indices->getHandle();
}

}  // namespace Jetstream::Render
