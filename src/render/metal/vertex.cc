#include "jetstream/render/metal/buffer.hh"
#include "jetstream/render/metal/vertex.hh"

namespace Jetstream::Render {

using Implementation = VertexImp<Device::Metal>;

Implementation::VertexImp(const Config& config) : Vertex(config) {
    for (const auto& [buffer, stride] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::Metal>>(buffer), stride}
        );
    }

    if (config.indices) {
        indices = std::dynamic_pointer_cast<BufferImp<Device::Metal>>(config.indices);
    }
}

Result Implementation::create(MTL::VertexDescriptor* vertDesc, const U64& offset) {
    JST_DEBUG("Creating Metal vertex.");

    // Offset vertex buffers for buffer attachments.
    indexOffset = offset;

    U32 bindingCount = 0;
    for (const auto& [buffer, stride] : buffers) {
        vertexCount = buffer->size() / stride;

        MTL::VertexFormat bindingFormat = MTL::VertexFormat::VertexFormatInvalid;
        
        switch (stride) {
            case 1:
                bindingFormat = MTL::VertexFormat::VertexFormatFloat;
                break;       
            case 2:
                bindingFormat = MTL::VertexFormat::VertexFormatFloat2;
                break;       
            case 3:
                bindingFormat = MTL::VertexFormat::VertexFormatFloat3;
                break;       
            case 4:
                bindingFormat = MTL::VertexFormat::VertexFormatFloat4;
                break;       
        }

        auto attr = vertDesc->attributes()->object(bindingCount)->init();
        attr->setFormat(bindingFormat);
        attr->setBufferIndex(bindingCount + indexOffset);
        attr->setOffset(0);

        auto layout = vertDesc->layouts()->object(bindingCount + indexOffset)->init();
        layout->setStride(stride * sizeof(F32));
        layout->setStepRate(1);
        layout->setStepFunction(MTL::VertexStepFunctionPerVertex);

        bindingCount += 1;
    }

    if (indices) {
        vertexCount = indices->size();
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal vertex.");

    return Result::SUCCESS;
}

Result Implementation::encode(MTL::RenderCommandEncoder* encoder) {
    U64 index = indexOffset;
    for (const auto& [buffer, stride] : buffers) {
        encoder->setVertexBuffer(buffer->getHandle(), 0, index++);
    }

    return Result::SUCCESS;
}

const MTL::Buffer* Implementation::getIndexBuffer() {
    return indices->getHandle();
}

}  // namespace Jetstream::Render
