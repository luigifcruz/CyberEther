#include "jetstream/render/devices/webgpu/transfer.hh"

#include <limits>

#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/texture.hh"

namespace Jetstream::Render {

using Implementation = TransferImp<DeviceType::WebGPU>;

Result Implementation::ensureCapacity(const U64& required) {
    if (required <= capacity) {
        return Result::SUCCESS;
    }

    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();
    WGPULimits limits = WGPU_LIMITS_INIT;
    if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success) {
        JST_ERROR("[WebGPU] Failed to query device limits.");
        return Result::ERROR;
    }
    if (required > limits.maxBufferSize) {
        JST_ERROR("[WebGPU] Required transfer buffer exceeds the device limit.");
        return Result::ERROR;
    }

    U64 replacementCapacity = 0;
    if (!calculateCapacity(required, 4, replacementCapacity)) {
        return Result::ERROR;
    }
    if (replacementCapacity > limits.maxBufferSize &&
        !calculateAlignedSize(required, 4, replacementCapacity)) {
        return Result::ERROR;
    }
    if (replacementCapacity > limits.maxBufferSize) {
        JST_ERROR("[WebGPU] Aligned transfer buffer exceeds the device limit.");
        return Result::ERROR;
    }

    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.size = replacementCapacity;
    descriptor.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer replacement = wgpuDeviceCreateBuffer(device, &descriptor);
    if (!replacement) {
        JST_ERROR("[WebGPU] Failed to create a {} byte transfer buffer.", replacementCapacity);
        return Result::ERROR;
    }

    if (buffer) {
        wgpuBufferRelease(buffer);
    }
    buffer = replacement;
    capacity = replacementCapacity;

    JST_DEBUG("[WebGPU] Grew transfer buffer to {:.2f} MB.",
              static_cast<F32>(capacity) / JST_MB);
    return Result::SUCCESS;
}

Result Implementation::encode(Transfer::Batch& batch,
                              WGPUQueue queue,
                              WGPUCommandEncoder encoder) {
    if (!queue || !encoder) {
        return Result::ERROR;
    }

    struct BufferCopy {
        std::shared_ptr<BufferImp<DeviceType::WebGPU>> destination;
        U64 sourceOffset;
        U64 destinationOffset;
        U64 size;
        const U8* source;
    };

    struct TextureCopy {
        std::shared_ptr<TextureImp<DeviceType::WebGPU>> destination;
        U64 sourceOffset;
        U64 destinationRow;
        U64 rowCount;
        U64 width;
        U64 rowByteSize;
        U64 encodedRowByteSize;
        const U8* source;
    };

    std::vector<BufferCopy> bufferCopies;
    std::vector<TextureCopy> textureCopies;
    U64 required = 0;

    for (const auto& transfer : batch.buffers()) {
        const U64 size = transfer.upload.data.size();
        if ((transfer.destinationOffset % 4) != 0 || (size % 4) != 0) {
            JST_ERROR("[WebGPU] Buffer transfer offsets and sizes must be four-byte aligned.");
            return Result::ERROR;
        }

        auto destination = std::dynamic_pointer_cast<BufferImp<DeviceType::WebGPU>>(
            transfer.destination);
        if (!destination) {
            JST_ERROR("[WebGPU] Cannot encode a buffer from another render device.");
            return Result::ERROR;
        }

        U64 sourceOffset = 0;
        if (!reserveRange(required, size, 4, sourceOffset)) {
            return Result::ERROR;
        }

        bufferCopies.push_back({destination,
                                sourceOffset,
                                transfer.destinationOffset,
                                size,
                                transfer.upload.data.data()});
    }

    for (const auto& transfer : batch.textures()) {
        auto destination = std::dynamic_pointer_cast<TextureImp<DeviceType::WebGPU>>(
            transfer.destination);
        if (!destination) {
            JST_ERROR("[WebGPU] Cannot encode a texture from another render device.");
            return Result::ERROR;
        }

        U64 rowCount = 0;
        if (!validateTexture(transfer, rowCount)) {
            JST_ERROR("[WebGPU] Invalid texture transfer.");
            return Result::ERROR;
        }

        U64 encodedRowByteSize = 0;
        if (!calculateAlignedSize(transfer.rowByteSize,
                                  256,
                                  encodedRowByteSize)) {
            return Result::ERROR;
        }
        if (rowCount > std::numeric_limits<U32>::max() ||
            encodedRowByteSize > std::numeric_limits<U32>::max() ||
            transfer.destinationSize.x > std::numeric_limits<U32>::max() ||
            transfer.destinationRow > std::numeric_limits<U32>::max() ||
            rowCount > std::numeric_limits<U64>::max() / encodedRowByteSize) {
            return Result::ERROR;
        }

        U64 sourceOffset = 0;
        if (!reserveRange(required,
                          rowCount * encodedRowByteSize,
                          256,
                          sourceOffset)) {
            return Result::ERROR;
        }

        textureCopies.push_back({destination,
                                 sourceOffset,
                                  transfer.destinationRow,
                                  rowCount,
                                  transfer.destinationSize.x,
                                  transfer.rowByteSize,
                                 encodedRowByteSize,
                                 transfer.upload.data.data()});
    }

    JST_CHECK(ensureCapacity(required));

    if (required > std::numeric_limits<size_t>::max()) {
        return Result::ERROR;
    }
    data.resize(static_cast<size_t>(required));

    for (const auto& copy : bufferCopies) {
        if (!copyBuffer(data.data() + copy.sourceOffset,
                        copy.source,
                        copy.size)) {
            return Result::ERROR;
        }
    }
    for (const auto& copy : textureCopies) {
        if (!copyTextureRows(data.data() + copy.sourceOffset,
                             copy.source,
                             copy.rowCount,
                             copy.rowByteSize,
                             copy.encodedRowByteSize)) {
            return Result::ERROR;
        }
    }

    wgpuQueueWriteBuffer(queue, buffer, 0, data.data(), data.size());

    for (const auto& copy : bufferCopies) {
        wgpuCommandEncoderCopyBufferToBuffer(encoder,
                                             buffer,
                                             copy.sourceOffset,
                                             copy.destination->buffer,
                                             copy.destinationOffset,
                                             copy.size);
    }

    for (const auto& copy : textureCopies) {
        WGPUTexelCopyBufferInfo source = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
        source.buffer = buffer;
        source.layout.offset = copy.sourceOffset;
        source.layout.bytesPerRow = static_cast<U32>(copy.encodedRowByteSize);
        source.layout.rowsPerImage = static_cast<U32>(copy.rowCount);

        WGPUTexelCopyTextureInfo destination = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        destination.texture = copy.destination->texture;
        destination.mipLevel = 0;
        destination.origin = WGPUOrigin3D{0, static_cast<U32>(copy.destinationRow), 0};
        destination.aspect = WGPUTextureAspect_All;

        WGPUExtent3D extent = {
            static_cast<U32>(copy.width),
            static_cast<U32>(copy.rowCount),
            1,
        };
        wgpuCommandEncoderCopyBufferToTexture(encoder, &source, &destination, &extent);
    }

    return Result::SUCCESS;
}

void Implementation::destroy() {
    if (buffer) {
        wgpuBufferRelease(buffer);
        buffer = nullptr;
    }
    capacity = 0;
    data.clear();
}

}  // namespace Jetstream::Render
