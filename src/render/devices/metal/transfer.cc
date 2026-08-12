#include "jetstream/render/devices/metal/transfer.hh"

#include <algorithm>
#include <cstring>
#include <limits>

#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/texture.hh"

namespace Jetstream::Render {

using Implementation = TransferImp<DeviceType::Metal>;

Result Implementation::ensureCapacity(Arena& arena, const U64& required) {
    if (required <= arena.capacity) {
        return Result::SUCCESS;
    }

    auto device = Backend::State<DeviceType::Metal>()->getDevice();
    U64 capacity = 0;
    if (!calculateCapacity(required, 1, capacity)) {
        return Result::ERROR;
    }

    if (capacity > device->maxBufferLength() ||
        capacity > std::numeric_limits<NS::UInteger>::max()) {
        JST_ERROR("[METAL] Required transfer arena exceeds the device buffer limit.");
        return Result::ERROR;
    }

    MTL::Buffer* replacement = device->newBuffer(static_cast<NS::UInteger>(capacity),
                                                 MTL::ResourceStorageModeShared);
    if (!replacement) {
        JST_ERROR("[METAL] Failed to create a {} byte transfer arena.", capacity);
        return Result::ERROR;
    }

    destroyArena(arena);
    arena.buffer = replacement;
    arena.capacity = capacity;

    JST_DEBUG("[METAL] Grew frame transfer arena to {:.2f} MB.",
              static_cast<F32>(capacity) / JST_MB);
    return Result::SUCCESS;
}

Result Implementation::encode(Transfer::Batch& batch,
                              MTL::CommandBuffer* commandBuffer,
                              const size_t frameIndex) {
    if (!commandBuffer || frameIndex >= arenas.size()) {
        return Result::ERROR;
    }

    struct BufferCopy {
        std::shared_ptr<BufferImp<DeviceType::Metal>> destination;
        U64 sourceOffset;
        U64 destinationOffset;
        U64 size;
        const U8* source;
    };

    struct TextureCopy {
        std::shared_ptr<TextureImp<DeviceType::Metal>> destination;
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
            JST_ERROR("[METAL] Buffer transfer offsets and sizes must be four-byte aligned.");
            return Result::ERROR;
        }

        auto destination = std::dynamic_pointer_cast<BufferImp<DeviceType::Metal>>(
            transfer.destination);
        if (!destination) {
            JST_ERROR("[METAL] Cannot encode a buffer from another render device.");
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

    auto device = Backend::State<DeviceType::Metal>()->getDevice();
    for (const auto& transfer : batch.textures()) {
        auto destination = std::dynamic_pointer_cast<TextureImp<DeviceType::Metal>>(
            transfer.destination);
        if (!destination) {
            JST_ERROR("[METAL] Cannot encode a texture from another render device.");
            return Result::ERROR;
        }
        U64 rowCount = 0;
        if (!validateTexture(transfer, rowCount)) {
            JST_ERROR("[METAL] Invalid texture transfer.");
            return Result::ERROR;
        }

        const U64 alignment = std::max<U64>(
            device->minimumLinearTextureAlignmentForPixelFormat(destination->pixelFormat), 1);
        U64 encodedRowByteSize = 0;
        if (!calculateAlignedSize(transfer.rowByteSize,
                                  alignment,
                                  encodedRowByteSize)) {
            return Result::ERROR;
        }
        if (rowCount > std::numeric_limits<NS::UInteger>::max() ||
            encodedRowByteSize > std::numeric_limits<NS::UInteger>::max() ||
            rowCount > std::numeric_limits<U64>::max() / encodedRowByteSize) {
            return Result::ERROR;
        }

        U64 sourceOffset = 0;
        if (!reserveRange(required,
                          rowCount * encodedRowByteSize,
                          alignment,
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

    auto& arena = arenas[frameIndex];
    JST_CHECK(ensureCapacity(arena, required));
    auto* mapped = static_cast<U8*>(arena.buffer->contents());

    for (const auto& copy : bufferCopies) {
        if (!copyBuffer(mapped + copy.sourceOffset, copy.source, copy.size)) {
            return Result::ERROR;
        }
    }
    for (const auto& copy : textureCopies) {
        if (!copyTextureRows(mapped + copy.sourceOffset,
                             copy.source,
                             copy.rowCount,
                             copy.rowByteSize,
                             copy.encodedRowByteSize)) {
            return Result::ERROR;
        }
    }

    auto* blit = commandBuffer->blitCommandEncoder();
    JST_ASSERT(blit, "[METAL] Failed to create transfer command encoder.");

    for (const auto& copy : bufferCopies) {
        blit->copyFromBuffer(arena.buffer,
                             copy.sourceOffset,
                             copy.destination->buffer,
                             copy.destinationOffset,
                             copy.size);
    }

    for (const auto& copy : textureCopies) {
        blit->copyFromBuffer(arena.buffer,
                             copy.sourceOffset,
                             copy.encodedRowByteSize,
                             copy.encodedRowByteSize * copy.rowCount,
                             MTL::Size(copy.width, copy.rowCount, 1),
                             copy.destination->texture,
                             0,
                             0,
                             MTL::Origin(0, copy.destinationRow, 0));
    }

    blit->endEncoding();
    return Result::SUCCESS;
}

void Implementation::destroyArena(Arena& arena) {
    if (arena.buffer) {
        arena.buffer->release();
        arena = {};
    }
}

void Implementation::destroy() {
    for (auto& arena : arenas) {
        destroyArena(arena);
    }
}

}  // namespace Jetstream::Render
