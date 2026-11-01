#include "jetstream/render/devices/webgpu/transfer.hh"

#include <algorithm>
#include <limits>

#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/texture.hh"

namespace Jetstream::Render {

using Implementation = TransferImp<DeviceType::WebGPU>;

Result Implementation::ensureCapacity(const U64& page,
                                      const U64& required,
                                      const U64& maximumBufferSize) {
    if (stagingBuffers.size() <= page) {
        stagingBuffers.resize(page + 1);
    }
    auto& staging = stagingBuffers[page];
    if (required <= staging.capacity) {
        return Result::SUCCESS;
    }

    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();
    U64 replacementCapacity = 0;
    if (!calculateCapacity(required, 4, replacementCapacity)) {
        return Result::ERROR;
    }
    if (replacementCapacity > maximumBufferSize &&
        !calculateAlignedSize(required, 4, replacementCapacity)) {
        return Result::ERROR;
    }
    if (replacementCapacity > maximumBufferSize) {
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

    if (staging.buffer) {
        wgpuBufferRelease(staging.buffer);
    }
    staging.buffer = replacement;
    staging.capacity = replacementCapacity;

    JST_DEBUG("[WebGPU] Grew transfer buffer to {:.2f} MB.",
              static_cast<F32>(staging.capacity) / JST_MB);
    return Result::SUCCESS;
}

Result Implementation::encode(Transfer::Batch& batch,
                              WGPUQueue queue,
                              WGPUCommandEncoder encoder) {
    if (!queue || !encoder) {
        return Result::ERROR;
    }

    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();
    WGPULimits limits = WGPU_LIMITS_INIT;
    if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success) {
        JST_ERROR("[WebGPU] Failed to query device limits.");
        return Result::ERROR;
    }
    const U64 pageLimit = std::min<U64>(limits.maxBufferSize, MaximumPageSize) /
                          RowAlignment * RowAlignment;
    if (pageLimit == 0) {
        JST_ERROR("[WebGPU] Device buffer limit is too small for transfer staging.");
        return Result::ERROR;
    }

    struct BufferCopy {
        std::shared_ptr<BufferImp<DeviceType::WebGPU>> destination;
        U64 page;
        U64 sourceOffset;
        U64 destinationOffset;
        U64 size;
        const U8* source;
    };

    struct TextureCopy {
        std::shared_ptr<TextureImp<DeviceType::WebGPU>> destination;
        U64 page;
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
    std::vector<U64> pages;

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

        for (U64 copied = 0; copied < size;) {
            const U64 chunkSize = std::min(size - copied, pageLimit);
            U64 page = 0;
            U64 sourceOffset = 0;
            if (!reservePageRange(pages, chunkSize, 4, pageLimit, page, sourceOffset)) {
                return Result::ERROR;
            }
            bufferCopies.push_back({destination,
                                    page,
                                    sourceOffset,
                                    transfer.destinationOffset + copied,
                                    chunkSize,
                                    transfer.upload.data.data() + copied});
            copied += chunkSize;
        }
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
                                  RowAlignment,
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

        if (encodedRowByteSize > pageLimit) {
            JST_ERROR("[WebGPU] Texture transfer row exceeds the staging buffer limit.");
            return Result::ERROR;
        }
        for (U64 copiedRows = 0; copiedRows < rowCount;) {
            const U64 chunkRows = std::min(rowCount - copiedRows, pageLimit / encodedRowByteSize);
            U64 page = 0;
            U64 sourceOffset = 0;
            if (!reservePageRange(pages, chunkRows * encodedRowByteSize,
                                  RowAlignment, pageLimit, page, sourceOffset)) {
                return Result::ERROR;
            }
            textureCopies.push_back({destination,
                                     page,
                                     sourceOffset,
                                     transfer.destinationRow + copiedRows,
                                     chunkRows,
                                     transfer.destinationSize.x,
                                     transfer.rowByteSize,
                                     encodedRowByteSize,
                                     transfer.upload.data.data() + copiedRows * transfer.rowByteSize});
            copiedRows += chunkRows;
        }
    }

    for (U64 page = 0; page < pages.size(); ++page) {
        JST_CHECK(ensureCapacity(page, pages[page], limits.maxBufferSize));
    }

    for (U64 page = 0; page < pages.size(); ++page) {
        const auto buffer = stagingBuffers[page].buffer;
        data.resize(static_cast<size_t>(pages[page]));

        for (const auto& copy : bufferCopies) {
            if (copy.page != page) {
                continue;
            }
            if (!copyBuffer(data.data() + copy.sourceOffset,
                            copy.source,
                            copy.size)) {
                return Result::ERROR;
            }
        }
        for (const auto& copy : textureCopies) {
            if (copy.page != page) {
                continue;
            }
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
            if (copy.page != page) {
                continue;
            }
            wgpuCommandEncoderCopyBufferToBuffer(encoder,
                                                 buffer,
                                                 copy.sourceOffset,
                                                 copy.destination->buffer,
                                                 copy.destinationOffset,
                                                 copy.size);
        }

        for (const auto& copy : textureCopies) {
            if (copy.page != page) {
                continue;
            }
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
    }

    return Result::SUCCESS;
}

void Implementation::destroy() {
    for (auto& staging : stagingBuffers) {
        if (staging.buffer) {
            wgpuBufferRelease(staging.buffer);
        }
    }
    stagingBuffers.clear();
    data.clear();
}

}  // namespace Jetstream::Render
