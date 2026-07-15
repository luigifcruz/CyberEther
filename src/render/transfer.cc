#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/transfer.hh"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace Jetstream::Render {

namespace {

bool PendingUploadRange(const Transfer::PendingUpload& upload,
                        U64& extent,
                        U64& end) {
    if (upload.unitByteSize == 0 || upload.data.empty() ||
        upload.data.size() % upload.unitByteSize != 0) {
        return false;
    }

    const U64 byteSize = static_cast<U64>(upload.data.size());
    if (static_cast<size_t>(byteSize) != upload.data.size()) {
        return false;
    }

    extent = byteSize / upload.unitByteSize;
    if (extent == 0 ||
        upload.start > std::numeric_limits<U64>::max() - extent ||
        upload.start > std::numeric_limits<U64>::max() / upload.unitByteSize) {
        return false;
    }

    const U64 byteStart = upload.start * upload.unitByteSize;
    if (byteSize > std::numeric_limits<U64>::max() - byteStart) {
        return false;
    }

    end = upload.start + extent;
    return true;
}

bool CheckedMultiply(const U64 lhs, const U64 rhs, U64& result) {
    if (lhs != 0 && rhs > std::numeric_limits<U64>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

}  // namespace

Transfer::Transfer() = default;

Transfer::~Transfer() = default;

U64 Transfer::PendingUpload::extent() const {
    if (unitByteSize == 0 || data.size() % unitByteSize != 0) {
        return 0;
    }
    return static_cast<U64>(data.size() / unitByteSize);
}

bool Transfer::reserveRange(U64& used,
                            const U64& size,
                            const U64& alignment,
                            U64& offset) {
    if (alignment == 0) {
        return false;
    }

    const U64 remainder = used % alignment;
    const U64 padding = remainder == 0 ? 0 : alignment - remainder;
    if (used > std::numeric_limits<U64>::max() - padding ||
        used + padding > std::numeric_limits<U64>::max() - size) {
        return false;
    }

    offset = used + padding;
    used = offset + size;
    return true;
}

bool Transfer::calculateCapacity(const U64& required,
                                 const U64& alignment,
                                 U64& capacity) {
    capacity = MinimumBufferSize;
    while (capacity < required) {
        if (capacity > std::numeric_limits<U64>::max() / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }

    return calculateAlignedSize(capacity, alignment, capacity);
}

bool Transfer::calculateAlignedSize(const U64& size,
                                    const U64& alignment,
                                    U64& alignedSize) {
    if (alignment == 0) {
        return false;
    }

    const U64 remainder = size % alignment;
    const U64 padding = remainder == 0 ? 0 : alignment - remainder;
    if (size > std::numeric_limits<U64>::max() - padding) {
        return false;
    }

    alignedSize = size + padding;
    return true;
}

bool Transfer::validateTexture(const TextureTransfer& transfer, U64& rowCount) {
    if (!transfer.destination || transfer.destination->multisampled() ||
        transfer.rowByteSize == 0 || transfer.upload.data.empty() ||
        transfer.upload.data.size() % transfer.rowByteSize != 0 ||
        transfer.upload.generation !=
            transfer.destination->uploadGeneration.load(std::memory_order_acquire)) {
        return false;
    }

    rowCount = transfer.upload.data.size() / transfer.rowByteSize;
    const auto& size = transfer.destinationSize;
    return size.x > 0 && transfer.destinationRow <= size.y &&
           rowCount <= size.y - transfer.destinationRow;
}

bool Transfer::copyBuffer(U8* destination, const U8* source, const U64& size) {
    if (size == 0) {
        return true;
    }
    if (!destination || !source || size > std::numeric_limits<size_t>::max()) {
        return false;
    }

    std::memcpy(destination, source, static_cast<size_t>(size));
    return true;
}

bool Transfer::copyTextureRows(U8* destination,
                               const U8* source,
                               const U64& rowCount,
                               const U64& rowByteSize,
                               const U64& encodedRowByteSize) {
    if (!destination || !source || rowByteSize == 0 || encodedRowByteSize < rowByteSize ||
        rowCount > std::numeric_limits<U64>::max() / encodedRowByteSize ||
        rowCount > std::numeric_limits<U64>::max() / rowByteSize ||
        rowCount * encodedRowByteSize > std::numeric_limits<size_t>::max() ||
        rowCount * rowByteSize > std::numeric_limits<size_t>::max()) {
        return false;
    }

    if (rowByteSize == encodedRowByteSize) {
        return copyBuffer(destination, source, rowCount * rowByteSize);
    }

    for (U64 row = 0; row < rowCount; ++row) {
        std::memcpy(destination + row * encodedRowByteSize,
                    source + row * rowByteSize,
                    rowByteSize);
    }
    return true;
}

Result Transfer::PendingUploadQueue::queue(const U64& start,
                                           const U64& count,
                                           const U64& totalCount,
                                           const U64& unitByteSize,
                                           const U8* source,
                                           const U64& generation) {
    if (count == 0) {
        return Result::SUCCESS;
    }

    if (!source || unitByteSize == 0 || start > totalCount || count > totalCount - start) {
        return Result::ERROR;
    }

    if (start > std::numeric_limits<U64>::max() / unitByteSize ||
        count > std::numeric_limits<U64>::max() / unitByteSize ||
        totalCount > std::numeric_limits<U64>::max() / unitByteSize) {
        return Result::ERROR;
    }

    const U64 sourceOffset = start * unitByteSize;
    const U64 byteSize = count * unitByteSize;
    if (sourceOffset > std::numeric_limits<size_t>::max() ||
        byteSize > std::numeric_limits<size_t>::max() ||
        totalCount * unitByteSize > std::numeric_limits<size_t>::max()) {
        return Result::ERROR;
    }

    std::lock_guard lock(mutex);
    if (!uploads.empty() &&
        (uploads.front().unitByteSize != unitByteSize ||
         uploads.front().generation != generation)) {
        return Result::ERROR;
    }
    return queueLocked(start, unitByteSize, source + sourceOffset, byteSize, generation);
}

std::vector<Transfer::PendingUpload> Transfer::PendingUploadQueue::take() {
    std::lock_guard lock(mutex);
    auto result = std::move(uploads);
    uploads.clear();
    return result;
}

Result Transfer::PendingUploadQueue::restore(std::vector<PendingUpload> restored) {
    if (restored.empty()) {
        return Result::SUCCESS;
    }

    const U64 restoredUnitByteSize = restored.front().unitByteSize;
    const U64 restoredGeneration = restored.front().generation;
    for (const auto& upload : restored) {
        U64 extent = 0;
        U64 end = 0;
        if (!PendingUploadRange(upload, extent, end) ||
            upload.unitByteSize != restoredUnitByteSize ||
            upload.generation != restoredGeneration) {
            return Result::ERROR;
        }
    }

    std::lock_guard lock(mutex);

    for (const auto& upload : uploads) {
        U64 extent = 0;
        U64 end = 0;
        if (!PendingUploadRange(upload, extent, end) ||
            upload.unitByteSize != uploads.front().unitByteSize ||
            upload.generation != uploads.front().generation) {
            return Result::ERROR;
        }
    }

    if (!uploads.empty() &&
        (uploads.front().unitByteSize != restoredUnitByteSize ||
         uploads.front().generation != restoredGeneration)) {
        return Result::SUCCESS;
    }

    // Updates queued after take() are newer and must win over restored data.
    auto current = std::move(uploads);
    uploads.clear();
    for (const auto& upload : restored) {
        const Result result = queueLocked(upload.start,
                                          upload.unitByteSize,
                                          upload.data.data(),
                                          upload.data.size(),
                                          upload.generation);
        if (result != Result::SUCCESS) {
            uploads = std::move(current);
            return result;
        }
    }
    for (const auto& upload : current) {
        const Result result = queueLocked(upload.start,
                                          upload.unitByteSize,
                                          upload.data.data(),
                                          upload.data.size(),
                                          upload.generation);
        if (result != Result::SUCCESS) {
            uploads = std::move(current);
            return result;
        }
    }

    return Result::SUCCESS;
}

void Transfer::PendingUploadQueue::clear() {
    std::lock_guard lock(mutex);
    uploads.clear();
}

bool Transfer::PendingUploadQueue::empty() const {
    std::lock_guard lock(mutex);
    return uploads.empty();
}

Result Transfer::PendingUploadQueue::queueLocked(const U64& start,
                                                 const U64& unitByteSize,
                                                 const U8* source,
                                                 const U64& byteSize,
                                                 const U64& generation) {
    if (!source || unitByteSize == 0 || byteSize == 0 ||
        byteSize % unitByteSize != 0 ||
        byteSize > std::numeric_limits<size_t>::max() ||
        start > std::numeric_limits<U64>::max() / unitByteSize) {
        return Result::ERROR;
    }

    const U64 count = byteSize / unitByteSize;
    if (start > std::numeric_limits<U64>::max() - count) {
        return Result::ERROR;
    }

    for (const auto& upload : uploads) {
        U64 extent = 0;
        U64 end = 0;
        if (!PendingUploadRange(upload, extent, end) ||
            upload.unitByteSize != unitByteSize ||
            upload.generation != generation) {
            return Result::ERROR;
        }
    }

    try {
        if (uploads.size() == uploads.capacity()) {
            uploads.reserve(uploads.size() + 1);
        }
    } catch (const std::bad_alloc&) {
        return Result::ERROR;
    } catch (const std::length_error&) {
        return Result::ERROR;
    }

    U64 mergedStart = start;
    U64 mergedEnd = start + count;

    auto first = uploads.begin();
    while (first != uploads.end()) {
        U64 extent = 0;
        U64 end = 0;
        PendingUploadRange(*first, extent, end);
        if (end >= mergedStart) {
            break;
        }
        ++first;
    }

    auto last = first;
    while (last != uploads.end() && last->start <= mergedEnd) {
        U64 extent = 0;
        U64 end = 0;
        PendingUploadRange(*last, extent, end);
        mergedStart = std::min(mergedStart, last->start);
        mergedEnd = std::max(mergedEnd, end);
        ++last;
    }

    U64 mergedByteSize = 0;
    if (!CheckedMultiply(mergedEnd - mergedStart,
                         unitByteSize,
                         mergedByteSize) ||
        mergedByteSize > std::numeric_limits<size_t>::max()) {
        return Result::ERROR;
    }

    PendingUpload merged;
    merged.start = mergedStart;
    merged.unitByteSize = unitByteSize;
    merged.generation = generation;
    if (mergedByteSize > merged.data.max_size()) {
        return Result::ERROR;
    }
    try {
        merged.data.resize(static_cast<size_t>(mergedByteSize));
    } catch (const std::bad_alloc&) {
        return Result::ERROR;
    } catch (const std::length_error&) {
        return Result::ERROR;
    }

    for (auto upload = first; upload != last; ++upload) {
        U64 offset = 0;
        if (!CheckedMultiply(upload->start - mergedStart,
                             unitByteSize,
                             offset) ||
            offset > mergedByteSize ||
            upload->data.size() > mergedByteSize - offset) {
            return Result::ERROR;
        }
        std::memcpy(merged.data.data() + offset,
                    upload->data.data(),
                    upload->data.size());
    }

    U64 offset = 0;
    if (!CheckedMultiply(start - mergedStart, unitByteSize, offset) ||
        offset > mergedByteSize || byteSize > mergedByteSize - offset) {
        return Result::ERROR;
    }
    std::memcpy(merged.data.data() + offset, source, static_cast<size_t>(byteSize));

    const auto position = uploads.erase(first, last);
    uploads.insert(position, std::move(merged));
    return Result::SUCCESS;
}

Transfer::Batch::Batch() = default;

Transfer::Batch::~Batch() {
    if (committed) {
        return;
    }

    for (auto& transfer : bufferTransfers) {
        std::vector<PendingUpload> uploads;
        uploads.push_back(std::move(transfer.upload));
        transfer.destination->restorePendingUploads(std::move(uploads));
    }

    for (auto& transfer : textureTransfers) {
        std::vector<PendingUpload> uploads;
        uploads.push_back(std::move(transfer.upload));
        transfer.destination->restorePendingUploads(std::move(uploads));
    }
}

bool Transfer::Batch::empty() const {
    return bufferTransfers.empty() && textureTransfers.empty();
}

bool Transfer::Batch::contains(const std::shared_ptr<Buffer>& buffer) const {
    return buffer && pendingBuffers.contains(buffer.get());
}

bool Transfer::Batch::contains(const std::shared_ptr<Texture>& texture) const {
    return texture && pendingTextures.contains(texture.get());
}

void Transfer::Batch::collect(const std::shared_ptr<Buffer>& buffer) {
    if (!buffer || !collectedBuffers.insert(buffer.get()).second) {
        return;
    }

    auto uploads = buffer->pendingUploads.take();
    if (!uploads.empty()) {
        pendingBuffers.insert(buffer.get());
    }
    for (auto& upload : uploads) {
        BufferTransfer transfer;
        transfer.destination = buffer;
        transfer.destinationOffset = upload.start * upload.unitByteSize;
        transfer.upload = std::move(upload);
        bufferTransfers.push_back(std::move(transfer));
    }
}

void Transfer::Batch::collect(const std::shared_ptr<Texture>& texture) {
    if (!texture || !collectedTextures.insert(texture.get()).second) {
        return;
    }

    std::lock_guard lock(texture->uploadMutex);
    auto uploads = texture->pendingUploads.take();
    if (!uploads.empty()) {
        pendingTextures.insert(texture.get());
    }
    for (auto& upload : uploads) {
        TextureTransfer transfer;
        transfer.destination = texture;
        transfer.destinationRow = upload.start;
        transfer.rowByteSize = upload.unitByteSize;
        transfer.destinationSize = texture->config.size;
        transfer.upload = std::move(upload);
        textureTransfers.push_back(std::move(transfer));
    }
}

void Transfer::Batch::commit() {
    committed = true;
    bufferTransfers.clear();
    textureTransfers.clear();
    collectedBuffers.clear();
    collectedTextures.clear();
    pendingBuffers.clear();
    pendingTextures.clear();
}

const std::vector<Transfer::BufferTransfer>& Transfer::Batch::buffers() const {
    return bufferTransfers;
}

const std::vector<Transfer::TextureTransfer>& Transfer::Batch::textures() const {
    return textureTransfers;
}

}  // namespace Jetstream::Render
