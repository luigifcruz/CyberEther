#include "jetstream/render/base/buffer.hh"

#include <limits>
#include <utility>

namespace Jetstream::Render {

Buffer::Buffer(const Config& config) : config(config) {}

Result Buffer::update() {
    return update(0, config.size);
}

Result Buffer::update(const U64& offset, const U64& size) {
    const Result validation = validateUpdate(offset, size);
    if (validation != Result::SUCCESS) {
        return validation;
    }

    if (size == 0) {
        return Result::SUCCESS;
    }

    if (config.enableZeroCopy) {
        return Result::SUCCESS;
    }

    return pendingUploads.queue(offset,
                                size,
                                config.size,
                                config.elementByteSize,
                                static_cast<const U8*>(config.buffer));
}

Result Buffer::validateUpdate(const U64& offset, const U64& size) const {
    if (size == 0 || config.enableZeroCopy) {
        return Result::SUCCESS;
    }

    if (!config.buffer || config.elementByteSize == 0 ||
        offset > config.size || size > config.size - offset ||
        offset > std::numeric_limits<U64>::max() / config.elementByteSize ||
        size > std::numeric_limits<U64>::max() / config.elementByteSize) {
        return Result::ERROR;
    }

    if ((offset * config.elementByteSize) % UploadAlignment != 0 ||
        (size * config.elementByteSize) % UploadAlignment != 0) {
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

void Buffer::restorePendingUploads(std::vector<Transfer::PendingUpload> uploads) {
    if (pendingUploads.restore(std::move(uploads)) != Result::SUCCESS) {
        JST_ERROR("[BUFFER] Failed to restore pending uploads.");
    }
}

}  // namespace Jetstream::Render
