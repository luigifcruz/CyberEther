#include "jetstream/render/base/texture.hh"

#include <algorithm>
#include <limits>
#include <utility>

namespace Jetstream::Render {

Texture::Texture(const Config& config) : config(config) {}

bool Texture::size(const Extent2D<U64>& size) {
    if (size <= Extent2D<U64>{1, 1}) {
        return false;
    }

    std::lock_guard lock(uploadMutex);
    if (config.size != size) {
        uploadGeneration.fetch_add(1, std::memory_order_release);
        pendingUploads.clear();
        config.size = size;
        return true;
    }

    return false;
}

Result Texture::fill() {
    std::lock_guard lock(uploadMutex);
    return fillRowLocked(0, config.size.y);
}

Result Texture::fillRow(const U64& y, const U64& height) {
    std::lock_guard lock(uploadMutex);
    return fillRowLocked(y, height);
}

Result Texture::fillRowLocked(const U64& y, const U64& height) {
    const Result validation = validateFillRow(y, height);
    if (validation != Result::SUCCESS || height == 0) {
        return validation;
    }

    const U64 bytesPerPixel = pixelByteSize();
    return pendingUploads.queue(y,
                                height,
                                config.size.y,
                                config.size.x * bytesPerPixel,
                                config.buffer,
                                uploadGeneration.load(std::memory_order_acquire));
}

Result Texture::validateFillRow(const U64& y, const U64& height) const {
    if (height == 0) {
        return Result::SUCCESS;
    }

    const U64 bytesPerPixel = pixelByteSize();
    if (!config.buffer || config.multisampled ||
        y > config.size.y || height > config.size.y - y ||
        config.size.x > std::numeric_limits<U64>::max() / bytesPerPixel) {
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

U64 Texture::pixelByteSize() const {
    const U64 channels = config.pfmt == PixelFormat::RGBA ? 4 : 1;
    const U64 channelByteSize = config.ptype == PixelType::F32 ? 4 : 1;
    return channels * channelByteSize;
}

void Texture::restorePendingUploads(std::vector<Transfer::PendingUpload> uploads) {
    std::lock_guard lock(uploadMutex);
    const U64 generation = uploadGeneration.load(std::memory_order_acquire);
    std::erase_if(uploads, [&](const auto& upload) {
        return upload.generation != generation;
    });
    if (pendingUploads.restore(std::move(uploads)) != Result::SUCCESS) {
        JST_ERROR("[TEXTURE] Failed to restore pending uploads.");
    }
}

}  // namespace Jetstream::Render
