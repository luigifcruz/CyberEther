#include "jetstream/render/devices/metal/buffer.hh"

namespace Jetstream::Render {

using Implementation = BufferImp<Device::Metal>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[Metal] Creating buffer.");

    // TODO: Add usage hints.
    auto device = Backend::State<Device::Metal>()->getDevice();
    const auto& byteSize = config.size * config.elementByteSize;

    if (config.enableZeroCopy) {
        buffer = reinterpret_cast<MTL::Buffer*>(config.buffer);
    } else {
        buffer = device->newBuffer(config.buffer,
                                   byteSize, 
                                   MTL::ResourceStorageModeShared); 
    }
    JST_ASSERT(buffer);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[Metal] Destroying buffer.");

    if (!config.enableZeroCopy && buffer) {
        buffer->release();
    }
    buffer = nullptr;

    return Result::SUCCESS;
}

Result Implementation::update() {
    return update(0, config.size);
}

Result Implementation::update(const U64& offset, const U64& size) {
    if (size == 0 || config.enableZeroCopy) {
        return Result::SUCCESS;
    }

    const auto& byteOffset = offset * config.elementByteSize;
    const auto& byteSize = size * config.elementByteSize;

    uint8_t* ptr = static_cast<uint8_t*>(buffer->contents());
    memcpy(ptr + byteOffset, (uint8_t*)config.buffer + byteOffset, byteSize);
#if !defined(TARGET_OS_IOS)
    buffer->didModifyRange(NS::Range(byteOffset, byteOffset + byteSize));
#endif

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
