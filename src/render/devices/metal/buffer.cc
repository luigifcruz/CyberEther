#include "jetstream/render/devices/metal/buffer.hh"

namespace Jetstream::Render {

using Implementation = BufferImp<DeviceType::Metal>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[METAL] Creating buffer.");

    if (config.buffer && !config.enableZeroCopy) {
        JST_CHECK(validateUpdate(0, config.size));
    }

    // TODO: Add usage hints.
    auto device = Backend::State<DeviceType::Metal>()->getDevice();
    const auto& byteSize = config.size * config.elementByteSize;

    if (config.enableZeroCopy) {
        buffer = static_cast<MTL::Buffer*>(config.buffer);
        buffer->retain();
    } else {
        buffer = device->newBuffer(byteSize, MTL::ResourceStorageModeShared);
    }
    JST_ASSERT(buffer, "Failed to create buffer.");

    if (config.buffer && !config.enableZeroCopy) {
        JST_CHECK(update());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[METAL] Destroying buffer.");

    if (buffer) {
        buffer->release();
    }
    buffer = nullptr;

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
