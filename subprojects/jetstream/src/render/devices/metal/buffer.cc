#include "jetstream/render/metal/buffer.hh"

namespace Jetstream::Render {

using Implementation = BufferImp<Device::Metal>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
    JST_INFO("Greetings from the Buffer Metal thingy.");
}

const Result Implementation::create() {
    JST_DEBUG("Creating Metal buffer.");

    // TODO: Add usage hints.
    const auto& byteSize = config.size * config.elementByteSize;
    auto device = Backend::State<Device::Metal>()->getDevice();
    buffer = device->newBuffer(config.buffer, byteSize, 
                               MTL::ResourceStorageModeShared); 
    JST_ASSERT(buffer);

    return Result::SUCCESS;
}

const Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal buffer.");

    if (buffer) {
        buffer->release();
    }
    buffer = nullptr;

    return Result::SUCCESS;
}

const Result Implementation::update() {
    return update(0, config.size);
}

const Result Implementation::update(const U64& offset, const U64& size) {
    const auto& byteOffset = offset * config.elementByteSize;
    const auto& byteSize = size * config.elementByteSize;

    uint8_t* ptr = static_cast<uint8_t*>(buffer->contents());
    memcpy(ptr + byteOffset, (uint8_t*)config.buffer + byteOffset, byteSize);
    buffer->didModifyRange(NS::Range(byteOffset, byteOffset + byteSize));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
