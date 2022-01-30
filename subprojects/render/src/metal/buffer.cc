#include "render/metal/buffer.hpp"

namespace Render {

Metal::Buffer::Buffer(const Config& config, const Metal& instance)
         : Render::Buffer(config), instance(instance) {
}

Result Metal::Buffer::create() {
    const auto& byteSize = config.size * config.elementByteSize;

    // TODO: Add usage hints.
    buffer = instance.getDevice()->newBuffer(config.buffer, byteSize,
            MTL::ResourceStorageModeShared);
    RENDER_ASSERT(buffer);

    return Result::SUCCESS;
}

Result Metal::Buffer::destroy() {
    if (buffer) {
        buffer->release();
    }
    buffer = nullptr;

    return Result::SUCCESS;
}

Result Metal::Buffer::update() {
    return this->update(0, config.size);
}

Result Metal::Buffer::update(const std::size_t& offset, const std::size_t& size) {
    const auto& byteOffset = offset * config.elementByteSize;
    const auto& byteSize = size * config.elementByteSize;

    uint8_t* ptr = static_cast<uint8_t*>(buffer->contents());
    memcpy(ptr + byteOffset, (uint8_t*)config.buffer + byteOffset, byteSize);
    buffer->didModifyRange(NS::Range(byteOffset, byteOffset + byteSize));

    return Result::SUCCESS;
}

}  // namespace Render
