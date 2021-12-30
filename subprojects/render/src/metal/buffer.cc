#include "render/metal/buffer.hpp"

namespace Render {

Metal::Buffer::Buffer(const Config& config, const Metal& instance)
         : Render::Buffer(config), instance(instance) {
}

Result Metal::Buffer::create() {
    const auto& byteSize = config.size * config.elementByteSize;

    // TODO: Add usage hints.
    buffer = instance.getDevice()->newBuffer(byteSize, MTL::ResourceStorageModeShared);
    RENDER_ASSERT(buffer);

    return Result::SUCCESS;
}

Result Metal::Buffer::destroy() {
    buffer->release();

    return Result::SUCCESS;
}

void* Metal::Buffer::raw() {
    return buffer;
}

Result Metal::Buffer::fill() {
    return this->fill(0, config.size);
}

Result Metal::Buffer::fill(const std::size_t& offset, const std::size_t& size) {
    const auto& byteOffset = offset * config.elementByteSize;
    const auto& byteSize = size * config.elementByteSize;

    uint8_t* ptr = static_cast<uint8_t*>(buffer->contents());
    memcpy(ptr + byteOffset, config.buffer + byteOffset, byteSize);
    buffer->didModifyRange(NS::Range(byteOffset, byteOffset + byteSize));

    return Result::SUCCESS;
}

Result Metal::Buffer::pour() {
    // TODO: Implement it.
    return Result::SUCCESS;
}

}  // namespace Render
