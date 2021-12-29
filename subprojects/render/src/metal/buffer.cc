#include "render/metal/buffer.hpp"

namespace Render {

Metal::Buffer::Buffer(const Config& config, const Metal& instance)
         : Render::Buffer(config), instance(instance) {
}

Result Metal::Buffer::create() {
    buffer = instance.getDevice()->newBuffer(config.size, MTL::ResourceStorageModeManaged);
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
    memcpy((uint8_t*)buffer->contents() + offset, config.buffer + offset, size);
    buffer->didModifyRange(NS::Range(0, size));

    return Result::SUCCESS;
}

Result Metal::Buffer::pour() {
    // TODO: Implement it.
    return Result::SUCCESS;
}

}  // namespace Render
