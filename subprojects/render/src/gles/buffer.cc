#include "render/gles/buffer.hpp"

namespace Render {

GLES::Buffer::Buffer(const Config& config, const GLES& instance)
         : Render::Buffer(config), instance(instance) {
    switch (config.target) {
        case Target::VERTEX:
            target = GL_ARRAY_BUFFER;
            break;
        case Target::VERTEX_INDICES:
            target = GL_ELEMENT_ARRAY_BUFFER;
            break;
        case Target::STORAGE:
            target = GL_SHADER_STORAGE_BUFFER;
            break;
    }
}

Result GLES::Buffer::create() {
    const auto& byteSize = config.size * config.elementByteSize;

    // TODO: Add usage hint.
    glGenBuffers(1, &id);
    CHECK(this->begin());
    glBufferData(target, byteSize, config.buffer, GL_DYNAMIC_DRAW);
    CHECK(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Buffer::destroy() {
    glDeleteBuffers(1, &id);

    return Result::SUCCESS;
}

Result GLES::Buffer::begin() {
    glBindBuffer(target, id);

    return Result::SUCCESS;
}

Result GLES::Buffer::end() {
    glBindBuffer(target, 0);

    return Result::SUCCESS;
}

Result GLES::Buffer::update() {
    return this->update(0, config.size);
}

Result GLES::Buffer::update(const std::size_t& offset, const std::size_t& size) {
    const auto& byteOffset = offset * config.elementByteSize;
    const auto& byteSize = size * config.elementByteSize;

    CHECK(this->begin());
    glBufferSubData(target, byteOffset, byteSize, (uint8_t*)config.buffer + byteOffset);
    CHECK(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

}  // namespace Render
