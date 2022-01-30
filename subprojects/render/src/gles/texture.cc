#include "render/gles/texture.hpp"

namespace Render {

GLES::Texture::Texture(const Config& config, const GLES& instance)
         : Render::Texture(config), instance(instance) {
}

Result GLES::Texture::create() {
    pfmt = convertPixelFormat(config.pfmt);
    ptype = convertPixelType(config.ptype);
    dfmt = convertDataFormat(config.dfmt);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, config.size.width, config.size.height, 0, pfmt, ptype, config.buffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::destroy() {
    glDeleteTextures(1, &tex);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::begin() {
    glBindTexture(GL_TEXTURE_2D, tex);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::end() {
    glBindTexture(GL_TEXTURE_2D, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

// After this method, user needs to recreate the texture
bool GLES::Texture::size(const Size2D<int>& size) {
    if (size <= Size2D<int>{1, 1}) {
        return false;
    }

    if (config.size != size) {
        config.size = size;
        return true;
    }

    return false;
}

void* GLES::Texture::raw() {
    return reinterpret_cast<void*>(tex);
}

Result GLES::Texture::fill() {
    CHECK(this->begin());
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, config.size.width, config.size.height, 0, pfmt, ptype, config.buffer);
    CHECK(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::fillRow(const std::size_t& y, const std::size_t& height) {
    if (height < 1) {
        return Result::SUCCESS;
    }

    CHECK(this->begin());
    size_t offset = y * config.size.width * ((config.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint8_t));
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, config.size.width, height, pfmt, ptype, config.buffer + offset);
    CHECK(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

}  // namespace Render
