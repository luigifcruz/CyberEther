#include "gles/texture.hpp"

namespace Render {

Result GLES::Texture::create() {
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cfg.width, cfg.height, 0,
            GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::destroy() {
    glDeleteTextures(1, &tex);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::start() {
    glBindTexture(GL_TEXTURE_2D, tex);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::end() {
    glBindTexture(GL_TEXTURE_2D, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

uint GLES::Texture::raw() {
    return tex;
}

Result GLES::Texture::fill(const uint8_t*) {
    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::pour(const uint8_t*) {
    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
