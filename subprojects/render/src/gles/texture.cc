#include "render/gles/texture.hpp"

namespace Render {

Result GLES::Texture::create() {
    pfmt = convertPixelFormat(cfg.pfmt);
    ptype = convertPixelType(cfg.ptype);
    dfmt = convertDataFormat(cfg.dfmt);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.width, cfg.height, 0, pfmt, ptype, cfg.buffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
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

Result GLES::Texture::fill() {
    RENDER_ASSERT_SUCCESS(this->start());
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.width, cfg.height, 0, pfmt, ptype, cfg.buffer);
    RENDER_ASSERT_SUCCESS(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::fill(int yo, int xo, int w, int h) {
    RENDER_ASSERT_SUCCESS(this->start());
    size_t offset = yo * w * ((cfg.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint));
    glTexSubImage2D(GL_TEXTURE_2D, 0, xo, yo, w, h, pfmt, ptype, cfg.buffer + offset);
    RENDER_ASSERT_SUCCESS(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::pour() {
    // TBD
    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
