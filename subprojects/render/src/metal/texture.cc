#include "render/metal/texture.hpp"

namespace Render {

Result Metal::Texture::create() {
    /*
    if (!Metal::cudaInteropSupported() && cfg.cudaInterop) {
        cfg.cudaInterop = false;
        return Result::ERROR;
    }

    pfmt = convertPixelFormat(cfg.pfmt);
    ptype = convertPixelType(cfg.ptype);
    dfmt = convertDataFormat(cfg.dfmt);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    auto ptr = (cfg.cudaInterop) ? nullptr : cfg.buffer;
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.size.width, cfg.size.height, 0, pfmt, ptype, ptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::destroy() {
    /*
    glDeleteTextures(1, &tex);

    if (cfg.cudaInterop) {
#ifdef RENDER_CUDA_AVAILABLE
        cudaGraphicsUnregisterResource(cuda_tex_resource);
        cudaStreamDestroy(stream);
#endif
    }
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::begin() {
    //glBindTexture(GL_TEXTURE_2D, tex);

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::end() {
    //glBindTexture(GL_TEXTURE_2D, 0);

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

// After this method, user needs to recreate the texture
bool Metal::Texture::size(const Size2D<int>& size) {
    if (size <= Size2D<int>{1, 1}) {
        return false;
    }

    if (cfg.size != size) {
        cfg.size = size;
        return true;
    }

    return false;
}

uint Metal::Texture::raw() {
    return tex;
}

Result Metal::Texture::fill() {
    /*
    if (cfg.cudaInterop) {
        return this->_cudaCopyToTexture(0, 0, cfg.size.width, cfg.size.height);
    }

    CHECK(this->begin());
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.size.width, cfg.size.height, 0, pfmt, ptype, cfg.buffer);
    CHECK(this->end());
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::fill(int yo, int xo, int w, int h) {
    /*
    if (cfg.cudaInterop) {
        return this->_cudaCopyToTexture(yo, xo, w, h);
    }

    CHECK(this->begin());
    size_t offset = yo * w * ((cfg.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint));
    glTexSubImage2D(GL_TEXTURE_2D, 0, xo, yo, w, h, pfmt, ptype, cfg.buffer + offset);
    CHECK(this->end());
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::pour() {
    // TBD
    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
