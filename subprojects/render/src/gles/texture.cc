#include "render/gles/texture.hpp"

namespace Render {

Result GLES::Texture::create() {
    if (!GLES::cudaInteropSupported() && cfg.cudaInterop) {
        cfg.cudaInterop = false;
        return Result::ERROR;
    }

    pfmt = convertPixelFormat(cfg.pfmt);
    ptype = convertPixelType(cfg.ptype);
    dfmt = convertDataFormat(cfg.dfmt);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    auto ptr = (cfg.cudaInterop) ? nullptr : cfg.buffer;
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.width, cfg.height, 0, pfmt, ptype, ptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (cfg.cudaInterop) {
#ifdef RENDER_CUDA_INTEROP_AVAILABLE
        cudaGraphicsGLRegisterImage(&cuda_tex_resource, tex, GL_TEXTURE_2D,
                cudaGraphicsMapFlagsWriteDiscard);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
#endif
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::destroy() {
    glDeleteTextures(1, &tex);

    if (cfg.cudaInterop) {
#ifdef RENDER_CUDA_INTEROP_AVAILABLE
        cudaGraphicsUnregisterResource(cuda_tex_resource);
        cudaStreamDestroy(stream);
#endif
    }

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

Result GLES::Texture::_cudaCopyToTexture(int yo, int xo, int w, int h) {
#ifdef RENDER_CUDA_INTEROP_AVAILABLE
    size_t i = (cfg.pfmt == PixelFormat::RED) ? 1 : 3;
    size_t m = i * ((cfg.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint));

    cudaArray *texture_ptr;
    cudaGraphicsMapResources(1, &cuda_tex_resource, stream);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0);
    cudaMemcpy2DToArrayAsync(texture_ptr, xo*m, yo, cfg.buffer, w*m, w*m, h, cudaMemcpyDeviceToDevice, stream);
    cudaGraphicsUnmapResources(1, &cuda_tex_resource, stream);

    return Result::SUCCESS;
#endif
    return Result::ERROR;
}


Result GLES::Texture::fill() {
    if (cfg.cudaInterop) {
        return this->_cudaCopyToTexture(0, 0, cfg.width, cfg.height);
    }

    RENDER_ASSERT_SUCCESS(this->start());
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.width, cfg.height, 0, pfmt, ptype, cfg.buffer);
    RENDER_ASSERT_SUCCESS(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::fill(int yo, int xo, int w, int h) {
    if (cfg.cudaInterop) {
        return this->_cudaCopyToTexture(yo, xo, w, h);
    }

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
