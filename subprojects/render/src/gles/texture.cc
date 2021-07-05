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
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.size.width, cfg.size.height, 0, pfmt, ptype, ptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (cfg.cudaInterop) {
#ifdef RENDER_CUDA_AVAILABLE
        int leastPriority = -1, greatestPriority = -1;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_CHECK(cudaGraphicsGLRegisterImage(&cuda_tex_resource, tex, GL_TEXTURE_2D,
                cudaGraphicsMapFlagsNone));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
#endif
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::destroy() {
    glDeleteTextures(1, &tex);

    if (cfg.cudaInterop) {
#ifdef RENDER_CUDA_AVAILABLE
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

// After this method, user needs to recreate the texture
bool GLES::Texture::size(const Size2D<int> & size) {
    if (size <= Size2D<int>{1, 1}) {
        return false;
    }

    if (cfg.size != size) {
        cfg.size = size;
        return true;
    }

    return false;
}

uint GLES::Texture::raw() {
    return tex;
}

Result GLES::Texture::_cudaCopyToTexture(int yo, int xo, int w, int h) {
#ifdef RENDER_CUDA_AVAILABLE
    size_t i = (cfg.pfmt == PixelFormat::RED) ? 1 : 3;
    size_t m = i * ((cfg.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint));
    size_t o = (yo * w * m) + (xo * m);

    cudaArray *texture_ptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_tex_resource, stream));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));
    CUDA_CHECK(cudaMemcpy2DToArrayAsync(texture_ptr, xo*m, yo, cfg.buffer+o, w*m, w*m, h,
            cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_tex_resource, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Result::SUCCESS;
#endif
    return Result::ERROR;
}


Result GLES::Texture::fill() {
    if (cfg.cudaInterop) {
        return this->_cudaCopyToTexture(0, 0, cfg.size.width, cfg.size.height);
    }

    CHECK(this->start());
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.size.width, cfg.size.height, 0, pfmt, ptype, cfg.buffer);
    CHECK(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::fill(int yo, int xo, int w, int h) {
    if (cfg.cudaInterop) {
        return this->_cudaCopyToTexture(yo, xo, w, h);
    }

    CHECK(this->start());
    size_t offset = yo * w * ((cfg.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint));
    glTexSubImage2D(GL_TEXTURE_2D, 0, xo, yo, w, h, pfmt, ptype, cfg.buffer + offset);
    CHECK(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::pour() {
    // TBD
    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
