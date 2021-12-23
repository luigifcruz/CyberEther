#include "render/gles/texture.hpp"

namespace Render {

GLES::Texture::Texture(const Config& config, const GLES& instance)
         : Render::Texture(config), instance(instance) {
}

Result GLES::Texture::create() {
    if (!GLES::cudaInteropSupported() && config.cudaInterop) {
        config.cudaInterop = false;
        return Result::ERROR;
    }

    pfmt = convertPixelFormat(config.pfmt);
    ptype = convertPixelType(config.ptype);
    dfmt = convertDataFormat(config.dfmt);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    auto ptr = (config.cudaInterop) ? nullptr : config.buffer;
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, config.size.width, config.size.height, 0, pfmt, ptype, ptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (config.cudaInterop) {
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

    if (config.cudaInterop) {
#ifdef RENDER_CUDA_AVAILABLE
        cudaGraphicsUnregisterResource(cuda_tex_resource);
        cudaStreamDestroy(stream);
#endif
    }

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

Result GLES::Texture::cudaCopyToTexture(int yo, int xo, int w, int h) {
#ifdef RENDER_CUDA_AVAILABLE
    size_t i = (config.pfmt == PixelFormat::RED) ? 1 : 3;
    size_t m = i * ((config.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint));
    size_t o = (yo * w * m) + (xo * m);

    cudaArray *texture_ptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_tex_resource, stream));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));
    CUDA_CHECK(cudaMemcpy2DToArrayAsync(texture_ptr, xo*m, yo, config.buffer+o, w*m, w*m, h,
            cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_tex_resource, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Result::SUCCESS;
#endif
    return Result::ERROR;
}


Result GLES::Texture::fill() {
    if (config.cudaInterop) {
        return cudaCopyToTexture(0, 0, config.size.width, config.size.height);
    }

    CHECK(this->begin());
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, config.size.width, config.size.height, 0, pfmt, ptype, config.buffer);
    CHECK(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::fillRow(const std::size_t& y, const std::size_t& height) {
    if (config.cudaInterop) {
        return this->cudaCopyToTexture(y, 0, config.size.width, config.size.height);
    }

    CHECK(this->begin());
    size_t offset = y * config.size.width * ((config.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint));
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, config.size.width, config.size.height, pfmt, ptype, config.buffer + offset);
    CHECK(this->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Texture::pour() {
    // TODO: Implement it.
    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

}  // namespace Render
