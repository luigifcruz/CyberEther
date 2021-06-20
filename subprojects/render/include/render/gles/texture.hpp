#ifndef RENDER_GLES_TEXTURE_H
#define RENDER_GLES_TEXTURE_H

#include "render/gles/instance.hpp"

#ifdef RENDER_CUDA_AVAILABLE
#include <cuda_gl_interop.h>
#endif

namespace Render {

class GLES::Texture : public Render::Texture {
public:
    Texture(Config& cfg, GLES& i) : Render::Texture(cfg), inst(i) {};

    using Render::Texture::size;
    Size2D<int> size(const Size2D<int> &) final;

    uint raw();
    Result pour();
    Result fill();
    Result fill(int, int, int, int);

protected:
    GLES& inst;

    uint tex, pfmt, dfmt, ptype;

    Result create();
    Result destroy();
    Result start();
    Result end();

#ifdef RENDER_CUDA_AVAILABLE
    cudaGraphicsResource* cuda_tex_resource = nullptr;
    cudaStream_t stream;
#endif
    Result _cudaCopyToTexture(int, int, int, int);

    friend class GLES::Surface;
    friend class GLES::Program;
};

} // namespace Render

#endif
