#ifndef RENDER_GLES_VERTEX_H
#define RENDER_GLES_VERTEX_H

#include "render/gles/instance.hpp"

#ifdef RENDER_CUDA_AVAILABLE
#include <cuda_gl_interop.h>
#endif

namespace Render {

class GLES::Vertex : public Render::Vertex {
public:
    Vertex(const Config& cfg, const GLES& i) : Render::Vertex(cfg), inst(i) {};

    Result update() final;

protected:
    const GLES& inst;

    uint vao, ebo;
    uint vertex_count;

    Result create() final;
    Result destroy() final;
    Result begin() final;
    Result end() final;

    uint count();
    uint buffered();

#ifdef RENDER_CUDA_AVAILABLE
    cudaStream_t stream;
#endif

    friend class GLES::Draw;
};

} // namespace Render

#endif
