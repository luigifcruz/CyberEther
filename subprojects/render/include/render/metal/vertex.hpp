#ifndef RENDER_METAL_VERTEX_H
#define RENDER_METAL_VERTEX_H

#include "render/metal/instance.hpp"

#ifdef RENDER_CUDA_AVAILABLE
#include <cuda_gl_interop.h>
#endif

namespace Render {

class Metal::Vertex : public Render::Vertex {
public:
    Vertex(const Config& cfg, const Metal& i) : Render::Vertex(cfg), inst(i) {};

    Result update() final;

protected:
    const Metal& inst;

    uint vao, ebo;
    uint vertex_count;

    Result create() final;
    Result destroy() final;
    Result begin() final;
    Result end() final;

    uint count();
    uint buffered();

    friend class Metal::Draw;
};

} // namespace Render

#endif
