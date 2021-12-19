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
    Result create();
    Result destroy();
    Result encode(MTL::RenderCommandEncoder* encoder);

    constexpr const MTL::Buffer* getIndexBuffer() {
        return indexBuffer;
    }

    constexpr const std::size_t getVertexCount() {
        return vertex_count;
    }

    constexpr const bool isBuffered() {
        return indexBuffer != nullptr;
    }

private:
    const Metal& inst;

    std::size_t vertex_count;
    MTL::Buffer* indexBuffer = nullptr;
    std::vector<MTL::Buffer*> vertexBuffers;

    friend class Metal::Draw;
};

} // namespace Render

#endif
