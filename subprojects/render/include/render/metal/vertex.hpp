#ifndef RENDER_METAL_VERTEX_H
#define RENDER_METAL_VERTEX_H

#include <vector>

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Vertex : public Render::Vertex {
 public:
    explicit Vertex(const Config& config, const Metal& instance);

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
    const Metal& instance;

    std::size_t vertex_count;
    MTL::Buffer* indexBuffer = nullptr;
    std::vector<MTL::Buffer*> vertexBuffers;

    friend class Metal::Draw;
};

}  // namespace Render

#endif
