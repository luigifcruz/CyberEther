#ifndef RENDER_METAL_VERTEX_H
#define RENDER_METAL_VERTEX_H

#include <vector>

#include "render/metal/instance.hpp"
#include "render/metal/buffer.hpp"

namespace Render {

class Metal::Vertex : public Render::Vertex {
 public:
    explicit Vertex(const Config& config, const Metal& instance);

 protected:
    Result create();
    Result destroy();
    Result encode(MTL::RenderCommandEncoder* encoder, const std::size_t& offset);

    constexpr const MTL::Buffer* getIndexBuffer() {
        return indices->getHandle();
    }

    constexpr const std::size_t getVertexCount() {
        return vertex_count;
    }

    const bool isBuffered() {
        return indices != nullptr;
    }

 private:
    const Metal& instance;

    std::size_t vertex_count;
    std::vector<std::pair<std::shared_ptr<Metal::Buffer>, uint32_t>> buffers;
    std::shared_ptr<Metal::Buffer> indices;

    friend class Metal::Draw;
};

}  // namespace Render

#endif
