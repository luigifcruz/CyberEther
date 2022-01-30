#ifndef RENDER_GLES_VERTEX_H
#define RENDER_GLES_VERTEX_H

#include <vector>
#include <utility>
#include <memory>

#include "render/gles/instance.hpp"
#include "render/gles/buffer.hpp"

namespace Render {

class GLES::Vertex : public Render::Vertex {
 public:
    explicit Vertex(const Config& config, const GLES& instance);

 protected:
    Result create();
    Result destroy();
    Result begin();
    Result end();

    const uint* getIndexBuffer() const {
        return indices->getHandle();
    }

    constexpr const std::size_t getVertexCount() const {
        return vertex_count;
    }

    const bool isBuffered() const {
        return indices != nullptr;
    }

 private:
    const GLES& instance;

    uint vao;
    std::size_t vertex_count;
    std::vector<std::pair<std::shared_ptr<GLES::Buffer>, uint32_t>> buffers;
    std::shared_ptr<GLES::Buffer> indices;

    friend class GLES::Draw;
};

}  // namespace Render

#endif
