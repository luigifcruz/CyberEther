#ifndef RENDER_GLES_VERTEX_H
#define RENDER_GLES_VERTEX_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Vertex : public Render::Vertex {
 public:
    explicit Vertex(const Config& config, const GLES& instance);

 protected:
    Result create();
    Result destroy();
    Result begin();
    Result end();

    uint count();
    uint buffered();

 private:
    const GLES& instance;

    uint vao, ebo;
    uint vertex_count;

    friend class GLES::Draw;
};

}  // namespace Render

#endif
