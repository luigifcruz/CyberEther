#ifndef RENDER_GLES_VERTEX_H
#define RENDER_GLES_VERTEX_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Vertex : public Render::Vertex {
public:
    Vertex(Config& cfg, GLES& i) : Render::Vertex(cfg), inst(i) {};

    Result update();

protected:
    GLES& inst;

    uint vao, ebo;

    Result create();
    Result destroy();
    Result draw();

    friend class GLES::Program;
};

} // namespace Render

#endif
