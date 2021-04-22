#ifndef RENDER_GLES_VERTEX_H
#define RENDER_GLES_VERTEX_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Vertex : public Render::Vertex {
public:
    Vertex(Config& cfg, State& s) : Render::Vertex(cfg), state(s) {};

    Result create();
    Result destroy();
    Result start();
    Result end();

    Result update();

protected:
    State& state;

    uint vao, ebo;
};

} // namespace Render

#endif

