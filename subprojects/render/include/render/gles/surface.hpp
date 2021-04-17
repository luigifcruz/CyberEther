#ifndef RENDER_GLES_SURFACE_H
#define RENDER_GLES_SURFACE_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Surface : public Render::Surface {
public:
    Surface(Config& cfg, State& s) : Render::Surface(cfg), state(s) {};

    Result create();
    Result destroy();
    Result start();
    Result end();

protected:
    State& state;
    uint fbo;
};

} // namespace Render

#endif
