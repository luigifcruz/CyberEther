#ifndef RENDER_GLES_SURFACE_H
#define RENDER_GLES_SURFACE_H

#include "base/surface.hpp"
#include "gles/api.hpp"
#include "gles/state.hpp"

namespace Render {

class GLES::Surface : public Render::Surface {
public:
    Surface(Config& cfg, State& s) : Render::Surface(cfg), state(s) {};

    Result create();
    Result destroy();
    Result start();
    Result end();

    void* getRawTexture();

private:
    State& state;
    uint fbo, tex;
};

} // namespace Render

#endif
