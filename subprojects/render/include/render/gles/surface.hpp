#ifndef RENDER_GLES_SURFACE_H
#define RENDER_GLES_SURFACE_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Surface : public Render::Surface {
public:
    Surface(Config& cfg, GLES& i) : Render::Surface(cfg), inst(i) {};

    Result create();
    Result destroy();
    Result draw();

protected:
    GLES& inst;

    uint fbo = 0;
};

} // namespace Render

#endif
