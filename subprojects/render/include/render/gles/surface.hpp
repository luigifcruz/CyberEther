#ifndef RENDER_GLES_SURFACE_H
#define RENDER_GLES_SURFACE_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Surface : public Render::Surface {
public:
    Surface(Config& cfg, GLES& i) : Render::Surface(cfg), inst(i) {};

    Result resize(int, int);

protected:
    GLES& inst;

    uint fbo = 0;
    std::shared_ptr<GLES::Texture> framebuffer;
    std::vector<std::shared_ptr<GLES::Program>> programs;

    Result create();
    Result destroy();
    Result draw();

    Result _createFramebuffer();
    Result _destroyFramebuffer();

    friend class GLES;
};

} // namespace Render

#endif
