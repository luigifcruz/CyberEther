#ifndef RENDER_GLES_SURFACE_H
#define RENDER_GLES_SURFACE_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Surface : public Render::Surface {
public:
    Surface(Config& cfg, GLES& i) : Render::Surface(cfg), inst(i) {};

    std::shared_ptr<Render::Texture> bind(Render::Texture::Config&);
    std::shared_ptr<Render::Program> bind(Render::Program::Config&);

protected:
    GLES& inst;

    uint fbo = 0;
    std::shared_ptr<GLES::Texture> texture;
    std::vector<std::shared_ptr<GLES::Program>> programs;

    Result create();
    Result destroy();
    Result draw();

    friend class GLES;
};

} // namespace Render

#endif
