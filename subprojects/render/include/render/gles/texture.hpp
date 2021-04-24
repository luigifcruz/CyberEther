#ifndef RENDER_GLES_TEXTURE_H
#define RENDER_GLES_TEXTURE_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Texture : public Render::Texture {
public:
    Texture(Config& cfg, GLES& i) : Render::Texture(cfg), inst(i) {};

    uint raw();
    Result pour();
    Result fill();
    Result fill(int, int, int, int);

protected:
    GLES& inst;

    uint tex;

    Result create();
    Result destroy();
    Result start();
    Result end();

    friend class GLES::Surface;
    friend class GLES::Program;
};

} // namespace Render

#endif
