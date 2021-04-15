#ifndef RENDER_GLES_TEXTURE_H
#define RENDER_GLES_TEXTURE_H

#include "gles/api.hpp"
#include "gles/state.hpp"

namespace Render {

class GLES::Texture : public Render::Texture {
public:
    Texture(Config& cfg, State& s) : Render::Texture(cfg), state(s) {};

    Result create();
    Result destroy();
    Result start();
    Result end();

    uint raw();
    Result pour();
    Result fill();
    Result fill(int, int, int, int);

private:
    State& state;
    uint tex;
};

} // namespace Render

#endif
