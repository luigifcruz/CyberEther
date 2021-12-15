#ifndef RENDER_METAL_TEXTURE_H
#define RENDER_METAL_TEXTURE_H

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Texture : public Render::Texture {
public:
    Texture(const Config& cfg, const Metal& i) : Render::Texture(cfg), inst(i) {};

    using Render::Texture::size;
    bool size(const Size2D<int>&) final;

    uint raw() final;
    Result pour() final;
    Result fill() final;
    Result fill(int, int, int, int) final;

protected:
    const Metal& inst;

    uint tex, pfmt, dfmt, ptype;

    Result create() final;
    Result destroy() final;
    Result begin() final;
    Result end() final;

    friend class Metal::Surface;
    friend class Metal::Program;
};

} // namespace Render

#endif
