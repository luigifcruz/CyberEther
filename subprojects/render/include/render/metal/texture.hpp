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

    MTL::Texture* texture;

    constexpr const MTL::Texture* getTexture() {
        return texture;
    }

    Result create();
    Result destroy();

    friend class Metal::Surface;
    friend class Metal::Program;
};

} // namespace Render

#endif
