#ifndef RENDER_METAL_TEXTURE_H
#define RENDER_METAL_TEXTURE_H

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Texture : public Render::Texture {
public:
    explicit Texture(const Config& config, const Metal& instance);

    using Render::Texture::size;
    bool size(const Size2D<int>&) final;

    void* raw() final;
    Result pour() final;
    Result fill() final;
    Result fill(int, int, int, int) final;

protected:
    Result create();
    Result destroy();

    constexpr const MTL::PixelFormat getPixelFormat() {
        return pixelFormat;
    }

private:
    const Metal& instance;

    MTL::Texture* texture;
    MTL::PixelFormat pixelFormat;

    friend class Metal::Surface;
    friend class Metal::Program;
};

} // namespace Render

#endif
