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
    Result fill() final;
    Result fillRow(const std::size_t& y, const std::size_t& height) final;

 protected:
    Result create();
    Result destroy();

    constexpr const MTL::PixelFormat getPixelFormat() const {
        return pixelFormat;
    }

    constexpr const MTL::Texture* getHandle() const {
        return texture;
    }

 private:
    const Metal& instance;

    MTL::Texture* texture = nullptr;
    MTL::PixelFormat pixelFormat;

    friend class Metal::Surface;
    friend class Metal::Program;
};

}  // namespace Render

#endif
