#ifndef RENDER_GLES_TEXTURE_H
#define RENDER_GLES_TEXTURE_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Texture : public Render::Texture {
 public:
    explicit Texture(const Config& config, const GLES& instance);

    using Render::Texture::size;
    bool size(const Size2D<int>&) final;

    void* raw() final;
    Result fill() final;
    Result fillRow(const std::size_t& y, const std::size_t& height) final;

 protected:
    Result create();
    Result destroy();
    Result begin();
    Result end();

 private:
    const GLES& instance;

    uint tex, pfmt, dfmt, ptype;

    friend class GLES::Surface;
    friend class GLES::Program;
};

}  // namespace Render

#endif
