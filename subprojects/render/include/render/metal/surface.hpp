#ifndef RENDER_METAL_SURFACE_H
#define RENDER_METAL_SURFACE_H

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Surface : public Render::Surface {
public:
    Surface(const Config& cfg, const Metal& i) : Render::Surface(cfg), inst(i) {};

    Size2D<int> size(const Size2D<int>&) final;

protected:
    const Metal& inst;

    std::shared_ptr<Metal::Texture> framebuffer;
    std::vector<std::shared_ptr<Metal::Program>> programs;

    Result create() final;
    Result destroy() final;
    Result draw() final;

    Result _createFramebuffer();
    Result _destroyFramebuffer();

    friend class Metal;
};

} // namespace Render

#endif
