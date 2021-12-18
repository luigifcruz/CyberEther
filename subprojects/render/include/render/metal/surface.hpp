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

    MTL::RenderPassDescriptor* renderPassDesc;

    std::shared_ptr<Metal::Texture> framebuffer;
    std::vector<std::shared_ptr<Metal::Program>> programs;

    Result create();
    Result destroy();
    Result draw(MTL::CommandBuffer* commandBuffer);

    friend class Metal;
};

} // namespace Render

#endif
