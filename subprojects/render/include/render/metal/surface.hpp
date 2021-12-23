#ifndef RENDER_METAL_SURFACE_H
#define RENDER_METAL_SURFACE_H

#include <vector>
#include <memory>

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Surface : public Render::Surface {
 public:
    explicit Surface(const Config& config, const Metal& instance);

    Size2D<int> size(const Size2D<int>&) final;

 protected:
    Result create();
    Result destroy();
    Result draw(MTL::CommandBuffer* commandBuffer);

 private:
    const Metal& instance;

    MTL::RenderPassDescriptor* renderPassDesc = nullptr;

    std::shared_ptr<Metal::Texture> framebuffer;
    std::vector<std::shared_ptr<Metal::Program>> programs;

    Result createFramebuffer();
    Result destroyFramebuffer();

    friend class Metal;
};

}  // namespace Render

#endif
