#ifndef RENDER_METAL_DRAW_H
#define RENDER_METAL_DRAW_H

#include <memory>

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Draw : public Render::Draw {
 public:
    explicit Draw(const Config& config, const Metal& instance);

 protected:
    Result create();
    Result destroy();
    Result encode(MTL::RenderCommandEncoder* encode);

 private:
    const Metal& instance;

    std::shared_ptr<Metal::Vertex> buffer;

    friend class Metal::Program;
};

}  // namespace Render

#endif
