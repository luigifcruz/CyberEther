#ifndef RENDER_METAL_DRAW_H
#define RENDER_METAL_DRAW_H

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Draw : public Render::Draw {
public:
    Draw(const Config& cfg, const Metal& i) : Render::Draw(cfg), inst(i) {};

protected:
    Result create();
    Result destroy();
    Result encode(MTL::RenderCommandEncoder* encode);

private:
    const Metal& inst;

    std::shared_ptr<Metal::Vertex> buffer;

    friend class Metal::Program;
};

} // namespace Render

#endif
