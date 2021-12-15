#ifndef RENDER_METAL_DRAW_H
#define RENDER_METAL_DRAW_H

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Draw : public Render::Draw {
public:
    Draw(const Config& cfg, const Metal& i) : Render::Draw(cfg), inst(i) {};

protected:
    const Metal& inst;

    std::shared_ptr<Metal::Vertex> buffer;

    Result create() final;
    Result destroy() final;
    Result draw() final;

    friend class Metal::Program;
};

} // namespace Render

#endif
