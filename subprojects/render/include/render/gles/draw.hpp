#ifndef RENDER_GLES_DRAW_H
#define RENDER_GLES_DRAW_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Draw : public Render::Draw {
public:
    Draw(const Config& cfg, const GLES& i) : Render::Draw(cfg), inst(i) {};

protected:
    Result create();
    Result destroy();
    Result draw();

private:
    const GLES& inst;

    std::shared_ptr<GLES::Vertex> buffer;

    friend class GLES::Program;
};

} // namespace Render

#endif
