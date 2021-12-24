#ifndef RENDER_GLES_DRAW_H
#define RENDER_GLES_DRAW_H

#include <memory>

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Draw : public Render::Draw {
 public:
    explicit Draw(const Config& config, const GLES& instance);

 protected:
    Result create();
    Result destroy();
    Result draw();

 private:
    const GLES& instance;

    std::shared_ptr<GLES::Vertex> buffer;

    friend class GLES::Program;
};

}  // namespace Render

#endif
