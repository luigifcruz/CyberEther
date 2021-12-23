#ifndef RENDER_GLES_SURFACE_H
#define RENDER_GLES_SURFACE_H

#include <vector>
#include <memory>

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Surface : public Render::Surface {
 public:
    explicit Surface(const Config& config, const GLES& instance);

    Size2D<int> size(const Size2D<int>&) final;

 protected:
    Result create();
    Result destroy();
    Result draw();

 private:
    const GLES& instance;

    uint fbo = 0;
    std::shared_ptr<GLES::Texture> framebuffer;
    std::vector<std::shared_ptr<GLES::Program>> programs;

    Result createFramebuffer();
    Result destroyFramebuffer();

    friend class GLES;
};

}  // namespace Render

#endif
