#ifndef RENDER_GLES_PROGRAM_H
#define RENDER_GLES_PROGRAM_H

#include <vector>
#include <memory>

#include "render/gles/surface.hpp"

namespace Render {

class GLES::Program : public Render::Program {
 public:
    explicit Program(const Config& config, const GLES& instance);

 protected:
    Result create();
    Result destroy();
    Result draw();

 private:
    const GLES& instance;

    int i;
    uint shader;

    std::vector<std::shared_ptr<GLES::Draw>> draws;
    std::vector<std::shared_ptr<GLES::Texture>> textures;

    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);

    friend class GLES::Surface;
};

}  // namespace Render

#endif
