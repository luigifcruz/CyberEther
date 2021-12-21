#ifndef RENDER_GLES_PROGRAM_H
#define RENDER_GLES_PROGRAM_H

#include "render/gles/surface.hpp"

namespace Render {

class GLES::Program : public Render::Program {
public:
    Program(const Config& c, const GLES& i) : Render::Program(c), inst(i) {};

    Result setUniform(std::string, const std::vector<int>&);
    Result setUniform(std::string, const std::vector<float>&);

protected:
    Result create();
    Result destroy();
    Result draw();

private:
    const GLES& inst;

    int i;
    uint shader;

    std::vector<std::shared_ptr<GLES::Draw>> draws;
    std::vector<std::shared_ptr<GLES::Texture>> textures;

    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);

    friend class GLES::Surface;
};

} // namespace Render

#endif
