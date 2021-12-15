#ifndef RENDER_METAL_PROGRAM_H
#define RENDER_METAL_PROGRAM_H

#include "render/metal/surface.hpp"

namespace Render {

class Metal::Program : public Render::Program {
public:
    Program(const Config& c, const Metal& i) : Render::Program(c), inst(i) {};

    Result setUniform(std::string, const std::vector<int>&) final;
    Result setUniform(std::string, const std::vector<float>&) final;

protected:
    const Metal& inst;

    int i;
    uint shader;

    std::vector<std::shared_ptr<Metal::Draw>> draws;
    std::vector<std::shared_ptr<Metal::Texture>> textures;

    Result create() final;
    Result destroy() final;
    Result draw() final;

private:
    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);

    friend class Metal::Surface;
};

} // namespace Render

#endif
