#ifndef RENDER_GLES_PROGRAM_H
#define RENDER_GLES_PROGRAM_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Program : public Render::Program {
public:
    Program(Config& c, State& s) : Render::Program(c), state(s) {};

    Result create();
    Result destroy();
    Result draw();

    Result setUniform(std::string, const std::vector<int> &);
    Result setUniform(std::string, const std::vector<float> &);

protected:
    State& state;

    uint shader;
    int i;

    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);
};

} // namespace Render

#endif
