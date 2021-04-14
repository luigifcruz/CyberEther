#ifndef RENDER_GLES_PROGRAM_H
#define RENDER_GLES_PROGRAM_H

#include "gles/api.hpp"
#include "gles/state.hpp"

namespace Render {

class GLES::Program : public Render::Program {
public:
    Program(Config& c, State& s) : Render::Program(c), state(s) {};

    Result create();
    Result destroy();

    Result setUniform(std::string, const std::vector<int> &);
    Result setUniform(std::string, const std::vector<float> &);

    Result draw();

private:
    State& state;

    uint shaderProgram;

    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);
};

} // namespace Render

#endif
