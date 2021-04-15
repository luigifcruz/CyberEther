#ifndef RENDER_GLES_INSTANCE_H
#define RENDER_GLES_INSTANCE_H

#include "base/instance.hpp"
#include "gles/api.hpp"
#include "gles/state.hpp"
#include "gles/program.hpp"
#include "gles/surface.hpp"
#include "gles/texture.hpp"

namespace Render {

class GLES::Instance : public Render::Instance {
public:
    Instance(Config& c, State& s) : Render::Instance(c), state(s) {};

    Result init();
    Result terminate();

    Result clear();
    Result draw();
    Result step();

    bool keepRunning();

    std::shared_ptr<Program> _createProgram(Program::Config& cfg, State& state) {
        auto ptr = std::make_shared<Program>(cfg, state);
        programs.push_back(ptr);
        return ptr;
    };

private:
    State& state;

    ImGuiIO* io;
    ImGuiStyle* style;

    uint vao, vbo, ebo;

    std::vector<std::shared_ptr<Program>> programs;

    Result createBuffers();
    Result destroyBuffers();

    Result createImgui();
    Result destroyImgui();
    Result startImgui();
    Result endImgui();

    static Result getError(std::string, std::string, int);
};

} // namespace Render

#endif
