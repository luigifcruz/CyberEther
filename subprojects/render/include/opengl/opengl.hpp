#ifndef RENDER_OPENGL_BACKEND_CYBERETHER_H
#define RENDER_OPENGL_BACKEND_CYBERETHER_H

#include <iostream>
#include <unistd.h>
#define GLFW_INCLUDE_ES3
#include <GLFW/glfw3.h>

#include "types.hpp"
#include "magic_enum.hpp"
#include "base.hpp"

namespace Render {

class OpenGL : public Backend {
public:
    BackendId getBackendId();
    Result init(Config);
    Result clear();
    Result draw();
    Result step();
    Result setupSurface();
    Result setupShaders(const char*, const char*);
    bool keepRunning();

    static Result getError(std::string, std::string, int);
    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);

private:
    GLFWwindow* window;

    uint vbo, ebo;
    uint shaderProgram;
};

} // namespace Render

#endif
