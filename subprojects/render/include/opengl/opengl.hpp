#ifndef RENDER_OPENGL_BACKEND_CYBERETHER_H
#define RENDER_OPENGL_BACKEND_CYBERETHER_H

#include <iostream>
#include <unistd.h>

#define GLFW_INCLUDE_ES3
#include <GLFW/glfw3.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "base.hpp"
#include "types.hpp"
#include "magic_enum.hpp"

namespace Render {

class OpenGL : public Backend {
public:
    BackendId getBackendId();

    Result init(Config);
    Result terminate();

    Result clear();
    Result draw();
    Result step();

    bool keepRunning();

private:
    GLFWwindow* window;

    uint vbo, ebo;
    uint shaderProgram;

    Result createSurface();
    Result destroySurface();

    Result createShaders(const char*, const char*);
    Result destroyShaders();

    Result createImgui();
    Result destroyImgui();
    Result startImgui();
    Result endImgui();

    static Result getError(std::string, std::string, int);
    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);
};

} // namespace Render

#endif
