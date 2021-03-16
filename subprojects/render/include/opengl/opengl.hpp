#ifndef RENDER_OPENGL_BACKEND_CYBERETHER_H
#define RENDER_OPENGL_BACKEND_CYBERETHER_H

#include <iostream>

#include <GLFW/glfw3.h>

#include "types.hpp"
#include "magic_enum.hpp"
#include "base/base.hpp"

namespace Render {

class OpenGL : public Render {
public:
    Backend getBackend();
    Result init(Config);

private:
    GLFWwindow* window;
};

} // namespace Render

#endif
