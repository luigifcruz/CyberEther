#include "opengl/opengl.hpp"
#include <unistd.h>

namespace Render {

Backend OpenGL::getBackend() {
    return Backend::OPENGL;
}

Result OpenGL::init(Config cfg) {
    if (!glfwInit()) {
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(cfg.width, cfg.height, cfg.title.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    while (!glfwWindowShouldClose(window)) {
        sleep(1);
    }

    return Result::SUCCESS;
}

} // namespace Render
