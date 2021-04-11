#ifndef RENDER_GLES_STATE_H
#define RENDER_GLES_STATE_H

#include "types.hpp"
#include "gles/api.hpp"

namespace Render {

class GLES::State {
public:
    GLFWwindow* window;
    uint vbo, ebo;
    uint shaderProgram;
};

} // namespace Render

#endif
