#ifndef RENDER_GLES_STATE_H
#define RENDER_GLES_STATE_H

#include "types.hpp"
#include "gles/api.hpp"

namespace Render {

class GLES::State {
public:
    GLFWwindow* window;
};

} // namespace Render

#endif
