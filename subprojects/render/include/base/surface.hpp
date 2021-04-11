#ifndef RENDER_BASE_SURFACE_H
#define RENDER_BASE_SURFACE_H

#include "types.hpp"

namespace Render {

const float vertices[] = {
    // positions ////// // tex //////
    +1.0f, +1.0f, 0.0f, +0.0f, +0.0f, // top right
    +1.0f, -1.0f, 0.0f, +0.0f, +1.0f, // bottom right
    -1.0f, -1.0f, 0.0f, +1.0f, +1.0f, // bottom left
    -1.0f, +1.0f, 0.0f, +1.0f, +0.0f, // top left
};

const uint elements[] = {
    0, 1, 2,
    2, 3, 0
};

class Surface {
public:
    struct Config {
        int width;
        int height;
        bool main;
    };

    Surface(Config& c) : s(c) {};
    virtual ~Surface() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;

protected:
    Config& s;
};

} // namespace Render

#endif
