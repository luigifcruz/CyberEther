#ifndef RENDER_BASE_BACKEND_CYBERETHER_H
#define RENDER_BASE_BACKEND_CYBERETHER_H

#include <iostream>

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

typedef struct {
    int width;
    int height;
    bool resizable;
    std::string title;

    const char* vertexSource;
    const char* fragmentSource;

    bool enableImgui;
} Config;

class Backend {
public:
    virtual ~Backend() = default;
    virtual BackendId getBackendId() = 0;

    virtual Result init(Config) = 0;
    virtual Result terminate() = 0;

    virtual Result clear() = 0;
    virtual Result draw() = 0;
    virtual Result step() = 0;

    virtual bool keepRunning() = 0;

protected:
    Config config;

    virtual Result createSurface() = 0;
    virtual Result destroySurface() = 0;

    virtual Result createShaders(const char*, const char*) = 0;
    virtual Result destroyShaders() = 0;

    virtual Result createImgui() = 0;
    virtual Result destroyImgui() = 0;
    virtual Result startImgui() = 0;
    virtual Result endImgui() = 0;
};

} // namespace Render

#endif
