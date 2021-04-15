#ifndef RENDER_BASE_INSTANCE_H
#define RENDER_BASE_INSTANCE_H

#include "base/program.hpp"
#include "base/surface.hpp"

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

class Instance {
public:
    struct Config {
        std::string title = "Render";
        int width = 1280;
        int height = 720;
        bool resizable = false;
        bool enableImgui = false;
    };

    Instance(Config& c) : cfg(c) {};
    virtual ~Instance() = default;

    virtual Result init() = 0;
    virtual Result terminate() = 0;

    virtual Result clear() = 0;
    virtual Result draw() = 0;
    virtual Result step() = 0;

    virtual bool keepRunning() = 0;

protected:
    Config& cfg;

    virtual Result createBuffers() = 0;
    virtual Result destroyBuffers() = 0;

    virtual Result createImgui() = 0;
    virtual Result destroyImgui() = 0;
    virtual Result startImgui() = 0;
    virtual Result endImgui() = 0;
};

} // namespace Render

#endif
