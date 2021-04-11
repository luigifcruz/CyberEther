#ifndef RENDER_BASE_INSTANCE_H
#define RENDER_BASE_INSTANCE_H

#include "base/program.hpp"
#include "base/surface.hpp"

namespace Render {

class Instance {
public:
    struct Config {
        int width;
        int height;
        bool resizable;
        std::string title;
        bool enableImgui;
    };

    Instance(Config& c) : a(c) {};
    virtual ~Instance() = default;

    virtual Result init() = 0;
    virtual Result terminate() = 0;

    virtual Result clear() = 0;
    virtual Result draw() = 0;
    virtual Result step() = 0;

    virtual bool keepRunning() = 0;

protected:
    Config& a;

    virtual Result createBuffers() = 0;
    virtual Result destroyBuffers() = 0;

    virtual Result createImgui() = 0;
    virtual Result destroyImgui() = 0;
    virtual Result startImgui() = 0;
    virtual Result endImgui() = 0;
};

} // namespace Render

#endif
