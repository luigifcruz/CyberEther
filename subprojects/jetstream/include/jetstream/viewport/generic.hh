#ifndef JETSTREAM_VIEWPORT_GENERIC_HH
#define JETSTREAM_VIEWPORT_GENERIC_HH

#include "jetstream/logger.hh"
#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/render/types.hh"

#include "jetstream/render/tools/imgui.h"

namespace Jetstream::Viewport {

class Generic {
 public:
    struct Config {
        bool vsync = true;
        bool resizable = false;
        std::string title = "Render";
        Render::Size2D<U64> size = {1280, 720};
    };

    explicit Generic(const Config& config);   
    virtual ~Generic() = default;

    virtual const std::string name() const = 0;

    virtual const Result create() = 0;
    virtual const Result destroy() = 0;

    virtual const Result createImgui() = 0;
    virtual const Result destroyImgui() = 0;

    virtual void* nextDrawable() = 0;
    
    virtual const Result pollEvents() = 0;
    virtual const bool keepRunning() = 0;

    const Result addMousePosEvent(F32 x, F32 y);
    const Result addMouseButtonEvent(U64 button, bool down);

 protected:
    const Config config;
};

}

#endif
