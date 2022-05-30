#ifndef JETSTREAM_RENDER_BASE_WINDOW_HH
#define JETSTREAM_RENDER_BASE_WINDOW_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"

namespace Jetstream::Render {

class Window {
 public:
    struct Config {
        Size2D<U64> size = {1280, 720};
        std::string title = "Render";
        bool resizable = false;
        bool imgui = false;
        bool vsync = true;
        float scale = -1.0;
    };

    explicit Window(const Config& config) : config(config) {
        JST_DEBUG("Window initialized.");
    }
    virtual ~Window() = default;

    virtual const Result create() = 0;
    virtual const Result destroy() = 0;
    virtual const Result begin() = 0;
    virtual const Result end() = 0;

    virtual const Result synchronize() = 0;
    virtual const bool keepRunning() = 0;

    virtual const Result bind(const std::shared_ptr<Surface>& surface) = 0;
    virtual constexpr const Device implementation() const = 0;

    template<Device D> 
    static std::shared_ptr<Window> Factory(const Config& config) {
        return std::make_shared<WindowImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
