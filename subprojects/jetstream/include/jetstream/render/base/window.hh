#ifndef JETSTREAM_RENDER_BASE_WINDOW_HH
#define JETSTREAM_RENDER_BASE_WINDOW_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/render/tools/imgui.h"

namespace Jetstream::Render {

class Window {
 public:
    struct Config {
        float scale = 1.0;
        bool imgui = false;
    };

    struct Stats {
        U64 droppedFrames;
    };

    explicit Window(const Config& config,
                    std::shared_ptr<Viewport::Generic>& viewport)
         : config(config),
           viewport(viewport) {
        JST_DEBUG("Window initialized.");
    }
    virtual ~Window() = default;

    virtual const Result create() = 0;
    virtual const Result destroy() = 0;
    virtual const Result begin() = 0;
    virtual const Result end() = 0;

    virtual const Stats& stats() const = 0;
    virtual void drawDebugMessage() const = 0;

    virtual constexpr const Device device() const = 0;

    virtual const Result bind(const std::shared_ptr<Surface>& surface) = 0;

    template<class T>
    inline Result JETSTREAM_API build(std::shared_ptr<T>& member, 
                                      const auto& config) {
        switch (this->device()) {
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
            case Device::Metal:
                member = T::template Factory<Device::Metal>(config); 
                break;
#endif
            default:
                JST_FATAL("Backend not supported.");
                return Result::ERROR;
        }

        return Result::SUCCESS;
    }

 protected:
    Config config;
    std::shared_ptr<Viewport::Generic> viewport;

    static void ApplyImGuiTheme(const F32& scale);
};

}  // namespace Jetstream::Render

#endif
