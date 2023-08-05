#ifndef JETSTREAM_RENDER_BASE_WINDOW_HH
#define JETSTREAM_RENDER_BASE_WINDOW_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/tools/imgui.h"
#include "jetstream/render/tools/imnodes.h"

namespace Jetstream::Render {

class Window {
 public:
    struct Config {
        float scale = 1.0;
        bool imgui = true;

        JST_SERDES(
            JST_SERDES_VAL("scale", scale);
            JST_SERDES_VAL("imgui", imgui);
        );
    };

    struct Stats {
        U64 droppedFrames;
    };

    explicit Window(const Config& config) : config(config) {
        JST_DEBUG("Window initialized.");
    }
    virtual ~Window() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result begin() = 0;
    virtual Result end() = 0;

    virtual const Stats& stats() const = 0;
    virtual void drawDebugMessage() const = 0;

    virtual constexpr Device device() const = 0;

    virtual Result bind(const std::shared_ptr<Surface>& surface) = 0;

    template<class T>
    inline Result JETSTREAM_API build(std::shared_ptr<T>& member, 
                                      const auto& config) {
        switch (this->device()) {
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
            case Device::Metal:
                member = T::template Factory<Device::Metal>(config); 
                break;
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
            case Device::Vulkan:
                member = T::template Factory<Device::Vulkan>(config); 
                break;
#endif
#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
            case Device::WebGPU:
                member = T::template Factory<Device::WebGPU>(config); 
                break;
#endif
            default:
                JST_FATAL("Backend not supported yet.");
                return Result::ERROR;
        }

        return Result::SUCCESS;
    }

 protected:
    Config config;

    static void ApplyImGuiTheme(const F32& scale);
    static void ApplyImNodesTheme();
};

}  // namespace Jetstream::Render

#endif
