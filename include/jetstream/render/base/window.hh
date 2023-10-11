#ifndef JETSTREAM_RENDER_BASE_WINDOW_HH
#define JETSTREAM_RENDER_BASE_WINDOW_HH

#include <queue>
#include <memory>
#include <thread>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/tools/imgui.h"
#include "jetstream/render/tools/imgui_stdlib.h"
#include "jetstream/render/tools/imgui_internal.h"
#include "jetstream/render/tools/imnodes.h"
#include "jetstream/render/tools/imgui_icons_ext.hh"
#include "jetstream/render/tools/imgui_notify_ext.h"

namespace Jetstream::Render {

class Window {
 public:
    struct Config {
        F32 scale = 1.0f;
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

    virtual Result create();
    virtual Result destroy() = 0;

    virtual Result begin();
    virtual Result end() = 0;

    virtual const Stats& stats() const = 0;
    virtual void drawDebugMessage() const = 0;

    virtual constexpr Device device() const = 0;

    Result bind(const std::shared_ptr<Surface>& surface);
    Result unbind(const std::shared_ptr<Surface>& surface);

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
                JST_ERROR("Backend not supported yet.");
                return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    constexpr const F32& scalingFactor() const {
        return _scalingFactor;
    }

 protected:
    Config config;

    F32 _scalingFactor;
    F32 _previousScalingFactor;

    void ScaleStyle(const Viewport::Generic& viewport);

    virtual Result processSurfaceQueues() = 0;

    std::queue<std::shared_ptr<Surface>> surfaceBindQueue;
    std::queue<std::shared_ptr<Surface>> surfaceUnbindQueue;

 private:
    bool graphicalLoopThreadStarted;
    std::thread::id graphicalLoopThreadId;

    void ImGuiStyleSetup();
    void ImGuiStyleScale();
    void ImNodesStyleSetup();
    void ImNodesStyleScale();
};

}  // namespace Jetstream::Render

#endif
