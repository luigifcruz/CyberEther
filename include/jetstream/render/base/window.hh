#ifndef JETSTREAM_RENDER_BASE_WINDOW_HH
#define JETSTREAM_RENDER_BASE_WINDOW_HH

#include <queue>
#include <memory>
#include <thread>
#include <mutex>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/tools/imgui.h"
#include "jetstream/render/tools/imgui_stdlib.h"
#include "jetstream/render/tools/imgui_fmtlib.h"
#include "jetstream/render/tools/imgui_internal.h"

namespace Jetstream::Render {

class Window {
 public:
    struct Config {
        F32 scale = 1.0f;

        JST_SERDES(scale);
    };

    struct Stats {
        U64 droppedFrames;
    };

    explicit Window(const Config& config) : config(config) {}
    virtual ~Window() = default;

    const Config& getConfig() const {
        return config;
    }

    Result create();
    Result destroy();

    Result begin();
    Result end();

    Result synchronize();

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

    virtual Result bindSurface(const std::shared_ptr<Surface>& surface) = 0;
    virtual Result unbindSurface(const std::shared_ptr<Surface>& surface) = 0;

    virtual Result underlyingCreate() = 0;
    virtual Result underlyingDestroy() = 0;

    virtual Result underlyingBegin() = 0;
    virtual Result underlyingEnd() = 0;

    virtual Result underlyingSynchronize() = 0;

    // Style scaling.

    F32 _scalingFactor;
    F32 _previousScalingFactor;

    void scaleStyle(const Viewport::Generic& viewport);

 private:
    std::vector<std::function<void(const F32& scalingFactor)>> styleSetupCallbacks;
    std::vector<std::function<void(const F32& scalingFactor)>> styleScaleCallbacks;

    bool graphicalLoopThreadStarted;
    std::thread::id graphicalLoopThreadId;
    std::mutex newFrameQueueMutex;

    Result processSurfaceBindQueue();
    Result processSurfaceUnbindQueue();

    std::queue<std::shared_ptr<Surface>> surfaceBindQueue;
    std::queue<std::shared_ptr<Surface>> surfaceUnbindQueue;
};

}  // namespace Jetstream::Render

#endif
